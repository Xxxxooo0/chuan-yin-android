#!/usr/bin/env python3
"""Export complete I/P synthesis graphs with converter-safe normalization.

The rewrite is mathematically equivalent to the source model:
- GroupNorm variance uses the biased population variance;
- AdaGN quantizer variance preserves torch.var's default unbiased correction;
- PixelUnshuffle is represented by a fixed one-hot stride Conv2D.

No AdaGN, GroupNorm, or q_recon operation is left outside the exported graph.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from analyze_recon_neuron_support import analyze_one, find_ncc
from export_recon_diagnostic import PROJECT_ROOT, find_tool, load_i_model, load_p_model, sha256


class FixedGroupNorm(nn.Module):
    """GroupNorm without aten::var, grouped Conv, or channel-layout reshapes."""

    def __init__(self, source: nn.GroupNorm, height: int, width: int) -> None:
        super().__init__()
        channels = source.num_channels
        groups = source.num_groups
        channels_per_group = channels // groups
        if height >= 32 or width >= 64:
            self.pool = nn.Sequential(
                nn.AvgPool2d((8, 8), stride=(8, 8)),
                nn.AvgPool2d((height // 8, width // 8)),
            )
            self.pool_strategy = "hierarchical_8x8_then_global"
        else:
            self.pool = nn.AvgPool2d((height, width))
            self.pool_strategy = "single_global"
        # Grouped 1x1 Conv is lowered by the MTK converter to DepthwiseConv
        # with channel multiplier C/G. MDLA5.3 rejects multiplier 10 and 32,
        # so express the same fixed mapping as sparse ordinary Conv2D weights.
        self.reduce_channels = nn.Conv2d(channels, groups, kernel_size=1, bias=False)
        self.expand_channels = nn.Conv2d(groups, channels, kernel_size=1, bias=False)
        with torch.no_grad():
            self.reduce_channels.weight.zero_()
            self.expand_channels.weight.zero_()
            for group in range(groups):
                channel_start = group * channels_per_group
                channel_end = channel_start + channels_per_group
                self.reduce_channels.weight[
                    group, channel_start:channel_end, 0, 0
                ] = 1.0 / channels_per_group
                self.expand_channels.weight[
                    channel_start:channel_end, group, 0, 0
                ] = 1.0
        self.reduce_channels.weight.requires_grad_(False)
        self.expand_channels.weight.requires_grad_(False)
        self.eps = float(source.eps)
        if source.affine:
            self.register_buffer("affine_weight", source.weight.detach().reshape(1, channels, 1, 1).clone())
            self.register_buffer("affine_bias", source.bias.detach().reshape(1, channels, 1, 1).clone())
        else:
            self.affine_weight = None
            self.affine_bias = None

    def _group_average(self, value: torch.Tensor) -> torch.Tensor:
        return self.expand_channels(self.reduce_channels(self.pool(value)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        mean = self._group_average(value)
        mean_square = self._group_average(value * value)
        variance = torch.relu(mean_square - mean * mean)
        output = (value - mean) * torch.rsqrt(variance + self.eps)
        if self.affine_weight is not None:
            output = output * self.affine_weight + self.affine_bias
        return output


def linear_as_conv(source: nn.Linear) -> nn.Conv2d:
    conv = nn.Conv2d(source.in_features, source.out_features, kernel_size=1, bias=True)
    with torch.no_grad():
        conv.weight.copy_(source.weight.detach().reshape(source.out_features, source.in_features, 1, 1))
        conv.bias.copy_(source.bias.detach())
    return conv


class FixedAdaGN(nn.Module):
    """AdaptiveGroupNorm without aten::var, view, or flatten operators."""

    def __init__(self, source: nn.Module, height: int, width: int) -> None:
        super().__init__()
        self.norm = FixedGroupNorm(source.gn, height, width)
        self.quantizer_pool = nn.AvgPool2d((16, 32))
        self.gamma = linear_as_conv(source.gamma)
        self.beta = linear_as_conv(source.beta)
        self.eps = float(source.eps)
        self.unbiased_factor = float(16 * 32) / float(16 * 32 - 1)

    def forward(self, value: torch.Tensor, codeword: torch.Tensor) -> torch.Tensor:
        quantizer_mean = self.quantizer_pool(codeword)
        quantizer_mean_square = self.quantizer_pool(codeword * codeword)
        quantizer_variance = torch.relu(
            quantizer_mean_square - quantizer_mean * quantizer_mean,
        ) * self.unbiased_factor
        scale_base = quantizer_variance + self.eps
        scale = self.gamma(scale_base * torch.rsqrt(scale_base))
        bias = self.beta(quantizer_mean)
        return scale * self.norm(value) + bias


class FixedStageBlock(nn.Module):
    def __init__(self, source: nn.Module, height: int, width: int) -> None:
        super().__init__()
        self.blocks = source.blocks
        self.norms = nn.ModuleList(
            FixedGroupNorm(norm, height, width) for norm in source.norms
        )

    def forward(
        self,
        value: torch.Tensor,
        quant_step: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for index, block in enumerate(self.blocks):
            value = block(value)
            if index < len(self.norms):
                value = self.norms[index](value)
        if quant_step is not None:
            value = value * quant_step
        return value


class FixedGenerator(nn.Module):
    def __init__(self, model: nn.Module, qp: int) -> None:
        super().__init__()
        source = model.recon_generation_net.decoder
        self.conv_in = source.conv_in
        self.ada1 = FixedAdaGN(source.ada1, 16, 32)
        self.stage1 = FixedStageBlock(source.stage1, 16, 32)
        self.ada2 = FixedAdaGN(source.ada2, 16, 32)
        self.stage2 = FixedStageBlock(source.stage2, 16, 32)
        self.ada3 = FixedAdaGN(source.ada3, 16, 32)
        self.upsample = source.upsample
        self.stage3 = FixedStageBlock(source.stage3, 32, 64)
        self.ada4 = FixedAdaGN(source.ada4, 32, 64)
        self.stage4 = FixedStageBlock(source.stage4, 32, 64)
        self.ada_final = FixedAdaGN(source.ada_final, 32, 64)
        self.head = source.head
        self.register_buffer("q_recon", model.q_scale_recon[qp : qp + 1].detach().clone())

    def forward(self, codeword: torch.Tensor) -> torch.Tensor:
        value = self.conv_in(codeword)
        value = self.stage1(self.ada1(value, codeword))
        value = self.stage2(self.ada2(value, codeword))
        value = self.upsample(self.ada3(value, codeword))
        value = self.stage3(value)
        value = self.stage4(self.ada4(value, codeword), self.q_recon)
        value = self.head(self.ada_final(value, codeword))
        return torch.clamp(F.pixel_shuffle(value, 8), -1.0, 1.0)


class FixedPixelUnshuffle2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        factor = 2
        channels = 256
        conv = nn.Conv2d(
            channels,
            channels * factor * factor,
            kernel_size=factor,
            stride=factor,
            bias=False,
        )
        with torch.no_grad():
            conv.weight.zero_()
            for channel in range(channels):
                for row in range(factor):
                    for column in range(factor):
                        output_channel = channel * factor * factor + row * factor + column
                        conv.weight[output_channel, channel, row, column] = 1.0
        conv.weight.requires_grad_(False)
        self.conv = conv

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(value)


class FixedIFeatureDec(nn.Module):
    def __init__(self, model: nn.Module, qp: int) -> None:
        super().__init__()
        source = model.dec
        self.conv_in = source.conv_in
        rewritten = []
        for module in source.dec_1:
            if isinstance(module, nn.GroupNorm):
                rewritten.append(FixedGroupNorm(module, 16, 32))
            else:
                rewritten.append(module)
        self.body = nn.ModuleList(rewritten)
        self.conv_out = source.conv_out
        self.register_buffer("q_dec", model.q_scale_dec[qp : qp + 1].detach().clone())

    def forward(self, y_hat: torch.Tensor) -> torch.Tensor:
        value = self.conv_in(y_hat) * self.q_dec
        for module in self.body:
            value = module(value)
        value = value * torch.sigmoid(value)
        return torch.clamp(self.conv_out(value), -1.0, 1.0)


class IFullSynthesisNhwc(nn.Module):
    def __init__(self, model: nn.Module, qp: int) -> None:
        super().__init__()
        self.feature_dec = FixedIFeatureDec(model, qp)
        self.generator = FixedGenerator(model, qp)

    def forward(self, y_hat_nhwc: torch.Tensor) -> torch.Tensor:
        y_hat = y_hat_nhwc.permute(0, 3, 1, 2).contiguous()
        frame = self.generator(self.feature_dec(y_hat))
        return frame.permute(0, 2, 3, 1).contiguous()


class PFullSynthesisNhwc(nn.Module):
    def __init__(self, model: nn.Module, qp: int) -> None:
        super().__init__()
        self.feature_dec = model.dec
        self.pixel_unshuffle = FixedPixelUnshuffle2()
        mlp = model.recon_generation_net.mlp
        self.mlp_norm0 = FixedGroupNorm(mlp[0], 16, 32)
        self.mlp_block0 = mlp[1]
        self.mlp_norm1 = FixedGroupNorm(mlp[2], 16, 32)
        self.mlp_block1 = mlp[3]
        self.generator = FixedGenerator(model, qp)
        self.register_buffer("q_dec", model.q_scale_dec[qp : qp + 1].detach().clone())

    def forward(self, y_hat_nhwc: torch.Tensor, ctx_nhwc: torch.Tensor) -> torch.Tensor:
        y_hat = y_hat_nhwc.permute(0, 3, 1, 2).contiguous()
        ctx = ctx_nhwc.permute(0, 3, 1, 2).contiguous()
        feature = self.feature_dec(y_hat, ctx, self.q_dec)
        value = self.mlp_norm0(self.pixel_unshuffle(feature))
        value = self.mlp_block0(value)
        value = self.mlp_norm1(value)
        value = value * torch.sigmoid(value)
        codeword = self.mlp_block1(value)
        frame = self.generator(codeword)
        return frame.permute(0, 2, 3, 1).contiguous()


class IFeatureDecNhwc(nn.Module):
    def __init__(self, model: nn.Module, qp: int) -> None:
        super().__init__()
        self.feature_dec = FixedIFeatureDec(model, qp)

    def forward(self, y_hat_nhwc: torch.Tensor) -> torch.Tensor:
        y_hat = y_hat_nhwc.permute(0, 3, 1, 2).contiguous()
        codeword = self.feature_dec(y_hat)
        return codeword.permute(0, 2, 3, 1).contiguous()


class PFeatureDecNhwc(nn.Module):
    def __init__(self, model: nn.Module, qp: int) -> None:
        super().__init__()
        self.feature_dec = model.dec
        self.register_buffer("q_dec", model.q_scale_dec[qp : qp + 1].detach().clone())

    def forward(self, y_hat_nhwc: torch.Tensor, ctx_nhwc: torch.Tensor) -> torch.Tensor:
        y_hat = y_hat_nhwc.permute(0, 3, 1, 2).contiguous()
        ctx = ctx_nhwc.permute(0, 3, 1, 2).contiguous()
        feature = self.feature_dec(y_hat, ctx, self.q_dec)
        return feature.permute(0, 2, 3, 1).contiguous()


class PCodewordMlp0Nhwc(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        mlp = model.recon_generation_net.mlp
        self.pixel_unshuffle = FixedPixelUnshuffle2()
        self.norm = FixedGroupNorm(mlp[0], 16, 32)
        self.block = mlp[1]

    def forward(self, feature_nhwc: torch.Tensor) -> torch.Tensor:
        feature = feature_nhwc.permute(0, 3, 1, 2).contiguous()
        value = self.block(self.norm(self.pixel_unshuffle(feature)))
        return value.permute(0, 2, 3, 1).contiguous()


class PCodewordMlp1Nhwc(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        mlp = model.recon_generation_net.mlp
        self.norm = FixedGroupNorm(mlp[2], 16, 32)
        self.block = mlp[3]

    def forward(self, value_nhwc: torch.Tensor) -> torch.Tensor:
        value = value_nhwc.permute(0, 3, 1, 2).contiguous()
        value = self.norm(value)
        value = value * torch.sigmoid(value)
        codeword = self.block(value)
        return codeword.permute(0, 2, 3, 1).contiguous()


class GeneratorStage1Nhwc(nn.Module):
    def __init__(self, generator: FixedGenerator) -> None:
        super().__init__()
        self.conv_in = generator.conv_in
        self.ada = generator.ada1
        self.stage = generator.stage1

    def forward(self, codeword_nhwc: torch.Tensor) -> torch.Tensor:
        codeword = codeword_nhwc.permute(0, 3, 1, 2).contiguous()
        value = self.stage(self.ada(self.conv_in(codeword), codeword))
        return value.permute(0, 2, 3, 1).contiguous()


class GeneratorStageNhwc(nn.Module):
    def __init__(self, ada: nn.Module, stage: nn.Module) -> None:
        super().__init__()
        self.ada = ada
        self.stage = stage

    def forward(self, value_nhwc: torch.Tensor, codeword_nhwc: torch.Tensor) -> torch.Tensor:
        value = value_nhwc.permute(0, 3, 1, 2).contiguous()
        codeword = codeword_nhwc.permute(0, 3, 1, 2).contiguous()
        output = self.stage(self.ada(value, codeword))
        return output.permute(0, 2, 3, 1).contiguous()


class GeneratorStage3Nhwc(nn.Module):
    def __init__(self, generator: FixedGenerator) -> None:
        super().__init__()
        self.ada = generator.ada3
        self.upsample = generator.upsample
        self.stage = generator.stage3

    def forward(self, value_nhwc: torch.Tensor, codeword_nhwc: torch.Tensor) -> torch.Tensor:
        value = value_nhwc.permute(0, 3, 1, 2).contiguous()
        codeword = codeword_nhwc.permute(0, 3, 1, 2).contiguous()
        output = self.stage(self.upsample(self.ada(value, codeword)))
        return output.permute(0, 2, 3, 1).contiguous()


class GeneratorStage4Nhwc(nn.Module):
    def __init__(self, generator: FixedGenerator) -> None:
        super().__init__()
        self.ada = generator.ada4
        self.stage = generator.stage4
        self.register_buffer("q_recon", generator.q_recon.detach().clone())

    def forward(self, value_nhwc: torch.Tensor, codeword_nhwc: torch.Tensor) -> torch.Tensor:
        value = value_nhwc.permute(0, 3, 1, 2).contiguous()
        codeword = codeword_nhwc.permute(0, 3, 1, 2).contiguous()
        output = self.stage(self.ada(value, codeword), self.q_recon)
        return output.permute(0, 2, 3, 1).contiguous()


class GeneratorFinalNhwc(nn.Module):
    def __init__(self, generator: FixedGenerator) -> None:
        super().__init__()
        self.ada = generator.ada_final
        self.head = generator.head

    def forward(self, value_nhwc: torch.Tensor, codeword_nhwc: torch.Tensor) -> torch.Tensor:
        value = value_nhwc.permute(0, 3, 1, 2).contiguous()
        codeword = codeword_nhwc.permute(0, 3, 1, 2).contiguous()
        output = self.head(self.ada(value, codeword))
        frame = torch.clamp(F.pixel_shuffle(output, 8), -1.0, 1.0)
        return frame.permute(0, 2, 3, 1).contiguous()


class GroupNormProbeNhwc(nn.Module):
    def __init__(self, source: nn.GroupNorm, height: int = 16, width: int = 32) -> None:
        super().__init__()
        self.norm = FixedGroupNorm(source, height, width)

    def forward(self, value_nhwc: torch.Tensor) -> torch.Tensor:
        value = value_nhwc.permute(0, 3, 1, 2).contiguous()
        output = self.norm(value)
        return output.permute(0, 2, 3, 1).contiguous()


class AdaGNProbeNhwc(nn.Module):
    def __init__(self, source: nn.Module, height: int = 16, width: int = 32) -> None:
        super().__init__()
        self.ada = FixedAdaGN(source, height, width)

    def forward(
        self,
        value_nhwc: torch.Tensor,
        codeword_nhwc: torch.Tensor,
    ) -> torch.Tensor:
        value = value_nhwc.permute(0, 3, 1, 2).contiguous()
        codeword = codeword_nhwc.permute(0, 3, 1, 2).contiguous()
        output = self.ada(value, codeword)
        return output.permute(0, 2, 3, 1).contiguous()


def run(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        return subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        ).returncode


def export_one(
    name: str,
    module: nn.Module,
    samples: tuple[torch.Tensor, ...],
    converter: str,
    ncc: str,
    arch: str,
    output_dir: Path,
) -> dict:
    pt_path = output_dir / f"{name}.pt"
    with torch.no_grad():
        scripted = torch.jit.trace(module.cpu().eval(), samples, strict=False)
        actual_shape = list(scripted(*samples).shape)
        scripted.save(str(pt_path))
    tflite = output_dir / f"{name}.tflite"
    converter_log = output_dir / "logs" / f"{name}_converter.log"
    converter_rc = run(
        [
            converter,
            "--input_script_module_file", str(pt_path),
            "--output_file", str(tflite),
            "--input_shapes", ":".join(
                ",".join(str(dim) for dim in sample.shape) for sample in samples
            ),
        ],
        converter_log,
    )
    record = {
        "name": name,
        "input_shapes_nhwc": [list(sample.shape) for sample in samples],
        "actual_output_shape_nhwc": actual_shape,
        "torchscript": str(pt_path),
        "torchscript_sha256": sha256(pt_path),
        "converter_rc": converter_rc,
        "converter_log": str(converter_log),
        "tflite": str(tflite) if tflite.is_file() else None,
        "tflite_sha256": sha256(tflite) if tflite.is_file() else None,
    }
    if converter_rc != 0 or not tflite.is_file():
        record["status"] = "converter_failed"
        return record
    ncc_record = analyze_one(
        tflite=tflite,
        ncc=ncc,
        arch=arch,
        output_dir=output_dir / "ncc" / name,
        compile_dla=True,
        ncc_flags=["--opt-bw", "--relax-fp32"],
    )
    record["ncc"] = ncc_record
    record["offline_compile_ok"] = (
        ncc_record.get("check_target_rc") == 0
        and ncc_record.get("exec_plan_rc") == 0
        and ncc_record.get("compile_dla_rc") == 0
        and bool(ncc_record.get("dla"))
        and Path(ncc_record["dla"]).is_file()
    )
    record["status"] = "ok" if record["offline_compile_ok"] else "ncc_failed"
    return record


def publish_offline_assets(
    records: list[dict],
    android_root: Path,
    checkpoint_sha256: dict[str, str],
    qp: int,
    arch: str,
) -> Path:
    assets_dir = android_root / "app" / "src" / "mtkOffline" / "assets" / "offline_models"
    assets_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = assets_dir / "decoder_offline_manifest.json"
    published: dict[str, dict] = {}
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        published = {
            item["name"]: item
            for item in existing.get("models", [])
            if isinstance(item, dict) and item.get("name")
        }

    for record in records:
        if record.get("status") != "ok" or not record.get("offline_compile_ok"):
            continue
        ncc_record = record.get("ncc") or {}
        source_dla = Path(ncc_record.get("dla") or "")
        if not source_dla.is_file():
            continue
        target_dla = assets_dir / f"{record['name']}.dla"
        shutil.copy2(source_dla, target_dla)
        published[record["name"]] = {
            "name": record["name"],
            "asset": f"offline_models/{target_dla.name}",
            "dla_sha256": sha256(target_dla),
            "tflite_sha256": record.get("tflite_sha256"),
            "input_shapes_nhwc": record.get("input_shapes_nhwc"),
            "output_shape_nhwc": record.get("actual_output_shape_nhwc"),
            "offline_compile_verified": True,
            "precision_verified": False,
        }

    manifest = {
        "deployment_path": "mtk_offline",
        "component": "decoder_synthesis",
        "arch": arch,
        "qp": qp,
        "checkpoint_sha256": checkpoint_sha256,
        "models": [published[name] for name in sorted(published)],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--android-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pytorch-converter", default=None)
    parser.add_argument("--ncc-tflite", required=True)
    parser.add_argument("--arch", default="mdla5.3")
    parser.add_argument("--qp", type=int, default=0)
    parser.add_argument(
        "--copy-offline-assets",
        action="store_true",
        help="Copy only successfully compiled DLA files into the mtkOffline flavor",
    )
    parser.add_argument(
        "--targets",
        default="all",
        help="all, i_segments, p_segments, or comma-separated target names",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    android_root = args.android_root.resolve()
    output_dir = (
        args.output_dir or android_root / "outputs" / "decoder_full_norm_rewrite_nhwc"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = find_tool(args.pytorch_converter, "mtk_pytorch_converter")
    ncc = find_ncc(args.ncc_tflite)

    print("[decoder-full] loading I/P checkpoints", flush=True)
    i_model, _, i_sha = load_i_model(source_root)
    p_model, _, p_sha = load_p_model(source_root)
    i_segment_keys = [
        "i_featuredec", "i_stage1", "i_stage2", "i_stage3", "i_stage4", "i_final"
    ]
    p_segment_keys = [
        "p_latent", "p_mlp0", "p_mlp1",
        "p_stage1", "p_stage2", "p_stage3", "p_stage4", "p_final"
    ]
    valid_keys = {
        "groupnorm_probe", "adagn_probe",
        "groupnorm_highres_probe", "adagn_highres_probe",
        "i", "p", *i_segment_keys, *p_segment_keys,
    }
    if args.targets == "all":
        selected_keys = ["groupnorm_probe", "adagn_probe", "i", "p"]
    else:
        requested = [item.strip() for item in args.targets.split(",") if item.strip()]
        selected_keys = []
        for item in requested:
            if item == "i_segments":
                selected_keys.extend(i_segment_keys)
            elif item == "p_segments":
                selected_keys.extend(p_segment_keys)
            else:
                selected_keys.append(item)
        unknown = [item for item in selected_keys if item not in valid_keys]
        if unknown:
            raise ValueError(f"unknown --targets entries: {unknown}")

    i_generator = FixedGenerator(i_model.cpu().eval(), args.qp) if any(
        key in i_segment_keys for key in selected_keys
    ) else None
    p_generator = FixedGenerator(p_model.cpu().eval(), args.qp) if any(
        key in p_segment_keys for key in selected_keys
    ) else None
    all_candidates = {
        "groupnorm_probe": lambda: (
            "groupnorm512_norm_rewrite_probe_fp32",
            GroupNormProbeNhwc(i_model.dec.dec_1[1]),
            (torch.zeros((1, 16, 32, 512), dtype=torch.float32),),
        ),
        "adagn_probe": lambda: (
            "adagn512_norm_rewrite_probe_fp32",
            AdaGNProbeNhwc(i_model.recon_generation_net.decoder.ada1),
            (
                torch.zeros((1, 16, 32, 512), dtype=torch.float32),
                torch.zeros((1, 16, 32, 18), dtype=torch.float32),
            ),
        ),
        "groupnorm_highres_probe": lambda: (
            "groupnorm320_highres_norm_rewrite_probe_fp32",
            GroupNormProbeNhwc(
                i_model.recon_generation_net.decoder.stage3.norms[0], 32, 64
            ),
            (torch.zeros((1, 32, 64, 320), dtype=torch.float32),),
        ),
        "adagn_highres_probe": lambda: (
            "adagn320_highres_norm_rewrite_probe_fp32",
            AdaGNProbeNhwc(
                i_model.recon_generation_net.decoder.ada4, 32, 64
            ),
            (
                torch.zeros((1, 32, 64, 320), dtype=torch.float32),
                torch.zeros((1, 16, 32, 18), dtype=torch.float32),
            ),
        ),
        "i": lambda: (
            "i_decoder_synthesis_norm_rewrite_fp32",
            IFullSynthesisNhwc(i_model.cpu().eval(), args.qp),
            (torch.zeros((1, 16, 32, 256), dtype=torch.float32),),
        ),
        "p": lambda: (
            "p_decoder_synthesis_norm_rewrite_fp32",
            PFullSynthesisNhwc(p_model.cpu().eval(), args.qp),
            (
                torch.zeros((1, 16, 32, 128), dtype=torch.float32),
                torch.zeros((1, 32, 64, 256), dtype=torch.float32),
            ),
        ),
        "i_featuredec": lambda: (
            "i_featuredec_norm_rewrite_fp32",
            IFeatureDecNhwc(i_model, args.qp),
            (torch.zeros((1, 16, 32, 256), dtype=torch.float32),),
        ),
        "p_latent": lambda: (
            "p_featuredec_latent_norm_rewrite_fp32",
            PFeatureDecNhwc(p_model, args.qp),
            (
                torch.zeros((1, 16, 32, 128), dtype=torch.float32),
                torch.zeros((1, 32, 64, 256), dtype=torch.float32),
            ),
        ),
        "p_mlp0": lambda: (
            "p_featuredec_mlp0_norm_rewrite_fp32",
            PCodewordMlp0Nhwc(p_model),
            (torch.zeros((1, 32, 64, 256), dtype=torch.float32),),
        ),
        "p_mlp1": lambda: (
            "p_featuredec_mlp1_norm_rewrite_fp32",
            PCodewordMlp1Nhwc(p_model),
            (torch.zeros((1, 16, 32, 256), dtype=torch.float32),),
        ),
        "i_stage1": lambda: ("i_generator_stage1_norm_rewrite_fp32", GeneratorStage1Nhwc(i_generator), (torch.zeros((1, 16, 32, 18)),)),
        "i_stage2": lambda: ("i_generator_stage2_norm_rewrite_fp32", GeneratorStageNhwc(i_generator.ada2, i_generator.stage2), (torch.zeros((1, 16, 32, 512)), torch.zeros((1, 16, 32, 18)))),
        "i_stage3": lambda: ("i_generator_stage3_norm_rewrite_fp32", GeneratorStage3Nhwc(i_generator), (torch.zeros((1, 16, 32, 512)), torch.zeros((1, 16, 32, 18)))),
        "i_stage4": lambda: ("i_generator_stage4_norm_rewrite_fp32", GeneratorStage4Nhwc(i_generator), (torch.zeros((1, 32, 64, 320)), torch.zeros((1, 16, 32, 18)))),
        "i_final": lambda: ("i_generator_final_norm_rewrite_fp32", GeneratorFinalNhwc(i_generator), (torch.zeros((1, 32, 64, 320)), torch.zeros((1, 16, 32, 18)))),
        "p_stage1": lambda: ("p_generator_stage1_norm_rewrite_fp32", GeneratorStage1Nhwc(p_generator), (torch.zeros((1, 16, 32, 18)),)),
        "p_stage2": lambda: ("p_generator_stage2_norm_rewrite_fp32", GeneratorStageNhwc(p_generator.ada2, p_generator.stage2), (torch.zeros((1, 16, 32, 512)), torch.zeros((1, 16, 32, 18)))),
        "p_stage3": lambda: ("p_generator_stage3_norm_rewrite_fp32", GeneratorStage3Nhwc(p_generator), (torch.zeros((1, 16, 32, 512)), torch.zeros((1, 16, 32, 18)))),
        "p_stage4": lambda: ("p_generator_stage4_norm_rewrite_fp32", GeneratorStage4Nhwc(p_generator), (torch.zeros((1, 32, 64, 320)), torch.zeros((1, 16, 32, 18)))),
        "p_final": lambda: ("p_generator_final_norm_rewrite_fp32", GeneratorFinalNhwc(p_generator), (torch.zeros((1, 32, 64, 320)), torch.zeros((1, 16, 32, 18)))),
    }
    candidates = [all_candidates[key]() for key in selected_keys]
    records = []
    manifest_path = output_dir / "decoder_full_norm_rewrite_manifest.json"
    for index, (name, module, samples) in enumerate(candidates, start=1):
        print(f"[decoder-full] {index}/{len(candidates)} export={name}", flush=True)
        try:
            record = export_one(name, module, samples, converter, ncc, args.arch, output_dir)
        except Exception as exc:
            record = {"name": name, "status": "exception", "error": repr(exc)}
        records.append(record)
        manifest = {
            "tool": Path(__file__).name,
            "source_root": str(source_root),
            "checkpoint_sha256": {"i": i_sha, "p": p_sha},
            "qp": args.qp,
            "selected_targets": selected_keys,
            "rewrite": {
                "GroupNorm": "biased E[x^2]-E[x]^2, RELU, MUL(RSQRT), AvgPool and sparse ordinary Conv2D",
                "AdaGN": "unbiased variance N/(N-1), MUL(RSQRT), Linear mapped to Conv2D",
                "P PixelUnshuffle2": "fixed one-hot Conv2D kernel=2 stride=2",
                "high-resolution GroupNorm": "hierarchical AvgPool 8x8 then global AvgPool",
                "P feature decoder": "split into latent, MLP0, and MLP1 DLA graphs",
                "q_recon": "inside stage4",
            },
            "outside_graph": ["rANS", "bitstream parsing", "serial entropy-prior reconstruction"],
            "records": records,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[decoder-full] {name} status={record['status']}", flush=True)
    print(f"wrote {manifest_path}")
    if args.copy_offline_assets:
        offline_manifest = publish_offline_assets(
            records=records,
            android_root=android_root,
            checkpoint_sha256={"i": i_sha, "p": p_sha},
            qp=args.qp,
            arch=args.arch,
        )
        published_count = sum(
            1 for record in records
            if record.get("status") == "ok" and record.get("offline_compile_ok")
        )
        print(f"published_offline_models={published_count} manifest={offline_manifest}")


if __name__ == "__main__":
    main()
