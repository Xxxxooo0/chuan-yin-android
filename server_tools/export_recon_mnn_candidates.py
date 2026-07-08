#!/usr/bin/env python3
"""Export P recon ONNX candidates and optionally convert them to MNN.

Run this on the Linux server. The model is loaded directly from the GVC-RT
source tree passed by --source-root; no old Android ONNX/TFLite assets are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(command: list[str], log_path: Path, timeout_sec: int | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[mnn-export] command_log={log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        try:
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"command timed out after {timeout_sec}s; see {log_path}") from exc
    if result.returncode != 0:
        raise RuntimeError(f"command failed rc={result.returncode}; see {log_path}")


def find_mnn_convert(explicit: str | None) -> str:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    for name in ("MNNConvert", "mnnconvert"):
        found = shutil.which(name)
        if found:
            return found
        candidates.append(Path(sys.executable).resolve().parent / name)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "missing MNN converter; install MNN or pass --mnn-convert. "
        "Tried MNNConvert/mnnconvert in PATH and current Python bin."
    )


def force_exportable_torch_path(source_root: Path) -> None:
    sys.path.insert(0, str(source_root))
    import src.layers.cuda_inference as cuda_inference
    import src.layers.layers as layers
    import src.models.video_model_G_b as video_model

    cuda_inference.CUSTOMIZED_CUDA_INFERENCE = False
    layers.CUSTOMIZED_CUDA_INFERENCE = False
    video_model.CUSTOMIZED_CUDA_INFERENCE = False


def load_p_model(source_root: Path):
    print(f"[mnn-export] source_root={source_root}", flush=True)
    force_exportable_torch_path(source_root)
    from src.models.video_model_G_b import DMC

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = source_root / "ckpt" / "checkpoints" / "GVC-RT_B_P.pt"
    print(f"[mnn-export] checkpoint={ckpt}", flush=True)
    if not ckpt.is_file():
        raise FileNotFoundError(f"missing checkpoint: {ckpt}")
    print(f"[mnn-export] device={device}; creating DMC", flush=True)
    model = DMC().to(device).eval()
    print("[mnn-export] loading checkpoint", flush=True)
    checkpoint = torch.load(ckpt, map_location="cpu")
    state_dict = checkpoint.get(
        "student_ema",
        checkpoint.get("student", checkpoint.get("state_dict", checkpoint)),
    )
    print("[mnn-export] loading state_dict", flush=True)
    model.load_state_dict(state_dict, strict=True)
    print("[mnn-export] P model loaded", flush=True)
    return model, device, ckpt


def pixel_unshuffle_static(x: torch.Tensor, factor: int) -> torch.Tensor:
    n, c, h, w = x.shape
    return (
        x.reshape(n, c, h // factor, factor, w // factor, factor)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(n, c * factor * factor, h // factor, w // factor)
    )


class SpaceToDepth2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return pixel_unshuffle_static(x, 2)

    @staticmethod
    def symbolic(g, x):
        return g.op("SpaceToDepth", x, blocksize_i=2)


class PReconMlpOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.mlp = model.recon_generation_net.mlp

    def forward(self, p_reference_feature):
        out = pixel_unshuffle_static(p_reference_feature, 2)
        out = self.mlp[0](out)
        out = self.mlp[1](out)
        out = self.mlp[2](out)
        out = out * torch.sigmoid(out)
        return self.mlp[3](out)


class PReconUnshuffleOnly(nn.Module):
    def __init__(self, model=None):
        super().__init__()

    def forward(self, p_reference_feature):
        return pixel_unshuffle_static(p_reference_feature, 2)


class PReconUnshuffleSpaceToDepthOnly(nn.Module):
    def __init__(self, model=None):
        super().__init__()

    def forward(self, p_reference_feature):
        return SpaceToDepth2.apply(p_reference_feature)


class PDecoderStage3BlocksOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.blocks = model.recon_generation_net.decoder.stage3.blocks

    def forward(self, p_upsampled):
        out = p_upsampled
        for block in self.blocks:
            out = block(out)
        return out


class PDecoderStage4BlocksOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.blocks = model.recon_generation_net.decoder.stage4.blocks

    def forward(self, p_stage3_adagn):
        out = p_stage3_adagn
        for block in self.blocks:
            out = block(out)
        return out


class PReconFinalHeadNoAda(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.head = model.recon_generation_net.decoder.head

    def forward(self, p_stage4_adagn):
        out = self.head(p_stage4_adagn)
        out = F.pixel_shuffle(out, 8)
        return torch.clamp(out, -1.0, 1.0)


class PReconStage1Stage2(nn.Module):
    def __init__(self, model):
        super().__init__()
        decoder = model.recon_generation_net.decoder
        self.conv_in = decoder.conv_in
        self.ada1 = decoder.ada1
        self.stage1 = decoder.stage1
        self.ada2 = decoder.ada2
        self.stage2 = decoder.stage2

    def forward(self, p_codeword):
        out = self.conv_in(p_codeword)
        out = self.ada1(out, p_codeword)
        out = self.stage1(out)
        out = self.ada2(out, p_codeword)
        return self.stage2(out)


class PReconUpsampleStage3(nn.Module):
    def __init__(self, model):
        super().__init__()
        decoder = model.recon_generation_net.decoder
        self.ada3 = decoder.ada3
        self.upsample = decoder.upsample
        self.stage3 = decoder.stage3

    def forward(self, p_stage2, p_codeword):
        out = self.ada3(p_stage2, p_codeword)
        out = self.upsample(out)
        return self.stage3(out)


class PReconStage4Final(nn.Module):
    def __init__(self, model):
        super().__init__()
        decoder = model.recon_generation_net.decoder
        self.ada4 = decoder.ada4
        self.stage4 = decoder.stage4
        self.ada_final = decoder.ada_final
        self.head = decoder.head

    def forward(self, p_stage3, p_codeword, q_recon):
        out = self.ada4(p_stage3, p_codeword)
        out = self.stage4(out, q_recon)
        out = self.ada_final(out, p_codeword)
        out = self.head(out)
        out = F.pixel_shuffle(out, 8)
        return torch.clamp(out, -1.0, 1.0)


class PReconBackHalf(nn.Module):
    def __init__(self, model):
        super().__init__()
        decoder = model.recon_generation_net.decoder
        self.ada3 = decoder.ada3
        self.upsample = decoder.upsample
        self.stage3 = decoder.stage3
        self.ada4 = decoder.ada4
        self.stage4 = decoder.stage4
        self.ada_final = decoder.ada_final
        self.head = decoder.head

    def forward(self, p_stage2, p_codeword, q_recon):
        out = self.ada3(p_stage2, p_codeword)
        out = self.upsample(out)
        out = self.stage3(out)
        out = self.ada4(out, p_codeword)
        out = self.stage4(out, q_recon)
        out = self.ada_final(out, p_codeword)
        out = self.head(out)
        out = F.pixel_shuffle(out, 8)
        return torch.clamp(out, -1.0, 1.0)


SEGMENTS: dict[str, dict[str, object]] = {
    "p_recon_unshuffle_only": {
        "module": PReconUnshuffleOnly,
        "inputs": [("p_reference_feature", (1, 256, 32, 64))],
        "outputs": [("p_unshuffled", (1, 1024, 16, 32))],
    },
    "p_recon_unshuffle_spacetodepth_only": {
        "module": PReconUnshuffleSpaceToDepthOnly,
        "inputs": [("p_reference_feature", (1, 256, 32, 64))],
        "outputs": [("p_unshuffled", (1, 1024, 16, 32))],
    },
    "p_recon_mlp_only": {
        "module": PReconMlpOnly,
        "inputs": [("p_reference_feature", (1, 256, 32, 64))],
        "outputs": [("p_codeword", (1, 18, 16, 32))],
    },
    "p_decoder_stage3_blocks_only": {
        "module": PDecoderStage3BlocksOnly,
        "inputs": [("p_upsampled", (1, 512, 32, 64))],
        "outputs": [("p_stage3", (1, 320, 32, 64))],
    },
    "p_decoder_stage4_blocks_only": {
        "module": PDecoderStage4BlocksOnly,
        "inputs": [("p_stage3_adagn", (1, 320, 32, 64))],
        "outputs": [("p_stage4", (1, 320, 32, 64))],
    },
    "p_recon_final_head_no_ada": {
        "module": PReconFinalHeadNoAda,
        "inputs": [("p_stage4_adagn", (1, 320, 32, 64))],
        "outputs": [("p_recon_frame", (1, 3, 256, 512))],
    },
    "p_recon_stage1_stage2": {
        "module": PReconStage1Stage2,
        "inputs": [("p_codeword", (1, 18, 16, 32))],
        "outputs": [("p_stage2", (1, 512, 16, 32))],
    },
    "p_recon_upsample_stage3": {
        "module": PReconUpsampleStage3,
        "inputs": [("p_stage2", (1, 512, 16, 32)), ("p_codeword", (1, 18, 16, 32))],
        "outputs": [("p_stage3", (1, 320, 32, 64))],
    },
    "p_recon_stage4_final": {
        "module": PReconStage4Final,
        "inputs": [
            ("p_stage3", (1, 320, 32, 64)),
            ("p_codeword", (1, 18, 16, 32)),
            ("q_recon", (1, 320, 1, 1)),
        ],
        "outputs": [("p_recon_frame", (1, 3, 256, 512))],
    },
    "p_recon_back_half": {
        "module": PReconBackHalf,
        "inputs": [
            ("p_stage2", (1, 512, 16, 32)),
            ("p_codeword", (1, 18, 16, 32)),
            ("q_recon", (1, 320, 1, 1)),
        ],
        "outputs": [("p_recon_frame", (1, 3, 256, 512))],
    },
}


def parse_segments(raw: str) -> list[str]:
    if raw == "all":
        return list(SEGMENTS)
    selected = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in selected if item not in SEGMENTS]
    if unknown:
        raise ValueError(f"unknown segments: {unknown}; valid={list(SEGMENTS)}")
    return selected


def export_onnx(
    module: nn.Module,
    input_specs: list[tuple[str, tuple[int, ...]]],
    output_specs: list[tuple[str, tuple[int, ...]]],
    output_path: Path,
    opset: int,
    device: torch.device,
) -> None:
    module.to(device).eval()
    print(f"[mnn-export] export ONNX start {output_path.name}", flush=True)
    inputs = tuple(torch.randn(shape, dtype=torch.float32, device=device) for _, shape in input_specs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            module,
            inputs,
            str(output_path),
            input_names=[name for name, _ in input_specs],
            output_names=[name for name, _ in output_specs],
            opset_version=opset,
            do_constant_folding=True,
        )
    print(f"[mnn-export] export ONNX done {output_path.name}", flush=True)


def convert_mnn(
    converter: str,
    onnx_path: Path,
    mnn_path: Path,
    log_path: Path,
    fp16: bool,
    save_static: bool,
    optimize_prefer: int,
    timeout_sec: int,
) -> None:
    print(f"[mnn-export] convert MNN start {mnn_path.name}", flush=True)
    command = [
        converter,
        "-f",
        "ONNX",
        "--modelFile",
        str(onnx_path),
        "--MNNModel",
        str(mnn_path),
        "--bizCode",
        "GVC_RT",
        "--optimizePrefer",
        str(optimize_prefer),
    ]
    if fp16:
        command.append("--fp16")
    if save_static:
        command.append("--saveStaticModel")
    run_command(command, log_path, timeout_sec=timeout_sec)
    print(f"[mnn-export] convert MNN done {mnn_path.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "recon_mnn_candidates"))
    parser.add_argument("--segments", default="all")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--convert-mnn", action="store_true")
    parser.add_argument("--mnn-convert", default=None)
    parser.add_argument("--mnn-fp16", action="store_true")
    parser.add_argument("--mnn-timeout-sec", type=int, default=600)
    parser.add_argument("--no-save-static", action="store_true")
    parser.add_argument("--optimize-prefer", type=int, default=2)
    parser.add_argument("--copy-assets", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    assets_dir = PROJECT_ROOT / "app" / "src" / "main" / "assets" / "recon_mnn"
    logs_dir = output_dir / "logs"
    selected = parse_segments(args.segments)
    print(f"[mnn-export] selected={','.join(selected)}", flush=True)

    model, device, ckpt = load_p_model(source_root)
    converter = find_mnn_convert(args.mnn_convert) if args.convert_mnn else None
    if converter:
        print(f"[mnn-export] mnn_converter={converter}", flush=True)
    records = []

    for segment in selected:
        spec = SEGMENTS[segment]
        module_cls = spec["module"]
        assert isinstance(module_cls, type)
        module = module_cls(model)
        input_specs = list(spec["inputs"])  # type: ignore[arg-type]
        output_specs = list(spec["outputs"])  # type: ignore[arg-type]
        onnx_path = output_dir / f"{segment}.onnx"
        record = {
            "segment": segment,
            "inputs": [{"name": name, "shape": list(shape)} for name, shape in input_specs],
            "outputs": [{"name": name, "shape": list(shape)} for name, shape in output_specs],
            "onnx": str(onnx_path),
        }
        try:
            export_onnx(module, input_specs, output_specs, onnx_path, args.opset, device)
            record["onnx_sha256"] = sha256(onnx_path)
            record["onnx_export"] = "ok"
            if converter:
                mnn_path = output_dir / f"{segment}.mnn"
                convert_mnn(
                    converter,
                    onnx_path,
                    mnn_path,
                    logs_dir / f"{segment}_mnnconvert.log",
                    fp16=args.mnn_fp16,
                    save_static=not args.no_save_static,
                    optimize_prefer=args.optimize_prefer,
                    timeout_sec=args.mnn_timeout_sec,
                )
                record["mnn"] = str(mnn_path)
                record["mnn_sha256"] = sha256(mnn_path)
                record["mnn_convert"] = "ok"
                if args.copy_assets:
                    assets_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(mnn_path, assets_dir / mnn_path.name)
            print(f"ok {segment}")
        except Exception as exc:  # noqa: BLE001
            record["status"] = "failed"
            record["error"] = str(exc)
            print(f"failed {segment}: {exc}")
        else:
            record["status"] = "ok"
        records.append(record)

    manifest = {
        "tool": "export_recon_mnn_candidates.py",
        "source_root": str(source_root),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256(ckpt),
        "device": str(device),
        "opset": args.opset,
        "convert_mnn": bool(args.convert_mnn),
        "mnn_converter": converter,
        "mnn_fp16": bool(args.mnn_fp16),
        "mnn_timeout_sec": args.mnn_timeout_sec,
        "save_static": not args.no_save_static,
        "optimize_prefer": args.optimize_prefer,
        "segments": selected,
        "records": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "recon_mnn_candidates_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.copy_assets:
        assets_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, assets_dir / manifest_path.name)
    failed = [record for record in records if record.get("status") != "ok"]
    print(f"wrote {manifest_path}")
    print(f"ok_records={len(records) - len(failed)} failed_records={len(failed)}")


if __name__ == "__main__":
    main()
