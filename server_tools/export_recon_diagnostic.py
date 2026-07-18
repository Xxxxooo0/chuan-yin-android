#!/usr/bin/env python3
"""Export and convert minimal recon diagnostic graphs for MTK testing.

Run this script on the Linux server. It intentionally exports small PyTorch
wrappers from source code instead of slicing the old Android ONNX files, so the
MTK failures can be attributed to specific recon structures.
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
from typing import Any

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_unshuffle_static(x: torch.Tensor, downscale_factor: int) -> torch.Tensor:
    """ONNX-exportable static equivalent of F.pixel_unshuffle for fixed NCHW."""
    n, c, h, w = x.shape
    factor = downscale_factor
    return (
        x.reshape(n, c, h // factor, factor, w // factor, factor)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(n, c * factor * factor, h // factor, w // factor)
    )


def mtk_space_to_depth_nchw_pattern(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """MediaTek documented PyTorch NCHW SpaceToDepth pattern."""
    n, c, h, w = x.shape
    return (
        x.reshape(n, c, h // block_size, block_size, w // block_size, block_size)
        .permute(0, 3, 5, 1, 2, 4)
        .contiguous()
        .reshape(n, c * block_size * block_size, h // block_size, w // block_size)
    )


class SpaceToDepth2(torch.autograd.Function):
    """Export pixel_unshuffle(2) as ONNX SpaceToDepth instead of 6D reshape."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return pixel_unshuffle_static(x, 2)

    @staticmethod
    def symbolic(g, x):
        return g.op("SpaceToDepth", x, blocksize_i=2)


def run_command(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"command failed rc={result.returncode}; see {log_path}")


def find_tool(explicit: str | None, name: str) -> str:
    if explicit:
        candidate = Path(explicit)
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(f"{name} does not exist: {candidate}")
    existing = shutil.which(name)
    if existing:
        return existing
    candidate = Path(sys.executable).resolve().parent / name
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError(f"{name} is not in PATH or current Python bin")


def find_ncc(sdk_root: Path, platform: str) -> Path:
    if sdk_root.is_file():
        sdk_root.chmod(sdk_root.stat().st_mode | 0o111)
        return sdk_root
    candidates = [
        sdk_root / "neuron_sdk" / "host" / "bin" / "ncc-tflite",
        sdk_root / "neuron_sdk" / platform / "bin" / "ncc-tflite",
        sdk_root / "host" / "bin" / "ncc-tflite",
        sdk_root / platform / "bin" / "ncc-tflite",
    ]
    for candidate in candidates:
        if candidate.is_file():
            candidate.chmod(candidate.stat().st_mode | 0o111)
            return candidate
    matches = sorted(path for path in sdk_root.rglob("ncc-tflite") if path.is_file())
    if matches:
        matches[0].chmod(matches[0].stat().st_mode | 0o111)
        return matches[0]
    raise FileNotFoundError(
        "missing ncc-tflite; checked "
        + ", ".join(str(p) for p in candidates)
        + f"; also searched recursively under {sdk_root}"
    )


def force_exportable_torch_path(source_root: Path) -> None:
    sys.path.insert(0, str(source_root))
    import src.layers.cuda_inference as cuda_inference
    import src.layers.layers as layers
    import src.models.image_model_G_b as image_model
    import src.models.video_model_G_b as video_model

    cuda_inference.CUSTOMIZED_CUDA_INFERENCE = False
    layers.CUSTOMIZED_CUDA_INFERENCE = False
    image_model.CUSTOMIZED_CUDA_INFERENCE = False
    video_model.CUSTOMIZED_CUDA_INFERENCE = False


def load_i_model(source_root: Path):
    force_exportable_torch_path(source_root)
    from src.models.image_model_G_b import DMCI

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = source_root / "ckpt" / "checkpoints" / "GVC-RT_B_I.pt"
    model = DMCI(encoder_ckpt_path=str(ckpt)).to(device).eval()
    return model, device, sha256(ckpt)


def load_p_model(source_root: Path):
    force_exportable_torch_path(source_root)
    from src.models.video_model_G_b import DMC

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = source_root / "ckpt" / "checkpoints" / "GVC-RT_B_P.pt"
    model = DMC().to(device).eval()
    checkpoint = torch.load(ckpt, map_location="cpu")
    state_dict = checkpoint.get(
        "student_ema",
        checkpoint.get("student", checkpoint.get("state_dict", checkpoint)),
    )
    model.load_state_dict(state_dict, strict=True)
    return model, device, sha256(ckpt)


class ILatentDecoder(nn.Module):
    def __init__(self, model, qp: int):
        super().__init__()
        self.dec = model.dec
        self.register_buffer("q_dec", model.q_scale_dec[qp : qp + 1].detach().clone())

    def forward(self, i_y_hat):
        return self.dec(i_y_hat, self.q_dec)


class ILatentConvIn(nn.Module):
    """First deployable I latent-decoder segment: DCB front plus q_dec."""

    def __init__(self, model, qp: int):
        super().__init__()
        self.conv_in = model.dec.conv_in
        self.register_buffer("q_dec", model.q_scale_dec[qp : qp + 1].detach().clone())

    def forward(self, i_y_hat):
        return self.conv_in(i_y_hat) * self.q_dec


class PLatentDecoder(nn.Module):
    def __init__(self, model, qp: int):
        super().__init__()
        self.dec = model.dec
        self.register_buffer("q_dec", model.q_scale_dec[qp : qp + 1].detach().clone())

    def forward(self, p_y_hat, p_ctx):
        return self.dec(p_y_hat, p_ctx, self.q_dec)


class PReconMlpConvOnly(nn.Module):
    """Approximate the P recon MLP without GroupNorm for deployability probing."""

    def __init__(self, model):
        super().__init__()
        mlp = model.recon_generation_net.mlp
        self.block0 = mlp[1]
        self.block1 = mlp[3]

    def forward(self, p_feature_unshuffled):
        out = self.block0(p_feature_unshuffled)
        out = out * torch.sigmoid(out)
        return self.block1(out)


class PReconFeatureToCodeword(nn.Module):
    """Real P recon wrapper front: feature -> pixel_unshuffle -> MLP -> codeword."""

    def __init__(self, model):
        super().__init__()
        self.mlp = model.recon_generation_net.mlp

    def forward(self, p_reference_feature):
        out = pixel_unshuffle_static(p_reference_feature, 2)
        return self.mlp(out)


class PReconUnshuffleOnly(nn.Module):
    """Real P recon space-to-depth front without MLP."""

    def __init__(self, model):
        super().__init__()

    def forward(self, p_reference_feature):
        return pixel_unshuffle_static(p_reference_feature, 2)


class PReconUnshuffleConvOnly(nn.Module):
    """Space-to-depth implemented as fixed one-hot Conv2d(stride=2)."""

    def __init__(self, model):
        super().__init__()
        in_channels = 256
        factor = 2
        weight = torch.zeros(in_channels * factor * factor, in_channels, factor, factor)
        for channel in range(in_channels):
            for row in range(factor):
                for col in range(factor):
                    out_channel = channel * factor * factor + row * factor + col
                    weight[out_channel, channel, row, col] = 1.0
        self.register_buffer("weight", weight)

    def forward(self, p_reference_feature):
        return torch.nn.functional.conv2d(p_reference_feature, self.weight, stride=2)


class PReconUnshuffleSpaceToDepthOnly(nn.Module):
    """Space-to-depth exported as ONNX SpaceToDepth."""

    def __init__(self, model):
        super().__init__()

    def forward(self, p_reference_feature):
        return SpaceToDepth2.apply(p_reference_feature)


class PReconUnshuffleMtkNchwOnly(nn.Module):
    """MediaTek documented PyTorch NCHW SpaceToDepth pattern."""

    def __init__(self, model):
        super().__init__()

    def forward(self, p_reference_feature):
        return mtk_space_to_depth_nchw_pattern(p_reference_feature, 2)


class PReconMlpFull(nn.Module):
    """Real P recon MLP without the preceding pixel_unshuffle."""

    def __init__(self, model):
        super().__init__()
        self.mlp = model.recon_generation_net.mlp

    def forward(self, p_feature_unshuffled):
        return self.mlp(p_feature_unshuffled)


class PReconMlpNorm0(nn.Module):
    """First GroupNorm in the P recon MLP."""

    def __init__(self, model):
        super().__init__()
        self.norm = model.recon_generation_net.mlp[0]

    def forward(self, p_feature_unshuffled):
        return self.norm(p_feature_unshuffled)


class PReconMlpDcb0(nn.Module):
    """First DepthConvBlock in the P recon MLP."""

    def __init__(self, model):
        super().__init__()
        self.block = model.recon_generation_net.mlp[1]

    def forward(self, p_mlp_norm0):
        return self.block(p_mlp_norm0)


class ExplicitDepthConvBlock(nn.Module):
    """Math-equivalent static DCB form with explicit ops for converter probing."""

    def __init__(self, block: nn.Module):
        super().__init__()
        self.adaptor = block.adaptor
        self.shortcut = bool(block.shortcut)
        self.dc0 = block.dc[0]
        self.dc2 = block.dc[2]
        self.dc3 = block.dc[3]
        self.ffn0 = block.ffn[0]
        self.ffn2 = block.ffn[2]

    @staticmethod
    def wsilu(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(4.0 * x) * x

    @staticmethod
    def wsilu_chunk_add(x: torch.Tensor) -> torch.Tensor:
        out = ExplicitDepthConvBlock.wsilu(x)
        half = out.shape[1] // 2
        return out[:, :half, :, :] + out[:, half:, :, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.adaptor is not None:
            x = self.adaptor(x)
        out = self.dc0(x)
        out = self.wsilu(out)
        out = self.dc2(out)
        out = self.dc3(out)
        out = out + x
        ffn = self.ffn0(out)
        ffn = self.wsilu_chunk_add(ffn)
        out = self.ffn2(ffn) + out
        if self.shortcut:
            out = out + x
        return out


class ExplicitDepthConvSequence(nn.Module):
    """Sequential DepthConvBlocks rewritten as explicit static DCBs."""

    def __init__(self, blocks: nn.Module):
        super().__init__()
        self.blocks = nn.ModuleList([ExplicitDepthConvBlock(block) for block in blocks])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.blocks:
            out = block(out)
        return out


class PReconMlpDcb0Explicit(nn.Module):
    """First P recon MLP DCB with explicit static logic."""

    def __init__(self, model):
        super().__init__()
        self.block = ExplicitDepthConvBlock(model.recon_generation_net.mlp[1])

    def forward(self, p_mlp_norm0):
        return self.block(p_mlp_norm0)


class PReconMlpNorm1(nn.Module):
    """Second GroupNorm in the P recon MLP."""

    def __init__(self, model):
        super().__init__()
        self.norm = model.recon_generation_net.mlp[2]

    def forward(self, p_mlp_dcb0):
        return self.norm(p_mlp_dcb0)


class PReconMlpSilu(nn.Module):
    """SiLU gate between the second GroupNorm and final DepthConvBlock."""

    def __init__(self, model):
        super().__init__()

    def forward(self, p_mlp_norm1):
        return p_mlp_norm1 * torch.sigmoid(p_mlp_norm1)


class PReconMlpDcb1(nn.Module):
    """Final DepthConvBlock in the P recon MLP."""

    def __init__(self, model):
        super().__init__()
        self.block = model.recon_generation_net.mlp[3]

    def forward(self, p_mlp_silu):
        return self.block(p_mlp_silu)


class PReconMlpDcb1Explicit(nn.Module):
    """Final P recon MLP DCB with explicit static logic."""

    def __init__(self, model):
        super().__init__()
        self.block = ExplicitDepthConvBlock(model.recon_generation_net.mlp[3])

    def forward(self, p_mlp_silu):
        return self.block(p_mlp_silu)


class PReconMlpDcb0Adaptor(nn.Module):
    """Adaptor 1x1 Conv in the first P recon MLP DepthConvBlock."""

    def __init__(self, model):
        super().__init__()
        self.adaptor = model.recon_generation_net.mlp[1].adaptor

    def forward(self, p_mlp_norm0):
        return self.adaptor(p_mlp_norm0)


class PReconMlpDcb0Dc(nn.Module):
    """Depthwise-conv branch of the first P recon MLP DepthConvBlock."""

    def __init__(self, model):
        super().__init__()
        self.dc = model.recon_generation_net.mlp[1].dc

    def forward(self, p_dcb0_adapted):
        return self.dc(p_dcb0_adapted)


class PReconMlpDcb0DcAdd(nn.Module):
    """Depthwise-conv branch plus residual add in the first P recon MLP DCB."""

    def __init__(self, model):
        super().__init__()
        self.dc = model.recon_generation_net.mlp[1].dc

    def forward(self, p_dcb0_adapted):
        return self.dc(p_dcb0_adapted) + p_dcb0_adapted


class PReconMlpDcb0Ffn(nn.Module):
    """FFN branch of the first P recon MLP DepthConvBlock."""

    def __init__(self, model):
        super().__init__()
        self.ffn = model.recon_generation_net.mlp[1].ffn

    def forward(self, p_dcb0_dc_add):
        return self.ffn(p_dcb0_dc_add)


class PReconMlpDcb0FfnAdd(nn.Module):
    """FFN branch plus residual add in the first P recon MLP DCB."""

    def __init__(self, model):
        super().__init__()
        self.ffn = model.recon_generation_net.mlp[1].ffn

    def forward(self, p_dcb0_dc_add):
        return self.ffn(p_dcb0_dc_add) + p_dcb0_dc_add


class PReconMlpDcb1Adaptor(nn.Module):
    """Adaptor 1x1 Conv in the final P recon MLP DepthConvBlock."""

    def __init__(self, model):
        super().__init__()
        self.adaptor = model.recon_generation_net.mlp[3].adaptor

    def forward(self, p_mlp_silu):
        return self.adaptor(p_mlp_silu)


class PReconMlpDcb1Dc(nn.Module):
    """Depthwise-conv branch of the final P recon MLP DepthConvBlock."""

    def __init__(self, model):
        super().__init__()
        self.dc = model.recon_generation_net.mlp[3].dc

    def forward(self, p_dcb1_adapted):
        return self.dc(p_dcb1_adapted)


class PReconMlpDcb1DcAdd(nn.Module):
    """Depthwise-conv branch plus residual add in the final P recon MLP DCB."""

    def __init__(self, model):
        super().__init__()
        self.dc = model.recon_generation_net.mlp[3].dc

    def forward(self, p_dcb1_adapted):
        return self.dc(p_dcb1_adapted) + p_dcb1_adapted


class PReconMlpDcb1Ffn(nn.Module):
    """FFN branch of the final P recon MLP DepthConvBlock."""

    def __init__(self, model):
        super().__init__()
        self.ffn = model.recon_generation_net.mlp[3].ffn

    def forward(self, p_dcb1_dc_add):
        return self.ffn(p_dcb1_dc_add)


class PReconMlpDcb1FfnAdd(nn.Module):
    """FFN branch plus residual add in the final P recon MLP DCB."""

    def __init__(self, model):
        super().__init__()
        self.ffn = model.recon_generation_net.mlp[3].ffn

    def forward(self, p_dcb1_dc_add):
        return self.ffn(p_dcb1_dc_add) + p_dcb1_dc_add


class PReconMlpNorm0Dcb0(nn.Module):
    """First half of the P recon MLP: GroupNorm + DepthConvBlock."""

    def __init__(self, model):
        super().__init__()
        self.norm = model.recon_generation_net.mlp[0]
        self.block = model.recon_generation_net.mlp[1]

    def forward(self, p_feature_unshuffled):
        return self.block(self.norm(p_feature_unshuffled))


class PReconMlpNorm1SiluDcb1(nn.Module):
    """Second half of the P recon MLP: GroupNorm + SiLU + DepthConvBlock."""

    def __init__(self, model):
        super().__init__()
        self.norm = model.recon_generation_net.mlp[2]
        self.block = model.recon_generation_net.mlp[3]

    def forward(self, p_mlp_dcb0):
        out = self.norm(p_mlp_dcb0)
        out = out * torch.sigmoid(out)
        return self.block(out)


class PReconFeatureToCodewordConv(nn.Module):
    """P recon front with fixed-conv space-to-depth followed by the real MLP."""

    def __init__(self, model):
        super().__init__()
        self.unshuffle = PReconUnshuffleConvOnly(model)
        self.mlp = model.recon_generation_net.mlp

    def forward(self, p_reference_feature):
        return self.mlp(self.unshuffle(p_reference_feature))


class PReconFeatureToCodewordSpaceToDepth(nn.Module):
    """P recon front with ONNX SpaceToDepth followed by the real MLP."""

    def __init__(self, model):
        super().__init__()
        self.mlp = model.recon_generation_net.mlp

    def forward(self, p_reference_feature):
        return self.mlp(SpaceToDepth2.apply(p_reference_feature))


class PReconFeatureToCodewordMtkNchw(nn.Module):
    """MTK NCHW SpaceToDepth pattern followed by the real MLP.

    This is a compatibility/speed probe. The MTK pattern channel order differs
    from torch.nn.functional.pixel_unshuffle, so precision requires reordering
    the following MLP weights before this can replace the real pipeline.
    """

    def __init__(self, model):
        super().__init__()
        self.mlp = model.recon_generation_net.mlp

    def forward(self, p_reference_feature):
        return self.mlp(mtk_space_to_depth_nchw_pattern(p_reference_feature, 2))


class PDecoderStage1ConvOnly(nn.Module):
    """Decoder conv_in + stage1 blocks, skipping AdaGN and GroupNorm."""

    def __init__(self, model):
        super().__init__()
        decoder = model.recon_generation_net.decoder
        self.conv_in = decoder.conv_in
        self.blocks = decoder.stage1.blocks

    def forward(self, p_codeword):
        out = self.conv_in(p_codeword)
        for block in self.blocks:
            out = block(out)
        return out


class PDecoderStage2BlocksOnly(nn.Module):
    """Decoder stage2 blocks, skipping AdaGN and GroupNorm."""

    def __init__(self, model):
        super().__init__()
        self.blocks = model.recon_generation_net.decoder.stage2.blocks

    def forward(self, p_stage1_adagn):
        out = p_stage1_adagn
        for block in self.blocks:
            out = block(out)
        return out


class PDecoderStage2BlocksExplicit(nn.Module):
    """Decoder stage2 DCBs with explicit static logic."""

    def __init__(self, model):
        super().__init__()
        self.blocks = ExplicitDepthConvSequence(model.recon_generation_net.decoder.stage2.blocks)

    def forward(self, p_stage1_adagn):
        return self.blocks(p_stage1_adagn)


class PDecoderStage3BlocksOnly(nn.Module):
    """Decoder stage3 blocks after upsample, skipping GroupNorm."""

    def __init__(self, model):
        super().__init__()
        self.blocks = model.recon_generation_net.decoder.stage3.blocks

    def forward(self, p_upsampled):
        out = p_upsampled
        for block in self.blocks:
            out = block(out)
        return out


class PDecoderStage3BlocksExplicit(nn.Module):
    """Decoder stage3 DCBs with explicit static logic."""

    def __init__(self, model):
        super().__init__()
        self.blocks = ExplicitDepthConvSequence(model.recon_generation_net.decoder.stage3.blocks)

    def forward(self, p_upsampled):
        return self.blocks(p_upsampled)


class PDecoderStage3Block0Only(nn.Module):
    """First stage3 DepthConvBlock, 512 channels to 320 channels."""

    def __init__(self, model):
        super().__init__()
        self.block = model.recon_generation_net.decoder.stage3.blocks[0]

    def forward(self, p_upsampled):
        return self.block(p_upsampled)


class PDecoderStage3Block0Explicit(nn.Module):
    """First decoder stage3 DCB with explicit static logic."""

    def __init__(self, model):
        super().__init__()
        self.block = ExplicitDepthConvBlock(model.recon_generation_net.decoder.stage3.blocks[0])

    def forward(self, p_upsampled):
        return self.block(p_upsampled)


class PDecoderStage3Block0Adaptor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.adaptor = model.recon_generation_net.decoder.stage3.blocks[0].adaptor

    def forward(self, p_upsampled):
        return self.adaptor(p_upsampled)


class PDecoderStage3Block0Dc(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.dc = model.recon_generation_net.decoder.stage3.blocks[0].dc

    def forward(self, p_stage3_block0_adapted):
        return self.dc(p_stage3_block0_adapted)


class PDecoderStage3Block0DcAdd(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.dc = model.recon_generation_net.decoder.stage3.blocks[0].dc

    def forward(self, p_stage3_block0_adapted):
        return self.dc(p_stage3_block0_adapted) + p_stage3_block0_adapted


class PDecoderStage3Block0Ffn(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.ffn = model.recon_generation_net.decoder.stage3.blocks[0].ffn

    def forward(self, p_stage3_block0_dc_add):
        return self.ffn(p_stage3_block0_dc_add)


class PDecoderStage3Block0FfnConv1(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.conv1 = model.recon_generation_net.decoder.stage3.blocks[0].ffn[0]

    def forward(self, p_stage3_block0_dc_add):
        return self.conv1(p_stage3_block0_dc_add)


class PDecoderStage3Block0FfnAct(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.act = model.recon_generation_net.decoder.stage3.blocks[0].ffn[1]

    def forward(self, p_stage3_block0_ffn_conv1):
        return self.act(p_stage3_block0_ffn_conv1)


class PDecoderStage3Block0FfnConv2(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.conv2 = model.recon_generation_net.decoder.stage3.blocks[0].ffn[2]

    def forward(self, p_stage3_block0_ffn_act):
        return self.conv2(p_stage3_block0_ffn_act)


class PDecoderStage3Block0FfnAdd(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.ffn = model.recon_generation_net.decoder.stage3.blocks[0].ffn

    def forward(self, p_stage3_block0_dc_add):
        return self.ffn(p_stage3_block0_dc_add) + p_stage3_block0_dc_add


class PDecoderStage3Blocks1To3Only(nn.Module):
    """Remaining stage3 DepthConvBlocks, all 320 channels."""

    def __init__(self, model):
        super().__init__()
        self.blocks = model.recon_generation_net.decoder.stage3.blocks[1:]

    def forward(self, p_stage3_block0):
        out = p_stage3_block0
        for block in self.blocks:
            out = block(out)
        return out


class PDecoderStage3Block1Only(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.block = model.recon_generation_net.decoder.stage3.blocks[1]

    def forward(self, p_stage3_block0):
        return self.block(p_stage3_block0)


class PDecoderStage3Block2Only(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.block = model.recon_generation_net.decoder.stage3.blocks[2]

    def forward(self, p_stage3_block1):
        return self.block(p_stage3_block1)


class PDecoderStage3Block3Only(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.block = model.recon_generation_net.decoder.stage3.blocks[3]

    def forward(self, p_stage3_block2):
        return self.block(p_stage3_block2)


class PDecoderStage4BlocksOnly(nn.Module):
    """Decoder stage4 blocks, skipping AdaGN, GroupNorm and q_recon multiply."""

    def __init__(self, model):
        super().__init__()
        self.blocks = model.recon_generation_net.decoder.stage4.blocks

    def forward(self, p_stage3_adagn):
        out = p_stage3_adagn
        for block in self.blocks:
            out = block(out)
        return out


class PDecoderStage4BlocksExplicit(nn.Module):
    """Decoder stage4 DCBs with explicit static logic."""

    def __init__(self, model):
        super().__init__()
        self.blocks = ExplicitDepthConvSequence(model.recon_generation_net.decoder.stage4.blocks)

    def forward(self, p_stage3_adagn):
        return self.blocks(p_stage3_adagn)


class PDecoderStage1Full(nn.Module):
    def __init__(self, model):
        super().__init__()
        decoder = model.recon_generation_net.decoder
        self.conv_in = decoder.conv_in
        self.ada1 = decoder.ada1
        self.stage1 = decoder.stage1

    def forward(self, p_codeword):
        out = self.conv_in(p_codeword)
        out = self.ada1(out, p_codeword)
        return self.stage1(out)


class PDecoderStage2Full(nn.Module):
    def __init__(self, model):
        super().__init__()
        decoder = model.recon_generation_net.decoder
        self.ada2 = decoder.ada2
        self.stage2 = decoder.stage2

    def forward(self, p_stage1, p_codeword):
        out = self.ada2(p_stage1, p_codeword)
        return self.stage2(out)


class PDecoderUpsampleStage3Full(nn.Module):
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


class PDecoderStage4Full(nn.Module):
    def __init__(self, model):
        super().__init__()
        decoder = model.recon_generation_net.decoder
        self.ada4 = decoder.ada4
        self.stage4 = decoder.stage4

    def forward(self, p_stage3, p_codeword, q_recon):
        out = self.ada4(p_stage3, p_codeword)
        return self.stage4(out, q_recon)


class PDecoderUpsamplerOriginal(nn.Module):
    """Original recon upsampler that currently exports view/permute/view."""

    def __init__(self, model):
        super().__init__()
        self.upsample = model.recon_generation_net.decoder.upsample

    def forward(self, p_stage2):
        return self.upsample(p_stage2)


class PDecoderUpsamplerPixelShuffle(nn.Module):
    """Equivalent upsampler using torch PixelShuffle for TFLite DepthToSpace."""

    def __init__(self, model):
        super().__init__()
        source = model.recon_generation_net.decoder.upsample
        self.conv1 = source.conv1
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, p_stage2):
        return self.shuffle(self.conv1(p_stage2))


class PDecoderUpsamplerConvOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.conv1 = model.recon_generation_net.decoder.upsample.conv1

    def forward(self, p_stage2):
        return self.conv1(p_stage2)


class PDecoderUpsamplerDepthToSpaceOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.depth2space = model.recon_generation_net.decoder.upsample.depth2space

    def forward(self, p_stage2_conv):
        return self.depth2space(p_stage2_conv, block_size=2)


class PReconBigLatentMlp(nn.Module):
    """P latent decoder + real recon MLP front, used to test a larger fused recon block."""

    def __init__(self, model, qp: int):
        super().__init__()
        self.dec = model.dec
        self.mlp = model.recon_generation_net.mlp
        self.register_buffer("q_dec", model.q_scale_dec[qp : qp + 1].detach().clone())

    def forward(self, p_y_hat, p_ctx):
        out = self.dec(p_y_hat, p_ctx, self.q_dec)
        out = pixel_unshuffle_static(out, 2)
        return self.mlp(out)


class PReconBigStage1Stage2(nn.Module):
    """Decoder conv_in + AdaGN/stage1 + AdaGN/stage2 as one diagnostic block."""

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


class PReconBigUpsampleStage3(nn.Module):
    """Decoder AdaGN + upsample + stage3 as one diagnostic block."""

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


class PReconBigStage4Final(nn.Module):
    """Decoder AdaGN + stage4 + final AdaGN/head as one diagnostic block."""

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
        out = torch.nn.functional.pixel_shuffle(out, 8)
        return torch.clamp(out, -1.0, 1.0)


class PStage1Stage2NoNorm(nn.Module):
    """Non-equivalent speed probe: conv_in + stage1/stage2 DCBs without AdaGN."""

    def __init__(self, model):
        super().__init__()
        decoder = model.recon_generation_net.decoder
        self.conv_in = decoder.conv_in
        self.stage1 = decoder.stage1
        self.stage2 = decoder.stage2

    def forward(self, p_codeword):
        out = self.conv_in(p_codeword)
        out = self.stage1(out)
        return self.stage2(out)


class PUpsampleStage3NoNorm(nn.Module):
    """Non-equivalent speed probe: upsample + stage3 DCBs without AdaGN."""

    def __init__(self, model):
        super().__init__()
        decoder = model.recon_generation_net.decoder
        self.upsample = decoder.upsample
        self.stage3 = decoder.stage3

    def forward(self, p_stage2):
        return self.stage3(self.upsample(p_stage2))


class PStage4FinalNoNorm(nn.Module):
    """Non-equivalent speed probe: stage4 DCBs + final head without final AdaGN."""

    def __init__(self, model):
        super().__init__()
        decoder = model.recon_generation_net.decoder
        self.stage4_blocks = decoder.stage4.blocks
        self.head = decoder.head

    def forward(self, p_stage3):
        out = p_stage3
        for block in self.stage4_blocks:
            out = block(out)
        out = self.head(out)
        out = torch.nn.functional.pixel_shuffle(out, 8)
        return torch.clamp(out, -1.0, 1.0)


class PReconPrefixStage1(nn.Module):
    def __init__(self, model):
        super().__init__()
        decoder = model.recon_generation_net.decoder
        self.conv_in = decoder.conv_in
        self.ada1 = decoder.ada1
        self.stage1 = decoder.stage1

    def forward(self, p_codeword):
        out = self.conv_in(p_codeword)
        out = self.ada1(out, p_codeword)
        return self.stage1(out)


class PReconPrefixStage2(nn.Module):
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


class PReconPrefixUpsample(nn.Module):
    def __init__(self, model):
        super().__init__()
        decoder = model.recon_generation_net.decoder
        self.conv_in = decoder.conv_in
        self.ada1 = decoder.ada1
        self.stage1 = decoder.stage1
        self.ada2 = decoder.ada2
        self.stage2 = decoder.stage2
        self.ada3 = decoder.ada3
        self.upsample = decoder.upsample

    def forward(self, p_codeword):
        out = self.conv_in(p_codeword)
        out = self.ada1(out, p_codeword)
        out = self.stage1(out)
        out = self.ada2(out, p_codeword)
        out = self.stage2(out)
        out = self.ada3(out, p_codeword)
        return self.upsample(out)


class PReconPrefixStage3(nn.Module):
    def __init__(self, model):
        super().__init__()
        decoder = model.recon_generation_net.decoder
        self.conv_in = decoder.conv_in
        self.ada1 = decoder.ada1
        self.stage1 = decoder.stage1
        self.ada2 = decoder.ada2
        self.stage2 = decoder.stage2
        self.ada3 = decoder.ada3
        self.upsample = decoder.upsample
        self.stage3 = decoder.stage3

    def forward(self, p_codeword):
        out = self.conv_in(p_codeword)
        out = self.ada1(out, p_codeword)
        out = self.stage1(out)
        out = self.ada2(out, p_codeword)
        out = self.stage2(out)
        out = self.ada3(out, p_codeword)
        out = self.upsample(out)
        return self.stage3(out)


class PReconPrefixStage4(nn.Module):
    def __init__(self, model):
        super().__init__()
        decoder = model.recon_generation_net.decoder
        self.conv_in = decoder.conv_in
        self.ada1 = decoder.ada1
        self.stage1 = decoder.stage1
        self.ada2 = decoder.ada2
        self.stage2 = decoder.stage2
        self.ada3 = decoder.ada3
        self.upsample = decoder.upsample
        self.stage3 = decoder.stage3
        self.ada4 = decoder.ada4
        self.stage4 = decoder.stage4

    def forward(self, p_codeword, quant_step):
        out = self.conv_in(p_codeword)
        out = self.ada1(out, p_codeword)
        out = self.stage1(out)
        out = self.ada2(out, p_codeword)
        out = self.stage2(out)
        out = self.ada3(out, p_codeword)
        out = self.upsample(out)
        out = self.stage3(out)
        out = self.ada4(out, p_codeword)
        return self.stage4(out, quant_step)


class PReconFinalHead(nn.Module):
    def __init__(self, model):
        super().__init__()
        decoder = model.recon_generation_net.decoder
        self.ada_final = decoder.ada_final
        self.head = decoder.head

    def forward(self, p_stage4, p_codeword):
        out = self.ada_final(p_stage4, p_codeword)
        out = self.head(out)
        out = torch.nn.functional.pixel_shuffle(out, 8)
        return torch.clamp(out, -1.0, 1.0)


class PReconFinalHeadNoAda(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.head = model.recon_generation_net.decoder.head

    def forward(self, p_stage4_adagn):
        out = self.head(p_stage4_adagn)
        out = torch.nn.functional.pixel_shuffle(out, 8)
        return torch.clamp(out, -1.0, 1.0)


class PGroupNormProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.GroupNorm(32, 512, eps=1e-6)

    def forward(self, feature):
        return self.norm(feature)


class PAdaptiveGroupNormProbe(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.ada = model.recon_generation_net.decoder.ada1

    def forward(self, feature, p_codeword):
        return self.ada(feature, p_codeword)


class IFastCodewordToFrameProbe(nn.Module):
    """Non-equivalent speed probe: one tiny head from I codeword to RGB frame."""

    def __init__(self):
        super().__init__()
        self.up1 = nn.Sequential(nn.Conv2d(18, 18 * 4, kernel_size=1), nn.PixelShuffle(2))
        self.up2 = nn.Sequential(nn.Conv2d(18, 18 * 4, kernel_size=1), nn.PixelShuffle(2))
        self.up3 = nn.Sequential(nn.Conv2d(18, 18 * 4, kernel_size=1), nn.PixelShuffle(2))
        self.up4 = nn.Sequential(nn.Conv2d(18, 3 * 4, kernel_size=1), nn.PixelShuffle(2))

    def forward(self, i_codeword):
        out = self.up1(i_codeword)
        out = self.up2(out)
        out = self.up3(out)
        return self.up4(out)


class IFastCodewordToFrameBlocksProbe(nn.Module):
    """Non-equivalent speed probe: add a few deployable blocks before the fast head."""

    def __init__(self, block_count: int):
        super().__init__()
        from src.layers.layers import DepthConvBlock

        self.blocks = nn.Sequential(*[DepthConvBlock(18, 18) for _ in range(block_count)])
        self.head = IFastCodewordToFrameProbe()

    def forward(self, i_codeword):
        return self.head(self.blocks(i_codeword))


class IFastCodewordToFrame1BlockProbe(IFastCodewordToFrameBlocksProbe):
    def __init__(self):
        super().__init__(1)


class IFastCodewordToFrame2BlockProbe(IFastCodewordToFrameBlocksProbe):
    def __init__(self):
        super().__init__(2)


class IFastCodewordToFrame4BlockProbe(IFastCodewordToFrameBlocksProbe):
    def __init__(self):
        super().__init__(4)


class PFastFeatureToFrameProbe(nn.Module):
    """Non-equivalent speed probe: direct P reference feature to RGB frame."""

    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(256, 3 * 8 * 8, kernel_size=1)
        self.shuffle = nn.PixelShuffle(8)

    def forward(self, p_reference_feature):
        return self.shuffle(self.head(p_reference_feature))


SEGMENTS: dict[str, dict[str, Any]] = {
    "i_latent_conv_in": {
        "family": "i",
        "module": ILatentConvIn,
        "inputs": [("i_y_hat", (1, 256, 16, 32))],
        "outputs": [("i_dec_stage0", (1, 512, 16, 32))],
    },
    "i_latent_decoder": {
        "family": "i",
        "module": ILatentDecoder,
        "inputs": [("i_y_hat", (1, 256, 16, 32))],
        "outputs": [("i_codeword", (1, 18, 16, 32))],
    },
    "p_latent_decoder": {
        "family": "p",
        "module": PLatentDecoder,
        "inputs": [("p_y_hat", (1, 128, 16, 32)), ("p_ctx", (1, 256, 32, 64))],
        "outputs": [("p_reference_feature", (1, 256, 32, 64))],
    },
    "p_recon_mlp_conv_only": {
        "family": "p",
        "module": PReconMlpConvOnly,
        "inputs": [("p_feature_unshuffled", (1, 1024, 16, 32))],
        "outputs": [("p_codeword", (1, 18, 16, 32))],
    },
    "p_recon_feature_to_codeword": {
        "family": "p",
        "module": PReconFeatureToCodeword,
        "inputs": [("p_reference_feature", (1, 256, 32, 64))],
        "outputs": [("p_codeword", (1, 18, 16, 32))],
    },
    "p_recon_unshuffle_only": {
        "family": "p",
        "module": PReconUnshuffleOnly,
        "inputs": [("p_reference_feature", (1, 256, 32, 64))],
        "outputs": [("p_feature_unshuffled", (1, 1024, 16, 32))],
    },
    "p_recon_unshuffle_conv_only": {
        "family": "p",
        "module": PReconUnshuffleConvOnly,
        "inputs": [("p_reference_feature", (1, 256, 32, 64))],
        "outputs": [("p_feature_unshuffled", (1, 1024, 16, 32))],
    },
    "p_recon_unshuffle_spacetodepth_only": {
        "family": "p",
        "module": PReconUnshuffleSpaceToDepthOnly,
        "inputs": [("p_reference_feature", (1, 256, 32, 64))],
        "outputs": [("p_feature_unshuffled", (1, 1024, 16, 32))],
    },
    "p_recon_unshuffle_mtk_nchw_only": {
        "family": "p",
        "module": PReconUnshuffleMtkNchwOnly,
        "inputs": [("p_reference_feature", (1, 256, 32, 64))],
        "outputs": [("p_feature_unshuffled", (1, 1024, 16, 32))],
    },
    "p_recon_mlp_full": {
        "family": "p",
        "module": PReconMlpFull,
        "inputs": [("p_feature_unshuffled", (1, 1024, 16, 32))],
        "outputs": [("p_codeword", (1, 18, 16, 32))],
    },
    "p_recon_mlp_norm0": {
        "family": "p",
        "module": PReconMlpNorm0,
        "inputs": [("p_feature_unshuffled", (1, 1024, 16, 32))],
        "outputs": [("p_mlp_norm0", (1, 1024, 16, 32))],
    },
    "p_recon_mlp_dcb0": {
        "family": "p",
        "module": PReconMlpDcb0,
        "inputs": [("p_mlp_norm0", (1, 1024, 16, 32))],
        "outputs": [("p_mlp_dcb0", (1, 256, 16, 32))],
    },
    "p_recon_mlp_dcb0_explicit": {
        "family": "p",
        "module": PReconMlpDcb0Explicit,
        "inputs": [("p_mlp_norm0", (1, 1024, 16, 32))],
        "outputs": [("p_mlp_dcb0", (1, 256, 16, 32))],
    },
    "p_recon_mlp_norm1": {
        "family": "p",
        "module": PReconMlpNorm1,
        "inputs": [("p_mlp_dcb0", (1, 256, 16, 32))],
        "outputs": [("p_mlp_norm1", (1, 256, 16, 32))],
    },
    "p_recon_mlp_silu": {
        "family": "p",
        "module": PReconMlpSilu,
        "inputs": [("p_mlp_norm1", (1, 256, 16, 32))],
        "outputs": [("p_mlp_silu", (1, 256, 16, 32))],
    },
    "p_recon_mlp_dcb1": {
        "family": "p",
        "module": PReconMlpDcb1,
        "inputs": [("p_mlp_silu", (1, 256, 16, 32))],
        "outputs": [("p_codeword", (1, 18, 16, 32))],
    },
    "p_recon_mlp_dcb1_explicit": {
        "family": "p",
        "module": PReconMlpDcb1Explicit,
        "inputs": [("p_mlp_silu", (1, 256, 16, 32))],
        "outputs": [("p_codeword", (1, 18, 16, 32))],
    },
    "p_recon_mlp_dcb0_adaptor": {
        "family": "p",
        "module": PReconMlpDcb0Adaptor,
        "inputs": [("p_mlp_norm0", (1, 1024, 16, 32))],
        "outputs": [("p_dcb0_adapted", (1, 256, 16, 32))],
    },
    "p_recon_mlp_dcb0_dc": {
        "family": "p",
        "module": PReconMlpDcb0Dc,
        "inputs": [("p_dcb0_adapted", (1, 256, 16, 32))],
        "outputs": [("p_dcb0_dc", (1, 256, 16, 32))],
    },
    "p_recon_mlp_dcb0_dc_add": {
        "family": "p",
        "module": PReconMlpDcb0DcAdd,
        "inputs": [("p_dcb0_adapted", (1, 256, 16, 32))],
        "outputs": [("p_dcb0_dc_add", (1, 256, 16, 32))],
    },
    "p_recon_mlp_dcb0_ffn": {
        "family": "p",
        "module": PReconMlpDcb0Ffn,
        "inputs": [("p_dcb0_dc_add", (1, 256, 16, 32))],
        "outputs": [("p_dcb0_ffn", (1, 256, 16, 32))],
    },
    "p_recon_mlp_dcb0_ffn_add": {
        "family": "p",
        "module": PReconMlpDcb0FfnAdd,
        "inputs": [("p_dcb0_dc_add", (1, 256, 16, 32))],
        "outputs": [("p_mlp_dcb0", (1, 256, 16, 32))],
    },
    "p_recon_mlp_dcb1_adaptor": {
        "family": "p",
        "module": PReconMlpDcb1Adaptor,
        "inputs": [("p_mlp_silu", (1, 256, 16, 32))],
        "outputs": [("p_dcb1_adapted", (1, 18, 16, 32))],
    },
    "p_recon_mlp_dcb1_dc": {
        "family": "p",
        "module": PReconMlpDcb1Dc,
        "inputs": [("p_dcb1_adapted", (1, 18, 16, 32))],
        "outputs": [("p_dcb1_dc", (1, 18, 16, 32))],
    },
    "p_recon_mlp_dcb1_dc_add": {
        "family": "p",
        "module": PReconMlpDcb1DcAdd,
        "inputs": [("p_dcb1_adapted", (1, 18, 16, 32))],
        "outputs": [("p_dcb1_dc_add", (1, 18, 16, 32))],
    },
    "p_recon_mlp_dcb1_ffn": {
        "family": "p",
        "module": PReconMlpDcb1Ffn,
        "inputs": [("p_dcb1_dc_add", (1, 18, 16, 32))],
        "outputs": [("p_dcb1_ffn", (1, 18, 16, 32))],
    },
    "p_recon_mlp_dcb1_ffn_add": {
        "family": "p",
        "module": PReconMlpDcb1FfnAdd,
        "inputs": [("p_dcb1_dc_add", (1, 18, 16, 32))],
        "outputs": [("p_codeword", (1, 18, 16, 32))],
    },
    "p_recon_mlp_norm0_dcb0": {
        "family": "p",
        "module": PReconMlpNorm0Dcb0,
        "inputs": [("p_feature_unshuffled", (1, 1024, 16, 32))],
        "outputs": [("p_mlp_dcb0", (1, 256, 16, 32))],
    },
    "p_recon_mlp_norm1_silu_dcb1": {
        "family": "p",
        "module": PReconMlpNorm1SiluDcb1,
        "inputs": [("p_mlp_dcb0", (1, 256, 16, 32))],
        "outputs": [("p_codeword", (1, 18, 16, 32))],
    },
    "p_recon_feature_to_codeword_conv": {
        "family": "p",
        "module": PReconFeatureToCodewordConv,
        "inputs": [("p_reference_feature", (1, 256, 32, 64))],
        "outputs": [("p_codeword", (1, 18, 16, 32))],
    },
    "p_recon_feature_to_codeword_spacetodepth": {
        "family": "p",
        "module": PReconFeatureToCodewordSpaceToDepth,
        "inputs": [("p_reference_feature", (1, 256, 32, 64))],
        "outputs": [("p_codeword", (1, 18, 16, 32))],
    },
    "p_recon_feature_to_codeword_mtk_nchw": {
        "family": "p",
        "module": PReconFeatureToCodewordMtkNchw,
        "inputs": [("p_reference_feature", (1, 256, 32, 64))],
        "outputs": [("p_codeword", (1, 18, 16, 32))],
    },
    "p_decoder_stage1_conv_only": {
        "family": "p",
        "module": PDecoderStage1ConvOnly,
        "inputs": [("p_codeword", (1, 18, 16, 32))],
        "outputs": [("p_stage1", (1, 512, 16, 32))],
    },
    "p_decoder_stage2_blocks_only": {
        "family": "p",
        "module": PDecoderStage2BlocksOnly,
        "inputs": [("p_stage1_adagn", (1, 512, 16, 32))],
        "outputs": [("p_stage2", (1, 512, 16, 32))],
    },
    "p_decoder_stage2_blocks_explicit": {
        "family": "p",
        "module": PDecoderStage2BlocksExplicit,
        "inputs": [("p_stage1_adagn", (1, 512, 16, 32))],
        "outputs": [("p_stage2", (1, 512, 16, 32))],
    },
    "p_decoder_stage3_blocks_only": {
        "family": "p",
        "module": PDecoderStage3BlocksOnly,
        "inputs": [("p_upsampled", (1, 512, 32, 64))],
        "outputs": [("p_stage3", (1, 320, 32, 64))],
    },
    "p_decoder_stage3_blocks_explicit": {
        "family": "p",
        "module": PDecoderStage3BlocksExplicit,
        "inputs": [("p_upsampled", (1, 512, 32, 64))],
        "outputs": [("p_stage3", (1, 320, 32, 64))],
    },
    "p_decoder_stage3_block0_only": {
        "family": "p",
        "module": PDecoderStage3Block0Only,
        "inputs": [("p_upsampled", (1, 512, 32, 64))],
        "outputs": [("p_stage3_block0", (1, 320, 32, 64))],
    },
    "p_decoder_stage3_block0_explicit": {
        "family": "p",
        "module": PDecoderStage3Block0Explicit,
        "inputs": [("p_upsampled", (1, 512, 32, 64))],
        "outputs": [("p_stage3_block0", (1, 320, 32, 64))],
    },
    "p_decoder_stage3_block0_adaptor": {
        "family": "p",
        "module": PDecoderStage3Block0Adaptor,
        "inputs": [("p_upsampled", (1, 512, 32, 64))],
        "outputs": [("p_stage3_block0_adapted", (1, 320, 32, 64))],
    },
    "p_decoder_stage3_block0_dc": {
        "family": "p",
        "module": PDecoderStage3Block0Dc,
        "inputs": [("p_stage3_block0_adapted", (1, 320, 32, 64))],
        "outputs": [("p_stage3_block0_dc", (1, 320, 32, 64))],
    },
    "p_decoder_stage3_block0_dc_add": {
        "family": "p",
        "module": PDecoderStage3Block0DcAdd,
        "inputs": [("p_stage3_block0_adapted", (1, 320, 32, 64))],
        "outputs": [("p_stage3_block0_dc_add", (1, 320, 32, 64))],
    },
    "p_decoder_stage3_block0_ffn": {
        "family": "p",
        "module": PDecoderStage3Block0Ffn,
        "inputs": [("p_stage3_block0_dc_add", (1, 320, 32, 64))],
        "outputs": [("p_stage3_block0_ffn", (1, 320, 32, 64))],
    },
    "p_decoder_stage3_block0_ffn_conv1": {
        "family": "p",
        "module": PDecoderStage3Block0FfnConv1,
        "inputs": [("p_stage3_block0_dc_add", (1, 320, 32, 64))],
        "outputs": [("p_stage3_block0_ffn_conv1", (1, 1280, 32, 64))],
    },
    "p_decoder_stage3_block0_ffn_act": {
        "family": "p",
        "module": PDecoderStage3Block0FfnAct,
        "inputs": [("p_stage3_block0_ffn_conv1", (1, 1280, 32, 64))],
        "outputs": [("p_stage3_block0_ffn_act", (1, 640, 32, 64))],
    },
    "p_decoder_stage3_block0_ffn_conv2": {
        "family": "p",
        "module": PDecoderStage3Block0FfnConv2,
        "inputs": [("p_stage3_block0_ffn_act", (1, 640, 32, 64))],
        "outputs": [("p_stage3_block0_ffn", (1, 320, 32, 64))],
    },
    "p_decoder_stage3_block0_ffn_add": {
        "family": "p",
        "module": PDecoderStage3Block0FfnAdd,
        "inputs": [("p_stage3_block0_dc_add", (1, 320, 32, 64))],
        "outputs": [("p_stage3_block0", (1, 320, 32, 64))],
    },
    "p_decoder_stage3_blocks1_3_only": {
        "family": "p",
        "module": PDecoderStage3Blocks1To3Only,
        "inputs": [("p_stage3_block0", (1, 320, 32, 64))],
        "outputs": [("p_stage3", (1, 320, 32, 64))],
    },
    "p_decoder_stage3_block1_only": {
        "family": "p",
        "module": PDecoderStage3Block1Only,
        "inputs": [("p_stage3_block0", (1, 320, 32, 64))],
        "outputs": [("p_stage3_block1", (1, 320, 32, 64))],
    },
    "p_decoder_stage3_block2_only": {
        "family": "p",
        "module": PDecoderStage3Block2Only,
        "inputs": [("p_stage3_block1", (1, 320, 32, 64))],
        "outputs": [("p_stage3_block2", (1, 320, 32, 64))],
    },
    "p_decoder_stage3_block3_only": {
        "family": "p",
        "module": PDecoderStage3Block3Only,
        "inputs": [("p_stage3_block2", (1, 320, 32, 64))],
        "outputs": [("p_stage3_block3", (1, 320, 32, 64))],
    },
    "p_decoder_stage4_blocks_only": {
        "family": "p",
        "module": PDecoderStage4BlocksOnly,
        "inputs": [("p_stage3_adagn", (1, 320, 32, 64))],
        "outputs": [("p_stage4", (1, 320, 32, 64))],
    },
    "p_decoder_stage4_blocks_explicit": {
        "family": "p",
        "module": PDecoderStage4BlocksExplicit,
        "inputs": [("p_stage3_adagn", (1, 320, 32, 64))],
        "outputs": [("p_stage4", (1, 320, 32, 64))],
    },
    "p_decoder_stage1_full": {
        "family": "p",
        "module": PDecoderStage1Full,
        "inputs": [("p_codeword", (1, 18, 16, 32))],
        "outputs": [("p_stage1", (1, 512, 16, 32))],
    },
    "p_decoder_stage2_full": {
        "family": "p",
        "module": PDecoderStage2Full,
        "inputs": [("p_stage1", (1, 512, 16, 32)), ("p_codeword", (1, 18, 16, 32))],
        "outputs": [("p_stage2", (1, 512, 16, 32))],
    },
    "p_decoder_upsample_stage3_full": {
        "family": "p",
        "module": PDecoderUpsampleStage3Full,
        "inputs": [("p_stage2", (1, 512, 16, 32)), ("p_codeword", (1, 18, 16, 32))],
        "outputs": [("p_stage3", (1, 320, 32, 64))],
    },
    "p_decoder_stage4_full": {
        "family": "p",
        "module": PDecoderStage4Full,
        "inputs": [("p_stage3", (1, 320, 32, 64)), ("p_codeword", (1, 18, 16, 32)), ("q_recon", (1, 1, 1, 1))],
        "outputs": [("p_stage4", (1, 320, 32, 64))],
    },
    "p_upsampler_original": {
        "family": "p",
        "module": PDecoderUpsamplerOriginal,
        "inputs": [("p_stage2", (1, 512, 16, 32))],
        "outputs": [("p_stage2_up", (1, 512, 32, 64))],
    },
    "p_upsampler_pixelshuffle": {
        "family": "p",
        "module": PDecoderUpsamplerPixelShuffle,
        "inputs": [("p_stage2", (1, 512, 16, 32))],
        "outputs": [("p_stage2_up", (1, 512, 32, 64))],
    },
    "p_upsampler_conv_only": {
        "family": "p",
        "module": PDecoderUpsamplerConvOnly,
        "inputs": [("p_stage2", (1, 512, 16, 32))],
        "outputs": [("p_stage2_conv", (1, 2048, 16, 32))],
    },
    "p_upsampler_depth_to_space_only": {
        "family": "p",
        "module": PDecoderUpsamplerDepthToSpaceOnly,
        "inputs": [("p_stage2_conv", (1, 2048, 16, 32))],
        "outputs": [("p_stage2_up", (1, 512, 32, 64))],
    },
    "p_recon_big_latent_mlp": {
        "family": "p",
        "module": PReconBigLatentMlp,
        "inputs": [("p_y_hat", (1, 128, 16, 32)), ("p_ctx", (1, 256, 32, 64))],
        "outputs": [("p_codeword", (1, 18, 16, 32))],
    },
    "p_recon_big_stage1_stage2": {
        "family": "p",
        "module": PReconBigStage1Stage2,
        "inputs": [("p_codeword", (1, 18, 16, 32))],
        "outputs": [("p_stage2", (1, 512, 16, 32))],
    },
    "p_recon_big_upsample_stage3": {
        "family": "p",
        "module": PReconBigUpsampleStage3,
        "inputs": [("p_stage2", (1, 512, 16, 32)), ("p_codeword", (1, 18, 16, 32))],
        "outputs": [("p_stage3", (1, 320, 32, 64))],
    },
    "p_recon_big_stage4_final": {
        "family": "p",
        "module": PReconBigStage4Final,
        "inputs": [("p_stage3", (1, 320, 32, 64)), ("p_codeword", (1, 18, 16, 32)), ("q_recon", (1, 1, 1, 1))],
        "outputs": [("p_frame", (1, 3, 256, 512))],
    },
    "p_stage1_stage2_no_norm": {
        "family": "p",
        "module": PStage1Stage2NoNorm,
        "inputs": [("p_codeword", (1, 18, 16, 32))],
        "outputs": [("p_stage2", (1, 512, 16, 32))],
    },
    "p_upsample_stage3_no_norm": {
        "family": "p",
        "module": PUpsampleStage3NoNorm,
        "inputs": [("p_stage2", (1, 512, 16, 32))],
        "outputs": [("p_stage3", (1, 320, 32, 64))],
    },
    "p_stage4_final_no_norm": {
        "family": "p",
        "module": PStage4FinalNoNorm,
        "inputs": [("p_stage3", (1, 320, 32, 64))],
        "outputs": [("p_frame", (1, 3, 256, 512))],
    },
    "p_recon_prefix_stage1": {
        "family": "p",
        "module": PReconPrefixStage1,
        "inputs": [("p_codeword", (1, 18, 16, 32))],
        "outputs": [("p_stage1", (1, 512, 16, 32))],
    },
    "p_recon_prefix_stage2": {
        "family": "p",
        "module": PReconPrefixStage2,
        "inputs": [("p_codeword", (1, 18, 16, 32))],
        "outputs": [("p_stage2", (1, 512, 16, 32))],
    },
    "p_recon_prefix_upsample": {
        "family": "p",
        "module": PReconPrefixUpsample,
        "inputs": [("p_codeword", (1, 18, 16, 32))],
        "outputs": [("p_stage2_up", (1, 512, 32, 64))],
    },
    "p_recon_prefix_stage3": {
        "family": "p",
        "module": PReconPrefixStage3,
        "inputs": [("p_codeword", (1, 18, 16, 32))],
        "outputs": [("p_stage3", (1, 320, 32, 64))],
    },
    "p_recon_prefix_stage4": {
        "family": "p",
        "module": PReconPrefixStage4,
        "inputs": [("p_codeword", (1, 18, 16, 32)), ("q_recon", (1, 1, 1, 1))],
        "outputs": [("p_stage4", (1, 320, 32, 64))],
    },
    "p_recon_final_head": {
        "family": "p",
        "module": PReconFinalHead,
        "inputs": [("p_stage4", (1, 320, 32, 64)), ("p_codeword", (1, 18, 16, 32))],
        "outputs": [("p_frame", (1, 3, 256, 512))],
    },
    "p_recon_final_head_no_ada": {
        "family": "p",
        "module": PReconFinalHeadNoAda,
        "inputs": [("p_stage4_adagn", (1, 320, 32, 64))],
        "outputs": [("p_frame", (1, 3, 256, 512))],
    },
    "p_groupnorm_probe": {
        "family": "probe",
        "module": PGroupNormProbe,
        "inputs": [("feature", (1, 512, 16, 32))],
        "outputs": [("feature_normed", (1, 512, 16, 32))],
    },
    "p_adagn_probe": {
        "family": "p",
        "module": PAdaptiveGroupNormProbe,
        "inputs": [("feature", (1, 512, 16, 32)), ("p_codeword", (1, 18, 16, 32))],
        "outputs": [("feature_adagn", (1, 512, 16, 32))],
    },
    "i_fast_codeword_to_frame_probe": {
        "family": "probe",
        "module": IFastCodewordToFrameProbe,
        "inputs": [("i_codeword", (1, 18, 16, 32))],
        "outputs": [("i_fast_frame", (1, 3, 256, 512))],
    },
    "i_fast_codeword_to_frame_1block_probe": {
        "family": "probe",
        "module": IFastCodewordToFrame1BlockProbe,
        "inputs": [("i_codeword", (1, 18, 16, 32))],
        "outputs": [("i_fast_frame", (1, 3, 256, 512))],
    },
    "i_fast_codeword_to_frame_2block_probe": {
        "family": "probe",
        "module": IFastCodewordToFrame2BlockProbe,
        "inputs": [("i_codeword", (1, 18, 16, 32))],
        "outputs": [("i_fast_frame", (1, 3, 256, 512))],
    },
    "i_fast_codeword_to_frame_4block_probe": {
        "family": "probe",
        "module": IFastCodewordToFrame4BlockProbe,
        "inputs": [("i_codeword", (1, 18, 16, 32))],
        "outputs": [("i_fast_frame", (1, 3, 256, 512))],
    },
    "p_fast_feature_to_frame_probe": {
        "family": "probe",
        "module": PFastFeatureToFrameProbe,
        "inputs": [("p_reference_feature", (1, 256, 32, 64))],
        "outputs": [("p_fast_frame", (1, 3, 256, 512))],
    },
}


def shape_text(shape: tuple[int, ...]) -> str:
    return ",".join(str(dim) for dim in shape)


def build_module(segment: str, source_root: Path, qp: int):
    spec = SEGMENTS[segment]
    if spec["family"] == "i":
        model, device, checkpoint_sha = load_i_model(source_root)
        module = spec["module"](model, qp)
    elif spec["family"] == "p":
        model, device, checkpoint_sha = load_p_model(source_root)
        module_cls = spec["module"]
        module = module_cls(model, qp) if segment in {"p_latent_decoder", "p_recon_big_latent_mlp"} else module_cls(model)
    else:
        force_exportable_torch_path(source_root)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_sha = ""
        module = spec["module"]()
    return module.to(device).eval(), device, checkpoint_sha


def export_onnx(segment: str, source_root: Path, qp: int, output_root: Path) -> dict[str, str]:
    spec = SEGMENTS[segment]
    module, device, checkpoint_sha = build_module(segment, source_root, qp)
    samples = tuple(torch.zeros(shape, dtype=torch.float32, device=device) for _, shape in spec["inputs"])
    onnx_path = output_root / f"{segment}.onnx"
    with torch.no_grad():
        torch.onnx.export(
            module,
            samples[0] if len(samples) == 1 else samples,
            str(onnx_path),
            input_names=[name for name, _ in spec["inputs"]],
            output_names=[name for name, _ in spec["outputs"]],
            opset_version=13,
            do_constant_folding=True,
        )
    return {
        "onnx": str(onnx_path),
        "onnx_sha256": sha256(onnx_path),
        "checkpoint_sha256": checkpoint_sha,
        "input_names": ",".join(name for name, _ in spec["inputs"]),
        "input_shapes": ":".join(shape_text(shape) for _, shape in spec["inputs"]),
        "output_names": ",".join(name for name, _ in spec["outputs"]),
        "output_shapes": ":".join(shape_text(shape) for _, shape in spec["outputs"]),
    }


def convert_tflite(
    segment: str,
    variant: str,
    onnx_record: dict[str, str],
    output_root: Path,
    converter: str,
) -> dict[str, str]:
    tflite_path = output_root / f"{segment}_{variant}.tflite"
    log_path = output_root / "logs" / f"{segment}_{variant}_onnx_converter.log"
    cmd = [
        converter,
        "--input_model_file",
        onnx_record["onnx"],
        "--output_file",
        str(tflite_path),
        "--output_file_format",
        "tflite",
        "--input_names",
        onnx_record["input_names"],
        "--input_shapes",
        onnx_record["input_shapes"],
        "--output_names",
        onnx_record["output_names"],
        "--tflite_op_export_spec",
        "builtin_first",
    ]
    if variant == "fp16_weight":
        cmd += ["--convert_float32_weights_to_float16", "True"]
    run_command(cmd, log_path)
    return {
        "tflite": str(tflite_path),
        "tflite_sha256": sha256(tflite_path),
        "converter_log": str(log_path),
    }


def compile_dla(
    segment: str,
    variant: str,
    tflite_record: dict[str, str],
    output_root: Path,
    ncc: Path,
    arch: str,
    ncc_opt: str,
    ncc_flags: list[str],
) -> dict[str, str]:
    dla_path = output_root / f"{segment}_{variant}.dla"
    log_path = output_root / "logs" / f"{segment}_{variant}_ncc.log"
    cmd = [
        str(ncc),
        "--arch",
        arch,
        "--opt",
        ncc_opt,
        *ncc_flags,
        tflite_record["tflite"],
        "-o",
        str(dla_path),
    ]
    env = os.environ.copy()
    host_lib = ncc.parent.parent / "lib"
    env["LD_LIBRARY_PATH"] = str(host_lib) if not env.get("LD_LIBRARY_PATH") else f"{host_lib}:{env['LD_LIBRARY_PATH']}"
    run_command(cmd, log_path, env=env)
    return {
        "dla": str(dla_path),
        "dla_sha256": sha256(dla_path),
        "ncc_log": str(log_path),
    }


def parse_segments(raw: str) -> list[str]:
    if raw == "all":
        return list(SEGMENTS.keys())
    selected = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in selected if item not in SEGMENTS]
    if unknown:
        raise ValueError(f"unknown segments: {unknown}; valid={list(SEGMENTS)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--android-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--onnx-converter", default=None)
    parser.add_argument("--platform", default="mt6899")
    parser.add_argument("--arch", default="mdla5.3")
    parser.add_argument("--ncc-opt", default="3")
    parser.add_argument("--opt-bw", action="store_true", help="Pass --opt-bw to ncc-tflite DLA compile")
    parser.add_argument("--relax-fp32", action="store_true", help="Pass --relax-fp32 to ncc-tflite DLA compile")
    parser.add_argument(
        "--extra-ncc-flag",
        action="append",
        default=[],
        help="Extra raw flag passed through to ncc-tflite; can be repeated",
    )
    parser.add_argument("--qp", type=int, default=0)
    parser.add_argument("--segments", default="all", help="all or comma-separated segment names")
    parser.add_argument("--variant", choices=("fp32", "fp16_weight", "all"), default="fp32")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--copy-assets", action="store_true")
    args = parser.parse_args()

    android_root = args.android_root.resolve()
    source_root = args.source_root.resolve()
    sdk_root = args.sdk_root.resolve()
    output_root = (args.output_dir or android_root / "outputs" / "recon_diagnostic").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    converter = find_tool(args.onnx_converter, "mtk_onnx_converter")
    ncc = find_ncc(sdk_root, args.platform)
    segments = parse_segments(args.segments)
    variants = ("fp32", "fp16_weight") if args.variant == "all" else (args.variant,)

    print(f"using mtk_onnx_converter: {converter}")
    print(f"using ncc-tflite: {ncc}")
    ncc_flags = list(args.extra_ncc_flag)
    if args.opt_bw:
        ncc_flags.append("--opt-bw")
    if args.relax_fp32:
        ncc_flags.append("--relax-fp32")

    records: list[dict[str, Any]] = []
    for segment in segments:
        for variant in variants:
            record: dict[str, Any] = {"segment": segment, "variant": variant, "status": "started"}
            try:
                onnx_record = export_onnx(segment, source_root, args.qp, output_root)
                record.update(onnx_record)
                record["onnx_export"] = "ok"
            except Exception as exc:
                record.update({"status": "failed", "onnx_export": "failed", "error": str(exc)})
                print(f"failed {segment} {variant} ONNX export: {exc}")
                records.append(record)
                continue

            try:
                tflite_record = convert_tflite(segment, variant, record, output_root, converter)
                record.update(tflite_record)
                record["tflite_convert"] = "ok"
            except Exception as exc:
                record.update({"status": "failed", "tflite_convert": "failed", "error": str(exc)})
                print(f"failed {segment} {variant} TFLite convert: {exc}")
                records.append(record)
                continue

            try:
                dla_record = compile_dla(
                    segment,
                    variant,
                    record,
                    output_root,
                    ncc,
                    args.arch,
                    args.ncc_opt,
                    ncc_flags,
                )
                record.update(dla_record)
                record["ncc_compile"] = "ok"
            except Exception as exc:
                record["ncc_compile"] = "failed"
                record["ncc_error"] = str(exc)
                record["ncc_log"] = str(output_root / "logs" / f"{segment}_{variant}_ncc.log")
                print(f"failed {segment} {variant} NCC compile: {exc}")
            record["status"] = "ok" if record.get("ncc_compile") == "ok" else "partial"
            records.append(record)

    asset_records = [
        record for record in records
        if record.get("status") in ("ok", "partial") and record.get("tflite_convert") == "ok"
    ]
    ok_records = [record for record in records if record.get("status") == "ok"]
    if args.copy_assets and asset_records:
        assets_dir = android_root / "app" / "src" / "main" / "assets" / "recon_diagnostic"
        assets_dir.mkdir(parents=True, exist_ok=True)
        for record in asset_records:
            if "onnx" in record:
                shutil.copy2(record["onnx"], assets_dir / Path(record["onnx"]).name)
            if "tflite" in record:
                shutil.copy2(record["tflite"], assets_dir / Path(record["tflite"]).name)
            if "dla" in record:
                shutil.copy2(record["dla"], assets_dir / Path(record["dla"]).name)

    manifest = {
        "platform": args.platform,
        "arch": args.arch,
        "ncc_flags": ncc_flags,
        "qp": args.qp,
        "segments": segments,
        "variants": list(variants),
        "records": records,
    }
    manifest_path = output_root / "recon_diagnostic_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.copy_assets:
        assets_dir = android_root / "app" / "src" / "main" / "assets" / "recon_diagnostic"
        shutil.copy2(manifest_path, assets_dir / manifest_path.name)
    print(f"wrote {manifest_path}")
    partial_records = [record for record in records if record.get("status") == "partial"]
    failed_records = [record for record in records if record.get("status") == "failed"]
    print(
        f"ok_records={len(ok_records)} partial_records={len(partial_records)} "
        f"failed_records={len(failed_records)} copied_tflite_records={len(asset_records)}"
    )


if __name__ == "__main__":
    main()
