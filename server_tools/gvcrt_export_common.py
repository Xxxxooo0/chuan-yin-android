"""Shared source-loading helpers for the current GVC-RT exporters."""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    checkpoint = source_root / "ckpt" / "checkpoints" / "GVC-RT_B_I.pt"
    model = DMCI(encoder_ckpt_path=str(checkpoint)).to(device).eval()
    return model, device, sha256(checkpoint)


def load_p_model(source_root: Path):
    force_exportable_torch_path(source_root)
    from src.models.video_model_G_b import DMC

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = source_root / "ckpt" / "checkpoints" / "GVC-RT_B_P.pt"
    model = DMC().to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get(
        "student_ema",
        checkpoint.get("student", checkpoint.get("state_dict", checkpoint)),
    )
    model.load_state_dict(state_dict, strict=True)
    return model, device, sha256(checkpoint_path)


class ExplicitDepthConvBlock(nn.Module):
    """Source-equivalent DCB with standard operations for MTK conversion."""

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
        output = ExplicitDepthConvBlock.wsilu(x)
        half = output.shape[1] // 2
        return output[:, :half, :, :] + output[:, half:, :, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.adaptor is not None:
            x = self.adaptor(x)
        output = self.dc0(x)
        output = self.wsilu(output)
        output = self.dc2(output)
        output = self.dc3(output)
        output = output + x
        ffn = self.ffn0(output)
        ffn = self.wsilu_chunk_add(ffn)
        output = self.ffn2(ffn) + output
        if self.shortcut:
            output = output + x
        return output
