#!/usr/bin/env python3
"""Export P recon segment boundary tensors for Android native-pipeline checks.

Run this on the server/PyTorch environment. The local Windows machine should
only edit this script and package the output into Android assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=os.environ.get("GVC_RT_SOURCE_ROOT"))
    parser.add_argument("--assets-root", type=Path, default=PROJECT_ROOT / "app" / "src" / "main" / "assets")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "p_recon_segment_trace")
    parser.add_argument("--qp", type=int, default=0)
    parser.add_argument("--copy-assets", action="store_true")
    args = parser.parse_args()
    if args.source_root is None:
        parser.error("--source-root is required, or set GVC_RT_SOURCE_ROOT")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_tensor(path: Path, tensor: torch.Tensor) -> Dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = tensor.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
    arr.astype("<f4", copy=False).tofile(path)
    return {
        "path": str(path),
        "shape": list(arr.shape),
        "dtype": "float32",
        "sha256": sha256_file(path),
    }


def load_f32le(path: Path, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    arr = np.fromfile(path, dtype="<f4")
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"{path} has {arr.size} elements, expected {expected}")
    return torch.from_numpy(arr.reshape(shape)).to(device=device, dtype=torch.float32)


def pixel_unshuffle_static(x: torch.Tensor, downscale_factor: int) -> torch.Tensor:
    n, c, h, w = x.shape
    factor = downscale_factor
    return (
        x.reshape(n, c, h // factor, factor, w // factor, factor)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(n, c * factor * factor, h // factor, w // factor)
    )


def force_exportable_torch_path(source_root: Path) -> None:
    sys.path.insert(0, str(source_root))
    import src.layers.cuda_inference as cuda_inference
    import src.layers.layers as layers
    import src.models.image_model_G_b as image_model
    import src.models.video_model_G_b as video_model
    import torch.nn.functional as functional

    cuda_inference.CUSTOMIZED_CUDA_INFERENCE = False
    layers.CUSTOMIZED_CUDA_INFERENCE = False
    image_model.CUSTOMIZED_CUDA_INFERENCE = False
    video_model.CUSTOMIZED_CUDA_INFERENCE = False
    functional.pixel_unshuffle = pixel_unshuffle_static


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
    return model, device, ckpt


def run_blocks(blocks: Iterable[torch.nn.Module], x: torch.Tensor) -> torch.Tensor:
    out = x
    for block in blocks:
        out = block(out)
    return out


def trace_p_recon(model, p_y_hat: torch.Tensor, p_ctx: torch.Tensor, qp: int) -> Dict[str, torch.Tensor]:
    q_dec = model.q_scale_dec[qp : qp + 1]
    decoder = model.recon_generation_net.decoder
    mlp = model.recon_generation_net.mlp

    tensors: Dict[str, torch.Tensor] = {}
    feature = model.dec(p_y_hat, p_ctx, q_dec)
    tensors["p_reference_feature"] = feature

    unshuffled = pixel_unshuffle_static(feature, 2)
    tensors["p_feature_unshuffled"] = unshuffled

    mlp_norm0 = mlp[0](unshuffled)
    tensors["p_mlp_norm0"] = mlp_norm0
    mlp_dcb0 = mlp[1](mlp_norm0)
    tensors["p_mlp_dcb0"] = mlp_dcb0
    mlp_norm1 = mlp[2](mlp_dcb0)
    mlp_norm1_silu = mlp_norm1 * torch.sigmoid(mlp_norm1)
    tensors["p_mlp_norm1_silu"] = mlp_norm1_silu
    codeword = mlp[3](mlp_norm1_silu)
    tensors["p_codeword"] = codeword

    stage1_pre_ada = run_blocks(decoder.stage1.blocks, decoder.conv_in(codeword))
    tensors["p_stage1_blocks"] = stage1_pre_ada
    stage1 = decoder.ada1(stage1_pre_ada, codeword)
    tensors["p_stage1_adagn"] = stage1

    stage2_pre_ada = run_blocks(decoder.stage2.blocks, stage1)
    tensors["p_stage2_blocks"] = stage2_pre_ada
    stage2 = decoder.ada2(stage2_pre_ada, codeword)
    tensors["p_stage2_adagn"] = stage2

    upsampled_pre_ada = decoder.upsample(stage2)
    tensors["p_upsampled"] = upsampled_pre_ada
    upsampled = decoder.ada3(upsampled_pre_ada, codeword)
    tensors["p_upsampled_adagn"] = upsampled

    stage3_pre_ada = run_blocks(decoder.stage3.blocks, upsampled)
    tensors["p_stage3_blocks"] = stage3_pre_ada
    stage3 = decoder.ada4(stage3_pre_ada, codeword)
    tensors["p_stage3_adagn"] = stage3

    stage4_pre_final = run_blocks(decoder.stage4.blocks, stage3)
    tensors["p_stage4_blocks"] = stage4_pre_final
    final_adagn = decoder.ada_final(stage4_pre_final, codeword)
    tensors["p_stage4_adagn_final"] = final_adagn
    frame_logits = decoder.head(final_adagn)
    tensors["p_recon_head_logits"] = frame_logits
    frame = torch.nn.functional.pixel_shuffle(frame_logits, 8).clamp(-1.0, 1.0)
    tensors["encoder_p_reference_frame"] = frame
    return tensors


def main() -> None:
    args = parse_args()
    assets_root = args.assets_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model, device, ckpt = load_p_model(args.source_root)

    p_y_hat = load_f32le(assets_root / "baseline" / "tensors" / "p_y_hat.f32le", (1, 128, 16, 32), device)
    p_ctx = load_f32le(assets_root / "baseline" / "tensors" / "p_ctx.f32le", (1, 256, 32, 64), device)

    with torch.no_grad():
        tensors = trace_p_recon(model, p_y_hat, p_ctx, args.qp)

    records = {}
    for name, tensor in tensors.items():
        records[name] = save_tensor(output_dir / f"{name}.f32le", tensor)

    manifest = {
        "tool": Path(__file__).name,
        "source_root": str(args.source_root),
        "assets_root": str(assets_root),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "qp": args.qp,
        "records": records,
    }
    manifest_path = output_dir / "p_recon_segment_trace_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {manifest_path}")

    if args.copy_assets:
        target = assets_root / "baseline" / "recon_p_segments"
        target.mkdir(parents=True, exist_ok=True)
        for source in output_dir.glob("*.f32le"):
            target.joinpath(source.name).write_bytes(source.read_bytes())
        target.joinpath("p_recon_segment_trace_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"copied trace tensors to {target}")


if __name__ == "__main__":
    main()
