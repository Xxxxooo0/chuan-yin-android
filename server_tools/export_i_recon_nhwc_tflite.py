#!/usr/bin/env python3
"""Export the complete I recon path from the live PyTorch source to NHWC TFLite.

This is a server-only diagnostic exporter. It deliberately does not consume the
legacy i_recon.onnx asset: the TorchScript module directly reuses DMCI.dec and
DMCI.recon_generation_net from GVC-RT_B_I.pt.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn

from export_recon_diagnostic import PROJECT_ROOT, load_i_model, sha256


INPUT_SHAPE_NCHW = (1, 256, 16, 32)
INPUT_SHAPE_NHWC = (1, 16, 32, 256)
CODEWORD_SHAPE_NHWC = (1, 16, 32, 18)
FRAME_SHAPE_NHWC = (1, 256, 512, 3)
MODEL_NAME = "i_recon_nhwc_fp32"


class IReconNhwc(nn.Module):
    """Exact source I recon with an NHWC external tensor contract."""

    def __init__(self, model: nn.Module, qp: int) -> None:
        super().__init__()
        self.dec = model.dec
        self.recon_generation_net = model.recon_generation_net
        self.register_buffer("q_dec", model.q_scale_dec[qp : qp + 1].detach().clone())
        self.register_buffer("q_recon", model.q_scale_recon[qp : qp + 1].detach().clone())

    def forward(self, i_y_hat_nhwc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        i_y_hat = i_y_hat_nhwc.permute(0, 3, 1, 2).contiguous()
        codeword = self.dec(i_y_hat, self.q_dec)
        frame = self.recon_generation_net(codeword, self.q_recon)
        return (
            codeword.permute(0, 2, 3, 1).contiguous(),
            frame.permute(0, 2, 3, 1).contiguous(),
        )


def find_converter(explicit: str | None) -> str:
    if explicit:
        candidate = Path(explicit)
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(f"mtk_pytorch_converter does not exist: {candidate}")
    candidate = shutil.which("mtk_pytorch_converter")
    if candidate:
        return candidate
    candidate = Path(torch.__file__).resolve().parent.parent / "bin" / "mtk_pytorch_converter"
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError("mtk_pytorch_converter is not in PATH or the active environment bin")


def read_f32(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    expected = int(np.prod(shape))
    if values.size != expected:
        raise ValueError(f"{path} has {values.size} floats, expected {expected} for shape {shape}")
    return values.reshape(shape)


def write_f32(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values.astype("<f4", copy=False).tofile(path)


def run(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True).returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--i-y-hat-f32le", type=Path, required=True)
    parser.add_argument("--android-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pytorch-converter", default=None)
    parser.add_argument("--qp", type=int, default=0)
    parser.add_argument("--copy-assets", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    android_root = args.android_root.resolve()
    output_dir = (args.output_dir or android_root / "outputs" / "i_recon_nhwc").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model, device, checkpoint_sha = load_i_model(source_root)
    module = IReconNhwc(model, args.qp).to(device).eval()
    input_nchw = read_f32(args.i_y_hat_f32le.resolve(), INPUT_SHAPE_NCHW)
    input_nhwc = np.transpose(input_nchw, (0, 2, 3, 1)).copy()
    sample = torch.from_numpy(input_nhwc).to(device=device)

    with torch.no_grad():
        codeword, frame = module(sample)
    codeword_np = codeword.detach().cpu().numpy()
    frame_np = frame.detach().cpu().numpy()
    if tuple(codeword_np.shape) != CODEWORD_SHAPE_NHWC or tuple(frame_np.shape) != FRAME_SHAPE_NHWC:
        raise RuntimeError(f"unexpected output shapes codeword={codeword_np.shape} frame={frame_np.shape}")

    input_path = output_dir / "i_y_hat_nhwc.f32le"
    codeword_path = output_dir / "i_codeword_nhwc_expected.f32le"
    frame_path = output_dir / "encoder_i_reference_frame_nhwc_expected.f32le"
    write_f32(input_path, input_nhwc)
    write_f32(codeword_path, codeword_np)
    write_f32(frame_path, frame_np)

    script_path = output_dir / f"{MODEL_NAME}.pt"
    with torch.no_grad():
        scripted = torch.jit.trace(module, (sample,), strict=False)
        scripted.save(str(script_path))

    converter = find_converter(args.pytorch_converter)
    tflite_path = output_dir / f"{MODEL_NAME}.tflite"
    converter_log = output_dir / "logs" / f"{MODEL_NAME}_converter.log"
    converter_rc = run(
        [
            converter,
            "--input_script_module_file", str(script_path),
            "--output_file", str(tflite_path),
            "--input_shapes", ",".join(str(value) for value in INPUT_SHAPE_NHWC),
        ],
        converter_log,
    )

    record = {
        "name": MODEL_NAME,
        "route": "pytorch_torchscript_nhwc_source_direct",
        "source_root": str(source_root),
        "checkpoint_sha256": checkpoint_sha,
        "qp": args.qp,
        "input": {"path": str(input_path), "sha256": sha256(input_path), "shape": list(INPUT_SHAPE_NHWC), "dtype": "float32"},
        "outputs": [
            {"name": "i_codeword_nhwc", "path": str(codeword_path), "sha256": sha256(codeword_path), "shape": list(CODEWORD_SHAPE_NHWC), "dtype": "float32"},
            {"name": "encoder_i_reference_frame_nhwc", "path": str(frame_path), "sha256": sha256(frame_path), "shape": list(FRAME_SHAPE_NHWC), "dtype": "float32"},
        ],
        "script_module": str(script_path),
        "script_module_sha256": sha256(script_path),
        "converter": converter,
        "converter_rc": converter_rc,
        "converter_log": str(converter_log),
        "tflite": str(tflite_path) if tflite_path.is_file() else None,
        "tflite_sha256": sha256(tflite_path) if tflite_path.is_file() else None,
        "onnx_used": False,
    }
    manifest_path = output_dir / "i_recon_nhwc_manifest.json"
    manifest_path.write_text(json.dumps({"tool": Path(__file__).name, "record": record}, indent=2), encoding="utf-8")

    if args.copy_assets and converter_rc == 0 and tflite_path.is_file():
        assets = android_root / "app" / "src" / "main" / "assets" / "recon_i_nhwc"
        assets.mkdir(parents=True, exist_ok=True)
        for path in (tflite_path, input_path, codeword_path, frame_path, manifest_path):
            shutil.copy2(path, assets / path.name)

    print(f"wrote {manifest_path}")
    print(f"{MODEL_NAME} converter_rc={converter_rc} tflite={tflite_path if tflite_path.is_file() else 'missing'}")
    if converter_rc != 0:
        raise SystemExit(converter_rc)


if __name__ == "__main__":
    main()
