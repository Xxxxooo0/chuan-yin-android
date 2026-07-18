#!/usr/bin/env python3
"""Export I latent-decoder conv_in as an NHWC TorchScript TFLite model.

This is an Online Compile diagnostic asset. It follows the MediaTek PyTorch
NHWC wrapper example and intentionally bypasses the existing ONNX route.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import torch
from torch import nn

from export_recon_diagnostic import PROJECT_ROOT, find_tool, load_i_model, sha256


NAME = "i_latent_conv_in_nhwc_fp32"
INPUT_SHAPE = (1, 16, 32, 256)
OUTPUT_SHAPE = (1, 16, 32, 512)


class ILatentConvInNhwc(nn.Module):
    """NHWC boundary wrapper around the source I latent decoder front."""

    def __init__(self, model: nn.Module, qp: int) -> None:
        super().__init__()
        self.conv_in = model.dec.conv_in
        self.register_buffer("q_dec", model.q_scale_dec[qp : qp + 1].detach().clone())

    def forward(self, i_y_hat_nhwc: torch.Tensor) -> torch.Tensor:
        i_y_hat = i_y_hat_nhwc.permute(0, 3, 1, 2).contiguous()
        stage0 = self.conv_in(i_y_hat) * self.q_dec
        return stage0.permute(0, 2, 3, 1).contiguous()


def run(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True).returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--android-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pytorch-converter", default=None)
    parser.add_argument("--qp", type=int, default=0)
    parser.add_argument("--copy-assets", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    android_root = args.android_root.resolve()
    output_dir = (args.output_dir or android_root / "outputs" / "i_latent_conv_in_nhwc").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = find_tool(args.pytorch_converter, "mtk_pytorch_converter")

    model, _, checkpoint_sha256 = load_i_model(source_root)
    module = ILatentConvInNhwc(model.cpu().eval(), args.qp).cpu().eval()
    sample = torch.zeros(INPUT_SHAPE, dtype=torch.float32)

    script_path = output_dir / f"{NAME}.pt"
    with torch.no_grad():
        scripted = torch.jit.trace(module, (sample,), strict=False)
        actual_output_shape = list(scripted(sample).shape)
        scripted.save(str(script_path))

    tflite_path = output_dir / f"{NAME}.tflite"
    converter_log = output_dir / "logs" / f"{NAME}_converter.log"
    converter_rc = run(
        [
            converter,
            "--input_script_module_file", str(script_path),
            "--output_file", str(tflite_path),
            "--input_shapes", ",".join(str(value) for value in INPUT_SHAPE),
        ],
        converter_log,
    )

    record = {
        "name": NAME,
        "route": "pytorch_torchscript_nhwc",
        "source_root": str(source_root),
        "checkpoint_sha256": checkpoint_sha256,
        "qp": args.qp,
        "input_shape_nhwc": list(INPUT_SHAPE),
        "output_shape_nhwc": list(OUTPUT_SHAPE),
        "actual_torchscript_output_shape": actual_output_shape,
        "script_module": str(script_path),
        "script_module_sha256": sha256(script_path),
        "converter": converter,
        "converter_rc": converter_rc,
        "converter_log": str(converter_log),
        "tflite": str(tflite_path) if tflite_path.is_file() else None,
        "tflite_sha256": sha256(tflite_path) if tflite_path.is_file() else None,
        "onnx_used": False,
        "ncc_used": False,
    }
    manifest_path = output_dir / "i_latent_conv_in_nhwc_manifest.json"
    manifest_path.write_text(json.dumps({"tool": Path(__file__).name, "record": record}, indent=2), encoding="utf-8")

    if args.copy_assets and converter_rc == 0 and tflite_path.is_file():
        assets_dir = android_root / "app" / "src" / "main" / "assets" / "recon_diagnostic"
        assets_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tflite_path, assets_dir / tflite_path.name)
        shutil.copy2(manifest_path, assets_dir / manifest_path.name)

    print(f"wrote {manifest_path}")
    print(f"{NAME} converter_rc={converter_rc} tflite={tflite_path if tflite_path.is_file() else 'missing'}")
    if converter_rc != 0:
        raise SystemExit(converter_rc)


if __name__ == "__main__":
    main()
