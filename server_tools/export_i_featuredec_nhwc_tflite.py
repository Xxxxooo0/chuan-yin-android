#!/usr/bin/env python3
"""Export the complete I-frame feature decoder as a source-derived NHWC TFLite graph.

Run on the Linux server only. The graph is loaded from GVC-RT_B_I.pt and does
not reuse the legacy Android ONNX or TFLite assets.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from export_recon_diagnostic import PROJECT_ROOT, find_tool, load_i_model, sha256


NAME = "i_featuredec_nhwc_fp32"
INPUT_SHAPE_NCHW = (1, 256, 16, 32)
INPUT_SHAPE_NHWC = (1, 16, 32, 256)
OUTPUT_SHAPE_NHWC = (1, 16, 32, 18)
MAX_ABS_THRESHOLD = 5e-4


class IFeatureDecNhwc(nn.Module):
    def __init__(self, model: nn.Module, qp: int) -> None:
        super().__init__()
        self.dec = model.dec
        self.register_buffer("q_dec", model.q_scale_dec[qp : qp + 1].detach().clone())

    def forward(self, i_y_hat_nhwc: torch.Tensor) -> torch.Tensor:
        i_y_hat = i_y_hat_nhwc.permute(0, 3, 1, 2).contiguous()
        codeword = self.dec(i_y_hat, self.q_dec)
        return codeword.permute(0, 2, 3, 1).contiguous()


def read_f32(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    expected = int(np.prod(shape))
    if values.size != expected:
        raise ValueError(f"{path} contains {values.size} floats, expected {expected} for {shape}")
    return values.reshape(shape)


def write_f32(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values.astype("<f4", copy=False).tofile(path)


def compare(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
    }


def run(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True).returncode


def run_tflite(tflite: Path, input_nhwc: np.ndarray) -> np.ndarray:
    import mtk_converter

    outputs = mtk_converter.TFLiteExecutor(str(tflite)).run([input_nhwc])
    if len(outputs) != 1:
        raise RuntimeError(f"expected one TFLite output, got {len(outputs)}")
    output = np.asarray(outputs[0], dtype=np.float32)
    if tuple(output.shape) != OUTPUT_SHAPE_NHWC:
        raise RuntimeError(f"unexpected TFLite output shape {output.shape}, expected {OUTPUT_SHAPE_NHWC}")
    return output


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
    output_dir = (args.output_dir or android_root / "outputs" / "i_featuredec_nhwc").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = find_tool(args.pytorch_converter, "mtk_pytorch_converter")

    model, _device, checkpoint_sha = load_i_model(source_root)
    module = IFeatureDecNhwc(model.cpu().eval(), args.qp).cpu().eval()
    input_nchw = read_f32(args.i_y_hat_f32le.resolve(), INPUT_SHAPE_NCHW)
    input_nhwc = np.transpose(input_nchw, (0, 2, 3, 1)).copy()
    sample = torch.from_numpy(input_nhwc)
    with torch.no_grad():
        expected = module(sample).cpu().numpy()
    if tuple(expected.shape) != OUTPUT_SHAPE_NHWC:
        raise RuntimeError(f"unexpected PyTorch output shape {expected.shape}, expected {OUTPUT_SHAPE_NHWC}")

    input_path = output_dir / "i_y_hat_nhwc.f32le"
    expected_path = output_dir / "i_codeword_nhwc_expected.f32le"
    write_f32(input_path, input_nhwc)
    write_f32(expected_path, expected)

    script_path = output_dir / f"{NAME}.pt"
    with torch.no_grad():
        scripted = torch.jit.trace(module, (sample,), strict=False)
        scripted.save(str(script_path))

    tflite_path = output_dir / f"{NAME}.tflite"
    converter_log = output_dir / "logs" / f"{NAME}_converter.log"
    converter_rc = run(
        [
            converter,
            "--input_script_module_file", str(script_path),
            "--output_file", str(tflite_path),
            "--input_shapes", ",".join(str(value) for value in INPUT_SHAPE_NHWC),
        ],
        converter_log,
    )

    comparison = None
    verification_passed = False
    tflite_output_path = output_dir / "i_codeword_nhwc_tflite.f32le"
    tflite_error = None
    if converter_rc == 0 and tflite_path.is_file():
        try:
            tflite_output = run_tflite(tflite_path, input_nhwc)
            comparison = compare(tflite_output, expected)
            verification_passed = comparison["max_abs"] <= MAX_ABS_THRESHOLD
            write_f32(tflite_output_path, tflite_output)
        except Exception as error:
            tflite_error = f"{error.__class__.__name__}: {error}"

    record = {
        "name": NAME,
        "route": "pytorch_torchscript_nhwc_source_direct",
        "source_root": str(source_root),
        "checkpoint_sha256": checkpoint_sha,
        "qp": args.qp,
        "input": {"path": str(input_path), "sha256": sha256(input_path), "shape": list(INPUT_SHAPE_NHWC), "dtype": "float32"},
        "expected_output": {"path": str(expected_path), "sha256": sha256(expected_path), "shape": list(OUTPUT_SHAPE_NHWC), "dtype": "float32"},
        "script_module": str(script_path),
        "script_module_sha256": sha256(script_path),
        "converter": converter,
        "converter_rc": converter_rc,
        "converter_log": str(converter_log),
        "tflite": str(tflite_path) if tflite_path.is_file() else None,
        "tflite_sha256": sha256(tflite_path) if tflite_path.is_file() else None,
        "tflite_output": str(tflite_output_path) if tflite_output_path.is_file() else None,
        "tflite_vs_pytorch": comparison,
        "tflite_error": tflite_error,
        "max_abs_threshold": MAX_ABS_THRESHOLD,
        "verification_passed": verification_passed,
        "onnx_used": False,
    }
    manifest_path = output_dir / "i_featuredec_nhwc_manifest.json"
    manifest_path.write_text(json.dumps({"tool": Path(__file__).name, "record": record}, indent=2), encoding="utf-8")

    if args.copy_assets:
        if not verification_passed:
            raise RuntimeError("refusing asset copy because TFLite vs PyTorch verification did not pass")
        assets = android_root / "app" / "src" / "main" / "assets" / "featuredec_i_nhwc"
        assets.mkdir(parents=True, exist_ok=True)
        for path in (tflite_path, input_path, expected_path, manifest_path):
            shutil.copy2(path, assets / path.name)

    print(f"wrote {manifest_path}")
    print(f"{NAME} converter_rc={converter_rc} verification_passed={verification_passed}")
    if comparison:
        print(
            "tflite_vs_pytorch "
            f"max_abs={comparison['max_abs']:.8g} mean_abs={comparison['mean_abs']:.8g} rmse={comparison['rmse']:.8g}"
        )
    if converter_rc != 0:
        raise SystemExit(converter_rc)
    if not verification_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
