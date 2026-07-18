#!/usr/bin/env python3
"""Export I FeatureDec around the unsupported MTKEXT_SILU boundary.

Pipeline: NPU body -> exact native SiLU -> NPU tail. Run on the Linux server.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

from export_i_featuredec_nhwc_tflite import compare, read_f32, run, write_f32
from export_recon_diagnostic import PROJECT_ROOT, find_tool, load_i_model, sha256


INPUT_SHAPE_NCHW = (1, 256, 16, 32)
INPUT_SHAPE_NHWC = (1, 16, 32, 256)
BODY_SHAPE_NHWC = (1, 16, 32, 512)
OUTPUT_SHAPE_NHWC = (1, 16, 32, 18)
THRESHOLD = 5e-4


class IFeatureDecBodyNhwc(nn.Module):
    def __init__(self, model: nn.Module, qp: int) -> None:
        super().__init__()
        self.conv_in = model.dec.conv_in
        self.dec_1 = model.dec.dec_1
        self.register_buffer("q_dec", model.q_scale_dec[qp : qp + 1].detach().clone())

    def forward(self, i_y_hat_nhwc: torch.Tensor) -> torch.Tensor:
        value = i_y_hat_nhwc.permute(0, 3, 1, 2).contiguous()
        value = self.conv_in(value) * self.q_dec
        value = self.dec_1(value)
        return value.permute(0, 2, 3, 1).contiguous()


class IFeatureDecTailNhwc(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.conv_out = model.dec.conv_out

    def forward(self, activated_nhwc: torch.Tensor) -> torch.Tensor:
        value = activated_nhwc.permute(0, 3, 1, 2).contiguous()
        value = torch.clamp(self.conv_out(value), -1.0, 1.0)
        return value.permute(0, 2, 3, 1).contiguous()


def export_module(module: nn.Module, sample: torch.Tensor, name: str, output_dir: Path, converter: str) -> tuple[Path, Path, int]:
    script_path = output_dir / f"{name}.pt"
    with torch.no_grad():
        torch.jit.trace(module, (sample,), strict=False).save(str(script_path))
    tflite_path = output_dir / f"{name}.tflite"
    log_path = output_dir / "logs" / f"{name}_converter.log"
    rc = run(
        [
            converter,
            "--input_script_module_file", str(script_path),
            "--output_file", str(tflite_path),
            "--input_shapes", ",".join(str(value) for value in sample.shape),
        ],
        log_path,
    )
    return script_path, tflite_path, rc


def run_tflite(path: Path, value: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
    import mtk_converter

    outputs = mtk_converter.TFLiteExecutor(str(path)).run([value])
    if len(outputs) != 1:
        raise RuntimeError(f"{path.name}: expected one output, got {len(outputs)}")
    output = np.asarray(outputs[0], dtype=np.float32)
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(f"{path.name}: output shape {output.shape}, expected {expected_shape}")
    return output


def silu(value: np.ndarray) -> np.ndarray:
    return (value / (1.0 + np.exp(-value))).astype(np.float32)


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
    output_dir = (args.output_dir or android_root / "outputs" / "i_featuredec_split_nhwc").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = find_tool(args.pytorch_converter, "mtk_pytorch_converter")
    model, _device, checkpoint_sha = load_i_model(source_root)
    model = model.cpu().eval()
    body = IFeatureDecBodyNhwc(model, args.qp).eval()
    tail = IFeatureDecTailNhwc(model).eval()

    input_nchw = read_f32(args.i_y_hat_f32le.resolve(), INPUT_SHAPE_NCHW)
    input_nhwc = np.transpose(input_nchw, (0, 2, 3, 1)).copy()
    input_tensor = torch.from_numpy(input_nhwc)
    with torch.no_grad():
        body_expected_tensor = body(input_tensor)
        activated_expected_tensor = body_expected_tensor * torch.sigmoid(body_expected_tensor)
        output_expected_tensor = tail(activated_expected_tensor)
    body_expected = body_expected_tensor.numpy()
    activated_expected = activated_expected_tensor.numpy()
    output_expected = output_expected_tensor.numpy()

    body_script, body_tflite, body_rc = export_module(body, input_tensor, "i_featuredec_body_nhwc_fp32", output_dir, converter)
    tail_script, tail_tflite, tail_rc = export_module(tail, activated_expected_tensor, "i_featuredec_tail_nhwc_fp32", output_dir, converter)
    verification_error = None
    comparisons = None
    verification_passed = False
    if body_rc == 0 and tail_rc == 0 and body_tflite.is_file() and tail_tflite.is_file():
        try:
            body_actual = run_tflite(body_tflite, input_nhwc, BODY_SHAPE_NHWC)
            activated_actual = silu(body_actual)
            output_actual = run_tflite(tail_tflite, activated_actual, OUTPUT_SHAPE_NHWC)
            comparisons = {
                "body_tflite_vs_pytorch": compare(body_actual, body_expected),
                "activated_numpy_vs_pytorch": compare(activated_actual, activated_expected),
                "pipeline_tflite_vs_pytorch": compare(output_actual, output_expected),
            }
            verification_passed = all(item["max_abs"] <= THRESHOLD for item in comparisons.values())
            write_f32(output_dir / "i_codeword_nhwc_tflite_pipeline.f32le", output_actual)
        except Exception as error:
            verification_error = f"{error.__class__.__name__}: {error}"

    files = {
        "input": (output_dir / "i_y_hat_nhwc.f32le", input_nhwc),
        "body_expected": (output_dir / "i_featuredec_body_nhwc_expected.f32le", body_expected),
        "activated_expected": (output_dir / "i_featuredec_activated_nhwc_expected.f32le", activated_expected),
        "output_expected": (output_dir / "i_codeword_nhwc_expected.f32le", output_expected),
    }
    for path, value in files.values():
        write_f32(path, value)

    record = {
        "route": "source_direct_nhwc_split_at_silu",
        "checkpoint_sha256": checkpoint_sha,
        "qp": args.qp,
        "threshold": THRESHOLD,
        "body": {"tflite": str(body_tflite), "sha256": sha256(body_tflite) if body_tflite.is_file() else None, "converter_rc": body_rc, "input_shape": list(INPUT_SHAPE_NHWC), "output_shape": list(BODY_SHAPE_NHWC)},
        "native_boundary": "float32 x * sigmoid(x)",
        "tail": {"tflite": str(tail_tflite), "sha256": sha256(tail_tflite) if tail_tflite.is_file() else None, "converter_rc": tail_rc, "input_shape": list(BODY_SHAPE_NHWC), "output_shape": list(OUTPUT_SHAPE_NHWC)},
        "script_modules": {"body": str(body_script), "tail": str(tail_script)},
        "fixtures": {name: {"path": str(path), "sha256": sha256(path), "shape": list(value.shape)} for name, (path, value) in files.items()},
        "comparisons": comparisons,
        "verification_error": verification_error,
        "verification_passed": verification_passed,
        "onnx_used": False,
    }
    manifest = output_dir / "i_featuredec_split_nhwc_manifest.json"
    manifest.write_text(json.dumps({"tool": Path(__file__).name, "record": record}, indent=2), encoding="utf-8")

    if args.copy_assets:
        if not verification_passed:
            raise RuntimeError("refusing asset copy because split TFLite pipeline verification failed")
        assets = android_root / "app" / "src" / "main" / "assets" / "featuredec_i_split_nhwc"
        assets.mkdir(parents=True, exist_ok=True)
        for path in (body_tflite, tail_tflite, manifest, *(path for path, _value in files.values())):
            shutil.copy2(path, assets / path.name)

    print(f"wrote {manifest}")
    print(f"body_rc={body_rc} tail_rc={tail_rc} verification_passed={verification_passed}")
    if comparisons:
        for name, item in comparisons.items():
            print(f"{name} max_abs={item['max_abs']:.8g} mean_abs={item['mean_abs']:.8g} rmse={item['rmse']:.8g}")
    if body_rc != 0 or tail_rc != 0:
        raise SystemExit(1)
    if not verification_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
