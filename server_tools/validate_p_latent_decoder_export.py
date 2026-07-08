#!/usr/bin/env python3
"""Validate P latent decoder export against server PyTorch trace.

Run on the Linux server only. This script compares one boundary:

    p_y_hat + p_ctx -> p_reference_feature

It reports PyTorch, ONNX Runtime, and TFLite outputs against
baseline/recon_p_segments/p_reference_feature.f32le so we can tell whether the
first recon mismatch starts at ONNX export or ONNX->TFLite conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_f32(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    data = np.fromfile(path, dtype="<f4")
    expected = int(np.prod(shape))
    if data.size != expected:
        raise ValueError(f"{path} has {data.size} floats, expected {expected} for {shape}")
    return data.reshape(shape)


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    diff = actual.astype(np.float64) - expected.astype(np.float64)
    abs_diff = np.abs(diff)
    return {
        "max_abs": float(abs_diff.max()) if abs_diff.size else 0.0,
        "mean_abs": float(abs_diff.mean()) if abs_diff.size else 0.0,
        "rmse": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
        "exact": bool(np.array_equal(actual, expected)),
    }


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


class PLatentDecoder(torch.nn.Module):
    def __init__(self, model, qp: int):
        super().__init__()
        self.dec = model.dec
        self.register_buffer("q_dec", model.q_scale_dec[qp : qp + 1].detach().clone())

    def forward(self, p_y_hat, p_ctx):
        return self.dec(p_y_hat, p_ctx, self.q_dec)


def run_pytorch(source_root: Path, qp: int, p_y_hat: np.ndarray, p_ctx: np.ndarray) -> tuple[np.ndarray, str]:
    model, device, ckpt = load_p_model(source_root)
    wrapper = PLatentDecoder(model, qp).to(device).eval()
    with torch.no_grad():
        y = torch.from_numpy(p_y_hat).to(device)
        ctx = torch.from_numpy(p_ctx).to(device)
        out = wrapper(y, ctx).detach().float().cpu().numpy()
    return out, sha256(ckpt)


def run_onnx(onnx_path: Path, p_y_hat: np.ndarray, p_ctx: np.ndarray) -> np.ndarray | None:
    try:
        import onnxruntime as ort
    except Exception as exc:
        print(f"[warn] onnxruntime unavailable: {exc}")
        return None
    if not onnx_path.is_file():
        print(f"[warn] missing ONNX: {onnx_path}")
        return None
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"p_y_hat": p_y_hat, "p_ctx": p_ctx})
    return outputs[0]


def run_tflite(tflite_path: Path, p_y_hat: np.ndarray, p_ctx: np.ndarray) -> np.ndarray | None:
    try:
        import tensorflow as tf
    except Exception as exc:
        print(f"[warn] tensorflow unavailable for TFLite: {exc}")
        return None
    if not tflite_path.is_file():
        print(f"[warn] missing TFLite: {tflite_path}")
        return None
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    input_by_name = {item["name"].split(":")[0]: item for item in inputs}
    for name, value in (("p_y_hat", p_y_hat), ("p_ctx", p_ctx)):
        detail = input_by_name.get(name)
        if detail is None:
            raise KeyError(f"TFLite input {name} not found; inputs={[i['name'] for i in inputs]}")
        interpreter.set_tensor(detail["index"], value.astype(np.float32, copy=False))
    interpreter.invoke()
    output = interpreter.get_output_details()[0]
    return interpreter.get_tensor(output["index"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--android-root", type=Path, default=Path.cwd())
    parser.add_argument("--qp", type=int, default=0)
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--tflite", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    android_root = args.android_root.resolve()
    assets = android_root / "app" / "src" / "main" / "assets"
    p_y_hat = read_f32(assets / "baseline" / "tensors" / "p_y_hat.f32le", (1, 128, 16, 32))
    p_ctx = read_f32(assets / "baseline" / "tensors" / "p_ctx.f32le", (1, 256, 32, 64))
    expected = read_f32(
        assets / "baseline" / "recon_p_segments" / "p_reference_feature.f32le",
        (1, 256, 32, 64),
    )

    onnx_path = args.onnx or (android_root / "outputs" / "recon_diagnostic" / "p_latent_decoder.onnx")
    tflite_path = args.tflite or (assets / "recon_diagnostic" / "p_latent_decoder_fp32.tflite")

    report: dict[str, Any] = {
        "source_root": str(args.source_root.resolve()),
        "android_root": str(android_root),
        "qp": args.qp,
        "onnx": str(onnx_path),
        "onnx_sha256": sha256(onnx_path) if onnx_path.is_file() else None,
        "tflite": str(tflite_path),
        "tflite_sha256": sha256(tflite_path) if tflite_path.is_file() else None,
        "expected_sha256": sha256(assets / "baseline" / "recon_p_segments" / "p_reference_feature.f32le"),
        "results": {},
    }

    pytorch_out, ckpt_sha = run_pytorch(args.source_root.resolve(), args.qp, p_y_hat, p_ctx)
    report["checkpoint_sha256"] = ckpt_sha
    report["results"]["pytorch"] = metrics(pytorch_out, expected)

    onnx_out = run_onnx(onnx_path, p_y_hat, p_ctx)
    if onnx_out is not None:
        report["results"]["onnx_vs_expected"] = metrics(onnx_out, expected)
        report["results"]["onnx_vs_pytorch"] = metrics(onnx_out, pytorch_out)

    tflite_out = run_tflite(tflite_path, p_y_hat, p_ctx)
    if tflite_out is not None:
        report["results"]["tflite_vs_expected"] = metrics(tflite_out, expected)
        report["results"]["tflite_vs_pytorch"] = metrics(tflite_out, pytorch_out)
        if onnx_out is not None:
            report["results"]["tflite_vs_onnx"] = metrics(tflite_out, onnx_out)

    out_json = args.out_json or (android_root / "outputs" / "p_latent_decoder_validate.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
