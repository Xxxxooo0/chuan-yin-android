#!/usr/bin/env python3
"""Export the first I latent-decoder Conv2D with NHWC FP16 external I/O.

This is a minimal Neuron Online Compile reproduction for the first failing
operator. It deliberately exports only ``model.dec.conv_in.adaptor`` and does
not change the production recon graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

print("[op0-fp16] importing numpy", flush=True)
import numpy as np
print("[op0-fp16] importing tensorflow", flush=True)
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]


NAME = "i_latent_op0_nhwc_fp16"
INPUT_SHAPE = (1, 16, 32, 256)
OUTPUT_SHAPE = (1, 16, 32, 512)
INPUT_ASSET = "baseline/tensors/i_y_hat.f32le"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nchw_to_nhwc(data: np.ndarray) -> np.ndarray:
    return np.transpose(data, (0, 2, 3, 1)).copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--android-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--qp", type=int, default=0)
    parser.add_argument("--copy-assets", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    android_root = args.android_root.resolve()
    output_dir = (args.output_dir or android_root / "outputs" / "i_latent_op0_fp16").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[op0-fp16] output_dir={output_dir}", flush=True)

    checkpoint_path = source_root / "ckpt" / "checkpoints" / "GVC-RT_B_I.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing I checkpoint: {checkpoint_path}")

    input_path = android_root / "app" / "src" / "main" / "assets" / INPUT_ASSET
    if not input_path.is_file():
        raise FileNotFoundError(f"missing baseline input: {input_path}")
    weights_path = output_dir / "op0_weights.npz"
    reference_path = output_dir / "i_op0_output_nhwc.f32le"
    extractor = Path(__file__).with_name("extract_i_latent_op0_weights.py")
    print("[op0-fp16] launching torch-only weight extractor", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(extractor),
            "--checkpoint", str(checkpoint_path),
            "--input-nchw-f32le", str(input_path),
            "--weights-output", str(weights_path),
            "--reference-output-nhwc-f32le", str(reference_path),
        ],
        check=True,
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    with np.load(weights_path, allow_pickle=False) as values:
        adaptor_weight = values["weight"]
        adaptor_bias = values["bias"]
    input_nchw = np.fromfile(input_path, dtype="<f4").reshape((1, 256, 16, 32))
    input_nhwc_f16 = nchw_to_nhwc(input_nchw).astype(np.float16)
    print("[op0-fp16] baseline input prepared", flush=True)

    # TFLite Conv2D filter layout is [H, W, input_channel, output_channel].
    weight_fp32 = np.transpose(adaptor_weight, (2, 3, 1, 0))
    bias_fp32 = adaptor_bias
    weight = weight_fp32.astype(np.float16)
    bias = bias_fp32.astype(np.float16)
    tf_weight = tf.constant(weight, dtype=tf.float16)
    tf_bias = tf.constant(bias, dtype=tf.float16)
    print("[op0-fp16] TensorFlow Conv2D graph prepared", flush=True)

    @tf.function(input_signature=[tf.TensorSpec(INPUT_SHAPE, tf.float16, name="i_y_hat_nhwc")])
    def conv_op(i_y_hat_nhwc: tf.Tensor) -> tf.Tensor:
        output = tf.nn.conv2d(i_y_hat_nhwc, tf_weight, strides=1, padding="VALID", data_format="NHWC")
        return tf.nn.bias_add(output, tf_bias, data_format="NHWC", name="i_op0_output_nhwc")

    concrete = conv_op.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
    tflite_path = output_dir / f"{NAME}.tflite"
    print("[op0-fp16] converting TFLite", flush=True)
    tflite_path.write_bytes(converter.convert())
    print(f"[op0-fp16] TFLite written={tflite_path}", flush=True)

    print("[op0-fp16] validating TFLite external dtypes", flush=True)
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if input_detail["dtype"] != np.float16 or output_detail["dtype"] != np.float16:
        raise RuntimeError(
            "expected FP16 external I/O, got "
            f"input={input_detail['dtype']} output={output_detail['dtype']}"
        )
    interpreter.set_tensor(input_detail["index"], input_nhwc_f16)
    interpreter.invoke()
    output_nhwc_f16 = interpreter.get_tensor(output_detail["index"])
    if tuple(output_nhwc_f16.shape) != OUTPUT_SHAPE:
        raise RuntimeError(f"unexpected TFLite output shape: {tuple(output_nhwc_f16.shape)}")

    print("[op0-fp16] validating TFLite output against PyTorch", flush=True)
    reference_output_nhwc = np.fromfile(reference_path, dtype="<f4").reshape(OUTPUT_SHAPE)
    tflite_output_f32 = output_nhwc_f16.astype(np.float32)
    diff = np.abs(tflite_output_f32 - reference_output_nhwc)

    input_f16_path = output_dir / "i_y_hat_nhwc.f16le"
    expected_f16_path = output_dir / "i_op0_output_nhwc.f16le"
    input_nhwc_f16.astype("<f2").tofile(input_f16_path)
    output_nhwc_f16.astype("<f2").tofile(expected_f16_path)
    print("[op0-fp16] FP16 reference tensors written", flush=True)

    record = {
        "name": NAME,
        "route": "tensorflow_direct_conv2d_nhwc_fp16_io",
        "source_root": str(source_root),
        "checkpoint_sha256": checkpoint_sha256,
        "qp": args.qp,
        "input_shape_nhwc": list(INPUT_SHAPE),
        "output_shape_nhwc": list(OUTPUT_SHAPE),
        "input_dtype": "float16",
        "output_dtype": "float16",
        "operation": "DepthConvBlock.adaptor Conv2D(1x1,256->512)+BiasAdd",
        "tflite": str(tflite_path),
        "tflite_sha256": sha256_file(tflite_path),
        "input_f16": str(input_f16_path),
        "input_f16_sha256": sha256_bytes(input_f16_path.read_bytes()),
        "expected_output_f16": str(expected_f16_path),
        "expected_output_f16_sha256": sha256_bytes(expected_f16_path.read_bytes()),
        "tflite_vs_pytorch_fp32": {
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "rmse": float(np.sqrt(np.mean(np.square(diff)))),
            "passed": bool(diff.max() <= 5e-3),
            "max_abs_threshold": 5e-3,
        },
        "input_tensor_dtype": str(input_detail["dtype"]),
        "output_tensor_dtype": str(output_detail["dtype"]),
    }
    manifest_path = output_dir / f"{NAME}_manifest.json"
    manifest_path.write_text(json.dumps({"tool": Path(__file__).name, "record": record}, indent=2), encoding="utf-8")

    if args.copy_assets:
        print("[op0-fp16] copying Android assets", flush=True)
        assets_dir = android_root / "app" / "src" / "main" / "assets" / "recon_fp16_i_latent"
        assets_dir.mkdir(parents=True, exist_ok=True)
        for path in (tflite_path, input_f16_path, expected_f16_path, manifest_path):
            shutil.copy2(path, assets_dir / path.name)

    print(f"wrote {manifest_path}")
    print(
        f"{NAME} input_dtype={input_detail['dtype']} output_dtype={output_detail['dtype']} "
        f"max_abs={diff.max():.8f} passed={diff.max() <= 5e-3}",
    )


if __name__ == "__main__":
    main()
