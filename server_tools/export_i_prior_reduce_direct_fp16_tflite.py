#!/usr/bin/env python3
"""Export the I-prior reduction as a direct NHWC floating-point TFLite graph.

This is an NPU feasibility probe only. It extracts the exact 1x1 reduction
weights from GVC-RT_B_I.pt, but bypasses ONNX and the PyTorch converter so the
TFLite model is a single Conv2D plus BiasAdd. FP16 external tensors require
unsupported CAST operations on MDLA5.3, so the default probe is pure FP32.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from export_i_prior_npu_tflite import (
    COMMON_CHANNELS,
    HEIGHT,
    PROJECT_ROOT,
    WIDTH,
    check_ncc,
    find_ncc,
    load_model,
    sha256,
)


REDUCED_CHANNELS = 256


def convert(concrete_function, output_path: Path) -> None:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function])
    converter.optimizations = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(converter.convert())


def export_model(source_root: Path, output_path: Path, io_dtype: str) -> tuple[Path, str]:
    import tensorflow as tf

    model, checkpoint, _device = load_model(source_root, force_zero_thres=0.12)
    reduction = model.y_spatial_prior_reduction
    weight = reduction.weight.detach().cpu().numpy()
    bias = reduction.bias.detach().cpu().numpy()
    if tuple(weight.shape) != (REDUCED_CHANNELS, COMMON_CHANNELS, 1, 1):
        raise RuntimeError(f"unexpected I prior reduction weight shape: {weight.shape}")

    # PyTorch: [out, in, kh, kw]. TensorFlow: [kh, kw, in, out].
    dtype = tf.float16 if io_dtype == "fp16" else tf.float32
    tf_weight = tf.constant(np.transpose(weight, (2, 3, 1, 0)).astype(np.float32))
    tf_bias = tf.constant(bias.astype(np.float32))

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                [1, HEIGHT, WIDTH, COMMON_CHANNELS],
                dtype,
                name="i_common_params_nhwc",
            )
        ]
    )
    def reduction_graph(i_common_params_nhwc):
        input_tensor = tf.cast(i_common_params_nhwc, tf.float32) if io_dtype == "fp16" else i_common_params_nhwc
        result = tf.nn.conv2d(
            input_tensor,
            tf_weight,
            strides=[1, 1, 1, 1],
            padding="VALID",
            data_format="NHWC",
        )
        result = tf.nn.bias_add(result, tf_bias, data_format="NHWC")
        output_tensor = tf.cast(result, tf.float16) if io_dtype == "fp16" else result
        return {"i_reduced_common_params_nhwc": output_tensor}

    convert(reduction_graph.get_concrete_function(), output_path)
    return checkpoint, sha256(checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "i_prior_reduce_fp16")
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--ncc-tflite", default=None)
    parser.add_argument("--platform", default="mt6899")
    parser.add_argument("--arch", default="mdla5.3")
    parser.add_argument("--io-dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--copy-assets", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    name = f"i_prior_reduce_direct_nhwc_{args.io_dtype}"
    path = output_dir / f"{name}.tflite"
    checkpoint, checkpoint_sha = export_model(args.source_root.resolve(), path, args.io_dtype)
    ncc = find_ncc(args.ncc_tflite, args.sdk_root, args.platform)

    record: dict[str, Any] = {
        "name": name,
        "path": str(path),
        "sha256": sha256(path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "layout": "NHWC",
        "input_shape": [1, HEIGHT, WIDTH, COMMON_CHANNELS],
        "output_shape": [1, HEIGHT, WIDTH, REDUCED_CHANNELS],
        "input_dtype": args.io_dtype,
        "output_dtype": args.io_dtype,
        "operation": (
            "Conv2D(1x1)+BiasAdd"
            if args.io_dtype == "fp32"
            else "Cast(FP16->FP32)+Conv2D(1x1)+BiasAdd+Cast(FP32->FP16)"
        ),
    }
    record.update(check_ncc(path, ncc, args.arch))

    if args.copy_assets:
        asset_dir = PROJECT_ROOT / "app" / "src" / "main" / "assets" / "prior_npu_diagnostic"
        asset_dir.mkdir(parents=True, exist_ok=True)
        copied = asset_dir / path.name
        shutil.copy2(path, copied)
        record["copied_asset"] = str(copied)
        record["copied_asset_sha256"] = sha256(copied)

    manifest = output_dir / f"i_prior_reduce_direct_{args.io_dtype}_manifest.json"
    manifest.write_text(json.dumps({"tool": Path(__file__).name, "record": record}, indent=2), encoding="utf-8")
    print(f"wrote {manifest}")
    print(
        f"{name} check_rc={record.get('check_target_rc')} plan_rc={record.get('exec_plan_rc')} "
        f"ncc={record.get('ncc_check')} eligible={record.get('ncc_eligible')}"
    )


if __name__ == "__main__":
    main()
