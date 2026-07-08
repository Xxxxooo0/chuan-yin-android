#!/usr/bin/env python3
"""Export minimal TFLite SpaceToDepth probes without ONNX conversion.

Run this script only on the Linux server/TensorFlow environment. It directly
creates TFLite graphs with tf.nn.space_to_depth so we can test whether the MTK
TFLite/Neuron path supports the native SPACE_TO_DEPTH op.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(concrete_func, output_path: Path) -> None:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = []
    tflite_model = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)


def export_nhwc_space_to_depth(output_dir: Path) -> dict[str, object]:
    import tensorflow as tf

    @tf.function(input_signature=[tf.TensorSpec([1, 32, 64, 256], tf.float32, name="p_reference_feature_nhwc")])
    def model(x):
        return {"p_feature_unshuffled_nhwc": tf.nn.space_to_depth(x, block_size=2, data_format="NHWC")}

    name = "p_recon_unshuffle_tflite_spacetodepth_nhwc_fp32"
    path = output_dir / f"{name}.tflite"
    convert(model.get_concrete_function(), path)
    return {
        "name": name,
        "path": str(path),
        "sha256": sha256(path),
        "input_shape": [1, 32, 64, 256],
        "output_shape": [1, 16, 32, 1024],
        "layout": "NHWC",
    }


def export_nchw_wrapper_space_to_depth(output_dir: Path) -> dict[str, object]:
    import tensorflow as tf

    @tf.function(input_signature=[tf.TensorSpec([1, 256, 32, 64], tf.float32, name="p_reference_feature")])
    def model(x):
        nhwc = tf.transpose(x, [0, 2, 3, 1])
        y = tf.nn.space_to_depth(nhwc, block_size=2, data_format="NHWC")
        nchw = tf.transpose(y, [0, 3, 1, 2])
        return {"p_feature_unshuffled": nchw}

    name = "p_recon_unshuffle_tflite_spacetodepth_nchw_wrap_fp32"
    path = output_dir / f"{name}.tflite"
    convert(model.get_concrete_function(), path)
    return {
        "name": name,
        "path": str(path),
        "sha256": sha256(path),
        "input_shape": [1, 256, 32, 64],
        "output_shape": [1, 1024, 16, 32],
        "layout": "NCHW wrapper",
    }


def export_nhwc_space_to_depth_fp16(output_dir: Path) -> dict[str, object]:
    import tensorflow as tf

    @tf.function(input_signature=[tf.TensorSpec([1, 32, 64, 256], tf.float16, name="p_reference_feature_nhwc")])
    def model(x):
        return {"p_feature_unshuffled_nhwc": tf.nn.space_to_depth(x, block_size=2, data_format="NHWC")}

    name = "p_recon_unshuffle_tflite_spacetodepth_nhwc_fp16"
    path = output_dir / f"{name}.tflite"
    convert(model.get_concrete_function(), path)
    return {
        "name": name,
        "path": str(path),
        "sha256": sha256(path),
        "input_shape": [1, 32, 64, 256],
        "output_shape": [1, 16, 32, 1024],
        "layout": "NHWC",
        "dtype": "float16",
    }


def export_nchw_wrapper_space_to_depth_fp16(output_dir: Path) -> dict[str, object]:
    import tensorflow as tf

    @tf.function(input_signature=[tf.TensorSpec([1, 256, 32, 64], tf.float16, name="p_reference_feature")])
    def model(x):
        nhwc = tf.transpose(x, [0, 2, 3, 1])
        y = tf.nn.space_to_depth(nhwc, block_size=2, data_format="NHWC")
        nchw = tf.transpose(y, [0, 3, 1, 2])
        return {"p_feature_unshuffled": nchw}

    name = "p_recon_unshuffle_tflite_spacetodepth_nchw_wrap_fp16"
    path = output_dir / f"{name}.tflite"
    convert(model.get_concrete_function(), path)
    return {
        "name": name,
        "path": str(path),
        "sha256": sha256(path),
        "input_shape": [1, 256, 32, 64],
        "output_shape": [1, 1024, 16, 32],
        "layout": "NCHW wrapper",
        "dtype": "float16",
    }


def export_nhwc_space_to_depth_fp16_cast(output_dir: Path) -> dict[str, object]:
    import tensorflow as tf

    @tf.function(input_signature=[tf.TensorSpec([1, 32, 64, 256], tf.float16, name="p_reference_feature_nhwc")])
    def model(x):
        y = tf.nn.space_to_depth(tf.cast(x, tf.float32), block_size=2, data_format="NHWC")
        return {"p_feature_unshuffled_nhwc": tf.cast(y, tf.float16)}

    name = "p_recon_unshuffle_tflite_spacetodepth_nhwc_fp16_cast"
    path = output_dir / f"{name}.tflite"
    convert(model.get_concrete_function(), path)
    return {
        "name": name,
        "path": str(path),
        "sha256": sha256(path),
        "input_shape": [1, 32, 64, 256],
        "output_shape": [1, 16, 32, 1024],
        "layout": "NHWC",
        "dtype": "float16_io_float32_space_to_depth",
    }


def export_nchw_wrapper_space_to_depth_fp16_cast(output_dir: Path) -> dict[str, object]:
    import tensorflow as tf

    @tf.function(input_signature=[tf.TensorSpec([1, 256, 32, 64], tf.float16, name="p_reference_feature")])
    def model(x):
        x32 = tf.cast(x, tf.float32)
        nhwc = tf.transpose(x32, [0, 2, 3, 1])
        y = tf.nn.space_to_depth(nhwc, block_size=2, data_format="NHWC")
        nchw = tf.transpose(y, [0, 3, 1, 2])
        return {"p_feature_unshuffled": tf.cast(nchw, tf.float16)}

    name = "p_recon_unshuffle_tflite_spacetodepth_nchw_wrap_fp16_cast"
    path = output_dir / f"{name}.tflite"
    convert(model.get_concrete_function(), path)
    return {
        "name": name,
        "path": str(path),
        "sha256": sha256(path),
        "input_shape": [1, 256, 32, 64],
        "output_shape": [1, 1024, 16, 32],
        "layout": "NCHW wrapper",
        "dtype": "float16_io_float32_space_to_depth",
    }


def export_nhwc_space_to_depth_int8(output_dir: Path) -> dict[str, object]:
    import tensorflow as tf

    @tf.function(input_signature=[tf.TensorSpec([1, 32, 64, 256], tf.int8, name="p_reference_feature_nhwc")])
    def model(x):
        return {"p_feature_unshuffled_nhwc": tf.nn.space_to_depth(x, block_size=2, data_format="NHWC")}

    name = "p_recon_unshuffle_tflite_spacetodepth_nhwc_int8"
    path = output_dir / f"{name}.tflite"
    convert(model.get_concrete_function(), path)
    return {
        "name": name,
        "path": str(path),
        "sha256": sha256(path),
        "input_shape": [1, 32, 64, 256],
        "output_shape": [1, 16, 32, 1024],
        "layout": "NHWC",
        "dtype": "int8",
    }


EXPORTERS: dict[str, Callable[[Path], dict[str, object]]] = {
    "nhwc": export_nhwc_space_to_depth,
    "nchw_wrap": export_nchw_wrapper_space_to_depth,
    "nhwc_fp16": export_nhwc_space_to_depth_fp16,
    "nchw_wrap_fp16": export_nchw_wrapper_space_to_depth_fp16,
    "nhwc_fp16_cast": export_nhwc_space_to_depth_fp16_cast,
    "nchw_wrap_fp16_cast": export_nchw_wrapper_space_to_depth_fp16_cast,
    "nhwc_int8": export_nhwc_space_to_depth_int8,
}


def parse_targets(value: str) -> list[str]:
    if value == "all":
        return list(EXPORTERS)
    targets = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in targets if item not in EXPORTERS]
    if unknown:
        raise ValueError(f"unknown targets: {unknown}; valid={list(EXPORTERS)}")
    return targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("app/src/main/assets/recon_diagnostic"))
    parser.add_argument(
        "--targets",
        default="all",
        help="all or comma list: nhwc,nchw_wrap,nhwc_fp16,nchw_wrap_fp16,nhwc_fp16_cast,nchw_wrap_fp16_cast,nhwc_int8",
    )
    parser.add_argument("--manifest", type=Path, default=Path("outputs/recon_diagnostic/direct_spacetodepth_manifest.json"))
    args = parser.parse_args()

    records = []
    for target in parse_targets(args.targets):
        record = EXPORTERS[target](args.output_dir)
        records.append(record)
        print(f"wrote {record['path']} sha256={record['sha256']}")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    print(f"wrote {args.manifest}")


if __name__ == "__main__":
    main()
