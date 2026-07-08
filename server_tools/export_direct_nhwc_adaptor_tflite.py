#!/usr/bin/env python3
"""Export direct NHWC TFLite probes for the P recon MLP adaptor Conv2D.

Run this script only on the Linux server/TensorFlow/PyTorch environment. It
loads the real P checkpoint, extracts recon_generation_net.mlp[1].adaptor, and
creates TFLite Conv2D graphs without going through ONNX conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

from export_recon_diagnostic import load_p_model


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


def load_adaptor_weights(source_root: Path) -> tuple[np.ndarray, np.ndarray, str]:
    model, _device, checkpoint_sha = load_p_model(source_root)
    adaptor = model.recon_generation_net.mlp[1].adaptor
    weight = adaptor.weight.detach().cpu().numpy()
    bias = adaptor.bias.detach().cpu().numpy()
    # PyTorch Conv2d: [out_ch, in_ch, kh, kw].
    # TensorFlow Conv2D: [kh, kw, in_ch, out_ch].
    weight = np.transpose(weight, (2, 3, 1, 0))
    return weight, bias, checkpoint_sha


def export_nhwc_conv_fp32(source_root: Path, output_dir: Path) -> dict[str, object]:
    import tensorflow as tf

    weight, bias, checkpoint_sha = load_adaptor_weights(source_root)
    weight = tf.constant(weight.astype(np.float32))
    bias = tf.constant(bias.astype(np.float32))

    @tf.function(input_signature=[tf.TensorSpec([1, 16, 32, 1024], tf.float32, name="p_mlp_norm0_nhwc")])
    def model(x):
        y = tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding="VALID", data_format="NHWC")
        return {"p_dcb0_adapted_nhwc": tf.nn.bias_add(y, bias, data_format="NHWC")}

    name = "p_recon_mlp_dcb0_adaptor_nhwc_conv_fp32"
    path = output_dir / f"{name}.tflite"
    convert(model.get_concrete_function(), path)
    return {
        "name": name,
        "path": str(path),
        "sha256": sha256(path),
        "checkpoint_sha256": checkpoint_sha,
        "input_shape": [1, 16, 32, 1024],
        "output_shape": [1, 16, 32, 256],
        "layout": "NHWC",
        "dtype": "float32",
    }


def export_nhwc_conv_fp16(source_root: Path, output_dir: Path) -> dict[str, object]:
    import tensorflow as tf

    weight, bias, checkpoint_sha = load_adaptor_weights(source_root)
    weight = tf.constant(weight.astype(np.float16))
    bias = tf.constant(bias.astype(np.float16))

    @tf.function(input_signature=[tf.TensorSpec([1, 16, 32, 1024], tf.float16, name="p_mlp_norm0_nhwc")])
    def model(x):
        y = tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding="VALID", data_format="NHWC")
        return {"p_dcb0_adapted_nhwc": tf.nn.bias_add(y, bias, data_format="NHWC")}

    name = "p_recon_mlp_dcb0_adaptor_nhwc_conv_fp16"
    path = output_dir / f"{name}.tflite"
    convert(model.get_concrete_function(), path)
    return {
        "name": name,
        "path": str(path),
        "sha256": sha256(path),
        "checkpoint_sha256": checkpoint_sha,
        "input_shape": [1, 16, 32, 1024],
        "output_shape": [1, 16, 32, 256],
        "layout": "NHWC",
        "dtype": "float16",
    }


def export_flat_fc_fp32(source_root: Path, output_dir: Path) -> dict[str, object]:
    import tensorflow as tf

    weight, bias, checkpoint_sha = load_adaptor_weights(source_root)
    # TensorFlow matmul weight: [in_ch, out_ch].
    weight = tf.constant(weight[0, 0, :, :].astype(np.float32))
    bias = tf.constant(bias.astype(np.float32))

    @tf.function(input_signature=[tf.TensorSpec([512, 1024], tf.float32, name="p_mlp_norm0_flat")])
    def model(x):
        return {"p_dcb0_adapted_flat": tf.matmul(x, weight) + bias}

    name = "p_recon_mlp_dcb0_adaptor_flat_fc_fp32"
    path = output_dir / f"{name}.tflite"
    convert(model.get_concrete_function(), path)
    return {
        "name": name,
        "path": str(path),
        "sha256": sha256(path),
        "checkpoint_sha256": checkpoint_sha,
        "input_shape": [512, 1024],
        "output_shape": [512, 256],
        "layout": "flat",
        "dtype": "float32",
    }


def export_native_weights(source_root: Path, output_dir: Path) -> dict[str, object]:
    weight, bias, checkpoint_sha = load_adaptor_weights(source_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "p_recon_mlp_dcb0_adaptor_weights.bin"
    with path.open("wb") as handle:
        # Native fused kernel consumes row-major [out_ch, in_ch], followed by bias.
        native_weight = np.transpose(weight[0, 0, :, :], (1, 0)).astype(np.float32)
        handle.write(native_weight.tobytes())
        handle.write(bias.astype(np.float32).tobytes())
    return {
        "name": "p_recon_mlp_dcb0_adaptor_weights",
        "path": str(path),
        "sha256": sha256(path),
        "checkpoint_sha256": checkpoint_sha,
        "weight_shape": [256, 1024],
        "bias_shape": [256],
        "dtype": "float32",
    }


def export_native_adagn_weights(source_root: Path, output_dir: Path, attr_name: str) -> dict[str, object]:
    model, _device, checkpoint_sha = load_p_model(source_root)
    ada = getattr(model.recon_generation_net.decoder, attr_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"p_decoder_{attr_name}_weights.bin"
    tensors = [
        ada.gamma.weight.detach().cpu().numpy().astype(np.float32),
        ada.gamma.bias.detach().cpu().numpy().astype(np.float32),
        ada.beta.weight.detach().cpu().numpy().astype(np.float32),
        ada.beta.bias.detach().cpu().numpy().astype(np.float32),
    ]
    with path.open("wb") as handle:
        for tensor in tensors:
            handle.write(tensor.tobytes())
    out_channels = int(tensors[0].shape[0])
    return {
        "name": f"p_decoder_{attr_name}_weights",
        "path": str(path),
        "sha256": sha256(path),
        "checkpoint_sha256": checkpoint_sha,
        "gamma_weight_shape": [out_channels, 18],
        "gamma_bias_shape": [out_channels],
        "beta_weight_shape": [out_channels, 18],
        "beta_bias_shape": [out_channels],
        "dtype": "float32",
    }


def export_native_adagn_stage1_weights(source_root: Path, output_dir: Path) -> dict[str, object]:
    return export_native_adagn_weights(source_root, output_dir, "ada1")


def export_native_adagn_all_weights(source_root: Path, output_dir: Path) -> dict[str, object]:
    records = [
        export_native_adagn_weights(source_root, output_dir, attr_name)
        for attr_name in ("ada1", "ada2", "ada3", "ada4", "ada_final")
    ]
    return {
        "name": "p_decoder_adagn_all_weights",
        "records": records,
    }


EXPORTERS: dict[str, Callable[[Path, Path], dict[str, object]]] = {
    "fp32": export_nhwc_conv_fp32,
    "fp16": export_nhwc_conv_fp16,
    "flat_fc_fp32": export_flat_fc_fp32,
    "native_weights": export_native_weights,
    "native_adagn_stage1_weights": export_native_adagn_stage1_weights,
    "native_adagn_all_weights": export_native_adagn_all_weights,
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
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("app/src/main/assets/recon_diagnostic"))
    parser.add_argument(
        "--targets",
        default="all",
        help="all or comma list: fp32,fp16,flat_fc_fp32,native_weights,native_adagn_stage1_weights,native_adagn_all_weights",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/recon_diagnostic/direct_nhwc_adaptor_manifest.json"),
    )
    args = parser.parse_args()

    records = []
    for target in parse_targets(args.targets):
        record = EXPORTERS[target](args.source_root, args.output_dir)
        expanded = record.get("records", [record])
        records.extend(expanded)
        for item in expanded:
            print(f"wrote {item['path']} sha256={item['sha256']}")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    print(f"wrote {args.manifest}")


if __name__ == "__main__":
    main()
