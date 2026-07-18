#!/usr/bin/env python3
"""Export one I-frame spatial-prior stage as a direct FP32 NHWC TFLite graph.

This deliberately bypasses ONNX and mtk_onnx_converter.  It recreates the
source DepthConvBlock sequence from checkpoint weights, then verifies the
PyTorch, TensorFlow, and TFLite outputs on the exact server stage inputs.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from export_i_prior_npu_tflite import (
    HEIGHT,
    PROJECT_ROOT,
    WIDTH,
    Y_CHANNELS,
    check_ncc,
    find_ncc,
    load_model,
    sha256,
)


STAGE_OUTPUT_CHANNELS = Y_CHANNELS


def read_f32(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    value = np.fromfile(path, dtype="<f4")
    expected = int(np.prod(shape))
    if value.size != expected:
        raise ValueError(f"{path} has {value.size} floats; expected {expected} for {shape}")
    return value.reshape(shape)


def compare(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
    }


def create_tflite(concrete_function, output_path: Path) -> None:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function])
    converter.optimizations = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(converter.convert())


def conv2d_nhwc(tf, value, layer):
    weight = layer.weight.detach().cpu().numpy()
    bias = layer.bias.detach().cpu().numpy()
    if tuple(weight.shape[2:]) != (1, 1):
        raise ValueError(f"expected 1x1 Conv2d, got {weight.shape}")
    kernel = tf.constant(np.transpose(weight, (2, 3, 1, 0)).astype(np.float32))
    bias_tensor = tf.constant(bias.astype(np.float32))
    output = tf.nn.conv2d(value, kernel, strides=[1, 1, 1, 1], padding="VALID", data_format="NHWC")
    return tf.nn.bias_add(output, bias_tensor, data_format="NHWC")


def depthwise_conv3x3_nhwc(tf, value, layer):
    weight = layer.weight.detach().cpu().numpy()
    bias = layer.bias.detach().cpu().numpy()
    if weight.shape[1] != 1 or tuple(weight.shape[2:]) != (3, 3):
        raise ValueError(f"expected depthwise 3x3 Conv2d, got {weight.shape}")
    # PyTorch depthwise: [out_channels, 1, kh, kw]. TensorFlow: [kh, kw, in_channels, multiplier].
    kernel = tf.constant(np.transpose(weight, (2, 3, 0, 1)).astype(np.float32))
    bias_tensor = tf.constant(bias.astype(np.float32))
    padded = tf.pad(value, [[0, 0], [1, 1], [1, 1], [0, 0]])
    output = tf.nn.depthwise_conv2d(
        padded,
        kernel,
        strides=[1, 1, 1, 1],
        padding="VALID",
        data_format="NHWC",
    )
    return tf.nn.bias_add(output, bias_tensor, data_format="NHWC")


def wsilu(tf, value):
    return tf.math.sigmoid(4.0 * value) * value


def depth_conv_block(tf, value, block):
    if block.adaptor is not None:
        value = conv2d_nhwc(tf, value, block.adaptor)
    shortcut = value
    output = conv2d_nhwc(tf, value, block.dc[0])
    output = wsilu(tf, output)
    output = depthwise_conv3x3_nhwc(tf, output, block.dc[2])
    output = conv2d_nhwc(tf, output, block.dc[3])
    output = output + shortcut

    ffn = conv2d_nhwc(tf, output, block.ffn[0])
    first, second = tf.split(wsilu(tf, ffn), num_or_size_splits=2, axis=3)
    ffn = conv2d_nhwc(tf, first + second, block.ffn[2])
    output = ffn + output
    if block.shortcut:
        output = output + shortcut
    return output


def build_graph(model, stage: int):
    import tensorflow as tf

    adaptor = getattr(model, f"y_spatial_prior_adaptor_{stage}")
    spatial_blocks = list(model.y_spatial_prior[:-1])
    final_conv = model.y_spatial_prior[-1]
    if len(spatial_blocks) != 3:
        raise ValueError(f"expected three spatial DCBs, got {len(spatial_blocks)}")

    @tf.function(
        input_signature=[
            tf.TensorSpec([1, HEIGHT, WIDTH, Y_CHANNELS], tf.float32, name="i_y_hat_so_far_nhwc"),
            tf.TensorSpec([1, HEIGHT, WIDTH, Y_CHANNELS], tf.float32, name="i_reduced_common_params_nhwc"),
        ]
    )
    def graph(y_hat_so_far, reduced_common_params):
        output = tf.concat([y_hat_so_far, reduced_common_params], axis=3)
        output = depth_conv_block(tf, output, adaptor)
        for block in spatial_blocks:
            output = depth_conv_block(tf, output, block)
        output = conv2d_nhwc(tf, output, final_conv)
        scales, means = tf.split(output, num_or_size_splits=2, axis=3)
        return tf.identity(scales, name="i_scales_nhwc"), tf.identity(means, name="i_means_nhwc")

    return graph


def pytorch_reference(model, stage: int, y_hat_nchw: np.ndarray, reduced_nchw: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    import torch

    adaptor = getattr(model, f"y_spatial_prior_adaptor_{stage}")
    y_hat = torch.from_numpy(y_hat_nchw).to(device)
    reduced = torch.from_numpy(reduced_nchw).to(device)
    with torch.no_grad():
        scales, means = model.y_spatial_prior(adaptor(torch.cat((y_hat, reduced), dim=1))).chunk(2, 1)
    return scales.detach().cpu().numpy(), means.detach().cpu().numpy()


def tflite_output(path: Path, y_hat_nhwc: np.ndarray, reduced_nhwc: np.ndarray) -> list[tuple[str, np.ndarray]]:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 2 or len(output_details) != 2:
        raise RuntimeError(f"unexpected TFLite IO counts inputs={len(input_details)} outputs={len(output_details)}")
    for detail in input_details:
        name = detail["name"]
        if "y_hat" in name:
            value = y_hat_nhwc
        elif "reduced" in name:
            value = reduced_nhwc
        else:
            raise RuntimeError(f"unrecognized TFLite input name: {name}")
        interpreter.set_tensor(detail["index"], value)
    interpreter.invoke()
    return [(detail["name"], interpreter.get_tensor(detail["index"])) for detail in output_details]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--y-hat-f32le", type=Path, required=True)
    parser.add_argument("--reduced-f32le", type=Path, required=True)
    parser.add_argument("--expected-scales-f32le", type=Path, required=True)
    parser.add_argument("--expected-means-f32le", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "i_prior_direct_fp32")
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--ncc-tflite", default=None)
    parser.add_argument("--platform", default="mt6899")
    parser.add_argument("--arch", default="mdla5.3")
    parser.add_argument("--max-abs", type=float, default=5e-4)
    parser.add_argument("--copy-assets", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint, device = load_model(args.source_root.resolve(), force_zero_thres=0.12)
    y_hat_nchw = read_f32(args.y_hat_f32le.resolve(), (1, Y_CHANNELS, HEIGHT, WIDTH))
    reduced_nchw = read_f32(args.reduced_f32le.resolve(), (1, Y_CHANNELS, HEIGHT, WIDTH))
    expected_scales = read_f32(args.expected_scales_f32le.resolve(), (1, Y_CHANNELS, HEIGHT, WIDTH))
    expected_means = read_f32(args.expected_means_f32le.resolve(), (1, Y_CHANNELS, HEIGHT, WIDTH))
    y_hat_nhwc = np.transpose(y_hat_nchw, (0, 2, 3, 1)).copy()
    reduced_nhwc = np.transpose(reduced_nchw, (0, 2, 3, 1)).copy()

    graph = build_graph(model, args.stage)
    tf_outputs = graph(y_hat_nhwc, reduced_nhwc)
    tf_scales = np.transpose(tf_outputs[0].numpy(), (0, 3, 1, 2))
    tf_means = np.transpose(tf_outputs[1].numpy(), (0, 3, 1, 2))
    pytorch_scales, pytorch_means = pytorch_reference(model, args.stage, y_hat_nchw, reduced_nchw, device)

    name = f"i_prior_stage{args.stage}_direct_nhwc_fp32"
    tflite = output_dir / f"{name}.tflite"
    create_tflite(graph.get_concrete_function(), tflite)
    tflite_outputs = tflite_output(tflite, y_hat_nhwc, reduced_nhwc)
    raw_first = np.transpose(tflite_outputs[0][1], (0, 3, 1, 2))
    raw_second = np.transpose(tflite_outputs[1][1], (0, 3, 1, 2))
    direct_error = compare(raw_first, pytorch_scales)["mean_abs"] + compare(raw_second, pytorch_means)["mean_abs"]
    swapped_error = compare(raw_first, pytorch_means)["mean_abs"] + compare(raw_second, pytorch_scales)["mean_abs"]
    if swapped_error < direct_error:
        tflite_scales, tflite_means = raw_second, raw_first
        tflite_output_order = ["i_means_nhwc", "i_scales_nhwc"]
    else:
        tflite_scales, tflite_means = raw_first, raw_second
        tflite_output_order = ["i_scales_nhwc", "i_means_nhwc"]

    comparisons = {
        "pytorch_vs_expected_scales": compare(pytorch_scales, expected_scales),
        "pytorch_vs_expected_means": compare(pytorch_means, expected_means),
        "tensorflow_vs_pytorch_scales": compare(tf_scales, pytorch_scales),
        "tensorflow_vs_pytorch_means": compare(tf_means, pytorch_means),
        "tflite_vs_pytorch_scales": compare(tflite_scales, pytorch_scales),
        "tflite_vs_pytorch_means": compare(tflite_means, pytorch_means),
    }
    verification_passed = all(item["max_abs"] <= args.max_abs for item in comparisons.values())
    ncc = find_ncc(args.ncc_tflite, args.sdk_root, args.platform)
    record: dict[str, Any] = {
        "name": name,
        "path": str(tflite),
        "sha256": sha256(tflite),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "stage": args.stage,
        "layout": "NHWC",
        "input_shapes": [[1, HEIGHT, WIDTH, Y_CHANNELS], [1, HEIGHT, WIDTH, Y_CHANNELS]],
        "output_shapes": [[1, HEIGHT, WIDTH, STAGE_OUTPUT_CHANNELS], [1, HEIGHT, WIDTH, STAGE_OUTPUT_CHANNELS]],
        "dtype": "float32",
        "output_order": tflite_output_order,
        "raw_tflite_output_names": [name for name, _ in tflite_outputs],
        "max_abs_threshold": args.max_abs,
        "comparisons": comparisons,
        "verification_passed": verification_passed,
    }
    record.update(check_ncc(tflite, ncc, args.arch))

    if args.copy_assets:
        if not verification_passed:
            raise RuntimeError("refusing asset copy because direct FP32 verification failed")
        asset_dir = PROJECT_ROOT / "app" / "src" / "main" / "assets" / "prior_npu_diagnostic"
        asset_dir.mkdir(parents=True, exist_ok=True)
        copied = asset_dir / tflite.name
        shutil.copy2(tflite, copied)
        record["copied_asset"] = str(copied)
        record["copied_asset_sha256"] = sha256(copied)

    manifest = output_dir / f"{name}_manifest.json"
    manifest.write_text(json.dumps({"tool": Path(__file__).name, "record": record}, indent=2), encoding="utf-8")
    print(f"wrote {manifest}")
    for label, result in comparisons.items():
        print(f"{label} max_abs={result['max_abs']:.8g} mean_abs={result['mean_abs']:.8g} rmse={result['rmse']:.8g}")
    print(f"verification_passed={verification_passed} max_abs_threshold={args.max_abs}")
    print(f"ncc={record.get('ncc_check')} eligible={record.get('ncc_eligible')}")


if __name__ == "__main__":
    main()
