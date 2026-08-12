#!/usr/bin/env python3
"""Append the native I-frame rANS encoder to i_entropy_prior_merged.tflite.

The original ten graph outputs are preserved. Two outputs are appended:
`i_rans_payload_buffer` (fixed-capacity uint8) and `i_rans_payload_size` (int32).
CDF tables are embedded as constant tensors so Android needs no side-channel
configuration when registering the CPU custom op.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CUSTOM_OP = "GVC_RT_RANS_ENCODE"
EXPECTED_OUTPUTS = (
    "i_z_hat",
    "i_y_q_w_0",
    "i_y_q_w_1",
    "i_y_q_w_2",
    "i_y_q_w_3",
    "i_s_w_0",
    "i_s_w_1",
    "i_s_w_2",
    "i_s_w_3",
    "i_y_hat",
)
EXPECTED_OUTPUT_SHAPES = (
    [1, 4, 8, 128],
    *[[1, 16, 32, 64] for _ in range(8)],
    [1, 16, 32, 256],
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_i32(root: Path, relative: str, shape=None) -> np.ndarray:
    value = np.fromfile(str(root / relative), dtype="<i4")
    if shape is not None:
        value = value.reshape(tuple(shape))
    return value


def tensor_name(value) -> str:
    name = value.name
    return name.decode("utf-8") if isinstance(name, bytes) else str(name)


def append_rans(source: Path, output: Path, package_root: Path, manifest: dict, capacity: int) -> dict:
    import flatbuffers
    from tensorflow.lite.python import schema_py_generated as schema

    raw = source.read_bytes()
    model_t = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(raw, 0))
    if len(model_t.subgraphs) != 1:
        raise RuntimeError("expected one TFLite subgraph")
    graph = model_t.subgraphs[0]
    if len(graph.outputs) != len(EXPECTED_OUTPUTS):
        raise RuntimeError("expected {} outputs, got {}".format(len(EXPECTED_OUTPUTS), len(graph.outputs)))
    actual_names = tuple(tensor_name(graph.tensors[index]) for index in graph.outputs)
    actual_shapes = tuple(
        [int(dimension) for dimension in graph.tensors[index].shape]
        for index in graph.outputs
    )
    for index, (expected_shape, actual_shape) in enumerate(zip(EXPECTED_OUTPUT_SHAPES, actual_shapes)):
        if list(expected_shape) != list(actual_shape):
            raise RuntimeError(
                "output shape/order mismatch index={} logical_name={} tensor_name={} expected={} actual={}".format(
                    index, EXPECTED_OUTPUTS[index], actual_names[index], expected_shape, actual_shape
                )
            )

    i_info = manifest["i"]
    gaussian_info = i_info["gaussian"]
    z_info = i_info["z"]
    constants = (
        ("i_gaussian_cdf", load_i32(package_root, gaussian_info["cdf"], gaussian_info["shape"])),
        ("i_gaussian_cdf_lengths", load_i32(package_root, gaussian_info["cdf_lengths"])),
        ("i_gaussian_offsets", load_i32(package_root, gaussian_info["offsets"])),
        ("i_z_cdf", load_i32(package_root, z_info["cdf"], z_info["shape"])),
        ("i_z_cdf_lengths", load_i32(package_root, z_info["cdf_lengths"])),
        ("i_z_offsets", load_i32(package_root, z_info["offsets"])),
        ("i_z_start_offset", np.asarray([i_info["z_start_offset"]], dtype="<i4")),
        ("i_z_per_channel_size", np.asarray([i_info["z_per_channel_size"]], dtype="<i4")),
    )

    constant_indices = []
    for name, array in constants:
        array = np.ascontiguousarray(array, dtype="<i4")
        buffer_t = schema.BufferT()
        buffer_t.data = np.frombuffer(array.tobytes(), dtype=np.uint8)
        model_t.buffers.append(buffer_t)
        tensor_t = schema.TensorT()
        tensor_t.shape = list(array.shape)
        tensor_t.type = int(schema.TensorType.INT32)
        tensor_t.buffer = len(model_t.buffers) - 1
        tensor_t.name = name.encode("utf-8")
        tensor_t.isVariable = False
        graph.tensors.append(tensor_t)
        constant_indices.append(len(graph.tensors) - 1)

    empty_buffer = 0
    payload_t = schema.TensorT()
    payload_t.shape = [capacity]
    payload_t.type = int(schema.TensorType.UINT8)
    payload_t.buffer = empty_buffer
    payload_t.name = b"i_rans_payload_buffer"
    payload_t.isVariable = False
    graph.tensors.append(payload_t)
    payload_index = len(graph.tensors) - 1

    size_t = schema.TensorT()
    size_t.shape = [1]
    size_t.type = int(schema.TensorType.INT32)
    size_t.buffer = empty_buffer
    size_t.name = b"i_rans_payload_size"
    size_t.isVariable = False
    graph.tensors.append(size_t)
    size_index = len(graph.tensors) - 1

    opcode_t = schema.OperatorCodeT()
    opcode_t.builtinCode = int(schema.BuiltinOperator.CUSTOM)
    opcode_t.deprecatedBuiltinCode = int(schema.BuiltinOperator.CUSTOM)
    opcode_t.customCode = CUSTOM_OP.encode("utf-8")
    opcode_t.version = 1
    model_t.operatorCodes.append(opcode_t)

    operator_t = schema.OperatorT()
    operator_t.opcodeIndex = len(model_t.operatorCodes) - 1
    operator_t.inputs = list(graph.outputs[:9]) + constant_indices
    operator_t.outputs = [payload_index, size_index]
    graph.operators.append(operator_t)
    graph.outputs = list(graph.outputs) + [payload_index, size_index]

    builder = flatbuffers.Builder(0)
    model_offset = model_t.Pack(builder)
    builder.Finish(model_offset, file_identifier=b"TFL3")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bytes(builder.Output()))
    return {
        "name": "i_entropy_prior_merged_rans",
        "file": output.name,
        "sha256": sha256(output),
        "source_model": str(source),
        "source_sha256": sha256(source),
        "custom_op": CUSTOM_OP,
        "custom_op_backend": "CPU",
        "neuron_boundary": "merged entropy/prior outputs -> CPU rANS",
        "input_names": ["i_y_pre_prior"],
        "input_shapes_nhwc": [[1, 16, 32, 256]],
        "output_names": list(EXPECTED_OUTPUTS) + ["i_rans_payload_buffer", "i_rans_payload_size"],
        "source_tensor_names": list(actual_names),
        "source_output_shapes": [list(value) for value in actual_shapes],
        "payload_capacity_bytes": capacity,
        "z_start_offset": int(i_info["z_start_offset"]),
        "z_per_channel_size": int(i_info["z_per_channel_size"]),
        "cdf_embedded": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-model", type=Path, required=True)
    parser.add_argument("--entropy-package-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--payload-capacity", type=int, default=262144)
    args = parser.parse_args()
    if args.payload_capacity <= 0:
        raise ValueError("payload capacity must be positive")
    package_root = args.entropy_package_dir.resolve()
    manifest_path = package_root / "large_entropy_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = append_rans(
        args.merged_model.resolve(),
        args.output.resolve(),
        package_root,
        manifest,
        args.payload_capacity,
    )
    manifest_output = args.output.resolve().with_name("i_entropy_prior_merged_rans_manifest.json")
    manifest_output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("wrote {}".format(args.output.resolve()))
    print("sha256={}".format(record["sha256"]))
    print("custom_op={}".format(CUSTOM_OP))
    print("manifest={}".format(manifest_output))


if __name__ == "__main__":
    main()
