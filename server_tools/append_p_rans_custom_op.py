#!/usr/bin/env python3
"""Append the native P-frame rANS encoder to p_entropy_prior_merged.tflite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CUSTOM_OP = "GVC_RT_P_RANS_ENCODE"
EXPECTED_OUTPUTS = (
    "p_z_hat", "p_y_q_w_0", "p_y_q_w_1", "p_s_w_0", "p_s_w_1", "p_y_hat",
)
EXPECTED_OUTPUT_SHAPES = (
    [1, 4, 8, 128],
    *[[1, 16, 32, 64] for _ in range(4)],
    [1, 16, 32, 128],
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_i32(root: Path, relative: str, shape=None) -> np.ndarray:
    value = np.fromfile(str(root / relative), dtype="<i4")
    return value.reshape(tuple(shape)) if shape is not None else value


def tensor_name(value) -> str:
    name = value.name
    return name.decode("utf-8") if isinstance(name, bytes) else str(name)


def append_rans(source: Path, output: Path, package_root: Path, manifest: dict, capacity: int) -> dict:
    import flatbuffers
    from tensorflow.lite.python import schema_py_generated as schema

    model_t = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(source.read_bytes(), 0))
    if len(model_t.subgraphs) != 1:
        raise RuntimeError("expected one TFLite subgraph")
    graph = model_t.subgraphs[0]
    if len(graph.inputs) != 2 or len(graph.outputs) != len(EXPECTED_OUTPUTS):
        raise RuntimeError("expected two inputs and {} outputs".format(len(EXPECTED_OUTPUTS)))
    actual_names = tuple(tensor_name(graph.tensors[index]) for index in graph.outputs)
    actual_shapes = tuple([int(value) for value in graph.tensors[index].shape] for index in graph.outputs)
    if actual_shapes != tuple(EXPECTED_OUTPUT_SHAPES):
        raise RuntimeError("P merged encoder output shape/order mismatch: {}".format(actual_shapes))

    p_info = manifest["p"]
    gaussian = p_info["gaussian"]
    z_info = p_info["z"]
    constants = (
        ("p_gaussian_cdf", load_i32(package_root, gaussian["cdf"], gaussian["shape"])),
        ("p_gaussian_cdf_lengths", load_i32(package_root, gaussian["cdf_lengths"])),
        ("p_gaussian_offsets", load_i32(package_root, gaussian["offsets"])),
        ("p_z_cdf", load_i32(package_root, z_info["cdf"], z_info["shape"])),
        ("p_z_cdf_lengths", load_i32(package_root, z_info["cdf_lengths"])),
        ("p_z_offsets", load_i32(package_root, z_info["offsets"])),
        ("p_z_start_offset", np.asarray([p_info["z_start_offset"]], dtype="<i4")),
        ("p_z_per_channel_size", np.asarray([p_info["z_per_channel_size"]], dtype="<i4")),
    )
    constant_indices = []
    for name, value in constants:
        array = np.ascontiguousarray(value, dtype="<i4")
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

    payload_t = schema.TensorT()
    payload_t.shape = [capacity]
    payload_t.type = int(schema.TensorType.UINT8)
    payload_t.buffer = 0
    payload_t.name = b"p_rans_payload_buffer"
    graph.tensors.append(payload_t)
    payload_index = len(graph.tensors) - 1
    size_t = schema.TensorT()
    size_t.shape = [1]
    size_t.type = int(schema.TensorType.INT32)
    size_t.buffer = 0
    size_t.name = b"p_rans_payload_size"
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
    operator_t.inputs = list(graph.outputs[:5]) + constant_indices
    operator_t.outputs = [payload_index, size_index]
    graph.operators.append(operator_t)
    graph.outputs = list(graph.outputs) + [payload_index, size_index]

    builder = flatbuffers.Builder(0)
    model_offset = model_t.Pack(builder)
    builder.Finish(model_offset, file_identifier=b"TFL3")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bytes(builder.Output()))
    return {
        "name": "p_entropy_prior_merged_rans",
        "file": output.name,
        "sha256": sha256(output),
        "source_model": str(source),
        "source_sha256": sha256(source),
        "custom_op": CUSTOM_OP,
        "custom_op_backend": "CPU",
        "input_names": ["p_y_pre_prior", "p_ctx_t"],
        "input_shapes_nhwc": [[1, 16, 32, 128], [1, 32, 64, 256]],
        "output_names": list(EXPECTED_OUTPUTS) + ["p_rans_payload_buffer", "p_rans_payload_size"],
        "source_tensor_names": list(actual_names),
        "source_output_shapes": [list(value) for value in actual_shapes],
        "payload_capacity_bytes": capacity,
        "z_start_offset": int(p_info["z_start_offset"]),
        "z_per_channel_size": int(p_info["z_per_channel_size"]),
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
    manifest = json.loads((package_root / "large_entropy_manifest.json").read_text(encoding="utf-8"))
    if "p" not in manifest:
        raise RuntimeError(
            "entropy package has no P CDF metadata; regenerate it with package_large_online_entropy.py"
        )
    record = append_rans(args.merged_model.resolve(), args.output.resolve(), package_root, manifest, args.payload_capacity)
    manifest_path = args.output.resolve().with_name("p_entropy_prior_merged_rans_manifest.json")
    manifest_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("wrote {}".format(args.output.resolve()))
    print("sha256={}".format(record["sha256"]))
    print("custom_op={}".format(CUSTOM_OP))
    print("manifest={}".format(manifest_path))


if __name__ == "__main__":
    main()
