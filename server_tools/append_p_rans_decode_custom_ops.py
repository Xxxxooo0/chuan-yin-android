#!/usr/bin/env python3
"""Replace P entropy-decoder symbol placeholders with serial rANS custom ops."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


Z_CUSTOM_OP = "GVC_RT_P_RANS_DECODE_Z"
Y_CUSTOM_OP = "GVC_RT_P_RANS_DECODE_Y"
EXPECTED_INPUT_SHAPES = ([1, 4, 8, 128], [1, 16, 32, 64], [1, 16, 32, 64], [1, 32, 64, 256])
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


def add_constant(model_t, graph, schema, name: str, value: np.ndarray) -> int:
    array = np.ascontiguousarray(value, dtype="<i4")
    buffer_t = schema.BufferT()
    buffer_t.data = np.frombuffer(array.tobytes(), dtype=np.uint8)
    model_t.buffers.append(buffer_t)
    tensor_t = schema.TensorT()
    tensor_t.shape = list(array.shape)
    tensor_t.type = int(schema.TensorType.INT32)
    tensor_t.buffer = len(model_t.buffers) - 1
    tensor_t.name = name.encode("utf-8")
    graph.tensors.append(tensor_t)
    return len(graph.tensors) - 1


def add_input(graph, schema, name: str, shape: List[int], tensor_type: int) -> int:
    tensor_t = schema.TensorT()
    tensor_t.shape = shape
    tensor_t.type = tensor_type
    tensor_t.buffer = 0
    tensor_t.name = name.encode("utf-8")
    graph.tensors.append(tensor_t)
    return len(graph.tensors) - 1


def add_opcode(model_t, schema, name: str) -> int:
    opcode_t = schema.OperatorCodeT()
    opcode_t.builtinCode = int(schema.BuiltinOperator.CUSTOM)
    opcode_t.deprecatedBuiltinCode = int(schema.BuiltinOperator.CUSTOM)
    opcode_t.customCode = name.encode("utf-8")
    opcode_t.version = 1
    model_t.operatorCodes.append(opcode_t)
    return len(model_t.operatorCodes) - 1


def custom_operator(schema, opcode: int, inputs: List[int], outputs: List[int]):
    operator_t = schema.OperatorT()
    operator_t.opcodeIndex = opcode
    operator_t.inputs = inputs
    operator_t.outputs = outputs
    return operator_t


def append_decode_ops(source: Path, output: Path, package_root: Path, manifest: dict, capacity: int) -> dict:
    import flatbuffers
    from tensorflow.lite.python import schema_py_generated as schema

    model_t = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(source.read_bytes(), 0))
    if len(model_t.subgraphs) != 1:
        raise RuntimeError("expected one TFLite subgraph")
    graph = model_t.subgraphs[0]
    if len(graph.inputs) != 4 or len(graph.outputs) != 6:
        raise RuntimeError("expected four inputs and six outputs")
    input_indices = list(graph.inputs)
    input_shapes = tuple([int(value) for value in graph.tensors[index].shape] for index in input_indices)
    output_shapes = tuple([int(value) for value in graph.tensors[index].shape] for index in graph.outputs)
    if input_shapes != tuple(EXPECTED_INPUT_SHAPES):
        raise RuntimeError("P merged decoder input order mismatch: {}".format(input_shapes))
    if output_shapes != tuple(EXPECTED_OUTPUT_SHAPES):
        raise RuntimeError("P merged decoder output order mismatch: {}".format(output_shapes))

    p_info = manifest["p"]
    gaussian = p_info["gaussian"]
    z_info = p_info["z"]
    gaussian_constants = [
        add_constant(model_t, graph, schema, "p_decode_gaussian_cdf", load_i32(package_root, gaussian["cdf"], gaussian["shape"])),
        add_constant(model_t, graph, schema, "p_decode_gaussian_lengths", load_i32(package_root, gaussian["cdf_lengths"])),
        add_constant(model_t, graph, schema, "p_decode_gaussian_offsets", load_i32(package_root, gaussian["offsets"])),
    ]
    z_constants = [
        add_constant(model_t, graph, schema, "p_decode_z_cdf", load_i32(package_root, z_info["cdf"], z_info["shape"])),
        add_constant(model_t, graph, schema, "p_decode_z_lengths", load_i32(package_root, z_info["cdf_lengths"])),
        add_constant(model_t, graph, schema, "p_decode_z_offsets", load_i32(package_root, z_info["offsets"])),
        add_constant(model_t, graph, schema, "p_decode_z_start_offset", np.asarray([p_info["z_start_offset"]], dtype="<i4")),
        add_constant(model_t, graph, schema, "p_decode_z_per_channel_size", np.asarray([p_info["z_per_channel_size"]], dtype="<i4")),
    ]
    payload_index = add_input(graph, schema, "p_rans_payload_buffer", [capacity], int(schema.TensorType.UINT8))
    payload_size_index = add_input(graph, schema, "p_rans_payload_size", [1], int(schema.TensorType.INT32))
    graph.inputs = [payload_index, payload_size_index, input_indices[3]]

    z_opcode = add_opcode(model_t, schema, Z_CUSTOM_OP)
    y_opcode = add_opcode(model_t, schema, Y_CUSTOM_OP)
    z_op = custom_operator(schema, z_opcode, [payload_index, payload_size_index] + gaussian_constants + z_constants, [input_indices[0]])
    base_operators = list(graph.operators)
    producer: Dict[int, int] = {}
    consumers: Dict[int, List[int]] = {}
    for operator_index, operator in enumerate(base_operators):
        for tensor_index in operator.outputs:
            if tensor_index >= 0:
                producer[tensor_index] = operator_index
        for tensor_index in operator.inputs:
            if tensor_index >= 0:
                consumers.setdefault(tensor_index, []).append(operator_index)

    insertions: Dict[int, List[object]] = {0: [z_op]}
    insertion_records = []
    for stage in range(2):
        q_tensor = input_indices[stage + 1]
        scale_tensor = graph.outputs[stage + 3]
        q_consumers = consumers.get(q_tensor, [])
        scale_producer = producer.get(scale_tensor)
        if not q_consumers or scale_producer is None:
            raise RuntimeError("cannot locate P stage {} producer/consumer".format(stage))
        insertion = min(q_consumers)
        if scale_producer >= insertion:
            raise RuntimeError("P stage {} scale producer is not before symbol consumer".format(stage))
        y_op = custom_operator(schema, y_opcode, [scale_tensor] + gaussian_constants, [q_tensor])
        insertions.setdefault(insertion, []).append(y_op)
        insertion_records.append({
            "stage": stage,
            "operator_position": insertion,
            "scale_tensor": tensor_name(graph.tensors[scale_tensor]),
            "decoded_tensor": tensor_name(graph.tensors[q_tensor]),
        })

    rewritten = []
    for position, operator in enumerate(base_operators):
        rewritten.extend(insertions.get(position, []))
        rewritten.append(operator)
    rewritten.extend(insertions.get(len(base_operators), []))
    graph.operators = rewritten

    builder = flatbuffers.Builder(0)
    model_offset = model_t.Pack(builder)
    builder.Finish(model_offset, file_identifier=b"TFL3")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bytes(builder.Output()))
    return {
        "name": "p_entropy_decode_merged_rans",
        "file": output.name,
        "sha256": sha256(output),
        "source_model": str(source),
        "source_sha256": sha256(source),
        "custom_ops": [Z_CUSTOM_OP, Y_CUSTOM_OP],
        "custom_op_backend": "CPU",
        "input_names": ["p_rans_payload_buffer", "p_rans_payload_size", "p_ctx_t"],
        "input_shapes": [[capacity], [1], [1, 32, 64, 256]],
        "output_names": ["p_z_hat", "p_y_q_w_0", "p_y_q_w_1", "p_s_w_0", "p_s_w_1", "p_y_hat"],
        "output_shapes_nhwc": [list(value) for value in EXPECTED_OUTPUT_SHAPES],
        "payload_capacity_bytes": capacity,
        "z_start_offset": int(p_info["z_start_offset"]),
        "z_per_channel_size": int(p_info["z_per_channel_size"]),
        "cdf_embedded": True,
        "decode_insertions": insertion_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-base-model", type=Path, required=True)
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
    record = append_decode_ops(args.merged_base_model.resolve(), args.output.resolve(), package_root, manifest, args.payload_capacity)
    manifest_path = args.output.resolve().with_name("p_entropy_decode_merged_rans_manifest.json")
    manifest_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("wrote {}".format(args.output.resolve()))
    print("sha256={}".format(record["sha256"]))
    print("custom_ops={},{}".format(Z_CUSTOM_OP, Y_CUSTOM_OP))
    print("manifest={}".format(manifest_path))


if __name__ == "__main__":
    main()
