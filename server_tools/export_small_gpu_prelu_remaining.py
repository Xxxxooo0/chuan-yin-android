#!/usr/bin/env python3
"""Export and validate only the five remaining Small GPU PReLU candidates."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema

from export_small_temporal_gpu_prelu import checked_destination, interpreter, pack, save, sha256


MODEL_NAMES = (
    "temporal_from_feature",
    "encoder",
    "decoder",
    "entropy_encode_fused",
    "entropy_decode_fused",
)


def builtin_code(model, operator):
    code = model.operatorCodes[operator.opcodeIndex]
    return int(code.builtinCode)


def custom_name(model, operator):
    code = model.operatorCodes[operator.opcodeIndex]
    if builtin_code(model, operator) != schema.BuiltinOperator.CUSTOM:
        return None
    return code.customCode.decode() if isinstance(code.customCode, bytes) else str(code.customCode)


def expand_scalar_prelu(source):
    candidate = copy.deepcopy(source)
    graph = candidate.subgraphs[0]
    constants = {}
    changes = []
    for index, operator in enumerate(graph.operators):
        if builtin_code(candidate, operator) != schema.BuiltinOperator.PRELU:
            continue
        assert len(operator.inputs) == 2
        x_index, alpha_index = map(int, operator.inputs)
        x, alpha = graph.tensors[x_index], graph.tensors[alpha_index]
        assert x.type == alpha.type == schema.TensorType.FLOAT32
        assert len(x.shape) == 4 and list(alpha.shape) == [1, 1, 1, 1]
        channels = int(x.shape[-1])
        assert channels > 1
        assert all(alpha_index not in node.outputs for node in graph.operators)
        scalar_bytes = bytes(candidate.buffers[alpha.buffer].data)
        assert len(scalar_bytes) == 4
        key = (alpha_index, channels)
        if key not in constants:
            buffer = schema.BufferT()
            buffer.data = np.frombuffer(scalar_bytes * channels, dtype=np.uint8).copy()
            expanded = copy.deepcopy(alpha)
            expanded.name = alpha.name + ("_gpu_c%d" % channels).encode()
            expanded.shape = np.array([1, 1, 1, channels], dtype=np.int32)
            assert expanded.shapeSignature is None
            expanded.buffer = len(candidate.buffers)
            candidate.buffers.append(buffer)
            constants[key] = len(graph.tensors)
            graph.tensors.append(expanded)
        operator.inputs = operator.inputs.copy()
        operator.inputs[1] = constants[key]
        changes.append({"node": index, "old_alpha_tensor": alpha_index,
                        "new_alpha_tensor": int(operator.inputs[1]), "channels": channels,
                        "alpha": float(np.frombuffer(scalar_bytes, dtype="<f4")[0]),
                        "old_shape": [1, 1, 1, 1], "new_shape": [1, 1, 1, channels]})
    assert changes, "Expected scalar PReLU nodes in every selected Small graph"
    return candidate, changes, len(constants)


def assert_only_prelu_changes(source, rewritten, changes):
    restored = copy.deepcopy(schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(rewritten, 0)))
    for change in changes:
        operator = restored.subgraphs[0].operators[change["node"]]
        operator.inputs = operator.inputs.copy()
        operator.inputs[1] = change["old_alpha_tensor"]
    del restored.subgraphs[0].tensors[len(source.subgraphs[0].tensors):]
    del restored.buffers[len(source.buffers):]
    assert pack(restored) == pack(source), "Unexpected schema change outside PReLU alpha parameters"


def validation_view(model):
    """Remove rANS computation only for NN equivalence; expose its outputs as fixtures."""
    view = copy.deepcopy(model)
    graph = view.subgraphs[0]
    custom_ops = [(index, custom_name(view, operator)) for index, operator in enumerate(graph.operators)
                  if custom_name(view, operator) is not None]
    exposed = []
    for index, _ in custom_ops:
        for tensor_index in graph.operators[index].outputs:
            tensor_index = int(tensor_index)
            if tensor_index not in graph.inputs:
                exposed.append(tensor_index)
    graph.inputs = np.array(list(map(int, graph.inputs)) + exposed, dtype=np.int32)
    custom_indices = {index for index, _ in custom_ops}
    graph.operators = [operator for index, operator in enumerate(graph.operators)
                       if index not in custom_indices]
    return view, [{"node": index, "custom_op": name,
                   "outputs_exposed_as_validation_inputs": list(map(int, model.subgraphs[0].operators[index].outputs))}
                  for index, name in custom_ops]


def sample_for(detail, fixture, rng):
    shape = tuple(map(int, detail["shape"]))
    dtype = detail["dtype"]
    size = int(np.prod(shape))
    if dtype == np.float32:
        if fixture == "constant": return np.full(shape, 0.5, np.float32)
        if fixture == "zeros": return np.zeros(shape, np.float32)
        if fixture == "signed_ramp": return np.linspace(-1, 1, size, dtype=np.float32).reshape(shape)
        return rng.uniform(-1, 1, shape).astype(np.float32)
    if dtype == np.uint8:
        return np.zeros(shape, np.uint8) if fixture != "seeded_random" else rng.integers(0, 256, shape, dtype=np.uint8)
    if dtype == np.int32:
        return np.zeros(shape, np.int32)
    raise AssertionError("Unsupported validation input dtype: %s" % dtype)


def compare_model(name, source_bytes, candidate_bytes, root, has_custom_ops):
    source_model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(source_bytes, 0))
    candidate_model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(candidate_bytes, 0))
    custom_records = []
    if has_custom_ops:
        source_model, custom_records = validation_view(source_model)
        candidate_model, candidate_custom = validation_view(candidate_model)
        assert candidate_custom == custom_records
        source_bytes, candidate_bytes = pack(source_model), pack(candidate_model)
    if has_custom_ops:
        resolver = tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        source_runtime = tf.lite.Interpreter(
            model_content=source_bytes, num_threads=1, experimental_op_resolver_type=resolver)
        candidate_runtime = tf.lite.Interpreter(
            model_content=candidate_bytes, num_threads=1, experimental_op_resolver_type=resolver)
        source_runtime.allocate_tensors()
        candidate_runtime.allocate_tensors()
    else:
        source_runtime, candidate_runtime = interpreter(source_bytes), interpreter(candidate_bytes)
    source_inputs, candidate_inputs = source_runtime.get_input_details(), candidate_runtime.get_input_details()
    source_outputs, candidate_outputs = source_runtime.get_output_details(), candidate_runtime.get_output_details()
    assert len(source_inputs) == len(candidate_inputs) and len(source_outputs) == len(candidate_outputs)
    for left, right in zip(source_inputs + source_outputs, candidate_inputs + candidate_outputs):
        assert left["name"] == right["name"] and left["dtype"] == right["dtype"]
        assert np.array_equal(left["shape"], right["shape"])
    records = []
    for fixture_index, fixture in enumerate(("constant", "zeros", "signed_ramp", "seeded_random")):
        rng = np.random.default_rng(20260904 + fixture_index)
        input_dir = root / "inputs" / name / fixture
        assert checked_destination(input_dir).is_dir()
        for input_index, (before, after) in enumerate(zip(source_inputs, candidate_inputs)):
            value = sample_for(before, fixture, rng)
            source_runtime.set_tensor(before["index"], value)
            candidate_runtime.set_tensor(after["index"], value)
            save(input_dir / ("input_%d.bin" % input_index), value.tobytes())
        source_runtime.invoke()
        candidate_runtime.invoke()
        reference_dir = root / "cache/reference" / name / fixture
        assert checked_destination(reference_dir).is_dir()
        for output_index, (before, after) in enumerate(zip(source_outputs, candidate_outputs)):
            expected = source_runtime.get_tensor(before["index"])
            actual = candidate_runtime.get_tensor(after["index"])
            assert np.isfinite(expected).all() and np.isfinite(actual).all()
            difference = actual.astype(np.float64) - expected.astype(np.float64)
            record = {"fixture": fixture, "output_index": output_index, "name": before["name"],
                      "shape": list(expected.shape), "dtype": str(expected.dtype),
                      "max_abs": float(np.abs(difference).max()),
                      "mean_abs": float(np.abs(difference).mean()),
                      "rmse": float(np.sqrt(np.mean(difference * difference))),
                      "bit_exact": actual.tobytes() == expected.tobytes(),
                      "source_sha256": sha256(expected.tobytes()),
                      "candidate_sha256": sha256(actual.tobytes())}
            records.append(record)
            save(reference_dir / ("reference_output_%d.bin" % output_index), expected.tobytes())
            print(json.dumps({"model": name, **record}), flush=True)
    return records, custom_records, len(source_inputs), len(source_outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.test_dir.resolve()
    manifest = json.loads(args.source_manifest.read_text())
    records_by_name = {record["name"]: record for record in manifest["models"]}
    reports = []
    for name in MODEL_NAMES:
        model_record = records_by_name[name]
        original_path = args.source_dir / (name + ".tflite")
        original = original_path.read_bytes()
        assert sha256(original) == model_record["sha256"]
        source_model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(original, 0))
        assert len(source_model.subgraphs) == 1
        candidate_model, changes, added_constants = expand_scalar_prelu(source_model)
        rewritten = pack(candidate_model)
        assert_only_prelu_changes(source_model, rewritten, changes)
        candidate_path = root / "candidate" / (name + ".tflite")
        candidate_exists = candidate_path.exists()
        if candidate_exists:
            assert candidate_path.read_bytes() == rewritten, "Existing candidate differs"
        custom_ops = [{"node": index, "name": custom_name(source_model, operator)}
                      for index, operator in enumerate(source_model.subgraphs[0].operators)
                      if custom_name(source_model, operator) is not None]
        comparisons, validation_custom, validation_inputs, validation_outputs = compare_model(
            name, original, rewritten, root, bool(custom_ops))
        assert all(record["bit_exact"] for record in comparisons)
        assert sha256(original_path.read_bytes()) == model_record["sha256"]
        if not candidate_exists:
            save(candidate_path, rewritten)
        reports.append({"status": "PASS", "model": name, "source_sha256": model_record["sha256"],
                        "candidate_sha256": sha256(rewritten), "source_bytes": len(original),
                        "candidate_bytes": len(rewritten), "operator_count": len(source_model.subgraphs[0].operators),
                        "prelu_changes": changes, "added_tensors": added_constants,
                        "added_buffers": added_constants, "other_schema_fields_unchanged": True,
                        "custom_ops_unchanged": custom_ops, "validation_custom_ops": validation_custom,
                        "validation_scope": "full_graph" if not custom_ops else "all_nn_with_rans_outputs_as_external_validation_inputs",
                        "validation_input_count": validation_inputs, "validation_output_count": validation_outputs,
                        "comparisons": comparisons, "server_bit_exact": True, "android_gpu_validated": False})
    report = {"status": "PASS", "variant": "small", "models": reports}
    save(root / "candidate/gpu-prelu-manifest.json", json.dumps(report, indent=2).encode())
    print("small_remaining_gpu_prelu_export=PASS models=5 bit_exact=true gpu_validated=false", flush=True)


if __name__ == "__main__":
    main()
