#!/usr/bin/env python3
"""Export only the existing Small temporal_from_frame GPU PReLU candidate.

Run on the Linux server, not the PC. No reconversion, NN rewrite, or MTK
runtime changes: duplicate the scalar alpha bytes into two channel constants.
CPU reference inference here is offline equivalence validation, not Android
GPU inference fallback.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import flatbuffers
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema


SOURCE_SHA256 = "1506f04cd1db51dd83a76052668274b0d4b246efd25c2151041cbf9ef84cc9f5"
SERVER_ROOT = Path("/media/ltelab/D/weilingfeng")
PRELU_CHANNELS = {76: 96, 91: 192, 104: 192}


def checked_destination(path):
    # Server workspace policy requires readlink -f before every write/create.
    resolved = Path(subprocess.check_output(["readlink", "-f", str(path)], text=True).strip())
    assert SERVER_ROOT in resolved.parents, resolved
    return resolved


def save(path, data):
    checked_destination(path).write_bytes(data)


def pack(model):
    builder = flatbuffers.Builder(0)
    root = model.Pack(builder)
    builder.Finish(root, file_identifier=b"TFL3")
    return bytes(builder.Output())


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def interpreter(model_bytes):
    result = tf.lite.Interpreter(
        model_content=model_bytes,
        num_threads=1,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    result.allocate_tensors()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.test_dir.resolve()
    candidate_path = root / "candidate/temporal_from_frame.tflite"
    assert candidate_path.resolve() != args.source_model.resolve()
    assert not candidate_path.exists(), "Use a fresh isolated candidate directory"
    original = args.source_model.read_bytes()
    assert sha256(original) == SOURCE_SHA256, "Only the verified Small temporal model is in scope"
    manifest = json.loads(args.source_manifest.read_text())
    source_record = next(m for m in manifest["models"] if m["name"] == "temporal_from_frame")
    assert source_record["sha256"] == SOURCE_SHA256
    source = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(original, 0))
    assert len(source.subgraphs) == 1
    candidate = copy.deepcopy(source)
    graph = candidate.subgraphs[0]
    found = {i for i, op in enumerate(graph.operators)
             if candidate.operatorCodes[op.opcodeIndex].builtinCode == schema.BuiltinOperator.PRELU}
    assert found == set(PRELU_CHANNELS)
    constants = {}
    changes = []
    for index, channels in PRELU_CHANNELS.items():
        op = graph.operators[index]
        x_index, alpha_index = map(int, op.inputs)
        x, alpha = graph.tensors[x_index], graph.tensors[alpha_index]
        assert x.type == alpha.type == schema.TensorType.FLOAT32
        assert list(x.shape) == [1, 32, 64, channels]
        assert list(alpha.shape) == [1, 1, 1, 1]
        assert all(alpha_index not in node.outputs for node in graph.operators)
        scalar_bytes = bytes(candidate.buffers[alpha.buffer].data)
        assert len(scalar_bytes) == 4
        key = (alpha_index, channels)
        if key not in constants:
            buffer = schema.BufferT()
            # Repeat the exact FP32 bit pattern; no value conversion/rounding.
            buffer.data = np.frombuffer(scalar_bytes * channels, dtype=np.uint8).copy()
            expanded = copy.deepcopy(alpha)
            expanded.name = alpha.name + ("_gpu_c%d" % channels).encode()
            expanded.shape = np.array([1, 1, 1, channels], dtype=np.int32)
            assert expanded.shapeSignature is None
            expanded.buffer = len(candidate.buffers)
            candidate.buffers.append(buffer)
            constants[key] = len(graph.tensors)
            graph.tensors.append(expanded)
        op.inputs[1] = constants[key]
        changes.append({"node": index, "old_alpha_tensor": alpha_index,
                        "new_alpha_tensor": int(op.inputs[1]), "channels": channels,
                        "alpha": float(np.frombuffer(scalar_bytes, dtype="<f4")[0]),
                        "old_shape": [1, 1, 1, 1], "new_shape": [1, 1, 1, channels]})

    rewritten = pack(candidate)
    restored = copy.deepcopy(schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(rewritten, 0)))
    for change in changes:
        restored.subgraphs[0].operators[change["node"]].inputs[1] = change["old_alpha_tensor"]
    del restored.subgraphs[0].tensors[len(source.subgraphs[0].tensors):]
    del restored.buffers[len(source.buffers):]
    # Reverting only the declared alpha edits must restore the entire schema object.
    assert pack(restored) == pack(source), "Unexpected change outside PReLU alpha parameters"
    save(candidate_path, rewritten)

    original_runtime, candidate_runtime = interpreter(original), interpreter(rewritten)
    original_inputs, candidate_inputs = original_runtime.get_input_details(), candidate_runtime.get_input_details()
    original_outputs, candidate_outputs = original_runtime.get_output_details(), candidate_runtime.get_output_details()
    for before, after in ((original_inputs, candidate_inputs), (original_outputs, candidate_outputs)):
        assert len(before) == len(after)
        for left, right in zip(before, after):
            assert left["name"] == right["name"] and left["dtype"] == right["dtype"]
            assert np.array_equal(left["shape"], right["shape"])
    assert len(original_inputs) == 1 and list(original_inputs[0]["shape"]) == [1, 256, 512, 3]
    shape = tuple(original_inputs[0]["shape"])
    constant = np.frombuffer((root / "inputs/input_0.bin").read_bytes(), dtype="<f4").reshape(shape)
    samples = {"constant": constant, "zeros": np.zeros(shape, dtype=np.float32),
               "signed_ramp": np.linspace(-1, 1, int(np.prod(shape)), dtype=np.float32).reshape(shape),
               "seeded_random": np.random.default_rng(20260904).uniform(-1, 1, shape).astype(np.float32)}
    comparisons = []
    for name, sample in samples.items():
        if name != "constant":
            checked_destination(root / "inputs" / name).mkdir()
            save(root / "inputs" / name / "input_0.bin", sample.tobytes())
        fixture_dir = root / "cache" / name
        checked_destination(fixture_dir).mkdir()
        for runtime, details in ((original_runtime, original_inputs), (candidate_runtime, candidate_inputs)):
            runtime.set_tensor(details[0]["index"], sample)
            runtime.invoke()
        for output_index, (before, after) in enumerate(zip(original_outputs, candidate_outputs)):
            expected = original_runtime.get_tensor(before["index"])
            actual = candidate_runtime.get_tensor(after["index"])
            assert np.isfinite(expected).all() and np.isfinite(actual).all()
            difference = actual.astype(np.float64) - expected.astype(np.float64)
            record = {"fixture": name, "input_sha256": sha256(sample.tobytes()),
                      "output_index": output_index, "name": before["name"],
                      "shape": expected.shape, "dtype": str(expected.dtype),
                      "max_abs": float(np.abs(difference).max()),
                      "mean_abs": float(np.abs(difference).mean()),
                      "rmse": float(np.sqrt(np.mean(difference * difference))),
                      "bit_exact": actual.tobytes() == expected.tobytes(),
                      "original_sha256": sha256(expected.tobytes()),
                      "candidate_sha256": sha256(actual.tobytes())}
            comparisons.append(record)
            save(fixture_dir / ("reference_output_%d.bin" % output_index), expected.tobytes())
            print(json.dumps(record), flush=True)
    assert sha256(args.source_model.read_bytes()) == SOURCE_SHA256
    passed = all(record["bit_exact"] for record in comparisons)
    report = {"status": "PASS" if passed else "FAIL", "variant": "small",
              "model": "temporal_from_frame", "backend": "server_tflite_builtin_ref",
              "tensorflow_version": tf.__version__, "checkpoint_sha256": manifest["checkpoint_sha256"],
              "source_sha256": SOURCE_SHA256, "candidate_sha256": sha256(rewritten),
              "source_bytes": len(original), "candidate_bytes": len(rewritten),
              "operators_unchanged": len(graph.operators), "added_tensors": len(constants),
              "added_buffers": len(constants), "other_schema_fields_unchanged": True,
              "changes": changes, "inputs": source_record["inputs"], "outputs": source_record["outputs"],
              "comparisons": comparisons, "gpu_invoke_validated": False}
    save(root / "candidate/gpu-prelu-manifest.json", json.dumps(report, indent=2).encode())
    assert passed, "Stop: numerical equivalence failed; do not probe or rewrite further"
    print("small_temporal_gpu_prelu_export=PASS bit_exact=true gpu_validated=false", flush=True)


if __name__ == "__main__":
    main()
