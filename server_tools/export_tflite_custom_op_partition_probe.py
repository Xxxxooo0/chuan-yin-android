#!/usr/bin/env python3
"""Export Conv2D -> CPU custom identity -> Conv2D for Neuron partition testing."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


CUSTOM_OP = "GVC_RT_CPU_IDENTITY"
INPUT_SHAPE = [1, 8, 8, 8]


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_builtin_model(output_path):
    import numpy as np
    import tensorflow as tf

    channels = INPUT_SHAPE[-1]
    kernel = np.zeros((1, 1, channels, channels), dtype=np.float32)
    for channel in range(channels):
        kernel[0, 0, channel, channel] = 1.0
    kernel = tf.constant(kernel)

    class Probe(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(INPUT_SHAPE, tf.float32, name="input_nhwc")])
        def __call__(self, x):
            first = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding="SAME", name="conv_before")
            custom_placeholder = tf.math.abs(first, name="custom_identity_placeholder")
            output = tf.nn.conv2d(
                custom_placeholder,
                kernel,
                strides=[1, 1, 1, 1],
                padding="SAME",
                name="conv_after",
            )
            return {"output_nhwc": output}

    concrete = Probe().__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
    converter.optimizations = []
    output_path.write_bytes(converter.convert())


def replace_abs_with_custom(source_path, output_path):
    import flatbuffers
    from tensorflow.lite.python import schema_py_generated as schema

    raw = source_path.read_bytes()
    model_t = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(raw, 0))
    abs_code = int(schema.BuiltinOperator.ABS)
    custom_code = int(schema.BuiltinOperator.CUSTOM)
    matching = []
    for index, opcode in enumerate(model_t.operatorCodes):
        if int(opcode.builtinCode) == abs_code or int(opcode.deprecatedBuiltinCode) == abs_code:
            matching.append(index)
    if len(matching) != 1:
        raise RuntimeError("expected exactly one ABS opcode, found {}".format(matching))
    opcode = model_t.operatorCodes[matching[0]]
    opcode.builtinCode = custom_code
    opcode.deprecatedBuiltinCode = custom_code
    opcode.customCode = CUSTOM_OP.encode("utf-8")

    builder = flatbuffers.Builder(0)
    model_offset = model_t.Pack(builder)
    builder.Finish(model_offset, file_identifier=b"TFL3")
    output_path.write_bytes(bytes(builder.Output()))


def inspect_model(path):
    from tensorflow.lite.python import schema_py_generated as schema

    raw = path.read_bytes()
    model = schema.Model.GetRootAsModel(raw, 0)
    names = []
    subgraph = model.Subgraphs(0)
    for index in range(subgraph.OperatorsLength()):
        operator = subgraph.Operators(index)
        opcode = model.OperatorCodes(operator.OpcodeIndex())
        custom = opcode.CustomCode()
        names.append(custom.decode("utf-8") if custom else str(opcode.BuiltinCode()))
    if names.count(CUSTOM_OP) != 1:
        raise RuntimeError("custom op missing after patch: {}".format(names))
    return names


def write_fixture(output_dir):
    import numpy as np

    count = int(np.prod(INPUT_SHAPE))
    values = ((np.arange(count, dtype=np.float32) % 31) + 1) / 32.0
    values.reshape(INPUT_SHAPE).astype("<f4").tofile(str(output_dir / "input_nhwc.f32le"))
    values.reshape(INPUT_SHAPE).astype("<f4").tofile(str(output_dir / "expected_output_nhwc.f32le"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--android-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--copy-assets", action="store_true")
    args = parser.parse_args()

    android_root = args.android_root.resolve()
    output_dir = (args.output_dir or android_root / "outputs" / "tflite_custom_op_partition_probe").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    builtin_path = output_dir / "rans_custom_op_partition_probe_builtin.tflite"
    model_path = output_dir / "rans_custom_op_partition_probe.tflite"
    build_builtin_model(builtin_path)
    replace_abs_with_custom(builtin_path, model_path)
    operators = inspect_model(model_path)
    write_fixture(output_dir)

    copied_asset = None
    if args.copy_assets:
        copied_asset = android_root / "app" / "src" / "mtkOffline" / "assets" / "diagnostic" / model_path.name
        copied_asset.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(model_path), str(copied_asset))

    manifest = {
        "tool": Path(__file__).name,
        "purpose": "Neuron partition -> CPU custom op -> Neuron partition feasibility probe",
        "custom_op": CUSTOM_OP,
        "layout": "NHWC",
        "dtype": "float32",
        "input_shape": INPUT_SHAPE,
        "output_shape": INPUT_SHAPE,
        "model": model_path.name,
        "model_sha256": sha256(model_path),
        "operators": operators,
        "input_fixture": "input_nhwc.f32le",
        "expected_output": "expected_output_nhwc.f32le",
        "copied_asset": str(copied_asset) if copied_asset else None,
        "android_custom_op_implementation": "CPU byte-for-byte identity",
    }
    manifest_path = output_dir / "rans_custom_op_partition_probe_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("wrote {}".format(model_path))
    print("sha256={}".format(manifest["model_sha256"]))
    print("operators={}".format(operators))
    print("manifest={}".format(manifest_path))
    if copied_asset:
        print("copied_asset={}".format(copied_asset))


if __name__ == "__main__":
    main()
