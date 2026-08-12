#!/usr/bin/env python3
"""Build one Large online TFLite package with runtime-selectable QP scales."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


MODEL_MAP = {
    "temporal_from_frame_dynamic_qp_fp32.tflite": "temporal_from_frame.tflite",
    "temporal_from_feature_dynamic_qp_fp32.tflite": "temporal_from_feature.tflite",
    "i_encoder_dynamic_qp_fp32.tflite": "i_encoder.tflite",
    "p_encoder_dynamic_qp_fp32.tflite": "p_encoder.tflite",
    "i_decoder_dynamic_qp_fp32.tflite": "i_decoder.tflite",
    "p_decoder_dynamic_qp_fp32.tflite": "p_decoder.tflite",
}

ENTROPY_MODELS = (
    "i_entropy_prior_merged_rans.tflite",
    "p_entropy_prior_merged_rans.tflite",
    "i_entropy_decode_merged_rans.tflite",
    "p_entropy_decode_merged_rans.tflite",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksums(root: Path) -> None:
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "SHA256SUMS.txt"):
        lines.append("{}  {}".format(sha256(path), path.relative_to(root).as_posix()))
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-package-dir", type=Path, default=None)
    parser.add_argument("--dynamic-export-dir", type=Path, required=True)
    parser.add_argument("--entropy-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base = args.base_package_dir.resolve() if args.base_package_dir else None
    exported = args.dynamic_export_dir.resolve()
    entropy = args.entropy_model_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("output exists: {}".format(output))
    export_manifest_path = exported / "large_dynamic_qp_export_manifest.json"
    if not export_manifest_path.is_file():
        raise FileNotFoundError("missing dynamic export manifest: {}".format(export_manifest_path))
    export_manifest = json.loads(export_manifest_path.read_text(encoding="utf-8"))
    if any(not record.get("online_package_eligible") for record in export_manifest["records"]):
        raise RuntimeError("dynamic export contains models that cannot be used by online TFLite")
    if not all(record.get("passed") for record in export_manifest["source_verification"]):
        raise RuntimeError("dynamic export source verification failed")

    if base is not None:
        if not (base / "manifest.json").is_file():
            raise FileNotFoundError("invalid base package: {}".format(base))
        shutil.copytree(base, output)
    else:
        output.mkdir(parents=True)
    models_dir = output / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    published_models = []
    export_records = {record["name"]: record for record in export_manifest["records"]}
    for source_name, target_name in MODEL_MAP.items():
        source = exported / source_name
        if not source.is_file():
            raise FileNotFoundError("missing dynamic model: {}".format(source))
        target = models_dir / target_name
        shutil.copy2(source, target)
        record_name = source_name[:-len("_fp32.tflite")]
        record = export_records[record_name]
        published_models.append({
            "name": target.stem,
            "file": "models/{}".format(target.name),
            "sha256": sha256(target),
            "input_names": record["input_names"],
            "input_shapes_nhwc": record["input_shapes_nhwc"],
            "output_names": record["output_names"],
            "output_shapes_nhwc": record["actual_torchscript_output_shapes_nhwc"],
        })

    for name in ENTROPY_MODELS:
        source = entropy / name
        if not source.is_file():
            raise FileNotFoundError("missing entropy model: {}".format(source))
        shutil.copy2(source, models_dir / name)

    scale_records = {}
    for name, record in export_manifest["quant_scale_tables"].items():
        source = exported / record["file"]
        target = output / record["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        scale_records[name] = {
            **record,
            "sha256": sha256(target),
        }

    base_manifest_path = output / "manifest.json"
    base_manifest = (
        json.loads(base_manifest_path.read_text(encoding="utf-8"))
        if base_manifest_path.is_file()
        else {
            "resolution": {"height": 256, "width": 512},
            "layout": "NHWC",
            "io_dtype": "FP32",
            "target": "Official NeuronDelegate online compile",
        }
    )
    base_manifest.update({
        "package": "gvc-rt-large-tflite-online-dynamic-qp-270p",
        "qp": 0,
        "default_qp": 0,
        "dynamic_qp": True,
        "supported_qps": export_manifest["supported_qps"],
        "validated_qps": export_manifest["verification_qps"],
        "quant_scale_tables": scale_records,
        "models": published_models,
        "entropy_models": [
            {"name": Path(name).stem, "file": "models/{}".format(name), "sha256": sha256(models_dir / name)}
            for name in ENTROPY_MODELS
        ],
        "checkpoint_sha256": export_manifest["checkpoint_sha256"],
    })
    base_manifest_path.write_text(json.dumps(base_manifest, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# GVC-RT Large 在线动态 QP 模型包\n\n"
        "该模型包共用一套 TFLite 权重，Android 运行时可选择 QP 0、3、6、9。\n\n"
        "六张连续神经网络图从 `quant_scales/` 接收对应 QP 的量化尺度；"
        "四张 entropy+rANS 图共用模型，并由运行时传入 QP。\n",
        encoding="utf-8",
    )
    write_checksums(output)
    print("wrote {}".format(output))
    print("models=10 supported_qps=0,3,6,9 dynamic_qp=true")


if __name__ == "__main__":
    main()
