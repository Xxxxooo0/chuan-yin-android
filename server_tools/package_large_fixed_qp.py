#!/usr/bin/env python3
"""Build the canonical Large online TFLite package with QP 9 embedded."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


FIXED_QP = 9
FRONT_MODELS = {
    "temporal_from_frame_big": "temporal_from_frame.tflite",
    "temporal_from_feature_big": "temporal_from_feature.tflite",
    "i_encoder_analysis_big": "i_encoder.tflite",
    "p_encoder_analysis_big": "p_encoder.tflite",
}
DECODER_MODELS = {
    "i_decoder_synthesis_merged_fp32": "i_decoder.tflite",
    "p_decoder_synthesis_merged_fp32": "p_decoder.tflite",
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


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError("missing manifest: {}".format(path))
    return json.loads(path.read_text(encoding="utf-8"))


def require_qp9(manifest: dict, label: str) -> None:
    if manifest.get("qp") != FIXED_QP:
        raise RuntimeError("{} was exported for QP={}, expected QP={}".format(
            label, manifest.get("qp"), FIXED_QP
        ))


def record_by_name(manifest: dict, name: str) -> dict:
    for record in manifest.get("records", []):
        if record.get("name") == name:
            return record
    raise KeyError("missing export record: {}".format(name))


def copy_model(source: Path, target: Path, record: dict) -> dict:
    if record.get("status") != "ok" or record.get("converter_rc") != 0:
        raise RuntimeError("model export is not verified: {} status={} converter_rc={}".format(
            record.get("name"), record.get("status"), record.get("converter_rc")
        ))
    if not source.is_file():
        raise FileNotFoundError("missing TFLite model: {}".format(source))
    source_sha = sha256(source)
    recorded_sha = record.get("tflite_sha256")
    if recorded_sha and source_sha != recorded_sha:
        raise RuntimeError("TFLite SHA mismatch: {}".format(source))
    shutil.copy2(source, target)
    return {
        "name": target.stem,
        "file": "models/{}".format(target.name),
        "sha256": sha256(target),
        "input_names": record.get("input_names"),
        "input_shapes_nhwc": record.get("input_shapes_nhwc"),
        "output_names": record.get("output_names"),
        "output_shapes_nhwc": (
            record.get("actual_torchscript_output_shapes_nhwc")
            or record.get("actual_output_shapes_nhwc")
        ),
    }


def write_checksums(root: Path) -> None:
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt")
    lines = ["{}  {}".format(sha256(path), path.relative_to(root).as_posix()) for path in paths]
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--front-export-dir", type=Path, required=True)
    parser.add_argument("--decoder-export-dir", type=Path, required=True)
    parser.add_argument("--entropy-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    front = args.front_export_dir.resolve()
    decoder = args.decoder_export_dir.resolve()
    entropy = args.entropy_model_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("output exists: {}".format(output))

    front_manifest = load_json(front / "three_modules_offline_nhwc_manifest.json")
    decoder_manifest = load_json(decoder / "decoder_full_norm_rewrite_manifest.json")
    require_qp9(front_manifest, "front export")
    require_qp9(decoder_manifest, "decoder export")

    output.mkdir(parents=True)
    models_dir = output / "models"
    models_dir.mkdir()
    published = []

    for record_name, target_name in FRONT_MODELS.items():
        record = record_by_name(front_manifest, record_name)
        published.append(copy_model(front / "{}_fp32.tflite".format(record_name), models_dir / target_name, record))

    for record_name, target_name in DECODER_MODELS.items():
        record = record_by_name(decoder_manifest, record_name)
        precision = record.get("precision") or {}
        if not precision.get("passed"):
            raise RuntimeError("decoder precision verification failed: {}".format(record_name))
        published.append(copy_model(decoder / "{}.tflite".format(record_name), models_dir / target_name, record))

    entropy_records = []
    for name in ENTROPY_MODELS:
        source = entropy / name
        if not source.is_file():
            raise FileNotFoundError("missing entropy model: {}".format(source))
        target = models_dir / name
        shutil.copy2(source, target)
        entropy_records.append({
            "name": target.stem,
            "file": "models/{}".format(name),
            "sha256": sha256(target),
        })

    front_checkpoints = front_manifest.get("checkpoint_sha256") or {}
    decoder_checkpoints = decoder_manifest.get("checkpoint_sha256") or {}
    if front_checkpoints != decoder_checkpoints:
        raise RuntimeError("front/decoder checkpoint SHA mismatch")

    manifest = {
        "package": "gvc-rt-large-tflite-online-fixed-qp9-270p",
        "resolution": {"height": 256, "width": 512},
        "layout": "NHWC",
        "io_dtype": "FP32",
        "target": "Official NeuronDelegate online compile",
        "qp": FIXED_QP,
        "default_qp": FIXED_QP,
        "dynamic_qp": False,
        "models": published,
        "entropy_models": entropy_records,
        "checkpoint_sha256": front_checkpoints,
        "source_manifests": {
            "front": "three_modules_offline_nhwc_manifest.json",
            "decoder": "decoder_full_norm_rewrite_manifest.json",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.copy2(front / "three_modules_offline_nhwc_manifest.json", output / "three_modules_offline_nhwc_manifest.json")
    shutil.copy2(decoder / "decoder_full_norm_rewrite_manifest.json", output / "decoder_full_norm_rewrite_manifest.json")
    (output / "README.md").write_text(
        "# GVC-RT Large 在线固定 QP9 模型包\n\n"
        "该包固定输入尺寸为 `256x512`、布局为 NHWC、外部 I/O 为 FP32，QP 固定为 9。\n\n"
        "六张连续神经网络图已将 QP9 的量化尺度固化为常量；四张 entropy+rANS 图由运行时固定传入 QP9。"
        "Android 使用官方 `NeuronDelegate` 在线编译，不需要 `quant_scales/`。\n",
        encoding="utf-8",
    )
    write_checksums(output)
    print("wrote {}".format(output))
    print("models=10 qp=9 dynamic_qp=false")


if __name__ == "__main__":
    main()
