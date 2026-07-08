#!/usr/bin/env python3
"""Convert the existing full recon ONNX graphs to MTK TFLite.

This script is intentionally ONNX-in / TFLite-out.  It does not re-export from
PyTorch, so the first precision check compares the current Android ONNX recon
baseline against the converted TFLite graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


GRAPHS: dict[str, dict[str, Any]] = {
    "i_recon_full": {
        "onnx": "i_recon.onnx",
        "inputs": [("i_y_hat", (1, 256, 16, 32))],
        "outputs": [
            ("i_codeword", (1, 18, 16, 32)),
            ("encoder_i_reference_frame", (1, 3, 256, 512)),
        ],
    },
    "p_recon_full": {
        "onnx": "p_recon.onnx",
        "inputs": [
            ("p_y_hat", (1, 128, 16, 32)),
            ("p_ctx", (1, 256, 32, 64)),
        ],
        "outputs": [
            ("encoder_p_reference_feature", (1, 256, 32, 64)),
            ("encoder_p_reference_frame", (1, 3, 256, 512)),
        ],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shape_text(shape: tuple[int, ...]) -> str:
    return ",".join(str(dim) for dim in shape)


def find_converter(explicit: str | None) -> str:
    if explicit:
        candidate = Path(explicit)
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(f"mtk_onnx_converter does not exist: {candidate}")
    candidate = shutil.which("mtk_onnx_converter")
    if candidate:
        return candidate
    candidate = Path(sys.executable).resolve().parent / "mtk_onnx_converter"
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError("mtk_onnx_converter is not in PATH or current Python bin")


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed rc={result.returncode}; see {log_path}")


def parse_graphs(raw: str) -> list[str]:
    if raw == "all":
        return list(GRAPHS)
    selected = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in selected if item not in GRAPHS]
    if unknown:
        raise ValueError(f"unknown graphs: {unknown}; valid={list(GRAPHS)}")
    return selected


def convert_graph(
    name: str,
    variant: str,
    models_dir: Path,
    output_dir: Path,
    converter: str,
) -> dict[str, Any]:
    spec = GRAPHS[name]
    onnx_path = models_dir / spec["onnx"]
    if not onnx_path.is_file():
        raise FileNotFoundError(f"missing ONNX graph: {onnx_path}")

    tflite_path = output_dir / f"{name}_{variant}.tflite"
    log_path = output_dir / "logs" / f"{name}_{variant}_onnx_converter.log"
    command = [
        converter,
        "--input_model_file",
        str(onnx_path),
        "--output_file",
        str(tflite_path),
        "--output_file_format",
        "tflite",
        "--input_names",
        ",".join(item[0] for item in spec["inputs"]),
        "--input_shapes",
        ":".join(shape_text(item[1]) for item in spec["inputs"]),
        "--output_names",
        ",".join(item[0] for item in spec["outputs"]),
        "--tflite_op_export_spec",
        "builtin_first",
    ]
    if variant == "fp16_weight":
        command += ["--convert_float32_weights_to_float16", "True"]

    run_command(command, log_path)
    return {
        "name": name,
        "variant": variant,
        "status": "ok",
        "onnx": str(onnx_path),
        "onnx_sha256": sha256(onnx_path),
        "tflite": str(tflite_path),
        "tflite_sha256": sha256(tflite_path),
        "converter_log": str(log_path),
        "inputs": [{"name": n, "shape": list(s)} for n, s in spec["inputs"]],
        "outputs": [{"name": n, "shape": list(s)} for n, s in spec["outputs"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--android-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--onnx-converter", default=None)
    parser.add_argument("--graphs", default="all", help="all or comma-separated graph names")
    parser.add_argument("--variant", choices=("fp32", "fp16_weight", "all"), default="fp32")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--copy-assets", action="store_true")
    args = parser.parse_args()

    android_root = args.android_root.resolve()
    models_dir = android_root / "app" / "src" / "main" / "assets" / "models"
    output_dir = (args.output_dir or android_root / "outputs" / "full_recon_tflite").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = find_converter(args.onnx_converter)
    graphs = parse_graphs(args.graphs)
    variants = ("fp32", "fp16_weight") if args.variant == "all" else (args.variant,)

    print(f"using mtk_onnx_converter: {converter}")
    records: list[dict[str, Any]] = []
    for graph in graphs:
        for variant in variants:
            try:
                record = convert_graph(graph, variant, models_dir, output_dir, converter)
                print(f"converted {graph}_{variant}: {record['tflite']}")
            except Exception as exc:
                record = {"name": graph, "variant": variant, "status": "failed", "error": str(exc)}
                print(f"failed {graph}_{variant}: {exc}")
            records.append(record)

    manifest = {
        "route": "existing_onnx_to_tflite",
        "records": records,
    }
    manifest_path = output_dir / "full_recon_tflite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.copy_assets:
        assets_dir = android_root / "app" / "src" / "main" / "assets" / "recon_diagnostic"
        assets_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            if record.get("status") == "ok":
                shutil.copy2(record["tflite"], assets_dir / Path(record["tflite"]).name)
        shutil.copy2(manifest_path, assets_dir / manifest_path.name)

    ok = sum(1 for record in records if record.get("status") == "ok")
    failed = len(records) - ok
    print(f"wrote {manifest_path}")
    print(f"ok_records={ok} failed_records={failed}")


if __name__ == "__main__":
    main()
