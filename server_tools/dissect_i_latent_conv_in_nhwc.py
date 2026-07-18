#!/usr/bin/env python3
"""Create official MediaTek per-OP artifacts for the NHWC I decoder front.

Run only on the Linux server. The script uses mtk_converter.dissect_tflite_model
without rewriting the FlatBuffer graph, then records every generated artifact so
the Android Online Compile probe can consume the exact sub-models later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import mtk_converter
import numpy as np


INPUT_NCHW_SHAPE = (1, 256, 16, 32)
INPUT_NHWC_SHAPE = (1, 16, 32, 256)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_tree(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        record: dict[str, object] = {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        if path.suffix == ".npy":
            array = np.load(path, allow_pickle=False)
            record.update({"kind": "npy", "shape": list(array.shape), "dtype": str(array.dtype)})
        elif path.suffix == ".tflite":
            record["kind"] = "tflite_submodel"
        else:
            record["kind"] = "other"
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--android-root", type=Path, required=True)
    parser.add_argument("--tflite", type=Path, default=None)
    parser.add_argument("--input-f32le", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--copy-assets", action="store_true")
    args = parser.parse_args()

    android_root = args.android_root.resolve()
    tflite = (args.tflite or android_root / "app" / "src" / "main" / "assets" / "recon_diagnostic" / "i_latent_conv_in_nhwc_fp32.tflite").resolve()
    input_path = (args.input_f32le or android_root / "app" / "src" / "main" / "assets" / "baseline" / "tensors" / "i_y_hat.f32le").resolve()
    output_dir = (args.output_dir or android_root / "outputs" / "i_latent_conv_in_nhwc_dissect").resolve()
    dissect_dir = output_dir / "dissected"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = np.fromfile(input_path, dtype="<f4")
    expected_numel = int(np.prod(INPUT_NCHW_SHAPE))
    if raw.size != expected_numel:
        raise ValueError(f"input size mismatch: got={raw.size}, expected={expected_numel}, path={input_path}")
    input_nhwc = raw.reshape(INPUT_NCHW_SHAPE).transpose(0, 2, 3, 1).copy()
    np.save(output_dir / "input_i_y_hat_nhwc.npy", input_nhwc)
    input_nhwc.tofile(output_dir / "input_i_y_hat_nhwc.f32le")

    executor = mtk_converter.TFLiteExecutor(str(tflite))
    full_outputs = executor.run([input_nhwc])
    if len(full_outputs) != 1:
        raise RuntimeError(f"expected one full-model output, got={len(full_outputs)}")
    full_output = np.asarray(full_outputs[0], dtype=np.float32)
    np.save(output_dir / "full_output_i_dec_stage0_nhwc.npy", full_output)
    full_output.tofile(output_dir / "full_output_i_dec_stage0_nhwc.f32le")

    if dissect_dir.exists():
        shutil.rmtree(dissect_dir)
    mtk_converter.dissect_tflite_model(
        str(tflite),
        input_nhwc,
        str(dissect_dir),
        quiet=True,
    )

    manifest = {
        "tool": Path(__file__).name,
        "route": "official_mtk_converter_dissect_tflite_model",
        "model": str(tflite),
        "model_sha256": sha256(tflite),
        "input_nchw_source": str(input_path),
        "input_nhwc_shape": list(input_nhwc.shape),
        "input_nhwc_dtype": str(input_nhwc.dtype),
        "full_output_nhwc_shape": list(full_output.shape),
        "full_output_nhwc_dtype": str(full_output.dtype),
        "dissect_dir": str(dissect_dir),
        "artifacts": describe_tree(dissect_dir),
    }
    manifest_path = output_dir / "i_latent_conv_in_nhwc_dissect_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.copy_assets:
        assets_dir = android_root / "app" / "src" / "main" / "assets" / "recon_dissect_i_nhwc"
        if assets_dir.exists():
            shutil.rmtree(assets_dir)
        shutil.copytree(dissect_dir, assets_dir / "submodels")
        shutil.copy2(manifest_path, assets_dir / manifest_path.name)
        shutil.copy2(output_dir / "input_i_y_hat_nhwc.f32le", assets_dir / "input_i_y_hat_nhwc.f32le")
        shutil.copy2(output_dir / "full_output_i_dec_stage0_nhwc.f32le", assets_dir / "full_output_i_dec_stage0_nhwc.f32le")

    print(f"wrote {manifest_path}")
    print(f"dissected_submodels={sum(item['kind'] == 'tflite_submodel' for item in manifest['artifacts'])}")


if __name__ == "__main__":
    main()
