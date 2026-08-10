#!/usr/bin/env python3
"""Append P entropy assets and I/P CDF tables to a Large online package.

The I entropy/prior network is deployed as a separate merged TFLite file. The
seven obsolete split I graphs are never copied into the package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from gvcrt_export_common import load_i_model, load_p_model


OBSOLETE_I_MODELS = (
    "i_hyper_enc_continuous",
    "i_hyper_prior_shared",
    "i_prior_stage0_params",
    "i_prior_reduce",
    "i_prior_stage1_continuous",
    "i_prior_stage2_continuous",
    "i_prior_stage3_continuous",
)

P_MODELS = (
    "p_hyper_enc_continuous",
    "p_hyper_prior_shared",
    "p_prior_stage0_params",
    "p_prior_stage1_continuous",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_i32(path: Path, value) -> None:
    array = np.asarray(value, dtype="<i4")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(array.tobytes())


def write_cdf(root: Path, name: str, info) -> dict:
    cdf, lengths, offsets = info
    cdf = np.asarray(cdf, dtype="<i4")
    lengths = np.asarray(lengths, dtype="<i4").reshape(-1)
    offsets = np.asarray(offsets, dtype="<i4").reshape(-1)
    base = Path("entropy")
    cdf_path = root / f"{base}/{name}_cdf.i32le"
    lengths_path = root / f"{base}/{name}_cdf_lengths.i32le"
    offsets_path = root / f"{base}/{name}_cdf_offsets.i32le"
    write_i32(cdf_path, cdf)
    write_i32(lengths_path, lengths)
    write_i32(offsets_path, offsets)
    return {
        "cdf": str(cdf_path.relative_to(root)).replace("\\", "/"),
        "shape": list(cdf.shape),
        "cdf_lengths": str(lengths_path.relative_to(root)).replace("\\", "/"),
        "offsets": str(offsets_path.relative_to(root)).replace("\\", "/"),
    }


def write_checksums(root: Path) -> None:
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "SHA256SUMS.txt"):
        lines.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--base-package-dir", type=Path, required=True)
    parser.add_argument("--entropy-export-dir", type=Path, required=True)
    parser.add_argument("--i-merged-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qp", type=int, default=0)
    parser.add_argument("--force-zero-thres", type=float, default=0.12)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    base = args.base_package_dir.resolve()
    entropy_export = args.entropy_export_dir.resolve()
    i_merged_model = args.i_merged_model.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    if not (base / "models/i_encoder.tflite").is_file():
        raise FileNotFoundError(f"not a Large online package: {base}")
    if not i_merged_model.is_file():
        raise FileNotFoundError(f"missing standalone I merged model: {i_merged_model}")

    print("[large-online-entropy] copying base package", flush=True)
    shutil.copytree(base, output)
    model_dir = output / "models"
    for name in OBSOLETE_I_MODELS:
        obsolete = model_dir / f"{name}.tflite"
        if obsolete.exists():
            obsolete.unlink()
    p_records = []
    for name in P_MODELS:
        source = entropy_export / f"{name}_fp32.tflite"
        if not source.is_file():
            raise FileNotFoundError(f"missing entropy TFLite: {source}")
        target = model_dir / f"{name}.tflite"
        shutil.copy2(source, target)
        p_records.append({"name": name, "file": f"models/{target.name}", "sha256": sha256(target)})

    merged_record = {
        "name": "i_entropy_prior_merged",
        "file": "models/i_entropy_prior_merged.tflite",
        "sha256": sha256(i_merged_model),
        "packaging": "standalone_not_in_archive",
        "input_names": ["i_y_pre_prior"],
        "input_shapes_nhwc": [[1, 16, 32, 256]],
        "output_names": [
            "i_z_hat",
            "i_y_q_w_0", "i_y_q_w_1", "i_y_q_w_2", "i_y_q_w_3",
            "i_s_w_0", "i_s_w_1", "i_s_w_2", "i_s_w_3",
            "i_y_hat",
        ],
        "output_shapes_nhwc": [
            [1, 4, 8, 128],
            *[[1, 16, 32, 64] for _ in range(8)],
            [1, 16, 32, 256],
        ],
    }

    print("[large-online-entropy] extracting I/P CDF tables", flush=True)
    i_model, _, i_checkpoint_sha = load_i_model(source_root)
    i_model = i_model.cpu().eval()
    p_model, _, p_checkpoint_sha = load_p_model(source_root)
    p_model = p_model.cpu().eval()
    # CDF tables are built at runtime by the source models and are not stored
    # in the checkpoints. This must happen before get_cdf_info().
    i_model.update(force_zero_thres=args.force_zero_thres)
    p_model.update(force_zero_thres=args.force_zero_thres)
    i_gaussian = write_cdf(output, "i_gaussian", i_model.gaussian_encoder.get_cdf_info())
    i_z = write_cdf(output, "i_z", i_model.bit_estimator_z.get_cdf_info())
    p_gaussian = write_cdf(output, "p_gaussian", p_model.gaussian_encoder.get_cdf_info())
    p_z = write_cdf(output, "p_z", p_model.bit_estimator_z.get_cdf_info())
    force_zero = float(args.force_zero_thres)
    manifest = {
        "package": "gvc-rt-large-tflite-online-entropy-270p-qp0",
        "qp": args.qp,
        "layout": "NHWC FP32 external TFLite I/O; native entropy state is NCHW",
        "force_zero_thres": force_zero,
        "checkpoint_sha256": {"i": i_checkpoint_sha, "p": p_checkpoint_sha},
        "i": {
            "input_i_frame": "inputs/frame_000_i_encoder/input_i_frame.f32le",
            "merged_model": merged_record,
            "gaussian": i_gaussian,
            "z": i_z,
            "z_start_offset": args.qp * 128,
            "z_per_channel_size": 32,
        },
        "p": {
            "input_p_frame": "inputs/frame_001_p_encoder/input_p_frame.f32le",
            "gaussian": p_gaussian,
            "z": p_z,
            "z_start_offset": args.qp * 128,
            "z_per_channel_size": 32,
            "models": p_records,
        },
    }
    (output / "large_entropy_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# GVC-RT Large 在线熵编码包\n\n"
        "该压缩包在 Large 在线 TFLite 包基础上增加 P 帧 entropy/prior 图和 I/P CDF 表。\n\n"
        "I 帧只使用独立文件 `i_entropy_prior_merged.tflite`，该文件不进入本压缩包。"
        "部署时将它单独放到包目录的 `models/i_entropy_prior_merged.tflite`。"
        "旧的 7 张 I 帧 entropy/prior 分图已停止打包。rANS 保持原生实现。\n",
        encoding="utf-8",
    )
    write_checksums(output)
    print(f"wrote {output}")
    print("standalone_i_model=i_entropy_prior_merged sha256=" + merged_record["sha256"])
    print("packaged_p_models=" + ",".join(P_MODELS))
    print(f"checkpoint_sha256_i={i_checkpoint_sha}")
    print(f"checkpoint_sha256_p={p_checkpoint_sha}")


if __name__ == "__main__":
    main()
