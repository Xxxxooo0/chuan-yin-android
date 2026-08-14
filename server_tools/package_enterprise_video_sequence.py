#!/usr/bin/env python3
"""Package verified DLA models with a chained video-sequence input."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image


HEIGHT = 256
WIDTH = 512
SHAPE = [1, HEIGHT, WIDTH, 3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive_path: Path, output_dir: Path) -> Path:
    output_root = output_dir.resolve()
    with tarfile.open(str(archive_path), "r:gz") as archive:
        for member in archive.getmembers():
            target = (output_dir / member.name).resolve()
            if target != output_root and output_root not in target.parents:
                raise ValueError("unsafe archive path: {}".format(member.name))
        archive.extractall(str(output_dir))
    roots = [path for path in output_dir.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise RuntimeError("model archive must contain one root directory")
    return roots[0]


def frame_spec(model: str) -> Tuple[float, float, str]:
    if model == "large":
        return -1.0, 1.0, "rgb / 255 * 2 - 1"
    return 0.0, 1.0, "rgb / 255"


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(str(path)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if list(rgb.shape) != [HEIGHT, WIDTH, 3]:
        raise RuntimeError(
            "{} shape={} expected={}".format(path, list(rgb.shape), [HEIGHT, WIDTH, 3])
        )
    return rgb


def build_flow(model: str, model_manifest: Dict[str, Any]) -> Dict[str, Any]:
    if model == "large":
        return {
            "frame_0000": ["i_encoder", "i_decoder"],
            "frame_0001": ["temporal_from_frame", "p_encoder", "p_decoder"],
            "frame_0002_and_later": [
                "temporal_from_feature",
                "p_encoder",
                "p_decoder",
            ],
            "state_rule": (
                "frame 0 i_reference_frame feeds temporal_from_frame; each "
                "p_reference_feature feeds the next temporal_from_feature"
            ),
            "reconstruction_output": {
                "frame_0000": "i_decoder.i_reference_frame",
                "frame_0001_and_later": "p_decoder.p_reference_frame",
            },
        }
    model_names = {record.get("name") for record in model_manifest.get("models", [])}
    required_names = {"temporal_from_frame", "temporal_from_feature", "encoder", "decoder"}
    if model_names != required_names:
        raise RuntimeError(
            "Small sequence package requires four-model QP9 flow; got {}".format(
                sorted(model_names)
            )
        )
    reset = model_manifest.get("reference_reset", {})
    if reset.get("period") != 64 or reset.get("phase") != 1:
        raise RuntimeError("Small sequence package requires period=64, phase=1 reset")
    return {
        "frame_0000": ["temporal_from_frame", "encoder", "decoder"],
        "reset_frames": {
            "condition": "frame_index > 0 and frame_index % 64 == 1",
            "stages": ["temporal_from_frame", "encoder", "decoder"],
            "temporal_input": "previous decoder.reconstructed_frame",
        },
        "other_frames": ["temporal_from_feature", "encoder", "decoder"],
        "initial_state": model_manifest.get("initial_reference"),
        "state_rule": (
            "decoder.next_ref_feature feeds the next temporal_from_feature; "
            "decoder.reconstructed_frame feeds temporal_from_frame on reset frames"
        ),
        "reconstruction_output": "decoder.reconstructed_frame",
    }


def build_readme(package_name: str, model: str, frame_count: int) -> str:
    value_range = "[-1,1]" if model == "large" else "[0,1]"
    return """# {package_name}

本包用于连续视频序列 DLA 测试，包含已经验证的 NeuroPilot 7.0.8 / MDLA 5.0
模型和 ParkScene 前 {frame_count} 帧输入。模型文件没有修改或重新编译。

## 输入

- 尺寸：`256 x 512`
- 布局：`NHWC`
- 类型：小端 `FP32`
- 输入范围：`{value_range}`
- `sequence/frames_f32le/`：DLA 的权威输入
- `sequence/frames_png/`：用于人工核对的 RGB 图片

按照 `sequence_manifest.json` 的 `execution_flow` 连续执行，必须将上一帧返回的
参考状态传给下一帧，不能对每一帧重新初始化。

## 必须保存并回传

每一帧都保存以下结果：

- `vendor_outputs/reconstructed_tensors/frame_NNNN.f32le`
- `vendor_outputs/reconstructed_png/frame_NNNN.png`
- `vendor_outputs/timing.csv`

重建 tensor 必须保持模型原始输出范围、NHWC shape `[1,256,512,3]` 和小端
FP32，不要在保存 tensor 前做裁剪、量化或颜色通道交换。PNG 仅用于查看。
`timing.csv` 至少包含 `frame_index,frame_type,total_ms`，并尽量记录每张 DLA
的单独耗时。
""".format(
        package_name=package_name,
        model=model,
        frame_count=frame_count,
        value_range=value_range,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("large", "small"), required=True)
    parser.add_argument("--model-package", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--package-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_package = args.model_package.resolve()
    sequence_dir = args.sequence_dir.resolve()
    output = args.output.resolve()
    if not model_package.is_file():
        raise FileNotFoundError(model_package)
    if not sequence_dir.is_dir():
        raise FileNotFoundError(sequence_dir)
    if output.exists():
        raise FileExistsError("output already exists: {}".format(output))
    if args.frame_count < 2:
        raise ValueError("frame-count must be at least 2")

    frame_paths = sorted(sequence_dir.glob("*.png"))[: args.frame_count]
    if len(frame_paths) != args.frame_count:
        raise RuntimeError(
            "found {} PNG frames, expected {}".format(len(frame_paths), args.frame_count)
        )

    value_min, value_max, normalization = frame_spec(args.model)
    with tempfile.TemporaryDirectory(prefix="gvc_rt_video_sequence_") as temporary:
        temporary_root = Path(temporary)
        model_root = safe_extract(model_package, temporary_root / "model")
        delivery_root = temporary_root / args.package_name
        shutil.copytree(str(model_root), str(delivery_root))

        model_manifest_path = delivery_root / "manifest.json"
        model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        if model_manifest.get("resolution") != {"height": HEIGHT, "width": WIDTH}:
            raise RuntimeError("model package resolution is not 256x512")
        if model_manifest.get("layout") != "NHWC":
            raise RuntimeError("model package layout is not NHWC")
        model_manifest["package"] = args.package_name
        model_manifest["source_model_package_sha256"] = sha256(model_package)
        model_manifest["sequence_manifest"] = "sequence_manifest.json"
        model_manifest_path.write_text(
            json.dumps(model_manifest, indent=2), encoding="utf-8"
        )

        tensor_dir = delivery_root / "sequence" / "frames_f32le"
        png_dir = delivery_root / "sequence" / "frames_png"
        tensor_dir.mkdir(parents=True)
        png_dir.mkdir(parents=True)
        frames: List[Dict[str, Any]] = []
        for index, source_path in enumerate(frame_paths):
            rgb = load_rgb(source_path)
            normalized = rgb.astype(np.float32) / 255.0
            if args.model == "large":
                normalized = normalized * 2.0 - 1.0
            normalized = normalized.reshape(SHAPE).astype("<f4", copy=False)

            name = "frame_{:04d}".format(index)
            tensor_path = tensor_dir / (name + ".f32le")
            png_path = png_dir / (name + ".png")
            normalized.tofile(str(tensor_path))
            Image.fromarray(rgb, mode="RGB").save(str(png_path))
            frame_record = {
                    "index": index,
                    "type": "I" if args.model == "large" and index == 0 else "P",
                    "source_file": source_path.name,
                    "input_tensor": tensor_path.relative_to(delivery_root).as_posix(),
                    "input_png": png_path.relative_to(delivery_root).as_posix(),
                    "shape": SHAPE,
                    "dtype": "float32_le",
                    "bytes": tensor_path.stat().st_size,
                    "sha256": sha256(tensor_path),
                }
            if args.model == "small":
                reset_reference = index == 0 or index % 64 == 1
                frame_record.update(
                    {
                        "reference_reset": reset_reference,
                        "temporal_model": (
                            "temporal_from_frame" if reset_reference else "temporal_from_feature"
                        ),
                    }
                )
            frames.append(frame_record)

        sequence_manifest = {
            "schema_version": 1,
            "package": args.package_name,
            "purpose": "vendor_chained_video_sequence_test",
            "model": "gvc-rt-{}".format(args.model),
            "model_package_sha256": sha256(model_package),
            "sequence": sequence_dir.name,
            "frame_count": args.frame_count,
            "fps": args.fps,
            "resolution": {"height": HEIGHT, "width": WIDTH},
            "layout": "NHWC",
            "input_dtype": "float32_le",
            "input_value_range": [value_min, value_max],
            "normalization": normalization,
            "execution_flow": build_flow(args.model, model_manifest),
            "frames": frames,
            "required_vendor_outputs": {
                "tensor_pattern": "vendor_outputs/reconstructed_tensors/frame_NNNN.f32le",
                "tensor_shape": SHAPE,
                "tensor_dtype": "float32_le",
                "tensor_value_range": [value_min, value_max],
                "png_pattern": "vendor_outputs/reconstructed_png/frame_NNNN.png",
                "timing_csv": "vendor_outputs/timing.csv",
            },
        }
        (delivery_root / "sequence_manifest.json").write_text(
            json.dumps(sequence_manifest, indent=2), encoding="utf-8"
        )
        (delivery_root / "README.md").write_text(
            build_readme(args.package_name, args.model, args.frame_count),
            encoding="utf-8",
        )

        checksum_path = delivery_root / "SHA256SUMS.txt"
        checksum_files = sorted(
            path for path in delivery_root.rglob("*") if path.is_file() and path != checksum_path
        )
        checksum_path.write_text(
            "".join(
                "{}  {}\n".format(sha256(path), path.relative_to(delivery_root).as_posix())
                for path in checksum_files
            ),
            encoding="ascii",
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(str(output), "w:gz") as archive:
            archive.add(str(delivery_root), arcname=args.package_name)

    print("model={}".format(args.model))
    print("frames={}".format(args.frame_count))
    print("reconstruction_tensor_pattern=vendor_outputs/reconstructed_tensors/frame_NNNN.f32le")
    print("reconstruction_png_pattern=vendor_outputs/reconstructed_png/frame_NNNN.png")
    print("archive={}".format(output))
    print("archive_sha256={}".format(sha256(output)))


if __name__ == "__main__":
    main()
