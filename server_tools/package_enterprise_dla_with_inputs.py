#!/usr/bin/env python3
"""Combine an enterprise DLA package with input-only precision vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive_path: Path, output_dir: Path) -> Path:
    output_dir_resolved = output_dir.resolve()
    with tarfile.open(str(archive_path), "r:gz") as archive:
        for member in archive.getmembers():
            target = (output_dir / member.name).resolve()
            if target != output_dir_resolved and output_dir_resolved not in target.parents:
                raise ValueError("unsafe archive path: {}".format(member.name))
        archive.extractall(str(output_dir))
    roots = [path for path in output_dir.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise RuntimeError("archive must contain exactly one root directory: {}".format(archive_path))
    return roots[0]


def output_spec(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": record["name"],
        "shape": record["shape"],
        "dtype": "float32_le",
        "elements": record["elements"],
        "bytes": record["bytes"],
        "vendor_file": record["vendor_file"],
    }


def build_readme(package_name: str) -> str:
    return """# {package_name}

该交付包包含离线 DLA 模型，以及部署精度采集所需的固定输入 Tensor。所有
Tensor 均为 NHWC 布局、小端 FP32 存储。

对 `input_manifest.json` 中的每个 stage，按以下步骤执行：

1. 加载 `model` 指定的 DLA。
2. 读取列出的输入文件，不改变 shape、layout 或 dtype。
3. 调用一次 DLA。
4. 按 `vendor_file` 的相对路径保存全部输出。

请回传完整的厂商输出目录以便内部比对。本包不包含服务器参考输出和精度报告。
""".format(package_name=package_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-package", type=Path, required=True)
    parser.add_argument("--precision-package", type=Path, required=True)
    parser.add_argument("--package-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_package = args.model_package.resolve()
    precision_package = args.precision_package.resolve()
    output = args.output.resolve()
    if not model_package.is_file():
        raise FileNotFoundError(model_package)
    if not precision_package.is_file():
        raise FileNotFoundError(precision_package)
    if output.exists():
        raise FileExistsError("output already exists: {}".format(output))

    with tempfile.TemporaryDirectory(prefix="gvc_rt_delivery_") as temporary:
        temp_root = Path(temporary)
        model_root = safe_extract(model_package, temp_root / "model")
        precision_root = safe_extract(precision_package, temp_root / "precision")
        precision_manifest_path = precision_root / "precision_manifest.json"
        precision_manifest = json.loads(precision_manifest_path.read_text(encoding="utf-8"))

        delivery_root = temp_root / args.package_name
        shutil.copytree(str(model_root), str(delivery_root))
        model_manifest_path = delivery_root / "manifest.json"
        original_model_manifest_sha = sha256(model_manifest_path)
        model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        model_manifest["package"] = args.package_name
        model_manifest["input_manifest"] = "input_manifest.json"
        model_manifest_path.write_text(json.dumps(model_manifest, indent=2), encoding="utf-8")

        input_stages: List[Dict[str, Any]] = []
        copied_inputs = 0
        for stage in precision_manifest["stages"]:
            stage_inputs = []
            for record in stage["inputs"]:
                source = precision_root / record["file"]
                if not source.is_file():
                    raise FileNotFoundError(source)
                if source.stat().st_size != record["bytes"] or sha256(source) != record["sha256"]:
                    raise RuntimeError("precision input integrity failure: {}".format(source))
                relative = Path("inputs") / stage["id"] / (record["name"] + ".f32le")
                target = delivery_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(source), str(target))
                stage_inputs.append(
                    {
                        "name": record["name"],
                        "file": relative.as_posix(),
                        "shape": record["shape"],
                        "dtype": "float32_le",
                        "elements": record["elements"],
                        "bytes": record["bytes"],
                        "sha256": sha256(target),
                    }
                )
                copied_inputs += 1
            input_stages.append(
                {
                    "order": stage["order"],
                    "id": stage["id"],
                    "model": stage["model"],
                    "inputs": stage_inputs,
                    "outputs": [output_spec(record) for record in stage["expected_outputs"]],
                }
            )

        input_manifest = {
            "schema_version": 1,
            "package": args.package_name,
            "purpose": "vendor_dla_precision_input_collection",
            "resolution": precision_manifest["resolution"],
            "qp": precision_manifest["qp"],
            "layout": precision_manifest["layout"],
            "dtype": precision_manifest["dtype"],
            "frame_count": precision_manifest["frame_count"],
            "frame_value_range": precision_manifest["frame_value_range"],
            "source_model_package_sha256": sha256(model_package),
            "source_model_manifest_sha256": original_model_manifest_sha,
            "delivery_models": precision_manifest.get("delivery_models", []),
            "stages": input_stages,
        }
        (delivery_root / "input_manifest.json").write_text(
            json.dumps(input_manifest, indent=2), encoding="utf-8"
        )
        (delivery_root / "README.md").write_text(build_readme(args.package_name), encoding="utf-8")

        forbidden = [
            path
            for path in delivery_root.rglob("*")
            if path.is_file()
            and ("expected" in path.parts or path.name in {"precision_manifest.json", "compare_outputs.py"})
        ]
        if forbidden:
            raise RuntimeError("reference files leaked into delivery: {}".format(forbidden))

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

    print("package={}".format(args.package_name))
    print("stages={}".format(len(input_stages)))
    print("input_tensors={}".format(copied_inputs))
    print("archive={}".format(output))
    print("archive_sha256={}".format(sha256(output)))


if __name__ == "__main__":
    main()
