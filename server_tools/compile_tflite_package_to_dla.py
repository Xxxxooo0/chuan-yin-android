#!/usr/bin/env python3
"""Compile an existing enterprise TFLite package into an offline DLA package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: Sequence[str], log_path: Path, env: Dict[str, str]) -> Tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        check=False,
    )
    output = completed.stdout or ""
    log_path.write_text("$ " + " ".join(command) + "\n\n" + output, encoding="utf-8")
    return completed.returncode, output


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError("missing {}: {}".format(label, path))
    return path


def expected_target(arch: str) -> str:
    if not arch.startswith("mdla"):
        raise ValueError("arch must use the mdlaX.Y form: {}".format(arch))
    return "MDLA_" + arch[len("mdla") :].replace(".", "_")


def compile_model(
    source_tflite: Path,
    output_dla: Path,
    log_dir: Path,
    ncc: Path,
    ncc_env: Dict[str, str],
    arch: str,
) -> Dict[str, Any]:
    flags = ["--arch", arch, "--opt-bw", "--relax-fp32"]
    output_dla.parent.mkdir(parents=True, exist_ok=True)
    if output_dla.exists():
        output_dla.unlink()

    with tempfile.TemporaryDirectory(prefix="gvc-rt-dla-") as temporary:
        temporary_dir = Path(temporary)
        temporary_tflite = temporary_dir / "model.tflite"
        temporary_dla = temporary_dir / "model.dla"
        shutil.copy2(str(source_tflite), str(temporary_tflite))

        check_rc, check_output = run(
            [str(ncc), str(temporary_tflite)] + flags + ["--check-target-only"],
            log_dir / "check_target.log",
            ncc_env,
        )
        plan_rc, plan_output = run(
            [str(ncc), str(temporary_tflite)] + flags + ["--show-exec-plan", "--show-memory-summary"],
            log_dir / "execution_plan.log",
            ncc_env,
        )
        compile_rc, compile_output = run(
            [str(ncc), str(temporary_tflite)]
            + flags
            + [
                "--gen-debug-info",
                "--show-exec-plan",
                "--show-memory-summary",
                "-d",
                str(temporary_dla),
            ],
            log_dir / "compile_dla.log",
            ncc_env,
        )
        if compile_rc == 0 and temporary_dla.is_file():
            shutil.copy2(str(temporary_dla), str(output_dla))

    target_lines = [line.strip() for line in plan_output.splitlines() if "Target:" in line]
    required_target = expected_target(arch)
    mdla_only = bool(target_lines) and all(required_target in line for line in target_lines)
    ok = check_rc == 0 and plan_rc == 0 and compile_rc == 0 and output_dla.is_file() and mdla_only
    return {
        "check_rc": check_rc,
        "plan_rc": plan_rc,
        "compile_rc": compile_rc,
        "execution_targets": target_lines,
        "mdla_only": mdla_only,
        "offline_compile_ok": ok,
        "diagnostic_tail": (check_output + "\n" + plan_output + "\n" + compile_output)[-4000:],
    }


def build_readme(source_manifest: Dict[str, Any], target: str) -> str:
    reset = source_manifest.get("reference_reset", {})
    return """# GVC-RT-Small DLA 模型包

## 固定配置

- 分辨率：256 x 512
- QP index：{qp}
- Tensor 布局：NHWC
- 外部 Tensor 类型：FP32
- 离线编译目标：{target}
- 参考重置：`{condition}`

## 调用流程

1. 第 0 帧及满足重置条件的帧，调用 `temporal_from_frame.dla`。
2. 其他帧调用 `temporal_from_feature.dla`。
3. 当前帧与 `ctx` 输入 `encoder.dla`，得到 `latent_y`。
4. `latent_y + ctx + memory` 输入 `decoder.dla`，得到下一参考特征和重建帧。

模型输入输出名称、shape、文件 SHA256 以 `manifest.json` 为准。
""".format(
        qp=source_manifest.get("fixed_q_index"),
        target=target,
        condition=reset.get("condition", "见 manifest.json"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite-package-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--package-name", required=True)
    parser.add_argument("--ncc-tflite", type=Path, required=True)
    parser.add_argument("--ncc-lib-dir", type=Path, required=True)
    parser.add_argument("--arch", default="mdla5.0")
    args = parser.parse_args()

    source_dir = args.tflite_package_dir.resolve()
    source_manifest_path = require_file(source_dir / "manifest.json", "source manifest")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    records = source_manifest.get("models", [])
    required_names = {"temporal_from_frame", "temporal_from_feature", "encoder", "decoder"}
    actual_names = {record.get("name") for record in records}
    if actual_names != required_names:
        raise RuntimeError("expected four Small models {}, got {}".format(sorted(required_names), sorted(actual_names)))
    if source_manifest.get("fixed_q_index") != 9:
        raise RuntimeError("expected phone Small fixed_q_index=9")

    ncc = require_file(args.ncc_tflite.resolve(), "ncc-tflite")
    ncc_lib_dir = args.ncc_lib_dir.resolve()
    if not ncc_lib_dir.is_dir():
        raise FileNotFoundError("missing NCC library directory: {}".format(ncc_lib_dir))

    output_dir = args.output_dir.resolve()
    work_dir = output_dir / "work"
    package_dir = output_dir / args.package_name
    if work_dir.exists():
        shutil.rmtree(str(work_dir))
    if package_dir.exists():
        shutil.rmtree(str(package_dir))
    models_dir = package_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    ncc_env = os.environ.copy()
    ncc_env["LD_LIBRARY_PATH"] = str(ncc_lib_dir) + os.pathsep + ncc_env.get("LD_LIBRARY_PATH", "")
    published: List[Dict[str, Any]] = []
    build_records: List[Dict[str, Any]] = []

    for index, source_record in enumerate(records, start=1):
        name = source_record["name"]
        source_tflite = require_file(source_dir / source_record["file"], name + " TFLite")
        actual_sha = sha256(source_tflite)
        if actual_sha != source_record.get("sha256"):
            raise RuntimeError("{} TFLite SHA mismatch: {} != {}".format(name, actual_sha, source_record.get("sha256")))
        if source_tflite.stat().st_size != source_record.get("bytes"):
            raise RuntimeError("{} TFLite size mismatch".format(name))

        print("[small-dla] {}/{} compile {}".format(index, len(records), name), flush=True)
        dla_path = models_dir / (name + ".dla")
        result = compile_model(
            source_tflite,
            dla_path,
            work_dir / "logs" / name,
            ncc,
            ncc_env,
            args.arch,
        )
        build_record = {
            "name": name,
            "source_tflite": str(source_tflite),
            "source_tflite_bytes": source_tflite.stat().st_size,
            "source_tflite_sha256": actual_sha,
            **result,
        }
        if dla_path.is_file():
            build_record.update({"dla": str(dla_path), "dla_bytes": dla_path.stat().st_size, "dla_sha256": sha256(dla_path)})
        build_records.append(build_record)
        print(
            "[small-dla] {} check={} plan={} compile={} mdla_only={} status={}".format(
                name,
                result["check_rc"],
                result["plan_rc"],
                result["compile_rc"],
                result["mdla_only"],
                "ok" if result["offline_compile_ok"] else "failed",
            ),
            flush=True,
        )
        if not result["offline_compile_ok"]:
            continue

        published_record = dict(source_record)
        published_record.update(
            {
                "file": "models/" + dla_path.name,
                "bytes": dla_path.stat().st_size,
                "sha256": sha256(dla_path),
                "source_tflite_bytes": source_tflite.stat().st_size,
                "source_tflite_sha256": actual_sha,
                "offline_compile_verified": True,
                "mdla_only": True,
                "execution_targets": result["execution_targets"],
            }
        )
        published.append(published_record)

    build_manifest = {
        "tool": Path(__file__).name,
        "source_package": source_manifest.get("package"),
        "source_manifest_sha256": sha256(source_manifest_path),
        "arch": args.arch,
        "expected_execution_target": expected_target(args.arch),
        "records": build_records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dla_build_manifest.json").write_text(json.dumps(build_manifest, indent=2), encoding="utf-8")
    if len(published) != len(records):
        failed = [record["name"] for record in build_records if not record["offline_compile_ok"]]
        raise RuntimeError("DLA compilation failed: {}".format(", ".join(failed)))

    target_label = "MDLA " + args.arch[len("mdla") :]
    delivery_manifest = dict(source_manifest)
    delivery_manifest.update(
        {
            "package": args.package_name,
            "format": "DLA offline model",
            "target": target_label,
            "compiler": "NeuroPilot SDK 7.0.8 ncc-tflite",
            "source_tflite_package": source_manifest.get("package"),
            "source_tflite_manifest_sha256": sha256(source_manifest_path),
            "models": published,
        }
    )
    manifest_path = package_dir / "manifest.json"
    manifest_path.write_text(json.dumps(delivery_manifest, indent=2), encoding="utf-8")
    readme_path = package_dir / "README.md"
    readme_path.write_text(build_readme(source_manifest, target_label), encoding="utf-8")
    checksum_paths = sorted(list(models_dir.glob("*.dla")) + [manifest_path, readme_path])
    (package_dir / "SHA256SUMS.txt").write_text(
        "".join("{}  {}\n".format(sha256(path), path.relative_to(package_dir).as_posix()) for path in checksum_paths),
        encoding="ascii",
    )

    archive = output_dir / (args.package_name + ".tar.gz")
    if archive.exists():
        archive.unlink()
    with tarfile.open(str(archive), "w:gz") as handle:
        handle.add(str(package_dir), arcname=args.package_name)
    print("package={}".format(package_dir), flush=True)
    print("archive={}".format(archive), flush=True)
    print("archive_sha256={}".format(sha256(archive)), flush=True)


if __name__ == "__main__":
    main()
