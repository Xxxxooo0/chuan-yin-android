#!/usr/bin/env python3
"""Run MediaTek ncc-tflite diagnostics for recon TFLite assets.

This script does not run PyTorch, ONNX Runtime, or model inference. It only
invokes Neuron Compiler diagnostic modes on existing .tflite files so recon
failures can be attributed to unsupported OPs, execution-plan splits, and memory
pressure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_FOCUS = (
    "i_recon_full_fp32",
    "p_recon_full_fp32",
    "i_latent_decoder_fp32",
    "p_latent_decoder_fp32",
    "p_recon_mlp_conv_only_fp32",
    "p_recon_feature_to_codeword_fp32",
    "p_recon_unshuffle_only_fp32",
    "p_recon_unshuffle_conv_only_fp32",
    "p_recon_unshuffle_spacetodepth_only_fp32",
    "p_recon_unshuffle_mtk_nchw_only_fp32",
    "p_recon_unshuffle_tflite_spacetodepth_nhwc_fp32",
    "p_recon_unshuffle_tflite_spacetodepth_nchw_wrap_fp32",
    "p_recon_unshuffle_tflite_spacetodepth_nhwc_fp16",
    "p_recon_unshuffle_tflite_spacetodepth_nchw_wrap_fp16",
    "p_recon_unshuffle_tflite_spacetodepth_nhwc_fp16_cast",
    "p_recon_unshuffle_tflite_spacetodepth_nchw_wrap_fp16_cast",
    "p_recon_unshuffle_tflite_spacetodepth_nhwc_int8",
    "p_recon_mlp_full_fp32",
    "p_recon_mlp_norm0_fp32",
    "p_recon_mlp_dcb0_fp32",
    "p_recon_mlp_norm1_fp32",
    "p_recon_mlp_silu_fp32",
    "p_recon_mlp_dcb1_fp32",
    "p_recon_mlp_dcb0_adaptor_fp32",
    "p_recon_mlp_dcb0_dc_fp32",
    "p_recon_mlp_dcb0_dc_add_fp32",
    "p_recon_mlp_dcb0_ffn_fp32",
    "p_recon_mlp_dcb0_ffn_add_fp32",
    "p_recon_mlp_dcb1_adaptor_fp32",
    "p_recon_mlp_dcb1_dc_fp32",
    "p_recon_mlp_dcb1_dc_add_fp32",
    "p_recon_mlp_dcb1_ffn_fp32",
    "p_recon_mlp_dcb1_ffn_add_fp32",
    "p_recon_mlp_norm0_dcb0_fp32",
    "p_recon_mlp_norm1_silu_dcb1_fp32",
    "p_recon_feature_to_codeword_conv_fp32",
    "p_recon_feature_to_codeword_spacetodepth_fp32",
    "p_recon_feature_to_codeword_mtk_nchw_fp32",
    "p_decoder_stage1_conv_only_fp32",
    "p_decoder_stage1_full_fp32",
    "p_decoder_stage2_full_fp32",
    "p_decoder_stage3_blocks_only_fp16_weight",
    "p_decoder_upsample_stage3_full_fp32",
    "p_decoder_stage4_full_fp32",
    "p_upsampler_original_fp32",
    "p_upsampler_pixelshuffle_fp32",
    "p_recon_prefix_stage1_fp32",
    "p_recon_prefix_stage2_fp32",
    "p_recon_prefix_upsample_fp32",
    "p_recon_prefix_stage3_fp32",
    "p_recon_prefix_stage4_fp32",
    "p_recon_final_head_fp32",
    "p_groupnorm_probe_fp32",
    "p_adagn_probe_fp32",
    "i_fast_codeword_to_frame_probe_fp32",
    "i_fast_codeword_to_frame_1block_probe_fp32",
    "i_fast_codeword_to_frame_2block_probe_fp32",
    "i_fast_codeword_to_frame_4block_probe_fp32",
    "p_fast_feature_to_frame_probe_fp32",
)


UNSUPPORTED_PATTERNS = (
    re.compile(r"unsupported[^\n]*", re.IGNORECASE),
    re.compile(r"not support[^\n]*", re.IGNORECASE),
    re.compile(r"fail(?:ed|ure)?[^\n]*", re.IGNORECASE),
    re.compile(r"fallback[^\n]*", re.IGNORECASE),
    re.compile(r"software[^\n]*", re.IGNORECASE),
    re.compile(r"not found[^\n]*", re.IGNORECASE),
    re.compile(r"no such file[^\n]*", re.IGNORECASE),
    re.compile(r"cannot open shared object file[^\n]*", re.IGNORECASE),
    re.compile(r"permission denied[^\n]*", re.IGNORECASE),
    re.compile(r"command not found[^\n]*", re.IGNORECASE),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_ncc(explicit: str | None) -> str:
    if explicit:
        candidate = Path(explicit)
        if candidate.is_file():
            candidate.chmod(candidate.stat().st_mode | 0o111)
            return str(candidate)
        if candidate.is_dir():
            host_candidate = candidate / "neuron_sdk" / "host" / "bin" / "ncc-tflite"
            if host_candidate.is_file():
                host_candidate.chmod(host_candidate.stat().st_mode | 0o111)
                return str(host_candidate)
            matches = sorted(
                path for path in candidate.rglob("ncc-tflite") if path.is_file()
            )
            if matches:
                matches[0].chmod(matches[0].stat().st_mode | 0o111)
                return str(matches[0])
            raise FileNotFoundError(f"ncc-tflite was not found under directory: {candidate}")
        raise FileNotFoundError(
            f"ncc-tflite does not exist: {candidate}. "
            "Pass either the binary path or the SDK root directory."
        )
    found = shutil.which("ncc-tflite")
    if found:
        return found
    raise FileNotFoundError("ncc-tflite is not in PATH; pass --ncc-tflite")


def ncc_runtime_env(ncc: str) -> dict[str, str]:
    env = os.environ.copy()
    ncc_path = Path(ncc).resolve()
    roots = [ncc_path.parent]
    roots.extend(list(ncc_path.parents)[:5])
    suffixes = (
        Path("."),
        Path("lib"),
        Path("lib64"),
        Path("host/lib"),
        Path("host/lib64"),
        Path("neuron_sdk/host/lib"),
        Path("neuron_sdk/host/lib64"),
    )
    library_dirs: list[Path] = []
    for root in roots:
        for suffix in suffixes:
            candidate = root / suffix
            if candidate.is_dir() and candidate not in library_dirs:
                library_dirs.append(candidate)
    for root in roots[:3]:
        for pattern in ("libc++.so*", "libLLVM*.so*", "libomp*.so*"):
            for library in root.rglob(pattern):
                parent = library.parent
                if parent not in library_dirs:
                    library_dirs.append(parent)
    existing = [Path(path) for path in env.get("LD_LIBRARY_PATH", "").split(":") if path]
    env["LD_LIBRARY_PATH"] = ":".join(str(path) for path in library_dirs + existing)
    return env


def parse_labels(raw: str) -> list[str] | None:
    if raw == "all":
        return None
    if raw == "focus":
        return list(DEFAULT_FOCUS)
    labels = [item.strip() for item in raw.split(",") if item.strip()]
    if not labels:
        raise ValueError("--labels must be all, focus, or a comma-separated label list")
    return labels


def collect_assets(assets_dir: Path, labels: list[str] | None) -> list[Path]:
    if labels is None:
        return sorted(assets_dir.glob("*.tflite"))
    missing: list[str] = []
    selected: list[Path] = []
    for label in labels:
        path = assets_dir / f"{label}.tflite"
        if path.is_file():
            selected.append(path)
        else:
            missing.append(label)
    if missing:
        raise FileNotFoundError(f"missing recon diagnostic TFLite assets: {missing}")
    return selected


def run_ncc(command: list[str], log_path: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=ncc_runtime_env(command[0]),
        )
        returncode = result.returncode
        output = result.stdout
    except OSError as exc:
        returncode = -1
        output = (
            f"OSError: {exc}\n\n"
            "If this is Exec format error, use neuron_sdk/host/bin/ncc-tflite "
            "on the x86_64 server instead of neuron_sdk/mt*/bin/ncc-tflite.\n"
        )
    text = "$ " + " ".join(command) + "\n\n" + output
    log_path.write_text(text, encoding="utf-8")
    return returncode, text


def interesting_lines(text: str) -> list[str]:
    lines: list[str] = []
    current_op: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^OP\[\d+\]:", stripped):
            current_op = stripped
            continue
        if "Target:" in stripped and stripped not in lines:
            lines.append(stripped)
        if any(pattern.search(line) for pattern in UNSUPPORTED_PATTERNS):
            if current_op and current_op not in lines:
                lines.append(current_op)
            if stripped not in lines:
                lines.append(stripped)
    return lines[:80]


def analyze_one(
    tflite: Path,
    ncc: str,
    arch: str,
    output_dir: Path,
    compile_dla: bool,
    ncc_flags: list[str],
) -> dict[str, Any]:
    label = tflite.stem
    logs_dir = output_dir / "logs"
    support_log = logs_dir / f"{label}_check_target.log"
    plan_log = logs_dir / f"{label}_exec_plan.log"
    memory_log = logs_dir / f"{label}_memory_summary.log"

    support_cmd = [ncc, str(tflite), "--arch", arch, *ncc_flags, "--check-target-only"]
    support_rc, support_text = run_ncc(support_cmd, support_log)

    plan_cmd = [
        ncc,
        str(tflite),
        "--arch",
        arch,
        *ncc_flags,
        "--show-exec-plan",
        "--show-memory-summary",
    ]
    plan_rc, plan_text = run_ncc(plan_cmd, plan_log)
    memory_log.write_text(plan_text, encoding="utf-8")

    record: dict[str, Any] = {
        "label": label,
        "tflite": str(tflite),
        "tflite_sha256": sha256(tflite),
        "arch": arch,
        "check_target_rc": support_rc,
        "exec_plan_rc": plan_rc,
        "check_target_log": str(support_log),
        "exec_plan_log": str(plan_log),
        "memory_summary_log": str(memory_log),
        "diagnostic_lines": interesting_lines(support_text + "\n" + plan_text),
    }

    if compile_dla:
        dla_path = output_dir / "dla" / f"{label}.dla"
        dla_path.parent.mkdir(parents=True, exist_ok=True)
        dla_log = logs_dir / f"{label}_compile_dla.log"
        dla_cmd = [
            ncc,
            str(tflite),
            "--arch",
            arch,
            *ncc_flags,
            "--gen-debug-info",
            "--show-exec-plan",
            "--show-memory-summary",
            "-d",
            str(dla_path),
        ]
        dla_rc, dla_text = run_ncc(dla_cmd, dla_log)
        record.update(
            {
                "compile_dla_rc": dla_rc,
                "compile_dla_log": str(dla_log),
                "dla": str(dla_path) if dla_path.is_file() else None,
                "dla_sha256": sha256(dla_path) if dla_path.is_file() else None,
                "dla_diagnostic_lines": interesting_lines(dla_text),
            }
        )

    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--android-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--assets-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--ncc-tflite", default=None)
    parser.add_argument("--arch", default="mdla5.3")
    parser.add_argument("--labels", default="focus", help="focus, all, or comma-separated labels")
    parser.add_argument("--compile-dla", action="store_true")
    parser.add_argument("--opt-bw", action="store_true", help="Pass --opt-bw to ncc-tflite")
    parser.add_argument("--relax-fp32", action="store_true", help="Pass --relax-fp32 to ncc-tflite")
    parser.add_argument(
        "--extra-ncc-flag",
        action="append",
        default=[],
        help="Extra raw flag passed through to ncc-tflite; can be repeated",
    )
    args = parser.parse_args()

    android_root = args.android_root.resolve()
    assets_dir = (
        args.assets_dir
        or android_root / "app" / "src" / "mtkOffline" / "conversion_inputs" / "recon_diagnostic"
    ).resolve()
    output_dir = (args.output_dir or android_root / "outputs" / "recon_neuron_diagnostics").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ncc = find_ncc(args.ncc_tflite)
    tflites = collect_assets(assets_dir, parse_labels(args.labels))
    ncc_flags = list(args.extra_ncc_flag)
    if args.opt_bw:
        ncc_flags.append("--opt-bw")
    if args.relax_fp32:
        ncc_flags.append("--relax-fp32")
    records = [
        analyze_one(path, ncc, args.arch, output_dir, args.compile_dla, ncc_flags)
        for path in tflites
    ]
    summary = {
        "tool": "ncc-tflite",
        "ncc_tflite": ncc,
        "arch": args.arch,
        "ncc_flags": ncc_flags,
        "assets_dir": str(assets_dir),
        "output_dir": str(output_dir),
        "compile_dla": args.compile_dla,
        "records": records,
    }
    summary_path = output_dir / "recon_neuron_diagnostics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    failed = [record for record in records if record["check_target_rc"] != 0 or record["exec_plan_rc"] != 0]
    print(f"wrote {summary_path}")
    print(f"checked={len(records)} failed_or_warn={len(failed)}")
    for record in records:
        status = "OK" if record["check_target_rc"] == 0 and record["exec_plan_rc"] == 0 else "CHECK"
        print(f"{status} {record['label']} check_rc={record['check_target_rc']} plan_rc={record['exec_plan_rc']}")
        for line in record["diagnostic_lines"][:6]:
            print(f"  {line}")


if __name__ == "__main__":
    main()
