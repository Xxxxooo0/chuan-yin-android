#!/usr/bin/env python3
"""Export the P recon tail as an NHWC-IO TFLite candidate for MTK testing.

Run this script on the Linux server. It exports a single diagnostic graph from
the live GVC-RT source tree:

    p_stage2_nhwc + p_codeword_nhwc + q_recon -> p_recon_frame_nhwc

The wrapper keeps the mathematically real P recon tail while making the
external contract NHWC, which is the first controlled experiment for the MTK GPU
layout requirement. This script does not run inference; it only exports,
converts, and optionally checks the generated TFLite with ncc-tflite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEGMENT = "p_recon_nhwc_upsample_stage3_stage4_final"

INPUTS = (
    ("p_stage2_nhwc", (1, 16, 32, 512)),
    ("p_codeword_nhwc", (1, 16, 32, 18)),
    ("q_recon", (1, 1, 1, 1)),
)
OUTPUTS = (("p_recon_frame_nhwc", (1, 256, 512, 3)),)

UNSUPPORTED_PATTERNS = (
    re.compile(r"unsupported[^\n]*", re.IGNORECASE),
    re.compile(r"not support[^\n]*", re.IGNORECASE),
    re.compile(r"fail(?:ed|ure)?[^\n]*", re.IGNORECASE),
    re.compile(r"fallback[^\n]*", re.IGNORECASE),
    re.compile(r"software[^\n]*", re.IGNORECASE),
    re.compile(r"Float32 input[^\n]*", re.IGNORECASE),
    re.compile(r"Float32 output[^\n]*", re.IGNORECASE),
    re.compile(r"Transpose[^\n]*", re.IGNORECASE),
    re.compile(r"Reshape[^\n]*", re.IGNORECASE),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shape_text(shape: tuple[int, ...]) -> str:
    return ",".join(str(dim) for dim in shape)


def run_command(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    text = "$ " + " ".join(command) + "\n\n" + result.stdout
    log_path.write_text(text, encoding="utf-8")
    return result.returncode, text


def find_tool(explicit: str | None, name: str) -> str:
    if explicit:
        candidate = Path(explicit)
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(f"{name} does not exist: {candidate}")
    existing = shutil.which(name)
    if existing:
        return existing
    candidate = Path(sys.executable).resolve().parent / name
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError(f"{name} is not in PATH or current Python bin")


def find_ncc(explicit: str | None, sdk_root: Path | None, platform: str) -> str | None:
    if explicit:
        candidate = Path(explicit)
        if candidate.is_file():
            candidate.chmod(candidate.stat().st_mode | 0o111)
            return str(candidate)
        raise FileNotFoundError(f"ncc-tflite does not exist: {candidate}")
    if sdk_root:
        root = sdk_root.resolve()
        candidates = [
            root / "neuron_sdk" / "host" / "bin" / "ncc-tflite",
            root / "neuron_sdk" / platform / "bin" / "ncc-tflite",
            root / "host" / "bin" / "ncc-tflite",
            root / platform / "bin" / "ncc-tflite",
        ]
        for candidate in candidates:
            if candidate.is_file():
                candidate.chmod(candidate.stat().st_mode | 0o111)
                return str(candidate)
        matches = sorted(path for path in root.rglob("ncc-tflite") if path.is_file())
        if matches:
            matches[0].chmod(matches[0].stat().st_mode | 0o111)
            return str(matches[0])
        raise FileNotFoundError(f"missing ncc-tflite under {root}")
    existing = shutil.which("ncc-tflite")
    return existing


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
    existing = [Path(path) for path in env.get("LD_LIBRARY_PATH", "").split(":") if path]
    env["LD_LIBRARY_PATH"] = ":".join(str(path) for path in library_dirs + existing)
    return env


def interesting_lines(text: str) -> list[str]:
    lines: list[str] = []
    for line in text.splitlines():
        if any(pattern.search(line) for pattern in UNSUPPORTED_PATTERNS):
            lines.append(line.strip())
    return lines[:80]


def force_exportable_torch_path(source_root: Path) -> None:
    sys.path.insert(0, str(source_root))
    import src.layers.cuda_inference as cuda_inference
    import src.layers.layers as layers
    import src.models.image_model_G_b as image_model
    import src.models.video_model_G_b as video_model

    cuda_inference.CUSTOMIZED_CUDA_INFERENCE = False
    layers.CUSTOMIZED_CUDA_INFERENCE = False
    image_model.CUSTOMIZED_CUDA_INFERENCE = False
    video_model.CUSTOMIZED_CUDA_INFERENCE = False


def load_p_model(source_root: Path):
    force_exportable_torch_path(source_root)
    from src.models.video_model_G_b import DMC

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = source_root / "ckpt" / "checkpoints" / "GVC-RT_B_P.pt"
    model = DMC().to(device).eval()
    checkpoint = torch.load(ckpt, map_location="cpu")
    state_dict = checkpoint.get(
        "student_ema",
        checkpoint.get("student", checkpoint.get("state_dict", checkpoint)),
    )
    model.load_state_dict(state_dict, strict=True)
    return model, device, sha256(ckpt)


class PReconNhwcUpsampleStage3Stage4Final(nn.Module):
    """Real P recon tail with NHWC external tensors.

    Internally this wrapper reuses the source model modules, so precision should
    match the PyTorch tail once compared on the server. The outer NHWC contract
    is intended to test whether MTK conversion can reduce layout pressure for
    GPU/Neuron backends.
    """

    def __init__(self, model):
        super().__init__()
        decoder = model.recon_generation_net.decoder
        self.ada3 = decoder.ada3
        self.upsample = decoder.upsample
        self.stage3 = decoder.stage3
        self.ada4 = decoder.ada4
        self.stage4 = decoder.stage4
        self.ada_final = decoder.ada_final
        self.head = decoder.head

    def forward(self, p_stage2_nhwc, p_codeword_nhwc, q_recon):
        p_stage2 = p_stage2_nhwc.permute(0, 3, 1, 2).contiguous()
        p_codeword = p_codeword_nhwc.permute(0, 3, 1, 2).contiguous()
        out = self.ada3(p_stage2, p_codeword)
        out = self.upsample(out)
        out = self.stage3(out)
        out = self.ada4(out, p_codeword)
        out = self.stage4(out, q_recon)
        out = self.ada_final(out, p_codeword)
        out = self.head(out)
        out = torch.nn.functional.pixel_shuffle(out, 8)
        out = torch.clamp(out, -1.0, 1.0)
        return out.permute(0, 2, 3, 1).contiguous()


def export_onnx(source_root: Path, output_dir: Path, opset: int) -> dict[str, Any]:
    model, device, checkpoint_sha = load_p_model(source_root)
    module = PReconNhwcUpsampleStage3Stage4Final(model).to(device).eval()
    samples = tuple(torch.zeros(shape, dtype=torch.float32, device=device) for _, shape in INPUTS)
    onnx_path = output_dir / f"{SEGMENT}.onnx"
    with torch.no_grad():
        torch.onnx.export(
            module,
            samples,
            str(onnx_path),
            input_names=[name for name, _ in INPUTS],
            output_names=[name for name, _ in OUTPUTS],
            opset_version=opset,
            do_constant_folding=True,
        )
    return {
        "onnx": str(onnx_path),
        "onnx_sha256": sha256(onnx_path),
        "checkpoint_sha256": checkpoint_sha,
        "input_names": ",".join(name for name, _ in INPUTS),
        "input_shapes": ":".join(shape_text(shape) for _, shape in INPUTS),
        "output_names": ",".join(name for name, _ in OUTPUTS),
        "output_shapes": ":".join(shape_text(shape) for _, shape in OUTPUTS),
    }


def convert_tflite(record: dict[str, Any], output_dir: Path, converter: str, variant: str) -> dict[str, Any]:
    tflite_path = output_dir / f"{SEGMENT}_{variant}.tflite"
    log_path = output_dir / "logs" / f"{SEGMENT}_{variant}_onnx_converter.log"
    command = [
        converter,
        "--input_model_file",
        record["onnx"],
        "--output_file",
        str(tflite_path),
        "--output_file_format",
        "tflite",
        "--input_names",
        record["input_names"],
        "--input_shapes",
        record["input_shapes"],
        "--output_names",
        record["output_names"],
        "--tflite_op_export_spec",
        "builtin_first",
    ]
    if variant == "fp16_weight":
        command += ["--convert_float32_weights_to_float16", "True"]
    returncode, text = run_command(command, log_path)
    if returncode != 0:
        raise RuntimeError(f"mtk_onnx_converter failed rc={returncode}; see {log_path}")
    return {
        "variant": variant,
        "tflite": str(tflite_path),
        "tflite_sha256": sha256(tflite_path),
        "converter_log": str(log_path),
        "converter_diagnostic_lines": interesting_lines(text),
    }


def run_ncc_checks(tflite: Path, output_dir: Path, ncc: str, arch: str, ncc_flags: list[str]) -> dict[str, Any]:
    logs_dir = output_dir / "logs"
    label = tflite.stem
    check_log = logs_dir / f"{label}_check_target.log"
    plan_log = logs_dir / f"{label}_exec_plan.log"
    check_cmd = [ncc, str(tflite), "--arch", arch, *ncc_flags, "--check-target-only"]
    plan_cmd = [
        ncc,
        str(tflite),
        "--arch",
        arch,
        *ncc_flags,
        "--show-exec-plan",
        "--show-memory-summary",
    ]
    check_rc, check_text = run_command(check_cmd, check_log, env=ncc_runtime_env(ncc))
    plan_rc, plan_text = run_command(plan_cmd, plan_log, env=ncc_runtime_env(ncc))
    return {
        "ncc_arch": arch,
        "ncc_flags": ncc_flags,
        "check_target_rc": check_rc,
        "exec_plan_rc": plan_rc,
        "check_target_log": str(check_log),
        "exec_plan_log": str(plan_log),
        "diagnostic_lines": interesting_lines(check_text + "\n" + plan_text),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--android-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--onnx-converter", default=None)
    parser.add_argument("--sdk-root", type=Path, default=None)
    parser.add_argument("--ncc-tflite", default=None)
    parser.add_argument("--platform", default="mt6899")
    parser.add_argument("--arch", default="mdla5.3")
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--variant", choices=("fp32", "fp16_weight", "all"), default="all")
    parser.add_argument("--opt-bw", action="store_true")
    parser.add_argument("--relax-fp32", action="store_true")
    parser.add_argument("--extra-ncc-flag", action="append", default=[])
    parser.add_argument("--skip-ncc", action="store_true")
    parser.add_argument("--copy-assets", action="store_true")
    args = parser.parse_args()

    android_root = args.android_root.resolve()
    output_dir = (args.output_dir or android_root / "outputs" / "recon_nhwc_tail").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = find_tool(args.onnx_converter, "mtk_onnx_converter")
    variants = ("fp32", "fp16_weight") if args.variant == "all" else (args.variant,)
    ncc_flags = list(args.extra_ncc_flag)
    if args.opt_bw:
        ncc_flags.append("--opt-bw")
    if args.relax_fp32:
        ncc_flags.append("--relax-fp32")
    ncc = None if args.skip_ncc else find_ncc(args.ncc_tflite, args.sdk_root, args.platform)

    print(f"[nhwc-tail] segment={SEGMENT}")
    print(f"[nhwc-tail] source_root={args.source_root.resolve()}")
    print(f"[nhwc-tail] output_dir={output_dir}")
    print(f"[nhwc-tail] converter={converter}")
    print(f"[nhwc-tail] ncc={ncc or 'skipped'}")

    records: list[dict[str, Any]] = []
    base_record = export_onnx(args.source_root.resolve(), output_dir, args.opset)
    for variant in variants:
        record: dict[str, Any] = {"segment": SEGMENT, "variant": variant, **base_record}
        try:
            record.update(convert_tflite(base_record, output_dir, converter, variant))
            record["tflite_convert"] = "ok"
        except Exception as exc:
            record["tflite_convert"] = "failed"
            record["status"] = "failed"
            record["error"] = str(exc)
            print(f"[nhwc-tail] failed convert {variant}: {exc}")
            records.append(record)
            continue

        if ncc:
            try:
                record.update(run_ncc_checks(Path(record["tflite"]), output_dir, ncc, args.arch, ncc_flags))
                record["ncc_check"] = "ok" if record["check_target_rc"] == 0 and record["exec_plan_rc"] == 0 else "warn"
            except Exception as exc:
                record["ncc_check"] = "failed"
                record["ncc_error"] = str(exc)
                print(f"[nhwc-tail] failed ncc {variant}: {exc}")
        else:
            record["ncc_check"] = "skipped"
        record["status"] = "ok" if record.get("tflite_convert") == "ok" else "failed"
        records.append(record)

    manifest = {
        "tool": Path(__file__).name,
        "segment": SEGMENT,
        "layout_contract": "NHWC input/output; source model internals are reused from PyTorch",
        "source_root": str(args.source_root.resolve()),
        "output_dir": str(output_dir),
        "converter": converter,
        "ncc_tflite": ncc,
        "arch": args.arch,
        "opset": args.opset,
        "variants": list(variants),
        "records": records,
    }
    manifest_path = output_dir / "recon_nhwc_tail_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.copy_assets:
        assets_dir = android_root / "app" / "src" / "main" / "assets" / "recon_diagnostic"
        assets_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            if record.get("tflite_convert") == "ok":
                shutil.copy2(record["tflite"], assets_dir / Path(record["tflite"]).name)
        shutil.copy2(manifest_path, assets_dir / manifest_path.name)

    print(f"[nhwc-tail] wrote {manifest_path}")
    for record in records:
        print(
            f"[nhwc-tail] {record['variant']} convert={record.get('tflite_convert')} "
            f"ncc={record.get('ncc_check')} tflite={record.get('tflite')}"
        )
        for line in record.get("diagnostic_lines", [])[:8]:
            print(f"  {line}")


if __name__ == "__main__":
    main()
