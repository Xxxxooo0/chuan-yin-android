#!/usr/bin/env python3
"""Export direct-NHWC P recon segments and compile offline DLA models.

The external tensors are NHWC while the wrapped source modules keep their
original NCHW implementation. MediaTek's PyTorch converter can fold these
fixed boundary permutations, avoiding the Transpose operators produced by the
older ONNX conversion route.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch import nn

from analyze_recon_neuron_support import analyze_one, find_ncc
from export_recon_diagnostic import PROJECT_ROOT, SEGMENTS, find_tool, load_p_model, sha256


DEFAULT_SEGMENTS = (
    "p_latent_decoder",
    "p_recon_mlp_dcb0",
    "p_recon_mlp_dcb1",
    "p_decoder_stage1_conv_only",
    "p_decoder_stage2_blocks_only",
    "p_upsampler_original",
    "p_decoder_stage3_blocks_only",
    "p_decoder_stage4_blocks_explicit",
    "p_recon_final_head_no_ada",
)


def nchw_to_nhwc_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) != 4:
        raise ValueError(f"only fixed 4D tensors are supported, got {shape}")
    n, c, h, w = shape
    return n, h, w, c


class NhwcBoundary(nn.Module):
    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.source = source

    def forward(self, *inputs_nhwc: torch.Tensor):
        inputs_nchw = tuple(
            value.permute(0, 3, 1, 2).contiguous() for value in inputs_nhwc
        )
        outputs = self.source(*inputs_nchw)
        if isinstance(outputs, tuple):
            return tuple(value.permute(0, 2, 3, 1).contiguous() for value in outputs)
        return outputs.permute(0, 2, 3, 1).contiguous()


def run(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        return subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        ).returncode


def parse_segments(raw: str) -> list[str]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [name for name in names if name not in DEFAULT_SEGMENTS]
    if unknown:
        raise ValueError(f"unsupported --segments entries: {unknown}")
    if not names:
        raise ValueError("--segments must not be empty")
    return names


def build_source(name: str, model: nn.Module, qp: int) -> nn.Module:
    module_cls = SEGMENTS[name]["module"]
    if name == "p_latent_decoder":
        return module_cls(model, qp).cpu().eval()
    return module_cls(model).cpu().eval()


def export_one(
    name: str,
    model: nn.Module,
    checkpoint_sha256: str,
    qp: int,
    converter: str,
    ncc: str,
    arch: str,
    output_dir: Path,
) -> dict[str, Any]:
    spec = SEGMENTS[name]
    input_shapes_nchw = [shape for _, shape in spec["inputs"]]
    output_shapes_nchw = [shape for _, shape in spec["outputs"]]
    input_shapes_nhwc = [nchw_to_nhwc_shape(shape) for shape in input_shapes_nchw]
    expected_output_shapes_nhwc = [nchw_to_nhwc_shape(shape) for shape in output_shapes_nchw]
    samples = tuple(torch.zeros(shape, dtype=torch.float32) for shape in input_shapes_nhwc)
    module = NhwcBoundary(build_source(name, model, qp)).cpu().eval()
    label = f"{name}_direct_nhwc_fp32"
    logs_dir = output_dir / "logs"

    script_path = output_dir / f"{label}.pt"
    with torch.no_grad():
        scripted = torch.jit.trace(module, samples, strict=False)
        actual = scripted(*samples)
        actual_outputs = actual if isinstance(actual, tuple) else (actual,)
        actual_output_shapes = [list(value.shape) for value in actual_outputs]
        scripted.save(str(script_path))

    tflite_path = output_dir / f"{label}.tflite"
    converter_log = logs_dir / f"{label}_converter.log"
    converter_command = [
        converter,
        "--input_script_module_file", str(script_path),
        "--output_file", str(tflite_path),
        "--input_shapes", ":".join(
            ",".join(str(dim) for dim in shape) for shape in input_shapes_nhwc
        ),
    ]
    converter_rc = run(converter_command, converter_log)

    record: dict[str, Any] = {
        "name": label,
        "source_segment": name,
        "route": "pytorch_torchscript_direct_nhwc",
        "checkpoint_sha256": checkpoint_sha256,
        "qp": qp,
        "input_names": [item[0] for item in spec["inputs"]],
        "input_shapes_nchw": [list(shape) for shape in input_shapes_nchw],
        "input_shapes_nhwc": [list(shape) for shape in input_shapes_nhwc],
        "output_names": [item[0] for item in spec["outputs"]],
        "output_shapes_nchw": [list(shape) for shape in output_shapes_nchw],
        "expected_output_shapes_nhwc": [list(shape) for shape in expected_output_shapes_nhwc],
        "actual_torchscript_output_shapes_nhwc": actual_output_shapes,
        "script_module": str(script_path),
        "script_module_sha256": sha256(script_path),
        "converter_command": converter_command,
        "converter_rc": converter_rc,
        "converter_log": str(converter_log),
        "tflite": str(tflite_path) if tflite_path.is_file() else None,
        "tflite_sha256": sha256(tflite_path) if tflite_path.is_file() else None,
    }
    if converter_rc != 0 or not tflite_path.is_file():
        record["status"] = "converter_failed"
        return record

    ncc_record = analyze_one(
        tflite=tflite_path,
        ncc=ncc,
        arch=arch,
        output_dir=output_dir / "ncc",
        compile_dla=True,
        ncc_flags=["--opt-bw", "--relax-fp32"],
    )
    diagnostics = ncc_record.get("diagnostic_lines", []) + ncc_record.get(
        "dla_diagnostic_lines", []
    )
    record["ncc"] = ncc_record
    record["transpose_warning"] = any("transpose" in line.lower() for line in diagnostics)
    record["offline_compile_ok"] = (
        ncc_record.get("check_target_rc") == 0
        and ncc_record.get("exec_plan_rc") == 0
        and ncc_record.get("compile_dla_rc") == 0
        and bool(ncc_record.get("dla"))
        and Path(ncc_record["dla"]).is_file()
    )
    record["status"] = "ok" if record["offline_compile_ok"] else "ncc_failed"
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--android-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--segments", default=",".join(DEFAULT_SEGMENTS))
    parser.add_argument("--pytorch-converter", default=None)
    parser.add_argument("--ncc-tflite", required=True)
    parser.add_argument("--arch", default="mdla5.3")
    parser.add_argument("--qp", type=int, default=0)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    android_root = args.android_root.resolve()
    output_dir = (
        args.output_dir or android_root / "outputs" / "ip_recon_offline_nhwc"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = find_tool(args.pytorch_converter, "mtk_pytorch_converter")
    ncc = find_ncc(args.ncc_tflite)
    selected = parse_segments(args.segments)

    model, _, checkpoint_sha256 = load_p_model(source_root)
    model = model.cpu().eval()
    records = []
    for index, name in enumerate(selected, start=1):
        print(f"[offline-nhwc] {index}/{len(selected)} export {name}", flush=True)
        try:
            record = export_one(
                name,
                model,
                checkpoint_sha256,
                args.qp,
                converter,
                ncc,
                args.arch,
                output_dir,
            )
        except Exception as exc:
            record = {"source_segment": name, "status": "exception", "error": repr(exc)}
        records.append(record)
        print(
            f"[offline-nhwc] {name} status={record['status']} "
            f"transpose_warning={record.get('transpose_warning')}",
            flush=True,
        )

    manifest = {
        "tool": Path(__file__).name,
        "source_root": str(source_root),
        "checkpoint_sha256": checkpoint_sha256,
        "qp": args.qp,
        "layout": "NHWC external I/O; source operators unchanged",
        "dtype": "float32 external I/O; NCC --relax-fp32 enabled",
        "arch": args.arch,
        "ncc_flags": ["--opt-bw", "--relax-fp32", "--gen-debug-info"],
        "records": records,
        "summary": {
            "selected": len(records),
            "offline_compile_ok": sum(1 for item in records if item.get("offline_compile_ok")),
            "without_transpose_warning": sum(
                1
                for item in records
                if item.get("offline_compile_ok") and not item.get("transpose_warning")
            ),
            "failed": sum(1 for item in records if item.get("status") != "ok"),
        },
    }
    manifest_path = output_dir / "ip_recon_offline_nhwc_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")
    print(json.dumps(manifest["summary"], indent=2))

    if manifest["summary"]["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
