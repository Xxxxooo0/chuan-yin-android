#!/usr/bin/env python3
"""Export source-derived I-frame prior NPU candidates for MTK Neuron.

Run on the Linux GVC-RT server.  This exporter deliberately keeps all
discrete entropy behavior out of TFLite: the generated graphs only produce the
continuous tensors needed by the 4x spatial prior.  Android native code owns
masking, rounding, CDF selection, and rANS so the bitstream contract remains
observable and testable.
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

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEIGHT = 16
WIDTH = 32
Y_CHANNELS = 256
COMMON_CHANNELS = 514
UNSUPPORTED = re.compile(r"unsupported|not support|fallback|Float32 input|Float32 output", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_tool(explicit: str | None, name: str) -> str:
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return str(path)
        raise FileNotFoundError(f"{name} does not exist: {path}")
    found = shutil.which(name)
    if found:
        return found
    candidate = Path(sys.executable).resolve().parent / name
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError(f"{name} is not in PATH or current Python bin")


def find_ncc(explicit: str | None, sdk_root: Path | None, platform: str) -> str | None:
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return str(path)
        raise FileNotFoundError(f"ncc-tflite does not exist: {path}")
    if sdk_root is None:
        return shutil.which("ncc-tflite")
    root = sdk_root.resolve()
    candidates = (
        root / "neuron_sdk" / "host" / "bin" / "ncc-tflite",
        root / "neuron_sdk" / platform / "bin" / "ncc-tflite",
        root / "host" / "bin" / "ncc-tflite",
        root / platform / "bin" / "ncc-tflite",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    matches = sorted(root.rglob("ncc-tflite"))
    return str(matches[0]) if matches else None


def ncc_env(ncc: str) -> dict[str, str]:
    env = os.environ.copy()
    binary = Path(ncc).resolve()
    roots = [binary.parent, *list(binary.parents)[:5]]
    libraries: list[Path] = []
    for root in roots:
        for suffix in ("lib", "lib64", "host/lib", "host/lib64", "neuron_sdk/host/lib"):
            path = root / suffix
            if path.is_dir() and path not in libraries:
                libraries.append(path)
    inherited = [entry for entry in env.get("LD_LIBRARY_PATH", "").split(":") if entry]
    env["LD_LIBRARY_PATH"] = ":".join([*(str(path) for path in libraries), *inherited])
    return env


def run(command: list[str], log: Path, env: dict[str, str] | None = None) -> tuple[int, str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    text = "$ " + " ".join(command) + "\n\n" + result.stdout
    log.write_text(text, encoding="utf-8")
    return result.returncode, text


def source_imports(source_root: Path) -> None:
    sys.path.insert(0, str(source_root))
    import src.layers.cuda_inference as cuda_inference
    import src.layers.layers as layers
    import src.models.image_model_G_b as image_model

    cuda_inference.CUSTOMIZED_CUDA_INFERENCE = False
    layers.CUSTOMIZED_CUDA_INFERENCE = False
    image_model.CUSTOMIZED_CUDA_INFERENCE = False


def load_model(source_root: Path, force_zero_thres: float | None):
    source_imports(source_root)
    from src.models.image_model_G_b import DMCI

    if not torch.cuda.is_available():
        raise RuntimeError("FP16 I prior export requires a CUDA server; CPU torch does not implement sigmoid(Half)")
    device = torch.device("cuda")
    checkpoint = source_root / "ckpt" / "checkpoints" / "GVC-RT_B_I.pt"
    model = DMCI(encoder_ckpt_path=str(checkpoint)).to(device).eval()
    model.update(force_zero_thres)
    return model, checkpoint, device


class PriorReduceNhwc(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.reduction = model.y_spatial_prior_reduction

    def forward(self, i_common_params_nhwc):
        common = i_common_params_nhwc.permute(0, 3, 1, 2).contiguous()
        return self.reduction(common).permute(0, 2, 3, 1).contiguous()


class PriorStageNhwc(nn.Module):
    def __init__(self, model, stage: int):
        super().__init__()
        if stage not in (1, 2, 3):
            raise ValueError(f"invalid stage {stage}")
        self.adaptor = getattr(model, f"y_spatial_prior_adaptor_{stage}")
        self.spatial_prior = model.y_spatial_prior

    def forward(self, i_y_hat_so_far_nhwc, i_reduced_common_params_nhwc):
        y_hat = i_y_hat_so_far_nhwc.permute(0, 3, 1, 2).contiguous()
        reduced = i_reduced_common_params_nhwc.permute(0, 3, 1, 2).contiguous()
        scales, means = self.spatial_prior(self.adaptor(torch.cat((y_hat, reduced), dim=1))).chunk(2, 1)
        return (
            scales.permute(0, 2, 3, 1).contiguous(),
            means.permute(0, 2, 3, 1).contiguous(),
        )


def export_onnx(module: nn.Module, path: Path, inputs: tuple[torch.Tensor, ...], input_names: list[str], output_names: list[str], opset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            module.eval(), inputs, str(path), input_names=input_names, output_names=output_names,
            opset_version=opset, do_constant_folding=True,
        )


def convert(onnx: Path, tflite: Path, converter: str, input_names: list[str], input_shapes: list[tuple[int, ...]], output_names: list[str]) -> tuple[int, str, Path]:
    log = tflite.parent / "logs" / f"{tflite.stem}_convert.log"
    command = [
        converter, "--input_model_file", str(onnx), "--output_file", str(tflite),
        "--output_file_format", "tflite", "--input_names", ",".join(input_names),
        "--input_shapes", ":".join(",".join(str(dim) for dim in shape) for shape in input_shapes),
        "--output_names", ",".join(output_names), "--tflite_op_export_spec", "builtin_first",
        "--convert_float32_weights_to_float16", "True",
    ]
    rc, text = run(command, log)
    return rc, text, log


def check_ncc(tflite: Path, ncc: str | None, arch: str) -> dict[str, Any]:
    if ncc is None:
        return {"ncc_check": "skipped", "ncc_eligible": False}
    logs = tflite.parent / "logs"
    check_log = logs / f"{tflite.stem}_check_target.log"
    plan_log = logs / f"{tflite.stem}_exec_plan.log"
    flags = ["--arch", arch, "--opt-bw", "--relax-fp32"]
    check_rc, check_text = run([ncc, str(tflite), *flags, "--check-target-only"], check_log, ncc_env(ncc))
    plan_rc, plan_text = run([ncc, str(tflite), *flags, "--show-exec-plan", "--show-memory-summary"], plan_log, ncc_env(ncc))
    diagnostics = [line.strip() for line in (check_text + "\n" + plan_text).splitlines() if UNSUPPORTED.search(line)]
    eligible = check_rc == 0 and plan_rc == 0 and not diagnostics
    return {
        "ncc_check": "ok" if eligible else "warn",
        "ncc_eligible": eligible,
        "check_target_rc": check_rc,
        "exec_plan_rc": plan_rc,
        "check_target_log": str(check_log),
        "exec_plan_log": str(plan_log),
        "diagnostic_lines": diagnostics[:80],
    }


def save_f32(path: Path, value: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy().astype("<f4").tofile(path)


def write_trace(model, y: torch.Tensor, common: torch.Tensor, output: Path, force_zero_thres: float | None) -> None:
    q_enc, q_dec, scales, means = model.separate_prior(common, False)
    reduced = model.y_spatial_prior_reduction(common)
    masks = model.get_mask_4x(1, Y_CHANNELS, HEIGHT, WIDTH, y.dtype, y.device)
    y_scaled = y * q_enc
    y_hat_so_far = torch.zeros_like(y)
    save_f32(output / "reduced_common_params.f32le", reduced)
    save_f32(output / "stage0_scales.f32le", scales)
    save_f32(output / "stage0_means.f32le", means)
    for stage in range(4):
        if stage:
            adaptor = getattr(model, f"y_spatial_prior_adaptor_{stage}")
            scales, means = model.y_spatial_prior(adaptor(torch.cat((y_hat_so_far, reduced), dim=1))).chunk(2, 1)
            save_f32(output / f"stage{stage}_scales.f32le", scales)
            save_f32(output / f"stage{stage}_means.f32le", means)
        _, y_q, y_hat, s_hat = model.process_with_mask(y_scaled, scales, means, masks[stage])
        y_hat_so_far = y_hat if stage == 0 else y_hat_so_far + y_hat
        save_f32(output / f"i_y_q_w_{stage}.f32le", model.single_part_for_writing_4x(y_q))
        save_f32(output / f"i_s_w_{stage}.f32le", model.single_part_for_writing_4x(s_hat))
        save_f32(output / f"i_y_hat_so_far_{stage}.f32le", y_hat_so_far)
    save_f32(output / "i_y_hat.f32le", y_hat_so_far * q_dec)
    fixtures = {"ties_to_even": {"-1.5": -2.0, "-0.5": 0.0, "0.5": 0.0, "1.5": 2.0}}
    (output / "rounding_fixture.json").write_text(json.dumps(fixtures, indent=2), encoding="utf-8")


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
    parser.add_argument("--force-zero-thres", type=float, default=0.12)
    parser.add_argument("--i-y-f32le", type=Path, required=True)
    parser.add_argument("--i-common-params-f32le", type=Path, required=True)
    parser.add_argument("--copy-assets", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    android_root = args.android_root.resolve()
    output_dir = (args.output_dir or android_root / "outputs" / "i_prior_npu").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = find_tool(args.onnx_converter, "mtk_onnx_converter")
    ncc = find_ncc(args.ncc_tflite, args.sdk_root, args.platform)
    model, checkpoint, device = load_model(source_root, args.force_zero_thres)

    y = torch.from_numpy(np.fromfile(args.i_y_f32le, dtype="<f4").reshape(1, Y_CHANNELS, HEIGHT, WIDTH)).to(device)
    common = torch.from_numpy(np.fromfile(args.i_common_params_f32le, dtype="<f4").reshape(1, COMMON_CHANNELS, HEIGHT, WIDTH)).to(device)
    with torch.no_grad():
        reduced = model.y_spatial_prior_reduction(common)
    reduced_channels = int(reduced.shape[1])
    # This MTK converter release rejects ONNX TensorProto.FLOAT16 (KeyError: 10).
    # Keep ONNX tensors FP32, then let the converter lower only constant weights
    # to FP16. NCC diagnostics decide whether its FP32 external IO is acceptable.
    export_model = model.eval()
    records: list[dict[str, Any]] = []
    definitions = [
        ("i_prior_reduce_fp16_weight", PriorReduceNhwc(export_model), (torch.zeros((1, HEIGHT, WIDTH, COMMON_CHANNELS), dtype=torch.float32, device=device),), ["i_common_params_nhwc"], ["i_reduced_common_params_nhwc"]),
        *[
            (f"i_prior_stage{stage}_fp16_weight", PriorStageNhwc(export_model, stage), (torch.zeros((1, HEIGHT, WIDTH, Y_CHANNELS), dtype=torch.float32, device=device), torch.zeros((1, HEIGHT, WIDTH, reduced_channels), dtype=torch.float32, device=device)), ["i_y_hat_so_far_nhwc", "i_reduced_common_params_nhwc"], ["i_scales_nhwc", "i_means_nhwc"])
            for stage in (1, 2, 3)
        ],
    ]
    for name, module, samples, input_names, output_names in definitions:
        onnx = output_dir / f"{name}.onnx"
        tflite = output_dir / f"{name}.tflite"
        export_onnx(module, onnx, samples, input_names, output_names, args.opset)
        rc, text, log = convert(onnx, tflite, converter, input_names, [tuple(sample.shape) for sample in samples], output_names)
        record: dict[str, Any] = {
            "name": name, "onnx": str(onnx), "onnx_sha256": sha256(onnx), "tflite": str(tflite) if tflite.is_file() else None,
            "tflite_sha256": sha256(tflite) if tflite.is_file() else None, "converter_rc": rc, "converter_log": str(log),
            "converter_diagnostics": [line.strip() for line in text.splitlines() if UNSUPPORTED.search(line)][:80],
            "input_names": input_names, "input_shapes": [list(sample.shape) for sample in samples], "output_names": output_names,
        }
        if rc == 0 and tflite.is_file():
            record.update(check_ncc(tflite, ncc, args.arch))
        else:
            record.update({"ncc_check": "not_run", "ncc_eligible": False})
        records.append(record)
    trace_dir = output_dir / "baseline"
    write_trace(model, y, common, trace_dir, args.force_zero_thres)
    manifest = {
        "tool": Path(__file__).name, "checkpoint_sha256": sha256(checkpoint), "source_root": str(source_root),
        "qp": 0, "frame_shape": [1, 3, 256, 512], "y_shape_nchw": [1, Y_CHANNELS, HEIGHT, WIDTH],
        "common_params_shape_nchw": [1, COMMON_CHANNELS, HEIGHT, WIDTH], "reduced_channels": reduced_channels,
        "force_zero_thres": args.force_zero_thres, "layout": "NHWC", "onnx_io_dtype": "float32", "tflite_weight_precision": "float16",
        "android_runtime_compatible": False,
        "ncc_tflite": ncc, "records": records,
    }
    manifest_path = output_dir / "i_prior_npu_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.copy_assets:
        print("skipped asset copy: FP32 ONNX IO candidates are NCC diagnostics only")
    print(f"wrote {manifest_path}")
    for record in records:
        print(f"{record['name']} converter_rc={record['converter_rc']} ncc={record.get('ncc_check')} eligible={record.get('ncc_eligible')}")


if __name__ == "__main__":
    main()
