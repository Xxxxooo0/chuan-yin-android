#!/usr/bin/env python3
"""Export the continuous stage1 network of the P-frame 2x spatial prior.

P prior stage0 remains native because it performs mask/quantization. This
script exports only stage1's real SpatialPrior network for Neuron AUTO probes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn

from export_i_prior_npu_tflite import (
    HEIGHT,
    PROJECT_ROOT,
    WIDTH,
    check_ncc,
    export_onnx,
    find_ncc,
    find_tool,
    sha256,
)
from export_recon_nhwc_tail import load_p_model


Y_CHANNELS = 128
COMMON_CHANNELS = 384
NAME = "p_prior_stage1_fp16_weight"


class PPriorStage1Nhwc(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.spatial_prior = model.y_spatial_prior

    def forward(self, p_y_hat_so_far_nhwc, p_common_params_nhwc):
        y_hat = p_y_hat_so_far_nhwc.permute(0, 3, 1, 2).contiguous()
        common = p_common_params_nhwc.permute(0, 3, 1, 2).contiguous()
        scales, means = self.spatial_prior(torch.cat((y_hat, common), dim=1)).chunk(2, 1)
        return (
            scales.permute(0, 2, 3, 1).contiguous(),
            means.permute(0, 2, 3, 1).contiguous(),
        )


def convert(
    onnx: Path,
    tflite: Path,
    converter: str,
    input_names: list[str],
    input_shapes: list[tuple[int, ...]],
    output_names: list[str],
) -> tuple[int, str, Path]:
    from export_i_prior_npu_tflite import run

    log = tflite.parent / "logs" / f"{tflite.stem}_convert.log"
    command = [
        converter,
        "--input_model_file", str(onnx),
        "--output_file", str(tflite),
        "--output_file_format", "tflite",
        "--input_names", ",".join(input_names),
        "--input_shapes", ":".join(",".join(str(dim) for dim in shape) for shape in input_shapes),
        "--output_names", ",".join(output_names),
        "--tflite_op_export_spec", "builtin_first",
        "--convert_float32_weights_to_float16", "True",
    ]
    rc, text = run(command, log)
    return rc, text, log


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
    parser.add_argument("--copy-assets", action="store_true")
    args = parser.parse_args()

    android_root = args.android_root.resolve()
    output_dir = (args.output_dir or android_root / "outputs" / "p_prior_npu").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model, device, checkpoint_sha = load_p_model(args.source_root.resolve())
    module = PPriorStage1Nhwc(model).to(device).eval()
    input_names = ["p_y_hat_so_far_nhwc", "p_common_params_nhwc"]
    output_names = ["p_scales_nhwc", "p_means_nhwc"]
    samples = (
        torch.zeros((1, HEIGHT, WIDTH, Y_CHANNELS), dtype=torch.float32, device=device),
        torch.zeros((1, HEIGHT, WIDTH, COMMON_CHANNELS), dtype=torch.float32, device=device),
    )
    onnx = output_dir / f"{NAME}.onnx"
    tflite = output_dir / f"{NAME}.tflite"
    export_onnx(module, onnx, samples, input_names, output_names, args.opset)
    converter = find_tool(args.onnx_converter, "mtk_onnx_converter")
    rc, text, log = convert(onnx, tflite, converter, input_names, [tuple(value.shape) for value in samples], output_names)
    record: dict[str, Any] = {
        "name": NAME,
        "onnx": str(onnx),
        "onnx_sha256": sha256(onnx),
        "tflite": str(tflite) if tflite.is_file() else None,
        "tflite_sha256": sha256(tflite) if tflite.is_file() else None,
        "checkpoint_sha256": checkpoint_sha,
        "layout": "NHWC",
        "input_dtype": "float32",
        "output_dtype": "float32",
        "tflite_weight_precision": "float16",
        "input_names": input_names,
        "input_shapes": [list(value.shape) for value in samples],
        "output_names": output_names,
        "converter_rc": rc,
        "converter_log": str(log),
    }
    if rc == 0 and tflite.is_file():
        record.update(check_ncc(tflite, find_ncc(args.ncc_tflite, args.sdk_root, args.platform), args.arch))
    else:
        record.update({"ncc_check": "not_run", "ncc_eligible": False, "converter_output": text[-4000:]})
    if args.copy_assets and tflite.is_file():
        assets = android_root / "app" / "src" / "main" / "assets" / "prior_npu_diagnostic"
        assets.mkdir(parents=True, exist_ok=True)
        copied = assets / tflite.name
        shutil.copy2(tflite, copied)
        record["copied_asset"] = str(copied)
        record["copied_asset_sha256"] = sha256(copied)

    manifest = output_dir / "p_prior_npu_manifest.json"
    manifest.write_text(json.dumps({"tool": Path(__file__).name, "record": record}, indent=2), encoding="utf-8")
    print(f"wrote {manifest}")
    print(f"{NAME} converter_rc={rc} check_rc={record.get('check_target_rc')} plan_rc={record.get('exec_plan_rc')}")


if __name__ == "__main__":
    main()
