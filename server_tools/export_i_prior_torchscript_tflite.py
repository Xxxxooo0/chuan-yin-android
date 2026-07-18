#!/usr/bin/env python3
"""Export the I-prior reduction through the official TorchScript converter path.

This is a deliberately small feasibility probe. It avoids the MTK ONNX
importer's unsupported FLOAT16 tensor type and follows the Converter guide's
PyTorch ScriptModule plus NHWC wrapper route. It does not change Android's
canonical encoder.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from export_i_prior_npu_tflite import (
    COMMON_CHANNELS,
    HEIGHT,
    PROJECT_ROOT,
    WIDTH,
    PriorReduceNhwc,
    check_ncc,
    find_ncc,
    find_tool,
    sha256,
)


NAME = "i_prior_reduce_torchscript_fp32"


def load_model(source_root: Path):
    sys.path.insert(0, str(source_root))
    import src.layers.cuda_inference as cuda_inference
    import src.layers.layers as layers
    import src.models.image_model_G_b as image_model
    from src.models.image_model_G_b import DMCI

    cuda_inference.CUSTOMIZED_CUDA_INFERENCE = False
    layers.CUSTOMIZED_CUDA_INFERENCE = False
    image_model.CUSTOMIZED_CUDA_INFERENCE = False
    checkpoint = source_root / "ckpt" / "checkpoints" / "GVC-RT_B_I.pt"
    return DMCI(encoder_ckpt_path=str(checkpoint)).cpu().eval(), checkpoint


def run(command: list[str], log: Path) -> tuple[int, str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    text = "$ " + " ".join(command) + "\n\n" + result.stdout
    log.write_text(text, encoding="utf-8")
    return result.returncode, text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pytorch-converter", default=None)
    parser.add_argument("--sdk-root", type=Path, default=None)
    parser.add_argument("--ncc-tflite", default=None)
    parser.add_argument("--platform", default="mt6899")
    parser.add_argument("--arch", default="mdla5.3")
    args = parser.parse_args()

    output_dir = (args.output_dir or PROJECT_ROOT / "outputs" / "i_prior_torchscript").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_root = args.source_root.resolve()
    converter = find_tool(args.pytorch_converter, "mtk_pytorch_converter")
    ncc = find_ncc(args.ncc_tflite, args.sdk_root, args.platform)
    model, checkpoint = load_model(source_root)
    module = PriorReduceNhwc(model).cpu().eval()
    sample = torch.zeros((1, HEIGHT, WIDTH, COMMON_CHANNELS), dtype=torch.float32)
    script_path = output_dir / f"{NAME}.pt"
    with torch.no_grad():
        torch.jit.trace(module, (sample,), strict=False).save(str(script_path))

    tflite_path = output_dir / f"{NAME}.tflite"
    convert_log = output_dir / "logs" / f"{NAME}_convert.log"
    rc, text = run(
        [
            converter,
            "--input_script_module_file", str(script_path),
            "--output_file", str(tflite_path),
            "--input_shapes", f"1,{HEIGHT},{WIDTH},{COMMON_CHANNELS}",
        ],
        convert_log,
    )
    record: dict[str, Any] = {
        "name": NAME,
        "route": "torchscript_nhwc",
        "checkpoint_sha256": sha256(checkpoint),
        "script_module": str(script_path),
        "script_module_sha256": sha256(script_path),
        "tflite": str(tflite_path) if tflite_path.is_file() else None,
        "tflite_sha256": sha256(tflite_path) if tflite_path.is_file() else None,
        "converter": converter,
        "converter_rc": rc,
        "converter_log": str(convert_log),
        "input_shape_nhwc": [1, HEIGHT, WIDTH, COMMON_CHANNELS],
        "onnx_used": False,
    }
    if rc == 0 and tflite_path.is_file():
        record.update(check_ncc(tflite_path, ncc, args.arch))
    else:
        record.update({"ncc_check": "not_run", "ncc_eligible": False, "converter_output": text[-4000:]})
    manifest_path = output_dir / "i_prior_torchscript_manifest.json"
    manifest_path.write_text(json.dumps({"tool": Path(__file__).name, "record": record}, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")
    print(f"{NAME} converter_rc={rc} ncc={record.get('ncc_check')} eligible={record.get('ncc_eligible')}")


if __name__ == "__main__":
    main()
