#!/usr/bin/env python3
"""Torch-only helper for the I latent op0 FP16 TFLite export.

TensorFlow and this server PyTorch build cannot coexist in one process. Keep
all checkpoint access and PyTorch reference execution here, then pass arrays
to the TensorFlow-only exporter through files.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch


INPUT_SHAPE = (1, 256, 16, 32)


def state_dict_from_checkpoint(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"unsupported checkpoint type: {type(checkpoint).__name__}")
    for key in ("student", "ema_shadow", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            print(f"[op0-weights] selected checkpoint state_dict={key}", flush=True)
            return value
    print("[op0-weights] selected checkpoint state_dict=root", flush=True)
    return checkpoint


def state_tensor(state_dict: dict[str, torch.Tensor], suffix: str) -> torch.Tensor:
    matches = [(key, value) for key, value in state_dict.items() if key.replace("module.", "") == suffix]
    if len(matches) != 1:
        available = [key for key in state_dict if key.endswith(suffix)]
        raise RuntimeError(f"expected one tensor for {suffix}, matches={available}")
    key, value = matches[0]
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"checkpoint value is not a tensor: {key}")
    print(f"[op0-weights] loaded tensor={key} shape={tuple(value.shape)}", flush=True)
    return value.detach().cpu().contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-nchw-f32le", type=Path, required=True)
    parser.add_argument("--weights-output", type=Path, required=True)
    parser.add_argument("--reference-output-nhwc-f32le", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, default=None)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    input_path = args.input_nchw_f32le.resolve()
    print(f"[op0-weights] reading checkpoint on CPU: {checkpoint_path}", flush=True)
    state_dict = state_dict_from_checkpoint(checkpoint_path)
    weight = state_tensor(state_dict, "dec.conv_in.adaptor.weight")
    bias = state_tensor(state_dict, "dec.conv_in.adaptor.bias")
    if tuple(weight.shape) != (512, 256, 1, 1) or tuple(bias.shape) != (512,):
        raise RuntimeError(f"unexpected parameter shapes: weight={tuple(weight.shape)} bias={tuple(bias.shape)}")

    input_nchw = np.fromfile(input_path, dtype="<f4").reshape(INPUT_SHAPE)
    with torch.no_grad():
        reference_nchw = torch.nn.functional.conv2d(torch.from_numpy(input_nchw), weight, bias).numpy()
    reference_nhwc = np.transpose(reference_nchw, (0, 2, 3, 1)).copy()

    args.weights_output.parent.mkdir(parents=True, exist_ok=True)
    args.reference_output_nhwc_f32le.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.weights_output, weight=weight.numpy(), bias=bias.numpy())
    reference_nhwc.astype("<f4").tofile(args.reference_output_nhwc_f32le)
    print(f"[op0-weights] wrote weights={args.weights_output}", flush=True)
    print(f"[op0-weights] wrote reference={args.reference_output_nhwc_f32le}", flush=True)

    if args.asset_dir is not None:
        asset_dir = args.asset_dir.resolve()
        asset_dir.mkdir(parents=True, exist_ok=True)
        input_nhwc = np.transpose(input_nchw, (0, 2, 3, 1)).copy()
        weight_ohwi = np.transpose(weight.numpy(), (0, 2, 3, 1)).copy()
        asset_paths = {
            "i_y_hat_nhwc.f16le": input_nhwc.astype("<f2"),
            "i_op0_weight_ohwi.f16le": weight_ohwi.astype("<f2"),
            "i_op0_bias.f16le": bias.numpy().astype("<f2"),
            "i_op0_output_nhwc.f16le": reference_nhwc.astype("<f2"),
        }
        for name, values in asset_paths.items():
            values.tofile(asset_dir / name)
        shutil.copy2(args.weights_output, asset_dir / args.weights_output.name)
        print(f"[op0-weights] wrote FP16 Adapter assets={asset_dir}", flush=True)


if __name__ == "__main__":
    main()
