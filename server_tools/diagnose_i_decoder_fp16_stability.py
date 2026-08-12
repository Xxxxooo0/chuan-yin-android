#!/usr/bin/env python3
"""Locate the first FP16-sensitive boundary in the rewritten I decoder.

This script is intentionally diagnostic only. It reads the real Android I
latent, runs the source and rewritten synthesis paths on CUDA FP16, and emits
stage-level error and range statistics without exporting or modifying models.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from export_decoder_full_norm_rewrite_nhwc import FixedGenerator, FixedIFeatureDec
from gvcrt_export_common import load_i_model, sha256


FP16_MAX = 65504.0
FP16_SQUARE_LIMIT = math.sqrt(FP16_MAX)


def load_nhwc(path: Path, shape, device: torch.device) -> torch.Tensor:
    values = np.fromfile(str(path), dtype="<f4")
    expected = int(np.prod(shape))
    if values.size != expected:
        raise RuntimeError(
            "{} elements={} expected={}".format(path, values.size, expected)
        )
    nchw = values.reshape(shape).transpose(0, 3, 1, 2).copy()
    return torch.from_numpy(nchw).to(device=device, dtype=torch.float16)


def tensor_stats(value: torch.Tensor):
    detached = value.detach().float()
    finite = torch.isfinite(detached)
    finite_values = detached[finite]
    if finite_values.numel() == 0:
        minimum = maximum = max_abs = float("nan")
    else:
        minimum = float(finite_values.min().item())
        maximum = float(finite_values.max().item())
        max_abs = float(finite_values.abs().max().item())
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "min": minimum,
        "max": maximum,
        "max_abs": max_abs,
        "finite_fraction": float(finite.float().mean().item()),
        "square_overflow_risk": bool(max_abs > FP16_SQUARE_LIMIT),
    }


def comparison(expected: torch.Tensor, actual: torch.Tensor):
    expected_f = expected.detach().float()
    actual_f = actual.detach().float()
    difference = actual_f - expected_f
    finite = torch.isfinite(expected_f) & torch.isfinite(actual_f)
    if not bool(finite.all().item()):
        return {
            "finite": False,
            "expected_finite_fraction": float(torch.isfinite(expected_f).float().mean().item()),
            "actual_finite_fraction": float(torch.isfinite(actual_f).float().mean().item()),
        }
    return {
        "finite": True,
        "max_abs": float(difference.abs().max().item()),
        "mean_abs": float(difference.abs().mean().item()),
        "rmse": float(torch.sqrt(torch.mean(difference * difference)).item()),
    }


def append_stage(records, name, source_value, rewritten_value):
    record = {
        "name": name,
        "source": tensor_stats(source_value),
        "rewritten": tensor_stats(rewritten_value),
        "rewritten_vs_source": comparison(source_value, rewritten_value),
    }
    records.append(record)
    metrics = record["rewritten_vs_source"]
    print(
        "[fp16-diagnose] {} max_abs={} rmse={} source_max={} rewritten_max={}".format(
            name,
            metrics.get("max_abs", "non_finite"),
            metrics.get("rmse", "non_finite"),
            record["source"]["max_abs"],
            record["rewritten"]["max_abs"],
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--android-i-y-hat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qp", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the FP16 stability diagnostic")

    source_root = args.source_root.resolve()
    latent_path = args.android_i_y_hat.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, device, checkpoint_sha = load_i_model(source_root)
    rewritten_feature = FixedIFeatureDec(model, args.qp).to(device).eval()
    rewritten_generator = FixedGenerator(model, args.qp).to(device).eval()
    model = model.half().eval()
    rewritten_feature = rewritten_feature.half().eval()
    rewritten_generator = rewritten_generator.half().eval()

    y_hat = load_nhwc(latent_path, (1, 16, 32, 256), device)
    q_dec = model.q_scale_dec[args.qp : args.qp + 1]
    q_recon = model.q_scale_recon[args.qp : args.qp + 1]
    source_generator = model.recon_generation_net.decoder
    records = []
    norm_inputs = []

    def norm_hook(name):
        def hook(_module, inputs):
            value = inputs[0]
            stats = tensor_stats(value)
            stats["name"] = name
            norm_inputs.append(stats)
        return hook

    hooks = []
    for name, module in rewritten_generator.named_modules():
        if module.__class__.__name__ in {"FixedGroupNorm", "FixedAdaGN"}:
            hooks.append(module.register_forward_pre_hook(norm_hook(name)))
    for name, module in rewritten_feature.named_modules():
        if module.__class__.__name__ == "FixedGroupNorm":
            hooks.append(module.register_forward_pre_hook(norm_hook("feature_dec." + name)))

    with torch.no_grad():
        source_codeword = model.dec(y_hat, q_dec)
        rewritten_codeword = rewritten_feature(y_hat)
        append_stage(records, "feature_dec", source_codeword, rewritten_codeword)

        # Feed both generator implementations the same rewritten codeword so
        # the first generator divergence is not hidden by FeatureDec error.
        codeword = rewritten_codeword
        source_value = source_generator.conv_in(codeword)
        rewritten_value = rewritten_generator.conv_in(codeword)
        append_stage(records, "generator_conv_in", source_value, rewritten_value)

        source_value = source_generator.ada1(source_value, codeword)
        rewritten_value = rewritten_generator.ada1(rewritten_value, codeword)
        append_stage(records, "generator_ada1", source_value, rewritten_value)
        source_value = source_generator.stage1(source_value)
        rewritten_value = rewritten_generator.stage1(rewritten_value)
        append_stage(records, "generator_stage1", source_value, rewritten_value)

        source_value = source_generator.ada2(source_value, codeword)
        rewritten_value = rewritten_generator.ada2(rewritten_value, codeword)
        append_stage(records, "generator_ada2", source_value, rewritten_value)
        source_value = source_generator.stage2(source_value)
        rewritten_value = rewritten_generator.stage2(rewritten_value)
        append_stage(records, "generator_stage2", source_value, rewritten_value)

        source_value = source_generator.ada3(source_value, codeword)
        rewritten_value = rewritten_generator.ada3(rewritten_value, codeword)
        append_stage(records, "generator_ada3", source_value, rewritten_value)
        source_value = source_generator.upsample(source_value)
        rewritten_value = rewritten_generator.upsample(rewritten_value)
        append_stage(records, "generator_upsample", source_value, rewritten_value)
        source_value = source_generator.stage3(source_value)
        rewritten_value = rewritten_generator.stage3(rewritten_value)
        append_stage(records, "generator_stage3", source_value, rewritten_value)

        source_value = source_generator.ada4(source_value, codeword)
        rewritten_value = rewritten_generator.ada4(rewritten_value, codeword)
        append_stage(records, "generator_ada4", source_value, rewritten_value)
        source_value = source_generator.stage4(source_value, q_recon)
        rewritten_value = rewritten_generator.stage4(rewritten_value, q_recon)
        append_stage(records, "generator_stage4", source_value, rewritten_value)

        source_value = source_generator.ada_final(source_value, codeword)
        rewritten_value = rewritten_generator.ada_final(rewritten_value, codeword)
        append_stage(records, "generator_ada_final", source_value, rewritten_value)
        source_value = torch.clamp(
            torch.nn.functional.pixel_shuffle(source_generator.head(source_value), 8),
            -1.0,
            1.0,
        )
        rewritten_value = torch.clamp(
            torch.nn.functional.pixel_shuffle(rewritten_generator.head(rewritten_value), 8),
            -1.0,
            1.0,
        )
        append_stage(records, "reference_frame", source_value, rewritten_value)
        torch.cuda.synchronize()

    for hook in hooks:
        hook.remove()

    first_material_difference = None
    first_non_finite_stage = None
    first_max_abs_over_0_05 = None
    for record in records:
        metrics = record["rewritten_vs_source"]
        if not metrics.get("finite", False) or metrics.get("max_abs", 0.0) > 1e-3:
            if first_material_difference is None:
                first_material_difference = record["name"]
        if not metrics.get("finite", False) and first_non_finite_stage is None:
            first_non_finite_stage = record["name"]
        if metrics.get("max_abs", 0.0) > 0.05 and first_max_abs_over_0_05 is None:
            first_max_abs_over_0_05 = record["name"]

    report = {
        "source_root": str(source_root),
        "checkpoint_sha256": checkpoint_sha,
        "android_i_y_hat": str(latent_path),
        "android_i_y_hat_sha256": sha256(latent_path),
        "qp": args.qp,
        "precision": "cuda_fp16_diagnostic",
        "fp16_square_limit": FP16_SQUARE_LIMIT,
        "first_material_difference": first_material_difference,
        "first_non_finite_stage": first_non_finite_stage,
        "first_max_abs_over_0_05": first_max_abs_over_0_05,
        "stages": records,
        "rewritten_norm_inputs": norm_inputs,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[fp16-diagnose] first_material_difference={}".format(first_material_difference))
    print("[fp16-diagnose] first_non_finite_stage={}".format(first_non_finite_stage))
    print("[fp16-diagnose] first_max_abs_over_0_05={}".format(first_max_abs_over_0_05))
    print("wrote {}".format(output_path))


if __name__ == "__main__":
    main()
