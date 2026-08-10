#!/usr/bin/env python3
"""Export one merged I-frame entropy/prior graph for MTK evaluation.

The graph starts at the I encoder latent and includes hyper encode/decode,
four-stage masked prior inference, symbol quantization, and serial feedback.
CDF lookup and rANS remain outside the graph.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch import nn

from analyze_recon_neuron_support import find_ncc
from export_three_modules_offline_nhwc import (
    Candidate,
    export_candidate,
    write_manifest,
)
from gvcrt_export_common import PROJECT_ROOT, find_tool, load_i_model


Shape = Tuple[int, int, int, int]
OUTPUT_NAMES = (
    "i_z_hat",
    "i_y_q_w_0",
    "i_y_q_w_1",
    "i_y_q_w_2",
    "i_y_q_w_3",
    "i_s_w_0",
    "i_s_w_1",
    "i_s_w_2",
    "i_s_w_3",
    "i_y_hat",
)
OUTPUT_SHAPES: Tuple[Shape, ...] = (
    (1, 128, 4, 8),
    *((1, 64, 16, 32),) * 8,
    (1, 256, 16, 32),
)


def build_mask_4x(phase_order: Sequence[int]) -> torch.Tensor:
    phases: List[torch.Tensor] = []
    for phase in range(4):
        mask = torch.zeros((1, 64, 16, 32), dtype=torch.float32)
        row = phase // 2
        column = phase % 2
        mask[:, :, row::2, column::2] = 1.0
        phases.append(mask)
    return torch.cat(tuple(phases[index] for index in phase_order), dim=1)


class IEntropyMerged(nn.Module):
    """Source-equivalent fixed-shape I entropy path before CDF/rANS."""

    def __init__(self, model: nn.Module, force_zero_thres: float) -> None:
        super().__init__()
        self.hyper_enc = model.hyper_enc
        self.hyper_dec = model.hyper_dec
        self.prior_fusion = model.y_prior_fusion
        self.reduction = model.y_spatial_prior_reduction
        self.adaptor_1 = model.y_spatial_prior_adaptor_1
        self.adaptor_2 = model.y_spatial_prior_adaptor_2
        self.adaptor_3 = model.y_spatial_prior_adaptor_3
        self.spatial_prior = model.y_spatial_prior
        self.force_zero_thres = float(force_zero_thres)
        self.register_buffer("mask_0", build_mask_4x((0, 1, 2, 3)))
        self.register_buffer("mask_1", build_mask_4x((3, 2, 1, 0)))
        self.register_buffer("mask_2", build_mask_4x((2, 3, 0, 1)))
        self.register_buffer("mask_3", build_mask_4x((1, 0, 3, 2)))

    @staticmethod
    def pack_4x(value: torch.Tensor) -> torch.Tensor:
        part_0, part_1, part_2, part_3 = value.chunk(4, dim=1)
        return (part_0 + part_1) + (part_2 + part_3)

    def quantize_stage(
        self,
        y: torch.Tensor,
        scales: torch.Tensor,
        means: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scales_hat = scales * mask
        means_hat = means * mask
        y_q = torch.round((y - means_hat) * mask)
        keep = (scales_hat > self.force_zero_thres).to(dtype=y_q.dtype)
        y_q = torch.clamp(y_q * keep, -128.0, 127.0)
        return y_q, y_q + means_hat, scales_hat

    def forward(self, y: torch.Tensor):
        # The fixed 16x32 latent needs no replicate padding before hyper_enc.
        z = self.hyper_enc(y)
        z_hat = torch.clamp(torch.round(z), -128.0, 127.0)

        common = self.prior_fusion(self.hyper_dec(z_hat))
        common = common[:, :, :16, :32].contiguous()
        q = common[:, :2, :, :]
        q_enc, q_dec = (torch.sigmoid(q) * 1.5 + 0.5).chunk(2, dim=1)
        scales, means = common[:, 2:, :, :].chunk(2, dim=1)
        reduced = self.reduction(common)
        y_scaled = y * q_enc

        y_q_0, y_hat_0, s_hat_0 = self.quantize_stage(
            y_scaled, scales, means, self.mask_0
        )

        stage_1 = self.spatial_prior(
            self.adaptor_1(torch.cat((y_hat_0, reduced), dim=1))
        )
        scales, means = stage_1.chunk(2, dim=1)
        y_q_1, y_hat_1, s_hat_1 = self.quantize_stage(
            y_scaled, scales, means, self.mask_1
        )

        y_hat_so_far = y_hat_0 + y_hat_1
        stage_2 = self.spatial_prior(
            self.adaptor_2(torch.cat((y_hat_so_far, reduced), dim=1))
        )
        scales, means = stage_2.chunk(2, dim=1)
        y_q_2, y_hat_2, s_hat_2 = self.quantize_stage(
            y_scaled, scales, means, self.mask_2
        )

        y_hat_so_far = y_hat_so_far + y_hat_2
        stage_3 = self.spatial_prior(
            self.adaptor_3(torch.cat((y_hat_so_far, reduced), dim=1))
        )
        scales, means = stage_3.chunk(2, dim=1)
        y_q_3, y_hat_3, s_hat_3 = self.quantize_stage(
            y_scaled, scales, means, self.mask_3
        )
        y_hat = (y_hat_so_far + y_hat_3) * q_dec

        return (
            z_hat,
            self.pack_4x(y_q_0),
            self.pack_4x(y_q_1),
            self.pack_4x(y_q_2),
            self.pack_4x(y_q_3),
            self.pack_4x(s_hat_0),
            self.pack_4x(s_hat_1),
            self.pack_4x(s_hat_2),
            self.pack_4x(s_hat_3),
            y_hat,
        )


def source_outputs(
    model: nn.Module,
    y: torch.Tensor,
    force_zero_thres: float,
) -> Tuple[torch.Tensor, ...]:
    model.gaussian_encoder.force_zero_thres = float(force_zero_thres)
    z = model.hyper_enc(model.pad_for_y(y))
    z_hat = torch.clamp(torch.round(z), -128.0, 127.0)
    common = model.y_prior_fusion(model.hyper_dec(z_hat))
    common = common[:, :, : y.shape[2], : y.shape[3]].contiguous()
    prior_outputs = model.compress_prior_4x(
        y,
        common,
        model.y_spatial_prior_reduction,
        model.y_spatial_prior_adaptor_1,
        model.y_spatial_prior_adaptor_2,
        model.y_spatial_prior_adaptor_3,
        model.y_spatial_prior,
    )
    return (z_hat,) + tuple(prior_outputs)


def compare_outputs(
    actual: Sequence[torch.Tensor],
    expected: Sequence[torch.Tensor],
) -> List[dict]:
    comparisons: List[dict] = []
    for name, value, reference in zip(OUTPUT_NAMES, actual, expected):
        difference = (value.float() - reference.float()).abs()
        comparisons.append(
            {
                "name": name,
                "shape": list(value.shape),
                "max_abs": float(difference.max().item()),
                "mean_abs": float(difference.mean().item()),
                "exact": bool(torch.equal(value, reference)),
            }
        )
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--android-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pytorch-converter", default=None)
    parser.add_argument("--ncc-tflite", required=True)
    parser.add_argument("--arch", default="mdla5.3")
    parser.add_argument("--qp", type=int, default=0)
    parser.add_argument("--force-zero-thres", type=float, default=0.12)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    android_root = args.android_root.resolve()
    output_dir = (
        args.output_dir or android_root / "outputs" / "i_entropy_merged_nhwc"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "i_entropy_merged_nhwc_manifest.json"
    converter = find_tool(args.pytorch_converter, "mtk_pytorch_converter")
    ncc = find_ncc(args.ncc_tflite)

    print("[i-entropy-merged] loading I checkpoint", flush=True)
    model, _, checkpoint_sha = load_i_model(source_root)
    model = model.cpu().eval()
    merged = IEntropyMerged(model, args.force_zero_thres).cpu().eval()

    torch.manual_seed(20260811)
    verification_input = torch.randn((1, 256, 16, 32), dtype=torch.float32) * 0.25
    with torch.no_grad():
        rewritten_outputs = tuple(merged(verification_input))
        expected_outputs = source_outputs(
            model, verification_input, args.force_zero_thres
        )
    verification = compare_outputs(rewritten_outputs, expected_outputs)
    verification_passed = all(item["exact"] for item in verification)
    print(
        "[i-entropy-merged] source verification "
        f"exact={verification_passed}",
        flush=True,
    )

    candidate = Candidate(
        "i_entropy_prior_merged",
        "complete_encoder_entropy",
        "i",
        (("i_y_pre_prior", (1, 256, 16, 32)),),
        tuple(zip(OUTPUT_NAMES, OUTPUT_SHAPES)),
        lambda unused_model, unused_qp: merged,
    )
    metadata = {
        "tool": Path(__file__).name,
        "source_root": str(source_root),
        "scope": "I hyper encode/decode + serial 4-stage masked prior + symbol quantization",
        "outside_graph": ["CDF lookup", "rANS", "bitstream mux", "reconstruction"],
        "checkpoint_sha256": {"i": checkpoint_sha},
        "qp": args.qp,
        "force_zero_thres": args.force_zero_thres,
        "layout": "NHWC FP32 external I/O; source state is NCHW",
        "arch": args.arch,
        "ncc_flags": ["--opt-bw", "--relax-fp32"],
        "selected_candidates": [candidate.name],
        "source_verification_passed": verification_passed,
        "source_verification": verification,
    }
    if not verification_passed:
        record = {
            "name": candidate.name,
            "group": candidate.group,
            "family": candidate.family,
            "status": "source_precision_failed",
        }
    else:
        try:
            record = export_candidate(
                candidate,
                model,
                checkpoint_sha,
                args.qp,
                converter,
                ncc,
                args.arch,
                output_dir,
            )
            record["boundary_note"] = (
                "one merged I entropy graph including exact serial mask/round/clamp feedback; "
                "CDF and rANS excluded"
            )
            record["online_tflite_ready"] = bool(
                record.get("converter_rc") == 0 and record.get("tflite")
            )
            record["offline_dla_status"] = record["status"]
            diagnostics = (record.get("ncc") or {}).get("diagnostic_lines", [])
            diagnostics += (record.get("ncc") or {}).get("dla_diagnostic_lines", [])
            record["transpose_warning"] = any(
                line.strip().endswith(": TRANSPOSE") or "false Transpose" in line
                for line in diagnostics
            )
            if record["online_tflite_ready"]:
                record["status"] = "ok"
        except Exception as exc:
            record = {
                "name": candidate.name,
                "group": candidate.group,
                "family": candidate.family,
                "status": "exception",
                "error": repr(exc),
            }

    write_manifest(manifest_path, [record], metadata)
    print(
        f"[i-entropy-merged] {candidate.name} status={record['status']}",
        flush=True,
    )
    print(f"wrote {manifest_path}", flush=True)
    print(json.dumps(json.loads(manifest_path.read_text(encoding="utf-8"))["summary"], indent=2))
    gc.collect()


if __name__ == "__main__":
    main()
