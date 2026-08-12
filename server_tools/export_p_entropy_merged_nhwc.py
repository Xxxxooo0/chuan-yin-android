#!/usr/bin/env python3
"""Export one merged P-frame entropy/prior graph for MTK online execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch import nn

from analyze_recon_neuron_support import find_ncc
from export_three_modules_offline_nhwc import Candidate, export_candidate, write_manifest
from gvcrt_export_common import PROJECT_ROOT, find_tool, load_p_model


Shape = Tuple[int, int, int, int]
Y_SHAPE: Shape = (1, 128, 16, 32)
Z_SHAPE: Shape = (1, 128, 4, 8)
CTX_T_SHAPE: Shape = (1, 256, 32, 64)
PACKED_SHAPE: Shape = (1, 64, 16, 32)
OUTPUT_NAMES = (
    "p_z_hat",
    "p_y_q_w_0",
    "p_y_q_w_1",
    "p_s_w_0",
    "p_s_w_1",
    "p_y_hat",
)
OUTPUT_SHAPES = (Z_SHAPE, PACKED_SHAPE, PACKED_SHAPE, PACKED_SHAPE, PACKED_SHAPE, Y_SHAPE)


def checkerboard(pattern: Sequence[Sequence[int]]) -> torch.Tensor:
    base = torch.tensor(pattern, dtype=torch.float32)
    return base.repeat(8, 16).reshape(1, 1, 16, 32)


def build_mask_2x(reverse: bool) -> torch.Tensor:
    first = checkerboard(((1, 0), (0, 1)))
    second = checkerboard(((0, 1), (1, 0)))
    order = (second, first) if reverse else (first, second)
    return torch.cat((order[0].repeat(1, 64, 1, 1), order[1].repeat(1, 64, 1, 1)), dim=1)


class PEntropyMerged(nn.Module):
    def __init__(self, model: nn.Module, force_zero_thres: float) -> None:
        super().__init__()
        self.hyper_enc = model.hyper_enc
        self.hyper_dec = model.hyper_dec
        self.temporal_prior_encoder = model.temporal_prior_encoder
        self.prior_fusion = model.y_prior_fusion
        self.spatial_prior = model.y_spatial_prior
        self.force_zero_thres = float(force_zero_thres)
        self.register_buffer("mask_0", build_mask_2x(False))
        self.register_buffer("mask_1", build_mask_2x(True))

    @staticmethod
    def pack_2x(value: torch.Tensor) -> torch.Tensor:
        first, second = value.chunk(2, dim=1)
        return first + second

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

    def forward(self, y: torch.Tensor, ctx_t: torch.Tensor):
        # Fixed 16x32 latent is already aligned to the hyper encoder stride.
        z = self.hyper_enc(y)
        z_hat = torch.clamp(torch.round(z), -128.0, 127.0)
        hierarchical = self.hyper_dec(z_hat)
        temporal = self.temporal_prior_encoder(ctx_t)
        common = self.prior_fusion(
            torch.cat((hierarchical[:, :, : temporal.shape[2], : temporal.shape[3]], temporal), dim=1)
        )
        q_dec, scales, means = common.chunk(3, dim=1)
        q_dec = torch.clamp_min(q_dec, 0.5)
        y_scaled = y * torch.reciprocal(q_dec)

        y_q_0, y_hat_0, s_hat_0 = self.quantize_stage(
            y_scaled, scales, means, self.mask_0
        )
        scales, means = self.spatial_prior(torch.cat((y_hat_0, common), dim=1)).chunk(2, dim=1)
        y_q_1, y_hat_1, s_hat_1 = self.quantize_stage(
            y_scaled, scales, means, self.mask_1
        )
        y_hat = (y_hat_0 + y_hat_1) * q_dec
        return (
            z_hat,
            self.pack_2x(y_q_0),
            self.pack_2x(y_q_1),
            self.pack_2x(s_hat_0),
            self.pack_2x(s_hat_1),
            y_hat,
        )


def source_outputs(
    model: nn.Module,
    y: torch.Tensor,
    ctx_t: torch.Tensor,
    force_zero_thres: float,
) -> Tuple[torch.Tensor, ...]:
    model.gaussian_encoder.force_zero_thres = float(force_zero_thres)
    z = model.hyper_enc(model.pad_for_y(y))
    z_hat = torch.clamp(torch.round(z), -128.0, 127.0)
    common = model.res_prior_param_decoder(z_hat, ctx_t)
    return (z_hat,) + tuple(model.compress_prior_2x(y, common, model.y_spatial_prior))


def compare_outputs(actual: Sequence[torch.Tensor], expected: Sequence[torch.Tensor]) -> List[dict]:
    records: List[dict] = []
    for name, value, reference in zip(OUTPUT_NAMES, actual, expected):
        difference = (value.float() - reference.float()).abs()
        records.append({
            "name": name,
            "shape": list(value.shape),
            "max_abs": float(difference.max().item()),
            "mean_abs": float(difference.mean().item()),
            "exact": bool(torch.equal(value, reference)),
        })
    return records


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
    output_dir = (args.output_dir or args.android_root.resolve() / "outputs" / "p_entropy_merged_nhwc").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "p_entropy_merged_nhwc_manifest.json"
    converter = find_tool(args.pytorch_converter, "mtk_pytorch_converter")
    ncc = find_ncc(args.ncc_tflite)

    print("[p-entropy-merged] loading P checkpoint", flush=True)
    model, _, checkpoint_sha = load_p_model(source_root)
    model = model.cpu().eval()
    merged = PEntropyMerged(model, args.force_zero_thres).cpu().eval()
    torch.manual_seed(20260811)
    y = torch.randn(Y_SHAPE, dtype=torch.float32) * 0.25
    ctx_t = torch.randn(CTX_T_SHAPE, dtype=torch.float32) * 0.25
    with torch.no_grad():
        rewritten = tuple(merged(y, ctx_t))
        expected = source_outputs(model, y, ctx_t, args.force_zero_thres)
    verification = compare_outputs(rewritten, expected)
    verification_passed = all(item["exact"] for item in verification)
    print("[p-entropy-merged] source verification exact={}".format(verification_passed), flush=True)

    candidate = Candidate(
        "p_entropy_prior_merged",
        "complete_encoder_entropy",
        "p",
        (("p_y_pre_prior", Y_SHAPE), ("p_ctx_t", CTX_T_SHAPE)),
        tuple(zip(OUTPUT_NAMES, OUTPUT_SHAPES)),
        lambda unused_model, unused_qp: merged,
        check_trace=False,
    )
    metadata = {
        "tool": Path(__file__).name,
        "source_root": str(source_root),
        "scope": "P hyper encode/decode + serial 2-stage masked prior + symbol quantization",
        "outside_graph": ["CDF lookup", "rANS", "bitstream mux", "reconstruction"],
        "checkpoint_sha256": {"p": checkpoint_sha},
        "qp": args.qp,
        "force_zero_thres": args.force_zero_thres,
        "layout": "NHWC FP32 external I/O; source state is NCHW",
        "arch": args.arch,
        "ncc_flags": ["--opt-bw", "--relax-fp32"],
        "selected_candidates": [candidate.name],
        "source_verification_passed": verification_passed,
        "source_verification": verification,
    }
    if verification_passed:
        record = export_candidate(candidate, model, checkpoint_sha, args.qp, converter, ncc, args.arch, output_dir)
        record["online_tflite_ready"] = bool(record.get("converter_rc") == 0 and record.get("tflite"))
    else:
        record = {"name": candidate.name, "status": "source_precision_failed"}
    write_manifest(manifest_path, [record], metadata)
    print("wrote {}".format(manifest_path))
    print(json.dumps(json.loads(manifest_path.read_text(encoding="utf-8"))["summary"], indent=2))


if __name__ == "__main__":
    main()
