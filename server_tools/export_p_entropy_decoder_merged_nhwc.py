#!/usr/bin/env python3
"""Export a serial P-frame entropy decoder graph before rANS injection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch import nn

from analyze_recon_neuron_support import find_ncc
from export_p_entropy_merged_nhwc import (
    CTX_T_SHAPE,
    OUTPUT_NAMES,
    OUTPUT_SHAPES,
    PACKED_SHAPE,
    Y_SHAPE,
    Z_SHAPE,
    build_mask_2x,
)
from export_three_modules_offline_nhwc import Candidate, export_candidate, write_manifest
from gvcrt_export_common import PROJECT_ROOT, find_tool, load_p_model


class PEntropyDecoderMerged(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.hyper_dec = model.hyper_dec
        self.temporal_prior_encoder = model.temporal_prior_encoder
        self.prior_fusion = model.y_prior_fusion
        self.spatial_prior = model.y_spatial_prior
        self.register_buffer("mask_0", build_mask_2x(False))
        self.register_buffer("mask_1", build_mask_2x(True))

    @staticmethod
    def pack_2x(value: torch.Tensor) -> torch.Tensor:
        first, second = value.chunk(2, dim=1)
        return first + second

    @staticmethod
    def restore(symbols: torch.Tensor, packed_means: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        packed = symbols + packed_means
        return torch.cat((packed, packed), dim=1) * mask

    def forward(
        self,
        z_hat: torch.Tensor,
        y_q_0: torch.Tensor,
        y_q_1: torch.Tensor,
        ctx_t: torch.Tensor,
    ):
        hierarchical = self.hyper_dec(z_hat)
        temporal = self.temporal_prior_encoder(ctx_t)
        common = self.prior_fusion(
            torch.cat((hierarchical[:, :, : temporal.shape[2], : temporal.shape[3]], temporal), dim=1)
        )
        q_dec, scales, means = common.chunk(3, dim=1)
        q_dec = torch.clamp_min(q_dec, 0.5)
        scales_0 = self.pack_2x(scales * self.mask_0)
        means_0 = self.pack_2x(means * self.mask_0)
        y_hat_0 = self.restore(y_q_0, means_0, self.mask_0)

        scales, means = self.spatial_prior(torch.cat((y_hat_0, common), dim=1)).chunk(2, dim=1)
        scales_1 = self.pack_2x(scales * self.mask_1)
        means_1 = self.pack_2x(means * self.mask_1)
        y_hat_1 = self.restore(y_q_1, means_1, self.mask_1)
        y_hat = (y_hat_0 + y_hat_1) * q_dec
        return z_hat, y_q_0, y_q_1, scales_0, scales_1, y_hat


def compare(actual: torch.Tensor, expected: torch.Tensor, name: str) -> dict:
    difference = (actual.float() - expected.float()).abs()
    return {
        "name": name,
        "shape": list(actual.shape),
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "rmse": float(torch.sqrt(torch.mean(difference * difference)).item()),
        "exact": bool(torch.equal(actual, expected)),
    }


def verify(model: nn.Module, merged: PEntropyDecoderMerged) -> Tuple[bool, List[dict]]:
    torch.manual_seed(20260811)
    z_hat = torch.clamp(torch.round(torch.randn(Z_SHAPE)), -128.0, 127.0)
    y_stages = tuple(torch.clamp(torch.round(torch.randn(PACKED_SHAPE)), -8.0, 8.0) for _ in range(2))
    ctx_t = torch.randn(CTX_T_SHAPE, dtype=torch.float32) * 0.25
    with torch.no_grad():
        outputs = tuple(merged(z_hat, y_stages[0], y_stages[1], ctx_t))
        common = model.res_prior_param_decoder(z_hat, ctx_t)
        q_dec, scales, means = model.separate_prior_for_video_decoding(common)
        masks = model.get_mask_2x(1, 128, 16, 32, common.dtype, common.device)
        scales_0 = model.single_part_for_writing_2x(scales * masks[0])
        means_0 = model.single_part_for_writing_2x(means * masks[0])
        restored_0 = y_stages[0] + means_0
        y_hat_0 = torch.cat((restored_0, restored_0), dim=1) * masks[0]
        scales, means = model.y_spatial_prior(torch.cat((y_hat_0, common), dim=1)).chunk(2, dim=1)
        scales_1 = model.single_part_for_writing_2x(scales * masks[1])
        means_1 = model.single_part_for_writing_2x(means * masks[1])
        restored_1 = y_stages[1] + means_1
        y_hat_1 = torch.cat((restored_1, restored_1), dim=1) * masks[1]
        expected = (z_hat, y_stages[0], y_stages[1], scales_0, scales_1, (y_hat_0 + y_hat_1) * q_dec)
    comparisons = [compare(value, reference, name) for name, value, reference in zip(OUTPUT_NAMES, outputs, expected)]
    return all(item["exact"] for item in comparisons), comparisons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--android-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pytorch-converter", default=None)
    parser.add_argument("--ncc-tflite", required=True)
    parser.add_argument("--arch", default="mdla5.3")
    parser.add_argument("--qp", type=int, default=0)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_dir = (args.output_dir or args.android_root.resolve() / "outputs" / "p_entropy_decoder_merged_nhwc").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "p_entropy_decoder_merged_nhwc_manifest.json"
    converter = find_tool(args.pytorch_converter, "mtk_pytorch_converter")
    ncc = find_ncc(args.ncc_tflite)

    print("[p-entropy-decoder-merged] loading P checkpoint", flush=True)
    model, _, checkpoint_sha = load_p_model(source_root)
    model = model.cpu().eval()
    merged = PEntropyDecoderMerged(model).cpu().eval()
    verification_passed, verification = verify(model, merged)
    print("[p-entropy-decoder-merged] source verification exact={}".format(verification_passed), flush=True)

    candidate = Candidate(
        "p_entropy_decode_merged_base",
        "p_entropy_decoder",
        "p",
        (("p_z_hat", Z_SHAPE), ("p_y_q_w_0", PACKED_SHAPE), ("p_y_q_w_1", PACKED_SHAPE), ("p_ctx_t", CTX_T_SHAPE)),
        tuple(zip(OUTPUT_NAMES, OUTPUT_SHAPES)),
        lambda unused_model, unused_qp: merged,
        check_trace=False,
    )
    metadata = {
        "tool": Path(__file__).name,
        "source_root": str(source_root),
        "scope": "P serial hyper/prior decode skeleton before rANS custom-op injection",
        "checkpoint_sha256": {"p": checkpoint_sha},
        "qp": args.qp,
        "layout": "NHWC FP32 external I/O; source state is NCHW",
        "arch": args.arch,
        "selected_candidates": [candidate.name],
        "source_verification_passed": verification_passed,
        "source_verification": verification,
    }
    if verification_passed:
        record = export_candidate(candidate, model, checkpoint_sha, args.qp, converter, ncc, args.arch, output_dir)
    else:
        record = {"name": candidate.name, "status": "source_verification_failed"}
    write_manifest(manifest_path, [record], metadata)
    print("wrote {}".format(manifest_path))
    print(json.dumps(json.loads(manifest_path.read_text(encoding="utf-8"))["summary"], indent=2))


if __name__ == "__main__":
    main()
