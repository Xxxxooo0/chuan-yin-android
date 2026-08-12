#!/usr/bin/env python3
"""Export one I-frame entropy-decoder prior graph before rANS injection.

The temporary graph accepts decoded z and four decoded y phases. A second
tool replaces those five graph inputs with serial rANS custom operators, so
the final TFLite has only payload bytes and payload size as external inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch import nn

from analyze_recon_neuron_support import find_ncc
from export_three_modules_offline_nhwc import Candidate, export_candidate, write_manifest
from gvcrt_export_common import PROJECT_ROOT, find_tool, load_i_model


Shape = Tuple[int, int, int, int]
PACKED_SHAPE: Shape = (1, 64, 16, 32)
Y_SHAPE: Shape = (1, 256, 16, 32)
Z_SHAPE: Shape = (1, 128, 4, 8)
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
OUTPUT_SHAPES = (Z_SHAPE, *(PACKED_SHAPE for _ in range(8)), Y_SHAPE)


def build_mask_4x(phase_order: Sequence[int]) -> torch.Tensor:
    phases: List[torch.Tensor] = []
    for phase in range(4):
        mask = torch.zeros(PACKED_SHAPE, dtype=torch.float32)
        row = phase // 2
        column = phase % 2
        mask[:, :, row::2, column::2] = 1.0
        phases.append(mask)
    return torch.cat(tuple(phases[index] for index in phase_order), dim=1)


class IEntropyDecoderMerged(nn.Module):
    """Source-equivalent serial I prior with decoded symbols as placeholders."""

    PHASE_ORDERS = (
        (0, 1, 2, 3),
        (3, 2, 1, 0),
        (2, 3, 0, 1),
        (1, 0, 3, 2),
    )

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.hyper_dec = model.hyper_dec
        self.prior_fusion = model.y_prior_fusion
        self.reduction = model.y_spatial_prior_reduction
        self.adaptor_1 = model.y_spatial_prior_adaptor_1
        self.adaptor_2 = model.y_spatial_prior_adaptor_2
        self.adaptor_3 = model.y_spatial_prior_adaptor_3
        self.spatial_prior = model.y_spatial_prior
        for stage, order in enumerate(self.PHASE_ORDERS):
            self.register_buffer("mask_{}".format(stage), build_mask_4x(order))

    @staticmethod
    def pack_4x(value: torch.Tensor) -> torch.Tensor:
        part_0, part_1, part_2, part_3 = value.chunk(4, dim=1)
        return (part_0 + part_1) + (part_2 + part_3)

    @staticmethod
    def restore(symbols: torch.Tensor, packed_means: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        packed = symbols + packed_means
        return torch.cat((packed, packed, packed, packed), dim=1) * mask

    def stage_params(
        self,
        y_hat_so_far: torch.Tensor,
        reduced: torch.Tensor,
        adaptor: nn.Module,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scales, means = self.spatial_prior(
            adaptor(torch.cat((y_hat_so_far, reduced), dim=1))
        ).chunk(2, dim=1)
        return self.pack_4x(scales * mask), self.pack_4x(means * mask)

    def forward(
        self,
        z_hat: torch.Tensor,
        y_q_0: torch.Tensor,
        y_q_1: torch.Tensor,
        y_q_2: torch.Tensor,
        y_q_3: torch.Tensor,
    ):
        common = self.prior_fusion(self.hyper_dec(z_hat))[:, :, :16, :32].contiguous()
        q = common[:, :2, :, :]
        q_dec = (torch.sigmoid(q) * 1.5 + 0.5)[:, 1:2, :, :]
        scales, means = common[:, 2:, :, :].chunk(2, dim=1)
        reduced = self.reduction(common)

        scales_0 = self.pack_4x(scales * self.mask_0)
        means_0 = self.pack_4x(means * self.mask_0)
        y_hat_0 = self.restore(y_q_0, means_0, self.mask_0)

        scales_1, means_1 = self.stage_params(y_hat_0, reduced, self.adaptor_1, self.mask_1)
        y_hat_1 = self.restore(y_q_1, means_1, self.mask_1)
        y_hat_so_far = y_hat_0 + y_hat_1

        scales_2, means_2 = self.stage_params(y_hat_so_far, reduced, self.adaptor_2, self.mask_2)
        y_hat_2 = self.restore(y_q_2, means_2, self.mask_2)
        y_hat_so_far = y_hat_so_far + y_hat_2

        scales_3, means_3 = self.stage_params(y_hat_so_far, reduced, self.adaptor_3, self.mask_3)
        y_hat_3 = self.restore(y_q_3, means_3, self.mask_3)
        y_hat = (y_hat_so_far + y_hat_3) * q_dec

        return (
            z_hat,
            y_q_0,
            y_q_1,
            y_q_2,
            y_q_3,
            scales_0,
            scales_1,
            scales_2,
            scales_3,
            y_hat,
        )


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


def verify(model: nn.Module, merged: IEntropyDecoderMerged) -> Tuple[bool, List[dict]]:
    torch.manual_seed(20260811)
    z_hat = torch.clamp(torch.round(torch.randn(Z_SHAPE)), -128.0, 127.0)
    y_stages = tuple(torch.clamp(torch.round(torch.randn(PACKED_SHAPE)), -8.0, 8.0) for _ in range(4))
    with torch.no_grad():
        outputs = tuple(merged(z_hat, *y_stages))
        common = model.y_prior_fusion(model.hyper_dec(z_hat))[:, :, :16, :32].contiguous()
        _, q_dec, scales, means = model.separate_prior(common, False)
        reduced = model.y_spatial_prior_reduction(common)
        masks = model.get_mask_4x(1, 256, 16, 32, common.dtype, common.device)
        expected = [z_hat, *y_stages]
        y_hat_so_far = None
        expected_scales = []
        for stage in range(4):
            if stage > 0:
                params = torch.cat((y_hat_so_far, reduced), dim=1)
                scales, means = model.y_spatial_prior(
                    getattr(model, "y_spatial_prior_adaptor_{}".format(stage))(params)
                ).chunk(2, dim=1)
            packed_scales = model.single_part_for_writing_4x(scales * masks[stage])
            packed_means = model.single_part_for_writing_4x(means * masks[stage])
            expected_scales.append(packed_scales)
            restored = torch.cat(tuple(y_stages[stage] + packed_means for _ in range(4)), dim=1) * masks[stage]
            y_hat_so_far = restored if y_hat_so_far is None else y_hat_so_far + restored
        expected.extend(expected_scales)
        expected.append(y_hat_so_far * q_dec)
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
    android_root = args.android_root.resolve()
    output_dir = (args.output_dir or android_root / "outputs" / "i_entropy_decoder_merged_nhwc").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "i_entropy_decoder_merged_nhwc_manifest.json"
    converter = find_tool(args.pytorch_converter, "mtk_pytorch_converter")
    ncc = find_ncc(args.ncc_tflite)

    print("[i-entropy-decoder-merged] loading I checkpoint", flush=True)
    model, _, checkpoint_sha = load_i_model(source_root)
    model = model.cpu().eval()
    merged = IEntropyDecoderMerged(model).cpu().eval()
    verification_passed, verification = verify(model, merged)
    print("[i-entropy-decoder-merged] source verification exact={}".format(verification_passed), flush=True)

    candidate = Candidate(
        "i_entropy_decode_merged_base",
        "i_entropy_decoder",
        "i",
        (("i_z_hat", Z_SHAPE), *(("i_y_q_w_{}".format(stage), PACKED_SHAPE) for stage in range(4))),
        tuple(zip(OUTPUT_NAMES, OUTPUT_SHAPES)),
        lambda unused_model, unused_qp: merged,
    )
    metadata = {
        "tool": Path(__file__).name,
        "source_root": str(source_root),
        "scope": "I serial hyper/prior decode skeleton before rANS custom-op injection",
        "checkpoint_sha256": {"i": checkpoint_sha},
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
