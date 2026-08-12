#!/usr/bin/env python3
"""Export the six Large neural graphs with runtime quantization inputs."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from analyze_recon_neuron_support import find_ncc
from export_decoder_full_norm_rewrite_nhwc import (
    ExplicitDepthConvBlock,
    FixedGenerator,
    FixedGroupNorm,
    FixedIFeatureDec,
    FixedPFeatureDec,
    FixedPixelUnshuffle2,
    IFullSynthesisNhwc,
    PFullSynthesisNhwc,
    standard_silu_no_fusion,
)
from export_three_modules_offline_nhwc import (
    Candidate,
    FixedPixelUnshuffle8,
    IEncoderDirectNhwc,
    NhwcBoundary,
    PEncoderDirectNhwc,
    TemporalFromFeature,
    TemporalFromFrameDirectNhwc,
    export_candidate,
    write_manifest,
)
from gvcrt_export_common import PROJECT_ROOT, find_tool, load_i_model, load_p_model, sha256


Shape = Tuple[int, int, int, int]
SUPPORTED_QPS = tuple(range(10))
VERIFICATION_QPS = (0, 3, 6, 9)


def q_nchw(q_nhwc: torch.Tensor) -> torch.Tensor:
    return q_nhwc.permute(0, 3, 1, 2).contiguous()


class DynamicTemporalFromFrame(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.pixel_unshuffle = FixedPixelUnshuffle8()
        self.feature_adaptor = model.feature_adaptor_i
        self.feature_extractor = model.feature_extractor

    def forward(self, frame: torch.Tensor, q_feature: torch.Tensor):
        feature = self.feature_adaptor(self.pixel_unshuffle(frame))
        ctx, ctx_t = self.feature_extractor(feature, q_nchw(q_feature))
        return tuple(value.permute(0, 2, 3, 1).contiguous() for value in (feature, ctx, ctx_t))


class DynamicTemporalFromFeature(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.feature_adaptor = model.feature_adaptor_p
        self.feature_extractor = model.feature_extractor

    def forward(self, feature_nhwc: torch.Tensor, q_feature: torch.Tensor):
        feature = self.feature_adaptor(feature_nhwc.permute(0, 3, 1, 2).contiguous())
        ctx, ctx_t = self.feature_extractor(feature, q_nchw(q_feature))
        return tuple(value.permute(0, 2, 3, 1).contiguous() for value in (feature, ctx, ctx_t))


class DynamicIEncoder(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.pixel_unshuffle = FixedPixelUnshuffle8()
        self.encoder = model.enc

    def forward(self, frame: torch.Tensor, q_enc: torch.Tensor):
        y = self.encoder.forward_torch(self.pixel_unshuffle(frame), q_nchw(q_enc))
        return y.permute(0, 2, 3, 1).contiguous()


class DynamicPEncoder(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.pixel_unshuffle = FixedPixelUnshuffle8()
        self.encoder = model.enc

    def forward(self, frame: torch.Tensor, ctx_nhwc: torch.Tensor, q_enc: torch.Tensor):
        ctx = ctx_nhwc.permute(0, 3, 1, 2).contiguous()
        y = self.encoder.forward_torch(self.pixel_unshuffle(frame), ctx, q_nchw(q_enc))
        return y.permute(0, 2, 3, 1).contiguous()


class DynamicIDecoder(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.feature_decoder = FixedIFeatureDec(model, 0)
        self.generator = FixedGenerator(model, 0)

    def forward(self, y_hat_nhwc: torch.Tensor, q_dec: torch.Tensor, q_recon: torch.Tensor):
        y_hat = y_hat_nhwc.permute(0, 3, 1, 2).contiguous()
        codeword = self.feature_decoder(y_hat, q_nchw(q_dec))
        frame = self.generator(codeword, q_nchw(q_recon))
        return frame.permute(0, 2, 3, 1).contiguous()


class DynamicPDecoder(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.feature_decoder = FixedPFeatureDec(model, 0)
        self.pixel_unshuffle = FixedPixelUnshuffle2()
        mlp = model.recon_generation_net.mlp
        self.mlp_norm0 = FixedGroupNorm(mlp[0], 16, 32)
        self.mlp_block0 = ExplicitDepthConvBlock(mlp[1])
        self.mlp_norm1 = FixedGroupNorm(mlp[2], 16, 32)
        self.mlp_block1 = ExplicitDepthConvBlock(mlp[3])
        self.generator = FixedGenerator(model, 0)

    def forward(
        self,
        y_hat_nhwc: torch.Tensor,
        ctx_nhwc: torch.Tensor,
        q_dec: torch.Tensor,
        q_recon: torch.Tensor,
    ):
        y_hat = y_hat_nhwc.permute(0, 3, 1, 2).contiguous()
        ctx = ctx_nhwc.permute(0, 3, 1, 2).contiguous()
        feature = self.feature_decoder(y_hat, ctx, q_nchw(q_dec))
        value = self.mlp_block0(self.mlp_norm0(self.pixel_unshuffle(feature)))
        value = standard_silu_no_fusion(self.mlp_norm1(value))
        codeword = self.mlp_block1(value)
        frame = self.generator(codeword, q_nchw(q_recon))
        return feature.permute(0, 2, 3, 1).contiguous(), frame.permute(0, 2, 3, 1).contiguous()


def candidates(i_model: nn.Module, p_model: nn.Module) -> Tuple[Candidate, ...]:
    return (
        Candidate("temporal_from_frame_dynamic_qp", "temporal_reference", "p",
                  (("reference_frame", (1, 3, 256, 512)), ("q_feature", (1, 256, 1, 1))),
                  (("reference_feature", (1, 256, 32, 64)), ("ctx", (1, 256, 32, 64)), ("ctx_t", (1, 256, 32, 64))),
                  lambda unused, qp: DynamicTemporalFromFrame(p_model), True),
        Candidate("temporal_from_feature_dynamic_qp", "temporal_reference", "p",
                  (("reference_feature", (1, 256, 32, 64)), ("q_feature", (1, 256, 1, 1))),
                  (("adapted_feature", (1, 256, 32, 64)), ("ctx", (1, 256, 32, 64)), ("ctx_t", (1, 256, 32, 64))),
                  lambda unused, qp: DynamicTemporalFromFeature(p_model), True),
        Candidate("i_encoder_dynamic_qp", "complete_encoder", "i",
                  (("input_i_frame", (1, 3, 256, 512)), ("q_enc", (1, 368, 1, 1))),
                  (("i_y_pre_prior", (1, 256, 16, 32)),),
                  lambda unused, qp: DynamicIEncoder(i_model), True),
        Candidate("p_encoder_dynamic_qp", "complete_encoder", "p",
                  (("input_p_frame", (1, 3, 256, 512)), ("p_ctx", (1, 256, 32, 64)), ("q_enc", (1, 256, 1, 1))),
                  (("p_y_pre_prior", (1, 128, 16, 32)),),
                  lambda unused, qp: DynamicPEncoder(p_model), True),
        Candidate("i_decoder_dynamic_qp", "complete_decoder", "i",
                  (("i_y_hat", (1, 256, 16, 32)), ("q_dec", (1, 512, 1, 1)), ("q_recon", (1, 320, 1, 1))),
                  (("i_reference_frame", (1, 3, 256, 512)),),
                  lambda unused, qp: DynamicIDecoder(i_model), True, False),
        Candidate("p_decoder_dynamic_qp", "complete_decoder", "p",
                  (("p_y_hat", (1, 128, 16, 32)), ("p_ctx", (1, 256, 32, 64)), ("q_dec", (1, 256, 1, 1)), ("q_recon", (1, 320, 1, 1))),
                  (("p_reference_feature", (1, 256, 32, 64)), ("p_reference_frame", (1, 3, 256, 512))),
                  lambda unused, qp: DynamicPDecoder(p_model), True, False),
    )


def write_scale_table(output: Path, name: str, tensor: torch.Tensor) -> dict:
    value = tensor[: len(SUPPORTED_QPS)].detach().cpu().contiguous().numpy().astype("<f4")
    path = output / "quant_scales" / "{}.f32le".format(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value.tobytes())
    return {
        "file": path.relative_to(output).as_posix(),
        "shape": list(value.shape),
        "bytes_per_qp": int(value[0].nbytes),
        "sha256": sha256(path),
    }


def stats(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    difference = (actual.float() - expected.float()).abs()
    return {
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "rmse": float(torch.sqrt(torch.mean(difference * difference)).item()),
        "exact": bool(torch.equal(actual, expected)),
    }


def compare_outputs(actual, expected) -> list[dict]:
    actual_tuple = actual if isinstance(actual, tuple) else (actual,)
    expected_tuple = expected if isinstance(expected, tuple) else (expected,)
    if len(actual_tuple) != len(expected_tuple):
        raise RuntimeError("verification output count mismatch")
    return [stats(a, e) for a, e in zip(actual_tuple, expected_tuple)]


def verify_dynamic_models(i_model: nn.Module, p_model: nn.Module) -> list[dict]:
    torch.manual_seed(20260813)
    frame = torch.randn((1, 256, 512, 3)) * 0.1
    feature = torch.randn((1, 32, 64, 256)) * 0.1
    ctx = torch.randn((1, 32, 64, 256)) * 0.1
    i_y = torch.randn((1, 16, 32, 256)) * 0.1
    p_y = torch.randn((1, 16, 32, 128)) * 0.1
    records = []
    with torch.no_grad():
        for qp in VERIFICATION_QPS:
            p_q_feature = p_model.q_scale_feature[qp:qp + 1].permute(0, 2, 3, 1).contiguous()
            i_q_enc = i_model.q_scale_enc[qp:qp + 1].permute(0, 2, 3, 1).contiguous()
            p_q_enc = p_model.q_scale_enc[qp:qp + 1].permute(0, 2, 3, 1).contiguous()
            i_q_dec = i_model.q_scale_dec[qp:qp + 1].permute(0, 2, 3, 1).contiguous()
            i_q_recon = i_model.q_scale_recon[qp:qp + 1].permute(0, 2, 3, 1).contiguous()
            p_q_dec = p_model.q_scale_dec[qp:qp + 1].permute(0, 2, 3, 1).contiguous()
            p_q_recon = p_model.q_scale_recon[qp:qp + 1].permute(0, 2, 3, 1).contiguous()
            checks = {
                "temporal_from_frame": compare_outputs(
                    DynamicTemporalFromFrame(p_model).eval()(frame, p_q_feature),
                    TemporalFromFrameDirectNhwc(p_model, qp).eval()(frame),
                ),
                "temporal_from_feature": compare_outputs(
                    DynamicTemporalFromFeature(p_model).eval()(feature, p_q_feature),
                    NhwcBoundary(TemporalFromFeature(p_model, qp)).eval()(feature),
                ),
                "i_encoder": compare_outputs(
                    DynamicIEncoder(i_model).eval()(frame, i_q_enc),
                    IEncoderDirectNhwc(i_model, qp).eval()(frame),
                ),
                "p_encoder": compare_outputs(
                    DynamicPEncoder(p_model).eval()(frame, ctx, p_q_enc),
                    PEncoderDirectNhwc(p_model, qp).eval()(frame, ctx),
                ),
                "i_decoder": compare_outputs(
                    DynamicIDecoder(i_model).eval()(i_y, i_q_dec, i_q_recon),
                    IFullSynthesisNhwc(i_model, qp).eval()(i_y),
                ),
                "p_decoder": compare_outputs(
                    DynamicPDecoder(p_model).eval()(p_y, ctx, p_q_dec, p_q_recon),
                    PFullSynthesisNhwc(p_model, qp).eval()(p_y, ctx),
                ),
            }
            passed = all(metric["max_abs"] <= 1e-6 for values in checks.values() for metric in values)
            records.append({"qp": qp, "passed": passed, "models": checks})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--android-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pytorch-converter", default=None)
    parser.add_argument("--ncc-tflite", required=True)
    parser.add_argument("--arch", default="mdla5.3")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    converter = find_tool(args.pytorch_converter, "mtk_pytorch_converter")
    ncc = find_ncc(args.ncc_tflite)
    print("[dynamic-qp] loading I/P checkpoints", flush=True)
    i_model, _, i_sha = load_i_model(source_root)
    p_model, _, p_sha = load_p_model(source_root)
    i_model, p_model = i_model.cpu().eval(), p_model.cpu().eval()
    scale_tables = {
        "i_q_enc": write_scale_table(output, "i_q_enc", i_model.q_scale_enc),
        "i_q_dec": write_scale_table(output, "i_q_dec", i_model.q_scale_dec),
        "i_q_recon": write_scale_table(output, "i_q_recon", i_model.q_scale_recon),
        "p_q_feature": write_scale_table(output, "p_q_feature", p_model.q_scale_feature),
        "p_q_enc": write_scale_table(output, "p_q_enc", p_model.q_scale_enc),
        "p_q_dec": write_scale_table(output, "p_q_dec", p_model.q_scale_dec),
        "p_q_recon": write_scale_table(output, "p_q_recon", p_model.q_scale_recon),
    }
    source_verification = verify_dynamic_models(i_model, p_model)
    if not all(record["passed"] for record in source_verification):
        raise RuntimeError("dynamic QP wrappers do not match fixed-QP source paths")
    records = []
    models = {"i": i_model, "p": p_model}
    selected = candidates(i_model, p_model)
    metadata = {
        "tool": Path(__file__).name,
        "source_root": str(source_root),
        "dynamic_qp": True,
        "supported_qps": list(SUPPORTED_QPS),
        "verification_qps": list(VERIFICATION_QPS),
        "checkpoint_sha256": {"i": i_sha, "p": p_sha},
        "quant_scale_tables": scale_tables,
        "source_verification": source_verification,
        "selected_candidates": [item.name for item in selected],
    }
    manifest = output / "large_dynamic_qp_export_manifest.json"
    for index, candidate in enumerate(selected, 1):
        print("[dynamic-qp] {}/6 export {}".format(index, candidate.name), flush=True)
        try:
            record = export_candidate(candidate, models[candidate.family],
                                      {"i": i_sha, "p": p_sha}[candidate.family], 0,
                                      converter, ncc, args.arch, output)
            record["online_package_eligible"] = (
                record.get("converter_rc") == 0 and bool(record.get("tflite"))
            )
            if record["online_package_eligible"]:
                record["status"] = "ok"
        except Exception as error:
            record = {"name": candidate.name, "status": "exception", "error": repr(error)}
        records.append(record)
        write_manifest(manifest, records, metadata)
        print("[dynamic-qp] {} status={}".format(candidate.name, record["status"]), flush=True)
        gc.collect()
    print("wrote {}".format(manifest))
    print(json.dumps(json.loads(manifest.read_text(encoding="utf-8"))["summary"], indent=2))


if __name__ == "__main__":
    main()
