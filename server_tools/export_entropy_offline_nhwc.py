#!/usr/bin/env python3
"""Export and offline-compile the I/P entropy neural subgraphs for MTK DLA.

Only continuous neural and parameter transforms are exported. Mask generation,
rounding, symbol packing, CDF lookup, and rANS remain native CPU operations.
The hyper-prior and spatial-prior graphs are shared by encode and decode.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path
from typing import Callable, Tuple

import torch
from torch import nn

from analyze_recon_neuron_support import find_ncc
from gvcrt_export_common import PROJECT_ROOT, find_tool, load_i_model, load_p_model, sha256
from export_three_modules_offline_nhwc import Candidate, export_candidate, write_manifest


Shape = Tuple[int, int, int, int]
Build = Callable[[nn.Module, int], nn.Module]


class IHyperEncContinuous(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.model.hyper_enc(self.model.pad_for_y(y))


class IHyperPriorContinuous(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.hyper_dec = model.hyper_dec
        self.prior_fusion = model.y_prior_fusion

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        params = self.prior_fusion(self.hyper_dec(z_hat))
        return params[:, :, :16, :32].contiguous()


class IStage0Params(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, common: torch.Tensor):
        return self.model.separate_prior(common, False)


class IPriorReduce(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.reduction = model.y_spatial_prior_reduction

    def forward(self, common: torch.Tensor) -> torch.Tensor:
        return self.reduction(common)


class IPriorStage(nn.Module):
    def __init__(self, model: nn.Module, stage: int) -> None:
        super().__init__()
        self.adaptor = getattr(model, f"y_spatial_prior_adaptor_{stage}")
        self.spatial_prior = model.y_spatial_prior

    def forward(self, y_hat_so_far: torch.Tensor, reduced: torch.Tensor):
        return self.spatial_prior(self.adaptor(torch.cat((y_hat_so_far, reduced), dim=1))).chunk(2, 1)


class PHyperEncContinuous(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.model.hyper_enc(self.model.pad_for_y(y))


class PHyperPriorContinuous(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, z_hat: torch.Tensor, ctx_t: torch.Tensor) -> torch.Tensor:
        return self.model.res_prior_param_decoder(z_hat, ctx_t)


class PStage0Params(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, common: torch.Tensor):
        return self.model.separate_prior_for_video_decoding(common)


class PPriorStage1(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.spatial_prior = model.y_spatial_prior

    def forward(self, y_hat_so_far: torch.Tensor, common: torch.Tensor):
        return self.spatial_prior(torch.cat((y_hat_so_far, common), dim=1)).chunk(2, 1)


def candidate(
    name: str,
    family: str,
    inputs: tuple[tuple[str, Shape], ...],
    outputs: tuple[tuple[str, Shape], ...],
    build: Build,
) -> Candidate:
    return Candidate(name, "entropy_model", family, inputs, outputs, build)


CANDIDATES = (
    candidate(
        "i_hyper_enc_continuous", "i",
        (("i_y_pre_prior", (1, 256, 16, 32)),),
        (("i_z_pre_quant", (1, 128, 4, 8)),),
        lambda model, qp: IHyperEncContinuous(model),
    ),
    candidate(
        "i_hyper_prior_shared", "i",
        (("i_z_hat", (1, 128, 4, 8)),),
        (("i_common_params", (1, 514, 16, 32)),),
        lambda model, qp: IHyperPriorContinuous(model),
    ),
    candidate(
        "i_prior_stage0_params", "i",
        (("i_common_params", (1, 514, 16, 32)),),
        (
            ("i_q_enc", (1, 1, 16, 32)),
            ("i_q_dec", (1, 1, 16, 32)),
            ("i_stage0_scales", (1, 256, 16, 32)),
            ("i_stage0_means", (1, 256, 16, 32)),
        ),
        lambda model, qp: IStage0Params(model),
    ),
    candidate(
        "i_prior_reduce", "i",
        (("i_common_params", (1, 514, 16, 32)),),
        (("i_reduced_common_params", (1, 256, 16, 32)),),
        lambda model, qp: IPriorReduce(model),
    ),
    *tuple(
        candidate(
            f"i_prior_stage{stage}_continuous", "i",
            (
                ("i_y_hat_so_far", (1, 256, 16, 32)),
                ("i_reduced_common_params", (1, 256, 16, 32)),
            ),
            (
                (f"i_stage{stage}_scales", (1, 256, 16, 32)),
                (f"i_stage{stage}_means", (1, 256, 16, 32)),
            ),
            lambda model, qp, stage=stage: IPriorStage(model, stage),
        )
        for stage in (1, 2, 3)
    ),
    candidate(
        "p_hyper_enc_continuous", "p",
        (("p_y_pre_prior", (1, 128, 16, 32)),),
        (("p_z_pre_quant", (1, 128, 4, 8)),),
        lambda model, qp: PHyperEncContinuous(model),
    ),
    candidate(
        "p_hyper_prior_shared", "p",
        (
            ("p_z_hat", (1, 128, 4, 8)),
            ("p_ctx_t", (1, 256, 32, 64)),
        ),
        (("p_common_params", (1, 384, 16, 32)),),
        lambda model, qp: PHyperPriorContinuous(model),
    ),
    candidate(
        "p_prior_stage0_params", "p",
        (("p_common_params", (1, 384, 16, 32)),),
        (
            ("p_q_dec", (1, 128, 16, 32)),
            ("p_stage0_scales", (1, 128, 16, 32)),
            ("p_stage0_means", (1, 128, 16, 32)),
        ),
        lambda model, qp: PStage0Params(model),
    ),
    candidate(
        "p_prior_stage1_continuous", "p",
        (
            ("p_y_hat_so_far", (1, 128, 16, 32)),
            ("p_common_params", (1, 384, 16, 32)),
        ),
        (
            ("p_stage1_scales", (1, 128, 16, 32)),
            ("p_stage1_means", (1, 128, 16, 32)),
        ),
        lambda model, qp: PPriorStage1(model),
    ),
)


def publish(records: list[dict], android_root: Path, checkpoint_shas: dict, arch: str) -> Path:
    assets = android_root / "app" / "src" / "mtkOffline" / "assets" / "offline_models"
    assets.mkdir(parents=True, exist_ok=True)
    manifest_path = assets / "entropy_offline_manifest.json"
    models: list[dict] = []
    for record in records:
        if record.get("status") != "ok" or not record.get("offline_compile_ok"):
            continue
        source = Path((record.get("ncc") or {}).get("dla") or "")
        if not source.is_file():
            continue
        target = assets / f"{record['name']}_fp32.dla"
        shutil.copy2(source, target)
        models.append({
            "name": record["name"],
            "family": record["family"],
            "used_by": ["encoder"] if "hyper_enc" in record["name"] else ["encoder", "decoder"],
            "asset": f"offline_models/{target.name}",
            "dla_sha256": sha256(target),
            "tflite_sha256": record.get("tflite_sha256"),
            "input_names": record.get("input_names"),
            "input_shapes_nhwc": record.get("input_shapes_nhwc"),
            "output_names": record.get("output_names"),
            "output_shapes_nhwc": record.get("actual_torchscript_output_shapes_nhwc"),
            "offline_compile_verified": True,
            "precision_verified": False,
        })
    manifest_path.write_text(json.dumps({
        "deployment_path": "mtk_offline",
        "component": "entropy_neural_graphs",
        "arch": arch,
        "checkpoint_sha256": checkpoint_shas,
        "native_cpu_boundaries": ["mask", "round", "clamp", "symbol_pack", "cdf", "rans"],
        "models": sorted(models, key=lambda item: item["name"]),
    }, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--android-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pytorch-converter", default=None)
    parser.add_argument("--ncc-tflite", required=True)
    parser.add_argument("--arch", default="mdla5.3")
    parser.add_argument("--qp", type=int, default=0)
    parser.add_argument("--candidates", default="all")
    parser.add_argument("--copy-offline-assets", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    android_root = args.android_root.resolve()
    output_dir = (args.output_dir or android_root / "outputs" / "mtk_offline_entropy").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "entropy_offline_nhwc_manifest.json"
    converter = find_tool(args.pytorch_converter, "mtk_pytorch_converter")
    ncc = find_ncc(args.ncc_tflite)

    by_name = {item.name: item for item in CANDIDATES}
    names = list(by_name) if args.candidates == "all" else [x.strip() for x in args.candidates.split(",") if x.strip()]
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"unknown --candidates entries: {unknown}")
    selected = [by_name[name] for name in names]

    print("[entropy-offline] loading I/P checkpoints", flush=True)
    i_model, _, i_sha = load_i_model(source_root)
    p_model, _, p_sha = load_p_model(source_root)
    models = {"i": i_model.cpu().eval(), "p": p_model.cpu().eval()}
    shas = {"i": i_sha, "p": p_sha}
    metadata = {
        "tool": Path(__file__).name,
        "source_root": str(source_root),
        "scope": "I/P hyper encoder, shared hyper prior, and serial spatial-prior neural stages",
        "checkpoint_sha256": shas,
        "layout": "NHWC external I/O",
        "dtype": "FP32 external I/O; NCC --relax-fp32 enabled",
        "arch": args.arch,
        "native_cpu_boundaries": ["mask", "round", "clamp", "symbol_pack", "cdf", "rans"],
        "selected_candidates": names,
    }
    records: list[dict] = []
    for index, item in enumerate(selected, 1):
        print(f"[entropy-offline] {index}/{len(selected)} export={item.name}", flush=True)
        try:
            record = export_candidate(item, models[item.family], shas[item.family], args.qp, converter, ncc, args.arch, output_dir)
            record["used_by"] = ["encoder"] if "hyper_enc" in item.name else ["encoder", "decoder"]
        except Exception as exc:
            record = {"name": item.name, "family": item.family, "status": "exception", "error": repr(exc)}
        records.append(record)
        write_manifest(manifest_path, records, metadata)
        print(f"[entropy-offline] {item.name} status={record['status']}", flush=True)
        gc.collect()

    print(f"wrote {manifest_path}")
    print(json.dumps(json.loads(manifest_path.read_text(encoding="utf-8"))["summary"], indent=2))
    if args.copy_offline_assets:
        path = publish(records, android_root, shas, args.arch)
        print(f"published_manifest={path}")


if __name__ == "__main__":
    main()
