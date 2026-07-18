#!/usr/bin/env python3
"""Audit the largest stateless GVC-RT module boundaries for offline MTK DLA.

Entropy coding, bitstream parsing, and the serial masked-prior loops remain
outside these graphs. This exporter tests the largest continuous neural graph
on each side of those CPU/stateful boundaries.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import torch
from torch import nn

from analyze_recon_neuron_support import analyze_one, find_ncc
from export_clean_gvcrt_modules import (
    IRecon,
    PRecon,
    TemporalFromFeature,
)
from export_recon_diagnostic import PROJECT_ROOT, find_tool, load_i_model, load_p_model, sha256


Shape = Tuple[int, int, int, int]


class FixedPixelUnshuffle8(nn.Module):
    """Exact PixelUnshuffle(8) expressed as a fixed stride-8 Conv2D."""

    def __init__(self) -> None:
        super().__init__()
        factor = 8
        channels = 3
        conv = nn.Conv2d(
            channels,
            channels * factor * factor,
            kernel_size=factor,
            stride=factor,
            padding=0,
            bias=False,
        )
        with torch.no_grad():
            conv.weight.zero_()
            for channel in range(channels):
                for row in range(factor):
                    for column in range(factor):
                        output_channel = channel * factor * factor + row * factor + column
                        conv.weight[output_channel, channel, row, column] = 1.0
        conv.weight.requires_grad_(False)
        self.conv = conv

    def forward(self, frame_nhwc: torch.Tensor) -> torch.Tensor:
        frame_nchw = frame_nhwc.permute(0, 3, 1, 2).contiguous()
        return self.conv(frame_nchw)


class TemporalFromFrameDirectNhwc(nn.Module):
    def __init__(self, model: nn.Module, qp: int) -> None:
        super().__init__()
        self.feature_adaptor_i = model.feature_adaptor_i
        self.feature_extractor = model.feature_extractor
        self.pixel_unshuffle = FixedPixelUnshuffle8()
        self.register_buffer("q_feature", model.q_scale_feature[qp : qp + 1].detach().clone())

    def forward(self, reference_frame_nhwc: torch.Tensor):
        feature = self.feature_adaptor_i(self.pixel_unshuffle(reference_frame_nhwc))
        ctx, ctx_t = self.feature_extractor(feature, self.q_feature)
        return tuple(
            value.permute(0, 2, 3, 1).contiguous() for value in (feature, ctx, ctx_t)
        )


class IEncoderDirectNhwc(nn.Module):
    def __init__(self, model: nn.Module, qp: int) -> None:
        super().__init__()
        self.enc = model.enc
        self.pixel_unshuffle = FixedPixelUnshuffle8()
        self.register_buffer("q_enc", model.q_scale_enc[qp : qp + 1].detach().clone())

    def forward(self, frame_nhwc: torch.Tensor):
        feature = self.pixel_unshuffle(frame_nhwc)
        y = self.enc.forward_torch(feature, self.q_enc)
        return y.permute(0, 2, 3, 1).contiguous()


class PEncoderDirectNhwc(nn.Module):
    def __init__(self, model: nn.Module, qp: int) -> None:
        super().__init__()
        self.enc = model.enc
        self.pixel_unshuffle = FixedPixelUnshuffle8()
        self.register_buffer("q_enc", model.q_scale_enc[qp : qp + 1].detach().clone())

    def forward(self, frame_nhwc: torch.Tensor, ctx_nhwc: torch.Tensor):
        feature = self.pixel_unshuffle(frame_nhwc)
        ctx = ctx_nhwc.permute(0, 3, 1, 2).contiguous()
        y = self.enc.forward_torch(feature, ctx, self.q_enc)
        return y.permute(0, 2, 3, 1).contiguous()


@dataclass(frozen=True)
class Candidate:
    name: str
    group: str
    family: str
    inputs_nchw: tuple[tuple[str, Shape], ...]
    outputs_nchw: tuple[tuple[str, Shape], ...]
    build: Callable[[nn.Module, int], nn.Module]
    direct_nhwc: bool = False


CANDIDATES = (
    Candidate(
        "temporal_from_frame_big",
        "temporal_reference",
        "p",
        (("reference_frame", (1, 3, 256, 512)),),
        (
            ("reference_feature", (1, 256, 32, 64)),
            ("ctx", (1, 256, 32, 64)),
            ("ctx_t", (1, 256, 32, 64)),
        ),
        lambda model, qp: TemporalFromFrameDirectNhwc(model, qp),
        True,
    ),
    Candidate(
        "temporal_from_feature_big",
        "temporal_reference",
        "p",
        (("reference_feature", (1, 256, 32, 64)),),
        (
            ("adapted_feature", (1, 256, 32, 64)),
            ("ctx", (1, 256, 32, 64)),
            ("ctx_t", (1, 256, 32, 64)),
        ),
        lambda model, qp: TemporalFromFeature(model, qp),
    ),
    Candidate(
        "i_encoder_analysis_big",
        "complete_encoder",
        "i",
        (("input_i_frame", (1, 3, 256, 512)),),
        (("i_y_pre_prior", (1, 256, 16, 32)),),
        lambda model, qp: IEncoderDirectNhwc(model, qp),
        True,
    ),
    Candidate(
        "p_encoder_analysis_big",
        "complete_encoder",
        "p",
        (
            ("input_p_frame", (1, 3, 256, 512)),
            ("p_ctx", (1, 256, 32, 64)),
        ),
        (("p_y_pre_prior", (1, 128, 16, 32)),),
        lambda model, qp: PEncoderDirectNhwc(model, qp),
        True,
    ),
    Candidate(
        "i_decoder_synthesis_big",
        "complete_decoder",
        "i",
        (("i_y_hat", (1, 256, 16, 32)),),
        (
            ("i_codeword", (1, 18, 16, 32)),
            ("i_reference_frame", (1, 3, 256, 512)),
        ),
        lambda model, qp: IRecon(model, qp),
    ),
    Candidate(
        "p_decoder_synthesis_big",
        "complete_decoder",
        "p",
        (
            ("p_y_hat", (1, 128, 16, 32)),
            ("p_ctx", (1, 256, 32, 64)),
        ),
        (
            ("p_reference_feature", (1, 256, 32, 64)),
            ("p_reference_frame", (1, 3, 256, 512)),
        ),
        lambda model, qp: PRecon(model, qp),
    ),
)


def to_nhwc(shape: Shape) -> Shape:
    n, c, h, w = shape
    return n, h, w, c


class NhwcBoundary(nn.Module):
    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.source = source

    def forward(self, *inputs_nhwc: torch.Tensor):
        inputs_nchw = tuple(
            tensor.permute(0, 3, 1, 2).contiguous() for tensor in inputs_nhwc
        )
        outputs = self.source(*inputs_nchw)
        if isinstance(outputs, tuple):
            return tuple(item.permute(0, 2, 3, 1).contiguous() for item in outputs)
        return outputs.permute(0, 2, 3, 1).contiguous()


def run(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        return subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        ).returncode


def write_manifest(path: Path, records: list[dict], metadata: dict) -> None:
    summary = {
        "selected": len(metadata["selected_candidates"]),
        "finished": len(records),
        "converter_ok": sum(1 for item in records if item.get("converter_rc") == 0),
        "offline_compile_ok": sum(1 for item in records if item.get("offline_compile_ok")),
        "without_transpose_warning": sum(
            1
            for item in records
            if item.get("offline_compile_ok") and not item.get("transpose_warning")
        ),
        "failed": sum(1 for item in records if item.get("status") != "ok"),
    }
    path.write_text(
        json.dumps({**metadata, "summary": summary, "records": records}, indent=2),
        encoding="utf-8",
    )


def publish_offline_assets(
    records: list[dict],
    android_root: Path,
    checkpoint_sha256: dict[str, str],
    qp: int,
    arch: str,
) -> Path:
    assets_dir = android_root / "app" / "src" / "mtkOffline" / "assets" / "offline_models"
    assets_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = assets_dir / "module_offline_manifest.json"
    published: dict[str, dict] = {}
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        published = {
            item["name"]: item
            for item in existing.get("models", [])
            if isinstance(item, dict) and item.get("name")
        }

    for record in records:
        if record.get("status") != "ok" or not record.get("offline_compile_ok"):
            continue
        ncc_record = record.get("ncc") or {}
        source_dla = Path(ncc_record.get("dla") or "")
        if not source_dla.is_file():
            continue
        target_dla = assets_dir / f"{record['name']}_fp32.dla"
        shutil.copy2(source_dla, target_dla)
        published[record["name"]] = {
            "name": record["name"],
            "group": record.get("group"),
            "family": record.get("family"),
            "asset": f"offline_models/{target_dla.name}",
            "dla_sha256": sha256(target_dla),
            "tflite_sha256": record.get("tflite_sha256"),
            "input_names": record.get("input_names"),
            "input_shapes_nhwc": record.get("input_shapes_nhwc"),
            "output_names": record.get("output_names"),
            "output_shapes_nhwc": record.get("actual_torchscript_output_shapes_nhwc"),
            "offline_compile_verified": True,
            "precision_verified": False,
        }

    manifest = {
        "deployment_path": "mtk_offline",
        "component": "temporal_reference_and_encoder_analysis",
        "arch": arch,
        "qp": qp,
        "checkpoint_sha256": checkpoint_sha256,
        "models": [published[name] for name in sorted(published)],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def export_candidate(
    candidate: Candidate,
    model: nn.Module,
    checkpoint_sha256: str,
    qp: int,
    converter: str,
    ncc: str,
    arch: str,
    output_dir: Path,
) -> dict:
    input_shapes_nhwc = [to_nhwc(shape) for _, shape in candidate.inputs_nchw]
    expected_outputs_nhwc = [to_nhwc(shape) for _, shape in candidate.outputs_nchw]
    samples = tuple(torch.zeros(shape, dtype=torch.float32) for shape in input_shapes_nhwc)
    source = candidate.build(model, qp).cpu().eval()
    module = source if candidate.direct_nhwc else NhwcBoundary(source).cpu().eval()
    logs = output_dir / "logs"

    pt_path = output_dir / f"{candidate.name}_fp32.pt"
    with torch.no_grad():
        scripted = torch.jit.trace(module, samples, strict=False)
        actual = scripted(*samples)
        actual_tuple = actual if isinstance(actual, tuple) else (actual,)
        actual_shapes = [list(tensor.shape) for tensor in actual_tuple]
        scripted.save(str(pt_path))

    record = {
        "name": candidate.name,
        "group": candidate.group,
        "family": candidate.family,
        "boundary_note": "largest continuous neural graph; entropy/prior state loop excluded",
        "checkpoint_sha256": checkpoint_sha256,
        "qp": qp,
        "input_names": [name for name, _ in candidate.inputs_nchw],
        "input_shapes_nchw": [list(shape) for _, shape in candidate.inputs_nchw],
        "input_shapes_nhwc": [list(shape) for shape in input_shapes_nhwc],
        "output_names": [name for name, _ in candidate.outputs_nchw],
        "output_shapes_nchw": [list(shape) for _, shape in candidate.outputs_nchw],
        "expected_output_shapes_nhwc": [list(shape) for shape in expected_outputs_nhwc],
        "actual_torchscript_output_shapes_nhwc": actual_shapes,
        "torchscript": str(pt_path),
        "torchscript_sha256": sha256(pt_path),
    }

    tflite_path = output_dir / f"{candidate.name}_fp32.tflite"
    converter_command = [
        converter,
        "--input_script_module_file", str(pt_path),
        "--output_file", str(tflite_path),
        "--input_shapes", ":".join(
            ",".join(str(dim) for dim in shape) for shape in input_shapes_nhwc
        ),
    ]
    converter_log = logs / f"{candidate.name}_converter.log"
    converter_rc = run(converter_command, converter_log)
    record.update(
        {
            "converter_command": converter_command,
            "converter_rc": converter_rc,
            "converter_log": str(converter_log),
            "tflite": str(tflite_path) if tflite_path.is_file() else None,
            "tflite_sha256": sha256(tflite_path) if tflite_path.is_file() else None,
        }
    )
    if converter_rc != 0 or not tflite_path.is_file():
        record["status"] = "converter_failed"
        return record

    ncc_record = analyze_one(
        tflite=tflite_path,
        ncc=ncc,
        arch=arch,
        output_dir=output_dir / "ncc" / candidate.group,
        compile_dla=True,
        ncc_flags=["--opt-bw", "--relax-fp32"],
    )
    diagnostics = ncc_record.get("diagnostic_lines", []) + ncc_record.get(
        "dla_diagnostic_lines", []
    )
    record["ncc"] = ncc_record
    record["transpose_warning"] = any("transpose" in line.lower() for line in diagnostics)
    record["offline_compile_ok"] = (
        ncc_record.get("check_target_rc") == 0
        and ncc_record.get("exec_plan_rc") == 0
        and ncc_record.get("compile_dla_rc") == 0
        and bool(ncc_record.get("dla"))
        and Path(ncc_record["dla"]).is_file()
    )
    record["status"] = "ok" if record["offline_compile_ok"] else "ncc_failed"
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--android-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pytorch-converter", default=None)
    parser.add_argument("--ncc-tflite", required=True)
    parser.add_argument("--arch", default="mdla5.3")
    parser.add_argument("--qp", type=int, default=0)
    parser.add_argument(
        "--copy-offline-assets",
        action="store_true",
        help="Copy only successfully compiled DLA files into the mtkOffline flavor",
    )
    parser.add_argument(
        "--candidates",
        default="all",
        help="all or a comma-separated list of candidate names",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    android_root = args.android_root.resolve()
    output_dir = (
        args.output_dir or android_root / "outputs" / "three_modules_offline_nhwc"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "three_modules_offline_nhwc_manifest.json"
    converter = find_tool(args.pytorch_converter, "mtk_pytorch_converter")
    ncc = find_ncc(args.ncc_tflite)
    candidates_by_name = {candidate.name: candidate for candidate in CANDIDATES}
    if args.candidates == "all":
        selected = list(CANDIDATES)
    else:
        selected_names = [name.strip() for name in args.candidates.split(",") if name.strip()]
        unknown = [name for name in selected_names if name not in candidates_by_name]
        if unknown:
            raise ValueError(f"unknown --candidates entries: {unknown}")
        selected = [candidates_by_name[name] for name in selected_names]
    if not selected:
        raise ValueError("--candidates must select at least one candidate")

    print("[three-modules] loading I/P checkpoints", flush=True)
    i_model, _, i_checkpoint_sha = load_i_model(source_root)
    p_model, _, p_checkpoint_sha = load_p_model(source_root)
    models = {"i": i_model.cpu().eval(), "p": p_model.cpu().eval()}
    checkpoint_shas = {"i": i_checkpoint_sha, "p": p_checkpoint_sha}
    metadata = {
        "tool": Path(__file__).name,
        "source_root": str(source_root),
        "scope": {
            "temporal_reference": "from_frame and from_feature full neural paths",
            "complete_encoder": "I/P analysis transforms; hyper/prior/rANS excluded",
            "complete_decoder": "I/P synthesis and recon; rANS/prior reconstruction excluded",
        },
        "checkpoint_sha256": checkpoint_shas,
        "qp": args.qp,
        "layout": "NHWC external I/O; source operators unchanged",
        "frame_downsample": "exact PixelUnshuffle(8) via fixed one-hot Conv2D kernel=8 stride=8",
        "dtype": "FP32 external I/O; NCC --relax-fp32 enabled",
        "arch": args.arch,
        "ncc_flags": ["--opt-bw", "--relax-fp32", "--gen-debug-info"],
        "selected_candidates": [candidate.name for candidate in selected],
    }

    records: list[dict] = []
    for index, candidate in enumerate(selected, start=1):
        print(
            f"[three-modules] {index}/{len(selected)} "
            f"group={candidate.group} export={candidate.name}",
            flush=True,
        )
        try:
            record = export_candidate(
                candidate,
                models[candidate.family],
                checkpoint_shas[candidate.family],
                args.qp,
                converter,
                ncc,
                args.arch,
                output_dir,
            )
        except Exception as exc:
            record = {
                "name": candidate.name,
                "group": candidate.group,
                "family": candidate.family,
                "status": "exception",
                "error": repr(exc),
            }
        records.append(record)
        write_manifest(manifest_path, records, metadata)
        print(
            f"[three-modules] {candidate.name} status={record['status']} "
            f"transpose_warning={record.get('transpose_warning')}",
            flush=True,
        )
        gc.collect()

    print(f"wrote {manifest_path}")
    final = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(json.dumps(final["summary"], indent=2))
    if args.copy_offline_assets:
        offline_manifest = publish_offline_assets(
            records=records,
            android_root=android_root,
            checkpoint_sha256=checkpoint_shas,
            qp=args.qp,
            arch=args.arch,
        )
        published_count = sum(
            1 for record in records
            if record.get("status") == "ok" and record.get("offline_compile_ok")
        )
        print(f"published_offline_models={published_count} manifest={offline_manifest}")


if __name__ == "__main__":
    main()
