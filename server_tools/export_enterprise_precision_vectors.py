#!/usr/bin/env python3
"""Export chained precision vectors for the enterprise Large/Small DLA packages."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


TensorItem = Tuple[str, torch.Tensor]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_frame(index: int, value_min: float, value_max: float) -> torch.Tensor:
    height, width = 256, 512
    y = torch.linspace(0.0, 1.0, height, dtype=torch.float32).reshape(1, height, 1, 1)
    x = torch.linspace(0.0, 1.0, width, dtype=torch.float32).reshape(1, 1, width, 1)
    phase = float(index) * 0.071
    red = torch.remainder(x + phase, 1.0).expand(1, height, width, 1)
    green = torch.remainder(y + phase * 0.5, 1.0).expand(1, height, width, 1)
    blue = torch.remainder((x + y) * 0.5 + phase * 0.25, 1.0).expand(1, height, width, 1)
    frame = torch.cat((red, green, blue), dim=3)
    return frame * (value_max - value_min) + value_min


class VectorWriter:
    def __init__(
        self,
        root: Path,
        model_id: str,
        checkpoint_sha256: Any,
        delivery_manifest: Path,
        value_range: Sequence[float],
    ) -> None:
        self.root = root
        self.model_id = model_id
        self.records: List[Dict[str, Any]] = []
        self.delivery_manifest = delivery_manifest
        delivery = json.loads(delivery_manifest.read_text(encoding="utf-8"))
        if delivery.get("resolution") != {"height": 256, "width": 512}:
            raise ValueError("delivery manifest is not fixed to 256x512")
        delivery_qp = delivery.get("qp", delivery.get("fixed_q_index"))
        if delivery_qp != 0:
            raise ValueError("delivery manifest is not fixed to QP 0")
        self.manifest: Dict[str, Any] = {
            "schema_version": 1,
            "model": model_id,
            "boundary": "direct_latent_no_entropy",
            "resolution": {"height": 256, "width": 512},
            "qp": 0,
            "layout": "NHWC",
            "dtype": "float32_le",
            "frame_count": 3,
            "frame_value_range": list(value_range),
            "checkpoint_sha256": checkpoint_sha256,
            "delivery_manifest_sha256": sha256(delivery_manifest),
            "delivery_package": delivery.get("package"),
            "delivery_models": [
                {
                    "name": model.get("name"),
                    "file": model.get("file"),
                    "sha256": model.get("sha256"),
                }
                for model in delivery.get("models", [])
            ],
            "thresholds": {"max_abs": 1e-3, "rmse": 1e-4},
            "stages": self.records,
        }

    def _save(self, relative: Path, value: torch.Tensor) -> Dict[str, Any]:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        array = value.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        array.tofile(str(path))
        return {
            "file": relative.as_posix(),
            "shape": list(array.shape),
            "elements": int(array.size),
            "bytes": int(array.nbytes),
            "sha256": sha256(path),
        }

    def add_stage(
        self,
        stage_id: str,
        model_name: str,
        inputs: Iterable[TensorItem],
        outputs: Iterable[TensorItem],
    ) -> None:
        base = Path("vectors") / stage_id
        input_records = []
        for name, value in inputs:
            record = self._save(base / "inputs" / (name + ".f32le"), value)
            record["name"] = name
            input_records.append(record)
        output_records = []
        for name, value in outputs:
            record = self._save(base / "expected" / (name + ".f32le"), value)
            record.update(
                {
                    "name": name,
                    "vendor_file": (Path(stage_id) / (name + ".f32le")).as_posix(),
                    "is_reconstructed_frame": "reconstructed_frame" in name or "reference_frame" in name,
                }
            )
            output_records.append(record)
        self.records.append(
            {
                "order": len(self.records),
                "id": stage_id,
                "model": model_name,
                "inputs": input_records,
                "expected_outputs": output_records,
            }
        )

    def finish(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.root / "precision_manifest.json"
        manifest_path.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        readme = """# 企业 DLA 精度向量

按 `precision_manifest.json` 中的顺序运行各 stage。所有 Tensor 均为 NHWC、
小端 FP32。每个 DLA 输出必须按 `vendor_file` 指定的路径写入
`vendor_outputs/`，随后使用 `compare_outputs.py` 生成对比结果。

该向量验证直接 latent 重建边界。
"""
        (self.root / "README.md").write_text(readme, encoding="utf-8")
        compare_source = Path(__file__).with_name("compare_enterprise_precision_outputs.py")
        if not compare_source.is_file():
            raise FileNotFoundError("missing comparison script: {}".format(compare_source))
        (self.root / "compare_outputs.py").write_bytes(compare_source.read_bytes())
        checksums = sorted(path for path in self.root.rglob("*") if path.is_file())
        checksum_path = self.root / "SHA256SUMS.txt"
        checksum_path.write_text(
            "".join(
                "{}  {}\n".format(sha256(path), path.relative_to(self.root).as_posix())
                for path in checksums
                if path != checksum_path
            ),
            encoding="ascii",
        )
        return manifest_path


def as_tuple(value: Any) -> Tuple[torch.Tensor, ...]:
    return value if isinstance(value, tuple) else (value,)


def add_executed_stage(
    writer: VectorWriter,
    stage_id: str,
    model_name: str,
    module: torch.nn.Module,
    inputs: Sequence[TensorItem],
    output_names: Sequence[str],
) -> Tuple[torch.Tensor, ...]:
    values = tuple(value for _, value in inputs)
    with torch.no_grad():
        outputs = as_tuple(module(*values))
    if len(outputs) != len(output_names):
        raise RuntimeError("{} returned {} outputs, expected {}".format(stage_id, len(outputs), len(output_names)))
    writer.add_stage(stage_id, model_name, inputs, zip(output_names, outputs))
    return outputs


def import_small_helpers(source_root: Path):
    path = source_root / "server_tools" / "export_gvc_rt_small_enterprise_dla.py"
    if not path.is_file():
        raise FileNotFoundError("missing Small helper exporter: {}".format(path))
    spec = importlib.util.spec_from_file_location("gvc_rt_small_export_helpers", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export_small(args: argparse.Namespace, writer: VectorWriter) -> None:
    helpers = import_small_helpers(args.source_root)
    model = helpers.load_model(args.source_root, args.checkpoint).cpu().eval()
    temporal = helpers.TemporalReferenceNhwc(model, args.qp).cpu().eval()
    encoder = helpers.EncoderAnalysisNhwc(model, args.qp).cpu().eval()
    decoder = helpers.DecoderSynthesisNhwc(model, args.qp).cpu().eval()
    reference = torch.zeros((1, 32, 64, 96), dtype=torch.float32)
    for frame_index in range(3):
        prefix = "frame_{:03d}".format(frame_index)
        frame = deterministic_frame(frame_index, 0.0, 1.0)
        ctx, ctx_t, memory = add_executed_stage(
            writer,
            prefix + "_temporal_reference",
            "temporal_reference",
            temporal,
            (("ref_feature", reference),),
            ("ctx", "ctx_t", "memory"),
        )
        (latent,) = add_executed_stage(
            writer,
            prefix + "_encoder",
            "encoder",
            encoder,
            (("frame", frame), ("ctx", ctx)),
            ("latent_y",),
        )
        reference, reconstructed = add_executed_stage(
            writer,
            prefix + "_decoder",
            "decoder",
            decoder,
            (("latent_y", latent), ("ctx", ctx), ("memory", memory)),
            ("next_ref_feature", "reconstructed_frame"),
        )


def export_large(args: argparse.Namespace, writer: VectorWriter) -> None:
    from export_clean_gvcrt_modules import TemporalFromFeature
    from export_decoder_full_norm_rewrite_nhwc import IFullSynthesisNhwc, PFullSynthesisNhwc
    from gvcrt_export_common import load_i_model, load_p_model
    from export_three_modules_offline_nhwc import (
        IEncoderDirectNhwc,
        NhwcBoundary,
        PEncoderDirectNhwc,
        TemporalFromFrameDirectNhwc,
    )

    i_model, _, i_checkpoint_sha = load_i_model(args.source_root)
    p_model, _, p_checkpoint_sha = load_p_model(args.source_root)
    writer.manifest["checkpoint_sha256"] = {
        "i": i_checkpoint_sha,
        "p": p_checkpoint_sha,
    }
    i_model = i_model.cpu().eval()
    p_model = p_model.cpu().eval()
    i_encoder = IEncoderDirectNhwc(i_model, args.qp).cpu().eval()
    p_encoder = PEncoderDirectNhwc(p_model, args.qp).cpu().eval()
    temporal_from_frame = TemporalFromFrameDirectNhwc(p_model, args.qp).cpu().eval()
    temporal_from_feature = NhwcBoundary(TemporalFromFeature(p_model, args.qp)).cpu().eval()
    i_decoder = IFullSynthesisNhwc(i_model, args.qp).cpu().eval()
    p_decoder = PFullSynthesisNhwc(p_model, args.qp).cpu().eval()

    frame0 = deterministic_frame(0, -1.0, 1.0)
    (i_latent,) = add_executed_stage(
        writer,
        "frame_000_i_encoder",
        "i_encoder",
        i_encoder,
        (("input_i_frame", frame0),),
        ("i_y_pre_prior",),
    )
    (i_reference_frame,) = add_executed_stage(
        writer,
        "frame_000_i_decoder",
        "i_decoder",
        i_decoder,
        (("i_y_hat", i_latent),),
        ("i_reference_frame",),
    )

    reference_feature, ctx, ctx_t = add_executed_stage(
        writer,
        "frame_001_temporal_from_frame",
        "temporal_from_frame",
        temporal_from_frame,
        (("reference_frame", i_reference_frame),),
        ("reference_feature", "ctx", "ctx_t"),
    )
    frame1 = deterministic_frame(1, -1.0, 1.0)
    (p_latent,) = add_executed_stage(
        writer,
        "frame_001_p_encoder",
        "p_encoder",
        p_encoder,
        (("input_p_frame", frame1), ("p_ctx", ctx)),
        ("p_y_pre_prior",),
    )
    reference_feature, p_reference_frame = add_executed_stage(
        writer,
        "frame_001_p_decoder",
        "p_decoder",
        p_decoder,
        (("p_y_hat", p_latent), ("p_ctx", ctx)),
        ("p_reference_feature", "p_reference_frame"),
    )

    adapted_feature, ctx, ctx_t = add_executed_stage(
        writer,
        "frame_002_temporal_from_feature",
        "temporal_from_feature",
        temporal_from_feature,
        (("reference_feature", reference_feature),),
        ("adapted_feature", "ctx", "ctx_t"),
    )
    frame2 = deterministic_frame(2, -1.0, 1.0)
    (p_latent,) = add_executed_stage(
        writer,
        "frame_002_p_encoder",
        "p_encoder",
        p_encoder,
        (("input_p_frame", frame2), ("p_ctx", ctx)),
        ("p_y_pre_prior",),
    )
    add_executed_stage(
        writer,
        "frame_002_p_decoder",
        "p_decoder",
        p_decoder,
        (("p_y_hat", p_latent), ("p_ctx", ctx)),
        ("p_reference_feature", "p_reference_frame"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("large", "small"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--delivery-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--qp", type=int, default=0)
    args = parser.parse_args()
    args.source_root = args.source_root.resolve()
    args.delivery_manifest = args.delivery_manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.qp != 0:
        raise ValueError("enterprise precision vectors are fixed to QP 0")
    if not args.delivery_manifest.is_file():
        raise FileNotFoundError(args.delivery_manifest)
    if args.model == "small":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for Small")
        args.checkpoint = args.checkpoint.resolve()
        checkpoint_hash: Any = sha256(args.checkpoint)
        model_id = "gvc-rt-small"
        value_range = (0.0, 1.0)
    else:
        checkpoint_hash = json.loads(args.delivery_manifest.read_text(encoding="utf-8")).get(
            "checkpoint_sha256", "bound_by_delivery_manifest"
        )
        model_id = "gvc-rt-large"
        value_range = (-1.0, 1.0)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("output directory must be new or empty: {}".format(args.output_dir))
    writer = VectorWriter(
        args.output_dir,
        model_id,
        checkpoint_hash,
        args.delivery_manifest,
        value_range,
    )
    if args.model == "small":
        export_small(args, writer)
    else:
        export_large(args, writer)
    manifest_path = writer.finish()

    archive = (args.archive or args.output_dir.with_suffix(".tar.gz")).resolve()
    with tarfile.open(str(archive), "w:gz") as package:
        package.add(str(args.output_dir), arcname=args.output_dir.name)
    print("model={}".format(model_id))
    print("stages={}".format(len(writer.records)))
    print("manifest={}".format(manifest_path))
    print("archive={}".format(archive))
    print("archive_sha256={}".format(sha256(archive)))


if __name__ == "__main__":
    main()
