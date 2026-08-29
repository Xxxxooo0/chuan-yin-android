#!/usr/bin/env python3
"""Measure GVC-RT-S sequence quality with the source model on the server.

The enterprise Small graphs consume NHWC YCbCr 4:4:4 tensors in [0, 1].
This script deliberately keeps video decoding, model execution, metrics, and
reconstruction generation in that color domain so Android RGB conversion is
not mixed into the model-quality result.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--q-index", type=int, default=9)
    parser.add_argument(
        "--modes",
        default="backbone,source",
        help="Comma-separated: backbone, backbone_reset, and/or source.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def run(command: List[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def decode_video(input_video: Path, raw_path: Path, width: int, height: int, fps: float) -> None:
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_video),
            "-vf",
            "scale={}:{}:flags=bicubic,fps={}".format(width, height, fps),
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
            str(raw_path),
        ]
    )


def iter_yuv420(path: Path, width: int, height: int, max_frames: int) -> Iterable[np.ndarray]:
    y_size = width * height
    uv_size = y_size // 4
    frame_size = y_size + uv_size * 2
    with path.open("rb") as source:
        index = 0
        while max_frames <= 0 or index < max_frames:
            packed = source.read(frame_size)
            if not packed:
                return
            if len(packed) != frame_size:
                raise RuntimeError("truncated YUV frame {}: {} bytes".format(index, len(packed)))
            data = np.frombuffer(packed, dtype=np.uint8)
            y = data[:y_size].reshape(height, width)
            u = data[y_size : y_size + uv_size].reshape(height // 2, width // 2)
            v = data[y_size + uv_size :].reshape(height // 2, width // 2)
            yuv = np.stack((y, u.repeat(2, 0).repeat(2, 1), v.repeat(2, 0).repeat(2, 1)), axis=0)
            yield yuv.astype(np.float32) / 255.0
            index += 1


def ycbcr_to_rgb(value: torch.Tensor) -> torch.Tensor:
    y, cb, cr = value.chunk(3, dim=1)
    kr, kg, kb = 0.2126, 0.7152, 0.0722
    r = y + (2.0 - 2.0 * kr) * (cr - 0.5)
    b = y + (2.0 - 2.0 * kb) * (cb - 0.5)
    g = (y - kr * r - kb * b) / kg
    return torch.cat((r, g, b), dim=1).clamp(0.0, 1.0)


def psnr(reference: torch.Tensor, actual: torch.Tensor) -> float:
    mse = torch.mean(torch.square(reference.float() - actual.float())).item()
    return float("inf") if mse == 0.0 else -10.0 * math.log10(mse)


def write_yuv420(output, value: torch.Tensor) -> None:
    value = value.detach().float().clamp(0.0, 1.0).cpu()
    y = value[:, 0:1]
    u = F.avg_pool2d(value[:, 1:2], 2, 2)
    v = F.avg_pool2d(value[:, 2:3], 2, 2)
    for plane in (y, u, v):
        packed = torch.round(plane * 255.0).to(torch.uint8).numpy()
        output.write(packed.tobytes())


def summarize(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "final": float(array[-1]),
    }


def encode_reconstruction(raw_path: Path, mp4_path: Path, width: int, height: int, fps: float) -> None:
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
            "{}x{}".format(width, height),
            "-r",
            str(fps),
            "-i",
            str(raw_path),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "0",
            "-pix_fmt",
            "yuv420p",
            str(mp4_path),
        ]
    )


def main() -> None:
    args = parse_args()
    if args.q_index != 9:
        raise ValueError("this sequence validation is fixed to q-index 9")
    source_root = args.source_root.resolve()
    checkpoint = args.checkpoint.resolve()
    input_video = args.input_video.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    if not set(modes).issubset({"backbone", "backbone_reset", "source"}):
        raise ValueError("modes must contain only backbone/backbone_reset/source")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(source_root / "video"))
    sys.path.insert(0, str(source_root / "server_tools"))
    from export_gvc_rt_small_enterprise_dla import (  # type: ignore
        DecoderSynthesisNhwc,
        EncoderAnalysisNhwc,
        TemporalReferenceFromFeatureNhwc,
        TemporalReferenceFromFrameNhwc,
        load_model,
    )
    from src.metrics.msssim import MS_SSIM  # type: ignore

    device = torch.device(args.device)
    print("[quality] loading model {}".format(checkpoint), flush=True)
    model = load_model(source_root, checkpoint).to(device).eval()
    metric_msssim = MS_SSIM(channels=3, data_range=1).to(device).eval()

    with tempfile.TemporaryDirectory(prefix="gvcrt-small-quality-") as temporary:
        decoded_path = Path(temporary) / "input.yuv"
        decode_video(input_video, decoded_path, args.width, args.height, args.fps)
        frames = list(iter_yuv420(decoded_path, args.width, args.height, args.max_frames))
    if not frames:
        raise RuntimeError("input video produced no frames")
    print("[quality] decoded_frames={}".format(len(frames)), flush=True)

    report = {
        "input_video": str(input_video),
        "checkpoint": str(checkpoint),
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "q_index": args.q_index,
        "frame_count": len(frames),
        "input_color": "YCbCr444_[0,1]_from_YUV420_nearest_chroma",
        "modes": {},
    }

    for mode in modes:
        print("[quality] mode={} start".format(mode), flush=True)
        raw_output = output_dir / ("{}_reconstructed.yuv".format(mode))
        rgb_psnr_values: List[float] = []
        yuv_psnr_values: List[float] = []
        msssim_values: List[float] = []
        frame_times_ms: List[float] = []

        if mode in {"backbone", "backbone_reset"}:
            temporal = TemporalReferenceFromFeatureNhwc(model, args.q_index).to(device).eval()
            temporal_from_frame = TemporalReferenceFromFrameNhwc(model, args.q_index).to(device).eval()
            encoder = EncoderAnalysisNhwc(model, args.q_index).to(device).eval()
            decoder = DecoderSynthesisNhwc(model, args.q_index).to(device).eval()
            ref_feature = torch.zeros((1, 32, 64, 96), dtype=torch.float32, device=device)
            ref_frame = torch.full((1, 256, 512, 3), 0.5, dtype=torch.float32, device=device)
        else:
            dpb = None

        with raw_output.open("wb") as writer, torch.no_grad():
            for frame_index, frame_array in enumerate(frames):
                source = torch.from_numpy(frame_array).unsqueeze(0).to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                if mode in {"backbone", "backbone_reset"}:
                    frame_nhwc = source.permute(0, 2, 3, 1).contiguous()
                    reset_from_frame = mode == "backbone_reset" and (
                        frame_index == 0 or frame_index % 64 == 1
                    )
                    if reset_from_frame:
                        ctx, _, memory = temporal_from_frame(ref_frame)
                    else:
                        ctx, _, memory = temporal(ref_feature)
                    latent = encoder(frame_nhwc, ctx)
                    ref_feature, reconstructed_nhwc = decoder(latent, ctx, memory)
                    ref_frame = reconstructed_nhwc
                    reconstructed = reconstructed_nhwc.permute(0, 3, 1, 2).contiguous()
                else:
                    if dpb is None:
                        dpb = {"ref_frame": torch.full_like(source, 0.5), "ref_feature": None}
                    if frame_index % 64 == 1:
                        dpb["ref_feature"] = None
                    fa_idx = model.frame_index_map[frame_index % len(model.frame_index_map)]
                    result = model.compress_core(
                        source,
                        dpb,
                        args.q_index,
                        fa_idx=fa_idx,
                        do_shift_qp=False,
                        get_recon=True,
                    )
                    dpb = result["dpb"]
                    reconstructed = dpb["ref_frame"]
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                frame_times_ms.append((time.perf_counter() - started) * 1000.0)

                source_rgb = ycbcr_to_rgb(source)
                reconstructed_rgb = ycbcr_to_rgb(reconstructed)
                rgb_psnr_values.append(psnr(source_rgb, reconstructed_rgb))
                yuv_psnr_values.append(psnr(source, reconstructed))
                msssim_values.append(float(metric_msssim(source_rgb, reconstructed_rgb).item()))
                write_yuv420(writer, reconstructed)
                if (frame_index + 1) % 10 == 0 or frame_index + 1 == len(frames):
                    print(
                        "[quality] mode={} frame={}/{} rgb_psnr={:.3f} msssim={:.6f}".format(
                            mode,
                            frame_index + 1,
                            len(frames),
                            rgb_psnr_values[-1],
                            msssim_values[-1],
                        ),
                        flush=True,
                    )

        mp4_output = output_dir / ("{}_reconstructed.mp4".format(mode))
        encode_reconstruction(raw_output, mp4_output, args.width, args.height, args.fps)
        mode_report = {
            "rgb_psnr_db": summarize(rgb_psnr_values),
            "ycbcr444_psnr_db": summarize(yuv_psnr_values),
            "rgb_ms_ssim": summarize(msssim_values),
            "model_frame_ms": summarize(frame_times_ms),
            "reconstructed_yuv": str(raw_output),
            "reconstructed_mp4": str(mp4_output),
        }
        report["modes"][mode] = mode_report
        print("[quality] mode={} summary={}".format(mode, json.dumps(mode_report)), flush=True)

    report_path = output_dir / "sequence_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote {}".format(report_path), flush=True)


if __name__ == "__main__":
    main()
