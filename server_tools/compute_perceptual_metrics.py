#!/usr/bin/env python3
"""Compute common LPIPS and DISTS metrics for saved RGB reconstructions."""

import argparse
import json
from pathlib import Path

import lpips
import numpy as np
import torch
from DISTS_pytorch import DISTS
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--reconstruction-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--report-type", choices=("large", "small"), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_reference(path):
    with Image.open(str(path)) as image:
        value = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(value.transpose(2, 0, 1).copy())


def load_reconstruction(path):
    value = np.load(str(path)).astype(np.float32, copy=False)
    if value.ndim != 3 or value.shape[0] != 3:
        raise RuntimeError("{} has invalid shape {}".format(path, value.shape))
    return torch.from_numpy(value.copy()).clamp(0.0, 1.0)


def main():
    args = parse_args()
    reference_paths = sorted(args.sequence_dir.glob("*.png"))
    reconstruction_paths = sorted(args.reconstruction_dir.glob("frame_*.npy"))
    if len(reference_paths) != len(reconstruction_paths):
        raise RuntimeError(
            "reference/reconstruction count mismatch: {} vs {}".format(
                len(reference_paths), len(reconstruction_paths)
            )
        )
    if not reference_paths:
        raise RuntimeError("no frames found")

    device = torch.device(args.device)
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    dists_model = DISTS().to(device).eval()
    lpips_values = []
    dists_values = []

    with torch.no_grad():
        for start in range(0, len(reference_paths), args.batch_size):
            stop = min(start + args.batch_size, len(reference_paths))
            reference = torch.stack(
                [load_reference(path) for path in reference_paths[start:stop]]
            ).to(device)
            reconstruction = torch.stack(
                [load_reconstruction(path) for path in reconstruction_paths[start:stop]]
            ).to(device)
            lpips_batch = lpips_model(reference * 2.0 - 1.0, reconstruction * 2.0 - 1.0)
            dists_batch = dists_model(reference, reconstruction, require_grad=False, batch_average=False)
            lpips_values.extend(lpips_batch.reshape(-1).detach().cpu().tolist())
            dists_values.extend(dists_batch.reshape(-1).detach().cpu().tolist())
            print("[perceptual] frames={}/{}".format(stop, len(reference_paths)), flush=True)

    report = json.loads(args.report.read_text(encoding="utf-8"))
    if args.report_type == "large":
        target = report
    else:
        target = report["modes"]["source_entropy"]
    frames = target["frames"]
    if len(frames) != len(lpips_values):
        raise RuntimeError("report frame count does not match metrics")
    for frame, lpips_value, dists_value in zip(frames, lpips_values, dists_values):
        frame["lpips"] = float(lpips_value)
        frame["dists"] = float(dists_value)
    target["mean_lpips"] = float(np.mean(lpips_values))
    target["mean_dists"] = float(np.mean(dists_values))
    target["perceptual_metric_backend"] = {
        "lpips": "lpips.LPIPS(net=alex, version=0.1)",
        "dists": "DISTS_pytorch.DISTS",
        "input": "RGB_float32_[0,1]",
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "[perceptual] mean_lpips={:.8f} mean_dists={:.8f}".format(
            target["mean_lpips"], target["mean_dists"]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
