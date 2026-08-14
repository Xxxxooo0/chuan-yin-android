#!/usr/bin/env python3
"""Plot ParkScene rate-distortion curves from server benchmark reports."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_large(input_dir):
    points = []
    for path in sorted(input_dir.glob("large_qp*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        points.append({
            "q": int(report["qp"]),
            "bpp": float(report["average_bpp"]),
            "psnr": float(report["mean_psnr_db"]),
            "ms_ssim": float(report["mean_ms_ssim"]),
            "lpips": float(report["mean_lpips"]),
            "dists": float(report["mean_dists"]),
        })
    return sorted(points, key=lambda point: point["bpp"])


def load_small(input_dir):
    points = []
    for path in sorted(input_dir.glob("small_q*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        mode = report["modes"]["source_entropy"]
        points.append({
            "q": int(report["q_index"]),
            "bpp": float(mode["entropy_bitstream"]["average_bpp"]),
            "psnr": float(mode["rgb_psnr_db"]["mean"]),
            "ms_ssim": float(mode["rgb_ms_ssim"]["mean"]),
            "lpips": float(mode["mean_lpips"]),
            "dists": float(mode["mean_dists"]),
        })
    return sorted(points, key=lambda point: point["bpp"])


def draw_curve(axis, points, metric, label, color, marker):
    axis.plot(
        [point["bpp"] for point in points],
        [point[metric] for point in points],
        color=color,
        marker=marker,
        linewidth=2.2,
        markersize=6,
        label=label,
    )
    for point in points:
        axis.annotate(
            "Q{}".format(point["q"]),
            (point["bpp"], point[metric]),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )


def main():
    args = parse_args()
    large = load_large(args.input_dir)
    small = load_small(args.input_dir)
    if not large or not small:
        raise RuntimeError("both Large and Small reports are required")

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=160)
    figure.suptitle("ParkScene 512x256, 96 frames @ 24 fps", fontsize=15, fontweight="bold")

    for axis, metric, ylabel in (
        (axes[0, 0], "psnr", "RGB PSNR (dB) - higher is better"),
        (axes[0, 1], "ms_ssim", "RGB MS-SSIM - higher is better"),
        (axes[1, 0], "lpips", "RGB LPIPS - lower is better"),
        (axes[1, 1], "dists", "RGB DISTS - lower is better"),
    ):
        draw_curve(axis, large, metric, "GVC-RT-Large", "#d62728", "o")
        draw_curve(axis, small, metric, "GVC-RT-Small", "#2ca02c", "s")
        axis.set_xlabel("Entropy payload (bpp)")
        axis.set_ylabel(ylabel)
        axis.set_xlim(left=0.0)
        axis.grid(True, color="#c8c8c8", linewidth=0.8, alpha=0.75)

        large_match = next(point for point in large if point["q"] == 3)
        small_match = next(point for point in small if point["q"] == 11)
        axis.scatter(
            [large_match["bpp"], small_match["bpp"]],
            [large_match[metric], small_match[metric]],
            marker="*",
            s=150,
            color="#ffbf00",
            edgecolor="#333333",
            linewidth=0.8,
            zorder=5,
            label="Matched bitrate",
        )
        axis.legend(loc="best", frameon=True)

    figure.text(
        0.5,
        0.015,
        "Rate = y+z rANS payload only. Highlighted pair: Small Q11 9.488 kbps vs Large Q3 9.694 kbps.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout(rect=(0.02, 0.045, 0.98, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    print("wrote {}".format(args.output))


if __name__ == "__main__":
    main()
