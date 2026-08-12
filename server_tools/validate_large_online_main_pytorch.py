#!/usr/bin/env python3
"""Validate the Android Large I-P-P path with the original PyTorch codec."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_input(path, height, width, device):
    values = np.fromfile(str(path), dtype="<f4")
    expected = height * width * 3
    if values.size != expected:
        raise RuntimeError("input elements={} expected={}".format(values.size, expected))
    nchw = values.reshape(1, height, width, 3).transpose(0, 3, 1, 2).copy()
    return torch.from_numpy(nchw).to(device=device, dtype=torch.float16)


def load_nhwc_tensor(path, shape, device):
    values = np.fromfile(str(path), dtype="<f4")
    expected = int(np.prod(shape))
    if values.size != expected:
        raise RuntimeError("{} elements={} expected={}".format(path, values.size, expected))
    nchw = values.reshape(shape).transpose(0, 3, 1, 2).copy()
    return torch.from_numpy(nchw).to(device=device, dtype=torch.float16)


def load_models(source_root, device, force_zero_thres):
    sys.path.insert(0, str(source_root))
    from src.models.image_model_G_b import DMCI
    from src.models.video_model_G_b import DMC

    i_checkpoint = source_root / "ckpt" / "checkpoints" / "GVC-RT_B_I.pt"
    p_checkpoint = source_root / "ckpt" / "checkpoints" / "GVC-RT_B_P.pt"
    i_model = DMCI(encoder_ckpt_path=str(i_checkpoint)).to(device).eval()
    p_model = DMC()
    checkpoint = torch.load(str(p_checkpoint), map_location="cpu")
    state = checkpoint.get(
        "student_ema",
        checkpoint.get("student", checkpoint.get("state_dict", checkpoint)),
    )
    p_model.load_state_dict(state, strict=True)
    p_model = p_model.to(device).eval()
    i_model.update(force_zero_thres)
    p_model.update(force_zero_thres)
    i_model.half()
    p_model.half()
    return i_model, p_model, i_checkpoint, p_checkpoint


def display_tensor(value):
    return ((value.float().clamp(-1.0, 1.0) + 1.0) * 0.5).cpu()


def psnr(reference, actual):
    mse = torch.mean((display_tensor(reference) - display_tensor(actual)) ** 2).item()
    return float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def to_nhwc_numpy(value):
    return value.detach().float().cpu().numpy().transpose(0, 2, 3, 1).copy()


def compare_android(server_nhwc, android_path):
    android = np.fromfile(str(android_path), dtype="<f4").reshape(server_nhwc.shape)
    difference = np.abs(server_nhwc - android)
    return {
        "android_file": str(android_path),
        "android_sha256": sha256(android_path),
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "exact": bool(np.array_equal(server_nhwc, android)),
    }


def write_nhwc(value, output):
    nhwc = to_nhwc_numpy(value)
    nhwc.astype("<f4").tofile(str(output))
    return nhwc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--android-decoded-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--qp", type=int, default=0)
    parser.add_argument("--force-zero-thres", type=float, default=0.12)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because the original compress path uses CUDA events")
    source_root = args.source_root.resolve()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    frame = load_input(input_path, args.height, args.width, device)
    i_model, p_model, i_checkpoint, p_checkpoint = load_models(
        source_root, device, args.force_zero_thres
    )

    sps = {
        "height": args.height,
        "width": args.width,
        "ec_part": 0,
        "use_ada_i": 0,
    }
    with torch.no_grad():
        i_q_enc = i_model.q_scale_enc[args.qp : args.qp + 1]
        i_y = i_model.enc(frame, i_q_enc)
        i_z = i_model.hyper_enc(i_model.pad_for_y(i_y))
        i_z_hat = torch.clamp(torch.round(i_z), -128.0, 127.0)
        i_common = i_model.y_prior_fusion(i_model.hyper_dec(i_z_hat))
        i_common = i_common[:, :, : i_y.shape[2], : i_y.shape[3]].contiguous()
        i_prior = i_model.compress_prior_4x(
            i_y,
            i_common,
            i_model.y_spatial_prior_reduction,
            i_model.y_spatial_prior_adaptor_1,
            i_model.y_spatial_prior_adaptor_2,
            i_model.y_spatial_prior_adaptor_3,
            i_model.y_spatial_prior,
        )
        i_y_hat = i_prior[-1]

        encoded_i = i_model.compress(frame, args.qp)
        p_model.clear_dpb()
        p_model.set_curr_poc(0)
        p_model.add_ref_frame(None, encoded_i["x_hat"])
        encoded_p_1 = p_model.compress(frame, args.qp)
        encoded_p_2 = p_model.compress(frame, args.qp)

        decoded_i = i_model.decompress(encoded_i["bit_stream"], sps, args.qp)["x_hat"]
        p_model.clear_dpb()
        p_model.set_curr_poc(0)
        p_model.add_ref_frame(None, decoded_i)
        decoded_p_1 = p_model.decompress(encoded_p_1["bit_stream"], sps, args.qp)["x_hat"]
        decoded_p_2 = p_model.decompress(encoded_p_2["bit_stream"], sps, args.qp)["x_hat"]

        android_i_decoder_record = None
        if args.android_decoded_dir:
            android_y_hat_path = (
                args.android_decoded_dir.resolve()
                / "boundaries"
                / "android_i_y_hat_encode.nhwc.f32le"
            )
            if android_y_hat_path.is_file():
                android_i_y_hat = load_nhwc_tensor(
                    android_y_hat_path, (1, 16, 32, 256), device
                )
                android_codeword = i_model.dec(
                    android_i_y_hat,
                    i_model.q_scale_dec[args.qp : args.qp + 1],
                )
                server_from_android_y_hat = i_model.recon_generation_net(
                    android_codeword,
                    i_model.q_scale_recon[args.qp : args.qp + 1],
                )
                server_from_android_nhwc = to_nhwc_numpy(server_from_android_y_hat)
                server_from_android_path = (
                    output_dir / "server_decoder_from_android_i_y_hat.nhwc.f32le"
                )
                server_from_android_nhwc.astype("<f4").tofile(
                    str(server_from_android_path)
                )
                android_frame_path = (
                    args.android_decoded_dir.resolve()
                    / "decoded_frame_000.nhwc.f32le"
                )
                android_i_decoder_record = {
                    "input": str(android_y_hat_path),
                    "output": str(server_from_android_path),
                    "output_sha256": sha256(server_from_android_path),
                    "psnr_db": psnr(frame, server_from_android_y_hat),
                    "vs_server_canonical": compare_android(
                        to_nhwc_numpy(decoded_i), server_from_android_path
                    ),
                }
                if android_frame_path.is_file():
                    android_i_decoder_record["vs_android_decoder"] = compare_android(
                        server_from_android_nhwc, android_frame_path
                    )
        torch.cuda.synchronize()

    boundary_records = []
    for name, value, android_name in (
        ("server_i_y_pre_prior", i_y, "android_i_y_pre_prior.nhwc.f32le"),
        ("server_i_y_hat", i_y_hat, "android_i_y_hat_encode.nhwc.f32le"),
    ):
        output = output_dir / (name + ".nhwc.f32le")
        nhwc = write_nhwc(value, output)
        record = {
            "name": name,
            "output": str(output),
            "output_sha256": sha256(output),
        }
        if args.android_decoded_dir:
            android = args.android_decoded_dir.resolve() / "boundaries" / android_name
            if android.is_file():
                record["android_comparison"] = compare_android(nhwc, android)
        boundary_records.append(record)

    decoded = [decoded_i, decoded_p_1, decoded_p_2]
    payloads = [encoded_i["bit_stream"], encoded_p_1["bit_stream"], encoded_p_2["bit_stream"]]
    records = []
    for index, value in enumerate(decoded):
        nhwc = to_nhwc_numpy(value)
        output = output_dir / "server_decoded_frame_{:03d}.nhwc.f32le".format(index)
        nhwc.astype("<f4").tofile(str(output))
        record = {
            "frame": index,
            "type": "I" if index == 0 else "P",
            "psnr_db": psnr(frame, value),
            "payload_bytes": len(payloads[index]),
            "output": str(output),
            "output_sha256": sha256(output),
        }
        if args.android_decoded_dir:
            android = args.android_decoded_dir.resolve() / "decoded_frame_{:03d}.nhwc.f32le".format(index)
            if android.is_file():
                record["android_comparison"] = compare_android(nhwc, android)
        records.append(record)

    report = {
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "shape_nhwc": [1, args.height, args.width, 3],
        "qp": args.qp,
        "force_zero_thres": args.force_zero_thres,
        "precision": "fp16_compute_f32_dump",
        "i_checkpoint_sha256": sha256(i_checkpoint),
        "p_checkpoint_sha256": sha256(p_checkpoint),
        "i_boundaries": boundary_records,
        "server_decoder_from_android_i_y_hat": android_i_decoder_record,
        "frames": records,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("wrote {}".format(report_path))


if __name__ == "__main__":
    main()
