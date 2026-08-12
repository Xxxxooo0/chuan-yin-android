#!/usr/bin/env python3
"""Run the canonical Large PyTorch codec on the Android ParkScene sequence."""

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frame(path, height, width, device):
    with Image.open(str(path)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.shape != (height, width, 3):
        raise RuntimeError("{} shape={} expected={}".format(path, rgb.shape, (height, width, 3)))
    nchw = rgb.transpose(2, 0, 1)[None].astype(np.float32)
    nchw = nchw / 255.0 * 2.0 - 1.0
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


def frame_psnr(reference, reconstruction):
    reference = (reference.float().clamp(-1.0, 1.0) + 1.0) * 0.5
    reconstruction = (reconstruction.float().clamp(-1.0, 1.0) + 1.0) * 0.5
    mse = torch.mean((reference - reconstruction) ** 2).item()
    return float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def write_nhwc_tensor(value, path):
    array = value.detach().float().cpu().numpy().transpose(0, 2, 3, 1).copy()
    array.astype("<f4").tofile(str(path))


def dump_p_entropy_boundaries(model, frame, qp, output_dir, frame_number):
    q_encoder = model.q_scale_enc[qp : qp + 1]
    q_feature = model.q_scale_feature[qp : qp + 1]
    feature = model.apply_feature_adaptor()
    ctx, ctx_t = model.feature_extractor(feature, q_feature)
    y = model.enc(frame, ctx, q_encoder)
    z = model.hyper_enc(model.pad_for_y(y))
    z_hat = torch.clamp(torch.round(z), -128.0, 127.0)
    common = model.res_prior_param_decoder(z_hat, ctx_t)
    y_q_0, y_q_1, s_w_0, s_w_1, y_hat = model.compress_prior_2x(
        y, common, model.y_spatial_prior
    )
    prefix = "server_p_frame_{:03d}".format(frame_number)
    for name, value in (
        ("ctx", ctx),
        ("ctx_t", ctx_t),
        ("y_pre_prior", y),
        ("z_hat", z_hat),
        ("y_q_w_0", y_q_0),
        ("y_q_w_1", y_q_1),
        ("s_w_0", s_w_0),
        ("s_w_1", s_w_1),
        ("y_hat", y_hat),
    ):
        write_nhwc_tensor(value, output_dir / "{}_{}.nhwc.f32le".format(prefix, name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=240)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--qp", type=int, default=0)
    parser.add_argument("--reset-interval", type=int, default=32)
    parser.add_argument("--force-zero-thres", type=float, default=0.12)
    parser.add_argument("--dump-boundary-frames", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the original compress implementation")
    source_root = args.source_root.resolve()
    sequence_dir = args.sequence_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    boundary_frames = {
        int(value.strip())
        for value in args.dump_boundary_frames.split(",")
        if value.strip()
    }
    frame_paths = sorted(sequence_dir.glob("*.png"))[: args.frame_count]
    if len(frame_paths) != args.frame_count:
        raise RuntimeError("found {} PNG frames, expected {}".format(len(frame_paths), args.frame_count))

    device = torch.device("cuda:0")
    i_model, p_model, i_checkpoint, p_checkpoint = load_models(
        source_root, device, args.force_zero_thres
    )
    sps_base = {
        "height": args.height,
        "width": args.width,
        "ec_part": 0,
        "use_ada_i": 0,
    }

    payloads = []
    encode_ms = []
    p_model.clear_dpb()
    p_model.set_curr_poc(0)
    with torch.no_grad():
        for index, path in enumerate(frame_paths):
            frame = load_frame(path, args.height, args.width, device)
            torch.cuda.synchronize(device=device)
            started = time.time()
            if index == 0:
                encoded = i_model.compress(frame, args.qp)
                p_model.clear_dpb()
                p_model.add_ref_frame(None, encoded["x_hat"])
            else:
                if args.reset_interval > 0 and index % args.reset_interval == 1:
                    p_model.prepare_feature_adaptor_i(args.qp)
                if index + 1 in boundary_frames:
                    dump_p_entropy_boundaries(
                        p_model, frame, args.qp, output_dir, index + 1
                    )
                encoded = p_model.compress(frame, args.qp)
                if index + 1 in boundary_frames:
                    (output_dir / "server_p_frame_{:03d}_rans_payload.bin".format(index + 1)).write_bytes(
                        encoded["bit_stream"]
                    )
            torch.cuda.synchronize(device=device)
            elapsed_ms = (time.time() - started) * 1000.0
            payloads.append(encoded["bit_stream"])
            encode_ms.append(elapsed_ms)
            print(
                "[server-large] encode frame={}/{} type={} reset={} payload_bytes={} ms={:.3f}".format(
                    index + 1,
                    args.frame_count,
                    "I" if index == 0 else "P",
                    bool(
                        index > 0
                        and args.reset_interval > 0
                        and index % args.reset_interval == 1
                    ),
                    len(encoded["bit_stream"]),
                    elapsed_ms,
                ),
                flush=True,
            )

    records = []
    decode_ms = []
    p_model.clear_dpb()
    p_model.set_curr_poc(0)
    with torch.no_grad():
        for index, (path, payload) in enumerate(zip(frame_paths, payloads)):
            frame = load_frame(path, args.height, args.width, device)
            reset_reference = index > 0 and args.reset_interval > 0 and index % args.reset_interval == 1
            sps = dict(sps_base)
            sps["use_ada_i"] = 1 if reset_reference else 0
            torch.cuda.synchronize(device=device)
            started = time.time()
            if index == 0:
                decoded = i_model.decompress(payload, sps, args.qp)
                p_model.clear_dpb()
                p_model.add_ref_frame(None, decoded["x_hat"])
            else:
                if reset_reference:
                    p_model.reset_ref_feature()
                decoded = p_model.decompress(payload, sps, args.qp)
            torch.cuda.synchronize(device=device)
            elapsed_ms = (time.time() - started) * 1000.0
            value = frame_psnr(frame, decoded["x_hat"])
            decode_ms.append(elapsed_ms)
            record = {
                "frame": index + 1,
                "type": "I" if index == 0 else "P",
                "reset_reference": reset_reference,
                "payload_bytes": len(payload),
                "encode_ms": encode_ms[index],
                "decode_ms": elapsed_ms,
                "psnr_db": value,
            }
            records.append(record)
            print(
                "[server-large] decode frame={}/{} type={} reset={} psnr_db={:.3f} ms={:.3f}".format(
                    index + 1,
                    args.frame_count,
                    record["type"],
                    reset_reference,
                    value,
                    elapsed_ms,
                ),
                flush=True,
            )

    psnrs = [record["psnr_db"] for record in records]
    summary = {
        "source_root": str(source_root),
        "sequence_dir": str(sequence_dir),
        "frame_count": args.frame_count,
        "shape_nhwc": [1, args.height, args.width, 3],
        "qp": args.qp,
        "reset_interval": args.reset_interval,
        "force_zero_thres": args.force_zero_thres,
        "precision": "pytorch_fp16_cuda",
        "i_checkpoint_sha256": file_sha256(i_checkpoint),
        "p_checkpoint_sha256": file_sha256(p_checkpoint),
        "payload_bytes": sum(len(payload) for payload in payloads),
        "mean_encode_ms": float(np.mean(encode_ms)),
        "mean_decode_ms": float(np.mean(decode_ms)),
        "mean_psnr_db": float(np.mean(psnrs)),
        "min_psnr_db": float(np.min(psnrs)),
        "final_psnr_db": float(psnrs[-1]),
        "frames": records,
    }
    report_path = output_dir / "server_large_park_scene_240_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    csv_path = output_dir / "server_large_park_scene_240_frames.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print("[server-large] summary {}".format(json.dumps({
        "frames": args.frame_count,
        "payload_bytes": summary["payload_bytes"],
        "mean_psnr_db": summary["mean_psnr_db"],
        "min_psnr_db": summary["min_psnr_db"],
        "final_psnr_db": summary["final_psnr_db"],
        "mean_encode_ms": summary["mean_encode_ms"],
        "mean_decode_ms": summary["mean_decode_ms"],
    })), flush=True)
    print("wrote {} and {}".format(report_path, csv_path), flush=True)


if __name__ == "__main__":
    main()
