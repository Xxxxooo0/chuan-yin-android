#!/usr/bin/env python3
"""Export clean GVC-RT module ONNX graphs and server baselines from source.

This script is intentionally source-derived. It does not import or reuse any
previous Android exporter. Run it on the server/PyTorch environment, not as a
local PC inference test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=os.environ.get("GVC_RT_SOURCE_ROOT"))
    parser.add_argument(
        "--output-assets",
        default=str(PROJECT_ROOT / "outputs" / "onnx-intermediate"),
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--qp", type=int, default=0)
    parser.add_argument(
        "--precision",
        choices=["fp32"],
        default="fp32",
        help="Clean v1 uses one FP32 precision path for both baseline and ONNX graphs.",
    )
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--i-frame-f32le")
    parser.add_argument("--p-frame-f32le")
    parser.add_argument("--force-zero-thres", type=float, default=None)
    args = parser.parse_args()
    if not args.source_root:
        parser.error("--source-root is required, or set GVC_RT_SOURCE_ROOT")
    return args


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_tensor(path: Path, tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = tensor.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
    arr.astype("<f4", copy=False).tofile(path)


def save_array(path: Path, values, dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(values, dtype=dtype).tofile(path)


def load_f32le(path: str | None, shape: Tuple[int, ...], device, dtype):
    if path is None:
        total = 1
        for dim in shape:
            total *= dim
        data = torch.linspace(-1.0, 1.0, total, dtype=torch.float32).reshape(shape)
        return data.to(device=device, dtype=dtype)
    import numpy as np

    arr = np.fromfile(path, dtype="<f4")
    expected = 1
    for dim in shape:
        expected *= dim
    if arr.size != expected:
        raise ValueError(f"{path} has {arr.size} elements, expected {expected}")
    return torch.from_numpy(arr.reshape(shape)).to(device=device, dtype=dtype)


def force_exportable_torch_path() -> None:
    """Disable custom CUDA proxy branches while exporting ONNX.

    The deployed ONNX graph must be composed of standard PyTorch ops. Server
    CUDA is still used for tensor math when inputs are on CUDA; this only
    prevents non-exportable custom extension calls from entering the trace.
    """
    import src.layers.cuda_inference as cuda_inference
    import src.layers.layers as layers
    import src.models.image_model_G_b as image_model
    import src.models.video_model_G_b as video_model
    import torch.nn.functional as functional

    cuda_inference.CUSTOMIZED_CUDA_INFERENCE = False
    layers.CUSTOMIZED_CUDA_INFERENCE = False
    image_model.CUSTOMIZED_CUDA_INFERENCE = False
    video_model.CUSTOMIZED_CUDA_INFERENCE = False
    functional.pixel_unshuffle = pixel_unshuffle_static


def update_entropy_if_available(model, force_zero_thres, label: str) -> bool:
    try:
        model.update(force_zero_thres)
    except ModuleNotFoundError as exc:
        if exc.name != "MLCodec_extensions_cpp":
            raise
        print(f"[{label}] MLCodec_extensions_cpp not found; cannot export a complete entropy baseline.")
        return False
    return True


def asset_path(path: Path, output_assets: Path) -> str:
    return path.relative_to(output_assets).as_posix()


def export_entropy_assets(
    model,
    prefix: str,
    qp: int,
    z_hat,
    y_quantized: List,
    scales: List,
    baseline_dir: Path,
    output_assets: Path,
) -> Dict:
    entropy_dir = baseline_dir / "entropy"
    bitstream_dir = baseline_dir / "bitstream"

    def write_cdf(name: str, cdf_info) -> Dict:
        cdf, cdf_lengths, offsets = cdf_info
        cdf = np.asarray(cdf, dtype="<i4")
        cdf_lengths = np.asarray(cdf_lengths, dtype="<i4").reshape(-1)
        offsets = np.asarray(offsets, dtype="<i4").reshape(-1)
        cdf_path = entropy_dir / f"{prefix}_{name}_cdf.i32le"
        lengths_path = entropy_dir / f"{prefix}_{name}_cdf_lengths.i32le"
        offsets_path = entropy_dir / f"{prefix}_{name}_cdf_offsets.i32le"
        save_array(cdf_path, cdf, "<i4")
        save_array(lengths_path, cdf_lengths, "<i4")
        save_array(offsets_path, offsets, "<i4")
        return {
            "cdf": asset_path(cdf_path, output_assets),
            "shape": list(cdf.shape),
            "cdf_lengths": asset_path(lengths_path, output_assets),
            "offsets": asset_path(offsets_path, output_assets),
        }

    gaussian = write_cdf("gaussian", model.gaussian_encoder.get_cdf_info())
    z = write_cdf("z", model.bit_estimator_z.get_cdf_info())

    z_symbols = z_hat.detach().to("cpu", dtype=torch.int8).contiguous().numpy().reshape(-1)
    z_symbols_path = entropy_dir / f"{prefix}_z_symbols.i8"
    save_array(z_symbols_path, z_symbols, "i1")

    packed_paths = []
    for stage, (y_q, s_w) in enumerate(zip(y_quantized, scales)):
        packed = model.gaussian_encoder.build_indexes_encoder(
            y_q.detach().clone(),
            s_w.detach().clone(),
        )
        packed_path = entropy_dir / f"{prefix}_y_packed_{stage}.i16le"
        save_array(packed_path, packed.detach().to("cpu", dtype=torch.int16).contiguous().numpy(), "<i2")
        packed_paths.append(asset_path(packed_path, output_assets))

    model.entropy_coder.set_use_two_entropy_coders(False)
    model.entropy_coder.reset()
    model.bit_estimator_z.encode_z(z_hat.to(dtype=torch.int8), qp)
    for y_q, s_w in zip(y_quantized, scales):
        model.gaussian_encoder.encode_y(y_q.detach().clone(), s_w.detach().clone())
    model.entropy_coder.flush()
    payload = model.entropy_coder.get_encoded_stream()
    payload_path = bitstream_dir / f"{prefix}_rans_payload.bin"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes(payload)

    _, z_channels, z_height, z_width = z_hat.shape
    return {
        "gaussian": gaussian,
        "z": z,
        "z_symbols": asset_path(z_symbols_path, output_assets),
        "z_start_offset": qp * int(z_channels),
        "z_per_channel_size": int(z_height * z_width),
        "y_packed": packed_paths,
        "payload": asset_path(payload_path, output_assets),
        "two_entropy_coders": False,
    }


def export_muxed_stream(
    height: int,
    width: int,
    qp: int,
    i_payload: bytes,
    p_payload: bytes,
    baseline_dir: Path,
    output_assets: Path,
) -> Dict:
    from src.utils.stream_helper import SPSHelper, write_ip, write_sps

    output = BytesIO()
    sps_helper = SPSHelper()
    sps = {
        "sps_id": -1,
        "height": height,
        "width": width,
        "ec_part": 0,
        "use_ada_i": 0,
    }
    sps_id, is_new = sps_helper.get_sps_id(sps)
    assert is_new
    sps["sps_id"] = sps_id
    write_sps(output, sps)
    write_ip(output, True, sps_id, qp, i_payload)
    write_ip(output, False, sps_id, qp, p_payload)
    path = baseline_dir / "bitstream" / "encoded_ip.gvc"
    path.write_bytes(output.getvalue())
    return {
        "path": asset_path(path, output_assets),
        "height": height,
        "width": width,
        "qp": qp,
        "ec_part": sps["ec_part"],
        "use_ada_i": sps["use_ada_i"],
    }


def export_onnx(
    module,
    path: Path,
    inputs: Tuple,
    input_names: List[str],
    output_names: List[str],
    opset: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            inputs,
            str(path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            do_constant_folding=True,
        )


def pixel_unshuffle_static(x, downscale_factor: int):
    """ONNX-exportable PixelUnshuffle for fixed-size deployment graphs."""
    b, c, h, w = x.shape
    r = downscale_factor
    return (
        x.reshape(b, c, h // r, r, w // r, r)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(b, c * r * r, h // r, w // r)
    )


class IEncoderFront(torch.nn.Module):
    def __init__(self, model, qp: int):
        super().__init__()
        self.model = model
        self.register_buffer("q_enc", model.q_scale_enc[qp : qp + 1].detach().clone())

    def forward(self, input_i_frame):
        feature = pixel_unshuffle_static(input_i_frame, 8)
        return self.model.enc.forward_torch(feature, self.q_enc)


class IHyperEnc(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, i_y_pre_prior):
        z = self.model.hyper_enc(self.model.pad_for_y(i_y_pre_prior))
        z_hat = torch.clamp(torch.round(z), -128.0, 127.0)
        return z, z_hat


class IHyperPrior(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, i_z_hat, i_y_pre_prior):
        params = self.model.hyper_dec(i_z_hat)
        params = self.model.y_prior_fusion(params)
        _, _, h, w = i_y_pre_prior.shape
        return params[:, :, :h, :w].contiguous()


class IDecodeHyperPrior(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, i_z_hat):
        return self.model.y_prior_fusion(self.model.hyper_dec(i_z_hat))


class IPrior4x(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, i_y_pre_prior, i_common_params):
        return self.model.compress_prior_4x(
            i_y_pre_prior,
            i_common_params,
            self.model.y_spatial_prior_reduction,
            self.model.y_spatial_prior_adaptor_1,
            self.model.y_spatial_prior_adaptor_2,
            self.model.y_spatial_prior_adaptor_3,
            self.model.y_spatial_prior,
        )


class IDecodePriorStage0(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, i_common_params, i_y_q_w_0):
        _, _, scales, means = self.model.separate_prior(i_common_params, False)
        batch, channels, height, width = means.shape
        mask_0, _, _, _ = self.model.get_mask_4x(batch, channels, height, width, means.dtype, means.device)
        scales_for_rans = self.model.single_part_for_writing_4x(scales * mask_0)
        y_hat_so_far = (torch.cat((i_y_q_w_0, i_y_q_w_0, i_y_q_w_0, i_y_q_w_0), dim=1) + means) * mask_0
        return scales_for_rans, y_hat_so_far


class IDecodePriorStage(torch.nn.Module):
    def __init__(self, model, stage: int):
        super().__init__()
        if stage not in (1, 2, 3):
            raise ValueError(f"invalid I decode stage {stage}")
        self.model = model
        self.stage = stage

    def forward(self, i_common_params, i_y_hat_so_far, i_y_q_w):
        _, q_dec, _, means = self.model.separate_prior(i_common_params, False)
        batch, channels, height, width = means.shape
        _, mask_1, mask_2, mask_3 = self.model.get_mask_4x(batch, channels, height, width, means.dtype, means.device)
        masks = (mask_1, mask_2, mask_3)
        adaptors = (
            self.model.y_spatial_prior_adaptor_1,
            self.model.y_spatial_prior_adaptor_2,
            self.model.y_spatial_prior_adaptor_3,
        )
        spatial_params = torch.cat((i_y_hat_so_far, self.model.y_spatial_prior_reduction(i_common_params)), dim=1)
        scales, means = self.model.y_spatial_prior(adaptors[self.stage - 1](spatial_params)).chunk(2, 1)
        mask = masks[self.stage - 1]
        scales_for_rans = self.model.single_part_for_writing_4x(scales * mask)
        current = (torch.cat((i_y_q_w, i_y_q_w, i_y_q_w, i_y_q_w), dim=1) + means) * mask
        y_hat = i_y_hat_so_far + current
        if self.stage == 3:
            y_hat = y_hat * q_dec
        return scales_for_rans, y_hat


class IRecon(torch.nn.Module):
    def __init__(self, model, qp: int):
        super().__init__()
        self.model = model
        self.register_buffer("q_dec", model.q_scale_dec[qp : qp + 1].detach().clone())
        self.register_buffer("q_recon", model.q_scale_recon[qp : qp + 1].detach().clone())

    def forward(self, i_y_hat):
        codeword = self.model.dec(i_y_hat, self.q_dec)
        frame = self.model.recon_generation_net(codeword, self.q_recon)
        return codeword, frame


class TemporalFromFrame(torch.nn.Module):
    def __init__(self, model, qp: int):
        super().__init__()
        self.model = model
        self.register_buffer("q_feature", model.q_scale_feature[qp : qp + 1].detach().clone())

    def forward(self, reference_frame):
        feature = self.model.feature_adaptor_i(pixel_unshuffle_static(reference_frame, 8))
        ctx, ctx_t = self.model.feature_extractor(feature, self.q_feature)
        return feature, ctx, ctx_t


class TemporalFromFeature(torch.nn.Module):
    def __init__(self, model, qp: int):
        super().__init__()
        self.model = model
        self.register_buffer("q_feature", model.q_scale_feature[qp : qp + 1].detach().clone())

    def forward(self, reference_feature):
        feature = self.model.feature_adaptor_p(reference_feature)
        ctx, ctx_t = self.model.feature_extractor(feature, self.q_feature)
        return feature, ctx, ctx_t


class PEncoderFront(torch.nn.Module):
    def __init__(self, model, qp: int):
        super().__init__()
        self.model = model
        self.register_buffer("q_enc", model.q_scale_enc[qp : qp + 1].detach().clone())

    def forward(self, input_p_frame, p_ctx):
        feature = pixel_unshuffle_static(input_p_frame, 8)
        return self.model.enc.forward_torch(feature, p_ctx, self.q_enc)


class PHyperEnc(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, p_y_pre_prior):
        z = self.model.hyper_enc(self.model.pad_for_y(p_y_pre_prior))
        z_hat = torch.clamp(torch.round(z), -128.0, 127.0)
        return z, z_hat


class PHyperPrior(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, p_z_hat, p_ctx_t):
        return self.model.res_prior_param_decoder(p_z_hat, p_ctx_t)


class PPrior2x(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, p_y_pre_prior, p_common_params):
        return self.model.compress_prior_2x(p_y_pre_prior, p_common_params, self.model.y_spatial_prior)


class PDecodePriorStage0(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, p_common_params, p_y_q_w_0):
        _, scales, means = self.model.separate_prior_for_video_decoding(p_common_params)
        batch, channels, height, width = means.shape
        mask_0, _ = self.model.get_mask_2x(batch, channels, height, width, means.dtype, means.device)
        scales_for_rans = self.model.single_part_for_writing_2x(scales * mask_0)
        y_hat_so_far = (torch.cat((p_y_q_w_0, p_y_q_w_0), dim=1) + means) * mask_0
        return scales_for_rans, y_hat_so_far


class PDecodePriorStage1(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, p_common_params, p_y_hat_so_far, p_y_q_w_1):
        q_dec, _, means = self.model.separate_prior_for_video_decoding(p_common_params)
        batch, channels, height, width = means.shape
        _, mask_1 = self.model.get_mask_2x(batch, channels, height, width, means.dtype, means.device)
        scales, means = self.model.y_spatial_prior(torch.cat((p_y_hat_so_far, p_common_params), dim=1)).chunk(2, 1)
        scales_for_rans = self.model.single_part_for_writing_2x(scales * mask_1)
        current = (torch.cat((p_y_q_w_1, p_y_q_w_1), dim=1) + means) * mask_1
        return scales_for_rans, (p_y_hat_so_far + current) * q_dec


class PRecon(torch.nn.Module):
    def __init__(self, model, qp: int):
        super().__init__()
        self.model = model
        self.register_buffer("q_dec", model.q_scale_dec[qp : qp + 1].detach().clone())
        self.register_buffer("q_recon", model.q_scale_recon[qp : qp + 1].detach().clone())

    def forward(self, p_y_hat, p_ctx):
        feature = self.model.dec(p_y_hat, p_ctx, self.q_dec)
        frame = self.model.recon_generation_net(feature, self.q_recon)
        return feature, frame


class IFusedImageEncoder(torch.nn.Module):
    def __init__(self, model, qp: int):
        super().__init__()
        self.encoder = IEncoderFront(model, qp)
        self.hyper_enc = IHyperEnc(model)
        self.hyper_prior = IHyperPrior(model)
        self.prior = IPrior4x(model)
        self.recon = IRecon(model, qp)

    def forward(self, input_i_frame):
        y = self.encoder(input_i_frame)
        _, z_hat = self.hyper_enc(y)
        params = self.hyper_prior(z_hat, y)
        y_q0, y_q1, y_q2, y_q3, s0, s1, s2, s3, y_hat = self.prior(y, params)
        _, reference_frame = self.recon(y_hat)
        return z_hat, y_q0, y_q1, y_q2, y_q3, s0, s1, s2, s3, reference_frame


class PFusedImageEncoder(torch.nn.Module):
    def __init__(self, model, qp: int):
        super().__init__()
        self.temporal = TemporalFromFrame(model, qp)
        self.encoder = PEncoderFront(model, qp)
        self.hyper_enc = PHyperEnc(model)
        self.hyper_prior = PHyperPrior(model)
        self.prior = PPrior2x(model)
        self.recon = PRecon(model, qp)

    def forward(self, input_p_frame, reference_frame):
        _, ctx, ctx_t = self.temporal(reference_frame)
        y = self.encoder(input_p_frame, ctx)
        _, z_hat = self.hyper_enc(y)
        params = self.hyper_prior(z_hat, ctx_t)
        y_q0, y_q1, s0, s1, y_hat = self.prior(y, params)
        _, reconstructed_frame = self.recon(y_hat, ctx)
        return z_hat, y_q0, y_q1, s0, s1, reconstructed_frame


def tensor_spec(path: str, shape: Iterable[int]) -> Dict:
    return {"path": path, "shape": list(shape), "dtype": "float32"}


def source_spec(source: str) -> Dict:
    return {"source": source}


def output_spec(baseline: str, shape: Iterable[int]) -> Dict:
    return {"baseline": baseline, "shape": list(shape), "dtype": "float32"}


def graph_step(name: str, model: str, inputs: Dict, outputs: Dict) -> Dict:
    return {"name": name, "model": model, "inputs": inputs, "outputs": outputs}


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    out = Path(args.output_assets).resolve()
    models_dir = out / "models"
    baseline_dir = out / "baseline"
    if out.exists():
        shutil.rmtree(out)
    models_dir.mkdir(parents=True)
    baseline_dir.mkdir(parents=True)

    sys.path.insert(0, str(source_root))
    force_exportable_torch_path()
    from src.models.image_model_G_b import DMCI
    from src.models.video_model_G_b import DMC

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    i_ckpt = source_root / "ckpt" / "checkpoints" / "GVC-RT_B_I.pt"
    p_ckpt = source_root / "ckpt" / "checkpoints" / "GVC-RT_B_P.pt"

    i_model = DMCI(encoder_ckpt_path=str(i_ckpt)).to(device).eval()
    p_model = DMC().to(device).eval()
    p_checkpoint = torch.load(p_ckpt, map_location="cpu")
    p_sd = p_checkpoint.get("student_ema", p_checkpoint.get("student", p_checkpoint.get("state_dict", p_checkpoint)))
    p_model.load_state_dict(p_sd, strict=True)
    if not update_entropy_if_available(i_model, args.force_zero_thres, "DMCI"):
        raise RuntimeError("DMCI entropy extension is required for a complete encoder baseline")
    if not update_entropy_if_available(p_model, args.force_zero_thres, "DMC"):
        raise RuntimeError("DMC entropy extension is required for a complete encoder baseline")
    input_shape = (1, 3, args.height, args.width)
    input_i = load_f32le(args.i_frame_f32le, input_shape, device, dtype)
    input_p = load_f32le(args.p_frame_f32le, input_shape, device, dtype)
    save_tensor(baseline_dir / "inputs" / "input_i_frame.f32le", input_i)
    save_tensor(baseline_dir / "inputs" / "input_p_frame.f32le", input_p)

    i_encoder = IEncoderFront(i_model, args.qp).to(device).eval()
    i_hyper_enc = IHyperEnc(i_model).to(device).eval()
    i_hyper_prior = IHyperPrior(i_model).to(device).eval()
    i_decode_hyper_prior = IDecodeHyperPrior(i_model).to(device).eval()
    i_prior = IPrior4x(i_model).to(device).eval()
    i_decode_stage_0 = IDecodePriorStage0(i_model).to(device).eval()
    i_decode_stage_1 = IDecodePriorStage(i_model, 1).to(device).eval()
    i_decode_stage_2 = IDecodePriorStage(i_model, 2).to(device).eval()
    i_decode_stage_3 = IDecodePriorStage(i_model, 3).to(device).eval()
    i_recon = IRecon(i_model, args.qp).to(device).eval()
    temporal_frame = TemporalFromFrame(p_model, args.qp).to(device).eval()
    temporal_feature = TemporalFromFeature(p_model, args.qp).to(device).eval()
    p_encoder = PEncoderFront(p_model, args.qp).to(device).eval()
    p_hyper_enc = PHyperEnc(p_model).to(device).eval()
    p_hyper_prior = PHyperPrior(p_model).to(device).eval()
    p_prior = PPrior2x(p_model).to(device).eval()
    p_decode_stage_0 = PDecodePriorStage0(p_model).to(device).eval()
    p_decode_stage_1 = PDecodePriorStage1(p_model).to(device).eval()
    p_recon = PRecon(p_model, args.qp).to(device).eval()
    i_fused_image_encoder = IFusedImageEncoder(i_model, args.qp).to(device).eval()
    p_fused_image_encoder = PFusedImageEncoder(p_model, args.qp).to(device).eval()

    with torch.no_grad():
        i_y = i_encoder(input_i)
        i_z, i_z_hat = i_hyper_enc(i_y)
        i_params = i_hyper_prior(i_z_hat, i_y)
        i_y_q0, i_y_q1, i_y_q2, i_y_q3, i_s0, i_s1, i_s2, i_s3, i_y_hat = i_prior(i_y, i_params)
        i_decode_params = i_decode_hyper_prior(i_z_hat)
        i_decode_s0, i_decode_y0 = i_decode_stage_0(i_decode_params, i_y_q0)
        i_decode_s1, i_decode_y1 = i_decode_stage_1(i_decode_params, i_decode_y0, i_y_q1)
        i_decode_s2, i_decode_y2 = i_decode_stage_2(i_decode_params, i_decode_y1, i_y_q2)
        i_decode_s3, i_decode_y_hat = i_decode_stage_3(i_decode_params, i_decode_y2, i_y_q3)
        i_codeword, i_ref = i_recon(i_y_hat)

        p_feature_from_frame, p_ctx, p_ctx_t = temporal_frame(i_ref)
        p_y = p_encoder(input_p, p_ctx)
        p_z, p_z_hat = p_hyper_enc(p_y)
        p_params = p_hyper_prior(p_z_hat, p_ctx_t)
        p_y_q0, p_y_q1, p_s0, p_s1, p_y_hat = p_prior(p_y, p_params)
        p_decode_s0, p_decode_y0 = p_decode_stage_0(p_params, p_y_q0)
        p_decode_s1, p_decode_y_hat = p_decode_stage_1(p_params, p_decode_y0, p_y_q1)
        p_feature, p_ref = p_recon(p_y_hat, p_ctx)
        p_feature_from_feature, p_ctx_from_feature, p_ctx_t_from_feature = temporal_feature(p_feature)

    tensors = {
        "i_y_pre_prior": i_y,
        "i_z_pre_quant": i_z,
        "i_z_hat": i_z_hat,
        "i_common_params": i_params,
        "i_y_q_w_0": i_y_q0,
        "i_y_q_w_1": i_y_q1,
        "i_y_q_w_2": i_y_q2,
        "i_y_q_w_3": i_y_q3,
        "i_s_w_0": i_s0,
        "i_s_w_1": i_s1,
        "i_s_w_2": i_s2,
        "i_s_w_3": i_s3,
        "i_y_hat": i_y_hat,
        "i_decode_common_params": i_decode_params,
        "i_decode_s_w_0": i_decode_s0,
        "i_decode_s_w_1": i_decode_s1,
        "i_decode_s_w_2": i_decode_s2,
        "i_decode_s_w_3": i_decode_s3,
        "i_y_hat_so_far_0": i_decode_y0,
        "i_y_hat_so_far_1": i_decode_y1,
        "i_y_hat_so_far_2": i_decode_y2,
        "i_decode_y_hat": i_decode_y_hat,
        "i_codeword": i_codeword,
        "encoder_i_reference_frame": i_ref,
        "temporal_from_frame_feature": p_feature_from_frame,
        "p_ctx": p_ctx,
        "p_ctx_t": p_ctx_t,
        "p_y_pre_prior": p_y,
        "p_z_pre_quant": p_z,
        "p_z_hat": p_z_hat,
        "p_common_params": p_params,
        "p_y_q_w_0": p_y_q0,
        "p_y_q_w_1": p_y_q1,
        "p_s_w_0": p_s0,
        "p_s_w_1": p_s1,
        "p_y_hat": p_y_hat,
        "p_decode_s_w_0": p_decode_s0,
        "p_decode_s_w_1": p_decode_s1,
        "p_y_hat_so_far_0": p_decode_y0,
        "p_decode_y_hat": p_decode_y_hat,
        "encoder_p_reference_feature": p_feature,
        "encoder_p_reference_frame": p_ref,
        "temporal_from_feature_feature": p_feature_from_feature,
        "p_ctx_from_feature": p_ctx_from_feature,
        "p_ctx_t_from_feature": p_ctx_t_from_feature,
    }
    for name, tensor in tensors.items():
        save_tensor(baseline_dir / "tensors" / f"{name}.f32le", tensor)

    entropy_assets = {
        "i": export_entropy_assets(
            i_model,
            "i",
            args.qp,
            i_z_hat,
            [i_y_q0, i_y_q1, i_y_q2, i_y_q3],
            [i_s0, i_s1, i_s2, i_s3],
            baseline_dir,
            out,
        ),
        "p": export_entropy_assets(
            p_model,
            "p",
            args.qp,
            p_z_hat,
            [p_y_q0, p_y_q1],
            [p_s0, p_s1],
            baseline_dir,
            out,
        ),
    }
    stream = export_muxed_stream(
        args.height,
        args.width,
        args.qp,
        (out / entropy_assets["i"]["payload"]).read_bytes(),
        (out / entropy_assets["p"]["payload"]).read_bytes(),
        baseline_dir,
        out,
    )

    export_onnx(i_encoder, models_dir / "i_encoder_front.onnx", (input_i,), ["input_i_frame"], ["i_y_pre_prior"], args.opset)
    export_onnx(i_hyper_enc, models_dir / "i_hyper_enc.onnx", (i_y,), ["i_y_pre_prior"], ["i_z_pre_quant", "i_z_hat"], args.opset)
    export_onnx(i_hyper_prior, models_dir / "i_hyper_prior.onnx", (i_z_hat, i_y), ["i_z_hat", "i_y_pre_prior"], ["i_common_params"], args.opset)
    export_onnx(i_decode_hyper_prior, models_dir / "i_decode_hyper_prior.onnx", (i_z_hat,), ["i_z_hat"], ["i_common_params"], args.opset)
    export_onnx(
        i_prior,
        models_dir / "i_prior_4x.onnx",
        (i_y, i_params),
        ["i_y_pre_prior", "i_common_params"],
        ["i_y_q_w_0", "i_y_q_w_1", "i_y_q_w_2", "i_y_q_w_3", "i_s_w_0", "i_s_w_1", "i_s_w_2", "i_s_w_3", "i_y_hat"],
        args.opset,
    )
    export_onnx(i_decode_stage_0, models_dir / "i_decode_prior_stage_0.onnx", (i_decode_params, i_y_q0), ["i_common_params", "i_y_q_w_0"], ["i_s_w_0", "i_y_hat_so_far_0"], args.opset)
    export_onnx(i_decode_stage_1, models_dir / "i_decode_prior_stage_1.onnx", (i_decode_params, i_decode_y0, i_y_q1), ["i_common_params", "i_y_hat_so_far_0", "i_y_q_w_1"], ["i_s_w_1", "i_y_hat_so_far_1"], args.opset)
    export_onnx(i_decode_stage_2, models_dir / "i_decode_prior_stage_2.onnx", (i_decode_params, i_decode_y1, i_y_q2), ["i_common_params", "i_y_hat_so_far_1", "i_y_q_w_2"], ["i_s_w_2", "i_y_hat_so_far_2"], args.opset)
    export_onnx(i_decode_stage_3, models_dir / "i_decode_prior_stage_3.onnx", (i_decode_params, i_decode_y2, i_y_q3), ["i_common_params", "i_y_hat_so_far_2", "i_y_q_w_3"], ["i_s_w_3", "i_y_hat"], args.opset)
    export_onnx(i_recon, models_dir / "i_recon.onnx", (i_y_hat,), ["i_y_hat"], ["i_codeword", "encoder_i_reference_frame"], args.opset)
    export_onnx(temporal_frame, models_dir / "temporal_from_frame.onnx", (i_ref,), ["reference_frame"], ["temporal_from_frame_feature", "p_ctx", "p_ctx_t"], args.opset)
    export_onnx(temporal_feature, models_dir / "temporal_from_feature.onnx", (p_feature,), ["reference_feature"], ["temporal_from_feature_feature", "p_ctx_from_feature", "p_ctx_t_from_feature"], args.opset)
    export_onnx(p_encoder, models_dir / "p_encoder_front.onnx", (input_p, p_ctx), ["input_p_frame", "p_ctx"], ["p_y_pre_prior"], args.opset)
    export_onnx(p_hyper_enc, models_dir / "p_hyper_enc.onnx", (p_y,), ["p_y_pre_prior"], ["p_z_pre_quant", "p_z_hat"], args.opset)
    export_onnx(p_hyper_prior, models_dir / "p_hyper_prior.onnx", (p_z_hat, p_ctx_t), ["p_z_hat", "p_ctx_t"], ["p_common_params"], args.opset)
    export_onnx(
        p_prior,
        models_dir / "p_prior_2x.onnx",
        (p_y, p_params),
        ["p_y_pre_prior", "p_common_params"],
        ["p_y_q_w_0", "p_y_q_w_1", "p_s_w_0", "p_s_w_1", "p_y_hat"],
        args.opset,
    )
    export_onnx(p_decode_stage_0, models_dir / "p_decode_prior_stage_0.onnx", (p_params, p_y_q0), ["p_common_params", "p_y_q_w_0"], ["p_s_w_0", "p_y_hat_so_far_0"], args.opset)
    export_onnx(p_decode_stage_1, models_dir / "p_decode_prior_stage_1.onnx", (p_params, p_decode_y0, p_y_q1), ["p_common_params", "p_y_hat_so_far_0", "p_y_q_w_1"], ["p_s_w_1", "p_y_hat"], args.opset)
    export_onnx(p_recon, models_dir / "p_recon.onnx", (p_y_hat, p_ctx), ["p_y_hat", "p_ctx"], ["encoder_p_reference_feature", "encoder_p_reference_frame"], args.opset)
    export_onnx(
        i_fused_image_encoder,
        models_dir / "i_image_encoder_fused.onnx",
        (input_i,),
        ["input_i_frame"],
        ["i_z_hat", "i_y_q_w_0", "i_y_q_w_1", "i_y_q_w_2", "i_y_q_w_3", "i_s_w_0", "i_s_w_1", "i_s_w_2", "i_s_w_3", "encoder_i_reference_frame"],
        args.opset,
    )
    export_onnx(
        p_fused_image_encoder,
        models_dir / "p_image_encoder_fused.onnx",
        (input_p, i_ref),
        ["input_p_frame", "encoder_i_reference_frame"],
        ["p_z_hat", "p_y_q_w_0", "p_y_q_w_1", "p_s_w_0", "p_s_w_1", "encoder_p_reference_frame"],
        args.opset,
    )

    manifest = build_manifest(source_root, args, i_ckpt, p_ckpt, tensors, entropy_assets, stream)
    manifest["metadata"]["model_sha256"] = {
        p.name: sha256_file(p) for p in sorted(models_dir.glob("*.onnx"))
    }
    (out / "gvcrt_clean_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote assets to {out}")


def shape_of(tensors: Dict, name: str) -> List[int]:
    return [int(v) for v in tensors[name].shape]


def baseline(name: str) -> str:
    return f"baseline/tensors/{name}.f32le"


def build_manifest(
    source_root: Path,
    args: argparse.Namespace,
    i_ckpt: Path,
    p_ckpt: Path,
    tensors: Dict,
    entropy_assets: Dict,
    stream: Dict,
) -> Dict:
    def out(name: str) -> Dict:
        return output_spec(baseline(name), shape_of(tensors, name))

    i_steps = [
        graph_step(
            "i_encoder_front",
            "models/i_encoder_front.onnx",
            {"input_i_frame": tensor_spec("baseline/inputs/input_i_frame.f32le", [1, 3, args.height, args.width])},
            {"i_y_pre_prior": out("i_y_pre_prior")},
        ),
        graph_step("i_hyper_enc", "models/i_hyper_enc.onnx", {"i_y_pre_prior": source_spec("i_y_pre_prior")}, {"i_z_pre_quant": out("i_z_pre_quant"), "i_z_hat": out("i_z_hat")}),
        graph_step("i_hyper_prior", "models/i_hyper_prior.onnx", {"i_z_hat": source_spec("i_z_hat"), "i_y_pre_prior": source_spec("i_y_pre_prior")}, {"i_common_params": out("i_common_params")}),
        graph_step(
            "i_prior_4x",
            "models/i_prior_4x.onnx",
            {"i_y_pre_prior": source_spec("i_y_pre_prior"), "i_common_params": source_spec("i_common_params")},
            {
                "i_y_q_w_0": out("i_y_q_w_0"),
                "i_y_q_w_1": out("i_y_q_w_1"),
                "i_y_q_w_2": out("i_y_q_w_2"),
                "i_y_q_w_3": out("i_y_q_w_3"),
                "i_s_w_0": out("i_s_w_0"),
                "i_s_w_1": out("i_s_w_1"),
                "i_s_w_2": out("i_s_w_2"),
                "i_s_w_3": out("i_s_w_3"),
                "i_y_hat": out("i_y_hat"),
            },
        ),
        graph_step("i_recon", "models/i_recon.onnx", {"i_y_hat": source_spec("i_y_hat")}, {"i_codeword": out("i_codeword"), "encoder_i_reference_frame": out("encoder_i_reference_frame")}),
    ]
    p_steps = i_steps + [
        graph_step(
            "p_temporal_from_i_reference_frame",
            "models/temporal_from_frame.onnx",
            {"reference_frame": source_spec("encoder_i_reference_frame")},
            {"temporal_from_frame_feature": out("temporal_from_frame_feature"), "p_ctx": out("p_ctx"), "p_ctx_t": out("p_ctx_t")},
        ),
        graph_step(
            "p_encoder_front",
            "models/p_encoder_front.onnx",
            {"input_p_frame": tensor_spec("baseline/inputs/input_p_frame.f32le", [1, 3, args.height, args.width]), "p_ctx": source_spec("p_ctx")},
            {"p_y_pre_prior": out("p_y_pre_prior")},
        ),
        graph_step("p_hyper_enc", "models/p_hyper_enc.onnx", {"p_y_pre_prior": source_spec("p_y_pre_prior")}, {"p_z_pre_quant": out("p_z_pre_quant"), "p_z_hat": out("p_z_hat")}),
        graph_step("p_hyper_prior", "models/p_hyper_prior.onnx", {"p_z_hat": source_spec("p_z_hat"), "p_ctx_t": source_spec("p_ctx_t")}, {"p_common_params": out("p_common_params")}),
        graph_step(
            "p_prior_2x",
            "models/p_prior_2x.onnx",
            {"p_y_pre_prior": source_spec("p_y_pre_prior"), "p_common_params": source_spec("p_common_params")},
            {"p_y_q_w_0": out("p_y_q_w_0"), "p_y_q_w_1": out("p_y_q_w_1"), "p_s_w_0": out("p_s_w_0"), "p_s_w_1": out("p_s_w_1"), "p_y_hat": out("p_y_hat")},
        ),
        graph_step("p_recon", "models/p_recon.onnx", {"p_y_hat": source_spec("p_y_hat"), "p_ctx": source_spec("p_ctx")}, {"encoder_p_reference_feature": out("encoder_p_reference_feature"), "encoder_p_reference_frame": out("encoder_p_reference_frame")}),
    ]
    decoder = {
        "android_input": "outputs/encoded_ip.gvc",
        "fallback_input": stream["path"],
        "i": {
            "z_shape": shape_of(tensors, "i_z_hat"),
            "y_stage_shape": shape_of(tensors, "i_y_q_w_0"),
            "hyper_prior": graph_step(
                "decode_i_hyper_prior",
                "models/i_decode_hyper_prior.onnx",
                {"i_z_hat": source_spec("i_z_hat")},
                {"i_common_params": out("i_common_params")},
            ),
            "stages": [
                graph_step(
                    "decode_i_prior_stage_0",
                    "models/i_decode_prior_stage_0.onnx",
                    {"i_common_params": source_spec("i_common_params"), "i_y_q_w_0": source_spec("i_y_q_w_0")},
                    {"i_s_w_0": out("i_s_w_0"), "i_y_hat_so_far_0": out("i_y_hat_so_far_0")},
                ),
                graph_step(
                    "decode_i_prior_stage_1",
                    "models/i_decode_prior_stage_1.onnx",
                    {"i_common_params": source_spec("i_common_params"), "i_y_hat_so_far_0": source_spec("i_y_hat_so_far_0"), "i_y_q_w_1": source_spec("i_y_q_w_1")},
                    {"i_s_w_1": out("i_s_w_1"), "i_y_hat_so_far_1": out("i_y_hat_so_far_1")},
                ),
                graph_step(
                    "decode_i_prior_stage_2",
                    "models/i_decode_prior_stage_2.onnx",
                    {"i_common_params": source_spec("i_common_params"), "i_y_hat_so_far_1": source_spec("i_y_hat_so_far_1"), "i_y_q_w_2": source_spec("i_y_q_w_2")},
                    {"i_s_w_2": out("i_s_w_2"), "i_y_hat_so_far_2": out("i_y_hat_so_far_2")},
                ),
                graph_step(
                    "decode_i_prior_stage_3",
                    "models/i_decode_prior_stage_3.onnx",
                    {"i_common_params": source_spec("i_common_params"), "i_y_hat_so_far_2": source_spec("i_y_hat_so_far_2"), "i_y_q_w_3": source_spec("i_y_q_w_3")},
                    {"i_s_w_3": out("i_s_w_3"), "i_y_hat": out("i_y_hat")},
                ),
            ],
            "recon": graph_step(
                "decode_i_recon",
                "models/i_recon.onnx",
                {"i_y_hat": source_spec("i_y_hat")},
                {"i_codeword": out("i_codeword"), "encoder_i_reference_frame": out("encoder_i_reference_frame")},
            ),
        },
        "p": {
            "z_shape": shape_of(tensors, "p_z_hat"),
            "y_stage_shape": shape_of(tensors, "p_y_q_w_0"),
            "temporal": graph_step(
                "decode_p_temporal_from_i_reference_frame",
                "models/temporal_from_frame.onnx",
                {"reference_frame": source_spec("encoder_i_reference_frame")},
                {"temporal_from_frame_feature": out("temporal_from_frame_feature"), "p_ctx": out("p_ctx"), "p_ctx_t": out("p_ctx_t")},
            ),
            "hyper_prior": graph_step(
                "decode_p_hyper_prior",
                "models/p_hyper_prior.onnx",
                {"p_z_hat": source_spec("p_z_hat"), "p_ctx_t": source_spec("p_ctx_t")},
                {"p_common_params": out("p_common_params")},
            ),
            "stages": [
                graph_step(
                    "decode_p_prior_stage_0",
                    "models/p_decode_prior_stage_0.onnx",
                    {"p_common_params": source_spec("p_common_params"), "p_y_q_w_0": source_spec("p_y_q_w_0")},
                    {"p_s_w_0": out("p_s_w_0"), "p_y_hat_so_far_0": out("p_y_hat_so_far_0")},
                ),
                graph_step(
                    "decode_p_prior_stage_1",
                    "models/p_decode_prior_stage_1.onnx",
                    {"p_common_params": source_spec("p_common_params"), "p_y_hat_so_far_0": source_spec("p_y_hat_so_far_0"), "p_y_q_w_1": source_spec("p_y_q_w_1")},
                    {"p_s_w_1": out("p_s_w_1"), "p_y_hat": out("p_y_hat")},
                ),
            ],
            "recon": graph_step(
                "decode_p_recon",
                "models/p_recon.onnx",
                {"p_y_hat": source_spec("p_y_hat"), "p_ctx": source_spec("p_ctx")},
                {"encoder_p_reference_feature": out("encoder_p_reference_feature"), "encoder_p_reference_frame": out("encoder_p_reference_frame")},
            ),
        },
    }
    return {
        "schema_version": 1,
        "metadata": {
            "source_root": str(source_root),
            "source_policy": "source-derived; no previous Android exporter or benchmark logic reused",
            "height": args.height,
            "width": args.width,
            "qp": args.qp,
            "precision": args.precision,
            "baseline_execution": "server PyTorch FP32; every exported ONNX graph uses FP32 tensors and FP32 weights",
            "opset": args.opset,
            "i_checkpoint": str(i_ckpt),
            "p_checkpoint": str(p_ckpt),
            "i_checkpoint_sha256": sha256_file(i_ckpt),
            "p_checkpoint_sha256": sha256_file(p_ckpt),
            "note": "rANS tables, packed symbols, and reference payloads are exported from the same server model run.",
        },
        "entropy": entropy_assets,
        "stream": stream,
        "decoder": decoder,
        "modules": {
            "temporal_reference": [
                {
                    "name": "from_frame",
                    "steps": [
                        graph_step(
                            "temporal_from_frame",
                            "models/temporal_from_frame.onnx",
                            {"reference_frame": tensor_spec(baseline("encoder_i_reference_frame"), shape_of(tensors, "encoder_i_reference_frame"))},
                            {"temporal_from_frame_feature": out("temporal_from_frame_feature"), "p_ctx": out("p_ctx"), "p_ctx_t": out("p_ctx_t")},
                        )
                    ],
                    "binary_comparisons": [],
                },
                {
                    "name": "from_feature",
                    "steps": [
                        graph_step(
                            "temporal_from_feature",
                            "models/temporal_from_feature.onnx",
                            {"reference_feature": tensor_spec(baseline("encoder_p_reference_feature"), shape_of(tensors, "encoder_p_reference_feature"))},
                            {"temporal_from_feature_feature": out("temporal_from_feature_feature"), "p_ctx_from_feature": out("p_ctx_from_feature"), "p_ctx_t_from_feature": out("p_ctx_t_from_feature")},
                        )
                    ],
                    "binary_comparisons": [],
                },
            ],
            "complete_encoder": [
                {
                    "name": "canonical_i_then_p",
                    "steps": p_steps,
                    "binary_comparisons": [
                        {"android": "outputs/i_rans_payload.bin", "baseline": "baseline/bitstream/i_rans_payload.bin"},
                        {"android": "outputs/p_rans_payload.bin", "baseline": "baseline/bitstream/p_rans_payload.bin"},
                    ],
                }
            ],
            "image_inference_fused": [
                {
                    "name": "fused_i_then_p",
                    "steps": [
                        graph_step(
                            "i_image_encoder_fused",
                            "models/i_image_encoder_fused.onnx",
                            {"input_i_frame": tensor_spec("baseline/inputs/input_i_frame.f32le", [1, 3, args.height, args.width])},
                            {
                                "i_z_hat": out("i_z_hat"),
                                "i_y_q_w_0": out("i_y_q_w_0"),
                                "i_y_q_w_1": out("i_y_q_w_1"),
                                "i_y_q_w_2": out("i_y_q_w_2"),
                                "i_y_q_w_3": out("i_y_q_w_3"),
                                "i_s_w_0": out("i_s_w_0"),
                                "i_s_w_1": out("i_s_w_1"),
                                "i_s_w_2": out("i_s_w_2"),
                                "i_s_w_3": out("i_s_w_3"),
                                "encoder_i_reference_frame": out("encoder_i_reference_frame"),
                            },
                        ),
                        graph_step(
                            "p_image_encoder_fused",
                            "models/p_image_encoder_fused.onnx",
                            {
                                "input_p_frame": tensor_spec("baseline/inputs/input_p_frame.f32le", [1, 3, args.height, args.width]),
                                "encoder_i_reference_frame": source_spec("encoder_i_reference_frame"),
                            },
                            {
                                "p_z_hat": out("p_z_hat"),
                                "p_y_q_w_0": out("p_y_q_w_0"),
                                "p_y_q_w_1": out("p_y_q_w_1"),
                                "p_s_w_0": out("p_s_w_0"),
                                "p_s_w_1": out("p_s_w_1"),
                                "encoder_p_reference_frame": out("encoder_p_reference_frame"),
                            },
                        ),
                    ],
                    "binary_comparisons": [],
                }
            ],
            "complete_decoder": [
                {
                    "name": "full_bitstream_decode_v2",
                    "steps": [],
                    "binary_comparisons": [],
                }
            ],
        },
    }


if __name__ == "__main__":
    main()
