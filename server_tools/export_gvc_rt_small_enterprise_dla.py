#!/usr/bin/env python3
"""Export the fixed-QP9 GVC-RT-Small neural backbone.

The exported runtime path follows the source reference reset: frame zero and
frames whose zero-based index modulo 64 equals one rebuild temporal context
from the previous reconstructed frame. Other frames use the previous feature.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn


TensorSpec = Tuple[str, Tuple[int, int, int, int]]
PUBLIC_MODEL_ID = "gvc-rt-small"
PUBLIC_CHECKPOINT_NAME = "gvc-rt-small-psnr-v1.ckpt"
FORBIDDEN_DELIVERY_MARKERS = (b"mlvc", b"dmc61")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_forbidden_delivery_markers(path: Path) -> List[str]:
    content = path.read_bytes().lower()
    return [marker.decode("ascii") for marker in FORBIDDEN_DELIVERY_MARKERS if marker in content]


def compare(actual: np.ndarray, expected: np.ndarray) -> Dict[str, float]:
    actual64 = actual.astype(np.float64, copy=False).reshape(-1)
    expected64 = expected.astype(np.float64, copy=False).reshape(-1)
    diff = actual64 - expected64
    denominator = float(np.linalg.norm(actual64) * np.linalg.norm(expected64))
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "cosine": float(np.dot(actual64, expected64) / denominator) if denominator else 1.0,
    }


def run(command: Sequence[str], log_path: Path, env: Dict[str, str] = None) -> Tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        check=False,
    )
    text = "$ " + " ".join(command) + "\n\n" + completed.stdout
    log_path.write_text(text, encoding="utf-8")
    return completed.returncode, completed.stdout


def find_tool(explicit: str, name: str) -> str:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("missing {}: {}".format(name, path))
        return str(path)
    found = shutil.which(name)
    if not found:
        raise FileNotFoundError("{} is not in PATH".format(name))
    return found


def compile_dla(
    name: str,
    tflite_path: Path,
    dla_path: Path,
    logs: Path,
    ncc: str,
    ncc_env: Dict[str, str],
    arch: str,
) -> Tuple[Dict[str, Any], str]:
    ncc_flags = ["--arch", arch, "--opt-bw", "--relax-fp32"]
    with tempfile.TemporaryDirectory(prefix="gvc-rt-small-ncc-") as temporary:
        public_dir = Path(temporary)
        public_tflite = public_dir / (name + ".tflite")
        public_dla = public_dir / (name + ".dla")
        shutil.copy2(str(tflite_path), str(public_tflite))
        check_rc, check_text = run(
            [ncc, str(public_tflite)] + ncc_flags + ["--check-target-only"],
            logs / "ncc_check_target.log",
            ncc_env,
        )
        plan_rc, plan_text = run(
            [ncc, str(public_tflite)] + ncc_flags + ["--show-exec-plan", "--show-memory-summary"],
            logs / "ncc_exec_plan.log",
            ncc_env,
        )
        compile_rc, compile_text = run(
            [ncc, str(public_tflite)] + ncc_flags + [
                "--gen-debug-info", "--show-exec-plan", "--show-memory-summary", "-d", str(public_dla)
            ],
            logs / "ncc_compile_dla.log",
            ncc_env,
        )
        if compile_rc == 0 and public_dla.is_file():
            shutil.copy2(str(public_dla), str(dla_path))

    target_lines = [line.strip() for line in plan_text.splitlines() if "Target:" in line]
    expected_target = "MDLA_" + arch.replace("mdla", "", 1).replace(".", "_")
    mdla_only = bool(target_lines) and all(expected_target in line for line in target_lines)
    result = {
        "check_rc": check_rc,
        "plan_rc": plan_rc,
        "compile_rc": compile_rc,
        "execution_targets": target_lines,
        "mdla_only": mdla_only,
        "dla": str(dla_path) if dla_path.is_file() else None,
        "dla_sha256": sha256(dla_path) if dla_path.is_file() else None,
    }
    result["offline_compile_ok"] = (
        check_rc == 0 and plan_rc == 0 and compile_rc == 0 and dla_path.is_file() and mdla_only
    )
    return result, check_text + "\n" + plan_text + "\n" + compile_text


def nhwc_to_nchw(value: torch.Tensor) -> torch.Tensor:
    return value.permute(0, 3, 1, 2).contiguous()


def nchw_to_nhwc(value: torch.Tensor) -> torch.Tensor:
    return value.permute(0, 2, 3, 1).contiguous()


def install_torch_compatibility() -> None:
    from torch.nn.modules import utils as module_utils

    if hasattr(module_utils, "consume_prefix_in_state_dict_if_present"):
        return

    def consume_prefix_in_state_dict_if_present(state_dict: Dict[str, Any], prefix: str) -> None:
        for key in list(state_dict.keys()):
            if key.startswith(prefix):
                state_dict[key[len(prefix) :]] = state_dict.pop(key)
        if hasattr(state_dict, "_metadata"):
            for key in list(state_dict._metadata.keys()):
                if key and (key == prefix.replace(".", "") or key.startswith(prefix)):
                    state_dict._metadata[key[len(prefix) :]] = state_dict._metadata.pop(key)

    module_utils.consume_prefix_in_state_dict_if_present = consume_prefix_in_state_dict_if_present


def space_to_depth(value: torch.Tensor, factor: int) -> torch.Tensor:
    return torch.cat(
        tuple(value[:, :, row::factor, column::factor] for row in range(factor) for column in range(factor)),
        dim=1,
    )


class TemporalReferenceFromFeatureNhwc(nn.Module):
    def __init__(self, model: nn.Module, q_index: int) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("q_feature", model.q_feature[q_index : q_index + 1].detach().clone())

    def forward(self, ref_feature_nhwc: torch.Tensor):
        ref_feature = nhwc_to_nchw(ref_feature_nhwc)
        ctx, ctx_t, memory = self.model.context_generation({"ref_feature": ref_feature}, self.q_feature)
        return nchw_to_nhwc(ctx), nchw_to_nhwc(ctx_t), nchw_to_nhwc(memory)


class TemporalReferenceFromFrameNhwc(nn.Module):
    def __init__(self, model: nn.Module, q_index: int) -> None:
        super().__init__()
        self.model = model
        self.factor = int(model.pixel_shuffle_factor)
        self.feature_adaptor_i = copy.deepcopy(model.feature_adaptor_i)
        self.feature_adaptor_p = copy.deepcopy(model.feature_adaptor_p)
        self.feature_extractor = copy.deepcopy(model.feature_extractor)
        self.register_buffer("q_feature", model.q_feature[q_index : q_index + 1].detach().clone())

        adaptor = self.feature_adaptor_i.conv.adaptor
        if adaptor is None:
            raise RuntimeError("feature_adaptor_i requires an input adaptor convolution")
        factor_squared = self.factor**2
        input_channels = int(adaptor.in_channels) // factor_squared
        input_permutation = [
            channel * factor_squared + offset
            for offset in range(factor_squared)
            for channel in range(input_channels)
        ]
        adaptor.weight.data = adaptor.weight[:, input_permutation].detach().clone()

    def forward(self, ref_frame_nhwc: torch.Tensor):
        ref_frame = nhwc_to_nchw(ref_frame_nhwc)
        ref_frame = self.model.shift_input(ref_frame)
        feature = self.feature_adaptor_i(space_to_depth(ref_frame, self.factor))
        ref_feature, memory = feature.chunk(2, dim=1)
        if self.model.chain_feature_adaptors:
            ref_feature = self.feature_adaptor_p(ref_feature)
        ctx, ctx_t = self.feature_extractor(ref_feature, self.q_feature)
        return nchw_to_nhwc(ctx), nchw_to_nhwc(ctx_t), nchw_to_nhwc(memory)


class EncoderAnalysisNhwc(nn.Module):
    def __init__(self, model: nn.Module, q_index: int) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("q_encoder", model.q_encoder[q_index : q_index + 1].detach().clone())
        analysis = model.encoder
        factor_squared = analysis.pixel_shuffle_factor ** 2
        input_channels = int(analysis.conv1.in_channels) // factor_squared
        input_permutation = [
            channel * factor_squared + offset
            for offset in range(factor_squared)
            for channel in range(input_channels)
        ]
        self.register_buffer("conv1_weight", analysis.conv1.weight[:, input_permutation].detach().clone())
        self.register_buffer("conv1_bias", analysis.conv1.bias.detach().clone())

    def forward(self, frame_nhwc: torch.Tensor, ctx_nhwc: torch.Tensor):
        x = nhwc_to_nchw(frame_nhwc)
        ctx = nhwc_to_nchw(ctx_nhwc)
        x = self.model.shift_input(x)
        analysis = self.model.encoder
        y = space_to_depth(x, analysis.pixel_shuffle_factor)
        y = torch.nn.functional.conv2d(y, self.conv1_weight, self.conv1_bias)
        y = analysis.conv2(torch.cat((y, ctx), dim=1))
        y = analysis.conv3(y)
        y = analysis.down(y * self.q_encoder)
        return nchw_to_nhwc(y)


class DecoderSynthesisNhwc(nn.Module):
    def __init__(self, model: nn.Module, q_index: int) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("q_decoder", model.q_decoder[q_index : q_index + 1].detach().clone())
        self.register_buffer("q_recon", model.q_recon[q_index : q_index + 1].detach().clone())

    def forward(self, y_nhwc: torch.Tensor, ctx_nhwc: torch.Tensor, memory_nhwc: torch.Tensor):
        y = nhwc_to_nchw(y_nhwc)
        ctx = nhwc_to_nchw(ctx_nhwc)
        memory = nhwc_to_nchw(memory_nhwc)
        feature, memory = self.model.decoder(y, ctx, self.q_decoder, memory)
        frame = self.model.recon_generation_net(feature, self.q_recon)
        frame = self.model.clamp_x_hat(frame)
        frame = self.model.unshift_output(frame)
        ref_feature = torch.cat((feature, memory), dim=1)
        return nchw_to_nhwc(ref_feature), nchw_to_nhwc(frame)


def load_model(source_root: Path, checkpoint: Path) -> nn.Module:
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(source_root / "video"))
    install_torch_compatibility()
    from src.models.dmc_6.dmc_61sb import DMC
    from src.utils.stream_helper import get_state_dict

    model = DMC(
        depth_conv_block_params={
            "activation": "LeakyReLU",
            "zero_init_residual": True,
            "chunk_mode": "gated",
            "ffn_gate_activation": "ReLU1",
        },
        feature_channels=48,
        spatial_prior_channels=128,
        recon_channels=192,
        hidden_channels=192,
        hyperprior_num_blocks=2,
        y_scale_repeat=4,
        z_channels=48,
        y_channels=48,
        hyperprior_variant="mini",
        feature_extractor_num_conv1_layers=1,
        feature_extractor_num_conv2_layers=1,
        input_offset=-0.5,
        memory_activation="identity",
        chain_feature_adaptors=True,
    )
    model.load_state_dict(get_state_dict(str(checkpoint)), strict=True)
    model.eval()
    return model


def save_fixture(path: Path, value: torch.Tensor) -> None:
    value.detach().cpu().numpy().astype("<f4", copy=False).tofile(path)


def export_one(
    name: str,
    wrapper: nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    input_names: Tuple[str, ...],
    output_names: Tuple[str, ...],
    output_dir: Path,
    pytorch_converter: str,
    onnx_converter: str,
    tflite_python: str,
    runner: Path,
    ncc: str,
    ncc_env: Dict[str, str],
    arch: str,
) -> Dict[str, Any]:
    logs = output_dir / "logs" / name
    fixtures = output_dir / "fixtures" / name
    fixtures.mkdir(parents=True, exist_ok=True)
    wrapper = wrapper.cpu().eval()
    with torch.no_grad():
        expected_value = wrapper(*inputs)
        expected = expected_value if isinstance(expected_value, tuple) else (expected_value,)
        traced = torch.jit.trace(wrapper, inputs, strict=False)
        traced_value = traced(*inputs)
        traced_outputs = traced_value if isinstance(traced_value, tuple) else (traced_value,)

    pt_path = output_dir / (name + ".pt")
    tflite_path = output_dir / (name + ".tflite")
    dla_path = output_dir / (name + ".dla")
    traced.save(str(pt_path))

    input_records = []
    for tensor_name, value in zip(input_names, inputs):
        file_name = "input_{}.f32le".format(tensor_name)
        save_fixture(fixtures / file_name, value)
        input_records.append({"name": tensor_name, "shape": list(value.shape), "dtype": "float32", "file": file_name})
    output_records = []
    for tensor_name, value in zip(output_names, expected):
        file_name = "expected_{}.f32le".format(tensor_name)
        save_fixture(fixtures / file_name, value)
        output_records.append({"name": tensor_name, "shape": list(value.shape), "dtype": "float32", "file": file_name})

    record: Dict[str, Any] = {
        "name": name,
        "inputs": input_records,
        "outputs": output_records,
        "torchscript": str(pt_path),
        "torchscript_sha256": sha256(pt_path),
        "torchscript_vs_wrapper": {
            tensor_name: compare(actual.detach().numpy(), reference.detach().numpy())
            for tensor_name, actual, reference in zip(output_names, traced_outputs, expected)
        },
    }

    # NeuroPilot 7.0.8 ships MTK Converter 7.16, whose PyTorch importer
    # rejects torch >= 2.0. Export ONNX in the source-compatible process and
    # keep the MTK conversion isolated from that PyTorch version constraint.
    onnx_path = output_dir / (name + ".onnx")
    torch.onnx.export(
        wrapper,
        inputs,
        str(onnx_path),
        input_names=list(input_names),
        output_names=list(output_names),
        opset_version=12,
        do_constant_folding=True,
    )
    record.update({"onnx": str(onnx_path), "onnx_sha256": sha256(onnx_path)})
    converter_command = [
        onnx_converter,
        "--input_model_file", str(onnx_path),
        "--output_file", str(tflite_path),
        "--output_file_format", "tflite",
        "--input_names", ",".join(input_names),
        "--input_shapes", ":".join(",".join(str(dim) for dim in value.shape) for value in inputs),
        "--output_names", ",".join(output_names),
        "--tflite_op_export_spec", "builtin_first",
    ]
    converter_rc, _ = run(converter_command, logs / "converter.log")
    record.update({"converter_command": converter_command, "converter_rc": converter_rc})
    if converter_rc != 0 or not tflite_path.is_file():
        record["status"] = "converter_failed"
        return record
    record.update({"tflite": str(tflite_path), "tflite_sha256": sha256(tflite_path)})

    exchange = output_dir / "tflite_exchange" / name
    exchange.mkdir(parents=True, exist_ok=True)
    runner_command = [tflite_python, str(runner), "--model", str(tflite_path), "--output-dir", str(exchange)]
    for index, value in enumerate(inputs):
        input_path = exchange / "input_{}.npy".format(index)
        np.save(str(input_path), value.numpy())
        runner_command.extend(["--input", str(input_path)])
    runner_rc, runner_text = run(runner_command, logs / "tflite_runner.log")
    record["tflite_runner_rc"] = runner_rc
    if runner_rc == 0:
        metrics = {}
        for index, (tensor_name, reference) in enumerate(zip(output_names, expected)):
            actual = np.load(str(exchange / "output_{}.npy".format(index)))
            metrics[tensor_name] = compare(actual, reference.numpy())
        record["tflite_vs_pytorch"] = metrics
        record["precision_passed"] = all(
            value["max_abs"] <= 1e-3 and value["rmse"] <= 1e-4 for value in metrics.values()
        )
    else:
        record["precision_passed"] = False
        record["precision_error"] = runner_text[-2000:]

    compile_result, compile_text = compile_dla(name, tflite_path, dla_path, logs, ncc, ncc_env, arch)
    record.update(compile_result)
    record["status"] = "ok" if record["offline_compile_ok"] and record["precision_passed"] else "failed"
    if record["status"] != "ok":
        record["ncc_tail"] = compile_text[-4000:]
    return record


def build_readme(model_extension: str, runtime_description: str) -> str:
    return """# GVC-RT-Small 神经网络包（270p / QP9）

## 固定配置

- 模型：GVC-RT-Small PSNR v1
- 分辨率：256 x 512
- QP index：9（全序列固定）
- Tensor 布局：NHWC
- 外部 Tensor 类型：FP32
- 运行形式：{runtime_description}

## 调用流程

首帧将 `reference_frame` 初始化为全 0.5 的 `[1,256,512,3]`。第 0 帧及满足
`frame_index % 64 == 1` 的帧调用 `temporal_from_frame.{model_extension}`；其他帧调用
`temporal_from_feature.{model_extension}`。

```text
reference_frame / reference_feature
  -> temporal_from_frame.{model_extension} / temporal_from_feature.{model_extension}
  -> ctx + ctx_t + memory

frame + ctx
  -> encoder.{model_extension}
  -> latent_y

latent_y + ctx + memory
  -> decoder.{model_extension}
  -> next_ref_feature + reconstructed_frame
```

Decoder 的两个输出都必须保留：`next_ref_feature` 用于普通帧时序参考，
`reconstructed_frame` 用于参考重置和视频输出。具体 Tensor 名称、shape、SHA256
以 `manifest.json` 为准。
""".format(model_extension=model_extension, runtime_description=runtime_description)


def published_model_names(q_index: int, suffix: str) -> Dict[str, str]:
    return {
        "temporal_from_frame_qp{}".format(q_index): "temporal_from_frame.{}".format(suffix),
        "temporal_from_feature_qp{}".format(q_index): "temporal_from_feature.{}".format(suffix),
        "encoder_analysis_qp{}".format(q_index): "encoder.{}".format(suffix),
        "decoder_synthesis_qp{}".format(q_index): "decoder.{}".format(suffix),
    }


def package(
    output_dir: Path,
    records: List[Dict[str, Any]],
    checkpoint: Path,
    q_index: int,
    package_name: str,
    target_label: str,
) -> Path:
    delivery = output_dir / package_name
    models = delivery / "models"
    if delivery.exists():
        shutil.rmtree(str(delivery))
    models.mkdir(parents=True)
    publish_names = published_model_names(q_index, "dla")
    published = []
    for record in records:
        if record.get("status") != "ok":
            raise RuntimeError("cannot package failed model: {}".format(record["name"]))
        source = Path(record["dla"])
        published_path = models / publish_names[record["name"]]
        shutil.copy2(str(source), str(published_path))
        published.append(
            {
                "name": published_path.stem,
                "file": "models/" + published_path.name,
                "bytes": published_path.stat().st_size,
                "sha256": sha256(published_path),
                "inputs": record["inputs"],
                "outputs": record["outputs"],
                "offline_compile_verified": True,
                "mdla_only": record["mdla_only"],
            }
        )
    manifest = {
        "package": package_name,
        "source_model": PUBLIC_MODEL_ID,
        "checkpoint": PUBLIC_CHECKPOINT_NAME,
        "checkpoint_sha256": sha256(checkpoint),
        "resolution": {"height": 256, "width": 512},
        "fixed_q_index": 9,
        "layout": "NHWC",
        "io_dtype": "FP32",
        "target": target_label,
        "initial_reference": {"name": "reference_frame", "shape": [1, 256, 512, 3], "value": 0.5},
        "reference_reset": {
            "period": 64,
            "phase": 1,
            "condition": "frame_index == 0 or frame_index % 64 == 1",
        },
        "models": published,
    }
    (delivery / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (delivery / "README.md").write_text(build_readme("dla", target_label + " 离线 DLA"), encoding="utf-8")
    delivery_files = list(models.glob("*.dla")) + [delivery / "manifest.json", delivery / "README.md"]
    leaked = {}
    for path in delivery_files:
        markers = find_forbidden_delivery_markers(path)
        if markers:
            leaked[path.relative_to(delivery).as_posix()] = markers
    if leaked:
        raise RuntimeError(
            "delivery contains private model identifiers; use an output path without legacy names: {}".format(leaked)
        )
    checksum_paths = sorted(list(models.glob("*.dla")) + [delivery / "manifest.json", delivery / "README.md"])
    (delivery / "SHA256SUMS.txt").write_text(
        "".join("{}  {}\n".format(sha256(path), path.relative_to(delivery).as_posix()) for path in checksum_paths),
        encoding="ascii",
    )
    archive = output_dir / (package_name + ".tar.gz")
    with tarfile.open(str(archive), "w:gz") as handle:
        handle.add(str(delivery), arcname=package_name)
    return archive


def package_tflite(
    output_dir: Path,
    records: List[Dict[str, Any]],
    checkpoint: Path,
    q_index: int,
) -> Path:
    package_name = "gvc-rt-small_tflite_codec_270p_qp9"
    delivery = output_dir / package_name
    models = delivery / "models"
    if delivery.exists():
        shutil.rmtree(str(delivery))
    models.mkdir(parents=True)
    publish_names = published_model_names(q_index, "tflite")
    published = []
    for record in records:
        if record.get("status") != "ok":
            raise RuntimeError("cannot package failed model: {}".format(record["name"]))
        source = Path(record["tflite"])
        published_path = models / publish_names[record["name"]]
        shutil.copy2(str(source), str(published_path))
        published.append(
            {
                "name": published_path.stem,
                "file": "models/" + published_path.name,
                "bytes": published_path.stat().st_size,
                "sha256": sha256(published_path),
                "inputs": record["inputs"],
                "outputs": record["outputs"],
            }
        )
    manifest = {
        "package": package_name,
        "source_model": PUBLIC_MODEL_ID,
        "checkpoint": PUBLIC_CHECKPOINT_NAME,
        "checkpoint_sha256": sha256(checkpoint),
        "resolution": {"height": 256, "width": 512},
        "fixed_q_index": 9,
        "layout": "NHWC",
        "io_dtype": "FP32",
        "initial_reference": {"name": "reference_frame", "shape": [1, 256, 512, 3], "value": 0.5},
        "reference_reset": {
            "period": 64,
            "phase": 1,
            "condition": "frame_index == 0 or frame_index % 64 == 1",
        },
        "models": published,
    }
    (delivery / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (delivery / "README.md").write_text(
        build_readme("tflite", "MediaTek TFLite 在线编译"),
        encoding="utf-8",
    )
    delivery_files = list(models.glob("*.tflite")) + [delivery / "manifest.json", delivery / "README.md"]
    leaked = {}
    for path in delivery_files:
        markers = find_forbidden_delivery_markers(path)
        if markers:
            leaked[path.relative_to(delivery).as_posix()] = markers
    if leaked:
        raise RuntimeError(
            "delivery contains private model identifiers; use an output path without legacy names: {}".format(leaked)
        )
    checksum_paths = sorted(delivery_files)
    (delivery / "SHA256SUMS.txt").write_text(
        "".join("{}  {}\n".format(sha256(path), path.relative_to(delivery).as_posix()) for path in checksum_paths),
        encoding="ascii",
    )
    archive = output_dir / (package_name + ".tar.gz")
    with tarfile.open(str(archive), "w:gz") as handle:
        handle.add(str(delivery), arcname=package_name)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pytorch-converter", default=None)
    parser.add_argument("--onnx-converter", default=None)
    parser.add_argument("--tflite-python", default=None)
    parser.add_argument("--ncc-tflite", required=True)
    parser.add_argument("--ncc-lib-dir", type=Path, default=None)
    parser.add_argument("--arch", default="mdla5.3")
    parser.add_argument("--q-index", type=int, default=9)
    parser.add_argument("--package-name", default="gvc-rt-small_dla_codec_270p_qp9")
    parser.add_argument("--package-only", action="store_true")
    args = parser.parse_args()
    if args.q_index != 9:
        raise ValueError("this deployment is fixed to q-index 9")

    source_root = args.source_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = (args.output_dir or source_root.parent / "gvc-rt-small-export").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint.is_file():
        raise FileNotFoundError("missing checkpoint: {}".format(checkpoint))
    ncc = find_tool(args.ncc_tflite, "ncc-tflite")
    ncc_lib_dir = args.ncc_lib_dir or Path(ncc).resolve().parent.parent / "lib"
    ncc_env = os.environ.copy()
    ncc_env["LD_LIBRARY_PATH"] = str(ncc_lib_dir) + os.pathsep + ncc_env.get("LD_LIBRARY_PATH", "")
    if args.package_only:
        manifest_path = output_dir / "gvc-rt-small-export-manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("missing export manifest: {}".format(manifest_path))
        export_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = export_manifest.get("records", [])
        if len(records) != 4 or not all(record.get("status") == "ok" for record in records):
            raise RuntimeError("package-only requires four successful export records")
        for record in records:
            name = record["name"]
            compile_result, compile_text = compile_dla(
                name,
                Path(record["tflite"]),
                output_dir / (name + ".dla"),
                output_dir / "logs" / name,
                ncc,
                ncc_env,
                args.arch,
            )
            record.update(compile_result)
            record["status"] = (
                "ok" if record["offline_compile_ok"] and record.get("precision_passed") else "failed"
            )
            if record["status"] != "ok":
                record["ncc_tail"] = compile_text[-4000:]
        manifest_path.write_text(json.dumps(export_manifest, indent=2), encoding="utf-8")
        if not all(record.get("status") == "ok" for record in records):
            failed = [record["name"] for record in records if record.get("status") != "ok"]
            raise RuntimeError("sanitized DLA recompile failed: {}".format(", ".join(failed)))
        target_label = "MDLA " + args.arch.replace("mdla", "", 1)
        archive = package(
            output_dir,
            records,
            checkpoint,
            args.q_index,
            args.package_name,
            target_label,
        )
        print("package={}".format(output_dir / args.package_name), flush=True)
        print("archive={}".format(archive), flush=True)
        print("archive_sha256={}".format(sha256(archive)), flush=True)
        return
    pytorch_converter = find_tool(args.pytorch_converter, "mtk_pytorch_converter")
    onnx_converter = find_tool(args.onnx_converter, "mtk_onnx_converter")
    tflite_python = args.tflite_python or str(Path(pytorch_converter).resolve().parent / "python")
    runner = source_root / "video/conversion/_exporter/_mediatek_tflite_runner.py"
    if not runner.is_file():
        raise FileNotFoundError("missing MediaTek TFLite runner: {}".format(runner))
    torch.manual_seed(0)
    model = load_model(source_root, checkpoint).cpu()

    ref_frame = torch.full((1, 256, 512, 3), 0.5, dtype=torch.float32)
    ref_feature = torch.zeros((1, 32, 64, 96), dtype=torch.float32)
    frame = torch.rand((1, 256, 512, 3), dtype=torch.float32)
    temporal_from_frame = TemporalReferenceFromFrameNhwc(model, args.q_index)
    temporal_from_feature = TemporalReferenceFromFeatureNhwc(model, args.q_index)
    with torch.no_grad():
        ctx, ctx_t, memory = temporal_from_frame(ref_frame)
    encoder = EncoderAnalysisNhwc(model, args.q_index)
    with torch.no_grad():
        latent_y = encoder(frame, ctx)
    decoder = DecoderSynthesisNhwc(model, args.q_index)

    with torch.no_grad():
        q_feature = model.q_feature[args.q_index : args.q_index + 1]
        source_frame_ctx = model.context_generation(
            {"ref_frame": nhwc_to_nchw(ref_frame), "ref_feature": None},
            q_feature,
        )
        source_feature_ctx = model.context_generation(
            {"ref_feature": nhwc_to_nchw(ref_feature)},
            q_feature,
        )
        source_encoder = model.encoder(
            model.shift_input(nhwc_to_nchw(frame)),
            nhwc_to_nchw(ctx),
            model.q_encoder[args.q_index : args.q_index + 1],
        )
        source_decoded_feature, source_decoded_memory = model.decoder(
            source_encoder,
            nhwc_to_nchw(ctx),
            model.q_decoder[args.q_index : args.q_index + 1],
            nhwc_to_nchw(memory),
        )
        source_reconstructed = model.recon_generation_net(
            source_decoded_feature,
            model.q_recon[args.q_index : args.q_index + 1],
        )
        source_reconstructed = model.unshift_output(model.clamp_x_hat(source_reconstructed))
        source_reference_feature = torch.cat((source_decoded_feature, source_decoded_memory), dim=1)

    candidates = [
        (
            "temporal_from_frame_qp9",
            temporal_from_frame,
            (ref_frame,),
            ("reference_frame",),
            ("ctx", "ctx_t", "memory"),
            tuple(nchw_to_nhwc(value) for value in source_frame_ctx),
        ),
        (
            "temporal_from_feature_qp9",
            temporal_from_feature,
            (ref_feature,),
            ("reference_feature",),
            ("ctx", "ctx_t", "memory"),
            tuple(nchw_to_nhwc(value) for value in source_feature_ctx),
        ),
        (
            "encoder_analysis_qp9",
            encoder,
            (frame, ctx),
            ("frame", "ctx"),
            ("latent_y",),
            (nchw_to_nhwc(source_encoder),),
        ),
        (
            "decoder_synthesis_qp9",
            decoder,
            (latent_y, ctx, memory),
            ("latent_y", "ctx", "memory"),
            ("next_ref_feature", "reconstructed_frame"),
            (nchw_to_nhwc(source_reference_feature), nchw_to_nhwc(source_reconstructed)),
        ),
    ]
    records = []
    for index, (name, wrapper, inputs, input_names, output_names, source_outputs) in enumerate(candidates, start=1):
        print("[gvc-rt-small] {}/{} export {}".format(index, len(candidates), name), flush=True)
        record = export_one(
            name,
            wrapper,
            inputs,
            input_names,
            output_names,
            output_dir,
            pytorch_converter,
            onnx_converter,
            tflite_python,
            runner,
            ncc,
            ncc_env,
            args.arch,
        )
        with torch.no_grad():
            wrapper_value = wrapper(*inputs)
            wrapper_outputs = wrapper_value if isinstance(wrapper_value, tuple) else (wrapper_value,)
        source_metrics = {
            tensor_name: compare(actual.detach().numpy(), expected.detach().numpy())
            for tensor_name, actual, expected in zip(output_names, wrapper_outputs, source_outputs)
        }
        source_passed = all(
            value["max_abs"] <= 1e-5 and value["rmse"] <= 1e-6 for value in source_metrics.values()
        )
        record["wrapper_vs_source"] = source_metrics
        record["source_precision_passed"] = source_passed
        if not source_passed:
            record["status"] = "source_precision_failed"
        records.append(record)
        print("[gvc-rt-small] {} status={}".format(name, record["status"]), flush=True)

    export_manifest = {
        "tool": Path(__file__).name,
        "source_root": str(source_root),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "model": PUBLIC_MODEL_ID,
        "resolution": [256, 512],
        "fixed_q_index": 9,
        "reference_reset": {"period": 64, "phase": 1},
        "layout": "NHWC",
        "io_dtype": "float32",
        "records": records,
    }
    manifest_path = output_dir / "gvc-rt-small-export-manifest.json"
    manifest_path.write_text(json.dumps(export_manifest, indent=2), encoding="utf-8")
    print("wrote {}".format(manifest_path), flush=True)

    if all(record.get("status") == "ok" for record in records):
        target = "MDLA " + args.arch.replace("mdla", "", 1)
        archive = package(
            output_dir,
            records,
            checkpoint,
            args.q_index,
            args.package_name,
            target,
        )
        print("package={}".format(output_dir / args.package_name), flush=True)
        print("archive={}".format(archive), flush=True)
        print("archive_sha256={}".format(sha256(archive)), flush=True)
        tflite_archive = package_tflite(output_dir, records, checkpoint, args.q_index)
        print("tflite_package={}".format(output_dir / "gvc-rt-small_tflite_codec_270p_qp9"), flush=True)
        print("tflite_archive={}".format(tflite_archive), flush=True)
        print("tflite_archive_sha256={}".format(sha256(tflite_archive)), flush=True)
    else:
        failed = [record["name"] for record in records if record.get("status") != "ok"]
        raise RuntimeError("export failed: {}".format(", ".join(failed)))


if __name__ == "__main__":
    main()
