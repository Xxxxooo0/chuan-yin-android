#!/usr/bin/env python3
"""Package the verified 270p GVC-RT neural codec DLA graphs for delivery."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any


MODEL_SPECS = (
    (
        "front/three_modules_offline_nhwc_manifest.json",
        "temporal_from_frame_big",
        "temporal_from_frame.dla",
        "从重建帧初始化P帧时序参考",
    ),
    (
        "front/three_modules_offline_nhwc_manifest.json",
        "temporal_from_feature_big",
        "temporal_from_feature.dla",
        "从参考特征更新P帧时序参考",
    ),
    (
        "front/three_modules_offline_nhwc_manifest.json",
        "i_encoder_analysis_big",
        "i_encoder.dla",
        "I帧分析变换",
    ),
    (
        "front/three_modules_offline_nhwc_manifest.json",
        "p_encoder_analysis_big",
        "p_encoder.dla",
        "P帧分析变换",
    ),
    (
        "decoder/decoder_full_norm_rewrite_manifest.json",
        "i_decoder_synthesis_merged_fp32",
        "i_decoder.dla",
        "I帧合成与重建",
    ),
    (
        "decoder/decoder_full_norm_rewrite_manifest.json",
        "p_decoder_synthesis_merged_fp32",
        "p_decoder.dla",
        "P帧合成、重建与参考特征更新",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        record["name"]: record
        for record in data.get("records", [])
        if isinstance(record, dict) and record.get("name")
    }


def record_dla(record: dict[str, Any]) -> Path:
    ncc = record.get("ncc") or {}
    path = Path(ncc.get("dla") or "")
    if record.get("status") != "ok" or not record.get("offline_compile_ok"):
        raise RuntimeError(f"model is not offline-compile verified: {record.get('name')}")
    if not path.is_file():
        raise FileNotFoundError(f"manifest-referenced DLA is missing: {path}")
    return path


def output_shapes(record: dict[str, Any]) -> Any:
    return (
        record.get("actual_output_shapes_nhwc")
        or record.get("actual_torchscript_output_shapes_nhwc")
        or record.get("output_shapes_nhwc")
    )


def build_readme() -> str:
    return """# GVC-RT DLA编解码器（270p）

## 运行配置

- 分辨率：`256 x 512`
- QP: `0`
- Tensor布局：`NHWC`
- 外部Tensor类型：`FP32`
- 目标设备：`MDLA 5.3`

## 帧处理流程

### I帧

```text
归一化RGB帧
-> i_encoder.dla
-> Latent直接传递
-> i_decoder.dla
-> 重建RGB帧
```

I帧重建结果用于初始化第一个P帧参考。

### 第一个P帧

```text
I帧重建结果
-> temporal_from_frame.dla
-> ctx

当前归一化RGB帧 + ctx
-> p_encoder.dla
-> Latent直接传递
-> p_decoder.dla
-> 重建RGB帧 + 参考特征
```

### 后续P帧

```text
上一帧参考特征
-> temporal_from_feature.dla
-> ctx

当前归一化RGB帧 + ctx
-> p_encoder.dla
-> Latent直接传递
-> p_decoder.dla
-> 重建RGB帧 + 更新后的参考特征
```

## Tensor处理

输入RGB从 `[0,1]` 归一化到 `[-1,1]`。Encoder输出直接传入对应的Decoder，
不改变shape、layout或dtype。重建RGB范围为 `[-1,1]`，通过
`(value + 1) / 2` 转换回显示范围。

`manifest.json` 是输入输出Tensor接口的权威清单。测试前使用
`SHA256SUMS.txt` 校验所有交付文件。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--archive", type=Path, default=None)
    args = parser.parse_args()

    export_root = args.export_root.resolve()
    output_dir = (args.output_dir or export_root / "enterprise_delivery").resolve()
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    manifests: dict[Path, dict[str, dict[str, Any]]] = {}
    published = []
    for manifest_rel, source_name, target_name, description in MODEL_SPECS:
        manifest_path = export_root / manifest_rel
        if manifest_path not in manifests:
            manifests[manifest_path] = load_records(manifest_path)
        record = manifests[manifest_path].get(source_name)
        if record is None:
            raise KeyError(f"missing model record {source_name} in {manifest_path}")

        source = record_dla(record)
        target = models_dir / target_name
        shutil.copy2(source, target)
        published.append(
            {
                "name": target.stem,
                "description": description,
                "file": f"models/{target.name}",
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
                "input_names": record.get("input_names"),
                "input_shapes_nhwc": record.get("input_shapes_nhwc"),
                "output_names": record.get("output_names"),
                "output_shapes_nhwc": output_shapes(record),
                "offline_compile_verified": True,
                "mdla_only": record.get("mdla_only"),
            }
        )

    delivery_manifest = {
        "package": "gvcrt_dla_codec_270p_qp0",
        "resolution": {"height": 256, "width": 512},
        "qp": 0,
        "layout": "NHWC",
        "io_dtype": "FP32",
        "target": "MDLA 5.3",
        "latent_bridge": "direct",
        "models": published,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(delivery_manifest, indent=2), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(build_readme(), encoding="utf-8")

    checksum_paths = sorted([*models_dir.glob("*.dla"), output_dir / "manifest.json", output_dir / "README.md"])
    checksum_text = "".join(
        f"{sha256(path)}  {path.relative_to(output_dir).as_posix()}\n"
        for path in checksum_paths
    )
    (output_dir / "SHA256SUMS.txt").write_text(checksum_text, encoding="ascii")

    archive = (args.archive or export_root / "gvcrt_dla_codec_270p_qp0.tar.gz").resolve()
    with tarfile.open(archive, "w:gz") as package:
        package.add(output_dir, arcname=output_dir.name)

    print(f"packaged_models={len(published)}")
    print(f"output_dir={output_dir}")
    print(f"archive={archive}")
    print(f"archive_sha256={sha256(archive)}")


if __name__ == "__main__":
    main()
