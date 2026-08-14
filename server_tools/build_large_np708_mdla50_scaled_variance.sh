#!/usr/bin/env bash
# Build the 270p/QP0 Large enterprise package with the FP16-stable decoder.

set -euo pipefail

ANDROID_ROOT="${ANDROID_ROOT:-/media/ltelab/D/weilingfeng/GVC-RT_clean_android}"
SOURCE_ROOT="${SOURCE_ROOT:-/media/ltelab/D/weilingfeng/GVC-RT_inference}"
SOURCE_PYTHON="${SOURCE_PYTHON:-/media/ltelab/D/weilingfeng/conda_envs/weilingfeng/bin/python}"
NP708_ENV="${NP708_ENV:-/media/ltelab/D/weilingfeng/conda_envs/mtk708_py38}"
NP708_SDK="${NP708_SDK:-$ANDROID_ROOT/tools/neuropilot-sdk-premium-7.0.8-build20240807}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ANDROID_ROOT/outputs/gvc_rt_large_np708_mdla50_scaled_variance}"

MTK_CONVERTER="${MTK_CONVERTER:-$NP708_ENV/bin/mtk_pytorch_converter}"
NCC_TFLITE="${NCC_TFLITE:-$NP708_SDK/neuron_sdk/host/bin/ncc-tflite}"
NCC_LIB_DIR="${NCC_LIB_DIR:-$NP708_SDK/neuron_sdk/host/lib}"

PACKAGE_NAME="gvc-rt-large_dla_codec_270p_qp0_np708_mdla50_scaled_variance"
FINAL_PACKAGE_NAME="${PACKAGE_NAME}_with_inputs"
FRONT_DIR="$OUTPUT_ROOT/front"
DECODER_DIR="$OUTPUT_ROOT/decoder"
MODEL_DIR="$OUTPUT_ROOT/$PACKAGE_NAME"
MODEL_ARCHIVE="$OUTPUT_ROOT/$PACKAGE_NAME.tar.gz"
PRECISION_DIR="$OUTPUT_ROOT/gvc-rt-large_precision_270p_qp0_scaled_variance"
PRECISION_ARCHIVE="$OUTPUT_ROOT/gvc-rt-large_precision_270p_qp0_scaled_variance.tar.gz"
FINAL_ARCHIVE="$OUTPUT_ROOT/$FINAL_PACKAGE_NAME.tar.gz"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "missing required file: $1" >&2
    exit 1
  fi
}

require_file "$SOURCE_PYTHON"
require_file "$MTK_CONVERTER"
require_file "$NCC_TFLITE"
require_file "$SOURCE_ROOT/ckpt/checkpoints/GVC-RT_B_I.pt"
require_file "$SOURCE_ROOT/ckpt/checkpoints/GVC-RT_B_P.pt"

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "output already exists: $OUTPUT_ROOT" >&2
  echo "Set OUTPUT_ROOT to a new directory; existing exports are never overwritten." >&2
  exit 1
fi

mkdir -p "$FRONT_DIR" "$DECODER_DIR"
export LD_LIBRARY_PATH="$NCC_LIB_DIR:${LD_LIBRARY_PATH:-}"

cd "$ANDROID_ROOT"

echo "[large-np708] 1/5 export front graphs"
"$SOURCE_PYTHON" -u server_tools/export_three_modules_offline_nhwc.py \
  --source-root "$SOURCE_ROOT" \
  --android-root "$ANDROID_ROOT" \
  --output-dir "$FRONT_DIR" \
  --pytorch-converter "$MTK_CONVERTER" \
  --ncc-tflite "$NCC_TFLITE" \
  --arch mdla5.0 \
  --qp 0 \
  --candidates temporal_from_frame_big,temporal_from_feature_big,i_encoder_analysis_big,p_encoder_analysis_big

echo "[large-np708] 2/5 export scaled-variance merged decoders"
"$SOURCE_PYTHON" -u server_tools/export_decoder_full_norm_rewrite_nhwc.py \
  --source-root "$SOURCE_ROOT" \
  --android-root "$ANDROID_ROOT" \
  --output-dir "$DECODER_DIR" \
  --pytorch-converter "$MTK_CONVERTER" \
  --ncc-tflite "$NCC_TFLITE" \
  --arch mdla5.0 \
  --qp 0 \
  --targets i,p \
  --tflite-op-export-spec builtin_first

echo "[large-np708] validate exports"
"$SOURCE_PYTHON" - "$FRONT_DIR" "$DECODER_DIR" <<'PY'
import json
import sys
from pathlib import Path

front_dir = Path(sys.argv[1])
decoder_dir = Path(sys.argv[2])
front = json.loads((front_dir / "three_modules_offline_nhwc_manifest.json").read_text())
decoder = json.loads((decoder_dir / "decoder_full_norm_rewrite_manifest.json").read_text())

expected_front = {
    "temporal_from_frame_big",
    "temporal_from_feature_big",
    "i_encoder_analysis_big",
    "p_encoder_analysis_big",
}
front_records = {record["name"]: record for record in front.get("records", [])}
decoder_records = {record["name"]: record for record in decoder.get("records", [])}

for name in sorted(expected_front):
    record = front_records.get(name)
    if not record or record.get("status") != "ok" or not record.get("offline_compile_ok"):
        raise RuntimeError("front export is not offline ready: {}".format(name))

rewrite = (decoder.get("rewrite") or {}).get("GroupNorm", "")
if "s=1/16" not in rewrite:
    raise RuntimeError("decoder manifest does not contain the scaled-variance GroupNorm fix")

for name in ("i_decoder_synthesis_merged_fp32", "p_decoder_synthesis_merged_fp32"):
    record = decoder_records.get(name)
    precision = (record or {}).get("precision") or {}
    if (
        not record
        or record.get("status") != "ok"
        or not record.get("offline_compile_ok")
        or not record.get("mdla_only")
        or not precision.get("passed")
    ):
        raise RuntimeError("scaled-variance decoder is not ready: {}".format(name))

print("validated front=4 decoder=2 groupnorm=scaled_variance_s_1_16")
PY

echo "[large-np708] 3/5 package six DLA models"
"$SOURCE_PYTHON" -u server_tools/package_enterprise_dla_codec.py \
  --export-root "$OUTPUT_ROOT" \
  --output-dir "$MODEL_DIR" \
  --archive "$MODEL_ARCHIVE" \
  --package-name "$PACKAGE_NAME" \
  --target "MDLA 5.0 (NeuroPilot SDK 7.0.8; scaled-variance decoder)"

echo "[large-np708] 4/5 generate chained precision inputs"
"$SOURCE_PYTHON" -u server_tools/export_enterprise_precision_vectors.py \
  --model large \
  --source-root "$SOURCE_ROOT" \
  --delivery-manifest "$MODEL_DIR/manifest.json" \
  --output-dir "$PRECISION_DIR" \
  --archive "$PRECISION_ARCHIVE" \
  --qp 0

echo "[large-np708] 5/5 package DLA models with inputs"
"$SOURCE_PYTHON" -u server_tools/package_enterprise_dla_with_inputs.py \
  --model-package "$MODEL_ARCHIVE" \
  --precision-package "$PRECISION_ARCHIVE" \
  --package-name "$FINAL_PACKAGE_NAME" \
  --output "$FINAL_ARCHIVE"

echo "[large-np708] complete"
sha256sum "$MODEL_ARCHIVE" "$PRECISION_ARCHIVE" "$FINAL_ARCHIVE"
echo "delivery=$FINAL_ARCHIVE"
