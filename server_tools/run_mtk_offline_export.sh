#!/usr/bin/env bash
set -euo pipefail

ANDROID_ROOT="${ANDROID_ROOT:-/media/ltelab/D/weilingfeng/GVC-RT_clean_android}"
SOURCE_ROOT="${GVC_RT_SOURCE_ROOT:-/media/ltelab/D/weilingfeng/GVC-RT_inference}"
PYTHON="${PYTHON:-/media/ltelab/D/weilingfeng/conda_envs/weilingfeng/bin/python}"
SDK_ROOT="${NEUROPILOT_SDK_ROOT:-${ANDROID_ROOT}/neuropilot-sdk-premium-8.0.11-build20260211}"
NCC="${NCC_TFLITE:-${SDK_ROOT}/neuron_sdk/host/bin/ncc-tflite}"

export LD_LIBRARY_PATH="${SDK_ROOT}/neuron_sdk/host/lib:${LD_LIBRARY_PATH:-}"

cd "${ANDROID_ROOT}"

echo "[offline-export] temporal reference and encoder analysis"
"${PYTHON}" -u server_tools/export_three_modules_offline_nhwc.py \
  --source-root "${SOURCE_ROOT}" \
  --android-root "${ANDROID_ROOT}" \
  --output-dir "${ANDROID_ROOT}/outputs/mtk_offline_front" \
  --ncc-tflite "${NCC}" \
  --candidates temporal_from_frame_big,temporal_from_feature_big,i_encoder_analysis_big,p_encoder_analysis_big \
  --copy-offline-assets

echo "[offline-export] I/P entropy neural graphs"
"${PYTHON}" -u server_tools/export_entropy_offline_nhwc.py \
  --source-root "${SOURCE_ROOT}" \
  --android-root "${ANDROID_ROOT}" \
  --output-dir "${ANDROID_ROOT}/outputs/mtk_offline_entropy" \
  --ncc-tflite "${NCC}" \
  --copy-offline-assets

echo "[offline-export] high-resolution normalization probes"
"${PYTHON}" -u server_tools/export_decoder_full_norm_rewrite_nhwc.py \
  --source-root "${SOURCE_ROOT}" \
  --android-root "${ANDROID_ROOT}" \
  --output-dir "${ANDROID_ROOT}/outputs/mtk_offline_decoder_norm_probes" \
  --ncc-tflite "${NCC}" \
  --targets groupnorm_highres_probe,adagn_highres_probe

echo "[offline-export] I/P decoder segments"
"${PYTHON}" -u server_tools/export_decoder_full_norm_rewrite_nhwc.py \
  --source-root "${SOURCE_ROOT}" \
  --android-root "${ANDROID_ROOT}" \
  --output-dir "${ANDROID_ROOT}/outputs/mtk_offline_decoder_segments" \
  --ncc-tflite "${NCC}" \
  --targets i_segments,p_segments \
  --copy-offline-assets

echo "[offline-export] merged I/P decoder synthesis"
"${PYTHON}" -u server_tools/export_decoder_full_norm_rewrite_nhwc.py \
  --source-root "${SOURCE_ROOT}" \
  --android-root "${ANDROID_ROOT}" \
  --output-dir "${ANDROID_ROOT}/outputs/mtk_offline_decoder_merged" \
  --ncc-tflite "${NCC}" \
  --targets i,p \
  --copy-merged-assets

echo "[offline-export] published assets"
find "${ANDROID_ROOT}/app/src/mtkOffline/assets/offline_models" \
  -maxdepth 1 -type f \( -name '*.dla' -o -name '*manifest.json' \) \
  -printf '%f %s bytes\n' | sort
