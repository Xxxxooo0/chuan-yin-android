#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANDROID_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SOURCE_ROOT="${1:-/media/ltelab/D/weilingfeng/GVC-RT_inference}"
SDK_ROOT="${2:-/media/ltelab/D/weilingfeng/GVC-RT_clean_android/neuropilot-sdk-premium-8.0.11-build20260211}"
OUTPUT_DIR="${3:-${ANDROID_ROOT}/outputs/recon_hotspots_dla_check}"

SEGMENTS="p_decoder_stage1_conv_only,p_decoder_stage2_blocks_only,p_decoder_stage4_blocks_explicit,p_recon_final_head_no_ada"
LABELS="p_decoder_stage1_conv_only_fp16_weight,p_decoder_stage2_blocks_only_fp16_weight,p_decoder_stage4_blocks_explicit_fp16_weight,p_recon_final_head_no_ada_fp16_weight"

python "${SCRIPT_DIR}/export_recon_diagnostic.py" \
  --source-root "${SOURCE_ROOT}" \
  --android-root "${ANDROID_ROOT}" \
  --sdk-root "${SDK_ROOT}" \
  --segments "${SEGMENTS}" \
  --variant fp16_weight \
  --output-dir "${OUTPUT_DIR}" \
  --copy-assets \
  --opt-bw \
  --relax-fp32

python "${SCRIPT_DIR}/analyze_recon_neuron_support.py" \
  --android-root "${ANDROID_ROOT}" \
  --assets-dir "${ANDROID_ROOT}/app/src/main/assets/recon_diagnostic" \
  --output-dir "${OUTPUT_DIR}/ncc_check" \
  --ncc-tflite "${SDK_ROOT}" \
  --labels "${LABELS}" \
  --opt-bw \
  --relax-fp32 \
  --compile-dla
