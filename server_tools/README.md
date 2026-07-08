# Clean GVC-RT Export Tools

Run these scripts only on the server/PyTorch environment. The local PC is used
for editing, APK build, and adb installation only.

Clean v1 is intentionally FP32-only. The earlier FP16 export converted several
graphs to FP32 after recording an FP16 baseline, making the comparison invalid.
Use the command below exactly; FP16 is deferred until every graph has a
source-matched FP16 export path.

```bash
cd /media/ltelab/D/weilingfeng/GVC-RT_clean_android
export GVC_RT_SOURCE_ROOT=/media/ltelab/D/weilingfeng/GVC-RT_inference
python server_tools/export_clean_gvcrt_modules.py \
  --height 256 \
  --width 512 \
  --qp 0 \
  --precision fp32
```

The generated assets are:

- `models/*.onnx`
- `baseline/inputs/*.f32le`
- `baseline/tensors/*.f32le`
- `baseline/entropy/*_cdf*.i32le`, packed symbols, and z symbols
- `baseline/bitstream/i_rans_payload.bin`, `p_rans_payload.bin`, and `encoded_ip.gvc`
- `gvcrt_clean_manifest.json`

Copy the generated `app/src/main/assets` directory back into the Android clean
project before building the APK.

## Recon SpaceToDepth Check

Use this to test whether the P recon front can use a real `SpaceToDepth` op
instead of the failing 6D `Reshape -> Transpose -> Reshape` export or the slow
fixed-conv replacement.

```bash
cd /media/ltelab/D/weilingfeng/GVC-RT_clean_android
export GVC_RT_SOURCE_ROOT=/media/ltelab/D/weilingfeng/GVC-RT_inference

python server_tools/export_recon_diagnostic.py \
  --source-root "$GVC_RT_SOURCE_ROOT" \
  --sdk-root /media/ltelab/D/weilingfeng/GVC-RT_clean_android/neuropilot-sdk-premium-8.0.11-build20260211 \
  --segments p_recon_unshuffle_spacetodepth_only,p_recon_feature_to_codeword_spacetodepth \
  --variant fp32 \
  --copy-assets

python server_tools/analyze_recon_neuron_support.py \
  --assets-dir app/src/main/assets/recon_diagnostic \
  --arch mdla5.3 \
  --labels p_recon_unshuffle_spacetodepth_only_fp32,p_recon_feature_to_codeword_spacetodepth_fp32 \
  --ncc-tflite /media/ltelab/D/weilingfeng/GVC-RT_clean_android/neuropilot-sdk-premium-8.0.11-build20260211
```

If the MTK ONNX converter rejects ONNX `SpaceToDepth`, generate native TFLite
`SPACE_TO_DEPTH` probes directly with TensorFlow:

```bash
python server_tools/export_direct_spacetodepth_tflite.py \
  --output-dir app/src/main/assets/recon_diagnostic \
  --targets nhwc_fp16,nchw_wrap_fp16

python server_tools/analyze_recon_neuron_support.py \
  --assets-dir app/src/main/assets/recon_diagnostic \
  --arch mdla5.3 \
  --labels p_recon_unshuffle_tflite_spacetodepth_nhwc_fp16,p_recon_unshuffle_tflite_spacetodepth_nchw_wrap_fp16 \
  --ncc-tflite /media/ltelab/D/weilingfeng/GVC-RT_clean_android/neuropilot-sdk-premium-8.0.11-build20260211
```

To test the PyTorch NCHW pattern recommended by the MediaTek converter
documentation:

```bash
python server_tools/export_recon_diagnostic.py \
  --source-root "$GVC_RT_SOURCE_ROOT" \
  --sdk-root /media/ltelab/D/weilingfeng/GVC-RT_clean_android/neuropilot-sdk-premium-8.0.11-build20260211 \
  --segments p_recon_unshuffle_mtk_nchw_only,p_recon_feature_to_codeword_mtk_nchw \
  --variant fp32 \
  --copy-assets

python server_tools/analyze_recon_neuron_support.py \
  --assets-dir app/src/main/assets/recon_diagnostic \
  --arch mdla5.3 \
  --labels p_recon_unshuffle_mtk_nchw_only_fp32,p_recon_feature_to_codeword_mtk_nchw_fp32 \
  --ncc-tflite /media/ltelab/D/weilingfeng/GVC-RT_clean_android/neuropilot-sdk-premium-8.0.11-build20260211
```

## Recon MLP Split Check

After replacing P recon `pixel_unshuffle(2)` with the native path, use this to
find whether the remaining MLP cost comes from `GroupNorm`, `DepthConvBlock`, or
the SiLU gate.

```bash
cd /media/ltelab/D/weilingfeng/GVC-RT_clean_android
export GVC_RT_SOURCE_ROOT=/media/ltelab/D/weilingfeng/GVC-RT_inference

python server_tools/export_recon_diagnostic.py \
  --source-root "$GVC_RT_SOURCE_ROOT" \
  --sdk-root /media/ltelab/D/weilingfeng/GVC-RT_clean_android/neuropilot-sdk-premium-8.0.11-build20260211 \
  --segments p_recon_mlp_norm0,p_recon_mlp_dcb0,p_recon_mlp_norm1,p_recon_mlp_silu,p_recon_mlp_dcb1,p_recon_mlp_norm0_dcb0,p_recon_mlp_norm1_silu_dcb1 \
  --variant fp32 \
  --copy-assets

python server_tools/analyze_recon_neuron_support.py \
  --assets-dir app/src/main/assets/recon_diagnostic \
  --arch mdla5.3 \
  --labels p_recon_mlp_norm0_fp32,p_recon_mlp_dcb0_fp32,p_recon_mlp_norm1_fp32,p_recon_mlp_silu_fp32,p_recon_mlp_dcb1_fp32,p_recon_mlp_norm0_dcb0_fp32,p_recon_mlp_norm1_silu_dcb1_fp32 \
  --ncc-tflite /media/ltelab/D/weilingfeng/GVC-RT_clean_android/neuropilot-sdk-premium-8.0.11-build20260211
```

To split the two `DepthConvBlock` modules inside that MLP:

```bash
python server_tools/export_recon_diagnostic.py \
  --source-root "$GVC_RT_SOURCE_ROOT" \
  --sdk-root /media/ltelab/D/weilingfeng/GVC-RT_clean_android/neuropilot-sdk-premium-8.0.11-build20260211 \
  --segments p_recon_mlp_dcb0_adaptor,p_recon_mlp_dcb0_dc,p_recon_mlp_dcb0_dc_add,p_recon_mlp_dcb0_ffn,p_recon_mlp_dcb0_ffn_add,p_recon_mlp_dcb1_adaptor,p_recon_mlp_dcb1_dc,p_recon_mlp_dcb1_dc_add,p_recon_mlp_dcb1_ffn,p_recon_mlp_dcb1_ffn_add \
  --variant fp32 \
  --copy-assets

python server_tools/analyze_recon_neuron_support.py \
  --assets-dir app/src/main/assets/recon_diagnostic \
  --arch mdla5.3 \
  --labels p_recon_mlp_dcb0_adaptor_fp32,p_recon_mlp_dcb0_dc_fp32,p_recon_mlp_dcb0_dc_add_fp32,p_recon_mlp_dcb0_ffn_fp32,p_recon_mlp_dcb0_ffn_add_fp32,p_recon_mlp_dcb1_adaptor_fp32,p_recon_mlp_dcb1_dc_fp32,p_recon_mlp_dcb1_dc_add_fp32,p_recon_mlp_dcb1_ffn_fp32,p_recon_mlp_dcb1_ffn_add_fp32 \
  --ncc-tflite /media/ltelab/D/weilingfeng/GVC-RT_clean_android/neuropilot-sdk-premium-8.0.11-build20260211
```

## Recon MNN Candidates

This route reloads the P model directly from the server source tree and exports
fresh ONNX candidates before converting them to `.mnn`. It does not read old
Android ONNX/TFLite assets.

```bash
cd /media/ltelab/D/weilingfeng/GVC-RT_clean_android
export GVC_RT_SOURCE_ROOT=/media/ltelab/D/weilingfeng/GVC-RT_inference

/media/ltelab/D/weilingfeng/conda_envs/weilingfeng/bin/python \
  server_tools/export_recon_mnn_candidates.py \
  --source-root "$GVC_RT_SOURCE_ROOT" \
  --output-dir /media/ltelab/D/weilingfeng/GVC-RT_clean_android/outputs/recon_mnn_candidates \
  --segments p_recon_mlp_only,p_recon_stage1_stage2,p_recon_upsample_stage3,p_recon_stage4_final,p_recon_back_half \
  --opset 12 \
  --convert-mnn \
  --mnn-fp16 \
  --copy-assets
```

If `MNNConvert` is not in `PATH`, pass it explicitly:

```bash
  --mnn-convert /path/to/MNNConvert
```

After rebuilding and installing the APK, probe one candidate on Android:

```bash
adb shell am start -n com.gvcrt.clean/.MainActivity \
  --ez reconMnnDiagnosticTest true \
  --es reconMnnLabel p_recon_upsample_stage3 \
  --ei reconMnnWarmup 5 \
  --ei reconMnnMeasured 50
```

The current Android entry verifies `.mnn` assets and probes whether an MNN
runtime is linked. Real CPU/OpenCL/Vulkan timing requires adding the MNN Android
AAR/native libraries first.
