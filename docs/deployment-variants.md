# Deployment Variants

The Android app has two physically separated deployment flavors.

## ONNX Demo

- Source set: `app/src/onnxDemo/`
- Neural runtime: ONNX Runtime
- Models: `app/src/onnxDemo/assets/models/*.onnx`
- Canonical manifest: `app/src/onnxDemo/assets/gvcrt_clean_manifest.json`
- Build: `./gradlew :app:assembleOnnxDemoDebug`
- Application ID: `com.gvcrt.clean.onnxdemo`

This flavor is the stable demonstration path for image reconstruction, PSNR,
bitstream round trips, and module timing. Native rANS remains shared code.

## MTK Offline

- Source set: `app/src/mtkOffline/`
- Neural runtime: MediaTek offline DLA runtime
- Final models: `app/src/mtkOffline/assets/offline_models/*.dla`
- Conversion-only inputs: `app/src/mtkOffline/conversion_inputs/`
- Build: `./gradlew :app:assembleMtkOfflineDebug`
- Application ID: `com.gvcrt.clean.mtkoffline`

TFLite and ONNX conversion diagnostics stay under `conversion_inputs/` and are
not packaged into the APK. The final deployment path must load compiled `.dla`
models and must not fall back to ONNX, TFLite online compilation, NNAPI, or CPU
neural inference.

Common Kotlin, JNI, rANS, baselines, and sample images remain in `app/src/main/`.

## Server Offline Export

Upload the current exporters and run:

```bash
cd /media/ltelab/D/weilingfeng/GVC-RT_clean_android
bash server_tools/run_mtk_offline_export.sh
```

The script publishes only NCC-verified `.dla` files to
`app/src/mtkOffline/assets/offline_models/`. TFLite, TorchScript, logs, and
failed candidates remain under `outputs/` and are not packaged into the APK.
The published manifests deliberately keep `precision_verified=false` until a
separate server tensor comparison has passed.
