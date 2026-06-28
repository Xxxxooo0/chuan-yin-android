# GVC-RT Clean Android Deployment

This project is a clean, source-derived deployment scaffold for the three
GVC-RT modules:

1. Temporal reference: previous reference frame/feature to `ctx` and `ctx_t`.
2. Complete encoding: I encode, local I reconstruction, P temporal reference,
   P encode, and P reference update.
3. Complete decoding: parses I/P NAL payloads, decodes `z` and each spatial
   prior stage through native rANS, then reconstructs and updates references.

This Android project is self-contained for build and on-device runs: Gradle wrapper, Android SDK, ONNX assets, entropy tables, baselines, and sample images live under this project directory. The PyTorch source tree is only required when re-exporting assets; pass it with `--source-root` or `GVC_RT_SOURCE_ROOT`. Previous Android ONNX files, exporters, benchmark code, and diagnostic probes are intentionally not reused.

## Export On Server

Run `server_tools/export_clean_gvcrt_modules.py` on the server/PyTorch
environment. It imports `DMCI` and `DMC` directly from `src`, loads
`GVC-RT_B_I.pt` and `GVC-RT_B_P.pt`, exports ONNX subgraphs, and writes the
Android comparison baseline.

The clean v1 correctness baseline is FP32-only. Every generated baseline tensor
and ONNX graph uses FP32, so Android results are not compared against a
different-precision server path. FP16 deployment is deferred until it has its
own source-matched export and baseline.

## Build Locally

```powershell
cd D:\android\ceshi\GVC-RT_clean_android
.\gradlew.bat :app:assembleDebug
```

## Run On Android

```powershell
$adb = 'D:\android\ceshi\GVC-RT_clean_android\sdk\platform-tools\adb.exe'
& $adb install -r .\app\build\outputs\apk\debug\app-debug.apk
& $adb logcat -c
& $adb shell am force-stop com.gvcrt.clean
& $adb shell am start -n com.gvcrt.clean/.MainActivity --ez temporalReferenceTest true
& $adb logcat -d -s GVC_RT_CLEAN:I
```

Use `--ez completeEncoderTest true` or `--ez completeDecoderTest true` for the
other two module entries. Run the encoder first when validating Android's own
bitstream: the decoder then reads `outputs/encoded_ip.gvc`; without it, the
decoder explicitly falls back to the packaged server stream baseline.

Use `--ez fullProjectTest true` to run the clean project entry. It executes the
three modules in order: temporal reference, complete encoder, then complete
decoder. The decoder consumes the Android-generated `outputs/encoded_ip.gvc`
written by the encoder in the same run.

Use `--ez imageInferenceTest true` for one-image inference. Without an explicit
image path it uses `assets/sample/sample_input.png`; with `--es imagePath ...`
it decodes that PNG/JPEG path on the device, resizes to `256x512`, normalizes to
`[-1,1]`, runs encode/decode, and prints a short quality and timing summary.

```powershell
& $adb shell am start -n com.gvcrt.clean/.MainActivity --ez imageInferenceTest true
& $adb push .\your_image.png /sdcard/Download/gvcrt_input.png
& $adb shell am start -n com.gvcrt.clean/.MainActivity --ez imageInferenceTest true --es imagePath /sdcard/Download/gvcrt_input.png
```

## Speed Tests

The second button row runs device-only speed benchmarks. Each benchmark keeps
the ONNX sessions and CDF tables warm, performs 5 warmup runs, and reports 50
measured samples as mean, p50, p90, and fps. Session initialization, baseline
comparison, and output-file writes are excluded from measured samples.
Speed tests also print process RAM diagnostics as `memory_start`,
`memory_mark`, `memory_peak`, and `memory_end`. These include Android PSS,
Java/native heap, RSS/HWM, and system available memory.

```powershell
& $adb shell am start -n com.gvcrt.clean/.MainActivity --ez temporalReferenceSpeedTest true
& $adb shell am start -n com.gvcrt.clean/.MainActivity --ez completeEncoderSpeedTest true
& $adb shell am start -n com.gvcrt.clean/.MainActivity --ez completeDecoderSpeedTest true
```

## RAM And GPU Diagnostics

The app can reliably report its own RAM usage, but Android does not expose a
stable per-app GPU/NPU memory API to ordinary apps. GPU memory in app logs is
therefore reported as unavailable unless the device exposes it through system
dumpsys/procfs commands.

Use these adb commands after or during a benchmark run for best-effort external
diagnostics:

```powershell
& $adb shell dumpsys meminfo com.gvcrt.clean
& $adb shell dumpsys gfxinfo com.gvcrt.clean
& $adb shell dumpsys SurfaceFlinger
& $adb shell cat /proc/meminfo
& $adb shell cat /sys/kernel/dmabuf/buffers 2>/dev/null
```

On some devices, `dumpsys meminfo` contains `Graphics`, `GL`, `EGL mtrack`, or
dmabuf-related rows. If those fields are absent or permission denied, treat GPU
memory as `unavailable_on_this_device`; do not infer GPU/NPU usage only from
NNAPI being enabled.

## Current v1 Boundary

The complete encoder runs ONNX graph sequences, source-derived CDF-index
packing, native rANS, and SPS/I/P muxing. It compares I/P payloads and the
final `encoded_ip.gvc` byte-for-byte with the server baseline. The complete
decoder parses that stream, incrementally decodes I's four and P's two prior
stages from Android-generated CDF indexes, and compares decoded symbols,
latents, reconstructed frames, and reference state against the server baseline.
