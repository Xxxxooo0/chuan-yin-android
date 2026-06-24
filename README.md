# GVC-RT Clean Android Deployment

This project is a clean, source-derived deployment scaffold for the three
GVC-RT modules:

1. Temporal reference: previous reference frame/feature to `ctx` and `ctx_t`.
2. Complete encoding: I encode, local I reconstruction, P temporal reference,
   P encode, and P reference update.
3. Complete decoding: v1 covers the post-entropy neural decode path from
   server `y_hat` tensors to reconstructed reference outputs.

The source of truth is `D:\sever_chuanyin\GVC-RT_inference`. Previous Android
ONNX files, exporters, benchmark code, and diagnostic probes are intentionally
not reused.

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
$adb = 'D:\android\ceshi\GVC-RT_inference\Android\sdk\platform-tools\adb.exe'
& $adb install -r .\app\build\outputs\apk\debug\app-debug.apk
& $adb logcat -c
& $adb shell am force-stop com.gvcrt.clean
& $adb shell am start -n com.gvcrt.clean/.MainActivity --ez temporalReferenceTest true
& $adb logcat -d -s GVC_RT_CLEAN:I
```

Use `--ez completeEncoderTest true` or `--ez completeDecoderTest true` for the
other two module entries.

## Current v1 Boundary

The complete encoder runs ONNX graph sequences, source-derived CDF-index
packing, native rANS, and SPS/I/P muxing. It compares I/P payloads and the
final `encoded_ip.gvc` byte-for-byte with the server baseline. The complete
decoder remains a post-entropy neural decode test until its rANS-to-prior path
is connected.
