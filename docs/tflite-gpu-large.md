# Large 标准 TFLite GPU 十图路径

Large GPU 使用独立 `gpularge/` 模型包，不替换 `large/` MTK/NPU 包。显式 `backend=GPU` 时，`LargeOnlineCodecRunner` 从 `enterprise_tflite/gpularge` 创建并持有十张图，完成 I/P encode、payload、独立 decode 和重建闭环。

## 执行边界

六张连续神经网络图完整交给 GPU Delegate：

1. `temporal_from_frame.tflite`
2. `temporal_from_feature.tflite`
3. `i_encoder.tflite`
4. `p_encoder.tflite`
5. `i_decoder.tflite`
6. `p_decoder.tflite`

Infinix X6891 实测六张图均为：

```text
gpu_nodes=1
allowed_builtin_cpu_fallback_nodes=0
unexpected_cpu_nodes=0
```

四张 entropy+rANS 图为：

1. `i_entropy_prior_merged_rans.tflite`
2. `p_entropy_prior_merged_rans.tflite`
3. `i_entropy_decode_merged_rans.tflite`
4. `p_entropy_decode_merged_rans.tflite`

当前 GPU Delegate 未形成 entropy NN 分区，因此这些图的 builtin 使用普通 TFLite CPU kernel；已知 I/P rANS custom op 使用项目现有 native CPU 实现。该路径不启用 NNAPI、XNNPACK 或 MTK fallback。`CompatibilityList=false` 时，显式 GPU 仍进行 Delegate、Interpreter、coverage 和 invoke 探测。

## 模型包

归档位于：

```text
gpu_models/large/gvc-rt-large_tflite_gpu_270p_qp9_ten_graphs.tar.gz
```

包固定为 `256×512`、QP9、NHWC FP32，顶层目录为 `gpularge/`。四张前端图使用 QP9 checkpoint 与 ONNX 标准 DCB/WSiLU 展开；图像输入模型使用数学等价的固定 stride-8 Conv2D PixelUnshuffle。两张 decoder 使用已验证的 GPU 等价改写。MTK 模型和运行路径未改变。

## 部署与运行

```powershell
$adb = '.\sdk\platform-tools\adb.exe'
$deviceRoot = '/sdcard/Android/data/com.gvcrt.clean.mtkoffline/files'
New-Item -ItemType Directory -Force .\tmp\large-gpu-package | Out-Null
tar -xzf .\gpu_models\large\gvc-rt-large_tflite_gpu_270p_qp9_ten_graphs.tar.gz -C .\tmp\large-gpu-package
& $adb push .\tmp\large-gpu-package\gpularge "$deviceRoot/enterprise_tflite/"
& $adb shell am start -n com.gvcrt.clean.mtkoffline/com.gvcrt.clean.MainActivity `
  --ez largeOnlineMainTest true `
  --es backend GPU `
  --es imagePath asset:sample/park_scene_im00001.png `
  --ei largeOnlineQp 9 `
  --ei largeOnlineWarmup 0 `
  --ei largeOnlineMeasured 1
& $adb logcat -d -s GVC_RT_CLEAN:I '*:S'
```

## Infinix X6891 结果

相同 APK、QP9 和输入 SHA256 `7de985fea1b5a1ca0c354d174b865eaedc2afb745b3ed0e62e0aa36ffac21522`：

| 后端 | I 帧 | P 帧 2 | P 帧 3 | 平均 PSNR | 总耗时 | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MTK NPU | 22.537 dB | 22.919 dB | 22.989 dB | 22.815 dB | 724.570 ms | 4.140 |
| GPU 十图 | 22.505 dB | 22.930 dB | 22.978 dB | 22.804 dB | 3795.169 ms | 0.790 |

GPU 与 MTK 的平均质量差为 `-0.011 dB`。六张连续图均有实际 GPU invoke；当前总速度主要受 I/P entropy encode/decode 的 TFLite CPU 执行影响。

将正式归档部署到 `enterprise_tflite/gpularge/` 后，又使用设备序列 `maintenance_large_20260903` 做了三帧 I/P/P 回归：

```powershell
& $adb shell am start `
  -n com.gvcrt.clean.mtkoffline/com.gvcrt.clean.MainActivity `
  --ez largeOnlineVideoTest true `
  --es backend GPU `
  --es sequenceDir "$deviceRoot/sequences/maintenance_large_20260903" `
  --ei sequenceFrames 3 `
  --ei largeOnlineQp 9 `
  --ei largeOnlineWarmup 0 `
  --ei largeOnlineMeasured 1
```

日志确认 `models=10`，六张连续图均为 `gpu_nodes=1`、`allowed_builtin_cpu_fallback_nodes=0`、`unexpected_cpu_nodes=0`，并以 `all_models_exercised=true` 结束。结果为平均 `25.552 dB`、`0.807 FPS`；该输入首帧 SHA256 为 `6bcfc10a8b365b3fe177f1cc14768fe5c107c1fcd525a88517c261ba18175b7a`。

正式归档 SHA256：

```text
52d9753644f9d8aa75eb1aaba9015049e3efbd2d44022c873b405b718f79e7b4
```
