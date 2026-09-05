# Small 离线视频测试

该入口把普通视频逐帧转换为固定 `256x512`、数值范围 `[0,1]` 的 NHWC FP32
YCbCr444 张量。全序列固定使用 QP index 9，并按源码的参考重置规则执行：

```text
第 0 帧及 frame_index % 64 == 1：上一重建 reference_frame
  -> temporal_from_frame.tflite
其他帧：上一帧 reference_feature
  -> temporal_from_feature.tflite
当前帧 + ctx
  -> encoder.tflite
latent_y + ctx + memory
  -> decoder.tflite
  -> reconstructed_frame + 下一帧 reference_feature
```

四张 TFLite 均使用官方 MediaTek `NeuronDelegate`，配置为 `mtk-neuron`、
`--relax-fp32`、允许 FP16 计算、`FAST_SINGLE_ANSWER`。首个 reference frame 为全
0.5；Decoder 输出的 reference frame 和 reference feature 都会保留。该路径不生成
压缩码流；输出 MP4 仅用于观看重建结果。

## App 操作

安装 `mtkOfflineDebug` APK 并部署 Small 模型包后，点击 `Small video offline` 选择视频。测试期间可点击 `Stop video`。结束后界面显示最后一帧原图、重建图、模型 FPS 和平均 PSNR。

## adb 操作

先把视频放进应用可读目录，再启动测试：

```powershell
adb push input.mp4 /sdcard/Android/data/com.gvcrt.clean.mtkoffline/files/input.mp4
adb logcat -c
adb shell am force-stop com.gvcrt.clean.mtkoffline
adb shell am start -n com.gvcrt.clean.mtkoffline/com.gvcrt.clean.MainActivity `
  --ez smallOfflineVideoTest true `
  --es videoPath /sdcard/Android/data/com.gvcrt.clean.mtkoffline/files/input.mp4 `
  --ei smallOfflineVideoSeconds 60 `
  --ei smallOfflineVideoBitrate 8000000
adb logcat -d -s GVC_RT_CLEAN:I
```

结果位于应用外部文件目录的 `enterprise_tflite_codec/small/video_demo/<时间戳>/`，包括：

- `reconstructed.mp4`
- `input_last.png`
- `reconstructed_last.png`
- `run_report.json`

PSNR 在 H.264 写出之前，将模型 YCbCr444 输入输出转换到 BT.709 RGB 后计算；
模型耗时不包含视频解码、MP4 写入和模型创建。

## PNG 序列与逐图计时

`smallOnlineVideoTest` 的 PNG 序列入口现已对齐上述快速配置（`allowFp16=true`、
`mtk-neuron`、`--relax-fp32`、`FAST_SINGLE_ANSWER`），复用相同的快速配置缓存。
外部张量仍为 NHWC FP32；输入域、参考重置及模型文件不变。

```powershell
adb -s 167412565L102127 shell am start -n com.gvcrt.clean.mtkoffline/com.gvcrt.clean.MainActivity --ez smallOnlineVideoTest true --es sequenceDir /sdcard/Android/data/com.gvcrt.clean.mtkoffline/files/sequences/maintenance_small_fast_20260903 --ei sequenceFrames 96
```

四图的 `small_online_video_speed` 日志记录 mean/P50/P90；当前没有预热，包含首次
invoke，runtime.run 的 I/O 拷贝计入耗时，图片读取、PSNR 与显示不计入。
2026-09-03 在 Infinix X6891 / MT6899 上，先通过同输入设备 PSNR 回归，再独立测速，
得到 `18.463 ms/帧`、`54.161 FPS`，平均 PSNR `25.228 dB`。
命令、模型/APK/输入 SHA、逐图数据和限制见
`model_test/20260903-1633-small-fast/test-note.md`；此结果不替代服务器或 tensor exact 验收。
