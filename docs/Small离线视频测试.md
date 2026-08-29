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
