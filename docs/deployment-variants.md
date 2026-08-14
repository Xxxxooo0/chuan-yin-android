# 部署路线

## 当前路线 1：ONNX Demo

- Source set：`app/src/onnxDemo/`
- 神经网络运行时：ONNX Runtime
- 模型：`models/onnx-demo/assets/models/*.onnx`
- 构建：`./gradlew :app:assembleOnnxDemoDebug`
- Application ID：`com.gvcrt.clean.onnxdemo`

该路线用于图片重建、PSNR、码流回环和模块演示。rANS 使用共享原生实现。

## 当前路线 2：MTK 在线部署（Large / Small）

- Source set：`app/src/mtkOffline/`
- 神经网络运行时：TFLite + MediaTek 官方 `NeuronDelegate`
- 构建：`./gradlew :app:assembleMtkOfflineDebug`
- Application ID：`com.gvcrt.clean.mtkoffline`

Large 与 Small 均使用同一 APK；模型包位于独立的本地模型分支，位置见 [模型索引](../models/README.md)。设备按下表从应用私有目录或 external files 目录读取对应包内的 `models/*.tflite`，再由 `NeuronDelegate` 在线编译。

| 模型变体 | 模型包 | 设备目录 | 测试参数 |
| --- | --- | --- | --- |
| Large | `gvc-rt-large_tflite_online_fixed_qp9_270p.tar.gz`（QP9，含 6 张连续图 + 4 张 entropy+rANS 图） | `enterprise_tflite/large/` | `enterpriseTfliteVariant=large` |
| Small | `gvc-rt-small_tflite_codec_270p_qp0_with_inputs.tar.gz`（QP0，3 张 TFLite + 固定测试输入） | `enterprise_tflite/small/` | `enterpriseTfliteVariant=small` |

也可使用 `enterpriseTfliteVariant=all` 依次检查两个变体。`mtkOffline` 是历史 source set 名称，不表示当前加载 `.dla`。

Large 固定 QP9 包自带 I/P 编码、解码四张 entropy+rANS 合图，设备部署时无需单独推送 entropy 模型；`models/large/online_entropy/` 保留公共图及其 manifest 作为诊断基线。旧的 I 7 张、P 4 张 entropy/prior 分图已弃用。完整打包与测试命令见 [Large 在线部署](large-online-deployment.md)。

该路线另含 1 分钟离线视频演示：系统文件选择器选视频 → `MediaCodec` 解码 → 两遍式 GVC 编码/独立解码 → 输出 `encoded_video.gvc`、`reconstructed.mp4` 与 `run_report.json`，并逐帧校验编码侧/解码侧重建 SHA。

## 后续路线：MTK 离线 DLA

服务器脚本可生成 NCC 验证的 `.dla` 企业交付物：

```bash
cd /media/ltelab/D/weilingfeng/GVC-RT_clean_android
bash server_tools/run_mtk_offline_export.sh
```

也可以使用 `server_tools/compile_tflite_package_to_dla.py` 将已验证的企业 TFLite 包整体编译为 DLA 包，再用 `package_enterprise_video_sequence.py` 打包链式视频序列输入；Large 的 7.0.8 工具链构建入口是 `server_tools/build_large_np708_mdla50_scaled_variance.sh`。该路线是后续企业离线集成方向，当前 Android 应用不加载 `.dla`。转换日志、TFLite、TorchScript 和失败候选均留在 `outputs/`，不打入 APK。
