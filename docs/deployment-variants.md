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

| 模型变体 | 交付包目录 | 设备目录 | 测试参数 |
| --- | --- | --- | --- |
| Large | `models/large/local_models/gvc-rt-large/` | `enterprise_tflite/large/` | `enterpriseTfliteVariant=large` |
| Small | `models/small/local_models/gvc-rt-small/` | `enterprise_tflite/small/` | `enterpriseTfliteVariant=small` |

也可使用 `enterpriseTfliteVariant=all` 依次检查两个变体。`mtkOffline` 是历史 source set 名称，不表示当前加载 `.dla`。

## 后续路线：MTK 离线 DLA

服务器脚本可生成 NCC 验证的 `.dla` 企业交付物：

```bash
cd /media/ltelab/D/weilingfeng/GVC-RT_clean_android
bash server_tools/run_mtk_offline_export.sh
```

该路线是后续企业离线集成方向，当前 Android 应用不加载 `.dla`。转换日志、TFLite、TorchScript 和失败候选均留在 `outputs/`，不打入 APK。
