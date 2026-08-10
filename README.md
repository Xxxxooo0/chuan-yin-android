# GVC-RT Android 部署

本仓库保留两条当前部署路线：

1. **ONNX Demo**：用于图片重建、码流回环、PSNR 和模块演示。
2. **MTK 在线部署**：使用 TFLite 与官方 `NeuronDelegate` 在设备侧在线编译；通过外置模型包选择 GVC-RT Large 或 Small。

后续将增加 **MTK 离线 DLA 部署**，用于企业侧离线集成；当前 Android APK 不使用 `.dla` 运行模型。

项目文档导航见 [docs/README.md](docs/README.md)。部署路线、源码集和应用 ID 见 [docs/deployment-variants.md](docs/deployment-variants.md)；项目目录职责见 [docs/repository-layout.md](docs/repository-layout.md)，全部模型位置与用途见 [models/README.md](models/README.md)。历史 Recon 分段诊断、MNN/GPU 探针和旧 native TFLite 诊断实验已移除，不再作为部署入口。

## 本机构建

```powershell
cd D:\android\ceshi\GVC-RT_clean_android
.\gradlew.bat -PgvcrtSkipAssets :app:assembleOnnxDemoDebug
.\gradlew.bat -PgvcrtSkipAssets :app:assembleMtkOfflineDebug
```

模型推理、精度比对和性能测试不在本机运行：Android 测试通过 `adb` 完成，PyTorch 导出和服务器精度验证通过 `server_tools/` 在远端完成。

## Android 安装

```powershell
$adb = '.\sdk\platform-tools\adb.exe'
& $adb install -r .\app\build\outputs\apk\onnxDemo\debug\app-onnxDemo-debug.apk
& $adb logcat -c
& $adb shell am start -n com.gvcrt.clean.onnxdemo/.MainActivity --ez imageInferenceTest true
& $adb logcat -d -s GVC_RT_CLEAN:I
```

MTK APK 的 application ID 为 `com.gvcrt.clean.mtkoffline`；该 flavor 名称沿用历史目录名，当前实际运行模式为在线编译。服务端导出、离线审计和企业交付命令见 [server_tools/README.md](server_tools/README.md)。
