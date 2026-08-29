# GVC-RT Android 部署

本仓库的 Android 应用只保留 **MTK 在线部署**：使用 TFLite 与官方 `NeuronDelegate` 在设备侧在线编译，通过外置模型包选择 GVC-RT Large 或 Small。

Android ONNX Demo 已从活动项目移除，本地归档位于 `local_archive/onnx-demo-android/`；服务器端为 TFLite/DLA 转换保留的 ONNX 导出能力仍位于 `server_tools/`。

后续将增加 **MTK 离线 DLA 部署**，用于企业侧离线集成；当前 Android APK 不使用 `.dla` 运行模型。

项目文档导航见 [docs/README.md](docs/README.md)。部署路线、源码集和应用 ID 见 [docs/deployment-variants.md](docs/deployment-variants.md)；项目目录职责见 [docs/repository-layout.md](docs/repository-layout.md)，全部模型位置与用途见 [models/README.md](models/README.md)。历史 Recon 分段诊断、MNN/GPU 探针和旧 native TFLite 诊断实验已移除，不再作为部署入口。

Large 在线 I 帧合并熵模型的独立部署、测速和精度 dump 命令见 [GVC-RT Large 在线部署](docs/large-online-deployment.md)。

## 本机构建

```powershell
cd D:\android\ceshi\GVC-RT_clean_android
.\gradlew.bat -PgvcrtSkipAssets :app:assembleMtkOfflineDebug
```

模型推理、精度比对和性能测试不在本机运行：Android 测试通过 `adb` 完成，PyTorch 导出和服务器精度验证通过 `server_tools/` 在远端完成。

## 模型测试工作区

新模型或新包必须先在 `model_test/<日期时间>-<large|small>-<目的>/` 下按 [AGENTS.md](AGENTS.md) 的规范完成精度与速度验证，通过后才允许替换 `models/large/`、`models/small/` 中的正式包，并同步更新对应模型目录的 `README.md` 与 `SHA256SUMS.txt`；测试目录随后整体删除。

## Android 安装

```powershell
$adb = '.\sdk\platform-tools\adb.exe'
& $adb install -r .\app\build\outputs\apk\mtkOffline\debug\app-mtkOffline-debug.apk
& $adb shell am start -n com.gvcrt.clean.mtkoffline/com.gvcrt.clean.MainActivity --ez enterpriseTfliteTest true --es enterpriseTfliteVariant large
& $adb logcat -d -s GVC_RT_CLEAN:I
```

MTK APK 的 application ID 为 `com.gvcrt.clean.mtkoffline`；该 flavor 名称沿用历史目录名，当前实际运行模式为在线编译。全部 adb 测试入口（含 Large 主流程与 1 分钟离线视频演示）见 [docs/large-online-deployment.md](docs/large-online-deployment.md)，服务端导出、离线审计和企业交付命令见 [server_tools/README.md](server_tools/README.md)。
