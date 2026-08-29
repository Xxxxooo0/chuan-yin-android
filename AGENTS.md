# 仓库规范

## 项目结构与模块组织

本仓库是用于 GVC-RT 推理与诊断的精简 Android 部署项目。Android 应用位于 `app/`。

- `app/src/main/java/com/gvcrt/clean/`：Kotlin 应用逻辑、TFLite/Neuron 运行器、模块测试、速度基准测试、内存采样和图像推理。
- `app/src/main/cpp/`：原生 C++/JNI 代码、rANS 集成、MTK TFLite 桥接和融合算子实验。
- `app/src/main/assets/`：打包进 APK 的运行时资源。除非明确需要，`models/` 和 `baseline/` 下的大型基线及模型资源应保持忽略状态。
- `app/src/main/jniLibs/`：Android 运行时需要的原生动态库。
- `server_tools/`：服务端导出、验证和诊断脚本。这些脚本用于远程 Linux 服务器环境，不用于本地 PC 推理。
- `docs/`：项目说明与辅助文档。

## 构建、测试与开发命令

在仓库根目录运行以下命令：

```powershell
.\gradlew.bat :app:compileDebugKotlin
```

编译 Kotlin，用于快速发现大部分 Android 源码错误。

```powershell
.\gradlew.bat :app:assembleDebug
```

构建调试版 APK，包括 CMake/JNI 代码和打包资源。

```powershell
.\sdk\platform-tools\adb.exe install -r .\app\build\outputs\apk\debug\app-debug.apk
```

将 APK 安装到已连接的 Android 设备。

## 编码风格与命名规范

应用编排使用 Kotlin，JNI 和原生算子使用 C++17。修改范围应限定在当前模块内。Kotlin 类名使用 `PascalCase`，函数和变量使用 `camelCase`；基准测试标签应采用稳定的蛇形命名，例如 `native_p_recon_stage_precision_probe`。日志中应明确记录张量 shape 和资源路径。

本仓库中的所有 `README.md` 必须使用中文编写。命令、路径、API 名称和标识符可以保留原文。

## 测试规范

本工作站只用于编辑和构建，不用于模型推理。禁止在本地 PC 上运行 PyTorch、ONNX Runtime、TFLite、精度或性能推理测试。应使用：

- Android 设备和 `adb` 执行 APK 测试、速度测试、内存日志采集及 Android 输出导出。
- 远程服务器上的 `server_tools/` 脚本执行 PyTorch、模型导出和精度验证。

比较模型输出时，必须记录实际使用的完整 `adb` 参数或服务器命令，并包含关键文件的哈希值。

## 模型测试工作区规范

每次测试新模型时，必须在仓库根目录的 `model_test/` 下建立独立测试目录，命名为 `<日期时间>-<large|small>-<简短目的>`，例如 `model_test/20260814-1530-large-decoder-qp9/`。禁止把候选模型、测试输入、日志、张量 dump、临时 APK 或其他中间文件直接放到仓库根目录、`models/`、`app/src/main/assets/` 或正式 `outputs/` 中。

每个测试目录使用以下固定结构：

```text
model_test/<test-id>/
├─ candidate/    待测模型包、清单和 SHA256
├─ inputs/       本次测试专用输入
├─ cache/        adb 拉取结果、日志、张量 dump、截图、APK 和中间文件
└─ test-note.md  测试边界、命令、设备、后端、验收标准和结论
```

`model_test/` 整体由 Git 忽略。正式流程输出继续放在 `outputs/`；模型测试产生的全部输出只能进入本次测试目录的 `cache/`，不得与正式输出混放。

模型测试与晋升必须遵守以下流程：

1. 测试前在 `test-note.md` 中记录 Large/Small 变体、模型边界、输入输出 shape、layout、dtype、QP、checkpoint SHA256、模型 SHA256、设备、后端和通过标准。
2. 在远程服务器执行源码参考、导出和精度验证；在 Android 设备上通过 `adb` 和目标运行时验证模型能够创建并完成 invoke。先验证精度，再测试速度；仅转换成功或 NCC 达标不能视为测试通过。
3. 测试未通过时，不得修改 `models/large/` 或 `models/small/` 中的正式包。未通过的测试可以暂留用于定位，但必须继续隔离在自己的测试目录内；确认放弃后删除整个测试目录。
4. 只有满足预先记录的通过标准后，才允许用候选包替换 `models/large/` 或 `models/small/` 中对应的正式包。替换时只删除被取代的包，并同步更新该模型目录的 `README.md`、`SHA256SUMS` 或等效清单；需由 Git LFS 管理的大模型文件必须保持正确跟踪。
5. 删除测试目录前，把最终使用的命令、设备、后端、关键 SHA256、精度和速度结论写入正式模型包的说明或相关项目文档，并重新校验正式包 checksum，执行与改动相称的 Android 构建。
6. 正式包替换和验证完成后，删除整个 `model_test/<test-id>/`，包括 `cache/` 中的所有测试输出，并清理设备端与该测试对应的临时输出。不得在仓库根目录或正式 `outputs/` 中遗留模型测试文件。

## 提交与拉取请求规范

提交历史使用简短的祈使句，例如 `Add recon diagnostics and MTK TFLite runtime` 和 `Optimize native rANS encoding path`。每个提交应聚焦单一目的，禁止混入无关实验。提交前运行 `.\gradlew.bat :app:assembleDebug`，并使用 `git status --short` 检查 SDK、MTK 工具、构建产物和缓存文件是否已正确忽略。

PR 或工作交接说明必须包含：改动目的、实际执行的 Android/服务器命令、使用的设备和后端、重要精度或速度结果，以及已知的回退行为或不支持的算子。

## 安全与配置注意事项

禁止提交本地 SDK、服务器凭据、生成的压缩包、构建产物或下载的 MTK 工具归档。`sdk/`、`mtk/`、`outputs/`、`app/build/` 和 `local.properties` 等路径必须保持在 Git 忽略范围内。
