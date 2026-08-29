# 仓库目录说明

```text
GVC-RT_clean_android/
├─ app/                 Android 应用、JNI 和 flavor 源码
├─ server_tools/        服务器导出、离线编译、精度与打包脚本
├─ docs/                部署流程、算子清单和架构文档；archive/ 存退役诊断文档
├─ models/              Large/Small 模型包（独立 worktree）
├─ model_test/          每次模型测试的独立工作区，整体忽略
├─ local_archive/       本机退役源码与运行资产归档，整体忽略
├─ sdk/                 本地 Android SDK，忽略
├─ mtk/                 本地 MediaTek 工具（NNBenchmark、npu_systrace 等），忽略
├─ outputs/             服务器或本机生成的导出结果，忽略
└─ tmp/、.tmp_*/        临时工作目录，忽略
```

## 部署路线

当前 Android 运行路线为 `app/src/mtkOffline/`：GVC-RT Large 与 Small 共用此 source set、APK 和 Kotlin 运行器，仅通过加载的企业 TFLite 包区分。

`mtkOffline` 是历史 source set 名称，不代表当前运行 `.dla`。

后续路线是 `.dla` 离线部署：服务器生成并审计 DLA 文件后交由企业侧集成，当前 Android APK 不使用这条路线。

MTK 在线路线还提供 1 分钟离线视频演示（`MediaCodec` 解码 + 两遍式 GVC 编码/独立解码 + MP4 重建），入口见 [Large 在线部署](large-online-deployment.md)。

`app/src/main/` 保存 MTK 在线路线的 UI、rANS、基础工具和公共代码；JNI 源码位于 `app/src/main/cpp/`。已退役的 Android ONNX Demo 源码和配置保存在本地 `local_archive/onnx-demo-android/`，不再参与构建。

## 模型与交付

模型统一位于根目录 `models/`。Large 与 Small 分别位于 `models/large/` 和 `models/small/`，二者是只保存模型包的独立 Git worktree。模型实体均被主项目忽略，不会混入主代码分支。具体包结构和当前推荐包见 [../models/README.md](../models/README.md)。

## 模型测试工作区与晋升

候选模型测试按根目录 `AGENTS.md` 执行：每次测试在 `model_test/<日期时间>-<large|small>-<目的>/` 下建立 `candidate/`、`inputs/`、`cache/`、`test-note.md` 结构。测试产物只能进入本测试目录的 `cache/`，不得混入 `outputs/` 或模型目录。

只有满足预先记录的通过标准后，才允许替换 `models/large/` 或 `models/small/` 中的正式包；替换时同步更新该模型目录的 `README.md` 和 `SHA256SUMS.txt`，随后删除整个测试目录并清理设备端临时输出。

## 清理原则

`sdk/`、`mtk/`、`outputs/`、`model_test/`、`tmp/`、模型压缩包、构建产物和转换中间件都是本地生成物。它们不应提交到代码分支；需要长期保留的模型包应只提交到相应的模型专用分支。
