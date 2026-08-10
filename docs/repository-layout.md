# 仓库目录说明

```text
GVC-RT_clean_android/
├─ app/                 Android 应用、JNI 和 flavor 源码
├─ server_tools/        服务器导出、离线编译、精度与打包脚本
├─ docs/                部署流程、算子清单和架构文档
├─ models/              ONNX 运行资产与 Large/Small 模型包
├─ sdk/                 本地 Android SDK，忽略
├─ mtk/                 本地 MediaTek 工具，忽略
├─ outputs/             服务器或本机生成的导出结果，忽略
└─ tmp/、.tmp_*/        临时工作目录，忽略
```

## 部署路线

当前可运行的路线有两条：

1. `app/src/onnxDemo/`：ONNX Runtime demo，适合图片重建、码流回环和展示。
2. `app/src/mtkOffline/`：MTK 在线部署。GVC-RT Large 与 Small 共用此 source set、APK 和 Kotlin 运行器，仅通过加载的企业 TFLite 包区分。

`mtkOffline` 是历史 source set 名称，不代表当前运行 `.dla`。

后续路线是 `.dla` 离线部署：服务器生成并审计 DLA 文件后交由企业侧集成，当前 Android APK 不使用这条路线。

`app/src/main/` 是三条当前路线共享的 UI、rANS、基础工具和公共代码。

## 模型与交付

模型统一位于根目录 `models/`。ONNX Demo 运行资产位于 `models/onnx-demo/assets/`；Large 与 Small 分别位于 `models/large/` 和 `models/small/`，二者是只检出模型目录的独立 Git worktree。模型实体均被主项目忽略，不会混入主代码分支。具体包结构和当前推荐包见 [../models/README.md](../models/README.md)。

## 清理原则

`sdk/`、`mtk/`、`outputs/`、`tmp/`、模型压缩包、构建产物和转换中间件都是本地生成物。它们不应提交到代码分支；需要长期保留的模型包应只提交到相应的模型专用分支。
