# 文档导航

本目录只保存可随源码版本维护的说明与图示；模型、转换产物、设备日志和压缩交付包不放在此处。

## 当前部署与交付

- [部署路线](deployment-variants.md)：ONNX Demo、MTK 在线 Large/Small，以及后续 DLA 交付的边界。
- [仓库目录说明](repository-layout.md)：各目录职责、本地生成物与模型 worktree 的位置。
- [模型索引](../models/README.md)：ONNX 运行资产、Large/Small 包、校验文件及使用原则。
- [服务端导出工具](../server_tools/README.md)：远端导出、离线编译、精度向量和结果审计命令。

## 历史诊断资料

- [I Recon FeatureDec and Generator Inventory](I_RECON_FEATUREDEC_GENERATOR_INVENTORY.md)：已移除的 Recon 分段诊断路线的算子、边界与验证记录，仅供追溯，不作为当前 Android 部署入口。
- [ONNX 版本流程图](ONNX版本流程图.md)：ONNX Demo 的版本流程图；PNG/SVG 均为该图的渲染文件。

## 文档维护规则

1. 根目录 [README](../README.md) 只保留构建、安装和部署入口。
2. 部署路线或交付结构变更时，同步更新本页、`deployment-variants.md` 和 `repository-layout.md`。
3. 所有 README 使用中文；命令、路径、API 名称和标识符可保留原样。
