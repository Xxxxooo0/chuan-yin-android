# 文档导航

本目录只保存可随源码版本维护的说明与图示；模型、转换产物、设备日志和压缩交付包不放在此处。已退役的历史诊断文档移入 `archive/`。

## 当前部署与交付

- [部署路线](deployment-variants.md)：MTK 在线 Large/Small 与后续 DLA 交付的边界。
- [GVC-RT Large 在线部署](large-online-deployment.md)：固定 QP9 打包、entropy+rANS 合图、Android 测试命令、1 分钟离线视频演示。
- [仓库目录说明](repository-layout.md)：各目录职责、模型测试工作区与本地生成物的位置。
- [模型索引](../models/README.md)：Large/Small 包、校验文件及使用原则。
- [服务端导出工具](../server_tools/README.md)：远端导出、离线编译、精度向量、视频序列与结果审计命令。

模型测试工作区（`model_test/`）的结构与模型晋升流程见仓库根目录 [AGENTS.md](../AGENTS.md)。

## 历史诊断资料

- [I Recon FeatureDec and Generator Inventory](archive/I_RECON_FEATUREDEC_GENERATOR_INVENTORY.md)：已移除的 Recon 分段诊断路线的算子、边界与验证记录，仅供追溯，不作为当前 Android 部署入口。

## 文档维护规则

1. 根目录 [README](../README.md) 只保留构建、安装和部署入口。
2. 部署路线或交付结构变更时，同步更新本页、`deployment-variants.md`、`repository-layout.md` 和 [模型索引](../models/README.md)。
3. 退役文档移入 `archive/` 并保留原始文件名，本页与相关文档的引用同步改为新路径。
4. 所有 README 使用中文；命令、路径、API 名称和标识符可保留原样。
