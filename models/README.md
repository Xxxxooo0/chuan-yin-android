# 模型索引

本目录集中管理本机部署使用的全部模型。主项目只保留这一份 Android 与导出代码，模型二进制不进入主代码分支。

```text
models/
├─ large/       GVC-RT Large 模型分支 worktree
└─ small/       GVC-RT Small 模型分支 worktree
```

## GVC-RT Large

- 模型分支：`GVC-RT-L-models`
- Worktree：`models/large/`
- 包目录：`models/large/`
- 当前推荐包：`gvc-rt-large_tflite_online_fixed_qp9_270p.tar.gz`
- 熵公共图：`models/large/online_entropy/`（四张 merged+rANS 图与 `manifest.json`）

固定 QP9 包包含六张连续神经网络图及 I/P 编解码四张 entropy+rANS 图，不需要
`quant_scales/`。`online_entropy/` 同时作为公共图备份与诊断基线，其中四张
merged+rANS 图与固定 QP9 包内容一致。旧在线包及旧的 7 张 I 与 4 张 P entropy/prior
分图不再使用。

## GVC-RT Small

- 模型分支：`GVC-RT-S-models`
- Worktree：`models/small/`
- 包目录：`models/small/`
- 当前企业流程：四模型 QP9（`temporal_from_frame`、`temporal_from_feature`、`encoder`、`decoder`），打包见 `server_tools/compile_tflite_package_to_dla.py` 与 `package_enterprise_video_sequence.py`，二者要求恰好这四张模型且 `fixed_q_index=9`
- 本地旧包：`gvc-rt-small_tflite_codec_270p_qp0_with_inputs.tar.gz`（三模型 QP0），新四模型包生成并通过 `model_test/` 验证后按晋升流程替换

## 使用原则

1. Large/Small 是只保存模型包的专用 worktree；Android、文档和导出代码统一使用主项目版本。
2. 交付或部署前根据包目录内的 `SHA256SUMS.txt` 校验完整性。Large 的清单还覆盖 `online_entropy/` 全部文件与目录 README；Small 的清单覆盖三个交付包。
3. Android 设备侧将解压后的包分别放入 `enterprise_tflite/large/` 或 `enterprise_tflite/small/`。
4. 模型输入固定为 `256x512`；Large 在线主线固定 QP9，其他模型的 QP、布局、精度和后端要求以包内 README 与 manifest 为准。
5. 候选模型必须先在 `model_test/<test-id>/` 按根目录 `AGENTS.md` 的流程完成精度与速度验证，通过后才允许替换正式包；替换时同步更新对应模型目录的 `README.md` 与 `SHA256SUMS.txt`，随后删除整个测试目录。
