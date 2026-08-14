# GVC-RT Small 本地模型快照

本目录仅提交到本地 `GVC-RT-S-models` 分支，不要将该分支推送到 Gitee。

当前 Small 企业交付流程为**四模型 QP9**：`temporal_from_frame`、`temporal_from_feature`、`encoder`、`decoder`（原 `temporal_reference` 已拆分为前两张图）。打包入口见主项目 `server_tools/compile_tflite_package_to_dla.py` 与 `package_enterprise_video_sequence.py`，二者均要求恰好这四张模型且 `fixed_q_index=9`。

本目录当前仍保存旧代际的三模型 QP0 包；新四模型包生成并通过 `model_test/` 验证后，按晋升流程替换：

- `gvc-rt-small_tflite_codec_270p_qp0_with_inputs.tar.gz`：旧三模型（temporal_reference、encoder、decoder）TFLite 及固定测试输入。
- `gvc-rt-small_dla_codec_270p_qp0_with_inputs.tar.gz`：旧三模型 DLA 及固定测试输入。
- `gvc-rt-small_precision_270p_qp0.tar.gz`：旧三模型服务器精度参考张量。

完整性校验见本目录 `SHA256SUMS.txt`；交付或部署前按清单核对。

正式包替换必须先在主项目 `model_test/<test-id>/` 按 `AGENTS.md` 流程完成精度与速度验证，通过后才允许替换；替换时同步更新本 README 与 `SHA256SUMS.txt`，随后删除整个测试目录。
