# GVC-RT Small 本地模型快照

本目录仅提交到本地 `GVC-RT-S-models` 分支，不要将该分支推送到 Gitee。

- `gvc-rt-small_tflite_codec_270p_qp0_with_inputs.tar.gz`：三个 TFLite 模型及固定测试输入。
- `gvc-rt-small_dla_codec_270p_qp0_with_inputs.tar.gz`：已编译的 DLA 模型及固定测试输入。
- `gvc-rt-small_precision_270p_qp0.tar.gz`：服务器精度参考张量。
当前部署固定输入尺寸为 `256x512`（270p 填充后的工作尺寸），QP 固定为 `0`。

完整性校验见本目录 `SHA256SUMS.txt`；交付或部署前按清单核对。

正式包替换必须先在主项目 `model_test/<test-id>/` 按 `AGENTS.md` 流程完成精度与速度验证，通过后才允许替换；替换时同步更新本 README 与 `SHA256SUMS.txt`，随后删除整个测试目录。
