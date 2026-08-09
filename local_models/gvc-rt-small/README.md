# GVC-RT Small 本地模型快照

本目录仅提交到本地 `GVC-RT-S-models` 分支，不要将该分支推送到 Gitee。

- `gvc-rt-small_tflite_codec_270p_qp0_with_inputs.tar.gz`：三个 TFLite 模型及固定测试输入。
- `gvc-rt-small_dla_codec_270p_qp0_with_inputs.tar.gz`：已编译的 DLA 模型及固定测试输入。
- `gvc-rt-small_precision_270p_qp0.tar.gz`：服务器精度参考张量。
- `gvc-rt-small_server_original.tar.gz`：服务器端 Small 原始包快照。

当前部署固定输入尺寸为 `256x512`（270p 填充后的工作尺寸），QP 固定为 `0`。
