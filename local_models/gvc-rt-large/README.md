# GVC-RT Large 本地模型快照

本目录仅提交到本地 `GVC-RT-L-models` 分支，不要将该分支推送到 Gitee。

- `gvc-rt-large_tflite_codec_270p_qp0_with_inputs_runtime-tested.tar.gz`：替换无 `MTKEXT_SILU` 的 I/P 解码器后，在 Android 官方 NeuronDelegate 路径创建并运行验证通过的六模型包。
- `online_entropy/i_entropy_prior_merged.tflite`：I 帧 hyper/prior、四阶段掩码量化与反馈合并图。Android 官方 NeuronDelegate 在线路径实测离散符号完全一致，10 次均值约 `37 ms`。
- `gvc-rt-large_dla_codec_270p_qp0_with_inputs.tar.gz`：六个已编译的 DLA 模型及固定测试输入。
- `gvc-rt-large_precision_270p_qp0.tar.gz`：服务器精度参考张量与对比脚本。

当前部署固定输入尺寸为 `256x512`（270p 填充后的工作尺寸），QP 固定为 `0`，Tensor 布局为 NHWC、数据类型为 FP32。
