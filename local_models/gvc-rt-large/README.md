# GVC-RT Large 本地模型快照

本目录仅提交到本地 `GVC-RT-L-models` 分支，不要将该分支推送到 Gitee。

- `gvc-rt-large_tflite_codec_270p_qp0_with_inputs_runtime-tested.tar.gz`：正式六模型包。I/P decoder 使用高分辨率 GroupNorm 的 `1/16` 缩放方差实现，避免 Neuron FP16 stage3 溢出。Android 官方 NeuronDelegate 实测 I/P PSNR 为 `20.107/21.308/21.235 dB`。
- `online_entropy/i_entropy_prior_merged.tflite`：不含 rANS 的 I 帧合图，仅保留作诊断基线。
- `online_entropy/i_entropy_prior_merged_rans.tflite`：正式 I 编码 entropy/prior+rANS 单图。
- `online_entropy/i_entropy_decode_merged_rans.tflite`：I 帧码流解码单图，输入 rANS payload，图内串行完成 z/y rANS 解码、四阶段 prior 和 `y_hat` 恢复。Neuron 执行连续网络，rANS custom op 使用原生 CPU。Android 实测全部离散符号、CDF index、`y_hat` 和重建帧 exact；warmup 3、正式 10 次时 entropy+rANS 均值 `32.721 ms`。
- `online_entropy/p_entropy_prior_merged_rans.tflite`：正式 P 编码 entropy/prior+rANS 单图。
- `online_entropy/p_entropy_decode_merged_rans.tflite`：正式 P 解码 entropy/prior+rANS 单图。
- `online_entropy/manifest.json`：上述四张正式模型的 SHA256、固定输入输出名称和 NHWC shape。
- `gvc-rt-large_dla_codec_270p_qp0_with_inputs.tar.gz`：六个已编译的 DLA 模型及固定测试输入。
- `gvc-rt-large_precision_270p_qp0.tar.gz`：服务器精度参考张量与对比脚本。

当前部署固定输入尺寸为 `256x512`（270p 填充后的工作尺寸），QP 固定为 `0`，Tensor 布局为 NHWC、数据类型为 FP32。
