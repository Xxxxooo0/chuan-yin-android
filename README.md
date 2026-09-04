# GVC-RT Small 正式模型包

本目录仅提交到本地 `GVC-RT-S-models` 分支，不要将该分支推送到 Gitee。

当前在线正式包为 `gvc-rt-small_tflite_codec_270p_qp9.tar.gz`，解压顶层目录为 `small/`。
包含时序frame/feature、Encoder、Decoder和两张fused entropy，共六张TFLite；QP9、实际尺寸512x256。
采用官方AAR Neuron：`allowFp16=true`、`--relax-fp32`、`mtk-neuron`、`FAST_SINGLE_ANSWER`。
rANS在图内CPU custom op执行，CDF已嵌入图中；必须配套支持这些custom op的项目APK。

2026-09-03，Infinix X6891、ParkScene96帧测试：entropy编解码y_hat全部exact，
PSNR平均25.628 dB、最低24.068 dB；模型平均33.746 ms/帧（29.633 FPS），entropy合计16.696 ms。
FPS不含创建、视频读取、PNG保存和逐帧精度比较。用户批准后替换旧在线TFLite包，默认入口为 `fused_relax_fp32`。
本次不保证与服务器FP32逐值一致，也未验证两个独立视频参考状态循环。

本包包含entropy模型，属于内部完整链路包，不是企业无熵模型的四模型交付包。

## GPU 四图包

`gvc-rt-small_tflite_gpu_270p_qp9_four_graphs.tar.gz` 是独立的标准 TFLite GPU 四图包，解压顶层目录为 `gpusmall/`，只包含：

- `temporal_from_frame`
- `temporal_from_feature`
- `encoder`
- `decoder`

四图均为保持原数学语义的 PReLU channel-wise GPU 候选，不包含 entropy/rANS，Encoder latent 直接送入 Decoder，因此不产生真实码流。2026-09-04 在 Infinix X6891 上显式 GPU forced probe 完成 96/96 帧；四图均为完整 GPU delegation、`unexpected_cpu_nodes=0`，平均 PSNR `25.230 dB`，平均 `76.168 ms/帧`、`13.129 FPS`。该结果仅作为四图 GPU 数值与速度基线，不能代替完整 entropy+rANS 编解码质量。

以下旧QP0档案保留不动，不能作为本次QP9六图包的配套精度参考：

- `gvc-rt-small_dla_codec_270p_qp0_with_inputs.tar.gz`：旧三模型DLA及固定测试输入。
- `gvc-rt-small_precision_270p_qp0.tar.gz`：旧三模型服务器参考张量。

完整性校验见本目录 `SHA256SUMS.txt`；交付或部署前按清单核对。

部署路径：`/sdcard/Android/data/com.gvcrt.clean.mtkoffline/files/enterprise_tflite/small/`。
模型不内嵌APK；重装或换设备时需同时部署模型包。
源代码、构建结果和完整验证记录位于
`D:/android/ceshi/GVC-RT-S-bitrate/docs/Small六图正式包验证.md`。
压缩包由现有 `*.tar.gz` Git LFS规则管理。

正式包替换必须先在主项目 `model_test/<test-id>/` 按 `AGENTS.md` 流程完成精度与速度验证，通过后才允许替换；替换时同步更新本 README 与 `SHA256SUMS.txt`，随后删除整个测试目录。
