# GVC-RT Small 正式模型包

本目录仅提交到本地 `GVC-RT-S-models` 分支，不要将该分支推送到 Gitee。

当前在线正式包为 `gvc-rt-small_tflite_codec_270p_qp9.tar.gz`，解压顶层目录为 `small/`。
包含时序frame/feature、Encoder、Decoder和两张fused entropy，共六张TFLite；QP9、实际尺寸512x256。
采用官方AAR Neuron：`allowFp16=true`、`--relax-fp32`、`mtk-neuron`、`FAST_SINGLE_ANSWER`。
rANS在图内CPU custom op执行，CDF已嵌入图中；必须配套支持这些custom op的项目APK。

2026-09-05，Infinix X6891、同一组 ParkScene RGB PNG 96 帧复测：完成 96/96 帧，
PSNR 平均/最低为 `22.946/20.094 dB`；模型平均 `34.107 ms/帧`（`29.319 FPS`），
entropy encode/decode 合计 `16.797 ms/帧`。同输入服务器源码 entropy 路径为 `22.944776 dB`，
手机 MTK 六图平均差仅 `+0.001453 dB`。FPS 不含创建、视频读取、PNG 保存和逐帧精度比较。

本包包含entropy模型，属于内部完整链路包，不是企业无熵模型的四模型交付包。

## GPU 四图包

`gvc-rt-small_tflite_gpu_270p_qp9_four_graphs.tar.gz` 是独立的标准 TFLite GPU 四图包，解压顶层目录为 `gpusmall/`，只包含：

- `temporal_from_frame`
- `temporal_from_feature`
- `encoder`
- `decoder`

四图均为保持原数学语义的 PReLU channel-wise GPU 候选，不包含 entropy/rANS，Encoder latent 直接送入 Decoder，因此不产生真实码流。2026-09-04 在 Infinix X6891 上显式 GPU forced probe 完成 96/96 帧；四图均为完整 GPU delegation、`unexpected_cpu_nodes=0`，平均 PSNR `25.230 dB`，平均 `76.168 ms/帧`、`13.129 FPS`。该结果仅作为四图 GPU 数值与速度基线，不能代替完整 entropy+rANS 编解码质量。

## GPU 六图包

`gvc-rt-small_tflite_gpu_270p_qp9_six_graphs.tar.gz` 是独立的标准 TFLite GPU 六图候选，解压顶层目录同样为 `gpusmall/`。它包含上述四张普通 GPU 图，以及 `entropy_encode_fused` 和 `entropy_decode_fused`：

- 四张普通模型的神经网络节点完整交给 GPU Delegate，`unexpected_cpu_nodes=0`。
- 两张 fused entropy 图中可支持的神经网络分区交给 GPU；当前 GPU Delegate 不支持的 builtin 允许由普通 TFLite CPU kernel 执行。
- 只允许三个已知 Small rANS custom op 使用本项目 native CPU 实现；不使用 MTK、NNAPI 或 XNNPACK fallback。
- Infinix X6891 的 `CompatibilityList=false`，显式 `backend=GPU` 会继续 forced probe；`AUTO` 不自动强制 GPU。

2026-09-04 同设备、同一组 96 帧真机闭环结果：完成 96/96 帧；fused encode coverage 为 `gpu_nodes=1 allowed_native_rans_nodes=1 allowed_builtin_cpu_fallback_nodes=58 unexpected_cpu_nodes=0`，fused decode 为 `1/3/35/0`。平均/最低 PSNR 为 `22.440/19.005 dB`，平均 `128.093 ms/帧`（`7.807 FPS`）。

同输入服务器源码 entropy 为 `22.944776 dB`，MTK 六图为 `22.946229 dB`，GPU 六图分别低约 `0.505 dB` 和 `0.506 dB`。因此该包已经验证六图 GPU 路径可以闭环运行，但仍有明确精度差距，当前作为 GPU 候选保留，不宣称为 MTK 的无损精度替代。归档内 README 记录执行边界、逐图耗时、部署命令和完整校验值。

以下旧QP0档案保留不动，不能作为本次QP9六图包的配套精度参考：

- `gvc-rt-small_dla_codec_270p_qp0_with_inputs.tar.gz`：旧三模型DLA及固定测试输入。
- `gvc-rt-small_precision_270p_qp0.tar.gz`：旧三模型服务器参考张量。

完整性校验见本目录 `SHA256SUMS.txt`；交付或部署前按清单核对。

MTK 包部署路径为 `/sdcard/Android/data/com.gvcrt.clean.mtkoffline/files/enterprise_tflite/small/`；GPU 包部署路径为 `/sdcard/Android/data/com.gvcrt.clean.mtkoffline/files/enterprise_tflite/gpusmall/`。模型不内嵌 APK；重装或换设备时需同时部署对应模型包。
压缩包由现有 `*.tar.gz` Git LFS规则管理。

正式包替换必须先在主项目 `model_test/<test-id>/` 按 `AGENTS.md` 流程完成精度与速度验证，通过后才允许替换；替换时同步更新本 README 与 `SHA256SUMS.txt`，随后删除整个测试目录。
