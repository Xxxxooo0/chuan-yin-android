# 服务端导出工具

这些脚本只在远端 Linux/PyTorch 环境运行。本机仅用于编辑、构建 APK 和通过 adb 安装测试。

## ONNX 中间图与基线导出

该脚本不再对应 Android 运行入口，仅在服务器保留，用于从源码导出转换与验证所需的 ONNX 中间图和基线。固定尺寸为 `256x512`、QP 为 `0`：

```bash
cd /media/ltelab/D/weilingfeng/GVC-RT_clean_android
export GVC_RT_SOURCE_ROOT=/media/ltelab/D/weilingfeng/GVC-RT_inference
python server_tools/export_clean_gvcrt_modules.py \
  --output-assets outputs/onnx-intermediate \
  --height 256 --width 512 --qp 0 --precision fp32
```

默认输出同样是 `outputs/onnx-intermediate/`；该目录由 Git 忽略，不参与 Android 构建。

## MTK Large 离线导出（企业交付备用）

该路径导出并离线编译 GVC-RT Large 的前端、熵模型和合并解码图，用于企业 `.dla` 交付。当前 Android 应用运行的是 TFLite + 官方 `NeuronDelegate` 在线编译路线，不直接加载这些 `.dla`。

```bash
cd /media/ltelab/D/weilingfeng/GVC-RT_clean_android
export GVC_RT_SOURCE_ROOT=/media/ltelab/D/weilingfeng/GVC-RT_inference
bash server_tools/run_mtk_offline_export.sh
```

## GVC-RT Small 离线导出（企业交付备用）

Small 模型使用独立源码和 checkpoint。`--ncc-tflite` 指向 SDK 内的 `ncc-tflite` 可执行文件。

```bash
python server_tools/export_gvc_rt_small_enterprise_dla.py \
  --source-root /media/ltelab/D/weilingfeng/mlvc \
  --checkpoint /path/to/checkpoint.pt \
  --ncc-tflite /path/to/neuron_sdk/host/bin/ncc-tflite
```

## 精度向量与交付包

`export_enterprise_precision_vectors.py` 为企业端 DLA 输出生成输入、参考输出和比较清单；`package_enterprise_dla_codec.py` 打包已验证模型；`package_enterprise_dla_with_inputs.py` 将模型包与输入精度包合并。脚本的 `--help` 给出必填路径。

```bash
python server_tools/export_enterprise_precision_vectors.py --help
python server_tools/package_enterprise_dla_codec.py --help
python server_tools/package_enterprise_dla_with_inputs.py --help
```

## Large 在线 entropy/prior

I/P 编码侧分别由 `export_i_entropy_merged_nhwc.py`、`export_p_entropy_merged_nhwc.py`
生成 prior 合图，再由 `append_i_rans_custom_op.py`、`append_p_rans_custom_op.py` 注入
原生 rANS。I/P 解码侧生成串行 prior 基图后，由对应
`append_*_rans_decode_custom_ops.py` 注入原生 rANS decode。正式 Android 路径不再使用
旧的 I 7 张分图、P 4 张分图或旧解码多图。完整命令、独立模型文件和测试入口见
[Large 在线部署](../docs/large-online-deployment.md)。

## TFLite CPU 自定义算子分区探针

`export_tflite_custom_op_partition_probe.py` 生成
`Conv2D -> GVC_RT_CPU_IDENTITY -> Conv2D` 测试图，用于验证同一个 Interpreter
能否在 Neuron 分区之间执行 CPU 自定义算子。该探针只验证运行机制，不包含真实 rANS。

```bash
python -u server_tools/export_tflite_custom_op_partition_probe.py \
  --android-root /media/ltelab/D/weilingfeng/GVC-RT_clean_android \
  --copy-assets
```

## I 帧 merged entropy + rANS 变体

`append_i_rans_custom_op.py` 不重新导出神经网络，只在已经验证的
`i_entropy_prior_merged.tflite` 尾部追加 CPU 自定义算子
`GVC_RT_RANS_ENCODE`。CDF 常量写入模型，原 10 个输出保留，并新增固定容量 payload
和实际 payload 长度。生成的 `i_entropy_prior_merged_rans.tflite` 是当前正式 I 编码
entropy 模型，源 merged 文件只用于导出和验证。

```bash
python -u server_tools/append_i_rans_custom_op.py \
  --merged-model outputs/i_entropy_merged_nhwc/i_entropy_prior_merged_fp32.tflite \
  --entropy-package-dir outputs/gvc-rt-large_tflite_online_entropy_270p_qp0 \
  --output outputs/i_entropy_merged_rans/i_entropy_prior_merged_rans.tflite
```

Android 将生成文件放入 Large 在线包的 `models/` 后，通过
`--ez largeIEntropyCodecTest true` 运行正式 I 编码路径。

## I 帧 merged entropy decoder + rANS 变体

`export_i_entropy_decoder_merged_nhwc.py` 先导出以 `z_hat/y_q_w_0..3` 为占位输入的
串行 prior 合图，并验证其与源码等价。`append_i_rans_decode_custom_ops.py` 再把这些输入
替换为 `GVC_RT_RANS_DECODE_Z/Y` custom op，使最终模型只接收 payload buffer 和实际
长度，输出 `z_hat`、四阶段符号/scales 和最终 `y_hat`。该变体用于减少四个 Interpreter
之间的调度和 tensor 拷贝，rANS 仍由原生 CPU 执行。

## Large 固定 QP9 在线包打包

固定 QP9 连续图由 `export_three_modules_offline_nhwc.py` 与
`export_decoder_full_norm_rewrite_nhwc.py` 从同一 I/P checkpoint 导出，`package_large_fixed_qp.py`
将六张连续图与四张 entropy+rANS 图组成正式包
`gvc-rt-large_tflite_online_fixed_qp9_270p.tar.gz`。完整命令见
[Large 在线部署](../docs/large-online-deployment.md)。

## 企业 TFLite 包转 DLA 与视频序列交付

`compile_tflite_package_to_dla.py` 将已验证的企业 TFLite 包整体编译为 NCC 验证的离线 DLA 包
（`--arch mdlaX.Y`、`--opt-bw`、`--relax-fp32`）；Small 输入包必须恰好包含四张模型且 `fixed_q_index=9`。`package_enterprise_video_sequence.py`
将已验证 DLA 模型与链式视频序列输入合并为交付包：Small 使用四模型 QP9 流程（`temporal_from_frame`、`temporal_from_feature`、`encoder`、`decoder`），Large 使用六模型流程。

```bash
python server_tools/compile_tflite_package_to_dla.py --help
python server_tools/package_enterprise_video_sequence.py --help
```

## 服务器基准与质量指标

`benchmark_large_park_scene_sequence.py` 在 ParkScene 序列上运行 Large PyTorch 主线，输出
码率、PSNR、MS-SSIM、LPIPS、DISTS 的 JSON 报告；`compute_perceptual_metrics.py` 对保存的
RGB 重建图计算 LPIPS/DISTS；`plot_park_rd_curve.py` 汇总各 QP 报告并绘制率失真曲线。

```bash
python server_tools/benchmark_large_park_scene_sequence.py --help
python server_tools/compute_perceptual_metrics.py --help
python server_tools/plot_park_rd_curve.py --help
```

## np708/mdla50 构建脚本

`build_large_np708_mdla50_scaled_variance.sh` 使用 NeuroPilot 7.0.8 工具链
（`mtk_pytorch_converter` + `ncc-tflite`）构建 270p/QP0 Large 企业包，采用 FP16 稳定的
缩放方差解码器，产物为 `gvc-rt-large_dla_codec_270p_qp0_np708_mdla50_scaled_variance*`。
路径通过环境变量覆盖，默认值见脚本头部。

## 结果审计

使用 `audit_mtk_offline_assets.py` 检查离线模型的 manifest、SHA 和 NCC 结果。`compare_enterprise_precision_outputs.py` 用企业回传的输出与参考向量生成精度报告。

输出目录 `outputs/`、SDK、模型资产和压缩包均为本地/服务器生成物，不提交到代码分支。
