# 服务端导出工具

这些脚本只在远端 Linux/PyTorch 环境运行。本机仅用于编辑、构建 APK 和通过 adb 安装测试。

## ONNX Demo 导出

从源码导出 ONNX demo 所需模型与基线，固定尺寸为 `256x512`、QP 为 `0`：

```bash
cd /media/ltelab/D/weilingfeng/GVC-RT_clean_android
export GVC_RT_SOURCE_ROOT=/media/ltelab/D/weilingfeng/GVC-RT_inference
python server_tools/export_clean_gvcrt_modules.py \
  --height 256 --width 512 --qp 0 --precision fp32
```

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

## 结果审计

使用 `audit_mtk_offline_assets.py` 检查离线模型的 manifest、SHA 和 NCC 结果。`compare_enterprise_precision_outputs.py` 用企业回传的输出与参考向量生成精度报告。

输出目录 `outputs/`、SDK、模型资产和压缩包均为本地/服务器生成物，不提交到代码分支。
