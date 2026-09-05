# Large GPU 模型包

本目录交付 Large 标准 TFLite GPU 十图包：

```text
gvc-rt-large_tflite_gpu_270p_qp9_ten_graphs.tar.gz
```

归档顶层目录为 `gpularge/`。六张连续神经网络图由 GPU Delegate 执行；四张 entropy+rANS 图当前使用普通 TFLite CPU builtin 与既有 native CPU rANS。显式 GPU 模式不使用 NNAPI、XNNPACK，也不会回退 MTK。

设备部署目录：

```text
/sdcard/Android/data/com.gvcrt.clean.mtkoffline/files/enterprise_tflite/gpularge/
```

Infinix X6891 十图闭环已通过。相同 APK、输入和 QP9 下，GPU 平均 PSNR 为 `22.804 dB`，MTK 为 `22.815 dB`；正式包三帧序列回归为平均 `25.552 dB`、`0.807 FPS`、`all_models_exercised=true`。速度瓶颈为四张 entropy 图的 TFLite CPU 执行。包校验值见 `SHA256SUMS.txt`，部署和运行命令见 [Large GPU 十图路径](../../docs/tflite-gpu-large.md)。
