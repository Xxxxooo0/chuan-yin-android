# Small GPU 模型包

本目录随 `codex/gpu-version` 代码分支交付 Small 标准 TFLite GPU 模型包：

```text
gvc-rt-small_tflite_gpu_270p_qp9_six_graphs.tar.gz
```

归档顶层目录为 `gpusmall/`，包含 `temporal_from_frame`、
`temporal_from_feature`、`encoder`、`entropy_encode_fused`、
`entropy_decode_fused` 和 `decoder` 六张 TFLite 图。完整编解码使用六图；
无 entropy 的离线重建演示只使用其中四张普通图，不生成真实码流。

设备部署目录：

```text
/sdcard/Android/data/com.gvcrt.clean.mtkoffline/files/enterprise_tflite/gpusmall/
```

该包复用已经在 `GVC-RT-S-models` 分支完成真机验证的同一文件，校验值见
`SHA256SUMS.txt`。GPU 六图相对服务器源码/MTK 六图仍有约 `0.505 dB` 的已知
PSNR 差距，因此当前用于 GPU 隔离演示，不作为 MTK NPU 的无精度差替代品。
