# 模型索引

本目录集中管理本机部署使用的全部模型。主项目只保留这一份 Android 与导出代码，模型二进制不进入主代码分支。

```text
models/
├─ onnx-demo/   ONNX Demo 的 APK 运行资产
├─ large/       GVC-RT Large 模型分支 worktree
└─ small/       GVC-RT Small 模型分支 worktree
```

## ONNX Demo

- 资产目录：`models/onnx-demo/assets/`
- 模型目录：`models/onnx-demo/assets/models/`
- 基线目录：`models/onnx-demo/assets/baseline/`
- 清单：`models/onnx-demo/assets/gvcrt_clean_manifest.json`
- Android 构建：`./gradlew :app:assembleOnnxDemoDebug`

Gradle 将该资产目录作为 `onnxDemo` flavor 的额外 assets source。使用 `-PgvcrtSkipAssets` 时不打包这些大模型，适合只做源码和 JNI 构建检查。

当前清单中的 19 个已存在 ONNX 文件均通过 SHA256 校验。清单还记录了未随本机资产保存的可选融合图 `i_image_encoder_fused.onnx` 和 `p_image_encoder_fused.onnx`；当前 Android 路径使用已验证的分图序列，不应将这两个缺失文件视为可用部署入口。

## GVC-RT Large

- 模型分支：`GVC-RT-L-models`
- Worktree：`models/large/`
- 包目录：`models/large/local_models/gvc-rt-large/`
- 当前推荐包：`gvc-rt-large_tflite_codec_270p_qp0_with_inputs_runtime-tested.tar.gz`
- 独立 I 熵模型：`models/large/local_models/gvc-rt-large/online_entropy/i_entropy_prior_merged.tflite`

I 熵合并模型不进入基础模型压缩包。部署时先解压基础包，再将该文件单独放入设备包的
`models/i_entropy_prior_merged.tflite`。旧的 7 张 I entropy/prior 分图不再使用。

## GVC-RT Small

- 模型分支：`GVC-RT-S-models`
- Worktree：`models/small/`
- 包目录：`models/small/local_models/gvc-rt-small/`
- 当前 TFLite 包：`gvc-rt-small_tflite_codec_270p_qp0_with_inputs.tar.gz`

## 使用原则

1. Large/Small worktree 使用稀疏检出，只保留各自的模型包；Android、文档和导出代码统一使用主项目版本。
2. 交付或部署前根据包目录内的 `SHA256SUMS.txt` 校验完整性。
3. Android 设备侧将解压后的包分别放入 `enterprise_tflite/large/` 或 `enterprise_tflite/small/`。
4. 模型输入固定为 `256x512`、QP 为 `0`；布局、精度和后端要求以包内 README 与 manifest 为准。
