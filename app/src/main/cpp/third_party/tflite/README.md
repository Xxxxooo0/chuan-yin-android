# TensorFlow Lite C ABI 头文件

这些未修改的头文件取自 TensorFlow `v2.17.0`，与项目声明的 TFLite 版本一致：

- https://github.com/tensorflow/tensorflow/tree/v2.17.0/tensorflow/lite/core/c
- https://github.com/tensorflow/tensorflow/blob/v2.17.0/tensorflow/lite/c/common.h

仅用于 `gvcrt_gpu_guard` 读取标准 GPU Delegate 应用后的 execution plan。
没有编译另一份 TFLite，没有 MTK 依赖，也没有算子计算实现。
许可证见 `LICENSE`。升级 TFLite 时需要同步检查这些 ABI 头文件。
