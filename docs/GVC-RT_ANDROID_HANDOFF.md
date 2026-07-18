# GVC-RT Android 部署交接文档

## 1. 文档目的

这份文档用于从零接手 GVC-RT Android 部署项目。阅读完后，应能够理解：

- 项目由哪三个模块组成；
- 服务器源码如何导出模型和测试基线；
- Android 如何加载模型并执行测试；
- 哪些部分已经基本正常；
- 当前 Recon、NNAPI、GPU/NPU 适配的主要问题；
- 后续应该从哪里继续优化。

当前项目只做 Android 部署和设备侧测试，不在本机电脑上运行 PyTorch、ONNX Runtime 推理或性能测试。

## 2. 项目路径

### Android 项目

Android 项目根目录中的主要部分：

```text
```
### 服务器源码

服务器 SSH 信息已省略。

服务器源码负责加载 PyTorch checkpoint、导出模型、生成输入输出基线和精度参考数据。
```

服务器源码负责加载 PyTorch checkpoint、导出模型、生成输入输出基线和精度参考数据。

## 3. 三个模块的边界

### 3.1 时序参考模块

时序参考模块只用于 P 帧，I 帧不使用。

上一帧可以提供两种参考状态：

```text
Reference Frame [1, 3, 256, 512]
    -> pixel_unshuffle(8)
    -> Feature Adaptor I
    -> Feature Extractor
    -> ctx / ctx_t [1, 256, 32, 64]
```

或者：

```text
Reference Feature [1, 256, 32, 64]
    -> Feature Adaptor P (1x1 Conv)
    -> Feature Extractor
    -> ctx / ctx_t [1, 256, 32, 64]
```

正式 Android 模型：

```text
models/temporal_from_frame.onnx
models/temporal_from_feature.onnx
```

主要算子：

- PixelUnshuffle；
- 1x1 Conv、3x3 Conv、stride Conv；
- GroupNorm；
- SiLU/WSiLU；
- Add、Mul；
- Reshape、Transpose。

### 3.2 完整编码模块

#### I 帧编码

```text
输入帧 [1, 3, 256, 512]
    -> i_encoder_front.onnx
    -> y [1, 256, 16, 32]
    -> i_hyper_enc.onnx
    -> z / z_hat
    -> i_hyper_prior.onnx
    -> i_prior_4x.onnx
    -> y 量化符号
    -> C++ JNI rANS
    -> I bitstream
```

I 帧同时生成参考帧：

```text
y_hat
    -> i_recon.onnx
    -> Reference Frame
```

#### P 帧编码

```text
当前帧 + Reference Frame
    -> temporal_from_frame.onnx
    -> ctx / ctx_t
    -> p_encoder_front.onnx
    -> y [1, 128, 16, 32]
    -> p_hyper_enc.onnx
    -> z / z_hat
    -> p_hyper_prior.onnx
    -> p_prior_2x.onnx
    -> y 量化符号
    -> C++ JNI rANS
    -> P bitstream
```

P 帧同时更新参考状态：

```text
p_y_hat + ctx
    -> p_recon.onnx
    -> Reference Feature / Reference Frame
```

正式 Android 模型：

```text
models/i_encoder_front.onnx
models/i_hyper_enc.onnx
models/i_hyper_prior.onnx
models/i_prior_4x.onnx
models/i_recon.onnx

models/p_encoder_front.onnx
models/p_hyper_enc.onnx
models/p_hyper_prior.onnx
models/p_prior_2x.onnx
models/p_recon.onnx
```

### 3.3 完整解码模块

#### I 帧解码

```text
I bitstream
    -> GVC/NAL 解析
    -> C++ JNI rANS 解码 z/y
    -> i_decode_hyper_prior.onnx
    -> i_decode_prior_stage_0.onnx
    -> i_decode_prior_stage_1.onnx
    -> i_decode_prior_stage_2.onnx
    -> i_decode_prior_stage_3.onnx
    -> i_recon.onnx
    -> Reference Frame
```

#### P 帧解码

```text
P bitstream + Reference state
    -> temporal_from_frame.onnx
    -> C++ JNI rANS 解码 z/y
    -> p_hyper_prior.onnx
    -> p_decode_prior_stage_0.onnx
    -> p_decode_prior_stage_1.onnx
    -> p_recon.onnx
    -> 新 Reference Feature / Reference Frame
```

Prior stage 不能简单看作普通卷积。每个 stage 都会结合当前已解码的 y 状态生成下一阶段的 scales/means，然后再进行下一组 rANS 解码。

## 4. Android 入口

### UI 按钮

主要入口在：

```text
app/src/main/java/com/gvcrt/clean/MainActivity.kt
```

当前主要按钮和 adb extra：

```text
时序参考精度：temporalReferenceTest
完整编码精度：completeEncoderTest
完整解码精度：completeDecoderTest

时序参考速度：temporalReferenceSpeedTest
完整编码速度：completeEncoderSpeedTest
完整解码速度：completeDecoderSpeedTest

完整项目：fullProjectTest
图片推理：imageInferenceTest
Recon 诊断：reconDiagnosticTest
```

### Android 构建

在项目根目录执行：

```powershell
.\gradlew.bat :app:compileDebugKotlin
.\gradlew.bat :app:assembleDebug
```

安装到手机：

```powershell
.\sdk\platform-tools\adb.exe install -r .\app\build\outputs\apk\debug\app-debug.apk
```

### Android 单轮测试

```powershell
$adb = (Resolve-Path '.\sdk\platform-tools\adb.exe').Path
& $adb logcat -c
& $adb shell am force-stop com.gvcrt.clean
& $adb shell am start -n com.gvcrt.clean/.MainActivity --ez completeEncoderTest true
& $adb logcat -d -s GVC_RT_CLEAN:I
```

把 `completeEncoderTest` 替换成其他 extra，即可运行其他入口。

## 5. 代码职责

### Kotlin

| 文件 | 作用 |
|---|---|
| `MainActivity.kt` | UI、按钮、adb extra 和测试调度 |
| `CleanModuleTests.kt` | 三模块精度测试和 baseline 比对 |
| `ModuleSpeedBenchmarks.kt` | 三模块速度、阶段耗时和内存测试 |
| `OnnxSessionRunner.kt` | ONNX Runtime session 创建和执行 |
| `ModuleManifest.kt` | 解析 `gvcrt_clean_manifest.json` |
| `MtkTfliteRuntime.kt` | MTK TFLite/native bridge 入口 |
| `OfficialNeuronRuntime.kt` | 官方 NeuronDelegate 入口 |
| `TfliteGpuRuntime.kt` | TFLite GPU Delegate 入口 |
| `ReconDiagnosticBenchmark.kt` | Recon 分段、GPU、Neuron、native 诊断 |
| `MemorySampler.kt` | App RAM、PSS、RSS 和系统内存采样 |
| `NativeRans.kt` | rANS 解码 JNI 包装 |
| `RansNativeEncoder.kt` | rANS 编码 JNI 包装 |
| `AssetStore.kt` | assets 读取、materialize 和 SHA256 |

### C++/JNI

| 文件 | 作用 |
|---|---|
| `mtk_tflite_jni.cpp` | TFLite、Neuron、OpenCL、Vulkan、native recon bridge |
| `rans_jni.cpp` | Kotlin 与 rANS C++ 实现之间的 JNI 接口 |
| `rans/rans.cpp` | rANS 核心实现 |
| `i_prior_npu_jni.cpp` | I prior NPU 实验接口，当前不是完整主链路 |
| `CMakeLists.txt` | native 库编译配置 |

## 6. 后端说明

### ONNX Runtime

正式 ONNX 流程由 `OnnxSessionRunner` 执行。

当前后端包括：

```text
CPU_ORT_ALL_OPT
NNAPI_FP16_ALLOW_FALLBACK
```

注意：

```text
FP32 ONNX 图 + USE_FP16
```

不等于严格 FP16，也不等于完整 NPU 执行。NNAPI session 创建成功也不代表所有节点都下沉到硬件。

### Neuron/NPU

Recon TFLite 诊断可以使用 `OfficialNeuronRuntime` 和 `NeuronDelegate`。

NPU 诊断中已经出现过：

```text
MDLA: Cannot support Float32 input
MDLA: Cannot support Float32 output
EDPA: false Transpose is not supported
EDPA: permSupport == perm 3D permutation not support
EDPA: unsupported operation
EDPA: false Unsupported data type Float32
```

因此，NPU 编译成功不代表大图完全下沉。必须同时确认：

- create 是否成功；
- invoke 是否成功；
- 是否 fully delegated；
- 是否发生 CPU fallback；
- 实际阶段耗时是否下降。

### GPU

项目中已有 TFLite GPU、OpenCL、Vulkan 和 native fused probe，但目前主要用于 recon 诊断，不应直接当作完整编码默认路径。

GPU 优先验证大子图：

```text
latent decoder
stage1 + stage2
upsampler + stage3
stage4 + final head
```

如果 GPU 只跑小段，CPU/GPU 之间的拷贝和同步可能抵消加速收益。

### rANS

rANS 使用 C++ JNI，在 CPU 上执行：

```text
量化符号
    -> CDF 查表
    -> rANS encode/decode
    -> flush
    -> payload
    -> bitstream mux
```

rANS 不属于 ONNX、NNAPI 或 recon 后端。分析完整编码速度时，必须把神经网络耗时和 rANS/symbol preparation CPU 耗时分开统计。

## 7. Recon 当前重点问题

### 7.1 主要速度热点

此前 native P recon 测试的参考结果：

| 阶段 | 平均耗时 |
|---|---:|
| Stage3 blocks | 16.507 ms |
| Stage4 blocks | 13.202 ms |
| Latent Decoder | 12.561 ms |
| Upsampler | 11.686 ms |
| Stage2 blocks | 9.061 ms |
| Stage1 Conv | 8.738 ms |
| Final Head | 7.830 ms |
| 完整 pipeline | 约 101.724 ms |

Stage3、Stage4、Latent Decoder 和 Upsampler 合计约 53.956 ms，是优先级最高的优化目标。

### 7.2 NPU 适配问题

主要问题不是 Conv 本身，而是：

- FP32 输入/输出不被 MDLA 完整支持；
- NCHW/NHWC 转换需要 Transpose；
- Transpose 和部分 3D permutation 不支持；
- Reshape 被诊断为 unsupported operation；
- DepthToSpace 的 FP32 版本不支持；
- GroupNorm/AdaGN 被展开成多个小算子；
- 多个 TFLite/Neuron 子图之间存在拷贝和同步。

所以大图可能实际变成：

```text
Conv -> NPU
Transpose/Reshape -> CPU
Conv -> NPU
AdaGN -> CPU
DepthToSpace -> CPU
```

这会让 NPU 加速收益很低，甚至比 native CPU/GPU 路径更慢。

### 7.3 Recon 路径差异

正式完整流程使用：

```text
i_recon.onnx
p_recon.onnx
```

native recon probe 使用：

```text
JNI + TFLite/native GroupNorm/AdaGN/PixelShuffle
```

两条路径的速度不能直接混用。要让 native recon 的优化结果影响完整编码，必须把它接入 I/P 编码中的参考帧或参考特征更新位置。

## 8. 精度测试方法

服务器负责生成 baseline，Android 只读取 baseline 并运行部署模型。

比较顺序：

```text
1. 输入 shape 和 SHA256
2. 模型资产 SHA256
3. 连续 tensor 的 max_abs/mean_abs/RMSE
4. z_hat 离散符号是否 exact
5. y_q_w 离散符号是否 exact
6. rANS payload 是否字节一致
7. 最终 encoded_ip.gvc 是否字节一致
8. Recon 输出帧误差和 PSNR
```

Recon 的 TFLite 某段已经出现过：

```text
max_abs ≈ 6.68e-6
mean_abs ≈ 4.96e-7
rmse ≈ 6.81e-7
```

这种情况属于很小的浮点差异，不能因为 `exact=false` 就直接判断模型错误。需要区分连续 tensor 误差和离散符号/码流错误。

## 9. 速度测试方法

速度测试必须在 Android 真机上执行，不能在本机电脑上跑推理。

每组测试建议：

```text
warmup 5 次
正式测试 50 次
输出 mean / p50 / p90
输出每个 stage 的耗时
输出 RAM start / peak / end
```

Recon 单独诊断阶段可以使用较短测试：

```text
warmup 3 次
正式测试 10 次
```

不要把以下时间混在一起：

- session 创建/编译时间；
- 单帧推理时间；
- tensor copy 时间；
- rANS 时间；
- bitstream mux 时间。

## 10. 服务器导出流程

服务器端基本流程：

```bash
cd /media/ltelab/D/weilingfeng/GVC-RT_clean_android
export GVC_RT_SOURCE_ROOT=/media/ltelab/D/weilingfeng/GVC-RT_inference

/media/ltelab/D/weilingfeng/conda_envs/weilingfeng/bin/python \
  server_tools/export_clean_gvcrt_modules.py \
  --source-root "$GVC_RT_SOURCE_ROOT" \
  --output-assets /media/ltelab/D/weilingfeng/GVC-RT_clean_android/app/src/main/assets \
  --height 256 \
  --width 512 \
  --qp 0 \
  --precision fp32 \
  --opset 12
```

如果要导出 recon 诊断模型，使用对应的 `server_tools/export_recon_*.py` 脚本，并把输出放到独立目录，不要覆盖正式模型。

## 11. 继续优化时的推荐顺序

### 第一步：冻结正确性基线

保留：

```text
i_recon.onnx
p_recon.onnx
```

它们作为完整编码/解码精度基线，不要因为速度实验直接覆盖。

### 第二步：只改一个 recon 大段

推荐顺序：

```text
Stage3 + Stage4
    -> Upsampler
    -> Latent Decoder
    -> Stage1 + Stage2
    -> Final Head
```

每次只替换一个候选实现，然后在 Android 上比较：

- create/invoke 是否成功；
- 是否 fully delegated；
- mean/p50/p90；
- 内存峰值；
- 最终重建输出。

### 第三步：减少边界

目标不是导出更多小模型，而是形成少量连续大子图：

```text
固定布局
统一数据类型
减少 Transpose/Reshape
融合 Conv + Norm + AdaGN
融合 Conv + PixelShuffle/DepthToSpace
减少 JNI/TFLite/Neuron 间拷贝
```

### 第四步：接回完整流程

只有单独 recon 大段验证通过后，才接入：

```text
I 编码 -> I reference frame
P 编码 -> P reference feature
I 解码 -> I reference frame
P 解码 -> P reference update
```

## 12. 接手检查清单

开始修改前，先确认：

- [ ] 当前工作目录是 `D:\android\ceshi\GVC-RT_clean_android`；
- [ ] 没有误用旧 Android 项目的 ONNX 或 benchmark；
- [ ] 正式模型位于 `app/src/main/assets/models/`；
- [ ] 当前 checkpoint 和 baseline 的 SHA256 已记录；
- [ ] 当前测试设备已通过 `adb devices` 识别；
- [ ] 测试命令明确指定了入口 extra；
- [ ] 日志中区分了 ORT、NNAPI、Neuron、GPU、native 和 CPU rANS；
- [ ] 没有把 session 创建时间当成单帧推理时间；
- [ ] 没有把 isolated recon 速度当成完整编码速度；
- [ ] 修改前先保存当前 Git 状态；
- [ ] 每次只做一个变量的后端或子图替换。

## 13. 当前结论

当前项目已经具备三模块 Android 测试入口、ONNX 正式模型、rANS JNI、Recon 诊断和多后端实验能力。

目前最需要解决的不是模型流程缺失，而是 Recon 的后端适配和执行边界：

```text
NCHW/NHWC 转换
 + FP32 NPU 限制
 + Transpose/Reshape 不支持
 + GroupNorm/AdaGN 未融合
 + PixelShuffle/DepthToSpace 适配问题
 + 多子图拷贝同步
```

下一阶段最合理的主路线是：保留 ONNX 作为正确性基线，优先把 Stage3、Stage4、Upsampler 和 Latent Decoder 组成较少的 native/GPU 大子图，再通过 Android 真机测试决定是否接入完整编码和解码流程。
