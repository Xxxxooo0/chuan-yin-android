# Small 标准 TFLite GPU 后端

本页说明 GVC-RT Small 六图 GPU 路径。现有 MTK/NPU 模型、`OfficialNeuronRuntime`、Neuron 参数、cache、tensor 路径和 rANS 码流均保持不变；GPU 使用独立 `gpusmall/` 模型目录和标准 TensorFlow Lite GPU Delegate。

## 当前执行结构

```text
temporal / encoder / decoder
        ↓
完整 GPU delegation

entropy encode / decode fused
        ↓
GPU 可支持的 NN 分区 + TFLite builtin CPU fallback
        ↓
已知 Small rANS custom op
        ↓
现有 native CPU rANS
```

Small 六图为：

1. `temporal_from_frame.tflite`
2. `temporal_from_feature.tflite`
3. `encoder.tflite`
4. `entropy_encode_fused.tflite`
5. `entropy_decode_fused.tflite`
6. `decoder.tflite`

四张普通神经网络图仍执行严格 guard：任何普通 TFLite builtin 留在 CPU 都会失败。两张 fused entropy 图允许 GPU Delegate 当前不支持的 builtin 使用默认 TFLite CPU kernel；只精确注册以下既有 rANS custom op，不开放其它 CUSTOM：

- `GVC_RT_SMALL_RANS_ENCODE`
- `GVC_RT_SMALL_RANS_DECODE_Z`
- `GVC_RT_SMALL_RANS_DECODE_Y`

GPU 路径不启用 NNAPI、XNNPACK 或 MTK fallback。rANS 复用 `app/src/main/cpp/rans/rans.cpp`，没有复制或修改算法。

## 后端选择

- `MTK_NPU`：保持原来的官方 AAR `NeuronDelegate` 路径。
- `AUTO`：保持保守策略；CompatibilityList 不支持时不自动强制 GPU，默认 MTK-first。
- `GPU`：即使 CompatibilityList 返回 `false`，也实际创建 GPU Delegate 和 Interpreter，检查 delegation coverage 并执行 invoke；任一阶段失败直接报错，不回退其它后端。

Infinix X6891 上会记录：

```text
compatibility_list_supported=false gpu_forced_probe=true
```

Delegate 构造成功本身不代表设备可用。只有创建 Interpreter、coverage 检查和真实 invoke 都成功，当前模型才算在该设备上可运行。

## 模型包

GPU 六图包位于 `models/small` 的 `GVC-RT-S-models` 分支：

```text
gvc-rt-small_tflite_gpu_270p_qp9_six_graphs.tar.gz
```

`codex/gpu-version` 代码分支同时在 `gpu_models/small/` 保留同一包及其
`SHA256SUMS.txt`，用于单独检出该分支时直接部署 GPU 演示。

归档顶层目录为 `gpusmall/`，部署到：

```text
/sdcard/Android/data/com.gvcrt.clean.mtkoffline/files/enterprise_tflite/gpusmall/
```

四张普通图中的共享 PReLU alpha 已显式展开为 channel-wise 常量，修复 GPU Delegate 对 `[1,1,1,1]` 共享斜率的解析限制；数学语义、I/O shape、layout、dtype 与 QP 不变。归档中的 `manifest.json`、`README.md` 和 `SHA256SUMS.txt` 给出完整定义与校验。

## Infinix X6891 真机结果

设备：Infinix X6891 / MT6899 / Android 16。输入为同一组 ParkScene 512×256 RGB PNG 96 帧，固定 QP9。

- 完成 96/96 帧，encode → payload → independent decode 闭环成功。
- 四张普通图全部 `unexpected_cpu_nodes=0`。
- fused encode：`gpu_nodes=1 allowed_native_rans_nodes=1 allowed_builtin_cpu_fallback_nodes=58 unexpected_cpu_nodes=0`。
- fused decode：`gpu_nodes=1 allowed_native_rans_nodes=3 allowed_builtin_cpu_fallback_nodes=35 unexpected_cpu_nodes=0`。
- rANS invoke：encode 96 次、decode Z 96 次、decode Y 192 次，全部成功。
- 平均/最低 PSNR：`22.440/19.005 dB`。
- 平均模型耗时：`128.093 ms/帧`，`7.807 FPS`；不含创建、视频读取、PNG 保存和逐帧精度比较。

| 模型 | GPU 六图平均耗时（ms） | MTK 六图平均耗时（ms） |
| --- | ---: | ---: |
| `temporal_from_frame` | 31.468 | 13.895（3 次 reference reset） |
| `temporal_from_feature` | 16.846 | 4.537 |
| `encoder` | 20.454 | 4.643 |
| `entropy_encode_fused` | 31.941 | 10.975 |
| `entropy_decode_fused` | 26.446 | 5.822 |
| `decoder` | 31.949 | 7.838 |
| 全链路 | 128.093 | 34.107 |

两轮均未做稳态预热，逐图速度用于当前设备上的路径对比，不代表其它设备。

## 精度说明

同一输入的结果为：

| 路径 | 平均 PSNR |
| --- | ---: |
| 服务器源码 raw latent（无 entropy） | 25.226425 dB |
| GPU 四图 raw latent（无 entropy） | 25.230 dB |
| 服务器源码 entropy | 22.944776 dB |
| MTK 六图 entropy+rANS | 22.946229 dB |
| GPU 六图 entropy+rANS | 22.440 dB |

Small 源码自身的 entropy quantization 会带来约 `2.282 dB` 的下降，这不是 rANS 错误；MTK 六图与同输入源码 entropy 对齐。GPU 六图相对源码/MTK 六图仍低约 `0.505 dB`，因此当前状态是“六图可闭环运行，存在已知精度差距”，不能宣称为 MTK 的无精度差替代品。

四图 raw latent 路径不产生码流，只用于普通 GPU 图的质量与速度 A/B；离线完整编解码应使用六图 entropy+rANS 路径。

## 构建和真机运行

本地只编译和打包，不运行模型推理：

```powershell
.\gradlew.bat -PgvcrtSkipAssets :app:compileMtkOfflineDebugKotlin :app:assembleMtkOfflineDebug --console=plain
```

安装并执行显式 GPU 六图：

```powershell
$adb = '.\sdk\platform-tools\adb.exe'
$deviceRoot = '/sdcard/Android/data/com.gvcrt.clean.mtkoffline/files'
& $adb install -r .\app\build\outputs\apk\mtkOffline\debug\app-mtkOffline-debug.apk
& $adb shell am start -n com.gvcrt.clean.mtkoffline/com.gvcrt.clean.MainActivity --ez smallOnlineVideoTest true --es backend GPU --ez smallOnlineUseEntropy true --es sequenceDir "$deviceRoot/inputs" --ei sequenceFrames 96
& $adb logcat -d -s GVC_RT_CLEAN:I tflite:I
```

只测试四张普通图时可传 `--ez smallOnlineUseEntropy false`。该参数只允许显式 GPU，不能作为真实编解码或码流验证结果。

## 关键日志

每张图依次检查：

```text
gpu_delegate_create_ok
gpu_interpreter_create_ok
gpu_delegation_checked ...
gpu_invoke_ok
```

coverage 日志区分：

```text
gpu_nodes=<数量>
allowed_native_rans_nodes=<数量>
allowed_builtin_cpu_fallback_nodes=<数量>
unexpected_cpu_nodes=<数量>
```

普通四图必须没有 CPU 节点；fused entropy 图只有已知 rANS 和明确启用的 builtin fallback 不计入 unexpected。未知 CUSTOM、其它 delegate 或未批准的执行节点仍会失败。

## 导出脚本

- `server_tools/export_small_temporal_gpu_prelu.py`：只修复 `temporal_from_frame` 的 channel-wise PReLU alpha，并做原图/候选数值一致性验证。
- `server_tools/export_small_gpu_prelu_remaining.py`：以相同规则处理其余五图；fused 图写出时仍保留 rANS custom op。

脚本只在服务器 TensorFlow 环境运行，候选输出必须先放入 `model_test/<test-id>/`，通过真机验证后才能晋升到模型分支。
