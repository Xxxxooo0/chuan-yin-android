# GVC-RT Large 在线部署

当前 Large 在线路径使用官方 `NeuronDelegate`。I/P entropy/prior 均使用单张合图，
rANS 通过图内原生 CPU custom op 执行，连续神经网络交给 NeuronDelegate：

- I 编码：`i_entropy_prior_merged_rans.tflite`
- I 解码：`i_entropy_decode_merged_rans.tflite`
- P 编码：`p_entropy_prior_merged_rans.tflite`
- P 解码：`p_entropy_decode_merged_rans.tflite`

旧的 I 7 张分图和 P 4 张分图不再用于正式 Android 路径。

## 服务器导出与打包

```bash
cd /media/ltelab/D/weilingfeng/GVC-RT_clean_android
export GVC_RT_SOURCE_ROOT=/media/ltelab/D/weilingfeng/GVC-RT_inference

python -u server_tools/export_i_entropy_merged_nhwc.py \
  --source-root "$GVC_RT_SOURCE_ROOT" \
  --android-root "$PWD" \
  --output-dir outputs/i_entropy_merged_nhwc

python -u server_tools/package_large_online_entropy.py \
  --source-root "$GVC_RT_SOURCE_ROOT" \
  --base-package-dir /path/to/unpacked/gvc-rt-large_tflite_codec_270p_qp0_with_inputs \
  --i-merged-model outputs/i_entropy_merged_nhwc/i_entropy_prior_merged_fp32.tflite \
  --output-dir outputs/gvc-rt-large_tflite_online_entropy_270p_qp0

python -u server_tools/append_i_rans_custom_op.py \
  --merged-model outputs/i_entropy_merged_nhwc/i_entropy_prior_merged_fp32.tflite \
  --entropy-package-dir outputs/gvc-rt-large_tflite_online_entropy_270p_qp0 \
  --output outputs/i_entropy_merged_rans/i_entropy_prior_merged_rans.tflite

python -u server_tools/export_i_entropy_decoder_merged_nhwc.py \
  --source-root "$GVC_RT_SOURCE_ROOT" \
  --android-root "$PWD" \
  --output-dir outputs/i_entropy_decoder_merged_nhwc \
  --ncc-tflite neuropilot-sdk-premium-8.0.11-build20260211/neuron_sdk/host/bin/ncc-tflite

python -u server_tools/append_i_rans_decode_custom_ops.py \
  --merged-base-model outputs/i_entropy_decoder_merged_nhwc/i_entropy_decode_merged_base_fp32.tflite \
  --entropy-package-dir outputs/gvc-rt-large_tflite_online_entropy_270p_qp0 \
  --output outputs/i_entropy_decoder_merged_rans/i_entropy_decode_merged_rans.tflite

python -u server_tools/export_p_entropy_merged_nhwc.py \
  --source-root "$GVC_RT_SOURCE_ROOT" --android-root "$PWD" \
  --output-dir outputs/p_entropy_merged_nhwc \
  --ncc-tflite neuropilot-sdk-premium-8.0.11-build20260211/neuron_sdk/host/bin/ncc-tflite

python -u server_tools/append_p_rans_custom_op.py \
  --merged-model outputs/p_entropy_merged_nhwc/p_entropy_prior_merged_fp32.tflite \
  --entropy-package-dir outputs/gvc-rt-large_tflite_online_entropy_270p_qp0 \
  --output outputs/p_entropy_merged_rans/p_entropy_prior_merged_rans.tflite

python -u server_tools/export_p_entropy_decoder_merged_nhwc.py \
  --source-root "$GVC_RT_SOURCE_ROOT" --android-root "$PWD" \
  --output-dir outputs/p_entropy_decoder_merged_nhwc \
  --ncc-tflite neuropilot-sdk-premium-8.0.11-build20260211/neuron_sdk/host/bin/ncc-tflite

python -u server_tools/append_p_rans_decode_custom_ops.py \
  --merged-base-model outputs/p_entropy_decoder_merged_nhwc/p_entropy_decode_merged_base_fp32.tflite \
  --entropy-package-dir outputs/gvc-rt-large_tflite_online_entropy_270p_qp0 \
  --output outputs/p_entropy_decoder_merged_rans/p_entropy_decode_merged_rans.tflite
```

打包目录不包含四张 merged+rANS 模型。这些模型单独保存到本地模型分支的
`models/large/local_models/gvc-rt-large/online_entropy/`。

## Android 部署

```powershell
$adb = '.\sdk\platform-tools\adb.exe'
$root = '/sdcard/Android/data/com.gvcrt.clean.mtkoffline/files/enterprise_tflite/large'

& $adb install -r .\app\build\outputs\apk\mtkOffline\debug\app-mtkOffline-debug.apk
& $adb push D:\path\to\unpacked-package\. $root
& $adb push .\models\large\local_models\gvc-rt-large\online_entropy\i_entropy_prior_merged_rans.tflite "$root/models/i_entropy_prior_merged_rans.tflite"
& $adb push .\models\large\local_models\gvc-rt-large\online_entropy\i_entropy_decode_merged_rans.tflite "$root/models/i_entropy_decode_merged_rans.tflite"
& $adb push .\models\large\local_models\gvc-rt-large\online_entropy\p_entropy_prior_merged_rans.tflite "$root/models/p_entropy_prior_merged_rans.tflite"
& $adb push .\models\large\local_models\gvc-rt-large\online_entropy\p_entropy_decode_merged_rans.tflite "$root/models/p_entropy_decode_merged_rans.tflite"
```

## 测试入口

```powershell
# I 帧单轮精度 dump，包含 rANS roundtrip
& $adb shell am start -n com.gvcrt.clean.mtkoffline/com.gvcrt.clean.MainActivity --ez largeIEntropyMergedPrecisionTest true

# I 帧稳态速度，不执行 roundtrip，不写 tensor dump
& $adb shell am start -n com.gvcrt.clean.mtkoffline/com.gvcrt.clean.MainActivity --ez largeIEntropyMergedSpeedTest true --ei largeIEntropyMergedWarmup 3 --ei largeIEntropyMergedMeasured 10

# merged entropy/prior 尾部接 CPU rANS custom op，新旧 payload exact 对比与速度
& $adb shell am start -n com.gvcrt.clean.mtkoffline/com.gvcrt.clean.MainActivity `
  --ez largeIEntropyRansMergedTest true `
  --ei largeIEntropyRansMergedWarmup 3 `
  --ei largeIEntropyRansMergedMeasured 10

# 完整 I -> 本地参考 -> P 编码/重建
& $adb shell am start -n com.gvcrt.clean.mtkoffline/com.gvcrt.clean.MainActivity --ez largeIpEntropyCodecTest true

# Large 统一主流程：同一输入执行 I、P(from frame)、P(from feature)，再从码流独立解码
& $adb shell am start -n com.gvcrt.clean.mtkoffline/com.gvcrt.clean.MainActivity `
  --ez largeOnlineMainTest true `
  --es imagePath asset:sample/park_scene_im00001.png `
  --ei largeOnlineWarmup 1 `
  --ei largeOnlineMeasured 1

# 先运行 I 编码生成 payload，再执行严格 FP32 串行码流解码
& $adb shell am start -n com.gvcrt.clean.mtkoffline/com.gvcrt.clean.MainActivity --ez largeIEntropyCodecTest true
& $adb shell am start -n com.gvcrt.clean.mtkoffline/com.gvcrt.clean.MainActivity --ez largeIEntropyDecodeTest true

# P 编码单图稳态速度
& $adb shell am start -n com.gvcrt.clean.mtkoffline/com.gvcrt.clean.MainActivity `
  --ez largePEntropyMergedSpeedTest true `
  --ei largePEntropyMergedWarmup 3 `
  --ei largePEntropyMergedMeasured 10

# P payload 解码精度与稳态速度
& $adb shell am start -n com.gvcrt.clean.mtkoffline/com.gvcrt.clean.MainActivity --ez largePEntropyDecodeTest true
& $adb shell am start -n com.gvcrt.clean.mtkoffline/com.gvcrt.clean.MainActivity `
  --ez largePEntropyDecodeSpeedTest true `
  --ei largePEntropyDecodeWarmup 3 `
  --ei largePEntropyDecodeMeasured 10

# 快速配置的串行解码速度：allowFp16=true、--relax-fp32
& $adb shell am start -n com.gvcrt.clean.mtkoffline/com.gvcrt.clean.MainActivity `
  --ez largeIEntropyDecodeSpeedTest true `
  --ei largeIEntropyDecodeWarmup 3 `
  --ei largeIEntropyDecodeMeasured 10

& $adb logcat -d -s GVC_RT_CLEAN:I
```

日志中 `create_ms` 是在线编译/创建时间；`large_i_codec_speed` 不包含创建时间。
`i_rans_roundtrip` 是精度诊断，不计入正式编码总耗时。I/P tensor 和 payload 输出到
应用 external files 下的 `enterprise_tflite_codec/large/`。

I 解码日志按 `i_rans_z / i_decode_init / i_rans_y_0..3 /
i_decode_stage1..3 / i_restore_y_0..3 / i_decoder / total` 分项输出。精度入口会将解码结果
与同一 Android 编码运行产生的 `z_hat、y_q_w、CDF index、y_hat、reference frame` 比较。
当前 CDF 模式固定为 `all_packed_scales_android_encoder_compatible`，用于验证 Android 自编码
payload 的闭环；与服务器 PyTorch 原生 force-zero skip 码流互通仍需单独验收，不能由该闭环替代。

## I 解码 entropy + rANS 单图变体

该模型把 `rANS z -> init prior -> rANS y0 -> stage1 -> rANS y1 -> stage2 ->
rANS y2 -> stage3 -> rANS y3 -> y_hat` 放入一张 TFLite 和一个 Interpreter。连续
prior 算子由 NeuronDelegate 处理，5 个串行 rANS custom op 保持原生 CPU。旧四图实现
已经删除，正式 `Large I decode` 入口只使用该单图。服务器生成命令见“服务器导出与打包”。

将结果放到设备 Large 包的 `models/i_entropy_decode_merged_rans.tflite` 后运行：

```powershell
& $adb shell am start -n com.gvcrt.clean.mtkoffline/com.gvcrt.clean.MainActivity `
  --ez largeIEntropyDecodeMergedTest true

& $adb shell am start -n com.gvcrt.clean.mtkoffline/com.gvcrt.clean.MainActivity `
  --ez largeIEntropyDecodeMergedSpeedTest true `
  --ei largeIEntropyDecodeMergedWarmup 3 `
  --ei largeIEntropyDecodeMergedMeasured 10
```

## Delegate 节点检查

```powershell
& $adb logcat -d TFLite:I tflite:I Neuron:I NeuroPilot:I *:S
```

当前样机实测：

- `i_encoder`：`110/110` 节点由 Delegate 接管，1 个分区。
- `i_entropy_prior_merged`：`446/455` 节点由 Delegate 接管，11 个分区；未接管的是
  5 个 `ROUND` 和 4 个 Boolean 相关 `CAST`。
- `i_decoder`：`840/840` 节点由 Delegate 接管，1 个分区。

在 warmup 3 次、正式 10 次、复用 DirectBuffer 的条件下，当前结果为：I encoder
`13.423 ms`、merged entropy `37.822 ms`、原生 rANS `5.829 ms`、I decoder
`33.402 ms`；真实串行 `i_pipeline_steady` 为 `91.492 ms`。该总时间不包含模型创建、
roundtrip 诊断和输出文件写入。

这些日志证明 TFLite Delegate 的分区情况，但不能单独证明每个 delegated partition 的
最终硬件一定是 MDLA。当前日志还显示 `Neuron adapter API exists: 0` 和 NNAPI delegate
创建信息；若需要 MDLA 硬件利用率，仍需厂商/root 性能工具补充确认。

当前 `setCacheDir/setModelToken` 没有生成缓存文件，App 冷启动仍会发生在线编译。
应保持 interpreter 在进程内复用，不能把 `create_ms` 计入稳态帧耗时。

I 解码单图在 Infinix X6891、`allowFp16=true`、`--relax-fp32`、warmup 3、正式 10 次下：
entropy+rANS `mean 32.721 ms`，I decoder `mean 36.918 ms`，总计 `mean 69.642 ms /`
`p50 64.893 ms / p90 73.041 ms`。同一 Android 编码 payload 下，`z_hat`、四阶段
符号、全部 scales/CDF index、`y_hat` 和最终重建帧均 exact。

## merged rANS 诊断结果

`i_entropy_prior_merged_rans.tflite` 在 merged entropy/prior 尾部追加
`GVC_RT_RANS_ENCODE` CPU custom op。Infinix X6891 实测 Neuron 仍接管 `446/456`
个节点，分区数由原图的 11 增至 12，说明只新增了末端 CPU 分区，没有扩大回退范围。
warmup 5、正式 50 次且输出复制不计时，`entropy/prior + rANS` 为
`mean 34.336 ms / p50 34.119 ms / p90 35.507 ms`。旧分离路径组件均值为
`37.822 + 5.829 = 43.651 ms`，减少约 `9.315 ms（21.3%）`。两条路径的
`i_y_hat` 完全一致，160 字节 rANS payload 逐字节一致。

## Large 统一主流程

正式入口 `largeOnlineMainTest` 持久创建并调用 10 张模型：6 张连续神经网络图和 I/P
编码、解码各一张 merged entropy+rANS 图。默认使用同一 ParkScene 输入执行 `I-P-P`
三帧，以同时覆盖 `temporal_from_frame` 和 `temporal_from_feature`。编码端生成 SPS、I、P、P
NAL，解码端只读取封装后的 payload，并维护独立的 Reference Frame/Feature，不复用编码端
`ctx/ctx_t`。每帧编码侧本地重建必须与独立解码结果逐字节一致，否则入口直接失败。

当前正式 I/P decoder 使用 FP16 稳定的缩放方差实现：高分辨率 GroupNorm 在计算平方前
对中心化特征乘 `1/16`，同步缩放 epsilon，避免 stage3 在 FP16 下溢出为 NaN。ParkScene
真机结果为 I `20.107 dB`、P1 `21.308 dB`、P2 `21.235 dB`，与服务器参考相差约
`0.03-0.04 dB`。warmup 2、正式 5 次时，I decoder 编码/解码均值为
`34.039/33.835 ms`，P decoder 为 `64.687/63.521 ms`，I-P-P 完整编码、封装和独立解码
均值为 `394.009 ms`。10 张模型全部调用，`status=PASS all_models_exercised=true`；首次在线创建
约 `110.3 s`，不计入稳态耗时。输出码流为
`enterprise_tflite_codec/large/main/encoded_i_p_p.gvc`，同时保存三帧解码 tensor 和最终重建图。
