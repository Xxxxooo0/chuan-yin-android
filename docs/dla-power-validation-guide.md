# GVC-RT Large/Small DLA 功耗验证控制要求说明

## 1. 适用范围

本文面向 NeuroPilot 7.0.8、MDLA 5.0 离线运行环境，适用于以下 240 帧 ParkScene 交付包：

| 变体 | 交付包 | SHA256 |
|---|---|---|
| Large | `gvc-rt-large_np708_mdla50_scaled_variance_parkscene_240frames.tar.gz` | `26044fe697976998f940e5dc8261e8e415b46e2512183a78297ed46697140c7f` |
| Small | `gvc-rt-small_dla_codec_270p_qp9_np708_mdla50_parkscene_240frames.tar.gz` | `b483bae6d28722caf732bf9ae7abed8d4be913dcadc49317bd6ac60a41a501cf` |

两者均固定为 `256x512`、NHWC、FP32 I/O。Large 使用 QP0、输入范围 `[-1,1]`；Small 使用 QP9、输入范围 `[0,1]`。

交付包只包含 DLA、输入序列和 manifest，不包含可执行 runner。下文的 `<runner>` 是测试端 DLA 运行程序的占位名称，需映射到实际接口。不得把 `.dla` 直接传给只接受 `.tflite` 的 API。

## 2. 模型执行边界

### 2.1 Large

Large 共加载 6 个 DLA：

```text
frame 0:
  frame -> i_encoder -> i_y_hat -> i_decoder -> reference_frame

frame 1:
  reference_frame -> temporal_from_frame -> ctx
  frame + ctx -> p_encoder -> p_y_hat
  p_y_hat + ctx -> p_decoder -> reference_feature + reconstructed_frame

frame 2..239:
  previous reference_feature -> temporal_from_feature -> ctx
  frame + ctx -> p_encoder -> p_y_hat
  p_y_hat + ctx -> p_decoder -> next reference_feature + reconstructed_frame
```

一轮 240 帧共调用 `2 + 239 x 3 = 719` 次 DLA。功耗包不包含熵模型和 rANS，Encoder latent 直接送入 Decoder。

### 2.2 Small

Small 共加载 4 个 DLA：`temporal_from_frame`、`temporal_from_feature`、`encoder`、`decoder`。

```text
reset frame:
  reference_frame -> temporal_from_frame -> ctx + memory
other frame:
  reference_feature -> temporal_from_feature -> ctx + memory

every frame:
  frame + ctx -> encoder -> latent_y
  latent_y + ctx + memory -> decoder -> next_ref_feature + reconstructed_frame
```

初始 `reference_frame` 全部填充 `0.5`。240 帧中的 reset frame 固定为 `0、1、65、129、193`；其他帧使用上一帧 `next_ref_feature`。一轮共调用 `240 x 3 = 720` 次 DLA。

## 3. 运行状态机

测试端 runner 应维护以下状态，不应仅以进程是否存在判断模型状态：

| 状态 | 含义 |
|---|---|
| `UNLOADED` | 无模型句柄、无输入输出 buffer |
| `PREPARING` | 校验 manifest/SHA，创建 DLA 句柄并分配 buffer |
| `READY` | 全部模型已加载，但没有连续送帧 |
| `RUNNING` | 送帧循环已使能；`invoke_inflight` 表示当前是否正在执行 DLA |
| `STOPPING` | 已拒绝新帧，等待当前整帧执行链完成 |
| `ERROR` | 任一加载、输入、invoke 或输出操作返回非零状态 |

正常转换为：

```text
UNLOADED -> PREPARING -> READY -> RUNNING -> STOPPING -> READY -> UNLOADED
```

`READY` 只代表模型已加载，不代表 MDLA 正在计算。硬件仅在 invoke 期间工作，帧间可能自动掉电或降频。

## 4. 使能、停止与释放

为完成规定的功耗验证，运行程序需要提供以下等价控制能力：

```bash
<runner> prepare --variant large|small --package <解压目录>
<runner> start --mode max-throughput|paced-24fps --loop
<runner> status --json
<runner> stop --at-frame-boundary
<runner> release
```

### 4.1 `prepare`

1. 执行 `sha256sum -c SHA256SUMS.txt`。
2. 读取 `manifest.json` 和 `sequence_manifest.json`，拒绝不匹配的 shape、layout、dtype、QP 或模型数量。
3. 明确选择 MDLA 后端并加载全部 DLA；一次测试内保持句柄常驻。
4. 分配并复用 tensor buffer，禁止每帧重复创建模型。
5. 全部句柄创建成功后进入 `READY`，记录每个 DLA SHA 和加载返回码。

### 4.2 `start`

1. 清零 frame/sequence/invoke/error 计数器并初始化参考状态。
2. 先 warmup，再打印 `POWER_WINDOW_BEGIN run_id=<id> monotonic_ns=<time>`。
3. `max-throughput` 不插入等待；`paced-24fps` 以 `41.667 ms` 为帧周期，若处理已超时则不额外 sleep。
4. 每次 invoke 返回 `0` 后才更新参考状态和计数器；返回非零立即进入 `ERROR`。

### 4.3 `stop`

NeuroPilot invoke 是同步调用，不能假设存在安全的异步取消接口。停止时应：

1. 设置 `stop_requested=true`，停止接收新帧。
2. 等待当前 invoke 返回，并完成当前帧剩余 DLA 链，避免留下半更新参考状态。
3. 打印 `POWER_WINDOW_END` 和最终计数器，进入 `READY`。

若进程被强制终止，下一次启动必须重新加载模型并重置整个序列状态，不能继续使用旧参考 tensor。

### 4.4 `release`

仅在没有 invoke 执行时释放模型句柄、输入输出 buffer 和工作线程，完成后进入 `UNLOADED`。对于 TFLite Shim 路径，对应生命周期语义是 `create -> invoke -> free`；直接 DLA runner 应使用其离线 runtime 对应的 load/invoke/unload API。

## 5. 状态查询与日志

`status --json` 至少输出：

```json
{
  "state": "RUNNING",
  "variant": "small",
  "run_id": "power-001",
  "models_loaded": 4,
  "sequence_index": 3,
  "frame_index": 87,
  "stage": "decoder",
  "invoke_inflight": true,
  "invoke_success": 2427,
  "invoke_failed": 0,
  "last_return_code": 0,
  "last_invoke_ms": 4.12,
  "last_progress_monotonic_ns": 1234567890,
  "stop_requested": false
}
```

判定“模型正在正常运行”必须同时满足：

1. 软件状态为 `RUNNING`，且 progress heartbeat 持续更新。
2. invoke 成功计数持续增加、失败计数为 0。
3. NeuroPilot/APUSYS trace 在测量窗口内存在 MDLA 执行记录。

只有进程存活、模型已加载或功耗升高，都不能单独证明 MDLA 正在运行。

## 6. MDLA 运行证据

本仓库提供 `mtk/npu_systrace_v1.7_20240730/`。DLA 已带 `--gen-debug-info`，可按工具说明采集：

```bash
export MTKNN_ENABLE_PROFILER=1
export MTKNN_PROFILER_LOG_PATH=/data/local/tmp/gvc_power.prof
export MTKNN_PER_OP_PROFILE=1
export MTKNN_MDLA_PER_OP_TRACE=1
<runner> start ...
```

测量前运行 `02-trace_start_all.bat`，结束后运行 `02-trace_stop_mdla_profiling.bat`。保留 `apusys.trace`、`system.trace`、`combine.trace` 和 profiler log。trace 仅用于确认硬件路径；正式功耗轮次可关闭 per-op profiling，避免追踪开销污染结果。

## 7. 推荐功耗测试步骤

同一变体至少重复 3 轮，并固定屏幕、网络、亮度、温度、散热方式、性能模式和电池状态。

1. **系统空闲基线**：runner 未启动，采样 60 秒。
2. **模型加载空闲**：执行 `prepare` 后停在 `READY`，采样 60 秒。
3. **预热**：执行 `start`，连续运行 60 秒，不计入功耗结果。
4. **工作窗口**：连续测量 180 秒；分别测试 `max-throughput` 和 `paced-24fps`。
5. **停止保持**：执行 `stop`，确认回到 `READY`，采样 60 秒。
6. **完全释放**：执行 `release`，确认 `UNLOADED`，采样 60 秒。

功耗窗口内禁止保存逐帧 PNG、tensor 或详细 per-op 日志，只保留计数器、耗时统计和滚动 checksum。逐帧重建结果应在单独的精度轮次保存，否则 CPU、存储和图片编码功耗会污染模型数据。

## 8. 回传结果

每个变体回传：

- 包 SHA、设备型号、系统/Neuron/NeuroPilot/MDLA 版本。
- 测试模式、温控和性能策略、输入帧率、循环次数。
- 空闲/加载空闲/运行/停止/释放阶段的平均功耗、峰值功耗和总能量。
- 完成帧数、有效 FPS、mean/p50/p90、invoke 成功/失败数。
- `status` 最终 JSON、运行日志和一轮硬件 trace。
- 单独精度轮次的 240 张重建 tensor/PNG；功耗轮次只回传 checksum。

验收时首先确认无 invoke 错误、帧数和 DLA 调用数正确，再比较 Large 与 Small 的稳态平均功耗和每帧能量。
