# I Recon FeatureDec and Generator Inventory

## Scope

This inventory follows the I-frame reconstruction path in
`src/models/image_model_G_b.py` and `src/models/improved_model_gvc_b.py`:

```text
y_hat [1,256,16,32]
  -> FeatureDec (IntraDecoder)
  -> codeword [1,18,16,32]
  -> Generator (PretrainedReconWrapper.decoder)
  -> frame [1,3,256,512]
```

Shapes below use NCHW in PyTorch. MTK/TFLite boundaries use the equivalent NHWC
shape. Parameter counts are calculated from the source layer definitions and
include biases and affine GroupNorm parameters.

## FeatureDec

| Order | Source operation | Input -> output (NCHW) | Main operators | Parameters |
|---:|---|---|---|---:|
| 0 | `conv_in` | `[1,256,16,32] -> [1,512,16,32]` | DCB with 1x1 adaptor | 2,237,440 |
| 1 | multiply `q_dec` | `[1,512,16,32] -> same` | `MUL` | 0 |
| 2-9 | `dec_1`: 8 repetitions | `[1,512,16,32] -> same` | DCB then GroupNorm(32) | 16,855,040 |
| 10 | final activation | `[1,512,16,32] -> same` | `x * sigmoid(x)` | 0 |
| 11 | `conv_out` | `[1,512,16,32] -> [1,18,16,32]` | DCB with 1x1 adaptor | 12,132 |
| 12 | output clamp | `[1,18,16,32] -> same` | `CLAMP[-1,1]` | 0 |
|  | **FeatureDec total** |  | **10 DCB + 8 GroupNorm** | **19,104,612** |

Each DCB expands to: optional adaptor `Conv1x1`, `Conv1x1`, weighted SiLU,
depthwise `Conv3x3`, `Conv1x1`, residual add, `Conv1x1` expansion, weighted
SiLU + channel split/add, `Conv1x1` projection, residual add.

### FeatureDec deployment status

- Server full NHWC TFLite conversion passed numerical validation:
  `max_abs=9.5963478e-06` versus PyTorch.
- Official Android Neuron creation failed on the full graph because of
  unresolved `MTKEXT_SILU`.
- Splitting at the final SiLU also passed server validation
  (`max_abs=9.6559525e-06`), but the 76 MB body failed delegate preparation with
  `ANEURALNETWORKS_BAD_DATA`.
- Recommended boundary: MTK DCB subgraphs separated by native GroupNorm, final
  native SiLU, then MTK `conv_out`. This mirrors the already working P recon
  mixed-pipeline structure and avoids one oversized graph.

## Generator

The Generator uses the 18-channel codeword both as the main input and as the
conditioning tensor for every AdaptiveGroupNorm (AdaGN).

| Order | Source operation | Input -> output (NCHW) | Main operators | Parameters |
|---:|---|---|---|---:|
| 0 | `conv_in` | `[1,18,16,32] -> [1,512,16,32]` | Conv3x3 | 83,456 |
| 1 | `ada1` | feature + codeword -> `[1,512,16,32]` | variance, sqrt, mean, 2 Linear, GroupNorm, mul/add | 19,456 |
| 2 | `stage1` | `[1,512,16,32] -> same` | 4 DCB + 3 GroupNorm | 8,426,496 |
| 3 | `ada2` | same + codeword -> same | AdaGN | 19,456 |
| 4 | `stage2` | `[1,512,16,32] -> same` | 4 DCB + 3 GroupNorm | 8,426,496 |
| 5 | `ada3` | same + codeword -> same | AdaGN | 19,456 |
| 6 | `upsample` | `[1,512,16,32] -> [1,512,32,64]` | Conv3x3 512->2048 + DepthToSpace(2) | 9,439,232 |
| 7 | `stage3` | `[1,512,32,64] -> [1,320,32,64]` | 4 DCB + 3 GroupNorm | 3,464,640 |
| 8 | `ada4` | feature + codeword -> `[1,320,32,64]` | AdaGN | 12,160 |
| 9 | `stage4` | `[1,320,32,64] -> same` | 4 DCB + 3 GroupNorm + `q_recon` multiply | 3,300,480 |
| 10 | `ada_final` | feature + codeword -> same | AdaGN | 12,160 |
| 11 | `head` | `[1,320,32,64] -> [1,192,32,64]` | Conv3x3 | 553,152 |
| 12 | output shuffle/clamp | `[1,192,32,64] -> [1,3,256,512]` | PixelShuffle(8), clamp | 0 |
|  | **Generator total** |  | **16 DCB + 12 GroupNorm inside stages + 5 AdaGN** | **33,776,640** |

AdaGN has no affine parameters in its internal GroupNorm. Its parameters are
the gamma and beta `Linear(18,C)` layers. The source uses `aten::var`, which is
the known blocker for direct full-Generator MTK PyTorch conversion.

## Processing Order and MTK Boundaries

1. Run FeatureDec `conv_in` as a small NHWC MTK graph.
2. Apply `q_dec`, GroupNorm, and unsupported activation math natively.
3. Run each contiguous DCB group on MTK; keep tensors resident in NHWC where
   possible to avoid NCHW transposes between every block.
4. Run FeatureDec `conv_out` on MTK and retain the 18-channel codeword.
5. For Generator, run Conv/DCB/Upsampler/Head segments on MTK.
6. Keep AdaGN native until its variance/mean path is rewritten into a device
   supported graph. Do not change its formula for speed.
7. Measure the complete mixed FeatureDec + Generator boundary; isolated stage
   times are not sufficient because delegate synchronization and copies add
   overhead.

## Acceptance Criteria

- Server TFLite output passes the recorded continuous-tensor threshold against
  the same checkpoint and input fixture.
- Android official Neuron `create` and `invoke` both succeed; converter or NCC
  success alone is insufficient.
- The reconstructed frame is compared against the existing ONNX-canonical I
  recon output before replacing the complete encoder path.
- Speed reporting separates model creation time from warmup and measured
  per-frame inference time.
