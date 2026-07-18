package com.gvcrt.clean

import android.content.Context
import android.os.SystemClock
import java.io.File
import java.util.Locale

class PEncoderPriorTfliteDiagnostic(
    context: Context,
    private val emit: (String) -> Unit,
) {
    private val store = AssetStore(context)
    private val cacheDir = File(context.cacheDir, "p_prior_tflite_diagnostic")

    fun run() {
        val asset = "$ASSET_DIR/p_prior_stage1_fp16_weight.tflite"
        if (!store.exists(asset)) {
            emit("p_prior_tflite_probe_result create_or_invoke_ok=false reason=missing_asset asset=$asset")
            return
        }
        val yHat = tensor("p_y_hat_so_far_0", Y_SHAPE)
        val common = tensor("p_common_params", COMMON_SHAPE)
        val sourceY = tensor("p_y_pre_prior", Y_SHAPE)
        emit("p_prior_tflite_probe_start mode=NEURON accelerator=AUTO io=float32 stage=stage1 warmup=5 measured=50")
        try {
            MtkTfliteRuntime.create(
                store.materialize(asset),
                accelerationMode = MtkTfliteRuntime.ACCELERATION_NEURON,
                acceleratorFlag = MtkTfliteRuntime.ACCELERATOR_AUTO,
                cacheDir = cacheDir,
            ).use { runtime ->
                emit(
                    "p_prior_tflite_create asset=$asset sha256=${store.sha256(asset)} create_ok=true " +
                        "fully_delegated=${runtime.fullyDelegated} options=${runtime.optionsSummary}",
                )
                val inputs = listOf(
                    PriorNpuTensorCodec.nchwToNhwcF32(yHat.data, 128, 16, 32),
                    PriorNpuTensorCodec.nchwToNhwcF32(common.data, 384, 16, 32),
                )
                val outputs = runtime.run(inputs, copyOutputs = true)
                emit("p_prior_tflite_invoke asset=$asset invoke_ok=true output_bytes=${outputs.joinToString(prefix = "[", postfix = "]") { it.size.toString() }}")
                emitPrecision(outputs, sourceY, common)
                repeat(WARMUP_RUNS) { runtime.run(inputs, copyOutputs = false) }
                val elapsed = LongArray(MEASURED_RUNS)
                repeat(MEASURED_RUNS) { index ->
                    val started = SystemClock.elapsedRealtimeNanos()
                    runtime.run(inputs, copyOutputs = false)
                    elapsed[index] = SystemClock.elapsedRealtimeNanos() - started
                }
                emitSpeed(elapsed)
            }
        } catch (error: Throwable) {
            emit(
                "p_prior_tflite_failure asset=$asset create_or_invoke_ok=false " +
                    "type=${error.javaClass.simpleName} message=${error.message}",
            )
        }
    }

    private fun tensor(name: String, shape: LongArray): TensorValue =
        TensorIO.readF32Le(name, shape, store.readBytes("baseline/tensors/$name.f32le"))

    private fun emitPrecision(outputs: List<ByteArray>, sourceY: TensorValue, common: TensorValue) {
        require(outputs.size == 2) { "P prior stage1 returned ${outputs.size} outputs" }
        val scales = PriorNpuTensorCodec.nhwcF32ToNchw(outputs[0], 128, 16, 32)
        val means = PriorNpuTensorCodec.nhwcF32ToNchw(outputs[1], 128, 16, 32)
        val expectedScales = tensor("p_s_w_1", STAGE_SHAPE)
        val expectedSymbols = tensor("p_y_q_w_1", STAGE_SHAPE)
        val actualScales = TensorValue("p_s_w_1", STAGE_SHAPE, packMasked(scales))
        val actualSymbols = TensorValue("p_y_q_w_1", STAGE_SHAPE, quantizeAndPack(sourceY.data, common.data, scales, means))
        emitDiff("scales", actualScales, expectedScales)
        emitDiff("symbols", actualSymbols, expectedSymbols)
        val cdfExact = EntropySymbols.indexesForScales(actualScales).contentEquals(
            EntropySymbols.indexesForScales(expectedScales),
        )
        emit("p_prior_tflite_precision kind=cdf_indexes exact=$cdfExact")
    }

    private fun packMasked(values: FloatArray): FloatArray {
        val packed = FloatArray(64 * 16 * 32)
        for (channel in 0 until 64) {
            for (row in 0 until 16) {
                for (column in 0 until 32) {
                    val index = index(channel, row, column)
                    packed[index] = values[index] * mask1(channel, row, column) +
                        values[index(64 + channel, row, column)] * mask1(64 + channel, row, column)
                }
            }
        }
        return packed
    }

    private fun quantizeAndPack(y: FloatArray, common: FloatArray, scales: FloatArray, means: FloatArray): FloatArray {
        val quantized = FloatArray(128 * 16 * 32)
        for (channel in 0 until 128) {
            for (row in 0 until 16) {
                for (column in 0 until 32) {
                    val index = index(channel, row, column)
                    val mask = mask1(channel, row, column)
                    val qDec = maxOf(common[index], 0.5f)
                    val residual = (y[index] / qDec - means[index] * mask) * mask
                    var symbol = Math.rint(residual.toDouble()).toFloat()
                    if (scales[index] * mask <= FORCE_ZERO_THRESHOLD) symbol = 0f
                    quantized[index] = symbol.coerceIn(-128f, 127f)
                }
            }
        }
        return packMasked(quantized)
    }

    private fun mask1(channel: Int, row: Int, column: Int): Float {
        val firstHalf = channel < 64
        val evenParity = (row + column) % 2 == 0
        return if (firstHalf) {
            if (evenParity) 0f else 1f
        } else {
            if (evenParity) 1f else 0f
        }
    }

    private fun index(channel: Int, row: Int, column: Int): Int = (channel * 16 + row) * 32 + column

    private fun emitDiff(kind: String, actual: TensorValue, expected: TensorValue) {
        val diff = TensorIO.diff(actual, expected)
        emit(
            "p_prior_tflite_precision kind=$kind exact=${diff.exact} max_abs=${format(diff.maxAbs)} " +
                "mean_abs=${format(diff.meanAbs)} rmse=${format(diff.rmse)} first_diff=${firstDifference(actual.data, expected.data)}",
        )
    }

    private fun emitSpeed(values: LongArray) {
        val ordered = values.sortedArray()
        emit(
            "p_prior_tflite_speed stage=stage1 samples=${values.size} mean_ms=${formatMs(values.average() / 1_000_000.0)} " +
                "p50_ms=${formatMs(ordered[(ordered.size - 1) / 2] / 1_000_000.0)} " +
                "p90_ms=${formatMs(ordered[((ordered.size - 1) * 90) / 100] / 1_000_000.0)}",
        )
    }

    private fun firstDifference(actual: FloatArray, expected: FloatArray): Int =
        actual.indices.firstOrNull { actual[it] != expected[it] } ?: -1

    private fun format(value: Float): String = String.format(Locale.US, "%.8f", value)
    private fun formatMs(value: Double): String = String.format(Locale.US, "%.3f", value)

    companion object {
        private const val ASSET_DIR = "prior_npu_diagnostic"
        private val Y_SHAPE = longArrayOf(1, 128, 16, 32)
        private val COMMON_SHAPE = longArrayOf(1, 384, 16, 32)
        private val STAGE_SHAPE = longArrayOf(1, 64, 16, 32)
        private const val FORCE_ZERO_THRESHOLD = 0.12f
        private const val WARMUP_RUNS = 5
        private const val MEASURED_RUNS = 50
    }
}
