package com.gvcrt.clean

import android.content.Context
import android.os.SystemClock
import kotlin.math.exp

/** I FeatureDec split into two official Neuron graphs around the unsupported SiLU. */
class IFeatureDecSplitOfficialNeuronProbe(
    private val context: Context,
    private val emit: (String) -> Unit,
) {
    private val store = AssetStore(context)

    fun run(
        warmupRuns: Int = 3,
        measuredRuns: Int = 10,
        allowFp16ForFp32: Boolean = false,
    ) {
        val required = listOf(BODY_MODEL, TAIL_MODEL, INPUT, BODY_EXPECTED, OUTPUT_EXPECTED)
        val missing = required.filterNot(store::exists)
        if (missing.isNotEmpty()) {
            emit("i_featuredec_split_skip reason=missing_assets assets=${missing.joinToString(":")}")
            return
        }
        val input = store.readBytes(INPUT)
        val bodyExpected = store.readBytes(BODY_EXPECTED)
        val outputExpected = store.readBytes(OUTPUT_EXPECTED)
        emit(
            "i_featuredec_split_start body_sha256=${store.sha256(BODY_MODEL)} tail_sha256=${store.sha256(TAIL_MODEL)} " +
                "input_sha256=${AssetStore.sha256(input)} output_sha256=${AssetStore.sha256(outputExpected)} " +
                "backend=official_aar_neuron allow_fp16=$allowFp16ForFp32 warmup=$warmupRuns measured=$measuredRuns",
        )
        try {
            val createStarted = SystemClock.elapsedRealtimeNanos()
            OfficialNeuronRuntime.create(
                store.materialize(BODY_MODEL),
                context.cacheDir.resolve("i_featuredec_body_official_cache"),
                allowFp16ForFp32 = allowFp16ForFp32,
            ).use { body ->
                emit("i_featuredec_split_body_create_ok=true options=${body.optionsSummary}")
                OfficialNeuronRuntime.create(
                    store.materialize(TAIL_MODEL),
                    context.cacheDir.resolve("i_featuredec_tail_official_cache"),
                    allowFp16ForFp32 = allowFp16ForFp32,
                ).use { tail ->
                    emit(
                        "i_featuredec_split_create_ok=true fully_delegated=not_reported " +
                            "body_options=${body.optionsSummary} tail_options=${tail.optionsSummary} " +
                            "create_ms=${format(elapsedMs(createStarted))}",
                    )
                    val bodyActual = body.run(listOf(input)).single()
                    emitComparison("body", BODY_SHAPE, bodyActual, bodyExpected)
                    val activated = nativeSilu(bodyActual)
                    val outputActual = tail.run(listOf(activated)).single()
                    val outputDiff = emitComparison("codeword", OUTPUT_SHAPE, outputActual, outputExpected)
                    if (outputDiff.maxAbs > TOLERANCE) {
                        emit("i_featuredec_split_speed_skip reason=precision_gate_failed")
                        return
                    }
                    repeat(warmupRuns) { runPipeline(body, tail, input) }
                    val total = LongArray(measuredRuns)
                    var bodyNs = 0L
                    var siluNs = 0L
                    var tailNs = 0L
                    repeat(measuredRuns) { index ->
                        val totalStarted = SystemClock.elapsedRealtimeNanos()
                        val bodyStarted = SystemClock.elapsedRealtimeNanos()
                        val bodyOutput = body.run(listOf(input)).single()
                        bodyNs += SystemClock.elapsedRealtimeNanos() - bodyStarted
                        val siluStarted = SystemClock.elapsedRealtimeNanos()
                        val siluOutput = nativeSilu(bodyOutput)
                        siluNs += SystemClock.elapsedRealtimeNanos() - siluStarted
                        val tailStarted = SystemClock.elapsedRealtimeNanos()
                        tail.run(listOf(siluOutput), copyOutputs = false)
                        tailNs += SystemClock.elapsedRealtimeNanos() - tailStarted
                        total[index] = SystemClock.elapsedRealtimeNanos() - totalStarted
                    }
                    val sorted = total.sorted()
                    emit(
                        "i_featuredec_split_speed body_mean_ms=${format(bodyNs / measuredRuns / 1_000_000.0)} " +
                            "native_silu_mean_ms=${format(siluNs / measuredRuns / 1_000_000.0)} " +
                            "tail_mean_ms=${format(tailNs / measuredRuns / 1_000_000.0)} " +
                            "total_mean_ms=${format(total.average() / 1_000_000.0)} " +
                            "p50_ms=${format(percentile(sorted, 0.50) / 1_000_000.0)} " +
                            "p90_ms=${format(percentile(sorted, 0.90) / 1_000_000.0)} samples=$measuredRuns",
                    )
                }
            }
        } catch (error: Throwable) {
            emit("i_featuredec_split_failure type=${error.javaClass.simpleName} message=${error.message}")
        }
    }

    private fun runPipeline(body: OfficialNeuronRuntime, tail: OfficialNeuronRuntime, input: ByteArray) {
        val activated = nativeSilu(body.run(listOf(input)).single())
        tail.run(listOf(activated), copyOutputs = false)
    }

    private fun nativeSilu(bytes: ByteArray): ByteArray {
        val tensor = TensorIO.readF32Le("i_featuredec_pre_silu", BODY_SHAPE, bytes)
        for (index in tensor.data.indices) {
            val value = tensor.data[index]
            tensor.data[index] = (value / (1.0 + exp(-value.toDouble()))).toFloat()
        }
        return TensorIO.f32Le(tensor)
    }

    private fun emitComparison(name: String, shape: LongArray, actual: ByteArray, expected: ByteArray): TensorDiff {
        val diff = TensorIO.diff(
            TensorIO.readF32Le("$name.actual", shape, actual),
            TensorIO.readF32Le("$name.expected", shape, expected),
        )
        emit(
            "i_featuredec_split_compare output=$name pass=${diff.maxAbs <= TOLERANCE} threshold=$TOLERANCE " +
                "max_abs=${format(diff.maxAbs)} mean_abs=${format(diff.meanAbs)} rmse=${format(diff.rmse)} exact=${diff.exact}",
        )
        return diff
    }

    private fun elapsedMs(started: Long): Double = (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0
    private fun percentile(sorted: List<Long>, fraction: Double): Long =
        sorted[((sorted.size - 1) * fraction).toInt().coerceIn(0, sorted.lastIndex)]
    private fun format(value: Double): String = "%.3f".format(java.util.Locale.US, value)
    private fun format(value: Float): String = "%.8f".format(java.util.Locale.US, value)

    companion object {
        private const val ROOT = "featuredec_i_split_nhwc"
        private const val BODY_MODEL = "$ROOT/i_featuredec_body_nhwc_fp32.tflite"
        private const val TAIL_MODEL = "$ROOT/i_featuredec_tail_nhwc_fp32.tflite"
        private const val INPUT = "$ROOT/i_y_hat_nhwc.f32le"
        private const val BODY_EXPECTED = "$ROOT/i_featuredec_body_nhwc_expected.f32le"
        private const val OUTPUT_EXPECTED = "$ROOT/i_codeword_nhwc_expected.f32le"
        private const val TOLERANCE = 5e-4f
        private val BODY_SHAPE = longArrayOf(1L, 16L, 32L, 512L)
        private val OUTPUT_SHAPE = longArrayOf(1L, 16L, 32L, 18L)
    }
}
