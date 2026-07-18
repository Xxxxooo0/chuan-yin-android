package com.gvcrt.clean

import android.content.Context
import android.os.SystemClock

/** Full I FeatureDec precision and short speed probe through the official NeuronDelegate AAR. */
class IFeatureDecOfficialNeuronProbe(
    private val context: Context,
    private val emit: (String) -> Unit,
) {
    private val store = AssetStore(context)

    fun run(warmupRuns: Int = 3, measuredRuns: Int = 10) {
        require(warmupRuns >= 0 && measuredRuns > 0)
        val required = listOf(MODEL, INPUT, EXPECTED)
        val missing = required.filterNot(store::exists)
        if (missing.isNotEmpty()) {
            emit("i_featuredec_official_skip reason=missing_assets assets=${missing.joinToString(":")}")
            return
        }
        val input = store.readBytes(INPUT)
        val expected = store.readBytes(EXPECTED)
        emit(
            "i_featuredec_official_start model_sha256=${store.sha256(MODEL)} " +
                "input_sha256=${AssetStore.sha256(input)} expected_sha256=${AssetStore.sha256(expected)} " +
                "backend=official_aar_neuron allow_fp16=false warmup=$warmupRuns measured=$measuredRuns",
        )
        try {
            val createStarted = SystemClock.elapsedRealtimeNanos()
            OfficialNeuronRuntime.create(
                store.materialize(MODEL),
                context.cacheDir.resolve("i_featuredec_official_neuron_cache"),
                allowFp16ForFp32 = false,
            ).use { runtime ->
                val createMs = elapsedMs(createStarted)
                require(runtime.inputSizes.contentEquals(longArrayOf(input.size.toLong()))) {
                    "input bytes mismatch runtime=${runtime.inputSizes.joinToString(":")} expected=${input.size}"
                }
                require(runtime.outputSizes.contentEquals(longArrayOf(expected.size.toLong()))) {
                    "output bytes mismatch runtime=${runtime.outputSizes.joinToString(":")} expected=${expected.size}"
                }
                val actual = runtime.run(listOf(input)).single()
                val diff = TensorIO.diff(
                    TensorIO.readF32Le("i_featuredec.actual", OUTPUT_SHAPE, actual),
                    TensorIO.readF32Le("i_featuredec.expected", OUTPUT_SHAPE, expected),
                )
                val passed = diff.maxAbs <= TOLERANCE
                emit(
                    "i_featuredec_official_create_invoke_ok=true fully_delegated=not_reported " +
                        "options=${runtime.optionsSummary} create_ms=${format(createMs)}",
                )
                emit(
                    "i_featuredec_official_compare pass=$passed threshold=$TOLERANCE " +
                        "max_abs=${format(diff.maxAbs)} mean_abs=${format(diff.meanAbs)} " +
                        "rmse=${format(diff.rmse)} exact=${diff.exact}",
                )
                if (!passed) {
                    emit("i_featuredec_official_speed_skip reason=precision_gate_failed")
                    return
                }
                repeat(warmupRuns) { runtime.run(listOf(input), copyOutputs = false) }
                val elapsed = LongArray(measuredRuns)
                repeat(measuredRuns) { index ->
                    val started = SystemClock.elapsedRealtimeNanos()
                    runtime.run(listOf(input), copyOutputs = false)
                    elapsed[index] = SystemClock.elapsedRealtimeNanos() - started
                }
                val sorted = elapsed.sorted()
                emit(
                    "i_featuredec_official_speed mean_ms=${format(elapsed.average() / 1_000_000.0)} " +
                        "p50_ms=${format(percentile(sorted, 0.50) / 1_000_000.0)} " +
                        "p90_ms=${format(percentile(sorted, 0.90) / 1_000_000.0)} samples=$measuredRuns",
                )
            }
        } catch (error: Throwable) {
            emit(
                "i_featuredec_official_failure create_or_invoke_ok=false " +
                    "type=${error.javaClass.simpleName} message=${error.message}",
            )
        }
    }

    private fun elapsedMs(started: Long): Double =
        (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0

    private fun percentile(sorted: List<Long>, fraction: Double): Long {
        val index = ((sorted.size - 1) * fraction).toInt().coerceIn(0, sorted.lastIndex)
        return sorted[index]
    }

    private fun format(value: Double): String = "%.3f".format(java.util.Locale.US, value)
    private fun format(value: Float): String = "%.8f".format(java.util.Locale.US, value)

    companion object {
        private const val ROOT = "featuredec_i_nhwc"
        private const val MODEL = "$ROOT/i_featuredec_nhwc_fp32.tflite"
        private const val INPUT = "$ROOT/i_y_hat_nhwc.f32le"
        private const val EXPECTED = "$ROOT/i_codeword_nhwc_expected.f32le"
        private const val TOLERANCE = 5e-4f
        private val OUTPUT_SHAPE = longArrayOf(1L, 16L, 32L, 18L)
    }
}
