package com.gvcrt.clean

import android.content.Context
import android.os.SystemClock

/** Validates source-derived full I recon TFLite through the official MTK NeuronDelegate AAR. */
class IReconOfficialNeuronProbe(
    private val context: Context,
    private val emit: (String) -> Unit,
) {
    private val store = AssetStore(context)

    fun run() {
        val required = listOf(MODEL, INPUT, CODEWORD, FRAME)
        val missing = required.filterNot(store::exists)
        if (missing.isNotEmpty()) {
            emit("i_recon_official_skip reason=missing_assets assets=${missing.joinToString(":")}")
            return
        }
        val input = store.readBytes(INPUT)
        val expectedCodeword = store.readBytes(CODEWORD)
        val expectedFrame = store.readBytes(FRAME)
        emit(
            "i_recon_official_start model_sha256=${store.sha256(MODEL)} input_sha256=${AssetStore.sha256(input)} " +
                "codeword_sha256=${AssetStore.sha256(expectedCodeword)} frame_sha256=${AssetStore.sha256(expectedFrame)} " +
                "backend=official_aar_neuron allow_fp16=false",
        )
        try {
            val createStarted = SystemClock.elapsedRealtimeNanos()
            OfficialNeuronRuntime.create(
                store.materialize(MODEL),
                context.cacheDir.resolve("i_recon_official_neuron_cache"),
                allowFp16ForFp32 = false,
            ).use { runtime ->
                val createMs = (SystemClock.elapsedRealtimeNanos() - createStarted) / 1_000_000.0
                require(runtime.inputSizes.contentEquals(longArrayOf(input.size.toLong()))) {
                    "input bytes mismatch runtime=${runtime.inputSizes.joinToString(":")} expected=${input.size}"
                }
                val invokeStarted = SystemClock.elapsedRealtimeNanos()
                val outputs = runtime.run(listOf(input))
                val invokeMs = (SystemClock.elapsedRealtimeNanos() - invokeStarted) / 1_000_000.0
                require(outputs.size == 2) { "output count mismatch runtime=${outputs.size} expected=2" }
                val codewordActual = outputs.singleOrNull { it.size == expectedCodeword.size }
                    ?: error("missing codeword output size=${expectedCodeword.size} actual=${outputs.map(ByteArray::size)}")
                val frameActual = outputs.singleOrNull { it.size == expectedFrame.size }
                    ?: error("missing frame output size=${expectedFrame.size} actual=${outputs.map(ByteArray::size)}")
                emit(
                    "i_recon_official_create_invoke_ok=true options=${runtime.optionsSummary} " +
                        "create_ms=${format(createMs)} invoke_ms=${format(invokeMs)} output_sizes=${outputs.joinToString(":") { it.size.toString() }}",
                )
                emitComparison("i_codeword", CODEWORD_SHAPE, codewordActual, expectedCodeword)
                emitComparison("encoder_i_reference_frame", FRAME_SHAPE, frameActual, expectedFrame)
            }
        } catch (error: Throwable) {
            emit(
                "i_recon_official_failure create_or_invoke_ok=false type=${error.javaClass.simpleName} " +
                    "message=${error.message}",
            )
        }
    }

    private fun emitComparison(name: String, shape: LongArray, actual: ByteArray, expected: ByteArray) {
        val diff = TensorIO.diff(
            TensorIO.readF32Le("$name.actual", shape, actual),
            TensorIO.readF32Le("$name.expected", shape, expected),
        )
        emit(
            "i_recon_official_compare output=$name pass=${diff.maxAbs <= TOLERANCE} threshold=$TOLERANCE " +
                "max_abs=${format(diff.maxAbs)} mean_abs=${format(diff.meanAbs)} rmse=${format(diff.rmse)} exact=${diff.exact}",
        )
    }

    private fun format(value: Double): String = "%.3f".format(java.util.Locale.US, value)
    private fun format(value: Float): String = "%.8f".format(java.util.Locale.US, value)

    companion object {
        private const val ROOT = "recon_i_nhwc"
        private const val MODEL = "$ROOT/i_recon_nhwc_fp32.tflite"
        private const val INPUT = "$ROOT/i_y_hat_nhwc.f32le"
        private const val CODEWORD = "$ROOT/i_codeword_nhwc_expected.f32le"
        private const val FRAME = "$ROOT/encoder_i_reference_frame_nhwc_expected.f32le"
        private const val TOLERANCE = 5e-4f
        private val CODEWORD_SHAPE = longArrayOf(1L, 16L, 32L, 18L)
        private val FRAME_SHAPE = longArrayOf(1L, 256L, 512L, 3L)
    }
}
