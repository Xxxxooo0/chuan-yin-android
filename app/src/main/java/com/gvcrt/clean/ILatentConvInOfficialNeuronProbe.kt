package com.gvcrt.clean

import android.content.Context
import kotlin.math.sqrt

/** Full NHWC I latent conv_in comparison through the official NeuronDelegate AAR. */
class ILatentConvInOfficialNeuronProbe(
    private val context: Context,
    private val emit: (String) -> Unit,
) {
    private val store = AssetStore(context)

    fun run() {
        val required = listOf(MODEL, INPUT, EXPECTED)
        val missing = required.filterNot(store::exists)
        if (missing.isNotEmpty()) {
            emit("i_latent_conv_in_official_skip reason=missing_assets assets=${missing.joinToString(":")}")
            return
        }
        val input = store.readBytes(INPUT)
        val expected = store.readBytes(EXPECTED)
        emit(
            "i_latent_conv_in_official_start model_sha256=${store.sha256(MODEL)} " +
                "input_sha256=${AssetStore.sha256(input)} expected_sha256=${AssetStore.sha256(expected)} " +
                "backend=official_aar_neuron allow_fp16=false",
        )
        try {
            OfficialNeuronRuntime.create(
                store.materialize(MODEL),
                context.cacheDir.resolve("i_latent_conv_in_official_neuron_cache"),
                allowFp16ForFp32 = false,
            ).use { runtime ->
                require(runtime.inputSizes.contentEquals(longArrayOf(input.size.toLong()))) {
                    "input bytes mismatch runtime=${runtime.inputSizes.joinToString(":")} expected=${input.size}"
                }
                val actual = runtime.run(listOf(input)).single()
                require(actual.size == expected.size) {
                    "output bytes mismatch actual=${actual.size} expected=${expected.size}"
                }
                val diff = TensorIO.diff(
                    TensorIO.readF32Le("i_latent_actual", OUTPUT_SHAPE, actual),
                    TensorIO.readF32Le("i_latent_expected", OUTPUT_SHAPE, expected),
                )
                emit("i_latent_conv_in_official_create_invoke_ok=true options=${runtime.optionsSummary}")
                emit(
                    "i_latent_conv_in_official_compare pass=${diff.maxAbs <= TOLERANCE} threshold=$TOLERANCE " +
                        "max_abs=${format(diff.maxAbs)} mean_abs=${format(diff.meanAbs)} " +
                        "rmse=${format(diff.rmse)} cosine=${format(cosine(actual, expected))} exact=${diff.exact}",
                )
            }
        } catch (error: Throwable) {
            emit(
                "i_latent_conv_in_official_failure create_or_invoke_ok=false " +
                    "type=${error.javaClass.simpleName} message=${error.message}",
            )
        }
    }

    private fun cosine(actualBytes: ByteArray, expectedBytes: ByteArray): Float {
        val actual = TensorIO.readF32Le("actual", OUTPUT_SHAPE, actualBytes).data
        val expected = TensorIO.readF32Le("expected", OUTPUT_SHAPE, expectedBytes).data
        var dot = 0.0
        var actualNorm = 0.0
        var expectedNorm = 0.0
        actual.indices.forEach { index ->
            dot += actual[index].toDouble() * expected[index]
            actualNorm += actual[index].toDouble() * actual[index]
            expectedNorm += expected[index].toDouble() * expected[index]
        }
        return if (actualNorm == 0.0 || expectedNorm == 0.0) 0f else (dot / sqrt(actualNorm * expectedNorm)).toFloat()
    }

    private fun format(value: Float): String = "%.8f".format(java.util.Locale.US, value)

    companion object {
        private const val MODEL = "recon_diagnostic/i_latent_conv_in_nhwc_fp32.tflite"
        private const val INPUT = "recon_dissect_i_nhwc/input_i_y_hat_nhwc.f32le"
        private const val EXPECTED = "recon_dissect_i_nhwc/full_output_i_dec_stage0_nhwc.f32le"
        private const val TOLERANCE = 5e-4f
        private val OUTPUT_SHAPE = longArrayOf(1L, 16L, 32L, 512L)
    }
}
