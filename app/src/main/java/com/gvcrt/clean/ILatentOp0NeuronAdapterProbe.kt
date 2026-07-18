package com.gvcrt.clean

import android.content.Context
import kotlin.math.sqrt

/** Direct Neuron Adapter FP16 Conv2D probe; intentionally bypasses TFLite. */
class ILatentOp0NeuronAdapterProbe(
    context: Context,
    private val emit: (String) -> Unit,
) {
    private val store = AssetStore(context)

    fun run() {
        val input = "$ROOT/i_y_hat_nhwc.f16le"
        val weights = "$ROOT/i_op0_weight_ohwi.f16le"
        val bias = "$ROOT/i_op0_bias.f16le"
        val expected = "$ROOT/i_op0_output_nhwc.f16le"
        val missing = listOf(input, weights, bias, expected).filterNot(store::exists)
        if (missing.isNotEmpty()) {
            emit("i_latent_op0_adapter_fp16_skip reason=missing_assets assets=${missing.joinToString(":")}")
            return
        }
        val inputBytes = store.readBytes(input)
        val weightBytes = store.readBytes(weights)
        val biasBytes = store.readBytes(bias)
        val expectedBytes = store.readBytes(expected)
        emit(
            "i_latent_op0_adapter_fp16_start input_sha256=${AssetStore.sha256(inputBytes)} " +
                "weight_sha256=${AssetStore.sha256(weightBytes)} bias_sha256=${AssetStore.sha256(biasBytes)} " +
                "expected_sha256=${AssetStore.sha256(expectedBytes)}",
        )
        try {
            val result = MtkTfliteRuntime.runNeuronAdapterFp16Conv(inputBytes, weightBytes, biasBytes)
            require(result.size == 2) { "unexpected native result size=${result.size}" }
            val actualBytes = result[0]
            val nativeStatus = result[1].toString(Charsets.UTF_8)
            require(actualBytes.size == expectedBytes.size) {
                "unexpected Adapter output bytes=${actualBytes.size}, expected=${expectedBytes.size}"
            }
            val actual = TensorIO.readF16Le("adapter_actual", OUTPUT_SHAPE, actualBytes)
            val reference = TensorIO.readF16Le("adapter_expected", OUTPUT_SHAPE, expectedBytes)
            val diff = TensorIO.diff(actual, reference)
            emit("i_latent_op0_adapter_fp16_create_invoke_ok=true $nativeStatus")
            emit(
                "i_latent_op0_adapter_fp16_compare pass=${diff.maxAbs <= TOLERANCE} threshold=$TOLERANCE " +
                    "max_abs=${format(diff.maxAbs)} mean_abs=${format(diff.meanAbs)} " +
                    "rmse=${format(diff.rmse)} cosine=${format(cosine(actual.data, reference.data))} exact=${diff.exact}",
            )
        } catch (error: Throwable) {
            emit(
                "i_latent_op0_adapter_fp16_failure create_or_invoke_ok=false " +
                    "type=${error.javaClass.simpleName} message=${error.message}",
            )
        }
    }

    private fun cosine(left: FloatArray, right: FloatArray): Float {
        var dot = 0.0
        var leftNorm = 0.0
        var rightNorm = 0.0
        left.indices.forEach { index ->
            dot += left[index].toDouble() * right[index]
            leftNorm += left[index].toDouble() * left[index]
            rightNorm += right[index].toDouble() * right[index]
        }
        return if (leftNorm == 0.0 || rightNorm == 0.0) 0f else (dot / sqrt(leftNorm * rightNorm)).toFloat()
    }

    private fun format(value: Float): String = "%.8f".format(java.util.Locale.US, value)

    companion object {
        private const val ROOT = "recon_adapter_i_op0"
        private const val TOLERANCE = 5e-3f
        private val OUTPUT_SHAPE = longArrayOf(1L, 16L, 32L, 512L)
    }
}
