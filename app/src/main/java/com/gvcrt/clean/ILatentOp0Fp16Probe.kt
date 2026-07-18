package com.gvcrt.clean

import android.content.Context
import kotlin.math.sqrt

/** Minimal FP16-I/O Online Compile probe for the first I latent Conv2D. */
class ILatentOp0Fp16Probe(
    private val context: Context,
    private val emit: (String) -> Unit,
) {
    private val store = AssetStore(context)

    fun run(accelerationMode: Int) {
        val modelAsset = "$ROOT/i_latent_op0_nhwc_fp16.tflite"
        val expectedInputAsset = "$ROOT/i_y_hat_nhwc.f16le"
        val expectedOutputAsset = "$ROOT/i_op0_output_nhwc.f16le"
        val requiredAssets = listOf(modelAsset, expectedInputAsset, expectedOutputAsset, BASELINE_INPUT)
        val missing = requiredAssets.filterNot(store::exists)
        if (missing.isNotEmpty()) {
            emit("i_latent_op0_fp16_skip reason=missing_assets assets=${missing.joinToString(":")}")
            return
        }

        val input = makeNhwcFp16Input()
        val expectedInput = store.readBytes(expectedInputAsset)
        val expectedOutput = store.readBytes(expectedOutputAsset)
        emit(
            "i_latent_op0_fp16_start backend=${backendLabel(accelerationMode)} " +
                "model_sha256=${store.sha256(modelAsset)} " +
                "input_sha256=${AssetStore.sha256(input)} " +
                "server_input_sha256=${AssetStore.sha256(expectedInput)} " +
                "expected_output_sha256=${AssetStore.sha256(expectedOutput)} " +
                "input_exact=${input.contentEquals(expectedInput)}",
        )
        if (!input.contentEquals(expectedInput)) {
            emit("i_latent_op0_fp16_skip reason=android_f32_to_f16_mismatch")
            return
        }

        try {
            MtkTfliteRuntime.create(
                store.materialize(modelAsset),
                accelerationMode = accelerationMode,
                cacheDir = context.cacheDir.resolve("i_latent_op0_fp16_cache"),
                allowFp16ForFp32 = false,
            ).use { runtime ->
                emit(
                    "i_latent_op0_fp16_create backend=${backendLabel(accelerationMode)} create_ok=true " +
                        "fully_delegated=${runtime.fullyDelegated} input_bytes=${runtime.inputSizes.joinToString(":")} " +
                        "output_bytes=${runtime.outputSizes.joinToString(":")} options=${runtime.optionsSummary}",
                )
                require(runtime.inputSizes.contentEquals(longArrayOf(input.size.toLong()))) {
                    "unexpected input bytes=${runtime.inputSizes.joinToString(":")}, expected=${input.size}"
                }
                val outputs = runtime.run(listOf(input))
                require(outputs.size == 1) { "unexpected output count=${outputs.size}" }
                require(outputs[0].size == expectedOutput.size) {
                    "unexpected output bytes=${outputs[0].size}, expected=${expectedOutput.size}"
                }
                emitDiff(outputs[0], expectedOutput)
                emit("i_latent_op0_fp16_complete backend=${backendLabel(accelerationMode)} invoke_ok=true")
            }
        } catch (error: Throwable) {
            emit(
                "i_latent_op0_fp16_failure backend=${backendLabel(accelerationMode)} create_or_invoke_ok=false " +
                    "type=${error.javaClass.simpleName} message=${error.message}",
            )
        }
    }

    private fun makeNhwcFp16Input(): ByteArray {
        val nchw = TensorIO.readF32Le("i_y_hat", NCHW_SHAPE, store.readBytes(BASELINE_INPUT)).data
        val nhwc = FloatArray(nchw.size)
        for (height in 0 until HEIGHT) {
            for (width in 0 until WIDTH) {
                for (channel in 0 until CHANNELS) {
                    val source = (channel * HEIGHT + height) * WIDTH + width
                    val target = (height * WIDTH + width) * CHANNELS + channel
                    nhwc[target] = nchw[source]
                }
            }
        }
        return TensorIO.f16Le(TensorValue("i_y_hat_nhwc_fp16", NHWC_INPUT_SHAPE, nhwc))
    }

    private fun emitDiff(actualBytes: ByteArray, expectedBytes: ByteArray) {
        val actual = TensorIO.readF16Le("i_op0_actual", NHWC_OUTPUT_SHAPE, actualBytes)
        val expected = TensorIO.readF16Le("i_op0_expected", NHWC_OUTPUT_SHAPE, expectedBytes)
        val diff = TensorIO.diff(actual, expected)
        emit(
            "i_latent_op0_fp16_compare pass=${diff.maxAbs <= TOLERANCE} threshold=$TOLERANCE " +
                "max_abs=${format(diff.maxAbs)} mean_abs=${format(diff.meanAbs)} " +
                "rmse=${format(diff.rmse)} cosine=${format(cosine(actual.data, expected.data))} exact=${diff.exact}",
        )
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

    private fun backendLabel(mode: Int): String = when (mode) {
        MtkTfliteRuntime.ACCELERATION_CPU -> "mtk_cpu"
        MtkTfliteRuntime.ACCELERATION_NEURON -> "mtk_neuron"
        else -> "unknown_$mode"
    }

    private fun format(value: Float): String = "%.8f".format(java.util.Locale.US, value)

    companion object {
        private const val ROOT = "recon_fp16_i_latent"
        private const val BASELINE_INPUT = "baseline/tensors/i_y_hat.f32le"
        private const val CHANNELS = 256
        private const val HEIGHT = 16
        private const val WIDTH = 32
        private const val TOLERANCE = 5e-3f
        private val NCHW_SHAPE = longArrayOf(1, CHANNELS.toLong(), HEIGHT.toLong(), WIDTH.toLong())
        private val NHWC_INPUT_SHAPE = longArrayOf(1, HEIGHT.toLong(), WIDTH.toLong(), CHANNELS.toLong())
        private val NHWC_OUTPUT_SHAPE = longArrayOf(1L, HEIGHT.toLong(), WIDTH.toLong(), 512L)
    }
}
