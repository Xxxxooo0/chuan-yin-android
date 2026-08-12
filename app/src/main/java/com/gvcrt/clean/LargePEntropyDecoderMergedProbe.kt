package com.gvcrt.clean

import android.content.Context
import android.os.SystemClock
import com.mediatek.neuropilot_V.neuron.NeuronDelegate
import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest
import java.util.Locale

/** Validates and benchmarks P payload -> merged entropy/rANS decode -> P decoder. */
class LargePEntropyDecoderMergedProbe(
    private val context: Context,
    private val emit: (String) -> Unit,
) {
    fun run(
        warmupRuns: Int = 0,
        measuredRuns: Int = 1,
        fastRelaxFp32: Boolean = true,
        validatePrecision: Boolean = true,
    ) {
        require(warmupRuns >= 0 && measuredRuns > 0)
        val packageRoot = findPackageRoot()
        val encoderOutput = context.getExternalFilesDir(null)!!.resolve("enterprise_tflite_codec/large/p")
        val payloadFile = encoderOutput.resolve("p_rans_payload.bin")
        val ctxTFile = encoderOutput.resolve("p_ctx_t.nchw.f32le")
        val ctxFile = encoderOutput.resolve("p_ctx.nchw.f32le")
        require(payloadFile.isFile && ctxTFile.isFile && ctxFile.isFile) {
            "missing P encoder output; run largePEntropyCodecTest first"
        }
        val payload = payloadFile.readBytes()
        val ctxT = TensorIO.readF32Le("p_ctx_t", CTX_SHAPE, ctxTFile.readBytes())
        val ctx = TensorIO.readF32Le("p_ctx", CTX_SHAPE, ctxFile.readBytes())
        val mergedModel = findModel(packageRoot, MERGED_MODEL)
        val decoderModel = packageRoot.resolve("models/p_decoder.tflite")
        require(decoderModel.isFile) { "missing P decoder: ${decoderModel.absolutePath}" }

        val createStarted = SystemClock.elapsedRealtimeNanos()
        PEntropyRansDecodeMergedRuntime.create(
            mergedModel,
            context.cacheDir.resolve("enterprise_tflite/large/p_entropy_decode_merged"),
            fastRelaxFp32,
        ).use { merged ->
            val mergedCreateMs = elapsedMs(createStarted)
            createDecoder(decoderModel, fastRelaxFp32).use { decoder ->
                emit(
                    "large_p_decode_merged_start mode=${if (fastRelaxFp32) "fast_relax_fp32" else "strict_fp32"} " +
                        "warmup=$warmupRuns measured=$measuredRuns payload_bytes=${payload.size} " +
                        "payload_sha256=${sha256(payload)} model_sha256=${sha256(mergedModel)} " +
                        "merged_create_ms=${format(mergedCreateMs)} options=${merged.optionsSummary}",
                )
                val ctxTNhwc = NhwcTensorCodec.toF32Le(ctxT)
                val ctxNhwc = NhwcTensorCodec.toF32Le(ctx)
                repeat(warmupRuns) { runCanonical(payload, ctxTNhwc, ctxNhwc, merged, decoder) }
                val measured = ArrayList<RunResult>(measuredRuns)
                repeat(measuredRuns) {
                    measured += runCanonical(payload, ctxTNhwc, ctxNhwc, merged, decoder)
                }
                emitStats(measured)

                var passed = true
                if (validatePrecision) {
                    val raw = merged.run(payload, ctxTNhwc)
                    require(raw.size == OUTPUT_SPECS.size) {
                        "P merged decoder outputs=${raw.size}, expected=${OUTPUT_SPECS.size}"
                    }
                    val outputs = OUTPUT_SPECS.mapIndexed { index, spec ->
                        NhwcTensorCodec.fromF32Le(spec.name, spec.shape, raw[index])
                    }
                    passed = compareOutputs(encoderOutput, outputs)
                    val decoded = runDecoder(decoder, raw.last(), ctxNhwc)
                    passed = compareTensor(encoderOutput, "p_reference_feature", decoded[0], false) && passed
                    passed = compareTensor(encoderOutput, "p_reference_frame", decoded[1], false) && passed
                }
                emit(
                    "large_p_decode_merged_complete status=${if (passed) "PASS" else "FAIL"} " +
                        "precision_checked=$validatePrecision payload_sha256=${sha256(payload)}",
                )
            }
        }
    }

    private fun runCanonical(
        payload: ByteArray,
        ctxT: ByteArray,
        ctx: ByteArray,
        merged: PEntropyRansDecodeMergedRuntime,
        decoder: OfficialNeuronRuntime,
    ): RunResult {
        val totalStarted = SystemClock.elapsedRealtimeNanos()
        val entropyStarted = SystemClock.elapsedRealtimeNanos()
        val yHat = merged.runCanonical(payload, ctxT)
        val entropyMs = elapsedMs(entropyStarted)
        val decoderStarted = SystemClock.elapsedRealtimeNanos()
        val decoded = runDecoder(decoder, yHat, ctx)
        val decoderMs = elapsedMs(decoderStarted)
        return RunResult(
            entropyMs,
            decoderMs,
            elapsedMs(totalStarted),
            decoded[0].data.sum() + decoded[1].data.sum(),
        )
    }

    private fun runDecoder(runtime: OfficialNeuronRuntime, yHat: ByteArray, ctx: ByteArray): List<TensorValue> {
        val outputs = runtime.run(listOf(yHat, ctx))
        require(outputs.size == 2) { "P decoder output count=${outputs.size}" }
        return listOf(
            NhwcTensorCodec.fromF32Le("p_reference_feature", CTX_SHAPE, outputs[0]),
            NhwcTensorCodec.fromF32Le("p_reference_frame", FRAME_SHAPE, outputs[1]),
        )
    }

    private fun compareOutputs(root: File, outputs: List<TensorValue>): Boolean {
        var passed = true
        outputs.forEachIndexed { index, value ->
            val exact = index <= 2
            passed = compareTensor(root, OUTPUT_SPECS[index].name, value, exact) && passed
            if (index in 3..4) {
                val expectedFile = root.resolve("p_s_w_${index - 3}.nchw.f32le")
                if (expectedFile.isFile) {
                    val expected = TensorIO.readF32Le(value.name, value.shape, expectedFile.readBytes())
                    val cdfExact = EntropySymbols.indexesForScales(value)
                        .contentEquals(EntropySymbols.indexesForScales(expected))
                    emit("large_p_decode_merged_compare tensor=${value.name} cdf_index_exact=$cdfExact")
                    passed = cdfExact && passed
                }
            }
        }
        return passed
    }

    private fun compareTensor(root: File, fileStem: String, actual: TensorValue, exact: Boolean): Boolean {
        val file = root.resolve("$fileStem.nchw.f32le")
        if (!file.isFile) {
            emit("large_p_decode_merged_compare tensor=$fileStem status=MISSING expected=${file.absolutePath}")
            return false
        }
        val expected = TensorIO.readF32Le(fileStem, actual.shape, file.readBytes())
        val diff = TensorIO.diff(actual, expected)
        val passed = if (exact) diff.exact else diff.maxAbs <= 1e-3f && diff.rmse <= 1e-4f
        emit(
            "large_p_decode_merged_compare tensor=$fileStem passed=$passed exact=${diff.exact} " +
                "max_abs=${format(diff.maxAbs.toDouble())} mean_abs=${format(diff.meanAbs.toDouble())} " +
                "rmse=${format(diff.rmse.toDouble())}",
        )
        return passed
    }

    private fun emitStats(results: List<RunResult>) {
        emitStats("p_entropy_decode_merged", results.map { it.entropyMs }.toDoubleArray())
        emitStats("p_decoder", results.map { it.decoderMs }.toDoubleArray())
        emitStats("total", results.map { it.totalMs }.toDoubleArray())
        emit("large_p_decode_merged_checksum=${format(results.last().checksum.toDouble())}")
    }

    private fun emitStats(label: String, values: DoubleArray) {
        emit(
            "large_p_decode_merged_speed stage=$label samples=${values.size} mean_ms=${format(values.average())} " +
                "p50_ms=${format(percentile(values, 0.50))} p90_ms=${format(percentile(values, 0.90))} includes_create=false",
        )
    }

    private fun createDecoder(model: File, fastRelaxFp32: Boolean): OfficialNeuronRuntime {
        val sha = sha256(model)
        return OfficialNeuronRuntime.create(
            tfliteFile = model,
            cacheDir = context.cacheDir.resolve("enterprise_tflite/large/p_decoder_merged_probe"),
            allowFp16ForFp32 = fastRelaxFp32,
            acceleratorName = "mtk-neuron",
            compileOptions = if (fastRelaxFp32) "--relax-fp32" else null,
            executionPreference = NeuronDelegate.Options.EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER,
            modelToken = "gvcrt_large_p_decoder_${sha.take(12)}_${if (fastRelaxFp32) "fast" else "strict"}",
        )
    }

    private fun findModel(packageRoot: File, name: String): File {
        val candidates = listOf(
            packageRoot.resolve("models/$name"),
            context.getExternalFilesDir(null)!!.resolve("enterprise_tflite_codec/large/$name"),
            context.filesDir.resolve("enterprise_tflite/large/$name"),
        )
        return candidates.firstOrNull(File::isFile)
            ?: error("missing $name; checked ${candidates.joinToString { it.absolutePath }}")
    }

    private fun findPackageRoot(): File {
        val internal = context.filesDir.resolve("enterprise_tflite/large")
        val external = context.getExternalFilesDir(null)?.resolve("enterprise_tflite/large")
        return listOfNotNull(internal, external).firstOrNull { it.resolve("large_entropy_manifest.json").isFile }
            ?: internal
    }

    private fun percentile(values: DoubleArray, fraction: Double): Double {
        val sorted = values.sortedArray()
        return sorted[((sorted.size - 1) * fraction).toInt().coerceIn(sorted.indices)]
    }

    private fun elapsedMs(started: Long): Double =
        (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0

    private fun sha256(file: File): String = FileInputStream(file).use { stream ->
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(1024 * 1024)
        while (true) {
            val count = stream.read(buffer)
            if (count < 0) break
            digest.update(buffer, 0, count)
        }
        digest.digest().joinToString("") { "%02x".format(Locale.US, it) }
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { "%02x".format(Locale.US, it) }

    private fun format(value: Double): String = "%.6f".format(Locale.US, value)

    private data class TensorSpec(val name: String, val shape: LongArray)
    private data class RunResult(
        val entropyMs: Double,
        val decoderMs: Double,
        val totalMs: Double,
        val checksum: Float,
    )

    private companion object {
        const val MERGED_MODEL = "p_entropy_decode_merged_rans.tflite"
        val OUTPUT_SPECS = listOf(
            TensorSpec("p_z_hat", longArrayOf(1, 128, 4, 8)),
            TensorSpec("p_y_q_w_0", longArrayOf(1, 64, 16, 32)),
            TensorSpec("p_y_q_w_1", longArrayOf(1, 64, 16, 32)),
            TensorSpec("p_s_w_0", longArrayOf(1, 64, 16, 32)),
            TensorSpec("p_s_w_1", longArrayOf(1, 64, 16, 32)),
            TensorSpec("p_y_hat", longArrayOf(1, 128, 16, 32)),
        )
        val CTX_SHAPE = longArrayOf(1, 256, 32, 64)
        val FRAME_SHAPE = longArrayOf(1, 3, 256, 512)
    }
}
