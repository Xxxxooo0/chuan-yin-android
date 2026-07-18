package com.gvcrt.clean

import android.content.Context
import android.os.SystemClock
import java.util.Locale

class IEncoderPriorNpuBenchmark(
    private val context: Context,
    private val emit: (String) -> Unit,
) {
    private val store = AssetStore(context)
    private val manifest = CleanManifest.parse(store.readBytes(MANIFEST).decodeToString())
    private val y = baseline("i_y_pre_prior", Y_SHAPE)
    private val common = baseline("i_common_params", COMMON_SHAPE)

    fun runPrecision() {
        emit("i_prior_npu_precision_start qp=0 shape=[1,3,256,512]")
        IEncoderPriorNpuRunner(context).use { runner ->
            emit("i_prior_npu_create_ok=true")
            emit("i_prior_npu_assets ${runner.assetSummary()}")
            emit("i_prior_npu_runtime ${runner.runtimeSummary()}")
            val output = runner.run(y, common)
            output.stageTimesNs.forEach { (stage, elapsed) -> emit("i_prior_npu_time stage=$stage ms=${formatMs(elapsed)}") }
            var allExact = true
            output.symbols.forEachIndexed { stage, actual ->
                allExact = emitExact(actual, baseline("i_y_q_w_$stage", STAGE_SHAPE)) && allExact
            }
            output.scales.forEachIndexed { stage, actual ->
                val expected = baseline("i_s_w_$stage", STAGE_SHAPE)
                emitContinuous(actual, expected)
                val indexExact = EntropySymbols.indexesForScales(actual).contentEquals(EntropySymbols.indexesForScales(expected))
                emit("i_prior_npu_cdf stage=$stage exact=$indexExact")
                allExact = indexExact && allExact
            }
            allExact = emitExact(output.yHat, baseline("i_y_hat", Y_SHAPE)) && allExact
            val payload = encodeI(output)
            val expectedPayload = store.readBytes(manifest.entropy.getValue("i").payload)
            val payloadExact = payload.contentEquals(expectedPayload)
            emit("i_prior_npu_payload exact=$payloadExact android_bytes=${payload.size} server_bytes=${expectedPayload.size} first_diff=${firstDifference(payload, expectedPayload)}")
            allExact = payloadExact && allExact
            val stream = manifest.stream ?: error("missing stream specification")
            val pPayload = store.readBytes(manifest.entropy.getValue("p").payload)
            val muxed = GvcStreamMuxer.mux(stream, payload, pPayload)
            val expectedStream = store.readBytes(stream.path)
            val streamExact = muxed.contentEquals(expectedStream)
            emit("i_prior_npu_bitstream exact=$streamExact android_bytes=${muxed.size} server_bytes=${expectedStream.size} first_diff=${firstDifference(muxed, expectedStream)}")
            emit("i_prior_npu_precision_result pass=${allExact && streamExact}")
        }
    }

    fun runSpeed(warmupRuns: Int = 5, measuredRuns: Int = 50) {
        emit("i_prior_npu_speed_start warmup=$warmupRuns measured=$measuredRuns qp=0")
        val onnxStep = manifest.modules.getValue("complete_encoder").single().steps.single { it.name == "i_prior_4x" }
        val onnxInputs = mapOf("i_y_pre_prior" to y, "i_common_params" to common)
        OnnxSessionRunner(store, OnnxBackend.NNAPI_FP16_ALLOW_FALLBACK).use { onnx ->
            IEncoderPriorNpuRunner(context).use { npu ->
                emit("i_prior_npu_create_ok=true")
                emit("i_prior_npu_assets ${npu.assetSummary()}")
                emit("i_prior_npu_runtime ${npu.runtimeSummary()}")
                val entropy = manifest.entropy.getValue("i")
                RansNativeEncoder.create(CdfTable.load(store, entropy.gaussian), CdfTable.load(store, entropy.z), entropy.twoEntropyCoders).use { encoder ->
                    repeat(warmupRuns) {
                        onnx.run(onnxStep, onnxInputs)
                        encodeI(encoder, npu.run(y, common))
                    }
                    val onnxTimes = ArrayList<Long>(measuredRuns)
                    val npuTimes = ArrayList<Long>(measuredRuns)
                    val ransTimes = ArrayList<Long>(measuredRuns)
                    val stages = linkedMapOf<String, MutableList<Long>>()
                    repeat(measuredRuns) {
                        val onnxStarted = SystemClock.elapsedRealtimeNanos()
                        onnx.run(onnxStep, onnxInputs)
                        onnxTimes += SystemClock.elapsedRealtimeNanos() - onnxStarted
                        val npuStarted = SystemClock.elapsedRealtimeNanos()
                        val output = npu.run(y, common)
                        npuTimes += SystemClock.elapsedRealtimeNanos() - npuStarted
                        val ransStarted = SystemClock.elapsedRealtimeNanos()
                        encodeI(encoder, output)
                        ransTimes += SystemClock.elapsedRealtimeNanos() - ransStarted
                        output.stageTimesNs.forEach { (name, value) -> stages.getOrPut(name) { ArrayList() } += value }
                    }
                    emitSummary("i_prior_onnx_nnapi", onnxTimes)
                    emitSummary("i_prior_npu_total", npuTimes)
                    emitSummary("i_prior_npu_rans", ransTimes)
                    stages.forEach { (name, values) -> emitSummary("i_prior_npu_$name", values) }
                    val onnxMean = onnxTimes.average()
                    val npuMean = npuTimes.average()
                    val gain = if (onnxMean == 0.0) 0.0 else (onnxMean - npuMean) * 100.0 / onnxMean
                    emit("i_prior_npu_speed_result improvement_pct=${String.format(Locale.US, "%.3f", gain)} promote=${gain >= 10.0}")
                }
            }
        }
    }

    private fun encodeI(output: IEncoderPriorNpuOutput): ByteArray {
        val entropy = manifest.entropy.getValue("i")
        val gaussian = CdfTable.load(store, entropy.gaussian)
        val zTable = CdfTable.load(store, entropy.z)
        return RansNativeEncoder.create(gaussian, zTable, entropy.twoEntropyCoders).use { encoder ->
            encodeI(encoder, output)
        }
    }

    private fun encodeI(encoder: RansNativeEncoder, output: IEncoderPriorNpuOutput): ByteArray {
        val entropy = manifest.entropy.getValue("i")
        encoder.encodeZ(EntropySymbols.zSymbols(baseline("i_z_hat", Z_SHAPE)), entropy.zStartOffset, entropy.zPerChannelSize)
        output.symbols.indices.forEach { stage -> encoder.encodeY(output.symbols[stage], output.scales[stage]) }
        return encoder.flush()
    }

    private fun emitExact(actual: TensorValue, expected: TensorValue): Boolean {
        val diff = TensorIO.diff(actual, expected)
        emit("i_prior_npu_compare tensor=${actual.name} kind=discrete exact=${diff.exact} first_diff=${firstDifference(actual.data, expected.data)}")
        return diff.exact
    }

    private fun emitContinuous(actual: TensorValue, expected: TensorValue) {
        val diff = TensorIO.diff(actual, expected)
        emit("i_prior_npu_compare tensor=${actual.name} kind=continuous exact=${diff.exact} max_abs=${format(diff.maxAbs)} mean_abs=${format(diff.meanAbs)} rmse=${format(diff.rmse)}")
    }

    private fun emitSummary(label: String, values: List<Long>) {
        val ordered = values.sorted()
        val mean = values.average()
        val p50 = ordered[(ordered.size - 1) / 2]
        val p90 = ordered[((ordered.size - 1) * 90) / 100]
        emit("i_prior_npu_speed stage=$label samples=${values.size} mean_ms=${formatMs(mean)} p50_ms=${formatMs(p50)} p90_ms=${formatMs(p90)}")
    }

    private fun baseline(name: String, shape: LongArray): TensorValue =
        TensorIO.readF32Le(name, shape, store.readBytes("baseline/tensors/$name.f32le"))

    private fun firstDifference(first: FloatArray, second: FloatArray): Int = first.indices.firstOrNull { first[it] != second[it] } ?: -1
    private fun firstDifference(first: ByteArray, second: ByteArray): Int = first.indices.firstOrNull { it >= second.size || first[it] != second[it] }
        ?: if (first.size == second.size) -1 else minOf(first.size, second.size)
    private fun format(value: Float): String = String.format(Locale.US, "%.8f", value)
    private fun formatMs(value: Number): String = String.format(Locale.US, "%.3f", value.toDouble() / 1_000_000.0)

    companion object {
        private const val MANIFEST = "gvcrt_clean_manifest.json"
        private val Y_SHAPE = longArrayOf(1, 256, 16, 32)
        private val COMMON_SHAPE = longArrayOf(1, 514, 16, 32)
        private val STAGE_SHAPE = longArrayOf(1, 64, 16, 32)
        private val Z_SHAPE = longArrayOf(1, 128, 4, 8)
    }
}
