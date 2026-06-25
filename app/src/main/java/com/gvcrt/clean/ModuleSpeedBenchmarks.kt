package com.gvcrt.clean

import android.content.Context
import android.os.SystemClock
import java.util.Locale
import kotlin.math.ceil

class ModuleSpeedBenchmarks(
    context: Context,
    private val emit: (String) -> Unit,
    private val backend: OnnxBackend = OnnxBackend.NNAPI_FP16_ALLOW_FALLBACK,
) {
    private val store = AssetStore(context)
    private val manifest = CleanManifest.parse(store.readBytes(MANIFEST).decodeToString())

    init {
        require(manifest.metadata.optString("precision") == "fp32") {
            "speed benchmark requires the source-matched fp32 manifest"
        }
    }

    fun runModule(moduleName: String) {
        emit("speed_module=$moduleName backend=${backend.label} warmup=$WARMUP_RUNS measured=$MEASURED_RUNS")
        when (moduleName) {
            "temporal_reference" -> runTemporalReference()
            "complete_encoder" -> runCompleteEncoder()
            "complete_decoder" -> runCompleteDecoder()
            else -> error("unsupported speed module '$moduleName'")
        }
    }

    private fun runTemporalReference() {
        val cases = manifest.modules["temporal_reference"] ?: error("missing temporal reference cases")
        OnnxSessionRunner(store, backend).use { runner ->
            cases.forEach { case ->
                val staticInputs = preloadStaticInputs(case.steps)
                benchmark("temporal_reference_${case.name}", 1) { timer ->
                    runGraphSteps(runner, case.steps, staticInputs, timer)
                }
            }
        }
    }

    private fun runCompleteEncoder() {
        val case = manifest.modules["complete_encoder"]?.singleOrNull()
            ?: error("missing canonical complete encoder case")
        val iEntropy = manifest.entropy["i"] ?: error("missing I rANS assets")
        val pEntropy = manifest.entropy["p"] ?: error("missing P rANS assets")
        val stream = manifest.stream ?: error("missing stream specification")
        val staticInputs = preloadStaticInputs(case.steps)
        val iRans = createRans(iEntropy)
        val pRans = createRans(pEntropy)
        try {
            OnnxSessionRunner(store, backend).use { runner ->
                benchmark("complete_encoder", 2) { timer ->
                    val tensors = runGraphSteps(runner, case.steps, staticInputs, timer)
                    val iPayload = encodeEntropy("i", iEntropy, tensors, 4, iRans, timer)
                    val pPayload = encodeEntropy("p", pEntropy, tensors, 2, pRans, timer)
                    timer.measure("stream_mux") { GvcStreamMuxer.mux(stream, iPayload, pPayload) }
                }
            }
        } finally {
            iRans.close()
            pRans.close()
        }
    }

    private fun runCompleteDecoder() {
        val decoder = manifest.decoder ?: error("missing decoder specification")
        val expectedStream = manifest.stream ?: error("missing stream specification")
        val usesAndroidOutput = store.outputExists(decoder.androidInput)
        val streamBytes = if (usesAndroidOutput) {
            store.readOutput(decoder.androidInput)
        } else {
            store.readBytes(decoder.fallbackInput)
        }
        emit(
            "speed_decoder_input=${if (usesAndroidOutput) decoder.androidInput else decoder.fallbackInput} " +
                "bytes=${streamBytes.size} sha256=${AssetStore.sha256(streamBytes)}"
        )
        val iEntropy = manifest.entropy["i"] ?: error("missing I rANS assets")
        val pEntropy = manifest.entropy["p"] ?: error("missing P rANS assets")
        val iRans = createRans(iEntropy)
        val pRans = createRans(pEntropy)
        try {
            OnnxSessionRunner(store, backend).use { runner ->
                benchmark("complete_decoder", 2) { timer ->
                    val parsed = timer.measure("stream_parse") { GvcStreamMuxer.demux(streamBytes) }
                    require(parsed.stream.height == expectedStream.height && parsed.stream.width == expectedStream.width) {
                        "decoder stream geometry differs from manifest"
                    }
                    require(parsed.stream.qp == expectedStream.qp) { "decoder stream QP differs from manifest" }
                    val tensors = linkedMapOf<String, TensorValue>()
                    decodeEntropy("i", parsed.iPayload, iEntropy, decoder.i, iRans, runner, tensors, timer)
                    tensors.putAll(runGraphStep(runner, decoder.i.recon, tensors, timer))
                    val temporal = decoder.p.temporal ?: error("missing P temporal decoder step")
                    tensors.putAll(runGraphStep(runner, temporal, tensors, timer))
                    decodeEntropy("p", parsed.pPayload, pEntropy, decoder.p, pRans, runner, tensors, timer)
                    tensors.putAll(runGraphStep(runner, decoder.p.recon, tensors, timer))
                }
            }
        } finally {
            iRans.close()
            pRans.close()
        }
    }

    private fun decodeEntropy(
        prefix: String,
        payload: ByteArray,
        entropy: EntropySpec,
        decoder: DecoderPathSpec,
        rans: NativeRans,
        runner: OnnxSessionRunner,
        tensors: MutableMap<String, TensorValue>,
        timer: StageTimer,
    ) {
        require(!entropy.twoEntropyCoders) { "two-entropy-coder mode is not enabled" }
        val zName = "${prefix}_z_hat"
        val zBytes = timer.measure("${prefix}_rans_z") {
            rans.beginDecode(payload)
            rans.decodeZ(decoder.zShape.elementCount(), entropy.zStartOffset, entropy.zPerChannelSize)
        }
        tensors[zName] = TensorIO.fromI8(zName, decoder.zShape, zBytes)
        tensors.putAll(runGraphStep(runner, decoder.hyperPrior, tensors, timer))

        decoder.stages.forEach { stage ->
            val yInput = stage.inputs.single { it.tensorName.contains("_y_q_w_") }.tensorName
            val scaleOutput = stage.outputs.single { it.tensorName.contains("_s_w_") }.tensorName
            tensors[yInput] = TensorValue(yInput, decoder.yStageShape, FloatArray(decoder.yStageShape.elementCount()))
            timer.measure(stage.name) {
                val scaleProbe = runner.run(stage, resolveInputs(stage, tensors))
                val decoded = rans.decodeY(EntropySymbols.indexesForScales(scaleProbe.getValue(scaleOutput)))
                tensors[yInput] = TensorIO.fromI8(yInput, decoder.yStageShape, decoded)
                tensors.putAll(runner.run(stage, resolveInputs(stage, tensors)))
            }
        }
    }

    private fun encodeEntropy(
        prefix: String,
        entropy: EntropySpec,
        tensors: Map<String, TensorValue>,
        stageCount: Int,
        rans: NativeRans,
        timer: StageTimer,
    ): ByteArray = timer.measure("${prefix}_rans") {
        val z = EntropySymbols.zSymbols(tensors.getValue("${prefix}_z_hat"))
        val packedStages = Array(stageCount) { stage ->
            EntropySymbols.packY(
                tensors.getValue("${prefix}_y_q_w_$stage"),
                tensors.getValue("${prefix}_s_w_$stage"),
            )
        }
        rans.encode(z, entropy.zStartOffset, entropy.zPerChannelSize, packedStages)
    }

    private fun createRans(entropy: EntropySpec): NativeRans =
        NativeRans.create(CdfTable.load(store, entropy.gaussian), CdfTable.load(store, entropy.z))

    private fun runGraphSteps(
        runner: OnnxSessionRunner,
        steps: List<GraphStep>,
        staticInputs: Map<String, TensorValue>,
        timer: StageTimer,
    ): MutableMap<String, TensorValue> {
        val tensors = linkedMapOf<String, TensorValue>()
        steps.forEach { step ->
            tensors.putAll(runGraphStep(runner, step, tensors, timer, staticInputs))
        }
        return tensors
    }

    private fun runGraphStep(
        runner: OnnxSessionRunner,
        step: GraphStep,
        tensors: Map<String, TensorValue>,
        timer: StageTimer,
        staticInputs: Map<String, TensorValue> = emptyMap(),
    ): Map<String, TensorValue> =
        timer.measure(step.name) { runner.run(step, resolveInputs(step, tensors, staticInputs)) }

    private fun preloadStaticInputs(steps: List<GraphStep>): Map<String, TensorValue> =
        steps.flatMap { it.inputs }
            .filter { it.source == null }
            .associate { input ->
                val path = input.path ?: error("static input ${input.tensorName} has no path")
                val shape = input.shape ?: error("static input ${input.tensorName} has no shape")
                input.tensorName to TensorIO.readF32Le(input.tensorName, shape, store.readBytes(path))
            }

    private fun resolveInputs(
        step: GraphStep,
        tensors: Map<String, TensorValue>,
        staticInputs: Map<String, TensorValue> = emptyMap(),
    ): Map<String, TensorValue> =
        step.inputs.associate { input ->
            val tensor = input.source?.let(tensors::getValue) ?: staticInputs.getValue(input.tensorName)
            input.tensorName to tensor
        }

    private fun benchmark(label: String, framesPerRun: Int, runOnce: (StageTimer) -> Unit) {
        emit("speed_start label=$label warmup=$WARMUP_RUNS measured=$MEASURED_RUNS frames_per_run=$framesPerRun")
        repeat(WARMUP_RUNS) { runOnce(StageTimer()) }

        val totals = ArrayList<Long>(MEASURED_RUNS)
        val stages = linkedMapOf<String, MutableList<Long>>()
        repeat(MEASURED_RUNS) {
            val timer = StageTimer()
            val started = SystemClock.elapsedRealtimeNanos()
            runOnce(timer)
            totals += SystemClock.elapsedRealtimeNanos() - started
            timer.values.forEach { (name, elapsed) -> stages.getOrPut(name) { ArrayList() } += elapsed }
        }

        emitSpeedSummary(label, "total", totals, framesPerRun)
        stages.forEach { (name, values) -> emitSpeedSummary(label, name, values, null) }
        emit("speed_complete label=$label")
    }

    private fun emitSpeedSummary(label: String, stage: String, values: List<Long>, framesPerRun: Int?) {
        val summary = TimingSummary.from(values)
        val fps = framesPerRun?.let { it * 1_000_000_000.0 / summary.meanNs }
        emit(
            "speed label=$label stage=$stage samples=${values.size} " +
                "mean_ms=${formatMs(summary.meanNs)} p50_ms=${formatMs(summary.p50Ns)} p90_ms=${formatMs(summary.p90Ns)}" +
                (fps?.let { " fps=${String.format(Locale.US, "%.3f", it)}" } ?: "")
        )
    }

    private fun formatMs(nanos: Double): String = String.format(Locale.US, "%.3f", nanos / 1_000_000.0)

    private class StageTimer {
        val values = linkedMapOf<String, Long>()

        inline fun <T> measure(name: String, block: () -> T): T {
            val started = SystemClock.elapsedRealtimeNanos()
            return try {
                block()
            } finally {
                values[name] = (values[name] ?: 0L) + (SystemClock.elapsedRealtimeNanos() - started)
            }
        }
    }

    private data class TimingSummary(
        val meanNs: Double,
        val p50Ns: Double,
        val p90Ns: Double,
    ) {
        companion object {
            fun from(values: List<Long>): TimingSummary {
                require(values.isNotEmpty()) { "cannot summarize an empty timing series" }
                val sorted = values.sorted()
                return TimingSummary(
                    values.average(),
                    percentile(sorted, 0.50),
                    percentile(sorted, 0.90),
                )
            }

            private fun percentile(sorted: List<Long>, quantile: Double): Double {
                val index = (ceil(sorted.size * quantile).toInt() - 1).coerceIn(0, sorted.lastIndex)
                return sorted[index].toDouble()
            }
        }
    }

    private fun LongArray.elementCount(): Int = fold(1L) { acc, value -> acc * value }.toInt()

    companion object {
        private const val MANIFEST = "gvcrt_clean_manifest.json"
        private const val WARMUP_RUNS = 5
        private const val MEASURED_RUNS = 50
    }
}
