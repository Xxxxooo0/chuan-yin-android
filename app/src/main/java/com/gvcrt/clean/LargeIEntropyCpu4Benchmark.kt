package com.gvcrt.clean

import android.content.Context
import android.os.SystemClock
import org.json.JSONObject
import java.io.File
import java.util.Locale

/** Four-thread CPU benchmark for the seven I-frame entropy TFLite graphs. */
class LargeIEntropyCpu4Benchmark(
    private val context: Context,
    private val emit: (String) -> Unit,
) {
    fun run(warmupRuns: Int = 3, measuredRuns: Int = 10) {
        require(warmupRuns >= 0 && measuredRuns > 0)
        val packageRoot = findPackageRoot()
        val manifest = JSONObject(packageRoot.resolve(MANIFEST).readText())
        val forceZero = manifest.getDouble("force_zero_thres").toFloat()
        emit(
            "large_i_entropy_cpu4_start graphs=7 threads=4 xnnpack=true nnapi=false delegate=none " +
                "warmup=$warmupRuns measured=$measuredRuns",
        )

        val y = TensorValue("i_y_pre_prior", Y_SHAPE, FloatArray(Y_SHAPE.product()))
        emit("large_i_entropy_cpu4_input_ready source=deterministic_zero measured=false")
        val runtimes = linkedMapOf<String, OfficialTfliteCpuRuntime>()
        try {
            GRAPH_NAMES.forEach { name ->
                val model = packageRoot.resolve("models/$name.tflite")
                require(model.isFile) { "missing TFLite model: ${model.absolutePath}" }
                emit("large_i_entropy_cpu4_create_start model=$name bytes=${model.length()}")
                val started = SystemClock.elapsedRealtimeNanos()
                runtimes[name] = OfficialTfliteCpuRuntime.create(model, THREADS).also {
                    emit(
                        "large_i_entropy_cpu4_create_ok model=$name create_ms=${format(elapsedMs(started))} " +
                            "options=${it.optionsSummary}",
                    )
                }
            }

            val graphTimes = GRAPH_NAMES.associateWith { mutableListOf<Double>() }
            val neuralTotals = mutableListOf<Double>()
            val quantTotals = mutableListOf<Double>()
            val pipelineTotals = mutableListOf<Double>()
            var lastChecksum = 0.0
            repeat(warmupRuns + measuredRuns) { runIndex ->
                val record = runIndex >= warmupRuns
                val pipelineStarted = SystemClock.elapsedRealtimeNanos()
                var neuralMs = 0.0
                var quantMs = 0.0

                fun graph(name: String, block: () -> List<TensorValue>): List<TensorValue> {
                    emit("large_i_entropy_cpu4_invoke_start run=$runIndex model=$name record=$record")
                    val started = SystemClock.elapsedRealtimeNanos()
                    val result = block()
                    val ms = elapsedMs(started)
                    emit(
                        "large_i_entropy_cpu4_invoke_done run=$runIndex model=$name " +
                            "record=$record elapsed_ms=${format(ms)}",
                    )
                    neuralMs += ms
                    if (record) graphTimes.getValue(name).add(ms)
                    return result
                }

                fun <T> quant(block: () -> T): T {
                    val started = SystemClock.elapsedRealtimeNanos()
                    val result = block()
                    quantMs += elapsedMs(started)
                    return result
                }

                val zPreQuant = graph("i_hyper_enc_continuous") {
                    runNchw(
                        runtimes.getValue("i_hyper_enc_continuous"),
                        listOf(NhwcTensorCodec.toF32Le(y)),
                        listOf(TensorSpec("i_z_pre_quant", Z_SHAPE)),
                    )
                }.single()
                val zHat = quant { quantizeInt8(zPreQuant, "i_z_hat") }
                val common = graph("i_hyper_prior_shared") {
                    runNchw(
                        runtimes.getValue("i_hyper_prior_shared"),
                        listOf(NhwcTensorCodec.toF32Le(zHat)),
                        listOf(TensorSpec("i_common_params", COMMON_SHAPE)),
                    )
                }.single()
                val stage0 = graph("i_prior_stage0_params") {
                    runNchw(
                        runtimes.getValue("i_prior_stage0_params"),
                        listOf(NhwcTensorCodec.toF32Le(common)),
                        listOf(
                            TensorSpec("i_q_enc", Q_SHAPE),
                            TensorSpec("i_q_dec", Q_SHAPE),
                            TensorSpec("i_stage0_scales", Y_SHAPE),
                            TensorSpec("i_stage0_means", Y_SHAPE),
                        ),
                    )
                }
                val reduced = graph("i_prior_reduce") {
                    runNchw(
                        runtimes.getValue("i_prior_reduce"),
                        listOf(NhwcTensorCodec.toF32Le(common)),
                        listOf(TensorSpec("i_reduced_common_params", Y_SHAPE)),
                    )
                }.single()
                val yScaled = quant { EntropyPriorQuantizer.multiply(y, stage0[0], "i_y_scaled") }
                var yHatSoFar = quant {
                    EntropyPriorQuantizer.quantize(yScaled, stage0[3], stage0[2], 0, 4, forceZero).yHat
                }
                for (stage in 1..3) {
                    val name = "i_prior_stage${stage}_continuous"
                    val params = graph(name) {
                        runNchw(
                            runtimes.getValue(name),
                            listOf(NhwcTensorCodec.toF32Le(yHatSoFar), NhwcTensorCodec.toF32Le(reduced)),
                            listOf(
                                TensorSpec("i_stage${stage}_scales", Y_SHAPE),
                                TensorSpec("i_stage${stage}_means", Y_SHAPE),
                            ),
                        )
                    }
                    yHatSoFar = quant {
                        val current = EntropyPriorQuantizer.quantize(
                            yScaled,
                            params[1],
                            params[0],
                            stage,
                            4,
                            forceZero,
                        ).yHat
                        EntropyPriorQuantizer.add(yHatSoFar, current, "i_y_hat_so_far_$stage")
                    }
                }
                val yHat = quant { EntropyPriorQuantizer.multiply(yHatSoFar, stage0[1], "i_y_hat") }
                lastChecksum = yHat.data.sumOf { it.toDouble() }
                if (record) {
                    neuralTotals += neuralMs
                    quantTotals += quantMs
                    pipelineTotals += elapsedMs(pipelineStarted)
                }
            }
            GRAPH_NAMES.forEach { name -> emitSpeed(name, graphTimes.getValue(name)) }
            emitSpeed("seven_graphs_total", neuralTotals)
            emitSpeed("native_quant_total", quantTotals)
            emitSpeed("seven_graphs_plus_quant", pipelineTotals)
            emit("large_i_entropy_cpu4_complete checksum=${format(lastChecksum)}")
        } finally {
            runtimes.values.forEach(OfficialTfliteCpuRuntime::close)
        }
    }

    private fun runNchw(
        runtime: OfficialTfliteCpuRuntime,
        inputs: List<ByteArray>,
        outputs: List<TensorSpec>,
    ): List<TensorValue> {
        val bytes = runtime.run(inputs)
        require(bytes.size == outputs.size)
        return outputs.mapIndexed { index, spec -> NhwcTensorCodec.fromF32Le(spec.name, spec.shape, bytes[index]) }
    }

    private fun quantizeInt8(input: TensorValue, name: String): TensorValue =
        TensorValue(name, input.shape, FloatArray(input.numel) { index ->
            Math.rint(input.data[index].toDouble()).toInt().coerceIn(-128, 127).toFloat()
        })

    private fun emitSpeed(stage: String, values: List<Double>) {
        val sorted = values.sorted()
        fun percentile(fraction: Double): Double = sorted[((sorted.size - 1) * fraction).toInt()]
        emit(
            "large_i_entropy_cpu4_speed stage=$stage samples=${values.size} " +
                "mean_ms=${format(values.average())} p50_ms=${format(percentile(0.5))} " +
                "p90_ms=${format(percentile(0.9))}",
        )
    }

    private fun findPackageRoot(): File {
        val internal = context.filesDir.resolve("enterprise_tflite/large")
        val external = context.getExternalFilesDir(null)?.resolve("enterprise_tflite/large")
        return listOfNotNull(internal, external).firstOrNull { it.resolve(MANIFEST).isFile }
            ?: error("missing $MANIFEST")
    }

    private fun elapsedMs(started: Long): Double =
        (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0

    private fun format(value: Double): String = "%.3f".format(Locale.US, value)

    private data class TensorSpec(val name: String, val shape: LongArray)

    private companion object {
        const val MANIFEST = "large_entropy_manifest.json"
        const val THREADS = 4
        val GRAPH_NAMES = listOf(
            "i_hyper_enc_continuous",
            "i_hyper_prior_shared",
            "i_prior_stage0_params",
            "i_prior_reduce",
            "i_prior_stage1_continuous",
            "i_prior_stage2_continuous",
            "i_prior_stage3_continuous",
        )
        val Y_SHAPE = longArrayOf(1, 256, 16, 32)
        val Z_SHAPE = longArrayOf(1, 128, 4, 8)
        val Q_SHAPE = longArrayOf(1, 1, 16, 32)
        val COMMON_SHAPE = longArrayOf(1, 514, 16, 32)

        private fun LongArray.product(): Int = fold(1L) { product, value -> product * value }.toInt()
    }
}
