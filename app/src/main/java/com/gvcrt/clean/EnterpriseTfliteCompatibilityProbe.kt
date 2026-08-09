package com.gvcrt.clean

import android.content.Context
import android.os.SystemClock
import com.mediatek.neuropilot_V.neuron.NeuronDelegate
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest
import java.util.Locale

/** Runs the enterprise TFLite handoff packages through the official NeuronDelegate. */
class EnterpriseTfliteCompatibilityProbe(
    private val context: Context,
    private val emit: (String) -> Unit,
) {
    fun run(
        variant: String,
        warmupRuns: Int = 3,
        measuredRuns: Int = 10,
        relaxFp32: Boolean = false,
    ) {
        require(warmupRuns >= 0) { "warmupRuns must be non-negative" }
        require(measuredRuns > 0) { "measuredRuns must be positive" }
        val variants = when (variant.lowercase(Locale.US)) {
            "large" -> listOf("large")
            "small" -> listOf("small")
            "all" -> listOf("large", "small")
            else -> error("enterpriseTfliteVariant must be large, small, or all")
        }
        variants.forEach { runVariant(it, warmupRuns, measuredRuns, relaxFp32) }
    }

    private fun runVariant(
        variant: String,
        warmupRuns: Int,
        measuredRuns: Int,
        relaxFp32: Boolean,
    ) {
        val internalRoot = context.filesDir.resolve("enterprise_tflite/$variant")
        val externalRoot = context.getExternalFilesDir(null)?.resolve("enterprise_tflite/$variant")
        val packageRoot = listOfNotNull(internalRoot, externalRoot)
            .firstOrNull { it.resolve("input_manifest.json").isFile }
            ?: internalRoot
        val inputManifestFile = packageRoot.resolve("input_manifest.json")
        require(inputManifestFile.isFile) {
            "missing input manifest: ${inputManifestFile.absolutePath}"
        }
        val manifest = JSONObject(inputManifestFile.readText())
        val stagesJson = manifest.getJSONArray("stages")
        val stages = (0 until stagesJson.length())
            .map { stagesJson.getJSONObject(it) }
            .sortedBy { it.getInt("order") }
        val modelNames = stages.map { it.getString("model") }.distinct()
        val runtimes = linkedMapOf<String, OfficialNeuronRuntime>()
        val failedModels = linkedSetOf<String>()
        var successfulStages = 0
        var failedStages = 0

        val runtimeProfile = if (relaxFp32) "mlvc_relax_fp32" else "strict_fp32"
        emit(
            "enterprise_tflite_start variant=$variant package=${manifest.optString("package")} " +
                "models=${modelNames.size} stages=${stages.size} backend=official_aar_neuron " +
                "runtime_profile=$runtimeProfile allow_fp16=$relaxFp32 " +
                "root=${packageRoot.absolutePath}",
        )
        try {
            stages.forEach { stage ->
                val stageId = stage.getString("id")
                val modelName = stage.getString("model")
                if (modelName in failedModels) {
                    failedStages++
                    emit("enterprise_tflite_stage_skip variant=$variant stage=$stageId model=$modelName reason=create_failed")
                    return@forEach
                }
                val runtime = runtimes[modelName] ?: createRuntime(
                    variant = variant,
                    modelName = modelName,
                    modelFile = packageRoot.resolve("models/$modelName.tflite"),
                    relaxFp32 = relaxFp32,
                )?.also { runtimes[modelName] = it } ?: run {
                    failedModels += modelName
                    failedStages++
                    return@forEach
                }
                try {
                    runStage(variant, packageRoot, stage, runtime)
                    successfulStages++
                } catch (error: Throwable) {
                    failedStages++
                    emit(
                        "enterprise_tflite_stage_failure variant=$variant stage=$stageId model=$modelName " +
                            "type=${error.javaClass.simpleName} message=${error.message}",
                    )
                }
            }
            if (variant == "small" && failedModels.isEmpty()) {
                runSmallChainedPrecision(
                    packageRoot = packageRoot,
                    stages = stages,
                    runtimes = runtimes,
                )
                runSmallPipelineBenchmark(
                    packageRoot = packageRoot,
                    stages = stages,
                    runtimes = runtimes,
                    warmupRuns = warmupRuns,
                    measuredRuns = measuredRuns,
                )
            }
        } finally {
            runtimes.values.forEach(OfficialNeuronRuntime::close)
        }
        emit(
            "enterprise_tflite_complete variant=$variant models_ok=${runtimes.size} " +
                "models_failed=${failedModels.size} stages_ok=$successfulStages stages_failed=$failedStages",
        )
    }

    private fun runSmallChainedPrecision(
        packageRoot: File,
        stages: List<JSONObject>,
        runtimes: Map<String, OfficialNeuronRuntime>,
    ) {
        val frameGroups = stages.groupBy { stage ->
            stage.getString("id").split("_").take(2).joinToString("_")
        }
            .toSortedMap()
        val firstTemporal = stages.first { it.getString("model") == "temporal_reference" }
        val temporal = runtimes.getValue("temporal_reference")
        val encoder = runtimes.getValue("encoder")
        val decoder = runtimes.getValue("decoder")
        var referenceFeature = readStageInput(packageRoot, firstTemporal, "ref_feature")
        val outputRoot = context.getExternalFilesDir(null)!!
            .resolve("enterprise_tflite_chained_outputs/small")

        emit("enterprise_tflite_chained_start variant=small frames=${frameGroups.size}")
        frameGroups.forEach { (frameId, frameStages) ->
            val temporalStage = frameStages.first { it.getString("model") == "temporal_reference" }
            val encoderStage = frameStages.first { it.getString("model") == "encoder" }
            val decoderStage = frameStages.first { it.getString("model") == "decoder" }
            val frame = readStageInput(packageRoot, encoderStage, "frame")
            val temporalOutputs = temporal.run(listOf(referenceFeature))
            require(temporalOutputs.size == 3) { "small temporal output count=${temporalOutputs.size}" }
            writeOutputs(outputRoot, temporalStage, temporalOutputs)
            val ctx = temporalOutputs[0]
            val memory = temporalOutputs[2]
            val encoderOutputs = encoder.run(listOf(frame, ctx))
            require(encoderOutputs.size == 1) { "small encoder output count=${encoderOutputs.size}" }
            writeOutputs(outputRoot, encoderStage, encoderOutputs)
            val decoderOutputs = decoder.run(listOf(encoderOutputs[0], ctx, memory))
            require(decoderOutputs.size == 2) { "small decoder output count=${decoderOutputs.size}" }
            writeOutputs(outputRoot, decoderStage, decoderOutputs)
            referenceFeature = decoderOutputs[0]
            emit("enterprise_tflite_chained_frame_ok variant=small frame=$frameId")
        }
        emit("enterprise_tflite_chained_complete variant=small frames=${frameGroups.size}")
    }

    private fun runSmallPipelineBenchmark(
        packageRoot: File,
        stages: List<JSONObject>,
        runtimes: Map<String, OfficialNeuronRuntime>,
        warmupRuns: Int,
        measuredRuns: Int,
    ) {
        val temporalStage = stages.first { it.getString("model") == "temporal_reference" }
        val encoderStage = stages.first { it.getString("model") == "encoder" }
        val temporal = runtimes.getValue("temporal_reference")
        val encoder = runtimes.getValue("encoder")
        val decoder = runtimes.getValue("decoder")
        var referenceFeature = readStageInput(packageRoot, temporalStage, "ref_feature")
        val frame = readStageInput(packageRoot, encoderStage, "frame")
        val temporalTimes = mutableListOf<Double>()
        val encoderTimes = mutableListOf<Double>()
        val decoderTimes = mutableListOf<Double>()
        val totalTimes = mutableListOf<Double>()

        emit(
            "enterprise_tflite_speed_start variant=small mode=chained warmup=$warmupRuns measured=$measuredRuns",
        )
        repeat(warmupRuns + measuredRuns) { runIndex ->
            val totalStarted = SystemClock.elapsedRealtimeNanos()
            val temporalStarted = SystemClock.elapsedRealtimeNanos()
            val temporalOutputs = temporal.run(listOf(referenceFeature))
            val temporalMs = elapsedMs(temporalStarted)
            require(temporalOutputs.size == 3) { "small temporal output count=${temporalOutputs.size}" }
            val ctx = temporalOutputs[0]
            val memory = temporalOutputs[2]

            val encoderStarted = SystemClock.elapsedRealtimeNanos()
            val encoderOutputs = encoder.run(listOf(frame, ctx))
            val encoderMs = elapsedMs(encoderStarted)
            require(encoderOutputs.size == 1) { "small encoder output count=${encoderOutputs.size}" }

            val decoderStarted = SystemClock.elapsedRealtimeNanos()
            val decoderOutputs = decoder.run(listOf(encoderOutputs[0], ctx, memory))
            val decoderMs = elapsedMs(decoderStarted)
            require(decoderOutputs.size == 2) { "small decoder output count=${decoderOutputs.size}" }
            referenceFeature = decoderOutputs[0]
            val totalMs = elapsedMs(totalStarted)

            if (runIndex >= warmupRuns) {
                temporalTimes += temporalMs
                encoderTimes += encoderMs
                decoderTimes += decoderMs
                totalTimes += totalMs
            }
        }
        emitSpeed("temporal_reference", temporalTimes)
        emitSpeed("encoder", encoderTimes)
        emitSpeed("decoder", decoderTimes)
        emitSpeed("total", totalTimes)
        emit(
            "enterprise_tflite_speed_complete variant=small mean_fps=${format(1000.0 / totalTimes.average())}",
        )
    }

    private fun readStageInput(packageRoot: File, stage: JSONObject, name: String): ByteArray {
        val inputs = stage.getJSONArray("inputs")
        val record = (0 until inputs.length())
            .map { inputs.getJSONObject(it) }
            .first { it.getString("name") == name }
        return packageRoot.resolve(record.getString("file")).readBytes()
    }

    private fun emitSpeed(label: String, values: List<Double>) {
        val sorted = values.sorted()
        fun percentile(fraction: Double): Double {
            val index = ((sorted.size - 1) * fraction).toInt().coerceIn(sorted.indices)
            return sorted[index]
        }
        emit(
            "enterprise_tflite_speed variant=small stage=$label samples=${values.size} " +
                "mean_ms=${format(values.average())} p50_ms=${format(percentile(0.50))} " +
                "p90_ms=${format(percentile(0.90))}",
        )
    }

    private fun createRuntime(
        variant: String,
        modelName: String,
        modelFile: File,
        relaxFp32: Boolean,
    ): OfficialNeuronRuntime? {
        require(modelFile.isFile) { "missing model: ${modelFile.absolutePath}" }
        val sha = sha256(modelFile)
        emit(
            "enterprise_tflite_create_start variant=$variant model=$modelName bytes=${modelFile.length()} " +
                "sha256=$sha",
        )
        return try {
            val started = SystemClock.elapsedRealtimeNanos()
            val runtime = OfficialNeuronRuntime.create(
                tfliteFile = modelFile,
                cacheDir = context.cacheDir.resolve(
                    "enterprise_tflite/$variant/${if (relaxFp32) "mlvc_relax_fp32" else "strict_fp32"}/$modelName",
                ),
                allowFp16ForFp32 = relaxFp32,
                acceleratorName = if (relaxFp32) "mtk-neuron" else null,
                compileOptions = if (relaxFp32) "--relax-fp32" else null,
                executionPreference = if (relaxFp32) {
                    NeuronDelegate.Options.EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER
                } else {
                    NeuronDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED
                },
                modelToken = if (relaxFp32) {
                    "gvcrt_${variant}_${modelName}_${sha.take(12)}"
                } else {
                    null
                },
            )
            emit(
                "enterprise_tflite_create_ok variant=$variant model=$modelName " +
                    "create_ms=${format(elapsedMs(started))} inputs=${runtime.inputSizes.joinToString(":")} " +
                    "outputs=${runtime.outputSizes.joinToString(":")} options=${runtime.optionsSummary}",
            )
            runtime
        } catch (error: Throwable) {
            emit(
                "enterprise_tflite_create_failure variant=$variant model=$modelName " +
                    "type=${error.javaClass.simpleName} message=${error.message}",
            )
            null
        }
    }

    private fun runStage(
        variant: String,
        packageRoot: File,
        stage: JSONObject,
        runtime: OfficialNeuronRuntime,
    ) {
        val stageId = stage.getString("id")
        val modelName = stage.getString("model")
        val inputsJson = stage.getJSONArray("inputs")
        val inputs = (0 until inputsJson.length()).map { index ->
            val input = inputsJson.getJSONObject(index)
            val file = packageRoot.resolve(input.getString("file"))
            require(file.isFile) { "missing input: ${file.absolutePath}" }
            require(file.length() == input.getLong("bytes")) {
                "input bytes mismatch name=${input.getString("name")} file=${file.length()} manifest=${input.getLong("bytes")}"
            }
            file.readBytes()
        }
        val expectedInputSizes = inputs.map { it.size.toLong() }.toLongArray()
        require(runtime.inputSizes.contentEquals(expectedInputSizes)) {
            "runtime input sizes=${runtime.inputSizes.joinToString(":")} manifest=${expectedInputSizes.joinToString(":")}"
        }

        val started = SystemClock.elapsedRealtimeNanos()
        val outputs = runtime.run(inputs)
        val invokeMs = elapsedMs(started)
        val outputsJson = stage.getJSONArray("outputs")
        require(outputs.size == outputsJson.length()) {
            "output count mismatch runtime=${outputs.size} manifest=${outputsJson.length()}"
        }
        val outputRoot = context.getExternalFilesDir(null)!!
            .resolve("enterprise_tflite_outputs/$variant")
        writeOutputs(outputRoot, stage, outputs)
        emit(
            "enterprise_tflite_stage_ok variant=$variant stage=$stageId model=$modelName " +
                "invoke_ms=${format(invokeMs)} input_sizes=${inputs.joinToString(":") { it.size.toString() }} " +
                "output_sizes=${outputs.joinToString(":") { it.size.toString() }}",
        )
    }

    private fun writeOutputs(outputRoot: File, stage: JSONObject, outputs: List<ByteArray>) {
        val outputsJson = stage.getJSONArray("outputs")
        require(outputs.size == outputsJson.length()) {
            "output count mismatch runtime=${outputs.size} manifest=${outputsJson.length()}"
        }
        outputs.forEachIndexed { index, bytes ->
            val output = outputsJson.getJSONObject(index)
            require(bytes.size.toLong() == output.getLong("bytes")) {
                "output bytes mismatch name=${output.getString("name")} runtime=${bytes.size} manifest=${output.getLong("bytes")}"
            }
            val target = outputRoot.resolve(output.getString("vendor_file"))
            target.parentFile?.mkdirs()
            target.writeBytes(bytes)
        }
    }

    private fun elapsedMs(started: Long): Double =
        (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0

    private fun format(value: Double): String = "%.3f".format(Locale.US, value)

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
