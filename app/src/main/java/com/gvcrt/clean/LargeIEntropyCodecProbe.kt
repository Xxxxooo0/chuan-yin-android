package com.gvcrt.clean

import android.content.Context
import android.os.SystemClock
import com.mediatek.neuropilot_V.neuron.NeuronDelegate
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest
import java.util.Locale

/** I-frame online codec probe using the single merged entropy/prior TFLite model. */
class LargeIEntropyCodecProbe(
    private val context: Context,
    private val emit: (String) -> Unit,
) {
    fun run(
        warmupRuns: Int = 0,
        measuredRuns: Int = 1,
        validateRoundtrip: Boolean = true,
        dumpOutputs: Boolean = true,
    ) {
        require(warmupRuns >= 0 && measuredRuns > 0) {
            "invalid benchmark warmup=$warmupRuns measured=$measuredRuns"
        }
        val packageRoot = findPackageRoot()
        val manifestFile = packageRoot.resolve(MANIFEST)
        require(manifestFile.isFile) {
            "missing $MANIFEST; install the Large online package first: ${manifestFile.absolutePath}"
        }
        val manifest = JSONObject(manifestFile.readText())
        val i = manifest.getJSONObject("i")
        val mergedModel = i.getJSONObject("merged_model")
        require(mergedModel.getString("name") == MERGED_MODEL) {
            "unsupported I entropy model: ${mergedModel.getString("name")}"
        }
        val mergedFile = resolvePackageFile(packageRoot, mergedModel.getString("file"))
        require(mergedFile.isFile) {
            "missing standalone merged model: ${mergedFile.absolutePath}"
        }
        val mergedSha = sha256(mergedFile)
        require(mergedSha.equals(mergedModel.getString("sha256"), ignoreCase = true)) {
            "merged model SHA mismatch actual=$mergedSha expected=${mergedModel.getString("sha256")}"
        }

        val runtimes = linkedMapOf<String, OfficialNeuronRuntime>()
        try {
            emit(
                "large_i_codec_start backend=official_aar_neuron entropy_io=nhwc_merged " +
                    "allow_fp16=true compile_options=--relax-fp32 preference=FAST_SINGLE_ANSWER " +
                    "warmup=$warmupRuns measured=$measuredRuns roundtrip=$validateRoundtrip " +
                    "qp=${manifest.getInt("qp")} package=${manifest.getString("package")} root=${packageRoot.absolutePath}",
            )
            emit(
                "large_i_codec_model name=$MERGED_MODEL file=${mergedFile.absolutePath} " +
                    "bytes=${mergedFile.length()} sha256=$mergedSha packaging=standalone",
            )

            fun runtime(name: String, explicitModel: File? = null): OfficialNeuronRuntime = runtimes.getOrPut(name) {
                createRuntime(packageRoot, name, explicitModel)
            }

            val inputFrame = resolvePackageFile(packageRoot, i.getString("input_i_frame")).readBytes()
            val encoderRuntime = runtime("i_encoder")
            val encoder = runRepeated(
                label = "i_encoder",
                runtime = encoderRuntime,
                inputs = listOf(inputFrame),
                warmupRuns = warmupRuns,
                measuredRuns = measuredRuns,
            )
            val y = NhwcTensorCodec.fromF32Le("i_y_pre_prior", Y_SHAPE, encoder.outputs.single())

            val entropyRuntime = runtime(MERGED_MODEL, mergedFile)
            val entropyRun = runRepeated(
                label = "i_entropy_merged",
                runtime = entropyRuntime,
                inputs = listOf(NhwcFloatTensor.fromNchw(y, "i_y_pre_prior").toF32Le()),
                warmupRuns = warmupRuns,
                measuredRuns = measuredRuns,
            )
            val entropyOutputs = decodeMergedOutputs(entropyRun.outputs)
            val entropy = readEntropySpec(packageRoot, i)
            val rans = runRansRepeated(
                entropyOutputs = entropyOutputs,
                entropy = entropy,
                warmupRuns = warmupRuns,
                measuredRuns = measuredRuns,
            )

            val roundtripStatus = if (validateRoundtrip) {
                validateRansRoundtrip(rans.payload, entropyOutputs, entropy)
                "PASS"
            } else {
                emit("large_i_codec_diagnostic stage=i_rans_roundtrip status=SKIPPED excluded_from_encode_total=true")
                "SKIPPED"
            }

            val decoderRuntime = runtime("i_decoder")
            val decoder = runRepeated(
                label = "i_decoder",
                runtime = decoderRuntime,
                inputs = listOf(NhwcTensorCodec.toF32Le(entropyOutputs.yHat)),
                warmupRuns = warmupRuns,
                measuredRuns = measuredRuns,
            )
            val frame = NhwcTensorCodec.fromF32Le("i_reference_frame", FRAME_SHAPE, decoder.outputs.single())
            val steadyComponentSum = encoder.stats.meanMs + entropyRun.stats.meanMs + rans.stats.meanMs + decoder.stats.meanMs
            emit(
                "large_i_codec_speed stage=component_sum samples=$measuredRuns " +
                    "mean_ms=${formatMs(steadyComponentSum)} " +
                    "note=component_means_excludes_create_roundtrip_dump_and_layout_boundaries",
            )
            if (warmupRuns > 0 || measuredRuns > 1) {
                runPipelineRepeated(
                    encoderRuntime = encoderRuntime,
                    entropyRuntime = entropyRuntime,
                    decoderRuntime = decoderRuntime,
                    inputFrame = inputFrame,
                    entropy = entropy,
                    warmupRuns = warmupRuns,
                    measuredRuns = measuredRuns,
                )
            }

            val outputRoot = context.getExternalFilesDir(null)!!
                .resolve("enterprise_tflite_codec/large/i")
            if (dumpOutputs) {
                writeOutputs(outputRoot, entropyOutputs, rans.payload, frame)
            }
            emit(
                "large_i_codec_complete payload_bytes=${rans.payload.size} payload_sha256=${sha256(rans.payload)} " +
                    "rans_roundtrip=$roundtripStatus dump_outputs=$dumpOutputs output=${outputRoot.absolutePath}",
            )
        } finally {
            runtimes.values.forEach(OfficialNeuronRuntime::close)
        }
    }

    private fun runRepeated(
        label: String,
        runtime: OfficialNeuronRuntime,
        inputs: List<ByteArray>,
        warmupRuns: Int,
        measuredRuns: Int,
    ): RuntimeRun {
        repeat(warmupRuns) { runtime.run(inputs, copyOutputs = false) }
        val times = DoubleArray(measuredRuns)
        var outputs: List<ByteArray> = emptyList()
        repeat(measuredRuns) { index ->
            val started = SystemClock.elapsedRealtimeNanos()
            outputs = runtime.run(inputs, copyOutputs = index == measuredRuns - 1)
            times[index] = elapsedMs(started)
        }
        val stats = emitSpeed(label, times, "runtime_direct_buffers_reused")
        return RuntimeRun(outputs, stats)
    }

    private fun runRansRepeated(
        entropyOutputs: IEntropyOutputs,
        entropy: LocalEntropySpec,
        warmupRuns: Int,
        measuredRuns: Int,
    ): RansRun {
        repeat(warmupRuns) { encodePayload(entropyOutputs, entropy) }
        val times = DoubleArray(measuredRuns)
        var payload = ByteArray(0)
        repeat(measuredRuns) { index ->
            val started = SystemClock.elapsedRealtimeNanos()
            payload = encodePayload(entropyOutputs, entropy)
            times[index] = elapsedMs(started)
        }
        return RansRun(payload, emitSpeed("i_rans", times, "not_applicable"))
    }

    private fun runPipelineRepeated(
        encoderRuntime: OfficialNeuronRuntime,
        entropyRuntime: OfficialNeuronRuntime,
        decoderRuntime: OfficialNeuronRuntime,
        inputFrame: ByteArray,
        entropy: LocalEntropySpec,
        warmupRuns: Int,
        measuredRuns: Int,
    ) {
        fun runOnce() {
            val yBytes = encoderRuntime.run(listOf(inputFrame)).single()
            val y = NhwcTensorCodec.fromF32Le("i_y_pre_prior", Y_SHAPE, yBytes)
            val mergedBytes = entropyRuntime.run(
                listOf(NhwcFloatTensor.fromNchw(y, "i_y_pre_prior").toF32Le()),
            )
            val entropyOutputs = decodeMergedOutputs(mergedBytes)
            encodePayload(entropyOutputs, entropy)
            decoderRuntime.run(
                listOf(NhwcTensorCodec.toF32Le(entropyOutputs.yHat)),
                copyOutputs = false,
            )
        }

        repeat(warmupRuns) { runOnce() }
        val times = DoubleArray(measuredRuns)
        repeat(measuredRuns) { index ->
            val started = SystemClock.elapsedRealtimeNanos()
            runOnce()
            times[index] = elapsedMs(started)
        }
        emitSpeed("i_pipeline_steady", times, "runtime_direct_buffers_reused")
    }

    private fun encodePayload(entropyOutputs: IEntropyOutputs, entropy: LocalEntropySpec): ByteArray =
        RansNativeEncoder.create(entropy.gaussian, entropy.z, useTwoEncoders = false).use { encoder ->
            encoder.encodeZ(
                EntropySymbols.zSymbols(entropyOutputs.zHat),
                entropy.zStartOffset,
                entropy.zPerChannelSize,
            )
            entropyOutputs.quantized.forEach { encoder.encodeY(it.symbols, it.scales) }
            encoder.flush()
        }

    private fun validateRansRoundtrip(
        payload: ByteArray,
        entropyOutputs: IEntropyOutputs,
        entropy: LocalEntropySpec,
    ) {
        val started = SystemClock.elapsedRealtimeNanos()
        NativeRans.create(entropy.gaussian, entropy.z).use { decoder ->
            val decoded = decoder.decode(
                payload,
                entropyOutputs.zHat.numel,
                entropy.zStartOffset,
                entropy.zPerChannelSize,
                entropyOutputs.quantized.map { EntropySymbols.indexesForScales(it.scales) }.toTypedArray(),
            )
            require(decoded[0].contentEquals(EntropySymbols.zSymbols(entropyOutputs.zHat))) {
                "i rANS z roundtrip mismatch"
            }
            entropyOutputs.quantized.forEachIndexed { index, stage ->
                require(decoded[index + 1].contentEquals(symbolBytes(stage.symbols))) {
                    "i rANS y stage=$index roundtrip mismatch"
                }
            }
        }
        emit(
            "large_i_codec_diagnostic stage=i_rans_roundtrip status=PASS " +
                "elapsed_ms=${formatMs(elapsedMs(started))} excluded_from_encode_total=true",
        )
    }

    private fun decodeMergedOutputs(outputBytes: List<ByteArray>): IEntropyOutputs {
        require(outputBytes.size == 10) {
            "$MERGED_MODEL output count=${outputBytes.size}, expected=10"
        }
        val zHat = NhwcFloatTensor.fromF32Le(
            "i_z_hat_nhwc",
            Z_NHWC_SHAPE,
            outputBytes[0],
        ).toNchw("i_z_hat")
        val quantized = (0 until 4).map { stage ->
            val symbols = NhwcFloatTensor.fromF32Le(
                "i_y_q_w_${stage}_nhwc",
                PACKED_NHWC_SHAPE,
                outputBytes[1 + stage],
            ).toNchw("i_y_q_w_$stage")
            val scales = NhwcFloatTensor.fromF32Le(
                "i_s_w_${stage}_nhwc",
                PACKED_NHWC_SHAPE,
                outputBytes[5 + stage],
            ).toNchw("i_s_w_$stage")
            PriorSymbols(symbols, scales)
        }
        val yHat = NhwcFloatTensor.fromF32Le(
            "i_y_hat_nhwc",
            Y_NHWC_SHAPE,
            outputBytes[9],
        ).toNchw("i_y_hat")
        return IEntropyOutputs(zHat, quantized, yHat)
    }

    private fun writeOutputs(
        outputRoot: File,
        entropyOutputs: IEntropyOutputs,
        payload: ByteArray,
        frame: TensorValue,
    ) {
        outputRoot.mkdirs()
        val tensors = linkedMapOf<String, TensorValue>()
        tensors["i_z_hat"] = entropyOutputs.zHat
        entropyOutputs.quantized.forEachIndexed { index, stage ->
            tensors["i_y_q_w_$index"] = stage.symbols
            tensors["i_s_w_$index"] = stage.scales
        }
        tensors["i_y_hat"] = entropyOutputs.yHat
        tensors["i_reference_frame"] = frame
        val records = JSONArray()
        tensors.forEach { (name, tensor) ->
            val bytes = TensorIO.f32Le(tensor)
            val fileName = "$name.nchw.f32le"
            outputRoot.resolve(fileName).writeBytes(bytes)
            records.put(
                JSONObject()
                    .put("name", name)
                    .put("file", fileName)
                    .put("shape", JSONArray(tensor.shape.toList()))
                    .put("sha256", sha256(bytes)),
            )
            emit(
                "large_i_codec_output tensor=$name shape=${TensorIO.shapeText(tensor.shape)} " +
                    "sha256=${sha256(bytes)}",
            )
        }
        outputRoot.resolve("i_rans_payload.bin").writeBytes(payload)
        val manifest = JSONObject()
            .put("layout", "NCHW")
            .put("dtype", "float32 little-endian")
            .put("tensors", records)
            .put("payload", JSONObject().put("file", "i_rans_payload.bin").put("sha256", sha256(payload)))
        outputRoot.resolve("output_manifest.json").writeText(manifest.toString(2))
    }

    private fun createRuntime(packageRoot: File, name: String, explicitModel: File? = null): OfficialNeuronRuntime {
        val model = explicitModel ?: packageRoot.resolve("models/$name.tflite")
        require(model.isFile) { "missing TFLite model: ${model.absolutePath}" }
        val sha = sha256(model)
        val cacheDir = context.cacheDir.resolve("enterprise_tflite/large/online_relax_fp32/$name")
        val cacheFilesBefore = cacheDir.listFiles()?.size ?: 0
        val started = SystemClock.elapsedRealtimeNanos()
        return OfficialNeuronRuntime.create(
            tfliteFile = model,
            cacheDir = cacheDir,
            allowFp16ForFp32 = true,
            acceleratorName = "mtk-neuron",
            compileOptions = "--relax-fp32",
            executionPreference = NeuronDelegate.Options.EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER,
            modelToken = "gvcrt_large_${name}_${sha.take(12)}",
        ).also {
            val cacheFilesAfter = cacheDir.listFiles()?.size ?: 0
            emit(
                "large_i_codec_create model=$name create_ms=${formatMs(elapsedMs(started))} " +
                    "cache_files_before=$cacheFilesBefore cache_files_after=$cacheFilesAfter options=${it.optionsSummary}",
            )
        }
    }

    private fun emitSpeed(label: String, times: DoubleArray, bufferMode: String): SpeedStats {
        val stats = SpeedStats(
            meanMs = times.average(),
            p50Ms = percentile(times, 0.50),
            p90Ms = percentile(times, 0.90),
        )
        emit(
                "large_i_codec_speed stage=$label samples=${times.size} mean_ms=${formatMs(stats.meanMs)} " +
                    "p50_ms=${formatMs(stats.p50Ms)} p90_ms=${formatMs(stats.p90Ms)} " +
                    "includes_create=false buffers=$bufferMode",
        )
        return stats
    }

    private fun readEntropySpec(root: File, json: JSONObject): LocalEntropySpec = LocalEntropySpec(
        gaussian = readCdf(root, json.getJSONObject("gaussian")),
        z = readCdf(root, json.getJSONObject("z")),
        zStartOffset = json.getInt("z_start_offset"),
        zPerChannelSize = json.getInt("z_per_channel_size"),
    )

    private fun readCdf(root: File, json: JSONObject): CdfTable {
        val shape = json.getJSONArray("shape")
        val rows = shape.getInt(0)
        val stride = shape.getInt(1)
        val values = TensorIO.readI32Le(json.getString("cdf"), resolvePackageFile(root, json.getString("cdf")).readBytes())
        val lengths = TensorIO.readI32Le(
            json.getString("cdf_lengths"),
            resolvePackageFile(root, json.getString("cdf_lengths")).readBytes(),
        )
        val offsets = TensorIO.readI32Le(
            json.getString("offsets"),
            resolvePackageFile(root, json.getString("offsets")).readBytes(),
        )
        require(values.size == rows * stride && lengths.size == rows && offsets.size == rows) { "invalid CDF table" }
        return CdfTable(values, rows, stride, lengths, offsets)
    }

    private fun resolvePackageFile(root: File, relative: String): File {
        val canonicalRoot = root.canonicalFile
        val target = canonicalRoot.resolve(relative).canonicalFile
        require(target.path == canonicalRoot.path || target.path.startsWith(canonicalRoot.path + File.separator)) {
            "package path escapes root: $relative"
        }
        return target
    }

    private fun symbolBytes(symbols: TensorValue): ByteArray =
        ByteArray(symbols.numel) { index -> symbols.data[index].toInt().toByte() }

    private fun findPackageRoot(): File {
        val internal = context.filesDir.resolve("enterprise_tflite/large")
        val external = context.getExternalFilesDir(null)?.resolve("enterprise_tflite/large")
        return listOfNotNull(internal, external).firstOrNull { it.resolve(MANIFEST).isFile } ?: internal
    }

    private fun elapsedMs(started: Long): Double =
        (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0

    private fun sha256(file: File): String = FileInputStream(file).use { input ->
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(1024 * 1024)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            digest.update(buffer, 0, count)
        }
        digest.digest().joinToString("") { "%02x".format(Locale.US, it) }
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { "%02x".format(Locale.US, it) }

    private fun formatMs(value: Double): String = "%.3f".format(Locale.US, value)

    private fun percentile(values: DoubleArray, fraction: Double): Double {
        val sorted = values.sortedArray()
        val index = ((sorted.size - 1) * fraction).toInt().coerceIn(sorted.indices)
        return sorted[index]
    }

    private data class RuntimeRun(val outputs: List<ByteArray>, val stats: SpeedStats)
    private data class RansRun(val payload: ByteArray, val stats: SpeedStats)
    private data class SpeedStats(val meanMs: Double, val p50Ms: Double, val p90Ms: Double)
    private data class PriorSymbols(val symbols: TensorValue, val scales: TensorValue)
    private data class IEntropyOutputs(
        val zHat: TensorValue,
        val quantized: List<PriorSymbols>,
        val yHat: TensorValue,
    )
    private data class LocalEntropySpec(
        val gaussian: CdfTable,
        val z: CdfTable,
        val zStartOffset: Int,
        val zPerChannelSize: Int,
    )

    private companion object {
        const val MANIFEST = "large_entropy_manifest.json"
        const val MERGED_MODEL = "i_entropy_prior_merged"
        val FRAME_SHAPE = longArrayOf(1, 3, 256, 512)
        val Y_SHAPE = longArrayOf(1, 256, 16, 32)
        val Z_NHWC_SHAPE = longArrayOf(1, 4, 8, 128)
        val Y_NHWC_SHAPE = longArrayOf(1, 16, 32, 256)
        val PACKED_NHWC_SHAPE = longArrayOf(1, 16, 32, 64)
    }
}
