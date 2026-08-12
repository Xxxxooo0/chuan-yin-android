package com.gvcrt.clean

import android.content.Context
import android.os.SystemClock
import com.mediatek.neuropilot_V.neuron.NeuronDelegate
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.Locale

/** P-frame online TFLite codec probe with native masked quantization and rANS. */
class LargePEntropyCodecProbe(
    private val context: Context,
    private val emit: (String) -> Unit,
) {
    fun runMergedSpeed(warmupRuns: Int = 3, measuredRuns: Int = 10) {
        require(warmupRuns >= 0 && measuredRuns > 0)
        val packageRoot = findPackageRoot()
        val outputRoot = context.getExternalFilesDir(null)!!.resolve("enterprise_tflite_codec/large/p")
        val yFile = outputRoot.resolve("p_y_pre_prior.nchw.f32le")
        val ctxTFile = outputRoot.resolve("p_ctx_t.nchw.f32le")
        require(yFile.isFile && ctxTFile.isFile) {
            "missing P entropy inputs; run largePEntropyCodecTest first"
        }
        val y = TensorIO.readF32Le("p_y_pre_prior", Y_SHAPE, yFile.readBytes())
        val ctxT = TensorIO.readF32Le("p_ctx_t", CTX_SHAPE, ctxTFile.readBytes())
        val yNhwc = NhwcTensorCodec.toF32Le(y)
        val ctxTNhwc = NhwcTensorCodec.toF32Le(ctxT)
        val model = findMergedModel(packageRoot, ENCODER_MERGED_MODEL)
        val createStarted = SystemClock.elapsedRealtimeNanos()
        PEntropyRansMergedRuntime.create(
            model,
            context.cacheDir.resolve("enterprise_tflite/large/p_entropy_prior_merged"),
        ).use { runtime ->
            emit(
                "large_p_entropy_merged_speed_start warmup=$warmupRuns measured=$measuredRuns " +
                    "create_ms=${format(elapsedMs(createStarted))} model_sha256=${sha256(model)} " +
                    "options=${runtime.optionsSummary}",
            )
            repeat(warmupRuns) { runtime.run(yNhwc, ctxTNhwc, copyOutputs = false) }
            val samples = DoubleArray(measuredRuns) {
                val started = SystemClock.elapsedRealtimeNanos()
                runtime.run(yNhwc, ctxTNhwc, copyOutputs = false)
                elapsedMs(started)
            }
            val canonical = runtime.runCanonical(yNhwc, ctxTNhwc)
            emit(
                "large_p_entropy_merged_speed samples=${samples.size} mean_ms=${format(samples.average())} " +
                    "p50_ms=${format(percentile(samples, 0.50))} p90_ms=${format(percentile(samples, 0.90))} " +
                    "includes_create=false payload_bytes=${canonical[1].size} payload_sha256=${sha256(canonical[1])}",
            )
        }
    }

    fun run() {
        val packageRoot = findPackageRoot()
        val manifestFile = packageRoot.resolve(MANIFEST)
        require(manifestFile.isFile) { "missing $MANIFEST: ${manifestFile.absolutePath}" }
        val manifest = JSONObject(manifestFile.readText())
        val p = manifest.optJSONObject("p")
        val runtimes = linkedMapOf<String, OfficialNeuronRuntime>()
        try {
            emit(
                "large_p_codec_start backend=official_aar_neuron profile=mlvc_relax_fp32 " +
                    "qp=${manifest.getInt("qp")} package=${manifest.getString("package")} root=${packageRoot.absolutePath}",
            )
            fun runtime(name: String): OfficialNeuronRuntime = runtimes.getOrPut(name) {
                createRuntime(packageRoot, name)
            }

            val iReferenceFile = context.getExternalFilesDir(null)!!
                .resolve("enterprise_tflite_codec/large/i/i_reference_frame.nchw.f32le")
            require(iReferenceFile.isFile) {
                "missing I reference frame; run largeIEntropyCodecTest first: ${iReferenceFile.absolutePath}"
            }
            val iReference = TensorIO.readF32Le("i_reference_frame", FRAME_SHAPE, iReferenceFile.readBytes())
            val temporalRuntime = runtime("temporal_from_frame")
            val temporal = timed("p_temporal_from_frame") {
                runNchw(
                    temporalRuntime,
                    listOf(NhwcTensorCodec.toF32Le(iReference)),
                    listOf(
                        TensorSpec("p_reference_feature_initial", CTX_SHAPE),
                        TensorSpec("p_ctx", CTX_SHAPE),
                        TensorSpec("p_ctx_t", CTX_SHAPE),
                    ),
                )
            }
            val ctx = temporal[1]
            val ctxT = temporal[2]
            val inputFrame = packageRoot.resolve(findPInputPath(packageRoot, p)).readBytes()
            val encoderRuntime = runtime("p_encoder")
            val y = timed("p_encoder") {
                runNchw(
                    encoderRuntime,
                    listOf(inputFrame, NhwcTensorCodec.toF32Le(ctx)),
                    listOf(TensorSpec("p_y_pre_prior", Y_SHAPE)),
                ).single()
            }
            val mergedModel = findMergedModel(packageRoot, ENCODER_MERGED_MODEL)
            val mergedCreateStarted = SystemClock.elapsedRealtimeNanos()
            val merged = PEntropyRansMergedRuntime.create(
                mergedModel,
                context.cacheDir.resolve("enterprise_tflite/large/p_entropy_prior_merged"),
            )
            emit(
                "large_p_codec_create model=$ENCODER_MERGED_MODEL " +
                    "create_ms=${format(elapsedMs(mergedCreateStarted))} options=${merged.optionsSummary}",
            )
            val mergedRaw = merged.use {
                timed("p_entropy_prior_merged_rans") {
                    it.run(NhwcTensorCodec.toF32Le(y), NhwcTensorCodec.toF32Le(ctxT))
                }
            }
            require(mergedRaw.size == 8) { "P merged entropy outputs=${mergedRaw.size}, expected=8" }
            val zHat = NhwcTensorCodec.fromF32Le("p_z_hat", Z_SHAPE, mergedRaw[0])
            val ySymbols = listOf(
                NhwcTensorCodec.fromF32Le("p_y_q_w_0", PACKED_SHAPE, mergedRaw[1]),
                NhwcTensorCodec.fromF32Le("p_y_q_w_1", PACKED_SHAPE, mergedRaw[2]),
            )
            val yScales = listOf(
                NhwcTensorCodec.fromF32Le("p_s_w_0", PACKED_SHAPE, mergedRaw[3]),
                NhwcTensorCodec.fromF32Le("p_s_w_1", PACKED_SHAPE, mergedRaw[4]),
            )
            val yHat = NhwcTensorCodec.fromF32Le("p_y_hat", Y_SHAPE, mergedRaw[5])
            val payloadSize = ByteBuffer.wrap(mergedRaw[7]).order(ByteOrder.LITTLE_ENDIAN).int
            require(payloadSize in 1..mergedRaw[6].size) { "invalid P merged payload size=$payloadSize" }
            val payload = mergedRaw[6].copyOf(payloadSize)
            val roundtripStatus = if (p != null) {
                val entropy = readEntropySpec(packageRoot, p)
                timed("p_rans_roundtrip") {
                    NativeRans.create(entropy.gaussian, entropy.z).use { decoder ->
                        val decoded = decoder.decode(
                            payload,
                            zHat.numel,
                            entropy.zStartOffset,
                            entropy.zPerChannelSize,
                            yScales.map { EntropySymbols.indexesForScales(it) }.toTypedArray(),
                        )
                        require(decoded[0].contentEquals(EntropySymbols.zSymbols(zHat))) {
                            "p rANS z roundtrip mismatch"
                        }
                        ySymbols.forEachIndexed { index, symbols ->
                            require(decoded[index + 1].contentEquals(symbolBytes(symbols))) {
                                "p rANS y stage=$index roundtrip mismatch"
                            }
                        }
                    }
                }
                "PASS"
            } else {
                emit("large_p_codec_diagnostic stage=p_rans_roundtrip status=SKIPPED reason=external_p_cdf_missing")
                "SKIPPED"
            }
            val decoderRuntime = runtime("p_decoder")
            val decoded = timed("p_decoder") {
                runNchw(
                    decoderRuntime,
                    listOf(NhwcTensorCodec.toF32Le(yHat), NhwcTensorCodec.toF32Le(ctx)),
                    listOf(
                        TensorSpec("p_reference_feature", CTX_SHAPE),
                        TensorSpec("p_reference_frame", FRAME_SHAPE),
                    ),
                )
            }
            val outputRoot = context.getExternalFilesDir(null)!!
                .resolve("enterprise_tflite_codec/large/p")
            writeOutputs(
                outputRoot = outputRoot,
                ctx = ctx,
                ctxT = ctxT,
                y = y,
                zHat = zHat,
                ySymbols = ySymbols,
                yScales = yScales,
                yHat = yHat,
                referenceFeature = decoded[0],
                referenceFrame = decoded[1],
                payload = payload,
            )
            emit(
                "large_p_codec_complete payload_bytes=${payload.size} payload_sha256=${sha256(payload)} " +
                    "rans_roundtrip=$roundtripStatus output=${outputRoot.absolutePath}",
            )
        } finally {
            runtimes.values.forEach(OfficialNeuronRuntime::close)
        }
    }

    private fun writeOutputs(
        outputRoot: File,
        ctx: TensorValue,
        ctxT: TensorValue,
        y: TensorValue,
        zHat: TensorValue,
        ySymbols: List<TensorValue>,
        yScales: List<TensorValue>,
        yHat: TensorValue,
        referenceFeature: TensorValue,
        referenceFrame: TensorValue,
        payload: ByteArray,
    ) {
        outputRoot.mkdirs()
        val tensors = linkedMapOf<String, TensorValue>()
        tensors["p_ctx"] = ctx
        tensors["p_ctx_t"] = ctxT
        tensors["p_y_pre_prior"] = y
        tensors["p_z_hat"] = zHat
        ySymbols.forEachIndexed { index, symbols ->
            tensors["p_y_q_w_$index"] = symbols
            tensors["p_s_w_$index"] = yScales[index]
        }
        tensors["p_y_hat"] = yHat
        tensors["p_reference_feature"] = referenceFeature
        tensors["p_reference_frame"] = referenceFrame
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
                "large_p_codec_output tensor=$name shape=${TensorIO.shapeText(tensor.shape)} " +
                    "sha256=${sha256(bytes)}",
            )
        }
        outputRoot.resolve("p_rans_payload.bin").writeBytes(payload)
        val manifest = JSONObject()
            .put("layout", "NCHW")
            .put("dtype", "float32 little-endian")
            .put("tensors", records)
            .put("payload", JSONObject().put("file", "p_rans_payload.bin").put("sha256", sha256(payload)))
        outputRoot.resolve("output_manifest.json").writeText(manifest.toString(2))
    }

    private fun runNchw(
        runtime: OfficialNeuronRuntime,
        inputs: List<ByteArray>,
        outputs: List<TensorSpec>,
    ): List<TensorValue> {
        val bytes = runtime.run(inputs)
        require(bytes.size == outputs.size) { "TFLite output count=${bytes.size}, expected=${outputs.size}" }
        return outputs.mapIndexed { index, spec -> NhwcTensorCodec.fromF32Le(spec.name, spec.shape, bytes[index]) }
    }

    private fun createRuntime(packageRoot: File, name: String): OfficialNeuronRuntime {
        val model = packageRoot.resolve("models/$name.tflite")
        require(model.isFile) { "missing TFLite model: ${model.absolutePath}" }
        val sha = sha256(model)
        val started = SystemClock.elapsedRealtimeNanos()
        return OfficialNeuronRuntime.create(
            tfliteFile = model,
            cacheDir = context.cacheDir.resolve("enterprise_tflite/large/mlvc_relax_fp32/$name"),
            allowFp16ForFp32 = true,
            acceleratorName = "mtk-neuron",
            compileOptions = "--relax-fp32",
            executionPreference = NeuronDelegate.Options.EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER,
            modelToken = "gvcrt_large_entropy_${name}_${sha.take(12)}",
        ).also {
            emit("large_p_codec_create model=$name create_ms=${format(elapsedMs(started))} options=${it.optionsSummary}")
        }
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
        val values = TensorIO.readI32Le(json.getString("cdf"), root.resolve(json.getString("cdf")).readBytes())
        val lengths = TensorIO.readI32Le(json.getString("cdf_lengths"), root.resolve(json.getString("cdf_lengths")).readBytes())
        val offsets = TensorIO.readI32Le(json.getString("offsets"), root.resolve(json.getString("offsets")).readBytes())
        require(values.size == rows * stride && lengths.size == rows && offsets.size == rows) { "invalid CDF table" }
        return CdfTable(values, rows, stride, lengths, offsets)
    }

    private fun symbolBytes(symbols: TensorValue): ByteArray =
        ByteArray(symbols.numel) { index -> symbols.data[index].toInt().toByte() }

    private fun findPackageRoot(): File {
        val internal = context.filesDir.resolve("enterprise_tflite/large")
        val external = context.getExternalFilesDir(null)?.resolve("enterprise_tflite/large")
        return listOfNotNull(internal, external).firstOrNull { it.resolve(MANIFEST).isFile } ?: internal
    }

    private fun findPInputPath(packageRoot: File, entropyInfo: JSONObject?): String {
        entropyInfo?.optString("input_p_frame")?.takeIf(String::isNotBlank)?.let { return it }
        val inputManifest = packageRoot.resolve("input_manifest.json")
        require(inputManifest.isFile) { "missing input_manifest.json: ${inputManifest.absolutePath}" }
        val stages = JSONObject(inputManifest.readText()).getJSONArray("stages")
        for (index in 0 until stages.length()) {
            val stage = stages.getJSONObject(index)
            if (stage.optString("model") != "p_encoder") continue
            val inputs = stage.getJSONArray("inputs")
            for (inputIndex in 0 until inputs.length()) {
                val input = inputs.getJSONObject(inputIndex)
                if (input.optString("name") == "input_p_frame") return input.getString("file")
            }
        }
        error("input_manifest.json contains no P encoder frame input")
    }

    private fun findMergedModel(packageRoot: File, name: String): File {
        val candidates = listOf(
            packageRoot.resolve("models/$name"),
            context.getExternalFilesDir(null)!!.resolve("enterprise_tflite_codec/large/$name"),
            context.filesDir.resolve("enterprise_tflite/large/$name"),
        )
        return candidates.firstOrNull(File::isFile)
            ?: error("missing $name; checked ${candidates.joinToString { it.absolutePath }}")
    }

    private fun <T> timed(label: String, block: () -> T): T {
        val started = SystemClock.elapsedRealtimeNanos()
        return block().also {
            emit("large_p_codec_speed stage=$label elapsed_ms=${format(elapsedMs(started))} includes_create=false")
        }
    }

    private fun elapsedMs(started: Long): Double =
        (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0

    private fun percentile(values: DoubleArray, fraction: Double): Double {
        val sorted = values.sortedArray()
        return sorted[((sorted.size - 1) * fraction).toInt().coerceIn(sorted.indices)]
    }

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

    private fun format(value: Double): String = "%.3f".format(Locale.US, value)

    private data class TensorSpec(val name: String, val shape: LongArray)
    private data class LocalEntropySpec(
        val gaussian: CdfTable,
        val z: CdfTable,
        val zStartOffset: Int,
        val zPerChannelSize: Int,
    )

    private companion object {
        const val MANIFEST = "large_entropy_manifest.json"
        val FRAME_SHAPE = longArrayOf(1, 3, 256, 512)
        val CTX_SHAPE = longArrayOf(1, 256, 32, 64)
        val Y_SHAPE = longArrayOf(1, 128, 16, 32)
        val Z_SHAPE = longArrayOf(1, 128, 4, 8)
        val PACKED_SHAPE = longArrayOf(1, 64, 16, 32)
        const val ENCODER_MERGED_MODEL = "p_entropy_prior_merged_rans.tflite"
    }
}
