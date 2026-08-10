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

/** P-frame online TFLite codec probe with native masked quantization and rANS. */
class LargePEntropyCodecProbe(
    private val context: Context,
    private val emit: (String) -> Unit,
) {
    fun run() {
        val packageRoot = findPackageRoot()
        val manifestFile = packageRoot.resolve(MANIFEST)
        require(manifestFile.isFile) { "missing $MANIFEST: ${manifestFile.absolutePath}" }
        val manifest = JSONObject(manifestFile.readText())
        require(manifest.has("p")) { "package does not contain P-frame entropy assets" }
        val p = manifest.getJSONObject("p")
        val forceZero = manifest.getDouble("force_zero_thres").toFloat()
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
            val inputFrame = packageRoot.resolve(p.getString("input_p_frame")).readBytes()
            val encoderRuntime = runtime("p_encoder")
            val y = timed("p_encoder") {
                runNchw(
                    encoderRuntime,
                    listOf(inputFrame, NhwcTensorCodec.toF32Le(ctx)),
                    listOf(TensorSpec("p_y_pre_prior", Y_SHAPE)),
                ).single()
            }
            val hyperEncRuntime = runtime("p_hyper_enc_continuous")
            val zPreQuant = timed("p_hyper_enc") {
                runNchw(
                    hyperEncRuntime,
                    listOf(NhwcTensorCodec.toF32Le(y)),
                    listOf(TensorSpec("p_z_pre_quant", Z_SHAPE)),
                ).single()
            }
            val zHat = timed("p_z_quant") { quantizeInt8(zPreQuant, "p_z_hat") }
            val hyperPriorRuntime = runtime("p_hyper_prior_shared")
            val common = timed("p_hyper_prior") {
                runNchw(
                    hyperPriorRuntime,
                    listOf(NhwcTensorCodec.toF32Le(zHat), NhwcTensorCodec.toF32Le(ctxT)),
                    listOf(TensorSpec("p_common_params", COMMON_SHAPE)),
                ).single()
            }
            val priorStage0Runtime = runtime("p_prior_stage0_params")
            val stage0Params = timed("p_prior_stage0") {
                runNchw(
                    priorStage0Runtime,
                    listOf(NhwcTensorCodec.toF32Le(common)),
                    listOf(
                        TensorSpec("p_q_dec", Y_SHAPE),
                        TensorSpec("p_stage0_scales", Y_SHAPE),
                        TensorSpec("p_stage0_means", Y_SHAPE),
                    ),
                )
            }
            val yScaled = EntropyPriorQuantizer.divide(y, stage0Params[0], "p_y_scaled")
            val quantized = ArrayList<EntropyPriorStage>(2)
            var yHatSoFar = timed("p_quant_stage0") {
                EntropyPriorQuantizer.quantize(
                    yScaled,
                    stage0Params[2],
                    stage0Params[1],
                    phase = 0,
                    groups = 2,
                    forceZeroThreshold = forceZero,
                ).also(quantized::add).yHat
            }
            val priorStage1Runtime = runtime("p_prior_stage1_continuous")
            val stage1Params = timed("p_prior_stage1") {
                runNchw(
                    priorStage1Runtime,
                    listOf(NhwcTensorCodec.toF32Le(yHatSoFar), NhwcTensorCodec.toF32Le(common)),
                    listOf(
                        TensorSpec("p_stage1_scales", Y_SHAPE),
                        TensorSpec("p_stage1_means", Y_SHAPE),
                    ),
                )
            }
            val stage1 = timed("p_quant_stage1") {
                EntropyPriorQuantizer.quantize(
                    yScaled,
                    stage1Params[1],
                    stage1Params[0],
                    phase = 1,
                    groups = 2,
                    forceZeroThreshold = forceZero,
                ).also(quantized::add)
            }
            yHatSoFar = EntropyPriorQuantizer.add(yHatSoFar, stage1.yHat, "p_y_hat_so_far")
            val yHat = EntropyPriorQuantizer.multiply(yHatSoFar, stage0Params[0], "p_y_hat")
            val entropy = readEntropySpec(packageRoot, p)
            val payload = timed("p_rans") {
                RansNativeEncoder.create(entropy.gaussian, entropy.z, useTwoEncoders = false).use { encoder ->
                    encoder.encodeZ(EntropySymbols.zSymbols(zHat), entropy.zStartOffset, entropy.zPerChannelSize)
                    quantized.forEach { encoder.encodeY(it.symbols, it.scales) }
                    encoder.flush()
                }
            }
            timed("p_rans_roundtrip") {
                NativeRans.create(entropy.gaussian, entropy.z).use { decoder ->
                    val decoded = decoder.decode(
                        payload,
                        zHat.numel,
                        entropy.zStartOffset,
                        entropy.zPerChannelSize,
                        quantized.map { EntropySymbols.indexesForScales(it.scales) }.toTypedArray(),
                    )
                    require(decoded[0].contentEquals(EntropySymbols.zSymbols(zHat))) { "p rANS z roundtrip mismatch" }
                    quantized.forEachIndexed { index, stage ->
                        require(decoded[index + 1].contentEquals(symbolBytes(stage.symbols))) {
                            "p rANS y stage=$index roundtrip mismatch"
                        }
                    }
                }
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
                zHat = zHat,
                quantized = quantized,
                yHat = yHat,
                referenceFeature = decoded[0],
                referenceFrame = decoded[1],
                payload = payload,
            )
            emit(
                "large_p_codec_complete payload_bytes=${payload.size} payload_sha256=${sha256(payload)} " +
                    "rans_roundtrip=PASS output=${outputRoot.absolutePath}",
            )
        } finally {
            runtimes.values.forEach(OfficialNeuronRuntime::close)
        }
    }

    private fun writeOutputs(
        outputRoot: File,
        ctx: TensorValue,
        ctxT: TensorValue,
        zHat: TensorValue,
        quantized: List<EntropyPriorStage>,
        yHat: TensorValue,
        referenceFeature: TensorValue,
        referenceFrame: TensorValue,
        payload: ByteArray,
    ) {
        outputRoot.mkdirs()
        val tensors = linkedMapOf<String, TensorValue>()
        tensors["p_ctx"] = ctx
        tensors["p_ctx_t"] = ctxT
        tensors["p_z_hat"] = zHat
        quantized.forEachIndexed { index, stage ->
            tensors["p_y_q_w_$index"] = stage.symbols
            tensors["p_s_w_$index"] = stage.scales
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

    private fun quantizeInt8(input: TensorValue, name: String): TensorValue =
        TensorValue(name, input.shape, FloatArray(input.numel) { index ->
            Math.rint(input.data[index].toDouble()).toInt().coerceIn(-128, 127).toFloat()
        })

    private fun symbolBytes(symbols: TensorValue): ByteArray =
        ByteArray(symbols.numel) { index -> symbols.data[index].toInt().toByte() }

    private fun findPackageRoot(): File {
        val internal = context.filesDir.resolve("enterprise_tflite/large")
        val external = context.getExternalFilesDir(null)?.resolve("enterprise_tflite/large")
        return listOfNotNull(internal, external).firstOrNull { it.resolve(MANIFEST).isFile } ?: internal
    }

    private fun <T> timed(label: String, block: () -> T): T {
        val started = SystemClock.elapsedRealtimeNanos()
        return block().also {
            emit("large_p_codec_speed stage=$label elapsed_ms=${format(elapsedMs(started))} includes_create=false")
        }
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
        val COMMON_SHAPE = longArrayOf(1, 384, 16, 32)
    }
}
