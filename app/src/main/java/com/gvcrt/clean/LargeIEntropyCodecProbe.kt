package com.gvcrt.clean

import android.content.Context
import android.os.SystemClock
import com.mediatek.neuropilot_V.neuron.NeuronDelegate
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest
import java.util.Locale

/**
 * I-frame online TFLite codec probe.
 *
 * Continuous entropy networks run through NeuronDelegate. Masked quantization,
 * CDF selection and rANS stay in the existing native CPU implementation.
 */
class LargeIEntropyCodecProbe(
    private val context: Context,
    private val emit: (String) -> Unit,
) {
    fun run(
        useNhwcEntropy: Boolean = false,
        useMergedEntropy: Boolean = false,
        compareMergedAgainstNhwc: Boolean = true,
        mergedWarmupRuns: Int = 3,
        mergedMeasuredRuns: Int = 10,
    ) {
        require(!(useNhwcEntropy && useMergedEntropy)) {
            "NHWC split and merged entropy modes are mutually exclusive"
        }
        val packageRoot = findPackageRoot()
        val manifestFile = packageRoot.resolve(MANIFEST)
        require(manifestFile.isFile) {
            "missing $MANIFEST; export and install the Large entropy package first: ${manifestFile.absolutePath}"
        }
        val manifest = JSONObject(manifestFile.readText())
        val i = manifest.getJSONObject("i")
        val forceZero = manifest.getDouble("force_zero_thres").toFloat()
        val runtimes = linkedMapOf<String, OfficialNeuronRuntime>()
        try {
            emit(
                "large_i_codec_start backend=official_aar_neuron profile=mlvc_relax_fp32 " +
                    "entropy_io=${when {
                        useMergedEntropy -> "nhwc_merged"
                        useNhwcEntropy -> "nhwc_direct"
                        else -> "legacy_nchw"
                    }} " +
                    "qp=${manifest.getInt("qp")} package=${manifest.getString("package")} root=${packageRoot.absolutePath}",
            )
            fun runtime(name: String): OfficialNeuronRuntime = runtimes.getOrPut(name) {
                createRuntime(packageRoot, name)
            }

            val inputFrame = packageRoot.resolve(i.getString("input_i_frame")).readBytes()
            val y = timed("i_encoder") {
                runNchw(
                    runtime("i_encoder"),
                    listOf(inputFrame),
                    listOf(TensorSpec("i_y_pre_prior", Y_SHAPE)),
                ).single()
            }
            val entropyOutputs = when {
                useMergedEntropy -> {
                    val merged = runEntropyMerged(
                        y,
                        ::runtime,
                        mergedWarmupRuns,
                        mergedMeasuredRuns,
                    )
                    if (compareMergedAgainstNhwc) {
                        val split = runEntropyNhwc(y, forceZero, ::runtime)
                        emitMergedComparison(merged, split)
                    }
                    merged
                }
                useNhwcEntropy -> runEntropyNhwc(y, forceZero, ::runtime)
                else -> runEntropyLegacy(y, forceZero, ::runtime)
            }
            val zHat = entropyOutputs.zHat
            val quantized = entropyOutputs.quantized
            val yHat = entropyOutputs.yHat
            val entropy = readEntropySpec(packageRoot, i)
            val payload = timed("i_rans") {
                RansNativeEncoder.create(entropy.gaussian, entropy.z, useTwoEncoders = false).use { encoder ->
                    encoder.encodeZ(EntropySymbols.zSymbols(zHat), entropy.zStartOffset, entropy.zPerChannelSize)
                    quantized.forEach { encoder.encodeY(it.symbols, it.scales) }
                    encoder.flush()
                }
            }
            timed("i_rans_roundtrip") {
                NativeRans.create(entropy.gaussian, entropy.z).use { decoder ->
                    val decoded = decoder.decode(
                        payload,
                        zHat.numel,
                        entropy.zStartOffset,
                        entropy.zPerChannelSize,
                        quantized.map { EntropySymbols.indexesForScales(it.scales) }.toTypedArray(),
                    )
                    require(decoded[0].contentEquals(EntropySymbols.zSymbols(zHat))) { "i rANS z roundtrip mismatch" }
                    quantized.forEachIndexed { index, stage ->
                        require(decoded[index + 1].contentEquals(symbolBytes(stage.symbols))) {
                            "i rANS y stage=$index roundtrip mismatch"
                        }
                    }
                }
            }
            val frame = timed("i_decoder") {
                runNchw(
                    runtime("i_decoder"),
                    listOf(NhwcTensorCodec.toF32Le(yHat)),
                    listOf(TensorSpec("i_reference_frame", FRAME_SHAPE)),
                ).single()
            }
            val outputRoot = context.getExternalFilesDir(null)!!
                .resolve("enterprise_tflite_codec/large/i")
            outputRoot.mkdirs()
            outputRoot.resolve("i_rans_payload.bin").writeBytes(payload)
            outputRoot.resolve("i_y_hat.nchw.f32le").writeBytes(TensorIO.f32Le(yHat))
            outputRoot.resolve("i_reference_frame.nchw.f32le").writeBytes(TensorIO.f32Le(frame))
            emit(
                "large_i_codec_complete payload_bytes=${payload.size} payload_sha256=${sha256(payload)} " +
                    "rans_roundtrip=PASS output=${outputRoot.absolutePath}",
            )
        } finally {
            runtimes.values.forEach(OfficialNeuronRuntime::close)
        }
    }

    private fun runEntropyLegacy(
        y: TensorValue,
        forceZero: Float,
        runtime: (String) -> OfficialNeuronRuntime,
    ): IEntropyOutputs {
        val zPreQuant = timed("i_hyper_enc") {
            runNchw(
                runtime("i_hyper_enc_continuous"),
                listOf(NhwcTensorCodec.toF32Le(y)),
                listOf(TensorSpec("i_z_pre_quant", Z_SHAPE)),
            ).single()
        }
        val zHat = timed("i_z_quant") { quantizeInt8(zPreQuant, "i_z_hat") }
        val common = timed("i_hyper_prior") {
            runNchw(
                runtime("i_hyper_prior_shared"),
                listOf(NhwcTensorCodec.toF32Le(zHat)),
                listOf(TensorSpec("i_common_params", COMMON_SHAPE)),
            ).single()
        }
        val stage0 = timed("i_prior_stage0") {
            runNchw(
                runtime("i_prior_stage0_params"),
                listOf(NhwcTensorCodec.toF32Le(common)),
                listOf(
                    TensorSpec("i_q_enc", Q_SHAPE),
                    TensorSpec("i_q_dec", Q_SHAPE),
                    TensorSpec("i_stage0_scales", Y_SHAPE),
                    TensorSpec("i_stage0_means", Y_SHAPE),
                ),
            )
        }
        val reduced = timed("i_prior_reduce") {
            runNchw(
                runtime("i_prior_reduce"),
                listOf(NhwcTensorCodec.toF32Le(common)),
                listOf(TensorSpec("i_reduced_common_params", Y_SHAPE)),
            ).single()
        }
        val yScaled = EntropyPriorQuantizer.multiply(y, stage0[0], "i_y_scaled")
        val quantized = ArrayList<PriorSymbols>(4)
        var yHatSoFar = timed("i_quant_stage0") {
            EntropyPriorQuantizer.quantize(yScaled, stage0[3], stage0[2], 0, 4, forceZero).also {
                quantized += PriorSymbols(it.symbols, it.scales)
            }.yHat
        }
        for (stage in 1..3) {
            val params = timed("i_prior_stage$stage") {
                runNchw(
                    runtime("i_prior_stage${stage}_continuous"),
                    listOf(NhwcTensorCodec.toF32Le(yHatSoFar), NhwcTensorCodec.toF32Le(reduced)),
                    listOf(
                        TensorSpec("i_stage${stage}_scales", Y_SHAPE),
                        TensorSpec("i_stage${stage}_means", Y_SHAPE),
                    ),
                )
            }
            val current = timed("i_quant_stage$stage") {
                EntropyPriorQuantizer.quantize(yScaled, params[1], params[0], stage, 4, forceZero).also {
                    quantized += PriorSymbols(it.symbols, it.scales)
                }
            }
            yHatSoFar = EntropyPriorQuantizer.add(yHatSoFar, current.yHat, "i_y_hat_so_far_$stage")
        }
        return IEntropyOutputs(
            zHat,
            quantized,
            EntropyPriorQuantizer.multiply(yHatSoFar, stage0[1], "i_y_hat"),
        )
    }

    private fun runEntropyNhwc(
        y: TensorValue,
        forceZero: Float,
        runtime: (String) -> OfficialNeuronRuntime,
    ): IEntropyOutputs {
        val yNhwc = NhwcFloatTensor.fromNchw(y)
        val zPreQuant = timed("i_hyper_enc") {
            NhwcFloatTensor.fromF32Le(
                "i_z_pre_quant",
                Z_NHWC_SHAPE,
                runtime("i_hyper_enc_continuous").run(listOf(yNhwc.toF32Le())).single(),
            )
        }
        val (zHatNhwc, zHat) = timed("i_z_quant") {
            NhwcEntropyPriorQuantizer.quantizeInt8(zPreQuant, "i_z_hat_nhwc", "i_z_hat")
        }
        val commonBytes = timed("i_hyper_prior") {
            runtime("i_hyper_prior_shared").run(listOf(zHatNhwc.toF32Le())).single()
        }
        val stage0Bytes = timed("i_prior_stage0") {
            runtime("i_prior_stage0_params").run(listOf(commonBytes))
        }
        require(stage0Bytes.size == 4) { "i_prior_stage0 output count=${stage0Bytes.size}" }
        val qEnc = NhwcFloatTensor.fromF32Le("i_q_enc", Q_NHWC_SHAPE, stage0Bytes[0])
        val qDec = NhwcFloatTensor.fromF32Le("i_q_dec", Q_NHWC_SHAPE, stage0Bytes[1])
        val stage0Scales = NhwcFloatTensor.fromF32Le("i_stage0_scales", Y_NHWC_SHAPE, stage0Bytes[2])
        val stage0Means = NhwcFloatTensor.fromF32Le("i_stage0_means", Y_NHWC_SHAPE, stage0Bytes[3])
        val reducedBytes = timed("i_prior_reduce") {
            runtime("i_prior_reduce").run(listOf(commonBytes)).single()
        }
        require(reducedBytes.size == Y_NHWC_SHAPE.fold(1L) { product, value -> product * value }.toInt() * 4) {
            "i_reduced_common_params byte count=${reducedBytes.size}"
        }
        val yScaled = NhwcEntropyPriorQuantizer.multiply(yNhwc, qEnc, "i_y_scaled")
        val quantized = ArrayList<PriorSymbols>(4)
        var yHatSoFar = timed("i_quant_stage0") {
            NhwcEntropyPriorQuantizer.quantize(yScaled, stage0Means, stage0Scales, 0, 4, forceZero).also {
                quantized += PriorSymbols(it.symbols, it.scales)
            }.yHat
        }
        for (stage in 1..3) {
            val params = timed("i_prior_stage$stage") {
                runtime("i_prior_stage${stage}_continuous").run(
                    listOf(yHatSoFar.toF32Le(), reducedBytes),
                )
            }
            require(params.size == 2) { "i_prior_stage$stage output count=${params.size}" }
            val scales = NhwcFloatTensor.fromF32Le("i_stage${stage}_scales", Y_NHWC_SHAPE, params[0])
            val means = NhwcFloatTensor.fromF32Le("i_stage${stage}_means", Y_NHWC_SHAPE, params[1])
            val current = timed("i_quant_stage$stage") {
                NhwcEntropyPriorQuantizer.quantize(yScaled, means, scales, stage, 4, forceZero).also {
                    quantized += PriorSymbols(it.symbols, it.scales)
                }
            }
            yHatSoFar = NhwcEntropyPriorQuantizer.add(yHatSoFar, current.yHat, "i_y_hat_so_far_$stage")
        }
        val yHat = NhwcEntropyPriorQuantizer.multiply(yHatSoFar, qDec, "i_y_hat_nhwc").toNchw("i_y_hat")
        return IEntropyOutputs(zHat, quantized, yHat)
    }

    private fun runEntropyMerged(
        y: TensorValue,
        runtime: (String) -> OfficialNeuronRuntime,
        warmupRuns: Int,
        measuredRuns: Int,
    ): IEntropyOutputs {
        require(warmupRuns >= 0 && measuredRuns > 0) {
            "invalid merged benchmark warmup=$warmupRuns measured=$measuredRuns"
        }
        val mergedRuntime = runtime("i_entropy_prior_merged")
        val input = NhwcFloatTensor.fromNchw(y, "i_y_pre_prior").toF32Le()
        repeat(warmupRuns) { mergedRuntime.run(listOf(input), copyOutputs = false) }
        val times = DoubleArray(measuredRuns)
        var outputBytes: List<ByteArray> = emptyList()
        repeat(measuredRuns) { index ->
            val started = SystemClock.elapsedRealtimeNanos()
            outputBytes = mergedRuntime.run(
                listOf(input),
                copyOutputs = index == measuredRuns - 1,
            )
            times[index] = elapsedMs(started)
        }
        emit(
            "large_i_codec_speed stage=i_entropy_merged samples=$measuredRuns " +
                "mean_ms=${format(times.average())} p50_ms=${format(percentile(times, 0.50))} " +
                "p90_ms=${format(percentile(times, 0.90))}",
        )
        require(outputBytes.size == 10) {
            "i_entropy_prior_merged output count=${outputBytes.size}, expected=10"
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

    private fun emitMergedComparison(merged: IEntropyOutputs, split: IEntropyOutputs) {
        emitComparison("i_z_hat", merged.zHat, split.zHat, discrete = true)
        merged.quantized.indices.forEach { stage ->
            emitComparison(
                "i_y_q_w_$stage",
                merged.quantized[stage].symbols,
                split.quantized[stage].symbols,
                discrete = true,
            )
            emitComparison(
                "i_s_w_$stage",
                merged.quantized[stage].scales,
                split.quantized[stage].scales,
                discrete = false,
            )
        }
        emitComparison("i_y_hat", merged.yHat, split.yHat, discrete = false)
    }

    private fun emitComparison(
        name: String,
        actual: TensorValue,
        expected: TensorValue,
        discrete: Boolean,
    ) {
        val diff = TensorIO.diff(actual, expected)
        emit(
            "large_i_entropy_merged_compare tensor=$name kind=${if (discrete) "discrete" else "continuous"} " +
                "exact=${diff.exact} max_abs=${format(diff.maxAbs.toDouble())} " +
                "mean_abs=${format(diff.meanAbs.toDouble())} rmse=${format(diff.rmse.toDouble())}",
        )
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
            emit("large_i_codec_create model=$name create_ms=${format(elapsedMs(started))} options=${it.optionsSummary}")
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
        return block().also { emit("large_i_codec_speed stage=$label elapsed_ms=${format(elapsedMs(started))}") }
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

    private fun percentile(values: DoubleArray, fraction: Double): Double {
        val sorted = values.sortedArray()
        val index = ((sorted.size - 1) * fraction).toInt().coerceIn(sorted.indices)
        return sorted[index]
    }

    private data class TensorSpec(val name: String, val shape: LongArray)
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
        val FRAME_SHAPE = longArrayOf(1, 3, 256, 512)
        val Y_SHAPE = longArrayOf(1, 256, 16, 32)
        val Z_SHAPE = longArrayOf(1, 128, 4, 8)
        val Q_SHAPE = longArrayOf(1, 1, 16, 32)
        val COMMON_SHAPE = longArrayOf(1, 514, 16, 32)
        val Y_NHWC_SHAPE = longArrayOf(1, 16, 32, 256)
        val Z_NHWC_SHAPE = longArrayOf(1, 4, 8, 128)
        val Q_NHWC_SHAPE = longArrayOf(1, 16, 32, 1)
        val PACKED_NHWC_SHAPE = longArrayOf(1, 16, 32, 64)
    }
}
