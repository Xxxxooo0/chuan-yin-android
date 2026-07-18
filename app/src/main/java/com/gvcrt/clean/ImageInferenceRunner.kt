package com.gvcrt.clean

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.os.SystemClock
import java.io.File
import java.util.Locale
import kotlin.math.log10
import kotlin.math.sqrt

class ImageInferenceRunner(
    private val context: Context,
    private val emit: (String) -> Unit,
    private val backend: OnnxBackend = OnnxBackend.NNAPI_FP16_ALLOW_FALLBACK,
    private val showImages: ((File, File, Double) -> Unit)? = null,
) : AutoCloseable {
    private val store = AssetStore(context)
    private val manifest = CleanManifest.parse(store.readBytes(MANIFEST).decodeToString())
    private val runner = ProcessOnnxSessionCache.get(context, backend)

    init {
        require(manifest.metadata.optString("precision") == "fp32") {
            "image inference requires the source-matched fp32 manifest"
        }
    }

    fun run(imagePath: String?, decodeFromBitstream: Boolean = false) {
        val image = ImageTensorLoader.load(context, imagePath)
        emit(
            "image_inference_source=${image.source} original=${image.originalWidth}x${image.originalHeight} " +
                "tensor_shape=${TensorIO.shapeText(image.tensor.shape)} backend=${backend.label}"
        )
        val memory = MemorySampler(context, emit)

        val case = manifest.modules["complete_encoder"]?.singleOrNull()
            ?: error("missing image inference encoder case")
        emit("image_graph_variant=legacy_multi_session encoder_steps=${case.steps.size}")
        val decoder = manifest.decoder ?: error("missing decoder specification")
        val stream = manifest.stream ?: error("missing stream specification")
        val iEntropy = manifest.entropy["i"] ?: error("missing I rANS assets")
        val pEntropy = manifest.entropy["p"] ?: error("missing P rANS assets")
        val timer = StageTimer()
        var totalNs = 0L
        var coreCodecNs = 0L
        val iEncoderRans = createRansEncoder(iEntropy)
        val pEncoderRans = createRansEncoder(pEntropy)
        val iRans = if (decodeFromBitstream) createRans(iEntropy) else null
        val pRans = if (decodeFromBitstream) createRans(pEntropy) else null
        try {
            memory.begin("image_inference")
            emit("image_speed_note=onnx_sessions_reused_until_activity_destroyed")
            val started = SystemClock.elapsedRealtimeNanos()
            val inputI = image.tensor.renamed("input_i_frame")
            val inputP = image.tensor.renamed("input_p_frame")
            val staticInputs = timer.measure("static_inputs") {
                preloadStaticInputs(case.steps, mapOf(inputI.name to inputI, inputP.name to inputP))
            }
            val codecStarted = SystemClock.elapsedRealtimeNanos()
            val tensors = runGraphSteps(
                runner,
                case.steps,
                staticInputs,
                timer,
            )
            val iPayload = encodeEntropy("i", iEntropy, tensors, 4, iEncoderRans, timer)
            val pPayload = encodeEntropy("p", pEntropy, tensors, 2, pEncoderRans, timer)
            memory.mark("encode_complete")
            val bitstream = timer.measure("stream_mux") { GvcStreamMuxer.mux(stream, iPayload, pPayload) }
            store.writeOutput("outputs/image_inference_encoded_ip.gvc", bitstream)
            emit(
                "image_bitstream path=outputs/image_inference_encoded_ip.gvc bytes=${bitstream.size} " +
                    "sha256=${AssetStore.sha256(bitstream)}"
            )

            val reconstructed = if (decodeFromBitstream) {
                emit("image_reconstruction_source=decoded_bitstream")
                decodeBitstream(
                    bitstream,
                    decoder,
                    stream,
                    iEntropy,
                    pEntropy,
                    iRans ?: error("missing I rANS decoder"),
                    pRans ?: error("missing P rANS decoder"),
                    runner,
                    timer,
                )
            } else {
                emit("image_reconstruction_source=encoder_local_reference")
                tensors
            }
            coreCodecNs = SystemClock.elapsedRealtimeNanos() - codecStarted
            memory.mark("reconstruction_complete")
            timer.measure("quality_metrics") {
                emitQuality("i_recon_vs_input", inputI, reconstructed.getValue("encoder_i_reference_frame"))
            }
            val finalOutput = reconstructed.getValue("encoder_p_reference_frame")
            val finalPsnr = timer.measure("quality_metrics") {
                emitQuality("p_recon_vs_input", inputP, finalOutput)
            }
            val (inputFile, outputFile) = timer.measure("png_output") {
                writeTensorPng(inputP, "image_input.png") to
                    writeTensorPng(finalOutput, "image_reconstruction.png")
            }
            emit("image_input=${inputFile.absolutePath}")
            emit("image_output=${outputFile.absolutePath}")
            timer.measure("ui_image_publish") {
                showImages?.invoke(inputFile, outputFile, finalPsnr)
            }
            totalNs = SystemClock.elapsedRealtimeNanos() - started
        } finally {
            iEncoderRans.close()
            pEncoderRans.close()
            iRans?.close()
            pRans?.close()
            memory.close()
        }

        emit("image_speed stage=total ms=${formatMs(totalNs)}")
        emit("image_speed stage=core_codec ms=${formatMs(coreCodecNs)}")
        timer.values.forEach { (stage, elapsedNs) ->
            emit("image_speed stage=$stage ms=${formatMs(elapsedNs)}")
        }
        emit("image_inference_status=PASS")
    }

    override fun close() {
        // The process cache owns the runner so Activity recreation keeps sessions warm.
    }

    private fun decodeBitstream(
        bitstream: ByteArray,
        decoder: DecoderSpec,
        expectedStream: StreamSpec,
        iEntropy: EntropySpec,
        pEntropy: EntropySpec,
        iRans: NativeRans,
        pRans: NativeRans,
        runner: OnnxSessionRunner,
        timer: StageTimer,
    ): Map<String, TensorValue> {
        val parsed = timer.measure("stream_parse") { GvcStreamMuxer.demux(bitstream) }
        require(parsed.stream.height == expectedStream.height && parsed.stream.width == expectedStream.width) {
            "decoded stream geometry differs from manifest"
        }
        require(parsed.stream.qp == expectedStream.qp) { "decoded stream QP differs from manifest" }
        val tensors = linkedMapOf<String, TensorValue>()
        decodeEntropy("i", parsed.iPayload, iEntropy, decoder.i, iRans, runner, tensors, timer)
        tensors.putAll(runGraphStep(runner, decoder.i.recon, tensors, timer))
        val temporal = decoder.p.temporal ?: error("missing P temporal decoder step")
        tensors.putAll(runGraphStep(runner, temporal, tensors, timer))
        decodeEntropy("p", parsed.pPayload, pEntropy, decoder.p, pRans, runner, tensors, timer)
        tensors.putAll(runGraphStep(runner, decoder.p.recon, tensors, timer))
        return tensors
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
        rans: RansNativeEncoder,
        timer: StageTimer,
    ): ByteArray {
        val z = timer.measure("${prefix}_rans_z_prepare") {
            EntropySymbols.zSymbols(tensors.getValue("${prefix}_z_hat"))
        }
        timer.measure("${prefix}_rans_z") {
            rans.encodeZ(z, entropy.zStartOffset, entropy.zPerChannelSize)
        }
        repeat(stageCount) { stage ->
            timer.measure("${prefix}_rans_y_$stage") {
                rans.encodeY(
                    tensors.getValue("${prefix}_y_q_w_$stage"),
                    tensors.getValue("${prefix}_s_w_$stage"),
                )
            }
        }
        return timer.measure("${prefix}_rans_flush") {
            rans.flush()
        }
    }

    private fun createRansEncoder(entropy: EntropySpec): RansNativeEncoder =
        RansNativeEncoder.create(
            CdfTable.load(store, entropy.gaussian),
            CdfTable.load(store, entropy.z),
            entropy.twoEntropyCoders,
        )

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

    private fun preloadStaticInputs(
        steps: List<GraphStep>,
        overrides: Map<String, TensorValue>,
    ): Map<String, TensorValue> =
        steps.flatMap { it.inputs }
            .filter { it.source == null }
            .associate { input ->
                val tensor = overrides[input.tensorName] ?: run {
                    val path = input.path ?: error("static input ${input.tensorName} has no path")
                    val shape = input.shape ?: error("static input ${input.tensorName} has no shape")
                    TensorIO.readF32Le(input.tensorName, shape, store.readBytes(path))
                }
                input.tensorName to tensor
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

    private fun emitQuality(label: String, input: TensorValue, reconstruction: TensorValue): Double {
        require(input.data.size == reconstruction.data.size) {
            "$label element mismatch: ${input.data.size} vs ${reconstruction.data.size}"
        }
        var maxAbs = 0.0
        var sumAbs = 0.0
        var sumSq = 0.0
        for (index in input.data.indices) {
            val expected = input.data[index].toDisplayRange()
            val actual = reconstruction.data[index].toDisplayRange()
            val diff = kotlin.math.abs(actual - expected)
            if (diff > maxAbs) maxAbs = diff
            sumAbs += diff
            sumSq += diff * diff
        }
        val count = input.data.size.coerceAtLeast(1)
        val meanAbs = sumAbs / count
        val rmse = sqrt(sumSq / count)
        val psnr = if (rmse == 0.0) Double.POSITIVE_INFINITY else 20.0 * log10(1.0 / rmse)
        emit(
            "image_quality $label max_abs=${formatFloat(maxAbs)} mean_abs=${formatFloat(meanAbs)} " +
                "rmse=${formatFloat(rmse)} psnr_db=${formatFloat(psnr)}"
        )
        return psnr
    }

    private fun writeTensorPng(tensor: TensorValue, fileName: String): File {
        require(tensor.shape.size == 4 && tensor.shape[0] == 1L && tensor.shape[1] == 3L) {
            "expected NCHW RGB tensor, got ${TensorIO.shapeText(tensor.shape)}"
        }
        val height = tensor.shape[2].toInt()
        val width = tensor.shape[3].toInt()
        val plane = height * width
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(plane)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val offset = y * width + x
                pixels[offset] = Color.rgb(
                    tensor.data[offset].toByteRange(),
                    tensor.data[plane + offset].toByteRange(),
                    tensor.data[2 * plane + offset].toByteRange(),
                )
            }
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        val dir = File(context.getExternalFilesDir(null) ?: context.filesDir, "outputs")
        dir.mkdirs()
        val file = File(dir, fileName)
        file.outputStream().use { output ->
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, output)
        }
        bitmap.recycle()
        return file
    }

    private fun TensorValue.renamed(name: String): TensorValue = TensorValue(name, shape, data)

    private fun Float.toDisplayRange(): Double =
        (((coerceIn(-1.0f, 1.0f) + 1.0f) * 0.5f).toDouble()).coerceIn(0.0, 1.0)

    private fun Float.toByteRange(): Int =
        (((coerceIn(-1.0f, 1.0f) + 1.0f) * 0.5f) * 255.0f).toInt().coerceIn(0, 255)

    private fun formatFloat(value: Double): String = String.format(Locale.US, "%.6f", value)

    private fun formatMs(nanos: Long): String = String.format(Locale.US, "%.3f", nanos / 1_000_000.0)

    private fun LongArray.elementCount(): Int = fold(1L) { acc, value -> acc * value }.toInt()

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

    companion object {
        private const val MANIFEST = "gvcrt_clean_manifest.json"
    }
}
