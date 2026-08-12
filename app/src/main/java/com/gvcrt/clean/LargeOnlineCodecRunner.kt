package com.gvcrt.clean

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.os.SystemClock
import com.mediatek.neuropilot_V.neuron.NeuronDelegate
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest
import java.util.Locale
import kotlin.math.log10
import kotlin.math.sqrt

/** Canonical Large online path: three input frames -> GVC stream -> independent reconstruction. */
class LargeOnlineCodecRunner(
    private val context: Context,
    private val emit: (String) -> Unit,
    private val showImages: ((File, File, Double) -> Unit)? = null,
) : AutoCloseable {
    private var prepared: PreparedRuntimes? = null

    fun run(imagePath: String?, warmupRuns: Int = 1, measuredRuns: Int = 1) {
        require(warmupRuns >= 0 && measuredRuns > 0)
        val image = ImageTensorLoader.load(context, imagePath)
        val displayInput = image.tensor.renamed("input_frame")
        val frame = NhwcTensorCodec.toF32Le(displayInput)
        val frames = List(FRAME_COUNT) { frame }
        val runtimes = prepare()
        emit(
            "large_online_main_start source=${image.source} frames=$FRAME_COUNT pattern=I,P,P " +
                "qp=${runtimes.stream.qp} layout=NHWC io=FP32 model_input_range=-1_1 " +
                "warmup=$warmupRuns measured=$measuredRuns",
        )
        repeat(warmupRuns) { execute(frames, runtimes, collectTimings = false) }
        val results = ArrayList<RunResult>(measuredRuns)
        repeat(measuredRuns) { results += execute(frames, runtimes, collectTimings = true) }
        emitSummary(results)

        val result = results.last()
        val outputRoot = context.getExternalFilesDir(null)!!.resolve("enterprise_tflite_codec/large/main")
        outputRoot.mkdirs()
        val inputFile = outputRoot.resolve("input_frame.nhwc.f32le").apply { writeBytes(frame) }
        val streamFile = outputRoot.resolve("encoded_i_p_p.gvc").apply { writeBytes(result.stream) }
        val boundaryRoot = outputRoot.resolve("boundaries").apply { mkdirs() }
        result.iBoundaries.forEach { (name, bytes) ->
            boundaryRoot.resolve("$name.nhwc.f32le").writeBytes(bytes)
            emit("large_online_main_boundary name=$name bytes=${bytes.size} sha256=${sha256(bytes)}")
        }
        result.decodedFrames.forEachIndexed { index, bytes ->
            outputRoot.resolve("decoded_frame_${index.toString().padStart(3, '0')}.nhwc.f32le").writeBytes(bytes)
        }
        val framePsnr = frames.indices.map { index ->
            val output = NhwcTensorCodec.fromF32Le("decoded_frame_$index", FRAME_SHAPE, result.decodedFrames[index])
            calculatePsnr(displayInput, output).also { psnr ->
                emit("large_online_main_quality frame=$index type=${if (index == 0) "I" else "P"} psnr_db=${format(psnr)}")
            }
        }
        val inputTensor = displayInput
        val outputTensor = NhwcTensorCodec.fromF32Le("decoded_frame", FRAME_SHAPE, result.decodedFrames.last())
        val psnr = framePsnr.last()
        val inputPng = writeTensorPng(inputTensor, "large_online_input.png")
        val outputPng = writeTensorPng(outputTensor, "large_online_reconstruction.png")
        showImages?.invoke(inputPng, outputPng, psnr)
        emit(
            "large_online_main_output stream=${streamFile.absolutePath} bytes=${result.stream.size} " +
                "sha256=${sha256(result.stream)} final_psnr_db=${format(psnr)} " +
                "input=${inputFile.absolutePath} input_sha256=${sha256(frame)} " +
                "reconstruction=${outputPng.absolutePath}",
        )
        emit("large_online_main_complete status=PASS all_models_exercised=true")
    }

    private fun execute(
        inputs: List<ByteArray>,
        runtimes: PreparedRuntimes,
        collectTimings: Boolean,
    ): RunResult {
        val timings = linkedMapOf<String, Long>()
        fun <T> timed(name: String, block: () -> T): T {
            if (!collectTimings) return block()
            val started = SystemClock.elapsedRealtimeNanos()
            return block().also { timings[name] = (timings[name] ?: 0L) + elapsedNs(started) }
        }

        val totalStarted = SystemClock.elapsedRealtimeNanos()
        val payloads = ArrayList<GvcFramePayload>(inputs.size)
        val encoderReconstructions = ArrayList<ByteArray>(inputs.size)
        val iBoundaries = linkedMapOf<String, ByteArray>()
        var encoderReferenceFrame: ByteArray? = null
        var encoderReferenceFeature: ByteArray? = null

        inputs.forEachIndexed { index, input ->
            if (index == 0) {
                val y = timed("encode_i_encoder") { runtimes.iEncoder.run(listOf(input)).single() }
                val entropy = timed("encode_i_entropy_rans") { runtimes.iEntropyEncoder.runCanonical(y) }
                require(entropy.size == 2) { "I entropy canonical outputs=${entropy.size}" }
                if (collectTimings) {
                    iBoundaries["android_i_y_pre_prior"] = y
                    iBoundaries["android_i_y_hat_encode"] = entropy[0]
                }
                val reconstruction = timed("encode_i_decoder") {
                    runtimes.iDecoder.run(listOf(entropy[0])).single()
                }
                payloads += GvcFramePayload(true, entropy[1])
                encoderReconstructions += reconstruction
                encoderReferenceFrame = reconstruction
            } else {
                val temporal = timed(if (index == 1) "encode_temporal_from_frame" else "encode_temporal_from_feature") {
                    if (index == 1) {
                        runtimes.temporalFromFrame.run(listOf(encoderReferenceFrame ?: error("missing encoder I reference")))
                    } else {
                        runtimes.temporalFromFeature.run(
                            listOf(encoderReferenceFeature ?: error("missing encoder P reference feature")),
                        )
                    }
                }
                require(temporal.size == 3) { "encoder temporal outputs=${temporal.size}" }
                val ctx = temporal[1]
                val ctxT = temporal[2]
                val y = timed("encode_p_encoder") { runtimes.pEncoder.run(listOf(input, ctx)).single() }
                val entropy = timed("encode_p_entropy_rans") {
                    runtimes.pEntropyEncoder.runCanonical(y, ctxT)
                }
                require(entropy.size == 2) { "P entropy canonical outputs=${entropy.size}" }
                val reconstruction = timed("encode_p_decoder") {
                    runtimes.pDecoder.run(listOf(entropy[0], ctx))
                }
                require(reconstruction.size == 2) { "encoder P decoder outputs=${reconstruction.size}" }
                payloads += GvcFramePayload(false, entropy[1])
                encoderReferenceFeature = reconstruction[0]
                encoderReconstructions += reconstruction[1]
            }
        }

        val stream = timed("stream_mux") { GvcStreamMuxer.muxSequence(runtimes.stream, payloads) }
        val parsed = timed("stream_demux") { GvcStreamMuxer.demuxSequence(stream) }
        require(parsed.frames.size == inputs.size) { "decoded frame count=${parsed.frames.size}" }
        require(parsed.stream.qp == runtimes.stream.qp) { "decoded QP=${parsed.stream.qp}" }

        val decodedFrames = ArrayList<ByteArray>(parsed.frames.size)
        var decoderReferenceFrame: ByteArray? = null
        var decoderReferenceFeature: ByteArray? = null
        parsed.frames.forEachIndexed { index, framePayload ->
            if (framePayload.isIFrame) {
                require(index == 0) { "I payload must be frame zero" }
                val yHat = timed("decode_i_entropy_rans") {
                    runtimes.iEntropyDecoder.runCanonical(framePayload.payload)
                }
                if (collectTimings) iBoundaries["android_i_y_hat_decode"] = yHat
                val reconstruction = timed("decode_i_decoder") {
                    runtimes.iDecoder.run(listOf(yHat)).single()
                }
                decoderReferenceFrame = reconstruction
                decodedFrames += reconstruction
            } else {
                val temporal = timed(if (index == 1) "decode_temporal_from_frame" else "decode_temporal_from_feature") {
                    if (index == 1) {
                        runtimes.temporalFromFrame.run(listOf(decoderReferenceFrame ?: error("missing decoder I reference")))
                    } else {
                        runtimes.temporalFromFeature.run(
                            listOf(decoderReferenceFeature ?: error("missing decoder P reference feature")),
                        )
                    }
                }
                require(temporal.size == 3) { "decoder temporal outputs=${temporal.size}" }
                val ctx = temporal[1]
                val ctxT = temporal[2]
                val yHat = timed("decode_p_entropy_rans") {
                    runtimes.pEntropyDecoder.runCanonical(framePayload.payload, ctxT)
                }
                val reconstruction = timed("decode_p_decoder") {
                    runtimes.pDecoder.run(listOf(yHat, ctx))
                }
                require(reconstruction.size == 2) { "decoded P outputs=${reconstruction.size}" }
                decoderReferenceFeature = reconstruction[0]
                decodedFrames += reconstruction[1]
            }
        }

        encoderReconstructions.zip(decodedFrames).forEachIndexed { index, pair ->
            require(pair.first.contentEquals(pair.second)) {
                "encoder/decoder reconstruction mismatch at frame=$index"
            }
        }
        if (collectTimings) timings["total"] = elapsedNs(totalStarted)
        return RunResult(stream, decodedFrames, timings, iBoundaries)
    }

    private fun prepare(): PreparedRuntimes {
        prepared?.let { return it }
        val root = findPackageRoot()
        val manifest = JSONObject(root.resolve("manifest.json").readText())
        val resolution = manifest.getJSONObject("resolution")
        val stream = StreamSpec(
            path = "",
            height = resolution.getInt("height"),
            width = resolution.getInt("width"),
            qp = manifest.getInt("qp"),
            ecPart = 0,
            useAdaI = 0,
        )
        require(stream.height == HEIGHT && stream.width == WIDTH && stream.qp == 0) {
            "Large online main requires ${HEIGHT}x$WIDTH QP=0"
        }
        val createStarted = SystemClock.elapsedRealtimeNanos()
        val official = linkedMapOf<String, OfficialNeuronRuntime>()
        var iEncode: IEntropyRansMergedRuntime? = null
        var pEncode: PEntropyRansMergedRuntime? = null
        var iDecode: IEntropyRansDecodeMergedRuntime? = null
        var pDecode: PEntropyRansDecodeMergedRuntime? = null
        try {
            REQUIRED_OFFICIAL_MODELS.forEach { name -> official[name] = createOfficial(root, name) }
            iEncode = IEntropyRansMergedRuntime.create(
                model(root, I_ENTROPY_ENCODER),
                context.cacheDir.resolve("enterprise_tflite/large/main/$I_ENTROPY_ENCODER"),
            )
            pEncode = PEntropyRansMergedRuntime.create(
                model(root, P_ENTROPY_ENCODER),
                context.cacheDir.resolve("enterprise_tflite/large/main/$P_ENTROPY_ENCODER"),
            )
            iDecode = IEntropyRansDecodeMergedRuntime.create(
                model(root, I_ENTROPY_DECODER),
                context.cacheDir.resolve("enterprise_tflite/large/main/$I_ENTROPY_DECODER"),
                true,
            )
            pDecode = PEntropyRansDecodeMergedRuntime.create(
                model(root, P_ENTROPY_DECODER),
                context.cacheDir.resolve("enterprise_tflite/large/main/$P_ENTROPY_DECODER"),
                true,
            )
            return PreparedRuntimes(
                stream = stream,
                temporalFromFrame = official.getValue("temporal_from_frame"),
                temporalFromFeature = official.getValue("temporal_from_feature"),
                iEncoder = official.getValue("i_encoder"),
                pEncoder = official.getValue("p_encoder"),
                iDecoder = official.getValue("i_decoder"),
                pDecoder = official.getValue("p_decoder"),
                iEntropyEncoder = iEncode,
                pEntropyEncoder = pEncode,
                iEntropyDecoder = iDecode,
                pEntropyDecoder = pDecode,
            ).also {
                prepared = it
                emit(
                    "large_online_main_prepare models=10 create_ms=${format(elapsedMs(createStarted))} " +
                        "backend=official_aar_neuron fast_models=10 decoder_models=scaled_variance_fp16 " +
                        "preference=FAST_SINGLE_ANSWER root=${root.absolutePath}",
                )
            }
        } catch (error: Throwable) {
            official.values.forEach(OfficialNeuronRuntime::close)
            iEncode?.close()
            pEncode?.close()
            iDecode?.close()
            pDecode?.close()
            throw error
        }
    }

    private fun createOfficial(root: File, name: String): OfficialNeuronRuntime {
        val file = model(root, "$name.tflite")
        val sha = sha256(file)
        val decoderModel = name == "i_decoder" || name == "p_decoder"
        return OfficialNeuronRuntime.create(
            tfliteFile = file,
            cacheDir = context.cacheDir.resolve("enterprise_tflite/large/main/$name"),
            allowFp16ForFp32 = true,
            acceleratorName = "mtk-neuron",
            compileOptions = "--relax-fp32",
            executionPreference = NeuronDelegate.Options.EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER,
            modelToken = "gvcrt_large_main_${name}_${sha.take(12)}_${if (decoderModel) "scaled_fast" else "fast"}",
        )
    }

    private fun model(root: File, name: String): File = root.resolve("models/$name").also {
        require(it.isFile) { "missing Large online model: ${it.absolutePath}" }
    }

    private fun findPackageRoot(): File {
        val internal = context.filesDir.resolve("enterprise_tflite/large")
        val external = context.getExternalFilesDir(null)?.resolve("enterprise_tflite/large")
        val required = REQUIRED_OFFICIAL_MODELS.map { "models/$it.tflite" } + REQUIRED_MERGED_MODELS.map { "models/$it" }
        return listOfNotNull(internal, external).firstOrNull { root ->
            root.resolve("manifest.json").isFile && required.all { root.resolve(it).isFile }
        } ?: error("no complete Large online package found")
    }

    private fun emitSummary(results: List<RunResult>) {
        val labels = results.flatMap { it.timings.keys }.distinct()
        labels.forEach { label ->
            val values = results.mapNotNull { it.timings[label] }
            if (values.isNotEmpty()) {
                emit(
                    "large_online_main_speed stage=$label samples=${values.size} " +
                        "mean_ms=${format(values.average() / 1_000_000.0)} " +
                        "p50_ms=${format(percentile(values, 0.50) / 1_000_000.0)} " +
                        "p90_ms=${format(percentile(values, 0.90) / 1_000_000.0)} includes_create=false",
                )
            }
        }
    }

    private fun calculatePsnr(input: TensorValue, reconstruction: TensorValue): Double {
        var sumSq = 0.0
        input.data.indices.forEach { index ->
            val diff = reconstruction.data[index].displayValue() - input.data[index].displayValue()
            sumSq += diff * diff
        }
        val rmse = sqrt(sumSq / input.data.size)
        return if (rmse == 0.0) Double.POSITIVE_INFINITY else 20.0 * log10(1.0 / rmse)
    }

    private fun writeTensorPng(tensor: TensorValue, fileName: String): File {
        val plane = HEIGHT * WIDTH
        val pixels = IntArray(plane)
        for (offset in pixels.indices) {
            pixels[offset] = Color.rgb(
                tensor.data[offset].byteValue(),
                tensor.data[plane + offset].byteValue(),
                tensor.data[2 * plane + offset].byteValue(),
            )
        }
        val bitmap = Bitmap.createBitmap(WIDTH, HEIGHT, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(pixels, 0, WIDTH, 0, 0, WIDTH, HEIGHT)
        val output = context.getExternalFilesDir(null)!!.resolve("outputs/$fileName")
        output.parentFile?.mkdirs()
        output.outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
        bitmap.recycle()
        return output
    }

    override fun close() {
        prepared?.close()
        prepared = null
    }

    private fun Float.displayValue(): Double = ((coerceIn(-1f, 1f) + 1f) * 0.5f).toDouble()
    private fun Float.byteValue(): Int = (((coerceIn(-1f, 1f) + 1f) * 0.5f) * 255f).toInt().coerceIn(0, 255)
    private fun elapsedNs(started: Long): Long = SystemClock.elapsedRealtimeNanos() - started
    private fun elapsedMs(started: Long): Double = elapsedNs(started) / 1_000_000.0
    private fun format(value: Double): String = String.format(Locale.US, "%.3f", value)
    private fun percentile(values: List<Long>, fraction: Double): Long {
        val sorted = values.sorted()
        return sorted[((sorted.size - 1) * fraction).toInt().coerceIn(sorted.indices)]
    }

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

    private data class RunResult(
        val stream: ByteArray,
        val decodedFrames: List<ByteArray>,
        val timings: Map<String, Long>,
        val iBoundaries: Map<String, ByteArray>,
    )

    private data class PreparedRuntimes(
        val stream: StreamSpec,
        val temporalFromFrame: OfficialNeuronRuntime,
        val temporalFromFeature: OfficialNeuronRuntime,
        val iEncoder: OfficialNeuronRuntime,
        val pEncoder: OfficialNeuronRuntime,
        val iDecoder: OfficialNeuronRuntime,
        val pDecoder: OfficialNeuronRuntime,
        val iEntropyEncoder: IEntropyRansMergedRuntime,
        val pEntropyEncoder: PEntropyRansMergedRuntime,
        val iEntropyDecoder: IEntropyRansDecodeMergedRuntime,
        val pEntropyDecoder: PEntropyRansDecodeMergedRuntime,
    ) : AutoCloseable {
        override fun close() {
            temporalFromFrame.close()
            temporalFromFeature.close()
            iEncoder.close()
            pEncoder.close()
            iDecoder.close()
            pDecoder.close()
            iEntropyEncoder.close()
            pEntropyEncoder.close()
            iEntropyDecoder.close()
            pEntropyDecoder.close()
        }
    }

    private companion object {
        const val HEIGHT = 256
        const val WIDTH = 512
        const val FRAME_COUNT = 3
        const val I_ENTROPY_ENCODER = "i_entropy_prior_merged_rans.tflite"
        const val P_ENTROPY_ENCODER = "p_entropy_prior_merged_rans.tflite"
        const val I_ENTROPY_DECODER = "i_entropy_decode_merged_rans.tflite"
        const val P_ENTROPY_DECODER = "p_entropy_decode_merged_rans.tflite"
        val FRAME_SHAPE = longArrayOf(1, 3, HEIGHT.toLong(), WIDTH.toLong())
        val REQUIRED_OFFICIAL_MODELS = listOf(
            "temporal_from_frame",
            "temporal_from_feature",
            "i_encoder",
            "p_encoder",
            "i_decoder",
            "p_decoder",
        )
        val REQUIRED_MERGED_MODELS = listOf(
            I_ENTROPY_ENCODER,
            P_ENTROPY_ENCODER,
            I_ENTROPY_DECODER,
            P_ENTROPY_DECODER,
        )
    }
}
