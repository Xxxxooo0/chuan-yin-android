package com.gvcrt.clean

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.SystemClock
import com.mediatek.neuropilot_V.neuron.NeuronDelegate
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.Locale
import kotlin.math.log10
import kotlin.math.roundToInt

/** Runs the current four-model Small QP9 package over a continuous PNG sequence. */
class SmallOnlineSequenceRunner(
    private val context: Context,
    private val emit: (String) -> Unit,
    private val showFrame: (Bitmap, Bitmap, Double, Int) -> Unit,
) {
    fun runSequence(sequenceDir: String, frameCount: Int) {
        require(frameCount >= 2) { "Small video sequence requires at least two frames" }
        val frameFiles = File(sequenceDir).listFiles()
            ?.filter { it.isFile && it.extension.equals("png", ignoreCase = true) }
            ?.sortedBy { it.name }
            ?.take(frameCount)
            .orEmpty()
        require(frameFiles.size == frameCount) {
            "Small video sequence requires $frameCount PNG frames, found ${frameFiles.size} in $sequenceDir"
        }

        val packageRoot = findPackageRoot()
        val runtimes = createRuntimes(packageRoot)
        try {
            emit(
                "small_online_video_start source=$sequenceDir frames=$frameCount fixed_q_index=9 " +
                    "layout=NHWC io=FP32 model_input_range=0_1 reference_reset_interval=64",
            )
            var referenceFrame = initialReferenceFrame()
            var referenceFeature: ByteArray? = null
            val frameTimes = mutableListOf<Double>()
            val psnrs = mutableListOf<Double>()

            frameFiles.forEachIndexed { index, frameFile ->
                val frame = loadFrame(frameFile)
                val resetReference = index == 0 || index % REFERENCE_RESET_INTERVAL == 1
                val started = SystemClock.elapsedRealtimeNanos()
                val temporal = if (resetReference) {
                    runtimes.temporalFromFrame.run(listOf(referenceFrame))
                } else {
                    runtimes.temporalFromFeature.run(listOf(referenceFeature!!))
                }
                require(temporal.size == 3) { "Small temporal output count=${temporal.size}" }
                val encoded = runtimes.encoder.run(listOf(frame.tensor, temporal[0]))
                require(encoded.size == 1) { "Small encoder output count=${encoded.size}" }
                val decoded = runtimes.decoder.run(listOf(encoded[0], temporal[0], temporal[2]))
                require(decoded.size == 2) { "Small decoder output count=${decoded.size}" }
                val elapsedMs = (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0

                referenceFeature = decoded[0]
                referenceFrame = decoded[1]
                val psnr = calculatePsnr(frame.tensor, referenceFrame)
                frameTimes += elapsedMs
                psnrs += psnr
                emit(
                    "small_online_video_progress frame=${index + 1}/$frameCount " +
                        "temporal=${if (resetReference) "from_frame" else "from_feature"} " +
                        "total_ms=${format(elapsedMs)} psnr_db=${format(psnr)}",
                )
                showFrame(frame.bitmap, reconstructionBitmap(referenceFrame), psnr, index + 1)
            }
            emit(
                "small_online_video_summary frames=$frameCount mean_frame_ms=${format(frameTimes.average())} " +
                    "fps=${format(1000.0 / frameTimes.average())} mean_psnr_db=${format(psnrs.average())} " +
                    "min_psnr_db=${format(psnrs.minOrNull()!!)}",
            )
            emit("small_online_video_complete status=PASS all_models_exercised=true")
        } finally {
            runtimes.close()
        }
    }

    private fun findPackageRoot(): File {
        val internal = context.filesDir.resolve("enterprise_tflite/small")
        val external = context.getExternalFilesDir(null)?.resolve("enterprise_tflite/small")
        val required = REQUIRED_MODELS.map { "models/$it.tflite" }
        return listOfNotNull(internal, external).firstOrNull { root ->
            root.resolve("manifest.json").isFile && required.all { root.resolve(it).isFile }
        } ?: error("no complete Small QP9 package found")
    }

    private fun createRuntimes(packageRoot: File): Runtimes {
        val manifest = JSONObject(packageRoot.resolve("manifest.json").readText())
        require(manifest.optInt("fixed_q_index", -1) == 9) {
            "Small online sequence requires fixed_q_index=9"
        }
        require(manifest.optString("layout") == "NHWC") { "Small online sequence requires NHWC" }
        require(manifest.optString("io_dtype") == "FP32") { "Small online sequence requires FP32 I/O" }
        val resolution = manifest.getJSONObject("resolution")
        require(resolution.getInt("height") == HEIGHT && resolution.getInt("width") == WIDTH) {
            "Small online sequence requires ${HEIGHT}x$WIDTH"
        }
        val declaredModels = (0 until manifest.getJSONArray("models").length()).map { index ->
            manifest.getJSONArray("models").getJSONObject(index).getString("name")
        }.toSet()
        require(declaredModels == REQUIRED_MODELS.toSet()) {
            "Small online sequence model set=$declaredModels expected=$REQUIRED_MODELS"
        }
        val created = linkedMapOf<String, OfficialNeuronRuntime>()
        val started = SystemClock.elapsedRealtimeNanos()
        try {
            REQUIRED_MODELS.forEach { name ->
                val model = packageRoot.resolve("models/$name.tflite")
                val sha = sha256(model)
                val runtime = OfficialNeuronRuntime.create(
                    tfliteFile = model,
                    cacheDir = context.cacheDir.resolve("enterprise_tflite/small/sequence/$name"),
                    allowFp16ForFp32 = false,
                    executionPreference = NeuronDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED,
                    modelToken = "gvcrt_small_qp9_${name}_${sha.take(12)}",
                )
                emit(
                    "small_online_video_create model=$name sha256=$sha inputs=${runtime.inputSizes.joinToString(",")} " +
                        "outputs=${runtime.outputSizes.joinToString(",")} options=${runtime.optionsSummary}",
                )
                created[name] = runtime
            }
            emit(
                "small_online_video_prepare models=${created.size} create_ms=${format(elapsedMs(started))} " +
                    "backend=official_aar_neuron allow_fp16=false root=${packageRoot.absolutePath}",
            )
            return Runtimes(
                temporalFromFrame = created.getValue("temporal_from_frame"),
                temporalFromFeature = created.getValue("temporal_from_feature"),
                encoder = created.getValue("encoder"),
                decoder = created.getValue("decoder"),
            )
        } catch (error: Throwable) {
            created.values.forEach(OfficialNeuronRuntime::close)
            throw error
        }
    }

    private fun initialReferenceFrame(): ByteArray =
        ByteBuffer.allocate(FRAME_BYTES).order(ByteOrder.LITTLE_ENDIAN).apply {
            asFloatBuffer().put(FloatArray(FRAME_FLOATS) { 0.5f })
        }.array()

    private fun loadFrame(file: File): Frame {
        val bitmap = BitmapFactory.decodeFile(file.absolutePath)
            ?: error("cannot decode PNG frame: ${file.absolutePath}")
        require(bitmap.width == WIDTH && bitmap.height == HEIGHT) {
            "Small video frame ${file.name} is ${bitmap.width}x${bitmap.height}; expected ${WIDTH}x$HEIGHT"
        }
        val pixels = IntArray(WIDTH * HEIGHT)
        bitmap.getPixels(pixels, 0, WIDTH, 0, 0, WIDTH, HEIGHT)
        val tensor = ByteBuffer.allocate(FRAME_BYTES).order(ByteOrder.LITTLE_ENDIAN).apply {
            pixels.forEach { pixel ->
                putFloat(((pixel shr 16) and 0xFF) / 255f)
                putFloat(((pixel shr 8) and 0xFF) / 255f)
                putFloat((pixel and 0xFF) / 255f)
            }
        }.array()
        return Frame(bitmap, tensor)
    }

    private fun reconstructionBitmap(tensor: ByteArray): Bitmap {
        require(tensor.size == FRAME_BYTES) { "Small reconstructed frame bytes=${tensor.size}" }
        val values = ByteBuffer.wrap(tensor).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        val pixels = IntArray(WIDTH * HEIGHT)
        pixels.indices.forEach { index ->
            val red = (values.get().coerceIn(0f, 1f) * 255f).roundToInt()
            val green = (values.get().coerceIn(0f, 1f) * 255f).roundToInt()
            val blue = (values.get().coerceIn(0f, 1f) * 255f).roundToInt()
            pixels[index] = (0xFF shl 24) or (red shl 16) or (green shl 8) or blue
        }
        return Bitmap.createBitmap(pixels, WIDTH, HEIGHT, Bitmap.Config.ARGB_8888)
    }

    private fun calculatePsnr(input: ByteArray, output: ByteArray): Double {
        require(input.size == FRAME_BYTES && output.size == FRAME_BYTES) { "Small PSNR frame shape mismatch" }
        val inputs = ByteBuffer.wrap(input).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        val outputs = ByteBuffer.wrap(output).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        var squaredError = 0.0
        repeat(FRAME_FLOATS) {
            val difference = inputs.get() - outputs.get()
            squaredError += difference * difference
        }
        val mse = squaredError / FRAME_FLOATS
        return if (mse == 0.0) Double.POSITIVE_INFINITY else 10.0 * log10(1.0 / mse)
    }

    private fun elapsedMs(started: Long): Double = (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { input ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(Locale.US, it) }
    }

    private fun format(value: Double): String = String.format(Locale.US, "%.3f", value)

    private data class Frame(val bitmap: Bitmap, val tensor: ByteArray)

    private data class Runtimes(
        val temporalFromFrame: OfficialNeuronRuntime,
        val temporalFromFeature: OfficialNeuronRuntime,
        val encoder: OfficialNeuronRuntime,
        val decoder: OfficialNeuronRuntime,
    ) : AutoCloseable {
        override fun close() {
            decoder.close()
            encoder.close()
            temporalFromFeature.close()
            temporalFromFrame.close()
        }
    }

    companion object {
        private const val HEIGHT = 256
        private const val WIDTH = 512
        private const val CHANNELS = 3
        private const val FRAME_FLOATS = HEIGHT * WIDTH * CHANNELS
        private const val FRAME_BYTES = FRAME_FLOATS * Float.SIZE_BYTES
        private const val REFERENCE_RESET_INTERVAL = 64
        private val REQUIRED_MODELS = listOf(
            "temporal_from_frame",
            "temporal_from_feature",
            "encoder",
            "decoder",
        )
    }
}
