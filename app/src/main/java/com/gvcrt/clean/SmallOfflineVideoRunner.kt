package com.gvcrt.clean

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.os.SystemClock
import com.mediatek.neuropilot_V.neuron.NeuronDelegate
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.log10
import kotlin.math.sqrt

/** Runs the Small temporal/encoder/decoder loop directly on frames decoded from a video. */
internal class SmallOfflineVideoRunner(
    private val context: Context,
    private val emit: (String) -> Unit,
    private val showImages: ((File, File, Double) -> Unit)? = null,
) : AutoCloseable {
    private val cancelled = AtomicBoolean(false)

    fun cancel() {
        cancelled.set(true)
    }

    fun run(
        inputUri: Uri,
        maxDurationSeconds: Int = 60,
        h264Bitrate: Int = 8_000_000,
    ) {
        require(maxDurationSeconds > 0) { "maxDurationSeconds must be positive" }
        require(h264Bitrate > 0) { "h264Bitrate must be positive" }
        cancelled.set(false)

        val packageRoot = findPackageRoot()
        val outputRoot = context.getExternalFilesDir(null)!!
            .resolve("enterprise_tflite_codec/small/video_demo/${timestamp()}")
            .apply { mkdirs() }
        val mp4File = outputRoot.resolve("reconstructed.mp4")
        val reportFile = outputRoot.resolve("run_report.json")
        val runtimeCreateStarted = SystemClock.elapsedRealtimeNanos()
        val runtimes = prepareRuntimes(packageRoot)
        val runtimeCreateNs = elapsedNs(runtimeCreateStarted)
        val pipelineStarted = SystemClock.elapsedRealtimeNanos()
        var status = "PASS"

        emit(
            "small_offline_video_start uri=$inputUri max_seconds=$maxDurationSeconds " +
                "model=${WIDTH}x$HEIGHT h264_bitrate=$h264Bitrate audio=false " +
                "mode=single_pass_reference_feature backend=official_aar_neuron " +
                "runtime_profile=mlvc_relax_fp32",
        )
        try {
            var referenceFeature = ByteArray(REFERENCE_FEATURE_BYTES)
            val temporalTimes = mutableListOf<Long>()
            val encoderTimes = mutableListOf<Long>()
            val decoderTimes = mutableListOf<Long>()
            val modelTotalTimes = mutableListOf<Long>()
            val psnrs = mutableListOf<Double>()
            var sourceDecodeNs = 0L
            var mp4WriteNs = 0L
            var frameCount = 0
            var firstPtsUs: Long? = null
            var lastInput: ByteArray? = null
            var lastReconstruction: ByteArray? = null
            lateinit var videoInfo: OfflineVideoInfo

            OfflineVideoFrameReader(
                context = context,
                uri = inputUri,
                targetWidth = WIDTH,
                targetHeight = HEIGHT,
                maxDurationUs = maxDurationSeconds * 1_000_000L,
                tensorRange = VideoTensorRange.ZERO_TO_ONE,
            ).use { reader ->
                videoInfo = reader.info
                emit(
                    "small_offline_video_source width=${videoInfo.width} height=${videoInfo.height} " +
                        "rotation=${videoInfo.rotationDegrees} fps=${videoInfo.frameRate} " +
                        "duration_ms=${format(videoInfo.durationUs / 1000.0)} mime=${videoInfo.mime}",
                )
                ReconstructionMp4Writer(
                    output = mp4File,
                    width = WIDTH,
                    height = HEIGHT,
                    frameRate = videoInfo.frameRate,
                    bitrate = h264Bitrate,
                    tensorRange = VideoTensorRange.ZERO_TO_ONE,
                ).use { writer ->
                    while (true) {
                        checkNotCancelled()
                        val decodeStarted = SystemClock.elapsedRealtimeNanos()
                        val sourceFrame = reader.next()
                        sourceDecodeNs += elapsedNs(decodeStarted)
                        if (sourceFrame == null) break
                        if (firstPtsUs == null) firstPtsUs = sourceFrame.presentationTimeUs

                        val modelStarted = SystemClock.elapsedRealtimeNanos()
                        val temporalStarted = SystemClock.elapsedRealtimeNanos()
                        val temporalOutputs = runtimes.temporal.run(listOf(referenceFeature))
                        temporalTimes += elapsedNs(temporalStarted)
                        require(temporalOutputs.size == 3) {
                            "small temporal output count=${temporalOutputs.size}"
                        }
                        val ctx = temporalOutputs[0]
                        val memory = temporalOutputs[2]

                        val encoderStarted = SystemClock.elapsedRealtimeNanos()
                        val encoderOutputs = runtimes.encoder.run(listOf(sourceFrame.tensor, ctx))
                        encoderTimes += elapsedNs(encoderStarted)
                        require(encoderOutputs.size == 1) {
                            "small encoder output count=${encoderOutputs.size}"
                        }

                        val decoderStarted = SystemClock.elapsedRealtimeNanos()
                        val decoderOutputs = runtimes.decoder.run(listOf(encoderOutputs[0], ctx, memory))
                        decoderTimes += elapsedNs(decoderStarted)
                        require(decoderOutputs.size == 2) {
                            "small decoder output count=${decoderOutputs.size}"
                        }
                        modelTotalTimes += elapsedNs(modelStarted)
                        referenceFeature = decoderOutputs[0]
                        val reconstruction = decoderOutputs[1]
                        val psnr = calculatePsnr(sourceFrame.tensor, reconstruction)
                        psnrs += psnr

                        val writeStarted = SystemClock.elapsedRealtimeNanos()
                        writer.writeFrame(
                            reconstruction,
                            sourceFrame.presentationTimeUs - (firstPtsUs ?: sourceFrame.presentationTimeUs),
                        )
                        mp4WriteNs += elapsedNs(writeStarted)
                        frameCount++
                        lastInput = sourceFrame.tensor
                        lastReconstruction = reconstruction
                        if (frameCount == 1 || frameCount % PROGRESS_INTERVAL == 0) {
                            emit(
                                "small_offline_video_progress frame=$frameCount " +
                                    "pts_ms=${format((sourceFrame.presentationTimeUs - (firstPtsUs ?: 0L)) / 1000.0)} " +
                                    "psnr_db=${format(psnr)} model_ms=${format(modelTotalTimes.last() / 1_000_000.0)}",
                            )
                        }
                    }
                }
            }

            require(frameCount > 0) { "video decoder produced no frames" }
            val finalInput = requireNotNull(lastInput)
            val finalReconstruction = requireNotNull(lastReconstruction)
            val inputPng = writePng(finalInput, outputRoot.resolve("input_last.png"))
            val reconstructionPng = writePng(finalReconstruction, outputRoot.resolve("reconstructed_last.png"))
            showImages?.invoke(inputPng, reconstructionPng, psnrs.last())
            val pipelineWallNs = elapsedNs(pipelineStarted)
            val modelTotalNs = modelTotalTimes.sum()

            emitTiming("temporal_reference", temporalTimes)
            emitTiming("encoder", encoderTimes)
            emitTiming("decoder", decoderTimes)
            emitTiming("model_total", modelTotalTimes)
            emit(
                "small_offline_video_speed stage=model_summary frames=$frameCount " +
                    "mean_frame_ms=${format(modelTotalNs / frameCount / 1_000_000.0)} " +
                    "fps=${format(frameCount * 1_000_000_000.0 / modelTotalNs)} " +
                    "includes_source_decode=false includes_mp4=false includes_create=false",
            )
            emit(
                "small_offline_video_speed stage=io source_decode_ms=${format(sourceDecodeNs / 1_000_000.0)} " +
                    "mp4_write_ms=${format(mp4WriteNs / 1_000_000.0)} " +
                    "runtime_create_ms=${format(runtimeCreateNs / 1_000_000.0)} " +
                    "pipeline_wall_ms=${format(pipelineWallNs / 1_000_000.0)}",
            )

            val report = JSONObject().apply {
                put("input_uri", inputUri.toString())
                put("frame_count", frameCount)
                put("model_width", WIDTH)
                put("model_height", HEIGHT)
                put("source_width", videoInfo.width)
                put("source_height", videoInfo.height)
                put("source_frame_rate", videoInfo.frameRate)
                put("model_total_ms", modelTotalNs / 1_000_000.0)
                put("temporal_reference_mean_ms", temporalTimes.average() / 1_000_000.0)
                put("encoder_mean_ms", encoderTimes.average() / 1_000_000.0)
                put("decoder_mean_ms", decoderTimes.average() / 1_000_000.0)
                put("mean_model_frame_ms", modelTotalNs / frameCount / 1_000_000.0)
                put("model_fps", frameCount * 1_000_000_000.0 / modelTotalNs)
                put("source_decode_ms", sourceDecodeNs / 1_000_000.0)
                put("mp4_write_ms", mp4WriteNs / 1_000_000.0)
                put("runtime_create_ms", runtimeCreateNs / 1_000_000.0)
                put("pipeline_wall_ms_excluding_model_create", pipelineWallNs / 1_000_000.0)
                put("mean_psnr_db", psnrs.average())
                put("min_psnr_db", psnrs.minOrNull())
                put("final_psnr_db", psnrs.last())
                put("psnr_domain", "model_tensor_0_1_before_h264")
                put("reference_initialization", "zero_feature")
                put("reconstructed_mp4_path", mp4File.absolutePath)
                put("reconstructed_mp4_sha256", sha256(mp4File))
            }
            reportFile.writeText(report.toString(2))
            emit(
                "small_offline_video_quality frames=$frameCount mean_psnr_db=${format(psnrs.average())} " +
                    "min_psnr_db=${format(psnrs.minOrNull()!!)} final_psnr_db=${format(psnrs.last())} " +
                    "domain=model_tensor_0_1_before_h264",
            )
            emit(
                "small_offline_video_output mp4=${mp4File.absolutePath} mp4_bytes=${mp4File.length()} " +
                    "mp4_sha256=${sha256(mp4File)} input_png=${inputPng.absolutePath} " +
                    "reconstruction_png=${reconstructionPng.absolutePath} report=${reportFile.absolutePath}",
            )
        } catch (_: VideoCancelledException) {
            status = "CANCELLED"
        } catch (error: Throwable) {
            status = "FAILED"
            throw error
        } finally {
            runtimes.close()
            emit("small_offline_video_complete status=$status output=${outputRoot.absolutePath}")
        }
    }

    private fun prepareRuntimes(packageRoot: File): SmallRuntimes {
        fun create(modelName: String): OfficialNeuronRuntime {
            val modelFile = packageRoot.resolve("models/$modelName.tflite")
            require(modelFile.isFile) { "missing model: ${modelFile.absolutePath}" }
            val sha = sha256(modelFile)
            val started = SystemClock.elapsedRealtimeNanos()
            emit(
                "small_offline_video_create_start model=$modelName bytes=${modelFile.length()} sha256=$sha",
            )
            return OfficialNeuronRuntime.create(
                tfliteFile = modelFile,
                cacheDir = context.cacheDir.resolve("enterprise_tflite/small/mlvc_relax_fp32/$modelName"),
                allowFp16ForFp32 = true,
                acceleratorName = "mtk-neuron",
                compileOptions = "--relax-fp32",
                executionPreference = NeuronDelegate.Options.EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER,
                modelToken = "gvcrt_small_${modelName}_${sha.take(12)}",
            ).also { runtime ->
                emit(
                    "small_offline_video_create_ok model=$modelName " +
                        "create_ms=${format(elapsedNs(started) / 1_000_000.0)} " +
                        "inputs=${runtime.inputSizes.joinToString(":")} " +
                        "outputs=${runtime.outputSizes.joinToString(":")} options=${runtime.optionsSummary}",
                )
            }
        }

        val temporal = create("temporal_reference")
        val encoder = try {
            create("encoder")
        } catch (error: Throwable) {
            temporal.close()
            throw error
        }
        val decoder = try {
            create("decoder")
        } catch (error: Throwable) {
            encoder.close()
            temporal.close()
            throw error
        }
        return SmallRuntimes(temporal, encoder, decoder).also { runtimes ->
            try {
                require(temporal.inputSizes.contentEquals(longArrayOf(REFERENCE_FEATURE_BYTES.toLong())))
                require(
                    temporal.outputSizes.contentEquals(
                        longArrayOf(CTX_BYTES.toLong(), CTX_BYTES.toLong(), MEMORY_BYTES.toLong()),
                    ),
                )
                require(encoder.inputSizes.contentEquals(longArrayOf(FRAME_BYTES.toLong(), CTX_BYTES.toLong())))
                require(encoder.outputSizes.contentEquals(longArrayOf(LATENT_BYTES.toLong())))
                require(
                    decoder.inputSizes.contentEquals(
                        longArrayOf(LATENT_BYTES.toLong(), CTX_BYTES.toLong(), MEMORY_BYTES.toLong()),
                    ),
                )
                require(
                    decoder.outputSizes.contentEquals(
                        longArrayOf(REFERENCE_FEATURE_BYTES.toLong(), FRAME_BYTES.toLong()),
                    ),
                )
            } catch (error: Throwable) {
                runtimes.close()
                throw error
            }
        }
    }

    private fun findPackageRoot(): File {
        val internalRoot = context.filesDir.resolve("enterprise_tflite/small")
        val externalRoot = context.getExternalFilesDir(null)?.resolve("enterprise_tflite/small")
        return listOfNotNull(internalRoot, externalRoot).firstOrNull { root ->
            root.resolve("input_manifest.json").isFile &&
                listOf("temporal_reference", "encoder", "decoder").all { root.resolve("models/$it.tflite").isFile }
        } ?: error("Small TFLite package is not installed under ${internalRoot.absolutePath}")
    }

    private fun emitTiming(stage: String, times: List<Long>) {
        val sorted = times.sorted()
        fun percentile(fraction: Double): Double {
            val index = ((sorted.size - 1) * fraction).toInt().coerceIn(sorted.indices)
            return sorted[index] / 1_000_000.0
        }
        emit(
            "small_offline_video_speed stage=$stage samples=${times.size} " +
                "mean_ms=${format(times.average() / 1_000_000.0)} " +
                "p50_ms=${format(percentile(0.50))} p90_ms=${format(percentile(0.90))}",
        )
    }

    private fun calculatePsnr(input: ByteArray, reconstruction: ByteArray): Double {
        require(input.size == reconstruction.size && input.size % 4 == 0) { "PSNR tensor byte count mismatch" }
        val inputFloats = ByteBuffer.wrap(input).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        val reconstructionFloats = ByteBuffer.wrap(reconstruction).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        var sumSq = 0.0
        repeat(input.size / 4) { index ->
            val inputValue = inputFloats[index].coerceIn(0f, 1f).toDouble()
            val reconstructionValue = reconstructionFloats[index].coerceIn(0f, 1f).toDouble()
            val diff = reconstructionValue - inputValue
            sumSq += diff * diff
        }
        val rmse = sqrt(sumSq / (input.size / 4))
        return if (rmse == 0.0) Double.POSITIVE_INFINITY else 20.0 * log10(1.0 / rmse)
    }

    private fun writePng(tensor: ByteArray, output: File): File {
        output.parentFile?.mkdirs()
        val bitmap = VideoTensorCodec.toBitmap(tensor, WIDTH, HEIGHT, VideoTensorRange.ZERO_TO_ONE)
        output.outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
        bitmap.recycle()
        return output
    }

    private fun checkNotCancelled() {
        if (cancelled.get()) throw VideoCancelledException()
    }

    private fun elapsedNs(started: Long): Long = SystemClock.elapsedRealtimeNanos() - started

    private fun format(value: Double): String = "%.3f".format(Locale.US, value)

    private fun timestamp(): String = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())

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

    override fun close() {
        cancel()
    }

    private data class SmallRuntimes(
        val temporal: OfficialNeuronRuntime,
        val encoder: OfficialNeuronRuntime,
        val decoder: OfficialNeuronRuntime,
    ) : AutoCloseable {
        override fun close() {
            decoder.close()
            encoder.close()
            temporal.close()
        }
    }

    private class VideoCancelledException : RuntimeException()

    private companion object {
        const val WIDTH = 512
        const val HEIGHT = 256
        const val FRAME_BYTES = WIDTH * HEIGHT * 3 * 4
        const val REFERENCE_FEATURE_BYTES = 32 * 64 * 96 * 4
        const val CTX_BYTES = 32 * 64 * 192 * 4
        const val MEMORY_BYTES = 32 * 64 * 48 * 4
        const val LATENT_BYTES = 16 * 32 * 48 * 4
        const val PROGRESS_INTERVAL = 10
    }
}
