package com.gvcrt.clean

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.net.Uri
import android.os.SystemClock
import com.mediatek.neuropilot_V.neuron.NeuronDelegate
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.CancellationException
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.log10
import kotlin.math.sqrt

/** Canonical Large online path: input frames -> GVC stream -> independent reconstruction. */
class LargeOnlineCodecRunner(
    private val context: Context,
    private val emit: (String) -> Unit,
    private val showImages: ((File, File, Double) -> Unit)? = null,
    private val showVideoFrames: ((Bitmap, Bitmap, Double, Int) -> Unit)? = null,
) : AutoCloseable {
    private var prepared: PreparedRuntimes? = null
    private val videoCancelled = AtomicBoolean(false)

    fun run(imagePath: String?, warmupRuns: Int = 1, measuredRuns: Int = 1, qp: Int = DEFAULT_QP) {
        val image = ImageTensorLoader.load(context, imagePath)
        val displayInput = image.tensor.renamed("input_frame")
        runFrames(
            frames = List(DEFAULT_FRAME_COUNT) { NhwcTensorCodec.toF32Le(displayInput) },
            source = image.source,
            outputName = "three_frame",
            warmupRuns = warmupRuns,
            measuredRuns = measuredRuns,
            qp = qp,
        )
    }

    fun runSequence(
        sequenceDir: String,
        frameCount: Int,
        warmupRuns: Int = 0,
        measuredRuns: Int = 1,
        dumpPEntropyBoundaries: Boolean = false,
        qp: Int = DEFAULT_QP,
    ) {
        require(frameCount > 0) { "Large online video frame count must be positive" }
        val directory = File(sequenceDir)
        require(directory.isDirectory) { "Large online video directory does not exist: ${directory.absolutePath}" }
        val frameFiles = directory.listFiles()
            ?.filter { it.isFile && it.extension.equals("png", ignoreCase = true) }
            ?.sortedBy { it.name }
            ?.take(frameCount)
            .orEmpty()
        require(frameFiles.size == frameCount) {
            "Large online video requires $frameCount PNG frames, found ${frameFiles.size} in ${directory.absolutePath}"
        }
        emit("large_online_video_load_start directory=${directory.absolutePath} frames=$frameCount")
        runSequenceFiles(
            frameFiles = frameFiles,
            source = directory.absolutePath,
            outputName = "sequence_${frameCount}_frames",
            warmupRuns = warmupRuns,
            measuredRuns = measuredRuns,
            dumpPEntropyBoundaries = dumpPEntropyBoundaries,
            qp = qp,
        )
    }

    fun cancelVideo() {
        videoCancelled.set(true)
    }

    /** Runs a bounded-memory two-pass video test: GVC encode/mux, then independent demux/decode/MP4. */
    fun runOfflineVideo(
        inputUri: Uri,
        maxDurationSeconds: Int = 60,
        h264Bitrate: Int = 8_000_000,
        qp: Int = DEFAULT_QP,
    ) {
        require(maxDurationSeconds > 0)
        videoCancelled.set(false)
        val runtimes = prepare(qp)
        val pipelineStarted = SystemClock.elapsedRealtimeNanos()
        val outputRoot = context.getExternalFilesDir(null)!!
            .resolve("enterprise_tflite_codec/large/video_demo/${timestamp()}")
            .apply { mkdirs() }
        val streamFile = outputRoot.resolve("encoded_video.gvc")
        val mp4File = outputRoot.resolve("reconstructed.mp4")
        val maxDurationUs = maxDurationSeconds * 1_000_000L
        emit(
            "large_offline_video_start uri=$inputUri max_seconds=$maxDurationSeconds qp=${runtimes.stream.qp} " +
                "model=${WIDTH}x$HEIGHT h264_bitrate=$h264Bitrate audio=false mode=two_pass_streaming",
        )
        try {
            val encodeResult: VideoEncodeResult
            val videoInfo: OfflineVideoInfo
            OfflineVideoFrameReader(context, inputUri, WIDTH, HEIGHT, maxDurationUs).use { reader ->
                videoInfo = reader.info
                emit(
                    "large_offline_video_source width=${videoInfo.width} height=${videoInfo.height} " +
                        "rotation=${videoInfo.rotationDegrees} fps=${videoInfo.frameRate} " +
                        "duration_ms=${format(videoInfo.durationUs / 1000.0)} mime=${videoInfo.mime}",
                )
                encodeResult = encodeVideo(reader, runtimes)
            }
            require(encodeResult.payloads.isNotEmpty()) { "video decoder produced no frames" }
            val muxStarted = SystemClock.elapsedRealtimeNanos()
            val stream = GvcStreamMuxer.muxSequence(runtimes.stream, encodeResult.payloads)
            val muxNs = elapsedNs(muxStarted)
            streamFile.writeBytes(stream)
            val parsed = GvcStreamMuxer.demuxSequence(stream)
            require(parsed.frames.size == encodeResult.payloads.size)

            val decodeResult: VideoDecodeResult
            OfflineVideoFrameReader(context, inputUri, WIDTH, HEIGHT, maxDurationUs).use { reader ->
                ReconstructionMp4Writer(
                    output = mp4File,
                    width = WIDTH,
                    height = HEIGHT,
                    frameRate = videoInfo.frameRate,
                    bitrate = h264Bitrate,
                ).use { writer ->
                    decodeResult = decodeVideo(
                        reader = reader,
                        frames = parsed.frames,
                        expectedReconstructionHashes = encodeResult.reconstructionHashes,
                        expectedPresentationTimesUs = encodeResult.presentationTimesUs,
                        runtimes = runtimes,
                        writer = writer,
                    )
                }
            }

            val encodeModelNs = encodeResult.timings.values.sum()
            val decodeModelNs = decodeResult.timings.values.sum()
            emitVideoTimings("encode", encodeResult.timings, encodeResult.payloads.size)
            emitVideoTimings("decode", decodeResult.timings, parsed.frames.size)
            emit(
                "large_offline_video_speed phase=gvc_model_total frames=${parsed.frames.size} " +
                    "encode_ms=${format(encodeModelNs / 1_000_000.0)} " +
                    "decode_ms=${format(decodeModelNs / 1_000_000.0)} " +
                    "mean_frame_ms=${format((encodeModelNs + decodeModelNs) / parsed.frames.size / 1_000_000.0)} " +
                    "fps=${format(parsed.frames.size * 1_000_000_000.0 / (encodeModelNs + decodeModelNs))} " +
                    "includes_source_decode=false includes_mp4=false includes_create=false",
            )

            val inputPng = writeNhwcPng(decodeResult.lastInput, outputRoot.resolve("input_last.png"))
            val reconPng = writeNhwcPng(decodeResult.lastReconstruction, outputRoot.resolve("reconstructed_last.png"))
            showImages?.invoke(inputPng, reconPng, decodeResult.psnrs.last())
            val pipelineWallNs = elapsedNs(pipelineStarted)
            val report = JSONObject().apply {
                put("input_uri", inputUri.toString())
                put("qp", runtimes.stream.qp)
                put("frame_count", parsed.frames.size)
                put("source_width", videoInfo.width)
                put("source_height", videoInfo.height)
                put("source_frame_rate", videoInfo.frameRate)
                put("source_decode_encode_pass_ms", encodeResult.sourceDecodeNs / 1_000_000.0)
                put("source_decode_compare_pass_ms", decodeResult.sourceDecodeNs / 1_000_000.0)
                put("gvc_encode_model_ms", encodeModelNs / 1_000_000.0)
                put("gvc_decode_model_ms", decodeModelNs / 1_000_000.0)
                put("stream_mux_ms", muxNs / 1_000_000.0)
                put("mp4_write_ms", decodeResult.mp4WriteNs / 1_000_000.0)
                put("pipeline_wall_ms_excluding_model_create", pipelineWallNs / 1_000_000.0)
                put("mean_psnr_db", decodeResult.psnrs.average())
                put("min_psnr_db", decodeResult.psnrs.minOrNull())
                put("final_psnr_db", decodeResult.psnrs.last())
                put("gvc_bytes", stream.size)
                put("gvc_sha256", sha256(stream))
                put("gvc_path", streamFile.absolutePath)
                put("reconstructed_mp4_path", mp4File.absolutePath)
            }
            outputRoot.resolve("run_report.json").writeText(report.toString(2))
            emit(
                "large_offline_video_quality frames=${parsed.frames.size} " +
                    "mean_psnr_db=${format(decodeResult.psnrs.average())} " +
                    "min_psnr_db=${format(decodeResult.psnrs.minOrNull()!!)} " +
                    "final_psnr_db=${format(decodeResult.psnrs.last())} domain=model_tensor_before_h264",
            )
            emit(
                "large_offline_video_output gvc=${streamFile.absolutePath} gvc_bytes=${stream.size} " +
                    "gvc_sha256=${sha256(stream)} mp4=${mp4File.absolutePath} " +
                    "pipeline_wall_ms=${format(pipelineWallNs / 1_000_000.0)} " +
                    "report=${outputRoot.resolve("run_report.json").absolutePath}",
            )
            emit("large_offline_video_complete status=PASS")
        } catch (_: CancellationException) {
            emit("large_offline_video_complete status=CANCELLED output=${outputRoot.absolutePath}")
        }
    }

    private fun runSequenceFiles(
        frameFiles: List<File>,
        source: String,
        outputName: String,
        warmupRuns: Int,
        measuredRuns: Int,
        dumpPEntropyBoundaries: Boolean,
        qp: Int,
    ) {
        require(frameFiles.isNotEmpty())
        require(warmupRuns >= 0 && measuredRuns > 0)
        val runtimes = prepare(qp)
        fun loadFrame(index: Int): ByteArray {
            val tensor = ImageTensorLoader.load(context, frameFiles[index].absolutePath)
                .tensor
                .renamed("input_frame_$index")
            return NhwcTensorCodec.toF32Le(tensor)
        }
        emit(
            "large_online_main_start source=$source frames=${frameFiles.size} pattern=I,Px${frameFiles.size - 1} " +
                "qp=${runtimes.stream.qp} layout=NHWC io=FP32 model_input_range=-1_1 " +
                "warmup=$warmupRuns measured=$measuredRuns sequence_mode=streaming " +
                "reference_reset_interval=$REFERENCE_RESET_INTERVAL",
        )
        repeat(warmupRuns) {
            execute(
                inputCount = frameFiles.size,
                inputAt = ::loadFrame,
                runtimes = runtimes,
                collectTimings = false,
                retainDecodedFrames = false,
            )
        }
        val results = ArrayList<SequenceRunResult>(measuredRuns)
        repeat(measuredRuns) {
            val psnrs = ArrayList<Double>(frameFiles.size)
            var lastDecoded: ByteArray? = null
            val result = execute(
                inputCount = frameFiles.size,
                inputAt = ::loadFrame,
                runtimes = runtimes,
                collectTimings = true,
                retainDecodedFrames = false,
                decodedConsumer = { index, decoded ->
                    psnrs += calculatePsnr(loadFrame(index), decoded)
                    if (index == frameFiles.lastIndex) lastDecoded = decoded
                },
                pEntropyBoundaryFrames = if (dumpPEntropyBoundaries) setOf(1, 2) else emptySet(),
            )
            results += SequenceRunResult(result, psnrs, lastDecoded ?: error("missing final decoded frame"))
        }
        emitSummary(results.map(SequenceRunResult::run), frameFiles.size)

        val result = results.last()
        val outputRoot = context.getExternalFilesDir(null)!!
            .resolve("enterprise_tflite_codec/large/main/$outputName")
        outputRoot.mkdirs()
        val firstInput = loadFrame(0)
        val lastInput = loadFrame(frameFiles.lastIndex)
        val inputFile = outputRoot.resolve("input_frame_000.nhwc.f32le").apply { writeBytes(firstInput) }
        val streamFile = outputRoot.resolve("encoded_${frameFiles.size}_frames.gvc").apply {
            writeBytes(result.run.stream)
        }
        outputRoot.resolve("decoded_frame_${frameFiles.lastIndex.toString().padStart(3, '0')}.nhwc.f32le")
            .writeBytes(result.lastDecoded)
        val boundaryRoot = outputRoot.resolve("boundaries").apply { mkdirs() }
        result.run.iBoundaries.forEach { (name, bytes) ->
            boundaryRoot.resolve("$name.nhwc.f32le").writeBytes(bytes)
            emit("large_online_main_boundary name=$name bytes=${bytes.size} sha256=${sha256(bytes)}")
        }
        result.psnrs.forEachIndexed { index, psnr ->
            emit(
                "large_online_main_quality frame=${index + 1} type=${if (index == 0) "I" else "P"} " +
                    "psnr_db=${format(psnr)}",
            )
        }
        val inputTensor = NhwcTensorCodec.fromF32Le("input_frame", FRAME_SHAPE, lastInput)
        val outputTensor = NhwcTensorCodec.fromF32Le("decoded_frame", FRAME_SHAPE, result.lastDecoded)
        val finalPsnr = result.psnrs.last()
        val inputPng = writeTensorPng(inputTensor, "large_online_input.png")
        val outputPng = writeTensorPng(outputTensor, "large_online_reconstruction.png")
        showImages?.invoke(inputPng, outputPng, finalPsnr)
        emit(
            "large_online_main_output stream=${streamFile.absolutePath} bytes=${result.run.stream.size} " +
                "sha256=${sha256(result.run.stream)} final_psnr_db=${format(finalPsnr)} " +
                "mean_psnr_db=${format(result.psnrs.average())} min_psnr_db=${format(result.psnrs.minOrNull()!!)} " +
                "input=${inputFile.absolutePath} input_sha256=${sha256(firstInput)} " +
                "reconstruction=${outputPng.absolutePath} dump_mode=streaming_final_frame_only",
        )
        emit("large_online_main_complete status=PASS all_models_exercised=true")
    }

    private fun runFrames(
        frames: List<ByteArray>,
        source: String,
        outputName: String,
        warmupRuns: Int,
        measuredRuns: Int,
        qp: Int,
    ) {
        require(frames.isNotEmpty())
        require(warmupRuns >= 0 && measuredRuns > 0)
        val runtimes = prepare(qp)
        emit(
            "large_online_main_start source=$source frames=${frames.size} pattern=I,Px${frames.size - 1} " +
                "qp=${runtimes.stream.qp} layout=NHWC io=FP32 model_input_range=-1_1 " +
                "warmup=$warmupRuns measured=$measuredRuns",
        )
        repeat(warmupRuns) {
            execute(frames.size, frames::get, runtimes, collectTimings = false)
        }
        val results = ArrayList<RunResult>(measuredRuns)
        repeat(measuredRuns) {
            results += execute(frames.size, frames::get, runtimes, collectTimings = true)
        }
        emitSummary(results, frames.size)

        val result = results.last()
        val outputRoot = context.getExternalFilesDir(null)!!
            .resolve("enterprise_tflite_codec/large/main/$outputName")
        outputRoot.mkdirs()
        val inputFile = outputRoot.resolve("input_frame_000.nhwc.f32le").apply { writeBytes(frames.first()) }
        val streamFile = outputRoot.resolve("encoded_${frames.size}_frames.gvc").apply { writeBytes(result.stream) }
        val boundaryRoot = outputRoot.resolve("boundaries").apply { mkdirs() }
        result.iBoundaries.forEach { (name, bytes) ->
            boundaryRoot.resolve("$name.nhwc.f32le").writeBytes(bytes)
            emit("large_online_main_boundary name=$name bytes=${bytes.size} sha256=${sha256(bytes)}")
        }
        result.decodedFrames.forEachIndexed { index, bytes ->
            outputRoot.resolve("decoded_frame_${index.toString().padStart(3, '0')}.nhwc.f32le").writeBytes(bytes)
        }
        val framePsnr = frames.indices.map { index ->
            calculatePsnr(frames[index], result.decodedFrames[index]).also { psnr ->
                emit(
                    "large_online_main_quality frame=${index + 1} type=${if (index == 0) "I" else "P"} " +
                        "psnr_db=${format(psnr)}",
                )
            }
        }
        val inputTensor = NhwcTensorCodec.fromF32Le("input_frame", FRAME_SHAPE, frames.last())
        val outputTensor = NhwcTensorCodec.fromF32Le("decoded_frame", FRAME_SHAPE, result.decodedFrames.last())
        val psnr = framePsnr.last()
        val inputPng = writeTensorPng(inputTensor, "large_online_input.png")
        val outputPng = writeTensorPng(outputTensor, "large_online_reconstruction.png")
        showImages?.invoke(inputPng, outputPng, psnr)
        emit(
            "large_online_main_output stream=${streamFile.absolutePath} bytes=${result.stream.size} " +
                "sha256=${sha256(result.stream)} final_psnr_db=${format(psnr)} " +
                "mean_psnr_db=${format(framePsnr.average())} min_psnr_db=${format(framePsnr.minOrNull()!!)} " +
                "input=${inputFile.absolutePath} input_sha256=${sha256(frames.first())} " +
                "reconstruction=${outputPng.absolutePath}",
        )
        emit("large_online_main_complete status=PASS all_models_exercised=true")
    }

    private fun encodeVideo(reader: OfflineVideoFrameReader, runtimes: PreparedRuntimes): VideoEncodeResult {
        val payloads = ArrayList<GvcFramePayload>()
        val hashes = ArrayList<String>()
        val presentationTimes = ArrayList<Long>()
        val timings = linkedMapOf<String, Long>()
        var sourceDecodeNs = 0L
        var firstPtsUs: Long? = null
        var encoderReferenceFrame: ByteArray? = null
        var encoderReferenceFeature: ByteArray? = null

        fun <T> timed(name: String, block: () -> T): T {
            val started = SystemClock.elapsedRealtimeNanos()
            return block().also { timings[name] = (timings[name] ?: 0L) + elapsedNs(started) }
        }

        while (true) {
            throwIfVideoCancelled()
            val readStarted = SystemClock.elapsedRealtimeNanos()
            val frame = reader.next()
            sourceDecodeNs += elapsedNs(readStarted)
            if (frame == null) break
            val index = payloads.size
            if (firstPtsUs == null) firstPtsUs = frame.presentationTimeUs
            presentationTimes += (frame.presentationTimeUs - firstPtsUs!!).coerceAtLeast(0)
            if (index == 0) {
                val y = timed("i_encoder") {
                    runtimes.iEncoder.run(runtimes.neuralInputs(listOf(frame.tensor), "i_q_enc")).single()
                }
                val entropy = timed("i_entropy_rans") {
                    runtimes.iEntropyEncoder.runCanonical(y, qp = runtimes.stream.qp)
                }
                require(entropy.size == 2)
                val reconstruction = timed("i_decoder_reference") {
                    runtimes.iDecoder.run(
                        runtimes.neuralInputs(listOf(entropy[0]), "i_q_dec", "i_q_recon"),
                    ).single()
                }
                payloads += GvcFramePayload(true, entropy[1])
                hashes += sha256(reconstruction)
                encoderReferenceFrame = reconstruction
            } else {
                val resetReference = shouldResetReference(index)
                val temporal = timed(if (resetReference) "p_temporal_from_frame" else "p_temporal_from_feature") {
                    if (resetReference) {
                        runtimes.temporalFromFrame.run(
                            runtimes.neuralInputs(
                                listOf(encoderReferenceFrame ?: error("missing encoder frame reference")),
                                "p_q_feature",
                            ),
                        )
                    } else {
                        runtimes.temporalFromFeature.run(
                            runtimes.neuralInputs(
                                listOf(encoderReferenceFeature ?: error("missing encoder feature reference")),
                                "p_q_feature",
                            ),
                        )
                    }
                }
                require(temporal.size == 3)
                val ctx = temporal[1]
                val ctxT = temporal[2]
                val y = timed("p_encoder") {
                    runtimes.pEncoder.run(runtimes.neuralInputs(listOf(frame.tensor, ctx), "p_q_enc")).single()
                }
                val entropy = timed("p_entropy_rans") {
                    runtimes.pEntropyEncoder.runCanonical(y, ctxT, qp = runtimes.stream.qp)
                }
                require(entropy.size == 2)
                val reconstruction = timed("p_decoder_reference") {
                    runtimes.pDecoder.run(
                        runtimes.neuralInputs(listOf(entropy[0], ctx), "p_q_dec", "p_q_recon"),
                    )
                }
                require(reconstruction.size == 2)
                payloads += GvcFramePayload(false, entropy[1])
                encoderReferenceFeature = reconstruction[0]
                encoderReferenceFrame = reconstruction[1]
                hashes += sha256(reconstruction[1])
            }
            if (index == 0 || (index + 1) % VIDEO_PROGRESS_INTERVAL == 0) {
                emit(
                    "large_offline_video_progress phase=encode frame=${index + 1} " +
                        "pts_ms=${format(presentationTimes.last() / 1000.0)} payload_bytes=${payloads.last().payload.size}",
                )
            }
        }
        return VideoEncodeResult(payloads, hashes, presentationTimes, timings, sourceDecodeNs)
    }

    private fun decodeVideo(
        reader: OfflineVideoFrameReader,
        frames: List<GvcFramePayload>,
        expectedReconstructionHashes: List<String>,
        expectedPresentationTimesUs: List<Long>,
        runtimes: PreparedRuntimes,
        writer: ReconstructionMp4Writer,
    ): VideoDecodeResult {
        require(frames.size == expectedReconstructionHashes.size && frames.size == expectedPresentationTimesUs.size)
        val timings = linkedMapOf<String, Long>()
        val psnrs = ArrayList<Double>(frames.size)
        var sourceDecodeNs = 0L
        var mp4WriteNs = 0L
        var decoderReferenceFrame: ByteArray? = null
        var decoderReferenceFeature: ByteArray? = null
        var lastInput: ByteArray? = null
        var lastReconstruction: ByteArray? = null

        fun <T> timed(name: String, block: () -> T): T {
            val started = SystemClock.elapsedRealtimeNanos()
            return block().also { timings[name] = (timings[name] ?: 0L) + elapsedNs(started) }
        }

        frames.forEachIndexed { index, payload ->
            throwIfVideoCancelled()
            val readStarted = SystemClock.elapsedRealtimeNanos()
            val sourceFrame = reader.next() ?: error("source video ended before GVC frame=$index")
            sourceDecodeNs += elapsedNs(readStarted)
            val reconstruction: ByteArray
            if (payload.isIFrame) {
                require(index == 0)
                val yHat = timed("i_entropy_rans") {
                    runtimes.iEntropyDecoder.runCanonical(payload.payload, qp = runtimes.stream.qp)
                }
                reconstruction = timed("i_decoder") {
                    runtimes.iDecoder.run(
                        runtimes.neuralInputs(listOf(yHat), "i_q_dec", "i_q_recon"),
                    ).single()
                }
                decoderReferenceFrame = reconstruction
            } else {
                val resetReference = shouldResetReference(index)
                val temporal = timed(if (resetReference) "p_temporal_from_frame" else "p_temporal_from_feature") {
                    if (resetReference) {
                        runtimes.temporalFromFrame.run(
                            runtimes.neuralInputs(
                                listOf(decoderReferenceFrame ?: error("missing decoder frame reference")),
                                "p_q_feature",
                            ),
                        )
                    } else {
                        runtimes.temporalFromFeature.run(
                            runtimes.neuralInputs(
                                listOf(decoderReferenceFeature ?: error("missing decoder feature reference")),
                                "p_q_feature",
                            ),
                        )
                    }
                }
                require(temporal.size == 3)
                val ctx = temporal[1]
                val ctxT = temporal[2]
                val yHat = timed("p_entropy_rans") {
                    runtimes.pEntropyDecoder.runCanonical(payload.payload, ctxT, qp = runtimes.stream.qp)
                }
                val outputs = timed("p_decoder") {
                    runtimes.pDecoder.run(
                        runtimes.neuralInputs(listOf(yHat, ctx), "p_q_dec", "p_q_recon"),
                    )
                }
                require(outputs.size == 2)
                decoderReferenceFeature = outputs[0]
                decoderReferenceFrame = outputs[1]
                reconstruction = outputs[1]
            }
            require(sha256(reconstruction) == expectedReconstructionHashes[index]) {
                "encoder/independent-decoder reconstruction mismatch at video frame=$index"
            }
            val psnr = calculatePsnr(sourceFrame.tensor, reconstruction)
            psnrs += psnr
            val writeStarted = SystemClock.elapsedRealtimeNanos()
            writer.writeFrame(reconstruction, expectedPresentationTimesUs[index])
            mp4WriteNs += elapsedNs(writeStarted)
            lastInput = sourceFrame.tensor
            lastReconstruction = reconstruction
            if (showVideoFrames != null && (index == 0 || (index + 1) % VIDEO_PREVIEW_INTERVAL == 0 || index == frames.lastIndex)) {
                showVideoFrames.invoke(
                    VideoTensorCodec.toBitmap(sourceFrame.tensor, WIDTH, HEIGHT),
                    VideoTensorCodec.toBitmap(reconstruction, WIDTH, HEIGHT),
                    psnr,
                    index + 1,
                )
            }
            if (index == 0 || (index + 1) % VIDEO_PROGRESS_INTERVAL == 0 || index == frames.lastIndex) {
                emit(
                    "large_offline_video_progress phase=decode frame=${index + 1}/${frames.size} " +
                        "pts_ms=${format(expectedPresentationTimesUs[index] / 1000.0)} psnr_db=${format(psnr)}",
                )
            }
        }
        return VideoDecodeResult(
            timings = timings,
            psnrs = psnrs,
            sourceDecodeNs = sourceDecodeNs,
            mp4WriteNs = mp4WriteNs,
            lastInput = lastInput ?: error("missing final input frame"),
            lastReconstruction = lastReconstruction ?: error("missing final reconstruction"),
        )
    }

    private fun emitVideoTimings(phase: String, timings: Map<String, Long>, frameCount: Int) {
        timings.forEach { (stage, elapsed) ->
            emit(
                "large_offline_video_speed phase=$phase stage=$stage total_ms=${format(elapsed / 1_000_000.0)} " +
                    "per_sequence_frame_ms=${format(elapsed / frameCount / 1_000_000.0)} includes_create=false",
            )
        }
    }

    private fun throwIfVideoCancelled() {
        if (videoCancelled.get()) throw CancellationException("video test cancelled")
    }

    private fun writeNhwcPng(bytes: ByteArray, output: File): File {
        val bitmap = VideoTensorCodec.toBitmap(bytes, WIDTH, HEIGHT)
        output.parentFile?.mkdirs()
        output.outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
        bitmap.recycle()
        return output
    }

    private fun timestamp(): String = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())

    private fun execute(
        inputCount: Int,
        inputAt: (Int) -> ByteArray,
        runtimes: PreparedRuntimes,
        collectTimings: Boolean,
        retainDecodedFrames: Boolean = true,
        decodedConsumer: ((Int, ByteArray) -> Unit)? = null,
        pEntropyBoundaryFrames: Set<Int> = emptySet(),
    ): RunResult {
        require(inputCount > 0)
        val timings = linkedMapOf<String, Long>()
        var excludedNs = 0L
        fun <T> timed(name: String, block: () -> T): T {
            if (!collectTimings) return block()
            val started = SystemClock.elapsedRealtimeNanos()
            return block().also { timings[name] = (timings[name] ?: 0L) + elapsedNs(started) }
        }

        val totalStarted = SystemClock.elapsedRealtimeNanos()
        val payloads = ArrayList<GvcFramePayload>(inputCount)
        val encoderReconstructionHashes = ArrayList<String>(inputCount)
        val pEntropyFrameTimes = ArrayList<Long>((inputCount - 1).coerceAtLeast(0))
        val iBoundaries = linkedMapOf<String, ByteArray>()
        var encoderReferenceFrame: ByteArray? = null
        var encoderReferenceFeature: ByteArray? = null

        repeat(inputCount) { index ->
            val loadStarted = SystemClock.elapsedRealtimeNanos()
            val input = inputAt(index)
            if (collectTimings) excludedNs += elapsedNs(loadStarted)
            if (index == 0) {
                val y = timed("encode_i_encoder") {
                    runtimes.iEncoder.run(runtimes.neuralInputs(listOf(input), "i_q_enc")).single()
                }
                val entropy = timed("encode_i_entropy_rans") {
                    runtimes.iEntropyEncoder.runCanonical(y, qp = runtimes.stream.qp)
                }
                require(entropy.size == 2) { "I entropy canonical outputs=${entropy.size}" }
                if (collectTimings) {
                    iBoundaries["android_i_y_pre_prior"] = y
                    iBoundaries["android_i_y_hat_encode"] = entropy[0]
                }
                val reconstruction = timed("encode_i_decoder") {
                    runtimes.iDecoder.run(
                        runtimes.neuralInputs(listOf(entropy[0]), "i_q_dec", "i_q_recon"),
                    ).single()
                }
                payloads += GvcFramePayload(true, entropy[1])
                encoderReconstructionHashes += sha256(reconstruction)
                encoderReferenceFrame = reconstruction
            } else {
                val resetReference = shouldResetReference(index)
                val temporal = timed(if (resetReference) "encode_temporal_from_frame" else "encode_temporal_from_feature") {
                    if (resetReference) {
                        runtimes.temporalFromFrame.run(
                            runtimes.neuralInputs(
                                listOf(encoderReferenceFrame ?: error("missing encoder I reference")),
                                "p_q_feature",
                            ),
                        )
                    } else {
                        runtimes.temporalFromFeature.run(
                            runtimes.neuralInputs(
                                listOf(encoderReferenceFeature ?: error("missing encoder P reference feature")),
                                "p_q_feature",
                            ),
                        )
                    }
                }
                require(temporal.size == 3) { "encoder temporal outputs=${temporal.size}" }
                val ctx = temporal[1]
                val ctxT = temporal[2]
                val y = timed("encode_p_encoder") {
                    runtimes.pEncoder.run(runtimes.neuralInputs(listOf(input, ctx), "p_q_enc")).single()
                }
                val entropyStarted = SystemClock.elapsedRealtimeNanos()
                val entropy = if (index in pEntropyBoundaryFrames) {
                    val outputs = runtimes.pEntropyEncoder.run(y, ctxT, qp = runtimes.stream.qp)
                    require(outputs.size == 8) { "P entropy diagnostic outputs=${outputs.size}" }
                    val payloadSize = java.nio.ByteBuffer.wrap(outputs[7])
                        .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                        .int
                    require(payloadSize in 0..outputs[6].size) { "invalid P diagnostic payload size=$payloadSize" }
                    val prefix = "android_p_frame_${(index + 1).toString().padStart(3, '0')}"
                    iBoundaries["${prefix}_y_pre_prior"] = y
                    iBoundaries["${prefix}_ctx"] = ctx
                    iBoundaries["${prefix}_ctx_t"] = ctxT
                    iBoundaries["${prefix}_z_hat"] = outputs[0]
                    iBoundaries["${prefix}_y_q_w_0"] = outputs[1]
                    iBoundaries["${prefix}_y_q_w_1"] = outputs[2]
                    iBoundaries["${prefix}_s_w_0"] = outputs[3]
                    iBoundaries["${prefix}_s_w_1"] = outputs[4]
                    iBoundaries["${prefix}_y_hat"] = outputs[5]
                    iBoundaries["${prefix}_rans_payload"] = outputs[6].copyOf(payloadSize)
                    listOf(outputs[5], outputs[6].copyOf(payloadSize))
                } else {
                    runtimes.pEntropyEncoder.runCanonical(y, ctxT, qp = runtimes.stream.qp)
                }
                if (collectTimings) {
                    val entropyNs = elapsedNs(entropyStarted)
                    timings["encode_p_entropy_rans"] =
                        (timings["encode_p_entropy_rans"] ?: 0L) + entropyNs
                    pEntropyFrameTimes += entropyNs
                    if (index == 1 || resetReference || index % P_ENTROPY_LOG_INTERVAL == 0 || index == inputCount - 1) {
                        emit(
                            "large_online_p_entropy_progress frame=${index + 1} " +
                                "reference=${if (resetReference) "frame" else "feature"} " +
                                "elapsed_ms=${format(entropyNs / 1_000_000.0)} payload_bytes=${entropy[1].size}",
                        )
                    }
                }
                require(entropy.size == 2) { "P entropy canonical outputs=${entropy.size}" }
                val reconstruction = timed("encode_p_decoder") {
                    runtimes.pDecoder.run(
                        runtimes.neuralInputs(listOf(entropy[0], ctx), "p_q_dec", "p_q_recon"),
                    )
                }
                require(reconstruction.size == 2) { "encoder P decoder outputs=${reconstruction.size}" }
                payloads += GvcFramePayload(false, entropy[1])
                encoderReferenceFeature = reconstruction[0]
                encoderReferenceFrame = reconstruction[1]
                encoderReconstructionHashes += sha256(reconstruction[1])
            }
        }
        if (collectTimings && pEntropyFrameTimes.isNotEmpty()) {
            emit(
                "large_online_p_entropy_distribution frames=${pEntropyFrameTimes.size} " +
                    "mean_ms=${format(pEntropyFrameTimes.average() / 1_000_000.0)} " +
                    "p50_ms=${format(percentile(pEntropyFrameTimes, 0.50) / 1_000_000.0)} " +
                    "p90_ms=${format(percentile(pEntropyFrameTimes, 0.90) / 1_000_000.0)} " +
                    "max_ms=${format((pEntropyFrameTimes.maxOrNull() ?: 0L) / 1_000_000.0)}",
            )
        }

        val stream = timed("stream_mux") { GvcStreamMuxer.muxSequence(runtimes.stream, payloads) }
        val parsed = timed("stream_demux") { GvcStreamMuxer.demuxSequence(stream) }
        require(parsed.frames.size == inputCount) { "decoded frame count=${parsed.frames.size}" }
        require(parsed.stream.qp == runtimes.stream.qp) { "decoded QP=${parsed.stream.qp}" }

        val decodedFrames = ArrayList<ByteArray>(if (retainDecodedFrames) parsed.frames.size else 0)
        var decoderReferenceFrame: ByteArray? = null
        var decoderReferenceFeature: ByteArray? = null
        parsed.frames.forEachIndexed { index, framePayload ->
            if (framePayload.isIFrame) {
                require(index == 0) { "I payload must be frame zero" }
                val yHat = timed("decode_i_entropy_rans") {
                    runtimes.iEntropyDecoder.runCanonical(framePayload.payload, qp = runtimes.stream.qp)
                }
                if (collectTimings) iBoundaries["android_i_y_hat_decode"] = yHat
                val reconstruction = timed("decode_i_decoder") {
                    runtimes.iDecoder.run(
                        runtimes.neuralInputs(listOf(yHat), "i_q_dec", "i_q_recon"),
                    ).single()
                }
                decoderReferenceFrame = reconstruction
                require(encoderReconstructionHashes[index] == sha256(reconstruction)) {
                    "encoder/decoder reconstruction mismatch at frame=$index"
                }
                if (retainDecodedFrames) decodedFrames += reconstruction
                if (decodedConsumer != null) {
                    val consumerStarted = SystemClock.elapsedRealtimeNanos()
                    decodedConsumer(index, reconstruction)
                    if (collectTimings) excludedNs += elapsedNs(consumerStarted)
                }
            } else {
                val resetReference = shouldResetReference(index)
                val temporal = timed(if (resetReference) "decode_temporal_from_frame" else "decode_temporal_from_feature") {
                    if (resetReference) {
                        runtimes.temporalFromFrame.run(
                            runtimes.neuralInputs(
                                listOf(decoderReferenceFrame ?: error("missing decoder I reference")),
                                "p_q_feature",
                            ),
                        )
                    } else {
                        runtimes.temporalFromFeature.run(
                            runtimes.neuralInputs(
                                listOf(decoderReferenceFeature ?: error("missing decoder P reference feature")),
                                "p_q_feature",
                            ),
                        )
                    }
                }
                require(temporal.size == 3) { "decoder temporal outputs=${temporal.size}" }
                val ctx = temporal[1]
                val ctxT = temporal[2]
                val yHat = timed("decode_p_entropy_rans") {
                    runtimes.pEntropyDecoder.runCanonical(
                        framePayload.payload,
                        ctxT,
                        qp = runtimes.stream.qp,
                    )
                }
                val reconstruction = timed("decode_p_decoder") {
                    runtimes.pDecoder.run(
                        runtimes.neuralInputs(listOf(yHat, ctx), "p_q_dec", "p_q_recon"),
                    )
                }
                require(reconstruction.size == 2) { "decoded P outputs=${reconstruction.size}" }
                decoderReferenceFeature = reconstruction[0]
                decoderReferenceFrame = reconstruction[1]
                require(encoderReconstructionHashes[index] == sha256(reconstruction[1])) {
                    "encoder/decoder reconstruction mismatch at frame=$index"
                }
                if (retainDecodedFrames) decodedFrames += reconstruction[1]
                if (decodedConsumer != null) {
                    val consumerStarted = SystemClock.elapsedRealtimeNanos()
                    decodedConsumer(index, reconstruction[1])
                    if (collectTimings) excludedNs += elapsedNs(consumerStarted)
                }
            }
        }
        if (collectTimings) timings["total"] = elapsedNs(totalStarted) - excludedNs
        return RunResult(stream, decodedFrames, timings, iBoundaries)
    }

    private fun prepare(qp: Int): PreparedRuntimes {
        require(qp in LargeDynamicQuantScales.REQUIRED_QPS) {
            "Large online QP must be one of ${LargeDynamicQuantScales.REQUIRED_QPS.sorted()}"
        }
        prepared?.let {
            it.selectQp(qp)
            return it
        }
        val root = findPackageRoot()
        val manifest = JSONObject(root.resolve("manifest.json").readText())
        val quantScales = LargeDynamicQuantScales.load(root, manifest)
        val packagedQp = manifest.optInt("default_qp", manifest.getInt("qp"))
        if (quantScales == null) {
            require(qp == packagedQp) {
                "fixed Large package supports only QP=$packagedQp; install a dynamic-QP package"
            }
        } else {
            require(qp in quantScales.supportedQps) {
                "dynamic Large package does not support QP=$qp"
            }
        }
        val resolution = manifest.getJSONObject("resolution")
        val stream = StreamSpec(
            path = "",
            height = resolution.getInt("height"),
            width = resolution.getInt("width"),
            qp = qp,
            ecPart = 0,
            useAdaI = 0,
        )
        require(stream.height == HEIGHT && stream.width == WIDTH) {
            "Large online main requires ${HEIGHT}x$WIDTH"
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
                quantScales = quantScales,
                fixedPackageQp = if (quantScales == null) packagedQp else null,
            ).also {
                prepared = it
                emit(
                    "large_online_main_prepare models=10 create_ms=${format(elapsedMs(createStarted))} " +
                        "backend=official_aar_neuron fast_models=10 decoder_models=scaled_variance_fp16 " +
                        "dynamic_qp=${quantScales != null} qp=$qp " +
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

    private fun emitSummary(results: List<RunResult>, frameCount: Int) {
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
        val totalValues = results.mapNotNull { it.timings["total"] }
        if (totalValues.isNotEmpty()) {
            val meanFrameNs = totalValues.average() / frameCount
            emit(
                "large_online_video_summary frames=$frameCount i_frames=1 p_frames=${frameCount - 1} " +
                    "mean_sequence_ms=${format(totalValues.average() / 1_000_000.0)} " +
                    "mean_frame_ms=${format(meanFrameNs / 1_000_000.0)} " +
                    "fps=${format(1_000_000_000.0 / meanFrameNs)} includes_create=false",
            )
        }
    }

    private fun calculatePsnr(input: ByteArray, reconstruction: ByteArray): Double {
        require(input.size == reconstruction.size && input.size % 4 == 0) { "PSNR tensor byte count mismatch" }
        val inputFloats = java.nio.ByteBuffer.wrap(input).order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        val reconstructionFloats = java.nio.ByteBuffer.wrap(reconstruction)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN)
            .asFloatBuffer()
        var sumSq = 0.0
        repeat(input.size / 4) { index ->
            val diff = reconstructionFloats[index].displayValue() - inputFloats[index].displayValue()
            sumSq += diff * diff
        }
        val rmse = sqrt(sumSq / (input.size / 4))
        return if (rmse == 0.0) Double.POSITIVE_INFINITY else 20.0 * log10(1.0 / rmse)
    }

    private fun shouldResetReference(frameIndex: Int): Boolean =
        frameIndex > 0 && frameIndex % REFERENCE_RESET_INTERVAL == 1

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

    private data class SequenceRunResult(
        val run: RunResult,
        val psnrs: List<Double>,
        val lastDecoded: ByteArray,
    )

    private data class VideoEncodeResult(
        val payloads: List<GvcFramePayload>,
        val reconstructionHashes: List<String>,
        val presentationTimesUs: List<Long>,
        val timings: Map<String, Long>,
        val sourceDecodeNs: Long,
    )

    private data class VideoDecodeResult(
        val timings: Map<String, Long>,
        val psnrs: List<Double>,
        val sourceDecodeNs: Long,
        val mp4WriteNs: Long,
        val lastInput: ByteArray,
        val lastReconstruction: ByteArray,
    )

    private data class PreparedRuntimes(
        var stream: StreamSpec,
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
        val quantScales: LargeDynamicQuantScales?,
        val fixedPackageQp: Int?,
    ) : AutoCloseable {
        private var selectedQuantInputs = quantScales?.select(stream.qp)

        fun selectQp(qp: Int) {
            if (fixedPackageQp != null) {
                require(qp == fixedPackageQp) {
                    "fixed Large package supports only QP=$fixedPackageQp; install a dynamic-QP package"
                }
            } else {
                require(qp in (quantScales?.supportedQps ?: emptySet())) {
                    "dynamic Large package does not support QP=$qp"
                }
            }
            if (stream.qp != qp) {
                stream = stream.copy(qp = qp)
                selectedQuantInputs = quantScales?.select(qp)
            }
        }

        fun neuralInputs(inputs: List<ByteArray>, vararg scaleNames: String): List<ByteArray> {
            val scales = selectedQuantInputs ?: return inputs
            return inputs + scaleNames.map { name -> scales[name] ?: error("missing selected scale $name") }
        }

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
        const val DEFAULT_QP = 9
        const val DEFAULT_FRAME_COUNT = 3
        const val REFERENCE_RESET_INTERVAL = 32
        const val P_ENTROPY_LOG_INTERVAL = 16
        const val VIDEO_PROGRESS_INTERVAL = 24
        const val VIDEO_PREVIEW_INTERVAL = 8
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
