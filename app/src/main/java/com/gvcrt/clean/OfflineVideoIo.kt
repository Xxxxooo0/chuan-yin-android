package com.gvcrt.clean

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.ImageFormat
import android.media.Image
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaCodecList
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMuxer
import android.net.Uri
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

internal data class DecodedVideoFrame(
    val index: Int,
    val presentationTimeUs: Long,
    val tensor: ByteArray,
)

internal data class OfflineVideoInfo(
    val width: Int,
    val height: Int,
    val rotationDegrees: Int,
    val frameRate: Int,
    val durationUs: Long,
    val mime: String,
)

internal enum class VideoTensorRange {
    ZERO_TO_ONE,
    NEGATIVE_ONE_TO_ONE,
}

internal enum class VideoTensorColorSpace {
    RGB,
    YCBCR_BT709,
}

/** Sequential MediaCodec reader that converts decoded YUV frames to fixed NHWC FP32 tensors. */
internal class OfflineVideoFrameReader(
    context: Context,
    uri: Uri,
    private val targetWidth: Int,
    private val targetHeight: Int,
    private val maxDurationUs: Long,
    private val tensorRange: VideoTensorRange = VideoTensorRange.NEGATIVE_ONE_TO_ONE,
    private val tensorColorSpace: VideoTensorColorSpace = VideoTensorColorSpace.RGB,
) : AutoCloseable {
    private val extractor = MediaExtractor()
    private val decoder: MediaCodec
    private val bufferInfo = MediaCodec.BufferInfo()
    private var inputEnded = false
    private var outputEnded = false
    private var frameIndex = 0
    private var firstInputPtsUs: Long? = null
    private var firstOutputPtsUs: Long? = null

    val info: OfflineVideoInfo

    init {
        extractor.setDataSource(context, uri, null)
        val track = (0 until extractor.trackCount).firstOrNull { index ->
            extractor.getTrackFormat(index).getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true
        } ?: error("selected file contains no video track")
        extractor.selectTrack(track)
        val format = extractor.getTrackFormat(track)
        val mime = format.getString(MediaFormat.KEY_MIME) ?: error("video track has no MIME type")
        val width = format.getInteger(MediaFormat.KEY_WIDTH)
        val height = format.getInteger(MediaFormat.KEY_HEIGHT)
        val rotation = if (format.containsKey(MediaFormat.KEY_ROTATION)) {
            format.getInteger(MediaFormat.KEY_ROTATION)
        } else {
            0
        }
        val frameRate = if (format.containsKey(MediaFormat.KEY_FRAME_RATE)) {
            format.getNumber(MediaFormat.KEY_FRAME_RATE)?.toFloat()?.roundToInt()?.coerceAtLeast(1) ?: 24
        } else {
            24
        }
        val duration = if (format.containsKey(MediaFormat.KEY_DURATION)) {
            format.getLong(MediaFormat.KEY_DURATION)
        } else {
            maxDurationUs
        }
        info = OfflineVideoInfo(width, height, rotation, frameRate, duration, mime)
        format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
        decoder = MediaCodec.createDecoderByType(mime)
        decoder.configure(format, null, null, 0)
        decoder.start()
    }

    fun next(): DecodedVideoFrame? {
        if (outputEnded) return null
        while (true) {
            if (!inputEnded) feedInput()
            val outputIndex = decoder.dequeueOutputBuffer(bufferInfo, CODEC_TIMEOUT_US)
            when {
                outputIndex >= 0 -> {
                    val eos = bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                    if (bufferInfo.size > 0 && firstOutputPtsUs == null) firstOutputPtsUs = bufferInfo.presentationTimeUs
                    val withinLimit = bufferInfo.presentationTimeUs - (firstOutputPtsUs ?: 0L) < maxDurationUs
                    var frame: DecodedVideoFrame? = null
                    if (bufferInfo.size > 0 && withinLimit) {
                        val image = decoder.getOutputImage(outputIndex)
                            ?: error("decoder did not expose a YUV image; output format=${decoder.outputFormat}")
                        image.use {
                            require(it.format == ImageFormat.YUV_420_888) {
                                "unsupported decoder image format=${it.format}"
                            }
                            frame = DecodedVideoFrame(
                                index = frameIndex++,
                                presentationTimeUs = bufferInfo.presentationTimeUs,
                                tensor = VideoTensorCodec.fromYuv420Image(
                                    it,
                                    targetWidth,
                                    targetHeight,
                                    info.rotationDegrees,
                                    tensorRange,
                                    tensorColorSpace,
                                ),
                            )
                        }
                    }
                    decoder.releaseOutputBuffer(outputIndex, false)
                    if (eos || !withinLimit) outputEnded = true
                    if (frame != null) return frame
                    if (outputEnded) return null
                }
                outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> Unit
                outputIndex == MediaCodec.INFO_TRY_AGAIN_LATER && inputEnded -> Unit
            }
        }
    }

    private fun feedInput() {
        val inputIndex = decoder.dequeueInputBuffer(CODEC_TIMEOUT_US)
        if (inputIndex < 0) return
        val input = decoder.getInputBuffer(inputIndex) ?: error("decoder input buffer is unavailable")
        input.clear()
        val sampleTime = extractor.sampleTime
        if (sampleTime >= 0 && firstInputPtsUs == null) firstInputPtsUs = sampleTime
        val elapsedUs = if (sampleTime >= 0) sampleTime - (firstInputPtsUs ?: sampleTime) else Long.MAX_VALUE
        if (sampleTime < 0 || elapsedUs >= maxDurationUs) {
            decoder.queueInputBuffer(
                inputIndex,
                0,
                0,
                sampleTime.coerceAtLeast(0),
                MediaCodec.BUFFER_FLAG_END_OF_STREAM,
            )
            inputEnded = true
            return
        }
        val size = extractor.readSampleData(input, 0)
        if (size < 0) {
            decoder.queueInputBuffer(inputIndex, 0, 0, sampleTime, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
            inputEnded = true
        } else {
            decoder.queueInputBuffer(inputIndex, 0, size, sampleTime, extractor.sampleFlags)
            extractor.advance()
        }
    }

    override fun close() {
        runCatching { decoder.stop() }
        decoder.release()
        extractor.release()
    }

    private companion object {
        const val CODEC_TIMEOUT_US = 10_000L
    }
}

/** Byte-buffer H.264 writer. The MP4 is only a display artifact. */
internal class ReconstructionMp4Writer(
    output: File,
    private val width: Int,
    private val height: Int,
    frameRate: Int,
    bitrate: Int,
    private val tensorRange: VideoTensorRange = VideoTensorRange.NEGATIVE_ONE_TO_ONE,
    private val tensorColorSpace: VideoTensorColorSpace = VideoTensorColorSpace.RGB,
) : AutoCloseable {
    private val bufferInfo = MediaCodec.BufferInfo()
    private val encoder: MediaCodec
    private val muxer = MediaMuxer(output.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
    private val colorFormat: Int
    private var trackIndex = -1
    private var muxerStarted = false
    private var closed = false

    init {
        output.parentFile?.mkdirs()
        val codecInfo = MediaCodecList(MediaCodecList.ALL_CODECS).codecInfos.firstOrNull { codec ->
            codec.isEncoder && codec.supportedTypes.any { it.equals(MIME_AVC, ignoreCase = true) } &&
                codec.getCapabilitiesForType(MIME_AVC).colorFormats.any(SUPPORTED_COLOR_FORMATS::contains)
        } ?: error("device has no byte-buffer H.264 encoder")
        colorFormat = SUPPORTED_COLOR_FORMATS.first { candidate ->
            codecInfo.getCapabilitiesForType(MIME_AVC).colorFormats.contains(candidate)
        }
        val format = MediaFormat.createVideoFormat(MIME_AVC, width, height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, colorFormat)
            setInteger(MediaFormat.KEY_BIT_RATE, bitrate)
            setInteger(MediaFormat.KEY_FRAME_RATE, frameRate.coerceAtLeast(1))
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        }
        encoder = MediaCodec.createByCodecName(codecInfo.name)
        encoder.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        encoder.start()
    }

    fun writeFrame(nhwcF32Le: ByteArray, presentationTimeUs: Long) {
        check(!closed)
        val yuv = VideoTensorCodec.toYuv420(
            nhwcF32Le,
            width,
            height,
            semiPlanar = colorFormat == COLOR_YUV420_SEMIPLANAR,
            tensorRange = tensorRange,
            tensorColorSpace = tensorColorSpace,
        )
        while (true) {
            val inputIndex = encoder.dequeueInputBuffer(CODEC_TIMEOUT_US)
            if (inputIndex >= 0) {
                val input = encoder.getInputBuffer(inputIndex) ?: error("encoder input buffer is unavailable")
                require(input.capacity() >= yuv.size) { "encoder input capacity=${input.capacity()} yuv=${yuv.size}" }
                input.clear()
                input.put(yuv)
                encoder.queueInputBuffer(inputIndex, 0, yuv.size, presentationTimeUs, 0)
                break
            }
            drain(endOfStream = false)
        }
        drain(endOfStream = false)
    }

    override fun close() {
        if (closed) return
        closed = true
        var eosQueued = false
        while (!eosQueued) {
            val inputIndex = encoder.dequeueInputBuffer(CODEC_TIMEOUT_US)
            if (inputIndex >= 0) {
                encoder.queueInputBuffer(inputIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                eosQueued = true
            } else {
                drain(endOfStream = false)
            }
        }
        drain(endOfStream = true)
        runCatching { encoder.stop() }
        encoder.release()
        if (muxerStarted) muxer.stop()
        muxer.release()
    }

    private fun drain(endOfStream: Boolean) {
        while (true) {
            val outputIndex = encoder.dequeueOutputBuffer(bufferInfo, if (endOfStream) CODEC_TIMEOUT_US else 0)
            when {
                outputIndex == MediaCodec.INFO_TRY_AGAIN_LATER -> if (!endOfStream) return
                outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    check(!muxerStarted) { "encoder output format changed twice" }
                    trackIndex = muxer.addTrack(encoder.outputFormat)
                    muxer.start()
                    muxerStarted = true
                }
                outputIndex >= 0 -> {
                    val output = encoder.getOutputBuffer(outputIndex) ?: error("encoder output buffer is unavailable")
                    if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) bufferInfo.size = 0
                    if (bufferInfo.size > 0) {
                        check(muxerStarted) { "H.264 output arrived before format" }
                        output.position(bufferInfo.offset)
                        output.limit(bufferInfo.offset + bufferInfo.size)
                        muxer.writeSampleData(trackIndex, output, bufferInfo)
                    }
                    val eos = bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                    encoder.releaseOutputBuffer(outputIndex, false)
                    if (eos) return
                }
            }
        }
    }

    private companion object {
        const val MIME_AVC = "video/avc"
        const val CODEC_TIMEOUT_US = 10_000L
        const val COLOR_YUV420_SEMIPLANAR = MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar
        val SUPPORTED_COLOR_FORMATS = listOf(
            COLOR_YUV420_SEMIPLANAR,
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Planar,
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible,
        )
    }
}

internal object VideoTensorCodec {
    fun fromYuv420Image(
        image: Image,
        targetWidth: Int,
        targetHeight: Int,
        rotationDegrees: Int,
        tensorRange: VideoTensorRange = VideoTensorRange.NEGATIVE_ONE_TO_ONE,
        tensorColorSpace: VideoTensorColorSpace = VideoTensorColorSpace.RGB,
    ): ByteArray {
        val crop = image.cropRect
        val sourceWidth = crop.width()
        val sourceHeight = crop.height()
        val rotation = ((rotationDegrees % 360) + 360) % 360
        require(rotation == 0 || rotation == 90 || rotation == 180 || rotation == 270) {
            "unsupported video rotation=$rotationDegrees"
        }
        val displayWidth = if (rotation == 90 || rotation == 270) sourceHeight else sourceWidth
        val displayHeight = if (rotation == 90 || rotation == 270) sourceWidth else sourceHeight
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        return ByteArray(targetWidth * targetHeight * 3 * 4).also { bytes ->
            val output = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            for (dy in 0 until targetHeight) {
                val logicalY = (dy.toLong() * displayHeight / targetHeight).toInt().coerceAtMost(displayHeight - 1)
                for (dx in 0 until targetWidth) {
                    val logicalX = (dx.toLong() * displayWidth / targetWidth).toInt().coerceAtMost(displayWidth - 1)
                    val localX: Int
                    val localY: Int
                    when (rotation) {
                        90 -> {
                            localX = logicalY
                            localY = sourceHeight - 1 - logicalX
                        }
                        180 -> {
                            localX = sourceWidth - 1 - logicalX
                            localY = sourceHeight - 1 - logicalY
                        }
                        270 -> {
                            localX = sourceWidth - 1 - logicalY
                            localY = logicalX
                        }
                        else -> {
                            localX = logicalX
                            localY = logicalY
                        }
                    }
                    val sx = crop.left + localX
                    val sy = crop.top + localY
                    val y = sample(yPlane, sx, sy)
                    val u = sample(uPlane, sx / 2, sy / 2)
                    val v = sample(vPlane, sx / 2, sy / 2)
                    if (tensorColorSpace == VideoTensorColorSpace.YCBCR_BT709) {
                        output.putFloat(normalizeByte(y, tensorRange))
                        output.putFloat(normalizeByte(u, tensorRange))
                        output.putFloat(normalizeByte(v, tensorRange))
                    } else {
                        val yf = (y - 16).coerceAtLeast(0) * 1.164f
                        val r = (yf + 1.596f * (v - 128)).roundToInt().coerceIn(0, 255)
                        val g = (yf - 0.392f * (u - 128) - 0.813f * (v - 128)).roundToInt().coerceIn(0, 255)
                        val b = (yf + 2.017f * (u - 128)).roundToInt().coerceIn(0, 255)
                        output.putFloat(normalizeByte(r, tensorRange))
                        output.putFloat(normalizeByte(g, tensorRange))
                        output.putFloat(normalizeByte(b, tensorRange))
                    }
                }
            }
        }
    }

    fun toBitmap(
        nhwcF32Le: ByteArray,
        width: Int,
        height: Int,
        tensorRange: VideoTensorRange = VideoTensorRange.NEGATIVE_ONE_TO_ONE,
        tensorColorSpace: VideoTensorColorSpace = VideoTensorColorSpace.RGB,
    ): Bitmap {
        require(nhwcF32Le.size == width * height * 3 * 4)
        val values = ByteBuffer.wrap(nhwcF32Le).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        val pixels = IntArray(width * height)
        for (index in pixels.indices) {
            val base = index * 3
            pixels[index] = if (tensorColorSpace == VideoTensorColorSpace.YCBCR_BT709) {
                val y = displayUnit(values[base], tensorRange)
                val cb = displayUnit(values[base + 1], tensorRange)
                val cr = displayUnit(values[base + 2], tensorRange)
                val r = y + 1.5748f * (cr - 0.5f)
                val g = y - 0.187324f * (cb - 0.5f) - 0.468124f * (cr - 0.5f)
                val b = y + 1.8556f * (cb - 0.5f)
                Color.rgb(unitByte(r), unitByte(g), unitByte(b))
            } else {
                Color.rgb(
                    displayByte(values[base], tensorRange),
                    displayByte(values[base + 1], tensorRange),
                    displayByte(values[base + 2], tensorRange),
                )
            }
        }
        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).apply {
            setPixels(pixels, 0, width, 0, 0, width, height)
        }
    }

    fun toYuv420(
        nhwcF32Le: ByteArray,
        width: Int,
        height: Int,
        semiPlanar: Boolean,
        tensorRange: VideoTensorRange = VideoTensorRange.NEGATIVE_ONE_TO_ONE,
        tensorColorSpace: VideoTensorColorSpace = VideoTensorColorSpace.RGB,
    ): ByteArray {
        require(width % 2 == 0 && height % 2 == 0)
        require(nhwcF32Le.size == width * height * 3 * 4)
        val values = ByteBuffer.wrap(nhwcF32Le).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        val plane = width * height
        val output = ByteArray(plane * 3 / 2)
        if (tensorColorSpace == VideoTensorColorSpace.YCBCR_BT709) {
            for (y in 0 until height) for (x in 0 until width) {
                val base = (y * width + x) * 3
                output[y * width + x] = displayByte(values[base], tensorRange).toByte()
            }
            for (y in 0 until height step 2) for (x in 0 until width step 2) {
                var cb = 0f
                var cr = 0f
                for (yy in 0..1) for (xx in 0..1) {
                    val base = ((y + yy) * width + x + xx) * 3
                    cb += displayUnit(values[base + 1], tensorRange)
                    cr += displayUnit(values[base + 2], tensorRange)
                }
                val chromaIndex = (y / 2) * (width / 2) + x / 2
                val ub = unitByte(cb / 4f).toByte()
                val vb = unitByte(cr / 4f).toByte()
                if (semiPlanar) {
                    output[plane + chromaIndex * 2] = ub
                    output[plane + chromaIndex * 2 + 1] = vb
                } else {
                    output[plane + chromaIndex] = ub
                    output[plane + plane / 4 + chromaIndex] = vb
                }
            }
            return output
        }
        for (y in 0 until height) for (x in 0 until width) {
            val base = (y * width + x) * 3
            val r = displayByte(values[base], tensorRange).toFloat()
            val g = displayByte(values[base + 1], tensorRange).toFloat()
            val b = displayByte(values[base + 2], tensorRange).toFloat()
            output[y * width + x] = (16f + 0.257f * r + 0.504f * g + 0.098f * b)
                .roundToInt().coerceIn(0, 255).toByte()
        }
        for (y in 0 until height step 2) for (x in 0 until width step 2) {
            var u = 0f
            var v = 0f
            for (yy in 0..1) for (xx in 0..1) {
                val base = ((y + yy) * width + x + xx) * 3
                val r = displayByte(values[base], tensorRange).toFloat()
                val g = displayByte(values[base + 1], tensorRange).toFloat()
                val b = displayByte(values[base + 2], tensorRange).toFloat()
                u += 128f - 0.148f * r - 0.291f * g + 0.439f * b
                v += 128f + 0.439f * r - 0.368f * g - 0.071f * b
            }
            val chromaIndex = (y / 2) * (width / 2) + x / 2
            val ub = (u / 4f).roundToInt().coerceIn(0, 255).toByte()
            val vb = (v / 4f).roundToInt().coerceIn(0, 255).toByte()
            if (semiPlanar) {
                output[plane + chromaIndex * 2] = ub
                output[plane + chromaIndex * 2 + 1] = vb
            } else {
                output[plane + chromaIndex] = ub
                output[plane + plane / 4 + chromaIndex] = vb
            }
        }
        return output
    }

    private fun sample(plane: Image.Plane, x: Int, y: Int): Int {
        val index = plane.buffer.position() + y * plane.rowStride + x * plane.pixelStride
        return plane.buffer.get(index).toInt() and 0xff
    }

    private fun normalizeByte(value: Int, tensorRange: VideoTensorRange): Float =
        when (tensorRange) {
            VideoTensorRange.ZERO_TO_ONE -> value / 255f
            VideoTensorRange.NEGATIVE_ONE_TO_ONE -> value / 127.5f - 1f
        }

    private fun displayByte(value: Float, tensorRange: VideoTensorRange): Int =
        unitByte(displayUnit(value, tensorRange))

    private fun displayUnit(value: Float, tensorRange: VideoTensorRange): Float =
        when (tensorRange) {
            VideoTensorRange.ZERO_TO_ONE -> value.coerceIn(0f, 1f)
            VideoTensorRange.NEGATIVE_ONE_TO_ONE -> (value.coerceIn(-1f, 1f) + 1f) * 0.5f
        }

    private fun unitByte(value: Float): Int = (value.coerceIn(0f, 1f) * 255f).roundToInt()
}
