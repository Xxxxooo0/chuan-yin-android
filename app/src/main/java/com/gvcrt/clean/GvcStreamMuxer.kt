package com.gvcrt.clean

import java.io.ByteArrayOutputStream

data class ParsedGvcStream(
    val stream: StreamSpec,
    val iPayload: ByteArray,
    val pPayload: ByteArray,
)

data class GvcFramePayload(
    val isIFrame: Boolean,
    val payload: ByteArray,
)

data class ParsedGvcSequence(
    val stream: StreamSpec,
    val frames: List<GvcFramePayload>,
)

object GvcStreamMuxer {
    private const val NAL_SPS = 0
    private const val NAL_I = 1
    private const val NAL_P = 2

    fun mux(stream: StreamSpec, iPayload: ByteArray, pPayload: ByteArray): ByteArray {
        return muxSequence(
            stream,
            listOf(
                GvcFramePayload(isIFrame = true, payload = iPayload),
                GvcFramePayload(isIFrame = false, payload = pPayload),
            ),
        )
    }

    fun muxSequence(stream: StreamSpec, frames: List<GvcFramePayload>): ByteArray {
        require(stream.ecPart == 0) { "clean Android currently supports one rANS stream" }
        require(stream.useAdaI == 0) { "clean Android currently supports use_ada_i=0" }
        require(frames.isNotEmpty() && frames.first().isIFrame) { "GVC sequence must start with an I frame" }
        require(frames.drop(1).none(GvcFramePayload::isIFrame)) { "only the first sequence frame may be I" }
        val output = ByteArrayOutputStream()
        output.write((NAL_SPS shl 4) or 0)
        writeAdaptiveUInt(output, stream.height)
        writeAdaptiveUInt(output, stream.width)
        output.write((stream.ecPart shl 2) or stream.useAdaI)
        frames.forEach { frame ->
            writeIp(output, if (frame.isIFrame) NAL_I else NAL_P, stream.qp, frame.payload)
        }
        return output.toByteArray()
    }

    fun demux(input: ByteArray): ParsedGvcStream {
        val sequence = demuxSequence(input)
        require(sequence.frames.size == 2) { "expected one I and one P frame, got=${sequence.frames.size}" }
        require(sequence.frames[0].isIFrame && !sequence.frames[1].isIFrame) {
            "expected I/P frame order"
        }
        return ParsedGvcStream(
            sequence.stream,
            sequence.frames[0].payload,
            sequence.frames[1].payload,
        )
    }

    fun demuxSequence(input: ByteArray): ParsedGvcSequence {
        val reader = ByteReader(input)
        val spsHeader = reader.readU8()
        require((spsHeader ushr 4) == NAL_SPS && (spsHeader and 0x0f) == 0) {
            "expected SPS NAL at byte 0, got=0x${spsHeader.toString(16)}"
        }
        val height = reader.readAdaptiveUInt()
        val width = reader.readAdaptiveUInt()
        val flags = reader.readU8()
        val ecPart = flags ushr 2
        val useAdaI = flags and 0x03

        val frames = ArrayList<GvcFramePayload>()
        var qp: Int? = null
        while (!reader.exhausted()) {
            val nal = reader.readIp()
            require(nal.type == if (frames.isEmpty()) NAL_I else NAL_P) {
                "unexpected NAL type ${nal.type} at frame ${frames.size}"
            }
            require(qp == null || qp == nal.qp) { "sequence QP mismatch: $qp vs ${nal.qp}" }
            qp = nal.qp
            frames += GvcFramePayload(nal.type == NAL_I, nal.payload)
        }
        require(frames.isNotEmpty()) { "GVC sequence contains no frames" }
        return ParsedGvcSequence(
            stream = StreamSpec(
                path = "",
                height = height,
                width = width,
                qp = qp ?: error("missing sequence QP"),
                ecPart = ecPart,
                useAdaI = useAdaI,
            ),
            frames = frames,
        )
    }

    private fun writeIp(output: ByteArrayOutputStream, nalType: Int, qp: Int, payload: ByteArray) {
        require(qp in 0..255) { "QP must fit in uint8" }
        output.write(nalType shl 4)
        output.write(qp)
        writeAdaptiveUInt(output, payload.size)
        output.write(payload)
    }

    private fun writeAdaptiveUInt(output: ByteArrayOutputStream, value: Int) {
        require(value >= 0 && value < (1 shl 30)) { "adaptive uint outside supported range" }
        when {
            value < (1 shl 7) -> output.write(value)
            value < (1 shl 14) -> {
                output.write(((value ushr 8) and 0xff) or 0x80)
                output.write(value and 0xff)
            }
            else -> {
                output.write(((value ushr 24) and 0x3f) or 0xc0)
                output.write((value ushr 16) and 0xff)
                output.write((value ushr 8) and 0xff)
                output.write(value and 0xff)
            }
        }
    }

    private data class IpNal(val type: Int, val qp: Int, val payload: ByteArray)

    private class ByteReader(private val input: ByteArray) {
        var position: Int = 0
            private set

        fun exhausted(): Boolean = position == input.size

        fun readU8(): Int {
            require(position < input.size) { "unexpected end of stream at byte $position" }
            return input[position++].toInt() and 0xff
        }

        fun readAdaptiveUInt(): Int {
            val first = readU8()
            return when {
                first < 0x80 -> first
                first < 0xc0 -> ((first and 0x3f) shl 8) or readU8()
                else -> ((first and 0x3f) shl 24) or (readU8() shl 16) or (readU8() shl 8) or readU8()
            }
        }

        fun readIp(): IpNal {
            val header = readU8()
            val type = header ushr 4
            require(type == NAL_I || type == NAL_P) { "unsupported NAL type $type at byte ${position - 1}" }
            require((header and 0x0f) == 0) { "unsupported SPS id ${header and 0x0f}" }
            val qp = readU8()
            val size = readAdaptiveUInt()
            require(size <= input.size - position) {
                "NAL payload exceeds stream: size=$size remaining=${input.size - position}"
            }
            val payload = input.copyOfRange(position, position + size)
            position += size
            return IpNal(type, qp, payload)
        }
    }
}
