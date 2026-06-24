package com.gvcrt.clean

import java.io.ByteArrayOutputStream

object GvcStreamMuxer {
    private const val NAL_SPS = 0
    private const val NAL_I = 1
    private const val NAL_P = 2

    fun mux(stream: StreamSpec, iPayload: ByteArray, pPayload: ByteArray): ByteArray {
        require(stream.ecPart == 0) { "clean Android currently supports one rANS stream" }
        require(stream.useAdaI == 0) { "clean Android currently supports use_ada_i=0" }
        val output = ByteArrayOutputStream()
        output.write((NAL_SPS shl 4) or 0)
        writeAdaptiveUInt(output, stream.height)
        writeAdaptiveUInt(output, stream.width)
        output.write((stream.ecPart shl 2) or stream.useAdaI)
        writeIp(output, NAL_I, stream.qp, iPayload)
        writeIp(output, NAL_P, stream.qp, pPayload)
        return output.toByteArray()
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
}
