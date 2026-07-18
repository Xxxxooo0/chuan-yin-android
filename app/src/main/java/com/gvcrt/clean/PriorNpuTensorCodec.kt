package com.gvcrt.clean

import java.nio.ByteBuffer
import java.nio.ByteOrder

object PriorNpuTensorCodec {
    fun nchwToNhwcF32(data: FloatArray, channels: Int, height: Int, width: Int): ByteArray {
        require(data.size == channels * height * width) { "NCHW data size mismatch" }
        val bytes = ByteArray(data.size * 4)
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        for (row in 0 until height) {
            for (column in 0 until width) {
                for (channel in 0 until channels) {
                    buffer.putFloat(data[(channel * height + row) * width + column])
                }
            }
        }
        return bytes
    }

    fun nchwToNhwcFp16(data: FloatArray, channels: Int, height: Int, width: Int): ByteArray {
        require(data.size == channels * height * width) { "NCHW data size mismatch" }
        val bytes = ByteArray(data.size * 2)
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        for (row in 0 until height) {
            for (column in 0 until width) {
                for (channel in 0 until channels) {
                    buffer.putShort(TensorIO.floatToHalfBits(data[(channel * height + row) * width + column]))
                }
            }
        }
        return bytes
    }

    fun nhwcFp16ToNchw(bytes: ByteArray, channels: Int, height: Int, width: Int): FloatArray {
        require(bytes.size == channels * height * width * 2) { "NHWC FP16 byte size mismatch" }
        val output = FloatArray(channels * height * width)
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        for (row in 0 until height) {
            for (column in 0 until width) {
                for (channel in 0 until channels) {
                    output[(channel * height + row) * width + column] = TensorIO.halfBitsToFloat(buffer.short)
                }
            }
        }
        return output
    }

    fun nhwcF32ToNchw(bytes: ByteArray, channels: Int, height: Int, width: Int): FloatArray {
        require(bytes.size == channels * height * width * 4) { "NHWC FP32 byte size mismatch" }
        val output = FloatArray(channels * height * width)
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        for (row in 0 until height) {
            for (column in 0 until width) {
                for (channel in 0 until channels) {
                    output[(channel * height + row) * width + column] = buffer.float
                }
            }
        }
        return output
    }
}
