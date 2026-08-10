package com.gvcrt.clean

import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Converts FP32 tensors at the fixed TFLite NHWC boundary to the native NCHW layout. */
internal object NhwcTensorCodec {
    fun fromF32Le(name: String, nchwShape: LongArray, bytes: ByteArray): TensorValue {
        require(nchwShape.size == 4) { "$name requires a rank-4 NCHW shape" }
        val n = nchwShape[0].toInt()
        val c = nchwShape[1].toInt()
        val h = nchwShape[2].toInt()
        val w = nchwShape[3].toInt()
        require(bytes.size == n * c * h * w * 4) { "$name NHWC byte count mismatch" }
        val source = FloatArray(n * c * h * w)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(source)
        val destination = FloatArray(source.size)
        for (batch in 0 until n) for (row in 0 until h) for (column in 0 until w) for (channel in 0 until c) {
            val nhwc = ((batch * h + row) * w + column) * c + channel
            val nchw = ((batch * c + channel) * h + row) * w + column
            destination[nchw] = source[nhwc]
        }
        return TensorValue(name, nchwShape, destination)
    }

    fun toF32Le(tensor: TensorValue): ByteArray {
        val shape = tensor.shape
        require(shape.size == 4) { "${tensor.name} requires a rank-4 NCHW shape" }
        val n = shape[0].toInt()
        val c = shape[1].toInt()
        val h = shape[2].toInt()
        val w = shape[3].toInt()
        val destination = FloatArray(tensor.data.size)
        for (batch in 0 until n) for (row in 0 until h) for (column in 0 until w) for (channel in 0 until c) {
            val nhwc = ((batch * h + row) * w + column) * c + channel
            val nchw = ((batch * c + channel) * h + row) * w + column
            destination[nhwc] = tensor.data[nchw]
        }
        return ByteArray(destination.size * 4).also { bytes ->
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().put(destination)
        }
    }
}
