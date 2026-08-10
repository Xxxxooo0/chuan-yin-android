package com.gvcrt.clean

import java.lang.Math.rint
import java.nio.ByteBuffer
import java.nio.ByteOrder

internal data class NhwcFloatTensor(
    val name: String,
    val shape: LongArray,
    val data: FloatArray,
) {
    init {
        require(shape.size == 4 && shape.fold(1L) { product, value -> product * value } == data.size.toLong()) {
            "$name invalid NHWC tensor ${shape.contentToString()} data=${data.size}"
        }
    }

    fun toF32Le(): ByteArray = ByteArray(data.size * 4).also { bytes ->
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().put(data)
    }

    fun toNchw(newName: String): TensorValue {
        val n = shape[0].toInt()
        val h = shape[1].toInt()
        val w = shape[2].toInt()
        val c = shape[3].toInt()
        val destination = FloatArray(data.size)
        for (batch in 0 until n) for (row in 0 until h) for (column in 0 until w) for (channel in 0 until c) {
            val nhwc = ((batch * h + row) * w + column) * c + channel
            val nchw = ((batch * c + channel) * h + row) * w + column
            destination[nchw] = data[nhwc]
        }
        return TensorValue(newName, longArrayOf(n.toLong(), c.toLong(), h.toLong(), w.toLong()), destination)
    }

    companion object {
        fun fromF32Le(name: String, shape: LongArray, bytes: ByteArray): NhwcFloatTensor {
            val count = shape.fold(1L) { product, value -> product * value }.toInt()
            require(bytes.size == count * 4) { "$name NHWC byte count mismatch" }
            val data = FloatArray(count)
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(data)
            return NhwcFloatTensor(name, shape, data)
        }

        fun fromNchw(tensor: TensorValue, newName: String = tensor.name): NhwcFloatTensor {
            val shape = tensor.shape
            require(shape.size == 4) { "$newName requires rank-4 NCHW input" }
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
            return NhwcFloatTensor(newName, longArrayOf(n.toLong(), h.toLong(), w.toLong(), c.toLong()), destination)
        }
    }
}

internal data class NhwcPriorStage(
    val symbols: TensorValue,
    val scales: TensorValue,
    val yHat: NhwcFloatTensor,
)

/** Exact I-prior mask and quantization while continuous tensors stay NHWC. */
internal object NhwcEntropyPriorQuantizer {
    fun quantize(
        yScaled: NhwcFloatTensor,
        means: NhwcFloatTensor,
        scales: NhwcFloatTensor,
        phase: Int,
        groups: Int,
        forceZeroThreshold: Float,
    ): NhwcPriorStage {
        require(yScaled.shape.contentEquals(means.shape) && yScaled.shape.contentEquals(scales.shape)) {
            "prior y/means/scales NHWC shapes must match"
        }
        val shape = yScaled.shape
        require(shape[0] == 1L && shape[3] % groups == 0L && phase in 0 until groups)
        val height = shape[1].toInt()
        val width = shape[2].toInt()
        val channels = shape[3].toInt()
        val packedChannels = channels / groups
        val plane = height * width
        val symbols = FloatArray(packedChannels * plane)
        val packedScales = FloatArray(symbols.size)
        val yHat = FloatArray(yScaled.data.size)

        for (row in 0 until height) for (column in 0 until width) {
            val activeGroup = activeGroup(groups, phase, row, column)
            val spatial = row * width + column
            val channelBase = activeGroup * packedChannels
            for (packedChannel in 0 until packedChannels) {
                val channel = channelBase + packedChannel
                val fullIndex = spatial * channels + channel
                val packedIndex = packedChannel * plane + spatial
                val scale = scales.data[fullIndex]
                val symbol = if (scale <= forceZeroThreshold) {
                    0f
                } else {
                    rint((yScaled.data[fullIndex] - means.data[fullIndex]).toDouble())
                        .toInt().coerceIn(-128, 127).toFloat()
                }
                symbols[packedIndex] = symbol
                packedScales[packedIndex] = scale
                yHat[fullIndex] = symbol + means.data[fullIndex]
            }
        }
        val packedShape = longArrayOf(1, packedChannels.toLong(), height.toLong(), width.toLong())
        return NhwcPriorStage(
            TensorValue("y_q_w_$phase", packedShape, symbols),
            TensorValue("s_w_$phase", packedShape, packedScales),
            NhwcFloatTensor("y_hat_$phase", shape, yHat),
        )
    }

    fun multiply(left: NhwcFloatTensor, right: NhwcFloatTensor, name: String): NhwcFloatTensor {
        require(left.shape[0] == right.shape[0] && left.shape[1] == right.shape[1] && left.shape[2] == right.shape[2])
        val channels = left.shape[3].toInt()
        val rightChannels = right.shape[3].toInt()
        require(rightChannels == 1 || rightChannels == channels) { "$name invalid NHWC broadcast" }
        return NhwcFloatTensor(name, left.shape, FloatArray(left.data.size) { index ->
            val channel = index % channels
            val rightIndex = if (rightChannels == 1) index / channels else index
            left.data[index] * right.data[if (rightChannels == 1) rightIndex else index]
        })
    }

    fun add(left: NhwcFloatTensor, right: NhwcFloatTensor, name: String): NhwcFloatTensor {
        require(left.shape.contentEquals(right.shape)) { "$name NHWC shape mismatch" }
        return NhwcFloatTensor(name, left.shape, FloatArray(left.data.size) { left.data[it] + right.data[it] })
    }

    fun quantizeInt8(input: NhwcFloatTensor, nhwcName: String, nchwName: String): Pair<NhwcFloatTensor, TensorValue> {
        val quantized = FloatArray(input.data.size) { index ->
            rint(input.data[index].toDouble()).toInt().coerceIn(-128, 127).toFloat()
        }
        val nhwc = NhwcFloatTensor(nhwcName, input.shape, quantized)
        return nhwc to nhwc.toNchw(nchwName)
    }

    private fun activeGroup(groups: Int, phase: Int, row: Int, column: Int): Int =
        if (groups == 4) FOUR_PHASES[phase][((row and 1) shl 1) or (column and 1)]
        else TWO_PHASES[phase][(row + column) and 1]

    private val FOUR_PHASES = arrayOf(
        intArrayOf(0, 1, 2, 3),
        intArrayOf(3, 2, 1, 0),
        intArrayOf(2, 3, 0, 1),
        intArrayOf(1, 0, 3, 2),
    )
    private val TWO_PHASES = arrayOf(intArrayOf(0, 1), intArrayOf(1, 0))
}
