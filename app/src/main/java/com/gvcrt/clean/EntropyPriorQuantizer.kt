package com.gvcrt.clean

import java.lang.Math.rint

internal data class EntropyPriorStage(
    val symbols: TensorValue,
    val scales: TensorValue,
    val yHat: TensorValue,
)

/** Source-equivalent masked quantization; only the continuous prior runs in TFLite. */
internal object EntropyPriorQuantizer {
    fun restoreDecoded(
        decodedSymbols: ByteArray,
        packedMeans: TensorValue,
        packedScales: TensorValue,
        phase: Int,
        groups: Int,
        outputName: String,
    ): EntropyPriorStage {
        require(packedMeans.shape.contentEquals(packedScales.shape)) {
            "decoded prior means/scales shapes must match"
        }
        require(decodedSymbols.size == packedScales.numel) {
            "decoded symbol count=${decodedSymbols.size}, expected=${packedScales.numel}"
        }
        val packedShape = packedScales.shape
        require(packedShape.size == 4 && packedShape[0] == 1L && phase in 0 until groups) {
            "unsupported decoded prior tensor ${packedShape.contentToString()} phase=$phase groups=$groups"
        }
        val packedChannels = packedShape[1].toInt()
        val height = packedShape[2].toInt()
        val width = packedShape[3].toInt()
        val channels = packedChannels * groups
        val plane = height * width
        val symbolValues = FloatArray(decodedSymbols.size) { decodedSymbols[it].toFloat() }
        val yHat = FloatArray(channels * plane)

        for (row in 0 until height) for (column in 0 until width) {
            val activeGroup = activeGroup(groups, phase, row, column)
            val spatial = row * width + column
            val channelBase = activeGroup * packedChannels
            for (packedChannel in 0 until packedChannels) {
                val packedIndex = packedChannel * plane + spatial
                val fullIndex = (channelBase + packedChannel) * plane + spatial
                yHat[fullIndex] = symbolValues[packedIndex] + packedMeans.data[packedIndex]
            }
        }
        val fullShape = longArrayOf(1, channels.toLong(), height.toLong(), width.toLong())
        return EntropyPriorStage(
            TensorValue("${outputName}_symbols", packedShape, symbolValues),
            packedScales.renamed("${outputName}_scales"),
            TensorValue(outputName, fullShape, yHat),
        )
    }

    fun quantize(
        yScaled: TensorValue,
        means: TensorValue,
        scales: TensorValue,
        phase: Int,
        groups: Int,
        forceZeroThreshold: Float,
    ): EntropyPriorStage {
        require(yScaled.shape.contentEquals(means.shape) && yScaled.shape.contentEquals(scales.shape)) {
            "prior y/means/scales shapes must match"
        }
        val shape = yScaled.shape
        require(shape.size == 4 && shape[0] == 1L && shape[1] % groups == 0L) {
            "unsupported prior tensor ${shape.contentToString()} groups=$groups"
        }
        require(phase in 0 until groups) { "invalid phase=$phase for groups=$groups" }
        val channels = shape[1].toInt()
        val height = shape[2].toInt()
        val width = shape[3].toInt()
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
                val fullIndex = channel * plane + spatial
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
        return EntropyPriorStage(
            TensorValue("y_q_w_$phase", packedShape, symbols),
            TensorValue("s_w_$phase", packedShape, packedScales),
            TensorValue("y_hat_$phase", shape, yHat),
        )
    }

    fun multiply(left: TensorValue, right: TensorValue, name: String): TensorValue {
        require(left.data.size == right.data.size || right.shape[1] == 1L) { "$name invalid broadcast" }
        val result = FloatArray(left.data.size)
        val channels = left.shape[1].toInt()
        val plane = (left.shape[2] * left.shape[3]).toInt()
        for (channel in 0 until channels) for (index in 0 until plane) {
            val tensorIndex = channel * plane + index
            val rightIndex = if (right.data.size == left.data.size) tensorIndex else index
            result[tensorIndex] = left.data[tensorIndex] * right.data[rightIndex]
        }
        return TensorValue(name, left.shape, result)
    }

    fun divide(left: TensorValue, right: TensorValue, name: String): TensorValue {
        require(left.data.size == right.data.size || right.shape[1] == 1L) { "$name invalid broadcast" }
        val result = FloatArray(left.data.size)
        val channels = left.shape[1].toInt()
        val plane = (left.shape[2] * left.shape[3]).toInt()
        for (channel in 0 until channels) for (index in 0 until plane) {
            val tensorIndex = channel * plane + index
            val rightIndex = if (right.data.size == left.data.size) tensorIndex else index
            result[tensorIndex] = left.data[tensorIndex] / right.data[rightIndex].coerceAtLeast(0.5f)
        }
        return TensorValue(name, left.shape, result)
    }

    fun add(left: TensorValue, right: TensorValue, name: String): TensorValue {
        require(left.shape.contentEquals(right.shape)) { "$name shape mismatch" }
        return TensorValue(name, left.shape, FloatArray(left.data.size) { left.data[it] + right.data[it] })
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
