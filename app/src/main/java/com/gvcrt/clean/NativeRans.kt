package com.gvcrt.clean

import java.io.Closeable
import kotlin.math.ln

data class CdfTable(
    val values: IntArray,
    val rows: Int,
    val stride: Int,
    val lengths: IntArray,
    val offsets: IntArray,
) {
    companion object {
        fun load(store: AssetStore, spec: CdfSpec): CdfTable {
            require(spec.shape.size == 2) { "CDF must have [rows, stride] shape" }
            val rows = spec.shape[0].toInt()
            val stride = spec.shape[1].toInt()
            val values = TensorIO.readI32Le(spec.cdf, store.readBytes(spec.cdf))
            val lengths = TensorIO.readI32Le(spec.cdfLengths, store.readBytes(spec.cdfLengths))
            val offsets = TensorIO.readI32Le(spec.offsets, store.readBytes(spec.offsets))
            require(values.size == rows * stride) { "${spec.cdf} CDF element mismatch" }
            require(lengths.size == rows) { "${spec.cdfLengths} CDF length mismatch" }
            require(offsets.size == rows) { "${spec.offsets} CDF offset mismatch" }
            return CdfTable(values, rows, stride, lengths, offsets)
        }
    }
}

class NativeRans private constructor(private var handle: Long) : Closeable {
    fun encode(
        zSymbols: ByteArray,
        zStartOffset: Int,
        zPerChannelSize: Int,
        packedYStages: Array<ShortArray>,
    ): ByteArray {
        check(handle != 0L) { "rANS session is closed" }
        return nativeEncode(handle, zSymbols, zStartOffset, zPerChannelSize, packedYStages)
    }

    fun decode(
        payload: ByteArray,
        zTotalSize: Int,
        zStartOffset: Int,
        zPerChannelSize: Int,
        yIndexes: Array<ByteArray>,
    ): Array<ByteArray> {
        check(handle != 0L) { "rANS session is closed" }
        return nativeDecode(handle, payload, zTotalSize, zStartOffset, zPerChannelSize, yIndexes)
    }

    fun beginDecode(payload: ByteArray) {
        check(handle != 0L) { "rANS session is closed" }
        nativeBeginDecode(handle, payload)
    }

    fun decodeZ(zTotalSize: Int, zStartOffset: Int, zPerChannelSize: Int): ByteArray {
        check(handle != 0L) { "rANS session is closed" }
        return nativeDecodeZ(handle, zTotalSize, zStartOffset, zPerChannelSize)
    }

    fun decodeY(indexes: ByteArray): ByteArray {
        check(handle != 0L) { "rANS session is closed" }
        return nativeDecodeY(handle, indexes)
    }

    override fun close() {
        if (handle != 0L) {
            nativeRelease(handle)
            handle = 0L
        }
    }

    companion object {
        init {
            System.loadLibrary("gvcrt_clean_rans")
        }

        fun create(gaussian: CdfTable, z: CdfTable): NativeRans =
            NativeRans(
                nativeCreate(
                    gaussian.values,
                    gaussian.rows,
                    gaussian.stride,
                    gaussian.lengths,
                    gaussian.offsets,
                    z.values,
                    z.rows,
                    z.stride,
                    z.lengths,
                    z.offsets,
                ),
            )

        @JvmStatic
        private external fun nativeCreate(
            gaussianCdf: IntArray,
            gaussianRows: Int,
            gaussianStride: Int,
            gaussianLengths: IntArray,
            gaussianOffsets: IntArray,
            zCdf: IntArray,
            zRows: Int,
            zStride: Int,
            zLengths: IntArray,
            zOffsets: IntArray,
        ): Long

        @JvmStatic
        private external fun nativeEncode(
            handle: Long,
            zSymbols: ByteArray,
            zStartOffset: Int,
            zPerChannelSize: Int,
            packedYStages: Array<ShortArray>,
        ): ByteArray

        @JvmStatic
        private external fun nativeDecode(
            handle: Long,
            payload: ByteArray,
            zTotalSize: Int,
            zStartOffset: Int,
            zPerChannelSize: Int,
            yIndexes: Array<ByteArray>,
        ): Array<ByteArray>

        @JvmStatic
        private external fun nativeBeginDecode(handle: Long, payload: ByteArray)

        @JvmStatic
        private external fun nativeDecodeZ(
            handle: Long,
            zTotalSize: Int,
            zStartOffset: Int,
            zPerChannelSize: Int,
        ): ByteArray

        @JvmStatic
        private external fun nativeDecodeY(handle: Long, indexes: ByteArray): ByteArray

        @JvmStatic
        private external fun nativeRelease(handle: Long)
    }
}

object EntropySymbols {
    private const val SCALE_MIN = 0.11f
    private const val SCALE_MAX = 16.0f
    private const val LOG_SCALE_MIN = -2.2072749f
    private const val LOG_STEP_RECIP = 25.502707f

    fun zSymbols(tensor: TensorValue): ByteArray =
        ByteArray(tensor.data.size) { index ->
            tensor.data[index].toExactInt("${tensor.name}[$index]").toByte()
        }

    fun packY(symbols: TensorValue, scales: TensorValue): ShortArray {
        require(symbols.data.size == scales.data.size) { "${symbols.name}/${scales.name} size mismatch" }
        return ShortArray(symbols.data.size) { index ->
            val symbol = symbols.data[index].toExactInt("${symbols.name}[$index]")
            require(symbol in -128..127) { "${symbols.name}[$index] outside int8 range" }
            val cdfIndex = cdfIndex(scales.data[index])
            ((symbol shl 8) + cdfIndex).toShort()
        }
    }

    fun indexesForScales(scales: TensorValue): ByteArray =
        ByteArray(scales.data.size) { index -> cdfIndex(scales.data[index]).toByte() }

    fun indexes(packed: ShortArray): ByteArray =
        ByteArray(packed.size) { index -> (packed[index].toInt() and 0xff).toByte() }

    private fun cdfIndex(rawScale: Float): Int {
        val scale = rawScale.coerceIn(SCALE_MIN, SCALE_MAX)
        return ((ln(scale) - LOG_SCALE_MIN) * LOG_STEP_RECIP).toInt().coerceIn(0, 127)
    }

    private fun Float.toExactInt(label: String): Int {
        val rounded = toInt()
        require(this == rounded.toFloat()) { "$label is not an integer: $this" }
        return rounded
    }
}
