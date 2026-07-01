package com.gvcrt.clean

class RansNativeEncoder private constructor(
    private var handle: Long,
    private val gaussianGroup: Int,
    private val zGroup: Int,
) : AutoCloseable {
    fun setUseTwoEncoders(enabled: Boolean) {
        nativeSetUseTwoEncoders(requireHandle(), enabled)
    }

    fun encodeZ(symbols: ByteArray, startOffset: Int, perChannelSize: Int) {
        nativeEncodeZ(requireHandle(), symbols, zGroup, startOffset, perChannelSize)
    }

    fun encodeY(symbols: ShortArray) {
        nativeEncodeY(requireHandle(), symbols, gaussianGroup)
    }

    fun encodeY(symbols: TensorValue, scales: TensorValue) {
        require(symbols.data.size == scales.data.size) {
            "${symbols.name}/${scales.name} size mismatch"
        }
        nativeEncodeYFromFloat(requireHandle(), symbols.data, scales.data, gaussianGroup)
    }

    fun flush(): ByteArray = nativeFlush(requireHandle())

    override fun close() {
        val current = handle
        if (current != 0L) {
            nativeDestroyEncoder(current)
            handle = 0L
        }
    }

    private fun requireHandle(): Long {
        check(handle != 0L) { "RansNativeEncoder has already been closed" }
        return handle
    }

    companion object {
        init {
            System.loadLibrary("gvcrt_clean_rans")
        }

        fun create(gaussian: CdfTable, z: CdfTable, useTwoEncoders: Boolean): RansNativeEncoder {
            val handle = nativeCreateEncoder()
            val gaussianGroup = nativeAddCdf(
                handle,
                gaussian.values,
                gaussian.rows,
                gaussian.stride,
                gaussian.lengths,
                gaussian.offsets,
            )
            val zGroup = nativeAddCdf(
                handle,
                z.values,
                z.rows,
                z.stride,
                z.lengths,
                z.offsets,
            )
            return RansNativeEncoder(handle, gaussianGroup, zGroup).also {
                it.setUseTwoEncoders(useTwoEncoders)
            }
        }

        @JvmStatic
        private external fun nativeCreateEncoder(): Long

        @JvmStatic
        private external fun nativeDestroyEncoder(handle: Long)

        @JvmStatic
        private external fun nativeSetUseTwoEncoders(handle: Long, enabled: Boolean)

        @JvmStatic
        private external fun nativeAddCdf(
            handle: Long,
            flatCdfs: IntArray,
            rows: Int,
            cols: Int,
            cdfSizes: IntArray,
            offsets: IntArray,
        ): Int

        @JvmStatic
        private external fun nativeEncodeY(handle: Long, symbols: ShortArray, cdfGroupIndex: Int)

        @JvmStatic
        private external fun nativeEncodeYFromFloat(
            handle: Long,
            symbols: FloatArray,
            scales: FloatArray,
            cdfGroupIndex: Int,
        )

        @JvmStatic
        private external fun nativeEncodeZ(
            handle: Long,
            symbols: ByteArray,
            cdfGroupIndex: Int,
            startOffset: Int,
            perChannelSize: Int,
        )

        @JvmStatic
        private external fun nativeFlush(handle: Long): ByteArray
    }
}
