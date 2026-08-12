package com.gvcrt.clean

import com.mediatek.neuropilot_V.neuron.NeuronDelegate
import java.io.File

/** Runs one I entropy-decoder graph with serial native rANS decode custom ops. */
class IEntropyRansDecodeMergedRuntime private constructor(
    private var nativeHandle: Long,
    private val delegate: NeuronDelegate,
    val optionsSummary: String,
) : AutoCloseable {
    @Synchronized
    fun run(payload: ByteArray, copyOutputs: Boolean = true, qp: Int = 0): List<ByteArray> {
        check(nativeHandle != 0L) { "merged entropy decoder is closed" }
        return nativeRun(nativeHandle, payload, qp, if (copyOutputs) OUTPUT_ALL else OUTPUT_NONE).toList()
    }

    /** Returns only i_y_hat while still executing the full serial entropy decode. */
    @Synchronized
    fun runCanonical(payload: ByteArray, qp: Int = 0): ByteArray {
        check(nativeHandle != 0L) { "merged entropy decoder is closed" }
        return nativeRun(nativeHandle, payload, qp, OUTPUT_CANONICAL).single()
    }

    override fun close() {
        val handle = nativeHandle
        nativeHandle = 0L
        if (handle != 0L) nativeClose(handle)
        delegate.close()
    }

    companion object {
        init {
            System.loadLibrary("gvcrt_clean_rans")
        }

        fun create(model: File, cacheDir: File, fastRelaxFp32: Boolean): IEntropyRansDecodeMergedRuntime {
            require(model.isFile) { "missing merged entropy decoder: ${model.absolutePath}" }
            cacheDir.mkdirs()
            val options = NeuronDelegate.Options()
                .setAllowFp16(fastRelaxFp32)
                .setAcceleratorName("mtk-neuron")
                .setExecutionPreference(NeuronDelegate.Options.EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER)
                .setMaxNumberOfDelegatedPartitions(8)
                .setCacheDir(cacheDir.absolutePath)
                .setModelToken("i_entropy_decode_merged_rans_${if (fastRelaxFp32) "fast" else "strict"}")
            if (fastRelaxFp32) options.setCompileOptions("--relax-fp32")
            val delegate = NeuronDelegate(options)
            return try {
                val handle = nativeCreate(model.absolutePath, delegate.nativeHandle)
                require(handle != 0L) { "failed to create merged entropy decoder runtime" }
                IEntropyRansDecodeMergedRuntime(
                    nativeHandle = handle,
                    delegate = delegate,
                    optionsSummary =
                        "NeuronDelegate(allowFp16=$fastRelaxFp32,preference=FAST_SINGLE_ANSWER," +
                            "accelerator=mtk-neuron,compileOptions=${if (fastRelaxFp32) "--relax-fp32" else "none"}," +
                            "maxPartitions=8)",
                )
            } catch (error: Throwable) {
                delegate.close()
                throw error
            }
        }

        @JvmStatic
        private external fun nativeCreate(modelPath: String, delegateHandle: Long): Long

        @JvmStatic
        private external fun nativeRun(handle: Long, payload: ByteArray, qp: Int, outputMode: Int): Array<ByteArray>

        @JvmStatic
        private external fun nativeClose(handle: Long)

        private const val OUTPUT_NONE = 0
        private const val OUTPUT_ALL = 1
        private const val OUTPUT_CANONICAL = 2
    }
}
