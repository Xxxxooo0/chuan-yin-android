package com.gvcrt.clean

import com.mediatek.neuropilot_V.neuron.NeuronDelegate
import java.io.File

/** Runs the merged P entropy encoder with its terminal native rANS custom op. */
class PEntropyRansMergedRuntime private constructor(
    private var nativeHandle: Long,
    private val delegate: NeuronDelegate,
    val optionsSummary: String,
) : AutoCloseable {
    @Synchronized
    fun run(y: ByteArray, ctxT: ByteArray, copyOutputs: Boolean = true, qp: Int = 0): List<ByteArray> {
        check(nativeHandle != 0L) { "P merged rANS runtime is closed" }
        return nativeRun(nativeHandle, y, ctxT, qp, if (copyOutputs) OUTPUT_ALL else OUTPUT_NONE).toList()
    }

    /** Returns P y_hat and the trimmed rANS payload. */
    @Synchronized
    fun runCanonical(y: ByteArray, ctxT: ByteArray, copyOutputs: Boolean = true, qp: Int = 0): List<ByteArray> {
        check(nativeHandle != 0L) { "P merged rANS runtime is closed" }
        return nativeRun(nativeHandle, y, ctxT, qp, if (copyOutputs) OUTPUT_CANONICAL else OUTPUT_NONE).toList()
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

        fun create(model: File, cacheDir: File): PEntropyRansMergedRuntime {
            require(model.isFile) { "missing P merged rANS model: ${model.absolutePath}" }
            cacheDir.mkdirs()
            val options = NeuronDelegate.Options()
                .setAllowFp16(true)
                .setCompileOptions("--relax-fp32")
                .setAcceleratorName("mtk-neuron")
                .setExecutionPreference(NeuronDelegate.Options.EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER)
                .setMaxNumberOfDelegatedPartitions(8)
                .setCacheDir(cacheDir.absolutePath)
                .setModelToken("p_entropy_prior_merged_rans")
            val delegate = NeuronDelegate(options)
            return try {
                val handle = nativeCreate(model.absolutePath, delegate.nativeHandle)
                require(handle != 0L) { "failed to create P merged rANS runtime" }
                PEntropyRansMergedRuntime(
                    nativeHandle = handle,
                    delegate = delegate,
                    optionsSummary =
                        "NeuronDelegate(allowFp16=true,preference=FAST_SINGLE_ANSWER," +
                            "accelerator=mtk-neuron,compileOptions=--relax-fp32,maxPartitions=8)",
                )
            } catch (error: Throwable) {
                delegate.close()
                throw error
            }
        }

        @JvmStatic
        private external fun nativeCreate(modelPath: String, delegateHandle: Long): Long

        @JvmStatic
        private external fun nativeRun(
            handle: Long,
            y: ByteArray,
            ctxT: ByteArray,
            qp: Int,
            outputMode: Int,
        ): Array<ByteArray>

        @JvmStatic
        private external fun nativeClose(handle: Long)

        private const val OUTPUT_NONE = 0
        private const val OUTPUT_ALL = 1
        private const val OUTPUT_CANONICAL = 2
    }
}
