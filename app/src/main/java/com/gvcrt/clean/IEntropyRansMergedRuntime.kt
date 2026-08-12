package com.gvcrt.clean

import com.mediatek.neuropilot_V.neuron.NeuronDelegate
import java.io.File

/** Runs i_entropy_prior_merged with the terminal CPU rANS custom op registered. */
class IEntropyRansMergedRuntime private constructor(
    private var nativeHandle: Long,
    private val delegate: NeuronDelegate,
    val optionsSummary: String,
) : AutoCloseable {
    @Synchronized
    fun run(input: ByteArray, copyOutputs: Boolean = true): List<ByteArray> {
        check(nativeHandle != 0L) { "merged rANS runtime is closed" }
        return nativeRun(nativeHandle, input, if (copyOutputs) OUTPUT_ALL else OUTPUT_NONE).toList()
    }

    /** Returns only i_y_hat and the trimmed rANS payload for the canonical codec path. */
    @Synchronized
    fun runCanonical(input: ByteArray, copyOutputs: Boolean = true): List<ByteArray> {
        check(nativeHandle != 0L) { "merged rANS runtime is closed" }
        return nativeRun(nativeHandle, input, if (copyOutputs) OUTPUT_CANONICAL else OUTPUT_NONE).toList()
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

        fun create(model: File, cacheDir: File): IEntropyRansMergedRuntime {
            require(model.isFile) { "missing merged rANS model: ${model.absolutePath}" }
            cacheDir.mkdirs()
            val options = NeuronDelegate.Options()
                .setAllowFp16(true)
                .setCompileOptions("--relax-fp32")
                .setAcceleratorName("mtk-neuron")
                .setExecutionPreference(NeuronDelegate.Options.EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER)
                .setMaxNumberOfDelegatedPartitions(8)
                .setCacheDir(cacheDir.absolutePath)
                .setModelToken("i_entropy_prior_merged_rans")
            val delegate = NeuronDelegate(options)
            return try {
                val handle = nativeCreate(model.absolutePath, delegate.nativeHandle)
                require(handle != 0L) { "failed to create merged rANS native runtime" }
                IEntropyRansMergedRuntime(
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
        private external fun nativeRun(handle: Long, input: ByteArray, outputMode: Int): Array<ByteArray>

        @JvmStatic
        private external fun nativeClose(handle: Long)

        private const val OUTPUT_NONE = 0
        private const val OUTPUT_ALL = 1
        private const val OUTPUT_CANONICAL = 2
    }
}
