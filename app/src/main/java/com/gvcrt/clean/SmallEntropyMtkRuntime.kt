package com.gvcrt.clean

import com.mediatek.neuropilot_V.neuron.NeuronDelegate
import java.io.File

/** Isolated MTK diagnostic runtime for the existing Small fused entropy graphs. */
class SmallEntropyMtkRuntime private constructor(
    private var nativeHandle: Long,
    private val delegate: NeuronDelegate,
    val inputSizes: LongArray,
    val outputSizes: LongArray,
    val optionsSummary: String,
) : AutoCloseable {
    private val ownerThread = Thread.currentThread()

    fun run(inputs: List<ByteArray>): List<ByteArray> {
        check(Thread.currentThread() === ownerThread) {
            "Small entropy MTK runtime must run on its creation thread"
        }
        check(nativeHandle != 0L) { "Small entropy MTK runtime is closed" }
        require(inputs.size == inputSizes.size) {
            "input count mismatch runtime=${inputSizes.size} actual=${inputs.size}"
        }
        inputs.forEachIndexed { index, bytes ->
            require(bytes.size.toLong() == inputSizes[index]) {
                "input[$index] bytes mismatch runtime=${inputSizes[index]} actual=${bytes.size}"
            }
        }
        return nativeRun(nativeHandle, inputs.toTypedArray()).toList().also { outputs ->
            require(outputs.size == outputSizes.size)
            outputs.forEachIndexed { index, bytes ->
                require(bytes.size.toLong() == outputSizes[index]) {
                    "output[$index] bytes mismatch runtime=${outputSizes[index]} actual=${bytes.size}"
                }
            }
        }
    }

    override fun close() {
        check(Thread.currentThread() === ownerThread) {
            "Small entropy MTK runtime must close on its creation thread"
        }
        val handle = nativeHandle
        nativeHandle = 0L
        if (handle != 0L) nativeClose(handle)
        delegate.close()
    }

    companion object {
        init {
            System.loadLibrary("gvcrt_small_gpu_rans")
        }

        fun create(
            model: File,
            kind: SmallEntropyGpuRuntime.Kind,
            cacheDir: File,
        ): SmallEntropyMtkRuntime {
            require(model.isFile) { "missing Small fused MTK model: ${model.absolutePath}" }
            cacheDir.mkdirs()
            val options = NeuronDelegate.Options()
                .setAllowFp16(true)
                .setCompileOptions("--relax-fp32")
                .setAcceleratorName("mtk-neuron")
                .setExecutionPreference(NeuronDelegate.Options.EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER)
                .setMaxNumberOfDelegatedPartitions(8)
                .setCacheDir(cacheDir.absolutePath)
                .setModelToken("small_entropy_${kind.name.lowercase()}_qp9")
            val delegate = NeuronDelegate(options)
            return try {
                val handle = nativeCreate(model.absolutePath, kind.nativeValue, delegate.nativeHandle)
                require(handle != 0L) { "failed to create Small entropy MTK runtime" }
                SmallEntropyMtkRuntime(
                    nativeHandle = handle,
                    delegate = delegate,
                    inputSizes = nativeInputSizes(handle),
                    outputSizes = nativeOutputSizes(handle),
                    optionsSummary =
                        "NeuronDelegate(allowFp16=true,preference=FAST_SINGLE_ANSWER," +
                            "accelerator=mtk-neuron,compileOptions=--relax-fp32,maxPartitions=8)",
                )
            } catch (error: Throwable) {
                delegate.close()
                throw error
            }
        }

        @JvmStatic private external fun nativeCreate(
            modelPath: String,
            kind: Int,
            delegateHandle: Long,
        ): Long

        @JvmStatic private external fun nativeInputSizes(handle: Long): LongArray
        @JvmStatic private external fun nativeOutputSizes(handle: Long): LongArray
        @JvmStatic private external fun nativeRun(handle: Long, inputs: Array<ByteArray>): Array<ByteArray>
        @JvmStatic private external fun nativeClose(handle: Long)
    }
}
