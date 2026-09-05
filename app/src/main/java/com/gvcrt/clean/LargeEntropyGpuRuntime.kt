package com.gvcrt.clean

import android.util.Log
import java.io.File

/** Official TFLite GPU runtime with only the existing Large native rANS ops registered. */
internal class LargeEntropyGpuRuntime private constructor(
    private val modelName: String,
    private val kind: Kind,
    private var nativeHandle: Long,
    private val guard: GpuDelegationGuard,
    val optionsSummary: String,
) : AutoCloseable {
    private val ownerThread = Thread.currentThread()
    private var invokeLogged = false

    fun run(inputs: List<ByteArray>, copyOutputs: Boolean = true, qp: Int = 0): List<ByteArray> =
        invoke(inputs, qp, if (copyOutputs) OUTPUT_ALL else OUTPUT_NONE)

    fun runCanonical(inputs: List<ByteArray>, qp: Int = 0): List<ByteArray> =
        invoke(inputs, qp, OUTPUT_CANONICAL)

    private fun invoke(inputs: List<ByteArray>, qp: Int, outputMode: Int): List<ByteArray> {
        check(Thread.currentThread() === ownerThread) {
            "Large entropy GPU runtime must run on its creation thread"
        }
        check(nativeHandle != 0L) { "Large entropy GPU runtime is closed" }
        require(inputs.size == kind.logicalInputCount) {
            "Large entropy GPU input count mismatch expected=${kind.logicalInputCount} actual=${inputs.size}"
        }
        val outputs = try {
            if (kind.isEncode) {
                nativeRunEncode(nativeHandle, inputs.toTypedArray(), qp, outputMode)
            } else {
                nativeRunDecode(nativeHandle, inputs.toTypedArray(), qp, outputMode)
            }.toList()
        } catch (error: Throwable) {
            throw IllegalStateException(
                "gpu_model_invoke_failed stage=invoke model=$modelName backend=tflite_gpu " +
                    "native_rans=true error=${error.message}",
                error,
            )
        }
        if (!invokeLogged) {
            Log.i(
                "GVC_RT_CLEAN",
                "gpu_invoke_ok model=$modelName native_rans=true precision_validated=false",
            )
            invokeLogged = true
        }
        return outputs
    }

    override fun close() {
        check(Thread.currentThread() === ownerThread) {
            "Large entropy GPU runtime must close on its creation thread"
        }
        try {
            if (nativeHandle != 0L) {
                if (kind.isEncode) nativeCloseEncode(nativeHandle) else nativeCloseDecode(nativeHandle)
                nativeHandle = 0L
            }
        } finally {
            guard.close()
        }
    }

    enum class Kind(
        val nativeKind: Int,
        val isEncode: Boolean,
        val logicalInputCount: Int,
        val expectedModelName: String,
    ) {
        I_ENCODE(0, true, 1, "i_entropy_prior_merged_rans.tflite"),
        P_ENCODE(1, true, 2, "p_entropy_prior_merged_rans.tflite"),
        I_DECODE(0, false, 1, "i_entropy_decode_merged_rans.tflite"),
        P_DECODE(1, false, 2, "p_entropy_decode_merged_rans.tflite"),
    }

    companion object {
        init {
            System.loadLibrary("gvcrt_large_gpu_entropy_encode")
            System.loadLibrary("gvcrt_large_gpu_entropy_decode")
        }

        fun create(
            model: File,
            kind: Kind,
        ): LargeEntropyGpuRuntime {
            require(model.isFile) { "missing Large GPU entropy model: ${model.absolutePath}" }
            require(model.name == kind.expectedModelName) {
                "Large GPU entropy model name=${model.name} expected=${kind.expectedModelName}"
            }
            var stage = "guard_create"
            var guard: GpuDelegationGuard? = null
            var nativeHandle = 0L
            try {
                val coverage = GpuDelegationGuard(
                    0L,
                    model.name,
                    allowBuiltinCpuFallback = true,
                )
                guard = coverage
                stage = "interpreter_create"
                nativeHandle = if (kind.isEncode) {
                    nativeCreateEncode(
                        model.absolutePath,
                        kind.nativeKind,
                        coverage.nativeHandle,
                        0L,
                    )
                } else {
                    nativeCreateDecode(
                        model.absolutePath,
                        kind.nativeKind,
                        coverage.nativeHandle,
                        0L,
                    )
                }
                check(nativeHandle != 0L) { "native Large entropy GPU interpreter handle is null" }
                Log.i(
                    "GVC_RT_CLEAN",
                    "large_entropy_interpreter_create_ok model=${model.name} " +
                        "backend=tflite_cpu_fallback native_rans=true",
                )
                stage = "delegation_check"
                coverage.requireFullyDelegated()
                Log.i(
                    "GVC_RT_CLEAN",
                    "large_entropy_execution_check_ok model=${model.name} gpu_nodes=0 " +
                        "cpu_builtin=true native_rans=true",
                )
                return LargeEntropyGpuRuntime(
                    modelName = model.name,
                    kind = kind,
                    nativeHandle = nativeHandle,
                    guard = coverage,
                    optionsSummary =
                        "TFLiteCpuFallback(native_rans=true,xnnpack=false,nnapi=false," +
                            "reason=entropy_encode_decode_cdf_consistency)",
                )
            } catch (error: Throwable) {
                if (nativeHandle != 0L) {
                    if (kind.isEncode) nativeCloseEncode(nativeHandle) else nativeCloseDecode(nativeHandle)
                }
                guard?.close()
                if (error.message?.startsWith("gpu_delegate_unsupported") == true) throw error
                throw IllegalStateException(
                    "gpu_model_create_failed stage=$stage model=${model.name} " +
                        "backend=tflite_gpu_entropy_cpu_fallback error=${error.message}",
                    error,
                )
            }
        }

        @JvmStatic private external fun nativeCreateEncode(
            modelPath: String,
            kind: Int,
            gpuDelegate: Long,
            guardDelegate: Long,
        ): Long

        @JvmStatic private external fun nativeRunEncode(
            handle: Long,
            inputs: Array<ByteArray>,
            qp: Int,
            outputMode: Int,
        ): Array<ByteArray>

        @JvmStatic private external fun nativeCloseEncode(handle: Long)

        @JvmStatic private external fun nativeCreateDecode(
            modelPath: String,
            kind: Int,
            gpuDelegate: Long,
            guardDelegate: Long,
        ): Long

        @JvmStatic private external fun nativeRunDecode(
            handle: Long,
            inputs: Array<ByteArray>,
            qp: Int,
            outputMode: Int,
        ): Array<ByteArray>

        @JvmStatic private external fun nativeCloseDecode(handle: Long)

        private const val OUTPUT_NONE = 0
        private const val OUTPUT_ALL = 1
        private const val OUTPUT_CANONICAL = 2
    }
}

internal class SelectedIEntropyEncoder private constructor(
    private val mtk: IEntropyRansMergedRuntime?,
    private val gpu: LargeEntropyGpuRuntime?,
) : AutoCloseable {
    fun runCanonical(input: ByteArray, copyOutputs: Boolean = true, qp: Int = 0): List<ByteArray> =
        mtk?.runCanonical(input, copyOutputs, qp)
            ?: if (copyOutputs) gpu!!.runCanonical(listOf(input), qp) else gpu!!.run(listOf(input), false, qp)

    override fun close() { mtk?.close() ?: gpu!!.close() }

    companion object {
        fun mtk(runtime: IEntropyRansMergedRuntime) = SelectedIEntropyEncoder(runtime, null)
        fun gpu(runtime: LargeEntropyGpuRuntime) = SelectedIEntropyEncoder(null, runtime)
    }
}

internal class SelectedPEntropyEncoder private constructor(
    private val mtk: PEntropyRansMergedRuntime?,
    private val gpu: LargeEntropyGpuRuntime?,
) : AutoCloseable {
    fun run(y: ByteArray, ctxT: ByteArray, copyOutputs: Boolean = true, qp: Int = 0): List<ByteArray> =
        mtk?.run(y, ctxT, copyOutputs, qp) ?: gpu!!.run(listOf(y, ctxT), copyOutputs, qp)

    fun runCanonical(y: ByteArray, ctxT: ByteArray, copyOutputs: Boolean = true, qp: Int = 0): List<ByteArray> =
        mtk?.runCanonical(y, ctxT, copyOutputs, qp)
            ?: if (copyOutputs) gpu!!.runCanonical(listOf(y, ctxT), qp) else gpu!!.run(listOf(y, ctxT), false, qp)

    override fun close() { mtk?.close() ?: gpu!!.close() }

    companion object {
        fun mtk(runtime: PEntropyRansMergedRuntime) = SelectedPEntropyEncoder(runtime, null)
        fun gpu(runtime: LargeEntropyGpuRuntime) = SelectedPEntropyEncoder(null, runtime)
    }
}

internal class SelectedIEntropyDecoder private constructor(
    private val mtk: IEntropyRansDecodeMergedRuntime?,
    private val gpu: LargeEntropyGpuRuntime?,
) : AutoCloseable {
    fun runCanonical(payload: ByteArray, qp: Int = 0): ByteArray =
        mtk?.runCanonical(payload, qp) ?: gpu!!.runCanonical(listOf(payload), qp).single()

    override fun close() { mtk?.close() ?: gpu!!.close() }

    companion object {
        fun mtk(runtime: IEntropyRansDecodeMergedRuntime) = SelectedIEntropyDecoder(runtime, null)
        fun gpu(runtime: LargeEntropyGpuRuntime) = SelectedIEntropyDecoder(null, runtime)
    }
}

internal class SelectedPEntropyDecoder private constructor(
    private val mtk: PEntropyRansDecodeMergedRuntime?,
    private val gpu: LargeEntropyGpuRuntime?,
) : AutoCloseable {
    fun run(payload: ByteArray, ctxT: ByteArray, copyOutputs: Boolean = true, qp: Int = 0): List<ByteArray> =
        mtk?.run(payload, ctxT, copyOutputs, qp) ?: gpu!!.run(listOf(payload, ctxT), copyOutputs, qp)

    fun runCanonical(payload: ByteArray, ctxT: ByteArray, qp: Int = 0): ByteArray =
        mtk?.runCanonical(payload, ctxT, qp)
            ?: gpu!!.runCanonical(listOf(payload, ctxT), qp).single()

    override fun close() { mtk?.close() ?: gpu!!.close() }

    companion object {
        fun mtk(runtime: PEntropyRansDecodeMergedRuntime) = SelectedPEntropyDecoder(runtime, null)
        fun gpu(runtime: LargeEntropyGpuRuntime) = SelectedPEntropyDecoder(null, runtime)
    }
}
