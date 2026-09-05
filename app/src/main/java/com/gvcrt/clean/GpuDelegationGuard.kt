package com.gvcrt.clean

import org.tensorflow.lite.Delegate

/** Requires GPU NN nodes; only exact, model-scoped native rANS registrations may stay on CPU. */
internal class GpuDelegationGuard(
    gpuHandle: Long,
    private val modelName: String,
    allowBuiltinCpuFallback: Boolean = false,
) : Delegate {
    private var handle = nativeCreate(gpuHandle, modelName, allowBuiltinCpuFallback)

    override fun getNativeHandle(): Long = nativeDelegate(handle)

    fun requireFullyDelegated() {
        val failure = nativeFailure(handle)
        check(failure.isEmpty()) {
            "gpu_delegate_unsupported stage=delegation_check model=$modelName backend=tflite_gpu $failure"
        }
    }

    override fun close() {
        if (handle != 0L) {
            nativeClose(handle)
            handle = 0L
        }
    }

    companion object {
        init {
            System.loadLibrary("gvcrt_gpu_guard")
        }

        @JvmStatic private external fun nativeCreate(
            gpuHandle: Long,
            modelName: String,
            allowBuiltinCpuFallback: Boolean,
        ): Long
        @JvmStatic private external fun nativeDelegate(handle: Long): Long
        @JvmStatic private external fun nativeFailure(handle: Long): String
        @JvmStatic private external fun nativeClose(handle: Long)
    }
}
