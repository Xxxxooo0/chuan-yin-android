package com.gvcrt.clean

import android.util.Log
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.File

/** Official-TFLite-only runtime for Small fused entropy+rANS GPU execution. */
class SmallEntropyGpuRuntime private constructor(
    private val modelName: String,
    private var nativeHandle: Long,
    private val delegate: GpuDelegate,
    private val guard: GpuDelegationGuard,
    val inputSizes: LongArray,
    val outputSizes: LongArray,
    val optionsSummary: String,
) : AutoCloseable {
    private val ownerThread = Thread.currentThread()
    private var invokeLogged = false

    fun run(inputs: List<ByteArray>): List<ByteArray> {
        check(Thread.currentThread() === ownerThread) {
            "Small entropy GPU runtime must run on its creation thread"
        }
        check(nativeHandle != 0L) { "Small entropy GPU runtime is closed" }
        require(inputs.size == inputSizes.size) {
            "input count mismatch runtime=${inputSizes.size} actual=${inputs.size}"
        }
        inputs.forEachIndexed { index, bytes ->
            require(bytes.size.toLong() == inputSizes[index]) {
                "input[$index] bytes mismatch runtime=${inputSizes[index]} actual=${bytes.size}"
            }
        }
        val outputs = try {
            nativeRun(nativeHandle, inputs.toTypedArray()).toList()
        } catch (error: Throwable) {
            throw IllegalStateException(
                "gpu_model_invoke_failed stage=invoke model=$modelName backend=tflite_gpu error=${error.message}",
                error,
            )
        }
        require(outputs.size == outputSizes.size) {
            "output count mismatch runtime=${outputSizes.size} actual=${outputs.size}"
        }
        outputs.forEachIndexed { index, bytes ->
            check(bytes.size.toLong() == outputSizes[index]) {
                "output[$index] bytes mismatch runtime=${outputSizes[index]} actual=${bytes.size}"
            }
        }
        if (!invokeLogged) {
            Log.i("GVC_RT_CLEAN", "gpu_invoke_ok model=$modelName native_rans=true precision_validated=false")
            invokeLogged = true
        }
        return outputs
    }

    override fun close() {
        check(Thread.currentThread() === ownerThread) {
            "Small entropy GPU runtime must close on its creation thread"
        }
        try {
            if (nativeHandle != 0L) {
                nativeClose(nativeHandle)
                nativeHandle = 0L
            }
        } finally {
            guard.close()
            delegate.close()
        }
    }

    enum class Kind(val nativeValue: Int) {
        ENCODE(0),
        DECODE(1),
    }

    companion object {
        init {
            System.loadLibrary("gvcrt_small_gpu_rans")
        }

        fun create(tfliteFile: File, kind: Kind): SmallEntropyGpuRuntime {
            require(tfliteFile.isFile) {
                "gpu_model_create_failed model=${tfliteFile.absolutePath} backend=tflite_gpu reason=missing_file"
            }
            var compatibilitySupported = false
            var stage = "compatibility_list"
            val delegate = try {
                CompatibilityList().use { compatibility ->
                    compatibilitySupported = compatibility.isDelegateSupportedOnThisDevice
                    val forcedProbe = !compatibilitySupported
                    Log.i(
                        "GVC_RT_CLEAN",
                        "gpu_compatibility model=${tfliteFile.name} " +
                            "compatibility_list_supported=$compatibilitySupported gpu_forced_probe=$forcedProbe",
                    )
                    stage = "delegate_create"
                    val options = if (compatibilitySupported) {
                        compatibility.bestOptionsForThisDevice
                    } else {
                        GpuDelegate.Options()
                    }
                    GpuDelegate(
                        options
                            .setPrecisionLossAllowed(false)
                            .setQuantizedModelsAllowed(false),
                    ).also {
                        Log.i("GVC_RT_CLEAN", "gpu_delegate_create_ok model=${tfliteFile.name}")
                    }
                }
            } catch (error: Throwable) {
                throw IllegalStateException(
                    "gpu_delegate_create_failed stage=$stage model=${tfliteFile.name} " +
                        "backend=tflite_gpu error=${error.message}",
                    error,
                )
            }

            var guard: GpuDelegationGuard? = null
            var handle = 0L
            try {
                stage = "guard_create"
                val coverage = GpuDelegationGuard(
                    delegate.nativeHandle,
                    tfliteFile.name,
                    allowBuiltinCpuFallback = true,
                )
                guard = coverage
                stage = "interpreter_create"
                handle = nativeCreate(
                    tfliteFile.absolutePath,
                    kind.nativeValue,
                    delegate.nativeHandle,
                    coverage.nativeHandle,
                )
                check(handle != 0L) { "native Small entropy GPU interpreter handle is null" }
                Log.i("GVC_RT_CLEAN", "gpu_interpreter_create_ok model=${tfliteFile.name}")
                stage = "delegation_check"
                coverage.requireFullyDelegated()
                Log.i(
                    "GVC_RT_CLEAN",
                    "gpu_delegation_check_ok model=${tfliteFile.name} " +
                        "cpu_fallback_allowed=true native_rans=true",
                )
                stage = "tensor_sizes"
                val inputSizes = nativeInputSizes(handle)
                val outputSizes = nativeOutputSizes(handle)
                val summary =
                    "TFLiteGpu(compatibility_list_supported=$compatibilitySupported," +
                        "gpu_forced_probe=${!compatibilitySupported},cpu_fallback_allowed=true," +
                        "native_rans=true,precision_loss_allowed=false,quantized=false," +
                        "xnnpack=false,nnapi=false,fallback=false)"
                Log.i(
                    "GVC_RT_CLEAN",
                    "gpu_model_created model=${tfliteFile.absolutePath} kind=${kind.name.lowercase()} " +
                        "inputs=${inputSizes.joinToString(",")} outputs=${outputSizes.joinToString(",")} " +
                        "cpu_fallback_allowed=true native_rans=true",
                )
                return SmallEntropyGpuRuntime(
                    modelName = tfliteFile.name,
                    nativeHandle = handle,
                    delegate = delegate,
                    guard = coverage,
                    inputSizes = inputSizes,
                    outputSizes = outputSizes,
                    optionsSummary = summary,
                )
            } catch (error: Throwable) {
                if (handle != 0L) nativeClose(handle)
                guard?.close()
                delegate.close()
                if (error.message?.startsWith("gpu_delegate_unsupported") == true) throw error
                throw IllegalStateException(
                    "gpu_model_create_failed stage=$stage model=${tfliteFile.name} " +
                        "backend=tflite_gpu error=${error.message}",
                    error,
                )
            }
        }

        @JvmStatic private external fun nativeCreate(
            modelPath: String,
            kind: Int,
            gpuDelegateHandle: Long,
            guardDelegateHandle: Long,
        ): Long

        @JvmStatic private external fun nativeInputSizes(handle: Long): LongArray
        @JvmStatic private external fun nativeOutputSizes(handle: Long): LongArray
        @JvmStatic private external fun nativeRun(handle: Long, inputs: Array<ByteArray>): Array<ByteArray>
        @JvmStatic private external fun nativeClose(handle: Long)
    }
}
