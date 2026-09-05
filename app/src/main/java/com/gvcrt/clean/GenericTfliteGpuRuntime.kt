package com.gvcrt.clean

import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class GenericTfliteGpuRuntime private constructor(
    private val modelName: String,
    private val interpreter: Interpreter,
    private val delegate: GpuDelegate,
    private val guard: GpuDelegationGuard,
    val inputSizes: LongArray,
    val outputSizes: LongArray,
    val optionsSummary: String,
) : AutoCloseable {
    private val ownerThread = Thread.currentThread()
    private val inputBuffers = inputSizes.mapIndexed { index, size ->
        allocateTensorBuffer("input[$index]", size)
    }
    private val outputBuffers = outputSizes.mapIndexed { index, size ->
        allocateTensorBuffer("output[$index]", size)
    }
    private val inputsArray = inputBuffers.toTypedArray()
    private val outputs = outputBuffers.mapIndexed { index, buffer -> index to buffer }.toMap()
    private var invokeLogged = false

    fun run(inputs: List<ByteArray>, copyOutputs: Boolean = true): List<ByteArray> {
        check(Thread.currentThread() === ownerThread) { "TFLite GPU must run on its creation thread" }
        require(inputs.size == inputSizes.size) {
            "input count mismatch runtime=${inputSizes.size} actual=${inputs.size}"
        }
        inputs.forEachIndexed { index, bytes ->
            require(bytes.size.toLong() == inputSizes[index]) {
                "input[$index] bytes mismatch runtime=${inputSizes[index]} actual=${bytes.size}"
            }
            inputBuffers[index].apply {
                clear()
                put(bytes)
                rewind()
            }
        }
        outputBuffers.forEach(ByteBuffer::clear)
        try {
            interpreter.runForMultipleInputsOutputs(inputsArray, outputs)
            outputSizes.forEachIndexed { index, size ->
                check(interpreter.getOutputTensor(index).numBytes().toLong() == size) {
                    "output[$index] byte size changed from $size"
                }
            }
            if (!invokeLogged) {
                Log.i("GVC_RT_CLEAN", "gpu_invoke_ok model=$modelName precision_validated=false")
                invokeLogged = true
            }
        } catch (error: Throwable) {
            throw IllegalStateException(
                "gpu_model_invoke_failed stage=invoke model=$modelName backend=tflite_gpu error=${error.message}",
                error,
            )
        }
        if (!copyOutputs) return emptyList()
        return outputBuffers.map { buffer ->
            val bytes = ByteArray(buffer.capacity())
            buffer.rewind()
            buffer.get(bytes)
            bytes
        }
    }

    override fun close() {
        check(Thread.currentThread() === ownerThread) { "TFLite GPU must close on its creation thread" }
        try {
            interpreter.close()
        } finally {
            guard.close()
            delegate.close()
        }
    }

    companion object {
        fun create(
            tfliteFile: File,
            allowUnsupportedDevice: Boolean = false,
            allowBuiltinCpuFallback: Boolean = false,
            allowPrecisionLoss: Boolean = true,
        ): GenericTfliteGpuRuntime {
            require(tfliteFile.isFile) {
                "gpu_model_create_failed model=${tfliteFile.absolutePath} backend=tflite_gpu reason=missing_file"
            }
            var compatibilitySupported = false
            var forcedProbe = false
            var stage = "compatibility_list"
            val delegate = try {
                CompatibilityList().use { compatibility ->
                    compatibilitySupported = compatibility.isDelegateSupportedOnThisDevice
                    forcedProbe = !compatibilitySupported && allowUnsupportedDevice
                    Log.i("GVC_RT_CLEAN", "gpu_compatibility model=${tfliteFile.name} compatibility_list_supported=$compatibilitySupported gpu_forced_probe=$forcedProbe")
                    check(compatibilitySupported || allowUnsupportedDevice) {
                        "gpu_delegate_unsupported model=${tfliteFile.name} backend=tflite_gpu reason=device_not_supported"
                    }
                    val options = if (compatibilitySupported) compatibility.bestOptionsForThisDevice else GpuDelegate.Options()
                    stage = "delegate_create"
                    GpuDelegate(
                        options
                            .setPrecisionLossAllowed(allowPrecisionLoss)
                            .setQuantizedModelsAllowed(false),
                    ).also {
                        Log.i("GVC_RT_CLEAN", "gpu_delegate_create_ok model=${tfliteFile.name}")
                    }
                }
            } catch (error: Throwable) {
                if (error.message?.startsWith("gpu_delegate_unsupported") == true) throw error
                throw IllegalStateException(
                    "gpu_delegate_create_failed stage=$stage model=${tfliteFile.name} backend=tflite_gpu error=${error.message}",
                    error,
                )
            }
            var guard: GpuDelegationGuard? = null
            var interpreter: Interpreter? = null
            try {
                stage = "guard_create"
                val coverage = GpuDelegationGuard(
                    delegate.nativeHandle,
                    tfliteFile.name,
                    allowBuiltinCpuFallback,
                )
                guard = coverage
                stage = "interpreter_create"
                val modelInterpreter = Interpreter(
                    tfliteFile,
                    Interpreter.Options()
                        .setUseXNNPACK(false)
                        .setUseNNAPI(false)
                        .addDelegate(delegate)
                        .addDelegate(coverage),
                )
                interpreter = modelInterpreter
                Log.i("GVC_RT_CLEAN", "gpu_interpreter_create_ok model=${tfliteFile.name}")
                stage = "allocate_tensors"
                modelInterpreter.allocateTensors()
                stage = "delegation_check"
                coverage.requireFullyDelegated()
                Log.i(
                    "GVC_RT_CLEAN",
                    "gpu_delegation_check_ok model=${tfliteFile.name} " +
                        "cpu_fallback_allowed=$allowBuiltinCpuFallback",
                )
                stage = "tensor_buffers"
                val inputInfo = (0 until modelInterpreter.inputTensorCount).joinToString(";") { index ->
                    modelInterpreter.getInputTensor(index).let { "${it.dataType()}:${it.shape().contentToString()}:${it.numBytes()}" }
                }
                val outputInfo = (0 until modelInterpreter.outputTensorCount).joinToString(";") { index ->
                    modelInterpreter.getOutputTensor(index).let { "${it.dataType()}:${it.shape().contentToString()}:${it.numBytes()}" }
                }
                Log.i(
                    "GVC_RT_CLEAN",
                    "gpu_model_created model=${tfliteFile.absolutePath} inputs=$inputInfo outputs=$outputInfo " +
                        "cpu_fallback_allowed=$allowBuiltinCpuFallback",
                )
                return GenericTfliteGpuRuntime(
                    modelName = tfliteFile.name,
                    interpreter = modelInterpreter,
                    delegate = delegate,
                    guard = coverage,
                    inputSizes = LongArray(modelInterpreter.inputTensorCount) { index ->
                        modelInterpreter.getInputTensor(index).numBytes().toLong()
                    },
                    outputSizes = LongArray(modelInterpreter.outputTensorCount) { index ->
                        modelInterpreter.getOutputTensor(index).numBytes().toLong()
                    },
                    optionsSummary =
                        "TFLiteGpu(compatibility_list_supported=$compatibilitySupported," +
                            "gpu_forced_probe=$forcedProbe,cpu_fallback_allowed=$allowBuiltinCpuFallback," +
                            "precision_loss_allowed=$allowPrecisionLoss," +
                            "quantized=false,xnnpack=false,nnapi=false)",
                )
            } catch (error: Throwable) {
                interpreter?.close()
                guard?.close()
                delegate.close()
                if (error.message?.startsWith("gpu_delegate_unsupported") == true) throw error
                throw IllegalStateException(
                    "gpu_model_create_failed stage=$stage model=${tfliteFile.name} backend=tflite_gpu error=${error.message}",
                    error,
                )
            }
        }

        private fun allocateTensorBuffer(label: String, size: Long): ByteBuffer {
            require(size in 0..Int.MAX_VALUE.toLong()) { "$label byte size is unsupported: $size" }
            return ByteBuffer.allocateDirect(size.toInt()).order(ByteOrder.nativeOrder())
        }
    }
}
