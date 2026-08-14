package com.gvcrt.clean

import com.mediatek.neuropilot_V.Interpreter
import com.mediatek.neuropilot_V.neuron.NeuronDelegate
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class OfficialNeuronRuntime private constructor(
    private val interpreter: Interpreter,
    private val delegate: NeuronDelegate,
    val inputSizes: LongArray,
    val outputSizes: LongArray,
    val optionsSummary: String,
) : AutoCloseable {
    private val inputBuffers = inputSizes.map { size ->
        ByteBuffer.allocateDirect(size.toInt()).order(ByteOrder.nativeOrder())
    }
    private val outputBuffers = outputSizes.map { size ->
        ByteBuffer.allocateDirect(size.toInt()).order(ByteOrder.nativeOrder())
    }
    private val outputBindings = outputBuffers.mapIndexed { index, buffer -> index to buffer }.toMap()

    @Synchronized
    fun run(inputs: List<ByteArray>, copyOutputs: Boolean = true): List<ByteArray> {
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
        interpreter.runForMultipleInputsOutputs(inputBuffers.toTypedArray(), outputBindings)
        if (!copyOutputs) {
            return emptyList()
        }
        return outputBuffers.map { buffer ->
            val bytes = ByteArray(buffer.capacity())
            buffer.rewind()
            buffer.get(bytes)
            bytes
        }
    }

    override fun close() {
        interpreter.close()
        delegate.close()
    }

    companion object {
        fun create(
            tfliteFile: File,
            cacheDir: File,
            allowFp16ForFp32: Boolean = true,
            acceleratorName: String? = null,
            compileOptions: String? = null,
            executionPreference: Int = NeuronDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED,
            modelToken: String? = null,
        ): OfficialNeuronRuntime {
            cacheDir.mkdirs()
            val delegateOptions = NeuronDelegate.Options()
                .setAllowFp16(allowFp16ForFp32)
                .setExecutionPreference(executionPreference)
            acceleratorName?.let(delegateOptions::setAcceleratorName)
            compileOptions?.let(delegateOptions::setCompileOptions)
            modelToken?.let {
                delegateOptions
                    .setCacheDir(cacheDir.absolutePath)
                    .setModelToken(it)
            }
            val delegate = NeuronDelegate(delegateOptions)
            val options = Interpreter.Options().addDelegate(delegate)
            val interpreter = try {
                Interpreter(tfliteFile, options)
            } catch (t: Throwable) {
                delegate.close()
                throw t
            }
            return OfficialNeuronRuntime(
                interpreter = interpreter,
                delegate = delegate,
                inputSizes = LongArray(interpreter.inputTensorCount) { index ->
                    interpreter.getInputTensor(index).numBytes().toLong()
                },
                outputSizes = LongArray(interpreter.outputTensorCount) { index ->
                    interpreter.getOutputTensor(index).numBytes().toLong()
                },
                optionsSummary = buildString {
                    append("NeuronDelegate(allowFp16=$allowFp16ForFp32")
                    append(",preference=$executionPreference")
                    acceleratorName?.let { append(",accelerator=$it") }
                    compileOptions?.let { append(",compileOptions=$it") }
                    modelToken?.let { append(",cache=true,token=$it") }
                    append(")")
                },
            )
        }
    }
}
