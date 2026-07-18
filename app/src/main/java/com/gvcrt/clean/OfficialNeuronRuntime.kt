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
    fun run(inputs: List<ByteArray>, copyOutputs: Boolean = true): List<ByteArray> {
        require(inputs.size == inputSizes.size) {
            "input count mismatch runtime=${inputSizes.size} actual=${inputs.size}"
        }
        val inputBuffers = inputs.mapIndexed { index, bytes ->
            require(bytes.size.toLong() == inputSizes[index]) {
                "input[$index] bytes mismatch runtime=${inputSizes[index]} actual=${bytes.size}"
            }
            directBuffer(bytes)
        }
        val outputBuffers = outputSizes.map { size ->
            ByteBuffer.allocateDirect(size.toInt()).order(ByteOrder.nativeOrder())
        }
        val outputs = outputBuffers.mapIndexed { index, buffer -> index to buffer }.toMap()
        interpreter.runForMultipleInputsOutputs(inputBuffers.toTypedArray(), outputs)
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
        ): OfficialNeuronRuntime {
            cacheDir.mkdirs()
            val delegateOptions = NeuronDelegate.Options()
                .setAllowFp16(allowFp16ForFp32)
                .setExecutionPreference(NeuronDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED)
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
                optionsSummary = "NeuronDelegate(allowFp16=$allowFp16ForFp32,sustainedSpeed)",
            )
        }

        private fun directBuffer(bytes: ByteArray): ByteBuffer =
            ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).also {
                it.put(bytes)
                it.rewind()
            }
    }
}
