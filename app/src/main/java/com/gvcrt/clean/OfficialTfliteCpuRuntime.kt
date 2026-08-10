package com.gvcrt.clean

import com.mediatek.neuropilot_V.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** MediaTek TFLite Shim running on CPU only, with no Neuron or NNAPI delegate. */
class OfficialTfliteCpuRuntime private constructor(
    private val interpreter: Interpreter,
    val inputSizes: LongArray,
    val outputSizes: LongArray,
    val optionsSummary: String,
) : AutoCloseable {
    fun run(inputs: List<ByteArray>): List<ByteArray> {
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
        return outputBuffers.map { buffer ->
            ByteArray(buffer.capacity()).also { bytes ->
                buffer.rewind()
                buffer.get(bytes)
            }
        }
    }

    override fun close() = interpreter.close()

    companion object {
        fun create(tfliteFile: File, threads: Int = 4): OfficialTfliteCpuRuntime {
            require(threads > 0) { "threads must be positive" }
            val options = Interpreter.Options()
                .setNumThreads(threads)
                .setUseXNNPACK(true)
                .setUseNNAPI(false)
            val interpreter = Interpreter(tfliteFile, options)
            return OfficialTfliteCpuRuntime(
                interpreter = interpreter,
                inputSizes = LongArray(interpreter.inputTensorCount) { index ->
                    interpreter.getInputTensor(index).numBytes().toLong()
                },
                outputSizes = LongArray(interpreter.outputTensorCount) { index ->
                    interpreter.getOutputTensor(index).numBytes().toLong()
                },
                optionsSummary = "TFLiteCPU(threads=$threads,xnnpack=true,nnapi=false,delegate=none)",
            )
        }

        private fun directBuffer(bytes: ByteArray): ByteBuffer =
            ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).also {
                it.put(bytes)
                it.rewind()
            }
    }
}
