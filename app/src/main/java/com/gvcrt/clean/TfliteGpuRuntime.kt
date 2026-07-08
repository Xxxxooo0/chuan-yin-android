package com.gvcrt.clean

import android.os.SystemClock
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import kotlin.math.ceil

class TfliteGpuRuntime private constructor(
    private val interpreter: Interpreter,
    private val delegate: GpuDelegate,
    val inputSizes: LongArray,
    val outputSizes: LongArray,
    val delegateOptions: String,
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
        fun isAvailable(): Boolean =
            runCatching { CompatibilityList().isDelegateSupportedOnThisDevice }.getOrDefault(false)

        fun create(tfliteFile: File, force: Boolean = false): TfliteGpuRuntime {
            val compatibility = CompatibilityList()
            val supported = compatibility.isDelegateSupportedOnThisDevice
            require(force || supported) {
                "TFLite GPU delegate unsupported on this device"
            }
            val delegateOptions = if (supported) compatibility.bestOptionsForThisDevice else null
            val delegate = if (delegateOptions != null) GpuDelegate(delegateOptions) else GpuDelegate()
            val options = Interpreter.Options().addDelegate(delegate)
            val interpreter = try {
                Interpreter(tfliteFile, options)
            } catch (t: Throwable) {
                delegate.close()
                throw t
            }
            return TfliteGpuRuntime(
                interpreter = interpreter,
                delegate = delegate,
                inputSizes = LongArray(interpreter.inputTensorCount) { index ->
                    tensorBytes(interpreter.getInputTensor(index).shape(), interpreter.getInputTensor(index).dataType())
                },
                outputSizes = LongArray(interpreter.outputTensorCount) { index ->
                    tensorBytes(interpreter.getOutputTensor(index).shape(), interpreter.getOutputTensor(index).dataType())
                },
                delegateOptions = if (supported) "compatibility_best_options" else "forced_default_options",
            )
        }

        fun percentile(sorted: List<Long>, percentile: Double): Double {
            if (sorted.isEmpty()) return 0.0
            val index = ceil(percentile * sorted.size).toInt().coerceIn(1, sorted.size) - 1
            return sorted[index].toDouble()
        }

        fun fmtMs(nanos: Double): String = String.format(Locale.US, "%.3f", nanos / 1_000_000.0)

        private fun tensorBytes(shape: IntArray, dataType: DataType): Long =
            shape.fold(1L) { acc, value -> acc * value.toLong() } * dataType.byteSize().toLong()

        private fun directBuffer(bytes: ByteArray): ByteBuffer =
            ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).also {
                it.put(bytes)
                it.rewind()
            }
    }
}
