package com.gvcrt.clean

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.nio.FloatBuffer
import java.nio.ShortBuffer

class OnnxSessionRunner(private val store: AssetStore) : AutoCloseable {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val sessions = mutableMapOf<String, OrtSession>()

    fun run(step: GraphStep, inputs: Map<String, TensorValue>): Map<String, TensorValue> {
        val session = sessions.getOrPut(step.model) {
            val options = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            env.createSession(store.materialize(step.model).absolutePath, options)
        }

        val ortInputs = linkedMapOf<String, OnnxTensor>()
        try {
            step.inputs.forEach { spec ->
                val tensor = inputs.getValue(spec.tensorName)
                val inputType = (session.inputInfo.getValue(spec.tensorName).info as TensorInfo).type
                ortInputs[spec.tensorName] = createTensor(tensor, inputType)
            }
            session.run(ortInputs).use { result ->
                val out = linkedMapOf<String, TensorValue>()
                step.outputs.forEach { spec ->
                    val value = result.get(spec.tensorName).orElseThrow {
                        IllegalStateException("missing ONNX output ${spec.tensorName} from ${step.name}")
                    } as OnnxTensor
                    val outputType = value.info.type
                    val data = readTensor(value, outputType)
                    out[spec.tensorName] = TensorValue(spec.tensorName, spec.shape, data)
                }
                return out
            }
        } finally {
            ortInputs.values.forEach { it.close() }
        }
    }

    private fun createTensor(tensor: TensorValue, type: OnnxJavaType): OnnxTensor {
        return when (type) {
            OnnxJavaType.FLOAT -> OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(tensor.data),
                tensor.shape,
            )
            OnnxJavaType.FLOAT16 -> {
                val half = ShortArray(tensor.data.size) { TensorIO.floatToHalfBits(tensor.data[it]) }
                OnnxTensor.createTensor(
                    env,
                    ShortBuffer.wrap(half),
                    tensor.shape,
                    OnnxJavaType.FLOAT16,
                )
            }
            else -> error("unsupported ONNX input type $type for ${tensor.name}")
        }
    }

    private fun readTensor(value: OnnxTensor, type: OnnxJavaType): FloatArray {
        return when (type) {
            OnnxJavaType.FLOAT -> {
                val buffer = value.floatBuffer
                FloatArray(buffer.remaining()).also { buffer.get(it) }
            }
            OnnxJavaType.FLOAT16 -> {
                val buffer = value.shortBuffer
                val raw = ShortArray(buffer.remaining())
                buffer.get(raw)
                FloatArray(raw.size) { TensorIO.halfBitsToFloat(raw[it]) }
            }
            else -> error("unsupported ONNX output type $type")
        }
    }

    override fun close() {
        sessions.values.forEach { it.close() }
        sessions.clear()
    }
}
