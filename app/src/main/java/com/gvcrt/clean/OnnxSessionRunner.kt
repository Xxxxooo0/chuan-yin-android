package com.gvcrt.clean

import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import ai.onnxruntime.providers.NNAPIFlags
import java.io.File
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import java.util.EnumSet
import java.util.Locale
import org.json.JSONArray

enum class OnnxBackend(val label: String) {
    CPU("CPU_ORT_ALL_OPT"),
    NNAPI_FP16_ALLOW_FALLBACK("NNAPI_FP16_ALLOW_FALLBACK"),
    NNAPI_FP16_CPU_DISABLED("NNAPI_FP16_CPU_DISABLED"),
}

class OnnxSessionRunner(
    private val store: AssetStore,
    private val backend: OnnxBackend = OnnxBackend.CPU,
    private val profilingDir: File? = null,
    private val emit: ((String) -> Unit)? = null,
) : AutoCloseable {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val sessions = mutableMapOf<String, OrtSession>()

    fun prepare(steps: Iterable<GraphStep>) {
        steps.forEach(::sessionFor)
    }

    fun evict(modelNames: Iterable<String>) {
        modelNames.distinct().forEach { modelName ->
            sessions.remove(modelName)?.close()
            Log.i(TAG, "onnx_session_evicted model=$modelName")
        }
    }

    fun run(step: GraphStep, inputs: Map<String, TensorValue>): Map<String, TensorValue> {
        val session = sessionFor(step)

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
                    out[spec.tensorName] = readTensor(
                        value,
                        outputType,
                        spec.tensorName,
                        spec.shape,
                    )
                }
                return out
            }
        } finally {
            ortInputs.values.forEach { it.close() }
        }
    }

    private fun sessionFor(step: GraphStep): OrtSession =
        sessions.getOrPut(step.model) {
            val model = step.model
            val createStartNs = System.nanoTime()
            Log.i(
                TAG,
                "onnx_session_create_start model=$model backend=${backend.label}",
            )
            val options = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                profilingDir?.let { directory ->
                    directory.mkdirs()
                    val prefix = File(directory, profileName(model)).absolutePath
                    enableProfiling(prefix)
                }
                when (backend) {
                    OnnxBackend.NNAPI_FP16_ALLOW_FALLBACK ->
                        addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
                    OnnxBackend.NNAPI_FP16_CPU_DISABLED ->
                        addNnapi(EnumSet.of(NNAPIFlags.USE_FP16, NNAPIFlags.CPU_DISABLED)).also {
                            addConfigEntry("session.disable_cpu_ep_fallback", "1")
                        }
                    OnnxBackend.CPU -> Unit
                }
            }
            env.createSession(store.materialize(model).absolutePath, options).also {
                val createMs = (System.nanoTime() - createStartNs) / 1_000_000.0
                Log.i(TAG, "onnx_session_create_complete model=$model ms=%.3f".format(createMs))
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
                OnnxTensor.createTensor(
                    env,
                    ShortBuffer.wrap(tensor.fp16Bits()),
                    tensor.shape,
                    OnnxJavaType.FLOAT16,
                )
            }
            else -> error("unsupported ONNX input type $type for ${tensor.name}")
        }
    }

    private fun readTensor(
        value: OnnxTensor,
        type: OnnxJavaType,
        name: String,
        shape: LongArray,
    ): TensorValue {
        return when (type) {
            OnnxJavaType.FLOAT -> {
                val buffer = value.floatBuffer
                TensorValue(name, shape, FloatArray(buffer.remaining()).also { buffer.get(it) })
            }
            OnnxJavaType.FLOAT16 -> {
                val buffer = value.shortBuffer
                val raw = ShortArray(buffer.remaining())
                buffer.get(raw)
                TensorValue.fromFp16(name, shape, raw)
            }
            else -> error("unsupported ONNX output type $type")
        }
    }

    override fun close() {
        sessions.forEach { (model, session) ->
            if (profilingDir != null) {
                runCatching {
                    val path = session.endProfiling()
                    emitProfileSummary(model, File(path))
                }
                    .onFailure { error ->
                        emit?.invoke(
                            "onnx_profile model=$model status=failed reason=${error.javaClass.simpleName}:${error.message}"
                        )
                    }
            }
            session.close()
        }
        sessions.clear()
    }

    private fun emitProfileSummary(model: String, profile: File) {
        val events = JSONArray(profile.readText())
        val providers = linkedMapOf<String, ProviderProfile>()
        var nodeEvents = 0
        var castEvents = 0
        for (index in 0 until events.length()) {
            val event = events.optJSONObject(index) ?: continue
            if (event.optString("cat") != "Node") continue
            val args = event.optJSONObject("args") ?: continue
            val provider = args.optString("provider")
            if (provider.isEmpty()) continue
            val opName = args.optString("op_name")
            val durationUs = event.optDouble("dur", 0.0)
            val summary = providers.getOrPut(provider) { ProviderProfile() }
            summary.events += 1
            summary.durationUs += durationUs
            nodeEvents += 1
            if (opName == "Cast") castEvents += 1
        }
        val providerText = providers.entries.joinToString(",") { (provider, summary) ->
            "$provider:${summary.events}:${String.format(Locale.US, "%.3f", summary.durationUs / 1000.0)}ms"
        }
        emit?.invoke(
            "onnx_profile model=$model status=ok path=${profile.absolutePath} " +
                "node_events=$nodeEvents cast_events=$castEvents providers=$providerText"
        )
    }

    private fun profileName(model: String): String =
        model.substringAfterLast('/').substringBeforeLast('.')

    private class ProviderProfile(
        var events: Int = 0,
        var durationUs: Double = 0.0,
    )

    private companion object {
        const val TAG = "GVC_RT_CLEAN"
    }
}
