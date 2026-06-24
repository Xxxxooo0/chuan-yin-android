package com.gvcrt.clean

import android.content.Context
import org.json.JSONObject

class CleanModuleTests(
    context: Context,
    private val emit: (String) -> Unit,
) {
    private val store = AssetStore(context)
    private val manifest = CleanManifest.parse(store.readBytes(MANIFEST).decodeToString())

    init {
        require(manifest.metadata.optString("precision") == "fp32") {
            "clean v1 requires a source-matched fp32 manifest; re-export assets with --precision fp32"
        }
    }

    fun runModule(moduleName: String) {
        emit("module=$moduleName")
        emitMetadata()
        val cases = manifest.modules[moduleName]
            ?: error("module '$moduleName' is not present in $MANIFEST")
        OnnxSessionRunner(store).use { runner ->
            cases.forEach { runCase(runner, moduleName, it) }
        }
    }

    private fun runCase(runner: OnnxSessionRunner, moduleName: String, case: ModuleCase) {
        emit("case=${case.name}")
        val tensors = linkedMapOf<String, TensorValue>()
        case.steps.forEach { step ->
            emit("step=${step.name} model=${step.model} sha256=${store.sha256(step.model)}")
            val inputs = step.inputs.associate { spec ->
                val tensor = when {
                    spec.source != null -> tensors.getValue(spec.source)
                    spec.path != null && spec.shape != null -> TensorIO.readF32Le(
                        spec.tensorName,
                        spec.shape,
                        store.readBytes(spec.path),
                    )
                    else -> error("input ${spec.tensorName} needs source or path+shape")
                }
                spec.tensorName to tensor
            }
            val outputs = runner.run(step, inputs)
            outputs.forEach { (name, tensor) ->
                tensors[name] = tensor
                emit("output=$name shape=${TensorIO.shapeText(tensor.shape)} elements=${tensor.numel}")
            }
            step.outputs.forEach { spec ->
                val baselinePath = spec.baseline ?: return@forEach
                val actual = outputs.getValue(spec.tensorName)
                val expected = TensorIO.readF32Le(spec.tensorName, spec.shape, store.readBytes(baselinePath))
                val diff = TensorIO.diff(actual, expected)
                val discrete = isDiscrete(spec.tensorName)
                val passed = if (discrete) diff.exact else diff.maxAbs <= CONTINUOUS_MAX_ABS_TOLERANCE
                emit(
                    "compare=${spec.tensorName} kind=${if (discrete) "discrete" else "continuous"} " +
                        "pass=$passed exact=${diff.exact} " +
                        "max_abs=${"%.8f".format(diff.maxAbs)} " +
                        "mean_abs=${"%.8f".format(diff.meanAbs)} " +
                        "rmse=${"%.8f".format(diff.rmse)}"
                )
            }
        }
        if (moduleName == "complete_encoder") {
            runEntropyEncoder(tensors)
        }
        case.binaryComparisons.forEach { (androidPath, baselinePath) ->
            if (store.outputExists(androidPath)) {
                emitBinaryComparison(androidPath, baselinePath)
            } else {
                emit("binary_compare android=$androidPath baseline=$baselinePath status=deferred")
            }
        }
    }

    private fun runEntropyEncoder(tensors: Map<String, TensorValue>) {
        runEntropyPath("i", manifest.entropy["i"] ?: error("missing I rANS assets"), tensors, 4)
        runEntropyPath("p", manifest.entropy["p"] ?: error("missing P rANS assets"), tensors, 2)
        val stream = manifest.stream ?: error("missing encoded IP stream baseline")
        val muxed = GvcStreamMuxer.mux(
            stream,
            store.readOutput("outputs/i_rans_payload.bin"),
            store.readOutput("outputs/p_rans_payload.bin"),
        )
        store.writeOutput("outputs/encoded_ip.gvc", muxed)
        emitBinaryComparison("outputs/encoded_ip.gvc", stream.path)
    }

    private fun runEntropyPath(
        prefix: String,
        entropy: EntropySpec,
        tensors: Map<String, TensorValue>,
        stageCount: Int,
    ) {
        require(!entropy.twoEntropyCoders) { "two-entropy-coder mode is not enabled in clean Android" }
        val z = EntropySymbols.zSymbols(tensors.getValue("${prefix}_z_hat"))
        val expectedZ = TensorIO.readI8(store.readBytes(entropy.zSymbols))
        emitDiscreteComparison("rans_${prefix}_z_symbols", z, expectedZ)

        val packedStages = Array(stageCount) { stage ->
            EntropySymbols.packY(
                tensors.getValue("${prefix}_y_q_w_$stage"),
                tensors.getValue("${prefix}_s_w_$stage"),
            )
        }
        require(entropy.yPacked.size == stageCount) { "${prefix} rANS stage count mismatch" }
        packedStages.forEachIndexed { stage, packed ->
            val expected = TensorIO.readI16Le(entropy.yPacked[stage], store.readBytes(entropy.yPacked[stage]))
            emitDiscreteComparison("rans_${prefix}_y_packed_$stage", packed, expected)
        }

        val gaussian = CdfTable.load(store, entropy.gaussian)
        val zTable = CdfTable.load(store, entropy.z)
        NativeRans.create(gaussian, zTable).use { rans ->
            val payload = rans.encode(z, entropy.zStartOffset, entropy.zPerChannelSize, packedStages)
            store.writeOutput("outputs/${prefix}_rans_payload.bin", payload)
            emitBinaryComparison("outputs/${prefix}_rans_payload.bin", entropy.payload)

            val decoded = rans.decode(
                payload,
                z.size,
                entropy.zStartOffset,
                entropy.zPerChannelSize,
                packedStages.map(EntropySymbols::indexes).toTypedArray(),
            )
            emitDiscreteComparison("rans_${prefix}_z_roundtrip", decoded[0], z)
            packedStages.forEachIndexed { stage, packed ->
                val expected = ByteArray(packed.size) { index -> (packed[index].toInt() shr 8).toByte() }
                emitDiscreteComparison("rans_${prefix}_y_roundtrip_$stage", decoded[stage + 1], expected)
            }
        }
    }

    private fun emitBinaryComparison(androidPath: String, baselinePath: String) {
        val actual = store.readOutput(androidPath)
        val expected = store.readBytes(baselinePath)
        val firstDiff = actual.indices.firstOrNull { index ->
            index >= expected.size || actual[index] != expected[index]
        } ?: if (actual.size == expected.size) null else minOf(actual.size, expected.size)
        emit(
            "binary_compare android=$androidPath baseline=$baselinePath " +
                "exact=${firstDiff == null} android_bytes=${actual.size} server_bytes=${expected.size} " +
                "first_diff=${firstDiff ?: -1}"
        )
    }

    private fun emitDiscreteComparison(name: String, actual: ByteArray, expected: ByteArray) {
        val firstDiff = actual.indices.firstOrNull { index ->
            index >= expected.size || actual[index] != expected[index]
        } ?: if (actual.size == expected.size) null else minOf(actual.size, expected.size)
        emit("compare=$name kind=discrete pass=${firstDiff == null} exact=${firstDiff == null} first_diff=${firstDiff ?: -1}")
    }

    private fun emitDiscreteComparison(name: String, actual: ShortArray, expected: ShortArray) {
        val firstDiff = actual.indices.firstOrNull { index ->
            index >= expected.size || actual[index] != expected[index]
        } ?: if (actual.size == expected.size) null else minOf(actual.size, expected.size)
        emit("compare=$name kind=discrete pass=${firstDiff == null} exact=${firstDiff == null} first_diff=${firstDiff ?: -1}")
    }

    private fun emitMetadata() {
        val metadata: JSONObject = manifest.metadata
        emit("manifest_sha256=${store.sha256(MANIFEST)}")
        metadata.keys().forEach { key ->
            emit("metadata.$key=${metadata.opt(key)}")
        }
    }

    private fun isDiscrete(tensorName: String): Boolean =
        tensorName.endsWith("_z_hat") || tensorName.contains("_y_q_w_")

    companion object {
        private const val MANIFEST = "gvcrt_clean_manifest.json"
        private const val CONTINUOUS_MAX_ABS_TOLERANCE = 2e-5f
    }
}
