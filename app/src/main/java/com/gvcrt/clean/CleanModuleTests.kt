package com.gvcrt.clean

import android.content.Context
import org.json.JSONObject

class CleanModuleTests(
    context: Context,
    private val emit: (String) -> Unit,
    private val backend: OnnxBackend = OnnxBackend.NNAPI_FP16_ALLOW_FALLBACK,
) {
    private val store = AssetStore(context)
    private val manifest = CleanManifest.parse(store.readBytes(MANIFEST).decodeToString())
    private val precisionFailures = mutableListOf<String>()

    init {
        require(manifest.metadata.optString("precision") == "fp32") {
            "clean v1 requires a source-matched fp32 manifest; re-export assets with --precision fp32"
        }
    }

    fun runModule(moduleName: String) {
        precisionFailures.clear()
        emit("module=$moduleName")
        emit("precision_backend=${backend.label}")
        emitMetadata()
        if (moduleName == "complete_decoder") {
            runCompleteDecoder()
        } else {
            val cases = manifest.modules[moduleName]
                ?: error("module '$moduleName' is not present in $MANIFEST")
            OnnxSessionRunner(store, backend).use { runner ->
                cases.forEach { runCase(runner, moduleName, it) }
            }
        }
        finishModule(moduleName)
    }

    private fun runCase(runner: OnnxSessionRunner, moduleName: String, case: ModuleCase) {
        emit("case=${case.name}")
        val tensors = linkedMapOf<String, TensorValue>()
        case.steps.forEach { step ->
            emit("step=${step.name} model=${step.model} sha256=${store.sha256(step.model)}")
            val inputs = resolveInputs(step, tensors)
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
                recordFailure("tensor:${spec.tensorName}", passed)
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
                recordFailure("binary:$androidPath:missing", false)
            }
        }
    }

    private fun runCompleteDecoder() {
        val decoder = manifest.decoder ?: error("missing decoder specification in $MANIFEST")
        val streamBaseline = manifest.stream ?: error("missing stream specification in $MANIFEST")
        val usesAndroidOutput = store.outputExists(decoder.androidInput)
        val streamBytes = if (usesAndroidOutput) {
            store.readOutput(decoder.androidInput)
        } else {
            store.readBytes(decoder.fallbackInput)
        }
        emit(
            "case=full_bitstream_decode_v2 input=${if (usesAndroidOutput) decoder.androidInput else decoder.fallbackInput} " +
                "bytes=${streamBytes.size} sha256=${AssetStore.sha256(streamBytes)}"
        )

        val parsed = GvcStreamMuxer.demux(streamBytes)
        require(parsed.stream.height == streamBaseline.height && parsed.stream.width == streamBaseline.width) {
            "decoder stream geometry ${parsed.stream.height}x${parsed.stream.width} does not match " +
                "manifest ${streamBaseline.height}x${streamBaseline.width}"
        }
        require(parsed.stream.qp == streamBaseline.qp) {
            "decoder stream QP ${parsed.stream.qp} does not match manifest ${streamBaseline.qp}"
        }
        require(parsed.stream.ecPart == streamBaseline.ecPart && parsed.stream.useAdaI == streamBaseline.useAdaI) {
            "decoder stream entropy flags do not match manifest"
        }
        val remuxed = GvcStreamMuxer.mux(parsed.stream, parsed.iPayload, parsed.pPayload)
        emitBinaryComparison("decoder_stream_remux", remuxed, streamBytes)

        val iEntropy = manifest.entropy["i"] ?: error("missing I rANS assets")
        val pEntropy = manifest.entropy["p"] ?: error("missing P rANS assets")
        emitBinaryComparison("decoder_i_payload", parsed.iPayload, store.readBytes(iEntropy.payload))
        emitBinaryComparison("decoder_p_payload", parsed.pPayload, store.readBytes(pEntropy.payload))

        val tensors = linkedMapOf<String, TensorValue>()
        OnnxSessionRunner(store, backend).use { runner ->
            decodeEntropyPath("i", parsed.iPayload, iEntropy, decoder.i, runner, tensors)
            val iRecon = runDecoderStep(runner, decoder.i.recon, tensors)
            tensors.putAll(iRecon)
            writeTensorOutput("outputs/decoded_i_y_hat.f32le", tensors.getValue("i_y_hat"))
            writeTensorOutput("outputs/decoder_i_reference_frame.f32le", tensors.getValue("encoder_i_reference_frame"))

            val temporal = decoder.p.temporal ?: error("missing P decoder temporal step")
            val temporalOutputs = runDecoderStep(runner, temporal, tensors)
            tensors.putAll(temporalOutputs)

            decodeEntropyPath("p", parsed.pPayload, pEntropy, decoder.p, runner, tensors)
            val pRecon = runDecoderStep(runner, decoder.p.recon, tensors)
            tensors.putAll(pRecon)
            writeTensorOutput("outputs/decoded_p_y_hat.f32le", tensors.getValue("p_y_hat"))
            writeTensorOutput("outputs/decoder_p_reference_feature.f32le", tensors.getValue("encoder_p_reference_feature"))
            writeTensorOutput("outputs/decoder_p_reference_frame.f32le", tensors.getValue("encoder_p_reference_frame"))
        }
    }

    private fun decodeEntropyPath(
        prefix: String,
        payload: ByteArray,
        entropy: EntropySpec,
        decoder: DecoderPathSpec,
        runner: OnnxSessionRunner,
        tensors: MutableMap<String, TensorValue>,
    ) {
        require(!entropy.twoEntropyCoders) { "two-entropy-coder mode is not enabled in clean Android" }
        val gaussian = CdfTable.load(store, entropy.gaussian)
        val zTable = CdfTable.load(store, entropy.z)
        NativeRans.create(gaussian, zTable).use { rans ->
            rans.beginDecode(payload)
            val zName = "${prefix}_z_hat"
            val z = TensorIO.fromI8(
                zName,
                decoder.zShape,
                rans.decodeZ(decoder.zShape.elementCount(), entropy.zStartOffset, entropy.zPerChannelSize),
            )
            tensors[zName] = z
            emitAndRequireDiscreteComparison(zName, z, baselineTensor(zName, decoder.zShape))

            tensors.putAll(runDecoderStep(runner, decoder.hyperPrior, tensors))
            decoder.stages.forEachIndexed { stageIndex, stage ->
                val yInput = stage.inputs.single { it.tensorName.contains("_y_q_w_") }.tensorName
                val scaleOutput = stage.outputs.single { it.tensorName.contains("_s_w_") }
                val stateOutput = stage.outputs.single { it.tensorName != scaleOutput.tensorName }

                tensors[yInput] = TensorValue(yInput, decoder.yStageShape, FloatArray(decoder.yStageShape.elementCount()))
                val scaleProbe = runDecoderStep(
                    runner,
                    stage,
                    tensors,
                    setOf(scaleOutput.tensorName),
                )
                val scales = scaleProbe.getValue(scaleOutput.tensorName)
                val decodedSymbols = rans.decodeY(EntropySymbols.indexesForScales(scales))
                val y = TensorIO.fromI8(yInput, decoder.yStageShape, decodedSymbols)
                tensors[yInput] = y
                emitAndRequireDiscreteComparison(yInput, y, baselineTensor(yInput, decoder.yStageShape))

                val stageOutputs = runDecoderStep(
                    runner,
                    stage,
                    tensors,
                    setOf(stateOutput.tensorName),
                )
                tensors.putAll(stageOutputs)
                emit("decoder_${prefix}_stage=$stageIndex symbols=${decodedSymbols.size}")
            }
        }
    }

    private fun runDecoderStep(
        runner: OnnxSessionRunner,
        step: GraphStep,
        tensors: Map<String, TensorValue>,
        compareOutputs: Set<String>? = null,
    ): Map<String, TensorValue> {
        emit("step=${step.name} model=${step.model} sha256=${store.sha256(step.model)}")
        val outputs = runner.run(step, resolveInputs(step, tensors))
        outputs.forEach { (name, tensor) ->
            emit("output=$name shape=${TensorIO.shapeText(tensor.shape)} elements=${tensor.numel}")
        }
        step.outputs.forEach { output ->
            val baselinePath = output.baseline ?: return@forEach
            if (compareOutputs == null || output.tensorName in compareOutputs) {
                emitTensorComparison(output.tensorName, outputs.getValue(output.tensorName), baselineTensor(output.tensorName, output.shape, baselinePath))
            }
        }
        return outputs
    }

    private fun resolveInputs(step: GraphStep, tensors: Map<String, TensorValue>): Map<String, TensorValue> =
        step.inputs.associate { spec ->
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

    private fun baselineTensor(name: String, shape: LongArray, path: String = "baseline/tensors/$name.f32le"): TensorValue =
        TensorIO.readF32Le(name, shape, store.readBytes(path))

    private fun writeTensorOutput(path: String, tensor: TensorValue) {
        store.writeOutput(path, TensorIO.f32Le(tensor))
        emit("wrote=$path elements=${tensor.numel}")
    }

    private fun emitAndRequireDiscreteComparison(name: String, actual: TensorValue, expected: TensorValue) {
        val diff = TensorIO.diff(actual, expected)
        emit("compare=$name kind=discrete pass=${diff.exact} exact=${diff.exact} first_diff=${firstDifference(actual, expected)}")
        recordFailure("tensor:$name", diff.exact)
    }

    private fun emitTensorComparison(name: String, actual: TensorValue, expected: TensorValue) {
        val diff = TensorIO.diff(actual, expected)
        val discrete = isDiscrete(name)
        val passed = if (discrete) diff.exact else diff.maxAbs <= CONTINUOUS_MAX_ABS_TOLERANCE
        emit(
            "compare=$name kind=${if (discrete) "discrete" else "continuous"} " +
                "pass=$passed exact=${diff.exact} max_abs=${"%.8f".format(diff.maxAbs)} " +
                "mean_abs=${"%.8f".format(diff.meanAbs)} rmse=${"%.8f".format(diff.rmse)}"
        )
        recordFailure("tensor:$name", passed)
    }

    private fun firstDifference(actual: TensorValue, expected: TensorValue): Int =
        actual.data.indices.firstOrNull { actual.data[it] != expected.data[it] } ?: -1

    private fun LongArray.elementCount(): Int = fold(1L) { acc, value -> acc * value }.toInt()

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
        RansNativeEncoder.create(gaussian, zTable, entropy.twoEntropyCoders).use { encoder ->
            encoder.encodeZ(z, entropy.zStartOffset, entropy.zPerChannelSize)
            repeat(stageCount) { stage ->
                encoder.encodeY(tensors.getValue("${prefix}_y_q_w_$stage"), tensors.getValue("${prefix}_s_w_$stage"))
            }
            val payload = encoder.flush()
            store.writeOutput("outputs/${prefix}_rans_payload.bin", payload)
            emitBinaryComparison("outputs/${prefix}_rans_payload.bin", entropy.payload)

        }

        NativeRans.create(gaussian, zTable).use { rans ->
            val decoded = rans.decode(
                store.readOutput("outputs/${prefix}_rans_payload.bin"),
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
        emitBinaryComparison("android=$androidPath baseline=$baselinePath", actual, expected)
    }

    private fun emitBinaryComparison(name: String, actual: ByteArray, expected: ByteArray) {
        val firstDiff = actual.indices.firstOrNull { index ->
            index >= expected.size || actual[index] != expected[index]
        } ?: if (actual.size == expected.size) null else minOf(actual.size, expected.size)
        emit(
            "binary_compare $name exact=${firstDiff == null} actual_bytes=${actual.size} expected_bytes=${expected.size} " +
                "first_diff=${firstDiff ?: -1}"
        )
        recordFailure("binary:$name", firstDiff == null)
    }

    private fun emitDiscreteComparison(name: String, actual: ByteArray, expected: ByteArray) {
        val firstDiff = actual.indices.firstOrNull { index ->
            index >= expected.size || actual[index] != expected[index]
        } ?: if (actual.size == expected.size) null else minOf(actual.size, expected.size)
        emit("compare=$name kind=discrete pass=${firstDiff == null} exact=${firstDiff == null} first_diff=${firstDiff ?: -1}")
        recordFailure("discrete:$name", firstDiff == null)
    }

    private fun emitDiscreteComparison(name: String, actual: ShortArray, expected: ShortArray) {
        val firstDiff = actual.indices.firstOrNull { index ->
            index >= expected.size || actual[index] != expected[index]
        } ?: if (actual.size == expected.size) null else minOf(actual.size, expected.size)
        emit("compare=$name kind=discrete pass=${firstDiff == null} exact=${firstDiff == null} first_diff=${firstDiff ?: -1}")
        recordFailure("discrete:$name", firstDiff == null)
    }

    private fun recordFailure(label: String, passed: Boolean) {
        if (!passed) precisionFailures += label
    }

    private fun finishModule(moduleName: String) {
        val failures = precisionFailures.distinct()
        val passed = failures.isEmpty()
        emit(
            "module_precision_status=${if (passed) "PASS" else "FAIL"} " +
                "module=$moduleName failures=${failures.size}" +
                (if (passed) "" else " failed_checks=${failures.joinToString(",")}")
        )
        require(passed) { "$moduleName precision failed at ${failures.joinToString(",")}" }
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
