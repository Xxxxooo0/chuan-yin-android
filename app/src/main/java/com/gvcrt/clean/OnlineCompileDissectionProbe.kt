package com.gvcrt.clean

import android.content.Context
import kotlin.math.sqrt

class OnlineCompileDissectionProbe(
    private val context: Context,
    private val emit: (String) -> Unit,
) {
    private val store = AssetStore(context)

    fun run(
        opName: String?,
        accelerationMode: Int = MtkTfliteRuntime.ACCELERATION_NEURON,
        useOfficialNeuronDelegate: Boolean = false,
        officialAllowFp16ForFp32: Boolean = true,
    ) {
        if (opName.isNullOrBlank()) {
            val operations = context.assets.list("$ROOT/submodels")?.sorted().orEmpty()
            emit("online_dissect_skip reason=missing_op available=${operations.joinToString(":")}")
            return
        }
        require(!opName.contains('/') && !opName.contains('\\')) { "invalid dissect operation name: $opName" }

        val opRoot = "$ROOT/submodels/$opName"
        val files = context.assets.list(opRoot)?.toList().orEmpty()
        val modelAsset = "$opRoot/model.tflite"
        val inputs = files.filter { INPUT_BIN.matches(it) }.sorted().map { "$opRoot/$it" }
        val expectedOutputs = files.filter { OUTPUT_BIN.matches(it) }.sorted().map { "$opRoot/$it" }
        if (!store.exists(modelAsset) || inputs.isEmpty() || expectedOutputs.isEmpty()) {
            emit(
                "online_dissect_skip op=$opName reason=missing_artifacts " +
                    "model=${store.exists(modelAsset)} inputs=${inputs.size} outputs=${expectedOutputs.size}",
            )
            return
        }

        val inputBytes = inputs.map(store::readBytes)
        emit(
            "online_dissect_op_start op=$opName backend=${backendLabel(accelerationMode, useOfficialNeuronDelegate)} " +
                "model_sha256=${store.sha256(modelAsset)} " +
                "input_bytes=${inputBytes.joinToString(":") { it.size.toString() }}",
        )
        try {
            if (useOfficialNeuronDelegate) {
                runOfficialNeuron(opName, modelAsset, inputBytes, expectedOutputs, officialAllowFp16ForFp32)
                return
            }
            MtkTfliteRuntime.create(
                store.materialize(modelAsset),
                accelerationMode = accelerationMode,
                cacheDir = context.cacheDir.resolve("online_dissect_cache"),
                allowFp16ForFp32 = false,
            ).use { runtime ->
                emit(
                    "online_dissect_create op=$opName create_ok=true " +
                        "fully_delegated=${runtime.fullyDelegated} options=${runtime.optionsSummary}",
                )
                require(runtime.inputSizes.size == inputBytes.size) {
                    "input count mismatch runtime=${runtime.inputSizes.size}, reference=${inputBytes.size}"
                }
                runtime.inputSizes.forEachIndexed { index, size ->
                    require(size == inputBytes[index].size.toLong()) {
                        "input[$index] bytes mismatch runtime=$size, reference=${inputBytes[index].size}"
                    }
                }
                val actualOutputs = runtime.run(inputBytes)
                require(actualOutputs.size == expectedOutputs.size) {
                    "output count mismatch runtime=${actualOutputs.size}, reference=${expectedOutputs.size}"
                }
                actualOutputs.forEachIndexed { index, actualBytes ->
                    val expectedBytes = store.readBytes(expectedOutputs[index])
                    emitComparison(opName, index, actualBytes, expectedBytes)
                }
                emit("online_dissect_complete op=$opName invoke_ok=true")
            }
        } catch (error: Throwable) {
            emit(
                "online_dissect_failure op=$opName create_or_invoke_ok=false " +
                    "type=${error.javaClass.simpleName} message=${error.message}",
            )
        }
    }

    private fun runOfficialNeuron(
        opName: String,
        modelAsset: String,
        inputBytes: List<ByteArray>,
        expectedOutputs: List<String>,
        allowFp16ForFp32: Boolean,
    ) {
        OfficialNeuronRuntime.create(
            store.materialize(modelAsset),
            context.cacheDir.resolve("online_dissect_official_neuron_cache"),
            allowFp16ForFp32,
        ).use { runtime ->
            emit(
                "online_dissect_create op=$opName create_ok=true fully_delegated=not_reported " +
                    "options=${runtime.optionsSummary}",
            )
            require(runtime.inputSizes.size == inputBytes.size) {
                "input count mismatch runtime=${runtime.inputSizes.size}, reference=${inputBytes.size}"
            }
            runtime.inputSizes.forEachIndexed { index, size ->
                require(size == inputBytes[index].size.toLong()) {
                    "input[$index] bytes mismatch runtime=$size, reference=${inputBytes[index].size}"
                }
            }
            val actualOutputs = runtime.run(inputBytes)
            require(actualOutputs.size == expectedOutputs.size) {
                "output count mismatch runtime=${actualOutputs.size}, reference=${expectedOutputs.size}"
            }
            actualOutputs.forEachIndexed { index, actualBytes ->
                emitComparison(opName, index, actualBytes, store.readBytes(expectedOutputs[index]))
            }
            emit("online_dissect_complete op=$opName invoke_ok=true")
        }
    }

    private fun emitComparison(opName: String, index: Int, actual: ByteArray, expected: ByteArray) {
        if (actual.size != expected.size || actual.size % Float.SIZE_BYTES != 0) {
            emit(
                "online_dissect_compare op=$opName output=$index pass=false reason=byte_shape " +
                    "actual_bytes=${actual.size} expected_bytes=${expected.size}",
            )
            return
        }
        val shape = longArrayOf((actual.size / Float.SIZE_BYTES).toLong())
        val actualTensor = TensorIO.readF32Le("$opName.actual.$index", shape, actual)
        val expectedTensor = TensorIO.readF32Le("$opName.expected.$index", shape, expected)
        val diff = TensorIO.diff(actualTensor, expectedTensor)
        emit(
            "online_dissect_compare op=$opName output=$index " +
                "pass=${diff.maxAbs <= TOLERANCE} threshold=$TOLERANCE " +
                "max_abs=${format(diff.maxAbs)} mean_abs=${format(diff.meanAbs)} " +
                "rmse=${format(diff.rmse)} cosine=${format(cosine(actualTensor.data, expectedTensor.data))} " +
                "exact=${diff.exact}",
        )
    }

    private fun cosine(left: FloatArray, right: FloatArray): Float {
        var dot = 0.0
        var leftNorm = 0.0
        var rightNorm = 0.0
        left.indices.forEach { index ->
            dot += left[index].toDouble() * right[index]
            leftNorm += left[index].toDouble() * left[index]
            rightNorm += right[index].toDouble() * right[index]
        }
        return if (leftNorm == 0.0 || rightNorm == 0.0) 0f else (dot / sqrt(leftNorm * rightNorm)).toFloat()
    }

    private fun format(value: Float): String = "%.8f".format(java.util.Locale.US, value)

    private fun backendLabel(accelerationMode: Int, useOfficialNeuronDelegate: Boolean): String = when {
        useOfficialNeuronDelegate -> "official_aar_neuron"
        accelerationMode == MtkTfliteRuntime.ACCELERATION_CPU -> "mtk_cpu"
        accelerationMode == MtkTfliteRuntime.ACCELERATION_NEURON -> "mtk_neuron"
        else -> "unknown_$accelerationMode"
    }

    companion object {
        private const val ROOT = "recon_dissect_i_nhwc"
        private const val TOLERANCE = 5e-4f
        private val INPUT_BIN = Regex("input_[0-9]+\\.bin")
        private val OUTPUT_BIN = Regex("output_[0-9]+\\.bin")
    }
}
