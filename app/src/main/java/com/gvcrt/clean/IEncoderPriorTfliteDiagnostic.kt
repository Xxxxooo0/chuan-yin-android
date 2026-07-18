package com.gvcrt.clean

import android.content.Context
import android.os.SystemClock
import java.io.File
import java.util.Locale

class IEncoderPriorTfliteDiagnostic(
    context: Context,
    private val emit: (String) -> Unit,
) {
    private val store = AssetStore(context)
    private val cacheDir = File(context.cacheDir, "i_prior_tflite_diagnostic")
    private val common = tensor("i_common_params", COMMON_SHAPE)
    private val reduced = tensor("i_prior_reduced_common_params", REDUCED_SHAPE, "prior_npu_diagnostic/reduced_common_params.f32le")

    fun run() {
        val directFp16Asset = "$ASSET_DIR/i_prior_reduce_direct_nhwc_fp16.tflite"
        val directFp32Asset = "$ASSET_DIR/i_prior_reduce_direct_nhwc_fp32.tflite"
        if (store.exists(directFp32Asset)) {
            emit("i_prior_tflite_probe_start mode=NEURON accelerator=AUTO io=float32 direct_reduce_stage1")
            probe(
                directFp32Asset,
                listOf(PriorNpuTensorCodec.nchwToNhwcF32(common.data, 514, 16, 32)),
            )
            val stage1Asset = "$ASSET_DIR/i_prior_stage1_fp16_weight.tflite"
            if (store.exists(stage1Asset)) {
                val yHat = tensor("i_y_hat_so_far_0", Y_SHAPE)
                probe(
                    stage1Asset,
                    listOf(
                        PriorNpuTensorCodec.nchwToNhwcF32(yHat.data, 256, 16, 32),
                        PriorNpuTensorCodec.nchwToNhwcF32(reduced.data, 256, 16, 32),
                    ),
                )
            } else {
                emit("i_prior_tflite_skip asset=$stage1Asset reason=missing_asset")
            }
            val stage2Asset = "$ASSET_DIR/i_prior_stage2_fp16_weight.tflite"
            if (store.exists(stage2Asset)) {
                val yHat = tensor("i_y_hat_so_far_1", Y_SHAPE)
                probe(
                    stage2Asset,
                    listOf(
                        PriorNpuTensorCodec.nchwToNhwcF32(yHat.data, 256, 16, 32),
                        PriorNpuTensorCodec.nchwToNhwcF32(reduced.data, 256, 16, 32),
                    ),
                )
            } else {
                emit("i_prior_tflite_skip asset=$stage2Asset reason=missing_asset")
            }
            val stage3Asset = "$ASSET_DIR/i_prior_stage3_fp16_weight.tflite"
            if (store.exists(stage3Asset)) {
                val yHat = tensor("i_y_hat_so_far_2", Y_SHAPE)
                probe(
                    stage3Asset,
                    listOf(
                        PriorNpuTensorCodec.nchwToNhwcF32(yHat.data, 256, 16, 32),
                        PriorNpuTensorCodec.nchwToNhwcF32(reduced.data, 256, 16, 32),
                    ),
                )
            } else {
                emit("i_prior_tflite_skip asset=$stage3Asset reason=missing_asset")
            }
            emit("i_prior_tflite_probe_complete direct_reduce_stage1_stage2_stage3")
            return
        }
        if (store.exists(directFp16Asset)) {
            emit("i_prior_tflite_probe_start mode=NEURON accelerator=AUTO io=float16 direct_reduce_only")
            probe(
                directFp16Asset,
                listOf(PriorNpuTensorCodec.nchwToNhwcFp16(common.data, 514, 16, 32)),
            )
            emit("i_prior_tflite_probe_complete direct_reduce_only")
            return
        }

        emit("i_prior_tflite_probe_start mode=NEURON accelerator=AUTO io=float32 legacy_weight_fp16")
        probe(
            "$ASSET_DIR/i_prior_reduce_fp16_weight.tflite",
            listOf(PriorNpuTensorCodec.nchwToNhwcF32(common.data, 514, 16, 32)),
        )
        (1..3).forEach { stage ->
            val yHat = tensor("i_y_hat_so_far_${stage - 1}", Y_SHAPE)
            probe(
                "$ASSET_DIR/i_prior_stage${stage}_fp16_weight.tflite",
                listOf(
                    PriorNpuTensorCodec.nchwToNhwcF32(yHat.data, 256, 16, 32),
                    PriorNpuTensorCodec.nchwToNhwcF32(reduced.data, 256, 16, 32),
                ),
            )
        }
        emit("i_prior_tflite_probe_complete")
    }

    fun runSpeed(warmupRuns: Int = 5, measuredRuns: Int = 50) {
        val models = directFp32Models()
        if (models.isEmpty()) {
            emit("i_prior_tflite_speed_result create_ok=false reason=missing_direct_fp32_or_stage_assets")
            return
        }
        emit(
            "i_prior_tflite_speed_start backend=mtk_neuron accelerator=AUTO warmup=$warmupRuns " +
                "measured=$measuredRuns scope=continuous_tflite_only",
        )
        val runtimes = ArrayList<Pair<SpeedModel, MtkTfliteRuntime>>(models.size)
        try {
            models.forEach { model ->
                val started = SystemClock.elapsedRealtimeNanos()
                val runtime = MtkTfliteRuntime.create(
                    store.materialize(model.asset),
                    accelerationMode = MtkTfliteRuntime.ACCELERATION_NEURON,
                    acceleratorFlag = MtkTfliteRuntime.ACCELERATOR_AUTO,
                    cacheDir = cacheDir,
                )
                val createMs = (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0
                emit(
                    "i_prior_tflite_speed_create stage=${model.stage} create_ok=true " +
                        "create_ms=${formatMs(createMs)} fully_delegated=${runtime.fullyDelegated} " +
                        "options=${runtime.optionsSummary}",
                )
                runtimes += model to runtime
            }
            runtimes.forEach { (model, runtime) ->
                repeat(warmupRuns) { runtime.run(model.inputs, copyOutputs = false) }
            }
            val totals = LongArray(measuredRuns)
            runtimes.forEach { (model, runtime) ->
                val values = LongArray(measuredRuns)
                repeat(measuredRuns) { index ->
                    val started = SystemClock.elapsedRealtimeNanos()
                    runtime.run(model.inputs, copyOutputs = false)
                    values[index] = SystemClock.elapsedRealtimeNanos() - started
                    totals[index] += values[index]
                }
                emitSummary(model.stage, values)
            }
            emitSummary("continuous_tflite_total", totals)
            emit("i_prior_tflite_speed_complete scope=continuous_tflite_only")
        } catch (error: Throwable) {
            emit(
                "i_prior_tflite_speed_result create_or_invoke_ok=false type=${error.javaClass.simpleName} " +
                    "message=${error.message}",
            )
        } finally {
            runtimes.forEach { (_, runtime) -> runtime.close() }
        }
    }

    fun runPrecision() {
        if (store.exists(DIRECT_STAGE1_ASSET)) {
            runDirectStage1Precision()
            return
        }
        val reduceAsset = "$ASSET_DIR/i_prior_reduce_direct_nhwc_fp32.tflite"
        val stageAssets = (1..3).map { "$ASSET_DIR/i_prior_stage${it}_fp16_weight.tflite" }
        if (!store.exists(reduceAsset) || stageAssets.any { !store.exists(it) }) {
            emit("i_prior_tflite_precision_result pass=false reason=missing_assets")
            return
        }
        val y = tensor("i_y_pre_prior", Y_SHAPE)
        emit("i_prior_tflite_precision_start backend=mtk_neuron accelerator=AUTO")
        val onnxCpuPass = verifyCanonicalOnnx(OnnxBackend.CPU)
        try {
            MtkTfliteRuntime.create(
                store.materialize(reduceAsset),
                MtkTfliteRuntime.ACCELERATION_NEURON,
                MtkTfliteRuntime.ACCELERATOR_AUTO,
                cacheDir,
            ).use { reduceRuntime ->
                val reduced = PriorNpuTensorCodec.nhwcF32ToNchw(
                    reduceRuntime.run(
                        listOf(PriorNpuTensorCodec.nchwToNhwcF32(common.data, 514, 16, 32)),
                        copyOutputs = true,
                    ).single(),
                    256,
                    16,
                    32,
                )
                emitContinuous("reduce", reduced, trace("reduced_common_params"))
                I4xPriorNative.create(y.data, common.data, FORCE_ZERO_THRESHOLD).use { native ->
                    var exactSymbols = true
                    var exactCdf = true
                    var exactYHat = true
                    var stageResult = native.runStage0()
                    emitContinuous("stage0_scales", stage0Tensor("scales"), trace("stage0_scales"))
                    emitContinuous("stage0_means", stage0Tensor("means"), trace("stage0_means"))
                    exactSymbols = compareDiscrete("stage0_symbols", stageResult.symbols, baseline("i_y_q_w_0", STAGE_SHAPE)) && exactSymbols
                    exactCdf = compareCdf("stage0", stageResult.scales, baseline("i_s_w_0", STAGE_SHAPE)) && exactCdf
                    exactYHat = compareDiscrete("stage0_y_hat", stageResult.yHatSoFar, baseline("i_y_hat_so_far_0", Y_SHAPE)) && exactYHat

                    stageAssets.forEachIndexed { index, asset ->
                        val stage = index + 1
                        MtkTfliteRuntime.create(
                            store.materialize(asset),
                            MtkTfliteRuntime.ACCELERATION_NEURON,
                            MtkTfliteRuntime.ACCELERATOR_AUTO,
                            cacheDir,
                        ).use { runtime ->
                            val outputs = runtime.run(
                                listOf(
                                    PriorNpuTensorCodec.nchwToNhwcF32(stageResult.yHatSoFar, 256, 16, 32),
                                    PriorNpuTensorCodec.nchwToNhwcF32(reduced, 256, 16, 32),
                                ),
                                copyOutputs = true,
                            )
                            val scales = PriorNpuTensorCodec.nhwcF32ToNchw(outputs[0], 256, 16, 32)
                            val means = PriorNpuTensorCodec.nhwcF32ToNchw(outputs[1], 256, 16, 32)
                            emitContinuous("stage${stage}_scales", scales, trace("stage${stage}_scales"))
                            emitContinuous("stage${stage}_means", means, trace("stage${stage}_means"))
                            stageResult = native.runStage(stage, scales, means)
                            exactSymbols = compareDiscrete("stage${stage}_symbols", stageResult.symbols, baseline("i_y_q_w_$stage", STAGE_SHAPE)) && exactSymbols
                            exactCdf = compareCdf("stage$stage", stageResult.scales, baseline("i_s_w_$stage", STAGE_SHAPE)) && exactCdf
                            if (stage < 3) {
                                exactYHat = compareDiscrete("stage${stage}_y_hat", stageResult.yHatSoFar, baseline("i_y_hat_so_far_$stage", Y_SHAPE)) && exactYHat
                            }
                        }
                    }
                    val finalYHat = native.finish()
                    exactYHat = compareDiscrete("final_y_hat", finalYHat, baseline("i_y_hat", Y_SHAPE)) && exactYHat
                    emit(
                        "i_prior_tflite_precision_result onnx_cpu_pass=$onnxCpuPass " +
                            "symbols_exact=$exactSymbols " +
                            "cdf_indexes_exact=$exactCdf y_hat_exact=$exactYHat " +
                            "pass=${exactSymbols && exactCdf && exactYHat}",
                    )
                }
            }
        } catch (error: Throwable) {
            emit("i_prior_tflite_precision_result pass=false type=${error.javaClass.simpleName} message=${error.message}")
        }
    }

    private fun runDirectStage1Precision() {
        val y = tensor("i_y_pre_prior", Y_SHAPE)
        val expectedScales = trace("stage1_scales")
        val expectedMeans = trace("stage1_means")
        val serverYHat0 = tensor("i_y_hat_so_far_0", Y_SHAPE)
        emit("i_prior_direct_stage1_precision_start backend=mtk_neuron accelerator=AUTO io=float32 allow_fp16=false")
        try {
            MtkTfliteRuntime.create(
                store.materialize(DIRECT_STAGE1_ASSET),
                MtkTfliteRuntime.ACCELERATION_NEURON,
                MtkTfliteRuntime.ACCELERATOR_AUTO,
                cacheDir,
                allowFp16ForFp32 = false,
            ).use { runtime ->
                val outputs = runtime.run(
                    listOf(
                        PriorNpuTensorCodec.nchwToNhwcF32(serverYHat0.data, 256, 16, 32),
                        PriorNpuTensorCodec.nchwToNhwcF32(reduced.data, 256, 16, 32),
                    ),
                    copyOutputs = true,
                )
                require(outputs.size == 2) { "direct stage1 returned ${outputs.size} outputs" }
                val first = PriorNpuTensorCodec.nhwcF32ToNchw(outputs[0], 256, 16, 32)
                val second = PriorNpuTensorCodec.nhwcF32ToNchw(outputs[1], 256, 16, 32)
                val directScore = meanAbs(first, expectedScales) + meanAbs(second, expectedMeans)
                val swappedScore = meanAbs(first, expectedMeans) + meanAbs(second, expectedScales)
                val scales: FloatArray
                val means: FloatArray
                val outputOrder: String
                if (swappedScore < directScore) {
                    scales = second
                    means = first
                    outputOrder = "means,scales"
                } else {
                    scales = first
                    means = second
                    outputOrder = "scales,means"
                }
                emit(
                    "i_prior_direct_stage1_create asset=$DIRECT_STAGE1_ASSET sha256=${store.sha256(DIRECT_STAGE1_ASSET)} " +
                        "create_ok=true fully_delegated=${runtime.fullyDelegated} output_order=$outputOrder " +
                        "options=${runtime.optionsSummary}",
                )
                emitContinuous("direct_stage1_scales", scales, expectedScales)
                emitContinuous("direct_stage1_means", means, expectedMeans)

                I4xPriorNative.create(y.data, common.data, FORCE_ZERO_THRESHOLD).use { native ->
                    val stage0 = native.runStage0()
                    val stage0Exact = compareDiscrete(
                        "direct_stage1_input_y_hat_so_far_0",
                        stage0.yHatSoFar,
                        serverYHat0.data,
                    )
                    val stage1 = native.runStage(1, scales, means)
                    val symbolsExact = compareDiscrete(
                        "direct_stage1_symbols",
                        stage1.symbols,
                        baseline("i_y_q_w_1", STAGE_SHAPE),
                    )
                    val cdfExact = compareCdf("direct_stage1", stage1.scales, baseline("i_s_w_1", STAGE_SHAPE))
                    val yHatExact = compareDiscrete(
                        "direct_stage1_y_hat_so_far",
                        stage1.yHatSoFar,
                        baseline("i_y_hat_so_far_1", Y_SHAPE),
                    )
                    emit(
                        "i_prior_direct_stage1_precision_result input_exact=$stage0Exact " +
                            "symbols_exact=$symbolsExact cdf_indexes_exact=$cdfExact " +
                            "y_hat_exact=$yHatExact pass=${stage0Exact && symbolsExact && cdfExact && yHatExact}",
                    )
                }
            }
        } catch (error: Throwable) {
            emit(
                "i_prior_direct_stage1_precision_result pass=false " +
                    "type=${error.javaClass.simpleName} message=${error.message}",
            )
        }
    }

    /** Rechecks the original full ONNX prior with exactly the server trace inputs. */
    private fun verifyCanonicalOnnx(backend: OnnxBackend): Boolean {
        val manifest = CleanManifest.parse(store.readBytes(MANIFEST).decodeToString())
        val step = manifest.modules.getValue("complete_encoder").single().steps
            .single { it.name == "i_prior_4x" }
        var pass = true
        try {
            OnnxSessionRunner(store, backend).use { runner ->
                val outputs = runner.run(
                    step,
                    mapOf(
                        "i_y_pre_prior" to tensor("i_y_pre_prior", Y_SHAPE),
                        "i_common_params" to common,
                    ),
                )
                step.outputs.forEach { spec ->
                    val actual = outputs.getValue(spec.tensorName)
                    val expected = TensorIO.readF32Le(
                        "server_${spec.tensorName}",
                        spec.shape,
                        store.readBytes(requireNotNull(spec.baseline)),
                    )
                    val diff = TensorIO.diff(actual, expected)
                    val isScale = spec.tensorName.startsWith("i_s_w_")
                    val isYHat = spec.tensorName == "i_y_hat"
                    val passForTensor = when {
                        isScale -> {
                        EntropySymbols.indexesForScales(actual).contentEquals(
                            EntropySymbols.indexesForScales(expected),
                        )
                        }
                        isYHat -> diff.maxAbs <= CONTINUOUS_Y_HAT_MAX_ABS
                        else -> diff.exact
                    }
                    pass = pass && passForTensor
                    emit(
                        "i_prior_onnx_precision backend=${backend.label} tensor=${spec.tensorName} " +
                            "kind=${when { isScale -> "cdf_indexes"; isYHat -> "continuous"; else -> "discrete" }} " +
                            "pass=$passForTensor " +
                            "max_abs=${formatValue(diff.maxAbs)} mean_abs=${formatValue(diff.meanAbs)} " +
                            "rmse=${formatValue(diff.rmse)}",
                    )
                }
            }
        } catch (error: Throwable) {
            pass = false
            emit(
                "i_prior_onnx_precision backend=${backend.label} pass=false " +
                    "type=${error.javaClass.simpleName} message=${error.message}",
            )
        }
        emit("i_prior_onnx_precision_result backend=${backend.label} pass=$pass")
        return pass
    }

    private fun probe(asset: String, inputs: List<ByteArray>) {
        try {
            MtkTfliteRuntime.create(
                store.materialize(asset),
                accelerationMode = MtkTfliteRuntime.ACCELERATION_NEURON,
                acceleratorFlag = MtkTfliteRuntime.ACCELERATOR_AUTO,
                cacheDir = cacheDir,
            ).use { runtime ->
                emit(
                    "i_prior_tflite_create asset=$asset sha256=${store.sha256(asset)} " +
                        "create_ok=true fully_delegated=${runtime.fullyDelegated} options=${runtime.optionsSummary}",
                )
                val outputs = runtime.run(inputs, copyOutputs = true)
                emit(
                    "i_prior_tflite_invoke asset=$asset invoke_ok=true output_bytes=" +
                        outputs.joinToString(prefix = "[", postfix = "]") { it.size.toString() },
                )
            }
        } catch (error: Throwable) {
            emit(
                "i_prior_tflite_failure asset=$asset create_or_invoke_ok=false " +
                    "type=${error.javaClass.simpleName} message=${error.message}",
            )
        }
    }

    private fun stage0Tensor(kind: String): FloatArray = trace("stage0_$kind")

    private fun trace(name: String): FloatArray =
        tensor(name, Y_SHAPE, "$ASSET_DIR/$name.f32le").data

    private fun baseline(name: String, shape: LongArray): FloatArray = tensor(name, shape).data

    private fun emitContinuous(label: String, actual: FloatArray, expected: FloatArray) {
        val diff = TensorIO.diff(TensorValue(label, longArrayOf(actual.size.toLong()), actual), TensorValue("server_$label", longArrayOf(expected.size.toLong()), expected))
        emit("i_prior_tflite_precision tensor=$label kind=continuous max_abs=${formatValue(diff.maxAbs)} mean_abs=${formatValue(diff.meanAbs)} rmse=${formatValue(diff.rmse)}")
    }

    private fun compareDiscrete(label: String, actual: FloatArray, expected: FloatArray): Boolean {
        val exact = actual.indices.all { actual[it] == expected[it] }
        val first = actual.indices.firstOrNull { actual[it] != expected[it] } ?: -1
        emit("i_prior_tflite_precision tensor=$label kind=discrete exact=$exact first_diff=$first")
        return exact
    }

    private fun compareCdf(stage: String, actual: FloatArray, expected: FloatArray): Boolean {
        val actualIndexes = EntropySymbols.indexesForScales(TensorValue("actual", STAGE_SHAPE, actual))
        val expectedIndexes = EntropySymbols.indexesForScales(TensorValue("expected", STAGE_SHAPE, expected))
        val firstDifference = actualIndexes.indices.firstOrNull { actualIndexes[it] != expectedIndexes[it] } ?: -1
        val mismatchCount = actualIndexes.indices.count { actualIndexes[it] != expectedIndexes[it] }
        val exact = mismatchCount == 0
        emit(
            "i_prior_tflite_precision tensor=${stage}_cdf_indexes exact=$exact " +
                "mismatch_count=$mismatchCount first_diff=$firstDifference",
        )
        return exact
    }

    private fun meanAbs(first: FloatArray, second: FloatArray): Float {
        require(first.size == second.size) { "cannot compare tensors with different sizes" }
        var sum = 0.0
        first.indices.forEach { index -> sum += kotlin.math.abs(first[index] - second[index]) }
        return (sum / first.size).toFloat()
    }

    private fun directFp32Models(): List<SpeedModel> {
        if (store.exists(DIRECT_STAGE1_ASSET)) {
            return listOf(
                SpeedModel(
                    "stage1_direct_fp32",
                    DIRECT_STAGE1_ASSET,
                    listOf(
                        PriorNpuTensorCodec.nchwToNhwcF32(tensor("i_y_hat_so_far_0", Y_SHAPE).data, 256, 16, 32),
                        PriorNpuTensorCodec.nchwToNhwcF32(reduced.data, 256, 16, 32),
                    ),
                ),
            )
        }
        val reduceAsset = "$ASSET_DIR/i_prior_reduce_direct_nhwc_fp32.tflite"
        val stageAssets = (1..3).map { "$ASSET_DIR/i_prior_stage${it}_fp16_weight.tflite" }
        if (!store.exists(reduceAsset) || stageAssets.any { !store.exists(it) }) return emptyList()
        val yHat = (0..2).map { stage -> tensor("i_y_hat_so_far_$stage", Y_SHAPE) }
        return listOf(
            SpeedModel(
                "reduce",
                reduceAsset,
                listOf(PriorNpuTensorCodec.nchwToNhwcF32(common.data, 514, 16, 32)),
            ),
            *stageAssets.mapIndexed { index, asset ->
                SpeedModel(
                    "stage${index + 1}",
                    asset,
                    listOf(
                        PriorNpuTensorCodec.nchwToNhwcF32(yHat[index].data, 256, 16, 32),
                        PriorNpuTensorCodec.nchwToNhwcF32(reduced.data, 256, 16, 32),
                    ),
                )
            }.toTypedArray(),
        )
    }

    private fun emitSummary(stage: String, values: LongArray) {
        val ordered = values.sortedArray()
        val mean = values.average() / 1_000_000.0
        val p50 = ordered[(ordered.size - 1) / 2] / 1_000_000.0
        val p90 = ordered[((ordered.size - 1) * 90) / 100] / 1_000_000.0
        emit(
            "i_prior_tflite_speed stage=$stage samples=${values.size} mean_ms=${formatMs(mean)} " +
                "p50_ms=${formatMs(p50)} p90_ms=${formatMs(p90)}",
        )
    }

    private fun formatMs(value: Double): String = String.format(Locale.US, "%.3f", value)
    private fun formatValue(value: Float): String = String.format(Locale.US, "%.8f", value)

    private data class SpeedModel(
        val stage: String,
        val asset: String,
        val inputs: List<ByteArray>,
    )

    private fun tensor(name: String, shape: LongArray, path: String = "baseline/tensors/$name.f32le"): TensorValue =
        TensorIO.readF32Le(name, shape, store.readBytes(path))

    companion object {
        private const val MANIFEST = "gvcrt_clean_manifest.json"
        private const val ASSET_DIR = "prior_npu_diagnostic"
        private const val DIRECT_STAGE1_ASSET = "$ASSET_DIR/i_prior_stage1_direct_nhwc_fp32.tflite"
        private val COMMON_SHAPE = longArrayOf(1, 514, 16, 32)
        private val Y_SHAPE = longArrayOf(1, 256, 16, 32)
        private val REDUCED_SHAPE = longArrayOf(1, 256, 16, 32)
        private val STAGE_SHAPE = longArrayOf(1, 64, 16, 32)
        private const val FORCE_ZERO_THRESHOLD = 0.12f
        private const val CONTINUOUS_Y_HAT_MAX_ABS = 1e-4f
    }
}
