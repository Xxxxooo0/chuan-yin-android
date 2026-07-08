package com.gvcrt.clean

import android.content.Context
import android.os.SystemClock
import java.io.File
import java.util.Locale
import kotlin.math.ceil

class ReconDiagnosticBenchmark(
    private val context: Context,
    private val emit: (String) -> Unit,
) {
    private val store = AssetStore(context)
    private val cacheDir = File(context.filesDir, "mtk_tflite_cache")

    fun run(
        labelFilter: String? = null,
        createOnly: Boolean = false,
        copyOutputs: Boolean = true,
        accelerationMode: Int = MtkTfliteRuntime.ACCELERATION_NEURON,
        warmupRuns: Int = WARMUP_RUNS,
        measuredRuns: Int = MEASURED_RUNS,
    ) {
        emit(
            "recon_diag_start route=mtk_tflite warmup=$warmupRuns measured=$measuredRuns " +
                "label_filter=${labelFilter ?: "all"} create_only=$createOnly copy_outputs=$copyOutputs " +
                "acceleration_mode=$accelerationMode"
        )
        if (labelFilter == NATIVE_PIXEL_UNSHUFFLE_LABEL) {
            runNativePixelUnshuffle(warmupRuns, measuredRuns)
            emit("recon_diag_complete")
            return
        }
        if (labelFilter == NATIVE_DEPTH_TO_SPACE_LABEL) {
            runNativeDepthToSpace(warmupRuns, measuredRuns)
            emit("recon_diag_complete")
            return
        }
        if (labelFilter == OPENCL_FUSED_UPSAMPLER_LABEL) {
            runOpenClFusedUpsampler(warmupRuns, measuredRuns)
            emit("recon_diag_complete")
            return
        }
        if (labelFilter == VULKAN_FUSED_UPSAMPLER_LABEL) {
            runVulkanFusedUpsampler(warmupRuns, measuredRuns)
            emit("recon_diag_complete")
            return
        }
        if (labelFilter == NATIVE_WSILU_CHUNK_ADD_LABEL) {
            runNativeWSiLUChunkAdd(warmupRuns, measuredRuns)
            emit("recon_diag_complete")
            return
        }
        if (labelFilter == NATIVE_FAST_WSILU_CHUNK_ADD_LABEL) {
            runNativeFastWSiLUChunkAdd(warmupRuns, measuredRuns)
            emit("recon_diag_complete")
            return
        }
        if (labelFilter == NATIVE_NEURON_EXTENSION_PROBE_LABEL) {
            runNativeNeuronExtensionProbe()
            emit("recon_diag_complete")
            return
        }
        if (labelFilter == NATIVE_AHWB_SYMBOL_PROBE_LABEL) {
            runNativeAhwbSymbolProbe()
            emit("recon_diag_complete")
            return
        }
        if (labelFilter == NATIVE_FUNCTIONAL_OPS_PROBE_LABEL) {
            runNativeFunctionalOpsProbe()
            emit("recon_diag_complete")
            return
        }
        if (labelFilter == NATIVE_P_RECON_PIPELINE_PROBE_LABEL) {
            runNativePReconPipelineProbe(
                label = NATIVE_P_RECON_PIPELINE_PROBE_LABEL,
                modelAssets = P_RECON_BASELINE_MODELS,
                warmupRuns = warmupRuns,
                measuredRuns = measuredRuns,
            )
            emit("recon_diag_complete")
            return
        }
        if (labelFilter == NATIVE_P_RECON_PRECISION_PROBE_LABEL) {
            runNativePReconPrecisionProbe()
            emit("recon_diag_complete")
            return
        }
        if (labelFilter == NATIVE_P_RECON_STAGE_PRECISION_PROBE_LABEL) {
            runNativePReconStagePrecisionProbe()
            emit("recon_diag_complete")
            return
        }
        if (labelFilter == NATIVE_P_RECON_PIPELINE_STAGE3_FP16_LABEL) {
            runNativePReconPipelineProbe(
                label = NATIVE_P_RECON_PIPELINE_STAGE3_FP16_LABEL,
                modelAssets = P_RECON_STAGE3_FP16_MODELS,
                warmupRuns = warmupRuns,
                measuredRuns = measuredRuns,
            )
            emit("recon_diag_complete")
            return
        }
        if (labelFilter == NATIVE_P_RECON_PIPELINE_STAGE4_FP16_LABEL) {
            runNativePReconPipelineProbe(
                label = NATIVE_P_RECON_PIPELINE_STAGE4_FP16_LABEL,
                modelAssets = P_RECON_STAGE4_FP16_MODELS,
                warmupRuns = warmupRuns,
                measuredRuns = measuredRuns,
            )
            emit("recon_diag_complete")
            return
        }
        if (labelFilter == NATIVE_P_RECON_PIPELINE_HOTSPOT_FP16_LABEL) {
            runNativePReconPipelineProbe(
                label = NATIVE_P_RECON_PIPELINE_HOTSPOT_FP16_LABEL,
                modelAssets = P_RECON_HOTSPOT_FP16_MODELS,
                warmupRuns = warmupRuns,
                measuredRuns = measuredRuns,
            )
            emit("recon_diag_complete")
            return
        }
        if (labelFilter == NATIVE_P_RECON_MIXED_MERGED_PROBE_LABEL) {
            runNativePReconMixedMergedProbe(warmupRuns, measuredRuns)
            emit("recon_diag_complete")
            return
        }
        if (labelFilter == NATIVE_P_RECON_BIG_PIPELINE_PROBE_LABEL) {
            runNativePReconBigPipelineProbe(warmupRuns, measuredRuns)
            emit("recon_diag_complete")
            return
        }
        if (labelFilter == RECON_NPU_MATRIX_LABEL) {
            runReconNpuMatrix(warmupRuns, measuredRuns)
            emit("recon_diag_complete")
            return
        }
        if (labelFilter == RECON_MTK_ACCELERATOR_MATRIX_LABEL) {
            runReconMtkAcceleratorMatrix(warmupRuns, measuredRuns)
            emit("recon_diag_complete")
            return
        }
        if (labelFilter == RECON_OFFICIAL_NEURON_MATRIX_LABEL) {
            runReconOfficialNeuronMatrix(warmupRuns, measuredRuns)
            emit("recon_diag_complete")
            return
        }
        if (labelFilter == RECON_GPU_BIG_MATRIX_LABEL) {
            runReconGpuBigMatrix(warmupRuns, measuredRuns)
            emit("recon_diag_complete")
            return
        }
        if (labelFilter == P_RECON_GPU_BIG_PIPELINE_LABEL) {
            runPReconGpuBigPipeline(warmupRuns, measuredRuns)
            emit("recon_diag_complete")
            return
        }
        if (labelFilter == NATIVE_DLA_RUNTIME_PROBE_LABEL) {
            runNativeDlaRuntimeProbe(
                label = NATIVE_DLA_RUNTIME_PROBE_LABEL,
                asset = "recon_diagnostic/p_latent_decoder_fp32.dla",
            )
            emit("recon_diag_complete")
            return
        }
        if (labelFilter == NATIVE_STAGE3_DLA_RUNTIME_PROBE_LABEL) {
            runNativeDlaRuntimeProbe(
                label = NATIVE_STAGE3_DLA_RUNTIME_PROBE_LABEL,
                asset = "recon_diagnostic/p_decoder_stage3_blocks_only_fp16_weight.dla",
            )
            emit("recon_diag_complete")
            return
        }
        if (labelFilter == NATIVE_PIXEL_UNSHUFFLE_ADAPTOR_LABEL) {
            runNativePixelUnshuffleAdaptor(NATIVE_PIXEL_UNSHUFFLE_ADAPTOR_LABEL, 8, warmupRuns, measuredRuns)
            emit("recon_diag_complete")
            return
        }
        if (labelFilter == DCB_NPU_SWEEP_LABEL) {
            runDcbNpuSweep(warmupRuns, measuredRuns)
            emit("recon_diag_complete")
            return
        }
        if (labelFilter != null) {
            NATIVE_PIXEL_UNSHUFFLE_ADAPTOR_THREAD_LABELS[labelFilter]?.let { threadCount ->
                runNativePixelUnshuffleAdaptor(labelFilter, threadCount, warmupRuns, measuredRuns)
                emit("recon_diag_complete")
                return
            }
            NATIVE_GROUPNORM512_THREAD_LABELS[labelFilter]?.let { threadCount ->
                runNativeGroupNorm512(labelFilter, threadCount, warmupRuns, measuredRuns)
                emit("recon_diag_complete")
                return
            }
            NATIVE_ADAGN512_STAGE1_THREAD_LABELS[labelFilter]?.let { threadCount ->
                runNativeAdaGn512Stage1(labelFilter, threadCount, warmupRuns, measuredRuns)
                emit("recon_diag_complete")
                return
            }
            NATIVE_ADAGN_LABELS[labelFilter]?.let { spec ->
                runNativeAdaGn(labelFilter, spec, warmupRuns, measuredRuns)
                emit("recon_diag_complete")
                return
            }
        }
        if (labelFilter == null) {
            runNativePReconPipelineProbe(
                label = NATIVE_P_RECON_PIPELINE_PROBE_LABEL,
                modelAssets = P_RECON_BASELINE_MODELS,
                warmupRuns = warmupRuns,
                measuredRuns = measuredRuns,
            )
            emit("recon_diag_complete")
            return
        }
        var found = false
        for (variant in ALL_VARIANTS) {
            for (model in MODELS) {
                val label = "${model.name}_$variant"
                if (label != labelFilter) {
                    continue
                }
                val asset = "recon_diagnostic/${model.name}_$variant.tflite"
                if (!store.exists(asset)) {
                    emit("recon_diag_skip label=$label reason=missing_asset")
                    continue
                }
                found = true
                runModel(
                    model = model,
                    variant = variant,
                    asset = asset,
                    createOnly = createOnly,
                    copyOutputs = copyOutputs,
                    accelerationMode = accelerationMode,
                    warmupRuns = warmupRuns,
                    measuredRuns = measuredRuns,
                )
            }
        }
        if (!found) {
            emit("recon_diag_no_assets expected=app/src/main/assets/recon_diagnostic/*.tflite")
        }
        emit("recon_diag_complete")
    }

    private fun runReconNpuMatrix(warmupRuns: Int, measuredRuns: Int) {
        emit(
            "recon_diag_npu_matrix_start backend=mtk_neuron warmup=$warmupRuns measured=$measuredRuns " +
                "copy_outputs=false variants=fp32,fp16_weight"
        )
        val results = ArrayList<ModelRunStats>()
        for (modelName in RECON_NPU_MATRIX_MODELS) {
            val model = MODELS.firstOrNull { it.name == modelName }
            if (model == null) {
                emit("recon_diag_npu_matrix_skip label=$modelName reason=missing_model_spec")
                continue
            }
            for (variant in RECON_MATRIX_VARIANTS) {
                val asset = "recon_diagnostic/${model.name}_$variant.tflite"
                if (!store.exists(asset)) {
                    emit("recon_diag_npu_matrix_skip label=${model.name}_$variant reason=missing_asset")
                    continue
                }
                runModel(
                    model = model,
                    variant = variant,
                    asset = asset,
                    createOnly = false,
                    copyOutputs = false,
                    accelerationMode = MtkTfliteRuntime.ACCELERATION_NEURON,
                    acceleratorFlag = MtkTfliteRuntime.ACCELERATOR_AUTO,
                    warmupRuns = warmupRuns,
                    measuredRuns = measuredRuns,
                )?.let { results += it }
            }
        }
        emitMatrixSummary("recon_diag_npu_matrix_summary", results)
    }

    private fun runReconMtkAcceleratorMatrix(warmupRuns: Int, measuredRuns: Int) {
        emit(
            "recon_diag_mtk_accelerator_matrix_start warmup=$warmupRuns measured=$measuredRuns " +
                "copy_outputs=false accelerators=${RECON_ACCELERATOR_FLAGS.keys.joinToString(",")}"
        )
        val results = ArrayList<ModelRunStats>()
        for ((acceleratorName, acceleratorFlag) in RECON_ACCELERATOR_FLAGS) {
            for (modelName in RECON_ACCELERATOR_MATRIX_MODELS) {
                val model = MODELS.firstOrNull { it.name == modelName }
                if (model == null) {
                    emit("recon_diag_mtk_accelerator_skip accelerator=$acceleratorName label=$modelName reason=missing_model_spec")
                    continue
                }
                val variant = acceleratorVariantFor(model.name)
                val asset = "recon_diagnostic/${model.name}_$variant.tflite"
                if (!store.exists(asset)) {
                    emit("recon_diag_mtk_accelerator_skip accelerator=$acceleratorName label=${model.name}_$variant reason=missing_asset")
                    continue
                }
                runModel(
                    model = model,
                    variant = variant,
                    asset = asset,
                    createOnly = false,
                    copyOutputs = false,
                    accelerationMode = MtkTfliteRuntime.ACCELERATION_NEURON,
                    acceleratorFlag = acceleratorFlag,
                    resultLabelPrefix = "${acceleratorName}_",
                    warmupRuns = warmupRuns,
                    measuredRuns = measuredRuns,
                )?.let { results += it }
            }
        }
        emitMatrixSummary("recon_diag_mtk_accelerator_matrix_summary", results)
    }

    private fun runReconOfficialNeuronMatrix(warmupRuns: Int, measuredRuns: Int) {
        emit(
            "recon_diag_official_neuron_matrix_start backend=official_aar_neuron " +
                "warmup=$warmupRuns measured=$measuredRuns copy_outputs=false"
        )
        val results = ArrayList<ModelRunStats>()
        for (modelName in RECON_OFFICIAL_NEURON_MATRIX_MODELS) {
            val model = MODELS.firstOrNull { it.name == modelName }
            if (model == null) {
                emit("recon_diag_official_neuron_skip label=$modelName reason=missing_model_spec")
                continue
            }
            val variant = acceleratorVariantFor(model.name)
            val asset = "recon_diagnostic/${model.name}_$variant.tflite"
            if (!store.exists(asset)) {
                emit("recon_diag_official_neuron_skip label=${model.name}_$variant reason=missing_asset")
                continue
            }
            runOfficialNeuronModel(
                model = model,
                variant = variant,
                asset = asset,
                copyOutputs = false,
                warmupRuns = warmupRuns,
                measuredRuns = measuredRuns,
            )?.let { results += it }
        }
        emitMatrixSummary("recon_diag_official_neuron_matrix_summary", results)
    }

    private fun acceleratorVariantFor(modelName: String): String =
        when (modelName) {
            "p_decoder_stage3_blocks_only",
            "p_decoder_stage4_blocks_explicit",
            "p_recon_final_head_no_ada" -> "fp16_weight"
            else -> "fp32"
        }

    private fun runReconGpuBigMatrix(warmupRuns: Int, measuredRuns: Int) {
        val available = TfliteGpuRuntime.isAvailable()
        emit(
            "recon_diag_gpu_matrix_start backend=tflite_gpu gpu_delegate_available=$available " +
                "gpu_delegate_forced=${!available} warmup=$warmupRuns measured=$measuredRuns " +
                "copy_outputs=false variants=fp32,fp16_weight"
        )
        val results = ArrayList<ModelRunStats>()
        for (modelName in RECON_GPU_BIG_MODELS) {
            val model = MODELS.firstOrNull { it.name == modelName }
            if (model == null) {
                emit("recon_diag_gpu_matrix_skip label=$modelName reason=missing_model_spec")
                continue
            }
            for (variant in RECON_MATRIX_VARIANTS) {
                val asset = "recon_diagnostic/${model.name}_$variant.tflite"
                if (!store.exists(asset)) {
                    emit("recon_diag_gpu_matrix_skip label=${model.name}_$variant reason=missing_asset")
                    continue
                }
                runGpuModel(
                    model = model,
                    variant = variant,
                    asset = asset,
                    copyOutputs = false,
                    warmupRuns = warmupRuns,
                    measuredRuns = measuredRuns,
                )?.let { results += it }
            }
        }
        emitMatrixSummary("recon_diag_gpu_matrix_summary", results)
    }

    private fun runDcbNpuSweep(warmupRuns: Int, measuredRuns: Int) {
        emit(
            "recon_diag_dcb_sweep_start labels=${DCB_NPU_SWEEP_MODELS.joinToString(",")} " +
                "variant=fp32 acceleration_mode=${MtkTfliteRuntime.ACCELERATION_NEURON} copy_outputs=false"
        )
        val results = ArrayList<ModelRunStats>()
        for (modelName in DCB_NPU_SWEEP_MODELS) {
            val model = MODELS.firstOrNull { it.name == modelName }
            if (model == null) {
                emit("recon_diag_dcb_sweep_skip label=${modelName}_fp32 reason=missing_model_spec")
                continue
            }
            val asset = "recon_diagnostic/${model.name}_fp32.tflite"
            if (!store.exists(asset)) {
                emit("recon_diag_dcb_sweep_skip label=${model.name}_fp32 reason=missing_asset")
                continue
            }
            runModel(
                model = model,
                variant = "fp32",
                asset = asset,
                createOnly = false,
                copyOutputs = false,
                accelerationMode = MtkTfliteRuntime.ACCELERATION_NEURON,
                warmupRuns = warmupRuns,
                measuredRuns = measuredRuns,
            )?.let { results += it }
        }
        val summary = results.joinToString(";") {
            "${it.label}:mean=${fmtMs(it.meanNanos)},p50=${fmtMs(it.p50Nanos)}," +
                "p90=${fmtMs(it.p90Nanos)},delegated=${it.fullyDelegated}"
        }
        emit(
            "recon_diag_dcb_sweep_summary count=${results.size} " +
                "copy_outputs=false summary=$summary"
        )
    }

    private fun runModel(
        model: ModelSpec,
        variant: String,
        asset: String,
        createOnly: Boolean,
        copyOutputs: Boolean,
        accelerationMode: Int,
        acceleratorFlag: Int = MtkTfliteRuntime.ACCELERATOR_AUTO,
        resultLabelPrefix: String = "",
        warmupRuns: Int,
        measuredRuns: Int,
    ): ModelRunStats? {
        val label = "$resultLabelPrefix${model.name}_$variant"
        var createOk = false
        var invokeOk = false
        try {
            emit(
                "recon_diag_attempt label=$label backend=mtk_neuron asset=$asset " +
                    "copy_outputs=$copyOutputs acceleration_mode=$accelerationMode accelerator_flag=$acceleratorFlag"
            )
            val file = store.materialize(asset)
            val inputs = model.inputs.map { ByteArray(it.bytes(variant)) }
            val createStarted = SystemClock.elapsedRealtimeNanos()
            MtkTfliteRuntime.create(file, accelerationMode, acceleratorFlag, cacheDir).use { runtime ->
                createOk = true
                val createMs = fmtMs((SystemClock.elapsedRealtimeNanos() - createStarted).toDouble())
                emit(
                    "recon_diag_model label=$label fully_delegated=${runtime.fullyDelegated} " +
                        "acceleration_mode=$accelerationMode accelerator_flag=$acceleratorFlag " +
                        "create_ms=$createMs input_sizes=${runtime.inputSizes.joinToString(":")} " +
                        "output_sizes=${runtime.outputSizes.joinToString(":")} " +
                        "options=${runtime.optionsSummary}"
                )
                require(runtime.inputSizes.size == inputs.size) {
                    "$label input count mismatch runtime=${runtime.inputSizes.size} expected=${inputs.size}"
                }
                runtime.inputSizes.forEachIndexed { index, size ->
                    require(size == inputs[index].size.toLong()) {
                        "$label input[$index] bytes mismatch runtime=$size expected=${inputs[index].size}"
                    }
                }
                if (createOnly) {
                    emit("recon_diag_create_only label=$label")
                } else {
                    repeat(warmupRuns) {
                        runtime.run(inputs, copyOutputs)
                    }
                    val elapsed = ArrayList<Long>(measuredRuns)
                    var outputSizes = emptyList<Int>()
                    repeat(measuredRuns) {
                        val started = SystemClock.elapsedRealtimeNanos()
                        outputSizes = runtime.run(inputs, copyOutputs).map { it.size }
                        elapsed += SystemClock.elapsedRealtimeNanos() - started
                    }
                    invokeOk = true
                    emitSpeed(label, elapsed)
                    emit(
                        "recon_diag_output label=$label copy_outputs=$copyOutputs " +
                            "output_bytes=${outputSizes.joinToString(":")}"
                    )
                    val sorted = elapsed.sorted()
                    emit(
                        "recon_diag_result label=$label backend=mtk_neuron create_ok=true invoke_ok=true " +
                            "fully_delegated=${runtime.fullyDelegated} mean=${fmtMs(elapsed.average())} " +
                            "p50=${fmtMs(percentile(sorted, 0.50))} p90=${fmtMs(percentile(sorted, 0.90))}"
                    )
                    return ModelRunStats(
                        label = label,
                        fullyDelegated = runtime.fullyDelegated,
                        meanNanos = elapsed.average(),
                        p50Nanos = percentile(sorted, 0.50),
                        p90Nanos = percentile(sorted, 0.90),
                    )
                }
            }
        } catch (t: Throwable) {
            emit(
                "recon_diag_error label=$label asset=$asset create_ok=$createOk invoke_ok=$invokeOk " +
                    "error=${t.message ?: t.javaClass.simpleName}"
            )
        }
        return null
    }

    private fun runGpuModel(
        model: ModelSpec,
        variant: String,
        asset: String,
        copyOutputs: Boolean,
        warmupRuns: Int,
        measuredRuns: Int,
    ): ModelRunStats? {
        val label = "${model.name}_$variant"
        var createOk = false
        var invokeOk = false
        try {
            val file = store.materialize(asset)
            val inputs = model.inputs.map { ByteArray(it.bytes(variant)) }
            val createStarted = SystemClock.elapsedRealtimeNanos()
            TfliteGpuRuntime.create(file, force = true).use { runtime ->
                createOk = true
                val createMs = fmtMs((SystemClock.elapsedRealtimeNanos() - createStarted).toDouble())
                emit(
                    "recon_diag_gpu_model label=$label backend=tflite_gpu create_ok=true " +
                        "create_ms=$createMs input_sizes=${runtime.inputSizes.joinToString(":")} " +
                        "output_sizes=${runtime.outputSizes.joinToString(":")} options=${runtime.delegateOptions}"
                )
                require(runtime.inputSizes.size == inputs.size) {
                    "$label input count mismatch runtime=${runtime.inputSizes.size} expected=${inputs.size}"
                }
                runtime.inputSizes.forEachIndexed { index, size ->
                    require(size == inputs[index].size.toLong()) {
                        "$label input[$index] bytes mismatch runtime=$size expected=${inputs[index].size}"
                    }
                }
                repeat(warmupRuns) {
                    runtime.run(inputs, copyOutputs)
                }
                val elapsed = ArrayList<Long>(measuredRuns)
                var outputSizes = emptyList<Int>()
                repeat(measuredRuns) {
                    val started = SystemClock.elapsedRealtimeNanos()
                    outputSizes = runtime.run(inputs, copyOutputs).map { it.size }
                    elapsed += SystemClock.elapsedRealtimeNanos() - started
                }
                invokeOk = true
                val sorted = elapsed.sorted()
                emitSpeed(label, elapsed)
                emit(
                    "recon_diag_gpu_output label=$label copy_outputs=$copyOutputs " +
                        "output_bytes=${outputSizes.joinToString(":")}"
                )
                emit(
                    "recon_diag_result label=$label backend=tflite_gpu create_ok=true invoke_ok=true " +
                        "fully_delegated=not_available mean=${fmtMs(elapsed.average())} " +
                        "p50=${fmtMs(TfliteGpuRuntime.percentile(sorted, 0.50))} " +
                        "p90=${fmtMs(TfliteGpuRuntime.percentile(sorted, 0.90))}"
                )
                return ModelRunStats(
                    label = label,
                    fullyDelegated = false,
                    meanNanos = elapsed.average(),
                    p50Nanos = TfliteGpuRuntime.percentile(sorted, 0.50),
                    p90Nanos = TfliteGpuRuntime.percentile(sorted, 0.90),
                )
            }
        } catch (t: Throwable) {
            emit(
                "recon_diag_gpu_error label=$label asset=$asset create_ok=$createOk invoke_ok=$invokeOk " +
                    "error=${t.message ?: t.javaClass.simpleName}"
            )
        }
        return null
    }

    private fun runOfficialNeuronModel(
        model: ModelSpec,
        variant: String,
        asset: String,
        copyOutputs: Boolean,
        warmupRuns: Int,
        measuredRuns: Int,
    ): ModelRunStats? {
        val label = "official_${model.name}_$variant"
        var createOk = false
        var invokeOk = false
        try {
            emit("recon_diag_official_attempt label=$label asset=$asset copy_outputs=$copyOutputs")
            val file = store.materialize(asset)
            val inputs = model.inputs.map { ByteArray(it.bytes(variant)) }
            val createStarted = SystemClock.elapsedRealtimeNanos()
            OfficialNeuronRuntime.create(file, File(context.filesDir, "official_neuron_cache")).use { runtime ->
                createOk = true
                val createMs = fmtMs((SystemClock.elapsedRealtimeNanos() - createStarted).toDouble())
                emit(
                    "recon_diag_official_model label=$label create_ok=true create_ms=$createMs " +
                        "input_sizes=${runtime.inputSizes.joinToString(":")} " +
                        "output_sizes=${runtime.outputSizes.joinToString(":")} options=${runtime.optionsSummary}"
                )
                require(runtime.inputSizes.size == inputs.size) {
                    "$label input count mismatch runtime=${runtime.inputSizes.size} expected=${inputs.size}"
                }
                runtime.inputSizes.forEachIndexed { index, size ->
                    require(size == inputs[index].size.toLong()) {
                        "$label input[$index] bytes mismatch runtime=$size expected=${inputs[index].size}"
                    }
                }
                repeat(warmupRuns) {
                    runtime.run(inputs, copyOutputs)
                }
                val elapsed = ArrayList<Long>(measuredRuns)
                var outputSizes = emptyList<Int>()
                repeat(measuredRuns) {
                    val started = SystemClock.elapsedRealtimeNanos()
                    outputSizes = runtime.run(inputs, copyOutputs).map { it.size }
                    elapsed += SystemClock.elapsedRealtimeNanos() - started
                }
                invokeOk = true
                val sorted = elapsed.sorted()
                emitSpeed(label, elapsed)
                emit(
                    "recon_diag_official_output label=$label copy_outputs=$copyOutputs " +
                        "output_bytes=${outputSizes.joinToString(":")}"
                )
                emit(
                    "recon_diag_result label=$label backend=official_aar_neuron create_ok=true invoke_ok=true " +
                        "fully_delegated=not_reported mean=${fmtMs(elapsed.average())} " +
                        "p50=${fmtMs(percentile(sorted, 0.50))} p90=${fmtMs(percentile(sorted, 0.90))}"
                )
                return ModelRunStats(
                    label = label,
                    fullyDelegated = false,
                    meanNanos = elapsed.average(),
                    p50Nanos = percentile(sorted, 0.50),
                    p90Nanos = percentile(sorted, 0.90),
                )
            }
        } catch (t: Throwable) {
            emit(
                "recon_diag_official_error label=$label asset=$asset create_ok=$createOk invoke_ok=$invokeOk " +
                    "error=${t.message ?: t.javaClass.simpleName}"
            )
        }
        return null
    }

    private fun runPReconGpuBigPipeline(warmupRuns: Int, measuredRuns: Int) {
        val available = TfliteGpuRuntime.isAvailable()
        emit(
            "recon_diag_gpu_big_pipeline_start label=$P_RECON_GPU_BIG_PIPELINE_LABEL " +
                "backend=tflite_gpu gpu_delegate_available=$available gpu_delegate_forced=${!available} " +
                "warmup=$warmupRuns measured=$measuredRuns"
        )
        val assets = listOf(
            "recon_diagnostic/p_recon_big_latent_mlp_fp32.tflite",
            "recon_diagnostic/p_recon_big_stage1_stage2_fp32.tflite",
            "recon_diagnostic/p_recon_big_upsample_stage3_fp32.tflite",
            "recon_diagnostic/p_recon_big_stage4_final_fp32.tflite",
        )
        val labels = listOf(
            "p_recon_big_latent_mlp",
            "p_recon_big_stage1_stage2",
            "p_recon_big_upsample_stage3",
            "p_recon_big_stage4_final",
        )
        for (asset in assets) {
            if (!store.exists(asset)) {
                emit("recon_diag_gpu_big_pipeline_summary create_ok=false invoke_ok=false reason=missing_asset asset=$asset")
                return
            }
        }
        var createOk = false
        var invokeOk = false
        try {
            val createStarted = SystemClock.elapsedRealtimeNanos()
            val runtimes = assets.map { TfliteGpuRuntime.create(store.materialize(it), force = true) }
            runtimes.useAll {
                createOk = true
                val createMs = fmtMs((SystemClock.elapsedRealtimeNanos() - createStarted).toDouble())
                runtimes.forEachIndexed { index, runtime ->
                    emit(
                        "recon_diag_gpu_big_pipeline_model stage=${labels[index]} create_ok=true " +
                            "input_sizes=${runtime.inputSizes.joinToString(":")} " +
                            "output_sizes=${runtime.outputSizes.joinToString(":")} options=${runtime.delegateOptions}"
                    )
                }
                emit("recon_diag_gpu_big_pipeline_created create_ms=$createMs")
                val pYHat = ByteArray((128 * 16 * 32) * 4)
                val pCtx = ByteArray((256 * 32 * 64) * 4)
                val qRecon = floatBytes(1.0f)
                repeat(warmupRuns) {
                    runGpuBigPipelineOnce(runtimes, pYHat, pCtx, qRecon)
                }
                val totals = ArrayList<Long>(measuredRuns)
                val stageTimings = LongArray(4)
                var frame = emptyList<ByteArray>()
                repeat(measuredRuns) {
                    val totalStarted = SystemClock.elapsedRealtimeNanos()
                    var codeword: List<ByteArray>
                    var stage2: List<ByteArray>
                    var stage3: List<ByteArray>
                    val s0 = SystemClock.elapsedRealtimeNanos()
                    codeword = runtimes[0].run(listOf(pYHat, pCtx), copyOutputs = true)
                    stageTimings[0] += SystemClock.elapsedRealtimeNanos() - s0
                    val s1 = SystemClock.elapsedRealtimeNanos()
                    stage2 = runtimes[1].run(codeword, copyOutputs = true)
                    stageTimings[1] += SystemClock.elapsedRealtimeNanos() - s1
                    val s2 = SystemClock.elapsedRealtimeNanos()
                    stage3 = runtimes[2].run(listOf(stage2[0], codeword[0]), copyOutputs = true)
                    stageTimings[2] += SystemClock.elapsedRealtimeNanos() - s2
                    val s3 = SystemClock.elapsedRealtimeNanos()
                    frame = runtimes[3].run(listOf(stage3[0], codeword[0], qRecon), copyOutputs = true)
                    stageTimings[3] += SystemClock.elapsedRealtimeNanos() - s3
                    totals += SystemClock.elapsedRealtimeNanos() - totalStarted
                }
                invokeOk = true
                val sorted = totals.sorted()
                emitSpeed(P_RECON_GPU_BIG_PIPELINE_LABEL, totals)
                labels.forEachIndexed { index, label ->
                    emit("recon_diag_gpu_big_pipeline_stage stage=$label mean=${fmtMs(stageTimings[index].toDouble() / measuredRuns)}")
                }
                emit(
                    "recon_diag_result label=$P_RECON_GPU_BIG_PIPELINE_LABEL backend=tflite_gpu " +
                        "create_ok=true invoke_ok=true fully_delegated=not_available " +
                        "mean=${fmtMs(totals.average())} p50=${fmtMs(TfliteGpuRuntime.percentile(sorted, 0.50))} " +
                        "p90=${fmtMs(TfliteGpuRuntime.percentile(sorted, 0.90))} output_bytes=${frame.joinToString(":") { it.size.toString() }}"
                )
            }
        } catch (t: Throwable) {
            emit(
                "recon_diag_gpu_big_pipeline_error create_ok=$createOk invoke_ok=$invokeOk " +
                    "error=${t.message ?: t.javaClass.simpleName}"
            )
        }
    }

    private fun runGpuBigPipelineOnce(
        runtimes: List<TfliteGpuRuntime>,
        pYHat: ByteArray,
        pCtx: ByteArray,
        qRecon: ByteArray,
    ): List<ByteArray> {
        val codeword = runtimes[0].run(listOf(pYHat, pCtx), copyOutputs = true)
        val stage2 = runtimes[1].run(codeword, copyOutputs = true)
        val stage3 = runtimes[2].run(listOf(stage2[0], codeword[0]), copyOutputs = true)
        return runtimes[3].run(listOf(stage3[0], codeword[0], qRecon), copyOutputs = true)
    }

    private fun List<AutoCloseable>.useAll(block: () -> Unit) {
        try {
            block()
        } finally {
            asReversed().forEach { it.close() }
        }
    }

    private fun floatBytes(value: Float): ByteArray =
        java.nio.ByteBuffer.allocate(4)
            .order(java.nio.ByteOrder.nativeOrder())
            .putFloat(value)
            .array()

    private fun emitMatrixSummary(prefix: String, results: List<ModelRunStats>) {
        val best = results.minByOrNull { it.meanNanos }
        val summary = results.joinToString(";") {
            "${it.label}:mean=${fmtMs(it.meanNanos)},p50=${fmtMs(it.p50Nanos)}," +
                "p90=${fmtMs(it.p90Nanos)},delegated=${it.fullyDelegated}"
        }
        emit(
            "$prefix count=${results.size} best=${best?.label ?: "none"} " +
                "best_mean=${best?.let { fmtMs(it.meanNanos) } ?: "na"} summary=$summary"
        )
    }

    private fun runNativePixelUnshuffle(warmupRuns: Int, measuredRuns: Int) {
        emit(
            "recon_diag_model label=$NATIVE_PIXEL_UNSHUFFLE_LABEL route=native_cpp " +
                "input_shape=1,256,32,64 output_shape=1,1024,16,32"
        )
        val elapsed = MtkTfliteRuntime.benchmarkNativePixelUnshuffle2(warmupRuns, measuredRuns).toList()
        emitSpeed(NATIVE_PIXEL_UNSHUFFLE_LABEL, elapsed)
        emit("recon_diag_output label=$NATIVE_PIXEL_UNSHUFFLE_LABEL copy_outputs=false output_bytes=")
    }

    private fun runNativeDepthToSpace(warmupRuns: Int, measuredRuns: Int) {
        emit(
            "recon_diag_model label=$NATIVE_DEPTH_TO_SPACE_LABEL route=native_cpp " +
                "input_shape=1,2048,16,32 output_shape=1,512,32,64"
        )
        val elapsed = MtkTfliteRuntime.benchmarkNativeDepthToSpace2(warmupRuns, measuredRuns).toList()
        emitSpeed(NATIVE_DEPTH_TO_SPACE_LABEL, elapsed)
        emit("recon_diag_output label=$NATIVE_DEPTH_TO_SPACE_LABEL copy_outputs=false output_bytes=")
    }

    private fun runOpenClFusedUpsampler(warmupRuns: Int, measuredRuns: Int) {
        emit(
            "recon_diag_model label=$OPENCL_FUSED_UPSAMPLER_LABEL route=opencl_native_fused " +
                "input_shape=1,512,16,32 output_shape=1,512,32,64 " +
                "op=conv3x3_512_to_2048_plus_pixelshuffle2 synthetic_weights=true"
        )
        MtkTfliteRuntime
            .benchmarkOpenClFusedUpsampler(warmupRuns, measuredRuns)
            .lines()
            .filter { it.isNotBlank() }
            .forEach { emit("recon_diag_opencl_fused_upsampler $it") }
        emit("recon_diag_output label=$OPENCL_FUSED_UPSAMPLER_LABEL copy_outputs=false output_bytes=")
    }

    private fun runVulkanFusedUpsampler(warmupRuns: Int, measuredRuns: Int) {
        emit(
            "recon_diag_model label=$VULKAN_FUSED_UPSAMPLER_LABEL route=vulkan_compute_fused " +
                "input_shape=1,512,16,32 output_shape=1,512,32,64 " +
                "op=conv3x3_512_to_2048_plus_pixelshuffle2 synthetic_weights=true"
        )
        MtkTfliteRuntime
            .benchmarkVulkanFusedUpsampler(warmupRuns, measuredRuns)
            .lines()
            .filter { it.isNotBlank() }
            .forEach { emit("recon_diag_vulkan_fused_upsampler $it") }
        emit("recon_diag_output label=$VULKAN_FUSED_UPSAMPLER_LABEL copy_outputs=false output_bytes=")
    }

    private fun runNativeWSiLUChunkAdd(warmupRuns: Int, measuredRuns: Int) {
        emit(
            "recon_diag_model label=$NATIVE_WSILU_CHUNK_ADD_LABEL route=native_cpp " +
                "input_shape=1,1280,32,64 output_shape=1,640,32,64"
        )
        val elapsed = MtkTfliteRuntime.benchmarkNativeWSiLUChunkAdd(warmupRuns, measuredRuns).toList()
        emitSpeed(NATIVE_WSILU_CHUNK_ADD_LABEL, elapsed)
        emit("recon_diag_output label=$NATIVE_WSILU_CHUNK_ADD_LABEL copy_outputs=false output_bytes=")
    }

    private fun runNativeFastWSiLUChunkAdd(warmupRuns: Int, measuredRuns: Int) {
        emit(
            "recon_diag_model label=$NATIVE_FAST_WSILU_CHUNK_ADD_LABEL route=native_cpp " +
                "input_shape=1,1280,32,64 output_shape=1,640,32,64 approximation=hard_sigmoid"
        )
        val elapsed = MtkTfliteRuntime.benchmarkNativeFastWSiLUChunkAdd(warmupRuns, measuredRuns).toList()
        emitSpeed(NATIVE_FAST_WSILU_CHUNK_ADD_LABEL, elapsed)
        emit("recon_diag_output label=$NATIVE_FAST_WSILU_CHUNK_ADD_LABEL copy_outputs=false output_bytes=")
    }

    private fun runNativeNeuronExtensionProbe() {
        val names = listOf(
            "GVC_RT_FUSED_DCB",
            "GVC_RT_FUSED_FFN",
            "GVC_RT_WSILU_CHUNK_ADD",
            "DCB",
            "FFN",
            "WSiLUChunkAdd",
            "MVPU",
            "MTKEXT",
        )
        emit(
            "recon_diag_model label=$NATIVE_NEURON_EXTENSION_PROBE_LABEL route=native_cpp " +
                "purpose=neuron_driver_extension_support names=${names.joinToString(":")}"
        )
        emit("recon_diag_extension_probe ${MtkTfliteRuntime.probeNeuronExtensions(names)}")
    }

    private fun runNativeAhwbSymbolProbe() {
        emit(
            "recon_diag_model label=$NATIVE_AHWB_SYMBOL_PROBE_LABEL route=native_cpp " +
                "purpose=tflite_shim_ahardwarebuffer_symbol_probe"
        )
        emit("recon_diag_ahwb_symbol_probe ${MtkTfliteRuntime.probeAhwbSymbols()}")
    }

    private fun runNativeFunctionalOpsProbe() {
        emit(
            "recon_diag_model label=$NATIVE_FUNCTIONAL_OPS_PROBE_LABEL route=native_cpp " +
                "purpose=functional_floatarray_bridge"
        )
        val pixelInput = FloatArray(256 * 32 * 64) { index -> ((index % 251) - 125) * 0.001f }
        val pixelStarted = SystemClock.elapsedRealtimeNanos()
        val pixelOutput = MtkTfliteRuntime.pixelUnshuffle2Nchw256(pixelInput)
        emit(
            "recon_diag_functional_op name=pixel_unshuffle2 output=${pixelOutput.size} " +
                "ms=${fmtMs((SystemClock.elapsedRealtimeNanos() - pixelStarted).toDouble())} " +
                "checksum=${fmtFloat(checksum(pixelOutput))}"
        )

        val gnInput = FloatArray(512 * 16 * 32) { index -> ((index % 127) - 63) * 0.002f }
        val gnStarted = SystemClock.elapsedRealtimeNanos()
        val gnOutput = MtkTfliteRuntime.groupNormNchw(gnInput, 512, 16, 32, 32, 1)
        emit(
            "recon_diag_functional_op name=groupnorm512 output=${gnOutput.size} " +
                "ms=${fmtMs((SystemClock.elapsedRealtimeNanos() - gnStarted).toDouble())} " +
                "checksum=${fmtFloat(checksum(gnOutput))}"
        )

        val weights = store.materialize("recon_diagnostic/p_decoder_ada1_weights.bin")
        val codeword = FloatArray(18 * 16 * 32) { index -> ((index % 113) - 56) * 0.002f }
        val adaStarted = SystemClock.elapsedRealtimeNanos()
        val adaOutput = MtkTfliteRuntime.adaGnNchw(gnInput, codeword, weights.absolutePath, 512, 16, 32, 1)
        emit(
            "recon_diag_functional_op name=adagn512 output=${adaOutput.size} " +
                "ms=${fmtMs((SystemClock.elapsedRealtimeNanos() - adaStarted).toDouble())} " +
                "checksum=${fmtFloat(checksum(adaOutput))}"
        )
    }

    private fun runNativePReconPipelineProbe(
        label: String,
        modelAssets: List<String>,
        warmupRuns: Int,
        measuredRuns: Int,
    ) {
        val missing = modelAssets
            .map { "recon_diagnostic/$it" }
            .filterNot { store.exists(it) }
        if (missing.isNotEmpty()) {
            emit("recon_diag_skip label=$label reason=missing_assets assets=${missing.joinToString(":")}")
            return
        }
        val modelPaths = modelAssets.map { store.materialize("recon_diagnostic/$it").absolutePath }
        val weightPaths = P_RECON_ADA_WEIGHT_ASSETS.map { store.materialize("recon_diagnostic/$it").absolutePath }
        emit(
            "recon_diag_model label=$label route=native_cpp_mtk_tflite " +
                "warmup=$warmupRuns measured=$measuredRuns models=${modelAssets.joinToString(":")}"
        )
        val report = MtkTfliteRuntime.runNativePReconPipelineProbe(
            modelPaths = modelPaths,
            adaWeightPaths = weightPaths,
            cacheDir = cacheDir,
            warmupRuns = warmupRuns,
            measuredRuns = measuredRuns,
        )
        report.lines().filter { it.isNotBlank() }.forEach { emit("recon_diag_pipeline $it") }
    }

    private fun runNativePReconPrecisionProbe() {
        val missing = P_RECON_BASELINE_MODELS
            .map { "recon_diagnostic/$it" }
            .filterNot { store.exists(it) }
        if (missing.isNotEmpty()) {
            emit("recon_precision_skip label=$NATIVE_P_RECON_PRECISION_PROBE_LABEL reason=missing_assets assets=${missing.joinToString(":")}")
            return
        }
        val modelPaths = P_RECON_BASELINE_MODELS.map { store.materialize("recon_diagnostic/$it").absolutePath }
        val weightPaths = P_RECON_ADA_WEIGHT_ASSETS.map { store.materialize("recon_diagnostic/$it").absolutePath }
        val pYHat = TensorIO.readF32Le(
            "p_y_hat",
            longArrayOf(1, 128, 16, 32),
            store.readBytes("baseline/tensors/p_y_hat.f32le"),
        )
        val pCtx = TensorIO.readF32Le(
            "p_ctx",
            longArrayOf(1, 256, 32, 64),
            store.readBytes("baseline/tensors/p_ctx.f32le"),
        )
        emit(
            "recon_precision_start label=$NATIVE_P_RECON_PRECISION_PROBE_LABEL " +
                "input=p_y_hat+p_ctx route=native_cpp_mtk_tflite"
        )
        val outputs = MtkTfliteRuntime.runNativePReconPipeline(
            modelPaths = modelPaths,
            adaWeightPaths = weightPaths,
            cacheDir = cacheDir,
            pYHat = pYHat.data,
            pCtx = pCtx.data,
        )
        require(outputs.size == 2) { "native P recon precision expected 2 outputs, got=${outputs.size}" }
        emitTensorDiff(
            "native_p_recon_reference_feature",
            TensorValue("native_p_recon_reference_feature", longArrayOf(1, 256, 32, 64), outputs[0]),
            TensorIO.readF32Le(
                "encoder_p_reference_feature",
                longArrayOf(1, 256, 32, 64),
                store.readBytes("baseline/tensors/encoder_p_reference_feature.f32le"),
            ),
        )
        emitTensorDiff(
            "native_p_recon_reference_frame",
            TensorValue("native_p_recon_reference_frame", longArrayOf(1, 3, 256, 512), outputs[1]),
            TensorIO.readF32Le(
                "encoder_p_reference_frame",
                longArrayOf(1, 3, 256, 512),
                store.readBytes("baseline/tensors/encoder_p_reference_frame.f32le"),
            ),
        )
    }

    private fun runNativePReconStagePrecisionProbe() {
        val missingAssets = listOf("recon_diagnostic/p_latent_decoder_fp32.tflite")
            .filterNot { store.exists(it) }
        val missingBaselines = listOf("baseline/recon_p_segments/p_reference_feature.f32le")
            .filterNot { store.exists(it) }
        if (missingAssets.isNotEmpty() || missingBaselines.isNotEmpty()) {
            emit(
                "recon_stage_precision_skip label=$NATIVE_P_RECON_STAGE_PRECISION_PROBE_LABEL " +
                    "missing_assets=${missingAssets.joinToString(":")} " +
                    "missing_baselines=${missingBaselines.joinToString(":")}"
            )
            return
        }

        val pYHat = TensorIO.readF32Le(
            "p_y_hat",
            longArrayOf(1, 128, 16, 32),
            store.readBytes("baseline/tensors/p_y_hat.f32le"),
        )
        val pCtx = TensorIO.readF32Le(
            "p_ctx",
            longArrayOf(1, 256, 32, 64),
            store.readBytes("baseline/tensors/p_ctx.f32le"),
        )
        emit("recon_stage_precision_start label=$NATIVE_P_RECON_STAGE_PRECISION_PROBE_LABEL stages=first_only")
        emit(
            "recon_stage_precision_asset model=p_latent_decoder_fp32 " +
                "sha256=${store.sha256("recon_diagnostic/p_latent_decoder_fp32.tflite")}"
        )
        val modelFile = store.materialize("recon_diagnostic/p_latent_decoder_fp32.tflite")
        val expected = TensorIO.readF32Le(
            "p_reference_feature",
            longArrayOf(1, 256, 32, 64),
            store.readBytes("baseline/recon_p_segments/p_reference_feature.f32le"),
        )
        val inputBytes = listOf(
            TensorIO.f32Le(pYHat),
            TensorIO.f32Le(pCtx),
        )
        val backends = listOf(
            "cpu" to MtkTfliteRuntime.ACCELERATION_CPU,
            "neuron" to MtkTfliteRuntime.ACCELERATION_NEURON,
        )
        var firstFailed: String? = null
        for ((backendName, mode) in backends) {
            MtkTfliteRuntime.create(
                modelFile,
                accelerationMode = mode,
                cacheDir = cacheDir,
            ).use { runtime ->
                emit(
                    "recon_stage_precision_model=p_latent_decoder_fp32 backend=$backendName " +
                        "fully_delegated=${runtime.fullyDelegated} options=${runtime.optionsSummary}"
                )
                val start = SystemClock.elapsedRealtimeNanos()
                val output = runtime.run(inputBytes).single()
                val elapsedMs = (SystemClock.elapsedRealtimeNanos() - start).toDouble() / 1_000_000.0
                emit("recon_stage_precision_invoke backend=$backendName elapsed_ms=${fmtMs(elapsedMs)}")
                val actual = TensorIO.readF32Le(
                    "native_p_reference_feature_$backendName",
                    longArrayOf(1, 256, 32, 64),
                    output,
                )
                emitTensorDiff("p_reference_feature_$backendName", actual, expected)
                val diff = TensorIO.diff(actual, expected)
                if (firstFailed == null && diff.maxAbs > 0.02f) {
                    firstFailed = "p_reference_feature_$backendName"
                }
            }
        }
        emit("recon_stage_precision_first_failed=${firstFailed ?: "none_in_first_stage"}")
    }

    private fun emitTensorDiff(name: String, actual: TensorValue, expected: TensorValue) {
        val diff = TensorIO.diff(actual, expected)
        emit(
            "recon_precision compare=$name pass=${diff.maxAbs <= 0.02f} " +
                "max_abs=${fmtFloat(diff.maxAbs.toDouble())} " +
                "mean_abs=${fmtFloat(diff.meanAbs.toDouble())} " +
                "rmse=${fmtFloat(diff.rmse.toDouble())} exact=${diff.exact}"
        )
    }

    private fun runNativePReconBigPipelineProbe(warmupRuns: Int, measuredRuns: Int) {
        val modelAssets = listOf(
            "p_recon_big_latent_mlp_fp32.tflite",
            "p_recon_big_stage1_stage2_fp32.tflite",
            "p_recon_big_upsample_stage3_fp32.tflite",
            "p_recon_big_stage4_final_fp32.tflite",
        )
        val modelPaths = modelAssets.map { store.materialize("recon_diagnostic/$it").absolutePath }
        emit(
            "recon_diag_model label=$NATIVE_P_RECON_BIG_PIPELINE_PROBE_LABEL route=native_cpp_mtk_tflite " +
                "warmup=$warmupRuns measured=$measuredRuns models=${modelAssets.joinToString(":")}"
        )
        val report = MtkTfliteRuntime.runNativePReconBigPipelineProbe(
            modelPaths = modelPaths,
            cacheDir = cacheDir,
            warmupRuns = warmupRuns,
            measuredRuns = measuredRuns,
        )
        report.lines().filter { it.isNotBlank() }.forEach { emit("recon_diag_big_pipeline $it") }
    }

    private fun runNativePReconMixedMergedProbe(warmupRuns: Int, measuredRuns: Int) {
        val modelAssets = listOf(
            "p_latent_decoder_fp32.tflite",
            "p_recon_mlp_dcb0_fp32.tflite",
            "p_recon_mlp_dcb1_fp32.tflite",
            "p_stage1_stage2_no_norm_fp32.tflite",
            "p_upsample_stage3_no_norm_fp32.tflite",
            "p_stage4_final_no_norm_fp32.tflite",
        )
        val weightAssets = listOf(
            "p_decoder_ada1_weights.bin",
            "p_decoder_ada2_weights.bin",
            "p_decoder_ada3_weights.bin",
            "p_decoder_ada4_weights.bin",
            "p_decoder_ada_final_weights.bin",
        )
        val modelPaths = modelAssets.map { store.materialize("recon_diagnostic/$it").absolutePath }
        val weightPaths = weightAssets.map { store.materialize("recon_diagnostic/$it").absolutePath }
        emit(
            "recon_diag_model label=$NATIVE_P_RECON_MIXED_MERGED_PROBE_LABEL route=native_cpp_mtk_tflite " +
                "non_equivalent_no_norm=true warmup=$warmupRuns measured=$measuredRuns models=${modelAssets.joinToString(":")}"
        )
        val report = MtkTfliteRuntime.runNativePReconMixedMergedProbe(
            modelPaths = modelPaths,
            adaWeightPaths = weightPaths,
            cacheDir = cacheDir,
            warmupRuns = warmupRuns,
            measuredRuns = measuredRuns,
        )
        report.lines().filter { it.isNotBlank() }.forEach { emit("recon_diag_mixed_merged $it") }
    }

    private fun runNativeDlaRuntimeProbe(label: String, asset: String) {
        val dla = store.materialize(asset)
        emit(
            "recon_diag_model label=$label route=native_neuron_runtime_v2 " +
                "dla=${dla.name} bytes=${dla.length()}"
        )
        emit("recon_diag_dla_probe label=$label ${MtkTfliteRuntime.probeDlaRuntime(dla.absolutePath)}")
    }

    private fun runNativePixelUnshuffleAdaptor(label: String, threadCount: Int, warmupRuns: Int, measuredRuns: Int) {
        val weights = store.materialize("recon_diagnostic/p_recon_mlp_dcb0_adaptor_weights.bin")
        emit(
            "recon_diag_model label=$label route=native_cpp " +
                "input_shape=1,256,32,64 output_shape=1,256,16,32 " +
                "weights=${weights.length()} threads=$threadCount"
        )
        val elapsed = MtkTfliteRuntime
            .benchmarkNativeFusedPixelUnshuffleAdaptor(weights.absolutePath, warmupRuns, measuredRuns, threadCount)
            .toList()
        emitSpeed(label, elapsed)
        emit("recon_diag_output label=$label copy_outputs=false output_bytes=")
    }

    private fun runNativeGroupNorm512(label: String, threadCount: Int, warmupRuns: Int, measuredRuns: Int) {
        emit(
            "recon_diag_model label=$label route=native_cpp " +
                "input_shape=1,512,16,32 output_shape=1,512,16,32 groups=32 threads=$threadCount"
        )
        val elapsed = MtkTfliteRuntime.benchmarkNativeGroupNorm512(warmupRuns, measuredRuns, threadCount).toList()
        emitSpeed(label, elapsed)
        emit("recon_diag_output label=$label copy_outputs=false output_bytes=")
    }

    private fun runNativeAdaGn512Stage1(label: String, threadCount: Int, warmupRuns: Int, measuredRuns: Int) {
        val weights = store.materialize("recon_diagnostic/p_decoder_ada1_weights.bin")
        emit(
            "recon_diag_model label=$label route=native_cpp " +
                "feature_shape=1,512,16,32 codeword_shape=1,18,16,32 " +
                "output_shape=1,512,16,32 weights=${weights.length()} threads=$threadCount"
        )
        val elapsed = MtkTfliteRuntime
            .benchmarkNativeAdaGn512Stage1(weights.absolutePath, warmupRuns, measuredRuns, threadCount)
            .toList()
        emitSpeed(label, elapsed)
        emit("recon_diag_output label=$label copy_outputs=false output_bytes=")
    }

    private fun runNativeAdaGn(label: String, spec: NativeAdaGnSpec, warmupRuns: Int, measuredRuns: Int) {
        val weights = store.materialize("recon_diagnostic/${spec.weightsAsset}")
        emit(
            "recon_diag_model label=$label route=native_cpp " +
                "feature_shape=1,${spec.channels},${spec.height},${spec.width} codeword_shape=1,18,16,32 " +
                "output_shape=1,${spec.channels},${spec.height},${spec.width} weights=${weights.length()} " +
                "threads=${spec.threadCount}"
        )
        val elapsed = MtkTfliteRuntime
            .benchmarkNativeAdaGn(
                weights.absolutePath,
                spec.channels,
                spec.height,
                spec.width,
                warmupRuns,
                measuredRuns,
                spec.threadCount,
            )
            .toList()
        emitSpeed(label, elapsed)
        emit("recon_diag_output label=$label copy_outputs=false output_bytes=")
    }

    private fun emitSpeed(label: String, values: List<Long>) {
        val sorted = values.sorted()
        val mean = values.average()
        val p50 = percentile(sorted, 0.50)
        val p90 = percentile(sorted, 0.90)
        emit(
            "recon_diag_speed label=$label samples=${values.size} " +
                "mean_ms=${fmtMs(mean)} p50_ms=${fmtMs(p50)} p90_ms=${fmtMs(p90)}"
        )
    }

    private fun percentile(sorted: List<Long>, quantile: Double): Double {
        val index = (ceil(sorted.size * quantile).toInt() - 1).coerceIn(0, sorted.lastIndex)
        return sorted[index].toDouble()
    }

    private fun fmtMs(nanos: Double): String = String.format(Locale.US, "%.3f", nanos / 1_000_000.0)

    private fun fmtFloat(value: Double): String = String.format(Locale.US, "%.6f", value)

    private fun checksum(values: FloatArray): Double {
        var sum = 0.0
        val step = (values.size / 4096).coerceAtLeast(1)
        var index = 0
        while (index < values.size) {
            sum += values[index].toDouble()
            index += step
        }
        return sum
    }

    private data class TensorSpec(val name: String, val shape: LongArray) {
        fun bytes(variant: String): Int {
            val bytesPerElement = if (variant == "fp16") 2 else 4
            return shape.fold(1L) { acc, value -> acc * value }.toInt() * bytesPerElement
        }
    }

    private data class ModelSpec(val name: String, val inputs: List<TensorSpec>)

    private data class ModelRunStats(
        val label: String,
        val fullyDelegated: Boolean,
        val meanNanos: Double,
        val p50Nanos: Double,
        val p90Nanos: Double,
    )

    private data class NativeAdaGnSpec(
        val weightsAsset: String,
        val channels: Int,
        val height: Int,
        val width: Int,
        val threadCount: Int,
    )

    private data class StageTraceSpec(val name: String, val shape: LongArray)

    companion object {
        private const val WARMUP_RUNS = 5
        private const val MEASURED_RUNS = 20
        private const val NATIVE_PIXEL_UNSHUFFLE_LABEL = "native_pixel_unshuffle2_fp32"
        private const val NATIVE_DEPTH_TO_SPACE_LABEL = "native_depth_to_space2_fp32"
        private const val OPENCL_FUSED_UPSAMPLER_LABEL = "opencl_fused_upsampler_probe"
        private const val VULKAN_FUSED_UPSAMPLER_LABEL = "vulkan_fused_upsampler_probe"
        private const val NATIVE_WSILU_CHUNK_ADD_LABEL = "native_wsilu_chunk_add_1280_fp32"
        private const val NATIVE_FAST_WSILU_CHUNK_ADD_LABEL = "native_fast_wsilu_chunk_add_1280_fp32"
        private const val NATIVE_NEURON_EXTENSION_PROBE_LABEL = "native_neuron_extension_probe"
        private const val NATIVE_AHWB_SYMBOL_PROBE_LABEL = "native_ahwb_symbol_probe"
        private const val NATIVE_FUNCTIONAL_OPS_PROBE_LABEL = "native_functional_ops_probe"
        private const val NATIVE_P_RECON_PIPELINE_PROBE_LABEL = "native_p_recon_pipeline_probe"
        private const val NATIVE_P_RECON_PRECISION_PROBE_LABEL = "native_p_recon_precision_probe"
        private const val NATIVE_P_RECON_STAGE_PRECISION_PROBE_LABEL = "native_p_recon_stage_precision_probe"
        private const val NATIVE_P_RECON_PIPELINE_STAGE3_FP16_LABEL = "native_p_recon_pipeline_stage3_fp16_probe"
        private const val NATIVE_P_RECON_PIPELINE_STAGE4_FP16_LABEL = "native_p_recon_pipeline_stage4_fp16_probe"
        private const val NATIVE_P_RECON_PIPELINE_HOTSPOT_FP16_LABEL = "native_p_recon_pipeline_hotspot_fp16_probe"
        private const val NATIVE_P_RECON_MIXED_MERGED_PROBE_LABEL = "native_p_recon_mixed_merged_probe"
        private const val NATIVE_P_RECON_BIG_PIPELINE_PROBE_LABEL = "native_p_recon_big_pipeline_probe"
        private const val RECON_NPU_MATRIX_LABEL = "recon_npu_matrix_probe"
        private const val RECON_MTK_ACCELERATOR_MATRIX_LABEL = "recon_mtk_accelerator_matrix_probe"
        private const val RECON_OFFICIAL_NEURON_MATRIX_LABEL = "recon_official_neuron_matrix_probe"
        private const val RECON_GPU_BIG_MATRIX_LABEL = "recon_gpu_big_matrix_probe"
        private const val P_RECON_GPU_BIG_PIPELINE_LABEL = "p_recon_gpu_big_pipeline_probe"
        private const val NATIVE_DLA_RUNTIME_PROBE_LABEL = "native_dla_runtime_probe"
        private const val NATIVE_STAGE3_DLA_RUNTIME_PROBE_LABEL = "native_stage3_dla_runtime_probe"
        private const val NATIVE_PIXEL_UNSHUFFLE_ADAPTOR_LABEL = "native_pixel_unshuffle2_adaptor_fp32"
        private const val DCB_NPU_SWEEP_LABEL = "dcb_npu_sweep"
        private val DCB_NPU_SWEEP_MODELS = listOf(
            "p_recon_mlp_dcb0",
            "p_recon_mlp_dcb0_adaptor",
            "p_recon_mlp_dcb0_dc",
            "p_recon_mlp_dcb0_dc_add",
            "p_recon_mlp_dcb0_ffn",
            "p_recon_mlp_dcb0_ffn_add",
            "p_recon_mlp_dcb1",
            "p_recon_mlp_dcb1_adaptor",
            "p_recon_mlp_dcb1_dc",
            "p_recon_mlp_dcb1_dc_add",
            "p_recon_mlp_dcb1_ffn",
            "p_recon_mlp_dcb1_ffn_add",
            "p_decoder_stage3_block0_only",
            "p_decoder_stage3_block0_adaptor",
            "p_decoder_stage3_block0_dc",
            "p_decoder_stage3_block0_dc_add",
            "p_decoder_stage3_block0_ffn",
            "p_decoder_stage3_block0_ffn_add",
        )
        private val NATIVE_PIXEL_UNSHUFFLE_ADAPTOR_THREAD_LABELS = mapOf(
            "native_pixel_unshuffle2_adaptor_t1_fp32" to 1,
            "native_pixel_unshuffle2_adaptor_t2_fp32" to 2,
            "native_pixel_unshuffle2_adaptor_t4_fp32" to 4,
            "native_pixel_unshuffle2_adaptor_t8_fp32" to 8,
        )
        private val NATIVE_GROUPNORM512_THREAD_LABELS = mapOf(
            "native_groupnorm512_t1_fp32" to 1,
            "native_groupnorm512_t2_fp32" to 2,
            "native_groupnorm512_t4_fp32" to 4,
            "native_groupnorm512_t8_fp32" to 8,
        )
        private val NATIVE_ADAGN512_STAGE1_THREAD_LABELS = mapOf(
            "native_adagn512_stage1_t1_fp32" to 1,
            "native_adagn512_stage1_t2_fp32" to 2,
            "native_adagn512_stage1_t4_fp32" to 4,
            "native_adagn512_stage1_t8_fp32" to 8,
        )
        private val NATIVE_ADAGN_LABELS = mapOf(
            "native_adagn_ada1_t1_fp32" to NativeAdaGnSpec("p_decoder_ada1_weights.bin", 512, 16, 32, 1),
            "native_adagn_ada2_t1_fp32" to NativeAdaGnSpec("p_decoder_ada2_weights.bin", 512, 16, 32, 1),
            "native_adagn_ada3_t1_fp32" to NativeAdaGnSpec("p_decoder_ada3_weights.bin", 512, 16, 32, 1),
            "native_adagn_ada4_t1_fp32" to NativeAdaGnSpec("p_decoder_ada4_weights.bin", 320, 32, 64, 1),
            "native_adagn_final_t1_fp32" to NativeAdaGnSpec("p_decoder_ada_final_weights.bin", 320, 32, 64, 1),
        )
        private val P_RECON_BASELINE_MODELS = listOf(
            "p_latent_decoder_fp32.tflite",
            "p_recon_mlp_dcb0_fp32.tflite",
            "p_recon_mlp_dcb1_fp32.tflite",
            "p_decoder_stage1_conv_only_fp32.tflite",
            "p_decoder_stage2_blocks_only_fp32.tflite",
            "p_upsampler_original_fp32.tflite",
            "p_decoder_stage3_blocks_only_fp32.tflite",
            "p_decoder_stage4_blocks_explicit_fp32.tflite",
            "p_recon_final_head_no_ada_fp32.tflite",
        )
        private val P_RECON_ADA_WEIGHT_ASSETS = listOf(
            "p_decoder_ada1_weights.bin",
            "p_decoder_ada2_weights.bin",
            "p_decoder_ada3_weights.bin",
            "p_decoder_ada4_weights.bin",
            "p_decoder_ada_final_weights.bin",
        )
        private val P_RECON_STAGE_TRACE_SPECS = listOf(
            StageTraceSpec("p_reference_feature", longArrayOf(1, 256, 32, 64)),
            StageTraceSpec("p_feature_unshuffled", longArrayOf(1, 1024, 16, 32)),
            StageTraceSpec("p_mlp_norm0", longArrayOf(1, 1024, 16, 32)),
            StageTraceSpec("p_mlp_dcb0", longArrayOf(1, 256, 16, 32)),
            StageTraceSpec("p_mlp_norm1_silu", longArrayOf(1, 256, 16, 32)),
            StageTraceSpec("p_codeword", longArrayOf(1, 18, 16, 32)),
            StageTraceSpec("p_stage1_blocks", longArrayOf(1, 512, 16, 32)),
            StageTraceSpec("p_stage1_adagn", longArrayOf(1, 512, 16, 32)),
            StageTraceSpec("p_stage2_blocks", longArrayOf(1, 512, 16, 32)),
            StageTraceSpec("p_stage2_adagn", longArrayOf(1, 512, 16, 32)),
            StageTraceSpec("p_upsampled", longArrayOf(1, 512, 32, 64)),
            StageTraceSpec("p_upsampled_adagn", longArrayOf(1, 512, 32, 64)),
            StageTraceSpec("p_stage3_blocks", longArrayOf(1, 320, 32, 64)),
            StageTraceSpec("p_stage3_adagn", longArrayOf(1, 320, 32, 64)),
            StageTraceSpec("p_stage4_blocks", longArrayOf(1, 320, 32, 64)),
            StageTraceSpec("p_stage4_adagn_final", longArrayOf(1, 320, 32, 64)),
            StageTraceSpec("encoder_p_reference_frame", longArrayOf(1, 3, 256, 512)),
        )
        private val P_RECON_STAGE3_FP16_MODELS = P_RECON_BASELINE_MODELS.map {
            if (it == "p_decoder_stage3_blocks_only_fp32.tflite") {
                "p_decoder_stage3_blocks_only_fp16_weight.tflite"
            } else {
                it
            }
        }
        private val P_RECON_STAGE4_FP16_MODELS = P_RECON_BASELINE_MODELS.map {
            if (it == "p_decoder_stage4_blocks_explicit_fp32.tflite") {
                "p_decoder_stage4_blocks_explicit_fp16_weight.tflite"
            } else {
                it
            }
        }
        private val P_RECON_HOTSPOT_FP16_MODELS = listOf(
            "p_latent_decoder_fp32.tflite",
            "p_recon_mlp_dcb0_fp32.tflite",
            "p_recon_mlp_dcb1_fp32.tflite",
            "p_decoder_stage1_conv_only_fp16_weight.tflite",
            "p_decoder_stage2_blocks_only_fp16_weight.tflite",
            "p_upsampler_original_fp32.tflite",
            "p_decoder_stage3_blocks_only_fp16_weight.tflite",
            "p_decoder_stage4_blocks_explicit_fp16_weight.tflite",
            "p_recon_final_head_no_ada_fp16_weight.tflite",
        )
        private val RECON_MATRIX_VARIANTS = listOf("fp32", "fp16_weight")
        private val RECON_NPU_MATRIX_MODELS = listOf(
            "p_upsampler_original",
            "p_decoder_stage3_blocks_only",
            "p_decoder_stage4_blocks_explicit",
            "p_recon_final_head_no_ada",
            "p_recon_big_latent_mlp",
            "p_recon_big_stage1_stage2",
            "p_recon_big_upsample_stage3",
            "p_recon_big_stage4_final",
        )
        private val RECON_GPU_BIG_MODELS = listOf(
            "p_recon_big_latent_mlp",
            "p_recon_big_stage1_stage2",
            "p_recon_big_upsample_stage3",
            "p_recon_big_stage4_final",
        )
        private val RECON_ACCELERATOR_FLAGS = linkedMapOf(
            "auto" to MtkTfliteRuntime.ACCELERATOR_AUTO,
            "gpu" to MtkTfliteRuntime.ACCELERATOR_GPU,
            "mdla" to MtkTfliteRuntime.ACCELERATOR_MDLA,
            "dsp" to MtkTfliteRuntime.ACCELERATOR_DSP,
        )
        private val RECON_ACCELERATOR_MATRIX_MODELS = listOf(
            "p_upsampler_original",
            "p_decoder_stage3_blocks_only",
            "p_decoder_stage4_blocks_explicit",
            "p_recon_final_head_no_ada",
        )
        private val RECON_OFFICIAL_NEURON_MATRIX_MODELS = listOf(
            "p_upsampler_original",
            "p_decoder_stage3_blocks_only",
            "p_decoder_stage4_blocks_explicit",
            "p_recon_final_head_no_ada",
        )
        private val DEFAULT_VARIANTS = listOf("fp32")
        private val ALL_VARIANTS = listOf("fp32", "fp16_weight", "fp16")
        private val MODELS = listOf(
            ModelSpec("i_recon_full", listOf(TensorSpec("i_y_hat", longArrayOf(1, 256, 16, 32)))),
            ModelSpec(
                "p_recon_full",
                listOf(
                    TensorSpec("p_y_hat", longArrayOf(1, 128, 16, 32)),
                    TensorSpec("p_ctx", longArrayOf(1, 256, 32, 64)),
                ),
            ),
            ModelSpec("i_latent_decoder", listOf(TensorSpec("i_y_hat", longArrayOf(1, 256, 16, 32)))),
            ModelSpec(
                "p_latent_decoder",
                listOf(
                    TensorSpec("p_y_hat", longArrayOf(1, 128, 16, 32)),
                    TensorSpec("p_ctx", longArrayOf(1, 256, 32, 64)),
                ),
            ),
            ModelSpec(
                "p_recon_big_latent_mlp",
                listOf(
                    TensorSpec("p_y_hat", longArrayOf(1, 128, 16, 32)),
                    TensorSpec("p_ctx", longArrayOf(1, 256, 32, 64)),
                ),
            ),
            ModelSpec("p_recon_big_stage1_stage2", listOf(TensorSpec("p_codeword", longArrayOf(1, 18, 16, 32)))),
            ModelSpec(
                "p_recon_big_upsample_stage3",
                listOf(
                    TensorSpec("p_stage2", longArrayOf(1, 512, 16, 32)),
                    TensorSpec("p_codeword", longArrayOf(1, 18, 16, 32)),
                ),
            ),
            ModelSpec(
                "p_recon_big_stage4_final",
                listOf(
                    TensorSpec("p_stage3", longArrayOf(1, 320, 32, 64)),
                    TensorSpec("p_codeword", longArrayOf(1, 18, 16, 32)),
                    TensorSpec("q_recon", longArrayOf(1)),
                ),
            ),
            ModelSpec("p_stage1_stage2_no_norm", listOf(TensorSpec("p_codeword", longArrayOf(1, 18, 16, 32)))),
            ModelSpec("p_upsample_stage3_no_norm", listOf(TensorSpec("p_stage2", longArrayOf(1, 512, 16, 32)))),
            ModelSpec("p_stage4_final_no_norm", listOf(TensorSpec("p_stage3", longArrayOf(1, 320, 32, 64)))),
            ModelSpec("p_recon_mlp_conv_only", listOf(TensorSpec("p_feature_unshuffled", longArrayOf(1, 1024, 16, 32)))),
            ModelSpec("p_recon_feature_to_codeword", listOf(TensorSpec("p_reference_feature", longArrayOf(1, 256, 32, 64)))),
            ModelSpec("p_recon_unshuffle_only", listOf(TensorSpec("p_reference_feature", longArrayOf(1, 256, 32, 64)))),
            ModelSpec("p_recon_unshuffle_conv_only", listOf(TensorSpec("p_reference_feature", longArrayOf(1, 256, 32, 64)))),
            ModelSpec("p_recon_unshuffle_spacetodepth_only", listOf(TensorSpec("p_reference_feature", longArrayOf(1, 256, 32, 64)))),
            ModelSpec("p_recon_unshuffle_mtk_nchw_only", listOf(TensorSpec("p_reference_feature", longArrayOf(1, 256, 32, 64)))),
            ModelSpec("p_recon_unshuffle_tflite_spacetodepth_nhwc", listOf(TensorSpec("p_reference_feature_nhwc", longArrayOf(1, 32, 64, 256)))),
            ModelSpec("p_recon_unshuffle_tflite_spacetodepth_nchw_wrap", listOf(TensorSpec("p_reference_feature", longArrayOf(1, 256, 32, 64)))),
            ModelSpec("p_recon_mlp_full", listOf(TensorSpec("p_feature_unshuffled", longArrayOf(1, 1024, 16, 32)))),
            ModelSpec("p_recon_feature_to_codeword_conv", listOf(TensorSpec("p_reference_feature", longArrayOf(1, 256, 32, 64)))),
            ModelSpec("p_recon_feature_to_codeword_spacetodepth", listOf(TensorSpec("p_reference_feature", longArrayOf(1, 256, 32, 64)))),
            ModelSpec("p_recon_feature_to_codeword_mtk_nchw", listOf(TensorSpec("p_reference_feature", longArrayOf(1, 256, 32, 64)))),
            ModelSpec("p_recon_mlp_norm0", listOf(TensorSpec("p_feature_unshuffled", longArrayOf(1, 1024, 16, 32)))),
            ModelSpec("p_recon_mlp_dcb0", listOf(TensorSpec("p_mlp_norm0", longArrayOf(1, 1024, 16, 32)))),
            ModelSpec("p_recon_mlp_dcb0_explicit", listOf(TensorSpec("p_mlp_norm0", longArrayOf(1, 1024, 16, 32)))),
            ModelSpec("p_recon_mlp_norm1", listOf(TensorSpec("p_mlp_dcb0", longArrayOf(1, 256, 16, 32)))),
            ModelSpec("p_recon_mlp_silu", listOf(TensorSpec("p_mlp_norm1", longArrayOf(1, 256, 16, 32)))),
            ModelSpec("p_recon_mlp_dcb1", listOf(TensorSpec("p_mlp_silu", longArrayOf(1, 256, 16, 32)))),
            ModelSpec("p_recon_mlp_dcb1_explicit", listOf(TensorSpec("p_mlp_silu", longArrayOf(1, 256, 16, 32)))),
            ModelSpec("p_recon_mlp_dcb0_adaptor", listOf(TensorSpec("p_mlp_norm0", longArrayOf(1, 1024, 16, 32)))),
            ModelSpec("p_recon_mlp_dcb0_dc", listOf(TensorSpec("p_dcb0_adapted", longArrayOf(1, 256, 16, 32)))),
            ModelSpec("p_recon_mlp_dcb0_dc_add", listOf(TensorSpec("p_dcb0_adapted", longArrayOf(1, 256, 16, 32)))),
            ModelSpec("p_recon_mlp_dcb0_ffn", listOf(TensorSpec("p_dcb0_dc_add", longArrayOf(1, 256, 16, 32)))),
            ModelSpec("p_recon_mlp_dcb0_ffn_add", listOf(TensorSpec("p_dcb0_dc_add", longArrayOf(1, 256, 16, 32)))),
            ModelSpec("p_recon_mlp_dcb0_adaptor_nhwc_conv", listOf(TensorSpec("p_mlp_norm0_nhwc", longArrayOf(1, 16, 32, 1024)))),
            ModelSpec("p_recon_mlp_dcb0_adaptor_flat_fc", listOf(TensorSpec("p_mlp_norm0_flat", longArrayOf(512, 1024)))),
            ModelSpec("p_recon_mlp_dcb1_adaptor", listOf(TensorSpec("p_mlp_silu", longArrayOf(1, 256, 16, 32)))),
            ModelSpec("p_recon_mlp_dcb1_dc", listOf(TensorSpec("p_dcb1_adapted", longArrayOf(1, 18, 16, 32)))),
            ModelSpec("p_recon_mlp_dcb1_dc_add", listOf(TensorSpec("p_dcb1_adapted", longArrayOf(1, 18, 16, 32)))),
            ModelSpec("p_recon_mlp_dcb1_ffn", listOf(TensorSpec("p_dcb1_dc_add", longArrayOf(1, 18, 16, 32)))),
            ModelSpec("p_recon_mlp_dcb1_ffn_add", listOf(TensorSpec("p_dcb1_dc_add", longArrayOf(1, 18, 16, 32)))),
            ModelSpec("p_recon_mlp_norm0_dcb0", listOf(TensorSpec("p_feature_unshuffled", longArrayOf(1, 1024, 16, 32)))),
            ModelSpec("p_recon_mlp_norm1_silu_dcb1", listOf(TensorSpec("p_mlp_dcb0", longArrayOf(1, 256, 16, 32)))),
            ModelSpec("p_decoder_stage1_conv_only", listOf(TensorSpec("p_codeword", longArrayOf(1, 18, 16, 32)))),
            ModelSpec("p_decoder_stage2_blocks_only", listOf(TensorSpec("p_stage1_adagn", longArrayOf(1, 512, 16, 32)))),
            ModelSpec("p_decoder_stage2_blocks_explicit", listOf(TensorSpec("p_stage1_adagn", longArrayOf(1, 512, 16, 32)))),
            ModelSpec("p_decoder_stage3_blocks_only", listOf(TensorSpec("p_upsampled", longArrayOf(1, 512, 32, 64)))),
            ModelSpec("p_decoder_stage3_blocks_explicit", listOf(TensorSpec("p_upsampled", longArrayOf(1, 512, 32, 64)))),
            ModelSpec("p_decoder_stage3_block0_only", listOf(TensorSpec("p_upsampled", longArrayOf(1, 512, 32, 64)))),
            ModelSpec("p_decoder_stage3_block0_explicit", listOf(TensorSpec("p_upsampled", longArrayOf(1, 512, 32, 64)))),
            ModelSpec("p_decoder_stage3_block0_adaptor", listOf(TensorSpec("p_upsampled", longArrayOf(1, 512, 32, 64)))),
            ModelSpec("p_decoder_stage3_block0_dc", listOf(TensorSpec("p_stage3_block0_adapted", longArrayOf(1, 320, 32, 64)))),
            ModelSpec("p_decoder_stage3_block0_dc_add", listOf(TensorSpec("p_stage3_block0_adapted", longArrayOf(1, 320, 32, 64)))),
            ModelSpec("p_decoder_stage3_block0_ffn", listOf(TensorSpec("p_stage3_block0_dc_add", longArrayOf(1, 320, 32, 64)))),
            ModelSpec("p_decoder_stage3_block0_ffn_conv1", listOf(TensorSpec("p_stage3_block0_dc_add", longArrayOf(1, 320, 32, 64)))),
            ModelSpec("p_decoder_stage3_block0_ffn_act", listOf(TensorSpec("p_stage3_block0_ffn_conv1", longArrayOf(1, 1280, 32, 64)))),
            ModelSpec("p_decoder_stage3_block0_ffn_conv2", listOf(TensorSpec("p_stage3_block0_ffn_act", longArrayOf(1, 640, 32, 64)))),
            ModelSpec("p_decoder_stage3_block0_ffn_add", listOf(TensorSpec("p_stage3_block0_dc_add", longArrayOf(1, 320, 32, 64)))),
            ModelSpec("p_decoder_stage3_blocks1_3_only", listOf(TensorSpec("p_stage3_block0", longArrayOf(1, 320, 32, 64)))),
            ModelSpec("p_decoder_stage3_block1_only", listOf(TensorSpec("p_stage3_block0", longArrayOf(1, 320, 32, 64)))),
            ModelSpec("p_decoder_stage3_block2_only", listOf(TensorSpec("p_stage3_block1", longArrayOf(1, 320, 32, 64)))),
            ModelSpec("p_decoder_stage3_block3_only", listOf(TensorSpec("p_stage3_block2", longArrayOf(1, 320, 32, 64)))),
            ModelSpec("p_decoder_stage4_blocks_only", listOf(TensorSpec("p_stage3_adagn", longArrayOf(1, 320, 32, 64)))),
            ModelSpec("p_decoder_stage4_blocks_explicit", listOf(TensorSpec("p_stage3_adagn", longArrayOf(1, 320, 32, 64)))),
            ModelSpec("p_decoder_stage1_full", listOf(TensorSpec("p_codeword", longArrayOf(1, 18, 16, 32)))),
            ModelSpec(
                "p_decoder_stage2_full",
                listOf(
                    TensorSpec("p_stage1", longArrayOf(1, 512, 16, 32)),
                    TensorSpec("p_codeword", longArrayOf(1, 18, 16, 32)),
                ),
            ),
            ModelSpec(
                "p_decoder_upsample_stage3_full",
                listOf(
                    TensorSpec("p_stage2", longArrayOf(1, 512, 16, 32)),
                    TensorSpec("p_codeword", longArrayOf(1, 18, 16, 32)),
                ),
            ),
            ModelSpec(
                "p_decoder_stage4_full",
                listOf(
                    TensorSpec("p_stage3", longArrayOf(1, 320, 32, 64)),
                    TensorSpec("p_codeword", longArrayOf(1, 18, 16, 32)),
                    TensorSpec("q_recon", longArrayOf(1, 1, 1, 1)),
                ),
            ),
            ModelSpec(
                "p_recon_final_head",
                listOf(
                    TensorSpec("p_stage4", longArrayOf(1, 320, 32, 64)),
                    TensorSpec("p_codeword", longArrayOf(1, 18, 16, 32)),
                ),
            ),
            ModelSpec("p_recon_final_head_no_ada", listOf(TensorSpec("p_stage4_adagn", longArrayOf(1, 320, 32, 64)))),
            ModelSpec("p_upsampler_original", listOf(TensorSpec("p_stage2", longArrayOf(1, 512, 16, 32)))),
            ModelSpec("p_upsampler_pixelshuffle", listOf(TensorSpec("p_stage2", longArrayOf(1, 512, 16, 32)))),
            ModelSpec("p_upsampler_conv_only", listOf(TensorSpec("p_stage2", longArrayOf(1, 512, 16, 32)))),
            ModelSpec("p_upsampler_depth_to_space_only", listOf(TensorSpec("p_stage2_conv", longArrayOf(1, 2048, 16, 32)))),
            ModelSpec("p_groupnorm_probe", listOf(TensorSpec("feature", longArrayOf(1, 512, 16, 32)))),
            ModelSpec(
                "p_adagn_probe",
                listOf(
                    TensorSpec("feature", longArrayOf(1, 512, 16, 32)),
                    TensorSpec("p_codeword", longArrayOf(1, 18, 16, 32)),
                ),
            ),
            ModelSpec("i_fast_codeword_to_frame_probe", listOf(TensorSpec("i_codeword", longArrayOf(1, 18, 16, 32)))),
            ModelSpec(
                "i_fast_codeword_to_frame_1block_probe",
                listOf(TensorSpec("i_codeword", longArrayOf(1, 18, 16, 32))),
            ),
            ModelSpec(
                "i_fast_codeword_to_frame_2block_probe",
                listOf(TensorSpec("i_codeword", longArrayOf(1, 18, 16, 32))),
            ),
            ModelSpec(
                "i_fast_codeword_to_frame_4block_probe",
                listOf(TensorSpec("i_codeword", longArrayOf(1, 18, 16, 32))),
            ),
            ModelSpec(
                "p_fast_feature_to_frame_probe",
                listOf(TensorSpec("p_reference_feature", longArrayOf(1, 256, 32, 64))),
            ),
        )
    }
}
