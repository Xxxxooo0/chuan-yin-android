package com.gvcrt.clean

import android.content.Context

class MnnReconDiagnosticBenchmark(
    private val context: Context,
    private val emit: (String) -> Unit,
) {
    private val store = AssetStore(context)

    fun run(
        labelFilter: String? = null,
        warmupRuns: Int = 5,
        measuredRuns: Int = 20,
    ) {
        emit(
            "recon_mnn_start route=mnn_isolated label_filter=${labelFilter ?: "all"} " +
                "warmup=$warmupRuns measured=$measuredRuns"
        )
        emit(runtimeProbeLine())

        var found = false
        for (spec in SPECS) {
            if (labelFilter != null && labelFilter != spec.label) continue
            val asset = "recon_mnn/${spec.label}.mnn"
            if (!store.exists(asset)) {
                emit("recon_mnn_skip label=${spec.label} reason=missing_asset asset=$asset")
                continue
            }
            found = true
            val file = store.materialize(asset)
            emit(
                "recon_mnn_asset label=${spec.label} asset=$asset size_bytes=${file.length()} " +
                    "sha256=${store.sha256(asset)} inputs=${spec.inputs} outputs=${spec.outputs}"
            )
            emit(
                "recon_mnn_runtime_unavailable label=${spec.label} " +
                    "reason=MNN Android runtime is not linked yet; add MNN AAR/native libs before timing CPU/OpenCL/Vulkan"
            )
        }
        if (!found) {
            emit("recon_mnn_no_assets expected=app/src/main/assets/recon_mnn/*.mnn")
        }
        emit("recon_mnn_complete")
    }

    private fun runtimeProbeLine(): String {
        val classNames = listOf(
            "com.taobao.android.mnn.MNNNetInstance",
            "com.alibaba.mnn.MNNNetInstance",
            "com.alibaba.mnn.Interpreter",
        )
        val status = classNames.joinToString(",") { name ->
            "$name=${if (classExists(name)) "present" else "missing"}"
        }
        return "recon_mnn_runtime_probe $status"
    }

    private fun classExists(name: String): Boolean =
        runCatching { Class.forName(name) }.isSuccess

    private data class Spec(
        val label: String,
        val inputs: String,
        val outputs: String,
    )

    companion object {
        private val SPECS = listOf(
            Spec(
                "p_recon_mlp_only",
                "[p_reference_feature:1x256x32x64]",
                "[p_codeword:1x18x16x32]",
            ),
            Spec(
                "p_recon_stage1_stage2",
                "[p_codeword:1x18x16x32]",
                "[p_stage2:1x512x16x32]",
            ),
            Spec(
                "p_recon_upsample_stage3",
                "[p_stage2:1x512x16x32,p_codeword:1x18x16x32]",
                "[p_stage3:1x320x32x64]",
            ),
            Spec(
                "p_recon_stage4_final",
                "[p_stage3:1x320x32x64,p_codeword:1x18x16x32,q_recon:1x320x1x1]",
                "[p_recon_frame:1x3x256x512]",
            ),
            Spec(
                "p_recon_back_half",
                "[p_stage2:1x512x16x32,p_codeword:1x18x16x32,q_recon:1x320x1x1]",
                "[p_recon_frame:1x3x256x512]",
            ),
        )
    }
}
