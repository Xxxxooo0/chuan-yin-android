package com.gvcrt.clean

import java.util.Locale

enum class RuntimeBackend(val logName: String) {
    AUTO("auto"),
    MTK_NPU("mtk_npu"),
    TFLITE_GPU("tflite_gpu"),
    ;

    /** The caller owns cleanup of a partially created model set before throwing. */
    fun <T> create(emit: (String) -> Unit, createModels: (RuntimeBackend) -> T): T {
        if (this != AUTO) {
            emit("selected_backend=$logName")
            return createModels(this)
        }
        emit("selected_backend=mtk_npu")
        return try {
            createModels(MTK_NPU)
        } catch (mtkError: Throwable) {
            emit("auto_mtk_create_failed error=${mtkError.message}")
            emit("selected_backend=tflite_gpu")
            try {
                createModels(TFLITE_GPU)
            } catch (gpuError: Throwable) {
                throw IllegalStateException(
                    "no_supported_gpu_npu_backend mtk_error=${mtkError.message} gpu_error=${gpuError.message}",
                    gpuError,
                )
            }
        }
    }

    companion object {
        fun parse(value: String?): RuntimeBackend = when (value?.trim()?.lowercase(Locale.US)) {
            null, "", "mtk", "npu", "mtk_npu" -> MTK_NPU
            "auto" -> AUTO
            "gpu", "tflite_gpu" -> TFLITE_GPU
            else -> error("unsupported backend=$value; expected auto, mtk_npu, or gpu")
        }
    }
}
