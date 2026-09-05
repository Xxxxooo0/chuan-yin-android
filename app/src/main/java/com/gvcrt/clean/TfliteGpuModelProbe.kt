package com.gvcrt.clean

import android.os.SystemClock
import java.io.File

/** One actual model and ordered binary fixtures; no alternate inference backend. */
class TfliteGpuModelProbe(private val emit: (String) -> Unit) {
    fun run(
        model: File,
        inputDir: File,
        outputDir: File,
        allowUnsupportedDevice: Boolean = true,
        allowBuiltinCpuFallback: Boolean = false,
    ) {
        emit("selected_backend=tflite_gpu model=${model.absolutePath}")
        val createStarted = SystemClock.elapsedRealtimeNanos()
        GenericTfliteGpuRuntime.create(
            model,
            allowUnsupportedDevice,
            allowBuiltinCpuFallback,
        ).use { runtime ->
            emit(
                "gpu_model_test_create create_ms=${elapsedMs(createStarted)} " +
                    "inputs=${runtime.inputSizes.joinToString(",")} outputs=${runtime.outputSizes.joinToString(",")} " +
                    "options=${runtime.optionsSummary}",
            )
            val inputs = runtime.inputSizes.indices.map { index ->
                val file = inputDir.resolve("input_$index.bin")
                file.readBytes().also { bytes ->
                    emit("gpu_model_test_input index=$index path=${file.absolutePath} bytes=${bytes.size} sha256=${AssetStore.sha256(bytes)}")
                }
            }
            val invokeStarted = SystemClock.elapsedRealtimeNanos()
            val outputs = runtime.run(inputs)
            emit("gpu_model_test_invoke invoke_ms=${elapsedMs(invokeStarted)} first_invoke=true includes_io_copy=true includes_create=false includes_dump=false")
            outputDir.mkdirs()
            outputs.forEachIndexed { index, bytes ->
                val file = outputDir.resolve("output_$index.bin")
                file.writeBytes(bytes)
                emit("gpu_model_test_output index=$index path=${file.absolutePath} bytes=${bytes.size} sha256=${AssetStore.sha256(bytes)}")
            }
            emit(
                "gpu_model_test_complete status=PASS " +
                    "cpu_fallback_allowed=$allowBuiltinCpuFallback precision_validated=false",
            )
        }
    }

    private fun elapsedMs(started: Long): Double = (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0
}
