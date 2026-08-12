package com.gvcrt.clean

import android.content.Context
import com.mediatek.neuropilot_V.neuron.NeuronDelegate

/** Tests one Interpreter containing Neuron partitions separated by one CPU custom op. */
class TfliteCustomOpPartitionProbe(
    private val context: Context,
    private val emit: (String) -> Unit,
) {
    fun run(warmupRuns: Int = 3, measuredRuns: Int = 10) {
        require(warmupRuns >= 0 && measuredRuns > 0)
        val store = AssetStore(context)
        require(store.exists(MODEL_ASSET)) {
            "missing $MODEL_ASSET; generate it with server_tools/export_tflite_custom_op_partition_probe.py"
        }
        val model = store.materialize(MODEL_ASSET)
        val options = NeuronDelegate.Options()
            .setAllowFp16(true)
            .setCompileOptions("--relax-fp32")
            .setAcceleratorName("mtk-neuron")
            .setExecutionPreference(NeuronDelegate.Options.EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER)
            .setMaxNumberOfDelegatedPartitions(8)
        emit(
            "custom_op_partition_start asset=$MODEL_ASSET sha256=${store.sha256(MODEL_ASSET)} " +
                "graph=conv2d_custom_cpu_identity_conv2d io=nhwc_fp32 " +
                "allow_fp16=true compile_options=--relax-fp32 accelerator=mtk-neuron " +
                "preference=FAST_SINGLE_ANSWER max_delegate_partitions=8 " +
                "warmup=$warmupRuns measured=$measuredRuns",
        )
        NeuronDelegate(options).use { delegate ->
            val result = nativeRun(
                model.absolutePath,
                delegate.nativeHandle,
                warmupRuns,
                measuredRuns,
            )
            emit("custom_op_partition_result create_ok=true invoke_ok=true $result")
            val invokeCount = RESULT_INVOKE_COUNT.find(result)?.groupValues?.get(1)?.toIntOrNull() ?: 0
            require(invokeCount == warmupRuns + measuredRuns) {
                "custom op invoke count mismatch expected=${warmupRuns + measuredRuns} actual=$invokeCount"
            }
            val maxAbs = RESULT_MAX_ABS.find(result)?.groupValues?.get(1)?.toDoubleOrNull()
                ?: error("native result does not contain max_abs")
            require(maxAbs <= MAX_ABS_THRESHOLD) {
                "custom op partition output mismatch max_abs=$maxAbs threshold=$MAX_ABS_THRESHOLD"
            }
            emit(
                "custom_op_partition_complete status=PASS custom_cpu_op_executed=true " +
                    "note=confirm_neuron_partition_count_from_device_neuron_logs",
            )
        }
    }

    private companion object {
        const val MODEL_ASSET = "diagnostic/rans_custom_op_partition_probe.tflite"
        val RESULT_INVOKE_COUNT = Regex("custom_invoke_count=(\\d+)")
        val RESULT_MAX_ABS = Regex("max_abs=([0-9.eE+-]+)")
        const val MAX_ABS_THRESHOLD = 1e-3

        init {
            System.loadLibrary("gvcrt_clean_rans")
        }

        @JvmStatic
        external fun nativeRun(
            modelPath: String,
            delegateHandle: Long,
            warmupRuns: Int,
            measuredRuns: Int,
        ): String
    }
}
