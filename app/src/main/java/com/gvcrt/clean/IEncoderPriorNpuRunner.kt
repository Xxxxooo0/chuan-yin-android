package com.gvcrt.clean

import android.content.Context
import android.os.SystemClock
import org.json.JSONObject
import java.io.File

data class IEncoderPriorNpuOutput(
    val symbols: List<TensorValue>,
    val scales: List<TensorValue>,
    val yHat: TensorValue,
    val stageTimesNs: Map<String, Long>,
)

class IEncoderPriorNpuRunner(context: Context) : AutoCloseable {
    private val store = AssetStore(context)
    private val config = PriorNpuConfig.parse(store.readBytes(MANIFEST).decodeToString())
    private val cacheDir = File(context.cacheDir, "i_prior_npu")
    private val reduce = createRuntime(config.reduceAsset)
    private val stages = config.stageAssets.map(::createRuntime)

    init {
        require(reduce.fullyDelegated && stages.all { it.fullyDelegated }) {
            "I prior NPU graph is not fully delegated"
        }
    }

    fun runtimeSummary(): String =
        (listOf("reduce" to reduce) + stages.mapIndexed { index, runtime -> "stage${index + 1}" to runtime })
            .joinToString(" ") { (name, runtime) ->
                "$name{delegated=${runtime.fullyDelegated},options=${runtime.optionsSummary}}"
            }

    fun assetSummary(): String =
        config.assetNames.joinToString(" ") { asset -> "$asset{sha256=${store.sha256(asset)}}" }

    fun run(y: TensorValue, commonParams: TensorValue): IEncoderPriorNpuOutput {
        require(y.shape.contentEquals(Y_SHAPE)) { "unexpected I prior y shape ${TensorIO.shapeText(y.shape)}" }
        require(commonParams.shape.contentEquals(COMMON_SHAPE)) { "unexpected I common params shape ${TensorIO.shapeText(commonParams.shape)}" }
        val timings = linkedMapOf<String, Long>()
        val reducedBytes = measure(timings, "reduce") {
            reduce.run(listOf(PriorNpuTensorCodec.nchwToNhwcFp16(commonParams.data, COMMON_CHANNELS, HEIGHT, WIDTH))).single()
        }
        val reduced = PriorNpuTensorCodec.nhwcFp16ToNchw(reducedBytes, config.reducedChannels, HEIGHT, WIDTH)
        I4xPriorNative.create(y.data, commonParams.data, config.forceZeroThreshold).use { native ->
            val stageResults = ArrayList<I4xPriorStageResult>(4)
            stageResults += measure(timings, "native_stage0") { native.runStage0() }
            for (stage in 1..3) {
                val input = stageResults.last().yHatSoFar
                val outputs = measure(timings, "stage$stage") {
                    stages[stage - 1].run(
                        listOf(
                            PriorNpuTensorCodec.nchwToNhwcFp16(input, Y_CHANNELS, HEIGHT, WIDTH),
                            PriorNpuTensorCodec.nchwToNhwcFp16(reduced, config.reducedChannels, HEIGHT, WIDTH),
                        ),
                    )
                }
                require(outputs.size == 2) { "I prior stage$stage returned ${outputs.size} outputs" }
                val scales = PriorNpuTensorCodec.nhwcFp16ToNchw(outputs[0], Y_CHANNELS, HEIGHT, WIDTH)
                val means = PriorNpuTensorCodec.nhwcFp16ToNchw(outputs[1], Y_CHANNELS, HEIGHT, WIDTH)
                stageResults += measure(timings, "native_stage$stage") { native.runStage(stage, scales, means) }
            }
            val yHat = measure(timings, "native_finish") { native.finish() }
            return IEncoderPriorNpuOutput(
                symbols = stageResults.mapIndexed { stage, result -> TensorValue("i_y_q_w_$stage", STAGE_SHAPE, result.symbols) },
                scales = stageResults.mapIndexed { stage, result -> TensorValue("i_s_w_$stage", STAGE_SHAPE, result.scales) },
                yHat = TensorValue("i_y_hat", Y_SHAPE, yHat),
                stageTimesNs = timings,
            )
        }
    }

    override fun close() {
        reduce.close()
        stages.forEach { it.close() }
    }

    private fun createRuntime(asset: String): MtkTfliteRuntime =
        MtkTfliteRuntime.create(
            store.materialize(asset),
            accelerationMode = MtkTfliteRuntime.ACCELERATION_NEURON,
            acceleratorFlag = MtkTfliteRuntime.ACCELERATOR_MDLA,
            cacheDir = cacheDir,
        )

    private inline fun <T> measure(values: MutableMap<String, Long>, name: String, block: () -> T): T {
        val started = SystemClock.elapsedRealtimeNanos()
        return block().also { values[name] = SystemClock.elapsedRealtimeNanos() - started }
    }

    private data class PriorNpuConfig(
        val reduceAsset: String,
        val stageAssets: List<String>,
        val reducedChannels: Int,
        val forceZeroThreshold: Float,
        val assetNames: List<String>,
    ) {
        companion object {
            fun parse(text: String): PriorNpuConfig {
                val root = JSONObject(text)
                require(root.getString("layout") == "NHWC") { "I prior NPU assets must use NHWC" }
                require(root.getString("dtype") == "float16") { "I prior NPU assets must use FP16" }
                val records = root.getJSONArray("records")
                val byName = (0 until records.length()).associate { index ->
                    val record = records.getJSONObject(index)
                    record.getString("name") to record
                }
                fun asset(name: String): String {
                    val record = byName[name] ?: error("missing I prior NPU record $name")
                    require(record.getBoolean("ncc_eligible")) { "$name is not NCC eligible" }
                    return "prior_npu/" + File(record.getString("tflite")).name
                }
                val reduceAsset = asset("i_prior_reduce_fp16")
                val stageAssets = (1..3).map { stage -> asset("i_prior_stage${stage}_fp16") }
                return PriorNpuConfig(
                    reduceAsset = reduceAsset,
                    stageAssets = stageAssets,
                    reducedChannels = root.getInt("reduced_channels"),
                    forceZeroThreshold = root.getDouble("force_zero_thres").toFloat(),
                    assetNames = listOf(reduceAsset, *stageAssets.toTypedArray()),
                )
            }
        }
    }

    companion object {
        private const val MANIFEST = "prior_npu/i_prior_npu_manifest.json"
        private const val HEIGHT = 16
        private const val WIDTH = 32
        private const val Y_CHANNELS = 256
        private const val COMMON_CHANNELS = 514
        private val Y_SHAPE = longArrayOf(1, Y_CHANNELS.toLong(), HEIGHT.toLong(), WIDTH.toLong())
        private val COMMON_SHAPE = longArrayOf(1, COMMON_CHANNELS.toLong(), HEIGHT.toLong(), WIDTH.toLong())
        private val STAGE_SHAPE = longArrayOf(1, 64, HEIGHT.toLong(), WIDTH.toLong())
    }
}
