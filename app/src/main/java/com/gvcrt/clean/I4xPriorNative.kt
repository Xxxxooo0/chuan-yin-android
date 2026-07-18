package com.gvcrt.clean

data class I4xPriorStageResult(
    val symbols: FloatArray,
    val scales: FloatArray,
    val yHatSoFar: FloatArray,
)

class I4xPriorNative private constructor(private var handle: Long) : AutoCloseable {
    fun runStage0(): I4xPriorStageResult = unpack(nativeRunStage0(requireHandle()))

    fun runStage(stage: Int, scales: FloatArray, means: FloatArray): I4xPriorStageResult =
        unpack(nativeRunStage(requireHandle(), scales, means, stage))

    fun finish(): FloatArray = nativeFinish(requireHandle())

    override fun close() {
        if (handle != 0L) {
            nativeRelease(handle)
            handle = 0L
        }
    }

    private fun requireHandle(): Long = checkNotNull(handle.takeIf { it != 0L }) { "I4xPriorNative is closed" }

    private fun unpack(values: Array<FloatArray>): I4xPriorStageResult {
        require(values.size == 3) { "native prior stage returned ${values.size} tensors" }
        return I4xPriorStageResult(values[0], values[1], values[2])
    }

    companion object {
        init {
            System.loadLibrary("gvcrt_clean_rans")
        }

        fun create(y: FloatArray, commonParams: FloatArray, forceZeroThreshold: Float): I4xPriorNative {
            val handle = nativeCreate(y, commonParams, forceZeroThreshold)
            check(handle != 0L) { "failed to create I4xPriorNative" }
            return I4xPriorNative(handle)
        }

        private external fun nativeCreate(y: FloatArray, commonParams: FloatArray, forceZeroThreshold: Float): Long
        private external fun nativeRunStage0(handle: Long): Array<FloatArray>
        private external fun nativeRunStage(handle: Long, scales: FloatArray, means: FloatArray, stage: Int): Array<FloatArray>
        private external fun nativeFinish(handle: Long): FloatArray
        private external fun nativeRelease(handle: Long)
    }
}
