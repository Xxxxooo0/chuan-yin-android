package com.gvcrt.clean

import java.io.File

class MtkTfliteRuntime private constructor(
    private var handle: Long,
    val inputSizes: LongArray,
    val outputSizes: LongArray,
    val fullyDelegated: Boolean,
    val optionsSummary: String,
) : AutoCloseable {
    fun run(inputs: List<ByteArray>, copyOutputs: Boolean = true): List<ByteArray> {
        require(handle != 0L) { "runtime is already closed" }
        return nativeRun(handle, inputs.toTypedArray(), copyOutputs).toList()
    }

    override fun close() {
        if (handle != 0L) {
            nativeRelease(handle)
            handle = 0L
        }
    }

    companion object {
        const val ACCELERATION_CPU = 0
        const val ACCELERATION_NEURON = 1
        const val ACCELERATOR_AUTO = 0
        const val ACCELERATOR_GPU = 1
        const val ACCELERATOR_DSP = 2
        const val ACCELERATOR_MDLA = 4

        init {
            System.loadLibrary("gvcrt_clean_rans")
        }

        fun create(
            tfliteFile: File,
            accelerationMode: Int = ACCELERATION_NEURON,
            acceleratorFlag: Int = ACCELERATOR_AUTO,
            cacheDir: File? = null,
            allowFp16ForFp32: Boolean = true,
        ): MtkTfliteRuntime {
            cacheDir?.mkdirs()
            val handle = nativeCreate(
                tfliteFile.absolutePath,
                accelerationMode,
                acceleratorFlag,
                cacheDir?.absolutePath,
                allowFp16ForFp32,
            )
            return MtkTfliteRuntime(
                handle = handle,
                inputSizes = nativeInputSizes(handle),
                outputSizes = nativeOutputSizes(handle),
                fullyDelegated = nativeFullyDelegated(handle),
                optionsSummary = nativeOptionsSummary(handle),
            )
        }

        fun benchmarkNativePixelUnshuffle2(warmupRuns: Int, measuredRuns: Int): LongArray =
            nativePixelUnshuffle2Benchmark(warmupRuns, measuredRuns)

        fun benchmarkNativeDepthToSpace2(warmupRuns: Int, measuredRuns: Int): LongArray =
            nativeDepthToSpace2Benchmark(warmupRuns, measuredRuns)

        fun benchmarkOpenClFusedUpsampler(warmupRuns: Int, measuredRuns: Int): String =
            nativeOpenClFusedUpsamplerBenchmark(warmupRuns, measuredRuns)

        fun benchmarkVulkanFusedUpsampler(warmupRuns: Int, measuredRuns: Int): String =
            nativeVulkanFusedUpsamplerBenchmark(warmupRuns, measuredRuns)

        fun benchmarkNativeWSiLUChunkAdd(warmupRuns: Int, measuredRuns: Int): LongArray =
            nativeWSiLUChunkAddBenchmark(warmupRuns, measuredRuns)

        fun benchmarkNativeFastWSiLUChunkAdd(warmupRuns: Int, measuredRuns: Int): LongArray =
            nativeFastWSiLUChunkAddBenchmark(warmupRuns, measuredRuns)

        fun benchmarkNativeFusedPixelUnshuffleAdaptor(
            weightsPath: String,
            warmupRuns: Int,
            measuredRuns: Int,
            threadCount: Int,
        ): LongArray = nativeFusedPixelUnshuffleAdaptorBenchmark(weightsPath, warmupRuns, measuredRuns, threadCount)

        fun benchmarkNativeGroupNorm512(warmupRuns: Int, measuredRuns: Int, threadCount: Int): LongArray =
            nativeGroupNorm512Benchmark(warmupRuns, measuredRuns, threadCount)

        fun benchmarkNativeAdaGn512Stage1(
            weightsPath: String,
            warmupRuns: Int,
            measuredRuns: Int,
            threadCount: Int,
        ): LongArray = nativeAdaGn512Stage1Benchmark(weightsPath, warmupRuns, measuredRuns, threadCount)

        fun benchmarkNativeAdaGn(
            weightsPath: String,
            channels: Int,
            height: Int,
            width: Int,
            warmupRuns: Int,
            measuredRuns: Int,
            threadCount: Int,
        ): LongArray = nativeAdaGnBenchmark(weightsPath, channels, height, width, warmupRuns, measuredRuns, threadCount)

        fun probeNeuronExtensions(names: List<String>): String =
            nativeProbeNeuronExtensions(names.toTypedArray())

        fun probeAhwbSymbols(): String =
            nativeProbeAhwbSymbols()

        fun runNeuronAdapterFp16Conv(
            input: ByteArray,
            weightsOhwi: ByteArray,
            bias: ByteArray,
        ): Array<ByteArray> = nativeNeuronAdapterFp16Conv(input, weightsOhwi, bias)

        fun pixelUnshuffle2Nchw256(input: FloatArray): FloatArray =
            nativePixelUnshuffle2Nchw256(input)

        fun groupNormNchw(
            input: FloatArray,
            channels: Int,
            height: Int,
            width: Int,
            groups: Int = 32,
            threadCount: Int = 1,
        ): FloatArray = nativeGroupNormNchw(input, channels, height, width, groups, threadCount)

        fun adaGnNchw(
            feature: FloatArray,
            codeword: FloatArray,
            weightsPath: String,
            channels: Int,
            height: Int,
            width: Int,
            threadCount: Int = 1,
        ): FloatArray = nativeAdaGnNchw(feature, codeword, weightsPath, channels, height, width, threadCount)

        fun runNativePReconPipelineProbe(
            modelPaths: List<String>,
            adaWeightPaths: List<String>,
            cacheDir: File,
            warmupRuns: Int,
            measuredRuns: Int,
        ): String {
            cacheDir.mkdirs()
            return nativePReconPipelineProbe(
                modelPaths.toTypedArray(),
                adaWeightPaths.toTypedArray(),
                cacheDir.absolutePath,
                warmupRuns,
                measuredRuns,
            )
        }

        fun runNativePReconPipeline(
            modelPaths: List<String>,
            adaWeightPaths: List<String>,
            cacheDir: File,
            pYHat: FloatArray,
            pCtx: FloatArray,
        ): Array<FloatArray> {
            cacheDir.mkdirs()
            return nativePReconPipelineRun(
                modelPaths.toTypedArray(),
                adaWeightPaths.toTypedArray(),
                cacheDir.absolutePath,
                pYHat,
                pCtx,
            )
        }

        fun runNativePReconPipelineTrace(
            modelPaths: List<String>,
            adaWeightPaths: List<String>,
            cacheDir: File,
            pYHat: FloatArray,
            pCtx: FloatArray,
        ): Array<FloatArray> {
            cacheDir.mkdirs()
            return nativePReconPipelineTrace(
                modelPaths.toTypedArray(),
                adaWeightPaths.toTypedArray(),
                cacheDir.absolutePath,
                pYHat,
                pCtx,
            )
        }

        fun runNativePReconBigPipelineProbe(
            modelPaths: List<String>,
            cacheDir: File,
            warmupRuns: Int,
            measuredRuns: Int,
        ): String {
            cacheDir.mkdirs()
            return nativePReconBigPipelineProbe(
                modelPaths.toTypedArray(),
                cacheDir.absolutePath,
                warmupRuns,
                measuredRuns,
            )
        }

        fun runNativePReconMixedMergedProbe(
            modelPaths: List<String>,
            adaWeightPaths: List<String>,
            cacheDir: File,
            warmupRuns: Int,
            measuredRuns: Int,
        ): String {
            cacheDir.mkdirs()
            return nativePReconMixedMergedProbe(
                modelPaths.toTypedArray(),
                adaWeightPaths.toTypedArray(),
                cacheDir.absolutePath,
                warmupRuns,
                measuredRuns,
            )
        }

        fun probeDlaRuntime(dlaPath: String): String =
            nativeDlaRuntimeProbe(dlaPath)

        private external fun nativeCreate(
            path: String,
            accelerationMode: Int,
            acceleratorFlag: Int,
            cacheDir: String?,
            allowFp16ForFp32: Boolean,
        ): Long
        private external fun nativeRelease(handle: Long)
        private external fun nativeInputSizes(handle: Long): LongArray
        private external fun nativeOutputSizes(handle: Long): LongArray
        private external fun nativeOptionsSummary(handle: Long): String
        private external fun nativeFullyDelegated(handle: Long): Boolean
        private external fun nativeRun(handle: Long, inputs: Array<ByteArray>, copyOutputs: Boolean): Array<ByteArray>
        private external fun nativePixelUnshuffle2Benchmark(warmupRuns: Int, measuredRuns: Int): LongArray
        private external fun nativeDepthToSpace2Benchmark(warmupRuns: Int, measuredRuns: Int): LongArray
        private external fun nativeOpenClFusedUpsamplerBenchmark(warmupRuns: Int, measuredRuns: Int): String
        private external fun nativeVulkanFusedUpsamplerBenchmark(warmupRuns: Int, measuredRuns: Int): String
        private external fun nativeWSiLUChunkAddBenchmark(warmupRuns: Int, measuredRuns: Int): LongArray
        private external fun nativeFastWSiLUChunkAddBenchmark(warmupRuns: Int, measuredRuns: Int): LongArray
        private external fun nativeFusedPixelUnshuffleAdaptorBenchmark(
            weightsPath: String,
            warmupRuns: Int,
            measuredRuns: Int,
            threadCount: Int,
        ): LongArray
        private external fun nativeGroupNorm512Benchmark(
            warmupRuns: Int,
            measuredRuns: Int,
            threadCount: Int,
        ): LongArray
        private external fun nativeAdaGn512Stage1Benchmark(
            weightsPath: String,
            warmupRuns: Int,
            measuredRuns: Int,
            threadCount: Int,
        ): LongArray
        private external fun nativeAdaGnBenchmark(
            weightsPath: String,
            channels: Int,
            height: Int,
            width: Int,
            warmupRuns: Int,
            measuredRuns: Int,
            threadCount: Int,
        ): LongArray
        private external fun nativeProbeNeuronExtensions(names: Array<String>): String
        private external fun nativeProbeAhwbSymbols(): String
        private external fun nativeNeuronAdapterFp16Conv(
            input: ByteArray,
            weightsOhwi: ByteArray,
            bias: ByteArray,
        ): Array<ByteArray>
        private external fun nativePixelUnshuffle2Nchw256(input: FloatArray): FloatArray
        private external fun nativeGroupNormNchw(
            input: FloatArray,
            channels: Int,
            height: Int,
            width: Int,
            groups: Int,
            threadCount: Int,
        ): FloatArray
        private external fun nativeAdaGnNchw(
            feature: FloatArray,
            codeword: FloatArray,
            weightsPath: String,
            channels: Int,
            height: Int,
            width: Int,
            threadCount: Int,
        ): FloatArray
        private external fun nativePReconPipelineProbe(
            modelPaths: Array<String>,
            adaWeightPaths: Array<String>,
            cacheDir: String,
            warmupRuns: Int,
            measuredRuns: Int,
        ): String
        private external fun nativePReconPipelineRun(
            modelPaths: Array<String>,
            adaWeightPaths: Array<String>,
            cacheDir: String,
            pYHat: FloatArray,
            pCtx: FloatArray,
        ): Array<FloatArray>
        private external fun nativePReconPipelineTrace(
            modelPaths: Array<String>,
            adaWeightPaths: Array<String>,
            cacheDir: String,
            pYHat: FloatArray,
            pCtx: FloatArray,
        ): Array<FloatArray>
        private external fun nativePReconBigPipelineProbe(
            modelPaths: Array<String>,
            cacheDir: String,
            warmupRuns: Int,
            measuredRuns: Int,
        ): String
        private external fun nativePReconMixedMergedProbe(
            modelPaths: Array<String>,
            adaWeightPaths: Array<String>,
            cacheDir: String,
            warmupRuns: Int,
            measuredRuns: Int,
        ): String
        private external fun nativeDlaRuntimeProbe(dlaPath: String): String
    }
}
