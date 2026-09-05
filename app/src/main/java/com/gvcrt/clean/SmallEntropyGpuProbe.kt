package com.gvcrt.clean

import android.os.SystemClock
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ln
import kotlin.math.sqrt

/** adb-only encode -> payload -> independent decode diagnostic for the Small fused candidates. */
class SmallEntropyGpuProbe(private val emit: (String) -> Unit) {
    fun run(encodeModel: File, decodeModel: File, fixtureDir: File, outputDir: File) {
        val latent = readFixture(fixtureDir.resolve("input_latent_y.f32le"))
        val ctxT = readFixture(fixtureDir.resolve("input_ctx_t.f32le"))
        outputDir.mkdirs()
        emit(
            "small_entropy_gpu_test_start backend=tflite_gpu encode=${encodeModel.absolutePath} " +
                "decode=${decodeModel.absolutePath} fixture_dir=${fixtureDir.absolutePath}",
        )

        val encodeOutputs = createAndRun(
            model = encodeModel,
            kind = SmallEntropyGpuRuntime.Kind.ENCODE,
            expectedInputs = ENCODE_INPUT_SIZES,
            expectedOutputs = ENCODE_OUTPUT_SIZES,
            inputs = listOf(latent, ctxT),
        )
        dumpOutputs(outputDir, "encode", encodeOutputs)
        val payloadSize = ByteBuffer.wrap(encodeOutputs[7]).order(ByteOrder.LITTLE_ENDIAN).int
        require(payloadSize in 1..encodeOutputs[6].size) {
            "invalid encoded payload size=$payloadSize capacity=${encodeOutputs[6].size}"
        }
        val payload = encodeOutputs[6].copyOf(payloadSize)
        outputDir.resolve("payload.bin").writeBytes(payload)
        emit(
            "small_entropy_gpu_payload path=${outputDir.resolve("payload.bin").absolutePath} " +
                "bytes=${payload.size} sha256=${AssetStore.sha256(payload)}",
        )

        val decodeOutputs = createAndRun(
            model = decodeModel,
            kind = SmallEntropyGpuRuntime.Kind.DECODE,
            expectedInputs = DECODE_INPUT_SIZES,
            expectedOutputs = DECODE_OUTPUT_SIZES,
            inputs = listOf(encodeOutputs[6], encodeOutputs[7], ctxT),
        )
        dumpOutputs(outputDir, "decode", decodeOutputs)
        emitExactDifference("z_hat", encodeOutputs[0], decodeOutputs[0])
        emitExactDifference("y_q_0", encodeOutputs[1], decodeOutputs[1])
        emitExactDifference("y_q_1", encodeOutputs[3], decodeOutputs[2])
        require(floatValuesEqual(encodeOutputs[0], decodeOutputs[0])) { "independent decode z_hat differs from encode" }
        require(floatValuesEqual(encodeOutputs[1], decodeOutputs[1])) { "independent decode y_q_0 differs from encode" }
        require(floatValuesEqual(encodeOutputs[3], decodeOutputs[2])) { "independent decode y_q_1 differs from encode" }
        validateServerReference(fixtureDir, encodeOutputs, decodeOutputs, payload)
        emit(
            "small_entropy_gpu_roundtrip discrete_exact=true " +
                "encode_y_hat_sha256=${AssetStore.sha256(encodeOutputs[5])} " +
                "decode_y_hat_sha256=${AssetStore.sha256(decodeOutputs[5])}",
        )
        emit(
            "small_entropy_gpu_test_complete status=PASS cpu_fallback_allowed=true " +
                "native_rans=true precision_validated=false",
        )
    }

    private fun createAndRun(
        model: File,
        kind: SmallEntropyGpuRuntime.Kind,
        expectedInputs: LongArray,
        expectedOutputs: LongArray,
        inputs: List<ByteArray>,
    ): List<ByteArray> {
        val createStarted = SystemClock.elapsedRealtimeNanos()
        SmallEntropyGpuRuntime.create(model, kind).use { runtime ->
            require(runtime.inputSizes.contentEquals(expectedInputs)) {
                "${kind.name.lowercase()} input sizes=${runtime.inputSizes.contentToString()} " +
                    "expected=${expectedInputs.contentToString()}"
            }
            require(runtime.outputSizes.contentEquals(expectedOutputs)) {
                "${kind.name.lowercase()} output sizes=${runtime.outputSizes.contentToString()} " +
                    "expected=${expectedOutputs.contentToString()}"
            }
            emit(
                "small_entropy_gpu_${kind.name.lowercase()}_create_ok " +
                    "create_ms=${elapsedMs(createStarted)} options=${runtime.optionsSummary}",
            )
            val invokeStarted = SystemClock.elapsedRealtimeNanos()
            return runtime.run(inputs).also {
                emit(
                    "small_entropy_gpu_${kind.name.lowercase()}_invoke_ok " +
                        "invoke_ms=${elapsedMs(invokeStarted)} includes_io_copy=true " +
                        "includes_create=false includes_dump=false",
                )
            }
        }
    }

    private fun readFixture(file: File): ByteArray {
        require(file.isFile) { "missing Small entropy fixture: ${file.absolutePath}" }
        return file.readBytes().also { bytes ->
            emit(
                "small_entropy_gpu_fixture path=${file.absolutePath} bytes=${bytes.size} " +
                    "sha256=${AssetStore.sha256(bytes)}",
            )
        }
    }

    private fun dumpOutputs(outputDir: File, prefix: String, outputs: List<ByteArray>) {
        outputs.forEachIndexed { index, bytes ->
            val file = outputDir.resolve("${prefix}_output_$index.bin")
            file.writeBytes(bytes)
            emit(
                "small_entropy_gpu_output graph=$prefix index=$index path=${file.absolutePath} " +
                    "bytes=${bytes.size} sha256=${AssetStore.sha256(bytes)}",
            )
        }
    }

    private fun emitExactDifference(label: String, expected: ByteArray, actual: ByteArray) {
        require(expected.size == actual.size && expected.size % 4 == 0)
        val expectedBuffer = ByteBuffer.wrap(expected).order(ByteOrder.LITTLE_ENDIAN)
        val actualBuffer = ByteBuffer.wrap(actual).order(ByteOrder.LITTLE_ENDIAN)
        var different = 0
        var first = -1
        var maxAbs = 0.0
        repeat(expected.size / 4) { index ->
            val expectedValue = expectedBuffer.float
            val actualValue = actualBuffer.float
            if (expectedValue.toRawBits() != actualValue.toRawBits()) {
                ++different
                if (first < 0) first = index
                maxAbs = maxOf(maxAbs, kotlin.math.abs(expectedValue - actualValue).toDouble())
            }
        }
        emit(
            "small_entropy_gpu_roundtrip_detail tensor=$label elements=${expected.size / 4} " +
                "different_elements=$different first_different_index=$first max_abs=$maxAbs",
        )
    }

    private fun floatValuesEqual(expected: ByteArray, actual: ByteArray): Boolean {
        if (expected.size != actual.size || expected.size % 4 != 0) return false
        val expectedBuffer = ByteBuffer.wrap(expected).order(ByteOrder.LITTLE_ENDIAN)
        val actualBuffer = ByteBuffer.wrap(actual).order(ByteOrder.LITTLE_ENDIAN)
        repeat(expected.size / 4) {
            if (expectedBuffer.float != actualBuffer.float) return false
        }
        return true
    }

    private fun validateServerReference(
        fixtureDir: File,
        encodeOutputs: List<ByteArray>,
        decodeOutputs: List<ByteArray>,
        payload: ByteArray,
    ) {
        val names = listOf("z_hat", "y_q_0", "scales_0", "y_q_1", "scales_1", "y_hat")
        val expected = names.map { name ->
            fixtureDir.resolve("expected_$name.f32le").also {
                require(it.isFile) { "missing Small entropy reference: ${it.absolutePath}" }
            }.readBytes()
        }
        names.forEachIndexed { index, name ->
            emitFloatComparison("encode_$name", expected[index], encodeOutputs[index])
        }
        emitFloatComparison("decode_z_hat", expected[0], decodeOutputs[0])
        emitFloatComparison("decode_y_q_0", expected[1], decodeOutputs[1])
        emitFloatComparison("decode_y_q_1", expected[3], decodeOutputs[2])
        emitFloatComparison("decode_y_hat", expected[5], decodeOutputs[5])
        compareCdfIndexes("scales_0", expected[2], encodeOutputs[2])
        compareCdfIndexes("scales_1", expected[4], encodeOutputs[4])
        val expectedPayload = fixtureDir.resolve("expected_payload.bin").also {
            require(it.isFile) { "missing Small entropy reference payload: ${it.absolutePath}" }
        }.readBytes()
        emit(
            "small_entropy_gpu_reference_payload exact=${payload.contentEquals(expectedPayload)} " +
                "actual_bytes=${payload.size} expected_bytes=${expectedPayload.size} " +
                "actual_sha256=${AssetStore.sha256(payload)} expected_sha256=${AssetStore.sha256(expectedPayload)}",
        )
    }

    private fun emitFloatComparison(label: String, expected: ByteArray, actual: ByteArray) {
        require(expected.size == actual.size && expected.size % 4 == 0)
        val expectedBuffer = ByteBuffer.wrap(expected).order(ByteOrder.LITTLE_ENDIAN)
        val actualBuffer = ByteBuffer.wrap(actual).order(ByteOrder.LITTLE_ENDIAN)
        var different = 0
        var maxAbs = 0.0
        var squared = 0.0
        repeat(expected.size / 4) {
            val delta = actualBuffer.float.toDouble() - expectedBuffer.float.toDouble()
            if (delta != 0.0) ++different
            maxAbs = maxOf(maxAbs, kotlin.math.abs(delta))
            squared += delta * delta
        }
        emit(
            "small_entropy_gpu_reference tensor=$label elements=${expected.size / 4} " +
                "different_elements=$different max_abs=$maxAbs rmse=${sqrt(squared / (expected.size / 4))}",
        )
    }

    private fun compareCdfIndexes(label: String, expected: ByteArray, actual: ByteArray) {
        require(expected.size == actual.size && expected.size % 4 == 0)
        val expectedBuffer = ByteBuffer.wrap(expected).order(ByteOrder.LITTLE_ENDIAN)
        val actualBuffer = ByteBuffer.wrap(actual).order(ByteOrder.LITTLE_ENDIAN)
        var different = 0
        repeat(expected.size / 4) {
            if (cdfIndex(expectedBuffer.float) != cdfIndex(actualBuffer.float)) ++different
        }
        emit(
            "small_entropy_gpu_reference_cdf_index tensor=$label elements=${expected.size / 4} " +
                "different_elements=$different exact=${different == 0}",
        )
    }

    private fun cdfIndex(value: Float): Int {
        val scale = value.coerceIn(0.11f, 16.0f)
        return ((ln(scale.toDouble()) - (-2.2072749f).toDouble()) * 25.502707f.toDouble())
            .toInt()
            .coerceIn(0, 127)
    }

    private fun elapsedMs(started: Long): Double =
        (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0

    companion object {
        private val ENCODE_INPUT_SIZES = longArrayOf(98_304, 1_572_864)
        private val ENCODE_OUTPUT_SIZES = longArrayOf(
            6_144, 49_152, 49_152, 49_152, 49_152, 98_304, 65_536, 4,
        )
        private val DECODE_INPUT_SIZES = longArrayOf(65_536, 4, 1_572_864)
        private val DECODE_OUTPUT_SIZES = longArrayOf(
            6_144, 49_152, 49_152, 49_152, 49_152, 98_304,
        )
    }
}
