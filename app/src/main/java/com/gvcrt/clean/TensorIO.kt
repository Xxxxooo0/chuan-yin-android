package com.gvcrt.clean

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

class TensorValue private constructor(
    val name: String,
    val shape: LongArray,
    private var floatStorage: FloatArray?,
    private val halfStorage: ShortArray?,
) {
    constructor(name: String, shape: LongArray, data: FloatArray) :
        this(name, shape, data, null)

    val data: FloatArray
        get() = floatStorage ?: FloatArray(halfStorage!!.size) {
            TensorIO.halfBitsToFloat(halfStorage[it])
        }.also { floatStorage = it }

    val numel: Int get() = floatStorage?.size ?: halfStorage!!.size

    fun fp16Bits(): ShortArray {
        val floats = floatStorage
        return if (floats == null) {
            halfStorage!!
        } else {
            ShortArray(floats.size) { TensorIO.floatToHalfBits(floats[it]) }
        }
    }

    fun renamed(newName: String): TensorValue =
        if (floatStorage == null) {
            fromFp16(newName, shape, halfStorage!!)
        } else {
            TensorValue(newName, shape, floatStorage!!)
        }

    companion object {
        fun fromFp16(name: String, shape: LongArray, data: ShortArray): TensorValue =
            TensorValue(name, shape, null, data)
    }
}

data class TensorDiff(
    val maxAbs: Float,
    val meanAbs: Float,
    val rmse: Float,
    val exact: Boolean,
)

object TensorIO {
    fun readF32Le(name: String, shape: LongArray, bytes: ByteArray): TensorValue {
        val expectedBytes = shape.fold(1L) { acc, v -> acc * v } * 4L
        require(bytes.size.toLong() == expectedBytes) {
            "$name byte size mismatch: got=${bytes.size}, expected=$expectedBytes"
        }
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val data = FloatArray((expectedBytes / 4L).toInt())
        for (i in data.indices) data[i] = bb.float
        return TensorValue(name, shape, data)
    }

    fun readI32Le(name: String, bytes: ByteArray): IntArray {
        require(bytes.size % 4 == 0) { "$name byte size must be divisible by 4" }
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return IntArray(bytes.size / 4) { bb.int }
    }

    fun readI16Le(name: String, bytes: ByteArray): ShortArray {
        require(bytes.size % 2 == 0) { "$name byte size must be divisible by 2" }
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return ShortArray(bytes.size / 2) { bb.short }
    }

    fun readI8(bytes: ByteArray): ByteArray = bytes.copyOf()

    fun fromI8(name: String, shape: LongArray, bytes: ByteArray): TensorValue {
        val expected = shape.fold(1L) { acc, value -> acc * value }
        require(bytes.size.toLong() == expected) {
            "$name byte size mismatch: got=${bytes.size}, expected=$expected"
        }
        return TensorValue(name, shape, FloatArray(bytes.size) { bytes[it].toFloat() })
    }

    fun f32Le(tensor: TensorValue): ByteArray {
        val bytes = ByteArray(tensor.data.size * 4)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().put(tensor.data)
        return bytes
    }

    fun readF16Le(name: String, shape: LongArray, bytes: ByteArray): TensorValue {
        val expectedBytes = shape.fold(1L) { acc, value -> acc * value } * 2L
        require(bytes.size.toLong() == expectedBytes) {
            "$name byte size mismatch: got=${bytes.size}, expected=$expectedBytes"
        }
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val data = FloatArray((expectedBytes / 2L).toInt())
        for (index in data.indices) data[index] = halfBitsToFloat(bb.short)
        return TensorValue(name, shape, data)
    }

    fun f16Le(tensor: TensorValue): ByteArray {
        val bytes = ByteArray(tensor.data.size * 2)
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        tensor.data.forEach { value -> bb.putShort(floatToHalfBits(value)) }
        return bytes
    }

    fun diff(a: TensorValue, b: TensorValue): TensorDiff {
        require(a.data.size == b.data.size) {
            "${a.name}/${b.name} element mismatch: ${a.data.size} vs ${b.data.size}"
        }
        var maxAbs = 0f
        var sumAbs = 0.0
        var sumSq = 0.0
        var exact = true
        for (i in a.data.indices) {
            val d = kotlin.math.abs(a.data[i] - b.data[i])
            if (d != 0f) exact = false
            if (d > maxAbs) maxAbs = d
            sumAbs += d.toDouble()
            sumSq += d.toDouble() * d.toDouble()
        }
        val n = a.data.size.coerceAtLeast(1)
        return TensorDiff(
            maxAbs = maxAbs,
            meanAbs = (sumAbs / n).toFloat(),
            rmse = sqrt(sumSq / n).toFloat(),
            exact = exact,
        )
    }

    fun shapeText(shape: LongArray): String = shape.joinToString(prefix = "[", postfix = "]")

    fun floatToHalfBits(value: Float): Short {
        val f = java.lang.Float.floatToIntBits(value)
        val sign = (f ushr 16) and 0x8000
        var mantissa = f and 0x007fffff
        val exp = (f ushr 23) and 0xff
        val half: Int = when {
            exp == 0xff -> {
                if (mantissa == 0) sign or 0x7c00 else sign or 0x7c00 or (mantissa ushr 13).coerceAtLeast(1)
            }
            exp > 142 -> sign or 0x7c00
            exp < 113 -> {
                if (exp < 103) {
                    sign
                } else {
                    mantissa = mantissa or 0x00800000
                    val shift = 125 - exp
                    val rounded = (mantissa + (1 shl (shift - 1))) ushr shift
                    sign or rounded
                }
            }
            else -> {
                val halfExp = exp - 112
                val roundedMantissa = mantissa + 0x00001000
                if ((roundedMantissa and 0x00800000) != 0) {
                    val adjustedExp = halfExp + 1
                    if (adjustedExp >= 31) sign or 0x7c00 else sign or (adjustedExp shl 10)
                } else {
                    sign or (halfExp shl 10) or (roundedMantissa ushr 13)
                }
            }
        }
        return half.toShort()
    }

    fun halfBitsToFloat(bits: Short): Float {
        val h = bits.toInt() and 0xffff
        val sign = (h and 0x8000) shl 16
        val exp = (h ushr 10) and 0x1f
        val mantissa = h and 0x03ff
        val f = when (exp) {
            0 -> {
                if (mantissa == 0) {
                    sign
                } else {
                    var m = mantissa
                    var e = -1
                    while ((m and 0x0400) == 0) {
                        m = m shl 1
                        e--
                    }
                    m = m and 0x03ff
                    sign or ((127 - 15 + 1 + e) shl 23) or (m shl 13)
                }
            }
            31 -> sign or 0x7f800000 or (mantissa shl 13)
            else -> sign or ((exp + 127 - 15) shl 23) or (mantissa shl 13)
        }
        return java.lang.Float.intBitsToFloat(f)
    }
}
