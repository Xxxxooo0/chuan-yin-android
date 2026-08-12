package com.gvcrt.clean

import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest

/** Runtime QP scale tables for the Large dynamic-QP TFLite package. */
class LargeDynamicQuantScales private constructor(
    private val tables: Map<String, ScaleTable>,
    val supportedQps: Set<Int>,
) {
    fun slice(name: String, qp: Int): ByteArray {
        require(qp in supportedQps) { "unsupported Large online QP=$qp supported=${supportedQps.sorted()}" }
        val table = tables[name] ?: error("missing quant scale table: $name")
        val offset = qp * table.bytesPerQp
        require(offset + table.bytesPerQp <= table.bytes.size) {
            "quant scale table $name does not contain QP=$qp"
        }
        return table.bytes.copyOfRange(offset, offset + table.bytesPerQp)
    }

    fun select(qp: Int): Map<String, ByteArray> = tables.keys.associateWith { name -> slice(name, qp) }

    companion object {
        fun load(packageRoot: File, manifest: JSONObject): LargeDynamicQuantScales? {
            if (!manifest.optBoolean("dynamic_qp", false)) return null
            val supported = manifest.getJSONArray("supported_qps")
            val supportedQps = buildSet {
                repeat(supported.length()) { add(supported.getInt(it)) }
            }
            require(supportedQps.containsAll(REQUIRED_QPS)) {
                "dynamic package must support QP ${REQUIRED_QPS.sorted()}, actual=${supportedQps.sorted()}"
            }
            val records = manifest.getJSONObject("quant_scale_tables")
            val tables = REQUIRED_TABLES.associateWith { name ->
                val record = records.getJSONObject(name)
                val file = packageRoot.resolve(record.getString("file"))
                require(file.isFile) { "missing quant scale table: ${file.absolutePath}" }
                val bytes = file.readBytes()
                val bytesPerQp = record.getInt("bytes_per_qp")
                require(bytesPerQp > 0 && bytes.size >= bytesPerQp * (supportedQps.maxOrNull()!! + 1)) {
                    "invalid quant scale table $name bytes=${bytes.size} bytesPerQp=$bytesPerQp"
                }
                val expectedSha = record.getString("sha256").lowercase()
                val actualSha = sha256(file)
                require(actualSha == expectedSha) {
                    "quant scale SHA mismatch name=$name expected=$expectedSha actual=$actualSha"
                }
                ScaleTable(bytes, bytesPerQp)
            }
            return LargeDynamicQuantScales(tables, supportedQps)
        }

        private fun sha256(file: File): String = FileInputStream(file).use { stream ->
            val digest = MessageDigest.getInstance("SHA-256")
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
            digest.digest().joinToString("") { "%02x".format(it) }
        }

        val REQUIRED_QPS = setOf(0, 3, 6, 9)
        private val REQUIRED_TABLES = setOf(
            "i_q_enc",
            "i_q_dec",
            "i_q_recon",
            "p_q_feature",
            "p_q_enc",
            "p_q_dec",
            "p_q_recon",
        )
    }

    private data class ScaleTable(val bytes: ByteArray, val bytesPerQp: Int)
}
