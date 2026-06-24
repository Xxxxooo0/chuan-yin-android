package com.gvcrt.clean

import android.content.Context
import java.io.File
import java.security.MessageDigest

class AssetStore(private val context: Context) {
    fun readBytes(assetPath: String): ByteArray =
        context.assets.open(assetPath).use { it.readBytes() }

    fun materialize(assetPath: String): File {
        val out = File(context.filesDir, "assets/$assetPath")
        out.parentFile?.mkdirs()
        context.assets.open(assetPath).use { input ->
            out.outputStream().use { output -> input.copyTo(output) }
        }
        return out
    }

    fun writeOutput(outputPath: String, bytes: ByteArray): File {
        val out = File(context.filesDir, outputPath)
        out.parentFile?.mkdirs()
        out.writeBytes(bytes)
        return out
    }

    fun readOutput(outputPath: String): ByteArray =
        File(context.filesDir, outputPath).readBytes()

    fun outputExists(outputPath: String): Boolean =
        File(context.filesDir, outputPath).isFile

    fun sha256(assetPath: String): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        context.assets.open(assetPath).use { input ->
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    companion object {
        fun sha256(data: ByteArray): String {
            val digest = MessageDigest.getInstance("SHA-256").digest(data)
            return digest.joinToString("") { "%02x".format(it) }
        }
    }
}
