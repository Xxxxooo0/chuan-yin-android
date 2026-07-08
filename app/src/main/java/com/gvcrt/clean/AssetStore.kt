package com.gvcrt.clean

import android.content.Context
import java.io.File
import java.security.MessageDigest

class AssetStore(private val context: Context) {
    fun readBytes(assetPath: String): ByteArray =
        externalAsset(assetPath).takeIf { it.isFile }?.readBytes()
            ?: context.assets.open(assetPath).use { it.readBytes() }

    fun materialize(assetPath: String): File {
        externalAsset(assetPath).takeIf { it.isFile }?.let { return it }
        val out = materializedAsset(assetPath)
        if (out.isFile) {
            return out
        }
        out.parentFile?.mkdirs()
        context.assets.open(assetPath).use { input ->
            out.outputStream().use { output -> input.copyTo(output) }
        }
        return out
    }

    fun exists(assetPath: String): Boolean =
        externalAsset(assetPath).isFile || materializedAsset(assetPath).isFile || runCatching {
            context.assets.open(assetPath).close()
        }.isSuccess

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
        externalAsset(assetPath).takeIf { it.isFile }?.let { file ->
            return sha256(file.readBytes())
        }
        materializedAsset(assetPath).takeIf { it.isFile }?.let { file ->
            return sha256(file.readBytes())
        }
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

    private fun externalAsset(assetPath: String): File =
        File(context.getExternalFilesDir(null), "assets/$assetPath")

    private fun materializedAsset(assetPath: String): File =
        File(context.filesDir, "${installedAssetCacheDirName()}/$assetPath")

    private fun installedAssetCacheDirName(): String {
        val info = context.packageManager.getPackageInfo(context.packageName, 0)
        return "assets_${info.lastUpdateTime}"
    }

    companion object {
        fun sha256(data: ByteArray): String {
            val digest = MessageDigest.getInstance("SHA-256").digest(data)
            return digest.joinToString("") { "%02x".format(it) }
        }
    }
}
