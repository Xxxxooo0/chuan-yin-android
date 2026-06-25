package com.gvcrt.clean

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import java.io.File

data class ImageTensorInput(
    val tensor: TensorValue,
    val source: String,
    val originalWidth: Int,
    val originalHeight: Int,
)

object ImageTensorLoader {
    private const val DEFAULT_ASSET = "sample/sample_input.png"
    private const val HEIGHT = 256
    private const val WIDTH = 512

    fun load(context: Context, imagePath: String?): ImageTensorInput {
        val source = when {
            imagePath.isNullOrBlank() -> "asset:$DEFAULT_ASSET"
            imagePath.startsWith("asset:") -> imagePath
            else -> imagePath
        }
        val bitmap = decodeBitmap(context, source)
        val scaled = if (bitmap.width == WIDTH && bitmap.height == HEIGHT) {
            bitmap
        } else {
            Bitmap.createScaledBitmap(bitmap, WIDTH, HEIGHT, true)
        }
        val data = FloatArray(3 * HEIGHT * WIDTH)
        val plane = HEIGHT * WIDTH
        for (y in 0 until HEIGHT) {
            for (x in 0 until WIDTH) {
                val pixel = scaled.getPixel(x, y)
                val offset = y * WIDTH + x
                data[offset] = Color.red(pixel).toModelInput()
                data[plane + offset] = Color.green(pixel).toModelInput()
                data[2 * plane + offset] = Color.blue(pixel).toModelInput()
            }
        }
        if (scaled !== bitmap) scaled.recycle()
        return ImageTensorInput(
            TensorValue("input_image", longArrayOf(1, 3, HEIGHT.toLong(), WIDTH.toLong()), data),
            source,
            bitmap.width,
            bitmap.height,
        )
    }

    private fun decodeBitmap(context: Context, source: String): Bitmap {
        val bitmap = if (source.startsWith("asset:")) {
            val assetPath = source.removePrefix("asset:")
            context.assets.open(assetPath).use(BitmapFactory::decodeStream)
        } else {
            BitmapFactory.decodeFile(File(source).absolutePath)
        }
        return bitmap ?: error("failed to decode image: $source")
    }

    private fun Int.toModelInput(): Float = (this.toFloat() / 255.0f) * 2.0f - 1.0f
}
