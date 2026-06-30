package com.gvcrt.clean

import android.app.Activity
import android.graphics.BitmapFactory
import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.io.File
import java.util.Locale

class MainActivity : Activity() {
    private lateinit var output: TextView
    private lateinit var moduleButtons: List<Button>
    private lateinit var imageComparison: LinearLayout
    private lateinit var inputImage: ImageView
    private lateinit var reconImage: ImageView
    private lateinit var reconTitle: TextView
    private lateinit var imageSummary: TextView
    private val imageSummaryLines = linkedMapOf<String, String>()
    private var running = false
    private var imageRunner: ImageInferenceRunner? = null
    private var imageRunnerBackend: OnnxBackend? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        output = TextView(this).apply {
            textSize = 13f
            setPadding(24, 24, 24, 24)
            text = "GVC-RT clean deployment\nSelect one module test.\n"
        }

        val temporalButton = moduleButton("Temporal\nreference") {
            startTests(listOf("temporal_reference"))
        }
        val encoderButton = moduleButton("Complete\nencoder") {
            startTests(listOf("complete_encoder"))
        }
        val decoderButton = moduleButton("Complete\ndecoder") {
            startTests(listOf("complete_decoder"))
        }
        val temporalSpeedButton = moduleButton("Temporal\nspeed") {
            startTests(listOf("temporal_reference_speed"))
        }
        val encoderSpeedButton = moduleButton("Encoder\nspeed") {
            startTests(listOf("complete_encoder_speed"))
        }
        val decoderSpeedButton = moduleButton("Decoder\nspeed") {
            startTests(listOf("complete_decoder_speed"))
        }
        val fullProjectButton = moduleButton("Full\nproject") {
            startTests(FULL_PROJECT_MODULES)
        }
        val imageInferenceButton = moduleButton("Image\ninfer") {
            startTests(listOf("image_inference"))
        }
        moduleButtons = listOf(
            temporalButton,
            encoderButton,
            decoderButton,
            temporalSpeedButton,
            encoderSpeedButton,
            decoderSpeedButton,
            fullProjectButton,
            imageInferenceButton,
        )

        val controls = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(16, 16, 16, 0)
            addView(temporalButton, buttonLayoutParams())
            addView(encoderButton, buttonLayoutParams())
            addView(decoderButton, buttonLayoutParams())
        }
        val speedControls = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(16, 0, 16, 0)
            addView(temporalSpeedButton, buttonLayoutParams())
            addView(encoderSpeedButton, buttonLayoutParams())
            addView(decoderSpeedButton, buttonLayoutParams())
        }
        val projectControls = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(16, 0, 16, 0)
            addView(fullProjectButton, buttonLayoutParams())
            addView(imageInferenceButton, buttonLayoutParams())
        }
        inputImage = comparisonImageView()
        reconImage = comparisonImageView()
        reconTitle = comparisonTitle("Reconstruction")
        imageComparison = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setPadding(16, 8, 16, 8)
            visibility = View.GONE
            addView(imagePanel(comparisonTitle("Input"), inputImage), imagePanelLayoutParams())
            addView(imagePanel(reconTitle, reconImage), imagePanelLayoutParams())
        }
        imageSummary = TextView(this).apply {
            textSize = 12f
            setPadding(24, 0, 24, 8)
            visibility = View.GONE
        }
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(controls)
            addView(speedControls)
            addView(projectControls)
            addView(
                imageComparison,
                LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    dp(180),
                ),
            )
            addView(
                imageSummary,
                LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                ),
            )
            addView(
                ScrollView(this@MainActivity).apply { addView(output) },
                LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    0,
                    1f,
                ),
            )
        }
        setContentView(root)

        val requested = requestedModules(intent)
        if (requested.isNotEmpty()) {
            startTests(requested)
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        val requested = requestedModules(intent)
        if (requested.isNotEmpty()) {
            startTests(requested)
        }
    }

    private fun requestedModules(intent: Intent): List<String> =
        when {
            intent.getBooleanExtra("temporalReferenceTest", false) -> listOf("temporal_reference")
            intent.getBooleanExtra("completeEncoderTest", false) -> listOf("complete_encoder")
            intent.getBooleanExtra("completeDecoderTest", false) -> listOf("complete_decoder")
            intent.getBooleanExtra("temporalReferenceSpeedTest", false) -> listOf("temporal_reference_speed")
            intent.getBooleanExtra("completeEncoderSpeedTest", false) -> listOf("complete_encoder_speed")
            intent.getBooleanExtra("completeDecoderSpeedTest", false) -> listOf("complete_decoder_speed")
            intent.getBooleanExtra("fullProjectTest", false) -> FULL_PROJECT_MODULES
            intent.getBooleanExtra("imageInferenceTest", false) -> listOf("image_inference")
            else -> emptyList()
        }

    private fun moduleButton(label: String, onClick: () -> Unit): Button =
        Button(this).apply {
            text = label
            isAllCaps = false
            textSize = 12f
            minWidth = 0
            minHeight = 0
            setPadding(8, 0, 8, 0)
            setOnClickListener { onClick() }
        }

    private fun buttonLayoutParams(): LinearLayout.LayoutParams =
        LinearLayout.LayoutParams(0, dp(56), 1f).apply {
            setMargins(4, 0, 4, 0)
        }

    private fun comparisonImageView(): ImageView =
        ImageView(this).apply {
            adjustViewBounds = true
            scaleType = ImageView.ScaleType.FIT_CENTER
            setBackgroundColor(0xFF101010.toInt())
        }

    private fun comparisonTitle(title: String): TextView =
        TextView(this).apply {
            text = title
            gravity = Gravity.CENTER
            textSize = 12f
        }

    private fun imagePanel(titleView: TextView, imageView: ImageView): LinearLayout =
        LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(
                titleView,
                LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    dp(24),
                ),
            )
            addView(
                imageView,
                LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    0,
                    1f,
                ),
            )
        }

    private fun imagePanelLayoutParams(): LinearLayout.LayoutParams =
        LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, 1f).apply {
            setMargins(4, 0, 4, 0)
        }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    private fun startTests(requested: List<String>) {
        if (running) {
            Log.w(TAG, "ignored request while another module test is running")
            return
        }
        running = true
        moduleButtons.forEach { it.isEnabled = false }
        output.text = "GVC-RT clean deployment\n"
        imageComparison.visibility = View.GONE
        imageSummary.visibility = View.GONE
        imageSummary.text = ""
        imageSummaryLines.clear()
        inputImage.setImageDrawable(null)
        reconImage.setImageDrawable(null)
        reconTitle.text = "Reconstruction"
        Thread {
            val result = StringBuilder()
            fun emit(line: String) {
                Log.i(TAG, line)
                result.append(line).append('\n')
                runOnUiThread {
                    updateImageSummary(line)
                    output.text = result.toString()
                }
            }

            try {
                emit("requested_modules=${requested.joinToString(",")}")
                val fullProjectRun = requested == FULL_PROJECT_MODULES
                requested.forEach { moduleName ->
                    when {
                        moduleName == "image_inference" -> {
                            val backend = if (intent.getBooleanExtra("imageInferenceCpu", false)) {
                                OnnxBackend.CPU
                            } else {
                                OnnxBackend.NNAPI_FP16_ALLOW_FALLBACK
                            }
                            imageRunnerFor(backend, ::emit).run(intent.getStringExtra("imagePath"))
                        }
                        moduleName.endsWith("_speed") -> {
                            ModuleSpeedBenchmarks(this, ::emit).runModule(moduleName.removeSuffix("_speed"))
                        }
                        else -> {
                            if (fullProjectRun) {
                                MemorySampler(this, ::emit).use { memory ->
                                    memory.begin("full_project_$moduleName")
                                    CleanModuleTests(this, ::emit).runModule(moduleName)
                                    memory.mark("${moduleName}_complete")
                                }
                            } else {
                                CleanModuleTests(this, ::emit).runModule(moduleName)
                            }
                        }
                    }
                }
            } catch (t: Throwable) {
                emit("FAILED: ${t.javaClass.simpleName}: ${t.message}")
                Log.e(TAG, "clean test failed", t)
            } finally {
                runOnUiThread {
                    running = false
                    moduleButtons.forEach { it.isEnabled = true }
                }
            }
        }.start()
    }

    private fun imageRunnerFor(
        backend: OnnxBackend,
        emit: (String) -> Unit,
    ): ImageInferenceRunner {
        if (imageRunnerBackend != backend) {
            imageRunner?.close()
            imageRunner = null
            imageRunnerBackend = backend
        }
        return imageRunner ?: ImageInferenceRunner(this, emit, backend, ::showImageComparison).also {
            imageRunner = it
        }
    }

    private fun showImageComparison(inputFile: File, reconFile: File, psnr: Double) {
        val input = BitmapFactory.decodeFile(inputFile.absolutePath)
        val recon = BitmapFactory.decodeFile(reconFile.absolutePath)
        runOnUiThread {
            inputImage.setImageBitmap(input)
            reconImage.setImageBitmap(recon)
            reconTitle.text = "Reconstruction PSNR ${String.format(Locale.US, "%.2f", psnr)} dB"
        imageComparison.visibility = View.VISIBLE
        }
    }

    private fun updateImageSummary(line: String) {
        when {
            line.startsWith("image_inference_source=") -> {
                val backend = line.substringAfter("backend=", "")
                if (backend.isNotBlank()) imageSummaryLines["Backend"] = "Backend: $backend"
            }
            line.startsWith("image_quality p_recon_vs_input") -> {
                imageSummaryLines["PSNR"] = "PSNR: ${line.valueAfter("psnr_db")} dB"
            }
            line.startsWith("image_speed stage=total") -> {
                imageSummaryLines["Total"] = "Total: ${line.valueAfter("ms")} ms"
            }
            line.startsWith("image_memory_peak") ||
                line.startsWith("memory_peak label=image_inference") -> {
                imageSummaryLines["Peak"] =
                    "Peak RAM: PSS ${line.valueAfter("total_pss_mb")} MB, " +
                        "native ${line.valueAfter("native_pss_mb")} MB, RSS ${line.valueAfter("rss_mb")} MB"
            }
            line.startsWith("image_memory_end") ||
                line.startsWith("memory_end label=image_inference") -> {
                imageSummaryLines["End"] =
                    "End RAM: PSS ${line.valueAfter("total_pss_mb")} MB, " +
                        "native heap ${line.valueAfter("native_heap_mb")} MB, low_memory=${line.valueAfter("low_memory")}"
            }
            line.startsWith("gpu_memory=") -> {
                imageSummaryLines["GPU"] = "GPU memory: unavailable from app API; use adb dumpsys meminfo"
            }
        }
        if (imageSummaryLines.isNotEmpty()) {
            imageSummary.text = imageSummaryLines.values.joinToString("\n")
            imageSummary.visibility = View.VISIBLE
        }
    }

    private fun String.valueAfter(key: String): String {
        val prefix = "$key="
        return substringAfter(prefix, "").substringBefore(" ").ifBlank { "n/a" }
    }

    override fun onDestroy() {
        imageRunner?.close()
        imageRunner = null
        super.onDestroy()
    }

    companion object {
        const val TAG = "GVC_RT_CLEAN"
        private val FULL_PROJECT_MODULES = listOf(
            "temporal_reference",
            "complete_encoder",
            "complete_decoder",
        )
    }
}
