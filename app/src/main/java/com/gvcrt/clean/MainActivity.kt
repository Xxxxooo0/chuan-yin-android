package com.gvcrt.clean

import android.app.Activity
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

class MainActivity : Activity() {
    private lateinit var output: TextView
    private lateinit var moduleButtons: List<Button>
    private var running = false

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
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(controls)
            addView(speedControls)
            addView(projectControls)
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

        val requested = when {
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
        if (requested.isNotEmpty()) {
            startTests(requested)
        }
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

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    private fun startTests(requested: List<String>) {
        if (running) {
            Log.w(TAG, "ignored request while another module test is running")
            return
        }
        running = true
        moduleButtons.forEach { it.isEnabled = false }
        output.text = "GVC-RT clean deployment\n"
        Thread {
            val result = StringBuilder()
            fun emit(line: String) {
                Log.i(TAG, line)
                result.append(line).append('\n')
                runOnUiThread { output.text = result.toString() }
            }

            try {
                emit("requested_modules=${requested.joinToString(",")}")
                requested.forEach { moduleName ->
                    when {
                        moduleName == "image_inference" -> {
                            val backend = if (intent.getBooleanExtra("imageInferenceCpu", false)) {
                                OnnxBackend.CPU
                            } else {
                                OnnxBackend.NNAPI_FP16_ALLOW_FALLBACK
                            }
                            ImageInferenceRunner(this, ::emit, backend).run(intent.getStringExtra("imagePath"))
                        }
                        moduleName.endsWith("_speed") -> {
                            ModuleSpeedBenchmarks(this, ::emit).runModule(moduleName.removeSuffix("_speed"))
                        }
                        else -> {
                            CleanModuleTests(this, ::emit).runModule(moduleName)
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

    companion object {
        const val TAG = "GVC_RT_CLEAN"
        private val FULL_PROJECT_MODULES = listOf(
            "temporal_reference",
            "complete_encoder",
            "complete_decoder",
        )
    }
}
