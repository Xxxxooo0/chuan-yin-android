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
        moduleButtons = listOf(temporalButton, encoderButton, decoderButton)

        val controls = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(16, 16, 16, 0)
            addView(temporalButton, buttonLayoutParams())
            addView(encoderButton, buttonLayoutParams())
            addView(decoderButton, buttonLayoutParams())
        }
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(controls)
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
                val runner = CleanModuleTests(this, ::emit)
                emit("requested_modules=${requested.joinToString(",")}")
                requested.forEach { runner.runModule(it) }
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
    }
}
