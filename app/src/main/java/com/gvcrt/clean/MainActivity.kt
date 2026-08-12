package com.gvcrt.clean

import android.app.Activity
import android.graphics.BitmapFactory
import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
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
    private var largeOnlineRunner: LargeOnlineCodecRunner? = null
    private var selectedLargeQp = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val isOnnxDemo = BuildConfig.DEPLOYMENT_PATH == "onnx_demo"
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
        val largeIPrecisionButton = moduleButton("Large I\nprecision") {
            startTests(listOf("large_i_entropy_merged_precision"))
        }
        val largeISpeedButton = moduleButton("Large I\nspeed") {
            startTests(listOf("large_i_entropy_merged_speed"))
        }
        val largeIpCodecButton = moduleButton("Large I/P\ncodec") {
            startTests(listOf("large_i_entropy_codec", "large_p_entropy_codec"))
        }
        val largeIDecodeButton = moduleButton("Large I/P\ndecode") {
            startTests(listOf("large_i_entropy_decode", "large_p_entropy_decode"))
        }
        val largeMainButton = moduleButton("Large full\ncodec") {
            startTests(listOf("large_online_main"))
        }
        moduleButtons = buildList {
            addAll(
                listOf(
                    temporalButton,
                    encoderButton,
                    decoderButton,
                    temporalSpeedButton,
                    encoderSpeedButton,
                    decoderSpeedButton,
                    fullProjectButton,
                    imageInferenceButton,
                )
            )
            if (!isOnnxDemo) {
                add(largeIPrecisionButton)
                add(largeISpeedButton)
                add(largeIpCodecButton)
                add(largeIDecodeButton)
                add(largeMainButton)
            }
        }

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
        val largeControls = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(16, 0, 16, 0)
            addView(largeIPrecisionButton, buttonLayoutParams())
            addView(largeISpeedButton, buttonLayoutParams())
            addView(largeIpCodecButton, buttonLayoutParams())
            addView(largeIDecodeButton, buttonLayoutParams())
            addView(largeMainButton, buttonLayoutParams())
        }
        val largeQpSelector = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                listOf("QP 0", "QP 3", "QP 6", "QP 9"),
            )
            onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                    selectedLargeQp = LARGE_QPS[position]
                }

                override fun onNothingSelected(parent: AdapterView<*>?) = Unit
            }
        }
        val largeQpControls = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setPadding(16, 0, 16, 0)
            addView(TextView(this@MainActivity).apply { text = "Large QP" })
            addView(largeQpSelector)
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
            if (!isOnnxDemo) {
                addView(largeQpControls)
                addView(largeControls)
            }
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

    private fun requestedModules(intent: Intent): List<String> {
        val requested = when {
            intent.getBooleanExtra("temporalReferenceTest", false) -> listOf("temporal_reference")
            intent.getBooleanExtra("completeEncoderTest", false) -> listOf("complete_encoder")
            intent.getBooleanExtra("completeDecoderTest", false) -> listOf("complete_decoder")
            intent.getBooleanExtra("temporalReferenceSpeedTest", false) -> listOf("temporal_reference_speed")
            intent.getBooleanExtra("completeEncoderSpeedTest", false) -> listOf("complete_encoder_speed")
            intent.getBooleanExtra("completeDecoderSpeedTest", false) -> listOf("complete_decoder_speed")
            intent.getBooleanExtra("fullProjectTest", false) -> FULL_PROJECT_MODULES
            intent.getBooleanExtra("imageInferenceTest", false) -> listOf("image_inference")
            intent.getBooleanExtra("sequenceInferenceTest", false) -> listOf("sequence_inference")
            intent.getBooleanExtra("largeIpEntropyCodecTest", false) ->
                listOf("large_i_entropy_codec", "large_p_entropy_codec")
            intent.getBooleanExtra("largeIEntropyMergedSpeedTest", false) -> listOf("large_i_entropy_merged_speed")
            intent.getBooleanExtra("largeIEntropyMergedPrecisionTest", false) ||
                intent.getBooleanExtra("largeIEntropyMergedTest", false) -> listOf("large_i_entropy_merged_precision")
            intent.getBooleanExtra("largeIEntropyCodecTest", false) -> listOf("large_i_entropy_codec")
            intent.getBooleanExtra("largeIEntropyRansMergedTest", false) -> listOf("large_i_entropy_rans_merged")
            intent.getBooleanExtra("largeIEntropyDecodeSpeedTest", false) -> listOf("large_i_entropy_decode_speed")
            intent.getBooleanExtra("largeIEntropyDecodeTest", false) -> listOf("large_i_entropy_decode")
            intent.getBooleanExtra("largeIEntropyDecodeMergedSpeedTest", false) ->
                listOf("large_i_entropy_decode_merged_speed")
            intent.getBooleanExtra("largeIEntropyDecodeMergedTest", false) ->
                listOf("large_i_entropy_decode_merged")
            intent.getBooleanExtra("largePEntropyCodecTest", false) -> listOf("large_p_entropy_codec")
            intent.getBooleanExtra("largePEntropyMergedSpeedTest", false) ->
                listOf("large_p_entropy_merged_speed")
            intent.getBooleanExtra("largePEntropyDecodeSpeedTest", false) ->
                listOf("large_p_entropy_decode_speed")
            intent.getBooleanExtra("largePEntropyDecodeTest", false) -> listOf("large_p_entropy_decode")
            intent.getBooleanExtra("largeOnlineVideoTest", false) -> listOf("large_online_video")
            intent.getBooleanExtra("largeOnlineMainTest", false) -> listOf("large_online_main")
            intent.getBooleanExtra("ransCustomOpPartitionTest", false) -> listOf("rans_custom_op_partition")
            intent.getBooleanExtra("enterpriseTfliteTest", false) -> listOf("enterprise_tflite")
            else -> emptyList()
        }
        return if (BuildConfig.DEPLOYMENT_PATH != "onnx_demo" || requested.all(ONNX_DEMO_MODULES::contains)) {
            requested
        } else {
            emptyList()
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
                if (isMemoryDiagnosticLine(line)) return
                result.append(line).append('\n')
                runOnUiThread {
                    updateImageSummary(line)
                    output.text = result.toString()
                }
            }

            try {
                emit("deployment_path=${BuildConfig.DEPLOYMENT_PATH}")
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
                            imageRunnerFor(backend, ::emit).run(
                                intent.getStringExtra("imagePath"),
                                decodeFromBitstream = intent.getBooleanExtra("imageInferenceDecodeBitstream", true),
                            )
                        }
                        moduleName == "sequence_inference" -> {
                            imageRunnerFor(OnnxBackend.NNAPI_FP16_ALLOW_FALLBACK, ::emit).runSequence(
                                sequenceDir = intent.getStringExtra("sequenceDir")
                                    ?: error("sequenceInferenceTest requires sequenceDir"),
                                frameCount = intent.getIntExtra("sequenceFrames", 96),
                            )
                        }
                        moduleName == "enterprise_tflite" -> {
                            EnterpriseTfliteCompatibilityProbe(this, ::emit).run(
                                variant = intent.getStringExtra("enterpriseTfliteVariant") ?: "all",
                                warmupRuns = intent.getIntExtra("enterpriseTfliteWarmup", 3),
                                measuredRuns = intent.getIntExtra("enterpriseTfliteMeasured", 10),
                                relaxFp32 = intent.getBooleanExtra("enterpriseTfliteRelaxFp32", false),
                            )
                        }
                        moduleName == "large_i_entropy_codec" -> {
                            LargeIEntropyCodecProbe(this, ::emit).run()
                        }
                        moduleName == "large_i_entropy_rans_merged" -> {
                            LargeIEntropyCodecProbe(this, ::emit).runRansMerged(
                                warmupRuns = intent.getIntExtra("largeIEntropyRansMergedWarmup", 3),
                                measuredRuns = intent.getIntExtra("largeIEntropyRansMergedMeasured", 10),
                            )
                        }
                        moduleName == "large_i_entropy_merged_precision" -> {
                            LargeIEntropyCodecProbe(this, ::emit).run(
                                warmupRuns = 0,
                                measuredRuns = 1,
                                validateRoundtrip = true,
                                dumpOutputs = true,
                            )
                        }
                        moduleName == "large_i_entropy_merged_speed" -> {
                            LargeIEntropyCodecProbe(this, ::emit).run(
                                warmupRuns = intent.getIntExtra("largeIEntropyMergedWarmup", 3),
                                measuredRuns = intent.getIntExtra("largeIEntropyMergedMeasured", 10),
                                validateRoundtrip = false,
                                dumpOutputs = false,
                            )
                        }
                        moduleName == "large_p_entropy_codec" -> {
                            LargePEntropyCodecProbe(this, ::emit).run()
                        }
                        moduleName == "large_p_entropy_merged_speed" -> {
                            LargePEntropyCodecProbe(this, ::emit).runMergedSpeed(
                                warmupRuns = intent.getIntExtra("largePEntropyMergedWarmup", 3),
                                measuredRuns = intent.getIntExtra("largePEntropyMergedMeasured", 10),
                            )
                        }
                        moduleName == "large_p_entropy_decode" -> {
                            LargePEntropyDecoderMergedProbe(this, ::emit).run(
                                fastRelaxFp32 = intent.getBooleanExtra("largePEntropyDecodeFast", true),
                                validatePrecision = true,
                            )
                        }
                        moduleName == "large_p_entropy_decode_speed" -> {
                            LargePEntropyDecoderMergedProbe(this, ::emit).run(
                                warmupRuns = intent.getIntExtra("largePEntropyDecodeWarmup", 3),
                                measuredRuns = intent.getIntExtra("largePEntropyDecodeMeasured", 10),
                                fastRelaxFp32 = true,
                                validatePrecision = false,
                            )
                        }
                        moduleName == "large_online_main" -> {
                            largeOnlineRunner(::emit).run(
                                imagePath = intent.getStringExtra("imagePath"),
                                warmupRuns = intent.getIntExtra("largeOnlineWarmup", 1),
                                measuredRuns = intent.getIntExtra("largeOnlineMeasured", 1),
                                qp = intent.getIntExtra("largeOnlineQp", selectedLargeQp),
                            )
                        }
                        moduleName == "large_online_video" -> {
                            largeOnlineRunner(::emit).runSequence(
                                sequenceDir = intent.getStringExtra("sequenceDir")
                                    ?: error("largeOnlineVideoTest requires sequenceDir"),
                                frameCount = intent.getIntExtra("sequenceFrames", 24),
                                warmupRuns = intent.getIntExtra("largeOnlineWarmup", 0),
                                measuredRuns = intent.getIntExtra("largeOnlineMeasured", 1),
                                dumpPEntropyBoundaries = intent.getBooleanExtra(
                                    "largeOnlineEntropyDiagnostics",
                                    false,
                                ),
                                qp = intent.getIntExtra("largeOnlineQp", selectedLargeQp),
                            )
                        }
                        moduleName == "rans_custom_op_partition" -> {
                            TfliteCustomOpPartitionProbe(this, ::emit).run(
                                warmupRuns = intent.getIntExtra("ransCustomOpWarmup", 3),
                                measuredRuns = intent.getIntExtra("ransCustomOpMeasured", 10),
                            )
                        }
                        moduleName == "large_i_entropy_decode" -> {
                            LargeIEntropyDecoderMergedProbe(this, ::emit).run(
                                fastRelaxFp32 = intent.getBooleanExtra("largeIEntropyDecodeFast", true),
                                validatePrecision = true,
                            )
                        }
                        moduleName == "large_i_entropy_decode_speed" -> {
                            LargeIEntropyDecoderMergedProbe(this, ::emit).run(
                                warmupRuns = intent.getIntExtra("largeIEntropyDecodeWarmup", 3),
                                measuredRuns = intent.getIntExtra("largeIEntropyDecodeMeasured", 10),
                                fastRelaxFp32 = true,
                                validatePrecision = false,
                            )
                        }
                        moduleName == "large_i_entropy_decode_merged" -> {
                            LargeIEntropyDecoderMergedProbe(this, ::emit).run(
                                fastRelaxFp32 = intent.getBooleanExtra("largeIEntropyDecodeMergedFast", false),
                                validatePrecision = true,
                            )
                        }
                        moduleName == "large_i_entropy_decode_merged_speed" -> {
                            LargeIEntropyDecoderMergedProbe(this, ::emit).run(
                                warmupRuns = intent.getIntExtra("largeIEntropyDecodeMergedWarmup", 3),
                                measuredRuns = intent.getIntExtra("largeIEntropyDecodeMergedMeasured", 10),
                                fastRelaxFp32 = true,
                                validatePrecision = false,
                            )
                        }
                        moduleName.endsWith("_speed") -> {
                            val speedBackend = if (
                                intent.getBooleanExtra("speedNnapiCpuDisabled", false)
                            ) {
                                OnnxBackend.NNAPI_FP16_CPU_DISABLED
                            } else {
                                OnnxBackend.NNAPI_FP16_ALLOW_FALLBACK
                            }
                            ModuleSpeedBenchmarks(
                                this,
                                ::emit,
                                backend = speedBackend,
                                warmupRuns = intent.getIntExtra("speedWarmup", 5),
                                measuredRuns = intent.getIntExtra("speedMeasured", 50),
                                enableProfiling = intent.getBooleanExtra("onnxProfiling", false),
                            ).runModule(moduleName.removeSuffix("_speed"))
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
            line.startsWith("image_speed stage=core_codec") -> {
                imageSummaryLines["Inference"] = "Core codec: ${line.valueAfter("ms")} ms"
            }
            line.startsWith("image_speed stage=encode_core") -> {
                imageSummaryLines["Encode"] = "Encode core: ${line.valueAfter("ms")} ms"
            }
            line.startsWith("image_speed stage=decode_core") -> {
                imageSummaryLines["Decode"] = "Decode core: ${line.valueAfter("ms")} ms"
            }
        }
        if (imageSummaryLines.isNotEmpty()) {
            imageSummary.text = imageSummaryLines.values.joinToString("\n")
            imageSummary.visibility = View.VISIBLE
        }
    }

    private fun largeOnlineRunner(emit: (String) -> Unit): LargeOnlineCodecRunner =
        largeOnlineRunner ?: LargeOnlineCodecRunner(this, emit, ::showImageComparison).also {
            largeOnlineRunner = it
        }

    private fun isMemoryDiagnosticLine(line: String): Boolean =
        line.startsWith("memory_") ||
            line.startsWith("image_memory_") ||
            line.startsWith("gpu_memory=")

    private fun String.valueAfter(key: String): String {
        val prefix = "$key="
        return substringAfter(prefix, "").substringBefore(" ").ifBlank { "n/a" }
    }

    override fun onDestroy() {
        imageRunner?.close()
        imageRunner = null
        largeOnlineRunner?.close()
        largeOnlineRunner = null
        super.onDestroy()
    }

    companion object {
        const val TAG = "GVC_RT_CLEAN"
        private val LARGE_QPS = intArrayOf(0, 3, 6, 9)
        private val ONNX_DEMO_MODULES = setOf(
            "temporal_reference",
            "complete_encoder",
            "complete_decoder",
            "temporal_reference_speed",
            "complete_encoder_speed",
            "complete_decoder_speed",
            "image_inference",
            "sequence_inference",
        )
        private val FULL_PROJECT_MODULES = listOf(
            "temporal_reference",
            "complete_encoder",
            "complete_decoder",
        )
    }
}
