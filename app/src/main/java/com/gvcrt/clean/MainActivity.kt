package com.gvcrt.clean

import android.app.Activity
import android.graphics.BitmapFactory
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.WindowManager
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
    private var smallOfflineVideoRunner: SmallOfflineVideoRunner? = null
    private var pendingVideoUri: Uri? = null
    private lateinit var videoStopButton: Button

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
        val reconMnnButton = moduleButton("Recon\nMNN") {
            startTests(listOf("recon_mnn_diagnostic"))
        }
        val priorNpuPrecisionButton = moduleButton("I Prior\nprecision") {
            startTests(listOf("i_prior_npu_precision"))
        }
        val priorNpuSpeedButton = moduleButton("I Prior\nNPU speed") {
            startTests(listOf("i_prior_npu_speed"))
        }
        val priorTfliteProbeButton = moduleButton("I Prior\nTFLite probe") {
            startTests(listOf("i_prior_tflite_probe"))
        }
        val pPriorTfliteProbeButton = moduleButton("P Prior\nTFLite probe") {
            startTests(listOf("p_prior_tflite_probe"))
        }
        val smallVideoButton = moduleButton("Small video\noffline") {
            openVideoPicker()
        }
        videoStopButton = moduleButton("Stop\nvideo") {
            smallOfflineVideoRunner?.cancel()
        }.apply { isEnabled = false }
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
                addAll(
                    listOf(
                        reconMnnButton,
                        priorNpuPrecisionButton,
                        priorNpuSpeedButton,
                        priorTfliteProbeButton,
                        pPriorTfliteProbeButton,
                        smallVideoButton,
                    )
                )
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
            if (!isOnnxDemo) {
                addView(reconMnnButton, buttonLayoutParams())
            }
        }
        val priorNpuControls = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(16, 0, 16, 0)
            addView(priorNpuPrecisionButton, buttonLayoutParams())
            addView(priorNpuSpeedButton, buttonLayoutParams())
            addView(priorTfliteProbeButton, buttonLayoutParams())
            addView(pPriorTfliteProbeButton, buttonLayoutParams())
        }
        val smallVideoControls = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(16, 0, 16, 0)
            addView(smallVideoButton, buttonLayoutParams())
            addView(videoStopButton, buttonLayoutParams())
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
            fitsSystemWindows = true
            addView(controls)
            addView(speedControls)
            addView(projectControls)
            if (!isOnnxDemo) {
                addView(priorNpuControls)
                addView(smallVideoControls)
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

    @Deprecated("Uses the platform document picker for broad Android compatibility")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != VIDEO_PICK_REQUEST || resultCode != RESULT_OK) return
        val uri = data?.data ?: return
        val takeFlags = data.flags and Intent.FLAG_GRANT_READ_URI_PERMISSION
        runCatching { contentResolver.takePersistableUriPermission(uri, takeFlags) }
        pendingVideoUri = uri
        startTests(listOf("small_offline_video"))
    }

    private fun openVideoPicker() {
        startActivityForResult(
            Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE)
                type = "video/*"
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
            },
            VIDEO_PICK_REQUEST,
        )
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
            intent.getBooleanExtra("reconDiagnosticTest", false) -> listOf("recon_diagnostic")
            intent.getBooleanExtra("reconDissectTest", false) -> listOf("recon_dissect")
            intent.getBooleanExtra("iLatentOp0Fp16PrecisionTest", false) -> listOf("i_latent_op0_fp16")
            intent.getBooleanExtra("iLatentOp0AdapterFp16Test", false) -> listOf("i_latent_op0_adapter_fp16")
            intent.getBooleanExtra("iLatentConvInOfficialNeuronTest", false) -> listOf("i_latent_conv_in_official")
            intent.getBooleanExtra("iFeatureDecOfficialNeuronTest", false) -> listOf("i_featuredec_official")
            intent.getBooleanExtra("iFeatureDecSplitOfficialNeuronTest", false) -> listOf("i_featuredec_split_official")
            intent.getBooleanExtra("iReconOfficialNeuronTest", false) -> listOf("i_recon_official")
            intent.getBooleanExtra("reconMnnDiagnosticTest", false) -> listOf("recon_mnn_diagnostic")
            intent.getBooleanExtra("iPriorNpuPrecisionTest", false) -> listOf("i_prior_npu_precision")
            intent.getBooleanExtra("iPriorNpuSpeedTest", false) -> listOf("i_prior_npu_speed")
            intent.getBooleanExtra("iPriorTfliteProbe", false) -> listOf("i_prior_tflite_probe")
            intent.getBooleanExtra("pPriorTfliteProbe", false) -> listOf("p_prior_tflite_probe")
            intent.getBooleanExtra("enterpriseTfliteTest", false) -> listOf("enterprise_tflite")
            intent.getBooleanExtra("smallOfflineVideoTest", false) -> listOf("small_offline_video")
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
        videoStopButton.isEnabled = requested.contains("small_offline_video")
        if (requested.contains("small_offline_video")) {
            window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
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
                        moduleName == "recon_diagnostic" -> {
                            ReconDiagnosticBenchmark(this, ::emit).run(
                                labelFilter = intent.getStringExtra("reconDiagLabel"),
                                createOnly = intent.getBooleanExtra("reconDiagCreateOnly", false),
                                copyOutputs = !intent.getBooleanExtra("reconDiagNoOutputCopy", false),
                                accelerationMode = intent.getIntExtra(
                                    "reconDiagAccelerationMode",
                                    MtkTfliteRuntime.ACCELERATION_NEURON,
                                ),
                                warmupRuns = intent.getIntExtra("reconDiagWarmup", 5),
                                measuredRuns = intent.getIntExtra("reconDiagMeasured", 20),
                            )
                        }
                        moduleName == "recon_dissect" -> {
                            OnlineCompileDissectionProbe(this, ::emit).run(
                                opName = intent.getStringExtra("reconDissectOp"),
                                accelerationMode = if (intent.getBooleanExtra("reconDissectCpu", false)) {
                                    MtkTfliteRuntime.ACCELERATION_CPU
                                } else {
                                    MtkTfliteRuntime.ACCELERATION_NEURON
                                },
                                useOfficialNeuronDelegate = intent.getBooleanExtra("reconDissectOfficialNeuron", false),
                                officialAllowFp16ForFp32 = !intent.getBooleanExtra("reconDissectOfficialFp32", false),
                            )
                        }
                        moduleName == "i_latent_op0_fp16" -> {
                            ILatentOp0Fp16Probe(this, ::emit).run(
                                accelerationMode = if (intent.getBooleanExtra("iLatentOp0Fp16Cpu", false)) {
                                    MtkTfliteRuntime.ACCELERATION_CPU
                                } else {
                                    MtkTfliteRuntime.ACCELERATION_NEURON
                                },
                            )
                        }
                        moduleName == "i_latent_op0_adapter_fp16" -> {
                            ILatentOp0NeuronAdapterProbe(this, ::emit).run()
                        }
                        moduleName == "i_latent_conv_in_official" -> {
                            ILatentConvInOfficialNeuronProbe(this, ::emit).run()
                        }
                        moduleName == "i_featuredec_official" -> {
                            IFeatureDecOfficialNeuronProbe(this, ::emit).run(
                                warmupRuns = intent.getIntExtra("iFeatureDecWarmup", 3),
                                measuredRuns = intent.getIntExtra("iFeatureDecMeasured", 10),
                            )
                        }
                        moduleName == "i_featuredec_split_official" -> {
                            IFeatureDecSplitOfficialNeuronProbe(this, ::emit).run(
                                warmupRuns = intent.getIntExtra("iFeatureDecWarmup", 3),
                                measuredRuns = intent.getIntExtra("iFeatureDecMeasured", 10),
                            )
                        }
                        moduleName == "i_recon_official" -> {
                            IReconOfficialNeuronProbe(this, ::emit).run()
                        }
                        moduleName == "recon_mnn_diagnostic" -> {
                            MnnReconDiagnosticBenchmark(this, ::emit).run(
                                labelFilter = intent.getStringExtra("reconMnnLabel"),
                                warmupRuns = intent.getIntExtra("reconMnnWarmup", 5),
                                measuredRuns = intent.getIntExtra("reconMnnMeasured", 20),
                            )
                        }
                        moduleName == "i_prior_npu_precision" -> {
                            IEncoderPriorTfliteDiagnostic(this, ::emit).runPrecision()
                        }
                        moduleName == "i_prior_npu_speed" -> {
                            IEncoderPriorTfliteDiagnostic(this, ::emit).runSpeed(
                                warmupRuns = intent.getIntExtra("iPriorNpuWarmup", 5),
                                measuredRuns = intent.getIntExtra("iPriorNpuMeasured", 50),
                            )
                        }
                        moduleName == "i_prior_tflite_probe" -> {
                            IEncoderPriorTfliteDiagnostic(this, ::emit).run()
                        }
                        moduleName == "p_prior_tflite_probe" -> {
                            PEncoderPriorTfliteDiagnostic(this, ::emit).run()
                        }
                        moduleName == "enterprise_tflite" -> {
                            EnterpriseTfliteCompatibilityProbe(this, ::emit).run(
                                variant = intent.getStringExtra("enterpriseTfliteVariant") ?: "all",
                                warmupRuns = intent.getIntExtra("enterpriseTfliteWarmup", 3),
                                measuredRuns = intent.getIntExtra("enterpriseTfliteMeasured", 10),
                                relaxFp32 = intent.getBooleanExtra("enterpriseTfliteRelaxFp32", false),
                            )
                        }
                        moduleName == "small_offline_video" -> {
                            val uri = pendingVideoUri
                                ?: intent.getStringExtra("videoUri")?.let(Uri::parse)
                                ?: intent.getStringExtra("videoPath")?.let { Uri.fromFile(File(it)) }
                                ?: error("smallOfflineVideoTest requires videoUri/videoPath or a picked video")
                            val runner = SmallOfflineVideoRunner(this, ::emit, ::showImageComparison)
                            smallOfflineVideoRunner = runner
                            MemorySampler(this, ::emit).use { memory ->
                                memory.begin("small_offline_video")
                                runner.run(
                                    inputUri = uri,
                                    maxDurationSeconds = intent.getIntExtra("smallOfflineVideoSeconds", 60),
                                    h264Bitrate = intent.getIntExtra("smallOfflineVideoBitrate", 8_000_000),
                                )
                                memory.mark("video_complete")
                            }
                        }
                        moduleName.endsWith("_speed") -> {
                            ModuleSpeedBenchmarks(
                                this,
                                ::emit,
                                warmupRuns = intent.getIntExtra("speedWarmup", 5),
                                measuredRuns = intent.getIntExtra("speedMeasured", 50),
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
                    videoStopButton.isEnabled = false
                    window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                }
                smallOfflineVideoRunner = null
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
            line.startsWith("small_offline_video_speed stage=model_summary") -> {
                imageSummaryLines["Small speed"] =
                    "Small model: ${line.valueAfter("mean_frame_ms")} ms/frame, ${line.valueAfter("fps")} FPS"
            }
            line.startsWith("small_offline_video_quality") -> {
                imageSummaryLines["Small PSNR"] =
                    "Video PSNR: ${line.valueAfter("mean_psnr_db")} dB (mean)"
            }
        }
        if (imageSummaryLines.isNotEmpty()) {
            imageSummary.text = imageSummaryLines.values.joinToString("\n")
            imageSummary.visibility = View.VISIBLE
        }
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
        smallOfflineVideoRunner?.close()
        smallOfflineVideoRunner = null
        imageRunner?.close()
        imageRunner = null
        super.onDestroy()
    }

    companion object {
        const val TAG = "GVC_RT_CLEAN"
        private const val VIDEO_PICK_REQUEST = 4102
        private val ONNX_DEMO_MODULES = setOf(
            "temporal_reference",
            "complete_encoder",
            "complete_decoder",
            "temporal_reference_speed",
            "complete_encoder_speed",
            "complete_decoder_speed",
            "image_inference",
        )
        private val FULL_PROJECT_MODULES = listOf(
            "temporal_reference",
            "complete_encoder",
            "complete_decoder",
        )
    }
}
