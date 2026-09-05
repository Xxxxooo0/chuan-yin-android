package com.gvcrt.clean

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.drawable.BitmapDrawable
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.DocumentsContract
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.WindowManager
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
    private var largeOnlineRunner: LargeOnlineCodecRunner? = null
    private var smallOnlineRunner: SmallOnlineSequenceRunner? = null
    private var smallOfflineVideoRunner: SmallOfflineVideoRunner? = null
    private var pendingVideoUri: Uri? = null
    private lateinit var variantSelector: Spinner
    private lateinit var backendSelector: Spinner
    private lateinit var videoStopButton: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        output = TextView(this).apply {
            textSize = 13f
            setPadding(24, 24, 24, 24)
            text = "GVC-RT GPU / MTK NPU deployment\n选择模型、后端和视频序列。\n"
        }

        val sequenceTestButton = moduleButton("测试\n视频序列") {
            openSequencePicker(selectedEnterpriseVariant())
        }
        val smallVideoButton = moduleButton("Small video\noffline") {
            openVideoPicker()
        }
        videoStopButton = moduleButton("Stop\nvideo") {
            smallOfflineVideoRunner?.cancel()
        }.apply { isEnabled = false }
        moduleButtons = listOf(sequenceTestButton, smallVideoButton)
        variantSelector = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                listOf("Large", "Small"),
            )
        }
        backendSelector = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                RuntimeBackend.entries.map(RuntimeBackend::logName),
            )
            setSelection(RuntimeBackend.entries.indexOf(RuntimeBackend.MTK_NPU))
        }
        val sequenceControls = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setPadding(16, 0, 16, 0)
            addView(TextView(this@MainActivity).apply { text = "模型" })
            addView(variantSelector)
            addView(backendSelector)
            addView(sequenceTestButton, buttonLayoutParams())
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
            addView(sequenceControls)
            addView(smallVideoControls)
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
        if (resultCode != RESULT_OK) return
        val uri = data?.data ?: return
        when (requestCode) {
            VIDEO_PICK_REQUEST -> {
                val takeFlags = data.flags and Intent.FLAG_GRANT_READ_URI_PERMISSION
                runCatching { contentResolver.takePersistableUriPermission(uri, takeFlags) }
                pendingVideoUri = uri
                startTests(listOf("small_offline_video"))
            }
            SEQUENCE_PICK_REQUEST -> stageSequenceForTest(uri, pendingSequenceVariant)
        }
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

    private var pendingSequenceVariant: String = "large"
    private var pendingSequenceBackend: RuntimeBackend = RuntimeBackend.MTK_NPU

    private fun openSequencePicker(variant: String) {
        pendingSequenceVariant = variant
        pendingSequenceBackend = RuntimeBackend.parse(backendSelector.selectedItem.toString())
        startActivityForResult(
            Intent(Intent.ACTION_OPEN_DOCUMENT_TREE).apply {
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
            },
            SEQUENCE_PICK_REQUEST,
        )
    }

    private fun stageSequenceForTest(treeUri: Uri, variant: String) {
        Thread {
            try {
                val takeFlags = Intent.FLAG_GRANT_READ_URI_PERMISSION
                runCatching { contentResolver.takePersistableUriPermission(treeUri, takeFlags) }
                val staged = stagePngSequence(treeUri)
                val module = if (variant == "large") "large_online_video" else "small_online_video"
                runOnUiThread {
                    startTests(
                        requested = listOf(module),
                        sequenceDir = staged.directory.absolutePath,
                        sequenceFrames = staged.frameCount,
                        runtimeBackend = pendingSequenceBackend,
                    )
                }
            } catch (error: Throwable) {
                Log.e(TAG, "could not stage video sequence", error)
                runOnUiThread {
                    output.text = "GVC-RT clean deployment\nFAILED: ${error.javaClass.simpleName}: ${error.message}\n"
                }
            }
        }.start()
    }

    private fun stagePngSequence(treeUri: Uri): StagedSequence {
        val childrenUri = DocumentsContract.buildChildDocumentsUriUsingTree(
            treeUri,
            DocumentsContract.getTreeDocumentId(treeUri),
        )
        val entries = mutableListOf<Pair<String, Uri>>()
        contentResolver.query(
            childrenUri,
            arrayOf(
                DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                DocumentsContract.Document.COLUMN_DISPLAY_NAME,
                DocumentsContract.Document.COLUMN_MIME_TYPE,
            ),
            null,
            null,
            null,
        )?.use { cursor ->
            val documentIdColumn = cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_DOCUMENT_ID)
            val displayNameColumn = cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_DISPLAY_NAME)
            val mimeTypeColumn = cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_MIME_TYPE)
            while (cursor.moveToNext()) {
                val name = cursor.getString(displayNameColumn)
                val mimeType = cursor.getString(mimeTypeColumn)
                if (mimeType == "image/png" || name.endsWith(".png", ignoreCase = true)) {
                    entries += name to DocumentsContract.buildDocumentUriUsingTree(
                        treeUri,
                        cursor.getString(documentIdColumn),
                    )
                }
            }
        }
        require(entries.size >= 2) { "selected directory needs at least two PNG frames" }
        val selected = entries.sortedBy { it.first }.take(SEQUENCE_FRAME_LIMIT)
        val destination = cacheDir.resolve("sequence_inputs/${System.currentTimeMillis()}").apply { mkdirs() }
        selected.forEachIndexed { index, (name, uri) ->
            val safeName = File(name).name.ifBlank { "frame_$index.png" }
            val target = destination.resolve("${index.toString().padStart(4, '0')}_$safeName")
            contentResolver.openInputStream(uri)?.use { input ->
                target.outputStream().use(input::copyTo)
            } ?: error("cannot read selected frame: $name")
        }
        return StagedSequence(destination, selected.size)
    }

    private fun requestedModules(intent: Intent): List<String> {
        return when {
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
            intent.getBooleanExtra("smallOnlineVideoTest", false) -> listOf("small_online_video")
            intent.getBooleanExtra("smallEntropyGpuTest", false) -> listOf("small_entropy_gpu")
            intent.getBooleanExtra("gpuModelTest", false) -> listOf("gpu_model")
            intent.getBooleanExtra("smallOfflineVideoTest", false) -> listOf("small_offline_video")
            intent.getBooleanExtra("largeOfflineVideoTest", false) -> listOf("large_offline_video")
            intent.getBooleanExtra("largeOnlineMainTest", false) -> listOf("large_online_main")
            intent.getBooleanExtra("ransCustomOpPartitionTest", false) -> listOf("rans_custom_op_partition")
            intent.getBooleanExtra("enterpriseTfliteTest", false) -> listOf("enterprise_tflite")
            else -> emptyList()
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

    private fun startTests(
        requested: List<String>,
        enterpriseVariant: String? = null,
        sequenceDir: String? = null,
        sequenceFrames: Int? = null,
        runtimeBackend: RuntimeBackend? = null,
    ) {
        if (running) {
            Log.w(TAG, "ignored request while another module test is running")
            return
        }
        running = true
        moduleButtons.forEach { it.isEnabled = false }
        if (::variantSelector.isInitialized) variantSelector.isEnabled = false
        if (::backendSelector.isInitialized) backendSelector.isEnabled = false
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
                val requestedBackend = runtimeBackend ?: RuntimeBackend.parse(intent.getStringExtra("backend"))
                emit("requested_modules=${requested.joinToString(",")}")
                requested.forEach { moduleName ->
                    require(
                        requestedBackend == RuntimeBackend.MTK_NPU || moduleName in listOf(
                            "gpu_model", "small_entropy_gpu", "small_online_video",
                            "large_online_main", "large_online_video", "large_offline_video",
                        ),
                    ) {
                        "gpu_delegate_unsupported module=$moduleName backend=${requestedBackend.logName} reason=mtk_only_entry"
                    }
                    when {
                        moduleName == "small_entropy_gpu" -> {
                            require(requestedBackend == RuntimeBackend.TFLITE_GPU) {
                                "smallEntropyGpuTest requires backend=GPU"
                            }
                            SmallEntropyGpuProbe(::emit).run(
                                encodeModel = File(
                                    intent.getStringExtra("smallEntropyEncodeModelPath")
                                        ?: error("smallEntropyGpuTest requires smallEntropyEncodeModelPath"),
                                ),
                                decodeModel = File(
                                    intent.getStringExtra("smallEntropyDecodeModelPath")
                                        ?: error("smallEntropyGpuTest requires smallEntropyDecodeModelPath"),
                                ),
                                fixtureDir = File(
                                    intent.getStringExtra("smallEntropyFixtureDir")
                                        ?: error("smallEntropyGpuTest requires smallEntropyFixtureDir"),
                                ),
                                outputDir = File(
                                    intent.getStringExtra("smallEntropyOutputDir")
                                        ?: error("smallEntropyGpuTest requires smallEntropyOutputDir"),
                                ),
                            )
                        }
                        moduleName == "gpu_model" -> {
                            TfliteGpuModelProbe(::emit).run(
                                model = File(intent.getStringExtra("gpuModelPath") ?: error("gpuModelTest requires gpuModelPath")),
                                inputDir = File(intent.getStringExtra("gpuInputDir") ?: error("gpuModelTest requires gpuInputDir")),
                                outputDir = File(intent.getStringExtra("gpuOutputDir") ?: error("gpuModelTest requires gpuOutputDir")),
                                allowUnsupportedDevice = requestedBackend != RuntimeBackend.AUTO,
                                allowBuiltinCpuFallback = intent.getBooleanExtra(
                                    "gpuAllowBuiltinCpuFallback",
                                    false,
                                ),
                            )
                        }
                        moduleName == "enterprise_tflite" -> {
                            EnterpriseTfliteCompatibilityProbe(this, ::emit).run(
                                variant = enterpriseVariant
                                    ?: intent.getStringExtra("enterpriseTfliteVariant")
                                    ?: "all",
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
                            largeOnlineRunner(::emit, requestedBackend).run(
                                imagePath = intent.getStringExtra("imagePath"),
                                warmupRuns = intent.getIntExtra("largeOnlineWarmup", 1),
                                measuredRuns = intent.getIntExtra("largeOnlineMeasured", 1),
                                qp = fixedLargeQp(),
                            )
                        }
                        moduleName == "large_online_video" -> {
                            largeOnlineRunner(::emit, requestedBackend).runSequence(
                                sequenceDir = sequenceDir
                                    ?: intent.getStringExtra("sequenceDir")
                                    ?: error("largeOnlineVideoTest requires sequenceDir"),
                                frameCount = sequenceFrames
                                    ?: intent.getIntExtra("sequenceFrames", SEQUENCE_FRAME_LIMIT),
                                warmupRuns = intent.getIntExtra("largeOnlineWarmup", 0),
                                measuredRuns = intent.getIntExtra("largeOnlineMeasured", 1),
                                dumpPEntropyBoundaries = intent.getBooleanExtra(
                                    "largeOnlineEntropyDiagnostics",
                                    false,
                                ),
                                qp = fixedLargeQp(),
                            )
                        }
                        moduleName == "small_online_video" -> {
                            val useEntropy = intent.getBooleanExtra("smallOnlineUseEntropy", true)
                            smallOnlineRunner(::emit, requestedBackend, useEntropy).runSequence(
                                sequenceDir = sequenceDir
                                    ?: intent.getStringExtra("sequenceDir")
                                    ?: error("smallOnlineVideoTest requires sequenceDir"),
                                frameCount = sequenceFrames
                                    ?: intent.getIntExtra("sequenceFrames", SEQUENCE_FRAME_LIMIT),
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
                        moduleName == "large_offline_video" -> {
                            val uri = pendingVideoUri
                                ?: intent.getStringExtra("videoUri")?.let(Uri::parse)
                                ?: intent.getStringExtra("videoPath")?.let { Uri.fromFile(File(it)) }
                                ?: error("largeOfflineVideoTest requires videoUri/videoPath or a picked video")
                            MemorySampler(this, ::emit).use { memory ->
                                memory.begin("large_offline_video")
                                largeOnlineRunner(::emit, requestedBackend).runOfflineVideo(
                                    inputUri = uri,
                                    maxDurationSeconds = intent.getIntExtra("largeOfflineVideoSeconds", 60),
                                    h264Bitrate = intent.getIntExtra("largeOfflineVideoBitrate", 8_000_000),
                                    qp = fixedLargeQp(),
                                )
                                memory.mark("video_complete")
                            }
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
                        else -> error("unsupported module: $moduleName")
                    }
                }
            } catch (t: Throwable) {
                emit("FAILED: ${t.javaClass.simpleName}: ${t.message}")
                Log.e(TAG, "clean test failed", t)
            } finally {
                runOnUiThread {
                    running = false
                    moduleButtons.forEach { it.isEnabled = true }
                    if (::variantSelector.isInitialized) variantSelector.isEnabled = true
                    if (::backendSelector.isInitialized) backendSelector.isEnabled = true
                    videoStopButton.isEnabled = false
                    window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                }
                smallOfflineVideoRunner = null
            }
        }.start()
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

    private fun showVideoComparison(input: Bitmap, reconstruction: Bitmap, psnr: Double, frame: Int) {
        runOnUiThread {
            val oldInput = (inputImage.drawable as? BitmapDrawable)?.bitmap
            val oldReconstruction = (reconImage.drawable as? BitmapDrawable)?.bitmap
            inputImage.setImageBitmap(input)
            reconImage.setImageBitmap(reconstruction)
            if (oldInput !== input && oldInput?.isRecycled == false) oldInput.recycle()
            if (oldReconstruction !== reconstruction && oldReconstruction?.isRecycled == false) {
                oldReconstruction.recycle()
            }
            reconTitle.text = "Frame $frame PSNR ${String.format(Locale.US, "%.2f", psnr)} dB"
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
            line.startsWith("large_offline_video_progress phase=decode") -> {
                imageSummaryLines["VideoProgress"] =
                    "Video: frame ${line.valueAfter("frame")}, PSNR ${line.valueAfter("psnr_db")} dB"
            }
            line.startsWith("large_offline_video_speed phase=gvc_model_total") -> {
                imageSummaryLines["VideoSpeed"] =
                    "GVC model: ${line.valueAfter("mean_frame_ms")} ms/frame, ${line.valueAfter("fps")} fps"
            }
            line.startsWith("large_offline_video_quality") -> {
                imageSummaryLines["VideoQuality"] =
                    "Mean PSNR: ${line.valueAfter("mean_psnr_db")} dB"
            }
            line.startsWith("large_online_main_output") -> {
                imageSummaryLines["VideoQuality"] =
                    "Mean PSNR: ${line.valueAfter("mean_psnr_db")} dB"
            }
            line.startsWith("large_online_video_summary") || line.startsWith("small_online_video_summary") -> {
                imageSummaryLines["VideoSpeed"] =
                    "Sequence: ${line.valueAfter("mean_frame_ms")} ms/frame, ${line.valueAfter("fps")} fps"
                imageSummaryLines["VideoQuality"] =
                    "Mean PSNR: ${line.valueAfter("mean_psnr_db")} dB"
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

    private fun largeOnlineRunner(emit: (String) -> Unit, backend: RuntimeBackend): LargeOnlineCodecRunner {
        largeOnlineRunner?.takeIf { it.backend == backend }?.let { return it }
        largeOnlineRunner?.close()
        return LargeOnlineCodecRunner(this, emit, ::showImageComparison, ::showVideoComparison, backend).also {
            largeOnlineRunner = it
        }
    }

    private fun smallOnlineRunner(
        emit: (String) -> Unit,
        backend: RuntimeBackend,
        useEntropy: Boolean = true,
    ): SmallOnlineSequenceRunner =
        smallOnlineRunner?.takeIf { it.backend == backend && it.useEntropy == useEntropy }
            ?: SmallOnlineSequenceRunner(this, emit, ::showVideoComparison, backend, useEntropy).also {
                smallOnlineRunner = it
            }

    private fun selectedEnterpriseVariant(): String =
        variantSelector.selectedItem.toString().lowercase(Locale.US)

    private fun fixedLargeQp(): Int {
        val requested = intent.getIntExtra("largeOnlineQp", FIXED_LARGE_QP)
        require(requested == FIXED_LARGE_QP) {
            "GVC-RT Large mainline is fixed to QP=$FIXED_LARGE_QP; requested QP=$requested"
        }
        return FIXED_LARGE_QP
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
        largeOnlineRunner?.close()
        largeOnlineRunner = null
        smallOnlineRunner = null
        super.onDestroy()
    }

    companion object {
        const val TAG = "GVC_RT_CLEAN"
        private const val FIXED_LARGE_QP = 9
        private const val VIDEO_PICK_REQUEST = 4109
        private const val SEQUENCE_PICK_REQUEST = 4110
        private const val SEQUENCE_FRAME_LIMIT = 24
        private data class StagedSequence(val directory: File, val frameCount: Int)
    }
}
