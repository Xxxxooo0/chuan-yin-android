package com.gvcrt.clean

import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import android.os.Process
import java.io.File
import java.util.Locale
import kotlin.math.max
import kotlin.math.min

class MemorySampler(
    context: Context,
    private val emit: (String) -> Unit,
    private val intervalMs: Long = 200L,
) : AutoCloseable {
    private val appContext = context.applicationContext
    private val activityManager = appContext.getSystemService(ActivityManager::class.java)
    private val lock = Object()
    @Volatile private var running = false
    private var label = ""
    private var peak = MemorySnapshot.empty()
    private var thread: Thread? = null

    fun begin(label: String) {
        synchronized(lock) {
            if (running) return
            this.label = label
            running = true
            peak = sample()
            emit("memory_start label=$label ${peak.format()}")
            emit("gpu_memory=unavailable_on_this_device source=android_app_no_stable_gpu_memory_api")
            thread = Thread({ sampleLoop() }, "gvcrt-memory-sampler").apply {
                isDaemon = true
                start()
            }
        }
    }

    fun mark(stage: String) {
        if (!running) return
        val snapshot = sampleAndUpdatePeak()
        emit("memory_mark label=$label stage=$stage ${snapshot.format()}")
    }

    fun end() {
        val currentLabel: String
        synchronized(lock) {
            if (!running) return
            running = false
            currentLabel = label
        }
        thread?.join(500L)
        thread = null
        val end = sampleAndUpdatePeak()
        emit("memory_peak label=$currentLabel ${peak.format()}")
        emit("memory_end label=$currentLabel ${end.format()}")
    }

    override fun close() {
        end()
    }

    private fun sampleLoop() {
        while (running) {
            sampleAndUpdatePeak()
            try {
                Thread.sleep(intervalMs)
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
                return
            }
        }
    }

    private fun sampleAndUpdatePeak(): MemorySnapshot {
        val snapshot = sample()
        synchronized(lock) {
            peak = peak.maxOf(snapshot)
        }
        return snapshot
    }

    private fun sample(): MemorySnapshot {
        val debugInfo = Debug.MemoryInfo()
        Debug.getMemoryInfo(debugInfo)
        val runtime = Runtime.getRuntime()
        val systemInfo = ActivityManager.MemoryInfo().also(activityManager::getMemoryInfo)
        val procStatus = ProcStatus.read()
        return MemorySnapshot(
            totalPssKb = debugInfo.totalPss,
            nativePssKb = debugInfo.nativePss,
            dalvikPssKb = debugInfo.dalvikPss,
            otherPssKb = debugInfo.otherPss,
            javaHeapUsedBytes = runtime.totalMemory() - runtime.freeMemory(),
            javaHeapMaxBytes = runtime.maxMemory(),
            nativeHeapAllocatedBytes = Debug.getNativeHeapAllocatedSize(),
            vmRssKb = procStatus.vmRssKb,
            vmHwmKb = procStatus.vmHwmKb,
            systemAvailBytes = systemInfo.availMem,
            systemTotalBytes = systemInfo.totalMem,
            lowMemory = systemInfo.lowMemory,
        )
    }

    private data class ProcStatus(
        val vmRssKb: Long,
        val vmHwmKb: Long,
    ) {
        companion object {
            fun read(): ProcStatus {
                var vmRss = 0L
                var vmHwm = 0L
                File("/proc/${Process.myPid()}/status").forEachLine { line ->
                    when {
                        line.startsWith("VmRSS:") -> vmRss = line.kbValue()
                        line.startsWith("VmHWM:") -> vmHwm = line.kbValue()
                    }
                }
                return ProcStatus(vmRss, vmHwm)
            }

            private fun String.kbValue(): Long =
                split(Regex("\\s+")).firstNotNullOfOrNull { it.toLongOrNull() } ?: 0L
        }
    }

    private data class MemorySnapshot(
        val totalPssKb: Int,
        val nativePssKb: Int,
        val dalvikPssKb: Int,
        val otherPssKb: Int,
        val javaHeapUsedBytes: Long,
        val javaHeapMaxBytes: Long,
        val nativeHeapAllocatedBytes: Long,
        val vmRssKb: Long,
        val vmHwmKb: Long,
        val systemAvailBytes: Long,
        val systemTotalBytes: Long,
        val lowMemory: Boolean,
    ) {
        fun maxOf(other: MemorySnapshot): MemorySnapshot =
            MemorySnapshot(
                max(totalPssKb, other.totalPssKb),
                max(nativePssKb, other.nativePssKb),
                max(dalvikPssKb, other.dalvikPssKb),
                max(otherPssKb, other.otherPssKb),
                max(javaHeapUsedBytes, other.javaHeapUsedBytes),
                max(javaHeapMaxBytes, other.javaHeapMaxBytes),
                max(nativeHeapAllocatedBytes, other.nativeHeapAllocatedBytes),
                max(vmRssKb, other.vmRssKb),
                max(vmHwmKb, other.vmHwmKb),
                min(systemAvailBytes, other.systemAvailBytes),
                max(systemTotalBytes, other.systemTotalBytes),
                lowMemory || other.lowMemory,
            )

        fun format(): String =
            "total_pss_mb=${mb(totalPssKb * 1024L)} " +
                "native_pss_mb=${mb(nativePssKb * 1024L)} " +
                "dalvik_pss_mb=${mb(dalvikPssKb * 1024L)} " +
                "other_pss_mb=${mb(otherPssKb * 1024L)} " +
                "java_heap_used_mb=${mb(javaHeapUsedBytes)} " +
                "java_heap_max_mb=${mb(javaHeapMaxBytes)} " +
                "native_heap_mb=${mb(nativeHeapAllocatedBytes)} " +
                "rss_mb=${mb(vmRssKb * 1024L)} " +
                "hwm_mb=${mb(vmHwmKb * 1024L)} " +
                "system_avail_mb=${mb(systemAvailBytes)} " +
                "system_total_mb=${mb(systemTotalBytes)} " +
                "low_memory=$lowMemory"

        companion object {
            fun empty(): MemorySnapshot =
                MemorySnapshot(0, 0, 0, 0, 0L, 0L, 0L, 0L, 0L, 0L, 0L, false)

            private fun mb(bytes: Long): String =
                String.format(Locale.US, "%.2f", bytes / (1024.0 * 1024.0))
        }
    }
}
