package com.gvcrt.clean

import android.content.Context

/** Keeps image-inference ORT sessions alive for the lifetime of the app process. */
object ProcessOnnxSessionCache {
    private val runners = mutableMapOf<OnnxBackend, OnnxSessionRunner>()

    @Synchronized
    fun get(context: Context, backend: OnnxBackend): OnnxSessionRunner =
        runners.getOrPut(backend) {
            OnnxSessionRunner(AssetStore(context.applicationContext), backend)
        }
}
