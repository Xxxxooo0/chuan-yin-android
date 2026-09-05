package com.gvcrt.clean

class SelectedSmallEntropyRuntime private constructor(
    private val mtk: SmallEntropyMtkRuntime?,
    private val gpu: SmallEntropyGpuRuntime?,
) : AutoCloseable {
    val inputSizes: LongArray = mtk?.inputSizes ?: gpu!!.inputSizes
    val outputSizes: LongArray = mtk?.outputSizes ?: gpu!!.outputSizes
    val optionsSummary: String = mtk?.optionsSummary ?: gpu!!.optionsSummary

    fun run(inputs: List<ByteArray>): List<ByteArray> =
        mtk?.run(inputs) ?: gpu!!.run(inputs)

    override fun close() {
        mtk?.close() ?: gpu!!.close()
    }

    companion object {
        fun mtk(runtime: SmallEntropyMtkRuntime): SelectedSmallEntropyRuntime =
            SelectedSmallEntropyRuntime(mtk = runtime, gpu = null)

        fun gpu(runtime: SmallEntropyGpuRuntime): SelectedSmallEntropyRuntime =
            SelectedSmallEntropyRuntime(mtk = null, gpu = runtime)
    }
}
