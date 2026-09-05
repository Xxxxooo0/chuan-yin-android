package com.gvcrt.clean

class SelectedModelRuntime private constructor(
    private val neuron: OfficialNeuronRuntime?,
    private val gpu: GenericTfliteGpuRuntime?,
) : AutoCloseable {
    val inputSizes: LongArray = neuron?.inputSizes ?: gpu!!.inputSizes
    val outputSizes: LongArray = neuron?.outputSizes ?: gpu!!.outputSizes
    val optionsSummary: String = neuron?.optionsSummary ?: gpu!!.optionsSummary

    fun run(inputs: List<ByteArray>, copyOutputs: Boolean = true): List<ByteArray> =
        neuron?.run(inputs, copyOutputs) ?: gpu!!.run(inputs, copyOutputs)

    override fun close() {
        neuron?.close() ?: gpu!!.close()
    }

    companion object {
        fun neuron(runtime: OfficialNeuronRuntime): SelectedModelRuntime =
            SelectedModelRuntime(neuron = runtime, gpu = null)

        fun gpu(runtime: GenericTfliteGpuRuntime): SelectedModelRuntime =
            SelectedModelRuntime(neuron = null, gpu = runtime)
    }
}
