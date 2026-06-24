package com.gvcrt.clean

import org.json.JSONArray
import org.json.JSONObject

data class InputSpec(
    val tensorName: String,
    val source: String?,
    val path: String?,
    val shape: LongArray?,
)

data class OutputSpec(
    val tensorName: String,
    val shape: LongArray,
    val baseline: String?,
)

data class GraphStep(
    val name: String,
    val model: String,
    val inputs: List<InputSpec>,
    val outputs: List<OutputSpec>,
)

data class ModuleCase(
    val name: String,
    val steps: List<GraphStep>,
    val binaryComparisons: List<Pair<String, String>>,
)

data class CdfSpec(
    val cdf: String,
    val shape: LongArray,
    val cdfLengths: String,
    val offsets: String,
)

data class EntropySpec(
    val gaussian: CdfSpec,
    val z: CdfSpec,
    val zSymbols: String,
    val zStartOffset: Int,
    val zPerChannelSize: Int,
    val yPacked: List<String>,
    val payload: String,
    val twoEntropyCoders: Boolean,
)

data class StreamSpec(
    val path: String,
    val height: Int,
    val width: Int,
    val qp: Int,
    val ecPart: Int,
    val useAdaI: Int,
)

data class CleanManifest(
    val metadata: JSONObject,
    val modules: Map<String, List<ModuleCase>>,
    val entropy: Map<String, EntropySpec>,
    val stream: StreamSpec?,
) {
    companion object {
        fun parse(text: String): CleanManifest {
            val root = JSONObject(text)
            val modulesJson = root.getJSONObject("modules")
            val modules = mutableMapOf<String, List<ModuleCase>>()
            for (moduleName in modulesJson.keys()) {
                val cases = modulesJson.getJSONArray(moduleName)
                modules[moduleName] = (0 until cases.length()).map { idx ->
                    parseCase(cases.getJSONObject(idx))
                }
            }
            val entropy = root.optJSONObject("entropy")?.let { entropyJson ->
                entropyJson.keys().asSequence().associateWith { key ->
                    parseEntropy(entropyJson.getJSONObject(key))
                }
            } ?: emptyMap()
            return CleanManifest(
                root.optJSONObject("metadata") ?: JSONObject(),
                modules,
                entropy,
                root.optJSONObject("stream")?.let(::parseStream),
            )
        }

        private fun parseStream(json: JSONObject): StreamSpec =
            StreamSpec(
                path = json.getString("path"),
                height = json.getInt("height"),
                width = json.getInt("width"),
                qp = json.getInt("qp"),
                ecPart = json.getInt("ec_part"),
                useAdaI = json.getInt("use_ada_i"),
            )

        private fun parseEntropy(json: JSONObject): EntropySpec =
            EntropySpec(
                gaussian = parseCdf(json.getJSONObject("gaussian")),
                z = parseCdf(json.getJSONObject("z")),
                zSymbols = json.getString("z_symbols"),
                zStartOffset = json.getInt("z_start_offset"),
                zPerChannelSize = json.getInt("z_per_channel_size"),
                yPacked = json.getJSONArray("y_packed").toStringList(),
                payload = json.getString("payload"),
                twoEntropyCoders = json.getBoolean("two_entropy_coders"),
            )

        private fun parseCdf(json: JSONObject): CdfSpec =
            CdfSpec(
                cdf = json.getString("cdf"),
                shape = json.getJSONArray("shape").toLongArray(),
                cdfLengths = json.getString("cdf_lengths"),
                offsets = json.getString("offsets"),
            )

        private fun parseCase(json: JSONObject): ModuleCase {
            val stepsJson = json.optJSONArray("steps") ?: JSONArray()
            val steps = (0 until stepsJson.length()).map { parseStep(stepsJson.getJSONObject(it)) }
            val bin = json.optJSONArray("binary_comparisons") ?: JSONArray()
            val binaryComparisons = (0 until bin.length()).map {
                val item = bin.getJSONObject(it)
                item.getString("android") to item.getString("baseline")
            }
            return ModuleCase(json.getString("name"), steps, binaryComparisons)
        }

        private fun parseStep(json: JSONObject): GraphStep {
            val inputsJson = json.getJSONObject("inputs")
            val inputs = inputsJson.keys().asSequence().map { tensorName ->
                val spec = inputsJson.getJSONObject(tensorName)
                InputSpec(
                    tensorName = tensorName,
                    source = spec.optString("source").ifBlank { null },
                    path = spec.optString("path").ifBlank { null },
                    shape = spec.optJSONArray("shape")?.toLongArray(),
                )
            }.toList()

            val outputsJson = json.getJSONObject("outputs")
            val outputs = outputsJson.keys().asSequence().map { tensorName ->
                val spec = outputsJson.getJSONObject(tensorName)
                OutputSpec(
                    tensorName = tensorName,
                    shape = spec.getJSONArray("shape").toLongArray(),
                    baseline = spec.optString("baseline").ifBlank { null },
                )
            }.toList()

            return GraphStep(
                name = json.getString("name"),
                model = json.getString("model"),
                inputs = inputs,
                outputs = outputs,
            )
        }

        private fun JSONArray.toLongArray(): LongArray =
            LongArray(length()) { getLong(it) }

        private fun JSONArray.toStringList(): List<String> =
            List(length()) { getString(it) }
    }
}
