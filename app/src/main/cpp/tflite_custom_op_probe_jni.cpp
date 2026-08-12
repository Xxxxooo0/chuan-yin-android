#include <jni.h>

#include <android/log.h>
#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr char kLogTag[] = "GVC_RT_CLEAN";
constexpr char kCustomOpName[] = "GVC_RT_CPU_IDENTITY";
constexpr int kTfLiteOk = 0;
constexpr int kTfLiteBuiltinCustom = 32;

struct TfLiteModel;
struct TfLiteInterpreterOptions;
struct TfLiteInterpreter;
struct TfLiteTensor;
struct TfLiteRegistrationExternal;
struct TfLiteOpaqueContext;
struct TfLiteOpaqueNode;
struct TfLiteOpaqueTensor;

using TfLiteStatus = int;

struct TfliteApi {
    void* library = nullptr;

    TfLiteModel* (*modelCreateFromFile)(const char*) = nullptr;
    void (*modelDelete)(TfLiteModel*) = nullptr;
    TfLiteInterpreterOptions* (*optionsCreate)() = nullptr;
    void (*optionsDelete)(TfLiteInterpreterOptions*) = nullptr;
    void (*optionsAddDelegate)(TfLiteInterpreterOptions*, void*) = nullptr;
    void (*optionsAddRegistration)(TfLiteInterpreterOptions*, TfLiteRegistrationExternal*) = nullptr;
    TfLiteRegistrationExternal* (*registrationCreate)(int, const char*, int) = nullptr;
    void (*registrationDelete)(TfLiteRegistrationExternal*) = nullptr;
    void (*registrationSetInvoke)(
        TfLiteRegistrationExternal*,
        TfLiteStatus (*)(TfLiteOpaqueContext*, TfLiteOpaqueNode*)) = nullptr;
    TfLiteInterpreter* (*interpreterCreate)(const TfLiteModel*, const TfLiteInterpreterOptions*) = nullptr;
    void (*interpreterDelete)(TfLiteInterpreter*) = nullptr;
    TfLiteStatus (*allocateTensors)(TfLiteInterpreter*) = nullptr;
    TfLiteStatus (*invoke)(TfLiteInterpreter*) = nullptr;
    int32_t (*inputCount)(const TfLiteInterpreter*) = nullptr;
    int32_t (*outputCount)(const TfLiteInterpreter*) = nullptr;
    TfLiteTensor* (*inputTensor)(const TfLiteInterpreter*, int32_t) = nullptr;
    const TfLiteTensor* (*outputTensor)(const TfLiteInterpreter*, int32_t) = nullptr;
    size_t (*tensorByteSize)(const TfLiteTensor*) = nullptr;
    TfLiteStatus (*tensorCopyFromBuffer)(TfLiteTensor*, const void*, size_t) = nullptr;
    TfLiteStatus (*tensorCopyToBuffer)(const TfLiteTensor*, void*, size_t) = nullptr;
    const char* (*version)() = nullptr;
    int (*opaqueInputCount)(const TfLiteOpaqueNode*) = nullptr;
    int (*opaqueOutputCount)(const TfLiteOpaqueNode*) = nullptr;
    const TfLiteOpaqueTensor* (*opaqueInput)(
        const TfLiteOpaqueContext*, const TfLiteOpaqueNode*, int) = nullptr;
    TfLiteOpaqueTensor* (*opaqueOutput)(
        TfLiteOpaqueContext*, const TfLiteOpaqueNode*, int) = nullptr;
    size_t (*opaqueByteSize)(const TfLiteOpaqueTensor*) = nullptr;
    void* (*opaqueData)(const TfLiteOpaqueTensor*) = nullptr;
};

TfliteApi* gApi = nullptr;
std::atomic<int> gCustomInvokeCount{0};

template <typename T>
T loadSymbol(void* library, const char* name) {
    dlerror();
    void* symbol = dlsym(library, name);
    const char* error = dlerror();
    if (error != nullptr || symbol == nullptr) {
        throw std::runtime_error(std::string("missing MTK TFLite symbol ") + name);
    }
    return reinterpret_cast<T>(symbol);
}

TfliteApi loadApi() {
    TfliteApi api;
    api.library = dlopen("libtensorflowlite_jni_mtk.so", RTLD_NOW | RTLD_LOCAL);
    if (api.library == nullptr) {
        throw std::runtime_error(std::string("dlopen libtensorflowlite_jni_mtk.so failed: ") + dlerror());
    }
    try {
        api.modelCreateFromFile = loadSymbol<decltype(api.modelCreateFromFile)>(api.library, "TfLiteModelCreateFromFile");
        api.modelDelete = loadSymbol<decltype(api.modelDelete)>(api.library, "TfLiteModelDelete");
        api.optionsCreate = loadSymbol<decltype(api.optionsCreate)>(api.library, "TfLiteInterpreterOptionsCreate");
        api.optionsDelete = loadSymbol<decltype(api.optionsDelete)>(api.library, "TfLiteInterpreterOptionsDelete");
        api.optionsAddDelegate = loadSymbol<decltype(api.optionsAddDelegate)>(api.library, "TfLiteInterpreterOptionsAddDelegate");
        api.optionsAddRegistration = loadSymbol<decltype(api.optionsAddRegistration)>(
            api.library, "TfLiteInterpreterOptionsAddRegistrationExternal");
        api.registrationCreate = loadSymbol<decltype(api.registrationCreate)>(api.library, "TfLiteRegistrationExternalCreate");
        api.registrationDelete = loadSymbol<decltype(api.registrationDelete)>(api.library, "TfLiteRegistrationExternalDelete");
        api.registrationSetInvoke = loadSymbol<decltype(api.registrationSetInvoke)>(api.library, "TfLiteRegistrationExternalSetInvoke");
        api.interpreterCreate = loadSymbol<decltype(api.interpreterCreate)>(api.library, "TfLiteInterpreterCreate");
        api.interpreterDelete = loadSymbol<decltype(api.interpreterDelete)>(api.library, "TfLiteInterpreterDelete");
        api.allocateTensors = loadSymbol<decltype(api.allocateTensors)>(api.library, "TfLiteInterpreterAllocateTensors");
        api.invoke = loadSymbol<decltype(api.invoke)>(api.library, "TfLiteInterpreterInvoke");
        api.inputCount = loadSymbol<decltype(api.inputCount)>(api.library, "TfLiteInterpreterGetInputTensorCount");
        api.outputCount = loadSymbol<decltype(api.outputCount)>(api.library, "TfLiteInterpreterGetOutputTensorCount");
        api.inputTensor = loadSymbol<decltype(api.inputTensor)>(api.library, "TfLiteInterpreterGetInputTensor");
        api.outputTensor = loadSymbol<decltype(api.outputTensor)>(api.library, "TfLiteInterpreterGetOutputTensor");
        api.tensorByteSize = loadSymbol<decltype(api.tensorByteSize)>(api.library, "TfLiteTensorByteSize");
        api.tensorCopyFromBuffer = loadSymbol<decltype(api.tensorCopyFromBuffer)>(api.library, "TfLiteTensorCopyFromBuffer");
        api.tensorCopyToBuffer = loadSymbol<decltype(api.tensorCopyToBuffer)>(api.library, "TfLiteTensorCopyToBuffer");
        api.version = loadSymbol<decltype(api.version)>(api.library, "TfLiteVersion");
        api.opaqueInputCount = loadSymbol<decltype(api.opaqueInputCount)>(api.library, "TfLiteOpaqueNodeNumberOfInputs");
        api.opaqueOutputCount = loadSymbol<decltype(api.opaqueOutputCount)>(api.library, "TfLiteOpaqueNodeNumberOfOutputs");
        api.opaqueInput = loadSymbol<decltype(api.opaqueInput)>(api.library, "TfLiteOpaqueNodeGetInput");
        api.opaqueOutput = loadSymbol<decltype(api.opaqueOutput)>(api.library, "TfLiteOpaqueNodeGetOutput");
        api.opaqueByteSize = loadSymbol<decltype(api.opaqueByteSize)>(api.library, "TfLiteOpaqueTensorByteSize");
        api.opaqueData = loadSymbol<decltype(api.opaqueData)>(api.library, "TfLiteOpaqueTensorData");
        return api;
    } catch (...) {
        dlclose(api.library);
        throw;
    }
}

TfLiteStatus customIdentityInvoke(TfLiteOpaqueContext* context, TfLiteOpaqueNode* node) {
    TfliteApi* api = gApi;
    if (api == nullptr || api->opaqueInputCount(node) != 1 || api->opaqueOutputCount(node) != 1) {
        return 1;
    }
    const TfLiteOpaqueTensor* input = api->opaqueInput(context, node, 0);
    TfLiteOpaqueTensor* output = api->opaqueOutput(context, node, 0);
    if (input == nullptr || output == nullptr) {
        return 1;
    }
    const size_t inputBytes = api->opaqueByteSize(input);
    const size_t outputBytes = api->opaqueByteSize(output);
    if (inputBytes != outputBytes) {
        return 1;
    }
    void* inputData = api->opaqueData(input);
    void* outputData = api->opaqueData(output);
    if (inputData == nullptr || outputData == nullptr) {
        return 1;
    }
    std::memcpy(outputData, inputData, inputBytes);
    ++gCustomInvokeCount;
    return kTfLiteOk;
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>((values.size() - 1) * fraction);
    return values[index];
}

std::string runProbe(const std::string& modelPath, void* delegate, int warmupRuns, int measuredRuns) {
    if (delegate == nullptr) {
        throw std::runtime_error("Neuron delegate handle is null");
    }
    if (warmupRuns < 0 || measuredRuns <= 0) {
        throw std::runtime_error("invalid warmup/measured runs");
    }

    TfliteApi api = loadApi();
    gApi = &api;
    TfLiteModel* model = nullptr;
    TfLiteInterpreterOptions* options = nullptr;
    TfLiteRegistrationExternal* registration = nullptr;
    TfLiteInterpreter* interpreter = nullptr;
    try {
        model = api.modelCreateFromFile(modelPath.c_str());
        if (model == nullptr) throw std::runtime_error("TfLiteModelCreateFromFile failed");
        options = api.optionsCreate();
        if (options == nullptr) throw std::runtime_error("TfLiteInterpreterOptionsCreate failed");
        registration = api.registrationCreate(kTfLiteBuiltinCustom, kCustomOpName, 1);
        if (registration == nullptr) throw std::runtime_error("custom registration create failed");
        api.registrationSetInvoke(registration, customIdentityInvoke);
        api.optionsAddRegistration(options, registration);
        api.optionsAddDelegate(options, delegate);
        interpreter = api.interpreterCreate(model, options);
        if (interpreter == nullptr) throw std::runtime_error("TfLiteInterpreterCreate failed");
        if (api.allocateTensors(interpreter) != kTfLiteOk) {
            throw std::runtime_error("TfLiteInterpreterAllocateTensors failed");
        }
        if (api.inputCount(interpreter) != 1 || api.outputCount(interpreter) != 1) {
            throw std::runtime_error("probe requires one input and one output");
        }
        TfLiteTensor* inputTensor = api.inputTensor(interpreter, 0);
        const TfLiteTensor* outputTensor = api.outputTensor(interpreter, 0);
        const size_t inputBytes = api.tensorByteSize(inputTensor);
        const size_t outputBytes = api.tensorByteSize(outputTensor);
        if (inputBytes != outputBytes || inputBytes % sizeof(float) != 0) {
            throw std::runtime_error("probe input/output byte size mismatch");
        }
        std::vector<float> input(inputBytes / sizeof(float));
        std::vector<float> output(outputBytes / sizeof(float));
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<float>((i % 31) + 1) / 32.0f;
        }
        if (api.tensorCopyFromBuffer(inputTensor, input.data(), inputBytes) != kTfLiteOk) {
            throw std::runtime_error("TfLiteTensorCopyFromBuffer failed");
        }

        gCustomInvokeCount = 0;
        for (int i = 0; i < warmupRuns; ++i) {
            if (api.invoke(interpreter) != kTfLiteOk) throw std::runtime_error("warmup invoke failed");
        }
        std::vector<double> times;
        times.reserve(measuredRuns);
        for (int i = 0; i < measuredRuns; ++i) {
            const auto start = std::chrono::steady_clock::now();
            if (api.invoke(interpreter) != kTfLiteOk) throw std::runtime_error("measured invoke failed");
            const auto end = std::chrono::steady_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }
        if (api.tensorCopyToBuffer(outputTensor, output.data(), outputBytes) != kTfLiteOk) {
            throw std::runtime_error("TfLiteTensorCopyToBuffer failed");
        }
        double maxAbs = 0.0;
        double sumAbs = 0.0;
        for (size_t i = 0; i < input.size(); ++i) {
            const double error = std::abs(static_cast<double>(output[i]) - input[i]);
            maxAbs = std::max(maxAbs, error);
            sumAbs += error;
        }
        const double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        std::ostringstream result;
        result << "tflite_version=" << api.version()
               << " custom_op=" << kCustomOpName
               << " custom_invoke_count=" << gCustomInvokeCount.load()
               << " warmup=" << warmupRuns
               << " measured=" << measuredRuns
               << " mean_ms=" << mean
               << " p50_ms=" << percentile(times, 0.50)
               << " p90_ms=" << percentile(times, 0.90)
               << " max_abs=" << maxAbs
               << " mean_abs=" << (sumAbs / input.size())
               << " output_count=" << output.size();

        api.interpreterDelete(interpreter);
        api.registrationDelete(registration);
        api.optionsDelete(options);
        api.modelDelete(model);
        gApi = nullptr;
        dlclose(api.library);
        return result.str();
    } catch (...) {
        if (interpreter != nullptr) api.interpreterDelete(interpreter);
        if (registration != nullptr) api.registrationDelete(registration);
        if (options != nullptr) api.optionsDelete(options);
        if (model != nullptr) api.modelDelete(model);
        gApi = nullptr;
        if (api.library != nullptr) dlclose(api.library);
        throw;
    }
}

void throwJava(JNIEnv* env, const std::string& message) {
    jclass exception = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(exception, message.c_str());
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_gvcrt_clean_TfliteCustomOpPartitionProbe_nativeRun(
    JNIEnv* env,
    jclass,
    jstring modelPath,
    jlong delegateHandle,
    jint warmupRuns,
    jint measuredRuns) {
    const char* pathChars = env->GetStringUTFChars(modelPath, nullptr);
    if (pathChars == nullptr) return nullptr;
    const std::string path(pathChars);
    env->ReleaseStringUTFChars(modelPath, pathChars);
    try {
        const std::string result = runProbe(
            path,
            reinterpret_cast<void*>(static_cast<uintptr_t>(delegateHandle)),
            warmupRuns,
            measuredRuns);
        __android_log_print(ANDROID_LOG_INFO, kLogTag, "custom_op_partition_native %s", result.c_str());
        return env->NewStringUTF(result.c_str());
    } catch (const std::exception& error) {
        __android_log_print(ANDROID_LOG_ERROR, kLogTag, "custom_op_partition_native_failed %s", error.what());
        throwJava(env, error.what());
        return nullptr;
    }
}
