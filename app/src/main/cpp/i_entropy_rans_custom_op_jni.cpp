#include <jni.h>

#include <android/log.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "rans/rans.h"

namespace {

constexpr char kLogTag[] = "GVC_RT_CLEAN";
constexpr char kICustomOpName[] = "GVC_RT_RANS_ENCODE";
constexpr char kPCustomOpName[] = "GVC_RT_P_RANS_ENCODE";
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
    void* (*tensorData)(const TfLiteTensor*) = nullptr;
    TfLiteStatus (*tensorCopyFromBuffer)(TfLiteTensor*, const void*, size_t) = nullptr;
    TfLiteStatus (*tensorCopyToBuffer)(const TfLiteTensor*, void*, size_t) = nullptr;
    int (*opaqueInputCount)(const TfLiteOpaqueNode*) = nullptr;
    int (*opaqueOutputCount)(const TfLiteOpaqueNode*) = nullptr;
    const TfLiteOpaqueTensor* (*opaqueInput)(
        const TfLiteOpaqueContext*, const TfLiteOpaqueNode*, int) = nullptr;
    TfLiteOpaqueTensor* (*opaqueOutput)(
        TfLiteOpaqueContext*, const TfLiteOpaqueNode*, int) = nullptr;
    size_t (*opaqueByteSize)(const TfLiteOpaqueTensor*) = nullptr;
    void* (*opaqueData)(const TfLiteOpaqueTensor*) = nullptr;
};

struct CdfData {
    std::shared_ptr<std::vector<std::vector<int32_t>>> cdfs;
    std::shared_ptr<std::vector<int32_t>> lengths;
    std::shared_ptr<std::vector<int32_t>> offsets;
};

class CustomRansEncoder {
public:
    CustomRansEncoder(const CdfData& gaussian, const CdfData& z)
        : encoder_(std::make_shared<RansEncoderLibMultiThread>())
        , gaussianGroup_(encoder_->add_cdf(gaussian.cdfs, gaussian.lengths, gaussian.offsets))
        , zGroup_(encoder_->add_cdf(z.cdfs, z.lengths, z.offsets)) {}

    std::vector<uint8_t> encode(
        const std::shared_ptr<std::vector<int8_t>>& z,
        int startOffset,
        int perChannelSize,
        const std::vector<std::shared_ptr<std::vector<int16_t>>>& yStages) {
        encoder_->encode_z(z, zGroup_, startOffset, perChannelSize);
        for (const auto& stage : yStages) {
            encoder_->encode_y(stage, gaussianGroup_);
        }
        encoder_->flush();
        const std::vector<uint8_t> output = *encoder_->get_encoded_stream();
        encoder_->reset();
        return output;
    }

private:
    std::shared_ptr<RansEncoderLib> encoder_;
    int gaussianGroup_;
    int zGroup_;
};

struct RansCustomState {
    std::unique_ptr<CustomRansEncoder> encoder;
    int zStartOffset = 0;
    int zPerChannelSize = 0;
};

struct Runtime {
    TfliteApi api;
    TfLiteModel* model = nullptr;
    TfLiteInterpreterOptions* options = nullptr;
    TfLiteRegistrationExternal* registration = nullptr;
    TfLiteInterpreter* interpreter = nullptr;
};

TfliteApi* gApi = nullptr;
std::unique_ptr<RansCustomState> gRansState;
std::mutex gRansMutex;

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
        api.optionsAddRegistration = loadSymbol<decltype(api.optionsAddRegistration)>(api.library, "TfLiteInterpreterOptionsAddRegistrationExternal");
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
        api.tensorData = loadSymbol<decltype(api.tensorData)>(api.library, "TfLiteTensorData");
        api.tensorCopyFromBuffer = loadSymbol<decltype(api.tensorCopyFromBuffer)>(api.library, "TfLiteTensorCopyFromBuffer");
        api.tensorCopyToBuffer = loadSymbol<decltype(api.tensorCopyToBuffer)>(api.library, "TfLiteTensorCopyToBuffer");
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

const TfLiteOpaqueTensor* customInput(TfLiteOpaqueContext* context, TfLiteOpaqueNode* node, int index) {
    const TfLiteOpaqueTensor* tensor = gApi->opaqueInput(context, node, index);
    if (tensor == nullptr || gApi->opaqueData(tensor) == nullptr) {
        throw std::runtime_error("rANS custom op input is unavailable");
    }
    return tensor;
}

template <typename T>
const T* tensorData(const TfLiteOpaqueTensor* tensor) {
    return static_cast<const T*>(gApi->opaqueData(tensor));
}

CdfData copyCdf(
    const TfLiteOpaqueTensor* cdfTensor,
    const TfLiteOpaqueTensor* lengthsTensor,
    const TfLiteOpaqueTensor* offsetsTensor) {
    const size_t rowCount = gApi->opaqueByteSize(lengthsTensor) / sizeof(int32_t);
    const size_t offsetCount = gApi->opaqueByteSize(offsetsTensor) / sizeof(int32_t);
    const size_t cdfCount = gApi->opaqueByteSize(cdfTensor) / sizeof(int32_t);
    if (rowCount == 0 || rowCount != offsetCount || cdfCount % rowCount != 0) {
        throw std::runtime_error("invalid embedded CDF tensor sizes");
    }
    const size_t stride = cdfCount / rowCount;
    const int32_t* cdf = tensorData<int32_t>(cdfTensor);
    const int32_t* lengths = tensorData<int32_t>(lengthsTensor);
    const int32_t* offsets = tensorData<int32_t>(offsetsTensor);
    auto cdfs = std::make_shared<std::vector<std::vector<int32_t>>>(rowCount);
    for (size_t row = 0; row < rowCount; ++row) {
        cdfs->at(row).assign(cdf + row * stride, cdf + (row + 1) * stride);
    }
    return {
        cdfs,
        std::make_shared<std::vector<int32_t>>(lengths, lengths + rowCount),
        std::make_shared<std::vector<int32_t>>(offsets, offsets + rowCount),
    };
}

std::shared_ptr<std::vector<int8_t>> copyZFromNhwc(const float* source) {
    constexpr int height = 4;
    constexpr int width = 8;
    constexpr int channels = 128;
    auto output = std::make_shared<std::vector<int8_t>>(height * width * channels);
    size_t destination = 0;
    for (int channel = 0; channel < channels; ++channel) {
        for (int row = 0; row < height; ++row) {
            for (int column = 0; column < width; ++column) {
                const float value = source[(row * width + column) * channels + channel];
                const int symbol = static_cast<int>(value);
                if (value != static_cast<float>(symbol) || symbol < -128 || symbol > 127) {
                    throw std::runtime_error("non-integral or out-of-range z symbol");
                }
                output->at(destination++) = static_cast<int8_t>(symbol);
            }
        }
    }
    return output;
}

std::shared_ptr<std::vector<int16_t>> packYFromNhwc(const float* symbols, const float* scales) {
    constexpr int height = 16;
    constexpr int width = 32;
    constexpr int channels = 64;
    constexpr float scaleMin = 0.11f;
    constexpr float scaleMax = 16.0f;
    constexpr float logScaleMin = -2.2072749f;
    constexpr float logStepRecip = 25.502707f;
    auto output = std::make_shared<std::vector<int16_t>>(height * width * channels);
    size_t destination = 0;
    for (int channel = 0; channel < channels; ++channel) {
        for (int row = 0; row < height; ++row) {
            for (int column = 0; column < width; ++column) {
                const size_t sourceIndex = static_cast<size_t>((row * width + column) * channels + channel);
                const float value = symbols[sourceIndex];
                const int symbol = static_cast<int>(value);
                if (value != static_cast<float>(symbol) || symbol < -128 || symbol > 127) {
                    throw std::runtime_error("non-integral or out-of-range y symbol");
                }
                const float scale = std::max(scaleMin, std::min(scaleMax, scales[sourceIndex]));
                int cdfIndex = static_cast<int>((std::log(scale) - logScaleMin) * logStepRecip);
                cdfIndex = std::max(0, std::min(127, cdfIndex));
                output->at(destination++) = static_cast<int16_t>(symbol * 256 + cdfIndex);
            }
        }
    }
    return output;
}

TfLiteStatus ransInvoke(TfLiteOpaqueContext* context, TfLiteOpaqueNode* node) {
    std::lock_guard<std::mutex> lock(gRansMutex);
    try {
        const int inputCount = gApi == nullptr ? 0 : gApi->opaqueInputCount(node);
        if ((inputCount != 13 && inputCount != 17) || gApi->opaqueOutputCount(node) != 2) {
            throw std::runtime_error("rANS custom op input/output count mismatch");
        }
        const int stageCount = (inputCount - 9) / 2;
        const int scalesStart = 1 + stageCount;
        const int constantsStart = 1 + stageCount * 2;
        std::vector<const TfLiteOpaqueTensor*> inputs;
        inputs.reserve(inputCount);
        for (int index = 0; index < inputCount; ++index) {
            inputs.push_back(customInput(context, node, index));
        }
        if (!gRansState) {
            const CdfData gaussian = copyCdf(
                inputs[constantsStart], inputs[constantsStart + 1], inputs[constantsStart + 2]);
            const CdfData z = copyCdf(
                inputs[constantsStart + 3], inputs[constantsStart + 4], inputs[constantsStart + 5]);
            gRansState = std::make_unique<RansCustomState>();
            gRansState->encoder = std::make_unique<CustomRansEncoder>(gaussian, z);
            gRansState->zStartOffset = tensorData<int32_t>(inputs[constantsStart + 6])[0];
            gRansState->zPerChannelSize = tensorData<int32_t>(inputs[constantsStart + 7])[0];
            if (gRansState->zPerChannelSize <= 0) {
                throw std::runtime_error("invalid embedded z per-channel size");
            }
        }
        auto z = copyZFromNhwc(tensorData<float>(inputs[0]));
        std::vector<std::shared_ptr<std::vector<int16_t>>> yStages;
        yStages.reserve(stageCount);
        for (int stage = 0; stage < stageCount; ++stage) {
            yStages.push_back(packYFromNhwc(
                tensorData<float>(inputs[1 + stage]),
                tensorData<float>(inputs[scalesStart + stage])));
        }
        const std::vector<uint8_t> payload = gRansState->encoder->encode(
            z, gRansState->zStartOffset, gRansState->zPerChannelSize, yStages);
        TfLiteOpaqueTensor* payloadOutput = gApi->opaqueOutput(context, node, 0);
        TfLiteOpaqueTensor* sizeOutput = gApi->opaqueOutput(context, node, 1);
        if (payloadOutput == nullptr || sizeOutput == nullptr ||
            gApi->opaqueData(payloadOutput) == nullptr || gApi->opaqueData(sizeOutput) == nullptr) {
            throw std::runtime_error("rANS custom op output is unavailable");
        }
        const size_t capacity = gApi->opaqueByteSize(payloadOutput);
        if (payload.size() > capacity || gApi->opaqueByteSize(sizeOutput) != sizeof(int32_t)) {
            throw std::runtime_error("rANS payload exceeds graph output capacity");
        }
        std::memcpy(gApi->opaqueData(payloadOutput), payload.data(), payload.size());
        *static_cast<int32_t*>(gApi->opaqueData(sizeOutput)) = static_cast<int32_t>(payload.size());
        return kTfLiteOk;
    } catch (const std::exception& error) {
        __android_log_print(ANDROID_LOG_ERROR, kLogTag, "entropy_rans_custom_op_failed %s", error.what());
        return 1;
    }
}

void destroyRuntime(Runtime* runtime) {
    if (runtime == nullptr) return;
    if (runtime->interpreter != nullptr) runtime->api.interpreterDelete(runtime->interpreter);
    if (runtime->registration != nullptr) runtime->api.registrationDelete(runtime->registration);
    if (runtime->options != nullptr) runtime->api.optionsDelete(runtime->options);
    if (runtime->model != nullptr) runtime->api.modelDelete(runtime->model);
    {
        std::lock_guard<std::mutex> lock(gRansMutex);
        gRansState.reset();
        gApi = nullptr;
    }
    if (runtime->api.library != nullptr) dlclose(runtime->api.library);
    delete runtime;
}

Runtime* createRuntime(
    const std::string& modelPath,
    void* delegate,
    const char* customOpName,
    int expectedInputs,
    int expectedOutputs) {
    if (delegate == nullptr) throw std::runtime_error("Neuron delegate handle is null");
    auto runtime = std::make_unique<Runtime>();
    runtime->api = loadApi();
    gApi = &runtime->api;
    try {
        runtime->model = runtime->api.modelCreateFromFile(modelPath.c_str());
        if (runtime->model == nullptr) throw std::runtime_error("TfLiteModelCreateFromFile failed");
        runtime->options = runtime->api.optionsCreate();
        if (runtime->options == nullptr) throw std::runtime_error("TfLiteInterpreterOptionsCreate failed");
        runtime->registration = runtime->api.registrationCreate(kTfLiteBuiltinCustom, customOpName, 1);
        if (runtime->registration == nullptr) throw std::runtime_error("rANS registration create failed");
        runtime->api.registrationSetInvoke(runtime->registration, ransInvoke);
        runtime->api.optionsAddRegistration(runtime->options, runtime->registration);
        runtime->api.optionsAddDelegate(runtime->options, delegate);
        runtime->interpreter = runtime->api.interpreterCreate(runtime->model, runtime->options);
        if (runtime->interpreter == nullptr) throw std::runtime_error("TfLiteInterpreterCreate failed");
        if (runtime->api.allocateTensors(runtime->interpreter) != kTfLiteOk) {
            throw std::runtime_error("TfLiteInterpreterAllocateTensors failed");
        }
        if (runtime->api.inputCount(runtime->interpreter) != expectedInputs ||
            runtime->api.outputCount(runtime->interpreter) != expectedOutputs) {
            throw std::runtime_error("merged rANS model input/output count mismatch");
        }
        return runtime.release();
    } catch (...) {
        destroyRuntime(runtime.release());
        throw;
    }
}

void copyJavaInput(JNIEnv* env, Runtime* runtime, int index, jbyteArray input) {
    TfLiteTensor* tensor = runtime->api.inputTensor(runtime->interpreter, index);
    const size_t bytes = runtime->api.tensorByteSize(tensor);
    if (env->GetArrayLength(input) != static_cast<jsize>(bytes)) {
        throw std::runtime_error("merged rANS input byte size mismatch");
    }
    std::vector<jbyte> data(bytes);
    env->GetByteArrayRegion(input, 0, static_cast<jsize>(bytes), data.data());
    if (runtime->api.tensorCopyFromBuffer(tensor, data.data(), bytes) != kTfLiteOk) {
        throw std::runtime_error("merged rANS input copy failed");
    }
}

jobjectArray collectOutputs(JNIEnv* env, Runtime* runtime, jint outputMode) {
    jclass byteArrayClass = env->FindClass("[B");
    if (outputMode == 0) {
        jobjectArray result = env->NewObjectArray(0, byteArrayClass, nullptr);
        env->DeleteLocalRef(byteArrayClass);
        return result;
    }
    const int outputCount = runtime->api.outputCount(runtime->interpreter);
    if (outputMode == 2) {
        const int yHatIndex = outputCount - 3;
        const TfLiteTensor* yHatTensor = runtime->api.outputTensor(runtime->interpreter, yHatIndex);
        const TfLiteTensor* payloadTensor = runtime->api.outputTensor(runtime->interpreter, yHatIndex + 1);
        const TfLiteTensor* sizeTensor = runtime->api.outputTensor(runtime->interpreter, yHatIndex + 2);
        int32_t payloadSize = 0;
        if (runtime->api.tensorCopyToBuffer(sizeTensor, &payloadSize, sizeof(payloadSize)) != kTfLiteOk ||
            payloadSize < 0 || static_cast<size_t>(payloadSize) > runtime->api.tensorByteSize(payloadTensor)) {
            throw std::runtime_error("invalid canonical rANS payload size");
        }
        jobjectArray result = env->NewObjectArray(2, byteArrayClass, nullptr);
        const size_t yHatBytes = runtime->api.tensorByteSize(yHatTensor);
        std::vector<uint8_t> yHat(yHatBytes);
        if (runtime->api.tensorCopyToBuffer(yHatTensor, yHat.data(), yHatBytes) != kTfLiteOk) {
            throw std::runtime_error("canonical y_hat copy failed");
        }
        jbyteArray yHatArray = env->NewByteArray(static_cast<jsize>(yHatBytes));
        env->SetByteArrayRegion(yHatArray, 0, static_cast<jsize>(yHatBytes), reinterpret_cast<const jbyte*>(yHat.data()));
        env->SetObjectArrayElement(result, 0, yHatArray);
        env->DeleteLocalRef(yHatArray);
        const void* payloadData = runtime->api.tensorData(payloadTensor);
        if (payloadData == nullptr) throw std::runtime_error("canonical payload data is unavailable");
        jbyteArray payloadArray = env->NewByteArray(payloadSize);
        env->SetByteArrayRegion(payloadArray, 0, payloadSize, reinterpret_cast<const jbyte*>(payloadData));
        env->SetObjectArrayElement(result, 1, payloadArray);
        env->DeleteLocalRef(payloadArray);
        env->DeleteLocalRef(byteArrayClass);
        return result;
    }
    if (outputMode != 1) throw std::runtime_error("unsupported merged rANS output mode");
    jobjectArray result = env->NewObjectArray(outputCount, byteArrayClass, nullptr);
    for (int index = 0; index < outputCount; ++index) {
        const TfLiteTensor* tensor = runtime->api.outputTensor(runtime->interpreter, index);
        const size_t bytes = runtime->api.tensorByteSize(tensor);
        std::vector<uint8_t> output(bytes);
        if (runtime->api.tensorCopyToBuffer(tensor, output.data(), bytes) != kTfLiteOk) {
            throw std::runtime_error("merged rANS output copy failed");
        }
        jbyteArray array = env->NewByteArray(static_cast<jsize>(bytes));
        env->SetByteArrayRegion(array, 0, static_cast<jsize>(bytes), reinterpret_cast<const jbyte*>(output.data()));
        env->SetObjectArrayElement(result, index, array);
        env->DeleteLocalRef(array);
    }
    env->DeleteLocalRef(byteArrayClass);
    return result;
}

void throwJava(JNIEnv* env, const std::string& message) {
    jclass exception = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(exception, message.c_str());
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_gvcrt_clean_IEntropyRansMergedRuntime_nativeCreate(
    JNIEnv* env, jclass, jstring modelPath, jlong delegateHandle) {
    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    if (path == nullptr) return 0;
    const std::string model(path);
    env->ReleaseStringUTFChars(modelPath, path);
    try {
        return reinterpret_cast<jlong>(createRuntime(
            model,
            reinterpret_cast<void*>(static_cast<uintptr_t>(delegateHandle)),
            kICustomOpName,
            1,
            12));
    } catch (const std::exception& error) {
        throwJava(env, error.what());
        return 0;
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_gvcrt_clean_IEntropyRansMergedRuntime_nativeRun(
    JNIEnv* env, jclass, jlong handle, jbyteArray input, jint outputMode) {
    try {
        auto* runtime = reinterpret_cast<Runtime*>(handle);
        if (runtime == nullptr) throw std::runtime_error("merged rANS runtime is closed");
        copyJavaInput(env, runtime, 0, input);
        if (runtime->api.invoke(runtime->interpreter) != kTfLiteOk) {
            throw std::runtime_error("merged rANS invoke failed");
        }
        return collectOutputs(env, runtime, outputMode);
    } catch (const std::exception& error) {
        throwJava(env, error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_gvcrt_clean_IEntropyRansMergedRuntime_nativeClose(JNIEnv*, jclass, jlong handle) {
    destroyRuntime(reinterpret_cast<Runtime*>(handle));
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_gvcrt_clean_PEntropyRansMergedRuntime_nativeCreate(
    JNIEnv* env, jclass, jstring modelPath, jlong delegateHandle) {
    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    if (path == nullptr) return 0;
    const std::string model(path);
    env->ReleaseStringUTFChars(modelPath, path);
    try {
        return reinterpret_cast<jlong>(createRuntime(
            model,
            reinterpret_cast<void*>(static_cast<uintptr_t>(delegateHandle)),
            kPCustomOpName,
            2,
            8));
    } catch (const std::exception& error) {
        throwJava(env, error.what());
        return 0;
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_gvcrt_clean_PEntropyRansMergedRuntime_nativeRun(
    JNIEnv* env,
    jclass,
    jlong handle,
    jbyteArray y,
    jbyteArray ctxT,
    jint outputMode) {
    try {
        auto* runtime = reinterpret_cast<Runtime*>(handle);
        if (runtime == nullptr) throw std::runtime_error("P merged rANS runtime is closed");
        copyJavaInput(env, runtime, 0, y);
        copyJavaInput(env, runtime, 1, ctxT);
        if (runtime->api.invoke(runtime->interpreter) != kTfLiteOk) {
            throw std::runtime_error("P merged rANS invoke failed");
        }
        return collectOutputs(env, runtime, outputMode);
    } catch (const std::exception& error) {
        throwJava(env, error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_gvcrt_clean_PEntropyRansMergedRuntime_nativeClose(JNIEnv*, jclass, jlong handle) {
    destroyRuntime(reinterpret_cast<Runtime*>(handle));
}
