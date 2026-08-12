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
constexpr char kIDecodeZOp[] = "GVC_RT_RANS_DECODE_Z";
constexpr char kIDecodeYOp[] = "GVC_RT_RANS_DECODE_Y";
constexpr char kPDecodeZOp[] = "GVC_RT_P_RANS_DECODE_Z";
constexpr char kPDecodeYOp[] = "GVC_RT_P_RANS_DECODE_Y";
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
    void (*registrationSetInvoke)(TfLiteRegistrationExternal*, TfLiteStatus (*)(TfLiteOpaqueContext*, TfLiteOpaqueNode*)) = nullptr;
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
    const TfLiteOpaqueTensor* (*opaqueInput)(const TfLiteOpaqueContext*, const TfLiteOpaqueNode*, int) = nullptr;
    TfLiteOpaqueTensor* (*opaqueOutput)(TfLiteOpaqueContext*, const TfLiteOpaqueNode*, int) = nullptr;
    size_t (*opaqueByteSize)(const TfLiteOpaqueTensor*) = nullptr;
    void* (*opaqueData)(const TfLiteOpaqueTensor*) = nullptr;
};

struct CdfData {
    std::shared_ptr<std::vector<std::vector<int32_t>>> cdfs;
    std::shared_ptr<std::vector<int32_t>> lengths;
    std::shared_ptr<std::vector<int32_t>> offsets;
};

struct DecodeState {
    RansDecoderLib decoder;
    int gaussianGroup = -1;
    int zGroup = -1;
    int zStartOffset = 0;
    int zPerChannelSize = 0;

    DecodeState(const CdfData& gaussian, const CdfData& z, int startOffset, int perChannelSize)
        : gaussianGroup(decoder.add_cdf(gaussian.cdfs, gaussian.lengths, gaussian.offsets)),
          zGroup(decoder.add_cdf(z.cdfs, z.lengths, z.offsets)),
          zStartOffset(startOffset),
          zPerChannelSize(perChannelSize) {}
};

struct Runtime {
    TfliteApi api;
    TfLiteModel* model = nullptr;
    TfLiteInterpreterOptions* options = nullptr;
    TfLiteRegistrationExternal* decodeZRegistration = nullptr;
    TfLiteRegistrationExternal* decodeYRegistration = nullptr;
    TfLiteInterpreter* interpreter = nullptr;
};

TfliteApi* gApi = nullptr;
std::unique_ptr<DecodeState> gDecodeState;
std::mutex gDecodeMutex;

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
#define LOAD_API(member, symbol) api.member = loadSymbol<decltype(api.member)>(api.library, symbol)
        LOAD_API(modelCreateFromFile, "TfLiteModelCreateFromFile");
        LOAD_API(modelDelete, "TfLiteModelDelete");
        LOAD_API(optionsCreate, "TfLiteInterpreterOptionsCreate");
        LOAD_API(optionsDelete, "TfLiteInterpreterOptionsDelete");
        LOAD_API(optionsAddDelegate, "TfLiteInterpreterOptionsAddDelegate");
        LOAD_API(optionsAddRegistration, "TfLiteInterpreterOptionsAddRegistrationExternal");
        LOAD_API(registrationCreate, "TfLiteRegistrationExternalCreate");
        LOAD_API(registrationDelete, "TfLiteRegistrationExternalDelete");
        LOAD_API(registrationSetInvoke, "TfLiteRegistrationExternalSetInvoke");
        LOAD_API(interpreterCreate, "TfLiteInterpreterCreate");
        LOAD_API(interpreterDelete, "TfLiteInterpreterDelete");
        LOAD_API(allocateTensors, "TfLiteInterpreterAllocateTensors");
        LOAD_API(invoke, "TfLiteInterpreterInvoke");
        LOAD_API(inputCount, "TfLiteInterpreterGetInputTensorCount");
        LOAD_API(outputCount, "TfLiteInterpreterGetOutputTensorCount");
        LOAD_API(inputTensor, "TfLiteInterpreterGetInputTensor");
        LOAD_API(outputTensor, "TfLiteInterpreterGetOutputTensor");
        LOAD_API(tensorByteSize, "TfLiteTensorByteSize");
        LOAD_API(tensorData, "TfLiteTensorData");
        LOAD_API(tensorCopyFromBuffer, "TfLiteTensorCopyFromBuffer");
        LOAD_API(tensorCopyToBuffer, "TfLiteTensorCopyToBuffer");
        LOAD_API(opaqueInputCount, "TfLiteOpaqueNodeNumberOfInputs");
        LOAD_API(opaqueOutputCount, "TfLiteOpaqueNodeNumberOfOutputs");
        LOAD_API(opaqueInput, "TfLiteOpaqueNodeGetInput");
        LOAD_API(opaqueOutput, "TfLiteOpaqueNodeGetOutput");
        LOAD_API(opaqueByteSize, "TfLiteOpaqueTensorByteSize");
        LOAD_API(opaqueData, "TfLiteOpaqueTensorData");
#undef LOAD_API
        return api;
    } catch (...) {
        dlclose(api.library);
        throw;
    }
}

const TfLiteOpaqueTensor* inputTensor(TfLiteOpaqueContext* context, TfLiteOpaqueNode* node, int index) {
    const TfLiteOpaqueTensor* tensor = gApi->opaqueInput(context, node, index);
    if (tensor == nullptr || gApi->opaqueData(tensor) == nullptr) {
        throw std::runtime_error("rANS decode custom input unavailable");
    }
    return tensor;
}

TfLiteOpaqueTensor* outputTensor(TfLiteOpaqueContext* context, TfLiteOpaqueNode* node, int index) {
    TfLiteOpaqueTensor* tensor = gApi->opaqueOutput(context, node, index);
    if (tensor == nullptr || gApi->opaqueData(tensor) == nullptr) {
        throw std::runtime_error("rANS decode custom output unavailable");
    }
    return tensor;
}

template <typename T>
const T* data(const TfLiteOpaqueTensor* tensor) {
    return static_cast<const T*>(gApi->opaqueData(tensor));
}

CdfData copyCdf(
    const TfLiteOpaqueTensor* cdfTensor,
    const TfLiteOpaqueTensor* lengthsTensor,
    const TfLiteOpaqueTensor* offsetsTensor) {
    const size_t rows = gApi->opaqueByteSize(lengthsTensor) / sizeof(int32_t);
    const size_t offsets = gApi->opaqueByteSize(offsetsTensor) / sizeof(int32_t);
    const size_t values = gApi->opaqueByteSize(cdfTensor) / sizeof(int32_t);
    if (rows == 0 || rows != offsets || values % rows != 0) {
        throw std::runtime_error("invalid embedded rANS decode CDF");
    }
    const size_t stride = values / rows;
    const int32_t* cdf = data<int32_t>(cdfTensor);
    const int32_t* lengths = data<int32_t>(lengthsTensor);
    const int32_t* cdfOffsets = data<int32_t>(offsetsTensor);
    auto cdfs = std::make_shared<std::vector<std::vector<int32_t>>>(rows);
    for (size_t row = 0; row < rows; ++row) {
        cdfs->at(row).assign(cdf + row * stride, cdf + (row + 1) * stride);
    }
    return {
        cdfs,
        std::make_shared<std::vector<int32_t>>(lengths, lengths + rows),
        std::make_shared<std::vector<int32_t>>(cdfOffsets, cdfOffsets + rows),
    };
}

void writeNhwc(const std::vector<int8_t>& source, int height, int width, int channels, float* output) {
    if (source.size() != static_cast<size_t>(height * width * channels)) {
        throw std::runtime_error("decoded symbol count mismatch");
    }
    size_t sourceIndex = 0;
    for (int channel = 0; channel < channels; ++channel) {
        for (int row = 0; row < height; ++row) {
            for (int column = 0; column < width; ++column) {
                output[(row * width + column) * channels + channel] = static_cast<float>(source[sourceIndex++]);
            }
        }
    }
}

std::shared_ptr<std::vector<uint8_t>> indexesFromNhwc(const float* scales) {
    constexpr int height = 16;
    constexpr int width = 32;
    constexpr int channels = 64;
    constexpr float scaleMin = 0.11f;
    constexpr float scaleMax = 16.0f;
    constexpr float logScaleMin = -2.2072749f;
    constexpr float logStepRecip = 25.502707f;
    auto indexes = std::make_shared<std::vector<uint8_t>>(height * width * channels);
    size_t destination = 0;
    for (int channel = 0; channel < channels; ++channel) {
        for (int row = 0; row < height; ++row) {
            for (int column = 0; column < width; ++column) {
                const size_t source = static_cast<size_t>((row * width + column) * channels + channel);
                const float scale = std::max(scaleMin, std::min(scaleMax, scales[source]));
                int index = static_cast<int>((std::log(scale) - logScaleMin) * logStepRecip);
                indexes->at(destination++) = static_cast<uint8_t>(std::max(0, std::min(127, index)));
            }
        }
    }
    return indexes;
}

TfLiteStatus decodeZInvoke(TfLiteOpaqueContext* context, TfLiteOpaqueNode* node) {
    std::lock_guard<std::mutex> lock(gDecodeMutex);
    try {
        if (gApi == nullptr || gApi->opaqueInputCount(node) != 10 || gApi->opaqueOutputCount(node) != 1) {
            throw std::runtime_error("rANS Z custom op input/output count mismatch");
        }
        std::vector<const TfLiteOpaqueTensor*> inputs;
        for (int index = 0; index < 10; ++index) inputs.push_back(inputTensor(context, node, index));
        if (!gDecodeState) {
            const CdfData gaussian = copyCdf(inputs[2], inputs[3], inputs[4]);
            const CdfData z = copyCdf(inputs[5], inputs[6], inputs[7]);
            const int startOffset = data<int32_t>(inputs[8])[0];
            const int perChannelSize = data<int32_t>(inputs[9])[0];
            if (perChannelSize <= 0) throw std::runtime_error("invalid z per-channel size");
            gDecodeState = std::make_unique<DecodeState>(gaussian, z, startOffset, perChannelSize);
        }
        const int32_t payloadSize = data<int32_t>(inputs[1])[0];
        const size_t payloadCapacity = gApi->opaqueByteSize(inputs[0]);
        if (payloadSize <= 0 || static_cast<size_t>(payloadSize) > payloadCapacity) {
            throw std::runtime_error("invalid rANS payload size");
        }
        const uint8_t* payload = data<uint8_t>(inputs[0]);
        gDecodeState->decoder.set_stream(
            std::make_shared<std::vector<uint8_t>>(payload, payload + payloadSize));
        gDecodeState->decoder.decode_z(1 * 128 * 4 * 8, gDecodeState->zGroup,
                                      gDecodeState->zStartOffset, gDecodeState->zPerChannelSize);
        auto symbols = gDecodeState->decoder.get_decoded_tensor();
        TfLiteOpaqueTensor* output = outputTensor(context, node, 0);
        if (gApi->opaqueByteSize(output) != symbols->size() * sizeof(float)) {
            throw std::runtime_error("rANS Z output byte size mismatch");
        }
        writeNhwc(*symbols, 4, 8, 128, static_cast<float*>(gApi->opaqueData(output)));
        return kTfLiteOk;
    } catch (const std::exception& error) {
        __android_log_print(ANDROID_LOG_ERROR, kLogTag, "i_entropy_rans_decode_z_failed %s", error.what());
        return 1;
    }
}

TfLiteStatus decodeYInvoke(TfLiteOpaqueContext* context, TfLiteOpaqueNode* node) {
    std::lock_guard<std::mutex> lock(gDecodeMutex);
    try {
        if (gApi == nullptr || !gDecodeState || gApi->opaqueInputCount(node) != 4 ||
            gApi->opaqueOutputCount(node) != 1) {
            throw std::runtime_error("rANS Y custom op state or input/output count mismatch");
        }
        const TfLiteOpaqueTensor* scales = inputTensor(context, node, 0);
        if (gApi->opaqueByteSize(scales) != static_cast<size_t>(1 * 16 * 32 * 64 * sizeof(float))) {
            throw std::runtime_error("rANS Y scales byte size mismatch");
        }
        gDecodeState->decoder.decode_y(indexesFromNhwc(data<float>(scales)), gDecodeState->gaussianGroup);
        auto symbols = gDecodeState->decoder.get_decoded_tensor();
        TfLiteOpaqueTensor* output = outputTensor(context, node, 0);
        if (gApi->opaqueByteSize(output) != symbols->size() * sizeof(float)) {
            throw std::runtime_error("rANS Y output byte size mismatch");
        }
        writeNhwc(*symbols, 16, 32, 64, static_cast<float*>(gApi->opaqueData(output)));
        return kTfLiteOk;
    } catch (const std::exception& error) {
        __android_log_print(ANDROID_LOG_ERROR, kLogTag, "i_entropy_rans_decode_y_failed %s", error.what());
        return 1;
    }
}

void destroyRuntime(Runtime* runtime) {
    if (runtime == nullptr) return;
    if (runtime->interpreter != nullptr) runtime->api.interpreterDelete(runtime->interpreter);
    if (runtime->decodeZRegistration != nullptr) runtime->api.registrationDelete(runtime->decodeZRegistration);
    if (runtime->decodeYRegistration != nullptr) runtime->api.registrationDelete(runtime->decodeYRegistration);
    if (runtime->options != nullptr) runtime->api.optionsDelete(runtime->options);
    if (runtime->model != nullptr) runtime->api.modelDelete(runtime->model);
    {
        std::lock_guard<std::mutex> lock(gDecodeMutex);
        gDecodeState.reset();
        gApi = nullptr;
    }
    if (runtime->api.library != nullptr) dlclose(runtime->api.library);
    delete runtime;
}

Runtime* createRuntime(
    const std::string& path,
    void* delegate,
    const char* decodeZOp,
    const char* decodeYOp,
    int expectedInputs,
    int expectedOutputs) {
    if (delegate == nullptr) throw std::runtime_error("Neuron delegate handle is null");
    auto runtime = std::make_unique<Runtime>();
    runtime->api = loadApi();
    gApi = &runtime->api;
    try {
        runtime->model = runtime->api.modelCreateFromFile(path.c_str());
        if (runtime->model == nullptr) throw std::runtime_error("TfLiteModelCreateFromFile failed");
        runtime->options = runtime->api.optionsCreate();
        runtime->decodeZRegistration = runtime->api.registrationCreate(kTfLiteBuiltinCustom, decodeZOp, 1);
        runtime->decodeYRegistration = runtime->api.registrationCreate(kTfLiteBuiltinCustom, decodeYOp, 1);
        if (runtime->options == nullptr || runtime->decodeZRegistration == nullptr || runtime->decodeYRegistration == nullptr) {
            throw std::runtime_error("rANS decode registration create failed");
        }
        runtime->api.registrationSetInvoke(runtime->decodeZRegistration, decodeZInvoke);
        runtime->api.registrationSetInvoke(runtime->decodeYRegistration, decodeYInvoke);
        runtime->api.optionsAddRegistration(runtime->options, runtime->decodeZRegistration);
        runtime->api.optionsAddRegistration(runtime->options, runtime->decodeYRegistration);
        runtime->api.optionsAddDelegate(runtime->options, delegate);
        runtime->interpreter = runtime->api.interpreterCreate(runtime->model, runtime->options);
        if (runtime->interpreter == nullptr) throw std::runtime_error("TfLiteInterpreterCreate failed");
        if (runtime->api.allocateTensors(runtime->interpreter) != kTfLiteOk) {
            throw std::runtime_error("TfLiteInterpreterAllocateTensors failed");
        }
        if (runtime->api.inputCount(runtime->interpreter) != expectedInputs ||
            runtime->api.outputCount(runtime->interpreter) != expectedOutputs) {
            throw std::runtime_error("merged entropy decoder input/output count mismatch");
        }
        return runtime.release();
    } catch (...) {
        destroyRuntime(runtime.release());
        throw;
    }
}

void copyJavaInput(JNIEnv* env, Runtime* runtime, int index, jbyteArray input) {
    if (input == nullptr) throw std::runtime_error("merged entropy decoder input is null");
    TfLiteTensor* tensor = runtime->api.inputTensor(runtime->interpreter, index);
    const size_t bytes = runtime->api.tensorByteSize(tensor);
    if (env->GetArrayLength(input) != static_cast<jsize>(bytes)) {
        throw std::runtime_error("merged entropy decoder input byte size mismatch");
    }
    std::vector<jbyte> data(bytes);
    env->GetByteArrayRegion(input, 0, static_cast<jsize>(bytes), data.data());
    if (runtime->api.tensorCopyFromBuffer(tensor, data.data(), bytes) != kTfLiteOk) {
        throw std::runtime_error("merged entropy decoder input copy failed");
    }
}

jobjectArray runRuntime(
    JNIEnv* env,
    Runtime* runtime,
    jbyteArray payload,
    jbyteArray contextInput,
    jint outputMode) {
    if (runtime == nullptr) throw std::runtime_error("merged entropy decoder is closed");
    TfLiteTensor* payloadTensor = runtime->api.inputTensor(runtime->interpreter, 0);
    TfLiteTensor* sizeTensor = runtime->api.inputTensor(runtime->interpreter, 1);
    const size_t capacity = runtime->api.tensorByteSize(payloadTensor);
    const jsize payloadSize = env->GetArrayLength(payload);
    if (payloadSize <= 0 || static_cast<size_t>(payloadSize) > capacity) {
        throw std::runtime_error("merged entropy decoder payload exceeds input capacity");
    }
    std::vector<uint8_t> padded(capacity, 0);
    env->GetByteArrayRegion(payload, 0, payloadSize, reinterpret_cast<jbyte*>(padded.data()));
    const int32_t size = payloadSize;
    if (runtime->api.tensorCopyFromBuffer(payloadTensor, padded.data(), padded.size()) != kTfLiteOk ||
        runtime->api.tensorCopyFromBuffer(sizeTensor, &size, sizeof(size)) != kTfLiteOk) {
        throw std::runtime_error("merged entropy decoder payload copy failed");
    }
    if (contextInput != nullptr) copyJavaInput(env, runtime, 2, contextInput);
    if (runtime->api.invoke(runtime->interpreter) != kTfLiteOk) {
        throw std::runtime_error("merged entropy decoder invoke failed");
    }
    jclass byteArrayClass = env->FindClass("[B");
    if (outputMode == 0) {
        jobjectArray empty = env->NewObjectArray(0, byteArrayClass, nullptr);
        env->DeleteLocalRef(byteArrayClass);
        return empty;
    }
    const int outputCount = runtime->api.outputCount(runtime->interpreter);
    const int first = outputMode == 2 ? outputCount - 1 : 0;
    const int count = outputMode == 2 ? 1 : outputCount;
    if (outputMode != 1 && outputMode != 2) throw std::runtime_error("unsupported output mode");
    jobjectArray result = env->NewObjectArray(count, byteArrayClass, nullptr);
    for (int resultIndex = 0; resultIndex < count; ++resultIndex) {
        const TfLiteTensor* tensor = runtime->api.outputTensor(runtime->interpreter, first + resultIndex);
        const size_t bytes = runtime->api.tensorByteSize(tensor);
        std::vector<uint8_t> output(bytes);
        if (runtime->api.tensorCopyToBuffer(tensor, output.data(), bytes) != kTfLiteOk) {
            throw std::runtime_error("merged entropy decoder output copy failed");
        }
        jbyteArray array = env->NewByteArray(static_cast<jsize>(bytes));
        env->SetByteArrayRegion(array, 0, static_cast<jsize>(bytes), reinterpret_cast<const jbyte*>(output.data()));
        env->SetObjectArrayElement(result, resultIndex, array);
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
Java_com_gvcrt_clean_IEntropyRansDecodeMergedRuntime_nativeCreate(
    JNIEnv* env, jclass, jstring modelPath, jlong delegateHandle) {
    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    if (path == nullptr) return 0;
    const std::string model(path);
    env->ReleaseStringUTFChars(modelPath, path);
    try {
        return reinterpret_cast<jlong>(createRuntime(
            model,
            reinterpret_cast<void*>(static_cast<uintptr_t>(delegateHandle)),
            kIDecodeZOp,
            kIDecodeYOp,
            2,
            10));
    } catch (const std::exception& error) {
        throwJava(env, error.what());
        return 0;
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_gvcrt_clean_IEntropyRansDecodeMergedRuntime_nativeRun(
    JNIEnv* env, jclass, jlong handle, jbyteArray payload, jint outputMode) {
    try {
        return runRuntime(env, reinterpret_cast<Runtime*>(handle), payload, nullptr, outputMode);
    } catch (const std::exception& error) {
        throwJava(env, error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_gvcrt_clean_IEntropyRansDecodeMergedRuntime_nativeClose(
    JNIEnv*, jclass, jlong handle) {
    destroyRuntime(reinterpret_cast<Runtime*>(handle));
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_gvcrt_clean_PEntropyRansDecodeMergedRuntime_nativeCreate(
    JNIEnv* env, jclass, jstring modelPath, jlong delegateHandle) {
    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    if (path == nullptr) return 0;
    const std::string model(path);
    env->ReleaseStringUTFChars(modelPath, path);
    try {
        return reinterpret_cast<jlong>(createRuntime(
            model,
            reinterpret_cast<void*>(static_cast<uintptr_t>(delegateHandle)),
            kPDecodeZOp,
            kPDecodeYOp,
            3,
            6));
    } catch (const std::exception& error) {
        throwJava(env, error.what());
        return 0;
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_gvcrt_clean_PEntropyRansDecodeMergedRuntime_nativeRun(
    JNIEnv* env,
    jclass,
    jlong handle,
    jbyteArray payload,
    jbyteArray ctxT,
    jint outputMode) {
    try {
        return runRuntime(env, reinterpret_cast<Runtime*>(handle), payload, ctxT, outputMode);
    } catch (const std::exception& error) {
        throwJava(env, error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_gvcrt_clean_PEntropyRansDecodeMergedRuntime_nativeClose(
    JNIEnv*, jclass, jlong handle) {
    destroyRuntime(reinterpret_cast<Runtime*>(handle));
}
