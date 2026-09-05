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
constexpr char kEncodeOp[] = "GVC_RT_SMALL_RANS_ENCODE";
constexpr char kDecodeZOp[] = "GVC_RT_SMALL_RANS_DECODE_Z";
constexpr char kDecodeYOp[] = "GVC_RT_SMALL_RANS_DECODE_Y";
constexpr int kTfLiteOk = 0;
constexpr int kTfLiteBuiltinCustom = 32;
constexpr int kEncodeKind = 0;
constexpr int kDecodeKind = 1;

constexpr int kZHeight = 4;
constexpr int kZWidth = 8;
constexpr int kZChannels = 48;
constexpr int kYHeight = 16;
constexpr int kYWidth = 32;
constexpr int kYChannels = 24;
constexpr int kZStartOffset = 432;
constexpr int kZPerChannelSize = 32;
constexpr size_t kPayloadCapacity = 65536;

struct TfLiteModel;
struct TfLiteInterpreterOptions;
struct TfLiteInterpreter;
struct TfLiteTensor;
struct TfLiteOperator;
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
    void (*optionsAddOperator)(TfLiteInterpreterOptions*, TfLiteOperator*) = nullptr;
    void (*optionsAddRegistration)(TfLiteInterpreterOptions*, TfLiteRegistrationExternal*) = nullptr;
    TfLiteOperator* (*operatorCreate)(int, const char*, int, void*) = nullptr;
    void (*operatorDelete)(TfLiteOperator*) = nullptr;
    TfLiteStatus (*operatorSetInvokeWithData)(
        TfLiteOperator*,
        TfLiteStatus (*)(void*, TfLiteOpaqueContext*, TfLiteOpaqueNode*)) = nullptr;
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
    int (*opaqueInputCount)(const TfLiteOpaqueNode*) = nullptr;
    int (*opaqueOutputCount)(const TfLiteOpaqueNode*) = nullptr;
    const TfLiteOpaqueTensor* (*opaqueInput)(
        const TfLiteOpaqueContext*, const TfLiteOpaqueNode*, int) = nullptr;
    TfLiteOpaqueTensor* (*opaqueOutput)(TfLiteOpaqueContext*, const TfLiteOpaqueNode*, int) = nullptr;
    size_t (*opaqueByteSize)(const TfLiteOpaqueTensor*) = nullptr;
    void* (*opaqueData)(const TfLiteOpaqueTensor*) = nullptr;
};

struct CdfData {
    std::shared_ptr<std::vector<std::vector<int32_t>>> cdfs;
    std::shared_ptr<std::vector<int32_t>> lengths;
    std::shared_ptr<std::vector<int32_t>> offsets;
};

struct EncodeState {
    RansEncoderLib encoder;
    int gaussianGroup;
    int zGroup;

    EncodeState(const CdfData& gaussian, const CdfData& z)
        : gaussianGroup(encoder.add_cdf(gaussian.cdfs, gaussian.lengths, gaussian.offsets)),
          zGroup(encoder.add_cdf(z.cdfs, z.lengths, z.offsets)) {}
};

struct DecodeState {
    RansDecoderLib decoder;
    int gaussianGroup;
    int zGroup;

    DecodeState(const CdfData& gaussian, const CdfData& z)
        : gaussianGroup(decoder.add_cdf(gaussian.cdfs, gaussian.lengths, gaussian.offsets)),
          zGroup(decoder.add_cdf(z.cdfs, z.lengths, z.offsets)) {}
};

struct Runtime {
    int kind = -1;
    bool isMtk = false;
    TfliteApi api;
    TfLiteModel* model = nullptr;
    TfLiteInterpreterOptions* options = nullptr;
    TfLiteOperator* encodeOperator = nullptr;
    TfLiteOperator* decodeZOperator = nullptr;
    TfLiteOperator* decodeYOperator = nullptr;
    TfLiteRegistrationExternal* encodeRegistration = nullptr;
    TfLiteRegistrationExternal* decodeZRegistration = nullptr;
    TfLiteRegistrationExternal* decodeYRegistration = nullptr;
    TfLiteInterpreter* interpreter = nullptr;
    std::unique_ptr<EncodeState> encodeState;
    std::unique_ptr<DecodeState> decodeState;
    int decodeYInvocations = 0;
};

Runtime* gActiveMtkRuntime = nullptr;
std::mutex gMtkInvokeMutex;

template <typename T>
T loadSymbol(void* library, const char* name) {
    dlerror();
    void* symbol = dlsym(library, name);
    const char* error = dlerror();
    if (error != nullptr || symbol == nullptr) {
        throw std::runtime_error(std::string("missing TFLite symbol ") + name);
    }
    return reinterpret_cast<T>(symbol);
}

TfliteApi loadApi(bool isMtk) {
    TfliteApi api;
    const char* libraryName = isMtk ? "libtensorflowlite_jni_mtk.so" : "libtensorflowlite_jni.so";
    api.library = dlopen(libraryName, RTLD_NOW | RTLD_LOCAL);
    if (api.library == nullptr) {
        throw std::runtime_error(std::string("dlopen ") + libraryName + " failed: " + dlerror());
    }
    try {
#define LOAD_API(member, symbol) api.member = loadSymbol<decltype(api.member)>(api.library, symbol)
        LOAD_API(modelCreateFromFile, "TfLiteModelCreateFromFile");
        LOAD_API(modelDelete, "TfLiteModelDelete");
        LOAD_API(optionsCreate, "TfLiteInterpreterOptionsCreate");
        LOAD_API(optionsDelete, "TfLiteInterpreterOptionsDelete");
        LOAD_API(optionsAddDelegate, "TfLiteInterpreterOptionsAddDelegate");
        if (isMtk) {
            LOAD_API(optionsAddRegistration, "TfLiteInterpreterOptionsAddRegistrationExternal");
            LOAD_API(registrationCreate, "TfLiteRegistrationExternalCreate");
            LOAD_API(registrationDelete, "TfLiteRegistrationExternalDelete");
            LOAD_API(registrationSetInvoke, "TfLiteRegistrationExternalSetInvoke");
        } else {
            LOAD_API(optionsAddOperator, "TfLiteInterpreterOptionsAddOperator");
            LOAD_API(operatorCreate, "TfLiteOperatorCreate");
            LOAD_API(operatorDelete, "TfLiteOperatorDelete");
            LOAD_API(operatorSetInvokeWithData, "TfLiteOperatorSetInvokeWithData");
        }
        LOAD_API(interpreterCreate, "TfLiteInterpreterCreate");
        LOAD_API(interpreterDelete, "TfLiteInterpreterDelete");
        LOAD_API(allocateTensors, "TfLiteInterpreterAllocateTensors");
        LOAD_API(invoke, "TfLiteInterpreterInvoke");
        LOAD_API(inputCount, "TfLiteInterpreterGetInputTensorCount");
        LOAD_API(outputCount, "TfLiteInterpreterGetOutputTensorCount");
        LOAD_API(inputTensor, "TfLiteInterpreterGetInputTensor");
        LOAD_API(outputTensor, "TfLiteInterpreterGetOutputTensor");
        LOAD_API(tensorByteSize, "TfLiteTensorByteSize");
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

const TfLiteOpaqueTensor* inputTensor(
    Runtime* runtime, TfLiteOpaqueContext* context, TfLiteOpaqueNode* node, int index) {
    const TfLiteOpaqueTensor* tensor = runtime->api.opaqueInput(context, node, index);
    if (tensor == nullptr || runtime->api.opaqueData(tensor) == nullptr) {
        throw std::runtime_error("Small rANS custom input unavailable");
    }
    return tensor;
}

TfLiteOpaqueTensor* outputTensor(
    Runtime* runtime, TfLiteOpaqueContext* context, TfLiteOpaqueNode* node, int index) {
    TfLiteOpaqueTensor* tensor = runtime->api.opaqueOutput(context, node, index);
    if (tensor == nullptr || runtime->api.opaqueData(tensor) == nullptr) {
        throw std::runtime_error("Small rANS custom output unavailable");
    }
    return tensor;
}

template <typename T>
const T* tensorData(Runtime* runtime, const TfLiteOpaqueTensor* tensor) {
    return static_cast<const T*>(runtime->api.opaqueData(tensor));
}

CdfData copyCdf(
    Runtime* runtime,
    const TfLiteOpaqueTensor* cdfTensor,
    const TfLiteOpaqueTensor* lengthsTensor,
    const TfLiteOpaqueTensor* offsetsTensor) {
    const size_t rows = runtime->api.opaqueByteSize(lengthsTensor) / sizeof(int32_t);
    const size_t offsetRows = runtime->api.opaqueByteSize(offsetsTensor) / sizeof(int32_t);
    const size_t values = runtime->api.opaqueByteSize(cdfTensor) / sizeof(int32_t);
    if (rows == 0 || rows != offsetRows || values % rows != 0) {
        throw std::runtime_error("invalid embedded Small rANS CDF tensor sizes");
    }
    const size_t stride = values / rows;
    const int32_t* cdf = tensorData<int32_t>(runtime, cdfTensor);
    const int32_t* lengths = tensorData<int32_t>(runtime, lengthsTensor);
    const int32_t* offsets = tensorData<int32_t>(runtime, offsetsTensor);
    auto cdfs = std::make_shared<std::vector<std::vector<int32_t>>>(rows);
    for (size_t row = 0; row < rows; ++row) {
        cdfs->at(row).assign(cdf + row * stride, cdf + (row + 1) * stride);
    }
    return {
        cdfs,
        std::make_shared<std::vector<int32_t>>(lengths, lengths + rows),
        std::make_shared<std::vector<int32_t>>(offsets, offsets + rows),
    };
}

void requireBytes(Runtime* runtime, const TfLiteOpaqueTensor* tensor, size_t expected, const char* label) {
    if (runtime->api.opaqueByteSize(tensor) != expected) {
        throw std::runtime_error(std::string(label) + " byte size mismatch");
    }
}

std::shared_ptr<std::vector<int8_t>> copyZFromNhwc(const float* source) {
    auto output = std::make_shared<std::vector<int8_t>>(kZHeight * kZWidth * kZChannels);
    size_t destination = 0;
    for (int channel = 0; channel < kZChannels; ++channel) {
        for (int row = 0; row < kZHeight; ++row) {
            for (int column = 0; column < kZWidth; ++column) {
                const float value = source[(row * kZWidth + column) * kZChannels + channel];
                const int symbol = static_cast<int>(value);
                if (value != static_cast<float>(symbol) || symbol < -128 || symbol > 127) {
                    throw std::runtime_error("non-integral or out-of-range Small z symbol");
                }
                output->at(destination++) = static_cast<int8_t>(symbol);
            }
        }
    }
    return output;
}

int scaleIndex(float value) {
    constexpr float kScaleMin = 0.11f;
    constexpr float kScaleMax = 16.0f;
    constexpr float kLogScaleMin = -2.2072749f;
    constexpr float kLogStepRecip = 25.502707f;
    const float scale = std::max(kScaleMin, std::min(kScaleMax, value));
    const int index = static_cast<int>((std::log(scale) - kLogScaleMin) * kLogStepRecip);
    return std::max(0, std::min(127, index));
}

template <typename T>
uint64_t valueHash(const std::vector<T>& values) {
    uint64_t hash = 1469598103934665603ULL;
    for (const T value : values) {
        const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
        for (size_t index = 0; index < sizeof(T); ++index) {
            hash ^= bytes[index];
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

uint64_t packedSymbolHash(const std::vector<int16_t>& values) {
    std::vector<int8_t> symbols(values.size());
    for (size_t index = 0; index < values.size(); ++index) {
        symbols[index] = static_cast<int8_t>(values[index] >> 8);
    }
    return valueHash(symbols);
}

uint64_t packedIndexHash(const std::vector<int16_t>& values) {
    std::vector<uint8_t> indexes(values.size());
    for (size_t index = 0; index < values.size(); ++index) {
        indexes[index] = static_cast<uint8_t>(values[index] & 0xff);
    }
    return valueHash(indexes);
}

std::shared_ptr<std::vector<int16_t>> packYFromNhwc(const float* symbols, const float* scales) {
    auto output = std::make_shared<std::vector<int16_t>>(kYHeight * kYWidth * kYChannels);
    size_t destination = 0;
    for (int channel = 0; channel < kYChannels; ++channel) {
        for (int row = 0; row < kYHeight; ++row) {
            for (int column = 0; column < kYWidth; ++column) {
                const size_t source = static_cast<size_t>((row * kYWidth + column) * kYChannels + channel);
                const float value = symbols[source];
                const int symbol = static_cast<int>(value);
                if (value != static_cast<float>(symbol) || symbol < -128 || symbol > 127) {
                    throw std::runtime_error("non-integral or out-of-range Small y symbol");
                }
                output->at(destination++) = static_cast<int16_t>(symbol * 256 + scaleIndex(scales[source]));
            }
        }
    }
    return output;
}

std::shared_ptr<std::vector<uint8_t>> indexesFromNhwc(const float* scales) {
    auto indexes = std::make_shared<std::vector<uint8_t>>(kYHeight * kYWidth * kYChannels);
    size_t destination = 0;
    for (int channel = 0; channel < kYChannels; ++channel) {
        for (int row = 0; row < kYHeight; ++row) {
            for (int column = 0; column < kYWidth; ++column) {
                const size_t source = static_cast<size_t>((row * kYWidth + column) * kYChannels + channel);
                indexes->at(destination++) = static_cast<uint8_t>(scaleIndex(scales[source]));
            }
        }
    }
    return indexes;
}

void writeNhwc(const std::vector<int8_t>& source, int height, int width, int channels, float* output) {
    if (source.size() != static_cast<size_t>(height * width * channels)) {
        throw std::runtime_error("Small decoded symbol count mismatch");
    }
    size_t sourceIndex = 0;
    for (int channel = 0; channel < channels; ++channel) {
        for (int row = 0; row < height; ++row) {
            for (int column = 0; column < width; ++column) {
                output[(row * width + column) * channels + channel] =
                    static_cast<float>(source[sourceIndex++]);
            }
        }
    }
}

TfLiteStatus encodeInvoke(void* userData, TfLiteOpaqueContext* context, TfLiteOpaqueNode* node) {
    auto* runtime = static_cast<Runtime*>(userData);
    try {
        if (runtime == nullptr || runtime->kind != kEncodeKind ||
            runtime->api.opaqueInputCount(node) != 13 || runtime->api.opaqueOutputCount(node) != 2) {
            throw std::runtime_error("Small encode custom op input/output count mismatch");
        }
        std::vector<const TfLiteOpaqueTensor*> inputs;
        inputs.reserve(13);
        for (int index = 0; index < 13; ++index) inputs.push_back(inputTensor(runtime, context, node, index));
        requireBytes(runtime, inputs[0], kZHeight * kZWidth * kZChannels * sizeof(float), "Small z_hat");
        for (int index = 1; index <= 4; ++index) {
            requireBytes(runtime, inputs[index], kYHeight * kYWidth * kYChannels * sizeof(float), "Small y/scales");
        }
        const int startOffset = tensorData<int32_t>(runtime, inputs[11])[0];
        const int perChannelSize = tensorData<int32_t>(runtime, inputs[12])[0];
        if (startOffset != kZStartOffset || perChannelSize != kZPerChannelSize) {
            throw std::runtime_error("Small embedded z parameters mismatch");
        }
        if (!runtime->encodeState) {
            runtime->encodeState = std::make_unique<EncodeState>(
                copyCdf(runtime, inputs[5], inputs[6], inputs[7]),
                copyCdf(runtime, inputs[8], inputs[9], inputs[10]));
        }
        const auto zSymbols = copyZFromNhwc(tensorData<float>(runtime, inputs[0]));
        const auto y0Packed = packYFromNhwc(
            tensorData<float>(runtime, inputs[1]), tensorData<float>(runtime, inputs[3]));
        const auto y1Packed = packYFromNhwc(
            tensorData<float>(runtime, inputs[2]), tensorData<float>(runtime, inputs[4]));
        runtime->encodeState->encoder.encode_z(
            zSymbols,
            runtime->encodeState->zGroup,
            kZStartOffset,
            kZPerChannelSize);
        runtime->encodeState->encoder.encode_y(y0Packed, runtime->encodeState->gaussianGroup);
        runtime->encodeState->encoder.encode_y(y1Packed, runtime->encodeState->gaussianGroup);
        runtime->encodeState->encoder.flush();
        const auto payload = runtime->encodeState->encoder.get_encoded_stream();
        TfLiteOpaqueTensor* payloadOutput = outputTensor(runtime, context, node, 0);
        TfLiteOpaqueTensor* sizeOutput = outputTensor(runtime, context, node, 1);
        requireBytes(runtime, payloadOutput, kPayloadCapacity, "Small payload output");
        requireBytes(runtime, sizeOutput, sizeof(int32_t), "Small payload size output");
        if (payload->empty() || payload->size() > kPayloadCapacity) {
            throw std::runtime_error("Small payload size is invalid");
        }
        const size_t payloadSize = payload->size();
        std::memset(runtime->api.opaqueData(payloadOutput), 0, kPayloadCapacity);
        std::memcpy(runtime->api.opaqueData(payloadOutput), payload->data(), payloadSize);
        *static_cast<int32_t*>(runtime->api.opaqueData(sizeOutput)) = static_cast<int32_t>(payloadSize);
        runtime->encodeState->encoder.reset();
        __android_log_print(
            ANDROID_LOG_INFO, kLogTag,
            "small_%s_rans_encode_invoke_ok z_symbols=%d y_stages=2 y_symbols_per_stage=%d payload_bytes=%zu z_hash=%016llx y0_hash=%016llx y0_index_hash=%016llx y1_hash=%016llx y1_index_hash=%016llx",
            runtime->isMtk ? "mtk" : "gpu",
            kZHeight * kZWidth * kZChannels, kYHeight * kYWidth * kYChannels, payloadSize,
            static_cast<unsigned long long>(valueHash(*zSymbols)),
            static_cast<unsigned long long>(packedSymbolHash(*y0Packed)),
            static_cast<unsigned long long>(packedIndexHash(*y0Packed)),
            static_cast<unsigned long long>(packedSymbolHash(*y1Packed)),
            static_cast<unsigned long long>(packedIndexHash(*y1Packed)));
        return kTfLiteOk;
    } catch (const std::exception& error) {
        if (runtime != nullptr && runtime->encodeState) runtime->encodeState->encoder.reset();
        __android_log_print(
            ANDROID_LOG_ERROR, kLogTag, "small_%s_rans_encode_invoke_failed error=%s",
            runtime != nullptr && runtime->isMtk ? "mtk" : "gpu", error.what());
        return 1;
    }
}

TfLiteStatus decodeZInvoke(void* userData, TfLiteOpaqueContext* context, TfLiteOpaqueNode* node) {
    auto* runtime = static_cast<Runtime*>(userData);
    try {
        if (runtime == nullptr || runtime->kind != kDecodeKind ||
            runtime->api.opaqueInputCount(node) != 10 || runtime->api.opaqueOutputCount(node) != 1) {
            throw std::runtime_error("Small decode Z custom op input/output count mismatch");
        }
        std::vector<const TfLiteOpaqueTensor*> inputs;
        inputs.reserve(10);
        for (int index = 0; index < 10; ++index) inputs.push_back(inputTensor(runtime, context, node, index));
        requireBytes(runtime, inputs[0], kPayloadCapacity, "Small payload input");
        requireBytes(runtime, inputs[1], sizeof(int32_t), "Small payload size input");
        const int startOffset = tensorData<int32_t>(runtime, inputs[8])[0];
        const int perChannelSize = tensorData<int32_t>(runtime, inputs[9])[0];
        if (startOffset != kZStartOffset || perChannelSize != kZPerChannelSize) {
            throw std::runtime_error("Small embedded decode z parameters mismatch");
        }
        if (!runtime->decodeState) {
            runtime->decodeState = std::make_unique<DecodeState>(
                copyCdf(runtime, inputs[2], inputs[3], inputs[4]),
                copyCdf(runtime, inputs[5], inputs[6], inputs[7]));
        }
        const int32_t payloadSize = tensorData<int32_t>(runtime, inputs[1])[0];
        if (payloadSize <= 0 || static_cast<size_t>(payloadSize) > kPayloadCapacity) {
            throw std::runtime_error("Small decode payload size is invalid");
        }
        const uint8_t* payload = tensorData<uint8_t>(runtime, inputs[0]);
        runtime->decodeState->decoder.set_stream(
            std::make_shared<std::vector<uint8_t>>(payload, payload + payloadSize));
        runtime->decodeState->decoder.decode_z(
            kZHeight * kZWidth * kZChannels,
            runtime->decodeState->zGroup,
            kZStartOffset,
            kZPerChannelSize);
        const auto symbols = runtime->decodeState->decoder.get_decoded_tensor();
        TfLiteOpaqueTensor* output = outputTensor(runtime, context, node, 0);
        requireBytes(runtime, output, symbols->size() * sizeof(float), "Small decoded z output");
        writeNhwc(*symbols, kZHeight, kZWidth, kZChannels,
                  static_cast<float*>(runtime->api.opaqueData(output)));
        runtime->decodeYInvocations = 0;
        __android_log_print(
            ANDROID_LOG_INFO, kLogTag,
            "small_%s_rans_decode_z_invoke_ok symbols=%zu payload_bytes=%d z_hash=%016llx",
            runtime->isMtk ? "mtk" : "gpu",
            symbols->size(), payloadSize,
            static_cast<unsigned long long>(valueHash(*symbols)));
        return kTfLiteOk;
    } catch (const std::exception& error) {
        __android_log_print(
            ANDROID_LOG_ERROR, kLogTag, "small_%s_rans_decode_z_invoke_failed error=%s",
            runtime != nullptr && runtime->isMtk ? "mtk" : "gpu", error.what());
        return 1;
    }
}

TfLiteStatus decodeYInvoke(void* userData, TfLiteOpaqueContext* context, TfLiteOpaqueNode* node) {
    auto* runtime = static_cast<Runtime*>(userData);
    try {
        if (runtime == nullptr || runtime->kind != kDecodeKind || !runtime->decodeState ||
            runtime->api.opaqueInputCount(node) != 4 || runtime->api.opaqueOutputCount(node) != 1) {
            throw std::runtime_error("Small decode Y custom op state or input/output count mismatch");
        }
        const TfLiteOpaqueTensor* scales = inputTensor(runtime, context, node, 0);
        requireBytes(runtime, scales, kYHeight * kYWidth * kYChannels * sizeof(float), "Small decode scales");
        const auto indexes = indexesFromNhwc(tensorData<float>(runtime, scales));
        runtime->decodeState->decoder.decode_y(indexes, runtime->decodeState->gaussianGroup);
        const auto symbols = runtime->decodeState->decoder.get_decoded_tensor();
        TfLiteOpaqueTensor* output = outputTensor(runtime, context, node, 0);
        requireBytes(runtime, output, symbols->size() * sizeof(float), "Small decoded y output");
        writeNhwc(*symbols, kYHeight, kYWidth, kYChannels,
                  static_cast<float*>(runtime->api.opaqueData(output)));
        ++runtime->decodeYInvocations;
        __android_log_print(
            ANDROID_LOG_INFO, kLogTag,
            "small_%s_rans_decode_y_invoke_ok stage=%d symbols=%zu symbol_hash=%016llx index_hash=%016llx",
            runtime->isMtk ? "mtk" : "gpu",
            runtime->decodeYInvocations - 1, symbols->size(),
            static_cast<unsigned long long>(valueHash(*symbols)),
            static_cast<unsigned long long>(valueHash(*indexes)));
        return kTfLiteOk;
    } catch (const std::exception& error) {
        __android_log_print(
            ANDROID_LOG_ERROR, kLogTag, "small_%s_rans_decode_y_invoke_failed error=%s",
            runtime != nullptr && runtime->isMtk ? "mtk" : "gpu", error.what());
        return 1;
    }
}

TfLiteStatus encodeInvokeMtk(TfLiteOpaqueContext* context, TfLiteOpaqueNode* node) {
    return encodeInvoke(gActiveMtkRuntime, context, node);
}

TfLiteStatus decodeZInvokeMtk(TfLiteOpaqueContext* context, TfLiteOpaqueNode* node) {
    return decodeZInvoke(gActiveMtkRuntime, context, node);
}

TfLiteStatus decodeYInvokeMtk(TfLiteOpaqueContext* context, TfLiteOpaqueNode* node) {
    return decodeYInvoke(gActiveMtkRuntime, context, node);
}

void destroyRuntime(Runtime* runtime) {
    if (runtime == nullptr) return;
    if (runtime->interpreter != nullptr) runtime->api.interpreterDelete(runtime->interpreter);
    if (runtime->encodeOperator != nullptr) runtime->api.operatorDelete(runtime->encodeOperator);
    if (runtime->decodeZOperator != nullptr) runtime->api.operatorDelete(runtime->decodeZOperator);
    if (runtime->decodeYOperator != nullptr) runtime->api.operatorDelete(runtime->decodeYOperator);
    if (runtime->encodeRegistration != nullptr) runtime->api.registrationDelete(runtime->encodeRegistration);
    if (runtime->decodeZRegistration != nullptr) runtime->api.registrationDelete(runtime->decodeZRegistration);
    if (runtime->decodeYRegistration != nullptr) runtime->api.registrationDelete(runtime->decodeYRegistration);
    if (runtime->options != nullptr) runtime->api.optionsDelete(runtime->options);
    if (runtime->model != nullptr) runtime->api.modelDelete(runtime->model);
    if (runtime->api.library != nullptr) dlclose(runtime->api.library);
    delete runtime;
}

void registerOperator(
    Runtime* runtime,
    TfLiteOperator*& target,
    TfLiteRegistrationExternal*& registrationTarget,
    const char* name,
    TfLiteStatus (*invoke)(void*, TfLiteOpaqueContext*, TfLiteOpaqueNode*),
    TfLiteStatus (*mtkInvoke)(TfLiteOpaqueContext*, TfLiteOpaqueNode*)) {
    if (runtime->isMtk) {
        registrationTarget = runtime->api.registrationCreate(kTfLiteBuiltinCustom, name, 1);
        if (registrationTarget == nullptr) {
            throw std::runtime_error(std::string("Small MTK rANS registration failed: ") + name);
        }
        runtime->api.registrationSetInvoke(registrationTarget, mtkInvoke);
        runtime->api.optionsAddRegistration(runtime->options, registrationTarget);
    } else {
        target = runtime->api.operatorCreate(kTfLiteBuiltinCustom, name, 1, runtime);
        if (target == nullptr || runtime->api.operatorSetInvokeWithData(target, invoke) != kTfLiteOk) {
            throw std::runtime_error(std::string("Small GPU rANS operator registration failed: ") + name);
        }
        runtime->api.optionsAddOperator(runtime->options, target);
    }
}

Runtime* createRuntime(
    const std::string& modelPath,
    int kind,
    void* primaryDelegate,
    void* guardDelegate,
    bool isMtk) {
    if (kind != kEncodeKind && kind != kDecodeKind) throw std::runtime_error("invalid Small rANS runtime kind");
    if (primaryDelegate == nullptr || (!isMtk && guardDelegate == nullptr)) {
        throw std::runtime_error("Small rANS delegate handle is null");
    }
    auto runtime = std::make_unique<Runtime>();
    runtime->kind = kind;
    runtime->isMtk = isMtk;
    runtime->api = loadApi(isMtk);
    try {
        runtime->model = runtime->api.modelCreateFromFile(modelPath.c_str());
        if (runtime->model == nullptr) throw std::runtime_error("TfLiteModelCreateFromFile failed");
        runtime->options = runtime->api.optionsCreate();
        if (runtime->options == nullptr) throw std::runtime_error("TfLiteInterpreterOptionsCreate failed");
        if (kind == kEncodeKind) {
            registerOperator(
                runtime.get(), runtime->encodeOperator, runtime->encodeRegistration,
                kEncodeOp, encodeInvoke, encodeInvokeMtk);
        } else {
            registerOperator(
                runtime.get(), runtime->decodeZOperator, runtime->decodeZRegistration,
                kDecodeZOp, decodeZInvoke, decodeZInvokeMtk);
            registerOperator(
                runtime.get(), runtime->decodeYOperator, runtime->decodeYRegistration,
                kDecodeYOp, decodeYInvoke, decodeYInvokeMtk);
        }
        __android_log_print(
            ANDROID_LOG_INFO, kLogTag,
            "small_custom_op_registration_ok model_kind=%s backend=%s",
            kind == kEncodeKind ? "encode" : "decode", isMtk ? "mtk_npu" : "tflite_gpu");
        runtime->api.optionsAddDelegate(runtime->options, primaryDelegate);
        if (guardDelegate != nullptr) runtime->api.optionsAddDelegate(runtime->options, guardDelegate);
        runtime->interpreter = runtime->api.interpreterCreate(runtime->model, runtime->options);
        if (runtime->interpreter == nullptr) throw std::runtime_error("TfLiteInterpreterCreate failed");
        if (runtime->api.allocateTensors(runtime->interpreter) != kTfLiteOk) {
            throw std::runtime_error("TfLiteInterpreterAllocateTensors failed");
        }
        const int expectedInputs = kind == kEncodeKind ? 2 : 3;
        const int expectedOutputs = kind == kEncodeKind ? 8 : 6;
        if (runtime->api.inputCount(runtime->interpreter) != expectedInputs ||
            runtime->api.outputCount(runtime->interpreter) != expectedOutputs) {
            throw std::runtime_error("Small fused entropy input/output count mismatch");
        }
        return runtime.release();
    } catch (...) {
        destroyRuntime(runtime.release());
        throw;
    }
}

jlongArray tensorSizes(JNIEnv* env, Runtime* runtime, bool inputs) {
    if (runtime == nullptr) throw std::runtime_error("Small GPU rANS runtime is closed");
    const int count = inputs
        ? runtime->api.inputCount(runtime->interpreter)
        : runtime->api.outputCount(runtime->interpreter);
    std::vector<jlong> sizes(count);
    for (int index = 0; index < count; ++index) {
        const TfLiteTensor* tensor = inputs
            ? runtime->api.inputTensor(runtime->interpreter, index)
            : runtime->api.outputTensor(runtime->interpreter, index);
        if (tensor == nullptr) throw std::runtime_error("Small GPU rANS tensor unavailable");
        sizes[index] = static_cast<jlong>(runtime->api.tensorByteSize(tensor));
    }
    jlongArray result = env->NewLongArray(count);
    env->SetLongArrayRegion(result, 0, count, sizes.data());
    return result;
}

jobjectArray runRuntime(JNIEnv* env, Runtime* runtime, jobjectArray javaInputs) {
    if (runtime == nullptr) throw std::runtime_error("Small GPU rANS runtime is closed");
    const int inputCount = runtime->api.inputCount(runtime->interpreter);
    if (javaInputs == nullptr || env->GetArrayLength(javaInputs) != inputCount) {
        throw std::runtime_error("Small GPU rANS input count mismatch");
    }
    for (int index = 0; index < inputCount; ++index) {
        auto input = static_cast<jbyteArray>(env->GetObjectArrayElement(javaInputs, index));
        if (input == nullptr) throw std::runtime_error("Small GPU rANS input is null");
        TfLiteTensor* tensor = runtime->api.inputTensor(runtime->interpreter, index);
        const size_t bytes = runtime->api.tensorByteSize(tensor);
        if (env->GetArrayLength(input) != static_cast<jsize>(bytes)) {
            env->DeleteLocalRef(input);
            throw std::runtime_error("Small GPU rANS input byte size mismatch at index " + std::to_string(index));
        }
        std::vector<jbyte> buffer(bytes);
        env->GetByteArrayRegion(input, 0, static_cast<jsize>(bytes), buffer.data());
        env->DeleteLocalRef(input);
        if (runtime->api.tensorCopyFromBuffer(tensor, buffer.data(), bytes) != kTfLiteOk) {
            throw std::runtime_error("Small GPU rANS input copy failed at index " + std::to_string(index));
        }
    }
    TfLiteStatus invokeStatus;
    if (runtime->isMtk) {
        std::lock_guard<std::mutex> lock(gMtkInvokeMutex);
        gActiveMtkRuntime = runtime;
        invokeStatus = runtime->api.invoke(runtime->interpreter);
        gActiveMtkRuntime = nullptr;
    } else {
        invokeStatus = runtime->api.invoke(runtime->interpreter);
    }
    if (invokeStatus != kTfLiteOk) {
        throw std::runtime_error("TfLiteInterpreterInvoke failed");
    }
    const int outputCount = runtime->api.outputCount(runtime->interpreter);
    jclass byteArrayClass = env->FindClass("[B");
    jobjectArray result = env->NewObjectArray(outputCount, byteArrayClass, nullptr);
    for (int index = 0; index < outputCount; ++index) {
        const TfLiteTensor* tensor = runtime->api.outputTensor(runtime->interpreter, index);
        const size_t bytes = runtime->api.tensorByteSize(tensor);
        std::vector<uint8_t> buffer(bytes);
        if (runtime->api.tensorCopyToBuffer(tensor, buffer.data(), bytes) != kTfLiteOk) {
            env->DeleteLocalRef(byteArrayClass);
            throw std::runtime_error("Small GPU rANS output copy failed at index " + std::to_string(index));
        }
        jbyteArray output = env->NewByteArray(static_cast<jsize>(bytes));
        env->SetByteArrayRegion(
            output, 0, static_cast<jsize>(bytes), reinterpret_cast<const jbyte*>(buffer.data()));
        env->SetObjectArrayElement(result, index, output);
        env->DeleteLocalRef(output);
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
Java_com_gvcrt_clean_SmallEntropyGpuRuntime_nativeCreate(
    JNIEnv* env,
    jclass,
    jstring modelPath,
    jint kind,
    jlong gpuDelegate,
    jlong guardDelegate) {
    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    if (path == nullptr) return 0;
    const std::string model(path);
    env->ReleaseStringUTFChars(modelPath, path);
    try {
        return reinterpret_cast<jlong>(createRuntime(
            model,
            kind,
            reinterpret_cast<void*>(static_cast<uintptr_t>(gpuDelegate)),
            reinterpret_cast<void*>(static_cast<uintptr_t>(guardDelegate)),
            false));
    } catch (const std::exception& error) {
        throwJava(env, error.what());
        return 0;
    }
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_gvcrt_clean_SmallEntropyGpuRuntime_nativeInputSizes(
    JNIEnv* env, jclass, jlong handle) {
    try {
        return tensorSizes(env, reinterpret_cast<Runtime*>(handle), true);
    } catch (const std::exception& error) {
        throwJava(env, error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_gvcrt_clean_SmallEntropyGpuRuntime_nativeOutputSizes(
    JNIEnv* env, jclass, jlong handle) {
    try {
        return tensorSizes(env, reinterpret_cast<Runtime*>(handle), false);
    } catch (const std::exception& error) {
        throwJava(env, error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_gvcrt_clean_SmallEntropyGpuRuntime_nativeRun(
    JNIEnv* env, jclass, jlong handle, jobjectArray inputs) {
    try {
        return runRuntime(env, reinterpret_cast<Runtime*>(handle), inputs);
    } catch (const std::exception& error) {
        throwJava(env, error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_gvcrt_clean_SmallEntropyGpuRuntime_nativeClose(
    JNIEnv*, jclass, jlong handle) {
    destroyRuntime(reinterpret_cast<Runtime*>(handle));
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_gvcrt_clean_SmallEntropyMtkRuntime_nativeCreate(
    JNIEnv* env,
    jclass,
    jstring modelPath,
    jint kind,
    jlong delegate) {
    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    if (path == nullptr) return 0;
    const std::string model(path);
    env->ReleaseStringUTFChars(modelPath, path);
    try {
        return reinterpret_cast<jlong>(createRuntime(
            model,
            kind,
            reinterpret_cast<void*>(static_cast<uintptr_t>(delegate)),
            nullptr,
            true));
    } catch (const std::exception& error) {
        throwJava(env, error.what());
        return 0;
    }
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_gvcrt_clean_SmallEntropyMtkRuntime_nativeInputSizes(
    JNIEnv* env, jclass, jlong handle) {
    try {
        return tensorSizes(env, reinterpret_cast<Runtime*>(handle), true);
    } catch (const std::exception& error) {
        throwJava(env, error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_gvcrt_clean_SmallEntropyMtkRuntime_nativeOutputSizes(
    JNIEnv* env, jclass, jlong handle) {
    try {
        return tensorSizes(env, reinterpret_cast<Runtime*>(handle), false);
    } catch (const std::exception& error) {
        throwJava(env, error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_gvcrt_clean_SmallEntropyMtkRuntime_nativeRun(
    JNIEnv* env, jclass, jlong handle, jobjectArray inputs) {
    try {
        return runRuntime(env, reinterpret_cast<Runtime*>(handle), inputs);
    } catch (const std::exception& error) {
        throwJava(env, error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_gvcrt_clean_SmallEntropyMtkRuntime_nativeClose(
    JNIEnv*, jclass, jlong handle) {
    destroyRuntime(reinterpret_cast<Runtime*>(handle));
}
