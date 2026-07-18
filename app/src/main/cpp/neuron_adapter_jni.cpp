#include <jni.h>

#include <dlfcn.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct NeuronModel;
struct NeuronCompilation;
struct NeuronExecution;
struct NeuronDevice;

struct NeuronOperandType {
    int32_t type;
    uint32_t dimensionCount;
    const uint32_t* dimensions;
    float scale;
    int32_t zeroPoint;
};

constexpr int32_t kTensorFloat16 = 8;
constexpr int32_t kInt32 = 1;
constexpr int32_t kConv2d = 3;
constexpr int32_t kPaddingValid = 2;
constexpr int32_t kFusedNone = 0;
constexpr size_t kInputBytes = 1 * 16 * 32 * 256 * sizeof(uint16_t);
constexpr size_t kWeightBytes = 512 * 1 * 1 * 256 * sizeof(uint16_t);
constexpr size_t kBiasBytes = 512 * sizeof(uint16_t);
constexpr size_t kOutputBytes = 1 * 16 * 32 * 512 * sizeof(uint16_t);

using ModelCreateFn = int (*)(NeuronModel**);
using ModelFreeFn = void (*)(NeuronModel*);
using ModelAddOperandFn = int (*)(NeuronModel*, const NeuronOperandType*);
using ModelSetOperandValueFn = int (*)(NeuronModel*, int32_t, const void*, size_t);
using ModelAddOperationFn = int (*)(NeuronModel*, int32_t, uint32_t, const uint32_t*, uint32_t, const uint32_t*);
using ModelIdentifyIoFn = int (*)(NeuronModel*, uint32_t, const uint32_t*, uint32_t, const uint32_t*);
using ModelFinishFn = int (*)(NeuronModel*);
using CompilationCreateForDevicesFn = int (*)(NeuronModel*, const NeuronDevice* const*, uint32_t, NeuronCompilation**);
using CompilationFinishFn = int (*)(NeuronCompilation*);
using CompilationFreeFn = void (*)(NeuronCompilation*);
using ExecutionCreateFn = int (*)(NeuronCompilation*, NeuronExecution**);
using ExecutionFreeFn = void (*)(NeuronExecution*);
using ExecutionSetInputFn = int (*)(NeuronExecution*, int32_t, const NeuronOperandType*, const void*, size_t);
using ExecutionSetOutputFn = int (*)(NeuronExecution*, int32_t, const NeuronOperandType*, void*, size_t);
using ExecutionComputeFn = int (*)(NeuronExecution*);
using GetDeviceCountFn = int (*)(uint32_t*);
using GetDeviceFn = int (*)(uint32_t, NeuronDevice**);
using DeviceGetNameFn = int (*)(const NeuronDevice*, const char**);

struct AdapterApi {
    void* library = nullptr;
    ModelCreateFn modelCreate = nullptr;
    ModelFreeFn modelFree = nullptr;
    ModelAddOperandFn modelAddOperand = nullptr;
    ModelSetOperandValueFn modelSetOperandValue = nullptr;
    ModelAddOperationFn modelAddOperation = nullptr;
    ModelIdentifyIoFn modelIdentifyIo = nullptr;
    ModelFinishFn modelFinish = nullptr;
    CompilationCreateForDevicesFn compilationCreateForDevices = nullptr;
    CompilationFinishFn compilationFinish = nullptr;
    CompilationFreeFn compilationFree = nullptr;
    ExecutionCreateFn executionCreate = nullptr;
    ExecutionFreeFn executionFree = nullptr;
    ExecutionSetInputFn executionSetInput = nullptr;
    ExecutionSetOutputFn executionSetOutput = nullptr;
    ExecutionComputeFn executionCompute = nullptr;
    GetDeviceCountFn getDeviceCount = nullptr;
    GetDeviceFn getDevice = nullptr;
    DeviceGetNameFn deviceGetName = nullptr;

    ~AdapterApi() {
        if (library != nullptr) dlclose(library);
    }
};

template <typename T>
T symbol(void* library, const char* name) {
    auto* value = dlsym(library, name);
    if (value == nullptr) throw std::runtime_error(std::string("missing Adapter symbol ") + name);
    return reinterpret_cast<T>(value);
}

std::shared_ptr<AdapterApi> loadApi() {
    static constexpr const char* kLibraries[] = {
        "libneuronusdk_adapter.mtk.so",
        "libneuron_adapter_mgvi.so",
        "libneuron_adapter.so",
    };
    auto api = std::make_shared<AdapterApi>();
    std::ostringstream errors;
    for (const char* name : kLibraries) {
        api->library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (api->library != nullptr) break;
        if (errors.tellp() > 0) errors << " | ";
        const char* error = dlerror();
        errors << name << '=' << (error == nullptr ? "unknown" : error);
    }
    if (api->library == nullptr) throw std::runtime_error("unable to load Neuron Adapter library: " + errors.str());
    api->modelCreate = symbol<ModelCreateFn>(api->library, "NeuronModel_create");
    api->modelFree = symbol<ModelFreeFn>(api->library, "NeuronModel_free");
    api->modelAddOperand = symbol<ModelAddOperandFn>(api->library, "NeuronModel_addOperand");
    api->modelSetOperandValue = symbol<ModelSetOperandValueFn>(api->library, "NeuronModel_setOperandValue");
    api->modelAddOperation = symbol<ModelAddOperationFn>(api->library, "NeuronModel_addOperation");
    api->modelIdentifyIo = symbol<ModelIdentifyIoFn>(api->library, "NeuronModel_identifyInputsAndOutputs");
    api->modelFinish = symbol<ModelFinishFn>(api->library, "NeuronModel_finish");
    api->compilationCreateForDevices = symbol<CompilationCreateForDevicesFn>(api->library, "NeuronCompilation_createForDevices");
    api->compilationFinish = symbol<CompilationFinishFn>(api->library, "NeuronCompilation_finish");
    api->compilationFree = symbol<CompilationFreeFn>(api->library, "NeuronCompilation_free");
    api->executionCreate = symbol<ExecutionCreateFn>(api->library, "NeuronExecution_create");
    api->executionFree = symbol<ExecutionFreeFn>(api->library, "NeuronExecution_free");
    api->executionSetInput = symbol<ExecutionSetInputFn>(api->library, "NeuronExecution_setInput");
    api->executionSetOutput = symbol<ExecutionSetOutputFn>(api->library, "NeuronExecution_setOutput");
    api->executionCompute = symbol<ExecutionComputeFn>(api->library, "NeuronExecution_compute");
    api->getDeviceCount = symbol<GetDeviceCountFn>(api->library, "Neuron_getDeviceCount");
    api->getDevice = symbol<GetDeviceFn>(api->library, "Neuron_getDevice");
    api->deviceGetName = symbol<DeviceGetNameFn>(api->library, "NeuronDevice_getName");
    return api;
}

void requireOk(int status, const char* operation) {
    if (status != 0) throw std::runtime_error(std::string(operation) + " failed status=" + std::to_string(status));
}

std::vector<uint8_t> bytes(JNIEnv* env, jbyteArray array) {
    const jsize size = env->GetArrayLength(array);
    std::vector<uint8_t> result(static_cast<size_t>(size));
    env->GetByteArrayRegion(array, 0, size, reinterpret_cast<jbyte*>(result.data()));
    return result;
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

struct Handles {
    std::shared_ptr<AdapterApi> api;
    NeuronModel* model = nullptr;
    NeuronCompilation* compilation = nullptr;
    NeuronExecution* execution = nullptr;

    ~Handles() {
        if (execution != nullptr) api->executionFree(execution);
        if (compilation != nullptr) api->compilationFree(compilation);
        if (model != nullptr) api->modelFree(model);
    }
};

jobjectArray result(JNIEnv* env, const std::vector<uint8_t>& output, const std::string& status) {
    jclass byteArrayClass = env->FindClass("[B");
    jobjectArray values = env->NewObjectArray(2, byteArrayClass, nullptr);
    jbyteArray outputArray = env->NewByteArray(static_cast<jsize>(output.size()));
    env->SetByteArrayRegion(outputArray, 0, static_cast<jsize>(output.size()), reinterpret_cast<const jbyte*>(output.data()));
    jbyteArray statusArray = env->NewByteArray(static_cast<jsize>(status.size()));
    env->SetByteArrayRegion(statusArray, 0, static_cast<jsize>(status.size()), reinterpret_cast<const jbyte*>(status.data()));
    env->SetObjectArrayElement(values, 0, outputArray);
    env->SetObjectArrayElement(values, 1, statusArray);
    return values;
}

}  // namespace

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativeNeuronAdapterFp16Conv(
    JNIEnv* env,
    jclass,
    jbyteArray inputArray,
    jbyteArray weightArray,
    jbyteArray biasArray) {
    try {
        const std::vector<uint8_t> input = bytes(env, inputArray);
        const std::vector<uint8_t> weights = bytes(env, weightArray);
        const std::vector<uint8_t> bias = bytes(env, biasArray);
        if (input.size() != kInputBytes || weights.size() != kWeightBytes || bias.size() != kBiasBytes) {
            throw std::runtime_error(
                "FP16 buffer size mismatch input=" + std::to_string(input.size()) +
                " weight=" + std::to_string(weights.size()) + " bias=" + std::to_string(bias.size()));
        }

        Handles handles;
        handles.api = loadApi();
        uint32_t deviceCount = 0;
        requireOk(handles.api->getDeviceCount(&deviceCount), "Neuron_getDeviceCount");
        std::vector<std::string> deviceNames;
        const NeuronDevice* mdlaDevice = nullptr;
        for (uint32_t index = 0; index < deviceCount; ++index) {
            NeuronDevice* device = nullptr;
            requireOk(handles.api->getDevice(index, &device), "Neuron_getDevice");
            const char* name = nullptr;
            requireOk(handles.api->deviceGetName(device, &name), "NeuronDevice_getName");
            const std::string deviceName = name == nullptr ? "<null>" : name;
            deviceNames.push_back(deviceName);
            if (lowercase(deviceName).find("mdla") != std::string::npos) mdlaDevice = device;
        }
        if (mdlaDevice == nullptr) {
            std::ostringstream message;
            message << "no MDLA device available devices=";
            for (size_t index = 0; index < deviceNames.size(); ++index) {
                if (index != 0) message << ':';
                message << deviceNames[index];
            }
            throw std::runtime_error(message.str());
        }

        requireOk(handles.api->modelCreate(&handles.model), "NeuronModel_create");
        const uint32_t inputDims[] = {1, 16, 32, 256};
        const uint32_t weightDims[] = {512, 1, 1, 256};
        const uint32_t biasDims[] = {512};
        const uint32_t outputDims[] = {1, 16, 32, 512};
        const NeuronOperandType inputType{kTensorFloat16, 4, inputDims, 0.0f, 0};
        const NeuronOperandType weightType{kTensorFloat16, 4, weightDims, 0.0f, 0};
        const NeuronOperandType biasType{kTensorFloat16, 1, biasDims, 0.0f, 0};
        const NeuronOperandType scalarType{kInt32, 0, nullptr, 0.0f, 0};
        const NeuronOperandType outputType{kTensorFloat16, 4, outputDims, 0.0f, 0};
        for (const NeuronOperandType* type : {&inputType, &weightType, &biasType, &scalarType, &scalarType, &scalarType, &scalarType, &outputType}) {
            requireOk(handles.api->modelAddOperand(handles.model, type), "NeuronModel_addOperand");
        }
        requireOk(handles.api->modelSetOperandValue(handles.model, 1, weights.data(), weights.size()), "set weight");
        requireOk(handles.api->modelSetOperandValue(handles.model, 2, bias.data(), bias.size()), "set bias");
        const int32_t padding = kPaddingValid;
        const int32_t stride = 1;
        const int32_t activation = kFusedNone;
        requireOk(handles.api->modelSetOperandValue(handles.model, 3, &padding, sizeof(padding)), "set padding");
        requireOk(handles.api->modelSetOperandValue(handles.model, 4, &stride, sizeof(stride)), "set stride_w");
        requireOk(handles.api->modelSetOperandValue(handles.model, 5, &stride, sizeof(stride)), "set stride_h");
        requireOk(handles.api->modelSetOperandValue(handles.model, 6, &activation, sizeof(activation)), "set activation");
        const uint32_t opInputs[] = {0, 1, 2, 3, 4, 5, 6};
        const uint32_t opOutputs[] = {7};
        requireOk(handles.api->modelAddOperation(handles.model, kConv2d, 7, opInputs, 1, opOutputs), "NeuronModel_addOperation(CONV_2D)");
        const uint32_t modelInputs[] = {0};
        const uint32_t modelOutputs[] = {7};
        requireOk(handles.api->modelIdentifyIo(handles.model, 1, modelInputs, 1, modelOutputs), "NeuronModel_identifyInputsAndOutputs");
        requireOk(handles.api->modelFinish(handles.model), "NeuronModel_finish");
        const NeuronDevice* devices[] = {mdlaDevice};
        requireOk(handles.api->compilationCreateForDevices(handles.model, devices, 1, &handles.compilation), "NeuronCompilation_createForDevices(MDLA)");
        requireOk(handles.api->compilationFinish(handles.compilation), "NeuronCompilation_finish");
        requireOk(handles.api->executionCreate(handles.compilation, &handles.execution), "NeuronExecution_create");
        std::vector<uint8_t> output(kOutputBytes);
        requireOk(handles.api->executionSetInput(handles.execution, 0, nullptr, input.data(), input.size()), "NeuronExecution_setInput");
        requireOk(handles.api->executionSetOutput(handles.execution, 0, nullptr, output.data(), output.size()), "NeuronExecution_setOutput");
        requireOk(handles.api->executionCompute(handles.execution), "NeuronExecution_compute");

        std::ostringstream status;
        status << "adapter_api=ok selected_device=mdla devices=";
        for (size_t index = 0; index < deviceNames.size(); ++index) {
            if (index != 0) status << ':';
            status << deviceNames[index];
        }
        return result(env, output, status.str());
    } catch (const std::exception& error) {
        jclass exception = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exception, error.what());
        return nullptr;
    }
}
