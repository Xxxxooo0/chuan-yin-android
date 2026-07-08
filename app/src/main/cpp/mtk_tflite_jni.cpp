#include <jni.h>

#include <dlfcn.h>
#include <android/log.h>

#include <cstdint>
#include <chrono>
#include <cstring>
#include <fstream>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <vulkan/vulkan.h>

#include "fused_upsampler_spv.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

void pixelUnshuffle2Nchw256x32x64(const float* input, float* output);
void nativeGroupNormOptimized(const float* input, float* output, int channels, int spatial, int requestedThreadCount);
void nativeAdaGnNchw(
    const float* feature,
    const float* codeword,
    const float* gammaWeight,
    const float* gammaBias,
    const float* betaWeight,
    const float* betaBias,
    float* normalized,
    float* output,
    int outputChannels,
    int spatial,
    int requestedThreadCount);

namespace {

constexpr uint32_t kInputBuffer = 0;
constexpr uint32_t kOutputBuffer = 1;
constexpr const char* kLogTag = "GVC_RT_MTK";
constexpr bool kVerboseRunLogging = false;

using OptionsCreateFn = int (*)(void**);
using OptionsFreeFn = void (*)(void*);
using OptionsSetAccelerationModeFn = int (*)(void*, int);
using OptionsSetAllowFp16Fn = int (*)(void*, bool);
using OptionsSetBoolFn = int (*)(void*, bool);
using OptionsSetIntFn = int (*)(void*, int);
using OptionsSetStringFn = int (*)(void*, const char*);
using OptionsSetUintFn = int (*)(void*, uint32_t);
using CreateAdvFn = int (*)(void**, const char*, void*);
using FreeFn = void (*)(void*);
using GetTensorCountFn = int (*)(void*, uint32_t, uint32_t*);
using GetTensorByteSizeFn = int (*)(void*, uint32_t, uint32_t, size_t*);
struct TFLiteTensorExt {
    uint32_t type = 0;
    int dimsSize = 0;
    int dims[16] = {};
    void* buffer = nullptr;
    size_t bufferSize = 0;
};
using GetTensorFn = int (*)(void*, uint32_t, TFLiteTensorExt*);
using GetTensorByIndexFn = int (*)(void*, uint32_t, TFLiteTensorExt*, int);
using SetTensorBufferFn = int (*)(void*, int, char*);
using SetInputTensorDataFn = int (*)(void*, uint32_t, const void*, size_t);
using InvokeFn = int (*)(void*);
using GetOutputTensorDataFn = int (*)(void*, uint32_t, void*, size_t);
using IsFullyDelegatedFn = int (*)(void*, bool*);
using NeuronDeviceGetExtensionSupportFn = int (*)(const char*, bool*);
using NeuronRuntimeV2CreateFn = int (*)(const char*, size_t, void**, size_t);
using NeuronRuntimeV2ReleaseFn = void (*)(void*);
using NeuronRuntimeV2GetCountFn = int (*)(void*, size_t*);

using cl_int = int;
using cl_uint = unsigned int;
using cl_ulong = unsigned long;
using cl_bool = cl_uint;
using cl_bitfield = cl_ulong;
using cl_device_type = cl_bitfield;
using cl_mem_flags = cl_bitfield;
using cl_platform_id = struct _cl_platform_id*;
using cl_device_id = struct _cl_device_id*;
using cl_context = struct _cl_context*;
using cl_command_queue = struct _cl_command_queue*;
using cl_program = struct _cl_program*;
using cl_kernel = struct _cl_kernel*;
using cl_mem = struct _cl_mem*;
using cl_event = struct _cl_event*;

constexpr cl_int CL_SUCCESS_VALUE = 0;
constexpr cl_bool CL_TRUE_VALUE = 1;
constexpr cl_device_type CL_DEVICE_TYPE_GPU_VALUE = 1 << 2;
constexpr cl_mem_flags CL_MEM_READ_ONLY_VALUE = 1 << 2;
constexpr cl_mem_flags CL_MEM_WRITE_ONLY_VALUE = 1 << 1;
constexpr cl_mem_flags CL_MEM_COPY_HOST_PTR_VALUE = 1 << 5;
constexpr cl_uint CL_PROGRAM_BUILD_LOG_VALUE = 0x1183;

using ClGetPlatformIDsFn = cl_int (*)(cl_uint, cl_platform_id*, cl_uint*);
using ClGetDeviceIDsFn = cl_int (*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
using ClCreateContextFn = cl_context (*)(const void*, cl_uint, const cl_device_id*, void (*)(const char*, const void*, size_t, void*), void*, cl_int*);
using ClReleaseContextFn = cl_int (*)(cl_context);
using ClCreateCommandQueueFn = cl_command_queue (*)(cl_context, cl_device_id, cl_bitfield, cl_int*);
using ClReleaseCommandQueueFn = cl_int (*)(cl_command_queue);
using ClCreateProgramWithSourceFn = cl_program (*)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
using ClBuildProgramFn = cl_int (*)(cl_program, cl_uint, const cl_device_id*, const char*, void (*)(cl_program, void*), void*);
using ClGetProgramBuildInfoFn = cl_int (*)(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*);
using ClReleaseProgramFn = cl_int (*)(cl_program);
using ClCreateKernelFn = cl_kernel (*)(cl_program, const char*, cl_int*);
using ClReleaseKernelFn = cl_int (*)(cl_kernel);
using ClCreateBufferFn = cl_mem (*)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
using ClReleaseMemObjectFn = cl_int (*)(cl_mem);
using ClSetKernelArgFn = cl_int (*)(cl_kernel, cl_uint, size_t, const void*);
using ClEnqueueNDRangeKernelFn = cl_int (*)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
using ClFinishFn = cl_int (*)(cl_command_queue);
using ClEnqueueReadBufferFn = cl_int (*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);

struct OpenClApi {
    void* library = nullptr;
    ClGetPlatformIDsFn getPlatformIDs = nullptr;
    ClGetDeviceIDsFn getDeviceIDs = nullptr;
    ClCreateContextFn createContext = nullptr;
    ClReleaseContextFn releaseContext = nullptr;
    ClCreateCommandQueueFn createCommandQueue = nullptr;
    ClReleaseCommandQueueFn releaseCommandQueue = nullptr;
    ClCreateProgramWithSourceFn createProgramWithSource = nullptr;
    ClBuildProgramFn buildProgram = nullptr;
    ClGetProgramBuildInfoFn getProgramBuildInfo = nullptr;
    ClReleaseProgramFn releaseProgram = nullptr;
    ClCreateKernelFn createKernel = nullptr;
    ClReleaseKernelFn releaseKernel = nullptr;
    ClCreateBufferFn createBuffer = nullptr;
    ClReleaseMemObjectFn releaseMemObject = nullptr;
    ClSetKernelArgFn setKernelArg = nullptr;
    ClEnqueueNDRangeKernelFn enqueueNDRangeKernel = nullptr;
    ClFinishFn finish = nullptr;
    ClEnqueueReadBufferFn enqueueReadBuffer = nullptr;

    ~OpenClApi() {
        if (library != nullptr) {
            dlclose(library);
        }
    }
};

struct Api {
    void* library = nullptr;
    OptionsCreateFn optionsCreate = nullptr;
    OptionsFreeFn optionsFree = nullptr;
    OptionsSetAccelerationModeFn optionsSetAccelerationMode = nullptr;
    OptionsSetAllowFp16Fn optionsSetAllowFp16 = nullptr;
    OptionsSetIntFn optionsSetPreference = nullptr;
    OptionsSetIntFn optionsSetExecutionPriority = nullptr;
    OptionsSetBoolFn optionsSetLowLatency = nullptr;
    OptionsSetBoolFn optionsSetDeepFusion = nullptr;
    OptionsSetIntFn optionsSetBoostHint = nullptr;
    OptionsSetIntFn optionsSetBoostDuration = nullptr;
    OptionsSetStringFn optionsSetCacheDir = nullptr;
    OptionsSetIntFn optionsSetMaxNumberDelegatedPartitions = nullptr;
    OptionsSetBoolFn optionsSetDisallowNnApiCpu = nullptr;
    OptionsSetUintFn optionsSetAcceleratorFlag = nullptr;
    CreateAdvFn createAdv = nullptr;
    FreeFn free = nullptr;
    GetTensorCountFn getTensorCount = nullptr;
    GetTensorByteSizeFn getTensorByteSize = nullptr;
    GetTensorFn getTensor = nullptr;
    GetTensorByIndexFn getTensorByIndex = nullptr;
    SetTensorBufferFn setTensorBuffer = nullptr;
    SetInputTensorDataFn setInputTensorData = nullptr;
    InvokeFn invoke = nullptr;
    GetOutputTensorDataFn getOutputTensorData = nullptr;
    IsFullyDelegatedFn isFullyDelegated = nullptr;

    ~Api() {
        if (library != nullptr) {
            dlclose(library);
        }
    }
};

struct Runtime {
    std::shared_ptr<Api> api;
    void* tflite = nullptr;
    int accelerationMode = 1;
    std::vector<size_t> inputSizes;
    std::vector<size_t> outputSizes;
    std::string optionsSummary;

    ~Runtime() {
        if (tflite != nullptr) {
            api->free(tflite);
        }
    }
};

[[noreturn]] void fail(JNIEnv* env, const std::string& message) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"), message.c_str());
    throw std::runtime_error(message);
}

template <typename T>
T loadSymbol(JNIEnv* env, void* library, const char* name) {
    auto* symbol = dlsym(library, name);
    if (symbol == nullptr) {
        fail(env, std::string("missing symbol ") + name + ": " + dlerror());
    }
    return reinterpret_cast<T>(symbol);
}

void check(JNIEnv* env, int status, const char* op) {
    if (status != 0) {
        fail(env, std::string(op) + " failed status=" + std::to_string(status));
    }
}

std::shared_ptr<Api> loadApi(JNIEnv* env) {
    auto api = std::make_shared<Api>();
    api->library = dlopen("libtflite_mtk.so", RTLD_NOW | RTLD_LOCAL);
    if (api->library == nullptr) {
        api->library = dlopen("/vendor/lib64/libtflite_mtk.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (api->library == nullptr) {
        fail(env, std::string("dlopen libtflite_mtk.so failed: ") + dlerror());
    }
    api->optionsCreate = loadSymbol<OptionsCreateFn>(env, api->library, "ANeuroPilotTFLiteOptions_create");
    api->optionsFree = loadSymbol<OptionsFreeFn>(env, api->library, "ANeuroPilotTFLiteOptions_free");
    api->optionsSetAccelerationMode = loadSymbol<OptionsSetAccelerationModeFn>(
        env, api->library, "ANeuroPilotTFLiteOptions_setAccelerationMode");
    api->optionsSetAllowFp16 = loadSymbol<OptionsSetAllowFp16Fn>(
        env, api->library, "ANeuroPilotTFLiteOptions_setAllowFp16PrecisionForFp32");
    api->optionsSetPreference = reinterpret_cast<OptionsSetIntFn>(
        dlsym(api->library, "ANeuroPilotTFLiteOptions_setPreference"));
    api->optionsSetExecutionPriority = reinterpret_cast<OptionsSetIntFn>(
        dlsym(api->library, "ANeuroPilotTFLiteOptions_setExecutionPriority"));
    api->optionsSetLowLatency = reinterpret_cast<OptionsSetBoolFn>(
        dlsym(api->library, "ANeuroPilotTFLiteOptions_setLowLatency"));
    api->optionsSetDeepFusion = reinterpret_cast<OptionsSetBoolFn>(
        dlsym(api->library, "ANeuroPilotTFLiteOptions_setDeepFusion"));
    api->optionsSetBoostHint = reinterpret_cast<OptionsSetIntFn>(
        dlsym(api->library, "ANeuroPilotTFLiteOptions_setBoostHint"));
    api->optionsSetBoostDuration = reinterpret_cast<OptionsSetIntFn>(
        dlsym(api->library, "ANeuroPilotTFLiteOptions_setBoostDuration"));
    api->optionsSetCacheDir = reinterpret_cast<OptionsSetStringFn>(
        dlsym(api->library, "ANeuroPilotTFLiteOptions_setCacheDir"));
    api->optionsSetMaxNumberDelegatedPartitions = reinterpret_cast<OptionsSetIntFn>(
        dlsym(api->library, "ANeuroPilotTFLiteOptions_setMaxNumberDelegatedPartitions"));
    api->optionsSetDisallowNnApiCpu = reinterpret_cast<OptionsSetBoolFn>(
        dlsym(api->library, "ANeuroPilotTFLiteOptions_setDisallowNnApiCpu"));
    api->optionsSetAcceleratorFlag = reinterpret_cast<OptionsSetUintFn>(
        dlsym(api->library, "ANeuroPilotTFLiteOptions_setAcceleratorFlag"));
    api->createAdv = loadSymbol<CreateAdvFn>(env, api->library, "ANeuroPilotTFLite_createAdv");
    api->free = loadSymbol<FreeFn>(env, api->library, "ANeuroPilotTFLite_free");
    api->getTensorCount = loadSymbol<GetTensorCountFn>(env, api->library, "ANeuroPilotTFLite_getTensorCount");
    api->getTensorByteSize = loadSymbol<GetTensorByteSizeFn>(env, api->library, "ANeuroPilotTFLite_getTensorByteSize");
    api->getTensor = loadSymbol<GetTensorFn>(env, api->library, "ANeuroPilotTFLite_getTensor");
    api->getTensorByIndex = loadSymbol<GetTensorByIndexFn>(env, api->library, "ANeuroPilotTFLite_getTensorByIndex");
    api->setTensorBuffer = loadSymbol<SetTensorBufferFn>(env, api->library, "ANeuroPilotTFLite_setTensorBuffer");
    api->setInputTensorData = loadSymbol<SetInputTensorDataFn>(env, api->library, "ANeuroPilotTFLite_setInputTensorData");
    api->invoke = loadSymbol<InvokeFn>(env, api->library, "ANeuroPilotTFLite_invoke");
    api->getOutputTensorData = loadSymbol<GetOutputTensorDataFn>(env, api->library, "ANeuroPilotTFLite_getOutputTensorData");
    api->isFullyDelegated = reinterpret_cast<IsFullyDelegatedFn>(dlsym(api->library, "ANeuroPilotTFLite_isFullyDelegated"));
    return api;
}

std::vector<size_t> querySizes(JNIEnv* env, Runtime& runtime, uint32_t bufferType) {
    uint32_t count = 0;
    check(env, runtime.api->getTensorCount(runtime.tflite, bufferType, &count), "ANeuroPilotTFLite_getTensorCount");
    std::vector<size_t> sizes(count);
    for (uint32_t i = 0; i < count; ++i) {
        check(env, runtime.api->getTensorByteSize(runtime.tflite, bufferType, i, &sizes[i]),
              "ANeuroPilotTFLite_getTensorByteSize");
    }
    return sizes;
}

Runtime* fromHandle(JNIEnv* env, jlong handle) {
    auto* runtime = reinterpret_cast<Runtime*>(handle);
    if (runtime == nullptr) {
        fail(env, "invalid MTK TFLite runtime handle");
    }
    return runtime;
}

jlongArray toLongArray(JNIEnv* env, const std::vector<size_t>& values) {
    auto result = env->NewLongArray(static_cast<jsize>(values.size()));
    std::vector<jlong> converted(values.begin(), values.end());
    env->SetLongArrayRegion(result, 0, static_cast<jsize>(converted.size()), converted.data());
    return result;
}

void appendOptionStatus(std::vector<std::string>& statuses, const std::string& name, int status) {
    statuses.push_back(name + "=" + (status == 0 ? "ok" : ("failed:" + std::to_string(status))));
}

template <typename Fn, typename Value>
void applyOptionalOption(std::vector<std::string>& statuses, const char* name, Fn fn, void* options, Value value) {
    if (fn == nullptr) {
        statuses.push_back(std::string(name) + "=missing");
        return;
    }
    appendOptionStatus(statuses, name, fn(options, value));
}

std::string joinStatuses(const std::vector<std::string>& statuses) {
    std::string joined;
    for (size_t i = 0; i < statuses.size(); ++i) {
        if (i != 0) {
            joined += ",";
        }
        joined += statuses[i];
    }
    return joined;
}

std::string probeNeuronExtensions(JNIEnv* env, jobjectArray names) {
    void* tfliteLibrary = dlopen("libtflite_mtk.so", RTLD_NOW | RTLD_LOCAL);
    std::string tfliteCustom = "tflite_custom_api=";
    if (tfliteLibrary == nullptr) {
        const char* error = dlerror();
        tfliteCustom += std::string("library_unavailable:") + (error != nullptr ? error : "unknown");
    } else {
        const char* customSymbols[] = {
            "ANeuroPilotTFLite_createCustom",
            "ANeuroPilotTFLite_createAdvCustom",
            "ANeuroPilotTFLite_createCustomWithBuffer",
            "ANeuroPilotTFLite_createAdvCustomWithBuffer",
        };
        for (size_t i = 0; i < sizeof(customSymbols) / sizeof(customSymbols[0]); ++i) {
            if (i != 0) {
                tfliteCustom += ",";
            }
            tfliteCustom += customSymbols[i];
            tfliteCustom += dlsym(tfliteLibrary, customSymbols[i]) != nullptr ? ":present" : ":missing";
        }
        dlclose(tfliteLibrary);
    }

    const char* libraryNames[] = {
        "libneuron_adapter.8.so",
        "libneuron_adapter.so.8.2.31",
        "libneuron_adapter.so",
        "/vendor/lib64/libneuron_adapter.8.so",
        "/vendor/lib64/libneuron_adapter.so",
    };
    void* library = nullptr;
    std::string loadedName;
    std::string lastError;
    for (const char* name : libraryNames) {
        library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (library != nullptr) {
            loadedName = name;
            break;
        }
        const char* error = dlerror();
        if (error != nullptr) {
            lastError = error;
        }
    }
    if (library == nullptr) {
        return tfliteCustom + " neuron_adapter_library=unavailable last_error=" + lastError;
    }

    auto getExtensionSupport = reinterpret_cast<NeuronDeviceGetExtensionSupportFn>(
        dlsym(library, "NeuronDevice_getExtensionSupport"));
    if (getExtensionSupport == nullptr) {
        std::string error = dlerror() != nullptr ? dlerror() : "unknown";
        dlclose(library);
        return tfliteCustom + " neuron_adapter_library=" + loadedName + " symbol=missing error=" + error;
    }

    std::string result = tfliteCustom + " neuron_adapter_library=" + loadedName + " symbol=ok";
    const jsize count = env->GetArrayLength(names);
    for (jsize i = 0; i < count; ++i) {
        auto item = static_cast<jstring>(env->GetObjectArrayElement(names, i));
        const char* chars = env->GetStringUTFChars(item, nullptr);
        bool supported = false;
        int status = getExtensionSupport(chars, &supported);
        result += " ";
        result += chars;
        result += "=status:";
        result += std::to_string(status);
        result += ",supported:";
        result += supported ? "true" : "false";
        env->ReleaseStringUTFChars(item, chars);
        env->DeleteLocalRef(item);
    }
    dlclose(library);
    return result;
}

std::string probeAhwbSymbols() {
    const char* libraryNames[] = {
        "libtflite_mtk.so",
        "/vendor/lib64/libtflite_mtk.so",
    };
    void* library = nullptr;
    std::string loadedName;
    std::string lastError;
    for (const char* name : libraryNames) {
        library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (library != nullptr) {
            loadedName = name;
            break;
        }
        const char* error = dlerror();
        if (error != nullptr) {
            lastError = error;
        }
    }
    if (library == nullptr) {
        return "library=unavailable last_error=" + lastError;
    }

    const char* symbols[] = {
        "ANeuralNetworksTFLiteOptions_setUseAhwb",
        "ANeuroPilotTFLiteWrapper_setBufferHandle",
        "ANeuroPilotTFLiteWrapper_makeAdvTFLite",
        "ANeuroPilotTFLiteWrapper_invoke",
        "ANeuroPilotTFLiteWrapper_free",
        "ANeuroPilotTFLiteWrapper_getInputTensorCount",
        "ANeuroPilotTFLiteWrapper_getOutputTensorCount",
        "ANeuroPilotTFLiteWrapper_getInputTensorSize",
        "ANeuroPilotTFLiteWrapper_getOutputTensorSize",
    };

    std::string result = "library=" + loadedName;
    for (const char* symbol : symbols) {
        result += " ";
        result += symbol;
        result += dlsym(library, symbol) != nullptr ? "=present" : "=missing";
    }
    dlclose(library);
    return result;
}

std::string probeDlaRuntimeV2(const std::string& dlaPath) {
    const char* libraryNames[] = {
        "libneuron_runtime.8.so",
        "libneuron_runtime.so.8.2.31",
        "libneuron_runtime.so",
        "/vendor/lib64/libneuron_runtime.8.so",
        "/vendor/lib64/libneuron_runtime.so",
    };
    void* library = nullptr;
    std::string loadedName;
    std::string lastError;
    for (const char* name : libraryNames) {
        library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (library != nullptr) {
            loadedName = name;
            break;
        }
        const char* error = dlerror();
        if (error != nullptr) {
            lastError = error;
        }
    }
    if (library == nullptr) {
        return "library=unavailable last_error=" + lastError;
    }

    auto create = reinterpret_cast<NeuronRuntimeV2CreateFn>(dlsym(library, "NeuronRuntimeV2_create"));
    auto release = reinterpret_cast<NeuronRuntimeV2ReleaseFn>(dlsym(library, "NeuronRuntimeV2_release"));
    auto getInputNumber =
        reinterpret_cast<NeuronRuntimeV2GetCountFn>(dlsym(library, "NeuronRuntimeV2_getInputNumber"));
    auto getOutputNumber =
        reinterpret_cast<NeuronRuntimeV2GetCountFn>(dlsym(library, "NeuronRuntimeV2_getOutputNumber"));
    if (create == nullptr || release == nullptr || getInputNumber == nullptr || getOutputNumber == nullptr) {
        std::string result = "library=" + loadedName;
        result += " create=" + std::string(create != nullptr ? "present" : "missing");
        result += " release=" + std::string(release != nullptr ? "present" : "missing");
        result += " getInputNumber=" + std::string(getInputNumber != nullptr ? "present" : "missing");
        result += " getOutputNumber=" + std::string(getOutputNumber != nullptr ? "present" : "missing");
        dlclose(library);
        return result;
    }

    void* runtime = nullptr;
    const auto started = std::chrono::steady_clock::now();
    const int createStatus = create(dlaPath.c_str(), 1, &runtime, 2048);
    const auto ended = std::chrono::steady_clock::now();
    std::string result = "library=" + loadedName;
    result += " symbols=present";
    result += " create_status=" + std::to_string(createStatus);
    result += " create_ms=" +
        std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(ended - started).count());
    if (createStatus == 0 && runtime != nullptr) {
        size_t inputCount = 0;
        size_t outputCount = 0;
        const int inputStatus = getInputNumber(runtime, &inputCount);
        const int outputStatus = getOutputNumber(runtime, &outputCount);
        result += " input_status=" + std::to_string(inputStatus);
        result += " input_count=" + std::to_string(inputCount);
        result += " output_status=" + std::to_string(outputStatus);
        result += " output_count=" + std::to_string(outputCount);
        release(runtime);
    }
    dlclose(library);
    return result;
}

template <typename T>
T loadOpenClSymbol(OpenClApi& api, const char* name) {
    auto* symbol = dlsym(api.library, name);
    if (symbol == nullptr) {
        throw std::runtime_error(std::string("missing OpenCL symbol ") + name + ": " + dlerror());
    }
    return reinterpret_cast<T>(symbol);
}

std::unique_ptr<OpenClApi> loadOpenClApi() {
    const char* names[] = {
        "libOpenCL.so",
        "/vendor/lib64/libOpenCL.so",
        "/system/vendor/lib64/libOpenCL.so",
    };
    auto api = std::make_unique<OpenClApi>();
    std::string lastError;
    for (const char* name : names) {
        api->library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (api->library != nullptr) {
            break;
        }
        const char* error = dlerror();
        if (error != nullptr) {
            lastError = error;
        }
    }
    if (api->library == nullptr) {
        throw std::runtime_error("OpenCL unavailable: " + lastError);
    }
    api->getPlatformIDs = loadOpenClSymbol<ClGetPlatformIDsFn>(*api, "clGetPlatformIDs");
    api->getDeviceIDs = loadOpenClSymbol<ClGetDeviceIDsFn>(*api, "clGetDeviceIDs");
    api->createContext = loadOpenClSymbol<ClCreateContextFn>(*api, "clCreateContext");
    api->releaseContext = loadOpenClSymbol<ClReleaseContextFn>(*api, "clReleaseContext");
    api->createCommandQueue = loadOpenClSymbol<ClCreateCommandQueueFn>(*api, "clCreateCommandQueue");
    api->releaseCommandQueue = loadOpenClSymbol<ClReleaseCommandQueueFn>(*api, "clReleaseCommandQueue");
    api->createProgramWithSource = loadOpenClSymbol<ClCreateProgramWithSourceFn>(*api, "clCreateProgramWithSource");
    api->buildProgram = loadOpenClSymbol<ClBuildProgramFn>(*api, "clBuildProgram");
    api->getProgramBuildInfo = loadOpenClSymbol<ClGetProgramBuildInfoFn>(*api, "clGetProgramBuildInfo");
    api->releaseProgram = loadOpenClSymbol<ClReleaseProgramFn>(*api, "clReleaseProgram");
    api->createKernel = loadOpenClSymbol<ClCreateKernelFn>(*api, "clCreateKernel");
    api->releaseKernel = loadOpenClSymbol<ClReleaseKernelFn>(*api, "clReleaseKernel");
    api->createBuffer = loadOpenClSymbol<ClCreateBufferFn>(*api, "clCreateBuffer");
    api->releaseMemObject = loadOpenClSymbol<ClReleaseMemObjectFn>(*api, "clReleaseMemObject");
    api->setKernelArg = loadOpenClSymbol<ClSetKernelArgFn>(*api, "clSetKernelArg");
    api->enqueueNDRangeKernel = loadOpenClSymbol<ClEnqueueNDRangeKernelFn>(*api, "clEnqueueNDRangeKernel");
    api->finish = loadOpenClSymbol<ClFinishFn>(*api, "clFinish");
    api->enqueueReadBuffer = loadOpenClSymbol<ClEnqueueReadBufferFn>(*api, "clEnqueueReadBuffer");
    return api;
}

std::string openClFusedUpsamplerBenchmark(int warmupRuns, int measuredRuns) {
    constexpr size_t inputElements = 512 * 16 * 32;
    constexpr size_t weightElements = 2048ull * 512ull * 3ull * 3ull;
    constexpr size_t biasElements = 2048;
    constexpr size_t outputElements = 512 * 32 * 64;

    auto api = loadOpenClApi();
    cl_uint platformCount = 0;
    cl_int status = api->getPlatformIDs(0, nullptr, &platformCount);
    if (status != CL_SUCCESS_VALUE || platformCount == 0) {
        throw std::runtime_error("clGetPlatformIDs failed status=" + std::to_string(status));
    }
    std::vector<cl_platform_id> platforms(platformCount);
    status = api->getPlatformIDs(platformCount, platforms.data(), nullptr);
    if (status != CL_SUCCESS_VALUE) {
        throw std::runtime_error("clGetPlatformIDs list failed status=" + std::to_string(status));
    }
    cl_device_id device = nullptr;
    for (cl_platform_id platform : platforms) {
        cl_uint deviceCount = 0;
        status = api->getDeviceIDs(platform, CL_DEVICE_TYPE_GPU_VALUE, 0, nullptr, &deviceCount);
        if (status == CL_SUCCESS_VALUE && deviceCount > 0) {
            std::vector<cl_device_id> devices(deviceCount);
            status = api->getDeviceIDs(platform, CL_DEVICE_TYPE_GPU_VALUE, deviceCount, devices.data(), nullptr);
            if (status == CL_SUCCESS_VALUE && !devices.empty()) {
                device = devices[0];
                break;
            }
        }
    }
    if (device == nullptr) {
        throw std::runtime_error("no OpenCL GPU device");
    }

    cl_int err = CL_SUCCESS_VALUE;
    cl_context context = api->createContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS_VALUE || context == nullptr) {
        throw std::runtime_error("clCreateContext failed status=" + std::to_string(err));
    }
    cl_command_queue queue = api->createCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS_VALUE || queue == nullptr) {
        api->releaseContext(context);
        throw std::runtime_error("clCreateCommandQueue failed status=" + std::to_string(err));
    }

    const char* source = R"CLC(
__kernel void fused_upsampler(
    __global const float* input,
    __global const float* weight,
    __global const float* bias,
    __global float* output) {
    const int gid = get_global_id(0);
    if (gid >= 512 * 32 * 64) return;
    const int ow = gid % 64;
    const int oh = (gid / 64) % 32;
    const int oc = gid / (32 * 64);
    const int ih_center = oh >> 1;
    const int iw_center = ow >> 1;
    const int phase = ((oh & 1) << 1) + (ow & 1);
    const int conv_oc = oc * 4 + phase;
    float sum = bias[conv_oc];
    for (int ic = 0; ic < 512; ++ic) {
        for (int kh = 0; kh < 3; ++kh) {
            const int ih = ih_center + kh - 1;
            if (ih < 0 || ih >= 16) continue;
            for (int kw = 0; kw < 3; ++kw) {
                const int iw = iw_center + kw - 1;
                if (iw < 0 || iw >= 32) continue;
                const int input_index = (ic * 16 + ih) * 32 + iw;
                const int weight_index = ((conv_oc * 512 + ic) * 3 + kh) * 3 + kw;
                sum += input[input_index] * weight[weight_index];
            }
        }
    }
    output[gid] = sum;
}
)CLC";
    const size_t sourceLength = std::strlen(source);
    cl_program program = api->createProgramWithSource(context, 1, &source, &sourceLength, &err);
    if (err != CL_SUCCESS_VALUE || program == nullptr) {
        api->releaseCommandQueue(queue);
        api->releaseContext(context);
        throw std::runtime_error("clCreateProgramWithSource failed status=" + std::to_string(err));
    }
    status = api->buildProgram(program, 1, &device, "", nullptr, nullptr);
    if (status != CL_SUCCESS_VALUE) {
        size_t logSize = 0;
        api->getProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG_VALUE, 0, nullptr, &logSize);
        std::string log(logSize, '\0');
        if (logSize > 0) {
            api->getProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG_VALUE, log.size(), log.data(), nullptr);
        }
        api->releaseProgram(program);
        api->releaseCommandQueue(queue);
        api->releaseContext(context);
        throw std::runtime_error("clBuildProgram failed status=" + std::to_string(status) + " log=" + log);
    }
    cl_kernel kernel = api->createKernel(program, "fused_upsampler", &err);
    if (err != CL_SUCCESS_VALUE || kernel == nullptr) {
        api->releaseProgram(program);
        api->releaseCommandQueue(queue);
        api->releaseContext(context);
        throw std::runtime_error("clCreateKernel failed status=" + std::to_string(err));
    }

    std::vector<float> input(inputElements);
    std::vector<float> weight(weightElements);
    std::vector<float> bias(biasElements);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>((i % 251) - 125) * 0.001f;
    }
    for (size_t i = 0; i < weight.size(); ++i) {
        weight[i] = static_cast<float>((i % 37) - 18) * 0.0001f;
    }
    for (size_t i = 0; i < bias.size(); ++i) {
        bias[i] = static_cast<float>((i % 31) - 15) * 0.0005f;
    }

    cl_mem inputBuffer = api->createBuffer(
        context, CL_MEM_READ_ONLY_VALUE | CL_MEM_COPY_HOST_PTR_VALUE,
        input.size() * sizeof(float), input.data(), &err);
    if (err != CL_SUCCESS_VALUE) throw std::runtime_error("clCreateBuffer input failed status=" + std::to_string(err));
    cl_mem weightBuffer = api->createBuffer(
        context, CL_MEM_READ_ONLY_VALUE | CL_MEM_COPY_HOST_PTR_VALUE,
        weight.size() * sizeof(float), weight.data(), &err);
    if (err != CL_SUCCESS_VALUE) throw std::runtime_error("clCreateBuffer weight failed status=" + std::to_string(err));
    cl_mem biasBuffer = api->createBuffer(
        context, CL_MEM_READ_ONLY_VALUE | CL_MEM_COPY_HOST_PTR_VALUE,
        bias.size() * sizeof(float), bias.data(), &err);
    if (err != CL_SUCCESS_VALUE) throw std::runtime_error("clCreateBuffer bias failed status=" + std::to_string(err));
    cl_mem outputBuffer = api->createBuffer(
        context, CL_MEM_WRITE_ONLY_VALUE, outputElements * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS_VALUE) throw std::runtime_error("clCreateBuffer output failed status=" + std::to_string(err));

    api->setKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    api->setKernelArg(kernel, 1, sizeof(cl_mem), &weightBuffer);
    api->setKernelArg(kernel, 2, sizeof(cl_mem), &biasBuffer);
    api->setKernelArg(kernel, 3, sizeof(cl_mem), &outputBuffer);

    const size_t global = outputElements;
    for (int i = 0; i < warmupRuns; ++i) {
        status = api->enqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
        if (status != CL_SUCCESS_VALUE) throw std::runtime_error("clEnqueueNDRangeKernel warmup failed status=" + std::to_string(status));
        api->finish(queue);
    }

    std::vector<long long> elapsed(static_cast<size_t>(measuredRuns));
    for (int i = 0; i < measuredRuns; ++i) {
        const auto started = std::chrono::steady_clock::now();
        status = api->enqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
        if (status != CL_SUCCESS_VALUE) throw std::runtime_error("clEnqueueNDRangeKernel failed status=" + std::to_string(status));
        api->finish(queue);
        const auto ended = std::chrono::steady_clock::now();
        elapsed[static_cast<size_t>(i)] =
            std::chrono::duration_cast<std::chrono::nanoseconds>(ended - started).count();
    }
    float checksum = 0.0f;
    status = api->enqueueReadBuffer(queue, outputBuffer, CL_TRUE_VALUE, 0, sizeof(float), &checksum, 0, nullptr, nullptr);
    if (status != CL_SUCCESS_VALUE) {
        checksum = -9999.0f;
    }

    api->releaseMemObject(outputBuffer);
    api->releaseMemObject(biasBuffer);
    api->releaseMemObject(weightBuffer);
    api->releaseMemObject(inputBuffer);
    api->releaseKernel(kernel);
    api->releaseProgram(program);
    api->releaseCommandQueue(queue);
    api->releaseContext(context);

    long long sum = 0;
    for (long long value : elapsed) sum += value;
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(3);
    out << "status=ok backend=opencl_gpu"
        << " warmup=" << warmupRuns
        << " measured=" << measuredRuns
        << " mean_ms=" << (static_cast<double>(sum) / measuredRuns / 1'000'000.0)
        << " input_floats=" << inputElements
        << " weight_mb=" << (static_cast<double>(weightElements * sizeof(float)) / (1024.0 * 1024.0))
        << " output_floats=" << outputElements
        << " checksum0=" << checksum;
    return out.str();
}

void vkCheck(VkResult result, const char* op) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(op) + " failed status=" + std::to_string(result));
    }
}

uint32_t findMemoryType(
    const VkPhysicalDeviceMemoryProperties& props,
    uint32_t typeBits,
    VkMemoryPropertyFlags flags) {
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) != 0 && (props.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    throw std::runtime_error("no matching Vulkan memory type");
}

struct VulkanBuffer {
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;

    void destroy() {
        if (buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
        }
        if (memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, memory, nullptr);
            memory = VK_NULL_HANDLE;
        }
    }

    ~VulkanBuffer() {
        destroy();
    }
};

VulkanBuffer createHostBuffer(
    VkDevice device,
    const VkPhysicalDeviceMemoryProperties& memoryProps,
    VkDeviceSize size,
    const void* initialData,
    VkBufferUsageFlags usage) {
    VulkanBuffer result;
    result.device = device;
    result.size = size;
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCheck(vkCreateBuffer(device, &bufferInfo, nullptr, &result.buffer), "vkCreateBuffer");

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device, result.buffer, &req);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = req.size;
    allocInfo.memoryTypeIndex = findMemoryType(
        memoryProps,
        req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkCheck(vkAllocateMemory(device, &allocInfo, nullptr, &result.memory), "vkAllocateMemory");
    vkCheck(vkBindBufferMemory(device, result.buffer, result.memory, 0), "vkBindBufferMemory");
    if (initialData != nullptr) {
        void* mapped = nullptr;
        vkCheck(vkMapMemory(device, result.memory, 0, size, 0, &mapped), "vkMapMemory");
        std::memcpy(mapped, initialData, static_cast<size_t>(size));
        vkUnmapMemory(device, result.memory);
    }
    return result;
}

std::string vulkanFusedUpsamplerBenchmark(int warmupRuns, int measuredRuns) {
    constexpr size_t inputElements = 512 * 16 * 32;
    constexpr size_t weightElements = 2048ull * 512ull * 3ull * 3ull;
    constexpr size_t biasElements = 2048;
    constexpr size_t outputElements = 512 * 32 * 64;

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "GVC_RT_FUSED_UPSAMPLER";
    appInfo.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;
    VkInstance instance = VK_NULL_HANDLE;
    vkCheck(vkCreateInstance(&instanceInfo, nullptr, &instance), "vkCreateInstance");

    uint32_t physicalCount = 0;
    vkCheck(vkEnumeratePhysicalDevices(instance, &physicalCount, nullptr), "vkEnumeratePhysicalDevices count");
    if (physicalCount == 0) {
        vkDestroyInstance(instance, nullptr);
        throw std::runtime_error("no Vulkan physical device");
    }
    std::vector<VkPhysicalDevice> physicalDevices(physicalCount);
    vkCheck(vkEnumeratePhysicalDevices(instance, &physicalCount, physicalDevices.data()), "vkEnumeratePhysicalDevices");
    VkPhysicalDevice physicalDevice = physicalDevices[0];

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
    uint32_t queueFamily = UINT32_MAX;
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
        if ((queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
            queueFamily = i;
            break;
        }
    }
    if (queueFamily == UINT32_MAX) {
        vkDestroyInstance(instance, nullptr);
        throw std::runtime_error("no Vulkan compute queue");
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;
    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    VkDevice device = VK_NULL_HANDLE;
    vkCheck(vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device), "vkCreateDevice");
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, queueFamily, 0, &queue);

    VkPhysicalDeviceMemoryProperties memoryProps{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProps);

    std::vector<float> input(inputElements);
    std::vector<float> weight(weightElements);
    std::vector<float> bias(biasElements);
    for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>((i % 251) - 125) * 0.001f;
    for (size_t i = 0; i < weight.size(); ++i) weight[i] = static_cast<float>((i % 37) - 18) * 0.0001f;
    for (size_t i = 0; i < bias.size(); ++i) bias[i] = static_cast<float>((i % 31) - 15) * 0.0005f;

    VulkanBuffer inputBuffer = createHostBuffer(device, memoryProps, input.size() * sizeof(float), input.data(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VulkanBuffer weightBuffer = createHostBuffer(device, memoryProps, weight.size() * sizeof(float), weight.data(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VulkanBuffer biasBuffer = createHostBuffer(device, memoryProps, bias.size() * sizeof(float), bias.data(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VulkanBuffer outputBuffer = createHostBuffer(device, memoryProps, outputElements * sizeof(float), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    VkDescriptorSetLayoutBinding bindings[4]{};
    for (uint32_t i = 0; i < 4; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo setLayoutInfo{};
    setLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    setLayoutInfo.bindingCount = 4;
    setLayoutInfo.pBindings = bindings;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    vkCheck(vkCreateDescriptorSetLayout(device, &setLayoutInfo, nullptr, &setLayout), "vkCreateDescriptorSetLayout");

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &setLayout;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    vkCheck(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout), "vkCreatePipelineLayout");

    VkShaderModuleCreateInfo shaderInfo{};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.codeSize = kFusedUpsamplerSpvSize;
    shaderInfo.pCode = kFusedUpsamplerSpv;
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    vkCheck(vkCreateShaderModule(device, &shaderInfo, nullptr, &shaderModule), "vkCreateShaderModule");

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = pipelineLayout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    vkCheck(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline), "vkCreateComputePipelines");

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 4;
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    vkCheck(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool), "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo setAlloc{};
    setAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setAlloc.descriptorPool = descriptorPool;
    setAlloc.descriptorSetCount = 1;
    setAlloc.pSetLayouts = &setLayout;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    vkCheck(vkAllocateDescriptorSets(device, &setAlloc, &descriptorSet), "vkAllocateDescriptorSets");

    VkDescriptorBufferInfo bufferInfos[4]{
        {inputBuffer.buffer, 0, inputBuffer.size},
        {weightBuffer.buffer, 0, weightBuffer.size},
        {biasBuffer.buffer, 0, biasBuffer.size},
        {outputBuffer.buffer, 0, outputBuffer.size},
    };
    VkWriteDescriptorSet writes[4]{};
    for (uint32_t i = 0; i < 4; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptorSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufferInfos[i];
    }
    vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);

    VkCommandPoolCreateInfo commandPoolInfo{};
    commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolInfo.queueFamilyIndex = queueFamily;
    commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    vkCheck(vkCreateCommandPool(device, &commandPoolInfo, nullptr, &commandPool), "vkCreateCommandPool");

    VkCommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAlloc.commandPool = commandPool;
    cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAlloc.commandBufferCount = 1;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    vkCheck(vkAllocateCommandBuffers(device, &cmdAlloc, &commandBuffer), "vkAllocateCommandBuffers");

    auto runOnce = [&]() {
        vkCheck(vkResetCommandBuffer(commandBuffer, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkCheck(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        vkCmdDispatch(commandBuffer, static_cast<uint32_t>((outputElements + 63) / 64), 1, 1);
        vkCheck(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        vkCheck(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE), "vkQueueSubmit");
        vkCheck(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
    };

    for (int i = 0; i < warmupRuns; ++i) runOnce();
    std::vector<long long> elapsed(static_cast<size_t>(measuredRuns));
    for (int i = 0; i < measuredRuns; ++i) {
        const auto started = std::chrono::steady_clock::now();
        runOnce();
        const auto ended = std::chrono::steady_clock::now();
        elapsed[static_cast<size_t>(i)] =
            std::chrono::duration_cast<std::chrono::nanoseconds>(ended - started).count();
    }

    float checksum = 0.0f;
    void* mapped = nullptr;
    vkCheck(vkMapMemory(device, outputBuffer.memory, 0, sizeof(float), 0, &mapped), "vkMapMemory output");
    std::memcpy(&checksum, mapped, sizeof(float));
    vkUnmapMemory(device, outputBuffer.memory);

    long long sum = 0;
    for (long long value : elapsed) sum += value;

    vkDeviceWaitIdle(device);
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyShaderModule(device, shaderModule, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
    outputBuffer.destroy();
    biasBuffer.destroy();
    weightBuffer.destroy();
    inputBuffer.destroy();
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(3);
    out << "status=ok backend=vulkan_compute"
        << " warmup=" << warmupRuns
        << " measured=" << measuredRuns
        << " mean_ms=" << (static_cast<double>(sum) / measuredRuns / 1'000'000.0)
        << " input_floats=" << inputElements
        << " weight_mb=" << (static_cast<double>(weightElements * sizeof(float)) / (1024.0 * 1024.0))
        << " output_floats=" << outputElements
        << " checksum0=" << checksum;
    return out.str();
}

std::vector<std::string> getStringArray(JNIEnv* env, jobjectArray array) {
    std::vector<std::string> values;
    const jsize count = env->GetArrayLength(array);
    values.reserve(static_cast<size_t>(count));
    for (jsize i = 0; i < count; ++i) {
        auto item = static_cast<jstring>(env->GetObjectArrayElement(array, i));
        const char* chars = env->GetStringUTFChars(item, nullptr);
        values.emplace_back(chars);
        env->ReleaseStringUTFChars(item, chars);
        env->DeleteLocalRef(item);
    }
    return values;
}

std::unique_ptr<Runtime> createPipelineRuntime(JNIEnv* env, const std::string& modelPath, const std::string& cacheDirPath) {
    auto runtime = std::make_unique<Runtime>();
    runtime->accelerationMode = 1;
    runtime->api = loadApi(env);
    std::vector<std::string> optionStatuses;
    void* options = nullptr;
    check(env, runtime->api->optionsCreate(&options), "ANeuroPilotTFLiteOptions_create");
    appendOptionStatus(optionStatuses, "accelerationMode", runtime->api->optionsSetAccelerationMode(options, 1));
    appendOptionStatus(optionStatuses, "allowFp16", runtime->api->optionsSetAllowFp16(options, true));
    applyOptionalOption(optionStatuses, "preferenceSustainedSpeed", runtime->api->optionsSetPreference, options, 2);
    applyOptionalOption(optionStatuses, "priorityHigh", runtime->api->optionsSetExecutionPriority, options, 110);
    applyOptionalOption(optionStatuses, "lowLatency", runtime->api->optionsSetLowLatency, options, true);
    applyOptionalOption(optionStatuses, "deepFusion", runtime->api->optionsSetDeepFusion, options, true);
    applyOptionalOption(optionStatuses, "boostHint", runtime->api->optionsSetBoostHint, options, 100);
    applyOptionalOption(optionStatuses, "boostDuration", runtime->api->optionsSetBoostDuration, options, 3000);
    applyOptionalOption(optionStatuses, "maxDelegatedPartitions", runtime->api->optionsSetMaxNumberDelegatedPartitions, options, 32);
    applyOptionalOption(optionStatuses, "disallowNnApiCpu", runtime->api->optionsSetDisallowNnApiCpu, options, true);
    if (!cacheDirPath.empty() && runtime->api->optionsSetCacheDir != nullptr) {
        appendOptionStatus(optionStatuses, "cacheDir", runtime->api->optionsSetCacheDir(options, cacheDirPath.c_str()));
    }
    runtime->optionsSummary = joinStatuses(optionStatuses);
    const int status = runtime->api->createAdv(&runtime->tflite, modelPath.c_str(), options);
    runtime->api->optionsFree(options);
    check(env, status, "ANeuroPilotTFLite_createAdv pipeline");
    runtime->inputSizes = querySizes(env, *runtime, kInputBuffer);
    runtime->outputSizes = querySizes(env, *runtime, kOutputBuffer);
    return runtime;
}

std::vector<float> runPipelineTflite(JNIEnv* env, Runtime& runtime, const std::vector<const std::vector<float>*>& inputs) {
    if (inputs.size() != runtime.inputSizes.size()) {
        fail(env, "pipeline input count mismatch");
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        const size_t bytes = inputs[i]->size() * sizeof(float);
        if (bytes != runtime.inputSizes[i]) {
            fail(env, "pipeline input bytes mismatch");
        }
        check(
            env,
            runtime.api->setTensorBuffer(runtime.tflite, static_cast<int>(i), reinterpret_cast<char*>(const_cast<float*>(inputs[i]->data()))),
            "ANeuroPilotTFLite_setTensorBuffer pipeline");
    }
    check(env, runtime.api->invoke(runtime.tflite), "ANeuroPilotTFLite_invoke pipeline");
    if (runtime.outputSizes.size() != 1) {
        fail(env, "pipeline expects single-output tflite segment");
    }
    std::vector<float> output(runtime.outputSizes[0] / sizeof(float));
    check(
        env,
        runtime.api->getOutputTensorData(runtime.tflite, 0, output.data(), runtime.outputSizes[0]),
        "ANeuroPilotTFLite_getOutputTensorData pipeline");
    return output;
}

void siluInPlace(std::vector<float>& values) {
    for (float& value : values) {
        value = value / (1.0f + std::exp(-value));
    }
}

using Clock = std::chrono::steady_clock;

template <typename Fn>
void measurePipelineStage(std::vector<long long>& timings, size_t index, Fn&& fn) {
    const auto started = Clock::now();
    fn();
    const auto ended = Clock::now();
    timings[index] += std::chrono::duration_cast<std::chrono::nanoseconds>(ended - started).count();
}

struct PReconPipelineOutputs {
    std::vector<float> feature;
    std::vector<float> frame;
};

PReconPipelineOutputs runPReconPipelineOnceWithInputs(
    JNIEnv* env,
    std::vector<std::unique_ptr<Runtime>>& runtimes,
    const std::vector<std::vector<float>>& adaWeights,
    const std::vector<float>& pYHat,
    const std::vector<float>& pCtx,
    std::vector<long long>& timings,
    std::vector<std::vector<float>>* trace = nullptr) {
    constexpr size_t pYHatElements = 128 * 16 * 32;
    constexpr size_t pCtxElements = 256 * 32 * 64;
    constexpr size_t codewordElements = 18 * 16 * 32;
    if (pYHat.size() != pYHatElements) {
        fail(env, "native p recon p_y_hat size mismatch");
    }
    if (pCtx.size() != pCtxElements) {
        fail(env, "native p recon p_ctx size mismatch");
    }

    std::vector<float> feature;
    measurePipelineStage(timings, 0, [&] {
        feature = runPipelineTflite(env, *runtimes[0], {&pYHat, &pCtx});
    });
    if (trace != nullptr) trace->push_back(feature);

    std::vector<float> unshuffled(1024 * 16 * 32);
    measurePipelineStage(timings, 1, [&] {
        pixelUnshuffle2Nchw256x32x64(feature.data(), unshuffled.data());
    });
    if (trace != nullptr) trace->push_back(unshuffled);

    std::vector<float> mlpNorm0(unshuffled.size());
    measurePipelineStage(timings, 2, [&] {
        nativeGroupNormOptimized(unshuffled.data(), mlpNorm0.data(), 1024, 16 * 32, 1);
    });
    if (trace != nullptr) trace->push_back(mlpNorm0);

    std::vector<float> dcb0;
    measurePipelineStage(timings, 3, [&] {
        dcb0 = runPipelineTflite(env, *runtimes[1], {&mlpNorm0});
    });
    if (trace != nullptr) trace->push_back(dcb0);

    std::vector<float> mlpNorm1(dcb0.size());
    measurePipelineStage(timings, 4, [&] {
        nativeGroupNormOptimized(dcb0.data(), mlpNorm1.data(), 256, 16 * 32, 1);
        siluInPlace(mlpNorm1);
    });
    if (trace != nullptr) trace->push_back(mlpNorm1);

    std::vector<float> codeword;
    measurePipelineStage(timings, 5, [&] {
        codeword = runPipelineTflite(env, *runtimes[2], {&mlpNorm1});
    });
    if (codeword.size() != codewordElements) {
        fail(env, "pipeline codeword size mismatch");
    }
    if (trace != nullptr) trace->push_back(codeword);

    std::vector<float> stage1;
    measurePipelineStage(timings, 6, [&] {
        stage1 = runPipelineTflite(env, *runtimes[3], {&codeword});
    });
    if (trace != nullptr) trace->push_back(stage1);
    std::vector<float> adagn(stage1.size());
    measurePipelineStage(timings, 7, [&] {
        nativeAdaGnNchw(
            stage1.data(), codeword.data(), adaWeights[0].data(), adaWeights[0].data() + 512 * 18,
            adaWeights[0].data() + 512 * 18 + 512,
            adaWeights[0].data() + 512 * 18 + 512 + 512 * 18,
            adagn.data(), stage1.data(), 512, 16 * 32, 1);
    });
    if (trace != nullptr) trace->push_back(stage1);

    std::vector<float> stage2;
    measurePipelineStage(timings, 8, [&] {
        stage2 = runPipelineTflite(env, *runtimes[4], {&stage1});
    });
    if (trace != nullptr) trace->push_back(stage2);
    measurePipelineStage(timings, 9, [&] {
        nativeAdaGnNchw(
            stage2.data(), codeword.data(), adaWeights[1].data(), adaWeights[1].data() + 512 * 18,
            adaWeights[1].data() + 512 * 18 + 512,
            adaWeights[1].data() + 512 * 18 + 512 + 512 * 18,
            adagn.data(), stage2.data(), 512, 16 * 32, 1);
    });
    if (trace != nullptr) trace->push_back(stage2);

    std::vector<float> upsampled;
    measurePipelineStage(timings, 10, [&] {
        upsampled = runPipelineTflite(env, *runtimes[5], {&stage2});
    });
    if (trace != nullptr) trace->push_back(upsampled);
    std::vector<float> adagn3(upsampled.size());
    measurePipelineStage(timings, 11, [&] {
        nativeAdaGnNchw(
            upsampled.data(), codeword.data(), adaWeights[2].data(), adaWeights[2].data() + 512 * 18,
            adaWeights[2].data() + 512 * 18 + 512,
            adaWeights[2].data() + 512 * 18 + 512 + 512 * 18,
            adagn3.data(), upsampled.data(), 512, 32 * 64, 1);
    });
    if (trace != nullptr) trace->push_back(upsampled);

    std::vector<float> stage3;
    measurePipelineStage(timings, 12, [&] {
        stage3 = runPipelineTflite(env, *runtimes[6], {&upsampled});
    });
    if (trace != nullptr) trace->push_back(stage3);
    std::vector<float> adagn4(stage3.size());
    measurePipelineStage(timings, 13, [&] {
        nativeAdaGnNchw(
            stage3.data(), codeword.data(), adaWeights[3].data(), adaWeights[3].data() + 320 * 18,
            adaWeights[3].data() + 320 * 18 + 320,
            adaWeights[3].data() + 320 * 18 + 320 + 320 * 18,
            adagn4.data(), stage3.data(), 320, 32 * 64, 1);
    });
    if (trace != nullptr) trace->push_back(stage3);

    std::vector<float> stage4;
    measurePipelineStage(timings, 14, [&] {
        stage4 = runPipelineTflite(env, *runtimes[7], {&stage3});
    });
    if (trace != nullptr) trace->push_back(stage4);
    std::vector<float> finalAdagn(stage4.size());
    measurePipelineStage(timings, 15, [&] {
        nativeAdaGnNchw(
            stage4.data(), codeword.data(), adaWeights[4].data(), adaWeights[4].data() + 320 * 18,
            adaWeights[4].data() + 320 * 18 + 320,
            adaWeights[4].data() + 320 * 18 + 320 + 320 * 18,
            finalAdagn.data(), stage4.data(), 320, 32 * 64, 1);
    });
    if (trace != nullptr) trace->push_back(stage4);

    std::vector<float> frame;
    measurePipelineStage(timings, 16, [&] {
        frame = runPipelineTflite(env, *runtimes[8], {&stage4});
    });
    if (trace != nullptr) trace->push_back(frame);
    return PReconPipelineOutputs{std::move(feature), std::move(frame)};
}

std::vector<float> runPReconPipelineOnce(
    JNIEnv* env,
    std::vector<std::unique_ptr<Runtime>>& runtimes,
    const std::vector<std::vector<float>>& adaWeights,
    std::vector<long long>& timings) {
    constexpr size_t pYHatElements = 128 * 16 * 32;
    constexpr size_t pCtxElements = 256 * 32 * 64;

    std::vector<float> pYHat(pYHatElements);
    std::vector<float> pCtx(pCtxElements);
    for (size_t i = 0; i < pYHat.size(); ++i) {
        pYHat[i] = static_cast<float>((i % 127) - 63) * 0.001f;
    }
    for (size_t i = 0; i < pCtx.size(); ++i) {
        pCtx[i] = static_cast<float>((i % 251) - 125) * 0.0005f;
    }

    return runPReconPipelineOnceWithInputs(env, runtimes, adaWeights, pYHat, pCtx, timings).frame;
}

std::vector<float> runPReconMixedMergedOnce(
    JNIEnv* env,
    std::vector<std::unique_ptr<Runtime>>& runtimes,
    const std::vector<std::vector<float>>& adaWeights,
    std::vector<long long>& timings) {
    constexpr size_t pYHatElements = 128 * 16 * 32;
    constexpr size_t pCtxElements = 256 * 32 * 64;
    constexpr size_t codewordElements = 18 * 16 * 32;

    std::vector<float> pYHat(pYHatElements);
    std::vector<float> pCtx(pCtxElements);
    for (size_t i = 0; i < pYHat.size(); ++i) {
        pYHat[i] = static_cast<float>((i % 127) - 63) * 0.001f;
    }
    for (size_t i = 0; i < pCtx.size(); ++i) {
        pCtx[i] = static_cast<float>((i % 251) - 125) * 0.0005f;
    }

    std::vector<float> feature;
    measurePipelineStage(timings, 0, [&] {
        feature = runPipelineTflite(env, *runtimes[0], {&pYHat, &pCtx});
    });

    std::vector<float> unshuffled(1024 * 16 * 32);
    measurePipelineStage(timings, 1, [&] {
        pixelUnshuffle2Nchw256x32x64(feature.data(), unshuffled.data());
    });

    std::vector<float> mlpNorm0(unshuffled.size());
    measurePipelineStage(timings, 2, [&] {
        nativeGroupNormOptimized(unshuffled.data(), mlpNorm0.data(), 1024, 16 * 32, 1);
    });

    std::vector<float> dcb0;
    measurePipelineStage(timings, 3, [&] {
        dcb0 = runPipelineTflite(env, *runtimes[1], {&mlpNorm0});
    });

    std::vector<float> mlpNorm1(dcb0.size());
    measurePipelineStage(timings, 4, [&] {
        nativeGroupNormOptimized(dcb0.data(), mlpNorm1.data(), 256, 16 * 32, 1);
        siluInPlace(mlpNorm1);
    });

    std::vector<float> codeword;
    measurePipelineStage(timings, 5, [&] {
        codeword = runPipelineTflite(env, *runtimes[2], {&mlpNorm1});
    });
    if (codeword.size() != codewordElements) {
        fail(env, "mixed pipeline codeword size mismatch");
    }

    std::vector<float> stage2;
    measurePipelineStage(timings, 6, [&] {
        stage2 = runPipelineTflite(env, *runtimes[3], {&codeword});
    });

    std::vector<float> adagn2(stage2.size());
    measurePipelineStage(timings, 7, [&] {
        nativeAdaGnNchw(
            stage2.data(), codeword.data(), adaWeights[1].data(), adaWeights[1].data() + 512 * 18,
            adaWeights[1].data() + 512 * 18 + 512,
            adaWeights[1].data() + 512 * 18 + 512 + 512 * 18,
            adagn2.data(), stage2.data(), 512, 16 * 32, 1);
    });

    std::vector<float> stage3;
    measurePipelineStage(timings, 8, [&] {
        stage3 = runPipelineTflite(env, *runtimes[4], {&stage2});
    });

    std::vector<float> adagn4(stage3.size());
    measurePipelineStage(timings, 9, [&] {
        nativeAdaGnNchw(
            stage3.data(), codeword.data(), adaWeights[3].data(), adaWeights[3].data() + 320 * 18,
            adaWeights[3].data() + 320 * 18 + 320,
            adaWeights[3].data() + 320 * 18 + 320 + 320 * 18,
            adagn4.data(), stage3.data(), 320, 32 * 64, 1);
    });

    std::vector<float> frame;
    measurePipelineStage(timings, 10, [&] {
        frame = runPipelineTflite(env, *runtimes[5], {&stage3});
    });
    return frame;
}

}  // namespace

void pixelUnshuffle2Nchw256x32x64(const float* input, float* output) {
    constexpr int channels = 256;
    constexpr int inputH = 32;
    constexpr int inputW = 64;
    constexpr int outputH = inputH / 2;
    constexpr int outputW = inputW / 2;
    constexpr int outputPlane = outputH * outputW;
    constexpr int inputPlane = inputH * inputW;

    for (int c = 0; c < channels; ++c) {
        const float* inputChannel = input + c * inputPlane;
        float* outputChannel = output + c * 4 * outputPlane;
        for (int oh = 0; oh < outputH; ++oh) {
            const float* row0 = inputChannel + (oh * 2) * inputW;
            const float* row1 = row0 + inputW;
            float* out00 = outputChannel + 0 * outputPlane + oh * outputW;
            float* out01 = outputChannel + 1 * outputPlane + oh * outputW;
            float* out10 = outputChannel + 2 * outputPlane + oh * outputW;
            float* out11 = outputChannel + 3 * outputPlane + oh * outputW;
            for (int ow = 0; ow < outputW; ++ow) {
                const int iw = ow * 2;
                out00[ow] = row0[iw];
                out01[ow] = row0[iw + 1];
                out10[ow] = row1[iw];
                out11[ow] = row1[iw + 1];
            }
        }
    }
}

void depthToSpace2Nchw2048x16x32(const float* input, float* output) {
    constexpr int outputChannels = 512;
    constexpr int inputH = 16;
    constexpr int inputW = 32;
    constexpr int outputH = inputH * 2;
    constexpr int outputW = inputW * 2;
    constexpr int inputPlane = inputH * inputW;
    constexpr int outputPlane = outputH * outputW;

    for (int c = 0; c < outputChannels; ++c) {
        const float* in00 = input + (c * 4 + 0) * inputPlane;
        const float* in01 = input + (c * 4 + 1) * inputPlane;
        const float* in10 = input + (c * 4 + 2) * inputPlane;
        const float* in11 = input + (c * 4 + 3) * inputPlane;
        float* outputChannel = output + c * outputPlane;
        for (int ih = 0; ih < inputH; ++ih) {
            float* row0 = outputChannel + (ih * 2) * outputW;
            float* row1 = row0 + outputW;
            const int inputRow = ih * inputW;
            for (int iw = 0; iw < inputW; ++iw) {
                const int outputCol = iw * 2;
                row0[outputCol] = in00[inputRow + iw];
                row0[outputCol + 1] = in01[inputRow + iw];
                row1[outputCol] = in10[inputRow + iw];
                row1[outputCol + 1] = in11[inputRow + iw];
            }
        }
    }
}

void wsiluChunkAdd1280x32x64(const float* input, float* output) {
    constexpr int outputChannels = 640;
    constexpr int spatial = 32 * 64;
    constexpr int planeElements = outputChannels * spatial;
    const float* secondHalf = input + planeElements;
    for (int i = 0; i < planeElements; ++i) {
        const float a = input[i];
        const float b = secondHalf[i];
        output[i] = a / (1.0f + std::exp(-4.0f * a)) + b / (1.0f + std::exp(-4.0f * b));
    }
}

inline float hardSigmoid4x(float x) {
    const float v = 0.5f + 0.8f * x;
    return std::min(1.0f, std::max(0.0f, v));
}

void fastWSiLUChunkAdd1280x32x64(const float* input, float* output) {
    constexpr int outputChannels = 640;
    constexpr int spatial = 32 * 64;
    constexpr int planeElements = outputChannels * spatial;
    const float* secondHalf = input + planeElements;
    for (int i = 0; i < planeElements; ++i) {
        const float a = input[i];
        const float b = secondHalf[i];
        output[i] = a * hardSigmoid4x(a) + b * hardSigmoid4x(b);
    }
}

std::vector<float> readFloatsFromFile(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open " + path);
    }
    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    input.seekg(0, std::ios::beg);
    if (size <= 0 || static_cast<size_t>(size) % sizeof(float) != 0) {
        throw std::runtime_error("invalid float file size " + path);
    }
    std::vector<float> values(static_cast<size_t>(size) / sizeof(float));
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(size));
    if (!input) {
        throw std::runtime_error("failed to read " + path);
    }
    return values;
}

void fusedPixelUnshuffle2Adaptor1024To256(
    const float* input,
    const float* weight,
    const float* bias,
    float* output) {
    constexpr int inputChannels = 256;
    constexpr int outputChannels = 256;
    constexpr int outputH = 16;
    constexpr int outputW = 32;
    constexpr int outputPixels = outputH * outputW;
    constexpr int inputW = 64;

    for (int oc = 0; oc < outputChannels; ++oc) {
        const float* weightOc = weight + oc * inputChannels * 4;
        float* outputOc = output + oc * outputPixels;
        for (int oh = 0; oh < outputH; ++oh) {
            for (int ow = 0; ow < outputW; ++ow) {
                float sum = bias[oc];
                const int baseH = oh * 2;
                const int baseW = ow * 2;
                for (int ic = 0; ic < inputChannels; ++ic) {
                    const float* inputChannel = input + ic * 32 * 64 + baseH * inputW + baseW;
                    const float* weightIc = weightOc + ic * 4;
                    sum += inputChannel[0] * weightIc[0];
                    sum += inputChannel[1] * weightIc[1];
                    sum += inputChannel[inputW] * weightIc[2];
                    sum += inputChannel[inputW + 1] * weightIc[3];
                }
                outputOc[oh * outputW + ow] = sum;
            }
        }
    }
}

std::vector<float> transposeAdaptorWeightToKc(const float* weightOcK) {
    constexpr int inputChannels = 1024;
    constexpr int outputChannels = 256;
    std::vector<float> weightKc(inputChannels * outputChannels);
    for (int oc = 0; oc < outputChannels; ++oc) {
        for (int k = 0; k < inputChannels; ++k) {
            weightKc[k * outputChannels + oc] = weightOcK[oc * inputChannels + k];
        }
    }
    return weightKc;
}

void adaptor1024To256PixelMajor(
    const float* unshuffled,
    const float* weightKc,
    const float* bias,
    float* output,
    int pixelBegin,
    int pixelEnd) {
    constexpr int inputChannels = 1024;
    constexpr int outputChannels = 256;
    constexpr int pixels = 16 * 32;
    alignas(16) float acc[outputChannels];
    for (int p = pixelBegin; p < pixelEnd; ++p) {
        std::memcpy(acc, bias, sizeof(acc));
        for (int k = 0; k < inputChannels; ++k) {
            const float x = unshuffled[k * pixels + p];
            const float* weight = weightKc + k * outputChannels;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
            const float32x4_t xv = vdupq_n_f32(x);
            for (int oc = 0; oc < outputChannels; oc += 4) {
                float32x4_t av = vld1q_f32(acc + oc);
                const float32x4_t wv = vld1q_f32(weight + oc);
                av = vmlaq_f32(av, xv, wv);
                vst1q_f32(acc + oc, av);
            }
#else
            for (int oc = 0; oc < outputChannels; ++oc) {
                acc[oc] += x * weight[oc];
            }
#endif
        }
        for (int oc = 0; oc < outputChannels; ++oc) {
            output[oc * pixels + p] = acc[oc];
        }
    }
}

void adaptor1024To256OutputChannelMajor(
    const float* unshuffled,
    const float* weight,
    const float* bias,
    float* output,
    int outputChannelBegin,
    int outputChannelEnd) {
    constexpr int inputChannels = 1024;
    constexpr int pixels = 16 * 32;
    for (int oc = outputChannelBegin; oc < outputChannelEnd; ++oc) {
        float* outputOc = output + oc * pixels;
        const float* weightOc = weight + oc * inputChannels;
        for (int p = 0; p < pixels; ++p) {
            outputOc[p] = bias[oc];
        }
        for (int k = 0; k < inputChannels; ++k) {
            const float* inputK = unshuffled + k * pixels;
            const float w = weightOc[k];
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
            const float32x4_t wv = vdupq_n_f32(w);
            for (int p = 0; p < pixels; p += 4) {
                float32x4_t ov = vld1q_f32(outputOc + p);
                const float32x4_t xv = vld1q_f32(inputK + p);
                ov = vmlaq_f32(ov, xv, wv);
                vst1q_f32(outputOc + p, ov);
            }
#else
            for (int p = 0; p < pixels; ++p) {
                outputOc[p] += inputK[p] * w;
            }
#endif
        }
    }
}

void fusedPixelUnshuffle2AdaptorOptimized(
    const float* input,
    const float* weight,
    const float* bias,
    float* unshuffled,
    float* output,
    int requestedThreadCount) {
    constexpr int outputChannels = 256;
    pixelUnshuffle2Nchw256x32x64(input, unshuffled);
    const unsigned int hardwareThreads = std::thread::hardware_concurrency();
    const unsigned int maxThreads = std::min<unsigned int>(std::max(1u, hardwareThreads), 8u);
    const int threadCount = std::max(1, std::min(requestedThreadCount, static_cast<int>(maxThreads)));
    if (threadCount == 1) {
        adaptor1024To256OutputChannelMajor(unshuffled, weight, bias, output, 0, outputChannels);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(threadCount));
    for (int threadIndex = 0; threadIndex < threadCount; ++threadIndex) {
        const int begin = outputChannels * threadIndex / threadCount;
        const int end = outputChannels * (threadIndex + 1) / threadCount;
        threads.emplace_back(adaptor1024To256OutputChannelMajor, unshuffled, weight, bias, output, begin, end);
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

void groupNorm512x16x32(
    const float* input,
    float* output,
    int groupBegin,
    int groupEnd) {
    constexpr int channels = 512;
    constexpr int groups = 32;
    constexpr int channelsPerGroup = channels / groups;
    constexpr int spatial = 16 * 32;
    constexpr int groupElements = channelsPerGroup * spatial;
    constexpr float eps = 1.0e-6f;

    for (int group = groupBegin; group < groupEnd; ++group) {
        const int channelBegin = group * channelsPerGroup;
        double sum = 0.0;
        double sumSq = 0.0;
        for (int c = 0; c < channelsPerGroup; ++c) {
            const float* inputChannel = input + (channelBegin + c) * spatial;
            for (int i = 0; i < spatial; ++i) {
                const float value = inputChannel[i];
                sum += value;
                sumSq += static_cast<double>(value) * static_cast<double>(value);
            }
        }
        const double mean = sum / groupElements;
        const double variance = std::max(0.0, sumSq / groupElements - mean * mean);
        const float invStd = 1.0f / std::sqrt(static_cast<float>(variance) + eps);
        for (int c = 0; c < channelsPerGroup; ++c) {
            const float* inputChannel = input + (channelBegin + c) * spatial;
            float* outputChannel = output + (channelBegin + c) * spatial;
            for (int i = 0; i < spatial; ++i) {
                outputChannel[i] = (inputChannel[i] - static_cast<float>(mean)) * invStd;
            }
        }
    }
}

void nativeGroupNorm512Optimized(const float* input, float* output, int requestedThreadCount) {
    constexpr int groups = 32;
    const unsigned int hardwareThreads = std::thread::hardware_concurrency();
    const unsigned int maxThreads = std::min<unsigned int>(std::max(1u, hardwareThreads), 8u);
    const int threadCount = std::max(1, std::min(requestedThreadCount, static_cast<int>(maxThreads)));
    if (threadCount == 1) {
        groupNorm512x16x32(input, output, 0, groups);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(threadCount));
    for (int threadIndex = 0; threadIndex < threadCount; ++threadIndex) {
        const int begin = groups * threadIndex / threadCount;
        const int end = groups * (threadIndex + 1) / threadCount;
        threads.emplace_back(groupNorm512x16x32, input, output, begin, end);
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

void linear18To512(
    const float* input,
    const float* weight,
    const float* bias,
    float* output) {
    constexpr int inputChannels = 18;
    constexpr int outputChannels = 512;
    for (int oc = 0; oc < outputChannels; ++oc) {
        const float* weightOc = weight + oc * inputChannels;
        float sum = bias[oc];
        for (int ic = 0; ic < inputChannels; ++ic) {
            sum += input[ic] * weightOc[ic];
        }
        output[oc] = sum;
    }
}

void codewordMeanStd18x16x32(const float* codeword, float* mean, float* stddev) {
    constexpr int channels = 18;
    constexpr int spatial = 16 * 32;
    constexpr float eps = 1.0e-6f;
    for (int c = 0; c < channels; ++c) {
        const float* channel = codeword + c * spatial;
        double sum = 0.0;
        double sumSq = 0.0;
        for (int i = 0; i < spatial; ++i) {
            const float value = channel[i];
            sum += value;
            sumSq += static_cast<double>(value) * static_cast<double>(value);
        }
        const double m = sum / spatial;
        const double variance = std::max(0.0, sumSq / spatial - m * m);
        mean[c] = static_cast<float>(m);
        stddev[c] = std::sqrt(static_cast<float>(variance) + eps);
    }
}

void applyAdaGn512(
    const float* normalized,
    const float* scale,
    const float* bias,
    float* output,
    int channelBegin,
    int channelEnd) {
    constexpr int spatial = 16 * 32;
    for (int c = channelBegin; c < channelEnd; ++c) {
        const float* in = normalized + c * spatial;
        float* out = output + c * spatial;
        const float s = scale[c];
        const float b = bias[c];
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        const float32x4_t sv = vdupq_n_f32(s);
        const float32x4_t bv = vdupq_n_f32(b);
        for (int i = 0; i < spatial; i += 4) {
            const float32x4_t xv = vld1q_f32(in + i);
            vst1q_f32(out + i, vmlaq_f32(bv, xv, sv));
        }
#else
        for (int i = 0; i < spatial; ++i) {
            out[i] = s * in[i] + b;
        }
#endif
    }
}

void nativeAdaGn512Stage1(
    const float* feature,
    const float* codeword,
    const float* gammaWeight,
    const float* gammaBias,
    const float* betaWeight,
    const float* betaBias,
    float* normalized,
    float* output,
    int requestedThreadCount) {
    constexpr int channels = 512;
    nativeGroupNorm512Optimized(feature, normalized, 1);
    float codewordMean[18];
    float codewordStd[18];
    float scale[channels];
    float bias[channels];
    codewordMeanStd18x16x32(codeword, codewordMean, codewordStd);
    linear18To512(codewordStd, gammaWeight, gammaBias, scale);
    linear18To512(codewordMean, betaWeight, betaBias, bias);

    const unsigned int hardwareThreads = std::thread::hardware_concurrency();
    const unsigned int maxThreads = std::min<unsigned int>(std::max(1u, hardwareThreads), 8u);
    const int threadCount = std::max(1, std::min(requestedThreadCount, static_cast<int>(maxThreads)));
    if (threadCount == 1) {
        applyAdaGn512(normalized, scale, bias, output, 0, channels);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(threadCount));
    for (int threadIndex = 0; threadIndex < threadCount; ++threadIndex) {
        const int begin = channels * threadIndex / threadCount;
        const int end = channels * (threadIndex + 1) / threadCount;
        threads.emplace_back(applyAdaGn512, normalized, scale, bias, output, begin, end);
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

void groupNormNchw(
    const float* input,
    float* output,
    int channels,
    int spatial,
    int groupBegin,
    int groupEnd) {
    constexpr int groups = 32;
    constexpr float eps = 1.0e-6f;
    const int channelsPerGroup = channels / groups;
    const int groupElements = channelsPerGroup * spatial;
    for (int group = groupBegin; group < groupEnd; ++group) {
        const int channelBegin = group * channelsPerGroup;
        double sum = 0.0;
        double sumSq = 0.0;
        for (int c = 0; c < channelsPerGroup; ++c) {
            const float* inputChannel = input + (channelBegin + c) * spatial;
            for (int i = 0; i < spatial; ++i) {
                const float value = inputChannel[i];
                sum += value;
                sumSq += static_cast<double>(value) * static_cast<double>(value);
            }
        }
        const double mean = sum / groupElements;
        const double variance = std::max(0.0, sumSq / groupElements - mean * mean);
        const float invStd = 1.0f / std::sqrt(static_cast<float>(variance) + eps);
        for (int c = 0; c < channelsPerGroup; ++c) {
            const float* inputChannel = input + (channelBegin + c) * spatial;
            float* outputChannel = output + (channelBegin + c) * spatial;
            for (int i = 0; i < spatial; ++i) {
                outputChannel[i] = (inputChannel[i] - static_cast<float>(mean)) * invStd;
            }
        }
    }
}

void nativeGroupNormOptimized(const float* input, float* output, int channels, int spatial, int requestedThreadCount) {
    constexpr int groups = 32;
    const unsigned int hardwareThreads = std::thread::hardware_concurrency();
    const unsigned int maxThreads = std::min<unsigned int>(std::max(1u, hardwareThreads), 8u);
    const int threadCount = std::max(1, std::min(requestedThreadCount, static_cast<int>(maxThreads)));
    if (threadCount == 1) {
        groupNormNchw(input, output, channels, spatial, 0, groups);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(threadCount));
    for (int threadIndex = 0; threadIndex < threadCount; ++threadIndex) {
        const int begin = groups * threadIndex / threadCount;
        const int end = groups * (threadIndex + 1) / threadCount;
        threads.emplace_back(groupNormNchw, input, output, channels, spatial, begin, end);
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

void linear18ToN(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int outputChannels) {
    constexpr int inputChannels = 18;
    for (int oc = 0; oc < outputChannels; ++oc) {
        const float* weightOc = weight + oc * inputChannels;
        float sum = bias[oc];
        for (int ic = 0; ic < inputChannels; ++ic) {
            sum += input[ic] * weightOc[ic];
        }
        output[oc] = sum;
    }
}

void applyAdaGnNchw(
    const float* normalized,
    const float* scale,
    const float* bias,
    float* output,
    int spatial,
    int channelBegin,
    int channelEnd) {
    for (int c = channelBegin; c < channelEnd; ++c) {
        const float* in = normalized + c * spatial;
        float* out = output + c * spatial;
        const float s = scale[c];
        const float b = bias[c];
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        const float32x4_t sv = vdupq_n_f32(s);
        const float32x4_t bv = vdupq_n_f32(b);
        int i = 0;
        for (; i + 3 < spatial; i += 4) {
            const float32x4_t xv = vld1q_f32(in + i);
            vst1q_f32(out + i, vmlaq_f32(bv, xv, sv));
        }
        for (; i < spatial; ++i) {
            out[i] = s * in[i] + b;
        }
#else
        for (int i = 0; i < spatial; ++i) {
            out[i] = s * in[i] + b;
        }
#endif
    }
}

void nativeAdaGnNchw(
    const float* feature,
    const float* codeword,
    const float* gammaWeight,
    const float* gammaBias,
    const float* betaWeight,
    const float* betaBias,
    float* normalized,
    float* output,
    int channels,
    int spatial,
    int requestedThreadCount) {
    nativeGroupNormOptimized(feature, normalized, channels, spatial, 1);
    float codewordMean[18];
    float codewordStd[18];
    std::vector<float> scale(static_cast<size_t>(channels));
    std::vector<float> bias(static_cast<size_t>(channels));
    codewordMeanStd18x16x32(codeword, codewordMean, codewordStd);
    linear18ToN(codewordStd, gammaWeight, gammaBias, scale.data(), channels);
    linear18ToN(codewordMean, betaWeight, betaBias, bias.data(), channels);

    const unsigned int hardwareThreads = std::thread::hardware_concurrency();
    const unsigned int maxThreads = std::min<unsigned int>(std::max(1u, hardwareThreads), 8u);
    const int threadCount = std::max(1, std::min(requestedThreadCount, static_cast<int>(maxThreads)));
    if (threadCount == 1) {
        applyAdaGnNchw(normalized, scale.data(), bias.data(), output, spatial, 0, channels);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(threadCount));
    for (int threadIndex = 0; threadIndex < threadCount; ++threadIndex) {
        const int begin = channels * threadIndex / threadCount;
        const int end = channels * (threadIndex + 1) / threadCount;
        threads.emplace_back(
            applyAdaGnNchw, normalized, scale.data(), bias.data(), output, spatial, begin, end);
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativeGroupNorm512Benchmark(
    JNIEnv* env, jclass, jint warmupRuns, jint measuredRuns, jint threadCount) {
    constexpr size_t elements = 512 * 16 * 32;
    std::vector<float> input(elements);
    std::vector<float> output(elements);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>((i % 251) - 125) * 0.001f;
    }
    for (int i = 0; i < warmupRuns; ++i) {
        nativeGroupNorm512Optimized(input.data(), output.data(), static_cast<int>(threadCount));
    }
    std::vector<jlong> elapsed(static_cast<size_t>(measuredRuns));
    volatile float checksum = 0.0f;
    for (int i = 0; i < measuredRuns; ++i) {
        auto started = std::chrono::steady_clock::now();
        nativeGroupNorm512Optimized(input.data(), output.data(), static_cast<int>(threadCount));
        auto ended = std::chrono::steady_clock::now();
        checksum += output[static_cast<size_t>(i) % output.size()];
        elapsed[static_cast<size_t>(i)] =
            std::chrono::duration_cast<std::chrono::nanoseconds>(ended - started).count();
    }
    if (checksum == 1234567.0f) {
        __android_log_print(ANDROID_LOG_INFO, kLogTag, "unexpected checksum=%f", static_cast<double>(checksum));
    }
    auto result = env->NewLongArray(static_cast<jsize>(elapsed.size()));
    env->SetLongArrayRegion(result, 0, static_cast<jsize>(elapsed.size()), elapsed.data());
    return result;
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativeAdaGn512Stage1Benchmark(
    JNIEnv* env, jclass, jstring weightsPath, jint warmupRuns, jint measuredRuns, jint threadCount) {
    try {
        const char* chars = env->GetStringUTFChars(weightsPath, nullptr);
        std::string path(chars);
        env->ReleaseStringUTFChars(weightsPath, chars);

        constexpr size_t gammaWeightElements = 512 * 18;
        constexpr size_t biasElements = 512;
        constexpr size_t totalElements = gammaWeightElements + biasElements + gammaWeightElements + biasElements;
        auto weights = readFloatsFromFile(path);
        if (weights.size() != totalElements) {
            fail(env, "native AdaGN weights size mismatch");
        }
        const float* gammaWeight = weights.data();
        const float* gammaBias = gammaWeight + gammaWeightElements;
        const float* betaWeight = gammaBias + biasElements;
        const float* betaBias = betaWeight + gammaWeightElements;

        constexpr size_t featureElements = 512 * 16 * 32;
        constexpr size_t codewordElements = 18 * 16 * 32;
        std::vector<float> feature(featureElements);
        std::vector<float> codeword(codewordElements);
        std::vector<float> normalized(featureElements);
        std::vector<float> output(featureElements);
        for (size_t i = 0; i < feature.size(); ++i) {
            feature[i] = static_cast<float>((i % 251) - 125) * 0.001f;
        }
        for (size_t i = 0; i < codeword.size(); ++i) {
            codeword[i] = static_cast<float>((i % 127) - 63) * 0.002f;
        }
        for (int i = 0; i < warmupRuns; ++i) {
            nativeAdaGn512Stage1(
                feature.data(), codeword.data(), gammaWeight, gammaBias, betaWeight, betaBias,
                normalized.data(), output.data(), static_cast<int>(threadCount));
        }
        std::vector<jlong> elapsed(static_cast<size_t>(measuredRuns));
        volatile float checksum = 0.0f;
        for (int i = 0; i < measuredRuns; ++i) {
            auto started = std::chrono::steady_clock::now();
            nativeAdaGn512Stage1(
                feature.data(), codeword.data(), gammaWeight, gammaBias, betaWeight, betaBias,
                normalized.data(), output.data(), static_cast<int>(threadCount));
            auto ended = std::chrono::steady_clock::now();
            checksum += output[static_cast<size_t>(i) % output.size()];
            elapsed[static_cast<size_t>(i)] =
                std::chrono::duration_cast<std::chrono::nanoseconds>(ended - started).count();
        }
        if (checksum == 1234567.0f) {
            __android_log_print(ANDROID_LOG_INFO, kLogTag, "unexpected checksum=%f", static_cast<double>(checksum));
        }
        auto result = env->NewLongArray(static_cast<jsize>(elapsed.size()));
        env->SetLongArrayRegion(result, 0, static_cast<jsize>(elapsed.size()), elapsed.data());
        return result;
    } catch (const std::exception& e) {
        fail(env, e.what());
    }
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativeAdaGnBenchmark(
    JNIEnv* env,
    jclass,
    jstring weightsPath,
    jint channels,
    jint height,
    jint width,
    jint warmupRuns,
    jint measuredRuns,
    jint threadCount) {
    try {
        const char* chars = env->GetStringUTFChars(weightsPath, nullptr);
        std::string path(chars);
        env->ReleaseStringUTFChars(weightsPath, chars);

        constexpr size_t codewordChannels = 18;
        const size_t outputChannels = static_cast<size_t>(channels);
        const size_t gammaWeightElements = outputChannels * codewordChannels;
        const size_t biasElements = outputChannels;
        const size_t totalElements = gammaWeightElements + biasElements + gammaWeightElements + biasElements;
        auto weights = readFloatsFromFile(path);
        if (weights.size() != totalElements) {
            fail(env, "native AdaGN weights size mismatch");
        }
        const float* gammaWeight = weights.data();
        const float* gammaBias = gammaWeight + gammaWeightElements;
        const float* betaWeight = gammaBias + biasElements;
        const float* betaBias = betaWeight + gammaWeightElements;

        const int spatial = static_cast<int>(height) * static_cast<int>(width);
        const size_t featureElements = outputChannels * static_cast<size_t>(spatial);
        constexpr size_t codewordElements = codewordChannels * 16 * 32;
        std::vector<float> feature(featureElements);
        std::vector<float> codeword(codewordElements);
        std::vector<float> normalized(featureElements);
        std::vector<float> output(featureElements);
        for (size_t i = 0; i < feature.size(); ++i) {
            feature[i] = static_cast<float>((i % 251) - 125) * 0.001f;
        }
        for (size_t i = 0; i < codeword.size(); ++i) {
            codeword[i] = static_cast<float>((i % 127) - 63) * 0.002f;
        }
        for (int i = 0; i < warmupRuns; ++i) {
            nativeAdaGnNchw(
                feature.data(), codeword.data(), gammaWeight, gammaBias, betaWeight, betaBias,
                normalized.data(), output.data(), static_cast<int>(channels), spatial, static_cast<int>(threadCount));
        }
        std::vector<jlong> elapsed(static_cast<size_t>(measuredRuns));
        volatile float checksum = 0.0f;
        for (int i = 0; i < measuredRuns; ++i) {
            auto started = std::chrono::steady_clock::now();
            nativeAdaGnNchw(
                feature.data(), codeword.data(), gammaWeight, gammaBias, betaWeight, betaBias,
                normalized.data(), output.data(), static_cast<int>(channels), spatial, static_cast<int>(threadCount));
            auto ended = std::chrono::steady_clock::now();
            checksum += output[static_cast<size_t>(i) % output.size()];
            elapsed[static_cast<size_t>(i)] =
                std::chrono::duration_cast<std::chrono::nanoseconds>(ended - started).count();
        }
        if (checksum == 1234567.0f) {
            __android_log_print(ANDROID_LOG_INFO, kLogTag, "unexpected checksum=%f", static_cast<double>(checksum));
        }
        auto result = env->NewLongArray(static_cast<jsize>(elapsed.size()));
        env->SetLongArrayRegion(result, 0, static_cast<jsize>(elapsed.size()), elapsed.data());
        return result;
    } catch (const std::exception& e) {
        fail(env, e.what());
    }
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativePixelUnshuffle2Benchmark(
    JNIEnv* env, jclass, jint warmupRuns, jint measuredRuns) {
    constexpr size_t inputElements = 256 * 32 * 64;
    constexpr size_t outputElements = 1024 * 16 * 32;
    std::vector<float> input(inputElements);
    std::vector<float> output(outputElements);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>((i % 251) - 125) * 0.001f;
    }

    for (int i = 0; i < warmupRuns; ++i) {
        pixelUnshuffle2Nchw256x32x64(input.data(), output.data());
    }

    std::vector<jlong> elapsed(static_cast<size_t>(measuredRuns));
    volatile float checksum = 0.0f;
    for (int i = 0; i < measuredRuns; ++i) {
        auto started = std::chrono::steady_clock::now();
        pixelUnshuffle2Nchw256x32x64(input.data(), output.data());
        auto ended = std::chrono::steady_clock::now();
        checksum += output[static_cast<size_t>(i) % output.size()];
        elapsed[static_cast<size_t>(i)] =
            std::chrono::duration_cast<std::chrono::nanoseconds>(ended - started).count();
    }
    if (checksum == 1234567.0f) {
        __android_log_print(ANDROID_LOG_INFO, kLogTag, "unexpected checksum=%f", static_cast<double>(checksum));
    }

    auto result = env->NewLongArray(static_cast<jsize>(elapsed.size()));
    env->SetLongArrayRegion(result, 0, static_cast<jsize>(elapsed.size()), elapsed.data());
    return result;
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativeDepthToSpace2Benchmark(
    JNIEnv* env, jclass, jint warmupRuns, jint measuredRuns) {
    constexpr size_t inputElements = 2048 * 16 * 32;
    constexpr size_t outputElements = 512 * 32 * 64;
    std::vector<float> input(inputElements);
    std::vector<float> output(outputElements);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>((i % 251) - 125) * 0.001f;
    }
    for (int i = 0; i < warmupRuns; ++i) {
        depthToSpace2Nchw2048x16x32(input.data(), output.data());
    }
    std::vector<jlong> elapsed(static_cast<size_t>(measuredRuns));
    volatile float checksum = 0.0f;
    for (int i = 0; i < measuredRuns; ++i) {
        auto started = std::chrono::steady_clock::now();
        depthToSpace2Nchw2048x16x32(input.data(), output.data());
        auto ended = std::chrono::steady_clock::now();
        checksum += output[static_cast<size_t>(i) % output.size()];
        elapsed[static_cast<size_t>(i)] =
            std::chrono::duration_cast<std::chrono::nanoseconds>(ended - started).count();
    }
    if (checksum == 1234567.0f) {
        __android_log_print(ANDROID_LOG_INFO, kLogTag, "unexpected checksum=%f", static_cast<double>(checksum));
    }
    auto result = env->NewLongArray(static_cast<jsize>(elapsed.size()));
    env->SetLongArrayRegion(result, 0, static_cast<jsize>(elapsed.size()), elapsed.data());
    return result;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativeOpenClFusedUpsamplerBenchmark(
    JNIEnv* env, jclass, jint warmupRuns, jint measuredRuns) {
    try {
        return env->NewStringUTF(openClFusedUpsamplerBenchmark(warmupRuns, measuredRuns).c_str());
    } catch (const std::exception& e) {
        std::string result = std::string("status=unavailable reason=") + e.what();
        return env->NewStringUTF(result.c_str());
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativeVulkanFusedUpsamplerBenchmark(
    JNIEnv* env, jclass, jint warmupRuns, jint measuredRuns) {
    try {
        return env->NewStringUTF(vulkanFusedUpsamplerBenchmark(warmupRuns, measuredRuns).c_str());
    } catch (const std::exception& e) {
        std::string result = std::string("status=unavailable reason=") + e.what();
        return env->NewStringUTF(result.c_str());
    }
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativeWSiLUChunkAddBenchmark(
    JNIEnv* env, jclass, jint warmupRuns, jint measuredRuns) {
    constexpr size_t inputElements = 1280 * 32 * 64;
    constexpr size_t outputElements = 640 * 32 * 64;
    std::vector<float> input(inputElements);
    std::vector<float> output(outputElements);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>((i % 251) - 125) * 0.001f;
    }
    for (int i = 0; i < warmupRuns; ++i) {
        wsiluChunkAdd1280x32x64(input.data(), output.data());
    }
    std::vector<jlong> elapsed(static_cast<size_t>(measuredRuns));
    volatile float checksum = 0.0f;
    for (int i = 0; i < measuredRuns; ++i) {
        auto started = std::chrono::steady_clock::now();
        wsiluChunkAdd1280x32x64(input.data(), output.data());
        auto ended = std::chrono::steady_clock::now();
        checksum += output[static_cast<size_t>(i) % output.size()];
        elapsed[static_cast<size_t>(i)] =
            std::chrono::duration_cast<std::chrono::nanoseconds>(ended - started).count();
    }
    if (checksum == 1234567.0f) {
        __android_log_print(ANDROID_LOG_INFO, kLogTag, "unexpected checksum=%f", static_cast<double>(checksum));
    }
    auto result = env->NewLongArray(static_cast<jsize>(elapsed.size()));
    env->SetLongArrayRegion(result, 0, static_cast<jsize>(elapsed.size()), elapsed.data());
    return result;
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativeFastWSiLUChunkAddBenchmark(
    JNIEnv* env, jclass, jint warmupRuns, jint measuredRuns) {
    constexpr size_t inputElements = 1280 * 32 * 64;
    constexpr size_t outputElements = 640 * 32 * 64;
    std::vector<float> input(inputElements);
    std::vector<float> output(outputElements);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>((i % 251) - 125) * 0.001f;
    }
    for (int i = 0; i < warmupRuns; ++i) {
        fastWSiLUChunkAdd1280x32x64(input.data(), output.data());
    }
    std::vector<jlong> elapsed(static_cast<size_t>(measuredRuns));
    volatile float checksum = 0.0f;
    for (int i = 0; i < measuredRuns; ++i) {
        auto started = std::chrono::steady_clock::now();
        fastWSiLUChunkAdd1280x32x64(input.data(), output.data());
        auto ended = std::chrono::steady_clock::now();
        checksum += output[static_cast<size_t>(i) % output.size()];
        elapsed[static_cast<size_t>(i)] =
            std::chrono::duration_cast<std::chrono::nanoseconds>(ended - started).count();
    }
    if (checksum == 1234567.0f) {
        __android_log_print(ANDROID_LOG_INFO, kLogTag, "unexpected checksum=%f", static_cast<double>(checksum));
    }
    auto result = env->NewLongArray(static_cast<jsize>(elapsed.size()));
    env->SetLongArrayRegion(result, 0, static_cast<jsize>(elapsed.size()), elapsed.data());
    return result;
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativeFusedPixelUnshuffleAdaptorBenchmark(
    JNIEnv* env, jclass, jstring weightsPath, jint warmupRuns, jint measuredRuns, jint threadCount) {
    try {
        const char* chars = env->GetStringUTFChars(weightsPath, nullptr);
        std::string path(chars);
        env->ReleaseStringUTFChars(weightsPath, chars);

        constexpr size_t inputElements = 256 * 32 * 64;
        constexpr size_t outputElements = 256 * 16 * 32;
        constexpr size_t weightElements = 256 * 1024;
        constexpr size_t biasElements = 256;
        auto weightsAndBias = readFloatsFromFile(path);
        if (weightsAndBias.size() != weightElements + biasElements) {
            fail(env, "native adaptor weights size mismatch");
        }
        const float* weight = weightsAndBias.data();
        const float* bias = weightsAndBias.data() + weightElements;
        std::vector<float> input(inputElements);
        std::vector<float> unshuffled(1024 * 16 * 32);
        std::vector<float> output(outputElements);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<float>((i % 251) - 125) * 0.001f;
        }

        for (int i = 0; i < warmupRuns; ++i) {
            fusedPixelUnshuffle2AdaptorOptimized(
                input.data(), weight, bias, unshuffled.data(), output.data(), static_cast<int>(threadCount));
        }

        std::vector<jlong> elapsed(static_cast<size_t>(measuredRuns));
        volatile float checksum = 0.0f;
        for (int i = 0; i < measuredRuns; ++i) {
            auto started = std::chrono::steady_clock::now();
            fusedPixelUnshuffle2AdaptorOptimized(
                input.data(), weight, bias, unshuffled.data(), output.data(), static_cast<int>(threadCount));
            auto ended = std::chrono::steady_clock::now();
            checksum += output[static_cast<size_t>(i) % output.size()];
            elapsed[static_cast<size_t>(i)] =
                std::chrono::duration_cast<std::chrono::nanoseconds>(ended - started).count();
        }
        if (checksum == 1234567.0f) {
            __android_log_print(ANDROID_LOG_INFO, kLogTag, "unexpected checksum=%f", static_cast<double>(checksum));
        }

        auto result = env->NewLongArray(static_cast<jsize>(elapsed.size()));
        env->SetLongArrayRegion(result, 0, static_cast<jsize>(elapsed.size()), elapsed.data());
        return result;
    } catch (const std::exception& e) {
        fail(env, e.what());
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativeCreate(
    JNIEnv* env, jclass, jstring path, jint accelerationMode, jint acceleratorFlag, jstring cacheDir) {
    try {
        const char* chars = env->GetStringUTFChars(path, nullptr);
        std::string modelPath(chars);
        env->ReleaseStringUTFChars(path, chars);
        std::string cacheDirPath;
        if (cacheDir != nullptr) {
            const char* cacheChars = env->GetStringUTFChars(cacheDir, nullptr);
            cacheDirPath = cacheChars;
            env->ReleaseStringUTFChars(cacheDir, cacheChars);
        }

        auto runtime = std::make_unique<Runtime>();
        runtime->accelerationMode = static_cast<int>(accelerationMode);
        runtime->api = loadApi(env);
        std::vector<std::string> optionStatuses;
        void* options = nullptr;
        check(env, runtime->api->optionsCreate(&options), "ANeuroPilotTFLiteOptions_create");
        appendOptionStatus(
            optionStatuses,
            "accelerationMode",
            runtime->api->optionsSetAccelerationMode(options, static_cast<int>(accelerationMode)));
        appendOptionStatus(
            optionStatuses,
            "allowFp16",
            runtime->api->optionsSetAllowFp16(options, true));
        applyOptionalOption(optionStatuses, "preferenceSustainedSpeed", runtime->api->optionsSetPreference, options, 2);
        applyOptionalOption(optionStatuses, "priorityHigh", runtime->api->optionsSetExecutionPriority, options, 110);
        applyOptionalOption(optionStatuses, "lowLatency", runtime->api->optionsSetLowLatency, options, true);
        applyOptionalOption(optionStatuses, "deepFusion", runtime->api->optionsSetDeepFusion, options, true);
        applyOptionalOption(optionStatuses, "boostHint", runtime->api->optionsSetBoostHint, options, 100);
        applyOptionalOption(optionStatuses, "boostDuration", runtime->api->optionsSetBoostDuration, options, 3000);
        applyOptionalOption(
            optionStatuses,
            "maxDelegatedPartitions",
            runtime->api->optionsSetMaxNumberDelegatedPartitions,
            options,
            32);
        applyOptionalOption(
            optionStatuses,
            "disallowNnApiCpu",
            runtime->api->optionsSetDisallowNnApiCpu,
            options,
            static_cast<int>(accelerationMode) != 0);
        if (acceleratorFlag != 0) {
            applyOptionalOption(
                optionStatuses,
                "acceleratorFlag",
                runtime->api->optionsSetAcceleratorFlag,
                options,
                static_cast<uint32_t>(acceleratorFlag));
        } else {
            optionStatuses.push_back("acceleratorFlag=auto");
        }
        if (!cacheDirPath.empty() && runtime->api->optionsSetCacheDir != nullptr) {
            appendOptionStatus(optionStatuses, "cacheDir", runtime->api->optionsSetCacheDir(options, cacheDirPath.c_str()));
        } else {
            optionStatuses.push_back(std::string("cacheDir=") + (cacheDirPath.empty() ? "unset" : "missing"));
        }
        runtime->optionsSummary = joinStatuses(optionStatuses);
        int status = runtime->api->createAdv(&runtime->tflite, modelPath.c_str(), options);
        runtime->api->optionsFree(options);
        check(env, status, "ANeuroPilotTFLite_createAdv");
        runtime->inputSizes = querySizes(env, *runtime, kInputBuffer);
        runtime->outputSizes = querySizes(env, *runtime, kOutputBuffer);
        return reinterpret_cast<jlong>(runtime.release());
    } catch (const std::exception&) {
        return 0;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativeRelease(JNIEnv*, jclass, jlong handle) {
    delete reinterpret_cast<Runtime*>(handle);
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativeInputSizes(JNIEnv* env, jclass, jlong handle) {
    return toLongArray(env, fromHandle(env, handle)->inputSizes);
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativeOutputSizes(JNIEnv* env, jclass, jlong handle) {
    return toLongArray(env, fromHandle(env, handle)->outputSizes);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativeOptionsSummary(JNIEnv* env, jclass, jlong handle) {
    return env->NewStringUTF(fromHandle(env, handle)->optionsSummary.c_str());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativeFullyDelegated(JNIEnv* env, jclass, jlong handle) {
    Runtime* runtime = fromHandle(env, handle);
    bool delegated = false;
    return runtime->api->isFullyDelegated != nullptr &&
           runtime->api->isFullyDelegated(runtime->tflite, &delegated) == 0 &&
           delegated;
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativeRun(
    JNIEnv* env, jclass, jlong handle, jobjectArray inputs, jboolean copyOutputs) {
    try {
        Runtime* runtime = fromHandle(env, handle);
        jsize inputCount = env->GetArrayLength(inputs);
        if (static_cast<size_t>(inputCount) != runtime->inputSizes.size()) {
            fail(env, "input count mismatch");
        }
        std::vector<std::vector<uint8_t>> inputBuffers(static_cast<size_t>(inputCount));
        for (jsize i = 0; i < inputCount; ++i) {
            auto input = static_cast<jbyteArray>(env->GetObjectArrayElement(inputs, i));
            jsize size = env->GetArrayLength(input);
            if (static_cast<size_t>(size) != runtime->inputSizes[i]) {
                fail(env, "input byte size mismatch");
            }
            inputBuffers[static_cast<size_t>(i)].resize(static_cast<size_t>(size));
            env->GetByteArrayRegion(
                input, 0, size, reinterpret_cast<jbyte*>(inputBuffers[static_cast<size_t>(i)].data()));
            if (kVerboseRunLogging) {
                __android_log_print(ANDROID_LOG_INFO, kLogTag, "set_tensor_buffer_start index=%d bytes=%d", i, size);
            }
            if (runtime->accelerationMode == 0) {
                check(env, runtime->api->setInputTensorData(
                               runtime->tflite,
                               static_cast<uint32_t>(i),
                               inputBuffers[static_cast<size_t>(i)].data(),
                               inputBuffers[static_cast<size_t>(i)].size()),
                      "ANeuroPilotTFLite_setInputTensorData input");
            } else {
                check(env, runtime->api->setTensorBuffer(
                               runtime->tflite, i,
                               reinterpret_cast<char*>(inputBuffers[static_cast<size_t>(i)].data())),
                      "ANeuroPilotTFLite_setTensorBuffer input");
            }
            if (kVerboseRunLogging) {
                __android_log_print(ANDROID_LOG_INFO, kLogTag, "set_tensor_buffer_done index=%d", i);
            }
            env->DeleteLocalRef(input);
        }

        if (kVerboseRunLogging) {
            __android_log_print(ANDROID_LOG_INFO, kLogTag, "invoke_start");
        }
        check(env, runtime->api->invoke(runtime->tflite), "ANeuroPilotTFLite_invoke");
        if (kVerboseRunLogging) {
            __android_log_print(ANDROID_LOG_INFO, kLogTag, "invoke_done");
        }

        auto byteArrayClass = env->FindClass("[B");
        if (!copyOutputs) {
            return env->NewObjectArray(0, byteArrayClass, nullptr);
        }
        auto outputs = env->NewObjectArray(static_cast<jsize>(runtime->outputSizes.size()), byteArrayClass, nullptr);
        for (size_t i = 0; i < runtime->outputSizes.size(); ++i) {
            std::vector<uint8_t> buffer(runtime->outputSizes[i]);
            if (kVerboseRunLogging) {
                __android_log_print(
                    ANDROID_LOG_INFO, kLogTag, "get_output_start index=%zu bytes=%zu", i, runtime->outputSizes[i]);
            }
            check(env, runtime->api->getOutputTensorData(runtime->tflite, i, buffer.data(), buffer.size()),
                  "ANeuroPilotTFLite_getOutputTensorData");
            if (kVerboseRunLogging) {
                __android_log_print(ANDROID_LOG_INFO, kLogTag, "get_output_done index=%zu", i);
            }
            auto output = env->NewByteArray(static_cast<jsize>(buffer.size()));
            env->SetByteArrayRegion(output, 0, static_cast<jsize>(buffer.size()), reinterpret_cast<jbyte*>(buffer.data()));
            env->SetObjectArrayElement(outputs, static_cast<jsize>(i), output);
            env->DeleteLocalRef(output);
        }
        return outputs;
    } catch (const std::exception&) {
        return nullptr;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativeProbeNeuronExtensions(
    JNIEnv* env, jclass, jobjectArray names) {
    try {
        return env->NewStringUTF(probeNeuronExtensions(env, names).c_str());
    } catch (const std::exception& e) {
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
        }
        return env->NewStringUTF((std::string("error=") + e.what()).c_str());
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativeProbeAhwbSymbols(
    JNIEnv* env, jclass) {
    try {
        return env->NewStringUTF(probeAhwbSymbols().c_str());
    } catch (const std::exception& e) {
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
        }
        return env->NewStringUTF((std::string("error=") + e.what()).c_str());
    }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativePixelUnshuffle2Nchw256(
    JNIEnv* env, jclass, jfloatArray inputArray) {
    try {
        constexpr size_t inputElements = 256 * 32 * 64;
        constexpr size_t outputElements = 1024 * 16 * 32;
        if (env->GetArrayLength(inputArray) != static_cast<jsize>(inputElements)) {
            fail(env, "native pixel_unshuffle2 input size mismatch");
        }
        std::vector<float> input(inputElements);
        std::vector<float> output(outputElements);
        env->GetFloatArrayRegion(inputArray, 0, static_cast<jsize>(inputElements), input.data());
        pixelUnshuffle2Nchw256x32x64(input.data(), output.data());
        auto result = env->NewFloatArray(static_cast<jsize>(outputElements));
        env->SetFloatArrayRegion(result, 0, static_cast<jsize>(outputElements), output.data());
        return result;
    } catch (const std::exception&) {
        return nullptr;
    }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativeGroupNormNchw(
    JNIEnv* env,
    jclass,
    jfloatArray inputArray,
    jint channels,
    jint height,
    jint width,
    jint groups,
    jint threadCount) {
    try {
        if (channels <= 0 || height <= 0 || width <= 0 || groups != 32 || channels % groups != 0) {
            fail(env, "native groupnorm invalid shape/groups");
        }
        const size_t elements = static_cast<size_t>(channels) * static_cast<size_t>(height) * static_cast<size_t>(width);
        if (env->GetArrayLength(inputArray) != static_cast<jsize>(elements)) {
            fail(env, "native groupnorm input size mismatch");
        }
        std::vector<float> input(elements);
        std::vector<float> output(elements);
        env->GetFloatArrayRegion(inputArray, 0, static_cast<jsize>(elements), input.data());
        nativeGroupNormOptimized(
            input.data(),
            output.data(),
            static_cast<int>(channels),
            static_cast<int>(height) * static_cast<int>(width),
            static_cast<int>(threadCount));
        auto result = env->NewFloatArray(static_cast<jsize>(elements));
        env->SetFloatArrayRegion(result, 0, static_cast<jsize>(elements), output.data());
        return result;
    } catch (const std::exception&) {
        return nullptr;
    }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativeAdaGnNchw(
    JNIEnv* env,
    jclass,
    jfloatArray featureArray,
    jfloatArray codewordArray,
    jstring weightsPath,
    jint channels,
    jint height,
    jint width,
    jint threadCount) {
    try {
        if (channels <= 0 || height <= 0 || width <= 0) {
            fail(env, "native adagn invalid shape");
        }
        const size_t featureElements =
            static_cast<size_t>(channels) * static_cast<size_t>(height) * static_cast<size_t>(width);
        constexpr size_t codewordElements = 18 * 16 * 32;
        if (env->GetArrayLength(featureArray) != static_cast<jsize>(featureElements)) {
            fail(env, "native adagn feature size mismatch");
        }
        if (env->GetArrayLength(codewordArray) != static_cast<jsize>(codewordElements)) {
            fail(env, "native adagn codeword size mismatch");
        }

        const char* chars = env->GetStringUTFChars(weightsPath, nullptr);
        std::string path(chars);
        env->ReleaseStringUTFChars(weightsPath, chars);
        auto weights = readFloatsFromFile(path);
        const size_t weightElements = static_cast<size_t>(channels) * 18;
        const size_t biasElements = static_cast<size_t>(channels);
        const size_t totalElements = weightElements + biasElements + weightElements + biasElements;
        if (weights.size() != totalElements) {
            fail(env, "native adagn weights size mismatch");
        }

        std::vector<float> feature(featureElements);
        std::vector<float> codeword(codewordElements);
        std::vector<float> normalized(featureElements);
        std::vector<float> output(featureElements);
        env->GetFloatArrayRegion(featureArray, 0, static_cast<jsize>(featureElements), feature.data());
        env->GetFloatArrayRegion(codewordArray, 0, static_cast<jsize>(codewordElements), codeword.data());

        const float* gammaWeight = weights.data();
        const float* gammaBias = gammaWeight + weightElements;
        const float* betaWeight = gammaBias + biasElements;
        const float* betaBias = betaWeight + weightElements;
        nativeAdaGnNchw(
            feature.data(),
            codeword.data(),
            gammaWeight,
            gammaBias,
            betaWeight,
            betaBias,
            normalized.data(),
            output.data(),
            static_cast<int>(channels),
            static_cast<int>(height) * static_cast<int>(width),
            static_cast<int>(threadCount));

        auto result = env->NewFloatArray(static_cast<jsize>(featureElements));
        env->SetFloatArrayRegion(result, 0, static_cast<jsize>(featureElements), output.data());
        return result;
    } catch (const std::exception&) {
        return nullptr;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativePReconPipelineProbe(
    JNIEnv* env,
    jclass,
    jobjectArray modelPathsArray,
    jobjectArray adaWeightPathsArray,
    jstring cacheDir,
    jint warmupRuns,
    jint measuredRuns) {
    try {
        auto modelPaths = getStringArray(env, modelPathsArray);
        auto adaWeightPaths = getStringArray(env, adaWeightPathsArray);
        if (modelPaths.size() != 9) {
            fail(env, "native p recon pipeline expects 9 model paths");
        }
        if (adaWeightPaths.size() != 5) {
            fail(env, "native p recon pipeline expects 5 AdaGN weight paths");
        }
        const char* cacheChars = env->GetStringUTFChars(cacheDir, nullptr);
        std::string cacheDirPath(cacheChars);
        env->ReleaseStringUTFChars(cacheDir, cacheChars);

        const auto createStarted = Clock::now();
        std::vector<std::unique_ptr<Runtime>> runtimes;
        runtimes.reserve(modelPaths.size());
        for (const auto& path : modelPaths) {
            runtimes.push_back(createPipelineRuntime(env, path, cacheDirPath));
        }
        const auto createEnded = Clock::now();

        std::vector<std::vector<float>> adaWeights;
        adaWeights.reserve(adaWeightPaths.size());
        for (const auto& path : adaWeightPaths) {
            adaWeights.push_back(readFloatsFromFile(path));
        }

        std::vector<long long> ignored(17, 0);
        for (int i = 0; i < warmupRuns; ++i) {
            runPReconPipelineOnce(env, runtimes, adaWeights, ignored);
        }

        std::vector<long long> timings(17, 0);
        std::vector<long long> totals;
        totals.reserve(static_cast<size_t>(measuredRuns));
        std::vector<float> frame;
        for (int i = 0; i < measuredRuns; ++i) {
            const auto started = Clock::now();
            frame = runPReconPipelineOnce(env, runtimes, adaWeights, timings);
            const auto ended = Clock::now();
            totals.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(ended - started).count());
        }

        const char* labels[] = {
            "p_latent_decoder",
            "native_pixel_unshuffle",
            "native_mlp_norm0",
            "p_mlp_dcb0",
            "native_mlp_norm1_silu",
            "p_mlp_dcb1",
            "p_decoder_stage1_conv",
            "native_ada1",
            "p_decoder_stage2_blocks",
            "native_ada2",
            "p_upsampler_original",
            "native_ada3",
            "p_decoder_stage3_blocks",
            "native_ada4",
            "p_decoder_stage4_blocks",
            "native_ada_final",
            "p_recon_final_head_no_ada",
        };

        auto meanMs = [](long long nanos, int count) {
            return static_cast<double>(nanos) / static_cast<double>(count) / 1'000'000.0;
        };
        long long totalSum = 0;
        for (long long value : totals) {
            totalSum += value;
        }
        double checksum = 0.0;
        const size_t step = std::max<size_t>(1, frame.size() / 4096);
        for (size_t i = 0; i < frame.size(); i += step) {
            checksum += frame[i];
        }

        std::ostringstream out;
        out.setf(std::ios::fixed);
        out.precision(3);
        out << "create_ms="
            << std::chrono::duration_cast<std::chrono::nanoseconds>(createEnded - createStarted).count() / 1'000'000.0
            << " measured=" << measuredRuns
            << " output_floats=" << frame.size()
            << " checksum=" << checksum
            << "\n";
        out << "stage=total mean_ms=" << meanMs(totalSum, measuredRuns) << "\n";
        for (size_t i = 0; i < timings.size(); ++i) {
            out << "stage=" << labels[i] << " mean_ms=" << meanMs(timings[i], measuredRuns) << "\n";
        }
        return env->NewStringUTF(out.str().c_str());
    } catch (const std::exception& e) {
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
        }
        return env->NewStringUTF((std::string("error=") + e.what()).c_str());
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativePReconPipelineRun(
    JNIEnv* env,
    jclass,
    jobjectArray modelPathsArray,
    jobjectArray adaWeightPathsArray,
    jstring cacheDir,
    jfloatArray pYHatArray,
    jfloatArray pCtxArray) {
    try {
        auto modelPaths = getStringArray(env, modelPathsArray);
        auto adaWeightPaths = getStringArray(env, adaWeightPathsArray);
        if (modelPaths.size() != 9) {
            fail(env, "native p recon pipeline run expects 9 model paths");
        }
        if (adaWeightPaths.size() != 5) {
            fail(env, "native p recon pipeline run expects 5 AdaGN weight paths");
        }
        const char* cacheChars = env->GetStringUTFChars(cacheDir, nullptr);
        std::string cacheDirPath(cacheChars);
        env->ReleaseStringUTFChars(cacheDir, cacheChars);

        std::vector<std::unique_ptr<Runtime>> runtimes;
        runtimes.reserve(modelPaths.size());
        for (const auto& path : modelPaths) {
            runtimes.push_back(createPipelineRuntime(env, path, cacheDirPath));
        }
        std::vector<std::vector<float>> adaWeights;
        adaWeights.reserve(adaWeightPaths.size());
        for (const auto& path : adaWeightPaths) {
            adaWeights.push_back(readFloatsFromFile(path));
        }

        constexpr size_t pYHatElements = 128 * 16 * 32;
        constexpr size_t pCtxElements = 256 * 32 * 64;
        if (env->GetArrayLength(pYHatArray) != static_cast<jsize>(pYHatElements)) {
            fail(env, "native p recon pipeline run p_y_hat size mismatch");
        }
        if (env->GetArrayLength(pCtxArray) != static_cast<jsize>(pCtxElements)) {
            fail(env, "native p recon pipeline run p_ctx size mismatch");
        }
        std::vector<float> pYHat(pYHatElements);
        std::vector<float> pCtx(pCtxElements);
        env->GetFloatArrayRegion(pYHatArray, 0, static_cast<jsize>(pYHatElements), pYHat.data());
        env->GetFloatArrayRegion(pCtxArray, 0, static_cast<jsize>(pCtxElements), pCtx.data());

        std::vector<long long> timings(17, 0);
        auto outputs = runPReconPipelineOnceWithInputs(env, runtimes, adaWeights, pYHat, pCtx, timings);

        jclass floatArrayClass = env->FindClass("[F");
        auto result = env->NewObjectArray(2, floatArrayClass, nullptr);
        auto featureArray = env->NewFloatArray(static_cast<jsize>(outputs.feature.size()));
        auto frameArray = env->NewFloatArray(static_cast<jsize>(outputs.frame.size()));
        env->SetFloatArrayRegion(featureArray, 0, static_cast<jsize>(outputs.feature.size()), outputs.feature.data());
        env->SetFloatArrayRegion(frameArray, 0, static_cast<jsize>(outputs.frame.size()), outputs.frame.data());
        env->SetObjectArrayElement(result, 0, featureArray);
        env->SetObjectArrayElement(result, 1, frameArray);
        env->DeleteLocalRef(featureArray);
        env->DeleteLocalRef(frameArray);
        return result;
    } catch (const std::exception&) {
        return nullptr;
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativePReconPipelineTrace(
    JNIEnv* env,
    jclass,
    jobjectArray modelPathsArray,
    jobjectArray adaWeightPathsArray,
    jstring cacheDir,
    jfloatArray pYHatArray,
    jfloatArray pCtxArray) {
    try {
        auto modelPaths = getStringArray(env, modelPathsArray);
        auto adaWeightPaths = getStringArray(env, adaWeightPathsArray);
        if (modelPaths.size() != 9) {
            fail(env, "native p recon pipeline trace expects 9 model paths");
        }
        if (adaWeightPaths.size() != 5) {
            fail(env, "native p recon pipeline trace expects 5 AdaGN weight paths");
        }
        const char* cacheChars = env->GetStringUTFChars(cacheDir, nullptr);
        std::string cacheDirPath(cacheChars);
        env->ReleaseStringUTFChars(cacheDir, cacheChars);

        std::vector<std::unique_ptr<Runtime>> runtimes;
        runtimes.reserve(modelPaths.size());
        for (const auto& path : modelPaths) {
            runtimes.push_back(createPipelineRuntime(env, path, cacheDirPath));
        }
        std::vector<std::vector<float>> adaWeights;
        adaWeights.reserve(adaWeightPaths.size());
        for (const auto& path : adaWeightPaths) {
            adaWeights.push_back(readFloatsFromFile(path));
        }

        constexpr size_t pYHatElements = 128 * 16 * 32;
        constexpr size_t pCtxElements = 256 * 32 * 64;
        if (env->GetArrayLength(pYHatArray) != static_cast<jsize>(pYHatElements)) {
            fail(env, "native p recon pipeline trace p_y_hat size mismatch");
        }
        if (env->GetArrayLength(pCtxArray) != static_cast<jsize>(pCtxElements)) {
            fail(env, "native p recon pipeline trace p_ctx size mismatch");
        }
        std::vector<float> pYHat(pYHatElements);
        std::vector<float> pCtx(pCtxElements);
        env->GetFloatArrayRegion(pYHatArray, 0, static_cast<jsize>(pYHatElements), pYHat.data());
        env->GetFloatArrayRegion(pCtxArray, 0, static_cast<jsize>(pCtxElements), pCtx.data());

        std::vector<long long> timings(17, 0);
        std::vector<std::vector<float>> trace;
        trace.reserve(17);
        runPReconPipelineOnceWithInputs(env, runtimes, adaWeights, pYHat, pCtx, timings, &trace);

        jclass floatArrayClass = env->FindClass("[F");
        auto result = env->NewObjectArray(static_cast<jsize>(trace.size()), floatArrayClass, nullptr);
        for (size_t i = 0; i < trace.size(); ++i) {
            auto item = env->NewFloatArray(static_cast<jsize>(trace[i].size()));
            env->SetFloatArrayRegion(item, 0, static_cast<jsize>(trace[i].size()), trace[i].data());
            env->SetObjectArrayElement(result, static_cast<jsize>(i), item);
            env->DeleteLocalRef(item);
        }
        return result;
    } catch (const std::exception&) {
        return nullptr;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativePReconBigPipelineProbe(
    JNIEnv* env,
    jclass,
    jobjectArray modelPathsArray,
    jstring cacheDir,
    jint warmupRuns,
    jint measuredRuns) {
    try {
        auto modelPaths = getStringArray(env, modelPathsArray);
        if (modelPaths.size() != 4) {
            fail(env, "native p recon big pipeline expects 4 model paths");
        }
        const char* cacheChars = env->GetStringUTFChars(cacheDir, nullptr);
        std::string cacheDirPath(cacheChars);
        env->ReleaseStringUTFChars(cacheDir, cacheChars);

        const auto createStarted = Clock::now();
        std::vector<std::unique_ptr<Runtime>> runtimes;
        runtimes.reserve(modelPaths.size());
        for (const auto& path : modelPaths) {
            runtimes.push_back(createPipelineRuntime(env, path, cacheDirPath));
        }
        const auto createEnded = Clock::now();

        constexpr size_t pYHatElements = 128 * 16 * 32;
        constexpr size_t pCtxElements = 256 * 32 * 64;
        constexpr size_t qReconElements = 1;
        std::vector<float> pYHat(pYHatElements);
        std::vector<float> pCtx(pCtxElements);
        std::vector<float> qRecon(qReconElements, 1.0f);
        for (size_t i = 0; i < pYHat.size(); ++i) {
            pYHat[i] = static_cast<float>((i % 127) - 63) * 0.001f;
        }
        for (size_t i = 0; i < pCtx.size(); ++i) {
            pCtx[i] = static_cast<float>((i % 251) - 125) * 0.0005f;
        }

        auto runOnce = [&]() {
            std::vector<float> codeword;
            std::vector<float> stage2;
            std::vector<float> stage3;
            std::vector<float> frame;
            codeword = runPipelineTflite(env, *runtimes[0], {&pYHat, &pCtx});
            stage2 = runPipelineTflite(env, *runtimes[1], {&codeword});
            stage3 = runPipelineTflite(env, *runtimes[2], {&stage2, &codeword});
            frame = runPipelineTflite(env, *runtimes[3], {&stage3, &codeword, &qRecon});
            return frame;
        };

        for (int i = 0; i < warmupRuns; ++i) {
            (void)runOnce();
        }

        const char* labels[] = {
            "p_recon_big_latent_mlp",
            "p_recon_big_stage1_stage2",
            "p_recon_big_upsample_stage3",
            "p_recon_big_stage4_final",
        };
        std::vector<long long> timings(4, 0);
        std::vector<long long> totals;
        totals.reserve(static_cast<size_t>(measuredRuns));
        std::vector<float> frame;
        for (int i = 0; i < measuredRuns; ++i) {
            std::vector<float> codeword;
            std::vector<float> stage2;
            std::vector<float> stage3;
            const auto totalStarted = Clock::now();
            measurePipelineStage(timings, 0, [&] {
                codeword = runPipelineTflite(env, *runtimes[0], {&pYHat, &pCtx});
            });
            measurePipelineStage(timings, 1, [&] {
                stage2 = runPipelineTflite(env, *runtimes[1], {&codeword});
            });
            measurePipelineStage(timings, 2, [&] {
                stage3 = runPipelineTflite(env, *runtimes[2], {&stage2, &codeword});
            });
            measurePipelineStage(timings, 3, [&] {
                frame = runPipelineTflite(env, *runtimes[3], {&stage3, &codeword, &qRecon});
            });
            const auto totalEnded = Clock::now();
            totals.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(totalEnded - totalStarted).count());
        }

        long long totalSum = 0;
        for (long long value : totals) {
            totalSum += value;
        }
        double checksum = 0.0;
        const size_t step = std::max<size_t>(1, frame.size() / 4096);
        for (size_t i = 0; i < frame.size(); i += step) {
            checksum += frame[i];
        }
        auto meanMs = [](long long nanos, int count) {
            return static_cast<double>(nanos) / static_cast<double>(count) / 1'000'000.0;
        };

        std::ostringstream out;
        out.setf(std::ios::fixed);
        out.precision(3);
        out << "create_ms="
            << std::chrono::duration_cast<std::chrono::nanoseconds>(createEnded - createStarted).count() / 1'000'000.0
            << " measured=" << measuredRuns
            << " output_floats=" << frame.size()
            << " checksum=" << checksum
            << "\n";
        out << "stage=total mean_ms=" << meanMs(totalSum, measuredRuns) << "\n";
        for (size_t i = 0; i < timings.size(); ++i) {
            out << "stage=" << labels[i] << " mean_ms=" << meanMs(timings[i], measuredRuns) << "\n";
        }
        return env->NewStringUTF(out.str().c_str());
    } catch (const std::exception& e) {
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
        }
        return env->NewStringUTF((std::string("error=") + e.what()).c_str());
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativePReconMixedMergedProbe(
    JNIEnv* env,
    jclass,
    jobjectArray modelPathsArray,
    jobjectArray adaWeightPathsArray,
    jstring cacheDir,
    jint warmupRuns,
    jint measuredRuns) {
    try {
        auto modelPaths = getStringArray(env, modelPathsArray);
        auto adaWeightPaths = getStringArray(env, adaWeightPathsArray);
        if (modelPaths.size() != 6) {
            fail(env, "native p recon mixed merged pipeline expects 6 model paths");
        }
        if (adaWeightPaths.size() != 5) {
            fail(env, "native p recon mixed merged pipeline expects 5 AdaGN weight paths");
        }
        const char* cacheChars = env->GetStringUTFChars(cacheDir, nullptr);
        std::string cacheDirPath(cacheChars);
        env->ReleaseStringUTFChars(cacheDir, cacheChars);

        const auto createStarted = Clock::now();
        std::vector<std::unique_ptr<Runtime>> runtimes;
        runtimes.reserve(modelPaths.size());
        for (const auto& path : modelPaths) {
            runtimes.push_back(createPipelineRuntime(env, path, cacheDirPath));
        }
        const auto createEnded = Clock::now();

        std::vector<std::vector<float>> adaWeights;
        adaWeights.reserve(adaWeightPaths.size());
        for (const auto& path : adaWeightPaths) {
            adaWeights.push_back(readFloatsFromFile(path));
        }

        std::vector<long long> ignored(11, 0);
        for (int i = 0; i < warmupRuns; ++i) {
            (void)runPReconMixedMergedOnce(env, runtimes, adaWeights, ignored);
        }

        std::vector<long long> timings(11, 0);
        std::vector<long long> totals;
        totals.reserve(static_cast<size_t>(measuredRuns));
        std::vector<float> frame;
        for (int i = 0; i < measuredRuns; ++i) {
            const auto started = Clock::now();
            frame = runPReconMixedMergedOnce(env, runtimes, adaWeights, timings);
            const auto ended = Clock::now();
            totals.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(ended - started).count());
        }

        const char* labels[] = {
            "p_latent_decoder",
            "native_pixel_unshuffle",
            "native_mlp_norm0",
            "p_mlp_dcb0",
            "native_mlp_norm1_silu",
            "p_mlp_dcb1",
            "p_stage1_stage2_no_norm",
            "native_ada2",
            "p_upsample_stage3_no_norm",
            "native_ada4",
            "p_stage4_final_no_norm",
        };

        auto meanMs = [](long long nanos, int count) {
            return static_cast<double>(nanos) / static_cast<double>(count) / 1'000'000.0;
        };
        long long totalSum = 0;
        for (long long value : totals) {
            totalSum += value;
        }
        double checksum = 0.0;
        const size_t step = std::max<size_t>(1, frame.size() / 4096);
        for (size_t i = 0; i < frame.size(); i += step) {
            checksum += frame[i];
        }

        std::ostringstream out;
        out.setf(std::ios::fixed);
        out.precision(3);
        out << "non_equivalent_no_norm=true create_ms="
            << std::chrono::duration_cast<std::chrono::nanoseconds>(createEnded - createStarted).count() / 1'000'000.0
            << " measured=" << measuredRuns
            << " output_floats=" << frame.size()
            << " checksum=" << checksum
            << "\n";
        out << "stage=total mean_ms=" << meanMs(totalSum, measuredRuns) << "\n";
        for (size_t i = 0; i < timings.size(); ++i) {
            out << "stage=" << labels[i] << " mean_ms=" << meanMs(timings[i], measuredRuns) << "\n";
        }
        return env->NewStringUTF(out.str().c_str());
    } catch (const std::exception& e) {
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
        }
        return env->NewStringUTF((std::string("error=") + e.what()).c_str());
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_gvcrt_clean_MtkTfliteRuntime_00024Companion_nativeDlaRuntimeProbe(
    JNIEnv* env, jclass, jstring dlaPath) {
    try {
        const char* chars = env->GetStringUTFChars(dlaPath, nullptr);
        std::string path(chars);
        env->ReleaseStringUTFChars(dlaPath, chars);
        return env->NewStringUTF(probeDlaRuntimeV2(path).c_str());
    } catch (const std::exception& e) {
        return env->NewStringUTF((std::string("error=") + e.what()).c_str());
    }
}
