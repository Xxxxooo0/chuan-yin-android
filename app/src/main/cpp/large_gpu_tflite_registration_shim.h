#pragma once

#include <dlfcn.h>

#include <cstring>
#include <stdexcept>
#include <string>

// The existing Large rANS bridge targets MTK's RegistrationExternal API. The
// official TFLite JNI bundled by the GPU runtime exposes the equivalent
// TfLiteOperator API instead. This shim adapts only those registration calls;
// the existing Large rANS implementation is included unchanged by the two GPU
// JNI translation units.
namespace gvcrt_large_gpu_shim {

using LegacyInvoke = int (*)(void*, void*);
using OperatorInvoke = int (*)(void*, void*, void*);

struct Registration {
    void* library = nullptr;
    void* operatorHandle = nullptr;
    LegacyInvoke invoke = nullptr;
};

inline thread_local void* pendingGuardDelegate = nullptr;
inline thread_local void* officialTfliteLibrary = nullptr;

inline void* systemDlopen(const char* name, int flags) {
    return ::dlopen(name, flags);
}

inline void* systemDlsym(void* library, const char* name) {
    return ::dlsym(library, name);
}

template <typename T>
T officialSymbol(void* library, const char* name) {
    void* symbol = systemDlsym(library, name);
    if (symbol == nullptr) {
        throw std::runtime_error(std::string("missing official TFLite symbol ") + name);
    }
    return reinterpret_cast<T>(symbol);
}

inline int operatorInvoke(void* userData, void* context, void* node) {
    auto* registration = static_cast<Registration*>(userData);
    return registration->invoke(context, node);
}

inline void* registrationCreate(int builtinCode, const char* name, int version) {
    if (officialTfliteLibrary == nullptr) {
        throw std::runtime_error("official TFLite library is unavailable during operator registration");
    }
    auto* registration = new Registration();
    registration->library = officialTfliteLibrary;
    try {
        const auto create = officialSymbol<void* (*)(int, const char*, int, void*)>(
            registration->library, "TfLiteOperatorCreate");
        registration->operatorHandle = create(builtinCode, name, version, registration);
        if (registration->operatorHandle == nullptr) {
            throw std::runtime_error(std::string("TfLiteOperatorCreate failed for ") + name);
        }
        return registration;
    } catch (...) {
        delete registration;
        throw;
    }
}

inline void registrationDelete(void* opaqueRegistration) {
    auto* registration = static_cast<Registration*>(opaqueRegistration);
    if (registration == nullptr) return;
    if (registration->operatorHandle != nullptr) {
        const auto destroy = officialSymbol<void (*)(void*)>(
            registration->library, "TfLiteOperatorDelete");
        destroy(registration->operatorHandle);
    }
    delete registration;
}

inline void registrationSetInvoke(void* opaqueRegistration, LegacyInvoke invoke) {
    auto* registration = static_cast<Registration*>(opaqueRegistration);
    registration->invoke = invoke;
    const auto setInvoke = officialSymbol<int (*)(void*, OperatorInvoke)>(
        registration->library, "TfLiteOperatorSetInvokeWithData");
    if (setInvoke(registration->operatorHandle, operatorInvoke) != 0) {
        throw std::runtime_error("TfLiteOperatorSetInvokeWithData failed");
    }
}

inline void optionsAddRegistration(void* options, void* opaqueRegistration) {
    auto* registration = static_cast<Registration*>(opaqueRegistration);
    const auto addOperator = officialSymbol<void (*)(void*, void*)>(
        registration->library, "TfLiteInterpreterOptionsAddOperator");
    addOperator(options, registration->operatorHandle);
}

inline void optionsAddDelegate(void* options, void* primaryDelegate) {
    const auto addDelegate = officialSymbol<void (*)(void*, void*)>(
        officialTfliteLibrary, "TfLiteInterpreterOptionsAddDelegate");
    addDelegate(options, primaryDelegate);
    if (pendingGuardDelegate != nullptr) addDelegate(options, pendingGuardDelegate);
}

inline void* dlopenAdapter(const char* name, int flags) {
    const char* actual = std::strcmp(name, "libtensorflowlite_jni_mtk.so") == 0
        ? "libtensorflowlite_jni.so"
        : name;
    void* library = systemDlopen(actual, flags);
    if (library != nullptr && std::strcmp(actual, "libtensorflowlite_jni.so") == 0) {
        officialTfliteLibrary = library;
    }
    return library;
}

inline void* dlsymAdapter(void* library, const char* name) {
    if (std::strcmp(name, "TfLiteInterpreterOptionsAddRegistrationExternal") == 0) {
        return reinterpret_cast<void*>(optionsAddRegistration);
    }
    if (std::strcmp(name, "TfLiteRegistrationExternalCreate") == 0) {
        return reinterpret_cast<void*>(registrationCreate);
    }
    if (std::strcmp(name, "TfLiteRegistrationExternalDelete") == 0) {
        return reinterpret_cast<void*>(registrationDelete);
    }
    if (std::strcmp(name, "TfLiteRegistrationExternalSetInvoke") == 0) {
        return reinterpret_cast<void*>(registrationSetInvoke);
    }
    if (std::strcmp(name, "TfLiteInterpreterOptionsAddDelegate") == 0) {
        return reinterpret_cast<void*>(optionsAddDelegate);
    }
    return systemDlsym(library, name);
}

class GuardDelegateScope {
public:
    explicit GuardDelegateScope(void* guard) { pendingGuardDelegate = guard; }

    ~GuardDelegateScope() { pendingGuardDelegate = nullptr; }
};

}  // namespace gvcrt_large_gpu_shim

#define dlopen gvcrt_large_gpu_shim::dlopenAdapter
#define dlsym gvcrt_large_gpu_shim::dlsymAdapter
