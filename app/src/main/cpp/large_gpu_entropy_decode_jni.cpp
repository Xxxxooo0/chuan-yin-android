#define Java_com_gvcrt_clean_IEntropyRansDecodeMergedRuntime_nativeCreate \
    Java_com_gvcrt_clean_LargeGpuIncludedIEntropyDecode_nativeCreate
#define Java_com_gvcrt_clean_IEntropyRansDecodeMergedRuntime_nativeRun \
    Java_com_gvcrt_clean_LargeGpuIncludedIEntropyDecode_nativeRun
#define Java_com_gvcrt_clean_IEntropyRansDecodeMergedRuntime_nativeClose \
    Java_com_gvcrt_clean_LargeGpuIncludedIEntropyDecode_nativeClose
#define Java_com_gvcrt_clean_PEntropyRansDecodeMergedRuntime_nativeCreate \
    Java_com_gvcrt_clean_LargeGpuIncludedPEntropyDecode_nativeCreate
#define Java_com_gvcrt_clean_PEntropyRansDecodeMergedRuntime_nativeRun \
    Java_com_gvcrt_clean_LargeGpuIncludedPEntropyDecode_nativeRun
#define Java_com_gvcrt_clean_PEntropyRansDecodeMergedRuntime_nativeClose \
    Java_com_gvcrt_clean_LargeGpuIncludedPEntropyDecode_nativeClose

#include "large_gpu_tflite_registration_shim.h"
#include "i_entropy_rans_decode_custom_op_jni.cpp"

#undef dlopen
#undef dlsym
#undef Java_com_gvcrt_clean_IEntropyRansDecodeMergedRuntime_nativeCreate
#undef Java_com_gvcrt_clean_IEntropyRansDecodeMergedRuntime_nativeRun
#undef Java_com_gvcrt_clean_IEntropyRansDecodeMergedRuntime_nativeClose
#undef Java_com_gvcrt_clean_PEntropyRansDecodeMergedRuntime_nativeCreate
#undef Java_com_gvcrt_clean_PEntropyRansDecodeMergedRuntime_nativeRun
#undef Java_com_gvcrt_clean_PEntropyRansDecodeMergedRuntime_nativeClose

namespace {

struct LargeGpuDecodeHandle {
    Runtime* runtime;
    int kind;
};

LargeGpuDecodeHandle* createLargeGpuDecode(
    const std::string& path, int kind, void* primaryDelegate, void* guardDelegate) {
    if (kind != 0 && kind != 1) throw std::runtime_error("invalid Large GPU entropy decode kind");
    if (primaryDelegate == nullptr) throw std::runtime_error("Large entropy primary delegate handle is null");
    gvcrt_large_gpu_shim::GuardDelegateScope guardScope(guardDelegate);
    auto* runtime = createRuntime(
        path,
        primaryDelegate,
        kind == 0 ? kIDecodeZOp : kPDecodeZOp,
        kind == 0 ? kIDecodeYOp : kPDecodeYOp,
        kind == 0 ? 2 : 3,
        kind == 0 ? 10 : 6);
    return new LargeGpuDecodeHandle{runtime, kind};
}

jobjectArray runLargeGpuDecode(
    JNIEnv* env, LargeGpuDecodeHandle* handle, jobjectArray inputs, jint qp, jint outputMode) {
    if (handle == nullptr || handle->runtime == nullptr) {
        throw std::runtime_error("Large GPU entropy decoder is closed");
    }
    const int expectedInputs = handle->kind == 0 ? 1 : 2;
    if (inputs == nullptr || env->GetArrayLength(inputs) != expectedInputs) {
        throw std::runtime_error("Large GPU entropy decode input count mismatch");
    }
    auto payload = static_cast<jbyteArray>(env->GetObjectArrayElement(inputs, 0));
    auto context = expectedInputs == 2
        ? static_cast<jbyteArray>(env->GetObjectArrayElement(inputs, 1))
        : nullptr;
    if (payload == nullptr || (expectedInputs == 2 && context == nullptr)) {
        if (payload != nullptr) env->DeleteLocalRef(payload);
        if (context != nullptr) env->DeleteLocalRef(context);
        throw std::runtime_error("Large GPU entropy decode input is null");
    }
    jobjectArray result = runRuntime(env, handle->runtime, payload, context, qp, outputMode);
    env->DeleteLocalRef(payload);
    if (context != nullptr) env->DeleteLocalRef(context);
    return result;
}

void closeLargeGpuDecode(LargeGpuDecodeHandle* handle) {
    if (handle == nullptr) return;
    destroyRuntime(handle->runtime);
    delete handle;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_gvcrt_clean_LargeEntropyGpuRuntime_nativeCreateDecode(
    JNIEnv* env, jclass, jstring modelPath, jint kind, jlong gpuDelegate, jlong guardDelegate) {
    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    if (path == nullptr) return 0;
    const std::string model(path);
    env->ReleaseStringUTFChars(modelPath, path);
    try {
        return reinterpret_cast<jlong>(createLargeGpuDecode(
            model,
            kind,
            reinterpret_cast<void*>(static_cast<uintptr_t>(gpuDelegate)),
            reinterpret_cast<void*>(static_cast<uintptr_t>(guardDelegate))));
    } catch (const std::exception& error) {
        throwJava(env, error.what());
        return 0;
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_gvcrt_clean_LargeEntropyGpuRuntime_nativeRunDecode(
    JNIEnv* env, jclass, jlong handle, jobjectArray inputs, jint qp, jint outputMode) {
    try {
        return runLargeGpuDecode(
            env, reinterpret_cast<LargeGpuDecodeHandle*>(handle), inputs, qp, outputMode);
    } catch (const std::exception& error) {
        throwJava(env, error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_gvcrt_clean_LargeEntropyGpuRuntime_nativeCloseDecode(
    JNIEnv*, jclass, jlong handle) {
    closeLargeGpuDecode(reinterpret_cast<LargeGpuDecodeHandle*>(handle));
}
