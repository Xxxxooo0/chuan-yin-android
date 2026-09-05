#define Java_com_gvcrt_clean_IEntropyRansMergedRuntime_nativeCreate \
    Java_com_gvcrt_clean_LargeGpuIncludedIEntropyEncode_nativeCreate
#define Java_com_gvcrt_clean_IEntropyRansMergedRuntime_nativeRun \
    Java_com_gvcrt_clean_LargeGpuIncludedIEntropyEncode_nativeRun
#define Java_com_gvcrt_clean_IEntropyRansMergedRuntime_nativeClose \
    Java_com_gvcrt_clean_LargeGpuIncludedIEntropyEncode_nativeClose
#define Java_com_gvcrt_clean_PEntropyRansMergedRuntime_nativeCreate \
    Java_com_gvcrt_clean_LargeGpuIncludedPEntropyEncode_nativeCreate
#define Java_com_gvcrt_clean_PEntropyRansMergedRuntime_nativeRun \
    Java_com_gvcrt_clean_LargeGpuIncludedPEntropyEncode_nativeRun
#define Java_com_gvcrt_clean_PEntropyRansMergedRuntime_nativeClose \
    Java_com_gvcrt_clean_LargeGpuIncludedPEntropyEncode_nativeClose

#include "large_gpu_tflite_registration_shim.h"
#include "i_entropy_rans_custom_op_jni.cpp"

#undef dlopen
#undef dlsym
#undef Java_com_gvcrt_clean_IEntropyRansMergedRuntime_nativeCreate
#undef Java_com_gvcrt_clean_IEntropyRansMergedRuntime_nativeRun
#undef Java_com_gvcrt_clean_IEntropyRansMergedRuntime_nativeClose
#undef Java_com_gvcrt_clean_PEntropyRansMergedRuntime_nativeCreate
#undef Java_com_gvcrt_clean_PEntropyRansMergedRuntime_nativeRun
#undef Java_com_gvcrt_clean_PEntropyRansMergedRuntime_nativeClose

namespace {

struct LargeGpuEncodeHandle {
    Runtime* runtime;
    int kind;
};

LargeGpuEncodeHandle* createLargeGpuEncode(
    const std::string& path, int kind, void* primaryDelegate, void* guardDelegate) {
    if (kind != 0 && kind != 1) throw std::runtime_error("invalid Large GPU entropy encode kind");
    if (primaryDelegate == nullptr) throw std::runtime_error("Large entropy primary delegate handle is null");
    gvcrt_large_gpu_shim::GuardDelegateScope guardScope(guardDelegate);
    auto* runtime = createRuntime(
        path,
        primaryDelegate,
        kind == 0 ? kICustomOpName : kPCustomOpName,
        kind == 0 ? 1 : 2,
        kind == 0 ? 12 : 8);
    return new LargeGpuEncodeHandle{runtime, kind};
}

jobjectArray runLargeGpuEncode(
    JNIEnv* env, LargeGpuEncodeHandle* handle, jobjectArray inputs, jint qp, jint outputMode) {
    if (handle == nullptr || handle->runtime == nullptr) {
        throw std::runtime_error("Large GPU entropy encoder is closed");
    }
    if (qp < 0 || qp > 9) throw std::runtime_error("Large GPU entropy encode QP must be in [0,9]");
    const int expectedInputs = handle->kind == 0 ? 1 : 2;
    if (inputs == nullptr || env->GetArrayLength(inputs) != expectedInputs) {
        throw std::runtime_error("Large GPU entropy encode input count mismatch");
    }
    handle->runtime->currentQp = qp;
    for (int index = 0; index < expectedInputs; ++index) {
        auto input = static_cast<jbyteArray>(env->GetObjectArrayElement(inputs, index));
        if (input == nullptr) throw std::runtime_error("Large GPU entropy encode input is null");
        copyJavaInput(env, handle->runtime, index, input);
        env->DeleteLocalRef(input);
    }
    if (invokeRuntime(handle->runtime) != kTfLiteOk) {
        throw std::runtime_error("Large GPU entropy encode invoke failed");
    }
    return collectOutputs(env, handle->runtime, outputMode);
}

void closeLargeGpuEncode(LargeGpuEncodeHandle* handle) {
    if (handle == nullptr) return;
    destroyRuntime(handle->runtime);
    delete handle;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_gvcrt_clean_LargeEntropyGpuRuntime_nativeCreateEncode(
    JNIEnv* env, jclass, jstring modelPath, jint kind, jlong gpuDelegate, jlong guardDelegate) {
    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    if (path == nullptr) return 0;
    const std::string model(path);
    env->ReleaseStringUTFChars(modelPath, path);
    try {
        return reinterpret_cast<jlong>(createLargeGpuEncode(
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
Java_com_gvcrt_clean_LargeEntropyGpuRuntime_nativeRunEncode(
    JNIEnv* env, jclass, jlong handle, jobjectArray inputs, jint qp, jint outputMode) {
    try {
        return runLargeGpuEncode(
            env, reinterpret_cast<LargeGpuEncodeHandle*>(handle), inputs, qp, outputMode);
    } catch (const std::exception& error) {
        throwJava(env, error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_gvcrt_clean_LargeEntropyGpuRuntime_nativeCloseEncode(
    JNIEnv*, jclass, jlong handle) {
    closeLargeGpuEncode(reinterpret_cast<LargeGpuEncodeHandle*>(handle));
}
