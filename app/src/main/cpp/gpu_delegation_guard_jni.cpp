#include <jni.h>
#include <android/log.h>
#include <cstring>
#include <string>

#include "tensorflow/lite/c/common.h"

namespace {

struct Guard {
    TfLiteDelegate delegate{};
    TfLiteDelegate* gpu = nullptr;
    std::string model;
    bool allowBuiltinCpuFallback = false;
    bool inspected = false;
    std::string failure;
};

// Only exact rANS registrations in their corresponding merged models.
// Ordinary NN models and arbitrary CUSTOM ops never receive a CPU exception.
bool isAllowedNativeRans(const Guard& guard, const TfLiteNode& node,
                         const TfLiteRegistration& registration) {
    constexpr int kBuiltinCustom = 32;
    if (node.delegate != nullptr || registration.builtin_code != kBuiltinCustom ||
        registration.version != 1 || registration.custom_name == nullptr) return false;
    const auto is = [&](const char* name) { return std::strcmp(registration.custom_name, name) == 0; };
    if (guard.model == "i_entropy_prior_merged_rans.tflite") return is("GVC_RT_RANS_ENCODE");
    if (guard.model == "p_entropy_prior_merged_rans.tflite") return is("GVC_RT_P_RANS_ENCODE");
    if (guard.model == "i_entropy_decode_merged_rans.tflite") {
        return is("GVC_RT_RANS_DECODE_Z") || is("GVC_RT_RANS_DECODE_Y");
    }
    if (guard.model == "p_entropy_decode_merged_rans.tflite") {
        return is("GVC_RT_P_RANS_DECODE_Z") || is("GVC_RT_P_RANS_DECODE_Y");
    }
    if (guard.model == "entropy_encode_fused.tflite") {
        return is("GVC_RT_SMALL_RANS_ENCODE");
    }
    if (guard.model == "entropy_decode_fused.tflite") {
        return is("GVC_RT_SMALL_RANS_DECODE_Z") || is("GVC_RT_SMALL_RANS_DECODE_Y");
    }
    return false;
}

TfLiteStatus inspectPlan(TfLiteContext* context, TfLiteDelegate* delegate) {
    auto* guard = static_cast<Guard*>(delegate->data_);
    guard->inspected = true;
    TfLiteIntArray* plan = nullptr;
    if (context->GetExecutionPlan(context, &plan) != kTfLiteOk) {
        guard->failure = "execution_plan_unavailable";
        return kTfLiteOk;
    }
    if (plan->size == 0) guard->failure = "empty_execution_plan";
    int gpuNodes = 0;
    int allowedRansNodes = 0;
    int allowedBuiltinCpuNodes = 0;
    int unexpectedCpuNodes = 0;
    for (int i = 0; i < plan->size; ++i) {
        TfLiteNode* node = nullptr;
        TfLiteRegistration* registration = nullptr;
        if (context->GetNodeAndRegistration(context, plan->data[i], &node, &registration) != kTfLiteOk) {
            guard->failure = "node_registration_unavailable";
            break;
        }
        if (node->delegate == guard->gpu) {
            ++gpuNodes;
        } else if (isAllowedNativeRans(*guard, *node, *registration)) {
            ++allowedRansNodes;
            __android_log_print(ANDROID_LOG_INFO, "GVC_RT_CLEAN",
                "gpu_allowed_native_rans model=%s node=%d custom_op=%s",
                guard->model.c_str(), plan->data[i], registration->custom_name);
        } else if (guard->allowBuiltinCpuFallback && node->delegate == nullptr &&
                   registration->builtin_code != 32) {
            ++allowedBuiltinCpuNodes;
        } else {
            ++unexpectedCpuNodes;
            const std::string op = registration->custom_name != nullptr
                ? registration->custom_name
                : "builtin_code_" + std::to_string(registration->builtin_code);
            const std::string detail = "unsupported_op=" + op + " node=" + std::to_string(plan->data[i]);
            if (guard->failure.empty()) guard->failure = detail;
            __android_log_print(ANDROID_LOG_ERROR, "GVC_RT_CLEAN",
                "gpu_delegate_unsupported model=%s backend=tflite_gpu %s",
                guard->model.c_str(), detail.c_str());
        }
    }
    // Counts are execution-plan nodes after delegation: a GPU partition counts as one.
    __android_log_print(guard->failure.empty() ? ANDROID_LOG_INFO : ANDROID_LOG_ERROR, "GVC_RT_CLEAN",
        "gpu_delegation_checked model=%s backend=tflite_gpu gpu_nodes=%d allowed_native_rans_nodes=%d allowed_builtin_cpu_fallback_nodes=%d unexpected_cpu_nodes=%d cpu_fallback_allowed=%s check_ok=%s",
        guard->model.c_str(), gpuNodes, allowedRansNodes, allowedBuiltinCpuNodes, unexpectedCpuNodes,
        guard->allowBuiltinCpuFallback ? "true" : "false",
        guard->failure.empty() ? "true" : "false");
    // The Java owner checks this result before invoking and closes a rejected interpreter.
    // This delegate never replaces or executes any node.
    return kTfLiteOk;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_gvcrt_clean_GpuDelegationGuard_nativeCreate(
    JNIEnv* env, jclass, jlong gpu, jstring model, jboolean allowBuiltinCpuFallback) {
    const char* name = env->GetStringUTFChars(model, nullptr);
    if (name == nullptr) return 0;
    auto* guard = new Guard();
    guard->model = name;
    env->ReleaseStringUTFChars(model, name);
    guard->gpu = reinterpret_cast<TfLiteDelegate*>(gpu);
    guard->allowBuiltinCpuFallback = allowBuiltinCpuFallback == JNI_TRUE;
    guard->delegate.data_ = guard;
    guard->delegate.Prepare = inspectPlan;
    return reinterpret_cast<jlong>(guard);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_gvcrt_clean_GpuDelegationGuard_nativeDelegate(JNIEnv*, jclass, jlong handle) {
    return reinterpret_cast<jlong>(&reinterpret_cast<Guard*>(handle)->delegate);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_gvcrt_clean_GpuDelegationGuard_nativeFailure(JNIEnv* env, jclass, jlong handle) {
    const auto* guard = reinterpret_cast<Guard*>(handle);
    return env->NewStringUTF(guard->inspected ? guard->failure.c_str() : "execution_plan_not_checked");
}

extern "C" JNIEXPORT void JNICALL
Java_com_gvcrt_clean_GpuDelegationGuard_nativeClose(JNIEnv*, jclass, jlong handle) {
    delete reinterpret_cast<Guard*>(handle);
}
