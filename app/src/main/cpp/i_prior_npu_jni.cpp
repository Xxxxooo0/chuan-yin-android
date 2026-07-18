#include <jni.h>

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kChannels = 256;
constexpr int kStageChannels = 64;
constexpr int kHeight = 16;
constexpr int kWidth = 32;
constexpr int kSpatial = kHeight * kWidth;
constexpr int kElements = kChannels * kSpatial;
constexpr int kCommonChannels = 514;

struct PriorState {
    std::vector<float> yScaled;
    std::vector<float> qDec;
    std::vector<float> yHat;
    std::vector<float> stage0Scales;
    std::vector<float> stage0Means;
    float forceZeroThreshold;
};

[[noreturn]] void throwIllegalState(JNIEnv* env, const char* message) {
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), message);
    throw std::runtime_error(message);
}

PriorState* fromHandle(JNIEnv* env, jlong handle) {
    if (handle == 0) {
        throwIllegalState(env, "I4xPriorNative is closed");
    }
    return reinterpret_cast<PriorState*>(handle);
}

std::vector<float> copyFloatArray(JNIEnv* env, jfloatArray array, int expected, const char* label) {
    if (env->GetArrayLength(array) != expected) {
        throwIllegalState(env, label);
    }
    std::vector<float> result(expected);
    env->GetFloatArrayRegion(array, 0, expected, result.data());
    return result;
}

jfloatArray toFloatArray(JNIEnv* env, const std::vector<float>& values) {
    auto output = env->NewFloatArray(static_cast<jsize>(values.size()));
    if (output == nullptr) {
        throw std::bad_alloc();
    }
    env->SetFloatArrayRegion(output, 0, static_cast<jsize>(values.size()), values.data());
    return output;
}

jobjectArray toStageResult(JNIEnv* env, const std::vector<float>& yQ, const std::vector<float>& scales, const std::vector<float>& yHat) {
    auto floatArrayClass = env->FindClass("[F");
    auto result = env->NewObjectArray(3, floatArrayClass, nullptr);
    env->SetObjectArrayElement(result, 0, toFloatArray(env, yQ));
    env->SetObjectArrayElement(result, 1, toFloatArray(env, scales));
    env->SetObjectArrayElement(result, 2, toFloatArray(env, yHat));
    return result;
}

float sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

bool isActive(int stage, int channel, int height, int width) {
    constexpr int kMaskPattern[4][4] = {
        {0, 1, 2, 3},
        {3, 2, 1, 0},
        {2, 3, 0, 1},
        {1, 0, 3, 2},
    };
    const int group = channel / kStageChannels;
    const int spatialPattern = ((height & 1) << 1) | (width & 1);
    return spatialPattern == kMaskPattern[stage][group];
}

jobjectArray processStage(JNIEnv* env, PriorState& state, const std::vector<float>& scales, const std::vector<float>& means, int stage) {
    if (stage < 0 || stage > 3 || scales.size() != kElements || means.size() != kElements) {
        throwIllegalState(env, "invalid I prior stage input");
    }
    std::vector<float> yQFull(kElements, 0.0f);
    std::vector<float> scalesFull(kElements, 0.0f);
    std::vector<float> yHatCurrent(kElements, 0.0f);
    std::fesetround(FE_TONEAREST);
    for (int channel = 0; channel < kChannels; ++channel) {
        for (int height = 0; height < kHeight; ++height) {
            for (int width = 0; width < kWidth; ++width) {
                const int index = (channel * kHeight + height) * kWidth + width;
                if (!isActive(stage, channel, height, width)) {
                    continue;
                }
                const float scale = scales[index];
                const float mean = means[index];
                float quantized = std::nearbyint(state.yScaled[index] - mean);
                if (scale <= state.forceZeroThreshold) {
                    quantized = 0.0f;
                }
                quantized = std::max(-128.0f, std::min(127.0f, quantized));
                yQFull[index] = quantized;
                scalesFull[index] = scale;
                yHatCurrent[index] = quantized + mean;
                state.yHat[index] += yHatCurrent[index];
            }
        }
    }
    std::vector<float> yQPacked(kStageChannels * kSpatial, 0.0f);
    std::vector<float> scalesPacked(kStageChannels * kSpatial, 0.0f);
    for (int channel = 0; channel < kStageChannels; ++channel) {
        for (int spatial = 0; spatial < kSpatial; ++spatial) {
            for (int group = 0; group < 4; ++group) {
                const int source = (channel + group * kStageChannels) * kSpatial + spatial;
                const int target = channel * kSpatial + spatial;
                yQPacked[target] += yQFull[source];
                scalesPacked[target] += scalesFull[source];
            }
        }
    }
    return toStageResult(env, yQPacked, scalesPacked, state.yHat);
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_gvcrt_clean_I4xPriorNative_nativeCreate(
    JNIEnv* env,
    jclass,
    jfloatArray y,
    jfloatArray commonParams,
    jfloat forceZeroThreshold)
{
    try {
        const auto yValues = copyFloatArray(env, y, kElements, "i_y_pre_prior shape mismatch");
        const auto common = copyFloatArray(env, commonParams, kCommonChannels * kSpatial, "i_common_params shape mismatch");
        auto state = std::make_unique<PriorState>();
        state->yScaled.resize(kElements);
        state->qDec.resize(kSpatial);
        state->yHat.assign(kElements, 0.0f);
        state->stage0Scales.resize(kElements);
        state->stage0Means.resize(kElements);
        state->forceZeroThreshold = forceZeroThreshold;
        for (int spatial = 0; spatial < kSpatial; ++spatial) {
            const float qEnc = sigmoid(common[spatial]) * 1.5f + 0.5f;
            state->qDec[spatial] = sigmoid(common[kSpatial + spatial]) * 1.5f + 0.5f;
            for (int channel = 0; channel < kChannels; ++channel) {
                const int index = channel * kSpatial + spatial;
                state->yScaled[index] = yValues[index] * qEnc;
                state->stage0Scales[index] = common[(channel + 2) * kSpatial + spatial];
                state->stage0Means[index] = common[(channel + 2 + kChannels) * kSpatial + spatial];
            }
        }
        return reinterpret_cast<jlong>(state.release());
    } catch (const std::exception&) {
        return 0;
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_gvcrt_clean_I4xPriorNative_nativeRunStage0(JNIEnv* env, jclass, jlong handle) {
    try {
        auto* state = fromHandle(env, handle);
        return processStage(env, *state, state->stage0Scales, state->stage0Means, 0);
    } catch (const std::exception&) {
        return nullptr;
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_gvcrt_clean_I4xPriorNative_nativeRunStage(
    JNIEnv* env,
    jclass,
    jlong handle,
    jfloatArray scales,
    jfloatArray means,
    jint stage)
{
    try {
        auto* state = fromHandle(env, handle);
        return processStage(
            env,
            *state,
            copyFloatArray(env, scales, kElements, "I prior scales shape mismatch"),
            copyFloatArray(env, means, kElements, "I prior means shape mismatch"),
            static_cast<int>(stage));
    } catch (const std::exception&) {
        return nullptr;
    }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_gvcrt_clean_I4xPriorNative_nativeFinish(JNIEnv* env, jclass, jlong handle) {
    try {
        auto* state = fromHandle(env, handle);
        std::vector<float> result(kElements);
        for (int channel = 0; channel < kChannels; ++channel) {
            for (int spatial = 0; spatial < kSpatial; ++spatial) {
                result[channel * kSpatial + spatial] = state->yHat[channel * kSpatial + spatial] * state->qDec[spatial];
            }
        }
        return toFloatArray(env, result);
    } catch (const std::exception&) {
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_gvcrt_clean_I4xPriorNative_nativeRelease(JNIEnv*, jclass, jlong handle) {
    delete reinterpret_cast<PriorState*>(handle);
}

// Kotlin declares these external methods in I4xPriorNative.Companion. Keep the
// original outer-class exports for compatibility and provide the generated
// Companion names used by the current Kotlin declaration.
extern "C" JNIEXPORT jlong JNICALL
Java_com_gvcrt_clean_I4xPriorNative_00024Companion_nativeCreate(
    JNIEnv* env, jclass clazz, jfloatArray y, jfloatArray commonParams, jfloat threshold) {
    return Java_com_gvcrt_clean_I4xPriorNative_nativeCreate(env, clazz, y, commonParams, threshold);
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_gvcrt_clean_I4xPriorNative_00024Companion_nativeRunStage0(JNIEnv* env, jclass clazz, jlong handle) {
    return Java_com_gvcrt_clean_I4xPriorNative_nativeRunStage0(env, clazz, handle);
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_gvcrt_clean_I4xPriorNative_00024Companion_nativeRunStage(
    JNIEnv* env, jclass clazz, jlong handle, jfloatArray scales, jfloatArray means, jint stage) {
    return Java_com_gvcrt_clean_I4xPriorNative_nativeRunStage(env, clazz, handle, scales, means, stage);
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_gvcrt_clean_I4xPriorNative_00024Companion_nativeFinish(JNIEnv* env, jclass clazz, jlong handle) {
    return Java_com_gvcrt_clean_I4xPriorNative_nativeFinish(env, clazz, handle);
}

extern "C" JNIEXPORT void JNICALL
Java_com_gvcrt_clean_I4xPriorNative_00024Companion_nativeRelease(JNIEnv* env, jclass clazz, jlong handle) {
    Java_com_gvcrt_clean_I4xPriorNative_nativeRelease(env, clazz, handle);
}
