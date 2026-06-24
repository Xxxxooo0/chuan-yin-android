#include <jni.h>

#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <vector>

#include "rans/rans.h"

namespace {

struct CdfData {
    std::shared_ptr<std::vector<std::vector<int32_t>>> cdfs;
    std::shared_ptr<std::vector<int32_t>> lengths;
    std::shared_ptr<std::vector<int32_t>> offsets;
};

struct RansSession {
    RansEncoderLib encoder;
    RansDecoderLib decoder;
    int gaussianGroup;
    int zGroup;

    RansSession(const CdfData& gaussian, const CdfData& z)
        : gaussianGroup(encoder.add_cdf(gaussian.cdfs, gaussian.lengths, gaussian.offsets))
        , zGroup(encoder.add_cdf(z.cdfs, z.lengths, z.offsets))
    {
        const int decoderGaussian = decoder.add_cdf(gaussian.cdfs, gaussian.lengths, gaussian.offsets);
        const int decoderZ = decoder.add_cdf(z.cdfs, z.lengths, z.offsets);
        if (decoderGaussian != gaussianGroup || decoderZ != zGroup) {
            throw std::runtime_error("rANS encoder/decoder CDF group mismatch");
        }
    }
};

void throwIllegalState(JNIEnv* env, const char* message)
{
    jclass exceptionClass = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(exceptionClass, message);
    env->DeleteLocalRef(exceptionClass);
}

CdfData copyCdf(JNIEnv* env, jintArray cdfValues, jint rows, jint stride, jintArray lengths, jintArray offsets)
{
    if (rows <= 0 || stride <= 1 || env->GetArrayLength(cdfValues) != rows * stride ||
        env->GetArrayLength(lengths) != rows || env->GetArrayLength(offsets) != rows) {
        throw std::invalid_argument("invalid rANS CDF table shape");
    }

    std::vector<jint> rawCdf(static_cast<size_t>(rows) * stride);
    std::vector<jint> rawLengths(rows);
    std::vector<jint> rawOffsets(rows);
    env->GetIntArrayRegion(cdfValues, 0, static_cast<jsize>(rawCdf.size()), rawCdf.data());
    env->GetIntArrayRegion(lengths, 0, rows, rawLengths.data());
    env->GetIntArrayRegion(offsets, 0, rows, rawOffsets.data());

    auto copiedCdfs = std::make_shared<std::vector<std::vector<int32_t>>>(rows);
    for (int row = 0; row < rows; ++row) {
        const auto begin = rawCdf.begin() + static_cast<size_t>(row) * stride;
        copiedCdfs->at(row) = std::vector<int32_t>(begin, begin + stride);
    }
    auto copiedLengths = std::make_shared<std::vector<int32_t>>(rawLengths.begin(), rawLengths.end());
    auto copiedOffsets = std::make_shared<std::vector<int32_t>>(rawOffsets.begin(), rawOffsets.end());
    return {copiedCdfs, copiedLengths, copiedOffsets};
}

std::shared_ptr<std::vector<int8_t>> copyI8(JNIEnv* env, jbyteArray source)
{
    const jsize size = env->GetArrayLength(source);
    std::vector<jbyte> raw(size);
    env->GetByteArrayRegion(source, 0, size, raw.data());
    return std::make_shared<std::vector<int8_t>>(raw.begin(), raw.end());
}

std::shared_ptr<std::vector<uint8_t>> copyU8(JNIEnv* env, jbyteArray source)
{
    const jsize size = env->GetArrayLength(source);
    std::vector<jbyte> raw(size);
    env->GetByteArrayRegion(source, 0, size, raw.data());
    auto result = std::make_shared<std::vector<uint8_t>>(size);
    for (jsize i = 0; i < size; ++i) {
        result->at(i) = static_cast<uint8_t>(raw[i]);
    }
    return result;
}

std::shared_ptr<std::vector<int16_t>> copyI16(JNIEnv* env, jshortArray source)
{
    const jsize size = env->GetArrayLength(source);
    std::vector<jshort> raw(size);
    env->GetShortArrayRegion(source, 0, size, raw.data());
    return std::make_shared<std::vector<int16_t>>(raw.begin(), raw.end());
}

jbyteArray makeByteArray(JNIEnv* env, const std::vector<int8_t>& values)
{
    jbyteArray output = env->NewByteArray(static_cast<jsize>(values.size()));
    env->SetByteArrayRegion(output, 0, static_cast<jsize>(values.size()),
                            reinterpret_cast<const jbyte*>(values.data()));
    return output;
}

jbyteArray makeByteArray(JNIEnv* env, const std::vector<uint8_t>& values)
{
    jbyteArray output = env->NewByteArray(static_cast<jsize>(values.size()));
    env->SetByteArrayRegion(output, 0, static_cast<jsize>(values.size()),
                            reinterpret_cast<const jbyte*>(values.data()));
    return output;
}

RansSession* checkedSession(JNIEnv* env, jlong handle)
{
    if (handle == 0) {
        throw std::invalid_argument("rANS session is closed");
    }
    return reinterpret_cast<RansSession*>(handle);
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_gvcrt_clean_NativeRans_nativeCreate(
    JNIEnv* env,
    jclass,
    jintArray gaussianCdf,
    jint gaussianRows,
    jint gaussianStride,
    jintArray gaussianLengths,
    jintArray gaussianOffsets,
    jintArray zCdf,
    jint zRows,
    jint zStride,
    jintArray zLengths,
    jintArray zOffsets)
{
    try {
        const CdfData gaussian = copyCdf(env, gaussianCdf, gaussianRows, gaussianStride, gaussianLengths, gaussianOffsets);
        const CdfData z = copyCdf(env, zCdf, zRows, zStride, zLengths, zOffsets);
        return reinterpret_cast<jlong>(new RansSession(gaussian, z));
    } catch (const std::exception& error) {
        throwIllegalState(env, error.what());
        return 0;
    }
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_gvcrt_clean_NativeRans_nativeEncode(
    JNIEnv* env,
    jclass,
    jlong handle,
    jbyteArray zSymbols,
    jint zStartOffset,
    jint zPerChannelSize,
    jobjectArray packedYStages)
{
    try {
        RansSession* session = checkedSession(env, handle);
        session->encoder.reset();
        session->encoder.encode_z(copyI8(env, zSymbols), session->zGroup, zStartOffset, zPerChannelSize);
        const jsize stageCount = env->GetArrayLength(packedYStages);
        for (jsize stage = 0; stage < stageCount; ++stage) {
            auto* symbols = static_cast<jshortArray>(env->GetObjectArrayElement(packedYStages, stage));
            session->encoder.encode_y(copyI16(env, symbols), session->gaussianGroup);
            env->DeleteLocalRef(symbols);
        }
        session->encoder.flush();
        return makeByteArray(env, *session->encoder.get_encoded_stream());
    } catch (const std::exception& error) {
        throwIllegalState(env, error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_gvcrt_clean_NativeRans_nativeDecode(
    JNIEnv* env,
    jclass,
    jlong handle,
    jbyteArray payload,
    jint zTotalSize,
    jint zStartOffset,
    jint zPerChannelSize,
    jobjectArray yIndexes)
{
    try {
        RansSession* session = checkedSession(env, handle);
        session->decoder.set_stream(copyU8(env, payload));
        session->decoder.decode_z(zTotalSize, session->zGroup, zStartOffset, zPerChannelSize);

        const jsize stageCount = env->GetArrayLength(yIndexes);
        jclass byteArrayClass = env->FindClass("[B");
        jobjectArray result = env->NewObjectArray(stageCount + 1, byteArrayClass, nullptr);
        env->DeleteLocalRef(byteArrayClass);

        jbyteArray decodedZ = makeByteArray(env, *session->decoder.get_decoded_tensor());
        env->SetObjectArrayElement(result, 0, decodedZ);
        env->DeleteLocalRef(decodedZ);

        for (jsize stage = 0; stage < stageCount; ++stage) {
            auto* indexes = static_cast<jbyteArray>(env->GetObjectArrayElement(yIndexes, stage));
            session->decoder.decode_y(copyU8(env, indexes), session->gaussianGroup);
            env->DeleteLocalRef(indexes);
            jbyteArray decoded = makeByteArray(env, *session->decoder.get_decoded_tensor());
            env->SetObjectArrayElement(result, stage + 1, decoded);
            env->DeleteLocalRef(decoded);
        }
        return result;
    } catch (const std::exception& error) {
        throwIllegalState(env, error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_gvcrt_clean_NativeRans_nativeRelease(JNIEnv*, jclass, jlong handle)
{
    delete reinterpret_cast<RansSession*>(handle);
}
