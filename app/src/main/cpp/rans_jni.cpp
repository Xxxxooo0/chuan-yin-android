#include <jni.h>

#include <algorithm>
#include <cmath>
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

class AndroidRansEncoder {
public:
    AndroidRansEncoder()
        : encoder0_(std::make_shared<RansEncoderLibMultiThread>())
        , encoder1_(std::make_shared<RansEncoderLibMultiThread>())
    {
    }

    void setUseTwoEncoders(bool enabled) { useTwoEncoders_ = enabled; }

    int addCdf(const CdfData& cdf)
    {
        const int idx0 = encoder0_->add_cdf(cdf.cdfs, cdf.lengths, cdf.offsets);
        encoder1_->add_cdf(cdf.cdfs, cdf.lengths, cdf.offsets);
        return idx0;
    }

    void encodeY(const std::shared_ptr<std::vector<int16_t>> symbols, int cdfGroupIndex)
    {
        if (useTwoEncoders_) {
            const int size0 = static_cast<int>(symbols->size()) / 2;
            auto symbols0 = std::make_shared<std::vector<int16_t>>(symbols->begin(), symbols->begin() + size0);
            auto symbols1 = std::make_shared<std::vector<int16_t>>(symbols->begin() + size0, symbols->end());
            encoder0_->encode_y(symbols0, cdfGroupIndex);
            encoder1_->encode_y(symbols1, cdfGroupIndex);
        } else {
            encoder0_->encode_y(symbols, cdfGroupIndex);
        }
    }

    void encodeZ(const std::shared_ptr<std::vector<int8_t>> symbols,
                 int cdfGroupIndex,
                 int startOffset,
                 int perChannelSize)
    {
        if (useTwoEncoders_) {
            const int size0 = static_cast<int>(symbols->size()) / 2;
            const int channelHalf = size0 / perChannelSize;
            auto symbols0 = std::make_shared<std::vector<int8_t>>(symbols->begin(), symbols->begin() + size0);
            auto symbols1 = std::make_shared<std::vector<int8_t>>(symbols->begin() + size0, symbols->end());
            encoder0_->encode_z(symbols0, cdfGroupIndex, startOffset, perChannelSize);
            encoder1_->encode_z(symbols1, cdfGroupIndex, startOffset + channelHalf, perChannelSize);
        } else {
            encoder0_->encode_z(symbols, cdfGroupIndex, startOffset, perChannelSize);
        }
    }

    std::vector<uint8_t> flush()
    {
        encoder0_->flush();
        encoder1_->flush();
        if (!useTwoEncoders_) {
            std::vector<uint8_t> output = *encoder0_->get_encoded_stream();
            encoder0_->reset();
            encoder1_->reset();
            return output;
        }

        auto result0 = encoder0_->get_encoded_stream();
        auto result1 = encoder1_->get_encoded_stream();
        const int nbytes0 = static_cast<int>(result0->size());
        const int nbytes1 = static_cast<int>(result1->size());

        int identicalBytes = 0;
        const int checkBytes = std::min(std::min(nbytes0, nbytes1), 8);
        for (int i = 0; i < checkBytes; ++i) {
            if (result0->at(nbytes0 - 1 - i) != 0 || result1->at(nbytes1 - 1 - i) != 0) {
                break;
            }
            identicalBytes++;
        }
        if (identicalBytes == 0 && nbytes0 > 0 && nbytes1 > 0 &&
            result0->at(nbytes0 - 1) == result1->at(nbytes1 - 1)) {
            identicalBytes = 1;
        }

        std::vector<uint8_t> stream(static_cast<size_t>(nbytes0 + nbytes1 - identicalBytes));
        std::copy(result0->begin(), result0->end(), stream.begin());
        std::copy(result1->rbegin() + identicalBytes, result1->rend(),
                  stream.begin() + static_cast<std::ptrdiff_t>(nbytes0));
        encoder0_->reset();
        encoder1_->reset();
        return stream;
    }

private:
    std::shared_ptr<RansEncoderLib> encoder0_;
    std::shared_ptr<RansEncoderLib> encoder1_;
    bool useTwoEncoders_{false};
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

std::shared_ptr<std::vector<int16_t>> packYFromFloat(JNIEnv* env, jfloatArray symbols, jfloatArray scales)
{
    const jsize size = env->GetArrayLength(symbols);
    if (env->GetArrayLength(scales) != size) {
        throw std::invalid_argument("symbol and scale size mismatch");
    }
    std::vector<jfloat> symbolValues(size);
    std::vector<jfloat> scaleValues(size);
    env->GetFloatArrayRegion(symbols, 0, size, symbolValues.data());
    env->GetFloatArrayRegion(scales, 0, size, scaleValues.data());

    constexpr float scaleMin = 0.11f;
    constexpr float scaleMax = 16.0f;
    constexpr float logScaleMin = -2.2072749f;
    constexpr float logStepRecip = 25.502707f;

    auto packed = std::make_shared<std::vector<int16_t>>(size);
    for (jsize i = 0; i < size; ++i) {
        const int symbol = static_cast<int>(symbolValues[i]);
        float scale = scaleValues[i];
        if (scale < scaleMin) {
            scale = scaleMin;
        } else if (scale > scaleMax) {
            scale = scaleMax;
        }
        int cdfIndex = static_cast<int>((std::log(scale) - logScaleMin) * logStepRecip);
        if (cdfIndex < 0) {
            cdfIndex = 0;
        } else if (cdfIndex > 127) {
            cdfIndex = 127;
        }
        packed->at(i) = static_cast<int16_t>((symbol << 8) + cdfIndex);
    }
    return packed;
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

AndroidRansEncoder* checkedEncoder(JNIEnv* env, jlong handle)
{
    if (handle == 0) {
        throw std::invalid_argument("rANS encoder is closed");
    }
    return reinterpret_cast<AndroidRansEncoder*>(handle);
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

extern "C" JNIEXPORT jlong JNICALL
Java_com_gvcrt_clean_RansNativeEncoder_nativeCreateEncoder(JNIEnv* env, jclass)
{
    try {
        return reinterpret_cast<jlong>(new AndroidRansEncoder());
    } catch (const std::exception& error) {
        throwIllegalState(env, error.what());
        return 0;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_gvcrt_clean_RansNativeEncoder_nativeDestroyEncoder(JNIEnv*, jclass, jlong handle)
{
    delete reinterpret_cast<AndroidRansEncoder*>(handle);
}

extern "C" JNIEXPORT void JNICALL
Java_com_gvcrt_clean_RansNativeEncoder_nativeSetUseTwoEncoders(
    JNIEnv* env,
    jclass,
    jlong handle,
    jboolean enabled)
{
    try {
        checkedEncoder(env, handle)->setUseTwoEncoders(enabled == JNI_TRUE);
    } catch (const std::exception& error) {
        throwIllegalState(env, error.what());
    }
}

extern "C" JNIEXPORT jint JNICALL
Java_com_gvcrt_clean_RansNativeEncoder_nativeAddCdf(
    JNIEnv* env,
    jclass,
    jlong handle,
    jintArray flatCdfs,
    jint rows,
    jint cols,
    jintArray cdfSizes,
    jintArray offsets)
{
    try {
        return checkedEncoder(env, handle)->addCdf(copyCdf(env, flatCdfs, rows, cols, cdfSizes, offsets));
    } catch (const std::exception& error) {
        throwIllegalState(env, error.what());
        return -1;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_gvcrt_clean_RansNativeEncoder_nativeEncodeY(
    JNIEnv* env,
    jclass,
    jlong handle,
    jshortArray symbols,
    jint cdfGroupIndex)
{
    try {
        checkedEncoder(env, handle)->encodeY(copyI16(env, symbols), cdfGroupIndex);
    } catch (const std::exception& error) {
        throwIllegalState(env, error.what());
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_gvcrt_clean_RansNativeEncoder_nativeEncodeYFromFloat(
    JNIEnv* env,
    jclass,
    jlong handle,
    jfloatArray symbols,
    jfloatArray scales,
    jint cdfGroupIndex)
{
    try {
        checkedEncoder(env, handle)->encodeY(packYFromFloat(env, symbols, scales), cdfGroupIndex);
    } catch (const std::exception& error) {
        throwIllegalState(env, error.what());
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_gvcrt_clean_RansNativeEncoder_nativeEncodeZ(
    JNIEnv* env,
    jclass,
    jlong handle,
    jbyteArray symbols,
    jint cdfGroupIndex,
    jint startOffset,
    jint perChannelSize)
{
    try {
        checkedEncoder(env, handle)->encodeZ(copyI8(env, symbols), cdfGroupIndex, startOffset, perChannelSize);
    } catch (const std::exception& error) {
        throwIllegalState(env, error.what());
    }
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_gvcrt_clean_RansNativeEncoder_nativeFlush(JNIEnv* env, jclass, jlong handle)
{
    try {
        return makeByteArray(env, checkedEncoder(env, handle)->flush());
    } catch (const std::exception& error) {
        throwIllegalState(env, error.what());
        return nullptr;
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
Java_com_gvcrt_clean_NativeRans_nativeBeginDecode(
    JNIEnv* env,
    jclass,
    jlong handle,
    jbyteArray payload)
{
    try {
        RansSession* session = checkedSession(env, handle);
        session->decoder.set_stream(copyU8(env, payload));
    } catch (const std::exception& error) {
        throwIllegalState(env, error.what());
    }
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_gvcrt_clean_NativeRans_nativeDecodeZ(
    JNIEnv* env,
    jclass,
    jlong handle,
    jint zTotalSize,
    jint zStartOffset,
    jint zPerChannelSize)
{
    try {
        RansSession* session = checkedSession(env, handle);
        session->decoder.decode_z(zTotalSize, session->zGroup, zStartOffset, zPerChannelSize);
        return makeByteArray(env, *session->decoder.get_decoded_tensor());
    } catch (const std::exception& error) {
        throwIllegalState(env, error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_gvcrt_clean_NativeRans_nativeDecodeY(
    JNIEnv* env,
    jclass,
    jlong handle,
    jbyteArray indexes)
{
    try {
        RansSession* session = checkedSession(env, handle);
        session->decoder.decode_y(copyU8(env, indexes), session->gaussianGroup);
        return makeByteArray(env, *session->decoder.get_decoded_tensor());
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
