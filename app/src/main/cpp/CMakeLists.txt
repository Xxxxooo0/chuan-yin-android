cmake_minimum_required(VERSION 3.22.1)
project(gvcrt_clean_rans)

add_library(
    gvcrt_clean_rans
    SHARED
    rans_jni.cpp
    rans/rans.cpp
    tflite_custom_op_probe_jni.cpp
    i_entropy_rans_custom_op_jni.cpp
    i_entropy_rans_decode_custom_op_jni.cpp
)

target_include_directories(gvcrt_clean_rans PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/rans)
target_compile_features(gvcrt_clean_rans PRIVATE cxx_std_17)
target_compile_options(gvcrt_clean_rans PRIVATE -O3)

find_library(log-lib log)
find_library(dl-lib dl)
target_link_libraries(gvcrt_clean_rans ${log-lib} ${dl-lib})

# Read-only execution-plan audit for the standard Java TFLite GPU runtime.
add_library(gvcrt_gpu_guard SHARED gpu_delegation_guard_jni.cpp)
target_include_directories(gvcrt_gpu_guard PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tflite)
target_compile_features(gvcrt_gpu_guard PRIVATE cxx_std_17)
target_link_libraries(gvcrt_gpu_guard ${log-lib})

# Independent Small fused entropy GPU probe. This intentionally uses the
# official TFLite JNI library at runtime and shares only the existing rANS core.
add_library(
    gvcrt_small_gpu_rans
    SHARED
    small_gpu_rans_jni.cpp
    rans/rans.cpp
)
target_include_directories(gvcrt_small_gpu_rans PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/rans)
target_compile_features(gvcrt_small_gpu_rans PRIVATE cxx_std_17)
target_compile_options(gvcrt_small_gpu_rans PRIVATE -O3)
target_link_libraries(gvcrt_small_gpu_rans ${log-lib} ${dl-lib})
