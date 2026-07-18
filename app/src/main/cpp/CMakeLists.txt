cmake_minimum_required(VERSION 3.22.1)
project(gvcrt_clean_rans)

add_library(
    gvcrt_clean_rans
    SHARED
    i_prior_npu_jni.cpp
    neuron_adapter_jni.cpp
    mtk_tflite_jni.cpp
    rans_jni.cpp
    rans/rans.cpp
)

target_include_directories(gvcrt_clean_rans PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/rans)
target_compile_features(gvcrt_clean_rans PRIVATE cxx_std_17)
target_compile_options(gvcrt_clean_rans PRIVATE -O3)

find_library(log-lib log)
find_library(vulkan-lib vulkan)
target_link_libraries(gvcrt_clean_rans ${log-lib} ${vulkan-lib})
