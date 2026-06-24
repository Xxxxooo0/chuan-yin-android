cmake_minimum_required(VERSION 3.22.1)
project(gvcrt_clean_rans)

add_library(
    gvcrt_clean_rans
    SHARED
    rans_jni.cpp
    rans/rans.cpp
)

target_include_directories(gvcrt_clean_rans PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/rans)
target_compile_features(gvcrt_clean_rans PRIVATE cxx_std_17)

find_library(log-lib log)
target_link_libraries(gvcrt_clean_rans ${log-lib})
