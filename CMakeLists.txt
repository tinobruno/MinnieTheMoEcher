cmake_minimum_required(VERSION 3.22)

# Allow newer MSVC compiler versions with CUDA
if(NOT DEFINED CMAKE_CUDA_FLAGS)
    set(CMAKE_CUDA_FLAGS "--allow-unsupported-compiler")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
endif()

project(moecher LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(MSVC)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /arch:AVX2 /utf-8 /EHsc /fp:fast")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/utf-8,/EHsc --allow-unsupported-compiler")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -funroll-loops")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

# CUDA architectures: sm_86 (RTX 3090), sm_89 (Ada/4090/6000), sm_90 (Hopper/Blackwell)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "86;89;90")
endif()

# ── Fetch header-only dependencies ──────────────────────────────────────────────
include(FetchContent)

FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.18.3
)
set(HTTPLIB_REQUIRE_OPENSSL OFF CACHE BOOL "" FORCE)
set(HTTPLIB_REQUIRE_ZLIB OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)

FetchContent_MakeAvailable(httplib nlohmann_json)

# ── Find CUDA libraries ────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── CUDA kernels library ───────────────────────────────────────────────────────
add_library(moecher_cuda STATIC
    src/cuda/activations.cu
)
target_include_directories(moecher_cuda PUBLIC src/cuda)
set_target_properties(moecher_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ── Main executable ────────────────────────────────────────────────────────────
add_executable(moecher
    src/server_single.cpp
)

if(WIN32)
    target_link_libraries(moecher PRIVATE
        moecher_cuda
        CUDA::cublas
        CUDA::cudart
        httplib::httplib
        nlohmann_json::nlohmann_json
        ws2_32
        crypt32
    )
else()
    target_link_libraries(moecher PRIVATE
        moecher_cuda
        CUDA::cublas
        CUDA::cudart
        httplib::httplib
        nlohmann_json::nlohmann_json
        pthread
    )
endif()

target_include_directories(moecher PRIVATE
    src/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)


