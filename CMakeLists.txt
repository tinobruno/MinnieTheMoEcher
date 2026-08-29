cmake_minimum_required(VERSION 3.22)

# Set CUDA architectures before project() initializes the CUDA toolchain
# sm_86 (Ampere/3090), sm_89 (Ada/4090/6000), sm_90 (Hopper), sm_100;sm_120 (Blackwell)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "86;89;90;100;120")
endif()

# Allow newer MSVC compiler versions with CUDA
if(NOT DEFINED CMAKE_CUDA_FLAGS)
    set(CMAKE_CUDA_FLAGS "--allow-unsupported-compiler")
endif()

project(moecher LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(MSVC)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /arch:AVX2 /utf-8 /EHsc")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler=/utf-8,/EHsc,/O2 --allow-unsupported-compiler")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -funroll-loops")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
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

# ── Main executable ────────────────────────────────────────────────────────────
set(MOECHER_SOURCES
    src/server_single.cpp
    src/cuda/activations.cu
)
if(WIN32)
    list(APPEND MOECHER_SOURCES src/moecher.rc)
endif()

add_executable(moecher ${MOECHER_SOURCES})

set_target_properties(moecher PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)

if(WIN32)
    target_link_libraries(moecher PRIVATE
        CUDA::cublas
        CUDA::cudart
        httplib::httplib
        nlohmann_json::nlohmann_json
        ws2_32
        crypt32
        winmm
    )
else()
    target_link_libraries(moecher PRIVATE
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


