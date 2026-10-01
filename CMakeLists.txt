cmake_minimum_required(VERSION 3.18)

if(APPLE)
    project(moecher LANGUAGES CXX OBJCXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

    add_compile_definitions(
        CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH=67108864
        CPPHTTPLIB_PAYLOAD_MAX_LENGTH=67108864
        CPPHTTPLIB_HEADER_MAX_LENGTH=1048576
        CPPHTTPLIB_REQUEST_URI_MAX_LENGTH=1048576
        USE_METAL
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -funroll-loops")
    set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -O3 -fobjc-arc -Wno-deprecated-declarations")
else()
    # Prefer CUDA 12.9 toolchain if present to support sm_120a (Blackwell native Tensor Cores)
    if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda-12.9/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.9/bin/nvcc")
    endif()

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "86;89;90;120a")
    endif()

    if(NOT DEFINED CMAKE_CUDA_FLAGS)
        set(CMAKE_CUDA_FLAGS "--allow-unsupported-compiler")
    endif()

    project(moecher LANGUAGES CXX CUDA)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD 17)
    if(MSVC)
        add_compile_definitions(
            NOMINMAX
            WIN32_LEAN_AND_MEAN
            _CRT_SECURE_NO_WARNINGS
            CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH=67108864
            CPPHTTPLIB_PAYLOAD_MAX_LENGTH=67108864
            CPPHTTPLIB_HEADER_MAX_LENGTH=1048576
            CPPHTTPLIB_REQUEST_URI_MAX_LENGTH=1048576
        )
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /arch:AVX2 /utf-8 /EHsc")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler=/utf-8,/EHsc,/O2 --allow-unsupported-compiler")
    else()
        add_compile_definitions(
            CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH=67108864
            CPPHTTPLIB_PAYLOAD_MAX_LENGTH=67108864
            CPPHTTPLIB_HEADER_MAX_LENGTH=1048576
            CPPHTTPLIB_REQUEST_URI_MAX_LENGTH=1048576
        )
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -funroll-loops")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    endif()
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

# ── Find CUDA libraries (non-Apple only) ─────────────────────────────────────────
if(NOT APPLE)
    find_package(CUDAToolkit REQUIRED)
endif()

# ── Main executable ────────────────────────────────────────────────────────────
if(APPLE)
    set(MOECHER_SOURCES
        src/server_single.cpp
        src/metal/metal_backend.mm
    )
else()
    set(MOECHER_SOURCES
        src/server_single.cpp
        src/cuda/activations.cu
        src/cuda/vision_kernels.cu
    )
    if(WIN32)
        list(APPEND MOECHER_SOURCES src/moecher.rc)
    endif()
endif()

add_executable(moecher ${MOECHER_SOURCES})

if(NOT APPLE)
    set_target_properties(moecher PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
    )
endif()

if(APPLE)
    target_link_libraries(moecher PRIVATE
        "-framework Metal"
        "-framework MetalPerformanceShaders"
        "-framework Foundation"
        "-framework Accelerate"
        httplib::httplib
        nlohmann_json::nlohmann_json
        pthread
    )
elseif(WIN32)
    target_link_libraries(moecher PRIVATE
        CUDA::cublas
        CUDA::cudart
        httplib::httplib
        nlohmann_json::nlohmann_json
        ws2_32
        crypt32
        winmm
        winhttp
        wininet
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
    src/metal
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(APPLE)
    add_executable(test_metal
        tests/test_metal.cpp
        src/metal/metal_backend.mm
    )
    target_link_libraries(test_metal PRIVATE
        "-framework Metal"
        "-framework MetalPerformanceShaders"
        "-framework Foundation"
        "-framework Accelerate"
        nlohmann_json::nlohmann_json
        pthread
    )
    target_include_directories(test_metal PRIVATE
        src/cuda
        src/metal
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
endif()
