#pragma once
// metal_backend.h — Metal runtime abstraction and kernel declarations for Apple Silicon

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>

#if defined(__APPLE__)

// ── 16-bit Float Types for Host / CPU ─────────────────────────────────────────

struct alignas(2) __nv_bfloat16 {
    uint16_t __x;
    __nv_bfloat16() : __x(0) {}
    constexpr __nv_bfloat16(uint16_t raw, bool) : __x(raw) {}
    
    static inline __nv_bfloat16 from_float(float f) {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        return __nv_bfloat16((uint16_t)(u >> 16), true);
    }
    inline float to_float() const {
        uint32_t u = ((uint32_t)__x) << 16;
        float f;
        std::memcpy(&f, &u, sizeof(f));
        return f;
    }
    explicit operator float() const { return to_float(); }
    bool operator==(const __nv_bfloat16& o) const { return __x == o.__x; }
    bool operator!=(const __nv_bfloat16& o) const { return __x != o.__x; }
};

struct alignas(4) __nv_bfloat162 {
    __nv_bfloat16 x;
    __nv_bfloat16 y;
};

struct alignas(2) half {
    uint16_t __x;
    half() : __x(0) {}
    constexpr half(uint16_t raw, bool) : __x(raw) {}
    
    static inline half from_float(float f) {
        // Fast float32 to float16 conversion
        uint32_t x;
        std::memcpy(&x, &f, sizeof(x));
        uint32_t sign = (x >> 16) & 0x8000;
        int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (x >> 13) & 0x3FF;
        if (exp <= 0) return half((uint16_t)sign, true);
        if (exp >= 31) return half((uint16_t)(sign | 0x7C00), true);
        return half((uint16_t)(sign | (exp << 10) | mant), true);
    }
    inline float to_float() const {
        uint32_t sign = ((uint32_t)(__x & 0x8000)) << 16;
        int32_t exp = ((__x >> 10) & 0x1F);
        uint32_t mant = (__x & 0x3FF);
        if (exp == 0) return 0.0f;
        if (exp == 31) exp = 255;
        else exp = exp - 15 + 127;
        uint32_t res = sign | (exp << 23) | (mant << 13);
        float f;
        std::memcpy(&f, &res, sizeof(f));
        return f;
    }
    explicit operator float() const { return to_float(); }
};

struct alignas(4) half2 {
    half x;
    half y;
};

// ── CUDA Runtime Compatibility Layer for Metal ───────────────────────────────

typedef void* cudaStream_t;
typedef void* cudaEvent_t;
typedef void* cudaGraph_t;
typedef void* cudaGraphExec_t;

#define __align__(n) alignas(n)

enum cudaError_t {
    cudaSuccess = 0,
    cudaErrorMemoryAllocation = 1,
    cudaErrorInitializationError = 2,
    cudaErrorLaunchFailure = 3,
    cudaErrorPriorLaunchFailure = 4,
    cudaErrorLaunchTimeout = 5,
    cudaErrorLaunchOutOfResources = 6,
    cudaErrorInvalidDeviceFunction = 7,
    cudaErrorInvalidConfiguration = 8,
    cudaErrorInvalidDevice = 9,
    cudaErrorInvalidValue = 10,
    cudaErrorInvalidPitchValue = 11,
    cudaErrorInvalidSymbol = 12,
    cudaErrorUnmapBufferObjectFailed = 13,
    cudaErrorArrayIsMapped = 14,
    cudaErrorAlreadyMapped = 15,
    cudaErrorNoDevice = 16,
    cudaErrorAlreadyAcquired = 17,
    cudaErrorNotPermitted = 18,
    cudaErrorNotSupported = 19,
    cudaErrorUnknown = 999
};

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

enum cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed = 2
};

enum cudaAccessProperty {
    cudaAccessPropertyNormal = 0,
    cudaAccessPropertyStreaming = 1,
    cudaAccessPropertyPersisting = 2
};

enum cudaLimit {
    cudaLimitStackSize = 0,
    cudaLimitPrintfFifoSize = 1,
    cudaLimitMallocHeapSize = 2,
    cudaLimitDevRuntimeSyncDepth = 3,
    cudaLimitDevRuntimePendingLaunchCount = 4,
    cudaLimitMaxL2FetchGranularity = 5,
    cudaLimitPersistingL2CacheSize = 6
};

#define cudaEventDisableTiming 0x02
#define cudaHostRegisterDefault 0x00
#define cudaDeviceScheduleSpin 0x01

struct cudaAccessPolicyWindow {
    void* base_ptr;
    size_t num_bytes;
    float hitRatio;
    cudaAccessProperty hitProp;
    cudaAccessProperty missProp;
};

struct cudaStreamAttrValue {
    cudaAccessPolicyWindow accessPolicyWindow;
};

enum cudaStreamAttrID {
    cudaStreamAttributeAccessPolicyWindow = 1
};

struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    size_t totalConstMem;
    int major;
    int minor;
    int clockRate;
    size_t textureAlignment;
    int multiProcessorCount;
    int integrated;
    int canMapHostMemory;
    size_t persistingL2CacheMaxSize;
};

// ── Metal Backend C Functions ───────────────────────────────────────────────

#ifdef __cplusplus
extern "C" {
#endif

void metal_init();
void* metal_malloc(size_t bytes);
void metal_free(void* ptr);
void metal_memcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind);
void metal_memcpy_async(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream);
void metal_memset(void* ptr, int value, size_t bytes);
void metal_memset_async(void* ptr, int value, size_t bytes, cudaStream_t stream);

cudaStream_t metal_stream_create();
void metal_stream_destroy(cudaStream_t stream);
void metal_stream_synchronize(cudaStream_t stream);

cudaEvent_t metal_event_create();
void metal_event_destroy(cudaEvent_t event);
void metal_event_record(cudaEvent_t event, cudaStream_t stream);
void metal_event_synchronize(cudaEvent_t event);
void metal_stream_wait_event(cudaStream_t stream, cudaEvent_t event, unsigned int flags);

void metal_get_device_properties(cudaDeviceProp* prop, int device);
void metal_get_mem_info(size_t* free_bytes, size_t* total_bytes);

#ifdef __cplusplus
}
#endif

// ── CUDA Runtime Shims ───────────────────────────────────────────────────────

template <typename T>
inline cudaError_t cudaMalloc(T** devPtr, size_t size) {
    *devPtr = (T*)metal_malloc(size);
    return *devPtr ? cudaSuccess : cudaErrorMemoryAllocation;
}

inline cudaError_t cudaFree(void* devPtr) {
    metal_free(devPtr);
    return cudaSuccess;
}

template <typename T>
inline cudaError_t cudaMallocHost(T** ptr, size_t size) {
    *ptr = (T*)metal_malloc(size);
    return *ptr ? cudaSuccess : cudaErrorMemoryAllocation;
}

inline cudaError_t cudaFreeHost(void* ptr) {
    metal_free(ptr);
    return cudaSuccess;
}

inline cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags) {
    (void)ptr; (void)size; (void)flags;
    return cudaSuccess;
}

inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    metal_memcpy(dst, src, count, kind);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0) {
    metal_memcpy_async(dst, src, count, kind, stream);
    return cudaSuccess;
}

inline cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
    metal_memset(devPtr, value, count);
    return cudaSuccess;
}

inline cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream = 0) {
    metal_memset_async(devPtr, value, count, stream);
    return cudaSuccess;
}

inline cudaError_t cudaStreamCreate(cudaStream_t* pStream) {
    *pStream = metal_stream_create();
    return cudaSuccess;
}

inline cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    metal_stream_destroy(stream);
    return cudaSuccess;
}

inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    metal_stream_synchronize(stream);
    return cudaSuccess;
}

inline cudaError_t cudaDeviceSynchronize() {
    metal_stream_synchronize(0);
    return cudaSuccess;
}

inline cudaError_t cudaEventCreate(cudaEvent_t* event) {
    *event = metal_event_create();
    return cudaSuccess;
}

inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
    (void)flags;
    *event = metal_event_create();
    return cudaSuccess;
}

inline cudaError_t cudaEventDestroy(cudaEvent_t event) {
    metal_event_destroy(event);
    return cudaSuccess;
}

inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0) {
    metal_event_record(event, stream);
    return cudaSuccess;
}

inline cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    metal_event_synchronize(event);
    return cudaSuccess;
}

inline cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags = 0) {
    metal_stream_wait_event(stream, event, flags);
    return cudaSuccess;
}

inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
    metal_get_device_properties(prop, device);
    return cudaSuccess;
}

inline cudaError_t cudaMemGetInfo(size_t* free_bytes, size_t* total_bytes) {
    metal_get_mem_info(free_bytes, total_bytes);
    return cudaSuccess;
}

inline cudaError_t cudaSetDevice(int device) { (void)device; return cudaSuccess; }
inline cudaError_t cudaSetDeviceFlags(unsigned int flags) { (void)flags; return cudaSuccess; }
inline cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value) { (void)limit; (void)value; return cudaSuccess; }
inline cudaError_t cudaStreamSetAttribute(cudaStream_t stream, cudaStreamAttrID attr, const cudaStreamAttrValue* value) {
    (void)stream; (void)attr; (void)value;
    return cudaSuccess;
}
inline const char* cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess: return "cudaSuccess";
        case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
        default: return "cudaErrorUnknown";
    }
}
inline cudaError_t cudaGetLastError() { return cudaSuccess; }

// CUDA Graph Stubs (always triggers graceful fallback to eager execution)
inline cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) {
    (void)stream; (void)mode;
    return cudaErrorNotSupported;
}
inline cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph) {
    (void)stream; (void)pGraph;
    return cudaErrorNotSupported;
}
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, void* a, void* b, size_t c) {
    (void)pGraphExec; (void)graph; (void)a; (void)b; (void)c;
    return cudaErrorNotSupported;
}
inline cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) {
    (void)graphExec; (void)stream;
    return cudaErrorNotSupported;
}
inline cudaError_t cudaGraphDestroy(cudaGraph_t graph) { (void)graph; return cudaSuccess; }
inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec) { (void)graphExec; return cudaSuccess; }

// ── cuBLAS Compatibility Layer ──────────────────────────────────────────────

typedef void* cublasHandle_t;
enum cublasStatus_t {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 2,
    CUBLAS_STATUS_INVALID_VALUE = 3,
    CUBLAS_STATUS_ARCH_MISMATCH = 4,
    CUBLAS_STATUS_MAPPING_ERROR = 5,
    CUBLAS_STATUS_EXECUTION_FAILED = 6,
    CUBLAS_STATUS_INTERNAL_ERROR = 7,
    CUBLAS_STATUS_NOT_SUPPORTED = 8,
    CUBLAS_STATUS_LICENSE_ERROR = 9
};

enum cublasOperation_t {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2
};

enum cudaDataType_t {
    CUDA_R_16F = 2,
    CUDA_R_32F = 0,
    CUDA_R_16BF = 14
};

enum cublasComputeType_t {
    CUBLAS_COMPUTE_16F = 64,
    CUBLAS_COMPUTE_32F = 68
};

enum cublasGemmAlgo_t {
    CUBLAS_GEMM_DEFAULT = -1
};

enum cublasMath_t {
    CUBLAS_DEFAULT_MATH = 0
};

inline cublasStatus_t cublasCreate(cublasHandle_t* handle) {
    *handle = (void*)0x1;
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasDestroy(cublasHandle_t handle) { (void)handle; return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t stream) { (void)handle; (void)stream; return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) { (void)handle; (void)mode; return CUBLAS_STATUS_SUCCESS; }

#ifdef __cplusplus
extern "C" {
#endif

cublasStatus_t metal_gemm_ex(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void* alpha,
    const void* A, cudaDataType_t Atype, int lda,
    const void* B, cudaDataType_t Btype, int ldb,
    const void* beta,
    void* C, cudaDataType_t Ctype, int ldc,
    cublasComputeType_t computeType,
    cublasGemmAlgo_t algo
);

cublasStatus_t metal_gemm_strided_batched_ex(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void* alpha,
    const void* A, cudaDataType_t Atype, int lda, long long strideA,
    const void* B, cudaDataType_t Btype, int ldb, long long strideB,
    const void* beta,
    void* C, cudaDataType_t Ctype, int ldc, long long strideC,
    int batchCount,
    cublasComputeType_t computeType,
    cublasGemmAlgo_t algo
);

#ifdef __cplusplus
}
#endif

inline cublasStatus_t cublasGemmStridedBatchedEx(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void* alpha,
    const void* A, cudaDataType_t Atype, int lda, long long strideA,
    const void* B, cudaDataType_t Btype, int ldb, long long strideB,
    const void* beta,
    void* C, cudaDataType_t Ctype, int ldc, long long strideC,
    int batchCount,
    cublasComputeType_t computeType,
    cublasGemmAlgo_t algo)
{
    return metal_gemm_strided_batched_ex(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo);
}

inline cublasStatus_t cublasGemmEx(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void* alpha,
    const void* A, cudaDataType_t Atype, int lda,
    const void* B, cudaDataType_t Btype, int ldb,
    const void* beta,
    void* C, cudaDataType_t Ctype, int ldc,
    cublasComputeType_t computeType,
    cublasGemmAlgo_t algo)
{
    return metal_gemm_ex(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);
}

#endif // __APPLE__
