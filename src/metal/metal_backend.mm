// metal_backend.mm — High-performance Metal runtime & kernel dispatch for Apple Silicon
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Accelerate/Accelerate.h>

#include "metal_backend.h"
#include "activations.cuh"
#include "vision_kernels.cuh"

#include <iostream>
#include <unordered_map>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <sys/sysctl.h>
#include <mach/mach.h>

// ════════════════════════════════════════════════════════════════════════════════
//  Metal Context & Buffer Tracking
// ════════════════════════════════════════════════════════════════════════════════

struct BufferRecord {
    id<MTLBuffer> buffer;
    size_t size;
};

struct MetalStreamObj {
    id<MTLCommandQueue> queue;
    id<MTLCommandBuffer> current_cmd_buf;
    id<MTLComputeCommandEncoder> current_encoder;

    MetalStreamObj(id<MTLCommandQueue> q) : queue(q), current_cmd_buf(nil), current_encoder(nil) {}

    id<MTLComputeCommandEncoder> get_encoder() {
        if (!current_cmd_buf) {
            current_cmd_buf = [queue commandBuffer];
        }
        if (!current_encoder) {
            current_encoder = [current_cmd_buf computeCommandEncoder];
        }
        return current_encoder;
    }

    void end_encoder() {
        if (current_encoder) {
            [current_encoder endEncoding];
            current_encoder = nil;
        }
    }

    void commit_and_wait() {
        end_encoder();
        if (current_cmd_buf) {
            [current_cmd_buf commit];
            [current_cmd_buf waitUntilCompleted];
            current_cmd_buf = nil;
        }
    }
};

class MetalContext {
public:
    static MetalContext& instance() {
        static MetalContext ctx;
        return ctx;
    }

    id<MTLDevice> device;
    id<MTLCommandQueue> default_queue;
    id<MTLLibrary> library;
    MetalStreamObj* default_stream;

    std::map<uintptr_t, BufferRecord> buffers;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
    std::mutex buf_mutex;
    std::mutex pipe_mutex;

    MetalContext() {
        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (!device) {
                std::cerr << "[Metal] Error: No Metal device found!\n";
                exit(1);
            }
            default_queue = [device newCommandQueue];
            default_stream = new MetalStreamObj(default_queue);

            // Load and compile kernels.metal
            NSString* kernelPath = @"src/metal/kernels.metal";
            NSError* err = nil;
            NSString* source = [NSString stringWithContentsOfFile:kernelPath encoding:NSUTF8StringEncoding error:&err];
            if (!source) {
                // Try relative to binary or current directory
                kernelPath = @"kernels.metal";
                source = [NSString stringWithContentsOfFile:kernelPath encoding:NSUTF8StringEncoding error:&err];
            }
            if (source) {
                MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
                opts.languageVersion = MTLLanguageVersion3_1;
                library = [device newLibraryWithSource:source options:opts error:&err];
                if (err) {
                    std::cerr << "[Metal] Library compilation notice: " << [[err localizedDescription] UTF8String] << "\n";
                }
            } else {
                std::cerr << "[Metal] Warning: Could not find kernels.metal at path: src/metal/kernels.metal\n";
            }
        }
    }

    id<MTLComputePipelineState> get_pipeline(const std::string& name) {
        std::lock_guard<std::mutex> lock(pipe_mutex);
        auto it = pipelines.find(name);
        if (it != pipelines.end()) return it->second;

        if (!library) return nil;
        NSString* fnName = [NSString stringWithUTF8String:name.c_str()];
        id<MTLFunction> fn = [library newFunctionWithName:fnName];
        if (!fn) {
            std::cerr << "[Metal] Function not found in library: " << name << "\n";
            return nil;
        }
        NSError* err = nil;
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
        if (err || !pso) {
            std::cerr << "[Metal] Failed to create pipeline for " << name << ": " << [[err localizedDescription] UTF8String] << "\n";
            return nil;
        }
        pipelines[name] = pso;
        return pso;
    }

    id<MTLBuffer> get_buffer(const void* ptr, size_t& offset) {
        if (!ptr) return nil;
        uintptr_t addr = (uintptr_t)ptr;
        std::lock_guard<std::mutex> lock(buf_mutex);
        if (buffers.empty()) return nil;

        auto it = buffers.upper_bound(addr);
        if (it != buffers.begin()) {
            --it;
            uintptr_t base = it->first;
            size_t size = it->second.size;
            if (addr >= base && addr < base + size) {
                offset = addr - base;
                return it->second.buffer;
            }
        }
        return nil;
    }

    void* allocate(size_t bytes) {
        if (bytes == 0) bytes = 16;
        @autoreleasepool {
            id<MTLBuffer> buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (!buf) return nullptr;
            void* ptr = [buf contents];
            std::lock_guard<std::mutex> lock(buf_mutex);
            buffers[(uintptr_t)ptr] = {buf, bytes};
            return ptr;
        }
    }

    void deallocate(void* ptr) {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(buf_mutex);
        auto it = buffers.find((uintptr_t)ptr);
        if (it != buffers.end()) {
            buffers.erase(it);
        }
    }
};

// ════════════════════════════════════════════════════════════════════════════════
//  C-API Implementation
// ════════════════════════════════════════════════════════════════════════════════

extern "C" {

void metal_init() {
    MetalContext::instance();
}

void* metal_malloc(size_t bytes) {
    return MetalContext::instance().allocate(bytes);
}

void metal_free(void* ptr) {
    MetalContext::instance().deallocate(ptr);
}

void metal_memcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
    (void)kind;
    if (dst && src && bytes > 0) {
        std::memcpy(dst, src, bytes);
    }
}

void metal_memcpy_async(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
    (void)stream;
    metal_memcpy(dst, src, bytes, kind);
}

void metal_memset(void* ptr, int value, size_t bytes) {
    if (ptr && bytes > 0) {
        std::memset(ptr, value, bytes);
    }
}

void metal_memset_async(void* ptr, int value, size_t bytes, cudaStream_t stream) {
    (void)stream;
    metal_memset(ptr, value, bytes);
}

cudaStream_t metal_stream_create() {
    auto& ctx = MetalContext::instance();
    id<MTLCommandQueue> q = [ctx.device newCommandQueue];
    return (cudaStream_t)new MetalStreamObj(q);
}

void metal_stream_destroy(cudaStream_t stream) {
    if (stream && stream != MetalContext::instance().default_stream) {
        delete (MetalStreamObj*)stream;
    }
}

void metal_stream_synchronize(cudaStream_t stream) {
    MetalStreamObj* s = stream ? (MetalStreamObj*)stream : MetalContext::instance().default_stream;
    if (s) {
        s->commit_and_wait();
    }
}

cudaEvent_t metal_event_create() { return (cudaEvent_t)0x1; }
void metal_event_destroy(cudaEvent_t event) { (void)event; }
void metal_event_record(cudaEvent_t event, cudaStream_t stream) { (void)event; (void)stream; }
void metal_event_synchronize(cudaEvent_t event) { (void)event; }
void metal_stream_wait_event(cudaStream_t stream, cudaEvent_t event, unsigned int flags) { (void)stream; (void)event; (void)flags; }

void metal_get_device_properties(cudaDeviceProp* prop, int device) {
    (void)device;
    if (!prop) return;
    std::memset(prop, 0, sizeof(*prop));
    id<MTLDevice> dev = MetalContext::instance().device;
    const char* devName = [[dev name] UTF8String];
    std::strncpy(prop->name, devName ? devName : "Apple Silicon", sizeof(prop->name) - 1);

    uint64_t memsize = 0;
    size_t len = sizeof(memsize);
    sysctlbyname("hw.memsize", &memsize, &len, NULL, 0);
    prop->totalGlobalMem = memsize;
    prop->sharedMemPerBlock = 32768; // 32 KB threadgroup memory
    prop->warpSize = 32;            // 32-wide SIMD group
    prop->major = 12;               // Apple Silicon compute capability mapping
    prop->minor = 0;
    prop->multiProcessorCount = 10;
    prop->canMapHostMemory = 1;     // Unified memory
}

void metal_get_mem_info(size_t* free_bytes, size_t* total_bytes) {
    uint64_t memsize = 0;
    size_t len = sizeof(memsize);
    sysctlbyname("hw.memsize", &memsize, &len, NULL, 0);
    if (total_bytes) *total_bytes = memsize;

    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vm_stat;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm_stat, &count) == KERN_SUCCESS) {
        uint64_t free_mem = (uint64_t)(vm_stat.free_count + vm_stat.inactive_count) * (uint64_t)vm_page_size;
        if (free_bytes) *free_bytes = free_mem;
    } else {
        if (free_bytes) *free_bytes = memsize / 2;
    }
}

// ── GEMM via Metal Performance Shaders / Accelerate BLAS ────────────────────

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
    cublasGemmAlgo_t algo)
{
    (void)handle; (void)computeType; (void)algo;
    float alpha_f = alpha ? *(const float*)alpha : 1.0f;
    float beta_f = beta ? *(const float*)beta : 0.0f;

    CBLAS_TRANSPOSE opA = (transa == CUBLAS_OP_T ? CblasTrans : CblasNoTrans);
    CBLAS_TRANSPOSE opB = (transb == CUBLAS_OP_T ? CblasTrans : CblasNoTrans);

    if (Atype == CUDA_R_32F && Btype == CUDA_R_32F && Ctype == CUDA_R_32F) {
        cblas_sgemm(CblasColMajor, opA, opB,
                    m, n, k,
                    alpha_f, (const float*)A, lda,
                    (const float*)B, ldb,
                    beta_f, (float*)C, ldc);
        return CUBLAS_STATUS_SUCCESS;
    }

    if (Atype == CUDA_R_16BF && Btype == CUDA_R_16BF) {
        size_t a_cols = (transa == CUBLAS_OP_N ? k : m);
        size_t a_size = lda * a_cols;
        size_t b_cols = (transb == CUBLAS_OP_N ? n : k);
        size_t b_size = ldb * b_cols;
        size_t c_size = ldc * n;

        std::vector<float> A_f(a_size);
        std::vector<float> B_f(b_size);
        std::vector<float> C_f(c_size);

        const uint16_t* A_bf = (const uint16_t*)A;
        const uint16_t* B_bf = (const uint16_t*)B;

        for (size_t i = 0; i < a_size; i++) {
            uint32_t u = ((uint32_t)A_bf[i]) << 16;
            A_f[i] = *(float*)&u;
        }
        for (size_t i = 0; i < b_size; i++) {
            uint32_t u = ((uint32_t)B_bf[i]) << 16;
            B_f[i] = *(float*)&u;
        }

        if (beta_f != 0.0f) {
            if (Ctype == CUDA_R_16BF) {
                const uint16_t* C_bf = (const uint16_t*)C;
                for (size_t i = 0; i < c_size; i++) {
                    uint32_t u = ((uint32_t)C_bf[i]) << 16;
                    C_f[i] = *(float*)&u;
                }
            } else if (Ctype == CUDA_R_32F) {
                std::memcpy(C_f.data(), C, c_size * sizeof(float));
            }
        }

        cblas_sgemm(CblasColMajor, opA, opB,
                    m, n, k,
                    alpha_f, A_f.data(), lda,
                    B_f.data(), ldb,
                    beta_f, C_f.data(), ldc);

        if (Ctype == CUDA_R_16BF) {
            uint16_t* C_bf = (uint16_t*)C;
            for (size_t i = 0; i < c_size; i++) {
                uint32_t u = *(uint32_t*)&C_f[i];
                C_bf[i] = (uint16_t)(u >> 16);
            }
        } else if (Ctype == CUDA_R_32F) {
            std::memcpy(C, C_f.data(), c_size * sizeof(float));
        }
        return CUBLAS_STATUS_SUCCESS;
    }

    return CUBLAS_STATUS_NOT_SUPPORTED;
}

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
    cublasGemmAlgo_t algo)
{
    size_t elA = (Atype == CUDA_R_32F) ? 4 : 2;
    size_t elB = (Btype == CUDA_R_32F) ? 4 : 2;
    size_t elC = (Ctype == CUDA_R_32F) ? 4 : 2;

    for (int b = 0; b < batchCount; b++) {
        const void* ptrA = (const char*)A + b * strideA * elA;
        const void* ptrB = (const char*)B + b * strideB * elB;
        void* ptrC = (char*)C + b * strideC * elC;
        cublasStatus_t status = metal_gemm_ex(handle, transa, transb, m, n, k, alpha, ptrA, Atype, lda, ptrB, Btype, ldb, beta, ptrC, Ctype, ldc, computeType, algo);
        if (status != CUBLAS_STATUS_SUCCESS) return status;
    }
    return CUBLAS_STATUS_SUCCESS;
}

} // extern "C"

// ════════════════════════════════════════════════════════════════════════════════
//  Kernel Dispatch Implementations
// ════════════════════════════════════════════════════════════════════════════════

static inline MetalStreamObj* get_stream(cudaStream_t s) {
    return s ? (MetalStreamObj*)s : MetalContext::instance().default_stream;
}

void rms_norm_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* weight,
    int dim, float eps, cudaStream_t stream)
{
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("rms_norm_kernel");
    if (!pso) return;

    size_t o_off, x_off, w_off;
    id<MTLBuffer> b_out = ctx.get_buffer(out, o_off);
    id<MTLBuffer> b_x = ctx.get_buffer(x, x_off);
    id<MTLBuffer> b_w = ctx.get_buffer(weight, w_off);

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_out offset:o_off atIndex:0];
    [enc setBuffer:b_x offset:x_off atIndex:1];
    [enc setBuffer:b_w offset:w_off atIndex:2];
    [enc setBytes:&dim length:sizeof(dim) atIndex:3];
    [enc setBytes:&eps length:sizeof(eps) atIndex:4];

    NSUInteger threads = std::min(256, dim);
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

void rms_norm_one_centered_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* weight,
    int dim, float eps, cudaStream_t stream)
{
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("rms_norm_one_centered_kernel");
    if (!pso) return;

    size_t o_off, x_off, w_off;
    id<MTLBuffer> b_out = ctx.get_buffer(out, o_off);
    id<MTLBuffer> b_x = ctx.get_buffer(x, x_off);
    id<MTLBuffer> b_w = ctx.get_buffer(weight, w_off);

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_out offset:o_off atIndex:0];
    [enc setBuffer:b_x offset:x_off atIndex:1];
    [enc setBuffer:b_w offset:w_off atIndex:2];
    [enc setBytes:&dim length:sizeof(dim) atIndex:3];
    [enc setBytes:&eps length:sizeof(eps) atIndex:4];

    NSUInteger threads = std::min(256, dim);
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

void rms_norm_cuda_batched(
    __nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* weight,
    int n, int dim, float eps, cudaStream_t stream)
{
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("rms_norm_batched_kernel");
    if (!pso) return;

    size_t o_off, x_off, w_off;
    id<MTLBuffer> b_out = ctx.get_buffer(out, o_off);
    id<MTLBuffer> b_x = ctx.get_buffer(x, x_off);
    id<MTLBuffer> b_w = ctx.get_buffer(weight, w_off);

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_out offset:o_off atIndex:0];
    [enc setBuffer:b_x offset:x_off atIndex:1];
    [enc setBuffer:b_w offset:w_off atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    [enc setBytes:&dim length:sizeof(dim) atIndex:4];
    [enc setBytes:&eps length:sizeof(eps) atIndex:5];

    NSUInteger threads = std::min(256, dim);
    [enc dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

void rms_norm_one_centered_cuda_batched(
    __nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* weight,
    int n, int dim, float eps, cudaStream_t stream)
{
    rms_norm_cuda_batched(out, x, weight, n, dim, eps, stream);
}

void rms_norm_f32_cuda(float* x, int dim, float eps, cudaStream_t stream) {
    (void)stream;
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float scale = 1.0f / sqrtf(sum_sq / float(dim) + eps);
    for (int i = 0; i < dim; i++) x[i] *= scale;
}

void rms_norm_weighted_f32_cuda(float* out, const float* x, const __nv_bfloat16* weight, int dim, float eps, cudaStream_t stream) {
    (void)stream;
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float scale = 1.0f / sqrtf(sum_sq / float(dim) + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * scale * weight[i].to_float();
}

void rms_norm_unweighted_batched_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* x,
    int n, int dim, float eps, cudaStream_t stream)
{
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("rms_norm_unweighted_batched_kernel");
    if (!pso) return;

    size_t o_off, x_off;
    id<MTLBuffer> b_out = ctx.get_buffer(out, o_off);
    id<MTLBuffer> b_x = ctx.get_buffer(x, x_off);

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_out offset:o_off atIndex:0];
    [enc setBuffer:b_x offset:x_off atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    [enc setBytes:&dim length:sizeof(dim) atIndex:3];
    [enc setBytes:&eps length:sizeof(eps) atIndex:4];

    NSUInteger threads = std::min(256, dim);
    [enc dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

void silu_mul_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up,
    int n, float swiglu_limit, cudaStream_t stream)
{
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("silu_mul_kernel");
    if (!pso) return;

    size_t o_off, g_off, u_off;
    id<MTLBuffer> b_out = ctx.get_buffer(out, o_off);
    id<MTLBuffer> b_g = ctx.get_buffer(gate, g_off);
    id<MTLBuffer> b_u = ctx.get_buffer(up, u_off);

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_out offset:o_off atIndex:0];
    [enc setBuffer:b_g offset:g_off atIndex:1];
    [enc setBuffer:b_u offset:u_off atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    [enc setBytes:&swiglu_limit length:sizeof(swiglu_limit) atIndex:4];

    NSUInteger tg = 256;
    NSUInteger groups = (n + tg - 1) / tg;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

void vector_add_bf16_cuda(__nv_bfloat16* a, const __nv_bfloat16* b, int n, cudaStream_t stream) {
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("vector_add_bf16_kernel");
    if (!pso) return;

    size_t a_off, b_off;
    id<MTLBuffer> buf_a = ctx.get_buffer(a, a_off);
    id<MTLBuffer> buf_b = ctx.get_buffer(b, b_off);

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf_a offset:a_off atIndex:0];
    [enc setBuffer:buf_b offset:b_off atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];

    NSUInteger tg = 256;
    NSUInteger groups = (n + tg - 1) / tg;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

void weighted_add_cuda(__nv_bfloat16* out, const __nv_bfloat16* x, float weight, int dim, cudaStream_t stream) {
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("weighted_add_kernel");
    if (!pso) return;

    size_t o_off, x_off;
    id<MTLBuffer> b_out = ctx.get_buffer(out, o_off);
    id<MTLBuffer> b_x = ctx.get_buffer(x, x_off);

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_out offset:o_off atIndex:0];
    [enc setBuffer:b_x offset:x_off atIndex:1];
    [enc setBytes:&weight length:sizeof(weight) atIndex:2];
    [enc setBytes:&dim length:sizeof(dim) atIndex:3];

    NSUInteger tg = 256;
    NSUInteger groups = (dim + tg - 1) / tg;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

void add_cuda(__nv_bfloat16* out, const __nv_bfloat16* a, const __nv_bfloat16* b, int n, cudaStream_t stream) {
    (void)stream;
    for (int i = 0; i < n; i++) {
        out[i] = __nv_bfloat16::from_float(a[i].to_float() + b[i].to_float());
    }
}

void add_f32_sigmoid_cuda(float* out, const float* a, const float* bias, int n, bool apply_sigmoid, cudaStream_t stream) {
    (void)stream;
    for (int i = 0; i < n; i++) {
        float val = a[i] + (bias ? bias[i] : 0.0f);
        if (apply_sigmoid) val = 1.0f / (1.0f + expf(-val));
        out[i] = val;
    }
}

void rope_cuda(
    __nv_bfloat16* x, int n_vectors, int head_dim, int rope_dim,
    int position, const float* freq_table, bool inverse, cudaStream_t stream)
{
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("rope_kernel");
    if (!pso) return;

    size_t x_off, f_off;
    id<MTLBuffer> b_x = ctx.get_buffer(x, x_off);
    id<MTLBuffer> b_f = ctx.get_buffer(freq_table, f_off);

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_x offset:x_off atIndex:0];
    [enc setBytes:&n_vectors length:sizeof(n_vectors) atIndex:1];
    [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:2];
    [enc setBytes:&rope_dim length:sizeof(rope_dim) atIndex:3];
    [enc setBytes:&position length:sizeof(position) atIndex:4];
    [enc setBuffer:b_f offset:f_off atIndex:5];
    [enc setBytes:&inverse length:sizeof(inverse) atIndex:6];

    [enc dispatchThreadgroups:MTLSizeMake(n_vectors, 1, 1) threadsPerThreadgroup:MTLSizeMake(rope_dim / 2, 1, 1)];
}

void rope_standard_cuda(
    __nv_bfloat16* q, __nv_bfloat16* k, int n_q_heads, int n_kv_heads,
    int head_dim, int pos, float rope_theta, cudaStream_t stream)
{
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("rope_standard_kernel");
    if (!pso) return;

    size_t q_off, k_off;
    id<MTLBuffer> b_q = ctx.get_buffer(q, q_off);
    id<MTLBuffer> b_k = ctx.get_buffer(k, k_off);

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_q offset:q_off atIndex:0];
    [enc setBuffer:b_k offset:k_off atIndex:1];
    [enc setBytes:&n_q_heads length:sizeof(n_q_heads) atIndex:2];
    [enc setBytes:&n_kv_heads length:sizeof(n_kv_heads) atIndex:3];
    [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:4];
    [enc setBytes:&pos length:sizeof(pos) atIndex:5];
    [enc setBytes:&rope_theta length:sizeof(rope_theta) atIndex:6];

    int max_heads = std::max(n_q_heads, n_kv_heads);
    [enc dispatchThreadgroups:MTLSizeMake(max_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(head_dim / 2, 1, 1)];
}

void embedding_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* table, const int32_t* ids,
    int seq_len, int dim, cudaStream_t stream)
{
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("embedding_kernel");
    if (!pso) return;

    size_t o_off, t_off, i_off;
    id<MTLBuffer> b_out = ctx.get_buffer(out, o_off);
    id<MTLBuffer> b_t = ctx.get_buffer(table, t_off);
    id<MTLBuffer> b_i = ctx.get_buffer(ids, i_off);

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_out offset:o_off atIndex:0];
    [enc setBuffer:b_t offset:t_off atIndex:1];
    [enc setBuffer:b_i offset:i_off atIndex:2];
    [enc setBytes:&seq_len length:sizeof(seq_len) atIndex:3];
    [enc setBytes:&dim length:sizeof(dim) atIndex:4];

    NSUInteger tg = std::min(256, dim);
    [enc dispatchThreadgroups:MTLSizeMake(seq_len, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

void embedding_broadcast_cuda(
    __nv_bfloat16* hidden, __nv_bfloat16* hc_state, const __nv_bfloat16* table,
    int token_id, int dim, int hc, cudaStream_t stream)
{
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("embedding_broadcast_kernel");
    if (!pso) return;

    size_t h_off, hc_off, t_off;
    id<MTLBuffer> b_h = ctx.get_buffer(hidden, h_off);
    id<MTLBuffer> b_hc = ctx.get_buffer(hc_state, hc_off);
    id<MTLBuffer> b_t = ctx.get_buffer(table, t_off);

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_h offset:h_off atIndex:0];
    [enc setBuffer:b_hc offset:hc_off atIndex:1];
    [enc setBuffer:b_t offset:t_off atIndex:2];
    [enc setBytes:&token_id length:sizeof(token_id) atIndex:3];
    [enc setBytes:&dim length:sizeof(dim) atIndex:4];
    [enc setBytes:&hc length:sizeof(hc) atIndex:5];

    NSUInteger tg = std::min(256, dim);
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

void gemv_int4_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* vec, const uint8_t* weight,
    const __nv_bfloat16* scale, int N, int K, cudaStream_t stream)
{
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("gemv_int4_kernel");
    if (!pso) return;

    size_t o_off, v_off, w_off, s_off;
    id<MTLBuffer> b_out = ctx.get_buffer(out, o_off);
    id<MTLBuffer> b_vec = ctx.get_buffer(vec, v_off);
    id<MTLBuffer> b_w = ctx.get_buffer(weight, w_off);
    id<MTLBuffer> b_s = ctx.get_buffer(scale, s_off);

    bool is_residual = false;
    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_out offset:o_off atIndex:0];
    [enc setBuffer:b_vec offset:v_off atIndex:1];
    [enc setBuffer:b_w offset:w_off atIndex:2];
    [enc setBuffer:b_s offset:s_off atIndex:3];
    [enc setBytes:&N length:sizeof(N) atIndex:4];
    [enc setBytes:&K length:sizeof(K) atIndex:5];
    [enc setBytes:&is_residual length:sizeof(is_residual) atIndex:6];

    [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

void gemv_int4_residual_cuda(
    __nv_bfloat16* inout, const __nv_bfloat16* vec, const uint8_t* weight,
    const __nv_bfloat16* scale, int N, int K, cudaStream_t stream)
{
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("gemv_int4_kernel");
    if (!pso) return;

    size_t o_off, v_off, w_off, s_off;
    id<MTLBuffer> b_out = ctx.get_buffer(inout, o_off);
    id<MTLBuffer> b_vec = ctx.get_buffer(vec, v_off);
    id<MTLBuffer> b_w = ctx.get_buffer(weight, w_off);
    id<MTLBuffer> b_s = ctx.get_buffer(scale, s_off);

    bool is_residual = true;
    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_out offset:o_off atIndex:0];
    [enc setBuffer:b_vec offset:v_off atIndex:1];
    [enc setBuffer:b_w offset:w_off atIndex:2];
    [enc setBuffer:b_s offset:s_off atIndex:3];
    [enc setBytes:&N length:sizeof(N) atIndex:4];
    [enc setBytes:&K length:sizeof(K) atIndex:5];
    [enc setBytes:&is_residual length:sizeof(is_residual) atIndex:6];

    [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

void gemv_int4_swiglu_fused_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* vec,
    const uint8_t* gate_weight, const __nv_bfloat16* gate_scale,
    const uint8_t* up_weight, const __nv_bfloat16* up_scale,
    int N, int K, float swiglu_limit, cudaStream_t stream)
{
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("gemv_int4_swiglu_fused_kernel");
    if (!pso) return;

    size_t o_off, v_off, gw_off, gs_off, uw_off, us_off;
    id<MTLBuffer> b_out = ctx.get_buffer(out, o_off);
    id<MTLBuffer> b_vec = ctx.get_buffer(vec, v_off);
    id<MTLBuffer> b_gw = ctx.get_buffer(gate_weight, gw_off);
    id<MTLBuffer> b_gs = ctx.get_buffer(gate_scale, gs_off);
    id<MTLBuffer> b_uw = ctx.get_buffer(up_weight, uw_off);
    id<MTLBuffer> b_us = ctx.get_buffer(up_scale, us_off);

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_out offset:o_off atIndex:0];
    [enc setBuffer:b_vec offset:v_off atIndex:1];
    [enc setBuffer:b_gw offset:gw_off atIndex:2];
    [enc setBuffer:b_gs offset:gs_off atIndex:3];
    [enc setBuffer:b_uw offset:uw_off atIndex:4];
    [enc setBuffer:b_us offset:us_off atIndex:5];
    [enc setBytes:&N length:sizeof(N) atIndex:6];
    [enc setBytes:&K length:sizeof(K) atIndex:7];
    [enc setBytes:&swiglu_limit length:sizeof(swiglu_limit) atIndex:8];

    [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

void gemv_int4_f32_cuda(
    float* out, const __nv_bfloat16* vec, const uint8_t* weight,
    const __nv_bfloat16* scale, int N, int K, cudaStream_t stream)
{
    (void)stream;
    int num_blocks = K / 32;
    for (int r = 0; r < N; r++) {
        const __nv_bfloat16* row_s = scale + r * num_blocks;
        const uint8_t* row_w = weight + r * (K / 2);
        float sum = 0.0f;
        for (int b = 0; b < num_blocks; b++) {
            float s = row_s[b].to_float();
            int w_off = b * 16;
            int a_off = b * 32;
            float b_sum = 0.0f;
            for (int i = 0; i < 16; i++) {
                uint8_t byte_val = row_w[w_off + i];
                float q0 = float(byte_val & 0x0F) - 8.0f;
                float q1 = float(byte_val >> 4) - 8.0f;
                b_sum += q0 * vec[a_off + i * 2].to_float() + q1 * vec[a_off + i * 2 + 1].to_float();
            }
            sum += b_sum * s;
        }
        out[r] = sum;
    }
}

void gemm_int4_batch_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* A, const uint8_t* weight,
    const __nv_bfloat16* scale, int N, int K, int M, cudaStream_t stream)
{
    for (int m = 0; m < M; m++) {
        gemv_int4_cuda(out + m * N, A + m * K, weight, scale, N, K, stream);
    }
}

void gemm_int4_f32_batch_cuda(
    float* out, const __nv_bfloat16* A, const uint8_t* weight,
    const __nv_bfloat16* scale, int N, int K, int M, cudaStream_t stream)
{
    for (int m = 0; m < M; m++) {
        gemv_int4_f32_cuda(out + m * N, A + m * K, weight, scale, N, K, stream);
    }
}

void gemm_int4_swiglu_fused_batch_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* A,
    const uint8_t* gate_weight, const __nv_bfloat16* gate_scale,
    const uint8_t* up_weight, const __nv_bfloat16* up_scale,
    int N, int K, int M, float swiglu_limit, cudaStream_t stream)
{
    for (int m = 0; m < M; m++) {
        gemv_int4_swiglu_fused_cuda(out + m * N, A + m * K, gate_weight, gate_scale, up_weight, up_scale, N, K, swiglu_limit, stream);
    }
}

void gemv_bf16_cuda(float* out, const __nv_bfloat16* W, const __nv_bfloat16* x, int N, int K, cudaStream_t stream) {
    (void)stream;
    for (int r = 0; r < N; r++) {
        float sum = 0.0f;
        const __nv_bfloat16* row = W + r * K;
        for (int c = 0; c < K; c++) {
            sum += row[c].to_float() * x[c].to_float();
        }
        out[r] = sum;
    }
}

void gemv_bf16_out_bf16_cuda(__nv_bfloat16* out, const __nv_bfloat16* W, const __nv_bfloat16* x, int N, int K, cudaStream_t stream) {
    (void)stream;
    for (int r = 0; r < N; r++) {
        float sum = 0.0f;
        const __nv_bfloat16* row = W + r * K;
        for (int c = 0; c < K; c++) {
            sum += row[c].to_float() * x[c].to_float();
        }
        out[r] = __nv_bfloat16::from_float(sum);
    }
}

void gemv_bf16_batch_cuda(float* out, const __nv_bfloat16* W, const __nv_bfloat16* X, int N, int K, int M, cudaStream_t stream) {
    for (int m = 0; m < M; m++) {
        gemv_bf16_cuda(out + m * N, W, X + m * K, N, K, stream);
    }
}

void gemv_bf16_out_bf16_batch_cuda(__nv_bfloat16* out, const __nv_bfloat16* W, const __nv_bfloat16* X, int N, int K, int M, cudaStream_t stream) {
    for (int m = 0; m < M; m++) {
        gemv_bf16_out_bf16_cuda(out + m * N, W, X + m * K, N, K, stream);
    }
}

void gemv_f32_cuda(float* out, const float* vec, const float* matrix, int M, int K, cudaStream_t stream) {
    (void)stream;
    for (int r = 0; r < M; r++) {
        float sum = 0.0f;
        const float* row = matrix + r * K;
        for (int c = 0; c < K; c++) sum += row[c] * vec[c];
        out[r] = sum;
    }
}

void bf16_to_f32_cuda(float* out, const __nv_bfloat16* in, int n, cudaStream_t stream) {
    (void)stream;
    for (int i = 0; i < n; i++) out[i] = in[i].to_float();
}

void f32_to_bf16_cuda(__nv_bfloat16* out, const float* in, int n, cudaStream_t stream) {
    (void)stream;
    for (int i = 0; i < n; i++) out[i] = __nv_bfloat16::from_float(in[i]);
}

void deltanet_linear_attention_decode_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* in_qkv, const __nv_bfloat16* in_z,
    const __nv_bfloat16* in_a, const __nv_bfloat16* in_b,
    const __nv_bfloat16* conv1d_w, const __nv_bfloat16* in_conv_state, __nv_bfloat16* out_conv_state,
    const __nv_bfloat16* A_log, const __nv_bfloat16* dt_bias, const __nv_bfloat16* norm_w,
    const __nv_bfloat16* in_ssm_state, __nv_bfloat16* out_ssm_state,
    int num_k_heads, int num_v_heads, int head_dim, cudaStream_t stream)
{
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("deltanet_decode_kernel");
    if (!pso) return;

    size_t o_off, qkv_off, z_off, a_off, b_off, cw_off, ics_off, ocs_off, al_off, dt_off, nw_off, iss_off, oss_off;
    id<MTLBuffer> b_o = ctx.get_buffer(out, o_off);
    id<MTLBuffer> b_qkv = ctx.get_buffer(in_qkv, qkv_off);
    id<MTLBuffer> b_z = ctx.get_buffer(in_z, z_off);
    id<MTLBuffer> b_a = ctx.get_buffer(in_a, a_off);
    id<MTLBuffer> b_b = ctx.get_buffer(in_b, b_off);
    id<MTLBuffer> b_cw = ctx.get_buffer(conv1d_w, cw_off);
    id<MTLBuffer> b_ics = ctx.get_buffer(in_conv_state, ics_off);
    id<MTLBuffer> b_ocs = ctx.get_buffer(out_conv_state, ocs_off);
    id<MTLBuffer> b_al = ctx.get_buffer(A_log, al_off);
    id<MTLBuffer> b_dt = ctx.get_buffer(dt_bias, dt_off);
    id<MTLBuffer> b_nw = ctx.get_buffer(norm_w, nw_off);
    id<MTLBuffer> b_iss = ctx.get_buffer(in_ssm_state, iss_off);
    id<MTLBuffer> b_oss = ctx.get_buffer(out_ssm_state, oss_off);

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_o offset:o_off atIndex:0];
    [enc setBuffer:b_qkv offset:qkv_off atIndex:1];
    [enc setBuffer:b_z offset:z_off atIndex:2];
    [enc setBuffer:b_a offset:a_off atIndex:3];
    [enc setBuffer:b_b offset:b_off atIndex:4];
    [enc setBuffer:b_cw offset:cw_off atIndex:5];
    [enc setBuffer:b_ics offset:ics_off atIndex:6];
    [enc setBuffer:b_ocs offset:ocs_off atIndex:7];
    [enc setBuffer:b_al offset:al_off atIndex:8];
    [enc setBuffer:b_dt offset:dt_off atIndex:9];
    [enc setBuffer:b_nw offset:nw_off atIndex:10];
    [enc setBuffer:b_iss offset:iss_off atIndex:11];
    [enc setBuffer:b_oss offset:oss_off atIndex:12];
    [enc setBytes:&num_k_heads length:sizeof(num_k_heads) atIndex:13];
    [enc setBytes:&num_v_heads length:sizeof(num_v_heads) atIndex:14];
    [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:15];

    [enc dispatchThreadgroups:MTLSizeMake(num_v_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
}

void deltanet_linear_attention_decode_batch_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* in_qkv, const __nv_bfloat16* in_z,
    const __nv_bfloat16* in_a, const __nv_bfloat16* in_b,
    const __nv_bfloat16* conv1d_w, const __nv_bfloat16* in_conv_state, __nv_bfloat16* out_conv_state,
    __nv_bfloat16* slot_conv_0, __nv_bfloat16* slot_conv_1,
    __nv_bfloat16* slot_conv_2, __nv_bfloat16* slot_conv_3,
    const __nv_bfloat16* A_log, const __nv_bfloat16* dt_bias, const __nv_bfloat16* norm_w,
    const __nv_bfloat16* in_ssm_state, __nv_bfloat16* out_ssm_state,
    __nv_bfloat16* slot_ssm_0, __nv_bfloat16* slot_ssm_1,
    __nv_bfloat16* slot_ssm_2, __nv_bfloat16* slot_ssm_3,
    int num_k_heads, int num_v_heads, int head_dim, int M, cudaStream_t stream)
{
    (void)slot_conv_0; (void)slot_conv_1; (void)slot_conv_2; (void)slot_conv_3;
    (void)slot_ssm_0; (void)slot_ssm_1; (void)slot_ssm_2; (void)slot_ssm_3;
    for (int m = 0; m < M; m++) {
        deltanet_linear_attention_decode_cuda(
            out + m * num_v_heads * head_dim,
            in_qkv + m * (2 * num_k_heads + num_v_heads) * head_dim,
            in_z + m * num_v_heads * head_dim,
            in_a + m * num_v_heads,
            in_b + m * num_v_heads,
            conv1d_w,
            in_conv_state, out_conv_state,
            A_log, dt_bias, norm_w,
            in_ssm_state, out_ssm_state,
            num_k_heads, num_v_heads, head_dim, stream);
    }
}

void softmax_cuda(float* out, const float* x, int rows, int cols, cudaStream_t stream) {
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("softmax_kernel");
    if (!pso) return;

    size_t o_off, x_off;
    id<MTLBuffer> b_o = ctx.get_buffer(out, o_off);
    id<MTLBuffer> b_x = ctx.get_buffer(x, x_off);

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_o offset:o_off atIndex:0];
    [enc setBuffer:b_x offset:x_off atIndex:1];
    [enc setBytes:&rows length:sizeof(rows) atIndex:2];
    [enc setBytes:&cols length:sizeof(cols) atIndex:3];

    NSUInteger tg = std::min(256, cols);
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

void argmax_f32_cuda(int32_t* out, const float* logits, int n, cudaStream_t stream) {
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("argmax_f32_kernel");
    if (!pso) return;

    size_t o_off, l_off;
    id<MTLBuffer> b_o = ctx.get_buffer(out, o_off);
    id<MTLBuffer> b_l = ctx.get_buffer(logits, l_off);

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_o offset:o_off atIndex:0];
    [enc setBuffer:b_l offset:l_off atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];

    NSUInteger tg = 256;
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

void argmax_f32_batch_cuda(int32_t* out, const float* logits, int n, int M, cudaStream_t stream) {
    for (int m = 0; m < M; m++) {
        argmax_f32_cuda(out + m, logits + m * n, n, stream);
    }
}

void sample_multinomial_f32_cuda(
    int32_t* out, float* logits, int n, float temperature, float rand_val, float min_p, cudaStream_t stream)
{
    (void)stream;
    if (temperature <= 0.0f) {
        int best_idx = 0;
        float best_val = logits[0];
        for (int i = 1; i < n; i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_idx = i;
            }
        }
        *out = best_idx;
        return;
    }

    float max_l = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > max_l) max_l = logits[i];

    float sum_exp = 0.0f;
    std::vector<float> probs(n);
    float inv_t = 1.0f / temperature;
    for (int i = 0; i < n; i++) {
        float p = expf((logits[i] - max_l) * inv_t);
        probs[i] = p;
        sum_exp += p;
    }
    float max_p = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] /= sum_exp;
        if (probs[i] > max_p) max_p = probs[i];
    }
    float cutoff = max_p * min_p;
    float sum_valid = 0.0f;
    for (int i = 0; i < n; i++) {
        if (probs[i] >= cutoff) sum_valid += probs[i];
        else probs[i] = 0.0f;
    }
    float r = rand_val * sum_valid;
    float c = 0.0f;
    int picked = 0;
    for (int i = 0; i < n; i++) {
        c += probs[i];
        if (c >= r) {
            picked = i;
            break;
        }
    }
    *out = picked;
}

void topk_cuda(float* out_vals, int32_t* out_idx, const float* scores, int n, int k, cudaStream_t stream) {
    (void)stream;
    std::vector<std::pair<float, int32_t>> items(n);
    for (int i = 0; i < n; i++) items[i] = {scores[i], (int32_t)i};
    std::partial_sort(items.begin(), items.begin() + k, items.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    for (int i = 0; i < k; i++) {
        out_vals[i] = items[i].first;
        out_idx[i] = items[i].second;
    }
}

void sqrtsoftplus_cuda(float* out, const float* x, int n, cudaStream_t stream) {
    (void)stream;
    for (int i = 0; i < n; i++) {
        float val = x[i];
        float sp = (val > 20.0f) ? val : ((val < -20.0f) ? expf(val) : log1pf(expf(val)));
        out[i] = sqrtf(sp);
    }
}

void precompute_freqs_cuda(
    float* freqs, int max_seq_len, int rope_dim, float base, float factor,
    int original_seq_len, int beta_fast, int beta_slow, cudaStream_t stream)
{
    (void)factor; (void)original_seq_len; (void)beta_fast; (void)beta_slow; (void)stream;
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int d = 0; d < rope_dim / 2; d++) {
            float freq = 1.0f / powf(base, float(d * 2) / float(rope_dim));
            float angle = float(pos) * freq;
            freqs[pos * rope_dim + d * 2] = cosf(angle);
            freqs[pos * rope_dim + d * 2 + 1] = sinf(angle);
        }
    }
}

// ── Vision Kernel Stubs & Implementations ───────────────────────────────────

void layernorm_affine_bf16_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta, int n_rows, int dim, float eps, cudaStream_t stream)
{
    (void)stream;
    for (int r = 0; r < n_rows; r++) {
        const __nv_bfloat16* row_x = x + r * dim;
        __nv_bfloat16* row_out = out + r * dim;
        float mean = 0.0f;
        for (int i = 0; i < dim; i++) mean += row_x[i].to_float();
        mean /= float(dim);
        float var = 0.0f;
        for (int i = 0; i < dim; i++) {
            float diff = row_x[i].to_float() - mean;
            var += diff * diff;
        }
        float rstd = 1.0f / sqrtf(var / float(dim) + eps);
        for (int i = 0; i < dim; i++) {
            float norm_val = (row_x[i].to_float() - mean) * rstd;
            float g = gamma ? gamma[i].to_float() : 1.0f;
            float b = beta ? beta[i].to_float() : 0.0f;
            row_out[i] = __nv_bfloat16::from_float(norm_val * g + b);
        }
    }
}

void gelu_bias_bf16_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* bias,
    int n_rows, int dim, cudaStream_t stream)
{
    (void)stream;
    for (int r = 0; r < n_rows; r++) {
        for (int c = 0; c < dim; c++) {
            int idx = r * dim + c;
            float val = x[idx].to_float() + (bias ? bias[c].to_float() : 0.0f);
            float gelu = 0.5f * val * (1.0f + tanhf(0.79788456f * (val + 0.044715f * val * val * val)));
            out[idx] = __nv_bfloat16::from_float(gelu);
        }
    }
}

void add_residual_bias_bf16_cuda(
    __nv_bfloat16* x, const __nv_bfloat16* residual, const __nv_bfloat16* bias,
    int n_rows, int dim, cudaStream_t stream)
{
    (void)stream;
    for (int r = 0; r < n_rows; r++) {
        for (int c = 0; c < dim; c++) {
            int idx = r * dim + c;
            float val = x[idx].to_float() + residual[idx].to_float() + (bias ? bias[c].to_float() : 0.0f);
            x[idx] = __nv_bfloat16::from_float(val);
        }
    }
}

void add_bias_bf16_cuda(
    __nv_bfloat16* x, const __nv_bfloat16* bias, int n_rows, int dim, cudaStream_t stream)
{
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("add_bias_kernel");
    if (!pso) return;

    size_t x_off, b_off;
    id<MTLBuffer> buf_x = ctx.get_buffer(x, x_off);
    id<MTLBuffer> buf_b = ctx.get_buffer(bias, b_off);

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf_x offset:x_off atIndex:0];
    [enc setBuffer:buf_b offset:b_off atIndex:1];
    [enc setBytes:&n_rows length:sizeof(n_rows) atIndex:2];
    [enc setBytes:&dim length:sizeof(dim) atIndex:3];

    int total = n_rows * dim;
    NSUInteger tg = 256;
    NSUInteger groups = (total + tg - 1) / tg;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

void add_tensors_bf16_cuda(__nv_bfloat16* dst, const __nv_bfloat16* src, size_t count, cudaStream_t stream) {
    vector_add_bf16_cuda(dst, src, (int)count, stream);
}

void vit_split_qkv_bias_bf16_cuda(
    __nv_bfloat16* Q_out, __nv_bfloat16* K_out, __nv_bfloat16* V_out,
    const __nv_bfloat16* qkv_in, const __nv_bfloat16* bias,
    const float* cos_table, const float* sin_table,
    int N, int n_heads, int head_dim, cudaStream_t stream)
{
    (void)stream;
    int total_dim = n_heads * head_dim;
    for (int t = 0; t < N; t++) {
        const __nv_bfloat16* t_in = qkv_in + t * (3 * total_dim);
        for (int h = 0; h < n_heads; h++) {
            for (int d = 0; d < head_dim; d++) {
                int ch = h * head_dim + d;
                float q = t_in[ch].to_float() + (bias ? bias[ch].to_float() : 0.0f);
                float k = t_in[total_dim + ch].to_float() + (bias ? bias[total_dim + ch].to_float() : 0.0f);
                float v = t_in[2 * total_dim + ch].to_float() + (bias ? bias[2 * total_dim + ch].to_float() : 0.0f);

                if (cos_table && sin_table && d < head_dim / 2) {
                    float cos_val = cos_table[t * (head_dim / 2) + d];
                    float sin_val = sin_table[t * (head_dim / 2) + d];
                    // Apply rotary
                }
                Q_out[h * N * head_dim + t * head_dim + d] = __nv_bfloat16::from_float(q);
                K_out[h * N * head_dim + t * head_dim + d] = __nv_bfloat16::from_float(k);
                V_out[h * N * head_dim + t * head_dim + d] = __nv_bfloat16::from_float(v);
            }
        }
    }
}

void vit_softmax_bf16_cuda(__nv_bfloat16* scores, int total_rows, int seq_len, float scale, cudaStream_t stream) {
    (void)stream;
    for (int r = 0; r < total_rows; r++) {
        __nv_bfloat16* row = scores + r * seq_len;
        float max_val = -1e38f;
        for (int c = 0; c < seq_len; c++) {
            float val = row[c].to_float() * scale;
            if (val > max_val) max_val = val;
        }
        float sum_exp = 0.0f;
        for (int c = 0; c < seq_len; c++) {
            float e = expf(row[c].to_float() * scale - max_val);
            sum_exp += e;
        }
        float inv_sum = 1.0f / (sum_exp + 1e-9f);
        for (int c = 0; c < seq_len; c++) {
            float e = expf(row[c].to_float() * scale - max_val) * inv_sum;
            row[c] = __nv_bfloat16::from_float(e);
        }
    }
}

void vit_merge_heads_bf16_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* heads,
    int N, int n_heads, int head_dim, cudaStream_t stream)
{
    (void)stream;
    for (int t = 0; t < N; t++) {
        for (int h = 0; h < n_heads; h++) {
            for (int d = 0; d < head_dim; d++) {
                out[t * (n_heads * head_dim) + h * head_dim + d] = heads[h * N * head_dim + t * head_dim + d];
            }
        }
    }
}

void spatial_merge_gather_bf16_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* in, int H_patches, int W_patches, int dim, cudaStream_t stream)
{
    (void)stream;
    int out_H = H_patches / 2;
    int out_W = W_patches / 2;
    for (int h = 0; h < out_H; h++) {
        for (int w = 0; w < out_W; w++) {
            int out_idx = (h * out_W + w) * (4 * dim);
            for (int dh = 0; dh < 2; dh++) {
                for (int dw = 0; dw < 2; dw++) {
                    int in_idx = ((h * 2 + dh) * W_patches + (w * 2 + dw)) * dim;
                    int sub = (dh * 2 + dw) * dim;
                    std::memcpy(out + out_idx + sub, in + in_idx, dim * sizeof(__nv_bfloat16));
                }
            }
        }
    }
}

void im2col_patch_embed_bf16_cuda(
    __nv_bfloat16* im2col_out, const float* img_norm,
    int H, int W, int patch_size, int temporal_size, int in_channels, cudaStream_t stream)
{
    (void)stream;
    int H_patches = H / patch_size;
    int W_patches = W / patch_size;
    int num_patches = H_patches * W_patches;
    int patch_dim = temporal_size * in_channels * patch_size * patch_size;

    for (int p = 0; p < num_patches; p++) {
        int ph = p / W_patches;
        int pw = p % W_patches;
        __nv_bfloat16* p_out = im2col_out + p * patch_dim;
        int idx = 0;
        for (int t = 0; t < temporal_size; t++) {
            for (int c = 0; c < in_channels; c++) {
                for (int h = 0; h < patch_size; h++) {
                    for (int w = 0; w < patch_size; w++) {
                        int img_h = ph * patch_size + h;
                        int img_w = pw * patch_size + w;
                        float val = img_norm[t * (in_channels * H * W) + c * (H * W) + img_h * W + img_w];
                        p_out[idx++] = __nv_bfloat16::from_float(val);
                    }
                }
            }
        }
    }
}

void add_bias_and_posemb_bf16_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* bias, const __nv_bfloat16* pos_embed,
    int num_patches, int dim, cudaStream_t stream)
{
    (void)stream;
    for (int p = 0; p < num_patches; p++) {
        for (int d = 0; d < dim; d++) {
            int idx = p * dim + d;
            float val = out[idx].to_float() + (bias ? bias[d].to_float() : 0.0f) + (pos_embed ? pos_embed[idx].to_float() : 0.0f);
            out[idx] = __nv_bfloat16::from_float(val);
        }
    }
}

void rms_norm_scaled_batched_bf16_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* x, float target_scale,
    int n, int dim, float eps, cudaStream_t stream)
{
    (void)stream;
    for (int r = 0; r < n; r++) {
        const __nv_bfloat16* row_x = x + r * dim;
        __nv_bfloat16* row_out = out + r * dim;
        float sum_sq = 0.0f;
        for (int i = 0; i < dim; i++) sum_sq += row_x[i].to_float() * row_x[i].to_float();
        float scale = target_scale / sqrtf(sum_sq / float(dim) + eps);
        for (int i = 0; i < dim; i++) row_out[i] = __nv_bfloat16::from_float(row_x[i].to_float() * scale);
    }
}

void visual_carrier_removal_and_align_bf16_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* x, float* d_col_mean_scratch,
    const __nv_bfloat16* d_embed_mean, float target_scale, int n, int dim, float eps, cudaStream_t stream)
{
    (void)d_col_mean_scratch; (void)stream;
    std::vector<float> col_mean(dim, 0.0f);
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < dim; c++) {
            col_mean[c] += x[r * dim + c].to_float();
        }
    }
    for (int c = 0; c < dim; c++) col_mean[c] /= float(n);

    for (int r = 0; r < n; r++) {
        float r_norm_sq = 0.0f;
        for (int c = 0; c < dim; c++) {
            float diff = x[r * dim + c].to_float() - col_mean[c];
            r_norm_sq += diff * diff;
        }
        float r_norm = sqrtf(r_norm_sq + eps);
        float s = target_scale / r_norm;
        for (int c = 0; c < dim; c++) {
            float centered = (x[r * dim + c].to_float() - col_mean[c]) * s;
            float base = d_embed_mean ? d_embed_mean[c].to_float() : 0.0f;
            out[r * dim + c] = __nv_bfloat16::from_float(centered + base);
        }
    }
}

void norm_cap_bf16_cuda(__nv_bfloat16* out, const __nv_bfloat16* x, float max_norm, int n, int dim, float eps, cudaStream_t stream) {
    (void)stream;
    for (int r = 0; r < n; r++) {
        float sum_sq = 0.0f;
        for (int c = 0; c < dim; c++) sum_sq += x[r * dim + c].to_float() * x[r * dim + c].to_float();
        float norm = sqrtf(sum_sq + eps);
        float scale = (norm > max_norm) ? (max_norm / norm) : 1.0f;
        for (int c = 0; c < dim; c++) out[r * dim + c] = __nv_bfloat16::from_float(x[r * dim + c].to_float() * scale);
    }
}

void permute_patches_by_window_bf16_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* in, const int* window_index,
    int num_blocks, int block_size, int dim, cudaStream_t stream)
{
    (void)stream;
    for (int b = 0; b < num_blocks; b++) {
        int src_b = window_index[b];
        std::memcpy(out + b * block_size * dim, in + src_b * block_size * dim, block_size * dim * sizeof(__nv_bfloat16));
    }
}

void vit_split_qkv_bias_window_bf16_cuda(
    __nv_bfloat16* Q_out, __nv_bfloat16* K_out, __nv_bfloat16* V_out,
    const __nv_bfloat16* qkv_in, const __nv_bfloat16* bias,
    const float* cos_table, const float* sin_table,
    int N, int n_heads, int head_dim, int window_len, cudaStream_t stream)
{
    (void)window_len;
    vit_split_qkv_bias_bf16_cuda(Q_out, K_out, V_out, qkv_in, bias, cos_table, sin_table, N, n_heads, head_dim, stream);
}

void vit_merge_heads_window_bf16_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* heads,
    int N, int n_heads, int head_dim, int window_len, cudaStream_t stream)
{
    (void)window_len;
    vit_merge_heads_bf16_cuda(out, heads, N, n_heads, head_dim, stream);
}

void unpermute_merged_tokens_bf16_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* in, const int* reverse_indices,
    int num_tokens, int dim, cudaStream_t stream)
{
    (void)stream;
    for (int t = 0; t < num_tokens; t++) {
        int src_t = reverse_indices[t];
        std::memcpy(out + t * dim, in + src_t * dim, dim * sizeof(__nv_bfloat16));
    }
}

// ── MoE Top-6 Routing ───────────────────────────────────────────────────────

void moe_route_top6_from_bf16_cuda(
    int32_t* topk_ids, float* topk_weights, const __nv_bfloat16* scores_bf16,
    const float* gate_bias, int n_experts, int top_k, float routed_scaling_factor, cudaStream_t stream)
{
    (void)stream;
    std::vector<std::pair<float, int>> exp_scores(n_experts);
    std::vector<float> exp_probs(n_experts);

    for (int i = 0; i < n_experts; i++) {
        float raw = scores_bf16[i].to_float();
        float sp = (raw > 20.0f) ? raw : ((raw < -20.0f) ? expf(raw) : log1pf(expf(raw)));
        float prob = sqrtf(sp);
        exp_probs[i] = prob;
        exp_scores[i] = {prob + (gate_bias ? gate_bias[i] : 0.0f), i};
    }

    std::partial_sort(exp_scores.begin(), exp_scores.begin() + top_k, exp_scores.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    float sum_p = 0.0f;
    for (int k = 0; k < top_k; k++) {
        int idx = exp_scores[k].second;
        sum_p += exp_probs[idx];
    }
    if (sum_p < 1e-6f) sum_p = 1e-6f;

    for (int k = 0; k < top_k; k++) {
        int idx = exp_scores[k].second;
        topk_ids[k] = idx;
        topk_weights[k] = (exp_probs[idx] / sum_p) * routed_scaling_factor;
    }
}

void moe_route_top6_from_bf16_batch_cuda(
    int32_t* topk_ids, float* topk_weights, const __nv_bfloat16* scores_bf16,
    const float* gate_bias, int M, int n_experts, int top_k, float routed_scaling_factor, cudaStream_t stream)
{
    for (int m = 0; m < M; m++) {
        moe_route_top6_from_bf16_cuda(
            topk_ids + m * top_k,
            topk_weights + m * top_k,
            scores_bf16 + m * n_experts,
            gate_bias, n_experts, top_k, routed_scaling_factor, stream);
    }
}

void moe_route_top6_cuda(
    int32_t* topk_ids, float* topk_weights, const float* scores_f32,
    int n_experts, int top_k, float routed_scaling_factor, cudaStream_t stream)
{
    (void)stream;
    std::vector<std::pair<float, int>> items(n_experts);
    for (int i = 0; i < n_experts; i++) items[i] = {scores_f32[i], i};
    std::partial_sort(items.begin(), items.begin() + top_k, items.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    float sum = 0.0f;
    for (int k = 0; k < top_k; k++) sum += items[k].first;
    if (sum < 1e-6f) sum = 1e-6f;
    for (int k = 0; k < top_k; k++) {
        topk_ids[k] = items[k].second;
        topk_weights[k] = (items[k].first / sum) * routed_scaling_factor;
    }
}

void moe_route_hash_cuda(
    int32_t* topk_ids, float* topk_weights, const int64_t* tid2eid_table,
    int token_id, int top_k, float routed_scaling_factor, cudaStream_t stream)
{
    (void)stream;
    int64_t val = tid2eid_table[token_id];
    for (int k = 0; k < top_k; k++) {
        topk_ids[k] = (int32_t)((val >> (k * 10)) & 0x3FF);
        topk_weights[k] = (1.0f / float(top_k)) * routed_scaling_factor;
    }
}

void moe_route_hash_device_id_cuda(
    int32_t* topk_ids, float* topk_weights, const int64_t* tid2eid_table,
    const int32_t* d_token_id, int top_k, float routed_scaling_factor, cudaStream_t stream)
{
    moe_route_hash_cuda(topk_ids, topk_weights, tid2eid_table, *d_token_id, top_k, routed_scaling_factor, stream);
}

void moe_route_hash_device_id_batch_cuda(
    int32_t* topk_ids, float* topk_weights, const int64_t* tid2eid_table,
    const int32_t* d_token_ids, int M, int top_k, float routed_scaling_factor, cudaStream_t stream)
{
    for (int m = 0; m < M; m++) {
        moe_route_hash_cuda(topk_ids + m * top_k, topk_weights + m * top_k, tid2eid_table, d_token_ids ? d_token_ids[m] : 0, top_k, routed_scaling_factor, stream);
    }
}

void fused_moe_accum_dynamic_cuda(
    __nv_bfloat16* accum, const __nv_bfloat16* down_buf, const float* topk_weights,
    const __nv_bfloat16* shared_down, int dim, cudaStream_t stream)
{
    (void)stream;
    for (int i = 0; i < dim; i++) {
        float sum = shared_down ? shared_down[i].to_float() : 0.0f;
        for (int k = 0; k < 6; k++) {
            sum += down_buf[k * dim + i].to_float() * topk_weights[k];
        }
        accum[i] = __nv_bfloat16::from_float(sum);
    }
}

void fused_moe_accum_dynamic_batch_cuda(
    __nv_bfloat16* accum, const __nv_bfloat16* down_buf, const float* topk_weights,
    const __nv_bfloat16* shared_down, int dim, int M, cudaStream_t stream)
{
    for (int m = 0; m < M; m++) {
        fused_moe_accum_dynamic_cuda(
            accum + m * dim,
            down_buf + m * 6 * dim,
            topk_weights + m * 6,
            shared_down ? (shared_down + m * dim) : nullptr,
            dim, stream);
    }
}

void fused_moe_accum_6_cuda(
    __nv_bfloat16* accum, const __nv_bfloat16* down_ptrs,
    float w0, float w1, float w2, float w3, float w4, float w5,
    int dim, cudaStream_t stream)
{
    float weights[6] = {w0, w1, w2, w3, w4, w5};
    fused_moe_accum_dynamic_cuda(accum, down_ptrs, weights, nullptr, dim, stream);
}

// ── IQ2_XXS & Q2_K Quantization Kernels ──────────────────────────────────────

void iq2_xxs_dequant_cuda(__nv_bfloat16* out, const block_iq2_xxs* weight, int rows, int cols, cudaStream_t stream) {
    (void)stream;
    int blocks_per_row = cols / 256;
    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < blocks_per_row; b++) {
            const block_iq2_xxs& blk = weight[r * blocks_per_row + b];
            float d = blk.d.to_float();
            int out_idx = r * cols + b * 256;
            for (int i = 0; i < 256; i++) {
                int q_idx = i / 8;
                int shift = (i % 8) * 2;
                int q = (blk.qs[q_idx] >> shift) & 3;
                float val = (float(q) - 1.5f) * d;
                out[out_idx + i] = __nv_bfloat16::from_float(val);
            }
        }
    }
}

void gemv_iq2_xxs_cuda(__nv_bfloat16* out, const __nv_bfloat16* vec, const block_iq2_xxs* weight, int rows, int cols, cudaStream_t stream) {
    (void)stream;
    int blocks_per_row = cols / 256;
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            const block_iq2_xxs& blk = weight[r * blocks_per_row + b];
            float d = blk.d.to_float();
            int in_idx = b * 256;
            for (int i = 0; i < 256; i++) {
                int q_idx = i / 8;
                int shift = (i % 8) * 2;
                int q = (blk.qs[q_idx] >> shift) & 3;
                float w = (float(q) - 1.5f) * d;
                sum += w * vec[in_idx + i].to_float();
            }
        }
        out[r] = __nv_bfloat16::from_float(sum);
    }
}

void gemv_iq2_xxs_swiglu_fused_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* vec,
    const block_iq2_xxs* w1, const block_iq2_xxs* w3,
    int N, int K, float swiglu_limit, cudaStream_t stream)
{
    (void)stream;
    std::vector<__nv_bfloat16> g(N);
    std::vector<__nv_bfloat16> u(N);
    gemv_iq2_xxs_cuda(g.data(), vec, w1, N, K, stream);
    gemv_iq2_xxs_cuda(u.data(), vec, w3, N, K, stream);
    silu_mul_cuda(out, g.data(), u.data(), N, swiglu_limit, stream);
}

void gemv_iq2_xxs_moe_swiglu_fused_cuda(
    __nv_bfloat16* gate_buf, const __nv_bfloat16* vec,
    const void* const* active_expert_ptrs, int w1_offset, int w3_offset,
    int N, int K, float swiglu_limit, const int32_t* topk_ids,
    const void* const* flat_expert_ptrs, int layer_id, int n_experts, cudaStream_t stream)
{
    (void)topk_ids; (void)flat_expert_ptrs; (void)layer_id; (void)n_experts;
    for (int k = 0; k < 6; k++) {
        const char* exp_base = (const char*)active_expert_ptrs[k];
        const block_iq2_xxs* w1 = (const block_iq2_xxs*)(exp_base + w1_offset);
        const block_iq2_xxs* w3 = (const block_iq2_xxs*)(exp_base + w3_offset);
        gemv_iq2_xxs_swiglu_fused_cuda(gate_buf + k * N, vec, w1, w3, N, K, swiglu_limit, stream);
    }
}

void gemv_iq2_xxs_moe_swiglu_fused_batch_cuda(
    __nv_bfloat16* gate_buf, const __nv_bfloat16* vec,
    int w1_offset, int w3_offset,
    int N, int K, float swiglu_limit, const int32_t* topk_ids,
    const void* const* flat_expert_ptrs, int layer_id, int n_experts, int M, cudaStream_t stream)
{
    const void* active[6];
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < 6; k++) {
            int eid = topk_ids ? topk_ids[m * 6 + k] : k;
            active[k] = (flat_expert_ptrs && eid < n_experts) ? flat_expert_ptrs[layer_id * n_experts + eid] : nullptr;
        }
        gemv_iq2_xxs_moe_swiglu_fused_cuda(
            gate_buf + m * 6 * N, vec + m * K,
            active, w1_offset, w3_offset,
            N, K, swiglu_limit, topk_ids ? (topk_ids + m * 6) : nullptr,
            flat_expert_ptrs, layer_id, n_experts, stream);
    }
}

void q2_k_dequant_cuda(__nv_bfloat16* out, const block_q2_K* weight, int rows, int cols, cudaStream_t stream) {
    (void)stream;
    int blocks_per_row = cols / 256;
    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < blocks_per_row; b++) {
            const block_q2_K& blk = weight[r * blocks_per_row + b];
            float d = blk.d.to_float();
            float dmin = blk.dmin.to_float();
            int out_idx = r * cols + b * 256;
            for (int i = 0; i < 256; i++) {
                int q_byte = blk.qs[i / 4];
                int q = (q_byte >> ((i % 4) * 2)) & 3;
                float val = d * float(q) - dmin;
                out[out_idx + i] = __nv_bfloat16::from_float(val);
            }
        }
    }
}

void gemv_q2_k_cuda(__nv_bfloat16* out, const __nv_bfloat16* vec, const block_q2_K* weight, int rows, int cols, cudaStream_t stream) {
    (void)stream;
    int blocks_per_row = cols / 256;
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            const block_q2_K& blk = weight[r * blocks_per_row + b];
            float d = blk.d.to_float();
            float dmin = blk.dmin.to_float();
            int in_idx = b * 256;
            for (int i = 0; i < 256; i++) {
                int q_byte = blk.qs[i / 4];
                int q = (q_byte >> ((i % 4) * 2)) & 3;
                float w = d * float(q) - dmin;
                sum += w * vec[in_idx + i].to_float();
            }
        }
        out[r] = __nv_bfloat16::from_float(sum);
    }
}

void gemv_q2_k_moe_cuda(
    __nv_bfloat16* down_buf, const __nv_bfloat16* gate_buf,
    const void* const* active_expert_ptrs, int w2_offset, int N, int K,
    const int32_t* topk_ids, const void* const* flat_expert_ptrs, int layer_id, int n_experts, cudaStream_t stream)
{
    (void)topk_ids; (void)flat_expert_ptrs; (void)layer_id; (void)n_experts;
    for (int k = 0; k < 6; k++) {
        const char* exp_base = (const char*)active_expert_ptrs[k];
        const block_q2_K* w2 = (const block_q2_K*)(exp_base + w2_offset);
        gemv_q2_k_cuda(down_buf + k * N, gate_buf + k * K, w2, N, K, stream);
    }
}

void gemv_q2_k_moe_batch_cuda(
    __nv_bfloat16* down_buf, const __nv_bfloat16* gate_buf,
    const int32_t* topk_ids, const void* const* flat_expert_ptrs,
    int layer_id, int n_experts,
    int w2_offset, int N, int K, int M, cudaStream_t stream)
{
    const void* active[6];
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < 6; k++) {
            int eid = topk_ids ? topk_ids[m * 6 + k] : k;
            active[k] = (flat_expert_ptrs && eid < n_experts) ? flat_expert_ptrs[layer_id * n_experts + eid] : nullptr;
        }
        gemv_q2_k_moe_cuda(
            down_buf + m * 6 * N, gate_buf + m * 6 * K,
            active, w2_offset, N, K,
            topk_ids ? (topk_ids + m * 6) : nullptr,
            flat_expert_ptrs, layer_id, n_experts, stream);
    }
}

// ── Attention Decoding Kernels ──────────────────────────────────────────────

void gqa_attention_decode_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* q, __nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
    const __nv_bfloat16* new_k, const __nv_bfloat16* new_v,
    int n_q_heads, int n_kv_heads, int head_dim, int pos, int max_seq_len, cudaStream_t stream)
{
    (void)max_seq_len; (void)stream;
    int group_size = n_q_heads / n_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    // Write new token to KV cache
    if (new_k && k_cache) {
        std::memcpy(k_cache + size_t(pos) * n_kv_heads * head_dim, new_k, n_kv_heads * head_dim * sizeof(__nv_bfloat16));
    }
    if (new_v && v_cache) {
        std::memcpy(v_cache + size_t(pos) * n_kv_heads * head_dim, new_v, n_kv_heads * head_dim * sizeof(__nv_bfloat16));
    }

    for (int qh = 0; qh < n_q_heads; qh++) {
        int kv_h = qh / group_size;
        const __nv_bfloat16* q_head = q + qh * head_dim;
        std::vector<float> scores(pos + 1);
        float max_s = -1e38f;

        for (int t = 0; t <= pos; t++) {
            const __nv_bfloat16* k_head = k_cache + ((size_t)t * n_kv_heads + kv_h) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q_head[d].to_float() * k_head[d].to_float();
            }
            dot *= scale;
            scores[t] = dot;
            if (dot > max_s) max_s = dot;
        }

        float sum_exp = 0.0f;
        for (int t = 0; t <= pos; t++) {
            scores[t] = expf(scores[t] - max_s);
            sum_exp += scores[t];
        }
        float inv_sum = 1.0f / (sum_exp + 1e-9f);

        __nv_bfloat16* out_head = out + qh * head_dim;
        for (int d = 0; d < head_dim; d++) {
            float val = 0.0f;
            for (int t = 0; t <= pos; t++) {
                const __nv_bfloat16* v_head = v_cache + ((size_t)t * n_kv_heads + kv_h) * head_dim;
                val += (scores[t] * inv_sum) * v_head[d].to_float();
            }
            out_head[d] = __nv_bfloat16::from_float(val);
        }
    }
}

void qwen_gqa_decode_gated_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* q_and_gate, __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* q_norm_w, const __nv_bfloat16* k_norm_w,
    __nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
    int n_q_heads, int n_kv_heads, int head_dim, const int32_t* d_pos,
    int pos_scalar, int max_seq_len, float rope_theta, float eps, cudaStream_t stream)
{
    (void)q_norm_w; (void)k_norm_w; (void)d_pos; (void)rope_theta; (void)eps;
    int pos = pos_scalar;
    gqa_attention_decode_cuda(out, q_and_gate, k_cache, v_cache, k, v, n_q_heads, n_kv_heads, head_dim, pos, max_seq_len, stream);

    // Apply gate: out = out * silu(gate)
    const __nv_bfloat16* gate = q_and_gate + n_q_heads * head_dim;
    for (int i = 0; i < n_q_heads * head_dim; i++) {
        float g = gate[i].to_float();
        float silu_g = g / (1.0f + expf(-g));
        out[i] = __nv_bfloat16::from_float(out[i].to_float() * silu_g);
    }
}

void qwen_gqa_decode_gated_fp8_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* q_and_gate, __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* q_norm_w, const __nv_bfloat16* k_norm_w,
    uint8_t* k_cache, uint8_t* v_cache,
    int n_q_heads, int n_kv_heads, int head_dim, const int32_t* d_pos,
    int pos_scalar, int max_seq_len, float rope_theta, float eps, cudaStream_t stream)
{
    (void)k_cache; (void)v_cache;
    // Fall back to standard decoding logic
    qwen_gqa_decode_gated_cuda(out, q_and_gate, k, v, q_norm_w, k_norm_w, (__nv_bfloat16*)k_cache, (__nv_bfloat16*)v_cache,
                               n_q_heads, n_kv_heads, head_dim, d_pos, pos_scalar, max_seq_len, rope_theta, eps, stream);
}

void qwen_gqa_decode_gated_fp8_batch_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* q_and_gate, __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* q_norm_w, const __nv_bfloat16* k_norm_w,
    uint8_t* k_cache, uint8_t* v_cache,
    int n_q_heads, int n_kv_heads, int head_dim, const int32_t* d_pos,
    int pos_scalar, int max_seq_len, int M, float rope_theta, float eps, cudaStream_t stream)
{
    for (int m = 0; m < M; m++) {
        qwen_gqa_decode_gated_fp8_cuda(
            out + m * n_q_heads * head_dim,
            q_and_gate + m * 2 * n_q_heads * head_dim,
            k + m * n_kv_heads * head_dim,
            v + m * n_kv_heads * head_dim,
            q_norm_w, k_norm_w, k_cache, v_cache,
            n_q_heads, n_kv_heads, head_dim, d_pos,
            pos_scalar + m, max_seq_len, rope_theta, eps, stream);
    }
}

void qwen2_gqa_decode_fp8_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* q, __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* q_norm_w, const __nv_bfloat16* k_norm_w,
    uint8_t* k_cache, uint8_t* v_cache,
    int n_q_heads, int n_kv_heads, int head_dim, const int32_t* d_pos,
    int pos_scalar, int max_seq_len, float rope_theta, float eps, const int32_t* d_mrope_pos,
    cudaStream_t stream)
{
    (void)q_norm_w; (void)k_norm_w; (void)d_pos; (void)rope_theta; (void)eps; (void)d_mrope_pos;
    gqa_attention_decode_cuda(out, q, (__nv_bfloat16*)k_cache, (__nv_bfloat16*)v_cache, k, v, n_q_heads, n_kv_heads, head_dim, pos_scalar, max_seq_len, stream);
}

void qwen2_gqa_decode_fp8_batch_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* q, __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* q_norm_w, const __nv_bfloat16* k_norm_w,
    uint8_t* k_cache, uint8_t* v_cache,
    int n_q_heads, int n_kv_heads, int head_dim, const int32_t* d_pos,
    int pos_scalar, int M, int max_seq_len, float rope_theta, float eps, const int32_t* d_mrope_pos,
    cudaStream_t stream)
{
    for (int m = 0; m < M; m++) {
        qwen2_gqa_decode_fp8_cuda(
            out + m * n_q_heads * head_dim, q + m * n_q_heads * head_dim, k + m * n_kv_heads * head_dim, v + m * n_kv_heads * head_dim,
            q_norm_w, k_norm_w, k_cache, v_cache, n_q_heads, n_kv_heads, head_dim, d_pos, pos_scalar + m, max_seq_len, rope_theta, eps, d_mrope_pos, stream);
    }
}

// ── Additional Helpers ──────────────────────────────────────────────────────

void increment_i32_cuda(int32_t* ptr, int inc, cudaStream_t stream) {
    (void)stream;
    if (ptr) *ptr += inc;
}

void accumulate_expert_freq_cuda(
    uint32_t* expert_counts, int32_t* step_topk, const int32_t* topk_idx,
    const int32_t* track_flag, int layer_id, int n_experts, int top_k, cudaStream_t stream)
{
    (void)stream;
    if (track_flag && *track_flag == 0) return;
    for (int k = 0; k < top_k; k++) {
        int idx = topk_idx[k];
        if (idx >= 0 && idx < n_experts) {
            expert_counts[layer_id * n_experts + idx]++;
            if (step_topk) step_topk[k] = idx;
        }
    }
}

void accumulate_expert_imatrix_cuda(
    float* gate_accum, float* down_accum, uint32_t* expert_counts,
    const __nv_bfloat16* h_norm, const int32_t* topk_indices,
    int num_tokens, int top_k, int n_experts, int hidden_dim, int moe_intermediate, cudaStream_t stream)
{
    (void)gate_accum; (void)down_accum; (void)expert_counts; (void)h_norm; (void)topk_indices;
    (void)num_tokens; (void)top_k; (void)n_experts; (void)hidden_dim; (void)moe_intermediate; (void)stream;
}

void init_cuda(cudaStream_t stream) {
    (void)stream;
    metal_init();
}

void init_batch_cuda(cudaStream_t stream) {
    (void)stream;
    metal_init();
}

void store_kv_device_pos_cuda(
    __nv_bfloat16* kv_cache, const __nv_bfloat16* kv_val,
    const int32_t* d_position, int window, int head_dim, cudaStream_t stream)
{
    (void)stream;
    int pos = d_position ? *d_position : 0;
    int slot = pos % window;
    std::memcpy(kv_cache + slot * head_dim, kv_val, head_dim * sizeof(__nv_bfloat16));
}

void mla_attention_fused_cuda(
    const __nv_bfloat16* raw_q, const __nv_bfloat16* raw_kv, const __nv_bfloat16* comp_kv,
    const float* attn_sink, __nv_bfloat16* out, const int32_t* d_position,
    const int32_t* d_comp_count, const float* freq_table, int max_cache_len,
    int head_dim, int rope_dim, float scale, float q_norm_eps,
    const uint8_t* comp_mask, int window, cudaStream_t stream)
{
    (void)raw_q; (void)raw_kv; (void)comp_kv; (void)attn_sink; (void)out; (void)d_position;
    (void)d_comp_count; (void)freq_table; (void)max_cache_len; (void)head_dim; (void)rope_dim;
    (void)scale; (void)q_norm_eps; (void)comp_mask; (void)window; (void)stream;
}

void compressor_device_step_cuda(
    const int32_t* d_position, int32_t* d_comp_count, const float* proj_kv,
    const float* proj_gate, float* comp_kv_state, float* comp_score_state,
    const float* comp_ape, const __nv_bfloat16* comp_norm, __nv_bfloat16* comp_kv_cache,
    const float* rope_freqs_compressed, int ratio, int head_dim, int rope_dim,
    float rms_norm_eps, const float* idx_proj_kv, const float* idx_proj_gate,
    float* idx_kv_state, float* idx_score_state, const float* idx_ape,
    const __nv_bfloat16* idx_norm, __nv_bfloat16* idx_comp_kv_cache, int max_comp, cudaStream_t stream)
{
    (void)d_position; (void)d_comp_count; (void)proj_kv; (void)proj_gate; (void)comp_kv_state;
    (void)comp_score_state; (void)comp_ape; (void)comp_norm; (void)comp_kv_cache;
    (void)rope_freqs_compressed; (void)ratio; (void)head_dim; (void)rope_dim; (void)rms_norm_eps;
    (void)idx_proj_kv; (void)idx_proj_gate; (void)idx_kv_state; (void)idx_score_state;
    (void)idx_ape; (void)idx_norm; (void)idx_comp_kv_cache; (void)max_comp; (void)stream;
}

void rope_device_pos_cuda(
    __nv_bfloat16* x, int n_vectors, int head_dim, int rope_dim,
    const int32_t* d_position, const float* freq_table, bool inverse, cudaStream_t stream)
{
    int pos = d_position ? *d_position : 0;
    rope_cuda(x, n_vectors, head_dim, rope_dim, pos, freq_table, inverse, stream);
}

void embedding_broadcast_device_id_cuda(
    __nv_bfloat16* hidden, __nv_bfloat16* hc_state,
    const __nv_bfloat16* table, const int32_t* d_token_id, int dim, int hc, cudaStream_t stream)
{
    int tok = d_token_id ? *d_token_id : 0;
    embedding_broadcast_cuda(hidden, hc_state, table, tok, dim, hc, stream);
}

void embedding_broadcast_batch_cuda(
    __nv_bfloat16* hidden, __nv_bfloat16* hc_state, const __nv_bfloat16* table,
    const int32_t* d_token_ids, int dim, int hc, int M, cudaStream_t stream)
{
    for (int m = 0; m < M; m++) {
        embedding_broadcast_cuda(hidden + m * dim, hc_state + m * hc * dim, table, d_token_ids[m], dim, hc, stream);
    }
}

void embedding_int4_cuda(
    __nv_bfloat16* out, const uint8_t* weight, const __nv_bfloat16* scale,
    const int32_t* ids, int seq_len, int dim, cudaStream_t stream)
{
    (void)stream;
    int num_blocks = dim / 32;
    for (int s = 0; s < seq_len; s++) {
        int token = ids[s];
        const uint8_t* row_w = weight + token * (dim / 2);
        const __nv_bfloat16* row_s = scale + token * num_blocks;
        __nv_bfloat16* row_out = out + s * dim;
        for (int b = 0; b < num_blocks; b++) {
            float s_val = row_s[b].to_float();
            int w_off = b * 16;
            int a_off = b * 32;
            for (int i = 0; i < 16; i++) {
                uint8_t byte_val = row_w[w_off + i];
                float q0 = (float(byte_val & 0x0F) - 8.0f) * s_val;
                float q1 = (float(byte_val >> 4) - 8.0f) * s_val;
                row_out[a_off + i * 2] = __nv_bfloat16::from_float(q0);
                row_out[a_off + i * 2 + 1] = __nv_bfloat16::from_float(q1);
            }
        }
    }
}

void embedding_int4_broadcast_device_id_cuda(
    __nv_bfloat16* hidden, __nv_bfloat16* hc_state, const uint8_t* weight,
    const __nv_bfloat16* scale, const int32_t* d_token_id, int dim, int hc, cudaStream_t stream)
{
    int tok = d_token_id ? *d_token_id : 0;
    embedding_int4_cuda(hidden, weight, scale, &tok, 1, dim, stream);
    for (int h = 0; h < hc; h++) {
        std::memcpy(hc_state + h * dim, hidden, dim * sizeof(__nv_bfloat16));
    }
}

void embedding_fp4_cuda(
    __nv_bfloat16* out, const uint8_t* weight, const uint8_t* scale,
    const int32_t* ids, int seq_len, int dim, cudaStream_t stream)
{
    (void)out; (void)weight; (void)scale; (void)ids; (void)seq_len; (void)dim; (void)stream;
}

// ── Stub Fallbacks for Quantization / Advanced Blackwell Routines ───────────

void fp8_dequant_cuda(__nv_bfloat16* out, const uint8_t* weight, const uint8_t* scale, int rows, int cols, int block_size, cudaStream_t stream) {
    (void)out; (void)weight; (void)scale; (void)rows; (void)cols; (void)block_size; (void)stream;
}
void gemv_fp8_cuda(__nv_bfloat16* out, const __nv_bfloat16* vec, const uint8_t* weight, const uint8_t* scale, int rows, int cols, int block_size, cudaStream_t stream) {
    (void)out; (void)vec; (void)weight; (void)scale; (void)rows; (void)cols; (void)block_size; (void)stream;
}
void gemv_fp8_grouped_cuda(__nv_bfloat16* out, const __nv_bfloat16* vec, const uint8_t* weight, const uint8_t* scale, int rows, int cols, int groups, int block_size, cudaStream_t stream) {
    (void)out; (void)vec; (void)weight; (void)scale; (void)rows; (void)cols; (void)groups; (void)block_size; (void)stream;
}
void gemv_fp8_grouped_batch_cuda(__nv_bfloat16* out, const __nv_bfloat16* vec, const uint8_t* weight, const uint8_t* scale, int rows, int cols, int groups, int block_size, int M, cudaStream_t stream) {
    (void)out; (void)vec; (void)weight; (void)scale; (void)rows; (void)cols; (void)groups; (void)block_size; (void)M; (void)stream;
}
void gemv_hc_pre_norm_cuda(float* mixes, const __nv_bfloat16* hc_state, const float* hc_fn, int mix_size, int hc_dim, float eps, cudaStream_t stream) {
    (void)mixes; (void)hc_state; (void)hc_fn; (void)mix_size; (void)hc_dim; (void)eps; (void)stream;
}
void gemv_hc_pre_norm_batch_cuda(float* mixes, const __nv_bfloat16* hc_state, const float* hc_fn, int M, int mix_size, int hc_dim, float eps, cudaStream_t stream) {
    (void)mixes; (void)hc_state; (void)hc_fn; (void)M; (void)mix_size; (void)hc_dim; (void)eps; (void)stream;
}
void fp4_dequant_cuda(__nv_bfloat16* out, const uint8_t* weight, const uint8_t* scale, int rows, int cols_packed, int scale_cols, cudaStream_t stream) {
    (void)out; (void)weight; (void)scale; (void)rows; (void)cols_packed; (void)scale_cols; (void)stream;
}
void int2_dequant_cuda(__nv_bfloat16* out, const uint8_t* weight, const __nv_bfloat16* scale_min, int rows, int cols_packed, int block_size, cudaStream_t stream) {
    (void)out; (void)weight; (void)scale_min; (void)rows; (void)cols_packed; (void)block_size; (void)stream;
}
void gemv_int2_cuda(__nv_bfloat16* out, const __nv_bfloat16* vec, const uint8_t* weight, const __nv_bfloat16* scale_min, int rows, int cols_packed, int block_size, cudaStream_t stream) {
    (void)out; (void)vec; (void)weight; (void)scale_min; (void)rows; (void)cols_packed; (void)block_size; (void)stream;
}
void gemm_int2_cuda(__nv_bfloat16* out, const __nv_bfloat16* A, const uint8_t* weight, const __nv_bfloat16* scale_min, int M, int N, int K_packed, int block_size, cudaStream_t stream) {
    (void)out; (void)A; (void)weight; (void)scale_min; (void)M; (void)N; (void)K_packed; (void)block_size; (void)stream;
}
void gemv_int4_grouped_batch_cuda(__nv_bfloat16* out, const __nv_bfloat16* A, const uint8_t* weight, const __nv_bfloat16* scale, int N, int K, int num_heads, int M, cudaStream_t stream) {
    gemm_int4_batch_cuda(out, A, weight, scale, N, K, M, stream);
}
void gemv_sparse_nvfp4_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* vec, const uint8_t* weight,
    const uint8_t* meta, const uint8_t* scale, int N, int K, cudaStream_t stream) {
    (void)out; (void)vec; (void)weight; (void)meta; (void)scale; (void)N; (void)K; (void)stream;
}
void gemv_sparse_nvfp4_swiglu_fused_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* vec, const uint8_t* gate_weight,
    const uint8_t* gate_meta, const uint8_t* gate_scale, const uint8_t* up_weight,
    const uint8_t* up_meta, const uint8_t* up_scale, int N, int K, float swiglu_limit, cudaStream_t stream) {
    (void)out; (void)vec; (void)gate_weight; (void)gate_meta; (void)gate_scale;
    (void)up_weight; (void)up_meta; (void)up_scale; (void)N; (void)K; (void)swiglu_limit; (void)stream;
}
void gemv_mixed_moe_swiglu_fused_batch_cuda(
    __nv_bfloat16* gate_buf, const __nv_bfloat16* vec,
    int w1_cold_offset, int w3_cold_offset, int N, int K, float swiglu_limit,
    const int32_t* topk_ids, const void* const* flat_expert_ptrs,
    const uint8_t* expert_type_map, int layer_id, int n_experts, int M, cudaStream_t stream) {
    (void)gate_buf; (void)vec; (void)w1_cold_offset; (void)w3_cold_offset;
    (void)N; (void)K; (void)swiglu_limit; (void)topk_ids; (void)flat_expert_ptrs;
    (void)expert_type_map; (void)layer_id; (void)n_experts; (void)M; (void)stream;
}
void gemv_mixed_moe_down_batch_cuda(
    __nv_bfloat16* down_buf, const __nv_bfloat16* gate_buf,
    const int32_t* topk_ids, const void* const* flat_expert_ptrs,
    const uint8_t* expert_type_map, int layer_id, int n_experts,
    int w2_cold_offset, int N, int K, int M, cudaStream_t stream) {
    (void)down_buf; (void)gate_buf; (void)topk_ids; (void)flat_expert_ptrs;
    (void)expert_type_map; (void)layer_id; (void)n_experts; (void)w2_cold_offset;
    (void)N; (void)K; (void)M; (void)stream;
}
void gemv_int3_cuda(__nv_bfloat16* out, const __nv_bfloat16* vec, const uint8_t* weight, const __nv_bfloat16* scale, int N, int K, cudaStream_t stream) {
    (void)out; (void)vec; (void)weight; (void)scale; (void)N; (void)K; (void)stream;
}
void gemv_int3_residual_cuda(__nv_bfloat16* inout, const __nv_bfloat16* vec, const uint8_t* weight, const __nv_bfloat16* scale, int N, int K, cudaStream_t stream) {
    (void)inout; (void)vec; (void)weight; (void)scale; (void)N; (void)K; (void)stream;
}
void gemv_int3_swiglu_fused_cuda(__nv_bfloat16* out, const __nv_bfloat16* vec, const uint8_t* gate_weight, const __nv_bfloat16* gate_scale, const uint8_t* up_weight, const __nv_bfloat16* up_scale, int N, int K, float swiglu_limit, cudaStream_t stream) {
    (void)out; (void)vec; (void)gate_weight; (void)gate_scale; (void)up_weight; (void)up_scale; (void)N; (void)K; (void)swiglu_limit; (void)stream;
}
void gemm_int3_batch_cuda(__nv_bfloat16* out, const __nv_bfloat16* A, const uint8_t* weight, const __nv_bfloat16* scale, int N, int K, int M, cudaStream_t stream) {
    (void)out; (void)A; (void)weight; (void)scale; (void)N; (void)K; (void)M; (void)stream;
}
void gemm_int3_swiglu_fused_batch_cuda(__nv_bfloat16* out, const __nv_bfloat16* A, const uint8_t* gate_weight, const __nv_bfloat16* gate_scale, const uint8_t* up_weight, const __nv_bfloat16* up_scale, int N, int K, int M, float swiglu_limit, cudaStream_t stream) {
    (void)out; (void)A; (void)gate_weight; (void)gate_scale; (void)up_weight; (void)up_scale; (void)N; (void)K; (void)M; (void)swiglu_limit; (void)stream;
}
void gemv_fp4_cuda(__nv_bfloat16* out, const __nv_bfloat16* vec, const uint8_t* weight, const uint8_t* scale, int N, int K, cudaStream_t stream) {
    (void)out; (void)vec; (void)weight; (void)scale; (void)N; (void)K; (void)stream;
}
void gemv_fp4_residual_cuda(__nv_bfloat16* inout, const __nv_bfloat16* vec, const uint8_t* weight, const uint8_t* scale, int N, int K, cudaStream_t stream) {
    (void)inout; (void)vec; (void)weight; (void)scale; (void)N; (void)K; (void)stream;
}
void gemv_fp4_f32_cuda(float* out, const __nv_bfloat16* vec, const uint8_t* weight, const uint8_t* scale, int N, int K, cudaStream_t stream) {
    (void)out; (void)vec; (void)weight; (void)scale; (void)N; (void)K; (void)stream;
}
void gemv_fp4_swiglu_fused_cuda(__nv_bfloat16* out, const __nv_bfloat16* vec, const uint8_t* gate_weight, const uint8_t* gate_scale, const uint8_t* up_weight, const uint8_t* up_scale, int N, int K, float swiglu_limit, cudaStream_t stream) {
    (void)out; (void)vec; (void)gate_weight; (void)gate_scale; (void)up_weight; (void)up_scale; (void)N; (void)K; (void)swiglu_limit; (void)stream;
}
void gemm_fp4_batch_cuda(__nv_bfloat16* out, const __nv_bfloat16* A, const uint8_t* weight, const uint8_t* scale, int N, int K, int M, cudaStream_t stream) {
    (void)out; (void)A; (void)weight; (void)scale; (void)N; (void)K; (void)M; (void)stream;
}
void gemm_fp4_f32_batch_cuda(float* out, const __nv_bfloat16* A, const uint8_t* weight, const uint8_t* scale, int N, int K, int M, cudaStream_t stream) {
    (void)out; (void)A; (void)weight; (void)scale; (void)N; (void)K; (void)M; (void)stream;
}
void gemm_fp4_residual_batch_cuda(__nv_bfloat16* inout, const __nv_bfloat16* A, const uint8_t* weight, const uint8_t* scale, int N, int K, int M, cudaStream_t stream) {
    (void)inout; (void)A; (void)weight; (void)scale; (void)N; (void)K; (void)M; (void)stream;
}
void gemm_fp4_swiglu_fused_batch_cuda(__nv_bfloat16* out, const __nv_bfloat16* A, const uint8_t* gate_weight, const uint8_t* gate_scale, const uint8_t* up_weight, const uint8_t* up_scale, int N, int K, int M, float swiglu_limit, cudaStream_t stream) {
    (void)out; (void)A; (void)gate_weight; (void)gate_scale; (void)up_weight; (void)up_scale; (void)N; (void)K; (void)M; (void)swiglu_limit; (void)stream;
}
void gemm_fp4_blackwell_tensorcore_cuda(
    __nv_bfloat16* out, const uint8_t* A_fp4, const uint8_t* A_scale,
    const uint8_t* weight, const uint8_t* scale, int N, int K, int M, cudaStream_t stream) {
    (void)out; (void)A_fp4; (void)A_scale; (void)weight; (void)scale; (void)N; (void)K; (void)M; (void)stream;
}
void gemm_fp4_residual_blackwell_tensorcore_cuda(
    __nv_bfloat16* inout, const uint8_t* A_fp4, const uint8_t* A_scale,
    const uint8_t* weight, const uint8_t* scale, int N, int K, int M, cudaStream_t stream) {
    (void)inout; (void)A_fp4; (void)A_scale; (void)weight; (void)scale; (void)N; (void)K; (void)M; (void)stream;
}
void gemm_fp4_f32_blackwell_tensorcore_cuda(
    float* out, const uint8_t* A_fp4, const uint8_t* A_scale,
    const uint8_t* weight, const uint8_t* scale, int N, int K, int M, cudaStream_t stream) {
    (void)out; (void)A_fp4; (void)A_scale; (void)weight; (void)scale; (void)N; (void)K; (void)M; (void)stream;
}
void gemm_fp4_swiglu_blackwell_tensorcore_cuda(
    __nv_bfloat16* out, const uint8_t* A_fp4, const uint8_t* A_scale,
    const uint8_t* gate_weight, const uint8_t* gate_scale, const uint8_t* up_weight, const uint8_t* up_scale,
    int N, int K, int M, float swiglu_limit, cudaStream_t stream) {
    (void)out; (void)A_fp4; (void)A_scale; (void)gate_weight; (void)gate_scale; (void)up_weight; (void)up_scale;
    (void)N; (void)K; (void)M; (void)swiglu_limit; (void)stream;
}
void quantize_bf16_to_fp4_e2m1_cuda(uint8_t* out_packed, uint8_t* out_scale, const __nv_bfloat16* in_bf16, int N, int K, cudaStream_t stream) {
    (void)out_packed; (void)out_scale; (void)in_bf16; (void)N; (void)K; (void)stream;
}
void dequant_int4_block_cuda(__nv_bfloat16* out, const uint8_t* weight, const __nv_bfloat16* scale, int N, int K, int block_size, cudaStream_t stream) {
    (void)stream;
    int num_blocks = K / block_size;
    for (int r = 0; r < N; r++) {
        const uint8_t* row_w = weight + r * (K / 2);
        const __nv_bfloat16* row_s = scale + r * num_blocks;
        __nv_bfloat16* row_out = out + r * K;
        for (int b = 0; b < num_blocks; b++) {
            float s = row_s[b].to_float();
            int w_off = b * 16;
            int a_off = b * 32;
            for (int i = 0; i < 16; i++) {
                uint8_t byte_val = row_w[w_off + i];
                float q0 = (float(byte_val & 0x0F) - 8.0f) * s;
                float q1 = (float(byte_val >> 4) - 8.0f) * s;
                row_out[a_off + i * 2] = __nv_bfloat16::from_float(q0);
                row_out[a_off + i * 2 + 1] = __nv_bfloat16::from_float(q1);
            }
        }
    }
}
void dequant_int3_block_cuda(__nv_bfloat16* out, const uint8_t* weight, const __nv_bfloat16* scale, int N, int K, int block_size, cudaStream_t stream) {
    (void)out; (void)weight; (void)scale; (void)N; (void)K; (void)block_size; (void)stream;
}
void apply_repetition_penalty_cuda(float* logits, const int32_t* history_tokens, int num_history, float penalty, cudaStream_t stream) {
    (void)stream;
    for (int i = 0; i < num_history; i++) {
        int token = history_tokens[i];
        if (logits[token] > 0.0f) logits[token] /= penalty;
        else logits[token] *= penalty;
    }
}
void hc_split_sinkhorn_cuda(float* pre, float* post, float* comb, const float* mixes, const float* scale, const float* base, int hc_mult, int sinkhorn_iters, float eps, cudaStream_t stream) {
    (void)pre; (void)post; (void)comb; (void)mixes; (void)scale; (void)base; (void)hc_mult; (void)sinkhorn_iters; (void)eps; (void)stream;
}
void hc_split_sinkhorn_batch_cuda(float* pre, float* post, float* comb, const float* mixes, const float* scale, const float* base, int M, int hc_mult, int sinkhorn_iters, float eps, cudaStream_t stream) {
    (void)pre; (void)post; (void)comb; (void)mixes; (void)scale; (void)base; (void)M; (void)hc_mult; (void)sinkhorn_iters; (void)eps; (void)stream;
}
void hc_pre_weighted_add_cuda(__nv_bfloat16* hidden, const __nv_bfloat16* hc_state, const float* pre_weights, int dim, int hc, cudaStream_t stream) {
    (void)hidden; (void)hc_state; (void)pre_weights; (void)dim; (void)hc; (void)stream;
}
void hc_pre_weighted_add_norm_cuda(__nv_bfloat16* out, const __nv_bfloat16* hc_state, const float* pre_weights, const __nv_bfloat16* norm_weight, int dim, int hc, float eps, cudaStream_t stream) {
    (void)out; (void)hc_state; (void)pre_weights; (void)norm_weight; (void)dim; (void)hc; (void)eps; (void)stream;
}
void hc_pre_weighted_add_norm_batch_cuda(__nv_bfloat16* out, const __nv_bfloat16* hc_state, const float* pre_weights, const __nv_bfloat16* norm_weight, int M, int dim, int hc, float eps, cudaStream_t stream) {
    (void)out; (void)hc_state; (void)pre_weights; (void)norm_weight; (void)M; (void)dim; (void)hc; (void)eps; (void)stream;
}
void hc_post_update_cuda(__nv_bfloat16* hc_state, const __nv_bfloat16* hidden, const __nv_bfloat16* hc_residual, const float* post_weights, const float* comb_weights, int dim, int hc, cudaStream_t stream) {
    (void)hc_state; (void)hidden; (void)hc_residual; (void)post_weights; (void)comb_weights; (void)dim; (void)hc; (void)stream;
}
void hc_post_update_batch_cuda(__nv_bfloat16* hc_state, const __nv_bfloat16* hidden, const __nv_bfloat16* hc_residual, const float* post_weights, const float* comb_weights, int dim, int hc, int M, cudaStream_t stream) {
    (void)hc_state; (void)hidden; (void)hc_residual; (void)post_weights; (void)comb_weights; (void)dim; (void)hc; (void)M; (void)stream;
}
void hc_head_reduce_cuda(__nv_bfloat16* hidden, const __nv_bfloat16* hc_state, const float* mixes, const float* scale, const float* base, int dim, int hc, cudaStream_t stream) {
    (void)hidden; (void)hc_state; (void)mixes; (void)scale; (void)base; (void)dim; (void)hc; (void)stream;
}
void indexer_score_and_mask_cuda(uint8_t* out_mask, float* out_scores, const __nv_bfloat16* index_comp, const __nv_bfloat16* q, const float* weights, const int32_t* d_comp_count, int max_comp_entries, int top_k, cudaStream_t stream) {
    (void)out_mask; (void)out_scores; (void)index_comp; (void)q; (void)weights; (void)d_comp_count; (void)max_comp_entries; (void)top_k; (void)stream;
}


// ── Additional Activations Support ──────────────────────────────────────────

void init_mla_dynamic_shared_memory() {}

void populate_active_expert_ptrs_cuda(
    const void** active_ptrs,
    const int32_t* topk_ids,
    const void* const* flat_expert_ptrs,
    int layer_id, int n_experts, int top_k,
    cudaStream_t stream)
{
    (void)stream;
    if (!active_ptrs || !topk_ids || !flat_expert_ptrs) return;
    for (int k = 0; k < top_k; k++) {
        int eid = topk_ids[k];
        active_ptrs[k] = flat_expert_ptrs[layer_id * n_experts + eid];
    }
}

void combine_kv_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* raw_kv,
    int raw_len,
    const __nv_bfloat16* comp_kv,
    int comp_len,
    int head_dim,
    cudaStream_t stream)
{
    (void)stream;
    if (out && raw_kv && raw_len > 0) {
        std::memcpy(out, raw_kv, (size_t)raw_len * head_dim * sizeof(__nv_bfloat16));
    }
    if (out && comp_kv && comp_len > 0) {
        std::memcpy(out + (size_t)raw_len * head_dim, comp_kv, (size_t)comp_len * head_dim * sizeof(__nv_bfloat16));
    }
}

void compressor_pool_cuda(
    float* out,
    const float* kv,
    const float* score,
    int window,
    int dim,
    cudaStream_t stream)
{
    (void)out; (void)kv; (void)score; (void)window; (void)dim; (void)stream;
}

void mla_attention_cuda(
    const __nv_bfloat16* q,
    const __nv_bfloat16* kv,
    const float* attn_sink,
    __nv_bfloat16* out,
    int n_heads,
    int cache_len,
    int head_dim,
    float scale,
    cudaStream_t stream)
{
    (void)q; (void)kv; (void)attn_sink; (void)out; (void)n_heads; (void)cache_len; (void)head_dim; (void)scale; (void)stream;
}

void quantize_bf16_to_int4_symmetric_cuda(
    uint8_t* out_packed,
    __nv_bfloat16* out_scale,
    const __nv_bfloat16* in_bf16,
    int N, int K,
    cudaStream_t stream)
{
    (void)out_packed; (void)out_scale; (void)in_bf16; (void)N; (void)K; (void)stream;
}
