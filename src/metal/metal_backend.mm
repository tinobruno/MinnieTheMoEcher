// metal_backend.mm — High-performance Metal runtime & kernel dispatch for Apple Silicon
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <arm_neon.h>

#include "metal_backend.h"
#include "activations.cuh"
#include "vision_kernels.cuh"

#include <iostream>
#include <unordered_map>
#include <unordered_set>
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
    id<MTLCommandBuffer> last_submitted_cmd_buf;
    int encoder_count;

    MetalStreamObj(id<MTLCommandQueue> q)
        : queue(q), current_cmd_buf(nil), current_encoder(nil), last_submitted_cmd_buf(nil), encoder_count(0) {}

    id<MTLComputeCommandEncoder> get_encoder() {
        if (!current_cmd_buf) {
            current_cmd_buf = [queue commandBuffer];
            encoder_count = 0;
        }
        if (!current_encoder) {
            current_encoder = [current_cmd_buf computeCommandEncoder];
            encoder_count++;
        }
        return current_encoder;
    }

    void end_encoder() {
        if (current_encoder) {
            [current_encoder endEncoding];
            current_encoder = nil;
        }
    }

    void commit_async() {
        end_encoder();
        if (current_cmd_buf) {
            [current_cmd_buf commit];
            last_submitted_cmd_buf = current_cmd_buf;
            current_cmd_buf = nil;
            encoder_count = 0;
        }
    }

    void end_encoder_and_maybe_commit(int batch_limit = 64) {
        end_encoder();
        if (encoder_count >= batch_limit) {
            commit_async();
        }
    }

    id<MTLCommandBuffer> get_command_buffer() {
        if (!current_cmd_buf) {
            current_cmd_buf = [queue commandBuffer];
            encoder_count = 0;
        }
        end_encoder();
        return current_cmd_buf;
    }

    void commit_and_wait() {
        commit_async();
        if (last_submitted_cmd_buf) {
            [last_submitted_cmd_buf waitUntilCompleted];
            last_submitted_cmd_buf = nil;
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

    id<MTLBuffer> mps_scratch_w_fp16 = nil;
    id<MTLBuffer> mps_scratch_w_up_fp16 = nil;
    id<MTLBuffer> mps_scratch_a_fp16 = nil;
    id<MTLBuffer> mps_scratch_c_fp16 = nil;
    id<MTLBuffer> mps_scratch_gate_fp16 = nil;
    std::mutex mps_mutex;

    void ensure_mps_scratch() {
        if (mps_scratch_w_fp16) return;
        std::lock_guard<std::mutex> lock(mps_mutex);
        if (mps_scratch_w_fp16) return;
        @autoreleasepool {
            mps_scratch_w_fp16 = [device newBufferWithLength:(size_t)17408 * 5120 * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            mps_scratch_w_up_fp16 = [device newBufferWithLength:(size_t)17408 * 5120 * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            mps_scratch_a_fp16 = [device newBufferWithLength:(size_t)512 * 17408 * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            mps_scratch_c_fp16 = [device newBufferWithLength:(size_t)512 * 17408 * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            mps_scratch_gate_fp16 = [device newBufferWithLength:(size_t)512 * 17408 * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        }
    }

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
    if (kind == cudaMemcpyDeviceToHost) {
        metal_stream_synchronize(stream);
    }
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
    s->end_encoder_and_maybe_commit();
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
    s->end_encoder_and_maybe_commit();
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
    s->end_encoder_and_maybe_commit();
}

void rms_norm_one_centered_cuda_batched(
    __nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* weight,
    int n, int dim, float eps, cudaStream_t stream)
{
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("rms_norm_one_centered_batched_kernel");
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
    s->end_encoder_and_maybe_commit();
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
    s->end_encoder_and_maybe_commit();
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
    s->end_encoder_and_maybe_commit();
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
    s->end_encoder_and_maybe_commit();
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
    s->end_encoder_and_maybe_commit();
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
    s->end_encoder_and_maybe_commit();
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
    s->end_encoder_and_maybe_commit();
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
    s->end_encoder_and_maybe_commit();
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
    s->end_encoder_and_maybe_commit();
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
    s->end_encoder_and_maybe_commit();
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
    if (!b_out || !b_vec || !b_w || !b_s) {
        int num_blocks = K / 32;
        dispatch_apply(N, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t r) {
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
            inout[r] = __nv_bfloat16::from_float(inout[r].to_float() + sum);
        });
        return;
    }

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
    s->end_encoder_and_maybe_commit();
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
    s->end_encoder_and_maybe_commit();
}

void gemv_int4_f32_cuda(
    float* out, const __nv_bfloat16* vec, const uint8_t* weight,
    const __nv_bfloat16* scale, int N, int K, cudaStream_t stream)
{
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("gemv_int4_f32_kernel");
    if (!pso) {
        int num_blocks = K / 32;
        dispatch_apply(N, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t r) {
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
        });
        return;
    }

    size_t o_off, v_off, w_off, s_off;
    id<MTLBuffer> b_out = ctx.get_buffer(out, o_off);
    id<MTLBuffer> b_vec = ctx.get_buffer(vec, v_off);
    id<MTLBuffer> b_w = ctx.get_buffer(weight, w_off);
    id<MTLBuffer> b_s = ctx.get_buffer(scale, s_off);
    if (!b_out || !b_vec || !b_w || !b_s) {
        int num_blocks = K / 32;
        dispatch_apply(N, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t r) {
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
        });
        return;
    }

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_out offset:o_off atIndex:0];
    [enc setBuffer:b_vec offset:v_off atIndex:1];
    [enc setBuffer:b_w offset:w_off atIndex:2];
    [enc setBuffer:b_s offset:s_off atIndex:3];
    [enc setBytes:&N length:sizeof(N) atIndex:4];
    [enc setBytes:&K length:sizeof(K) atIndex:5];

    [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    s->end_encoder_and_maybe_commit();
}

void gemm_int4_batch_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* A, const uint8_t* weight,
    const __nv_bfloat16* scale, int N, int K, int M, cudaStream_t stream)
{
    if (M <= 0) return;
    if (M == 1) {
        gemv_int4_cuda(out, A, weight, scale, N, K, stream);
        return;
    }

    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);

    // Ultra-fast Hardware MPS Tensor Core path for prefill (M >= 4)
    if (M >= 4) {
        id<MTLComputePipelineState> pso_dequant4 = ctx.get_pipeline("dequant_int4_to_fp16_kernel");
        id<MTLComputePipelineState> pso_b2f = ctx.get_pipeline("bf16_to_fp16_kernel");
        id<MTLComputePipelineState> pso_f2b = ctx.get_pipeline("fp16_to_bf16_kernel");

        size_t o_off, a_off, w_off, s_off;
        id<MTLBuffer> b_out = ctx.get_buffer(out, o_off);
        id<MTLBuffer> b_a = ctx.get_buffer(A, a_off);
        id<MTLBuffer> b_w = ctx.get_buffer(weight, w_off);
        id<MTLBuffer> b_s = ctx.get_buffer(scale, s_off);

        if (pso_dequant4 && pso_b2f && pso_f2b && b_out && b_a && b_w && b_s && N <= 17408 && K <= 17408 && M <= 512) {
            ctx.ensure_mps_scratch();

            // 1. Convert A to FP16 and Dequantize W to FP16 in 1 compute encoder
            id<MTLComputeCommandEncoder> enc = s->get_encoder();
            [enc setComputePipelineState:pso_b2f];
            [enc setBuffer:ctx.mps_scratch_a_fp16 offset:0 atIndex:0];
            [enc setBuffer:b_a offset:a_off atIndex:1];
            int count_a = M * K;
            [enc setBytes:&count_a length:4 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(count_a, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc setComputePipelineState:pso_dequant4];
            [enc setBuffer:ctx.mps_scratch_w_fp16 offset:0 atIndex:0];
            [enc setBuffer:b_w offset:w_off atIndex:1];
            [enc setBuffer:b_s offset:s_off atIndex:2];
            [enc setBytes:&N length:4 atIndex:3];
            [enc setBytes:&K length:4 atIndex:4];
            MTLSize grid = MTLSizeMake(K / 64, N, 1);
            MTLSize tg = MTLSizeMake(16, 8, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            // 2. Hardware MPS GEMM: C = A * W^T
            id<MTLCommandBuffer> cb = s->get_command_buffer();
            MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:K * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor* descW = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:K rowBytes:K * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:N * sizeof(uint16_t) dataType:MPSDataTypeFloat16];

            MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:ctx.mps_scratch_a_fp16 descriptor:descA];
            MPSMatrix* matW = [[MPSMatrix alloc] initWithBuffer:ctx.mps_scratch_w_fp16 descriptor:descW];
            MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:ctx.mps_scratch_c_fp16 descriptor:descC];

            MPSMatrixMultiplication* mul = [[MPSMatrixMultiplication alloc] initWithDevice:ctx.device transposeLeft:false transposeRight:true resultRows:M resultColumns:N interiorColumns:K alpha:1.0f beta:0.0f];
            [mul encodeToCommandBuffer:cb leftMatrix:matA rightMatrix:matW resultMatrix:matC];

            // 3. Convert C back to BF16 (into out)
            id<MTLComputeCommandEncoder> enc2 = s->get_encoder();
            [enc2 setComputePipelineState:pso_f2b];
            [enc2 setBuffer:b_out offset:o_off atIndex:0];
            [enc2 setBuffer:ctx.mps_scratch_c_fp16 offset:0 atIndex:1];
            int count_out = M * N;
            [enc2 setBytes:&count_out length:4 atIndex:2];
            bool is_res = false;
            [enc2 setBytes:&is_res length:sizeof(is_res) atIndex:3];
            [enc2 dispatchThreads:MTLSizeMake(count_out, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            s->end_encoder_and_maybe_commit();
            return;
        }
    }

    fprintf(stderr, "[WARN] gemm_int4_batch_cuda MPS BYPASSED! M=%d\n", M);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("gemm_int4_batch_kernel");
    if (!pso) {
        for (int m = 0; m < M; m++) {
            gemv_int4_cuda(out + (size_t)m * N, A + (size_t)m * K, weight, scale, N, K, stream);
        }
        return;
    }

    size_t o_off, a_off, w_off, s_off;
    id<MTLBuffer> b_out = ctx.get_buffer(out, o_off);
    id<MTLBuffer> b_a = ctx.get_buffer(A, a_off);
    id<MTLBuffer> b_w = ctx.get_buffer(weight, w_off);
    id<MTLBuffer> b_s = ctx.get_buffer(scale, s_off);

    if (!b_out || !b_a || !b_w || !b_s) {
        for (int m = 0; m < M; m++) {
            gemv_int4_cuda(out + (size_t)m * N, A + (size_t)m * K, weight, scale, N, K, stream);
        }
        return;
    }

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_out offset:o_off atIndex:0];
    [enc setBuffer:b_a offset:a_off atIndex:1];
    [enc setBuffer:b_w offset:w_off atIndex:2];
    [enc setBuffer:b_s offset:s_off atIndex:3];
    [enc setBytes:&N length:sizeof(N) atIndex:4];
    [enc setBytes:&K length:sizeof(K) atIndex:5];
    [enc setBytes:&M length:sizeof(M) atIndex:6];
    bool is_res = false;
    [enc setBytes:&is_res length:sizeof(is_res) atIndex:7];

    [enc dispatchThreadgroups:MTLSizeMake((N + 31) / 32, (M + 7) / 8, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    s->end_encoder_and_maybe_commit();
}

void gemm_int4_f32_batch_cuda(
    float* out, const __nv_bfloat16* A, const uint8_t* weight,
    const __nv_bfloat16* scale, int N, int K, int M, cudaStream_t stream)
{
    if (M <= 0) return;
    if (M == 1) {
        gemv_int4_f32_cuda(out, A, weight, scale, N, K, stream);
        return;
    }

    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("gemm_int4_f32_batch_kernel");
    if (!pso) {
        for (int m = 0; m < M; m++) {
            gemv_int4_f32_cuda(out + (size_t)m * N, A + (size_t)m * K, weight, scale, N, K, stream);
        }
        return;
    }

    size_t o_off, a_off, w_off, s_off;
    id<MTLBuffer> b_out = ctx.get_buffer(out, o_off);
    id<MTLBuffer> b_a = ctx.get_buffer(A, a_off);
    id<MTLBuffer> b_w = ctx.get_buffer(weight, w_off);
    id<MTLBuffer> b_s = ctx.get_buffer(scale, s_off);

    if (!b_out || !b_a || !b_w || !b_s) {
        for (int m = 0; m < M; m++) {
            gemv_int4_f32_cuda(out + (size_t)m * N, A + (size_t)m * K, weight, scale, N, K, stream);
        }
        return;
    }

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_out offset:o_off atIndex:0];
    [enc setBuffer:b_a offset:a_off atIndex:1];
    [enc setBuffer:b_w offset:w_off atIndex:2];
    [enc setBuffer:b_s offset:s_off atIndex:3];
    [enc setBytes:&N length:sizeof(N) atIndex:4];
    [enc setBytes:&K length:sizeof(K) atIndex:5];
    [enc setBytes:&M length:sizeof(M) atIndex:6];

    [enc dispatchThreadgroups:MTLSizeMake((N + 31) / 32, (M + 7) / 8, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    s->end_encoder_and_maybe_commit();
}

void gemm_int4_swiglu_fused_batch_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* A,
    const uint8_t* gate_weight, const __nv_bfloat16* gate_scale,
    const uint8_t* up_weight, const __nv_bfloat16* up_scale,
    int N, int K, int M, float swiglu_limit, cudaStream_t stream)
{
    if (M <= 0) return;
    if (M == 1) {
        gemv_int4_swiglu_fused_cuda(out, A, gate_weight, gate_scale, up_weight, up_scale, N, K, swiglu_limit, stream);
        return;
    }

    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("gemm_int4_swiglu_fused_batch_kernel");
    if (!pso) {
        for (int m = 0; m < M; m++) {
            gemv_int4_swiglu_fused_cuda(out + (size_t)m * N, A + (size_t)m * K, gate_weight, gate_scale, up_weight, up_scale, N, K, swiglu_limit, stream);
        }
        return;
    }

    size_t o_off, a_off, gw_off, gs_off, uw_off, us_off;
    id<MTLBuffer> b_out = ctx.get_buffer(out, o_off);
    id<MTLBuffer> b_a = ctx.get_buffer(A, a_off);
    id<MTLBuffer> b_gw = ctx.get_buffer(gate_weight, gw_off);
    id<MTLBuffer> b_gs = ctx.get_buffer(gate_scale, gs_off);
    id<MTLBuffer> b_uw = ctx.get_buffer(up_weight, uw_off);
    id<MTLBuffer> b_us = ctx.get_buffer(up_scale, us_off);

    if (!b_out || !b_a || !b_gw || !b_gs || !b_uw || !b_us) {
        for (int m = 0; m < M; m++) {
            gemv_int4_swiglu_fused_cuda(out + (size_t)m * N, A + (size_t)m * K, gate_weight, gate_scale, up_weight, up_scale, N, K, swiglu_limit, stream);
        }
        return;
    }

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_out offset:o_off atIndex:0];
    [enc setBuffer:b_a offset:a_off atIndex:1];
    [enc setBuffer:b_gw offset:gw_off atIndex:2];
    [enc setBuffer:b_gs offset:gs_off atIndex:3];
    [enc setBuffer:b_uw offset:uw_off atIndex:4];
    [enc setBuffer:b_us offset:us_off atIndex:5];
    [enc setBytes:&N length:sizeof(N) atIndex:6];
    [enc setBytes:&K length:sizeof(K) atIndex:7];
    [enc setBytes:&M length:sizeof(M) atIndex:8];
    [enc setBytes:&swiglu_limit length:sizeof(swiglu_limit) atIndex:9];

    [enc dispatchThreadgroups:MTLSizeMake((N + 31) / 32, (M + 7) / 8, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    s->end_encoder_and_maybe_commit();
}

void gemv_bf16_cuda(float* out, const __nv_bfloat16* W, const __nv_bfloat16* x, int N, int K, cudaStream_t stream) {
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("gemv_bf16_f32_kernel");
    if (!pso) {
        metal_stream_synchronize(stream);
        for (int r = 0; r < N; r++) {
            float sum = 0.0f;
            const __nv_bfloat16* row = W + (size_t)r * K;
            for (int c = 0; c < K; c++) {
                sum += row[c].to_float() * x[c].to_float();
            }
            out[r] = sum;
        }
        return;
    }

    size_t o_off, w_off, x_off;
    id<MTLBuffer> b_out = ctx.get_buffer(out, o_off);
    id<MTLBuffer> b_w = ctx.get_buffer(W, w_off);
    id<MTLBuffer> b_x = ctx.get_buffer(x, x_off);
    if (!b_out || !b_w || !b_x) {
        metal_stream_synchronize(stream);
        for (int r = 0; r < N; r++) {
            float sum = 0.0f;
            const __nv_bfloat16* row = W + (size_t)r * K;
            for (int c = 0; c < K; c++) {
                sum += row[c].to_float() * x[c].to_float();
            }
            out[r] = sum;
        }
        return;
    }

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_out offset:o_off atIndex:0];
    [enc setBuffer:b_w offset:w_off atIndex:1];
    [enc setBuffer:b_x offset:x_off atIndex:2];
    [enc setBytes:&N length:sizeof(N) atIndex:3];
    [enc setBytes:&K length:sizeof(K) atIndex:4];

    [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    s->end_encoder_and_maybe_commit();
}

void gemv_bf16_out_bf16_cuda(__nv_bfloat16* out, const __nv_bfloat16* W, const __nv_bfloat16* x, int N, int K, cudaStream_t stream) {
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("gemv_bf16_out_bf16_kernel");
    if (!pso) {
        metal_stream_synchronize(stream);
        for (int r = 0; r < N; r++) {
            float sum = 0.0f;
            const __nv_bfloat16* row = W + (size_t)r * K;
            for (int c = 0; c < K; c++) {
                sum += row[c].to_float() * x[c].to_float();
            }
            out[r] = __nv_bfloat16::from_float(sum);
        }
        return;
    }

    size_t o_off, w_off, x_off;
    id<MTLBuffer> b_out = ctx.get_buffer(out, o_off);
    id<MTLBuffer> b_w = ctx.get_buffer(W, w_off);
    id<MTLBuffer> b_x = ctx.get_buffer(x, x_off);
    if (!b_out || !b_w || !b_x) {
        metal_stream_synchronize(stream);
        for (int r = 0; r < N; r++) {
            float sum = 0.0f;
            const __nv_bfloat16* row = W + (size_t)r * K;
            for (int c = 0; c < K; c++) {
                sum += row[c].to_float() * x[c].to_float();
            }
            out[r] = __nv_bfloat16::from_float(sum);
        }
        return;
    }

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_out offset:o_off atIndex:0];
    [enc setBuffer:b_w offset:w_off atIndex:1];
    [enc setBuffer:b_x offset:x_off atIndex:2];
    [enc setBytes:&N length:sizeof(N) atIndex:3];
    [enc setBytes:&K length:sizeof(K) atIndex:4];

    [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    s->end_encoder_and_maybe_commit();
}

void gemv_bf16_batch_cuda(float* out, const __nv_bfloat16* W, const __nv_bfloat16* X, int N, int K, int M, cudaStream_t stream) {
    for (int m = 0; m < M; m++) {
        gemv_bf16_cuda(out + m * N, W, X + m * K, N, K, stream);
    }
}


void deltanet_in_proj_ab_batch_cuda(
    __nv_bfloat16* out_a, __nv_bfloat16* out_b,
    const __nv_bfloat16* Wa, const __nv_bfloat16* Wb, const __nv_bfloat16* X,
    int N, int K, int M, cudaStream_t stream)
{
    if (M <= 0) return;
    if (M == 1) {
        gemv_bf16_out_bf16_cuda(out_a, Wa, X, N, K, stream);
        gemv_bf16_out_bf16_cuda(out_b, Wb, X, N, K, stream);
        return;
    }

    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("deltanet_in_proj_ab_batch_kernel");
    if (!pso) {
        gemv_bf16_out_bf16_batch_cuda(out_a, Wa, X, N, K, M, stream);
        gemv_bf16_out_bf16_batch_cuda(out_b, Wb, X, N, K, M, stream);
        return;
    }

    size_t oa_off, ob_off, wa_off, wb_off, x_off;
    id<MTLBuffer> b_oa = ctx.get_buffer(out_a, oa_off);
    id<MTLBuffer> b_ob = ctx.get_buffer(out_b, ob_off);
    id<MTLBuffer> b_wa = ctx.get_buffer(Wa, wa_off);
    id<MTLBuffer> b_wb = ctx.get_buffer(Wb, wb_off);
    id<MTLBuffer> b_x  = ctx.get_buffer(X, x_off);

    if (!b_oa || !b_ob || !b_wa || !b_wb || !b_x) {
        gemv_bf16_out_bf16_batch_cuda(out_a, Wa, X, N, K, M, stream);
        gemv_bf16_out_bf16_batch_cuda(out_b, Wb, X, N, K, M, stream);
        return;
    }

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_oa offset:oa_off atIndex:0];
    [enc setBuffer:b_ob offset:ob_off atIndex:1];
    [enc setBuffer:b_wa offset:wa_off atIndex:2];
    [enc setBuffer:b_wb offset:wb_off atIndex:3];
    [enc setBuffer:b_x  offset:x_off  atIndex:4];
    [enc setBytes:&N length:sizeof(N) atIndex:5];
    [enc setBytes:&K length:sizeof(K) atIndex:6];
    [enc setBytes:&M length:sizeof(M) atIndex:7];

    [enc dispatchThreadgroups:MTLSizeMake(N, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    s->end_encoder_and_maybe_commit();
}

void gemv_bf16_out_bf16_batch_cuda(__nv_bfloat16* out, const __nv_bfloat16* W, const __nv_bfloat16* X, int N, int K, int M, cudaStream_t stream) {
    if (M <= 0) return;
    if (M == 1) {
        gemv_bf16_out_bf16_cuda(out, W, X, N, K, stream);
        return;
    }

    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("gemv_bf16_out_bf16_batch_kernel");
    if (!pso) {
        for (int m = 0; m < M; m++) {
            gemv_bf16_out_bf16_cuda(out + (size_t)m * N, W, X + (size_t)m * K, N, K, stream);
        }
        return;
    }

    size_t o_off, w_off, x_off;
    id<MTLBuffer> b_out = ctx.get_buffer(out, o_off);
    id<MTLBuffer> b_w = ctx.get_buffer(W, w_off);
    id<MTLBuffer> b_x = ctx.get_buffer(X, x_off);

    if (!b_out || !b_w || !b_x) {
        for (int m = 0; m < M; m++) {
            gemv_bf16_out_bf16_cuda(out + (size_t)m * N, W, X + (size_t)m * K, N, K, stream);
        }
        return;
    }

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_out offset:o_off atIndex:0];
    [enc setBuffer:b_w offset:w_off atIndex:1];
    [enc setBuffer:b_x offset:x_off atIndex:2];
    [enc setBytes:&N length:sizeof(N) atIndex:3];
    [enc setBytes:&K length:sizeof(K) atIndex:4];
    [enc setBytes:&M length:sizeof(M) atIndex:5];

    [enc dispatchThreadgroups:MTLSizeMake((N + 31) / 32, (M + 7) / 8, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    s->end_encoder_and_maybe_commit();
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
    int num_k_heads, int num_v_heads, int head_dim, cudaStream_t stream);

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
    if (M <= 0) return;
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso_conv = ctx.get_pipeline("deltanet_conv_kernel");
    id<MTLComputePipelineState> pso_ssm = ctx.get_pipeline("deltanet_ssm_step_kernel");
    int channels = (2 * num_k_heads + num_v_heads) * head_dim;
    int z_stride = num_v_heads * head_dim;

    size_t o_off, qkv_off, z_off, a_off, b_off, cw_off, ics_off, ocs_off, al_off, dt_off, nw_off, issm_off, ossm_off;
    id<MTLBuffer> b_out = ctx.get_buffer(out, o_off);
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
    id<MTLBuffer> b_issm = ctx.get_buffer(in_ssm_state, issm_off);
    id<MTLBuffer> b_ossm = ctx.get_buffer(out_ssm_state, ossm_off);

    if (pso_conv && pso_ssm && b_out && b_qkv && b_z && b_a && b_b && b_cw && b_ics && b_ocs && b_al && b_dt && b_nw && b_issm && b_ossm) {
        id<MTLComputeCommandEncoder> enc = s->get_encoder();
        for (int m = 0; m < M; m++) {
            // Conv step m
            [enc setComputePipelineState:pso_conv];
            [enc setBuffer:b_qkv offset:qkv_off + (size_t)m * channels * 2 atIndex:0];
            [enc setBuffer:b_qkv offset:qkv_off + (size_t)m * channels * 2 atIndex:1];
            [enc setBuffer:b_cw offset:cw_off atIndex:2];
            [enc setBuffer:(m == 0 ? b_ics : b_ocs) offset:(m == 0 ? ics_off : ocs_off) atIndex:3];
            [enc setBuffer:b_ocs offset:ocs_off atIndex:4];
            [enc setBytes:&channels length:sizeof(channels) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((channels + 127) / 128, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            // SSM step m
            [enc setComputePipelineState:pso_ssm];
            [enc setBuffer:b_out offset:o_off + (size_t)m * z_stride * 2 atIndex:0];
            [enc setBuffer:b_qkv offset:qkv_off + (size_t)m * channels * 2 atIndex:1];
            [enc setBuffer:b_z offset:z_off + (size_t)m * z_stride * 2 atIndex:2];
            [enc setBuffer:b_a offset:a_off + (size_t)m * num_v_heads * 2 atIndex:3];
            [enc setBuffer:b_b offset:b_off + (size_t)m * num_v_heads * 2 atIndex:4];
            [enc setBuffer:b_al offset:al_off atIndex:5];
            [enc setBuffer:b_dt offset:dt_off atIndex:6];
            [enc setBuffer:b_nw offset:nw_off atIndex:7];
            [enc setBuffer:(m == 0 ? b_issm : b_ossm) offset:(m == 0 ? issm_off : ossm_off) atIndex:8];
            [enc setBuffer:b_ossm offset:ossm_off atIndex:9];
            [enc setBytes:&num_k_heads length:sizeof(num_k_heads) atIndex:10];
            [enc setBytes:&num_v_heads length:sizeof(num_v_heads) atIndex:11];
            [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:12];
            [enc dispatchThreadgroups:MTLSizeMake(num_v_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        s->end_encoder_and_maybe_commit();
        return;
    }

    fprintf(stderr, "[WARN] deltanet_linear_attention_decode_batch_cuda FALLBACK TO CPU! M=%d\n", M);
    metal_stream_synchronize(stream);
    if (!out || !in_qkv || !in_z || !in_a || !in_b || !conv1d_w || !in_conv_state || !in_ssm_state || !norm_w || !A_log || !dt_bias) return;
    // 1. Conv1D 1x4 depthwise causal convolution across M tokens
    std::vector<__nv_bfloat16> conv_out((size_t)M * channels);
    __nv_bfloat16* conv_out_ptr = conv_out.data();

    dispatch_apply(channels, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t c) {
        const __nv_bfloat16* in_cs = in_conv_state + c * 4;
        const __nv_bfloat16* cw = conv1d_w + c * 4;

        float s0 = in_cs[1].to_float();
        float s1 = in_cs[2].to_float();
        float s2 = in_cs[3].to_float();

        float w0 = cw[0].to_float();
        float w1 = cw[1].to_float();
        float w2 = cw[2].to_float();
        float w3 = cw[3].to_float();

        for (int m = 0; m < M; m++) {
            float s3 = in_qkv[(size_t)m * channels + c].to_float();
            float val = s0 * w0 + s1 * w1 + s2 * w2 + s3 * w3;
            float silu_val = val / (1.0f + std::exp(-val));
            conv_out_ptr[(size_t)m * channels + c] = __nv_bfloat16::from_float(silu_val);

            if (m == 0 && slot_conv_0) {
                __nv_bfloat16* slot_cs = slot_conv_0 + c * 4;
                slot_cs[0] = __nv_bfloat16::from_float(s0);
                slot_cs[1] = __nv_bfloat16::from_float(s1);
                slot_cs[2] = __nv_bfloat16::from_float(s2);
                slot_cs[3] = __nv_bfloat16::from_float(s3);
            } else if (m == 1 && slot_conv_1) {
                __nv_bfloat16* slot_cs = slot_conv_1 + c * 4;
                slot_cs[0] = __nv_bfloat16::from_float(s0);
                slot_cs[1] = __nv_bfloat16::from_float(s1);
                slot_cs[2] = __nv_bfloat16::from_float(s2);
                slot_cs[3] = __nv_bfloat16::from_float(s3);
            } else if (m == 2 && slot_conv_2) {
                __nv_bfloat16* slot_cs = slot_conv_2 + c * 4;
                slot_cs[0] = __nv_bfloat16::from_float(s0);
                slot_cs[1] = __nv_bfloat16::from_float(s1);
                slot_cs[2] = __nv_bfloat16::from_float(s2);
                slot_cs[3] = __nv_bfloat16::from_float(s3);
            } else if (m == 3 && slot_conv_3) {
                __nv_bfloat16* slot_cs = slot_conv_3 + c * 4;
                slot_cs[0] = __nv_bfloat16::from_float(s0);
                slot_cs[1] = __nv_bfloat16::from_float(s1);
                slot_cs[2] = __nv_bfloat16::from_float(s2);
                slot_cs[3] = __nv_bfloat16::from_float(s3);
            }

            if (m == M - 1 && out_conv_state) {
                __nv_bfloat16* out_cs = out_conv_state + c * 4;
                out_cs[0] = __nv_bfloat16::from_float(s0);
                out_cs[1] = __nv_bfloat16::from_float(s1);
                out_cs[2] = __nv_bfloat16::from_float(s2);
                out_cs[3] = __nv_bfloat16::from_float(s3);
            }

            s0 = s1;
            s1 = s2;
            s2 = s3;
        }
    });

    // 2. SSM Recurrence per head across M tokens
    int qkv_stride = channels;
    dispatch_apply(num_v_heads, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t h) {
        int k_h = (int)h / (num_v_heads / num_k_heads);
        const __nv_bfloat16* in_state_h = in_ssm_state + h * head_dim * head_dim;
        std::vector<float> s_state(head_dim * head_dim);
        for (int i = 0; i < head_dim * head_dim; i++) {
            s_state[i] = in_state_h[i].to_float();
        }

        float dt_val = dt_bias[h].to_float();
        float a_log_val = A_log[h].to_float();

        int q_offset = k_h * head_dim;
        int k_offset = (num_k_heads * head_dim) + k_h * head_dim;
        int v_offset = (2 * num_k_heads * head_dim) + (int)h * head_dim;

        for (int m = 0; m < M; m++) {
            const __nv_bfloat16* m_conv = conv_out_ptr + (size_t)m * qkv_stride;
            const __nv_bfloat16* m_z = in_z + (size_t)m * z_stride;
            const __nv_bfloat16* m_a = in_a + (size_t)m * num_v_heads;
            const __nv_bfloat16* m_b = in_b + (size_t)m * num_v_heads;

            float a_val = m_a[h].to_float();
            float b_val = m_b[h].to_float();
            float beta = 1.0f / (1.0f + std::exp(-b_val));
            float val_a = a_val + dt_val;
            float softplus_a = (val_a > 20.0f) ? val_a : std::log1p(std::exp(val_a));
            float g = -std::exp(a_log_val) * softplus_a;
            float decay = std::exp(g);

            std::vector<float> s_q(head_dim), s_k(head_dim), s_v(head_dim), s_z(head_dim);
            float q_norm_sq = 0.0f, k_norm_sq = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                float qd = m_conv[q_offset + d].to_float();
                float kd = m_conv[k_offset + d].to_float();
                s_q[d] = qd;
                s_k[d] = kd;
                s_v[d] = m_conv[v_offset + d].to_float();
                s_z[d] = m_z[h * head_dim + d].to_float();
                q_norm_sq += qd * qd;
                k_norm_sq += kd * kd;
            }

            float r_q = (1.0f / std::sqrt(q_norm_sq + 1e-6f)) * (1.0f / std::sqrt((float)head_dim));
            float r_k = 1.0f / std::sqrt(k_norm_sq + 1e-6f);
            for (int d = 0; d < head_dim; d++) {
                s_q[d] *= r_q;
                s_k[d] *= r_k;
            }

            std::vector<float> s_out(head_dim, 0.0f);
            float out_norm_sq = 0.0f;

            for (int c = 0; c < head_dim; c++) {
                float mem = 0.0f;
                for (int r = 0; r < head_dim; r++) {
                    float s_val = s_state[r * head_dim + c];
                    mem += (decay * s_val) * s_k[r];
                }
                float delta_c = (s_v[c] - mem) * beta;
                float out_c = 0.0f;
                for (int r = 0; r < head_dim; r++) {
                    float new_s = decay * s_state[r * head_dim + c] + s_k[r] * delta_c;
                    s_state[r * head_dim + c] = new_s;
                    out_c += new_s * s_q[r];
                }
                s_out[c] = out_c;
                out_norm_sq += out_c * out_c;
            }

            if (m == 0 && slot_ssm_0) {
                __nv_bfloat16* slot_state_h = slot_ssm_0 + h * head_dim * head_dim;
                for (int i = 0; i < head_dim * head_dim; i++) slot_state_h[i] = __nv_bfloat16::from_float(s_state[i]);
            } else if (m == 1 && slot_ssm_1) {
                __nv_bfloat16* slot_state_h = slot_ssm_1 + h * head_dim * head_dim;
                for (int i = 0; i < head_dim * head_dim; i++) slot_state_h[i] = __nv_bfloat16::from_float(s_state[i]);
            } else if (m == 2 && slot_ssm_2) {
                __nv_bfloat16* slot_state_h = slot_ssm_2 + h * head_dim * head_dim;
                for (int i = 0; i < head_dim * head_dim; i++) slot_state_h[i] = __nv_bfloat16::from_float(s_state[i]);
            } else if (m == 3 && slot_ssm_3) {
                __nv_bfloat16* slot_state_h = slot_ssm_3 + h * head_dim * head_dim;
                for (int i = 0; i < head_dim * head_dim; i++) slot_state_h[i] = __nv_bfloat16::from_float(s_state[i]);
            }

            float r_out = 1.0f / std::sqrt(out_norm_sq / (float)head_dim + 1e-6f);
            __nv_bfloat16* m_out = out + (size_t)m * z_stride + h * head_dim;
            for (int d = 0; d < head_dim; d++) {
                float normed = s_out[d] * r_out * norm_w[d].to_float();
                float z = s_z[d];
                float silu_z = z / (1.0f + std::exp(-z));
                m_out[d] = __nv_bfloat16::from_float(normed * silu_z);
            }
        }

        if (out_ssm_state) {
            __nv_bfloat16* out_state_h = out_ssm_state + h * head_dim * head_dim;
            for (int i = 0; i < head_dim * head_dim; i++) {
                out_state_h[i] = __nv_bfloat16::from_float(s_state[i]);
            }
        }
    });
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
    id<MTLComputePipelineState> pso_conv = ctx.get_pipeline("deltanet_conv_kernel");
    id<MTLComputePipelineState> pso_ssm = ctx.get_pipeline("deltanet_ssm_step_kernel");
    int channels = (2 * num_k_heads + num_v_heads) * head_dim;

    if (pso_conv && pso_ssm) {
        size_t o_off, qkv_off, z_off, a_off, b_off, cw_off, ics_off, ocs_off, al_off, dt_off, nw_off, issm_off, ossm_off;
        id<MTLBuffer> b_out = ctx.get_buffer(out, o_off);
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
        id<MTLBuffer> b_issm = ctx.get_buffer(in_ssm_state, issm_off);
        id<MTLBuffer> b_ossm = ctx.get_buffer(out_ssm_state, ossm_off);

        if (b_out && b_qkv && b_z && b_a && b_b && b_cw && b_ics && b_ocs && b_al && b_dt && b_nw && b_issm && b_ossm) {
            // Stage 1: Conv1D causal convolution across all channels (updates out_conv_state and overwrites b_qkv with silu_val in-place)
            id<MTLComputeCommandEncoder> enc1 = s->get_encoder();
            [enc1 setComputePipelineState:pso_conv];
            [enc1 setBuffer:b_qkv offset:qkv_off atIndex:0]; // conv_out
            [enc1 setBuffer:b_qkv offset:qkv_off atIndex:1]; // in_qkv
            [enc1 setBuffer:b_cw offset:cw_off atIndex:2];
            [enc1 setBuffer:b_ics offset:ics_off atIndex:3];
            [enc1 setBuffer:b_ocs offset:ocs_off atIndex:4];
            [enc1 setBytes:&channels length:sizeof(channels) atIndex:5];

            NSUInteger tg = 256;
            NSUInteger groups = (channels + tg - 1) / tg;
            [enc1 dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            s->end_encoder_and_maybe_commit();

            // Stage 2: SSM recurrent step across all num_v_heads
            id<MTLComputeCommandEncoder> enc2 = s->get_encoder();
            [enc2 setComputePipelineState:pso_ssm];
            [enc2 setBuffer:b_out offset:o_off atIndex:0];
            [enc2 setBuffer:b_qkv offset:qkv_off atIndex:1]; // conv_out
            [enc2 setBuffer:b_z offset:z_off atIndex:2];
            [enc2 setBuffer:b_a offset:a_off atIndex:3];
            [enc2 setBuffer:b_b offset:b_off atIndex:4];
            [enc2 setBuffer:b_al offset:al_off atIndex:5];
            [enc2 setBuffer:b_dt offset:dt_off atIndex:6];
            [enc2 setBuffer:b_nw offset:nw_off atIndex:7];
            [enc2 setBuffer:b_issm offset:issm_off atIndex:8];
            [enc2 setBuffer:b_ossm offset:ossm_off atIndex:9];
            [enc2 setBytes:&num_k_heads length:sizeof(num_k_heads) atIndex:10];
            [enc2 setBytes:&num_v_heads length:sizeof(num_v_heads) atIndex:11];
            [enc2 setBytes:&head_dim length:sizeof(head_dim) atIndex:12];

            [enc2 dispatchThreadgroups:MTLSizeMake(num_v_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            s->end_encoder_and_maybe_commit();
            return;
        }
    }

    deltanet_linear_attention_decode_batch_cuda(
        out, in_qkv, in_z, in_a, in_b, conv1d_w,
        in_conv_state, out_conv_state,
        nullptr, nullptr, nullptr, nullptr,
        A_log, dt_bias, norm_w,
        in_ssm_state, out_ssm_state,
        nullptr, nullptr, nullptr, nullptr,
        num_k_heads, num_v_heads, head_dim, 1, stream);
}

void softmax_cuda(float* out, const float* x, int rows, int cols, cudaStream_t stream) {
    (void)stream;
    for (int r = 0; r < rows; r++) {
        const float* in_row = x + r * cols;
        float* out_row = out + r * cols;
        float max_val = in_row[0];
        for (int c = 1; c < cols; c++) if (in_row[c] > max_val) max_val = in_row[c];
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            float e = std::exp(in_row[c] - max_val);
            out_row[c] = e;
            sum += e;
        }
        float inv = 1.0f / (sum + 1e-9f);
        for (int c = 0; c < cols; c++) out_row[c] *= inv;
    }
}

void argmax_f32_cuda(int32_t* out, const float* logits, int n, cudaStream_t stream) {
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("argmax_f32_kernel");
    if (!pso) {
        metal_stream_synchronize(stream);
        int32_t best_idx = 0;
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

    size_t o_off, l_off;
    id<MTLBuffer> b_out = ctx.get_buffer(out, o_off);
    id<MTLBuffer> b_logits = ctx.get_buffer(logits, l_off);

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_out offset:o_off atIndex:0];
    [enc setBuffer:b_logits offset:l_off atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];

    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    s->end_encoder_and_maybe_commit();
}

void argmax_f32_batch_cuda(int32_t* out, const float* logits, int n, int M, cudaStream_t stream) {
    for (int m = 0; m < M; m++) {
        argmax_f32_cuda(out + m, logits + m * n, n, stream);
    }
}

void sample_multinomial_f32_cuda(
    int32_t* out, float* logits, int n, float temperature, float rand_val, float min_p,
    cudaStream_t stream, int top_k, float top_p)
{
    (void)top_k;
    (void)top_p;
    metal_stream_synchronize(stream);

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

    float inv_t = 1.0f / std::max(temperature, 1e-4f);

    float max_l = logits[0];
    int best_idx = 0;
    for (int i = 1; i < n; i++) {
        if (logits[i] > max_l) {
            max_l = logits[i];
            best_idx = i;
        }
    }

    thread_local std::vector<float> exps;
    if ((int)exps.size() < n) exps.resize(n);
    float sum_exp = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = std::exp((logits[i] - max_l) * inv_t);
        if (min_p > 0.0f && e < min_p) {
            e = 0.0f;
        }
        exps[i] = e;
        sum_exp += e;
    }

    if (sum_exp <= 0.0f) {
        *out = best_idx;
        return;
    }

    float target = rand_val * sum_exp;
    float cum = 0.0f;
    int picked = best_idx;
    for (int i = 0; i < n; i++) {
        cum += exps[i];
        if (cum >= target) {
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
    s->end_encoder_and_maybe_commit();
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

    if (new_k && k_cache) {
        std::memcpy(k_cache + size_t(pos) * n_kv_heads * head_dim, new_k, n_kv_heads * head_dim * sizeof(__nv_bfloat16));
    }
    if (new_v && v_cache) {
        std::memcpy(v_cache + size_t(pos) * n_kv_heads * head_dim, new_v, n_kv_heads * head_dim * sizeof(__nv_bfloat16));
    }

    static thread_local std::vector<float> scores_buf;
    if ((int)scores_buf.size() <= pos) scores_buf.resize((pos + 1) * 2);
    float* scores = scores_buf.data();

    std::vector<float> q_f(head_dim);
    std::vector<float> out_f(head_dim);

    for (int qh = 0; qh < n_q_heads; qh++) {
        int kv_h = qh / group_size;
        const __nv_bfloat16* q_head = q + qh * head_dim;

        for (int d = 0; d < head_dim; d++) {
            uint32_t u = ((uint32_t)*(const uint16_t*)&q_head[d]) << 16;
            q_f[d] = *(float*)&u;
        }

        float max_s = -1e38f;
        for (int t = 0; t <= pos; t++) {
            const __nv_bfloat16* k_head = k_cache + ((size_t)t * n_kv_heads + kv_h) * head_dim;
            float dot = 0.0f;
            int d = 0;
#if defined(__ARM_NEON)
            float32x4_t sum0 = vdupq_n_f32(0.0f);
            float32x4_t sum1 = vdupq_n_f32(0.0f);
            for (; d <= head_dim - 8; d += 8) {
                uint16x8_t kv = vld1q_u16((const uint16_t*)&k_head[d]);
                float32x4_t k0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(kv), 16));
                float32x4_t k1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(kv), 16));
                float32x4_t q0 = vld1q_f32(&q_f[d]);
                float32x4_t q1 = vld1q_f32(&q_f[d + 4]);
                sum0 = vmlaq_f32(sum0, k0, q0);
                sum1 = vmlaq_f32(sum1, k1, q1);
            }
            dot = vaddvq_f32(vaddq_f32(sum0, sum1));
#endif
            for (; d < head_dim; d++) {
                uint32_t u = ((uint32_t)*(const uint16_t*)&k_head[d]) << 16;
                dot += q_f[d] * (*(float*)&u);
            }
            dot *= scale;
            scores[t] = dot;
            if (dot > max_s) max_s = dot;
        }

        float sum_exp = 0.0f;
        for (int t = 0; t <= pos; t++) {
            float exp_val = expf(scores[t] - max_s);
            scores[t] = exp_val;
            sum_exp += exp_val;
        }
        float inv_sum = 1.0f / (sum_exp + 1e-9f);

        std::fill(out_f.begin(), out_f.end(), 0.0f);
        for (int t = 0; t <= pos; t++) {
            float w = scores[t] * inv_sum;
            const __nv_bfloat16* v_head = v_cache + ((size_t)t * n_kv_heads + kv_h) * head_dim;
            int d = 0;
#if defined(__ARM_NEON)
            float32x4_t wv = vdupq_n_f32(w);
            for (; d <= head_dim - 8; d += 8) {
                uint16x8_t vv = vld1q_u16((const uint16_t*)&v_head[d]);
                float32x4_t v0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vv), 16));
                float32x4_t v1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vv), 16));
                float32x4_t o0 = vld1q_f32(&out_f[d]);
                float32x4_t o1 = vld1q_f32(&out_f[d + 4]);
                vst1q_f32(&out_f[d], vmlaq_f32(o0, v0, wv));
                vst1q_f32(&out_f[d + 4], vmlaq_f32(o1, v1, wv));
            }
#endif
            for (; d < head_dim; d++) {
                uint32_t u = ((uint32_t)*(const uint16_t*)&v_head[d]) << 16;
                out_f[d] += w * (*(float*)&u);
            }
        }

        __nv_bfloat16* out_head = out + qh * head_dim;
        for (int d = 0; d < head_dim; d++) {
            out_head[d] = __nv_bfloat16::from_float(out_f[d]);
        }
    }
}



static inline float fp8_e4m3_to_float_host(uint8_t val) {
    if (val == 0) return 0.0f;
    uint32_t sign = (uint32_t)(val & 0x80) << 24;
    uint32_t body = ((uint32_t)(val & 0x7F) << 20) + 0x3C000000U;
    float f;
    uint32_t u = sign | body;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

static inline uint8_t float_to_fp8_e4m3_host(float val) {
    uint32_t u;
    std::memcpy(&u, &val, sizeof(u));
    uint32_t sign = (u >> 24) & 0x80;
    u &= 0x7FFFFFFF;
    if (u == 0) return (uint8_t)sign;

    u += (1U << 19);
    int exp = ((u >> 23) & 0xFF) - 127 + 7;
    uint32_t mant = (u >> 20) & 0x7;

    if (exp >= 15) {
        return (uint8_t)(sign | 0x7E);
    } else if (exp <= 0) {
        if (exp < -3) return (uint8_t)sign;
        mant = ((u & 0x007FFFFF) | 0x00800000) >> (21 - exp);
        return (uint8_t)(sign | mant);
    }
    return (uint8_t)(sign | (exp << 3) | mant);
}

template <typename TCache>
static void qwen_gqa_decode_gated_generic(
    __nv_bfloat16* out, const __nv_bfloat16* q_and_gate, __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* q_norm_w, const __nv_bfloat16* k_norm_w,
    TCache* k_cache, TCache* v_cache,
    int n_q_heads, int n_kv_heads, int head_dim, const int32_t* d_pos,
    int pos_scalar, int M, int max_seq_len, float rope_theta, float eps, cudaStream_t stream)
{
    metal_stream_synchronize(stream);
    int group_size = n_q_heads / n_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    int rotary_dim = 64; // Qwen 3.8 partial RoPE
    int half_rotary = rotary_dim / 2; // 32

    // 1. Process K and V for all M tokens and store into cache
    for (int m = 0; m < M; m++) {
        int pos = d_pos ? d_pos[m] : (pos_scalar + m);
        if (pos >= max_seq_len) pos = max_seq_len - 1;

        if (k && v && k_cache && v_cache) {
            for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) {
                const __nv_bfloat16* k_in = k + (size_t)m * n_kv_heads * head_dim + kv_h * head_dim;
                const __nv_bfloat16* v_in = v + (size_t)m * n_kv_heads * head_dim + kv_h * head_dim;
                std::vector<float> k_vec(head_dim);

                if (k_norm_w) {
                    float sum_sq = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        float val = k_in[d].to_float();
                        k_vec[d] = val;
                        sum_sq += val * val;
                    }
                    float rrms = 1.0f / sqrtf(sum_sq / (float)head_dim + eps);
                    for (int d = 0; d < head_dim; d++) {
                        k_vec[d] = k_vec[d] * rrms * (1.0f + k_norm_w[d].to_float());
                    }
                } else {
                    for (int d = 0; d < head_dim; d++) {
                        k_vec[d] = k_in[d].to_float();
                    }
                }

                for (int i = 0; i < half_rotary; i++) {
                    float freq = 1.0f / powf(rope_theta, (float)(2 * i) / (float)rotary_dim);
                    float angle = (float)pos * freq;
                    float cos_a = cosf(angle);
                    float sin_a = sinf(angle);
                    float k0 = k_vec[i];
                    float k1 = k_vec[i + half_rotary];
                    k_vec[i] = k0 * cos_a - k1 * sin_a;
                    k_vec[i + half_rotary] = k0 * sin_a + k1 * cos_a;
                }

                size_t cache_off = ((size_t)pos * n_kv_heads + kv_h) * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    if constexpr (std::is_same_v<TCache, uint8_t>) {
                        k_cache[cache_off + d] = float_to_fp8_e4m3_host(k_vec[d]);
                        v_cache[cache_off + d] = float_to_fp8_e4m3_host(v_in[d].to_float());
                    } else {
                        k_cache[cache_off + d] = __nv_bfloat16::from_float(k_vec[d]);
                        v_cache[cache_off + d] = v_in[d];
                    }
                }
            }
        }
    }

    // 2. Process Q and compute causal attention with Sigmoid Gate for all M tokens
    for (int m = 0; m < M; m++) {
        int pos = d_pos ? d_pos[m] : (pos_scalar + m);
        if (pos >= max_seq_len) pos = max_seq_len - 1;

        dispatch_apply(n_q_heads, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t qh) {
            int kv_h = (int)qh / group_size;
            size_t q_offset = (size_t)m * (2 * n_q_heads * head_dim) + (size_t)qh * (2 * head_dim);
            const __nv_bfloat16* q_in = q_and_gate + q_offset;
            const __nv_bfloat16* gate_in = q_in + head_dim;
            std::vector<float> q_vec(head_dim);

            if (q_norm_w) {
                float sum_sq = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    float val = q_in[d].to_float();
                    q_vec[d] = val;
                    sum_sq += val * val;
                }
                float rrms = 1.0f / sqrtf(sum_sq / (float)head_dim + eps);
                for (int d = 0; d < head_dim; d++) {
                    q_vec[d] = q_vec[d] * rrms * (1.0f + q_norm_w[d].to_float());
                }
            } else {
                for (int d = 0; d < head_dim; d++) {
                    q_vec[d] = q_in[d].to_float();
                }
            }

            for (int i = 0; i < half_rotary; i++) {
                float freq = 1.0f / powf(rope_theta, (float)(2 * i) / (float)rotary_dim);
                float angle = (float)pos * freq;
                float cos_a = cosf(angle);
                float sin_a = sinf(angle);
                float q0 = q_vec[i];
                float q1 = q_vec[i + half_rotary];
                q_vec[i] = q0 * cos_a - q1 * sin_a;
                q_vec[i + half_rotary] = q0 * sin_a + q1 * cos_a;
            }

            std::vector<float> scores(pos + 1);
            float max_s = -1e30f;
            for (int t = 0; t <= pos; t++) {
                const TCache* k_head = k_cache + ((size_t)t * n_kv_heads + kv_h) * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    float k_val;
                    if constexpr (std::is_same_v<TCache, uint8_t>) {
                        k_val = fp8_e4m3_to_float_host(k_head[d]);
                    } else {
                        k_val = k_head[d].to_float();
                    }
                    dot += q_vec[d] * k_val;
                }
                dot *= scale;
                scores[t] = dot;
                if (dot > max_s) max_s = dot;
            }

            float sum_exp = 0.0f;
            for (int t = 0; t <= pos; t++) {
                float e = expf(scores[t] - max_s);
                scores[t] = e;
                sum_exp += e;
            }
            float inv_sum = 1.0f / (sum_exp + 1e-9f);

            std::vector<float> out_f(head_dim, 0.0f);
            for (int t = 0; t <= pos; t++) {
                float w = scores[t] * inv_sum;
                const TCache* v_head = v_cache + ((size_t)t * n_kv_heads + kv_h) * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    float v_val;
                    if constexpr (std::is_same_v<TCache, uint8_t>) {
                        v_val = fp8_e4m3_to_float_host(v_head[d]);
                    } else {
                        v_val = v_head[d].to_float();
                    }
                    out_f[d] += w * v_val;
                }
            }

            __nv_bfloat16* out_head = out + (size_t)m * n_q_heads * head_dim + qh * head_dim;
            for (int d = 0; d < head_dim; d++) {
                float g = gate_in[d].to_float();
                float sig = 1.0f / (1.0f + expf(-g));
                out_head[d] = __nv_bfloat16::from_float(out_f[d] * sig);
            }
        });
    }
}

void qwen_gqa_decode_gated_fp8_batch_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* q_and_gate, __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* q_norm_w, const __nv_bfloat16* k_norm_w,
    uint8_t* k_cache, uint8_t* v_cache,
    int n_q_heads, int n_kv_heads, int head_dim, const int32_t* d_pos,
    int pos_scalar, int M, int max_seq_len, float rope_theta, float eps, cudaStream_t stream)
{
    if (M <= 0) return;
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso_kv = ctx.get_pipeline("qwen_gqa_write_kv_fp8_batch_kernel");
    id<MTLComputePipelineState> pso_q = ctx.get_pipeline("qwen_gqa_compute_attn_fp8_batch_kernel");

    if (pso_kv && pso_q) {
        size_t o_off, qg_off, k_off, v_off, qn_off = 0, kn_off = 0, kc_off, vc_off, dp_off = 0;
        id<MTLBuffer> b_out = ctx.get_buffer(out, o_off);
        id<MTLBuffer> b_qg = ctx.get_buffer(q_and_gate, qg_off);
        id<MTLBuffer> b_k = ctx.get_buffer(k, k_off);
        id<MTLBuffer> b_v = ctx.get_buffer(v, v_off);
        id<MTLBuffer> b_qn = q_norm_w ? ctx.get_buffer(q_norm_w, qn_off) : nil;
        id<MTLBuffer> b_kn = k_norm_w ? ctx.get_buffer(k_norm_w, kn_off) : nil;
        id<MTLBuffer> b_kc = ctx.get_buffer(k_cache, kc_off);
        id<MTLBuffer> b_vc = ctx.get_buffer(v_cache, vc_off);
        id<MTLBuffer> b_dp = d_pos ? ctx.get_buffer(d_pos, dp_off) : nil;

        if (b_out && b_qg && b_kc && b_vc) {
            int has_kn = (b_kn != nil) ? 1 : 0;
            int has_qn = (b_qn != nil) ? 1 : 0;
            int has_dp = (b_dp != nil) ? 1 : 0;

            // Stage 1: Write KV to FP8 cache in parallel
            if (b_k && b_v) {
                id<MTLComputeCommandEncoder> enc1 = s->get_encoder();
                [enc1 setComputePipelineState:pso_kv];
                [enc1 setBuffer:b_k offset:k_off atIndex:0];
                [enc1 setBuffer:b_v offset:v_off atIndex:1];
                [enc1 setBuffer:(b_kn ? b_kn : b_k) offset:(b_kn ? kn_off : k_off) atIndex:2];
                [enc1 setBuffer:b_kc offset:kc_off atIndex:3];
                [enc1 setBuffer:b_vc offset:vc_off atIndex:4];
                [enc1 setBytes:&n_kv_heads length:sizeof(n_kv_heads) atIndex:5];
                [enc1 setBytes:&head_dim length:sizeof(head_dim) atIndex:6];
                [enc1 setBuffer:(b_dp ? b_dp : b_k) offset:(b_dp ? dp_off : k_off) atIndex:7];
                [enc1 setBytes:&pos_scalar length:sizeof(pos_scalar) atIndex:8];
                [enc1 setBytes:&M length:sizeof(M) atIndex:9];
                [enc1 setBytes:&max_seq_len length:sizeof(max_seq_len) atIndex:10];
                [enc1 setBytes:&rope_theta length:sizeof(rope_theta) atIndex:11];
                [enc1 setBytes:&eps length:sizeof(eps) atIndex:12];
                [enc1 setBytes:&has_kn length:sizeof(has_kn) atIndex:13];
                [enc1 setBytes:&has_dp length:sizeof(has_dp) atIndex:14];

                [enc1 dispatchThreadgroups:MTLSizeMake(n_kv_heads, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                s->end_encoder_and_maybe_commit();
            }

            // Stage 2: Compute Query Attention against FP8 cache
            id<MTLComputeCommandEncoder> enc2 = s->get_encoder();
            [enc2 setComputePipelineState:pso_q];
            [enc2 setBuffer:b_out offset:o_off atIndex:0];
            [enc2 setBuffer:b_qg offset:qg_off atIndex:1];
            [enc2 setBuffer:(b_qn ? b_qn : b_out) offset:(b_qn ? qn_off : o_off) atIndex:2];
            [enc2 setBuffer:b_kc offset:kc_off atIndex:3];
            [enc2 setBuffer:b_vc offset:vc_off atIndex:4];
            [enc2 setBytes:&n_q_heads length:sizeof(n_q_heads) atIndex:5];
            [enc2 setBytes:&n_kv_heads length:sizeof(n_kv_heads) atIndex:6];
            [enc2 setBytes:&head_dim length:sizeof(head_dim) atIndex:7];
            [enc2 setBuffer:(b_dp ? b_dp : b_out) offset:(b_dp ? dp_off : o_off) atIndex:8];
            [enc2 setBytes:&pos_scalar length:sizeof(pos_scalar) atIndex:9];
            [enc2 setBytes:&M length:sizeof(M) atIndex:10];
            [enc2 setBytes:&max_seq_len length:sizeof(max_seq_len) atIndex:11];
            [enc2 setBytes:&rope_theta length:sizeof(rope_theta) atIndex:12];
            [enc2 setBytes:&eps length:sizeof(eps) atIndex:13];
            [enc2 setBytes:&has_qn length:sizeof(has_qn) atIndex:14];
            [enc2 setBytes:&has_dp length:sizeof(has_dp) atIndex:15];

            [enc2 dispatchThreadgroups:MTLSizeMake(n_q_heads, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            s->end_encoder_and_maybe_commit();
            return;
        }
    }

    qwen_gqa_decode_gated_generic<uint8_t>(
        out, q_and_gate, k, v, q_norm_w, k_norm_w, k_cache, v_cache,
        n_q_heads, n_kv_heads, head_dim, d_pos, pos_scalar, M, max_seq_len, rope_theta, eps, stream);
}

void qwen_gqa_decode_gated_fp8_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* q_and_gate, __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* q_norm_w, const __nv_bfloat16* k_norm_w,
    uint8_t* k_cache, uint8_t* v_cache,
    int n_q_heads, int n_kv_heads, int head_dim, const int32_t* d_pos,
    int pos_scalar, int max_seq_len, float rope_theta, float eps, cudaStream_t stream)
{
    qwen_gqa_decode_gated_fp8_batch_cuda(
        out, q_and_gate, k, v, q_norm_w, k_norm_w, k_cache, v_cache,
        n_q_heads, n_kv_heads, head_dim, d_pos, pos_scalar, 1, max_seq_len, rope_theta, eps, stream);
}

void qwen_gqa_decode_gated_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* q_and_gate, __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* q_norm_w, const __nv_bfloat16* k_norm_w,
    __nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
    int n_q_heads, int n_kv_heads, int head_dim, const int32_t* d_pos,
    int pos_scalar, int max_seq_len, float rope_theta, float eps, cudaStream_t stream)
{
    qwen_gqa_decode_gated_generic<__nv_bfloat16>(
        out, q_and_gate, k, v, q_norm_w, k_norm_w, k_cache, v_cache,
        n_q_heads, n_kv_heads, head_dim, d_pos, pos_scalar, 1, max_seq_len, rope_theta, eps, stream);
}



void qwen2_gqa_decode_fp8_batch_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* q, __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* q_norm_w, const __nv_bfloat16* k_norm_w,
    uint8_t* k_cache, uint8_t* v_cache,
    int n_q_heads, int n_kv_heads, int head_dim, const int32_t* d_pos,
    int pos_scalar, int M, int max_seq_len, float rope_theta, float eps, const int32_t* d_mrope_pos,
    cudaStream_t stream)
{
    metal_stream_synchronize(stream);
    int group_size = n_q_heads / n_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    int v_start = -1;
    int v_num = 0;
    if (d_mrope_pos) {
        v_start = d_mrope_pos[0];
        v_num = d_mrope_pos[1];
    }
    int rotary_dim = head_dim;
    int half_rotary = rotary_dim / 2;

    // 1. Process K and V for all M tokens and store into FP8 cache
    for (int m = 0; m < M; m++) {
        int pos = d_pos ? d_pos[m] : (pos_scalar + m);
        if (pos >= max_seq_len) pos = max_seq_len - 1;

        if (k && v && k_cache && v_cache) {
            for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) {
                const __nv_bfloat16* k_in = k + (size_t)m * n_kv_heads * head_dim + kv_h * head_dim;
                const __nv_bfloat16* v_in = v + (size_t)m * n_kv_heads * head_dim + kv_h * head_dim;
                std::vector<float> k_vec(head_dim);

                if (k_norm_w) {
                    float sum_sq = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        float val = k_in[d].to_float();
                        k_vec[d] = val;
                        sum_sq += val * val;
                    }
                    float rrms = 1.0f / sqrtf(sum_sq / (float)head_dim + eps);
                    for (int d = 0; d < head_dim; d++) {
                        k_vec[d] = k_vec[d] * rrms * (1.0f + k_norm_w[d].to_float());
                    }
                } else {
                    for (int d = 0; d < head_dim; d++) {
                        k_vec[d] = k_in[d].to_float();
                    }
                }

                for (int i = 0; i < half_rotary; i++) {
                    int eff_pos = pos;
                    if (v_num > 0 && v_start >= 0) {
                        if (pos >= v_start && pos < v_start + v_num) {
                            int v_idx = pos - v_start;
                            int grid_size = (v_num == 576) ? 24 : (int)roundf(sqrtf((float)v_num));
                            int r = v_idx / grid_size;
                            int c = v_idx % grid_size;
                            if (i < 16) eff_pos = v_start;
                            else if (i < 40) eff_pos = v_start + r;
                            else eff_pos = v_start + c;
                        } else if (pos >= v_start + v_num) {
                            int grid_size = (v_num == 576) ? 24 : (int)roundf(sqrtf((float)v_num));
                            int delta = v_num - grid_size;
                            eff_pos = pos - delta;
                        }
                    }
                    float freq = 1.0f / powf(rope_theta, (float)(2 * i) / (float)rotary_dim);
                    float angle = (float)eff_pos * freq;
                    float cos_a = cosf(angle);
                    float sin_a = sinf(angle);
                    float k0 = k_vec[i];
                    float k1 = k_vec[i + half_rotary];
                    k_vec[i] = k0 * cos_a - k1 * sin_a;
                    k_vec[i + half_rotary] = k0 * sin_a + k1 * cos_a;
                }

                size_t cache_off = ((size_t)pos * n_kv_heads + kv_h) * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    k_cache[cache_off + d] = float_to_fp8_e4m3_host(k_vec[d]);
                    v_cache[cache_off + d] = float_to_fp8_e4m3_host(v_in[d].to_float());
                }
            }
        }
    }

    // 2. Process Q and compute Attention for all M tokens
    for (int m = 0; m < M; m++) {
        int pos = d_pos ? d_pos[m] : (pos_scalar + m);
        if (pos >= max_seq_len) pos = max_seq_len - 1;

        dispatch_apply(n_q_heads, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t qh) {
            int kv_h = (int)qh / group_size;
            const __nv_bfloat16* q_in = q + (size_t)m * n_q_heads * head_dim + qh * head_dim;
            std::vector<float> q_vec(head_dim);

            if (q_norm_w) {
                float sum_sq = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    float val = q_in[d].to_float();
                    q_vec[d] = val;
                    sum_sq += val * val;
                }
                float rrms = 1.0f / sqrtf(sum_sq / (float)head_dim + eps);
                for (int d = 0; d < head_dim; d++) {
                    q_vec[d] = q_vec[d] * rrms * (1.0f + q_norm_w[d].to_float());
                }
            } else {
                for (int d = 0; d < head_dim; d++) {
                    q_vec[d] = q_in[d].to_float();
                }
            }

            for (int i = 0; i < half_rotary; i++) {
                int eff_pos = pos;
                if (v_num > 0 && v_start >= 0) {
                    if (pos >= v_start && pos < v_start + v_num) {
                        int v_idx = pos - v_start;
                        int grid_size = (v_num == 576) ? 24 : (int)roundf(sqrtf((float)v_num));
                        int r = v_idx / grid_size;
                        int c = v_idx % grid_size;
                        if (i < 16) eff_pos = v_start;
                        else if (i < 40) eff_pos = v_start + r;
                        else eff_pos = v_start + c;
                    } else if (pos >= v_start + v_num) {
                        int grid_size = (v_num == 576) ? 24 : (int)roundf(sqrtf((float)v_num));
                        int delta = v_num - grid_size;
                        eff_pos = pos - delta;
                    }
                }
                float freq = 1.0f / powf(rope_theta, (float)(2 * i) / (float)rotary_dim);
                float angle = (float)eff_pos * freq;
                float cos_a = cosf(angle);
                float sin_a = sinf(angle);
                float q0 = q_vec[i];
                float q1 = q_vec[i + half_rotary];
                q_vec[i] = q0 * cos_a - q1 * sin_a;
                q_vec[i + half_rotary] = q0 * sin_a + q1 * cos_a;
            }

            std::vector<float> scores(pos + 1);
            float max_s = -1e30f;
            for (int t = 0; t <= pos; t++) {
                const uint8_t* k_head = k_cache + ((size_t)t * n_kv_heads + kv_h) * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot += q_vec[d] * fp8_e4m3_to_float_host(k_head[d]);
                }
                dot *= scale;
                scores[t] = dot;
                if (dot > max_s) max_s = dot;
            }

            float sum_exp = 0.0f;
            for (int t = 0; t <= pos; t++) {
                float e = expf(scores[t] - max_s);
                scores[t] = e;
                sum_exp += e;
            }
            float inv_sum = 1.0f / (sum_exp + 1e-9f);

            std::vector<float> out_f(head_dim, 0.0f);
            for (int t = 0; t <= pos; t++) {
                float w = scores[t] * inv_sum;
                const uint8_t* v_head = v_cache + ((size_t)t * n_kv_heads + kv_h) * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    out_f[d] += w * fp8_e4m3_to_float_host(v_head[d]);
                }
            }

            __nv_bfloat16* out_head = out + (size_t)m * n_q_heads * head_dim + qh * head_dim;
            for (int d = 0; d < head_dim; d++) {
                out_head[d] = __nv_bfloat16::from_float(out_f[d]);
            }
        });
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
    qwen2_gqa_decode_fp8_batch_cuda(
        out, q, k, v, q_norm_w, k_norm_w,
        k_cache, v_cache, n_q_heads, n_kv_heads, head_dim,
        d_pos, pos_scalar, 1, max_seq_len, rope_theta, eps, d_mrope_pos, stream);
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
    if (seq_len <= 0) return;
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("embedding_int4_kernel");
    if (!pso) {
        int num_blocks = dim / 32;
        for (int s_idx = 0; s_idx < seq_len; s_idx++) {
            int token = ids[s_idx];
            const uint8_t* row_w = weight + (size_t)token * (dim / 2);
            const __nv_bfloat16* row_s = scale + (size_t)token * num_blocks;
            __nv_bfloat16* row_out = out + (size_t)s_idx * dim;
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
        return;
    }

    size_t o_off, w_off, s_off, i_off;
    id<MTLBuffer> b_out = ctx.get_buffer(out, o_off);
    id<MTLBuffer> b_w = ctx.get_buffer(weight, w_off);
    id<MTLBuffer> b_s = ctx.get_buffer(scale, s_off);
    id<MTLBuffer> b_ids = ctx.get_buffer(ids, i_off);

    if (!b_out || !b_w || !b_s || !b_ids) {
        int num_blocks = dim / 32;
        for (int s_idx = 0; s_idx < seq_len; s_idx++) {
            int token = ids[s_idx];
            const uint8_t* row_w = weight + (size_t)token * (dim / 2);
            const __nv_bfloat16* row_s = scale + (size_t)token * num_blocks;
            __nv_bfloat16* row_out = out + (size_t)s_idx * dim;
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
        return;
    }

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_out offset:o_off atIndex:0];
    [enc setBuffer:b_w offset:w_off atIndex:1];
    [enc setBuffer:b_s offset:s_off atIndex:2];
    [enc setBuffer:b_ids offset:i_off atIndex:3];
    [enc setBytes:&seq_len length:sizeof(seq_len) atIndex:4];
    [enc setBytes:&dim length:sizeof(dim) atIndex:5];

    NSUInteger tg = std::min(160, 256);
    [enc dispatchThreadgroups:MTLSizeMake(seq_len, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    s->end_encoder_and_maybe_commit();
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
void dequant_int3_block_cuda(
    __nv_bfloat16* out, const uint8_t* weight, const __nv_bfloat16* scale,
    int N, int K, int block_size, cudaStream_t stream)
{
    (void)block_size; (void)stream;
    int blocks_per_row = K / 32;
    dispatch_apply(N, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t r) {
        const uint8_t* row_w = weight + (size_t)r * ((size_t)K * 3 / 8);
        const __nv_bfloat16* row_s = scale + r * blocks_per_row;
        __nv_bfloat16* row_out = out + r * K;
        for (int b = 0; b < blocks_per_row; b++) {
            float s = row_s[b].to_float();
            const uint8_t* blk_w = row_w + b * 12;
            int out_idx = b * 32;
            for (int i = 0; i < 4; i++) {
                uint8_t b0 = blk_w[i * 3 + 0];
                uint8_t b1 = blk_w[i * 3 + 1];
                uint8_t b2 = blk_w[i * 3 + 2];

                float w0 = ((float)(b0 & 0x07) - 4.0f) * s;
                float w1 = ((float)((b0 >> 3) & 0x07) - 4.0f) * s;
                float w2 = ((float)((b0 >> 6) | ((b1 & 0x01) << 2)) - 4.0f) * s;
                float w3 = ((float)((b1 >> 1) & 0x07) - 4.0f) * s;
                float w4 = ((float)((b1 >> 4) & 0x07) - 4.0f) * s;
                float w5 = ((float)((b1 >> 7) | ((b2 & 0x03) << 1)) - 4.0f) * s;
                float w6 = ((float)((b2 >> 2) & 0x07) - 4.0f) * s;
                float w7 = ((float)((b2 >> 5) & 0x07) - 4.0f) * s;

                row_out[out_idx + i * 8 + 0] = __nv_bfloat16::from_float(w0);
                row_out[out_idx + i * 8 + 1] = __nv_bfloat16::from_float(w1);
                row_out[out_idx + i * 8 + 2] = __nv_bfloat16::from_float(w2);
                row_out[out_idx + i * 8 + 3] = __nv_bfloat16::from_float(w3);
                row_out[out_idx + i * 8 + 4] = __nv_bfloat16::from_float(w4);
                row_out[out_idx + i * 8 + 5] = __nv_bfloat16::from_float(w5);
                row_out[out_idx + i * 8 + 6] = __nv_bfloat16::from_float(w6);
                row_out[out_idx + i * 8 + 7] = __nv_bfloat16::from_float(w7);
            }
        }
    });
}

void gemv_int3_cuda(__nv_bfloat16* out, const __nv_bfloat16* vec, const uint8_t* weight, const __nv_bfloat16* scale, int N, int K, cudaStream_t stream) {
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("gemv_int3_kernel");
    size_t o_off, v_off, w_off, s_off;
    id<MTLBuffer> b_out = ctx.get_buffer(out, o_off);
    id<MTLBuffer> b_vec = ctx.get_buffer(vec, v_off);
    id<MTLBuffer> b_w = ctx.get_buffer(weight, w_off);
    id<MTLBuffer> b_s = ctx.get_buffer(scale, s_off);

    if (!pso || !b_out || !b_vec || !b_w || !b_s) {
        int blocks_per_row = K / 32;
        dispatch_apply(N, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t r) {
            const uint8_t* row_w = weight + (size_t)r * ((size_t)K * 3 / 8);
            const __nv_bfloat16* row_s = scale + r * blocks_per_row;
            float sum = 0.0f;
            for (int b = 0; b < blocks_per_row; b++) {
                float s_val = row_s[b].to_float();
                const uint8_t* blk_w = row_w + b * 12;
                int in_idx = b * 32;
                for (int i = 0; i < 4; i++) {
                    uint8_t b0 = blk_w[i * 3 + 0];
                    uint8_t b1 = blk_w[i * 3 + 1];
                    uint8_t b2 = blk_w[i * 3 + 2];

                    float w0 = ((float)(b0 & 0x07) - 4.0f) * s_val;
                    float w1 = ((float)((b0 >> 3) & 0x07) - 4.0f) * s_val;
                    float w2 = ((float)((b0 >> 6) | ((b1 & 0x01) << 2)) - 4.0f) * s_val;
                    float w3 = ((float)((b1 >> 1) & 0x07) - 4.0f) * s_val;
                    float w4 = ((float)((b1 >> 4) & 0x07) - 4.0f) * s_val;
                    float w5 = ((float)((b1 >> 7) | ((b2 & 0x03) << 1)) - 4.0f) * s_val;
                    float w6 = ((float)((b2 >> 2) & 0x07) - 4.0f) * s_val;
                    float w7 = ((float)((b2 >> 5) & 0x07) - 4.0f) * s_val;

                    sum += w0 * vec[in_idx + i * 8 + 0].to_float();
                    sum += w1 * vec[in_idx + i * 8 + 1].to_float();
                    sum += w2 * vec[in_idx + i * 8 + 2].to_float();
                    sum += w3 * vec[in_idx + i * 8 + 3].to_float();
                    sum += w4 * vec[in_idx + i * 8 + 4].to_float();
                    sum += w5 * vec[in_idx + i * 8 + 5].to_float();
                    sum += w6 * vec[in_idx + i * 8 + 6].to_float();
                    sum += w7 * vec[in_idx + i * 8 + 7].to_float();
                }
            }
            out[r] = __nv_bfloat16::from_float(sum);
        });
        return;
    }

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
    s->end_encoder_and_maybe_commit();
}

void gemv_int3_residual_cuda(__nv_bfloat16* inout, const __nv_bfloat16* vec, const uint8_t* weight, const __nv_bfloat16* scale, int N, int K, cudaStream_t stream) {
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("gemv_int3_kernel");
    if (!pso) {
        int blocks_per_row = K / 32;
        dispatch_apply(N, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t r) {
            const uint8_t* row_w = weight + (size_t)r * ((size_t)K * 3 / 8);
            const __nv_bfloat16* row_s = scale + r * blocks_per_row;
            float sum = 0.0f;
            for (int b = 0; b < blocks_per_row; b++) {
                float s_val = row_s[b].to_float();
                const uint8_t* blk_w = row_w + b * 12;
                int in_idx = b * 32;
                for (int i = 0; i < 4; i++) {
                    uint8_t b0 = blk_w[i * 3 + 0];
                    uint8_t b1 = blk_w[i * 3 + 1];
                    uint8_t b2 = blk_w[i * 3 + 2];

                    float w0 = ((float)(b0 & 0x07) - 4.0f) * s_val;
                    float w1 = ((float)((b0 >> 3) & 0x07) - 4.0f) * s_val;
                    float w2 = ((float)((b0 >> 6) | ((b1 & 0x01) << 2)) - 4.0f) * s_val;
                    float w3 = ((float)((b1 >> 1) & 0x07) - 4.0f) * s_val;
                    float w4 = ((float)((b1 >> 4) & 0x07) - 4.0f) * s_val;
                    float w5 = ((float)((b1 >> 7) | ((b2 & 0x03) << 1)) - 4.0f) * s_val;
                    float w6 = ((float)((b2 >> 2) & 0x07) - 4.0f) * s_val;
                    float w7 = ((float)((b2 >> 5) & 0x07) - 4.0f) * s_val;

                    sum += w0 * vec[in_idx + i * 8 + 0].to_float();
                    sum += w1 * vec[in_idx + i * 8 + 1].to_float();
                    sum += w2 * vec[in_idx + i * 8 + 2].to_float();
                    sum += w3 * vec[in_idx + i * 8 + 3].to_float();
                    sum += w4 * vec[in_idx + i * 8 + 4].to_float();
                    sum += w5 * vec[in_idx + i * 8 + 5].to_float();
                    sum += w6 * vec[in_idx + i * 8 + 6].to_float();
                    sum += w7 * vec[in_idx + i * 8 + 7].to_float();
                }
            }
            inout[r] = __nv_bfloat16::from_float(inout[r].to_float() + sum);
        });
        return;
    }

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
    s->end_encoder_and_maybe_commit();
}

void gemv_int3_swiglu_fused_cuda(__nv_bfloat16* out, const __nv_bfloat16* vec, const uint8_t* gate_weight, const __nv_bfloat16* gate_scale, const uint8_t* up_weight, const __nv_bfloat16* up_scale, int N, int K, float swiglu_limit, cudaStream_t stream) {
    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("gemv_int3_swiglu_fused_kernel");
    if (!pso) {
        int blocks_per_row = K / 32;
        dispatch_apply(N, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t r) {
            const uint8_t* g_row_w = gate_weight + (size_t)r * ((size_t)K * 3 / 8);
            const __nv_bfloat16* g_row_s = gate_scale + r * blocks_per_row;
            const uint8_t* u_row_w = up_weight + (size_t)r * ((size_t)K * 3 / 8);
            const __nv_bfloat16* u_row_s = up_scale + r * blocks_per_row;

            float sum_g = 0.0f;
            float sum_u = 0.0f;
            for (int b = 0; b < blocks_per_row; b++) {
                float sg = g_row_s[b].to_float();
                float su = u_row_s[b].to_float();
                const uint8_t* gb = g_row_w + b * 12;
                const uint8_t* ub = u_row_w + b * 12;
                int in_idx = b * 32;
                for (int i = 0; i < 4; i++) {
                    uint8_t gb0 = gb[i * 3 + 0], gb1 = gb[i * 3 + 1], gb2 = gb[i * 3 + 2];
                    uint8_t ub0 = ub[i * 3 + 0], ub1 = ub[i * 3 + 1], ub2 = ub[i * 3 + 2];

                    float gw0 = ((float)(gb0 & 0x07) - 4.0f) * sg;
                    float gw1 = ((float)((gb0 >> 3) & 0x07) - 4.0f) * sg;
                    float gw2 = ((float)((gb0 >> 6) | ((gb1 & 0x01) << 2)) - 4.0f) * sg;
                    float gw3 = ((float)((gb1 >> 1) & 0x07) - 4.0f) * sg;
                    float gw4 = ((float)((gb1 >> 4) & 0x07) - 4.0f) * sg;
                    float gw5 = ((float)((gb1 >> 7) | ((gb2 & 0x03) << 1)) - 4.0f) * sg;
                    float gw6 = ((float)((gb2 >> 2) & 0x07) - 4.0f) * sg;
                    float gw7 = ((float)((gb2 >> 5) & 0x07) - 4.0f) * sg;

                    float uw0 = ((float)(ub0 & 0x07) - 4.0f) * su;
                    float uw1 = ((float)((ub0 >> 3) & 0x07) - 4.0f) * su;
                    float uw2 = ((float)((ub0 >> 6) | ((ub1 & 0x01) << 2)) - 4.0f) * su;
                    float uw3 = ((float)((ub1 >> 1) & 0x07) - 4.0f) * su;
                    float uw4 = ((float)((ub1 >> 4) & 0x07) - 4.0f) * su;
                    float uw5 = ((float)((ub1 >> 7) | ((ub2 & 0x03) << 1)) - 4.0f) * su;
                    float uw6 = ((float)((ub2 >> 2) & 0x07) - 4.0f) * su;
                    float uw7 = ((float)((ub2 >> 5) & 0x07) - 4.0f) * su;

                    float v0 = vec[in_idx + i * 8 + 0].to_float();
                    float v1 = vec[in_idx + i * 8 + 1].to_float();
                    float v2 = vec[in_idx + i * 8 + 2].to_float();
                    float v3 = vec[in_idx + i * 8 + 3].to_float();
                    float v4 = vec[in_idx + i * 8 + 4].to_float();
                    float v5 = vec[in_idx + i * 8 + 5].to_float();
                    float v6 = vec[in_idx + i * 8 + 6].to_float();
                    float v7 = vec[in_idx + i * 8 + 7].to_float();

                    sum_g += gw0 * v0 + gw1 * v1 + gw2 * v2 + gw3 * v3 + gw4 * v4 + gw5 * v5 + gw6 * v6 + gw7 * v7;
                    sum_u += uw0 * v0 + uw1 * v1 + uw2 * v2 + uw3 * v3 + uw4 * v4 + uw5 * v5 + uw6 * v6 + uw7 * v7;
                }
            }
            if (swiglu_limit > 0.0f) {
                sum_g = std::min(sum_g, swiglu_limit);
                sum_u = std::clamp(sum_u, -swiglu_limit, swiglu_limit);
            }
            float silu_g = sum_g / (1.0f + std::exp(-sum_g));
            out[r] = __nv_bfloat16::from_float(silu_g * sum_u);
        });
        return;
    }

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
    s->end_encoder_and_maybe_commit();
}

void gemm_int3_batch_cuda(__nv_bfloat16* out, const __nv_bfloat16* A, const uint8_t* weight, const __nv_bfloat16* scale, int N, int K, int M, cudaStream_t stream) {
    if (M <= 0) return;
    if (M == 1) {
        gemv_int3_cuda(out, A, weight, scale, N, K, stream);
        return;
    }

    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);

    // Ultra-fast Hardware MPS Tensor Core path for prefill (M >= 4)
    if (M >= 4) {
        id<MTLComputePipelineState> pso_dequant3 = ctx.get_pipeline("dequant_int3_to_fp16_kernel");
        id<MTLComputePipelineState> pso_b2f = ctx.get_pipeline("bf16_to_fp16_kernel");
        id<MTLComputePipelineState> pso_f2b = ctx.get_pipeline("fp16_to_bf16_kernel");

        size_t o_off, a_off, w_off, s_off;
        id<MTLBuffer> b_out = ctx.get_buffer(out, o_off);
        id<MTLBuffer> b_a = ctx.get_buffer(A, a_off);
        id<MTLBuffer> b_w = ctx.get_buffer(weight, w_off);
        id<MTLBuffer> b_s = ctx.get_buffer(scale, s_off);

        if (pso_dequant3 && pso_b2f && pso_f2b && b_out && b_a && b_w && b_s && N <= 17408 && K <= 17408 && M <= 512) {
            ctx.ensure_mps_scratch();

            // 1. Convert A to FP16 and Dequantize W to FP16 in 1 compute encoder
            id<MTLComputeCommandEncoder> enc = s->get_encoder();
            [enc setComputePipelineState:pso_b2f];
            [enc setBuffer:ctx.mps_scratch_a_fp16 offset:0 atIndex:0];
            [enc setBuffer:b_a offset:a_off atIndex:1];
            int count_a = M * K;
            [enc setBytes:&count_a length:4 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(count_a, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc setComputePipelineState:pso_dequant3];
            [enc setBuffer:ctx.mps_scratch_w_fp16 offset:0 atIndex:0];
            [enc setBuffer:b_w offset:w_off atIndex:1];
            [enc setBuffer:b_s offset:s_off atIndex:2];
            [enc setBytes:&N length:4 atIndex:3];
            [enc setBytes:&K length:4 atIndex:4];
            MTLSize grid = MTLSizeMake(K / 32, N, 1);
            MTLSize tg = MTLSizeMake(32, 4, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            // 2. Hardware MPS GEMM: C = A * W^T
            id<MTLCommandBuffer> cb = s->get_command_buffer();
            MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:K * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor* descW = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:K rowBytes:K * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:N * sizeof(uint16_t) dataType:MPSDataTypeFloat16];

            MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:ctx.mps_scratch_a_fp16 descriptor:descA];
            MPSMatrix* matW = [[MPSMatrix alloc] initWithBuffer:ctx.mps_scratch_w_fp16 descriptor:descW];
            MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:ctx.mps_scratch_c_fp16 descriptor:descC];

            MPSMatrixMultiplication* mul = [[MPSMatrixMultiplication alloc] initWithDevice:ctx.device transposeLeft:false transposeRight:true resultRows:M resultColumns:N interiorColumns:K alpha:1.0f beta:0.0f];
            [mul encodeToCommandBuffer:cb leftMatrix:matA rightMatrix:matW resultMatrix:matC];

            // 3. Convert C back to BF16 (into out)
            id<MTLComputeCommandEncoder> enc2 = s->get_encoder();
            [enc2 setComputePipelineState:pso_f2b];
            [enc2 setBuffer:b_out offset:o_off atIndex:0];
            [enc2 setBuffer:ctx.mps_scratch_c_fp16 offset:0 atIndex:1];
            int count_out = M * N;
            [enc2 setBytes:&count_out length:4 atIndex:2];
            bool is_res = false;
            [enc2 setBytes:&is_res length:sizeof(is_res) atIndex:3];
            [enc2 dispatchThreads:MTLSizeMake(count_out, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            s->end_encoder_and_maybe_commit();
            return;
        }
    }

    fprintf(stderr, "[WARN] gemm_int3_batch_cuda MPS BYPASSED! M=%d\n", M);
    id<MTLComputePipelineState> pso = ctx.get_pipeline("gemm_int3_batch_kernel");
    if (!pso) {
        for (int m = 0; m < M; m++) {
            gemv_int3_cuda(out + (size_t)m * N, A + (size_t)m * K, weight, scale, N, K, stream);
        }
        return;
    }

    size_t o_off, a_off, w_off, s_off;
    id<MTLBuffer> b_out = ctx.get_buffer(out, o_off);
    id<MTLBuffer> b_a = ctx.get_buffer(A, a_off);
    id<MTLBuffer> b_w = ctx.get_buffer(weight, w_off);
    id<MTLBuffer> b_s = ctx.get_buffer(scale, s_off);

    if (!b_out || !b_a || !b_w || !b_s) {
        for (int m = 0; m < M; m++) {
            gemv_int3_cuda(out + (size_t)m * N, A + (size_t)m * K, weight, scale, N, K, stream);
        }
        return;
    }

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_out offset:o_off atIndex:0];
    [enc setBuffer:b_a offset:a_off atIndex:1];
    [enc setBuffer:b_w offset:w_off atIndex:2];
    [enc setBuffer:b_s offset:s_off atIndex:3];
    [enc setBytes:&N length:sizeof(N) atIndex:4];
    [enc setBytes:&K length:sizeof(K) atIndex:5];
    [enc setBytes:&M length:sizeof(M) atIndex:6];
    bool is_res = false;
    [enc setBytes:&is_res length:sizeof(is_res) atIndex:7];

    [enc dispatchThreadgroups:MTLSizeMake((N + 31) / 32, (M + 7) / 8, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    s->end_encoder_and_maybe_commit();
}

void gemm_int3_swiglu_fused_batch_cuda(__nv_bfloat16* out, const __nv_bfloat16* A, const uint8_t* gate_weight, const __nv_bfloat16* gate_scale, const uint8_t* up_weight, const __nv_bfloat16* up_scale, int N, int K, int M, float swiglu_limit, cudaStream_t stream) {
    if (M <= 0) return;
    if (M == 1) {
        gemv_int3_swiglu_fused_cuda(out, A, gate_weight, gate_scale, up_weight, up_scale, N, K, swiglu_limit, stream);
        return;
    }

    auto& ctx = MetalContext::instance();
    MetalStreamObj* s = get_stream(stream);

    // Ultra-fast Hardware MPS Tensor Core path for prefill (M >= 4)
    if (M >= 4) {
        id<MTLComputePipelineState> pso_dequant3 = ctx.get_pipeline("dequant_int3_to_fp16_kernel");
        id<MTLComputePipelineState> pso_b2f = ctx.get_pipeline("bf16_to_fp16_kernel");
        id<MTLComputePipelineState> pso_swiglu = ctx.get_pipeline("swiglu_fp16_to_bf16_kernel");

        size_t o_off, a_off, gw_off, gs_off, uw_off, us_off;
        id<MTLBuffer> b_out = ctx.get_buffer(out, o_off);
        id<MTLBuffer> b_a = ctx.get_buffer(A, a_off);
        id<MTLBuffer> b_gw = ctx.get_buffer(gate_weight, gw_off);
        id<MTLBuffer> b_gs = ctx.get_buffer(gate_scale, gs_off);
        id<MTLBuffer> b_uw = ctx.get_buffer(up_weight, uw_off);
        id<MTLBuffer> b_us = ctx.get_buffer(up_scale, us_off);

        if (pso_dequant3 && pso_b2f && pso_swiglu && b_out && b_a && b_gw && b_gs && b_uw && b_us && N <= 17408 && K <= 17408 && M <= 512) {
            ctx.ensure_mps_scratch();

            // 1. Convert A to FP16, Dequant Gate to FP16, Dequant Up to FP16 in 1 compute encoder
            id<MTLComputeCommandEncoder> enc = s->get_encoder();
            [enc setComputePipelineState:pso_b2f];
            [enc setBuffer:ctx.mps_scratch_a_fp16 offset:0 atIndex:0];
            [enc setBuffer:b_a offset:a_off atIndex:1];
            int count_a = M * K;
            [enc setBytes:&count_a length:4 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(count_a, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc setComputePipelineState:pso_dequant3];
            [enc setBuffer:ctx.mps_scratch_w_fp16 offset:0 atIndex:0];
            [enc setBuffer:b_gw offset:gw_off atIndex:1];
            [enc setBuffer:b_gs offset:gs_off atIndex:2];
            [enc setBytes:&N length:4 atIndex:3];
            [enc setBytes:&K length:4 atIndex:4];
            MTLSize grid = MTLSizeMake(K / 32, N, 1);
            MTLSize tg = MTLSizeMake(32, 4, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];

            [enc setBuffer:ctx.mps_scratch_w_up_fp16 offset:0 atIndex:0];
            [enc setBuffer:b_uw offset:uw_off atIndex:1];
            [enc setBuffer:b_us offset:us_off atIndex:2];
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            // 2. Hardware MPS GEMM: Gate = A * W_gate^T and Up = A * W_up^T
            id<MTLCommandBuffer> cb = s->get_command_buffer();
            MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:K * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor* descW = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:K rowBytes:K * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:N * sizeof(uint16_t) dataType:MPSDataTypeFloat16];

            MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:ctx.mps_scratch_a_fp16 descriptor:descA];
            MPSMatrix* matGw = [[MPSMatrix alloc] initWithBuffer:ctx.mps_scratch_w_fp16 descriptor:descW];
            MPSMatrix* matUw = [[MPSMatrix alloc] initWithBuffer:ctx.mps_scratch_w_up_fp16 descriptor:descW];
            MPSMatrix* matGateOut = [[MPSMatrix alloc] initWithBuffer:ctx.mps_scratch_gate_fp16 descriptor:descC];
            MPSMatrix* matUpOut = [[MPSMatrix alloc] initWithBuffer:ctx.mps_scratch_c_fp16 descriptor:descC];

            MPSMatrixMultiplication* mul = [[MPSMatrixMultiplication alloc] initWithDevice:ctx.device transposeLeft:false transposeRight:true resultRows:M resultColumns:N interiorColumns:K alpha:1.0f beta:0.0f];
            [mul encodeToCommandBuffer:cb leftMatrix:matA rightMatrix:matGw resultMatrix:matGateOut];
            [mul encodeToCommandBuffer:cb leftMatrix:matA rightMatrix:matUw resultMatrix:matUpOut];

            // 3. Fused SwiGLU activation + output conversion: out = bfloat(silu(gate) * up)
            id<MTLComputeCommandEncoder> enc2 = s->get_encoder();
            [enc2 setComputePipelineState:pso_swiglu];
            [enc2 setBuffer:b_out offset:o_off atIndex:0];
            [enc2 setBuffer:ctx.mps_scratch_gate_fp16 offset:0 atIndex:1];
            [enc2 setBuffer:ctx.mps_scratch_c_fp16 offset:0 atIndex:2];
            int count_out = M * N;
            [enc2 setBytes:&count_out length:4 atIndex:3];
            [enc2 setBytes:&swiglu_limit length:4 atIndex:4];
            [enc2 dispatchThreads:MTLSizeMake(count_out, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            s->end_encoder_and_maybe_commit();
            return;
        }
    }

    id<MTLComputePipelineState> pso = ctx.get_pipeline("gemm_int3_swiglu_fused_batch_kernel");
    if (!pso) {
        for (int m = 0; m < M; m++) {
            gemv_int3_swiglu_fused_cuda(out + (size_t)m * N, A + (size_t)m * K, gate_weight, gate_scale, up_weight, up_scale, N, K, swiglu_limit, stream);
        }
        return;
    }

    size_t o_off, a_off, gw_off, gs_off, uw_off, us_off;
    id<MTLBuffer> b_out = ctx.get_buffer(out, o_off);
    id<MTLBuffer> b_a = ctx.get_buffer(A, a_off);
    id<MTLBuffer> b_gw = ctx.get_buffer(gate_weight, gw_off);
    id<MTLBuffer> b_gs = ctx.get_buffer(gate_scale, gs_off);
    id<MTLBuffer> b_uw = ctx.get_buffer(up_weight, uw_off);
    id<MTLBuffer> b_us = ctx.get_buffer(up_scale, us_off);

    if (!b_out || !b_a || !b_gw || !b_gs || !b_uw || !b_us) {
        for (int m = 0; m < M; m++) {
            gemv_int3_swiglu_fused_cuda(out + (size_t)m * N, A + (size_t)m * K, gate_weight, gate_scale, up_weight, up_scale, N, K, swiglu_limit, stream);
        }
        return;
    }

    id<MTLComputeCommandEncoder> enc = s->get_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_out offset:o_off atIndex:0];
    [enc setBuffer:b_a offset:a_off atIndex:1];
    [enc setBuffer:b_gw offset:gw_off atIndex:2];
    [enc setBuffer:b_gs offset:gs_off atIndex:3];
    [enc setBuffer:b_uw offset:uw_off atIndex:4];
    [enc setBuffer:b_us offset:us_off atIndex:5];
    [enc setBytes:&N length:sizeof(N) atIndex:6];
    [enc setBytes:&K length:sizeof(K) atIndex:7];
    [enc setBytes:&M length:sizeof(M) atIndex:8];
    [enc setBytes:&swiglu_limit length:sizeof(swiglu_limit) atIndex:9];

    [enc dispatchThreadgroups:MTLSizeMake((N + 31) / 32, (M + 7) / 8, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    s->end_encoder_and_maybe_commit();
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

void apply_repetition_penalty_cuda(float* logits, const int32_t* history_tokens, int num_history, float penalty, cudaStream_t stream) {
    if (penalty == 1.0f || !logits || !history_tokens || num_history <= 0) return;
    metal_stream_synchronize(stream);
    std::unordered_set<int32_t> seen;
    for (int i = 0; i < num_history; i++) {
        int32_t token = history_tokens[i];
        if (token >= 0 && token < 250000 && seen.insert(token).second) {
            if (logits[token] > 0.0f) logits[token] /= penalty;
            else logits[token] *= penalty;
        }
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
