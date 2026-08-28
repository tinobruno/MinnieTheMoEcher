// activations.cu — CUDA kernels for moecher inference engine
// Custom kernels for operations not covered by cuBLAS:
//   RMSNorm, SiLU*mul (SwiGLU), RoPE, FP8/FP4 dequant, embedding,
//   softmax, top-k routing, HC Sinkhorn, etc.

#include "activations.cuh"
#include <cuda_bf16.h>
#include <cuda_bf16.hpp>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>
#include <cstdio>

// ════════════════════════════════════════════════════════════════════════════════
//  Helpers
// ════════════════════════════════════════════════════════════════════════════════

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float v) {
    return __float2bfloat16(v);
}

__device__ __forceinline__ float2 to_float2_bf162(const __nv_bfloat162& v) {
    return make_float2(__bfloat162float(v.x), __bfloat162float(v.y));
}

__device__ __forceinline__ __nv_bfloat162 to_bf162_float2(const float2& v) {
    __nv_bfloat162 t;
    t.x = __float2bfloat16(v.x);
    t.y = __float2bfloat16(v.y);
    return t;
}

// E8M0 (unsigned exponent only) -> float multiplier
// E8M0 has 8-bit exponent with bias 127, no mantissa: value = 2^(e-127)
__device__ __forceinline__ float e8m0_to_float(uint8_t bits) {
    if (bits == 0xFF) return 0.0f;
    return exp2f((float)bits - 127.0f);
}

// FP8 E4M3 -> float
// sign(1) | exponent(4) | mantissa(3), bias=7
__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t bits) {
    uint32_t sign = (bits >> 7) & 1;
    uint32_t exp  = (bits >> 3) & 0xF;
    uint32_t mant = bits & 0x7;
    float val;
    if (exp == 0) {
        val = ldexpf((float)mant, -9);
    } else if (exp == 15 && mant == 7) {
        val = __int_as_float(0x7FC00000);
    } else {
        val = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
    }
    return sign ? -val : val;
}

// FP4 E2M1 -> float (one nibble)
// sign(1) | exponent(2) | mantissa(1), bias=1
__device__ __forceinline__ float fp4_e2m1_to_float(uint8_t nibble) {
    uint32_t sign = (nibble >> 3) & 1;
    uint32_t exp  = (nibble >> 1) & 0x3;
    uint32_t mant = nibble & 0x1;
    float val;
    if (exp == 0) {
        // Subnormal: (-1)^sign * 0.5 * mant
        val = 0.5f * (float)mant;
    } else {
        // Normal: (-1)^sign * 2^(exp-1) * (1 + 0.5*mant)
        val = ldexpf(1.0f + 0.5f * (float)mant, (int)exp - 1);
    }
    return sign ? -val : val;
}

// ════════════════════════════════════════════════════════════════════════════════
//  RMSNorm
// ════════════════════════════════════════════════════════════════════════════════

// Single-row RMSNorm: out = weight * x * rsqrt(mean(x^2) + eps)
// Uses warp-level reduction for efficiency
__global__ void rms_norm_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    int dim, float eps)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    // Each thread accumulates partial sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = bf16_to_float(x[i]);
        sum_sq += v * v;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);

    if (lane == 0) sdata[warp] = sum_sq;
    __syncthreads();

    // Final reduction in first warp
    if (warp == 0) {
        sum_sq = (lane < (blockDim.x >> 5)) ? sdata[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
        if (lane == 0) sdata[0] = sum_sq;
    }
    __syncthreads();

    float rsqrt_val = rsqrtf(sdata[0] / (float)dim + eps);

    // Apply normalization and weight
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = bf16_to_float(x[i]) * rsqrt_val;
        float w = bf16_to_float(weight[i]);
        out[i] = float_to_bf16(w * v);
    }
}

void rms_norm_cuda(__nv_bfloat16* out, const __nv_bfloat16* x,
                   const __nv_bfloat16* weight, int dim, float eps,
                   cudaStream_t stream) {
    int threads = min(1024, ((dim + 31) / 32) * 32);
    int shared = (threads / 32) * sizeof(float);
    rms_norm_kernel<<<1, threads, shared, stream>>>(out, x, weight, dim, eps);
}

__global__ void rms_norm_one_centered_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    int dim, float eps)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    // Each thread accumulates partial sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = bf16_to_float(x[i]);
        sum_sq += v * v;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);

    if (lane == 0) sdata[warp] = sum_sq;
    __syncthreads();

    // Final reduction in first warp
    if (warp == 0) {
        sum_sq = (lane < (blockDim.x >> 5)) ? sdata[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
        if (lane == 0) sdata[0] = sum_sq;
    }
    __syncthreads();

    float rsqrt_val = rsqrtf(sdata[0] / (float)dim + eps);

    // Apply normalization and (1.0 + weight)
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = bf16_to_float(x[i]) * rsqrt_val;
        float w = 1.0f + bf16_to_float(weight[i]);
        out[i] = float_to_bf16(w * v);
    }
}

void rms_norm_one_centered_cuda(__nv_bfloat16* out, const __nv_bfloat16* x,
                                const __nv_bfloat16* weight, int dim, float eps,
                                cudaStream_t stream) {
    int threads = min(1024, ((dim + 31) / 32) * 32);
    int shared = (threads / 32) * sizeof(float);
    rms_norm_one_centered_kernel<<<1, threads, shared, stream>>>(out, x, weight, dim, eps);
}

__global__ void rms_norm_batched_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    int n, int dim, float eps)
{
    int row = blockIdx.x;
    if (row >= n) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    const __nv_bfloat16* x_row = x + (size_t)row * dim;
    __nv_bfloat16* out_row = out + (size_t)row * dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]);
        sum_sq += v * v;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    if (lane == 0) sdata[warp] = sum_sq;
    __syncthreads();

    if (warp == 0) {
        sum_sq = (lane < (blockDim.x >> 5)) ? sdata[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
        if (lane == 0) sdata[0] = sum_sq;
    }
    __syncthreads();

    float rsqrt_val = rsqrtf(sdata[0] / (float)dim + eps);
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]) * rsqrt_val;
        float w = bf16_to_float(weight[i]);
        out_row[i] = float_to_bf16(w * v);
    }
}

void rms_norm_cuda_batched(__nv_bfloat16* out, const __nv_bfloat16* x,
                           const __nv_bfloat16* weight, int n, int dim,
                           float eps, cudaStream_t stream) {
    int threads = min(1024, ((dim + 31) / 32) * 32);
    int shared = (threads / 32) * sizeof(float);
    rms_norm_batched_kernel<<<n, threads, shared, stream>>>(out, x, weight, n, dim, eps);
}

// Unweighted batched RMS norm: x * rsqrt(mean(x^2) + eps) per row
// Used for per-head Q normalization (DeepseekV4UnweightedRMSNorm)
__global__ void rms_norm_unweighted_batched_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    int n, int dim, float eps)
{
    int row = blockIdx.x;
    if (row >= n) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    const __nv_bfloat16* x_row = x + (size_t)row * dim;
    __nv_bfloat16* out_row = out + (size_t)row * dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]);
        sum_sq += v * v;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    if (lane == 0) sdata[warp] = sum_sq;
    __syncthreads();

    if (warp == 0) {
        sum_sq = (lane < (blockDim.x >> 5)) ? sdata[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
        if (lane == 0) sdata[0] = sum_sq;
    }
    __syncthreads();

    float rsqrt_val = rsqrtf(sdata[0] / (float)dim + eps);
    for (int i = tid; i < dim; i += blockDim.x) {
        out_row[i] = float_to_bf16(bf16_to_float(x_row[i]) * rsqrt_val);
    }
}

void rms_norm_unweighted_batched_cuda(__nv_bfloat16* out, const __nv_bfloat16* x,
                                       int n, int dim, float eps,
                                       cudaStream_t stream) {
    int threads = min(1024, ((dim + 31) / 32) * 32);
    int shared = (threads / 32) * sizeof(float);
    rms_norm_unweighted_batched_kernel<<<n, threads, shared, stream>>>(out, x, n, dim, eps);
}

// ════════════════════════════════════════════════════════════════════════════════
//  SiLU * mul (SwiGLU activation)
// ════════════════════════════════════════════════════════════════════════════════

__global__ void silu_mul_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    int n, float swiglu_limit)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = bf16_to_float(gate[idx]);
    float u = bf16_to_float(up[idx]);

    if (swiglu_limit > 0.0f) {
        u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
        g = fminf(g, swiglu_limit);
    }

    // SiLU(g) = g * sigmoid(g)
    float silu_g = g / (1.0f + expf(-g));
    out[idx] = float_to_bf16(silu_g * u);
}

void silu_mul_cuda(__nv_bfloat16* out, const __nv_bfloat16* gate,
                   const __nv_bfloat16* up, int n, float swiglu_limit,
                   cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_mul_kernel<<<blocks, threads, 0, stream>>>(out, gate, up, n, swiglu_limit);
}

// ════════════════════════════════════════════════════════════════════════════════
//  RoPE (Rotary Position Embeddings)
// ════════════════════════════════════════════════════════════════════════════════

__global__ void rope_kernel(
    __nv_bfloat16* __restrict__ x,   // [n_vectors, head_dim]
    int n_vectors, int head_dim, int rope_dim,
    int position,
    const float* __restrict__ freq_table,  // [max_seq_len, rope_dim/2] — stored as pairs (cos, sin)
    bool inverse)
{
    int vec_id = blockIdx.x;
    int pair_id = threadIdx.x;  // which pair within the rope_dim

    if (vec_id >= n_vectors || pair_id >= rope_dim / 2) return;

    int half_rope = rope_dim / 2;
    // The rope is applied to the LAST rope_dim elements
    int base_idx = vec_id * head_dim + (head_dim - rope_dim) + 2 * pair_id;

    float x0 = bf16_to_float(x[base_idx]);
    float x1 = bf16_to_float(x[base_idx + 1]);

    // freq_table stores precomputed cos and sin for each position and dimension
    float cos_val = freq_table[position * half_rope * 2 + pair_id * 2];
    float sin_val = freq_table[position * half_rope * 2 + pair_id * 2 + 1];

    if (inverse) sin_val = -sin_val;

    float y0 = x0 * cos_val - x1 * sin_val;
    float y1 = x0 * sin_val + x1 * cos_val;

    x[base_idx]     = float_to_bf16(y0);
    x[base_idx + 1] = float_to_bf16(y1);
}

void rope_cuda(__nv_bfloat16* x, int n_vectors, int head_dim, int rope_dim,
               int position, const float* freq_table, bool inverse,
               cudaStream_t stream) {
    int pairs = rope_dim / 2;
    rope_kernel<<<n_vectors, pairs, 0, stream>>>(
        x, n_vectors, head_dim, rope_dim, position, freq_table, inverse);
}

// ════════════════════════════════════════════════════════════════════════════════
//  FP8 E4M3 Dequantization
// ════════════════════════════════════════════════════════════════════════════════

__global__ void fp8_dequant_kernel(
    __nv_bfloat16* __restrict__ out,
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scale,
    int rows, int cols, int block_size,
    int scale_cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;

    int r = idx / cols;
    int c = idx % cols;

    // Find block indices
    int br = r / block_size;
    int bc = c / block_size;

    float s = e8m0_to_float(scale[br * scale_cols + bc]);
    float w = fp8_e4m3_to_float(weight[idx]);

    out[idx] = float_to_bf16(w * s);
}

void fp8_dequant_cuda(__nv_bfloat16* out, const uint8_t* weight,
                      const uint8_t* scale, int rows, int cols,
                      int block_size, cudaStream_t stream) {
    int scale_cols = (cols + block_size - 1) / block_size;
    int total = rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fp8_dequant_kernel<<<blocks, threads, 0, stream>>>(
        out, weight, scale, rows, cols, block_size, scale_cols);
}

// ════════════════════════════════════════════════════════════════════════════════
//  FP4 E2M1 Dequantization (packed 2 per byte)
// ════════════════════════════════════════════════════════════════════════════════

__global__ void fp4_dequant_kernel(
    __nv_bfloat16* __restrict__ out,
    const uint8_t* __restrict__ weight,  // [rows, cols_packed]
    const uint8_t* __restrict__ scale,   // [rows, scale_cols]
    int rows, int cols_packed,
    int scale_cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols_packed;
    if (idx >= total) return;

    int r = idx / cols_packed;
    int cp = idx % cols_packed;

    uint8_t packed = weight[idx];
    // Low nibble = first fp4, high nibble = second fp4
    uint8_t lo = packed & 0x0F;
    uint8_t hi = (packed >> 4) & 0x0F;

    int logical_col0 = cp * 2;
    int logical_col1 = cp * 2 + 1;

    // Scale block: 32 fp4 values share one scale = 16 bytes share one scale
    int sc0 = logical_col0 / 32;
    int sc1 = logical_col1 / 32;

    float s0 = e8m0_to_float(scale[r * scale_cols + sc0]);
    float s1 = e8m0_to_float(scale[r * scale_cols + sc1]);

    float v0 = fp4_e2m1_to_float(lo) * s0;
    float v1 = fp4_e2m1_to_float(hi) * s1;

    int out_col0 = r * (cols_packed * 2) + logical_col0;
    int out_col1 = r * (cols_packed * 2) + logical_col1;

    out[out_col0] = float_to_bf16(v0);
    out[out_col1] = float_to_bf16(v1);
}

void fp4_dequant_cuda(__nv_bfloat16* out, const uint8_t* weight,
                      const uint8_t* scale, int rows, int cols_packed,
                      int scale_cols, cudaStream_t stream) {
    int total = rows * cols_packed;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fp4_dequant_kernel<<<blocks, threads, 0, stream>>>(
        out, weight, scale, rows, cols_packed, scale_cols);
}

// ════════════════════════════════════════════════════════════════════════════════
//  Embedding lookup
// ════════════════════════════════════════════════════════════════════════════════

__global__ void embedding_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ table,
    const int32_t* __restrict__ ids,
    int seq_len, int dim)
{
    int s = blockIdx.x;
    int d = threadIdx.x + blockIdx.y * blockDim.x;
    if (s >= seq_len || d >= dim) return;

    int token_id = ids[s];
    out[s * dim + d] = table[(size_t)token_id * dim + d];
}

void embedding_cuda(__nv_bfloat16* out, const __nv_bfloat16* table,
                    const int32_t* ids, int seq_len, int dim,
                    cudaStream_t stream) {
    dim3 blocks(seq_len, (dim + 255) / 256);
    embedding_kernel<<<blocks, 256, 0, stream>>>(out, table, ids, seq_len, dim);
}

__global__ void embedding_broadcast_kernel(
    __nv_bfloat16* __restrict__ hidden,
    __nv_bfloat16* __restrict__ hc_state,
    const __nv_bfloat16* __restrict__ table,
    int token_id, int dim, int hc)
{
    int d = threadIdx.x + blockIdx.x * blockDim.x;
    if (d >= dim) return;

    __nv_bfloat16 val = table[(size_t)token_id * dim + d];
    hidden[d] = val;
    for (int h = 0; h < hc; h++) {
        hc_state[(size_t)h * dim + d] = val;
    }
}

void embedding_broadcast_cuda(
    __nv_bfloat16* hidden, __nv_bfloat16* hc_state,
    const __nv_bfloat16* table, int token_id, int dim, int hc,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    embedding_broadcast_kernel<<<blocks, threads, 0, stream>>>(hidden, hc_state, table, token_id, dim, hc);
}

// ════════════════════════════════════════════════════════════════════════════════
//  Softmax (row-wise)
// ════════════════════════════════════════════════════════════════════════════════

__global__ void softmax_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    const float* xr = x + (size_t)row * cols;
    float* outr = out + (size_t)row * cols;

    // Find max
    float max_val = -FLT_MAX;
    for (int c = tid; c < cols; c += blockDim.x)
        max_val = fmaxf(max_val, xr[c]);

    for (int offset = 16; offset > 0; offset >>= 1)
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    if (lane == 0) sdata[warp] = max_val;
    __syncthreads();
    if (warp == 0) {
        max_val = (lane < (blockDim.x >> 5)) ? sdata[lane] : -FLT_MAX;
        for (int offset = 16; offset > 0; offset >>= 1)
            max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
        if (lane == 0) sdata[0] = max_val;
    }
    __syncthreads();
    max_val = sdata[0];

    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float e = expf(xr[c] - max_val);
        outr[c] = e;
        sum_exp += e;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
    if (lane == 0) sdata[warp] = sum_exp;
    __syncthreads();
    if (warp == 0) {
        sum_exp = (lane < (blockDim.x >> 5)) ? sdata[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
        if (lane == 0) sdata[0] = sum_exp;
    }
    __syncthreads();
    float inv_sum = 1.0f / sdata[0];

    // Normalize
    for (int c = tid; c < cols; c += blockDim.x)
        outr[c] *= inv_sum;
}

void softmax_cuda(float* out, const float* x, int rows, int cols,
                  cudaStream_t stream) {
    int threads = min(1024, ((cols + 31) / 32) * 32);
    int shared = (threads / 32) * sizeof(float);
    softmax_kernel<<<rows, threads, shared, stream>>>(out, x, rows, cols);
}

// ════════════════════════════════════════════════════════════════════════════════
//  Residual add
// ════════════════════════════════════════════════════════════════════════════════

__global__ void add_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float va = bf16_to_float(a[idx]);
    float vb = bf16_to_float(b[idx]);
    out[idx] = float_to_bf16(va + vb);
}

void add_cuda(__nv_bfloat16* out, const __nv_bfloat16* a,
              const __nv_bfloat16* b, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads, 0, stream>>>(out, a, b, n);
}

void vector_add_bf16_cuda(__nv_bfloat16* a, const __nv_bfloat16* b, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads, 0, stream>>>(a, a, b, n);
}

__global__ void add_f32_sigmoid_kernel(
    float* __restrict__ out,
    const float* __restrict__ a,
    const float* __restrict__ bias,
    int n, bool apply_sigmoid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = a[idx];
    if (bias != nullptr) {
        v += bias[idx];
    }
    if (apply_sigmoid) {
        v = 1.0f / (1.0f + expf(-v));
    }
    out[idx] = v;
}

void add_f32_sigmoid_cuda(float* out, const float* a, const float* bias,
                          int n, bool apply_sigmoid, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_f32_sigmoid_kernel<<<blocks, threads, 0, stream>>>(out, a, bias, n, apply_sigmoid);
}

// ════════════════════════════════════════════════════════════════════════════════
//  Weighted add (for expert accumulation)
// ════════════════════════════════════════════════════════════════════════════════

__global__ void weighted_add_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    float weight, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v_out = bf16_to_float(out[idx]);
    float v_x = bf16_to_float(x[idx]);
    out[idx] = float_to_bf16(v_out + weight * v_x);
}

void weighted_add_cuda(__nv_bfloat16* out, const __nv_bfloat16* x,
                       float weight, int dim, cudaStream_t stream) {
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    weighted_add_kernel<<<blocks, threads, 0, stream>>>(out, x, weight, dim);
}

// ════════════════════════════════════════════════════════════════════════════════
//  Top-k selection (simple serial on GPU — small n=256)
// ════════════════════════════════════════════════════════════════════════════════

__global__ void topk_kernel(
    float* __restrict__ out_vals,
    int32_t* __restrict__ out_idx,
    const float* __restrict__ scores,
    int n, int k)
{
    // Single thread — n is small (256 experts)
    for (int ki = 0; ki < k; ki++) {
        float best_val = -FLT_MAX;
        int best_idx = -1;
        for (int i = 0; i < n; i++) {
            // Skip already selected
            bool skip = false;
            for (int j = 0; j < ki; j++) {
                if (out_idx[j] == i) { skip = true; break; }
            }
            if (skip) continue;
            if (scores[i] > best_val) {
                best_val = scores[i];
                best_idx = i;
            }
        }
        out_vals[ki] = best_val;
        out_idx[ki] = best_idx;
    }
}

void topk_cuda(float* out_vals, int32_t* out_idx, const float* scores,
               int n, int k, cudaStream_t stream) {
    topk_kernel<<<1, 1, 0, stream>>>(out_vals, out_idx, scores, n, k);
}

// ════════════════════════════════════════════════════════════════════════════════
//  sqrt(softplus(x)) scoring
// ════════════════════════════════════════════════════════════════════════════════

__global__ void sqrtsoftplus_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = x[idx];
    // softplus(v) = log(1 + exp(v))
    float sp = (v > 20.0f) ? v : logf(1.0f + expf(v));
    out[idx] = sqrtf(sp);
}

void sqrtsoftplus_cuda(float* out, const float* x, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sqrtsoftplus_kernel<<<blocks, threads, 0, stream>>>(out, x, n);
}

// ════════════════════════════════════════════════════════════════════════════════
//  HC Split + Sinkhorn
// ════════════════════════════════════════════════════════════════════════════════

// Single-thread kernel for the small HC matrices (hc_mult=4)
__global__ void hc_split_sinkhorn_kernel(
    float* __restrict__ pre,        // [hc_mult]
    float* __restrict__ post,       // [hc_mult]
    float* __restrict__ comb,       // [hc_mult * hc_mult]
    const float* __restrict__ mixes,// [(2+hc_mult)*hc_mult]
    const float* __restrict__ scale,// [3]
    const float* __restrict__ base, // [(2+hc_mult)*hc_mult]
    int hc_mult, int sinkhorn_iters, float eps)
{
    int tid = threadIdx.x;
    int hc = hc_mult;
    
    // Pre
    if (tid < hc) {
        float v = mixes[tid] * scale[0] + base[tid];
        pre[tid] = 1.0f / (1.0f + expf(-v)) + eps;
    }
    // Post
    if (tid >= hc && tid < 2*hc) {
        int i = tid - hc;
        float v = mixes[hc + i] * scale[1] + base[hc + i];
        post[i] = 2.0f / (1.0f + expf(-v));
    }

    __shared__ float s_comb[64]; // max hc=8
    __shared__ float s_row_sum[8];
    __shared__ float s_col_sum[8];

    // Comb logits
    if (tid < hc * hc) {
        int r = tid / hc;
        int c = tid % hc;
        float v = mixes[2 * hc + r * hc + c] * scale[2] + base[2 * hc + r * hc + c];
        s_comb[r * hc + c] = v;
    }
    __syncthreads();

    // Row max & softmax
    if (tid < hc) {
        int r = tid;
        float max_val = -1e38f;
        for (int c = 0; c < hc; c++) {
            max_val = fmaxf(max_val, s_comb[r * hc + c]);
        }
        float row_sum = 0.0f;
        for (int c = 0; c < hc; c++) {
            float e = expf(s_comb[r * hc + c] - max_val);
            s_comb[r * hc + c] = e;
            row_sum += e;
        }
        for (int c = 0; c < hc; c++) {
            s_comb[r * hc + c] = (s_comb[r * hc + c] / row_sum) + eps;
        }
    }
    __syncthreads();

    // Col normalize
    if (tid < hc) {
        int c = tid;
        float col_sum = 0.0f;
        for (int r = 0; r < hc; r++) {
            col_sum += s_comb[r * hc + c];
        }
        float inv = 1.0f / (col_sum + eps);
        for (int r = 0; r < hc; r++) {
            s_comb[r * hc + c] *= inv;
        }
    }
    __syncthreads();

    // Sinkhorn loop
    for (int iter = 0; iter < sinkhorn_iters - 1; iter++) {
        // Row norm
        if (tid < hc) {
            int r = tid;
            float row_sum = 0.0f;
            for (int c = 0; c < hc; c++) {
                row_sum += s_comb[r * hc + c];
            }
            float inv = 1.0f / (row_sum + eps);
            s_row_sum[r] = inv;
        }
        __syncthreads();
        if (tid < hc * hc) {
            int r = tid / hc;
            int c = tid % hc;
            s_comb[r * hc + c] *= s_row_sum[r];
        }
        __syncthreads();

        // Col norm
        if (tid < hc) {
            int c = tid;
            float col_sum = 0.0f;
            for (int r = 0; r < hc; r++) {
                col_sum += s_comb[r * hc + c];
            }
            float inv = 1.0f / (col_sum + eps);
            s_col_sum[c] = inv;
        }
        __syncthreads();
        if (tid < hc * hc) {
            int r = tid / hc;
            int c = tid % hc;
            s_comb[r * hc + c] *= s_col_sum[c];
        }
        __syncthreads();
    }

    if (tid < hc * hc) {
        comb[tid] = s_comb[tid];
    }
}

void hc_split_sinkhorn_cuda(float* pre, float* post, float* comb,
                             const float* mixes, const float* scale,
                             const float* base, int hc_mult,
                             int sinkhorn_iters, float eps,
                             cudaStream_t stream) {
    hc_split_sinkhorn_kernel<<<1, 32, 0, stream>>>(
        pre, post, comb, mixes, scale, base, hc_mult, sinkhorn_iters, eps);
}

// ════════════════════════════════════════════════════════════════════════════════
//  Precompute RoPE frequency table (YaRN)
// ════════════════════════════════════════════════════════════════════════════════

__global__ void precompute_freqs_kernel(
    float* __restrict__ freqs,      // [max_seq_len, rope_dim/2, 2]  (cos, sin pairs)
    int max_seq_len, int rope_dim, float base, float factor,
    int original_seq_len, int beta_fast, int beta_slow)
{
    int pos = blockIdx.x;
    int pair = threadIdx.x;
    int half_dim = rope_dim / 2;
    if (pos >= max_seq_len || pair >= half_dim) return;

    int i = pair * 2;
    float theta_extrap = (float)pos * powf(base, -((float)i) / (float)rope_dim);
    float freq_scale = 1.0f / factor;
    float theta = freq_scale * theta_extrap;
    float mscale = 1.0f;

    if (original_seq_len > 0) {
        float denom = 2.0f * logf(base);
        float corr0 = floorf((float)rope_dim * logf((float)original_seq_len / ((float)beta_fast * 2.0f * 3.141592653589793f)) / denom);
        float corr1 = ceilf((float)rope_dim * logf((float)original_seq_len / ((float)beta_slow * 2.0f * 3.141592653589793f)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(rope_dim - 1), corr1);

        float y = ((float)pair - corr0) / fmaxf(0.001f, corr1 - corr0);
        float ramp_mix = 1.0f - fminf(1.0f, fmaxf(0.0f, y));

        theta = (freq_scale * theta_extrap) * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
    }

    float cos_val = cosf(theta);
    float sin_val = sinf(theta);

    int out_idx = pos * half_dim * 2 + pair * 2;
    freqs[out_idx]     = cos_val;
    freqs[out_idx + 1] = sin_val;
}

void precompute_freqs_cuda(float* freqs, int max_seq_len, int rope_dim,
                           float base, float factor, int original_seq_len,
                           int beta_fast, int beta_slow, cudaStream_t stream) {
    int half_dim = rope_dim / 2;
    precompute_freqs_kernel<<<max_seq_len, half_dim, 0, stream>>>(
        freqs, max_seq_len, rope_dim, base, factor,
        original_seq_len, beta_fast, beta_slow);
}

// ════════════════════════════════════════════════════════════════════════════════
//  BF16 <-> F32 conversions
// ════════════════════════════════════════════════════════════════════════════════

__global__ void bf16_to_f32_kernel(float* out, const __nv_bfloat16* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = bf16_to_float(in[idx]);
}

void bf16_to_f32_cuda(float* out, const __nv_bfloat16* in, int n, cudaStream_t stream) {
    bf16_to_f32_kernel<<<(n+255)/256, 256, 0, stream>>>(out, in, n);
}

__global__ void f32_to_bf16_kernel(__nv_bfloat16* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = float_to_bf16(in[idx]);
}

void f32_to_bf16_cuda(__nv_bfloat16* out, const float* in, int n, cudaStream_t stream) {
    f32_to_bf16_kernel<<<(n+255)/256, 256, 0, stream>>>(out, in, n);
}

// ── Custom MLA Attention Kernel ─────────────────────────────────────────────

__global__ void mla_attention_kernel(
    const __nv_bfloat16* __restrict__ q,       // [n_heads, head_dim]
    const __nv_bfloat16* __restrict__ kv,      // [cache_len, head_dim]
    const float* __restrict__ attn_sink,       // [n_heads]
    __nv_bfloat16* __restrict__ out,           // [n_heads, head_dim]
    int cache_len,
    int head_dim,
    float scale
) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int n_threads = blockDim.x;

    // Dynamically allocated shared memory for scores
    extern __shared__ float s_mem[]; // scores: [cache_len]
    float* scores = s_mem;

    // Load Q for this head into shared memory
    __shared__ float s_q[512];
    for (int d = tid; d < head_dim; d += n_threads) {
        s_q[d] = __bfloat162float(q[h * head_dim + d]);
    }
    __syncthreads();

    // 1. Q @ KV.T
    for (int t = tid; t < cache_len; t += n_threads) {
        const __nv_bfloat16* kv_t = kv + (size_t)t * head_dim;
        float dot = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < head_dim; d++) {
            dot += s_q[d] * __bfloat162float(kv_t[d]);
        }
        scores[t] = dot * scale;
    }
    __syncthreads();

    // 2. Softmax
    float local_max = -1e38f;
    for (int t = tid; t < cache_len; t += n_threads) {
        local_max = fmaxf(local_max, scores[t]);
    }
    if (tid == 0) {
        local_max = fmaxf(local_max, attn_sink[h]);
    }

    // Warp-level max reduction
    float warp_max = local_max;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        warp_max = fmaxf(warp_max, __shfl_down_sync(0xffffffff, warp_max, offset));

    __shared__ float s_red[32];
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) s_red[warp] = warp_max;
    __syncthreads();

    float block_max = (tid < (n_threads >> 5)) ? s_red[tid] : -1e38f;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        block_max = fmaxf(block_max, __shfl_down_sync(0xffffffff, block_max, offset));
    if (tid == 0) s_red[0] = block_max;
    __syncthreads();
    block_max = s_red[0];

    // Exp and sum
    float local_sum = 0.0f;
    for (int t = tid; t < cache_len; t += n_threads) {
        float e = expf(scores[t] - block_max);
        scores[t] = e;
        local_sum += e;
    }
    if (tid == 0) {
        local_sum += expf(attn_sink[h] - block_max);
    }

    // Warp-level sum reduction
    float warp_sum = local_sum;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
    if (lane == 0) s_red[warp] = warp_sum;
    __syncthreads();

    float block_sum = (tid < (n_threads >> 5)) ? s_red[tid] : 0.0f;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
    if (tid == 0) s_red[0] = block_sum;
    __syncthreads();
    float inv_sum = 1.0f / s_red[0];

    // Pre-normalize scores in shared memory
    for (int t = tid; t < cache_len; t += n_threads) {
        scores[t] *= inv_sum;
    }
    __syncthreads();

    // 3. Normalize and compute out = scores @ KV
    for (int d = tid; d < head_dim; d += n_threads) {
        float v = 0.0f;
        for (int t = 0; t < cache_len; t++) {
            v += scores[t] * __bfloat162float(kv[(size_t)t * head_dim + d]);
        }
        out[(size_t)h * head_dim + d] = __float2bfloat16(v);
    }
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
    int n_threads = 256;
    size_t smem_bytes = cache_len * sizeof(float);
    if (smem_bytes > 49152) {
        static bool s_smem_configured = false;
        if (!s_smem_configured) {
            cudaFuncSetAttribute(mla_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
            s_smem_configured = true;
        }
    }
    mla_attention_kernel<<<n_heads, n_threads, smem_bytes, stream>>>(
        q, kv, attn_sink, out, cache_len, head_dim, scale
    );
}

// ════════════════════════════════════════════════════════════════════════════════
//  Compressor Kernels
// ════════════════════════════════════════════════════════════════════════════════

// BF16 GEMV: out[row] = dot(W[row, :], x[:]) for each row
// 1 Warp (32 threads) per row using 128-bit (8 BF16) vectorized memory transactions
template<typename TOut>
__global__ void gemv_bf16_generic_kernel(
    TOut* __restrict__ out,
    const __nv_bfloat16* __restrict__ W,
    const __nv_bfloat16* __restrict__ x,
    int N, int K)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;

    int lane = threadIdx.x;
    float sum = 0.0f;

    const uint4* w_vec4 = reinterpret_cast<const uint4*>(W + (size_t)row * K);
    const uint4* x_vec4 = reinterpret_cast<const uint4*>(x);

    int n_chunks = K / 8;
    for (int chunk = lane; chunk < n_chunks; chunk += 32) {
        uint4 w4 = w_vec4[chunk];
        uint4 x4 = x_vec4[chunk];

        const __nv_bfloat162* w_pairs = reinterpret_cast<const __nv_bfloat162*>(&w4);
        const __nv_bfloat162* x_pairs = reinterpret_cast<const __nv_bfloat162*>(&x4);

        #pragma unroll
        for (int p = 0; p < 4; p++) {
            float2 wf = to_float2_bf162(w_pairs[p]);
            float2 xf = to_float2_bf162(x_pairs[p]);
            sum += wf.x * xf.x + wf.y * xf.y;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0) {
        if constexpr (std::is_same_v<TOut, __nv_bfloat16>) {
            out[row] = __float2bfloat16(sum);
        } else {
            out[row] = sum;
        }
    }
}

void gemv_bf16_cuda(
    float* out,
    const __nv_bfloat16* W,
    const __nv_bfloat16* x,
    int N, int K,
    cudaStream_t stream)
{
    dim3 threads(32, 8);
    dim3 blocks((N + 7) / 8);
    gemv_bf16_generic_kernel<<<blocks, threads, 0, stream>>>(out, W, x, N, K);
}

void gemv_bf16_out_bf16_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* W,
    const __nv_bfloat16* x,
    int N, int K,
    cudaStream_t stream)
{
    dim3 threads(32, 8);
    dim3 blocks((N + 7) / 8);
    gemv_bf16_generic_kernel<<<blocks, threads, 0, stream>>>(out, W, x, N, K);
}


// Softmax-gated pooling: each thread handles one output dimension.
// For each dimension d:
//   1. Find max of score[i][d] across i in [0, window)
//   2. Compute softmax weights = exp(score[i][d] - max) / sum
//   3. out[d] = sum_i(kv[i][d] * weight_i)
__global__ void compressor_pool_kernel(
    float* __restrict__ out,
    const float* __restrict__ kv,
    const float* __restrict__ score,
    int window, int dim)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;

    // Find max for numerical stability
    float max_score = -1e30f;
    for (int i = 0; i < window; i++) {
        float s = score[i * dim + d];
        if (s > max_score) max_score = s;
    }

    // Compute softmax and weighted sum in one pass
    float sum_exp = 0.0f;
    float weighted_sum = 0.0f;
    for (int i = 0; i < window; i++) {
        float w = expf(score[i * dim + d] - max_score);
        sum_exp += w;
        weighted_sum += w * kv[i * dim + d];
    }

    out[d] = weighted_sum / (sum_exp + 1e-10f);
}

void compressor_pool_cuda(
    float* out,
    const float* kv,
    const float* score,
    int window, int dim,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    compressor_pool_kernel<<<blocks, threads, 0, stream>>>(out, kv, score, window, dim);
}


// Combine raw and compressed KV caches into a contiguous buffer
__global__ void combine_kv_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ raw_kv,
    int raw_len,
    const __nv_bfloat16* __restrict__ comp_kv,
    int comp_len,
    int head_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (raw_len + comp_len) * head_dim;
    if (idx >= total) return;

    int entry = idx / head_dim;
    int d = idx % head_dim;
    if (entry < raw_len) {
        out[idx] = raw_kv[entry * head_dim + d];
    } else {
        out[idx] = comp_kv[(entry - raw_len) * head_dim + d];
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
    int total = (raw_len + comp_len) * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    combine_kv_kernel<<<blocks, threads, 0, stream>>>(out, raw_kv, raw_len, comp_kv, comp_len, head_dim);
}

// FP8 GEMV Kernel
__device__ inline float fp8_e4m3_to_float_v2(uint8_t val) {
    if (val == 0) return 0.0f;
    uint32_t sign = (uint32_t)(val & 0x80) << 24;
    uint32_t exp  = (uint32_t)(val & 0x78) >> 3;
    uint32_t mant = (uint32_t)(val & 0x07);
    uint32_t f_exp = exp + 120;
    uint32_t res = sign | (f_exp << 23) | (mant << 20);
    return __uint_as_float(res);
}

__device__ inline float e8m0_to_float_v2(uint8_t val) {
    return __uint_as_float((uint32_t)val << 23);
}

// ── Hardware Asynchronous Copy (cp.async) Primitives ──────────────────────────
__device__ __forceinline__ void cp_async_16_bytes(void* smem_dst, const void* gmem_src) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    uint32_t smem_addr = __cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" : : "r"(smem_addr), "l"(gmem_src));
#else
    *(uint4*)smem_dst = *(const uint4*)gmem_src;
#endif
}

__device__ __forceinline__ void cp_async_commit_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

__global__ void gemv_fp8_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ vec,
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scale,
    int N, int K, int block_size)
{
    __shared__ float s_fp8_lut[256];
    __shared__ __nv_bfloat16 s_vec[2][256]; // Double-buffered vector cache (1 KB)

    int tid = threadIdx.y * 32 + threadIdx.x;
    if (tid < 256) s_fp8_lut[tid] = fp8_e4m3_to_float_v2(tid);

    // Pipeline prologue: prefetch first 256 elements of vec into s_vec[0]
    if (tid < 32) {
        cp_async_16_bytes((uint4*)s_vec[0] + tid, (const uint4*)vec + tid);
    }
    cp_async_commit_group();
    cp_async_wait_all();
    __syncthreads();

    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;

    int lane = threadIdx.x;
    float sum = 0.0f;
    int scale_cols = (K + block_size - 1) / block_size;
    int br = row / block_size;

    const uint2* w_vec8 = reinterpret_cast<const uint2*>(&weight[(size_t)row * K]);

    int n_tiles = K / 256;
    for (int t = 0; t < n_tiles; t++) {
        int curr_stage = t & 1;
        int next_stage = (t + 1) & 1;

        // Asynchronously prefetch next tile of input vector
        if (t + 1 < n_tiles && tid < 32) {
            cp_async_16_bytes((uint4*)s_vec[next_stage] + tid,
                              (const uint4*)(vec + ((t + 1) << 8)) + tid);
            cp_async_commit_group();
        }

        int chunk = (t << 5) + lane;
        int logical_col = chunk * 8;
        int bc = logical_col / block_size;
        float s_val = e8m0_to_float_v2(scale[br * scale_cols + bc]);

        uint2 w8 = w_vec8[chunk];
        const uint4* s_a_vec4 = reinterpret_cast<const uint4*>(s_vec[curr_stage]);
        uint4 a8 = s_a_vec4[lane];

        float2 f0 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&a8.x));
        float2 f1 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&a8.y));
        float2 f2 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&a8.z));
        float2 f3 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&a8.w));

        float chunk_sum = 0.0f;
        chunk_sum += s_fp8_lut[(w8.x) & 0xFF] * f0.x;
        chunk_sum += s_fp8_lut[(w8.x >> 8) & 0xFF] * f0.y;
        chunk_sum += s_fp8_lut[(w8.x >> 16) & 0xFF] * f1.x;
        chunk_sum += s_fp8_lut[(w8.x >> 24) & 0xFF] * f1.y;

        chunk_sum += s_fp8_lut[(w8.y) & 0xFF] * f2.x;
        chunk_sum += s_fp8_lut[(w8.y >> 8) & 0xFF] * f2.y;
        chunk_sum += s_fp8_lut[(w8.y >> 16) & 0xFF] * f3.x;
        chunk_sum += s_fp8_lut[(w8.y >> 24) & 0xFF] * f3.y;

        sum += chunk_sum * s_val;

        if (t + 1 < n_tiles) {
            cp_async_wait_all();
            __syncthreads();
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0) {
        out[row] = __float2bfloat16(sum);
    }
}

void gemv_fp8_cuda(__nv_bfloat16* out, const __nv_bfloat16* vec,
                   const uint8_t* weight, const uint8_t* scale,
                   int N, int K, int block_size, cudaStream_t stream) {
    dim3 threads(32, 8);
    dim3 blocks((N + 7) / 8);
    gemv_fp8_kernel<<<blocks, threads, 0, stream>>>(out, vec, weight, scale, N, K, block_size);
}

__global__ void gemv_fp8_grouped_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ vec,
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scale,
    int N, int K, int groups, int block_size)
{
    __shared__ float s_fp8_lut[256];
    __shared__ __nv_bfloat16 s_vec[2][256]; // Double-buffered vector cache (1 KB)

    int tid = threadIdx.y * 32 + threadIdx.x;
    if (tid < 256) s_fp8_lut[tid] = fp8_e4m3_to_float_v2(tid);

    int group = blockIdx.y;
    const __nv_bfloat16* g_vec = vec + (size_t)group * K;

    // Pipeline prologue: prefetch first 256 elements of g_vec into s_vec[0]
    if (tid < 32) {
        cp_async_16_bytes((uint4*)s_vec[0] + tid, (const uint4*)g_vec + tid);
    }
    cp_async_commit_group();
    cp_async_wait_all();
    __syncthreads();

    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (group >= groups || row >= N) return;

    int lane = threadIdx.x;
    float sum = 0.0f;
    int scale_cols = (K + block_size - 1) / block_size;
    int scale_rows_per_group = (N + block_size - 1) / block_size;
    int br = row / block_size;

    const uint8_t* g_weight = weight + (size_t)group * N * K;
    const uint8_t* g_scale = scale + (size_t)group * scale_rows_per_group * scale_cols;
    __nv_bfloat16* g_out = out + (size_t)group * N;

    const uint2* w_vec8 = reinterpret_cast<const uint2*>(&g_weight[(size_t)row * K]);

    int n_tiles = K / 256;
    for (int t = 0; t < n_tiles; t++) {
        int curr_stage = t & 1;
        int next_stage = (t + 1) & 1;

        // Asynchronously prefetch next tile of input vector
        if (t + 1 < n_tiles && tid < 32) {
            cp_async_16_bytes((uint4*)s_vec[next_stage] + tid,
                              (const uint4*)(g_vec + ((t + 1) << 8)) + tid);
            cp_async_commit_group();
        }

        int chunk = (t << 5) + lane;
        int logical_col = chunk * 8;
        int bc = logical_col / block_size;
        float s_val = e8m0_to_float_v2(g_scale[br * scale_cols + bc]);

        uint2 w8 = w_vec8[chunk];
        const uint4* s_a_vec4 = reinterpret_cast<const uint4*>(s_vec[curr_stage]);
        uint4 a8 = s_a_vec4[lane];

        float2 f0 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&a8.x));
        float2 f1 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&a8.y));
        float2 f2 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&a8.z));
        float2 f3 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&a8.w));

        float chunk_sum = 0.0f;
        chunk_sum += s_fp8_lut[(w8.x) & 0xFF] * f0.x;
        chunk_sum += s_fp8_lut[(w8.x >> 8) & 0xFF] * f0.y;
        chunk_sum += s_fp8_lut[(w8.x >> 16) & 0xFF] * f1.x;
        chunk_sum += s_fp8_lut[(w8.x >> 24) & 0xFF] * f1.y;

        chunk_sum += s_fp8_lut[(w8.y) & 0xFF] * f2.x;
        chunk_sum += s_fp8_lut[(w8.y >> 8) & 0xFF] * f2.y;
        chunk_sum += s_fp8_lut[(w8.y >> 16) & 0xFF] * f3.x;
        chunk_sum += s_fp8_lut[(w8.y >> 24) & 0xFF] * f3.y;

        sum += chunk_sum * s_val;

        if (t + 1 < n_tiles) {
            cp_async_wait_all();
            __syncthreads();
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0) {
        g_out[row] = __float2bfloat16(sum);
    }
}

void gemv_fp8_grouped_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* vec,
    const uint8_t* weight, const uint8_t* scale,
    int N, int K, int groups, int block_size,
    cudaStream_t stream)
{
    dim3 threads(32, 8);
    dim3 blocks((N + 7) / 8, groups);
    gemv_fp8_grouped_kernel<<<blocks, threads, 0, stream>>>(
        out, vec, weight, scale, N, K, groups, block_size);
}

__global__ void gemv_hc_pre_norm_kernel(
    float* __restrict__ mixes,
    const __nv_bfloat16* __restrict__ hc_state,
    const float* __restrict__ hc_fn,
    int mix_size, int hc_dim, float eps)
{
    int row = blockIdx.x;
    if (row >= mix_size) return;

    int tid = threadIdx.x;
    
    // Step 1: Compute sum of squares across hc_state in parallel (128-bit loads: 8 bf16s per uint4)
    float sum_sq = 0.0f;
    const uint4* hc_vec8 = reinterpret_cast<const uint4*>(hc_state);
    int n_vec8 = hc_dim / 8;
    for (int i = tid; i < n_vec8; i += blockDim.x) {
        uint4 u = hc_vec8[i];
        float2 f0 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
        float2 f1 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
        float2 f2 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&u.z));
        float2 f3 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&u.w));
        sum_sq += f0.x * f0.x + f0.y * f0.y + f1.x * f1.x + f1.y * f1.y +
                  f2.x * f2.x + f2.y * f2.y + f3.x * f3.x + f3.y * f3.y;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

    __shared__ float s_sq[32];
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) s_sq[warp] = sum_sq;
    __syncthreads();

    float b_sq = (lane < (blockDim.x / 32)) ? s_sq[lane] : 0.0f;
    if (warp == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            b_sq += __shfl_down_sync(0xffffffff, b_sq, offset);
        if (lane == 0) s_sq[0] = rsqrtf(b_sq / (float)hc_dim + eps);
    }
    __syncthreads();
    float rsqrt_val = s_sq[0];

    // Step 2: Compute dot product with row of hc_fn using 128-bit loads (float4 and uint2)
    const float4* fn_vec4 = reinterpret_cast<const float4*>(hc_fn + row * hc_dim);
    const uint2* hc_vec4 = reinterpret_cast<const uint2*>(hc_state);
    int n_vec4 = hc_dim / 4;
    float sum = 0.0f;
    for (int i = tid; i < n_vec4; i += blockDim.x) {
        float4 fn4 = fn_vec4[i];
        uint2 u = hc_vec4[i];
        float2 f0 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
        float2 f1 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
        sum += fn4.x * (f0.x * rsqrt_val) + fn4.y * (f0.y * rsqrt_val) +
               fn4.z * (f1.x * rsqrt_val) + fn4.w * (f1.y * rsqrt_val);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    __shared__ float s_sum[32];
    if (lane == 0) s_sum[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        float bsum = (lane < (blockDim.x / 32)) ? s_sum[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            bsum += __shfl_down_sync(0xffffffff, bsum, offset);
        if (lane == 0) mixes[row] = bsum;
    }
}

void gemv_hc_pre_norm_cuda(
    float* mixes, const __nv_bfloat16* hc_state,
    const float* hc_fn, int mix_size, int hc_dim, float eps, cudaStream_t stream)
{
    gemv_hc_pre_norm_kernel<<<mix_size, 256, 0, stream>>>(mixes, hc_state, hc_fn, mix_size, hc_dim, eps);
}

__global__ void gemv_int2_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ vec,
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ scale_min,
    int N, int K_packed, int block_size)
{
    int start_row = blockIdx.x * 4;
    int tid = threadIdx.x;
    int blocks = (K_packed * 4) / block_size;

    for (int r = 0; r < 4; r++) {
        int row = start_row + r;
        if (row >= N) break;

        const __nv_bfloat16* row_scales = scale_min + (row * blocks);
        const __nv_bfloat16* row_mins = scale_min + (N * blocks) + (row * blocks);

        __shared__ __nv_bfloat16 s_scale[128];
        __shared__ __nv_bfloat16 s_min[128];
        if (tid < blocks) {
            s_scale[tid] = row_scales[tid];
            s_min[tid] = row_mins[tid];
        }
        __syncthreads();

        float sum = 0.0f;
        const uint32_t* w_vec32 = reinterpret_cast<const uint32_t*>(&weight[row * K_packed]);
        const uint4* a_vec4 = reinterpret_cast<const uint4*>(vec);

        for (int chunk_idx = tid; chunk_idx < K_packed / 4; chunk_idx += blockDim.x) {
            int logical_col = chunk_idx * 16;
            int bc = logical_col / block_size;

            float s = __bfloat162float(s_scale[bc]);
            float m = __bfloat162float(s_min[bc]);

            uint32_t chunk = w_vec32[chunk_idx];
            uint4 a0 = a_vec4[chunk_idx * 2];
            uint4 a1 = a_vec4[chunk_idx * 2 + 1];
            uint32_t a_arr[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            
            float sum_w = 0.0f;
            float sum_a = 0.0f;

            #pragma unroll
            for (int j = 0; j < 8; j++) {
                uint32_t a_val = a_arr[j];
                __nv_bfloat162 bf2 = *reinterpret_cast<__nv_bfloat162*>(&a_val);
                float2 f2 = to_float2_bf162(bf2);

                int byte_idx = j / 2;
                int nibble_idx = j % 2;
                uint8_t b = (chunk >> (byte_idx * 8)) & 0xFF;

                float v0, v1;
                if (nibble_idx == 0) {
                    v0 = (float)(b & 0x03);
                    v1 = (float)((b >> 2) & 0x03);
                } else {
                    v0 = (float)((b >> 4) & 0x03);
                    v1 = (float)((b >> 6) & 0x03);
                }
                sum_w += v0 * f2.x + v1 * f2.y;
                sum_a += f2.x + f2.y;
            }
            sum += sum_w * s + sum_a * m;
        }
        // Warp reduce
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        
        __shared__ float s_sum[32];
        int lane = tid % 32;
        int warp = tid / 32;
        if (lane == 0) s_sum[warp] = sum;
        __syncthreads();
        
        sum = (tid < (blockDim.x / 32)) ? s_sum[lane] : 0.0f;
        if (warp == 0) {
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
                sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) out[row] = __float2bfloat16(sum);
        __syncthreads();
    }
}
void gemv_int2_cuda(__nv_bfloat16* out, const __nv_bfloat16* vec,
                    const uint8_t* weight, const __nv_bfloat16* scale_min,
                    int N, int K_packed, int block_size, cudaStream_t stream) {
    int max_chunks = K_packed / 4;
    int threads = max_chunks < 128 ? max_chunks : 128; // fallback limit
    int blocks_x = (N + 3) / 4;

    gemv_int2_kernel<<<blocks_x, threads, 0, stream>>>(out, vec, weight, scale_min, N, K_packed, block_size);
}

// ════════════════════════════════════════════════════════════════════════════════
//  GEMV INT4 Kernel (Block Size = 32, Symmetric with Zero-point 8)
// ════════════════════════════════════════════════════════════════════════════════
__global__ void gemv_int4_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ vec,
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ scale,
    int N, int K)
{
    int row = blockIdx.x * 2 + (threadIdx.y);
    if (row >= N) return;

    int tid = threadIdx.x; // 0..127
    int K_packed = K / 2;
    int num_blocks = K / 32;

    const __nv_bfloat16* row_scales = scale + (row * num_blocks);
    const uint32_t* w_vec32 = reinterpret_cast<const uint32_t*>(&weight[row * K_packed]);
    const uint4* a_vec4 = reinterpret_cast<const uint4*>(vec);

    float sum = 0.0f;

    for (int chunk_idx = tid; chunk_idx < K_packed / 4; chunk_idx += blockDim.x) {
        int logical_col = chunk_idx * 8;
        int bc = logical_col / 32;

        float s = __bfloat162float(row_scales[bc]);

        uint32_t chunk = w_vec32[chunk_idx];
        uint4 a_val4 = a_vec4[chunk_idx];

        uint32_t a_arr[4] = {a_val4.x, a_val4.y, a_val4.z, a_val4.w};
        
        float chunk_sum = 0.0f;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t aval = a_arr[j];
            __nv_bfloat162 bf2 = *reinterpret_cast<__nv_bfloat162*>(&aval);
            float2 f2 = to_float2_bf162(bf2);

            uint8_t b = (chunk >> (j * 8)) & 0xFF;
            float v0 = (float)(b & 0x0F) - 8.0f;
            float v1 = (float)((b >> 4) & 0x0F) - 8.0f;

            chunk_sum += v0 * f2.x + v1 * f2.y;
        }
        sum += chunk_sum * s;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    __shared__ float s_sum[2][4];
    int lane = tid % 32;
    int warp = tid / 32;
    if (lane == 0) s_sum[threadIdx.y][warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = (lane < 4) ? s_sum[threadIdx.y][lane] : 0.0f;
        #pragma unroll
        for (int offset = 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset);

        if (lane == 0) {
            out[row] = __float2bfloat16(sum);
        }
    }
}

void gemv_int4_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* vec,
    const uint8_t* weight,
    const __nv_bfloat16* scale,
    int N, int K,
    cudaStream_t stream)
{
    dim3 threads(128, 2);
    dim3 blocks((N + 1) / 2);
    gemv_int4_kernel<<<blocks, threads, 0, stream>>>(out, vec, weight, scale, N, K);
}

// ── IQ2_XXS Lookup Tables & CUDA Kernels ─────────────────────────────────────────
__device__ __constant__ static const uint8_t c_kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

__device__ __constant__ static const uint8_t c_ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

__device__ __constant__ static const uint64_t c_iq2xxs_grid[256] = {
    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL,
    0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x08080808082b0808ULL,
    0x08080808082b082bULL, 0x08080808082b2b08ULL, 0x08080808082b2b2bULL, 0x0808080819080819ULL,
    0x0808080819081908ULL, 0x0808080819190808ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL,
    0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b082b2bULL,
    0x080808082b2b082bULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x0808081908190808ULL,
    0x0808081908191919ULL, 0x0808081919080808ULL, 0x080808192b081908ULL, 0x080808192b192b08ULL,
    0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b082b082bULL, 0x0808082b2b08082bULL,
    0x0808190808080819ULL, 0x0808190808081908ULL, 0x0808190808190808ULL, 0x08081908082b0819ULL,
    0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819082b08ULL,
    0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL,
    0x080819082b2b1908ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908082b08ULL,
    0x08081919082b0808ULL, 0x080819191908192bULL, 0x08081919192b2b19ULL, 0x080819192b080808ULL,
    0x080819192b190819ULL, 0x0808192b08082b19ULL, 0x0808192b08190808ULL, 0x0808192b19080808ULL,
    0x0808192b2b081908ULL, 0x0808192b2b2b1908ULL, 0x08082b0808080808ULL, 0x08082b0808081919ULL,
    0x08082b0808082b08ULL, 0x08082b0808191908ULL, 0x08082b08082b2b08ULL, 0x08082b0819080819ULL,
    0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b081919082bULL, 0x08082b082b082b08ULL,
    0x08082b1908081908ULL, 0x08082b1919080808ULL, 0x08082b2b0808082bULL, 0x08082b2b08191908ULL,
    0x0819080808080819ULL, 0x0819080808081908ULL, 0x0819080808190808ULL, 0x08190808082b0819ULL,
    0x0819080819080808ULL, 0x08190808192b0808ULL, 0x081908082b081908ULL, 0x081908082b190808ULL,
    0x081908082b191919ULL, 0x0819081908080808ULL, 0x0819081908082b08ULL, 0x08190819082b0808ULL,
    0x0819081919190808ULL, 0x0819081919192b2bULL, 0x081908192b080808ULL, 0x0819082b082b1908ULL,
    0x0819082b19081919ULL, 0x0819190808080808ULL, 0x0819190808082b08ULL, 0x08191908082b0808ULL,
    0x08191908082b1919ULL, 0x0819190819082b19ULL, 0x081919082b080808ULL, 0x0819191908192b08ULL,
    0x08191919192b082bULL, 0x0819192b08080808ULL, 0x0819192b0819192bULL, 0x08192b0808080819ULL,
    0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b0819080808ULL, 0x08192b082b080819ULL,
    0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b192b2b0808ULL, 0x08192b2b19190819ULL,
    0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808082b2bULL, 0x082b080819081908ULL,
    0x082b0808192b0819ULL, 0x082b08082b080808ULL, 0x082b08082b08082bULL, 0x082b0819082b2b19ULL,
    0x082b081919082b08ULL, 0x082b082b08080808ULL, 0x082b082b0808082bULL, 0x082b190808080819ULL,
    0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b190819080808ULL, 0x082b19081919192bULL,
    0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b1919192b1908ULL, 0x082b192b2b190808ULL,
    0x082b2b0808082b08ULL, 0x082b2b08082b0808ULL, 0x082b2b082b191908ULL, 0x082b2b2b19081908ULL,
    0x1908080808080819ULL, 0x1908080808081908ULL, 0x1908080808190808ULL, 0x1908080808192b08ULL,
    0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL, 0x1908080819082b08ULL,
    0x190808081919192bULL, 0x19080808192b0808ULL, 0x190808082b080819ULL, 0x190808082b081908ULL,
    0x190808082b190808ULL, 0x1908081908080808ULL, 0x19080819082b0808ULL, 0x19080819192b0819ULL,
    0x190808192b080808ULL, 0x190808192b081919ULL, 0x1908082b08080819ULL, 0x1908082b08190808ULL,
    0x1908082b19082b08ULL, 0x1908082b1919192bULL, 0x1908082b192b2b08ULL, 0x1908190808080808ULL,
    0x1908190808082b08ULL, 0x19081908082b0808ULL, 0x190819082b080808ULL, 0x190819082b192b19ULL,
    0x190819190819082bULL, 0x19081919082b1908ULL, 0x1908192b08080808ULL, 0x19082b0808080819ULL,
    0x19082b0808081908ULL, 0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL,
    0x19082b1908080808ULL, 0x19082b1919192b08ULL, 0x19082b19192b0819ULL, 0x19082b192b08082bULL,
    0x19082b2b19081919ULL, 0x19082b2b2b190808ULL, 0x1919080808080808ULL, 0x1919080808082b08ULL,
    0x1919080808190819ULL, 0x1919080808192b19ULL, 0x19190808082b0808ULL, 0x191908082b080808ULL,
    0x191908082b082b08ULL, 0x1919081908081908ULL, 0x191908191908082bULL, 0x191908192b2b1908ULL,
    0x1919082b2b190819ULL, 0x191919082b190808ULL, 0x191919082b19082bULL, 0x1919191908082b2bULL,
    0x1919192b08080819ULL, 0x1919192b19191908ULL, 0x19192b0808080808ULL, 0x19192b0808190819ULL,
    0x19192b0808192b19ULL, 0x19192b08192b1908ULL, 0x19192b1919080808ULL, 0x19192b2b08082b08ULL,
    0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b0808192b2b08ULL,
    0x192b081908080808ULL, 0x192b081919191919ULL, 0x192b082b08192b08ULL, 0x192b082b192b0808ULL,
    0x192b190808080808ULL, 0x192b190808081919ULL, 0x192b191908190808ULL, 0x192b19190819082bULL,
    0x192b19192b081908ULL, 0x192b2b081908082bULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL,
    0x2b08080808082b2bULL, 0x2b08080819080819ULL, 0x2b0808082b08082bULL, 0x2b08081908081908ULL,
    0x2b08081908192b08ULL, 0x2b08081919080808ULL, 0x2b08082b08190819ULL, 0x2b08190808080819ULL,
    0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL,
    0x2b081908192b0808ULL, 0x2b08191908080808ULL, 0x2b0819191908192bULL, 0x2b0819192b191908ULL,
    0x2b08192b08082b19ULL, 0x2b08192b19080808ULL, 0x2b08192b192b0808ULL, 0x2b082b080808082bULL,
    0x2b082b1908081908ULL, 0x2b082b2b08190819ULL, 0x2b19080808081908ULL, 0x2b19080808190808ULL,
    0x2b190808082b1908ULL, 0x2b19080819080808ULL, 0x2b1908082b2b0819ULL, 0x2b1908190819192bULL,
    0x2b1908192b080808ULL, 0x2b19082b19081919ULL, 0x2b19190808080808ULL, 0x2b191908082b082bULL,
    0x2b19190819081908ULL, 0x2b19191919190819ULL, 0x2b192b082b080819ULL, 0x2b192b19082b0808ULL,
    0x2b2b08080808082bULL, 0x2b2b080819190808ULL, 0x2b2b08082b081919ULL, 0x2b2b081908082b19ULL,
    0x2b2b082b08080808ULL, 0x2b2b190808192b08ULL, 0x2b2b2b0819190808ULL, 0x2b2b2b1908081908ULL,
};

__global__ void gemv_iq2_xxs_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ vec,
    const block_iq2_xxs* __restrict__ weight,
    int N, int K)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;

    int lane = threadIdx.x;
    int n_blocks = K / 256;
    const block_iq2_xxs* row_w = weight + row * n_blocks;

    int l = lane / 8;
    int j = lane % 8;
    uint8_t kmask = 1 << j;

    float sum = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        const block_iq2_xxs* blk = &row_w[b];
        float d = __half2float(blk->d);
        const uint16_t* qs = blk->qs;

        #pragma unroll 4
        for (int ib32 = 0; ib32 < 8; ib32++) {
            uint32_t q0 = qs[ib32 * 4 + 0];
            uint32_t q1 = qs[ib32 * 4 + 1];
            uint32_t q2 = qs[ib32 * 4 + 2];
            uint32_t q3 = qs[ib32 * 4 + 3];

            uint32_t aux0 = q0 | (q1 << 16);
            uint32_t aux1 = q2 | (q3 << 16);
            float db = d * (0.5f + (aux1 >> 28)) * 0.25f;

            uint8_t grid_idx = (aux0 >> (8 * l)) & 0xFF;
            uint8_t sign_idx = (aux1 >> (7 * l)) & 0x7F;

            uint64_t grid_val = c_iq2xxs_grid[grid_idx];
            uint8_t sign_val = c_ksigns_iq2xs[sign_idx];

            uint8_t byte_val = (grid_val >> (8 * j)) & 0xFF;
            float sign_mult = (sign_val & kmask) ? -1.0f : 1.0f;
            float w = db * (float)byte_val * sign_mult;

            int col_idx = (b << 8) + (ib32 << 5) + lane;
            float a = __bfloat162float(vec[col_idx]);
            sum += w * a;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        out[row] = __float2bfloat16(sum);
    }
}

void gemv_iq2_xxs_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* vec,
    const block_iq2_xxs* weight,
    int N, int K,
    cudaStream_t stream)
{
    dim3 threads(32, 8);
    dim3 blocks((N + 7) / 8);
    gemv_iq2_xxs_kernel<<<blocks, threads, 0, stream>>>(out, vec, weight, N, K);
}

__global__ void iq2_xxs_dequant_kernel(
    __nv_bfloat16* __restrict__ out,
    const block_iq2_xxs* __restrict__ weight,
    int N, int K)
{
    int row = blockIdx.y;
    int col_blk = blockIdx.x * blockDim.x + threadIdx.x;
    int n_blocks = K / 256;
    if (row >= N || col_blk >= n_blocks) return;

    const block_iq2_xxs* blk = &weight[row * n_blocks + col_blk];
    float d = __half2float(blk->d);
    __nv_bfloat16* out_row = out + row * K + col_blk * 256;

    for (int ib32 = 0; ib32 < 8; ib32++) {
        uint32_t aux0 = (uint32_t)blk->qs[ib32 * 4 + 0] | ((uint32_t)blk->qs[ib32 * 4 + 1] << 16);
        uint32_t aux1 = (uint32_t)blk->qs[ib32 * 4 + 2] | ((uint32_t)blk->qs[ib32 * 4 + 3] << 16);
        float db = d * (0.5f + (aux1 >> 28)) * 0.25f;

        for (int l = 0; l < 4; l++) {
            uint8_t grid_idx = (aux0 >> (8 * l)) & 0xFF;
            uint8_t sign_idx = (aux1 >> (7 * l)) & 0x7F;
            uint64_t grid_val = c_iq2xxs_grid[grid_idx];
            uint8_t sign_val = c_ksigns_iq2xs[sign_idx];

            for (int j = 0; j < 8; j++) {
                uint8_t byte_val = (grid_val >> (8 * j)) & 0xFF;
                float sign_mult = (sign_val & c_kmask_iq2xs[j]) ? -1.0f : 1.0f;
                float w = db * (float)byte_val * sign_mult;
                out_row[ib32 * 32 + l * 8 + j] = __float2bfloat16(w);
            }
        }
    }
}

void iq2_xxs_dequant_cuda(
    __nv_bfloat16* out,
    const block_iq2_xxs* weight,
    int rows, int cols,
    cudaStream_t stream)
{
    int n_blocks = cols / 256;
    dim3 threads(64, 1);
    dim3 blocks((n_blocks + 63) / 64, rows);
    iq2_xxs_dequant_kernel<<<blocks, threads, 0, stream>>>(out, weight, rows, cols);
}

// ── Q2_K CUDA Kernels ────────────────────────────────────────────────────────────
__global__ void gemv_q2_k_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ vec,
    const block_q2_K* __restrict__ weight,
    int N, int K)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;

    int lane = threadIdx.x; // 0..31 (one warp per row)
    int n_blocks = K / 256;
    const block_q2_K* row_w = weight + row * n_blocks;

    float sum = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        const block_q2_K* blk = &row_w[b];
        float d = __half2float(blk->d);
        float min = __half2float(blk->dmin);

        #pragma unroll 4
        for (int iter = 0; iter < 8; iter++) {
            int idx = (iter << 5) + lane; // 0..255
            int group = idx >> 4;
            int l = idx & 15;
            int q_base = ((group >> 3) << 5) + ((group & 1) << 4);
            int shift = ((group >> 1) & 3) << 1;
            uint8_t q = (blk->qs[q_base + l] >> shift) & 0x03;
            uint8_t sc = blk->scales[group];
            float dl = d * (float)(sc & 0x0F);
            float ml = min * (float)(sc >> 4);
            float w = dl * (float)q - ml;

            int col_idx = (b << 8) + idx;
            float a = __bfloat162float(vec[col_idx]);
            sum += w * a;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        out[row] = __float2bfloat16(sum);
    }
}

void gemv_q2_k_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* vec,
    const block_q2_K* weight,
    int N, int K,
    cudaStream_t stream)
{
    dim3 threads(32, 8);
    dim3 blocks((N + 7) / 8);
    gemv_q2_k_kernel<<<blocks, threads, 0, stream>>>(out, vec, weight, N, K);
}

__global__ void q2_k_dequant_kernel(
    __nv_bfloat16* __restrict__ out,
    const block_q2_K* __restrict__ weight,
    int N, int K)
{
    int row = blockIdx.y;
    int col_blk = blockIdx.x * blockDim.x + threadIdx.x;
    int n_blocks = K / 256;
    if (row >= N || col_blk >= n_blocks) return;

    const block_q2_K* blk = &weight[row * n_blocks + col_blk];
    float d = __half2float(blk->d);
    float min = __half2float(blk->dmin);
    __nv_bfloat16* out_row = out + row * K + col_blk * 256;

    for (int group = 0; group < 16; group++) {
        uint8_t sc = blk->scales[group];
        float dl = d * (float)(sc & 0x0F);
        float ml = min * (float)(sc >> 4);
        int q_base = 32 * (group / 8) + 16 * (group & 1);
        int shift = ((group / 2) & 3) * 2;

        for (int l = 0; l < 16; l++) {
            uint8_t q = (blk->qs[q_base + l] >> shift) & 0x03;
            float w = dl * (float)q - ml;
            out_row[group * 16 + l] = __float2bfloat16(w);
        }
    }
}

void q2_k_dequant_cuda(
    __nv_bfloat16* out,
    const block_q2_K* weight,
    int rows, int cols,
    cudaStream_t stream)
{
    int n_blocks = cols / 256;
    dim3 threads(64, 1);
    dim3 blocks((n_blocks + 63) / 64, rows);
    q2_k_dequant_kernel<<<blocks, threads, 0, stream>>>(out, weight, rows, cols);
}

// ── HC Pre Weighted Add Kernel ──────────────────────────────────────────────────
__global__ void hc_pre_weighted_add_kernel(
    __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ hc_state,
    const float* __restrict__ pre_weights,
    int dim, int hc)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 8 >= dim) return;

    if (hc == 4) {
        float w0 = pre_weights[0];
        float w1 = pre_weights[1];
        float w2 = pre_weights[2];
        float w3 = pre_weights[3];

        const uint4* s0_ptr = reinterpret_cast<const uint4*>(hc_state + 0 * dim);
        const uint4* s1_ptr = reinterpret_cast<const uint4*>(hc_state + 1 * dim);
        const uint4* s2_ptr = reinterpret_cast<const uint4*>(hc_state + 2 * dim);
        const uint4* s3_ptr = reinterpret_cast<const uint4*>(hc_state + 3 * dim);

        uint4 u0 = s0_ptr[idx];
        uint4 u1 = s1_ptr[idx];
        uint4 u2 = s2_ptr[idx];
        uint4 u3 = s3_ptr[idx];

        float2 f0_0 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&u0.x));
        float2 f0_1 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&u0.y));
        float2 f0_2 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&u0.z));
        float2 f0_3 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&u0.w));

        float2 f1_0 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&u1.x));
        float2 f1_1 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&u1.y));
        float2 f1_2 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&u1.z));
        float2 f1_3 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&u1.w));

        float2 f2_0 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&u2.x));
        float2 f2_1 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&u2.y));
        float2 f2_2 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&u2.z));
        float2 f2_3 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&u2.w));

        float2 f3_0 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&u3.x));
        float2 f3_1 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&u3.y));
        float2 f3_2 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&u3.z));
        float2 f3_3 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&u3.w));

        __nv_bfloat162 res0 = to_bf162_float2(make_float2(
            w0 * f0_0.x + w1 * f1_0.x + w2 * f2_0.x + w3 * f3_0.x,
            w0 * f0_0.y + w1 * f1_0.y + w2 * f2_0.y + w3 * f3_0.y
        ));
        __nv_bfloat162 res1 = to_bf162_float2(make_float2(
            w0 * f0_1.x + w1 * f1_1.x + w2 * f2_1.x + w3 * f3_1.x,
            w0 * f0_1.y + w1 * f1_1.y + w2 * f2_1.y + w3 * f3_1.y
        ));
        __nv_bfloat162 res2 = to_bf162_float2(make_float2(
            w0 * f0_2.x + w1 * f1_2.x + w2 * f2_2.x + w3 * f3_2.x,
            w0 * f0_2.y + w1 * f1_2.y + w2 * f2_2.y + w3 * f3_2.y
        ));
        __nv_bfloat162 res3 = to_bf162_float2(make_float2(
            w0 * f0_3.x + w1 * f1_3.x + w2 * f2_3.x + w3 * f3_3.x,
            w0 * f0_3.y + w1 * f1_3.y + w2 * f2_3.y + w3 * f3_3.y
        ));

        uint4 out_u;
        *reinterpret_cast<__nv_bfloat162*>(&out_u.x) = res0;
        *reinterpret_cast<__nv_bfloat162*>(&out_u.y) = res1;
        *reinterpret_cast<__nv_bfloat162*>(&out_u.z) = res2;
        *reinterpret_cast<__nv_bfloat162*>(&out_u.w) = res3;

        reinterpret_cast<uint4*>(hidden)[idx] = out_u;
    }
}

void hc_pre_weighted_add_cuda(
    __nv_bfloat16* hidden, const __nv_bfloat16* hc_state, const float* pre_weights,
    int dim, int hc, cudaStream_t stream)
{
    int threads = 128;
    int blocks = (dim / 8 + threads - 1) / threads;
    hc_pre_weighted_add_kernel<<<blocks, threads, 0, stream>>>(hidden, hc_state, pre_weights, dim, hc);
}

// ── HC Post Update Kernel (Optimized 1D grid) ──────────────────────────────────
__global__ void hc_post_update_kernel(
    __nv_bfloat16* __restrict__ hc_state,
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ hc_residual,
    const float* __restrict__ post_weights,
    const float* __restrict__ comb_weights,
    int dim, int hc)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 8 >= dim) return;

    if (hc == 4) {
        float p0 = post_weights[0];
        float p1 = post_weights[1];
        float p2 = post_weights[2];
        float p3 = post_weights[3];

        float c00 = comb_weights[0 * 4 + 0], c01 = comb_weights[0 * 4 + 1], c02 = comb_weights[0 * 4 + 2], c03 = comb_weights[0 * 4 + 3];
        float c10 = comb_weights[1 * 4 + 0], c11 = comb_weights[1 * 4 + 1], c12 = comb_weights[1 * 4 + 2], c13 = comb_weights[1 * 4 + 3];
        float c20 = comb_weights[2 * 4 + 0], c21 = comb_weights[2 * 4 + 1], c22 = comb_weights[2 * 4 + 2], c23 = comb_weights[2 * 4 + 3];
        float c30 = comb_weights[3 * 4 + 0], c31 = comb_weights[3 * 4 + 1], c32 = comb_weights[3 * 4 + 2], c33 = comb_weights[3 * 4 + 3];

        uint4 uh = reinterpret_cast<const uint4*>(hidden)[idx];
        uint4 ur0 = reinterpret_cast<const uint4*>(hc_residual + 0 * dim)[idx];
        uint4 ur1 = reinterpret_cast<const uint4*>(hc_residual + 1 * dim)[idx];
        uint4 ur2 = reinterpret_cast<const uint4*>(hc_residual + 2 * dim)[idx];
        uint4 ur3 = reinterpret_cast<const uint4*>(hc_residual + 3 * dim)[idx];

        float2 fh_0 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&uh.x));
        float2 fh_1 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&uh.y));
        float2 fh_2 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&uh.z));
        float2 fh_3 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&uh.w));

        float2 fr0_0 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&ur0.x));
        float2 fr0_1 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&ur0.y));
        float2 fr0_2 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&ur0.z));
        float2 fr0_3 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&ur0.w));

        float2 fr1_0 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&ur1.x));
        float2 fr1_1 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&ur1.y));
        float2 fr1_2 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&ur1.z));
        float2 fr1_3 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&ur1.w));

        float2 fr2_0 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&ur2.x));
        float2 fr2_1 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&ur2.y));
        float2 fr2_2 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&ur2.z));
        float2 fr2_3 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&ur2.w));

        float2 fr3_0 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&ur3.x));
        float2 fr3_1 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&ur3.y));
        float2 fr3_2 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&ur3.z));
        float2 fr3_3 = to_float2_bf162(*reinterpret_cast<const __nv_bfloat162*>(&ur3.w));

        // Row 0
        uint4 out0;
        *reinterpret_cast<__nv_bfloat162*>(&out0.x) = to_bf162_float2(make_float2(p0 * fh_0.x + c00 * fr0_0.x + c10 * fr1_0.x + c20 * fr2_0.x + c30 * fr3_0.x, p0 * fh_0.y + c00 * fr0_0.y + c10 * fr1_0.y + c20 * fr2_0.y + c30 * fr3_0.y));
        *reinterpret_cast<__nv_bfloat162*>(&out0.y) = to_bf162_float2(make_float2(p0 * fh_1.x + c00 * fr0_1.x + c10 * fr1_1.x + c20 * fr2_1.x + c30 * fr3_1.x, p0 * fh_1.y + c00 * fr0_1.y + c10 * fr1_1.y + c20 * fr2_1.y + c30 * fr3_1.y));
        *reinterpret_cast<__nv_bfloat162*>(&out0.z) = to_bf162_float2(make_float2(p0 * fh_2.x + c00 * fr0_2.x + c10 * fr1_2.x + c20 * fr2_2.x + c30 * fr3_2.x, p0 * fh_2.y + c00 * fr0_2.y + c10 * fr1_2.y + c20 * fr2_2.y + c30 * fr3_2.y));
        *reinterpret_cast<__nv_bfloat162*>(&out0.w) = to_bf162_float2(make_float2(p0 * fh_3.x + c00 * fr0_3.x + c10 * fr1_3.x + c20 * fr2_3.x + c30 * fr3_3.x, p0 * fh_3.y + c00 * fr0_3.y + c10 * fr1_3.y + c20 * fr2_3.y + c30 * fr3_3.y));

        // Row 1
        uint4 out1;
        *reinterpret_cast<__nv_bfloat162*>(&out1.x) = to_bf162_float2(make_float2(p1 * fh_0.x + c01 * fr0_0.x + c11 * fr1_0.x + c21 * fr2_0.x + c31 * fr3_0.x, p1 * fh_0.y + c01 * fr0_0.y + c11 * fr1_0.y + c21 * fr2_0.y + c31 * fr3_0.y));
        *reinterpret_cast<__nv_bfloat162*>(&out1.y) = to_bf162_float2(make_float2(p1 * fh_1.x + c01 * fr0_1.x + c11 * fr1_1.x + c21 * fr2_1.x + c31 * fr3_1.x, p1 * fh_1.y + c01 * fr0_1.y + c11 * fr1_1.y + c21 * fr2_1.y + c31 * fr3_1.y));
        *reinterpret_cast<__nv_bfloat162*>(&out1.z) = to_bf162_float2(make_float2(p1 * fh_2.x + c01 * fr0_2.x + c11 * fr1_2.x + c21 * fr2_2.x + c31 * fr3_2.x, p1 * fh_2.y + c01 * fr0_2.y + c11 * fr1_2.y + c21 * fr2_2.y + c31 * fr3_2.y));
        *reinterpret_cast<__nv_bfloat162*>(&out1.w) = to_bf162_float2(make_float2(p1 * fh_3.x + c01 * fr0_3.x + c11 * fr1_3.x + c21 * fr2_3.x + c31 * fr3_3.x, p1 * fh_3.y + c01 * fr0_3.y + c11 * fr1_3.y + c21 * fr2_3.y + c31 * fr3_3.y));

        // Row 2
        uint4 out2;
        *reinterpret_cast<__nv_bfloat162*>(&out2.x) = to_bf162_float2(make_float2(p2 * fh_0.x + c02 * fr0_0.x + c12 * fr1_0.x + c22 * fr2_0.x + c32 * fr3_0.x, p2 * fh_0.y + c02 * fr0_0.y + c12 * fr1_0.y + c22 * fr2_0.y + c32 * fr3_0.y));
        *reinterpret_cast<__nv_bfloat162*>(&out2.y) = to_bf162_float2(make_float2(p2 * fh_1.x + c02 * fr0_1.x + c12 * fr1_1.x + c22 * fr2_1.x + c32 * fr3_1.x, p2 * fh_1.y + c02 * fr0_1.y + c12 * fr1_1.y + c22 * fr2_1.y + c32 * fr3_1.y));
        *reinterpret_cast<__nv_bfloat162*>(&out2.z) = to_bf162_float2(make_float2(p2 * fh_2.x + c02 * fr0_2.x + c12 * fr1_2.x + c22 * fr2_2.x + c32 * fr3_2.x, p2 * fh_2.y + c02 * fr0_2.y + c12 * fr1_2.y + c22 * fr2_2.y + c32 * fr3_2.y));
        *reinterpret_cast<__nv_bfloat162*>(&out2.w) = to_bf162_float2(make_float2(p2 * fh_3.x + c02 * fr0_3.x + c12 * fr1_3.x + c22 * fr2_3.x + c32 * fr3_3.x, p2 * fh_3.y + c02 * fr0_3.y + c12 * fr1_3.y + c22 * fr2_3.y + c32 * fr3_3.y));

        // Row 3
        uint4 out3;
        *reinterpret_cast<__nv_bfloat162*>(&out3.x) = to_bf162_float2(make_float2(p3 * fh_0.x + c03 * fr0_0.x + c13 * fr1_0.x + c23 * fr2_0.x + c33 * fr3_0.x, p3 * fh_0.y + c03 * fr0_0.y + c13 * fr1_0.y + c23 * fr2_0.y + c33 * fr3_0.y));
        *reinterpret_cast<__nv_bfloat162*>(&out3.y) = to_bf162_float2(make_float2(p3 * fh_1.x + c03 * fr0_1.x + c13 * fr1_1.x + c23 * fr2_1.x + c33 * fr3_1.x, p3 * fh_1.y + c03 * fr0_1.y + c13 * fr1_1.y + c23 * fr2_1.y + c33 * fr3_1.y));
        *reinterpret_cast<__nv_bfloat162*>(&out3.z) = to_bf162_float2(make_float2(p3 * fh_2.x + c03 * fr0_2.x + c13 * fr1_2.x + c23 * fr2_2.x + c33 * fr3_2.x, p3 * fh_2.y + c03 * fr0_2.y + c13 * fr1_2.y + c23 * fr2_2.y + c33 * fr3_2.y));
        *reinterpret_cast<__nv_bfloat162*>(&out3.w) = to_bf162_float2(make_float2(p3 * fh_3.x + c03 * fr0_3.x + c13 * fr1_3.x + c23 * fr2_3.x + c33 * fr3_3.x, p3 * fh_3.y + c03 * fr0_3.y + c13 * fr1_3.y + c23 * fr2_3.y + c33 * fr3_3.y));

        reinterpret_cast<uint4*>(hc_state + 0 * dim)[idx] = out0;
        reinterpret_cast<uint4*>(hc_state + 1 * dim)[idx] = out1;
        reinterpret_cast<uint4*>(hc_state + 2 * dim)[idx] = out2;
        reinterpret_cast<uint4*>(hc_state + 3 * dim)[idx] = out3;
    }
}

void hc_post_update_cuda(
    __nv_bfloat16* hc_state, const __nv_bfloat16* hidden, const __nv_bfloat16* hc_residual,
    const float* post_weights, const float* comb_weights,
    int dim, int hc, cudaStream_t stream)
{
    int threads = 128;
    int blocks = (dim / 8 + threads - 1) / threads;
    hc_post_update_kernel<<<blocks, threads, 0, stream>>>(
        hc_state, hidden, hc_residual, post_weights, comb_weights, dim, hc);
}

// ── HC Head Reduce Kernel ───────────────────────────────────────────────────────
__global__ void hc_head_reduce_kernel(
    __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ hc_state,
    const float* __restrict__ mixes,
    const float* __restrict__ scale,
    const float* __restrict__ base,
    int dim, int hc)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dim) return;

    float s = scale[0];
    float sum = 0.0f;
    for (int h = 0; h < hc; h++) {
        float mix = mixes[h];
        float b = base[h];
        // sigmoid(mix * s + b) + eps
        float arg = mix * s + b;
        float w = 1.0f / (1.0f + expf(-arg)) + 1e-6f;
        
        float val = __bfloat162float(hc_state[h * dim + tid]);
        sum += w * val;
    }
    hidden[tid] = __float2bfloat16(sum);
}

void hc_head_reduce_cuda(
    __nv_bfloat16* hidden, const __nv_bfloat16* hc_state,
    const float* mixes, const float* scale, const float* base,
    int dim, int hc, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    hc_head_reduce_kernel<<<blocks, threads, 0, stream>>>(
        hidden, hc_state, mixes, scale, base, dim, hc);
}
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void rms_norm_f32_kernel(float* x, int dim, float eps) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = x[i];
        sum_sq += v * v;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    if (lane == 0) sdata[warp] = sum_sq;
    __syncthreads();

    if (warp == 0) {
        sum_sq = (tid < (blockDim.x / 32)) ? sdata[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
        }
        if (tid == 0) {
            sdata[0] = rsqrtf(sum_sq / dim + eps);
        }
    }
    __syncthreads();

    float rsqrt_val = sdata[0];
    for (int i = tid; i < dim; i += blockDim.x) {
        x[i] *= rsqrt_val;
    }
}

void rms_norm_f32_cuda(float* x, int dim, float eps, cudaStream_t stream) {
    int threads = min(dim, 1024);
    int smem = (threads / 32) * sizeof(float);
    rms_norm_f32_kernel<<<1, threads, smem, stream>>>(x, dim, eps);
}

__global__ void rms_norm_weighted_f32_kernel(float* __restrict__ out, const float* __restrict__ x, const __nv_bfloat16* __restrict__ weight, int dim, float eps) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = x[i];
        sum_sq += v * v;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    if (lane == 0) sdata[warp] = sum_sq;
    __syncthreads();

    if (warp == 0) {
        sum_sq = (tid < (blockDim.x / 32)) ? sdata[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
        }
        if (tid == 0) {
            sdata[0] = rsqrtf(sum_sq / dim + eps);
        }
    }
    __syncthreads();

    float rsqrt_val = sdata[0];
    for (int i = tid; i < dim; i += blockDim.x) {
        float w = weight ? __bfloat162float(weight[i]) : 1.0f;
        out[i] = x[i] * rsqrt_val * w;
    }
}

void rms_norm_weighted_f32_cuda(float* out, const float* x, const __nv_bfloat16* weight, int dim, float eps, cudaStream_t stream) {
    int threads = min(dim, 1024);
    int smem = (threads / 32) * sizeof(float);
    rms_norm_weighted_f32_kernel<<<1, threads, smem, stream>>>(out, x, weight, dim, eps);
}

// ── Fused SwiGLU GEMV Kernel: w1(x) and w3(x) simultaneously + SiLU-Mul in registers ───
__global__ void gemv_iq2_xxs_swiglu_fused_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ vec,
    const block_iq2_xxs* __restrict__ w1,
    const block_iq2_xxs* __restrict__ w3,
    int N, int K, float swiglu_limit)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;

    int lane = threadIdx.x;
    int n_blocks = K / 256;
    const block_iq2_xxs* row_w1 = w1 + row * n_blocks;
    const block_iq2_xxs* row_w3 = w3 + row * n_blocks;

    int l = lane / 8;
    int j = lane % 8;
    uint8_t kmask = 1 << j;

    float sum1 = 0.0f;
    float sum3 = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        const block_iq2_xxs* blk1 = &row_w1[b];
        const block_iq2_xxs* blk3 = &row_w3[b];
        float d1 = __half2float(blk1->d);
        float d3 = __half2float(blk3->d);

        const uint16_t* qs1 = blk1->qs;
        const uint16_t* qs3 = blk3->qs;

        #pragma unroll 4
        for (int ib32 = 0; ib32 < 8; ib32++) {
            int q_off = ib32 * 4;
            uint32_t aux0_1 = (uint32_t)qs1[q_off + 0] | ((uint32_t)qs1[q_off + 1] << 16);
            uint32_t aux1_1 = (uint32_t)qs1[q_off + 2] | ((uint32_t)qs1[q_off + 3] << 16);
            float db1 = d1 * (0.5f + (aux1_1 >> 28)) * 0.25f;

            uint32_t aux0_3 = (uint32_t)qs3[q_off + 0] | ((uint32_t)qs3[q_off + 1] << 16);
            uint32_t aux1_3 = (uint32_t)qs3[q_off + 2] | ((uint32_t)qs3[q_off + 3] << 16);
            float db3 = d3 * (0.5f + (aux1_3 >> 28)) * 0.25f;

            uint8_t g_idx1 = (aux0_1 >> (8 * l)) & 0xFF;
            uint8_t s_idx1 = (aux1_1 >> (7 * l)) & 0x7F;
            uint64_t g_val1 = c_iq2xxs_grid[g_idx1];
            uint8_t s_val1 = c_ksigns_iq2xs[s_idx1];
            uint8_t byte1 = (g_val1 >> (8 * j)) & 0xFF;
            float sign1 = (s_val1 & kmask) ? -1.0f : 1.0f;
            float weight1 = db1 * (float)byte1 * sign1;

            uint8_t g_idx3 = (aux0_3 >> (8 * l)) & 0xFF;
            uint8_t s_idx3 = (aux1_3 >> (7 * l)) & 0x7F;
            uint64_t g_val3 = c_iq2xxs_grid[g_idx3];
            uint8_t s_val3 = c_ksigns_iq2xs[s_idx3];
            uint8_t byte3 = (g_val3 >> (8 * j)) & 0xFF;
            float sign3 = (s_val3 & kmask) ? -1.0f : 1.0f;
            float weight3 = db3 * (float)byte3 * sign3;

            int col_idx = (b << 8) + (ib32 << 5) + lane;
            float a = __bfloat162float(vec[col_idx]);
            sum1 += weight1 * a;
            sum3 += weight3 * a;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum1 += __shfl_down_sync(0xffffffff, sum1, offset);
        sum3 += __shfl_down_sync(0xffffffff, sum3, offset);
    }

    if (lane == 0) {
        float g = sum1;
        float u = sum3;
        if (swiglu_limit > 0.0f) {
            g = fminf(g, swiglu_limit);
            u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
        }
        float silu_g = g / (1.0f + expf(-g));
        out[row] = __float2bfloat16(silu_g * u);
    }
}

void gemv_iq2_xxs_swiglu_fused_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* vec,
    const block_iq2_xxs* w1,
    const block_iq2_xxs* w3,
    int N, int K, float swiglu_limit,
    cudaStream_t stream)
{
    dim3 threads(32, 8);
    dim3 blocks((N + 7) / 8);
    gemv_iq2_xxs_swiglu_fused_kernel<<<blocks, threads, 0, stream>>>(out, vec, w1, w3, N, K, swiglu_limit);
}

// ── Lightweight F32 GEMV Kernel for Hyper-Connections ─────────────────────────
__global__ void gemv_f32_kernel(
    float* __restrict__ out,
    const float* __restrict__ vec,
    const float* __restrict__ matrix,
    int M, int K)
{
    int row = blockIdx.x;
    if (row >= M) return;

    int tid = threadIdx.x;
    const float* row_ptr = matrix + row * K;
    float sum = 0.0f;

    for (int i = tid; i < K; i += blockDim.x) {
        sum += row_ptr[i] * vec[i];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float warp_sums[8];
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) {
        warp_sums[warp] = sum;
    }
    __syncthreads();

    if (warp == 0) {
        float bsum = (lane < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            bsum += __shfl_down_sync(0xffffffff, bsum, offset);
        }
        if (lane == 0) {
            out[row] = bsum;
        }
    }
}

void gemv_f32_cuda(float* out, const float* vec, const float* matrix, int M, int K, cudaStream_t stream) {
    gemv_f32_kernel<<<M, 256, 0, stream>>>(out, vec, matrix, M, K);
}

// ── Fused 6-way MoE accumulation ───────────────────────────────────────────────
__global__ void fused_moe_accum_6_kernel(
    __nv_bfloat16* __restrict__ accum,
    const __nv_bfloat16* __restrict__ down_ptrs, // [6 * dim]
    float w0, float w1, float w2, float w3, float w4, float w5,
    int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;

    float sum = 0.0f;
    sum += __bfloat162float(down_ptrs[0 * dim + idx]) * w0;
    sum += __bfloat162float(down_ptrs[1 * dim + idx]) * w1;
    sum += __bfloat162float(down_ptrs[2 * dim + idx]) * w2;
    sum += __bfloat162float(down_ptrs[3 * dim + idx]) * w3;
    sum += __bfloat162float(down_ptrs[4 * dim + idx]) * w4;
    sum += __bfloat162float(down_ptrs[5 * dim + idx]) * w5;

    accum[idx] = __float2bfloat16(sum);
}

void fused_moe_accum_6_cuda(
    __nv_bfloat16* accum,
    const __nv_bfloat16* down_ptrs,
    float w0, float w1, float w2, float w3, float w4, float w5,
    int dim, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    fused_moe_accum_6_kernel<<<blocks, threads, 0, stream>>>(
        accum, down_ptrs, w0, w1, w2, w3, w4, w5, dim);
}

// ── Fused 6-way MoE dynamic accumulation ─────────────────────────────────────
__global__ void fused_moe_accum_dynamic_kernel(
    __nv_bfloat16* __restrict__ accum,
    const __nv_bfloat16* __restrict__ down_buf, // [6 * dim]
    const float* __restrict__ topk_weights,     // [6]
    const __nv_bfloat16* __restrict__ shared_down, // [dim]
    int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;

    float sum = shared_down ? __bfloat162float(shared_down[idx]) : 0.0f;
    #pragma unroll
    for (int k = 0; k < 6; k++) {
        float w = topk_weights[k];
        sum += __bfloat162float(down_buf[k * dim + idx]) * w;
    }
    accum[idx] = __float2bfloat16(sum);
}

void fused_moe_accum_dynamic_cuda(
    __nv_bfloat16* accum,
    const __nv_bfloat16* down_buf,
    const float* topk_weights,
    const __nv_bfloat16* shared_down,
    int dim,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    fused_moe_accum_dynamic_kernel<<<blocks, threads, 0, stream>>>(accum, down_buf, topk_weights, shared_down, dim);
}

// ── GPU-Native MoE Top-6 Routing Kernel ───────────────────────────────────────
__global__ void moe_route_top6_from_bf16_kernel(
    int32_t* __restrict__ topk_ids,
    float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ scores_bf16,
    const float* __restrict__ gate_bias,
    int n_experts, int top_k, float routed_scaling_factor)
{
    __shared__ float s_probs[256];
    __shared__ float s_scores[256];
    __shared__ int   s_indices[256];

    int tid = threadIdx.x;
    if (tid < n_experts) {
        float raw = __bfloat162float(scores_bf16[tid]);
        float sp = (raw > 20.0f) ? raw : ((raw < -20.0f) ? expf(raw) : log1pf(expf(raw)));
        float prob = sqrtf(sp);
        s_probs[tid] = prob;
        s_scores[tid] = prob + (gate_bias ? gate_bias[tid] : 0.0f);
        s_indices[tid] = tid;
    } else {
        s_probs[tid] = 0.0f;
        s_scores[tid] = -1e38f;
        s_indices[tid] = -1;
    }
    __syncthreads();

    if (tid == 0) {
        for (int i = 0; i < top_k; i++) {
            int max_idx = i;
            float max_val = s_scores[i];
            for (int j = i + 1; j < n_experts; j++) {
                if (s_scores[j] > max_val) {
                    max_val = s_scores[j];
                    max_idx = j;
                }
            }
            float tmp_s = s_scores[i]; s_scores[i] = s_scores[max_idx]; s_scores[max_idx] = tmp_s;
            float tmp_p = s_probs[i];  s_probs[i]  = s_probs[max_idx];  s_probs[max_idx]  = tmp_p;
            int   tmp_id = s_indices[i]; s_indices[i] = s_indices[max_idx]; s_indices[max_idx] = tmp_id;
        }

        float weight_sum = 0.0f;
        for (int k = 0; k < top_k; k++) {
            weight_sum += s_probs[k];
        }
        if (weight_sum < 6.103515625e-5f) weight_sum = 6.103515625e-5f;

        for (int k = 0; k < top_k; k++) {
            topk_ids[k] = s_indices[k];
            topk_weights[k] = (s_probs[k] / weight_sum) * routed_scaling_factor;
        }
    }
}

void moe_route_top6_from_bf16_cuda(
    int32_t* topk_ids,
    float* topk_weights,
    const __nv_bfloat16* scores_bf16,
    const float* gate_bias,
    int n_experts, int top_k, float routed_scaling_factor,
    cudaStream_t stream)
{
    moe_route_top6_from_bf16_kernel<<<1, 256, 0, stream>>>(
        topk_ids, topk_weights, scores_bf16, gate_bias, n_experts, top_k, routed_scaling_factor);
}

__global__ void moe_route_top6_kernel(
    int32_t* __restrict__ topk_ids,
    float* __restrict__ topk_weights,
    const float* __restrict__ scores_f32,
    int n_experts, int top_k, float routed_scaling_factor)
{
    __shared__ float s_scores[256];
    __shared__ int   s_indices[256];

    int tid = threadIdx.x;
    if (tid < n_experts) {
        s_scores[tid] = scores_f32[tid];
        s_indices[tid] = tid;
    } else {
        s_scores[tid] = -1e38f;
        s_indices[tid] = -1;
    }
    __syncthreads();

    if (tid == 0) {
        for (int i = 0; i < top_k; i++) {
            int max_idx = i;
            float max_val = s_scores[i];
            for (int j = i + 1; j < n_experts; j++) {
                if (s_scores[j] > max_val) {
                    max_val = s_scores[j];
                    max_idx = j;
                }
            }
            float tmp_s = s_scores[i]; s_scores[i] = s_scores[max_idx]; s_scores[max_idx] = tmp_s;
            int   tmp_id = s_indices[i]; s_indices[i] = s_indices[max_idx]; s_indices[max_idx] = tmp_id;
        }

        float weight_sum = 0.0f;
        for (int k = 0; k < top_k; k++) {
            weight_sum += s_scores[k];
        }
        if (weight_sum < 6.103515625e-5f) weight_sum = 6.103515625e-5f;

        for (int k = 0; k < top_k; k++) {
            topk_ids[k] = s_indices[k];
            topk_weights[k] = (s_scores[k] / weight_sum) * routed_scaling_factor;
        }
    }
}

void moe_route_top6_cuda(
    int32_t* topk_ids,
    float* topk_weights,
    const float* scores_f32,
    int n_experts, int top_k, float routed_scaling_factor,
    cudaStream_t stream)
{
    moe_route_top6_kernel<<<1, 256, 0, stream>>>(
        topk_ids, topk_weights, scores_f32, n_experts, top_k, routed_scaling_factor);
}

// ── GPU-Native MoE Hash Routing Kernel (Layers 0..2) ─────────────────────────
__global__ void moe_route_hash_kernel(
    int32_t* __restrict__ topk_ids,
    float* __restrict__ topk_weights,
    const int64_t* __restrict__ tid2eid_table,
    int token_id, int top_k, float routed_scaling_factor)
{
    int tid = threadIdx.x;
    if (tid == 0) {
        const int64_t* eid_ptr = tid2eid_table ? (tid2eid_table + (size_t)token_id * top_k) : nullptr;
        float w = routed_scaling_factor / (float)top_k;
        for (int k = 0; k < top_k; k++) {
            topk_ids[k] = eid_ptr ? (int32_t)eid_ptr[k] : -1;
            topk_weights[k] = w;
        }
    }
}

void moe_route_hash_cuda(
    int32_t* topk_ids,
    float* topk_weights,
    const int64_t* tid2eid_table,
    int token_id, int top_k, float routed_scaling_factor,
    cudaStream_t stream)
{
    moe_route_hash_kernel<<<1, 32, 0, stream>>>(
        topk_ids, topk_weights, tid2eid_table,
        token_id, top_k, routed_scaling_factor);
}

__global__ void moe_route_hash_device_id_kernel(
    int32_t* __restrict__ topk_ids,
    float* __restrict__ topk_weights,
    const int64_t* __restrict__ tid2eid_table,
    const int32_t* __restrict__ d_token_id, int top_k, float routed_scaling_factor)
{
    int tid = threadIdx.x;
    if (tid == 0) {
        int token_id = *d_token_id;
        const int64_t* eid_ptr = tid2eid_table ? (tid2eid_table + (size_t)token_id * top_k) : nullptr;
        float w = routed_scaling_factor / (float)top_k;
        for (int k = 0; k < top_k; k++) {
            topk_ids[k] = eid_ptr ? (int32_t)eid_ptr[k] : -1;
            topk_weights[k] = w;
        }
    }
}

void moe_route_hash_device_id_cuda(
    int32_t* topk_ids,
    float* topk_weights,
    const int64_t* tid2eid_table,
    const int32_t* d_token_id, int top_k, float routed_scaling_factor,
    cudaStream_t stream)
{
    moe_route_hash_device_id_kernel<<<1, 32, 0, stream>>>(
        topk_ids, topk_weights, tid2eid_table,
        d_token_id, top_k, routed_scaling_factor);
}

// ── Populate Active Expert Pointers Kernel ────────────────────────────────────
__global__ void populate_active_expert_ptrs_kernel(
    const void** __restrict__ active_ptrs,
    const int32_t* __restrict__ topk_ids,
    const void* const* __restrict__ flat_expert_ptrs,
    int layer_id, int n_experts, int top_k)
{
    int k = threadIdx.x;
    if (k < top_k) {
        int eid = topk_ids[k];
        if (eid >= 0 && eid < n_experts && flat_expert_ptrs) {
            active_ptrs[k] = flat_expert_ptrs[layer_id * n_experts + eid];
        } else {
            active_ptrs[k] = nullptr;
        }
    }
}

void populate_active_expert_ptrs_cuda(
    const void** active_ptrs,
    const int32_t* topk_ids,
    const void* const* flat_expert_ptrs,
    int layer_id, int n_experts, int top_k,
    cudaStream_t stream)
{
    populate_active_expert_ptrs_kernel<<<1, top_k, 0, stream>>>(
        active_ptrs, topk_ids, flat_expert_ptrs, layer_id, n_experts, top_k);
}

// ── Batched All-6-Experts IQ2_XXS SwiGLU Fused Kernel ────────────────────────
__global__ void gemv_iq2_xxs_moe_swiglu_fused_kernel(
    __nv_bfloat16* __restrict__ gate_buf, // [6 * moe_inter]
    const __nv_bfloat16* __restrict__ vec,
    const void* const* __restrict__ active_expert_ptrs,
    const int32_t* __restrict__ topk_ids,
    const void* const* __restrict__ flat_expert_ptrs,
    int layer_id, int n_experts,
    int w1_offset, int w3_offset,
    int N, int K, float swiglu_limit)
{
    __shared__ uint64_t s_grid[256];
    __shared__ uint8_t s_signs[128];
    __shared__ __nv_bfloat16 s_vec[2][256]; // Double-buffered vector cache (1 KB)

    int tid = threadIdx.y * 32 + threadIdx.x;
    s_grid[tid] = c_iq2xxs_grid[tid];
    if (tid < 128) s_signs[tid] = c_ksigns_iq2xs[tid];

    // Pipeline prologue: prefetch first 256 elements of vec into s_vec[0]
    if (tid < 32) {
        cp_async_16_bytes((uint4*)s_vec[0] + tid, (const uint4*)vec + tid);
    }
    cp_async_commit_group();
    cp_async_wait_all();
    __syncthreads();

    int k = blockIdx.y;
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N || k >= 6) return;

    const void* p = nullptr;
    if (flat_expert_ptrs && topk_ids) {
        int eid = topk_ids[k];
        if (eid >= 0 && eid < n_experts) {
            p = flat_expert_ptrs[layer_id * n_experts + eid];
        }
    } else if (active_expert_ptrs) {
        p = active_expert_ptrs[k];
    }
    if (!p) return;

    const uint8_t* block = (const uint8_t*)p;
    const block_iq2_xxs* w1 = (const block_iq2_xxs*)(block + w1_offset);
    const block_iq2_xxs* w3 = (const block_iq2_xxs*)(block + w3_offset);

    int lane = threadIdx.x;
    int n_blocks = K / 256;
    const block_iq2_xxs* row_w1 = w1 + row * n_blocks;
    const block_iq2_xxs* row_w3 = w3 + row * n_blocks;

    int l = lane / 8;
    int j = lane % 8;
    uint8_t kmask = 1 << j;

    float sum1 = 0.0f;
    float sum3 = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        int curr_stage = b & 1;
        int next_stage = (b + 1) & 1;

        // Asynchronously prefetch next tile of vec during computation
        if (b + 1 < n_blocks && tid < 32) {
            cp_async_16_bytes((uint4*)s_vec[next_stage] + tid,
                              (const uint4*)(vec + ((b + 1) << 8)) + tid);
            cp_async_commit_group();
        }

        const block_iq2_xxs* blk1 = &row_w1[b];
        const block_iq2_xxs* blk3 = &row_w3[b];

        float d1 = (lane == 0) ? __half2float(blk1->d) : 0.0f;
        float d3 = (lane == 0) ? __half2float(blk3->d) : 0.0f;
        d1 = __shfl_sync(0xffffffff, d1, 0);
        d3 = __shfl_sync(0xffffffff, d3, 0);

        // Coalesced 32-thread register load for entire 32-element qs arrays
        uint16_t q1_lane = blk1->qs[lane];
        uint16_t q3_lane = blk3->qs[lane];

        #pragma unroll 4
        for (int ib32 = 0; ib32 < 8; ib32++) {
            int q_off = ib32 * 4;
            uint32_t aux0_1, aux1_1, aux0_3, aux1_3;

            // Broadcast aux descriptors from lane registers via shuffle
            uint16_t q1_0 = __shfl_sync(0xffffffff, q1_lane, q_off + 0);
            uint16_t q1_1 = __shfl_sync(0xffffffff, q1_lane, q_off + 1);
            uint16_t q1_2 = __shfl_sync(0xffffffff, q1_lane, q_off + 2);
            uint16_t q1_3 = __shfl_sync(0xffffffff, q1_lane, q_off + 3);

            uint16_t q3_0 = __shfl_sync(0xffffffff, q3_lane, q_off + 0);
            uint16_t q3_1 = __shfl_sync(0xffffffff, q3_lane, q_off + 1);
            uint16_t q3_2 = __shfl_sync(0xffffffff, q3_lane, q_off + 2);
            uint16_t q3_3 = __shfl_sync(0xffffffff, q3_lane, q_off + 3);

            aux0_1 = (uint32_t)q1_0 | ((uint32_t)q1_1 << 16);
            aux1_1 = (uint32_t)q1_2 | ((uint32_t)q1_3 << 16);
            aux0_3 = (uint32_t)q3_0 | ((uint32_t)q3_1 << 16);
            aux1_3 = (uint32_t)q3_2 | ((uint32_t)q3_3 << 16);

            float db1 = d1 * (0.5f + (aux1_1 >> 28)) * 0.25f;
            float db3 = d3 * (0.5f + (aux1_3 >> 28)) * 0.25f;

            uint8_t g_idx1 = (aux0_1 >> (8 * l)) & 0xFF;
            uint8_t s_idx1 = (aux1_1 >> (7 * l)) & 0x7F;
            uint64_t g_val1 = s_grid[g_idx1];
            uint8_t s_val1 = s_signs[s_idx1];
            uint8_t byte1 = (g_val1 >> (8 * j)) & 0xFF;
            float sign1 = (s_val1 & kmask) ? -1.0f : 1.0f;
            float weight1 = db1 * (float)byte1 * sign1;

            uint8_t g_idx3 = (aux0_3 >> (8 * l)) & 0xFF;
            uint8_t s_idx3 = (aux1_3 >> (7 * l)) & 0x7F;
            uint64_t g_val3 = s_grid[g_idx3];
            uint8_t s_val3 = s_signs[s_idx3];
            uint8_t byte3 = (g_val3 >> (8 * j)) & 0xFF;
            float sign3 = (s_val3 & kmask) ? -1.0f : 1.0f;
            float weight3 = db3 * (float)byte3 * sign3;

            int col_in_block = (ib32 << 5) + lane;
            float a = __bfloat162float(s_vec[curr_stage][col_in_block]);
            sum1 += weight1 * a;
            sum3 += weight3 * a;
        }

        if (b + 1 < n_blocks) {
            cp_async_wait_all();
            __syncthreads();
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum1 += __shfl_down_sync(0xffffffff, sum1, offset);
        sum3 += __shfl_down_sync(0xffffffff, sum3, offset);
    }

    if (lane == 0) {
        float g = sum1;
        float u = sum3;
        if (swiglu_limit > 0.0f) {
            g = fminf(g, swiglu_limit);
            u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
        }
        float silu_g = g / (1.0f + expf(-g));
        gate_buf[k * N + row] = __float2bfloat16(silu_g * u);
    }
}

void gemv_iq2_xxs_moe_swiglu_fused_cuda(
    __nv_bfloat16* gate_buf,
    const __nv_bfloat16* vec,
    const void* const* active_expert_ptrs,
    int w1_offset, int w3_offset,
    int N, int K, float swiglu_limit,
    const int32_t* topk_ids,
    const void* const* flat_expert_ptrs,
    int layer_id, int n_experts,
    cudaStream_t stream)
{
    dim3 threads(32, 8);
    dim3 blocks((N + 7) / 8, 6);
    gemv_iq2_xxs_moe_swiglu_fused_kernel<<<blocks, threads, 0, stream>>>(
        gate_buf, vec, active_expert_ptrs,
        topk_ids, flat_expert_ptrs, layer_id, n_experts,
        w1_offset, w3_offset, N, K, swiglu_limit);
}

// ── Batched All-6-Experts Q2_K w2 Down Projection Kernel ───────────────────────
__global__ void gemv_q2_k_moe_kernel(
    __nv_bfloat16* __restrict__ down_buf, // [6 * dim]
    const __nv_bfloat16* __restrict__ gate_buf, // [6 * moe_inter]
    const void* const* __restrict__ active_expert_ptrs,
    const int32_t* __restrict__ topk_ids,
    const void* const* __restrict__ flat_expert_ptrs,
    int layer_id, int n_experts,
    int w2_offset,
    int N, int K)
{
    __shared__ __nv_bfloat16 s_gate[2][256]; // Double-buffered gate/up input cache (1 KB)

    int tid = threadIdx.y * 32 + threadIdx.x;
    int k = blockIdx.y;
    const __nv_bfloat16* k_gate = gate_buf + (size_t)k * K;

    // Pipeline prologue: prefetch first 256 elements of k_gate into s_gate[0]
    if (tid < 32) {
        cp_async_16_bytes((uint4*)s_gate[0] + tid, (const uint4*)k_gate + tid);
    }
    cp_async_commit_group();
    cp_async_wait_all();
    __syncthreads();

    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N || k >= 6) return;

    const void* p = nullptr;
    if (flat_expert_ptrs && topk_ids) {
        int eid = topk_ids[k];
        if (eid >= 0 && eid < n_experts) {
            p = flat_expert_ptrs[layer_id * n_experts + eid];
        }
    } else if (active_expert_ptrs) {
        p = active_expert_ptrs[k];
    }
    if (!p) return;

    const uint8_t* block = (const uint8_t*)p;
    const block_q2_K* w2 = (const block_q2_K*)(block + w2_offset);

    int lane = threadIdx.x;
    int n_blocks = K / 256;
    const block_q2_K* row_w2 = w2 + row * n_blocks;

    int lane_half = lane >> 4;
    float sum = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        int curr_stage = b & 1;
        int next_stage = (b + 1) & 1;

        // Asynchronously prefetch next tile of gate input
        if (b + 1 < n_blocks && tid < 32) {
            cp_async_16_bytes((uint4*)s_gate[next_stage] + tid,
                              (const uint4*)(k_gate + ((b + 1) << 8)) + tid);
            cp_async_commit_group();
        }

        const block_q2_K* blk = &row_w2[b];
        float d = (lane == 0) ? __half2float(blk->d) : 0.0f;
        float min = (lane == 0) ? __half2float(blk->dmin) : 0.0f;
        d = __shfl_sync(0xffffffff, d, 0);
        min = __shfl_sync(0xffffffff, min, 0);

        uint8_t sc_val = (lane < 16) ? blk->scales[lane] : 0;

        uint8_t q0 = blk->qs[lane];
        uint8_t q1 = blk->qs[32 + lane];

        #pragma unroll 4
        for (int iter = 0; iter < 4; iter++) {
            uint8_t q = (q0 >> (iter * 2)) & 0x03;
            uint8_t sc = (uint8_t)__shfl_sync(0xffffffff, sc_val, (iter * 2) + lane_half);
            float dl = d * (float)(sc & 0x0F);
            float ml = min * (float)(sc >> 4);
            float w = dl * (float)q - ml;

            int col_in_block = (iter << 5) + lane;
            float a = __bfloat162float(s_gate[curr_stage][col_in_block]);
            sum += w * a;
        }

        #pragma unroll 4
        for (int iter = 0; iter < 4; iter++) {
            uint8_t q = (q1 >> (iter * 2)) & 0x03;
            uint8_t sc = (uint8_t)__shfl_sync(0xffffffff, sc_val, 8 + (iter * 2) + lane_half);
            float dl = d * (float)(sc & 0x0F);
            float ml = min * (float)(sc >> 4);
            float w = dl * (float)q - ml;

            int col_in_block = 128 + (iter << 5) + lane;
            float a = __bfloat162float(s_gate[curr_stage][col_in_block]);
            sum += w * a;
        }

        if (b + 1 < n_blocks) {
            cp_async_wait_all();
            __syncthreads();
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        down_buf[(size_t)k * N + row] = __float2bfloat16(sum);
    }
}

void gemv_q2_k_moe_cuda(
    __nv_bfloat16* down_buf,
    const __nv_bfloat16* gate_buf,
    const void* const* active_expert_ptrs,
    int w2_offset,
    int N, int K,
    const int32_t* topk_ids,
    const void* const* flat_expert_ptrs,
    int layer_id, int n_experts,
    cudaStream_t stream)
{
    dim3 threads(32, 8);
    dim3 blocks((N + 7) / 8, 6);
    gemv_q2_k_moe_kernel<<<blocks, threads, 0, stream>>>(
        down_buf, gate_buf, active_expert_ptrs,
        topk_ids, flat_expert_ptrs, layer_id, n_experts,
        w2_offset, N, K);
}

// ════════════════════════════════════════════════════════════════════════════════
//  Device-Driven Kernels for CUDA Graph Capture
// ════════════════════════════════════════════════════════════════════════════════

__global__ void embedding_broadcast_device_id_kernel(
    __nv_bfloat16* __restrict__ hidden,
    __nv_bfloat16* __restrict__ hc_state,
    const __nv_bfloat16* __restrict__ table,
    const int32_t* __restrict__ d_token_id, int dim, int hc)
{
    int d = threadIdx.x + blockIdx.x * blockDim.x;
    if (d >= dim) return;

    int token_id = *d_token_id;
    __nv_bfloat16 val = table[(size_t)token_id * dim + d];
    hidden[d] = val;
    for (int h = 0; h < hc; h++) {
        hc_state[(size_t)h * dim + d] = val;
    }
}

void embedding_broadcast_device_id_cuda(
    __nv_bfloat16* hidden, __nv_bfloat16* hc_state,
    const __nv_bfloat16* table, const int32_t* d_token_id, int dim, int hc,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    embedding_broadcast_device_id_kernel<<<blocks, threads, 0, stream>>>(
        hidden, hc_state, table, d_token_id, dim, hc);
}

__global__ void rope_device_pos_kernel(
    __nv_bfloat16* __restrict__ x,
    int n_vectors, int head_dim, int rope_dim,
    const int32_t* __restrict__ d_position,
    const float* __restrict__ freq_table,
    bool inverse)
{
    int vec_id = blockIdx.x;
    int pair_id = threadIdx.x;

    if (vec_id >= n_vectors || pair_id >= rope_dim / 2) return;

    int position = *d_position;
    int half_rope = rope_dim / 2;
    int base_idx = vec_id * head_dim + (head_dim - rope_dim) + 2 * pair_id;

    float x0 = bf16_to_float(x[base_idx]);
    float x1 = bf16_to_float(x[base_idx + 1]);

    float cos_val = freq_table[position * half_rope * 2 + pair_id * 2];
    float sin_val = freq_table[position * half_rope * 2 + pair_id * 2 + 1];

    if (inverse) sin_val = -sin_val;

    float y0 = x0 * cos_val - x1 * sin_val;
    float y1 = x0 * sin_val + x1 * cos_val;

    x[base_idx]     = float_to_bf16(y0);
    x[base_idx + 1] = float_to_bf16(y1);
}

void rope_device_pos_cuda(
    __nv_bfloat16* x, int n_vectors, int head_dim, int rope_dim,
    const int32_t* d_position, const float* freq_table, bool inverse,
    cudaStream_t stream)
{
    int pairs = rope_dim / 2;
    rope_device_pos_kernel<<<n_vectors, pairs, 0, stream>>>(
        x, n_vectors, head_dim, rope_dim, d_position, freq_table, inverse);
}

__global__ void store_kv_device_pos_kernel(
    __nv_bfloat16* __restrict__ kv_cache,
    const __nv_bfloat16* __restrict__ kv_val,
    const int32_t* __restrict__ d_position,
    int window, int head_dim)
{
    int d = threadIdx.x + blockIdx.x * blockDim.x;
    if (d >= head_dim) return;

    int position = *d_position;
    int cache_pos = position % window;
    kv_cache[(size_t)cache_pos * head_dim + d] = kv_val[d];
}

void store_kv_device_pos_cuda(
    __nv_bfloat16* kv_cache, const __nv_bfloat16* kv_val,
    const int32_t* d_position, int window, int head_dim,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (head_dim + threads - 1) / threads;
    store_kv_device_pos_kernel<<<blocks, threads, 0, stream>>>(
        kv_cache, kv_val, d_position, window, head_dim);
}

__global__ void prepare_combined_kv_kernel(
    __nv_bfloat16* __restrict__ combined_kv,
    int32_t* __restrict__ d_cache_len,
    const __nv_bfloat16* __restrict__ raw_kv_cache,
    const __nv_bfloat16* __restrict__ comp_kv_cache,
    const int32_t* __restrict__ d_position,
    const int32_t* __restrict__ d_comp_count,
    int window, int head_dim, int ratio)
{
    int position = *d_position;
    int raw_entries = (position + 1 < window) ? (position + 1) : window;
    int comp_entries = (ratio > 0 && d_comp_count != nullptr) ? *d_comp_count : 0;
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_cache_len = raw_entries + comp_entries;
    }

    int cache_pos = position % window;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_raw_elements = raw_entries * head_dim;

    if (position + 1 > window) {
        int after_pos = (cache_pos + 1) % window;
        int tail = window - after_pos;
        int tail_elements = tail * head_dim;

        for (int i = tid; i < total_raw_elements; i += blockDim.x * gridDim.x) {
            if (i < tail_elements) {
                combined_kv[i] = raw_kv_cache[(size_t)after_pos * head_dim + i];
            } else {
                int head_part_idx = i - tail_elements;
                combined_kv[i] = raw_kv_cache[head_part_idx];
            }
        }
    } else {
        for (int i = tid; i < total_raw_elements; i += blockDim.x * gridDim.x) {
            combined_kv[i] = raw_kv_cache[i];
        }
    }

    if (comp_entries > 0 && comp_kv_cache != nullptr) {
        int comp_elements = comp_entries * head_dim;
        for (int i = tid; i < comp_elements; i += blockDim.x * gridDim.x) {
            combined_kv[(size_t)total_raw_elements + i] = comp_kv_cache[i];
        }
    }
}

void prepare_combined_kv_cuda(
    __nv_bfloat16* combined_kv,
    int32_t* d_cache_len,
    const __nv_bfloat16* raw_kv_cache,
    const __nv_bfloat16* comp_kv_cache,
    const int32_t* d_position,
    const int32_t* d_comp_count,
    int window, int head_dim, int ratio,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = 64;
    prepare_combined_kv_kernel<<<blocks, threads, 0, stream>>>(
        combined_kv, d_cache_len, raw_kv_cache, comp_kv_cache,
        d_position, d_comp_count, window, head_dim, ratio);
}

__global__ void mla_attention_device_len_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ kv,
    const float* __restrict__ attn_sink,
    __nv_bfloat16* __restrict__ out,
    const int32_t* __restrict__ d_cache_len,
    int max_cache_len,
    int head_dim,
    float scale)
{
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int n_threads = blockDim.x;

    // Dynamically allocated shared memory for scores
    extern __shared__ float s_mem[];
    float* scores = s_mem;

    // Load Q for this head into shared memory
    __shared__ float s_q[512];
    for (int d = tid; d < head_dim; d += n_threads) {
        s_q[d] = __bfloat162float(q[h * head_dim + d]);
    }
    __syncthreads();

    int cache_len = (d_cache_len != nullptr) ? *d_cache_len : max_cache_len;
    if (cache_len > max_cache_len) cache_len = max_cache_len;
    if (cache_len < 1) cache_len = 1;

    // 1. Q @ KV.T
    for (int t = tid; t < cache_len; t += n_threads) {
        const __nv_bfloat16* kv_t = kv + (size_t)t * head_dim;
        float dot = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < head_dim; d++) {
            dot += s_q[d] * __bfloat162float(kv_t[d]);
        }
        scores[t] = dot * scale;
    }
    __syncthreads();

    // 2. Softmax
    float local_max = -1e38f;
    for (int t = tid; t < cache_len; t += n_threads) {
        local_max = fmaxf(local_max, scores[t]);
    }
    if (attn_sink != nullptr) {
        local_max = fmaxf(local_max, attn_sink[h]);
    }

    // Warp-level max reduction
    float warp_max = local_max;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        warp_max = fmaxf(warp_max, __shfl_down_sync(0xffffffff, warp_max, offset));

    __shared__ float s_red[32];
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) s_red[warp] = warp_max;
    __syncthreads();

    float block_max = (tid < (n_threads >> 5)) ? s_red[tid] : -1e38f;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        block_max = fmaxf(block_max, __shfl_down_sync(0xffffffff, block_max, offset));
    if (tid == 0) s_red[0] = block_max;
    __syncthreads();
    block_max = s_red[0];

    // Exp and sum
    float local_sum = 0.0f;
    for (int t = tid; t < cache_len; t += n_threads) {
        float e = expf(scores[t] - block_max);
        scores[t] = e;
        local_sum += e;
    }
    if (tid == 0 && attn_sink != nullptr) {
        local_sum += expf(attn_sink[h] - block_max);
    }

    // Warp-level sum reduction
    float warp_sum = local_sum;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
    if (lane == 0) s_red[warp] = warp_sum;
    __syncthreads();

    float block_sum = (tid < (n_threads >> 5)) ? s_red[tid] : 0.0f;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
    if (tid == 0) s_red[0] = block_sum;
    __syncthreads();
    float inv_sum = 1.0f / s_red[0];

    // Pre-normalize scores in shared memory
    for (int t = tid; t < cache_len; t += n_threads) {
        scores[t] *= inv_sum;
    }
    __syncthreads();

    // 3. Normalize and compute out = scores @ KV
    for (int d = tid; d < head_dim; d += n_threads) {
        float v = 0.0f;
        for (int t = 0; t < cache_len; t++) {
            v += scores[t] * __bfloat162float(kv[(size_t)t * head_dim + d]);
        }
        out[(size_t)h * head_dim + d] = __float2bfloat16(v);
    }
}

void mla_attention_device_len_cuda(
    const __nv_bfloat16* q,
    const __nv_bfloat16* kv,
    const float* attn_sink,
    __nv_bfloat16* out,
    const int32_t* d_cache_len,
    int max_cache_len,
    int head_dim,
    float scale,
    cudaStream_t stream)
{
    int n_threads = 256;
    size_t smem_bytes = max_cache_len * sizeof(float);
    if (smem_bytes > 49152) {
        static bool s_smem_device_configured = false;
        if (!s_smem_device_configured) {
            cudaFuncSetAttribute(mla_attention_device_len_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
            s_smem_device_configured = true;
        }
    }
    int n_heads = 64;
    mla_attention_device_len_kernel<<<n_heads, n_threads, smem_bytes, stream>>>(
        q, kv, attn_sink, out, d_cache_len, max_cache_len, head_dim, scale);
}

// ════════════════════════════════════════════════════════════════════════════════
//  Fused Flash-MLA Attention Kernel
//  Fuses:
//   1. Raw Q Load + Unweighted RMSNorm (in SM shared memory)
//   2. Forward RoPE (in SM shared memory)
//   3. Q @ KV Attention Dot Products + Scaling
//   4. Online Flash Softmax (Max + Exp + Sum Reductions + Attention Sink)
//   5. Weighted Value Reduction (scores @ KV) into SM shared memory
//   6. Inverse RoPE Rotation on output (in SM shared memory) + Output Store
// ════════════════════════════════════════════════════════════════════════════════

__global__ void mla_attention_fused_kernel(
    const __nv_bfloat16* __restrict__ raw_q,
    const __nv_bfloat16* __restrict__ kv,
    const float* __restrict__ attn_sink,
    __nv_bfloat16* __restrict__ out,
    const int32_t* __restrict__ d_cache_len,
    const int32_t* __restrict__ d_position,
    const float* __restrict__ freq_table,
    int max_cache_len,
    int head_dim,
    int rope_dim,
    float scale,
    float q_norm_eps)
{
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int n_threads = blockDim.x;

    extern __shared__ float s_mem[];
    float* scores = s_mem;

    __shared__ float s_q[512];
    __shared__ float s_out[512];
    __shared__ float s_red[32];

    // ── Step 1: Load raw Q & compute unweighted RMSNorm in shared memory ──────
    float local_sum_sq = 0.0f;
    for (int d = tid; d < head_dim; d += n_threads) {
        float val = bf16_to_float(raw_q[h * head_dim + d]);
        s_q[d] = val;
        local_sum_sq += val * val;
    }

    // Warp-level sum reduction for RMSNorm
    float warp_sum_sq = local_sum_sq;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        warp_sum_sq += __shfl_down_sync(0xffffffff, warp_sum_sq, offset);
    }
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) s_red[warp] = warp_sum_sq;
    __syncthreads();

    float block_sum_sq = (tid < (n_threads >> 5)) ? s_red[tid] : 0.0f;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        block_sum_sq += __shfl_down_sync(0xffffffff, block_sum_sq, offset);
    }
    if (tid == 0) s_red[0] = rsqrtf(block_sum_sq / (float)head_dim + q_norm_eps);
    __syncthreads();

    float rsqrt_val = s_red[0];
    for (int d = tid; d < head_dim; d += n_threads) {
        s_q[d] *= rsqrt_val;
    }
    __syncthreads();

    // ── Step 2: Apply forward RoPE rotation directly in s_q ───────────────────
    int pos = (d_position != nullptr) ? *d_position : 0;
    int half_rope = rope_dim / 2;
    if (tid < half_rope) {
        int pair_id = tid;
        int base_idx = (head_dim - rope_dim) + 2 * pair_id;
        float x0 = s_q[base_idx];
        float x1 = s_q[base_idx + 1];

        float cos_val = freq_table[pos * half_rope * 2 + pair_id * 2];
        float sin_val = freq_table[pos * half_rope * 2 + pair_id * 2 + 1];

        s_q[base_idx]     = x0 * cos_val - x1 * sin_val;
        s_q[base_idx + 1] = x0 * sin_val + x1 * cos_val;
    }
    __syncthreads();

    // ── Step 3: Attention scores Q @ KV.T ────────────────────────────────────
    int cache_len = (d_cache_len != nullptr) ? *d_cache_len : max_cache_len;
    if (cache_len > max_cache_len) cache_len = max_cache_len;
    if (cache_len < 1) cache_len = 1;

    for (int t = tid; t < cache_len; t += n_threads) {
        const __nv_bfloat16* kv_t = kv + (size_t)t * head_dim;
        float dot = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < head_dim; d++) {
            dot += s_q[d] * bf16_to_float(kv_t[d]);
        }
        scores[t] = dot * scale;
    }
    __syncthreads();

    // ── Step 4: Online Softmax Max + Exp + Sum ───────────────────────────────
    float local_max = -1e38f;
    for (int t = tid; t < cache_len; t += n_threads) {
        local_max = fmaxf(local_max, scores[t]);
    }
    if (attn_sink != nullptr) {
        local_max = fmaxf(local_max, attn_sink[h]);
    }

    float warp_max = local_max;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        warp_max = fmaxf(warp_max, __shfl_down_sync(0xffffffff, warp_max, offset));
    }
    if (lane == 0) s_red[warp] = warp_max;
    __syncthreads();

    float block_max = (tid < (n_threads >> 5)) ? s_red[tid] : -1e38f;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        block_max = fmaxf(block_max, __shfl_down_sync(0xffffffff, block_max, offset));
    }
    if (tid == 0) s_red[0] = block_max;
    __syncthreads();
    block_max = s_red[0];

    float local_sum = 0.0f;
    for (int t = tid; t < cache_len; t += n_threads) {
        float e = expf(scores[t] - block_max);
        scores[t] = e;
        local_sum += e;
    }
    if (tid == 0 && attn_sink != nullptr) {
        local_sum += expf(attn_sink[h] - block_max);
    }

    float warp_sum = local_sum;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
    }
    if (lane == 0) s_red[warp] = warp_sum;
    __syncthreads();

    float block_sum = (tid < (n_threads >> 5)) ? s_red[tid] : 0.0f;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
    }
    if (tid == 0) s_red[0] = 1.0f / block_sum;
    __syncthreads();
    float inv_sum = s_red[0];

    for (int t = tid; t < cache_len; t += n_threads) {
        scores[t] *= inv_sum;
    }
    __syncthreads();

    // ── Step 5: Weighted Value Reduction scores @ KV into s_out ──────────────
    for (int d = tid; d < head_dim; d += n_threads) {
        float v = 0.0f;
        #pragma unroll 4
        for (int t = 0; t < cache_len; t++) {
            v += scores[t] * bf16_to_float(kv[(size_t)t * head_dim + d]);
        }
        s_out[d] = v;
    }
    __syncthreads();

    // ── Step 6: Inverse RoPE Rotation & Output Store ─────────────────────────
    if (tid < half_rope) {
        int pair_id = tid;
        int base_idx = (head_dim - rope_dim) + 2 * pair_id;
        float y0 = s_out[base_idx];
        float y1 = s_out[base_idx + 1];

        float cos_val = freq_table[pos * half_rope * 2 + pair_id * 2];
        float sin_val = -freq_table[pos * half_rope * 2 + pair_id * 2 + 1]; // negative for inverse

        s_out[base_idx]     = y0 * cos_val - y1 * sin_val;
        s_out[base_idx + 1] = y0 * sin_val + y1 * cos_val;
    }
    __syncthreads();

    for (int d = tid; d < head_dim; d += n_threads) {
        out[h * head_dim + d] = float_to_bf16(s_out[d]);
    }
}

void mla_attention_fused_cuda(
    const __nv_bfloat16* raw_q,
    const __nv_bfloat16* kv,
    const float* attn_sink,
    __nv_bfloat16* out,
    const int32_t* d_cache_len,
    const int32_t* d_position,
    const float* freq_table,
    int max_cache_len,
    int head_dim,
    int rope_dim,
    float scale,
    float q_norm_eps,
    cudaStream_t stream)
{
    int n_threads = 256;
    size_t smem_bytes = max_cache_len * sizeof(float);
    if (smem_bytes > 49152) {
        static bool s_smem_fused_configured = false;
        if (!s_smem_fused_configured) {
            cudaFuncSetAttribute(mla_attention_fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
            s_smem_fused_configured = true;
        }
    }
    int n_heads = 64;
    mla_attention_fused_kernel<<<n_heads, n_threads, smem_bytes, stream>>>(
        raw_q, kv, attn_sink, out,
        d_cache_len, d_position, freq_table,
        max_cache_len, head_dim, rope_dim, scale, q_norm_eps);
}


__global__ void accumulate_expert_imatrix_kernel(
    float* __restrict__ gate_accum,
    float* __restrict__ down_accum,
    uint32_t* __restrict__ expert_counts,
    const __nv_bfloat16* __restrict__ h_norm,
    const int32_t* __restrict__ topk_indices,
    int num_tokens,
    int top_k,
    int n_experts,
    int hidden_dim,
    int moe_intermediate)
{
    int token_idx = blockIdx.x;
    int k_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (token_idx >= num_tokens || k_idx >= top_k) return;

    int expert_id = topk_indices[token_idx * top_k + k_idx];
    if (expert_id < 0 || expert_id >= n_experts) return;

    if (tid == 0) {
        atomicAdd(&expert_counts[expert_id], 1);
    }

    const __nv_bfloat16* token_norm = h_norm + (size_t)token_idx * hidden_dim;
    float* exp_gate = gate_accum + (size_t)expert_id * hidden_dim;
    float* exp_down = down_accum + (size_t)expert_id * moe_intermediate;

    // Gate & Up input channels accumulation: sum(x_j^2)
    for (int j = tid; j < hidden_dim; j += blockDim.x) {
        float x = __bfloat162float(token_norm[j]);
        float x_sq = x * x;
        atomicAdd(&exp_gate[j], x_sq);
    }

    // Down intermediate approximation: (|x_j| * sigmoid(x_j))^2
    for (int j = tid; j < moe_intermediate; j += blockDim.x) {
        float x = __bfloat162float(token_norm[j]);
        float sig = 1.0f / (1.0f + expf(-x));
        float d = fabsf(x) * sig;
        float d_sq = d * d;
        atomicAdd(&exp_down[j], d_sq);
    }
}

void accumulate_expert_imatrix_cuda(
    float* gate_accum,
    float* down_accum,
    uint32_t* expert_counts,
    const __nv_bfloat16* h_norm,
    const int32_t* topk_indices,
    int num_tokens,
    int top_k,
    int n_experts,
    int hidden_dim,
    int moe_intermediate,
    cudaStream_t stream)
{
    if (num_tokens <= 0) return;
    dim3 grid(num_tokens, top_k);
    int threads = 256;
    accumulate_expert_imatrix_kernel<<<grid, threads, 0, stream>>>(
        gate_accum, down_accum, expert_counts, h_norm, topk_indices,
        num_tokens, top_k, n_experts, hidden_dim, moe_intermediate);
}

// ════════════════════════════════════════════════════════════════════════════════
//  Qwen 3.8 Gated DeltaNet Linear Attention & Gated GQA Attention
// ════════════════════════════════════════════════════════════════════════════════

__global__ void deltanet_conv_kernel(
    __nv_bfloat16* __restrict__ conv_out,
    const __nv_bfloat16* __restrict__ in_qkv,
    const __nv_bfloat16* __restrict__ conv1d_w,
    __nv_bfloat16* __restrict__ conv_state,
    int channels)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;

    __nv_bfloat16* cs = conv_state + c * 4;
    const __nv_bfloat16* cw = conv1d_w + c * 4;

    float s0 = __bfloat162float(cs[1]);
    float s1 = __bfloat162float(cs[2]);
    float s2 = __bfloat162float(cs[3]);
    float s3 = __bfloat162float(in_qkv[c]);

    // Update state (shift left)
    cs[0] = __float2bfloat16(s0);
    cs[1] = __float2bfloat16(s1);
    cs[2] = __float2bfloat16(s2);
    cs[3] = __float2bfloat16(s3);

    float w0 = __bfloat162float(cw[0]);
    float w1 = __bfloat162float(cw[1]);
    float w2 = __bfloat162float(cw[2]);
    float w3 = __bfloat162float(cw[3]);

    float val = s0 * w0 + s1 * w1 + s2 * w2 + s3 * w3;
    float silu_val = val / (1.0f + __expf(-val));
    conv_out[c] = __float2bfloat16(silu_val);
}

__global__ void deltanet_ssm_step_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ conv_out,
    const __nv_bfloat16* __restrict__ in_z,
    const __nv_bfloat16* __restrict__ in_a,
    const __nv_bfloat16* __restrict__ in_b,
    const __nv_bfloat16* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ norm_w,
    float* __restrict__ ssm_state,
    int num_k_heads,
    int num_v_heads,
    int head_dim)
{
    int h = blockIdx.x;
    if (h >= num_v_heads) return;

    int tid = threadIdx.x;
    int k_h = h / (num_v_heads / num_k_heads); // 48 / 16 = 3

    __shared__ float s_q[128];
    __shared__ float s_k[128];
    __shared__ float s_v[128];
    __shared__ float s_z[128];
    __shared__ float s_kv_mem[128];
    __shared__ float s_out[128];
    __shared__ float s_q_norm_sq;
    __shared__ float s_k_norm_sq;
    __shared__ float s_out_norm_sq;

    int q_offset = k_h * head_dim;
    int k_offset = (num_k_heads * head_dim) + k_h * head_dim;
    int v_offset = (2 * num_k_heads * head_dim) + h * head_dim;

    const __nv_bfloat16* q_vec = conv_out + q_offset;
    const __nv_bfloat16* k_vec = conv_out + k_offset;
    const __nv_bfloat16* v_vec = conv_out + v_offset;
    const __nv_bfloat16* z_vec = in_z + h * head_dim;

    float* state_h = ssm_state + (size_t)h * head_dim * head_dim;

    float a_val = __bfloat162float(in_a[h]);
    float b_val = __bfloat162float(in_b[h]);
    float dt_val = __bfloat162float(dt_bias[h]);
    float a_log_val = __bfloat162float(A_log[h]);

    float beta = 1.0f / (1.0f + __expf(-b_val));
    float val_a = a_val + dt_val;
    float softplus_a = (val_a > 20.0f) ? val_a : log1pf(__expf(val_a));
    float g = -__expf(a_log_val) * softplus_a;
    float decay = __expf(g);

    if (tid < head_dim) {
        s_q[tid] = __bfloat162float(q_vec[tid]);
        s_k[tid] = __bfloat162float(k_vec[tid]);
        s_v[tid] = __bfloat162float(v_vec[tid]);
        s_z[tid] = __bfloat162float(z_vec[tid]);
    }
    if (tid == 0) {
        s_q_norm_sq = 0.0f;
        s_k_norm_sq = 0.0f;
        s_out_norm_sq = 0.0f;
    }
    __syncthreads();

    // 1. L2 Normalize Q and K
    float q_sq = (tid < head_dim) ? (s_q[tid] * s_q[tid]) : 0.0f;
    float k_sq = (tid < head_dim) ? (s_k[tid] * s_k[tid]) : 0.0f;

    for (int offset = 16; offset > 0; offset /= 2) {
        q_sq += __shfl_down_sync(0xFFFFFFFF, q_sq, offset);
        k_sq += __shfl_down_sync(0xFFFFFFFF, k_sq, offset);
    }
    if (tid % 32 == 0) {
        atomicAdd(&s_q_norm_sq, q_sq);
        atomicAdd(&s_k_norm_sq, k_sq);
    }
    __syncthreads();

    if (tid < head_dim) {
        float r_q = rsqrtf(s_q_norm_sq + 1e-6f) * (1.0f / sqrtf((float)head_dim));
        float r_k = rsqrtf(s_k_norm_sq + 1e-6f);
        s_q[tid] *= r_q;
        s_k[tid] *= r_k;
    }
    __syncthreads();

    // 2. Decay state and compute kv_mem[col] = sum_row (decay * S[row, col]) * k[row]
    if (tid < head_dim) {
        float mem = 0.0f;
        for (int r = 0; r < head_dim; r++) {
            float decayed_s = decay * state_h[r * head_dim + tid];
            state_h[r * head_dim + tid] = decayed_s;
            mem += decayed_s * s_k[r];
        }
        s_kv_mem[tid] = mem;
    }
    __syncthreads();

    // 3. State delta update: S[r, c] = decayed_S[r, c] + k[r] * delta[c]
    // and compute out[c] = sum_r S[r, c] * q[r]
    if (tid < head_dim) {
        float delta_c = (s_v[tid] - s_kv_mem[tid]) * beta;
        float out_c = 0.0f;
        for (int r = 0; r < head_dim; r++) {
            float new_s = state_h[r * head_dim + tid] + s_k[r] * delta_c;
            state_h[r * head_dim + tid] = new_s;
            out_c += new_s * s_q[r];
        }
        s_out[tid] = out_c;
    }
    __syncthreads();

    // 4. Output RMSNorm + Z-gating
    float out_sq = (tid < head_dim) ? (s_out[tid] * s_out[tid]) : 0.0f;
    for (int offset = 16; offset > 0; offset /= 2) {
        out_sq += __shfl_down_sync(0xFFFFFFFF, out_sq, offset);
    }
    if (tid % 32 == 0) atomicAdd(&s_out_norm_sq, out_sq);
    __syncthreads();

    float r_out = rsqrtf(s_out_norm_sq / (float)head_dim + 1e-6f);
    if (tid < head_dim) {
        float normed = s_out[tid] * r_out * __bfloat162float(norm_w[tid]);
        float z = s_z[tid];
        float silu_z = z / (1.0f + __expf(-z));
        out[h * head_dim + tid] = __float2bfloat16(normed * silu_z);
    }
}

void deltanet_linear_attention_decode_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* in_qkv,
    const __nv_bfloat16* in_z,
    const __nv_bfloat16* in_a,
    const __nv_bfloat16* in_b,
    const __nv_bfloat16* conv1d_w,
    __nv_bfloat16* conv_state,
    const __nv_bfloat16* A_log,
    const __nv_bfloat16* dt_bias,
    const __nv_bfloat16* norm_w,
    float* ssm_state,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    cudaStream_t stream)
{
    int channels = 10240;
    int threads = 256;
    int blocks = (channels + threads - 1) / threads;

    __nv_bfloat16* conv_out = const_cast<__nv_bfloat16*>(in_qkv);
    deltanet_conv_kernel<<<blocks, threads, 0, stream>>>(
        conv_out, in_qkv, conv1d_w, conv_state, channels);

    deltanet_ssm_step_kernel<<<num_v_heads, 128, 0, stream>>>(
        out, conv_out, in_z, in_a, in_b, A_log, dt_bias, norm_w, ssm_state,
        num_k_heads, num_v_heads, head_dim);
}

// ── Qwen 3.8 Gated GQA Attention Decode Kernels ───────────────────────────────

__global__ void qwen_gqa_write_kv_kernel(
    __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ k_norm_w,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    int n_kv_heads,
    int head_dim,
    int pos,
    float rope_theta,
    float eps)
{
    int kv_head = blockIdx.x;
    if (kv_head >= n_kv_heads) return;

    int tid = threadIdx.x;
    __nv_bfloat16* k_vec = k + kv_head * head_dim;
    const __nv_bfloat16* v_vec = v + kv_head * head_dim;

    // 1. RMSNorm on K
    __shared__ float s_k_sum;
    if (tid == 0) s_k_sum = 0.0f;
    __syncthreads();

    float k_sq = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float val = __bfloat162float(k_vec[i]);
        k_sq += val * val;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        k_sq += __shfl_down_sync(0xFFFFFFFF, k_sq, offset);
    }
    if (tid % 32 == 0) atomicAdd(&s_k_sum, k_sq);
    __syncthreads();

    float k_rrms = rsqrtf(s_k_sum / (float)head_dim + eps);
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float k_normed = __bfloat162float(k_vec[i]) * k_rrms * (1.0f + __bfloat162float(k_norm_w[i]));
        k_vec[i] = __float2bfloat16(k_normed);
    }
    __syncthreads();

    // 2. Apply partial RoPE to K (rotary_dim = 64)
    int rotary_dim = 64;
    int half_rotary = rotary_dim / 2; // 32
    for (int i = tid; i < half_rotary; i += blockDim.x) {
        float freq = 1.0f / powf(rope_theta, (float)(2 * i) / (float)rotary_dim);
        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        float k0 = __bfloat162float(k_vec[i]);
        float k1 = __bfloat162float(k_vec[i + half_rotary]);

        k_vec[i] = __float2bfloat16(k0 * cos_a - k1 * sin_a);
        k_vec[i + half_rotary] = __float2bfloat16(k0 * sin_a + k1 * cos_a);
    }
    __syncthreads();

    // 3. Store into KV cache
    size_t cache_offset = ((size_t)pos * n_kv_heads + kv_head) * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        k_cache[cache_offset + i] = k_vec[i];
        v_cache[cache_offset + i] = v_vec[i];
    }
}

__global__ void qwen_gqa_compute_attn_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q_and_gate,
    const __nv_bfloat16* __restrict__ q_norm_w,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int pos,
    float rope_theta,
    float eps)
{
    int q_head = blockIdx.x;
    if (q_head >= n_q_heads) return;

    int tid = threadIdx.x;
    int kv_head = q_head / (n_q_heads / n_kv_heads);

    __shared__ float s_q[256];
    __shared__ float s_q_sum;
    __shared__ float s_max_val;
    __shared__ float s_sum_exp;
    extern __shared__ float s_scores[];

    const __nv_bfloat16* q_in = q_and_gate + (size_t)q_head * (2 * head_dim);
    const __nv_bfloat16* gate_in = q_in + head_dim;
    __nv_bfloat16* out_vec = out + (size_t)q_head * head_dim;

    // 1. RMSNorm on Q per head
    if (tid == 0) s_q_sum = 0.0f;
    __syncthreads();

    float q_sq = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float val = __bfloat162float(q_in[i]);
        s_q[i] = val;
        q_sq += val * val;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        q_sq += __shfl_down_sync(0xFFFFFFFF, q_sq, offset);
    }
    if (tid % 32 == 0) atomicAdd(&s_q_sum, q_sq);
    __syncthreads();

    float q_rrms = rsqrtf(s_q_sum / (float)head_dim + eps);
    for (int i = tid; i < head_dim; i += blockDim.x) {
        s_q[i] = s_q[i] * q_rrms * (1.0f + __bfloat162float(q_norm_w[i]));
    }
    __syncthreads();

    // 2. Apply partial RoPE to Q (rotary_dim = 64)
    int rotary_dim = 64;
    int half_rotary = rotary_dim / 2;
    for (int i = tid; i < half_rotary; i += blockDim.x) {
        float freq = 1.0f / powf(rope_theta, (float)(2 * i) / (float)rotary_dim);
        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        float q0 = s_q[i];
        float q1 = s_q[i + half_rotary];

        s_q[i] = q0 * cos_a - q1 * sin_a;
        s_q[i + half_rotary] = q0 * sin_a + q1 * cos_a;
    }
    __syncthreads();

    // 3. Compute attention scores over history [0..pos]
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int t = tid; t <= pos; t += blockDim.x) {
        size_t k_offset = ((size_t)t * n_kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            dot += s_q[i] * __bfloat162float(k_cache[k_offset + i]);
        }
        s_scores[t] = dot * scale;
    }
    __syncthreads();

    // 4. Softmax over scores
    if (tid == 0) {
        float max_s = -1e38f;
        for (int t = 0; t <= pos; t++) {
            if (s_scores[t] > max_s) max_s = s_scores[t];
        }
        s_max_val = max_s;
        float sum_e = 0.0f;
        for (int t = 0; t <= pos; t++) {
            float e = __expf(s_scores[t] - max_s);
            s_scores[t] = e;
            sum_e += e;
        }
        s_sum_exp = sum_e;
    }
    __syncthreads();

    float inv_sum = 1.0f / (s_sum_exp + 1e-8f);
    for (int t = tid; t <= pos; t += blockDim.x) {
        s_scores[t] *= inv_sum;
    }
    __syncthreads();

    // 5. Output = sum_t (score_t * v_cache_t) * sigmoid(gate)
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float accum = 0.0f;
        for (int t = 0; t <= pos; t++) {
            size_t v_offset = ((size_t)t * n_kv_heads + kv_head) * head_dim;
            accum += s_scores[t] * __bfloat162float(v_cache[v_offset + i]);
        }
        float g_val = __bfloat162float(gate_in[i]);
        float sig_g = 1.0f / (1.0f + __expf(-g_val));
        out_vec[i] = __float2bfloat16(accum * sig_g);
    }
}

void qwen_gqa_decode_gated_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* q_and_gate,
    __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const __nv_bfloat16* q_norm_w,
    const __nv_bfloat16* k_norm_w,
    __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int pos,
    int max_seq_len,
    float rope_theta,
    float eps,
    cudaStream_t stream)
{
    int threads = 128;
    // Step 1: Write KV to cache
    qwen_gqa_write_kv_kernel<<<n_kv_heads, threads, 0, stream>>>(
        k, v, k_norm_w, k_cache, v_cache, n_kv_heads, head_dim, pos, rope_theta, eps);

    // Step 2: Compute Query Attention against cache and Gate
    size_t smem_size = (pos + 1) * sizeof(float);
    qwen_gqa_compute_attn_kernel<<<n_q_heads, threads, smem_size, stream>>>(
        out, q_and_gate, q_norm_w, k_cache, v_cache, n_q_heads, n_kv_heads, head_dim, pos, rope_theta, eps);
}

// ── 2-Stage Parallel Grid ArgMax across all GPU SMs ────────────────────────
__global__ void argmax_f32_stage1_kernel(
    float* __restrict__ block_maxes,
    int32_t* __restrict__ block_indices,
    const float* __restrict__ logits,
    int n)
{
    __shared__ float s_max[32];
    __shared__ int32_t s_idx[32];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float max_val = -1e38f;
    int32_t max_idx = 0;

    for (int i = gid; i < n; i += stride) {
        float v = logits[i];
        if (v > max_val) {
            max_val = v;
            max_idx = (int32_t)i;
        }
    }

    // Warp reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
        int32_t other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
        if (other_val > max_val) {
            max_val = other_val;
            max_idx = other_idx;
        }
    }

    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) {
        s_max[warp] = max_val;
        s_idx[warp] = max_idx;
    }
    __syncthreads();

    if (warp == 0) {
        int n_warps = blockDim.x / 32;
        max_val = (lane < n_warps) ? s_max[lane] : -1e38f;
        max_idx = (lane < n_warps) ? s_idx[lane] : 0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
            int32_t other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
            if (other_val > max_val) {
                max_val = other_val;
                max_idx = other_idx;
            }
        }
        if (lane == 0) {
            block_maxes[blockIdx.x] = max_val;
            block_indices[blockIdx.x] = max_idx;
        }
    }
}

__global__ void argmax_f32_stage2_kernel(
    int32_t* __restrict__ out,
    const float* __restrict__ block_maxes,
    const int32_t* __restrict__ block_indices,
    int n_blocks)
{
    __shared__ float s_max[32];
    __shared__ int32_t s_idx[32];

    int tid = threadIdx.x;
    float max_val = (tid < n_blocks) ? block_maxes[tid] : -1e38f;
    int32_t max_idx = (tid < n_blocks) ? block_indices[tid] : 0;

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
        int32_t other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
        if (other_val > max_val) {
            max_val = other_val;
            max_idx = other_idx;
        }
    }

    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) {
        s_max[warp] = max_val;
        s_idx[warp] = max_idx;
    }
    __syncthreads();

    if (warp == 0) {
        int n_warps = (blockDim.x + 31) / 32;
        max_val = (lane < n_warps) ? s_max[lane] : -1e38f;
        max_idx = (lane < n_warps) ? s_idx[lane] : 0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
            int32_t other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
            if (other_val > max_val) {
                max_val = other_val;
                max_idx = other_idx;
            }
        }
        if (lane == 0) {
            *out = max_idx;
        }
    }
}

void argmax_f32_cuda(int32_t* out, const float* logits, int n, cudaStream_t stream) {
    static float* s_d_block_maxes = nullptr;
    static int32_t* s_d_block_indices = nullptr;
    static const int N_BLOCKS = 128;
    if (!s_d_block_maxes) {
        cudaMalloc(&s_d_block_maxes, N_BLOCKS * sizeof(float));
        cudaMalloc(&s_d_block_indices, N_BLOCKS * sizeof(int32_t));
    }
    argmax_f32_stage1_kernel<<<N_BLOCKS, 256, 0, stream>>>(s_d_block_maxes, s_d_block_indices, logits, n);
    argmax_f32_stage2_kernel<<<1, 128, 0, stream>>>(out, s_d_block_maxes, s_d_block_indices, N_BLOCKS);
}

// ── GPU-Native Multinomial Softmax Sampling ─────────────────────────────────
__global__ void logits_exp_sum_stage1_kernel(
    float* __restrict__ block_sums,
    float* __restrict__ logits,
    int n,
    float inv_temp,
    const float* __restrict__ block_maxes,
    int n_blocks)
{
    __shared__ float s_sum[32];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float max_v = block_maxes ? block_maxes[0] : 0.0f;
    float sum = 0.0f;

    for (int i = gid; i < n; i += stride) {
        float v = logits[i];
        float e = expf((v - max_v) * inv_temp);
        logits[i] = e;
        sum += e;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) s_sum[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        int n_warps = blockDim.x / 32;
        sum = (lane < n_warps) ? s_sum[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) block_sums[blockIdx.x] = sum;
    }
}

__global__ void logits_sample_stage2_kernel(
    int32_t* __restrict__ out,
    const float* __restrict__ logits,
    const float* __restrict__ block_sums,
    int n,
    int n_blocks,
    float rand_val)
{
    __shared__ float s_sum[32];
    int tid = threadIdx.x;
    float total_sum = (tid < n_blocks) ? block_sums[tid] : 0.0f;

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        total_sum += __shfl_down_sync(0xffffffff, total_sum, offset);

    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) s_sum[warp] = total_sum;
    __syncthreads();

    if (warp == 0) {
        int n_warps = (blockDim.x + 31) / 32;
        total_sum = (lane < n_warps) ? s_sum[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            total_sum += __shfl_down_sync(0xffffffff, total_sum, offset);
        if (lane == 0) s_sum[0] = total_sum;
    }
    __syncthreads();

    total_sum = s_sum[0];
    if (total_sum <= 0.0f) {
        if (tid == 0) *out = 0;
        return;
    }

    if (tid == 0) {
        float target = rand_val * total_sum;
        float cum = 0.0f;
        int sampled_id = 0;
        for (int i = 0; i < n; i++) {
            cum += logits[i];
            if (cum >= target) {
                sampled_id = i;
                break;
            }
        }
        *out = sampled_id;
    }
}

void sample_multinomial_f32_cuda(
    int32_t* out,
    float* logits,
    int n,
    float temperature,
    float rand_val,
    cudaStream_t stream)
{
    static float* s_d_block_maxes = nullptr;
    static int32_t* s_d_block_indices = nullptr;
    static float* s_d_block_sums = nullptr;
    static const int N_BLOCKS = 128;
    if (!s_d_block_maxes) {
        cudaMalloc(&s_d_block_maxes, N_BLOCKS * sizeof(float));
        cudaMalloc(&s_d_block_indices, N_BLOCKS * sizeof(int32_t));
        cudaMalloc(&s_d_block_sums, N_BLOCKS * sizeof(float));
    }
    argmax_f32_stage1_kernel<<<N_BLOCKS, 256, 0, stream>>>(s_d_block_maxes, s_d_block_indices, logits, n);

    float inv_temp = 1.0f / fmaxf(temperature, 1e-4f);
    logits_exp_sum_stage1_kernel<<<N_BLOCKS, 256, 0, stream>>>(s_d_block_sums, logits, n, inv_temp, s_d_block_maxes, N_BLOCKS);
    logits_sample_stage2_kernel<<<1, 128, 0, stream>>>(out, logits, s_d_block_sums, n, N_BLOCKS, rand_val);
}
