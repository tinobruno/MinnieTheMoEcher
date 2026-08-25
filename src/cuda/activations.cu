// activations.cu — CUDA kernels for moecher inference engine
// Custom kernels for operations not covered by cuBLAS:
//   RMSNorm, SiLU*mul (SwiGLU), RoPE, FP8/FP4 dequant, embedding,
//   softmax, top-k routing, HC Sinkhorn, etc.

#include "activations.cuh"
#include <cuda_bf16.h>
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

    // Compute base frequency
    float freq = 1.0f / powf(base, (float)(2 * pair) / (float)rope_dim);

    // Apply YaRN scaling if original_seq_len > 0
    if (original_seq_len > 0) {
        float dim_f = (float)rope_dim;
        float low_f = dim_f * logf((float)original_seq_len / ((float)beta_fast * 2.0f * 3.14159265f)) / (2.0f * logf(base));
        float high_f = dim_f * logf((float)original_seq_len / ((float)beta_slow * 2.0f * 3.14159265f)) / (2.0f * logf(base));
        int low = max((int)floorf(low_f), 0);
        int high = min((int)ceilf(high_f), rope_dim - 1);

        float ramp;
        if (low == high) {
            ramp = (pair >= low) ? 1.0f : 0.0f;
        } else {
            ramp = fminf(fmaxf(((float)pair - (float)low) / ((float)high - (float)low), 0.0f), 1.0f);
        }
        float smooth = 1.0f - ramp;
        freq = freq / factor * (1.0f - smooth) + freq * smooth;
    }

    float angle = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

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
    mla_attention_kernel<<<n_heads, n_threads, smem_bytes, stream>>>(
        q, kv, attn_sink, out, cache_len, head_dim, scale
    );
}

// ════════════════════════════════════════════════════════════════════════════════
//  Compressor Kernels
// ════════════════════════════════════════════════════════════════════════════════

// BF16 GEMV: out[row] = dot(W[row, :], x[:]) for each row
// Each block handles one row. Threads cooperatively reduce across K.
__global__ void gemv_bf16_kernel(
    float* __restrict__ out,
    const __nv_bfloat16* __restrict__ W,
    const __nv_bfloat16* __restrict__ x,
    int N, int K)
{
    int row = blockIdx.x;
    if (row >= N) return;

    extern __shared__ float sdata_gemv[];

    int tid = threadIdx.x;
    float sum = 0.0f;
    const __nv_bfloat16* w_row = W + (size_t)row * K;
    for (int i = tid; i < K; i += blockDim.x) {
        sum += bf16_to_float(w_row[i]) * bf16_to_float(x[i]);
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) sdata_gemv[warp] = sum;
    __syncthreads();

    // Final reduction across warps
    int n_warps = (blockDim.x + 31) / 32;
    if (tid < n_warps) sum = sdata_gemv[tid];
    else sum = 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if (tid == 0) out[row] = sum;
}

void gemv_bf16_cuda(
    float* out,
    const __nv_bfloat16* W,
    const __nv_bfloat16* x,
    int N, int K,
    cudaStream_t stream)
{
    int n_threads = 256;
    int n_warps = (n_threads + 31) / 32;
    size_t smem = n_warps * sizeof(float);
    gemv_bf16_kernel<<<N, n_threads, smem, stream>>>(out, W, x, N, K);
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

__global__ void gemv_fp8_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ vec,
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scale,
    int N, int K, int block_size)
{
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;
    float sum = 0.0f;
    int scale_cols = (K + block_size - 1) / block_size;
    int br = row / block_size;

    // Process 8 elements at a time
    const uint2* w_vec8 = reinterpret_cast<const uint2*>(&weight[row * K]);
    const uint4* a_vec4 = reinterpret_cast<const uint4*>(vec); // uint4 = 16 bytes = 8 bfloat16

    for (int chunk = tid; chunk < K / 8; chunk += blockDim.x) {
        int logical_col = chunk * 8;
        int bc = logical_col / block_size;
        float s_val = e8m0_to_float_v2(scale[br * scale_cols + bc]);

        uint2 w8 = w_vec8[chunk];
        uint4 a8 = a_vec4[chunk];

        float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&a8.x));
        float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&a8.y));
        float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&a8.z));
        float2 f3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&a8.w));

        float chunk_sum = 0.0f;
        chunk_sum += fp8_e4m3_to_float_v2((w8.x) & 0xFF) * f0.x;
        chunk_sum += fp8_e4m3_to_float_v2((w8.x >> 8) & 0xFF) * f0.y;
        chunk_sum += fp8_e4m3_to_float_v2((w8.x >> 16) & 0xFF) * f1.x;
        chunk_sum += fp8_e4m3_to_float_v2((w8.x >> 24) & 0xFF) * f1.y;

        chunk_sum += fp8_e4m3_to_float_v2((w8.y) & 0xFF) * f2.x;
        chunk_sum += fp8_e4m3_to_float_v2((w8.y >> 8) & 0xFF) * f2.y;
        chunk_sum += fp8_e4m3_to_float_v2((w8.y >> 16) & 0xFF) * f3.x;
        chunk_sum += fp8_e4m3_to_float_v2((w8.y >> 24) & 0xFF) * f3.y;

        sum += chunk_sum * s_val;
    }
    // Warp reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    // Shared memory for block reduce
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
        if (tid == 0) out[row] = __float2bfloat16(sum);
    }
}
void gemv_fp8_cuda(__nv_bfloat16* out, const __nv_bfloat16* vec,
                   const uint8_t* weight, const uint8_t* scale,
                   int N, int K, int block_size, cudaStream_t stream) {
    int threads = 128;
    gemv_fp8_kernel<<<N, threads, 0, stream>>>(out, vec, weight, scale, N, K, block_size);
}

__global__ void gemv_fp8_grouped_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ vec,
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scale,
    int N, int K, int groups, int block_size)
{
    int group = blockIdx.y;
    int row = blockIdx.x;
    if (group >= groups || row >= N) return;

    int tid = threadIdx.x;
    float sum = 0.0f;
    int scale_cols = (K + block_size - 1) / block_size;
    int scale_rows_per_group = (N + block_size - 1) / block_size;
    int br = row / block_size;

    const uint8_t* g_weight = weight + (size_t)group * N * K;
    const uint8_t* g_scale = scale + (size_t)group * scale_rows_per_group * scale_cols;
    const __nv_bfloat16* g_vec = vec + (size_t)group * K;
    __nv_bfloat16* g_out = out + (size_t)group * N;

    const uint2* w_vec8 = reinterpret_cast<const uint2*>(&g_weight[row * K]);
    const uint4* a_vec4 = reinterpret_cast<const uint4*>(g_vec);

    for (int chunk = tid; chunk < K / 8; chunk += blockDim.x) {
        int logical_col = chunk * 8;
        int bc = logical_col / block_size;
        float s_val = e8m0_to_float_v2(g_scale[br * scale_cols + bc]);

        uint2 w8 = w_vec8[chunk];
        uint4 a8 = a_vec4[chunk];

        float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&a8.x));
        float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&a8.y));
        float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&a8.z));
        float2 f3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&a8.w));

        float chunk_sum = 0.0f;
        chunk_sum += fp8_e4m3_to_float_v2((w8.x) & 0xFF) * f0.x;
        chunk_sum += fp8_e4m3_to_float_v2((w8.x >> 8) & 0xFF) * f0.y;
        chunk_sum += fp8_e4m3_to_float_v2((w8.x >> 16) & 0xFF) * f1.x;
        chunk_sum += fp8_e4m3_to_float_v2((w8.x >> 24) & 0xFF) * f1.y;

        chunk_sum += fp8_e4m3_to_float_v2((w8.y) & 0xFF) * f2.x;
        chunk_sum += fp8_e4m3_to_float_v2((w8.y >> 8) & 0xFF) * f2.y;
        chunk_sum += fp8_e4m3_to_float_v2((w8.y >> 16) & 0xFF) * f3.x;
        chunk_sum += fp8_e4m3_to_float_v2((w8.y >> 24) & 0xFF) * f3.y;

        sum += chunk_sum * s_val;
    }

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
        if (tid == 0) g_out[row] = __float2bfloat16(sum);
    }
}

void gemv_fp8_grouped_cuda(
    __nv_bfloat16* out, const __nv_bfloat16* vec,
    const uint8_t* weight, const uint8_t* scale,
    int N, int K, int groups, int block_size, cudaStream_t stream)
{
    int threads = 128;
    dim3 blocks(N, groups);
    gemv_fp8_grouped_kernel<<<blocks, threads, 0, stream>>>(out, vec, weight, scale, N, K, groups, block_size);
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
        float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
        float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
        float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.z));
        float2 f3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.w));
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
        float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
        float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
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
                float2 f2 = __bfloat1622float2(bf2);

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

        float2 f0_0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u0.x));
        float2 f0_1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u0.y));
        float2 f0_2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u0.z));
        float2 f0_3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u0.w));

        float2 f1_0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u1.x));
        float2 f1_1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u1.y));
        float2 f1_2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u1.z));
        float2 f1_3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u1.w));

        float2 f2_0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u2.x));
        float2 f2_1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u2.y));
        float2 f2_2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u2.z));
        float2 f2_3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u2.w));

        float2 f3_0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u3.x));
        float2 f3_1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u3.y));
        float2 f3_2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u3.z));
        float2 f3_3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u3.w));

        __nv_bfloat162 res0 = __float22bfloat162_rn(make_float2(
            w0 * f0_0.x + w1 * f1_0.x + w2 * f2_0.x + w3 * f3_0.x,
            w0 * f0_0.y + w1 * f1_0.y + w2 * f2_0.y + w3 * f3_0.y
        ));
        __nv_bfloat162 res1 = __float22bfloat162_rn(make_float2(
            w0 * f0_1.x + w1 * f1_1.x + w2 * f2_1.x + w3 * f3_1.x,
            w0 * f0_1.y + w1 * f1_1.y + w2 * f2_1.y + w3 * f3_1.y
        ));
        __nv_bfloat162 res2 = __float22bfloat162_rn(make_float2(
            w0 * f0_2.x + w1 * f1_2.x + w2 * f2_2.x + w3 * f3_2.x,
            w0 * f0_2.y + w1 * f1_2.y + w2 * f2_2.y + w3 * f3_2.y
        ));
        __nv_bfloat162 res3 = __float22bfloat162_rn(make_float2(
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

        float2 fh_0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&uh.x));
        float2 fh_1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&uh.y));
        float2 fh_2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&uh.z));
        float2 fh_3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&uh.w));

        float2 fr0_0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ur0.x));
        float2 fr0_1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ur0.y));
        float2 fr0_2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ur0.z));
        float2 fr0_3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ur0.w));

        float2 fr1_0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ur1.x));
        float2 fr1_1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ur1.y));
        float2 fr1_2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ur1.z));
        float2 fr1_3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ur1.w));

        float2 fr2_0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ur2.x));
        float2 fr2_1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ur2.y));
        float2 fr2_2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ur2.z));
        float2 fr2_3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ur2.w));

        float2 fr3_0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ur3.x));
        float2 fr3_1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ur3.y));
        float2 fr3_2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ur3.z));
        float2 fr3_3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ur3.w));

        // Row 0
        uint4 out0;
        *reinterpret_cast<__nv_bfloat162*>(&out0.x) = __float22bfloat162_rn(make_float2(p0 * fh_0.x + c00 * fr0_0.x + c10 * fr1_0.x + c20 * fr2_0.x + c30 * fr3_0.x, p0 * fh_0.y + c00 * fr0_0.y + c10 * fr1_0.y + c20 * fr2_0.y + c30 * fr3_0.y));
        *reinterpret_cast<__nv_bfloat162*>(&out0.y) = __float22bfloat162_rn(make_float2(p0 * fh_1.x + c00 * fr0_1.x + c10 * fr1_1.x + c20 * fr2_1.x + c30 * fr3_1.x, p0 * fh_1.y + c00 * fr0_1.y + c10 * fr1_1.y + c20 * fr2_1.y + c30 * fr3_1.y));
        *reinterpret_cast<__nv_bfloat162*>(&out0.z) = __float22bfloat162_rn(make_float2(p0 * fh_2.x + c00 * fr0_2.x + c10 * fr1_2.x + c20 * fr2_2.x + c30 * fr3_2.x, p0 * fh_2.y + c00 * fr0_2.y + c10 * fr1_2.y + c20 * fr2_2.y + c30 * fr3_2.y));
        *reinterpret_cast<__nv_bfloat162*>(&out0.w) = __float22bfloat162_rn(make_float2(p0 * fh_3.x + c00 * fr0_3.x + c10 * fr1_3.x + c20 * fr2_3.x + c30 * fr3_3.x, p0 * fh_3.y + c00 * fr0_3.y + c10 * fr1_3.y + c20 * fr2_3.y + c30 * fr3_3.y));

        // Row 1
        uint4 out1;
        *reinterpret_cast<__nv_bfloat162*>(&out1.x) = __float22bfloat162_rn(make_float2(p1 * fh_0.x + c01 * fr0_0.x + c11 * fr1_0.x + c21 * fr2_0.x + c31 * fr3_0.x, p1 * fh_0.y + c01 * fr0_0.y + c11 * fr1_0.y + c21 * fr2_0.y + c31 * fr3_0.y));
        *reinterpret_cast<__nv_bfloat162*>(&out1.y) = __float22bfloat162_rn(make_float2(p1 * fh_1.x + c01 * fr0_1.x + c11 * fr1_1.x + c21 * fr2_1.x + c31 * fr3_1.x, p1 * fh_1.y + c01 * fr0_1.y + c11 * fr1_1.y + c21 * fr2_1.y + c31 * fr3_1.y));
        *reinterpret_cast<__nv_bfloat162*>(&out1.z) = __float22bfloat162_rn(make_float2(p1 * fh_2.x + c01 * fr0_2.x + c11 * fr1_2.x + c21 * fr2_2.x + c31 * fr3_2.x, p1 * fh_2.y + c01 * fr0_2.y + c11 * fr1_2.y + c21 * fr2_2.y + c31 * fr3_2.y));
        *reinterpret_cast<__nv_bfloat162*>(&out1.w) = __float22bfloat162_rn(make_float2(p1 * fh_3.x + c01 * fr0_3.x + c11 * fr1_3.x + c21 * fr2_3.x + c31 * fr3_3.x, p1 * fh_3.y + c01 * fr0_3.y + c11 * fr1_3.y + c21 * fr2_3.y + c31 * fr3_3.y));

        // Row 2
        uint4 out2;
        *reinterpret_cast<__nv_bfloat162*>(&out2.x) = __float22bfloat162_rn(make_float2(p2 * fh_0.x + c02 * fr0_0.x + c12 * fr1_0.x + c22 * fr2_0.x + c32 * fr3_0.x, p2 * fh_0.y + c02 * fr0_0.y + c12 * fr1_0.y + c22 * fr2_0.y + c32 * fr3_0.y));
        *reinterpret_cast<__nv_bfloat162*>(&out2.y) = __float22bfloat162_rn(make_float2(p2 * fh_1.x + c02 * fr0_1.x + c12 * fr1_1.x + c22 * fr2_1.x + c32 * fr3_1.x, p2 * fh_1.y + c02 * fr0_1.y + c12 * fr1_1.y + c22 * fr2_1.y + c32 * fr3_1.y));
        *reinterpret_cast<__nv_bfloat162*>(&out2.z) = __float22bfloat162_rn(make_float2(p2 * fh_2.x + c02 * fr0_2.x + c12 * fr1_2.x + c22 * fr2_2.x + c32 * fr3_2.x, p2 * fh_2.y + c02 * fr0_2.y + c12 * fr1_2.y + c22 * fr2_2.y + c32 * fr3_2.y));
        *reinterpret_cast<__nv_bfloat162*>(&out2.w) = __float22bfloat162_rn(make_float2(p2 * fh_3.x + c02 * fr0_3.x + c12 * fr1_3.x + c22 * fr2_3.x + c32 * fr3_3.x, p2 * fh_3.y + c02 * fr0_3.y + c12 * fr1_3.y + c22 * fr2_3.y + c32 * fr3_3.y));

        // Row 3
        uint4 out3;
        *reinterpret_cast<__nv_bfloat162*>(&out3.x) = __float22bfloat162_rn(make_float2(p3 * fh_0.x + c03 * fr0_0.x + c13 * fr1_0.x + c23 * fr2_0.x + c33 * fr3_0.x, p3 * fh_0.y + c03 * fr0_0.y + c13 * fr1_0.y + c23 * fr2_0.y + c33 * fr3_0.y));
        *reinterpret_cast<__nv_bfloat162*>(&out3.y) = __float22bfloat162_rn(make_float2(p3 * fh_1.x + c03 * fr0_1.x + c13 * fr1_1.x + c23 * fr2_1.x + c33 * fr3_1.x, p3 * fh_1.y + c03 * fr0_1.y + c13 * fr1_1.y + c23 * fr2_1.y + c33 * fr3_1.y));
        *reinterpret_cast<__nv_bfloat162*>(&out3.z) = __float22bfloat162_rn(make_float2(p3 * fh_2.x + c03 * fr0_2.x + c13 * fr1_2.x + c23 * fr2_2.x + c33 * fr3_2.x, p3 * fh_2.y + c03 * fr0_2.y + c13 * fr1_2.y + c23 * fr2_2.y + c33 * fr3_2.y));
        *reinterpret_cast<__nv_bfloat162*>(&out3.w) = __float22bfloat162_rn(make_float2(p3 * fh_3.x + c03 * fr0_3.x + c13 * fr1_3.x + c23 * fr2_3.x + c33 * fr3_3.x, p3 * fh_3.y + c03 * fr0_3.y + c13 * fr1_3.y + c23 * fr2_3.y + c33 * fr3_3.y));

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
        float gate = sum1;
        float up = sum3;
        float act = gate / (1.0f + expf(-gate));
        if (swiglu_limit > 0.0f) {
            act = fminf(fmaxf(act, -swiglu_limit), swiglu_limit);
        }
        out[row] = __float2bfloat16(act * up);
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
    __shared__ float s_scores[256];
    __shared__ int   s_indices[256];

    int tid = threadIdx.x;
    if (tid < n_experts) {
        float raw = __bfloat162float(scores_bf16[tid]);
        if (gate_bias) raw += gate_bias[tid];
        float sp = (raw > 20.0f) ? raw : logf(1.0f + expf(raw));
        s_scores[tid] = sqrtf(sp);
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
    int w1_offset, int w3_offset,
    int N, int K, float swiglu_limit)
{
    int k = blockIdx.y;
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N || k >= 6) return;

    const void* p = active_expert_ptrs ? active_expert_ptrs[k] : nullptr;
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
        float gate = sum1;
        float up = sum3;
        float act = gate / (1.0f + expf(-gate));
        if (swiglu_limit > 0.0f) {
            act = fminf(fmaxf(act, -swiglu_limit), swiglu_limit);
        }
        gate_buf[k * N + row] = __float2bfloat16(act * up);
    }
}

void gemv_iq2_xxs_moe_swiglu_fused_cuda(
    __nv_bfloat16* gate_buf,
    const __nv_bfloat16* vec,
    const void* const* active_expert_ptrs,
    int w1_offset, int w3_offset,
    int N, int K, float swiglu_limit,
    cudaStream_t stream)
{
    dim3 threads(32, 8);
    dim3 blocks((N + 7) / 8, 6);
    gemv_iq2_xxs_moe_swiglu_fused_kernel<<<blocks, threads, 0, stream>>>(
        gate_buf, vec, active_expert_ptrs,
        w1_offset, w3_offset, N, K, swiglu_limit);
}

// ── Batched All-6-Experts Q2_K w2 Down Projection Kernel ───────────────────────
__global__ void gemv_q2_k_moe_kernel(
    __nv_bfloat16* __restrict__ down_buf, // [6 * dim]
    const __nv_bfloat16* __restrict__ gate_buf, // [6 * moe_inter]
    const void* const* __restrict__ active_expert_ptrs,
    int w2_offset,
    int N, int K)
{
    int k = blockIdx.y;
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N || k >= 6) return;

    const void* p = active_expert_ptrs ? active_expert_ptrs[k] : nullptr;
    if (!p) return;

    const uint8_t* block = (const uint8_t*)p;
    const block_q2_K* w2 = (const block_q2_K*)(block + w2_offset);

    int lane = threadIdx.x;
    int n_blocks = K / 256;
    const block_q2_K* row_w2 = w2 + row * n_blocks;
    const __nv_bfloat16* k_gate = gate_buf + (size_t)k * K;

    float sum = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        const block_q2_K* blk = &row_w2[b];
        float d = __half2float(blk->d);
        float min = __half2float(blk->dmin);

        #pragma unroll 4
        for (int iter = 0; iter < 8; iter++) {
            int idx = (iter << 5) + lane;
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
            float a = __bfloat162float(k_gate[col_idx]);
            sum += w * a;
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
    cudaStream_t stream)
{
    dim3 threads(32, 8);
    dim3 blocks((N + 7) / 8, 6);
    gemv_q2_k_moe_kernel<<<blocks, threads, 0, stream>>>(
        down_buf, gate_buf, active_expert_ptrs,
        w2_offset, N, K);
}
