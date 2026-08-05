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
    // Special case: 0xFF is NaN in E8M0
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
        // Subnormal: (-1)^sign * 2^(-6) * (mant/8)
        val = ldexpf((float)mant, -9);  // mant * 2^-3 * 2^-6 = mant * 2^-9
    } else if (exp == 15 && mant == 7) {
        val = __int_as_float(0x7FC00000);  // NaN
    } else {
        // Normal: (-1)^sign * 2^(exp-7) * (1 + mant/8)
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
    // mixes layout: [pre(hc), post(hc), comb(hc*hc)]
    int hc = hc_mult;
    int mix_size = (2 + hc) * hc;  // 24 for hc=4

    // Apply sigmoid(mix * scale + base) for each section
    // Pre: indices [0, hc)
    for (int i = 0; i < hc; i++) {
        float v = mixes[i] * scale[0] + base[i];
        pre[i] = 1.0f / (1.0f + expf(-v)) + eps;
    }

    // Post: indices [hc, 2*hc)
    for (int i = 0; i < hc; i++) {
        float v = mixes[hc + i] * scale[1] + base[hc + i];
        post[i] = 1.0f / (1.0f + expf(-v)) + eps;
    }

    // Comb: indices [2*hc, (2+hc)*hc)
    for (int i = 0; i < hc * hc; i++) {
        float v = mixes[2 * hc + i] * scale[2] + base[2 * hc + i];
        comb[i] = 1.0f / (1.0f + expf(-v)) + eps;
    }

    // Sinkhorn normalization on comb[hc, hc]
    for (int iter = 0; iter < sinkhorn_iters; iter++) {
        // Row normalization
        for (int r = 0; r < hc; r++) {
            float row_sum = 0.0f;
            for (int c = 0; c < hc; c++)
                row_sum += comb[r * hc + c];
            float inv = 1.0f / (row_sum + 1e-12f);
            for (int c = 0; c < hc; c++)
                comb[r * hc + c] *= inv;
        }
        // Column normalization
        for (int c = 0; c < hc; c++) {
            float col_sum = 0.0f;
            for (int r = 0; r < hc; r++)
                col_sum += comb[r * hc + c];
            float inv = 1.0f / (col_sum + 1e-12f);
            for (int r = 0; r < hc; r++)
                comb[r * hc + c] *= inv;
        }
    }
}

void hc_split_sinkhorn_cuda(float* pre, float* post, float* comb,
                             const float* mixes, const float* scale,
                             const float* base, int hc_mult,
                             int sinkhorn_iters, float eps,
                             cudaStream_t stream) {
    hc_split_sinkhorn_kernel<<<1, 1, 0, stream>>>(
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

    // Use dynamically allocated shared memory for scores
    extern __shared__ float scores[]; // size: cache_len * sizeof(float)

    // 1. Q @ KV.T
    for (int t = tid; t < cache_len; t += n_threads) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            float qv = __bfloat162float(q[h * head_dim + d]);
            float kv_v = __bfloat162float(kv[t * head_dim + d]);
            dot += qv * kv_v;
        }
        scores[t] = dot * scale;
    }
    __syncthreads();

    // Add attn_sink
    if (tid == 0) {
        scores[0] += attn_sink[h];
    }
    __syncthreads();

    // 2. Softmax
    // Find max score
    float local_max = -1e38f;
    for (int t = tid; t < cache_len; t += n_threads) {
        local_max = fmaxf(local_max, scores[t]);
    }
    // Block-wide max reduction
    __shared__ float s_max[32];
    int lane = tid & 31;
    int warp = tid >> 5;
    float warp_max = local_max;
    for (int offset = 16; offset > 0; offset /= 2) warp_max = fmaxf(warp_max, __shfl_down_sync(0xffffffff, warp_max, offset));
    if (lane == 0) s_max[warp] = warp_max;
    __syncthreads();
    
    float block_max = -1e38f;
    if (tid < (n_threads >> 5)) block_max = s_max[tid];
    for (int offset = 16; offset > 0; offset /= 2) block_max = fmaxf(block_max, __shfl_down_sync(0xffffffff, block_max, offset));
    // Broadcast block_max
    if (tid == 0) s_max[0] = block_max;
    __syncthreads();
    block_max = s_max[0];

    // Exp and sum
    float local_sum = 0.0f;
    for (int t = tid; t < cache_len; t += n_threads) {
        float e = expf(scores[t] - block_max);
        scores[t] = e;
        local_sum += e;
    }
    // Block-wide sum reduction
    __shared__ float s_sum[32];
    float warp_sum = local_sum;
    for (int offset = 16; offset > 0; offset /= 2) warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
    if (lane == 0) s_sum[warp] = warp_sum;
    __syncthreads();

    float block_sum = 0.0f;
    if (tid < (n_threads >> 5)) block_sum = s_sum[tid];
    for (int offset = 16; offset > 0; offset /= 2) block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
    if (tid == 0) s_sum[0] = block_sum;
    __syncthreads();
    block_sum = s_sum[0];

    // Normalize and compute out = scores @ KV
    for (int d = tid; d < head_dim; d += n_threads) {
        float v = 0.0f;
        for (int t = 0; t < cache_len; t++) {
            v += (scores[t] / block_sum) * __bfloat162float(kv[t * head_dim + d]);
        }
        out[h * head_dim + d] = __float2bfloat16(v);
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
