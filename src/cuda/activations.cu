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

    // 2. Softmax
    // Find max score
    float local_max = -1e38f;
    for (int t = tid; t < cache_len; t += n_threads) {
        local_max = fmaxf(local_max, scores[t]);
    }
    // Include sink in max calculation
    if (tid == 0) {
        local_max = fmaxf(local_max, attn_sink[h]);
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
    // Include sink in sum calculation
    if (tid == 0) {
        local_sum += expf(attn_sink[h] - block_max);
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
    uint32_t sign = (val & 0x80) << 24;
    uint32_t exp  = (val & 0x78) >> 3;
    uint32_t mant = (val & 0x07);
    uint32_t f_exp = exp + 127 - 7;
    uint32_t f_mant = mant << 20;
    uint32_t res = sign | (f_exp << 23) | f_mant;
    return *((float*)&res);
}

__device__ inline float e8m0_to_float_v2(uint8_t val) {
    uint32_t res = (val + 127 - 127) << 23;
    return *((float*)&res);
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
        uint32_t w_arr[2] = {w8.x, w8.y};
        uint32_t a_arr[4] = {a8.x, a8.y, a8.z, a8.w};

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            uint32_t w_chunk = w_arr[i];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint8_t b = (w_chunk >> (j * 8)) & 0xFF;
                float w_f = fp8_e4m3_to_float_v2(b) * s_val;
                int a_idx = i * 2 + (j / 2);
                uint32_t a_val = a_arr[a_idx];
                __nv_bfloat162 bf2 = *reinterpret_cast<__nv_bfloat162*>(&a_val);
                float2 f2 = __bfloat1622float2(bf2);
                if ((j % 2) == 0) sum += w_f * f2.x;
                else sum += w_f * f2.y;
            }
        }
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

// ── HC Pre Weighted Add Kernel ──────────────────────────────────────────────────
__global__ void hc_pre_weighted_add_kernel(
    __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ hc_state,
    const float* __restrict__ pre_weights,
    int dim, int hc)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dim) return;

    float sum = 0.0f;
    for (int h = 0; h < hc; h++) {
        float w = pre_weights[h];
        if (fabsf(w) >= 1e-10f) {
            float val = __bfloat162float(hc_state[h * dim + tid]);
            sum += w * val;
        }
    }
    hidden[tid] = __float2bfloat16(sum);
}

void hc_pre_weighted_add_cuda(
    __nv_bfloat16* hidden, const __nv_bfloat16* hc_state, const float* pre_weights,
    int dim, int hc, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    hc_pre_weighted_add_kernel<<<blocks, threads, 0, stream>>>(hidden, hc_state, pre_weights, dim, hc);
}

// ── HC Post Update Kernel ───────────────────────────────────────────────────────
__global__ void hc_post_update_kernel(
    __nv_bfloat16* __restrict__ hc_state,
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ hc_residual,
    const float* __restrict__ post_weights,
    const float* __restrict__ comb_weights,
    int dim, int hc)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y; // Which HC head we are computing
    if (tid >= dim || h >= hc) return;

    // Start with post[h] * hidden
    float p_w = post_weights[h];
    float sum = p_w * __bfloat162float(hidden[tid]);

    // Add sum_j(comb[j * hc + h] * residual[j])
    for (int j = 0; j < hc; j++) {
        float c_w = comb_weights[j * hc + h];
        if (fabsf(c_w) >= 1e-10f) {
            float r_val = __bfloat162float(hc_residual[j * dim + tid]);
            sum += c_w * r_val;
        }
    }
    
    hc_state[h * dim + tid] = __float2bfloat16(sum);
}

void hc_post_update_cuda(
    __nv_bfloat16* hc_state, const __nv_bfloat16* hidden, const __nv_bfloat16* hc_residual,
    const float* post_weights, const float* comb_weights,
    int dim, int hc, cudaStream_t stream)
{
    int threads = 256;
    int blocks_x = (dim + threads - 1) / threads;
    dim3 blocks(blocks_x, hc);
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
