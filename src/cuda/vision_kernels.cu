#include "vision_kernels.cuh"
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>

static __device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

static __device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float v) {
    return __float2bfloat16(v);
}

// ── LayerNorm with Affine Transform ─────────────────────────────────────────

template <int BLOCK_SIZE>
__global__ void layernorm_affine_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta,
    int n_rows,
    int dim,
    float eps)
{
    int row = blockIdx.x;
    if (row >= n_rows) return;

    const __nv_bfloat16* row_x = x + row * dim;
    __nv_bfloat16* row_out = out + row * dim;

    // 1. Compute mean
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        thread_sum += bf16_to_f32(row_x[i]);
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    __shared__ float warp_sums[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[wid] = thread_sum;
    __syncthreads();

    int num_warps = BLOCK_SIZE / warpSize;
    float block_sum = (wid == 0 && threadIdx.x < num_warps) ? warp_sums[threadIdx.x] : 0.0f;
    if (wid == 0) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
    }
    __shared__ float s_mean;
    if (threadIdx.x == 0) s_mean = block_sum / (float)dim;
    __syncthreads();
    float mean = s_mean;

    // 2. Compute variance
    float thread_var = 0.0f;
    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        float diff = bf16_to_f32(row_x[i]) - mean;
        thread_var += diff * diff;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_var += __shfl_down_sync(0xffffffff, thread_var, offset);
    }
    if (lane == 0) warp_sums[wid] = thread_var;
    __syncthreads();

    float block_var = (wid == 0 && threadIdx.x < num_warps) ? warp_sums[threadIdx.x] : 0.0f;
    if (wid == 0) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            block_var += __shfl_down_sync(0xffffffff, block_var, offset);
        }
    }
    __shared__ float s_rstd;
    if (threadIdx.x == 0) {
        s_rstd = rsqrtf((block_var / (float)dim) + eps);
    }
    __syncthreads();
    float rstd = s_rstd;

    // 3. Normalize and scale
    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        float val = (bf16_to_f32(row_x[i]) - mean) * rstd;
        float g = gamma ? bf16_to_f32(gamma[i]) : 1.0f;
        float b = beta  ? bf16_to_f32(beta[i])  : 0.0f;
        row_out[i] = f32_to_bf16(val * g + b);
    }
}

void layernorm_affine_bf16_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* x,
    const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta,
    int n_rows,
    int dim,
    float eps,
    cudaStream_t stream)
{
    const int BLOCK_SIZE = 256;
    layernorm_affine_bf16_kernel<BLOCK_SIZE><<<n_rows, BLOCK_SIZE, 0, stream>>>(
        out, x, gamma, beta, n_rows, dim, eps);
}

// ── GELU with Bias ──────────────────────────────────────────────────────────

__global__ void gelu_bias_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    size_t total_elements,
    int dim)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int d = idx % dim;
    float v = bf16_to_f32(x[idx]);
    if (bias) v += bf16_to_f32(bias[d]);

    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float cdf = 0.5f * (1.0f + tanhf(0.79788456f * (v + 0.044715f * v * v * v)));
    out[idx] = f32_to_bf16(v * cdf);
}

void gelu_bias_bf16_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* x,
    const __nv_bfloat16* bias,
    int n_rows,
    int dim,
    cudaStream_t stream)
{
    size_t total = (size_t)n_rows * dim;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    gelu_bias_bf16_kernel<<<blocks, threads, 0, stream>>>(out, x, bias, total, dim);
}

// ── Residual Addition + Bias ────────────────────────────────────────────────

__global__ void add_residual_bias_bf16_kernel(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ bias,
    size_t total_elements,
    int dim)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int d = idx % dim;
    float v = bf16_to_f32(x[idx]) + bf16_to_f32(residual[idx]);
    if (bias) v += bf16_to_f32(bias[d]);
    x[idx] = f32_to_bf16(v);
}

void add_residual_bias_bf16_cuda(
    __nv_bfloat16* x,
    const __nv_bfloat16* residual,
    const __nv_bfloat16* bias,
    int n_rows,
    int dim,
    cudaStream_t stream)
{
    size_t total = (size_t)n_rows * dim;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    add_residual_bias_bf16_kernel<<<blocks, threads, 0, stream>>>(x, residual, bias, total, dim);
}

// ── Add Bias ────────────────────────────────────────────────────────────────

__global__ void add_bias_bf16_kernel(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    size_t total_elements,
    int dim)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int d = idx % dim;
    float v = bf16_to_f32(x[idx]) + bf16_to_f32(bias[d]);
    x[idx] = f32_to_bf16(v);
}

void add_bias_bf16_cuda(
    __nv_bfloat16* x,
    const __nv_bfloat16* bias,
    int n_rows,
    int dim,
    cudaStream_t stream)
{
    size_t total = (size_t)n_rows * dim;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    add_bias_bf16_kernel<<<blocks, threads, 0, stream>>>(x, bias, total, dim);
}

// ── Elementwise Add Tensors ─────────────────────────────────────────────────

__global__ void add_tensors_bf16_kernel(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    size_t count)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    float v = bf16_to_f32(dst[idx]) + bf16_to_f32(src[idx]);
    dst[idx] = f32_to_bf16(v);
}

void add_tensors_bf16_cuda(
    __nv_bfloat16* dst,
    const __nv_bfloat16* src,
    size_t count,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (int)((count + threads - 1) / threads);
    add_tensors_bf16_kernel<<<blocks, threads, 0, stream>>>(dst, src, count);
}

// ── ViT QKV Split and Layout Rearrange ──────────────────────────────────────

__global__ void vit_split_qkv_bias_kernel(
    __nv_bfloat16* __restrict__ Q_out,
    __nv_bfloat16* __restrict__ K_out,
    __nv_bfloat16* __restrict__ V_out,
    const __nv_bfloat16* __restrict__ qkv_in,
    const __nv_bfloat16* __restrict__ bias,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int N,
    int n_heads,
    int head_dim)
{
    int total_dim = n_heads * head_dim;
    int n = blockIdx.x; // patch index [0, N)
    if (n >= N) return;

    int h = blockIdx.y; // [0, n_heads)
    int d = threadIdx.x; // [0, head_dim)
    if (d >= head_dim) return;

    int in_base = n * (3 * total_dim);
    int head_offset = h * head_dim + d;

    float q_b = bias ? bf16_to_f32(bias[head_offset]) : 0.0f;
    float k_b = bias ? bf16_to_f32(bias[total_dim + head_offset]) : 0.0f;
    float v_b = bias ? bf16_to_f32(bias[2 * total_dim + head_offset]) : 0.0f;

    float q = bf16_to_f32(qkv_in[in_base + head_offset]) + q_b;
    float k = bf16_to_f32(qkv_in[in_base + total_dim + head_offset]) + k_b;
    float v = bf16_to_f32(qkv_in[in_base + 2 * total_dim + head_offset]) + v_b;

    // Apply Qwen 2D RoPE to Q and K if cos/sin tables provided
    if (cos_table && sin_table) {
        __shared__ float s_q[128];
        __shared__ float s_k[128];
        s_q[d] = q;
        s_k[d] = k;
        __syncthreads();

        int half_head = head_dim / 2;
        int partner = (d < half_head) ? (d + half_head) : (d - half_head);
        float sign = (d < half_head) ? -1.0f : 1.0f;
        float c = cos_table[n * head_dim + d];
        float s = sin_table[n * head_dim + d];

        q = q * c + sign * s_q[partner] * s;
        k = k * c + sign * s_k[partner] * s;
    }

    int out_idx = h * (N * head_dim) + n * head_dim + d;
    Q_out[out_idx] = f32_to_bf16(q);
    K_out[out_idx] = f32_to_bf16(k);
    V_out[out_idx] = f32_to_bf16(v);
}

void vit_split_qkv_bias_bf16_cuda(
    __nv_bfloat16* Q_out,
    __nv_bfloat16* K_out,
    __nv_bfloat16* V_out,
    const __nv_bfloat16* qkv_in,
    const __nv_bfloat16* bias,
    const float* cos_table,
    const float* sin_table,
    int N,
    int n_heads,
    int head_dim,
    cudaStream_t stream)
{
    dim3 grid(N, n_heads);
    int threads = head_dim; // 72 threads per block
    vit_split_qkv_bias_kernel<<<grid, threads, 0, stream>>>(
        Q_out, K_out, V_out, qkv_in, bias, cos_table, sin_table, N, n_heads, head_dim);
}

// ── Softmax for Attention Scores ────────────────────────────────────────────

template <int BLOCK_SIZE>
__global__ void vit_softmax_kernel(
    __nv_bfloat16* __restrict__ scores, // [total_rows, seq_len]
    int total_rows,
    int seq_len,
    float scale)
{
    int row = blockIdx.x;
    if (row >= total_rows) return;

    __nv_bfloat16* row_ptr = scores + row * seq_len;

    // 1. Find max
    float thread_max = -1e20f;
    for (int i = threadIdx.x; i < seq_len; i += BLOCK_SIZE) {
        float s = bf16_to_f32(row_ptr[i]) * scale;
        if (s > thread_max) thread_max = s;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }

    __shared__ float warp_reds[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if (lane == 0) warp_reds[wid] = thread_max;
    __syncthreads();

    int num_warps = BLOCK_SIZE / warpSize;
    float block_max = (wid == 0 && threadIdx.x < num_warps) ? warp_reds[threadIdx.x] : -1e20f;
    if (wid == 0) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            block_max = fmaxf(block_max, __shfl_down_sync(0xffffffff, block_max, offset));
        }
    }
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = block_max;
    __syncthreads();
    float max_val = s_max;

    // 2. Exponentiate and sum
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += BLOCK_SIZE) {
        float exp_val = expf(bf16_to_f32(row_ptr[i]) * scale - max_val);
        thread_sum += exp_val;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    if (lane == 0) warp_reds[wid] = thread_sum;
    __syncthreads();

    float block_sum = (wid == 0 && threadIdx.x < num_warps) ? warp_reds[threadIdx.x] : 0.0f;
    if (wid == 0) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
    }
    __shared__ float s_inv_sum;
    if (threadIdx.x == 0) s_inv_sum = 1.0f / fmaxf(block_sum, 1e-12f);
    __syncthreads();
    float inv_sum = s_inv_sum;

    // 3. Write normalized probabilities
    for (int i = threadIdx.x; i < seq_len; i += BLOCK_SIZE) {
        float p = expf(bf16_to_f32(row_ptr[i]) * scale - max_val) * inv_sum;
        row_ptr[i] = f32_to_bf16(p);
    }
}

void vit_softmax_bf16_cuda(
    __nv_bfloat16* scores,
    int total_rows,
    int seq_len,
    float scale,
    cudaStream_t stream)
{
    const int BLOCK_SIZE = 256;
    vit_softmax_kernel<BLOCK_SIZE><<<total_rows, BLOCK_SIZE, 0, stream>>>(
        scores, total_rows, seq_len, scale);
}

// ── Merge Heads [16, N, 72] -> [N, 1152] ───────────────────────────────────

__global__ void vit_merge_heads_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ heads,
    int N,
    int n_heads,
    int head_dim)
{
    int n = blockIdx.x;
    if (n >= N) return;

    int h = blockIdx.y;
    int d = threadIdx.x;
    if (d >= head_dim) return;

    int in_idx = h * (N * head_dim) + n * head_dim + d;
    int out_idx = n * (n_heads * head_dim) + h * head_dim + d;

    out[out_idx] = heads[in_idx];
}

void vit_merge_heads_bf16_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* heads,
    int N,
    int n_heads,
    int head_dim,
    cudaStream_t stream)
{
    dim3 grid(N, n_heads);
    vit_merge_heads_kernel<<<grid, head_dim, 0, stream>>>(out, heads, N, n_heads, head_dim);
}

// ── Spatial Merge Gather [48, 48, 1152] -> [24, 24, 4608] ──────────────────

__global__ void spatial_merge_gather_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ in,
    int H_merged,
    int W_merged,
    int dim)
{
    int my = blockIdx.y;
    int mx = blockIdx.x;
    if (my >= H_merged || mx >= W_merged) return;

    int m_idx = my * W_merged + mx;
    int out_base = m_idx * (4 * dim);

    int py0 = 2 * my;
    int px0 = 2 * mx;
    int W_patches = W_merged * 2;

    int p0 = py0 * W_patches + px0;
    int p1 = py0 * W_patches + (px0 + 1);
    int p2 = (py0 + 1) * W_patches + px0;
    int p3 = (py0 + 1) * W_patches + (px0 + 1);

    const __nv_bfloat16* src0 = in + p0 * dim;
    const __nv_bfloat16* src1 = in + p1 * dim;
    const __nv_bfloat16* src2 = in + p2 * dim;
    const __nv_bfloat16* src3 = in + p3 * dim;

    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        out[out_base + d]           = src0[d];
        out[out_base + dim + d]     = src1[d];
        out[out_base + 2 * dim + d] = src2[d];
        out[out_base + 3 * dim + d] = src3[d];
    }
}

void spatial_merge_gather_bf16_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* in,
    int H_merged,
    int W_merged,
    int dim,
    cudaStream_t stream)
{
    // Because input patches are already produced in Qwen block-major order
    // (each 2x2 spatial block is stored consecutively as 4 x dim features),
    // spatial merging is a direct contiguous memory copy / reshape!
    // Total elements = H_merged * W_merged * (4 * dim).
    size_t total_bytes = (size_t)H_merged * W_merged * (4 * dim) * sizeof(__nv_bfloat16);
    cudaMemcpyAsync(out, in, total_bytes, cudaMemcpyDeviceToDevice, stream);
}

// ── Conv3D Patch Embed with Bias and Positional Embeddings ──────────────────

// ── Im2Col extraction for Conv3D patch embedding ────────────────────────────

__global__ void im2col_patch_embed_kernel(
    __nv_bfloat16* __restrict__ im2col_out, // [P=2304, 1536]
    const float* __restrict__ img_norm,    // [T=2, C=3, H=768, W=768]
    int H, int W,
    int patch_size,
    int temporal_size,
    int in_channels)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t patch_dim = (size_t)in_channels * temporal_size * patch_size * patch_size; // 1536
    int W_patches = W / patch_size;
    size_t total = (size_t)(H / patch_size) * W_patches * patch_dim; // 2304 * 1536
    if (idx >= total) return;

    int p = idx / patch_dim;
    int k = idx % patch_dim;

    // Qwen Block-Major Patch Ordering:
    // Every 4 consecutive patches belong to a 2x2 spatial merge block.
    int W_blocks = W / (patch_size * 2); // 768 / 32 = 24
    int block_idx = p / 4;
    int my = block_idx / W_blocks;
    int mx = block_idx % W_blocks;
    int sub = p % 4;
    int py = my * 2 + (sub / 2);
    int px = mx * 2 + (sub % 2);

    int kw = k % patch_size;
    int rem1 = k / patch_size;
    int kh = rem1 % patch_size;
    int rem2 = rem1 / patch_size;
    int t = rem2 % temporal_size;
    int c = rem2 / temporal_size;

    int img_y = py * patch_size + kh;
    int img_x = px * patch_size + kw;

    size_t img_idx = ((size_t)t * in_channels * H * W) +
                     ((size_t)c * H * W) +
                     ((size_t)img_y * W) +
                     img_x;

    im2col_out[idx] = f32_to_bf16(img_norm[img_idx]);
}

void im2col_patch_embed_bf16_cuda(
    __nv_bfloat16* im2col_out,
    const float* img_norm,
    int H, int W,
    int patch_size,
    int temporal_size,
    int in_channels,
    cudaStream_t stream)
{
    size_t patch_dim = (size_t)in_channels * temporal_size * patch_size * patch_size; // 1536
    size_t num_patches = (size_t)(H / patch_size) * (W / patch_size); // 2304
    size_t total = num_patches * patch_dim;

    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    im2col_patch_embed_kernel<<<blocks, threads, 0, stream>>>(
        im2col_out, img_norm, H, W, patch_size, temporal_size, in_channels);
}

// ── Add Bias and Positional Embedding ───────────────────────────────────────

__global__ void add_bias_and_posemb_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ pos_embed,
    size_t total_elements,
    int dim)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int p = idx / dim;
    int d = idx % dim;

    // Map block-major patch index p to raster position row (py * 48 + px)
    int block_idx = p / 4;
    int W_blocks = 24; // 48 / 2
    int my = block_idx / W_blocks;
    int mx = block_idx % W_blocks;
    int sub = p % 4;
    int py = my * 2 + (sub / 2);
    int px = mx * 2 + (sub % 2);
    int pos_row = py * 48 + px;

    float v = bf16_to_f32(out[idx]);
    if (bias) v += bf16_to_f32(bias[d]);
    if (pos_embed) v += bf16_to_f32(pos_embed[pos_row * dim + d]);
    out[idx] = f32_to_bf16(v);
}

void add_bias_and_posemb_bf16_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* bias,
    const __nv_bfloat16* pos_embed,
    int num_patches,
    int dim,
    cudaStream_t stream)
{
    size_t total = (size_t)num_patches * dim;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    add_bias_and_posemb_kernel<<<blocks, threads, 0, stream>>>(
        out, bias, pos_embed, total, dim);
}

// ── Visual Token Scaled RMSNorm ─────────────────────────────────────────────

__global__ void rms_norm_scaled_batched_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    float target_scale,
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
        float v = bf16_to_f32(x_row[i]);
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

    float rsqrt_val = rsqrtf(sdata[0] / (float)dim + eps) * target_scale;
    for (int i = tid; i < dim; i += blockDim.x) {
        out_row[i] = f32_to_bf16(bf16_to_f32(x_row[i]) * rsqrt_val);
    }
}

void rms_norm_scaled_batched_bf16_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* x,
    float target_scale,
    int n, int dim, float eps,
    cudaStream_t stream)
{
    int threads = min(1024, ((dim + 31) / 32) * 32);
    int shared = (threads / 32) * sizeof(float);
    rms_norm_scaled_batched_kernel<<<n, threads, shared, stream>>>(
        out, x, target_scale, n, dim, eps);
}

// ── Visual Token Carrier Removal and Alignment ─────────────────────────────

__global__ void compute_col_mean_kernel(
    float* __restrict__ col_mean,
    const __nv_bfloat16* __restrict__ x,
    int n, int dim)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += bf16_to_f32(x[(size_t)i * dim + d]);
    }
    col_mean[d] = sum / (float)n;
}

__global__ void sub_col_mean_and_norm_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ col_mean,
    const __nv_bfloat16* __restrict__ embed_mean,
    float target_scale,
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
        float v = bf16_to_f32(x_row[i]) - col_mean[i];
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

    float rsqrt_val = rsqrtf(sdata[0] / (float)dim + eps) * target_scale;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = (bf16_to_f32(x_row[i]) - col_mean[i]) * rsqrt_val;
        if (embed_mean) v += bf16_to_f32(embed_mean[i]);
        out_row[i] = f32_to_bf16(v);
    }
}

void visual_carrier_removal_and_align_bf16_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* x,
    float* d_col_mean_scratch,
    const __nv_bfloat16* d_embed_mean,
    float target_scale,
    int n, int dim, float eps,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    compute_col_mean_kernel<<<blocks, threads, 0, stream>>>(d_col_mean_scratch, x, n, dim);

    int norm_threads = min(1024, ((dim + 31) / 32) * 32);
    int shared = (norm_threads / 32) * sizeof(float);
    sub_col_mean_and_norm_kernel<<<n, norm_threads, shared, stream>>>(
        out, x, d_col_mean_scratch, d_embed_mean, target_scale, n, dim, eps);
}

// ── Soft Norm Cap for Visual Tokens ──────────────────────────────────────────
// Caps token L2 norm at max_norm, preserving 100% of grounded token vectors (< max_norm)
// and clamping out-of-distribution background noise to prevent attention saturation.

__global__ void norm_cap_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    float max_norm,
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
        float v = bf16_to_f32(x_row[i]);
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

    float norm_val = sqrtf(sdata[0] + eps);
    float scale = (norm_val > max_norm) ? (max_norm / norm_val) : 1.0f;

    for (int i = tid; i < dim; i += blockDim.x) {
        out_row[i] = f32_to_bf16(bf16_to_f32(x_row[i]) * scale);
    }
}

void norm_cap_bf16_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* x,
    float max_norm,
    int n, int dim, float eps,
    cudaStream_t stream)
{
    int threads = min(1024, ((dim + 31) / 32) * 32);
    int shared = (threads / 32) * sizeof(float);
    norm_cap_bf16_kernel<<<n, threads, shared, stream>>>(
        out, x, max_norm, n, dim, eps);
}

// ── Window Attention Support Kernels ────────────────────────────────────────

__global__ void permute_patches_by_window_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ in,
    const int* __restrict__ window_index,
    int num_blocks,
    int block_elements)
{
    int b = blockIdx.x;
    if (b >= num_blocks) return;
    int src_b = window_index[b];
    const __nv_bfloat16* src_ptr = in + (size_t)src_b * block_elements;
    __nv_bfloat16* dst_ptr = out + (size_t)b * block_elements;
    for (int i = threadIdx.x; i < block_elements; i += blockDim.x) {
        dst_ptr[i] = src_ptr[i];
    }
}

void permute_patches_by_window_bf16_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* in,
    const int* window_index,
    int num_blocks,
    int block_size,
    int dim,
    cudaStream_t stream)
{
    int block_elements = block_size * dim;
    int threads = 256;
    permute_patches_by_window_kernel<<<num_blocks, threads, 0, stream>>>(
        out, in, window_index, num_blocks, block_elements);
}

__global__ void vit_split_qkv_bias_window_kernel(
    __nv_bfloat16* __restrict__ Q_out,
    __nv_bfloat16* __restrict__ K_out,
    __nv_bfloat16* __restrict__ V_out,
    const __nv_bfloat16* __restrict__ qkv_in,
    const __nv_bfloat16* __restrict__ bias,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int N,
    int n_heads,
    int head_dim,
    int window_len)
{
    int total_dim = n_heads * head_dim;
    int t = blockIdx.x; // patch index [0, N)
    if (t >= N) return;

    int h = blockIdx.y; // [0, n_heads)
    int d = threadIdx.x; // [0, head_dim)
    if (d >= head_dim) return;

    int in_base = t * (3 * total_dim);
    int head_offset = h * head_dim + d;

    float q_b = bias ? bf16_to_f32(bias[head_offset]) : 0.0f;
    float k_b = bias ? bf16_to_f32(bias[total_dim + head_offset]) : 0.0f;
    float v_b = bias ? bf16_to_f32(bias[2 * total_dim + head_offset]) : 0.0f;

    float q = bf16_to_f32(qkv_in[in_base + head_offset]) + q_b;
    float k = bf16_to_f32(qkv_in[in_base + total_dim + head_offset]) + k_b;
    float v = bf16_to_f32(qkv_in[in_base + 2 * total_dim + head_offset]) + v_b;

    // Apply Qwen 2D RoPE to Q and K if cos/sin tables provided
    if (cos_table && sin_table) {
        __shared__ float s_q[128];
        __shared__ float s_k[128];
        s_q[d] = q;
        s_k[d] = k;
        __syncthreads();

        int half_head = head_dim / 2;
        int partner = (d < half_head) ? (d + half_head) : (d - half_head);
        float sign = (d < half_head) ? -1.0f : 1.0f;
        float c = cos_table[t * head_dim + d];
        float s = sin_table[t * head_dim + d];

        q = q * c + sign * s_q[partner] * s;
        k = k * c + sign * s_k[partner] * s;
    }

    int w = t / window_len;
    int i = t % window_len;
    int batch_idx = w * n_heads + h;
    int out_idx = batch_idx * (window_len * head_dim) + i * head_dim + d;

    Q_out[out_idx] = f32_to_bf16(q);
    K_out[out_idx] = f32_to_bf16(k);
    V_out[out_idx] = f32_to_bf16(v);
}

void vit_split_qkv_bias_window_bf16_cuda(
    __nv_bfloat16* Q_out,
    __nv_bfloat16* K_out,
    __nv_bfloat16* V_out,
    const __nv_bfloat16* qkv_in,
    const __nv_bfloat16* bias,
    const float* cos_table,
    const float* sin_table,
    int N,
    int n_heads,
    int head_dim,
    int window_len,
    cudaStream_t stream)
{
    dim3 grid(N, n_heads);
    int threads = head_dim;
    vit_split_qkv_bias_window_kernel<<<grid, threads, 0, stream>>>(
        Q_out, K_out, V_out, qkv_in, bias, cos_table, sin_table, N, n_heads, head_dim, window_len);
}

__global__ void vit_merge_heads_window_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ heads,
    int N,
    int n_heads,
    int head_dim,
    int window_len)
{
    int t = blockIdx.x;
    if (t >= N) return;

    int h = blockIdx.y;
    int d = threadIdx.x;
    if (d >= head_dim) return;

    int w = t / window_len;
    int i = t % window_len;
    int batch_idx = w * n_heads + h;
    int in_idx = batch_idx * (window_len * head_dim) + i * head_dim + d;
    int out_idx = t * (n_heads * head_dim) + h * head_dim + d;

    out[out_idx] = heads[in_idx];
}

void vit_merge_heads_window_bf16_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* heads,
    int N,
    int n_heads,
    int head_dim,
    int window_len,
    cudaStream_t stream)
{
    dim3 grid(N, n_heads);
    vit_merge_heads_window_kernel<<<grid, head_dim, 0, stream>>>(
        out, heads, N, n_heads, head_dim, window_len);
}

__global__ void unpermute_merged_tokens_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ in,
    const int* __restrict__ reverse_indices,
    int num_tokens,
    int dim)
{
    int t = blockIdx.x;
    if (t >= num_tokens) return;

    int src_t = reverse_indices[t];
    const __nv_bfloat16* src_row = in + (size_t)src_t * dim;
    __nv_bfloat16* dst_row = out + (size_t)t * dim;

    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        dst_row[d] = src_row[d];
    }
}

void unpermute_merged_tokens_bf16_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* in,
    const int* reverse_indices,
    int num_tokens,
    int dim,
    cudaStream_t stream)
{
    int threads = 256;
    unpermute_merged_tokens_kernel<<<num_tokens, threads, 0, stream>>>(
        out, in, reverse_indices, num_tokens, dim);
}
