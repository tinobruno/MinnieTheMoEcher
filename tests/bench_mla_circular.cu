#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include <vector>
#include <cmath>

__global__ void mla_attention_circular_kernel(
    const __nv_bfloat16* __restrict__ q,          // [n_heads, head_dim]
    const __nv_bfloat16* __restrict__ raw_kv,     // [window, head_dim]
    const __nv_bfloat16* __restrict__ comp_kv,    // [max_comp, head_dim] (nullable)
    const int32_t* __restrict__ d_comp_count,     // [1] on GPU device (nullable)
    const float* __restrict__ attn_sink,          // [n_heads]
    __nv_bfloat16* __restrict__ out,              // [n_heads, head_dim]
    const int32_t* __restrict__ d_pos,            // [1] on GPU device
    int window,
    int head_dim,
    float scale
) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int n_threads = blockDim.x;

    int pos = *d_pos;
    int raw_len = (pos + 1 < window) ? (pos + 1) : window;
    int raw_start = (pos + 1 > window) ? ((pos + 1) % window) : 0;
    int comp_len = (comp_kv && d_comp_count && pos + 1 > window) ? *d_comp_count : 0;
    int total_len = raw_len + comp_len;

    extern __shared__ float s_mem[];
    float* scores = s_mem; // [total_len]

    // Load Q for this head into shared memory
    __shared__ float s_q[512];
    for (int d = tid; d < head_dim; d += n_threads) {
        s_q[d] = __bfloat162float(q[h * head_dim + d]);
    }
    __syncthreads();

    // 1. Q @ raw_KV
    for (int t = tid; t < raw_len; t += n_threads) {
        int slot = (raw_start + t) % window;
        const __nv_bfloat16* kv_t = raw_kv + (size_t)slot * head_dim;
        float dot = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < head_dim; d++) {
            dot += s_q[d] * __bfloat162float(kv_t[d]);
        }
        scores[t] = dot * scale;
    }

    // 1b. Q @ comp_KV
    for (int c = tid; c < comp_len; c += n_threads) {
        const __nv_bfloat16* kv_c = comp_kv + (size_t)c * head_dim;
        float dot = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < head_dim; d++) {
            dot += s_q[d] * __bfloat162float(kv_c[d]);
        }
        scores[raw_len + c] = dot * scale;
    }
    __syncthreads();

    // 2. Softmax max reduction
    float local_max = -1e38f;
    for (int t = tid; t < total_len; t += n_threads) {
        local_max = fmaxf(local_max, scores[t]);
    }
    if (tid == 0) {
        local_max = fmaxf(local_max, attn_sink[h]);
    }

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
    for (int t = tid; t < total_len; t += n_threads) {
        float e = expf(scores[t] - block_max);
        scores[t] = e;
        local_sum += e;
    }
    if (tid == 0) {
        local_sum += expf(attn_sink[h] - block_max);
    }

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

    for (int t = tid; t < total_len; t += n_threads) {
        scores[t] *= inv_sum;
    }
    __syncthreads();

    // 3. Out = scores @ (raw_KV + comp_KV)
    for (int d = tid; d < head_dim; d += n_threads) {
        float v = 0.0f;
        for (int t = 0; t < raw_len; t++) {
            int slot = (raw_start + t) % window;
            v += scores[t] * __bfloat162float(raw_kv[(size_t)slot * head_dim + d]);
        }
        for (int c = 0; c < comp_len; c++) {
            v += scores[raw_len + c] * __bfloat162float(comp_kv[(size_t)c * head_dim + d]);
        }
        out[(size_t)h * head_dim + d] = __float2bfloat16(v);
    }
}

int main() {
    int n_heads = 64;
    int head_dim = 512;
    int window = 128;
    int pos_val = 250;

    __nv_bfloat16 *d_q, *d_raw_kv, *d_comp_kv, *d_out;
    float *d_sink;
    int32_t *d_pos, *d_comp_count;

    cudaMalloc(&d_q, n_heads * head_dim * 2);
    cudaMalloc(&d_raw_kv, window * head_dim * 2);
    cudaMalloc(&d_comp_kv, 100 * head_dim * 2);
    cudaMalloc(&d_out, n_heads * head_dim * 2);
    cudaMalloc(&d_sink, n_heads * 4);
    cudaMalloc(&d_pos, 4);
    cudaMalloc(&d_comp_count, 4);

    cudaMemset(d_q, 0, n_heads * head_dim * 2);
    cudaMemset(d_raw_kv, 0, window * head_dim * 2);
    cudaMemset(d_comp_kv, 0, 100 * head_dim * 2);
    cudaMemset(d_sink, 0, n_heads * 4);

    int comp_count_val = 10;
    cudaMemcpy(d_pos, &pos_val, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_comp_count, &comp_count_val, 4, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int iters = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    size_t smem = (window + 100) * sizeof(float);
    cudaEventRecord(start, stream);
    for (int i = 0; i < iters; i++) {
        mla_attention_circular_kernel<<<n_heads, 256, smem, stream>>>(
            d_q, d_raw_kv, d_comp_kv, d_comp_count, d_sink, d_out, d_pos, window, head_dim, 0.044f
        );
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "mla_attention_circular latency per layer: " << ms / iters << " ms" << std::endl;
    std::cout << "43 layers total MLA: " << (ms / iters) * 43.0f << " ms" << std::endl;

    return 0;
}
