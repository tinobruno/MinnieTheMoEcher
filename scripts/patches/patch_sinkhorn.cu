#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

__global__ void hc_split_sinkhorn_kernel_v2(
    float* __restrict__ pre,
    float* __restrict__ post,
    float* __restrict__ comb,
    const float* __restrict__ mixes,
    const float* __restrict__ scale,
    const float* __restrict__ base,
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
