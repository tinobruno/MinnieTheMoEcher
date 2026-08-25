#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include "../src/cuda/activations.cuh"

__global__ void gemv_fp8_opt_kernel(
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

    // Process 16 elements at a time (uint4 = 16 bytes = 16 FP8 weights)
    const uint4* w_vec16 = reinterpret_cast<const uint4*>(&weight[(size_t)row * K]);
    const uint4* a_vec8  = reinterpret_cast<const uint4*>(vec); // uint4 = 16 bytes = 8 BF16 elements

    int chunks = K / 16;
    for (int chunk = tid; chunk < chunks; chunk += blockDim.x) {
        int logical_col = chunk * 16;
        int bc = logical_col / block_size;
        float s_val = __uint_as_float((uint32_t)scale[br * scale_cols + bc] << 23);

        uint4 w16 = w_vec16[chunk];
        uint4 a0 = a_vec8[chunk * 2 + 0];
        uint4 a1 = a_vec8[chunk * 2 + 1];

        const uint8_t* wb = reinterpret_cast<const uint8_t*>(&w16);
        const __nv_bfloat162* a_bf16_0 = reinterpret_cast<const __nv_bfloat162*>(&a0);
        const __nv_bfloat162* a_bf16_1 = reinterpret_cast<const __nv_bfloat162*>(&a1);

        float chunk_sum = 0.0f;
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            float2 af = __bfloat1622float2(a_bf16_0[p]);
            uint8_t b0 = wb[p * 2 + 0];
            uint8_t b1 = wb[p * 2 + 1];
            float w0 = (b0 == 0) ? 0.0f : __uint_as_float(((uint32_t)(b0 & 0x80) << 24) | (((uint32_t)(b0 & 0x78) >> 3) + 120 << 23) | ((uint32_t)(b0 & 0x07) << 20));
            float w1 = (b1 == 0) ? 0.0f : __uint_as_float(((uint32_t)(b1 & 0x80) << 24) | (((uint32_t)(b1 & 0x78) >> 3) + 120 << 23) | ((uint32_t)(b1 & 0x07) << 20));
            chunk_sum += w0 * af.x + w1 * af.y;
        }
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            float2 af = __bfloat1622float2(a_bf16_1[p]);
            uint8_t b0 = wb[8 + p * 2 + 0];
            uint8_t b1 = wb[8 + p * 2 + 1];
            float w0 = (b0 == 0) ? 0.0f : __uint_as_float(((uint32_t)(b0 & 0x80) << 24) | (((uint32_t)(b0 & 0x78) >> 3) + 120 << 23) | ((uint32_t)(b0 & 0x07) << 20));
            float w1 = (b1 == 0) ? 0.0f : __uint_as_float(((uint32_t)(b1 & 0x80) << 24) | (((uint32_t)(b1 & 0x78) >> 3) + 120 << 23) | ((uint32_t)(b1 & 0x07) << 20));
            chunk_sum += w0 * af.x + w1 * af.y;
        }

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
        if (tid == 0) out[row] = __float2bfloat16(sum);
    }
}

int main() {
    int N = 32768, K = 1536;
    __nv_bfloat16 *d_out, *d_vec;
    uint8_t *d_w, *d_s;
    cudaMalloc(&d_out, N * 2);
    cudaMalloc(&d_vec, K * 2);
    cudaMalloc(&d_w, (size_t)N * K);
    cudaMalloc(&d_s, (size_t)N * (K / 128));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iters = 100;
    // Current
    cudaEventRecord(start, stream);
    for (int i = 0; i < iters; i++) gemv_fp8_cuda(d_out, d_vec, d_w, d_s, N, K, 128, stream);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms_curr = 0;
    cudaEventElapsedTime(&ms_curr, start, stop);
    std::cout << "Current gemv_fp8_cuda: " << ms_curr / iters << " ms" << std::endl;

    // Optimized 16-byte
    cudaEventRecord(start, stream);
    for (int i = 0; i < iters; i++) gemv_fp8_opt_kernel<<<N, 128, 0, stream>>>(d_out, d_vec, d_w, d_s, N, K, 128);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms_opt = 0;
    cudaEventElapsedTime(&ms_opt, start, stop);
    std::cout << "Optimized 16-byte gemv_fp8: " << ms_opt / iters << " ms" << std::endl;

    return 0;
}
