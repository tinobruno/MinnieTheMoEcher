#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <iostream>

__global__ void gemv_bf16_f32_kernel(
    float* __restrict__ out,
    const __nv_bfloat16* __restrict__ vec,
    const __nv_bfloat16* __restrict__ weight,
    int N, int K)
{
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;

    const uint4* w_vec4 = reinterpret_cast<const uint4*>(&weight[(size_t)row * K]);
    const uint4* a_vec4 = reinterpret_cast<const uint4*>(vec);

    float sum = 0.0f;
    #pragma unroll 4
    for (int chunk = tid; chunk < K / 8; chunk += blockDim.x) {
        uint4 w4 = w_vec4[chunk];
        uint4 a4 = a_vec4[chunk];

        const __nv_bfloat162* w_bf16 = reinterpret_cast<const __nv_bfloat162*>(&w4);
        const __nv_bfloat162* a_bf16 = reinterpret_cast<const __nv_bfloat162*>(&a4);

        #pragma unroll
        for (int p = 0; p < 4; p++) {
            float2 w_f = __bfloat1622float2(w_bf16[p]);
            float2 a_f = __bfloat1622float2(a_bf16[p]);
            sum += w_f.x * a_f.x + w_f.y * a_f.y;
        }
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
        if (tid == 0) out[row] = sum;
    }
}

int main() {
    int dim = 4096;
    int vocab = 129280;

    __nv_bfloat16 *d_hidden, *d_head_weight;
    float *d_logits;

    cudaMalloc(&d_hidden, dim * 2);
    cudaMalloc(&d_head_weight, (size_t)vocab * dim * 2);
    cudaMalloc(&d_logits, vocab * 4);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iters = 50;

    // 1. cuBLAS
    cudaEventRecord(start, stream);
    for (int it = 0; it < iters; it++) {
        float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     vocab, 1, dim,
                     &alpha,
                     d_head_weight, CUDA_R_16BF, dim,
                     d_hidden, CUDA_R_16BF, dim,
                     &beta,
                     d_logits, CUDA_R_32F, vocab,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms_cublas = 0;
    cudaEventElapsedTime(&ms_cublas, start, stop);
    std::cout << "cuBLAS LM Head GEMM: " << ms_cublas / iters << " ms" << std::endl;

    // 2. Custom Fused GEMV
    cudaEventRecord(start, stream);
    for (int it = 0; it < iters; it++) {
        gemv_bf16_f32_kernel<<<vocab, 128, 0, stream>>>(d_logits, d_hidden, d_head_weight, vocab, dim);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms_custom = 0;
    cudaEventElapsedTime(&ms_custom, start, stop);
    std::cout << "Custom Fused 128-bit Vectorized LM Head GEMV: " << ms_custom / iters << " ms" << std::endl;

    return 0;
}
