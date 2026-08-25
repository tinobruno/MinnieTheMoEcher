#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <iostream>

__global__ void gemv_bf16_f32_gate_kernel(
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
    int N = 256;

    __nv_bfloat16 *d_vec, *d_w, *d_out_bf16;
    float *d_out_f32;
    cudaMalloc(&d_vec, dim * 2);
    cudaMalloc(&d_w, (size_t)N * dim * 2);
    cudaMalloc(&d_out_bf16, N * 2);
    cudaMalloc(&d_out_f32, N * 4);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iters = 100;
    // 40 layers of cuBLAS
    cudaEventRecord(start, stream);
    for (int it = 0; it < iters; it++) {
        for (int l = 0; l < 40; l++) {
            float alpha = 1.0f, beta = 0.0f;
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                         N, 1, dim,
                         &alpha,
                         d_w, CUDA_R_16BF, dim,
                         d_vec, CUDA_R_16BF, dim,
                         &beta,
                         d_out_bf16, CUDA_R_16BF, N,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        }
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms_cublas = 0;
    cudaEventElapsedTime(&ms_cublas, start, stop);
    std::cout << "40 layers cuBLAS gate_w: " << ms_cublas / iters << " ms" << std::endl;

    // 40 layers of custom gemv
    cudaEventRecord(start, stream);
    for (int it = 0; it < iters; it++) {
        for (int l = 0; l < 40; l++) {
            gemv_bf16_f32_gate_kernel<<<N, 128, 0, stream>>>(d_out_f32, d_vec, d_w, N, dim);
        }
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms_custom = 0;
    cudaEventElapsedTime(&ms_custom, start, stop);
    std::cout << "40 layers Custom gemv_bf16_f32 gate_w: " << ms_custom / iters << " ms" << std::endl;

    return 0;
}
