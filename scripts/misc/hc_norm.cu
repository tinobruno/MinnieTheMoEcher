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

extern "C" void rms_norm_f32_cuda(float* x, int dim, float eps, cudaStream_t stream) {
    int threads = min(dim, 1024);
    int smem = (threads / 32) * sizeof(float);
    rms_norm_f32_kernel<<<1, threads, smem, stream>>>(x, dim, eps);
}
