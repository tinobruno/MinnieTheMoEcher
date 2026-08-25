#include <iostream>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__global__ void gemv_int2_kernel_old(
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
}

__global__ void gemv_int2_kernel_new(
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
            for (int j = 0; j < 4; j++) {
                uint32_t byte_val = (chunk >> (j * 8)) & 0xFF;
                
                uint32_t a_val0 = a_arr[j * 2];
                __nv_bfloat162 bf2_0 = *reinterpret_cast<__nv_bfloat162*>(&a_val0);
                float2 f2_0 = __bfloat1622float2(bf2_0);
                
                uint32_t a_val1 = a_arr[j * 2 + 1];
                __nv_bfloat162 bf2_1 = *reinterpret_cast<__nv_bfloat162*>(&a_val1);
                float2 f2_1 = __bfloat1622float2(bf2_1);

                float v0 = (float)(byte_val & 0x03);
                float v1 = (float)((byte_val >> 2) & 0x03);
                float v2 = (float)((byte_val >> 4) & 0x03);
                float v3 = (float)(byte_val >> 6);

                sum_w += v0 * f2_0.x + v1 * f2_0.y + v2 * f2_1.x + v3 * f2_1.y;
                sum_a += f2_0.x + f2_0.y + f2_1.x + f2_1.y;
            }
            sum += sum_w * s + sum_a * m;
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
}

int main() {
    int N = 4096;
    int K = 4096;
    int K_packed = K / 4;
    int block_size = 256;

    __nv_bfloat16 *d_out;
    __nv_bfloat16 *d_vec;
    uint8_t *d_weight;
    __nv_bfloat16 *d_scale;

    cudaMalloc(&d_out, N * sizeof(__nv_bfloat16));
    cudaMalloc(&d_vec, K * sizeof(__nv_bfloat16));
    cudaMalloc(&d_weight, N * K_packed * sizeof(uint8_t));
    cudaMalloc(&d_scale, N * (K / block_size) * 2 * sizeof(__nv_bfloat16));

    int blocks_x = (N + 3) / 4;
    int threads = 128;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    gemv_int2_kernel_old<<<blocks_x, threads>>>(d_out, d_vec, d_weight, d_scale, N, K_packed, block_size);
    gemv_int2_kernel_new<<<blocks_x, threads>>>(d_out, d_vec, d_weight, d_scale, N, K_packed, block_size);
    cudaDeviceSynchronize();

    int iters = 1000;

    cudaEventRecord(start);
    for(int i=0; i<iters; i++) {
        gemv_int2_kernel_old<<<blocks_x, threads>>>(d_out, d_vec, d_weight, d_scale, N, K_packed, block_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_old;
    cudaEventElapsedTime(&ms_old, start, stop);

    cudaEventRecord(start);
    for(int i=0; i<iters; i++) {
        gemv_int2_kernel_new<<<blocks_x, threads>>>(d_out, d_vec, d_weight, d_scale, N, K_packed, block_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_new;
    cudaEventElapsedTime(&ms_new, start, stop);

    std::cout << "Old: " << ms_old / iters << " ms" << std::endl;
    std::cout << "\nNew: " << ms_new / iters << " ms\n";
}
