#include <cuda_bf16.h>
#include <stdint.h>

__global__ void gemv_int2_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ vec,
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ scale_min,
    int N, int K_packed, int block_size)
{
    int row = blockIdx.x;
    if (row >= N) return;

    int tid = threadIdx.x;
    float sum = 0.0f;
    int blocks = (K_packed * 4) / block_size;
    
    const __nv_bfloat16* row_scales = scale_min + (row * blocks);
    const __nv_bfloat16* row_mins = scale_min + (N * blocks) + (row * blocks);

    const uint4* w_vec4 = reinterpret_cast<const uint4*>(&weight[row * K_packed]);
    const uint4* a_vec4 = reinterpret_cast<const uint4*>(vec); 

    for (int cp_chunk = tid; cp_chunk < K_packed / 16; cp_chunk += blockDim.x) {
        int logical_col = cp_chunk * 64;
        int bc = logical_col / block_size;
        
        float s = __bfloat162float(row_scales[bc]);
        float m = __bfloat162float(row_mins[bc]);

        uint4 w4 = w_vec4[cp_chunk];
        uint32_t w_chunks[4] = {w4.x, w4.y, w4.z, w4.w};
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint32_t chunk = w_chunks[i]; 
            
            uint4 a0 = a_vec4[cp_chunk * 8 + i * 2];
            uint4 a1 = a_vec4[cp_chunk * 8 + i * 2 + 1];
            
            uint32_t a_arr[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            
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
                    v0 = (b & 0x03) * s + m;
                    v1 = ((b >> 2) & 0x03) * s + m;
                } else {
                    v0 = ((b >> 4) & 0x03) * s + m;
                    v1 = ((b >> 6) & 0x03) * s + m;
                }
                
                sum += v0 * f2.x + v1 * f2.y;
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    static __shared__ float shared_sum[32];
    int lane = tid % 32;
    int wid = tid / 32;

    if (lane == 0) {
        shared_sum[wid] = sum;
    }
    __syncthreads();

    if (wid == 0) {
        float val = (lane < (blockDim.x / 32)) ? shared_sum[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) {
            out[row] = __float2bfloat16(val);
        }
    }
}
