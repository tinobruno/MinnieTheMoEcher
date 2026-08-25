#include <iostream>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__global__ void fp4_dequant_kernel(__nv_bfloat16* __restrict__ out,
                                   const uint8_t* __restrict__ in_packed,
                                   const __nv_bfloat16* __restrict__ scale,
                                   int M, int N_packed, int scale_cols) {
    int row = blockIdx.x;
    int col_packed = threadIdx.x;
    
    if (row >= M || col_packed >= N_packed) return;
    
    uint8_t packed = in_packed[row * N_packed + col_packed];
    
    float v0 = (float)(packed & 0x0F) - 8.0f;
    float v1 = (float)((packed >> 4) & 0x0F) - 8.0f;
    
    int scale_col = (col_packed * 2) / 32; 
    if (scale_col >= scale_cols) scale_col = scale_cols - 1;
    
    float s = __bfloat162float(scale[row * scale_cols + scale_col]);
    
    out[row * (N_packed * 2) + col_packed * 2] = __float2bfloat16(v0 * s);
    out[row * (N_packed * 2) + col_packed * 2 + 1] = __float2bfloat16(v1 * s);
}

int main() {}
