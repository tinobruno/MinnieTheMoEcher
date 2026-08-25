import sys

content = open('src/cuda/activations.cu').read()

new_kernel = """
// Highly optimized BF16 GEMV with vectorized loads (uint4 = 8 BF16 elements)
__global__ void gemv_bf16_out_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ W,
    const __nv_bfloat16* __restrict__ x,
    int N, int K)
{
    int row = blockIdx.x;
    if (row >= N) return;

    int tid = threadIdx.x;
    float sum = 0.0f;
    
    // Each row has K elements. We process 8 elements at a time.
    const uint4* w_vec4 = reinterpret_cast<const uint4*>(&W[row * K]);
    const uint4* x_vec4 = reinterpret_cast<const uint4*>(x);
    
    int k_chunks = K / 8;
    for (int i = tid; i < k_chunks; i += blockDim.x) {
        uint4 w_val = w_vec4[i];
        uint4 x_val = x_vec4[i];
        
        uint32_t w_arr[4] = {w_val.x, w_val.y, w_val.z, w_val.w};
        uint32_t x_arr[4] = {x_val.x, x_val.y, x_val.z, x_val.w};
        
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            __nv_bfloat162 w_bf2 = *reinterpret_cast<__nv_bfloat162*>(&w_arr[j]);
            __nv_bfloat162 x_bf2 = *reinterpret_cast<__nv_bfloat162*>(&x_arr[j]);
            
            float2 w_f2 = __bfloat1622float2(w_bf2);
            float2 x_f2 = __bfloat1622float2(x_bf2);
            
            sum += w_f2.x * x_f2.x + w_f2.y * x_f2.y;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    extern __shared__ float shared_sum_bf16[];
    int lane = tid % 32;
    int wid = tid / 32;

    if (lane == 0) {
        shared_sum_bf16[wid] = sum;
    }
    __syncthreads();

    if (wid == 0) {
        float val = (lane < (blockDim.x / 32)) ? shared_sum_bf16[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) {
            out[row] = __float2bfloat16(val);
        }
    }
}

void gemv_bf16_out_bf16_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* W,
    const __nv_bfloat16* x,
    int N, int K,
    cudaStream_t stream)
{
    int n_threads = 256;
    size_t smem = (n_threads / 32) * sizeof(float);
    gemv_bf16_out_bf16_kernel<<<N, n_threads, smem, stream>>>(out, W, x, N, K);
}
"""

content += new_kernel

open('src/cuda/activations.cu', 'w').write(content)

h_content = open('src/cuda/activations.cuh').read()
h_content += """
void gemv_bf16_out_bf16_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* W,
    const __nv_bfloat16* x,
    int N, int K,
    cudaStream_t stream);
"""
open('src/cuda/activations.cuh', 'w').write(h_content)
