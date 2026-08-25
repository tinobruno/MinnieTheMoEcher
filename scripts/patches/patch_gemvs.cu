// FP8 GEMV Kernel
__device__ inline float fp8_e4m3_to_float_v2(uint8_t val) {
    if (val == 0) return 0.0f;
    uint32_t sign = (val & 0x80) << 24;
    uint32_t exp  = (val & 0x78) >> 3;
    uint32_t mant = (val & 0x07);
    uint32_t f_exp = exp + 127 - 7;
    uint32_t f_mant = mant << 20;
    uint32_t res = sign | (f_exp << 23) | f_mant;
    return *((float*)&res);
}

__device__ inline float e8m0_to_float_v2(uint8_t val) {
    uint32_t res = (val + 127 - 127) << 23;
    return *((float*)&res);
}

__global__ void gemv_fp8_kernel(
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

    // Process 8 elements at a time
    const uint2* w_vec8 = reinterpret_cast<const uint2*>(&weight[row * K]);
    const uint4* a_vec4 = reinterpret_cast<const uint4*>(vec); // uint4 = 16 bytes = 8 bfloat16

    for (int chunk = tid; chunk < K / 8; chunk += blockDim.x) {
        int logical_col = chunk * 8;
        int bc = logical_col / block_size;
        float s_val = e8m0_to_float_v2(scale[br * scale_cols + bc]);

        uint2 w8 = w_vec8[chunk];
        uint4 a8 = a_vec4[chunk];
        uint32_t w_arr[2] = {w8.x, w8.y};
        uint32_t a_arr[4] = {a8.x, a8.y, a8.z, a8.w};

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            uint32_t w_chunk = w_arr[i];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint8_t b = (w_chunk >> (j * 8)) & 0xFF;
                float w_f = fp8_e4m3_to_float_v2(b) * s_val;
                int a_idx = i * 2 + (j / 2);
                uint32_t a_val = a_arr[a_idx];
                __nv_bfloat162 bf2 = *reinterpret_cast<__nv_bfloat162*>(&a_val);
                float2 f2 = __bfloat1622float2(bf2);
                if ((j % 2) == 0) sum += w_f * f2.x;
                else sum += w_f * f2.y;
            }
        }
    }
    // Warp reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    // Shared memory for block reduce
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
void gemv_fp8_cuda(__nv_bfloat16* out, const __nv_bfloat16* vec,
                   const uint8_t* weight, const uint8_t* scale,
                   int N, int K, int block_size, cudaStream_t stream) {
    int threads = 128;
    gemv_fp8_kernel<<<N, threads, 0, stream>>>(out, vec, weight, scale, N, K, block_size);
}

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
    int scale_cols = (K_packed * 4 + block_size - 1) / block_size;
    int br = row / block_size;

    const __nv_bfloat16* row_scales = &scale_min[(br * scale_cols) * 2];
    const __nv_bfloat16* row_mins   = &scale_min[(br * scale_cols) * 2 + 1];

    const uint4* w_vec4 = reinterpret_cast<const uint4*>(&weight[row * K_packed]);
    const uint4* a_vec4 = reinterpret_cast<const uint4*>(vec);

    for (int cp_chunk = tid; cp_chunk < K_packed / 16; cp_chunk += blockDim.x) {
        int logical_col = cp_chunk * 64;
        int bc = logical_col / block_size;

        float s = __bfloat162float(row_scales[bc * 2]);
        float m = __bfloat162float(row_mins[bc * 2]);

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
                    v0 = (float)(b & 0x03) * s + m;
                    v1 = (float)((b >> 2) & 0x03) * s + m;
                } else {
                    v0 = (float)((b >> 4) & 0x03) * s + m;
                    v1 = (float)((b >> 6) & 0x03) * s + m;
                }
                sum += v0 * f2.x + v1 * f2.y;
            }
        }
    }
    // Warp reduce
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
void gemv_int2_cuda(__nv_bfloat16* out, const __nv_bfloat16* vec,
                    const uint8_t* weight, const __nv_bfloat16* scale_min,
                    int N, int K_packed, int block_size, cudaStream_t stream) {
    int threads = 128;
    gemv_int2_kernel<<<N, threads, 0, stream>>>(out, vec, weight, scale_min, N, K_packed, block_size);
}
