void gemv_mixed_moe_down_batch_cuda(
    __nv_bfloat16* down_buf,
    const __nv_bfloat16* gate_buf,
    const int32_t* topk_ids,
    const void* const* flat_expert_ptrs,
    const uint8_t* expert_type_map,
    int layer_id, int n_experts,
    int w2_cold_offset, int N, int K, int M,
    cudaStream_t stream)
{
    gemv_mixed_moe_down_grouped_pass_cuda(
        down_buf, gate_buf, flat_expert_ptrs,
        layer_id, n_experts, w2_cold_offset, N, K, M, true, stream);
    gemv_mixed_moe_down_grouped_pass_cuda(
        down_buf, gate_buf, flat_expert_ptrs,
        layer_id, n_experts, w2_cold_offset, N, K, M, false, stream);
}

// ════════════════════════════════════════════════════════════════════════════════
//  Blackwell sm_120a Native FP8 Tensor Core Accelerators
// ════════════════════════════════════════════════════════════════════════════════

__global__ void quantize_bf16_to_fp8_e4m3_kernel(
    uint8_t* __restrict__ out,
    const __nv_bfloat16* __restrict__ in,
    int total_elements)
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (idx >= total_elements) return;

    int remaining = total_elements - idx;
    if (remaining >= 8) {
        const uint4* in_u4 = reinterpret_cast<const uint4*>(in + idx);
        uint4 raw = *in_u4;
        const __nv_bfloat16* bf = reinterpret_cast<const __nv_bfloat16*>(&raw);

        __nv_fp8_e4m3 fp8_vals[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            fp8_vals[i] = __nv_fp8_e4m3(bf[i]);
        }

        uint64_t* out_u64 = reinterpret_cast<uint64_t*>(out + idx);
        *out_u64 = *reinterpret_cast<const uint64_t*>(&fp8_vals[0]);
    } else {
        for (int i = 0; i < remaining; i++) {
            out[idx + i] = (uint8_t)__nv_fp8_e4m3(in[idx + i]);
        }
    }
}

void quantize_bf16_to_fp8_e4m3_cuda(
    uint8_t* out,
    const __nv_bfloat16* in,
    int total_elements,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (total_elements / 8 + threads - 1) / threads;
    if (blocks == 0 && total_elements > 0) blocks = 1;
    quantize_bf16_to_fp8_e4m3_kernel<<<blocks, threads, 0, stream>>>(out, in, total_elements);
}

__global__ void dequant_int4_to_fp8_block_kernel(
    uint8_t* __restrict__ out,               // [N, K] FP8
    const uint8_t* __restrict__ weight,      // [N, K/2] packed INT4
    const __nv_bfloat16* __restrict__ scale, // [N, K/32] BF16
    int total_blocks)
{
    int warp_id = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    int warp_start_block = warp_id * 8;
    if (warp_start_block >= total_blocks) return;

    int cur_blocks = min(8, total_blocks - warp_start_block);

    uint32_t w = 0;
    if (lane < cur_blocks * 4) {
        const uint32_t* w32_ptr = reinterpret_cast<const uint32_t*>(weight + (size_t)warp_start_block * 16);
        w = w32_ptr[lane];
    }

    float s = 0.0f;
    if (lane < cur_blocks) {
        s = __bfloat162float(scale[warp_start_block + lane]);
    }

    int b = lane >> 2; // block 0..7
    int i = lane & 3;  // sub-block 0..3 (8 weights)

    uint32_t chunk = __shfl_sync(0xFFFFFFFF, w, b * 4 + i);
    float s_val = __shfl_sync(0xFFFFFFFF, s, b);

    uint32_t b0 = chunk & 0xFF;
    uint32_t b1 = (chunk >> 8) & 0xFF;
    uint32_t b2 = (chunk >> 16) & 0xFF;
    uint32_t b3 = (chunk >> 24) & 0xFF;

    float w0 = ((float)(b0 & 0x0F) - 8.0f) * s_val;
    float w1 = ((float)(b0 >> 4) - 8.0f) * s_val;
    float w2 = ((float)(b1 & 0x0F) - 8.0f) * s_val;
    float w3 = ((float)(b1 >> 4) - 8.0f) * s_val;
    float w4 = ((float)(b2 & 0x0F) - 8.0f) * s_val;
    float w5 = ((float)(b2 >> 4) - 8.0f) * s_val;
    float w6 = ((float)(b3 & 0x0F) - 8.0f) * s_val;
    float w7 = ((float)(b3 >> 4) - 8.0f) * s_val;

    __nv_fp8_e4m3 fp8_vals[8];
    fp8_vals[0] = __nv_fp8_e4m3(w0);
    fp8_vals[1] = __nv_fp8_e4m3(w1);
    fp8_vals[2] = __nv_fp8_e4m3(w2);
    fp8_vals[3] = __nv_fp8_e4m3(w3);
    fp8_vals[4] = __nv_fp8_e4m3(w4);
    fp8_vals[5] = __nv_fp8_e4m3(w5);
    fp8_vals[6] = __nv_fp8_e4m3(w6);
    fp8_vals[7] = __nv_fp8_e4m3(w7);

    if (b < cur_blocks) {
        uint64_t* out_u64 = reinterpret_cast<uint64_t*>(out + (size_t)warp_start_block * 32);
        out_u64[lane] = *reinterpret_cast<const uint64_t*>(&fp8_vals[0]);
    }
}

void dequant_int4_to_fp8_block_cuda(
    uint8_t* out,
    const uint8_t* weight,
    const __nv_bfloat16* scale,
    int N, int K,
    int block_size,
    cudaStream_t stream)
{
    int blocks_per_row = K / 32;
    int total_blocks = N * blocks_per_row;
    int threads = 256;
    int total_warps = (total_blocks + 7) / 8;
    int blocks = (total_warps + (threads / 32) - 1) / (threads / 32);
    dequant_int4_to_fp8_block_kernel<<<blocks, threads, 0, stream>>>(out, weight, scale, total_blocks);
}

__global__ void dequant_int3_to_fp8_block_kernel(
    uint8_t* __restrict__ out,               // [N, K] FP8
    const uint8_t* __restrict__ weight,      // packed INT3 (12 bytes per block)
    const __nv_bfloat16* __restrict__ scale, // [N, K/32] BF16
    int total_blocks)
{
    int warp_id = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    int warp_start_block = warp_id * 8;
    if (warp_start_block >= total_blocks) return;

    int cur_blocks = min(8, total_blocks - warp_start_block);

    uint32_t p = 0;
    if (lane < cur_blocks * 3) {
        const uint32_t* p32_ptr = reinterpret_cast<const uint32_t*>(weight + (size_t)warp_start_block * 12);
        p = p32_ptr[lane];
    }

    float s = 0.0f;
    if (lane < cur_blocks) {
        s = __bfloat162float(scale[warp_start_block + lane]);
    }

    int b = lane >> 2;
    int i = lane & 3;

    uint32_t p0 = __shfl_sync(0xFFFFFFFF, p, b * 3 + 0);
    uint32_t p1 = __shfl_sync(0xFFFFFFFF, p, b * 3 + 1);
    uint32_t p2 = __shfl_sync(0xFFFFFFFF, p, b * 3 + 2);
    float s_val = __shfl_sync(0xFFFFFFFF, s, b);

    uint32_t b0, b1, b2;
    if (i == 0) {
        b0 = p0 & 0xFF;
        b1 = (p0 >> 8) & 0xFF;
        b2 = (p0 >> 16) & 0xFF;
    } else if (i == 1) {
        b0 = (p0 >> 24) & 0xFF;
        b1 = p1 & 0xFF;
        b2 = (p1 >> 8) & 0xFF;
    } else if (i == 2) {
        b0 = (p1 >> 16) & 0xFF;
        b1 = (p1 >> 24) & 0xFF;
        b2 = p2 & 0xFF;
    } else {
        b0 = (p2 >> 8) & 0xFF;
        b1 = (p2 >> 16) & 0xFF;
        b2 = (p2 >> 24) & 0xFF;
    }

    float w0 = ((float)(b0 & 0x07) - 4.0f) * s_val;
    float w1 = ((float)((b0 >> 3) & 0x07) - 4.0f) * s_val;
    float w2 = ((float)(((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2)) - 4.0f) * s_val;
    float w3 = ((float)((b1 >> 1) & 0x07) - 4.0f) * s_val;
    float w4 = ((float)((b1 >> 4) & 0x07) - 4.0f) * s_val;
    float w5 = ((float)(((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1)) - 4.0f) * s_val;
    float w6 = ((float)((b2 >> 2) & 0x07) - 4.0f) * s_val;
    float w7 = ((float)((b2 >> 5) & 0x07) - 4.0f) * s_val;

    __nv_fp8_e4m3 fp8_vals[8];
    fp8_vals[0] = __nv_fp8_e4m3(w0);
    fp8_vals[1] = __nv_fp8_e4m3(w1);
    fp8_vals[2] = __nv_fp8_e4m3(w2);
    fp8_vals[3] = __nv_fp8_e4m3(w3);
    fp8_vals[4] = __nv_fp8_e4m3(w4);
    fp8_vals[5] = __nv_fp8_e4m3(w5);
    fp8_vals[6] = __nv_fp8_e4m3(w6);
    fp8_vals[7] = __nv_fp8_e4m3(w7);

    if (b < cur_blocks) {
        uint64_t* out_u64 = reinterpret_cast<uint64_t*>(out + (size_t)warp_start_block * 32);
        out_u64[lane] = *reinterpret_cast<const uint64_t*>(&fp8_vals[0]);
    }
}

void dequant_int3_to_fp8_block_cuda(
    uint8_t* out,
    const uint8_t* weight,
    const __nv_bfloat16* scale,
    int N, int K,
    int block_size,
    cudaStream_t stream)
{
    int blocks_per_row = K / 32;
    int total_blocks = N * blocks_per_row;
    int threads = 256;
    int total_warps = (total_blocks + 7) / 8;
    int blocks = (total_warps + (threads / 32) - 1) / (threads / 32);
    dequant_int3_to_fp8_block_kernel<<<blocks, threads, 0, stream>>>(out, weight, scale, total_blocks);
}

// ════════════════════════════════════════════════════════════════════════════════
//  Speculative Tree Attention Implementations
// ════════════════════════════════════════════════════════════════════════════════

__device__ __forceinline__ void update_top2_dev(float val, int32_t idx, float& m1, int32_t& i1, float& m2, int32_t& i2) {
    if (idx == i1) {
        if (val > m1) m1 = val;
        return;
    }
    if (idx == i2) {
        if (val > m2) m2 = val;
        if (m2 > m1) {
            float tm = m1; int32_t ti = i1;
            m1 = m2; i1 = i2;
            m2 = tm; i2 = ti;
        }
        return;
    }
    if (val > m1) {
        m2 = m1;
        i2 = i1;
        m1 = val;
        i1 = idx;
    } else if (val > m2) {
        m2 = val;
        i2 = idx;
    }
}

__device__ __forceinline__ void merge_top2_dev(float om1, int32_t oi1, float om2, int32_t oi2,
                                               float& m1, int32_t& i1, float& m2, int32_t& i2) {
    update_top2_dev(om1, oi1, m1, i1, m2, i2);
    update_top2_dev(om2, oi2, m1, i1, m2, i2);
}

__global__ void top2_f32_stage1_kernel(
    float* __restrict__ block_maxes_1,
    int32_t* __restrict__ block_indices_1,
    float* __restrict__ block_maxes_2,
    int32_t* __restrict__ block_indices_2,
    const float* __restrict__ logits,
    int n)
{
    __shared__ float s_m1[32], s_m2[32];
    __shared__ int32_t s_i1[32], s_i2[32];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float m1 = -1e38f, m2 = -1e38f;
    int32_t i1 = 0, i2 = 0;

    for (int i = gid; i < n; i += stride) {
        update_top2_dev(logits[i], (int32_t)i, m1, i1, m2, i2);
    }

    // Warp reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float om1 = __shfl_down_sync(0xffffffff, m1, offset);
        int32_t oi1 = __shfl_down_sync(0xffffffff, i1, offset);
        float om2 = __shfl_down_sync(0xffffffff, m2, offset);
        int32_t oi2 = __shfl_down_sync(0xffffffff, i2, offset);
        merge_top2_dev(om1, oi1, om2, oi2, m1, i1, m2, i2);
    }

    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) {
        s_m1[warp] = m1; s_i1[warp] = i1;
        s_m2[warp] = m2; s_i2[warp] = i2;
    }
    __syncthreads();

    if (warp == 0) {
        int n_warps = blockDim.x / 32;
        m1 = (lane < n_warps) ? s_m1[lane] : -1e38f;
        i1 = (lane < n_warps) ? s_i1[lane] : 0;
        m2 = (lane < n_warps) ? s_m2[lane] : -1e38f;
        i2 = (lane < n_warps) ? s_i2[lane] : 0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float om1 = __shfl_down_sync(0xffffffff, m1, offset);
            int32_t oi1 = __shfl_down_sync(0xffffffff, i1, offset);
            float om2 = __shfl_down_sync(0xffffffff, m2, offset);
            int32_t oi2 = __shfl_down_sync(0xffffffff, i2, offset);
            merge_top2_dev(om1, oi1, om2, oi2, m1, i1, m2, i2);
        }
        if (lane == 0) {
            block_maxes_1[blockIdx.x] = m1;
            block_indices_1[blockIdx.x] = i1;
            block_maxes_2[blockIdx.x] = m2;
            block_indices_2[blockIdx.x] = i2;
        }
    }
}

__global__ void top2_f32_stage2_kernel(
    int32_t* __restrict__ out_idx,
    float* __restrict__ out_vals,
    const float* __restrict__ block_maxes_1,
    const int32_t* __restrict__ block_indices_1,
    const float* __restrict__ block_maxes_2,
    const int32_t* __restrict__ block_indices_2,
    int n_blocks)
{
    __shared__ float s_m1[32], s_m2[32];
    __shared__ int32_t s_i1[32], s_i2[32];

    int tid = threadIdx.x;
    float m1 = -1e38f, m2 = -1e38f;
    int32_t i1 = 0, i2 = 0;

    for (int i = tid; i < n_blocks; i += blockDim.x) {
        update_top2_dev(block_maxes_1[i], block_indices_1[i], m1, i1, m2, i2);
        update_top2_dev(block_maxes_2[i], block_indices_2[i], m1, i1, m2, i2);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float om1 = __shfl_down_sync(0xffffffff, m1, offset);
        int32_t oi1 = __shfl_down_sync(0xffffffff, i1, offset);
        float om2 = __shfl_down_sync(0xffffffff, m2, offset);
        int32_t oi2 = __shfl_down_sync(0xffffffff, i2, offset);
        merge_top2_dev(om1, oi1, om2, oi2, m1, i1, m2, i2);
    }

    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) {
        s_m1[warp] = m1; s_i1[warp] = i1;
        s_m2[warp] = m2; s_i2[warp] = i2;
    }
    __syncthreads();

    if (warp == 0) {
        int n_warps = (blockDim.x + 31) / 32;
        m1 = (lane < n_warps) ? s_m1[lane] : -1e38f;
        i1 = (lane < n_warps) ? s_i1[lane] : 0;
        m2 = (lane < n_warps) ? s_m2[lane] : -1e38f;
        i2 = (lane < n_warps) ? s_i2[lane] : 0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float om1 = __shfl_down_sync(0xffffffff, m1, offset);
            int32_t oi1 = __shfl_down_sync(0xffffffff, i1, offset);
            float om2 = __shfl_down_sync(0xffffffff, m2, offset);
            int32_t oi2 = __shfl_down_sync(0xffffffff, i2, offset);
            merge_top2_dev(om1, oi1, om2, oi2, m1, i1, m2, i2);
        }
        if (lane == 0) {
            if (out_idx != nullptr) {
                out_idx[0] = i1;
                out_idx[1] = i2;
            }
            if (out_vals != nullptr) {
                out_vals[0] = m1;
                out_vals[1] = m2;
            }
        }
    }
}

void top2_f32_cuda(int32_t* out_indices, float* out_values, const float* logits, int n, cudaStream_t stream) {
    static float* s_d_bm1 = nullptr;
    static int32_t* s_d_bi1 = nullptr;
    static float* s_d_bm2 = nullptr;
    static int32_t* s_d_bi2 = nullptr;
    static const int N_BLOCKS = 128;
    if (!s_d_bm1) {
        cudaMalloc(&s_d_bm1, N_BLOCKS * sizeof(float));
        cudaMalloc(&s_d_bi1, N_BLOCKS * sizeof(int32_t));
        cudaMalloc(&s_d_bm2, N_BLOCKS * sizeof(float));
        cudaMalloc(&s_d_bi2, N_BLOCKS * sizeof(int32_t));
    }
    top2_f32_stage1_kernel<<<N_BLOCKS, 256, 0, stream>>>(s_d_bm1, s_d_bi1, s_d_bm2, s_d_bi2, logits, n);
    top2_f32_stage2_kernel<<<1, 128, 0, stream>>>(out_indices, out_values, s_d_bm1, s_d_bi1, s_d_bm2, s_d_bi2, N_BLOCKS);
}

// ── 2D Ancestor-Masked GQA Attention Kernel ───────────────────────────────────

__global__ void qwen_gqa_write_kv_tree_fp8_batch_kernel(
    __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ k_norm_w,
    uint8_t* __restrict__ k_cache,
    uint8_t* __restrict__ v_cache,
    int n_kv_heads,
    int head_dim,
    const int32_t* __restrict__ d_pos,
    int pos_base,
    int M,
    int max_seq_len,
    float rope_theta,
    float eps)
{
    int kv_head = blockIdx.x;
    int m = blockIdx.y;
    if (kv_head >= n_kv_heads || m >= M) return;

    int pos_rope = d_pos ? d_pos[m] : (pos_base + m);
    if (pos_rope >= max_seq_len) pos_rope = max_seq_len - 1;
    int current_base = d_pos ? d_pos[0] : pos_base;
    int pos_cache = current_base + m;
    if (pos_cache >= max_seq_len) pos_cache = max_seq_len - 1;

    int tid = threadIdx.x;
    __shared__ float s_k_sum;

    size_t k_offset = (size_t)m * (n_kv_heads * head_dim) + (size_t)kv_head * head_dim;
    __nv_bfloat16* k_vec = k + k_offset;
    const __nv_bfloat16* v_vec = v + k_offset;

    // 1. RMSNorm on K
    if (tid == 0) s_k_sum = 0.0f;
    __syncthreads();

    float k_sq = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float val = __bfloat162float(k_vec[i]);
        k_sq += val * val;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        k_sq += __shfl_down_sync(0xFFFFFFFF, k_sq, offset);
    }
    if (tid % 32 == 0) atomicAdd(&s_k_sum, k_sq);
    __syncthreads();

    float k_rrms = rsqrtf(s_k_sum / (float)head_dim + eps);
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float k_normed = __bfloat162float(k_vec[i]) * k_rrms * (1.0f + __bfloat162float(k_norm_w[i]));
        k_vec[i] = __float2bfloat16(k_normed);
    }
    __syncthreads();

    // 2. Partial RoPE with depth position pos_rope
    int rotary_dim = 64;
    int half_rotary = rotary_dim / 2;
    for (int i = tid; i < half_rotary; i += blockDim.x) {
        float freq = 1.0f / powf(rope_theta, (float)(2 * i) / (float)rotary_dim);
        float angle = (float)pos_rope * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        float k0 = __bfloat162float(k_vec[i]);
        float k1 = __bfloat162float(k_vec[i + half_rotary]);

        k_vec[i] = __float2bfloat16(k0 * cos_a - k1 * sin_a);
        k_vec[i + half_rotary] = __float2bfloat16(k0 * sin_a + k1 * cos_a);
    }
    __syncthreads();

    // 3. Store into temporary tree cache slot pos_cache
    size_t cache_offset = ((size_t)pos_cache * n_kv_heads + kv_head) * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        k_cache[cache_offset + i] = float_to_fp8_e4m3(__bfloat162float(k_vec[i]));
        v_cache[cache_offset + i] = float_to_fp8_e4m3(__bfloat162float(v_vec[i]));
    }
}

__global__ void qwen_gqa_compute_attn_tree_fp8_batch_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q_and_gate,
    const __nv_bfloat16* __restrict__ q_norm_w,
    const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    const int32_t* __restrict__ d_pos,
    const uint64_t* __restrict__ d_ancestor_mask,
    int pos_base,
    int M,
    int max_seq_len,
    float rope_theta,
    float eps)
{
    int q_head = blockIdx.x;
    int m = blockIdx.y;
    if (q_head >= n_q_heads || m >= M) return;

    int pos_rope = d_pos ? d_pos[m] : (pos_base + m);
    if (pos_rope >= max_seq_len) pos_rope = max_seq_len - 1;

    int current_base = d_pos ? d_pos[0] : pos_base;
    // Active tree KV range includes history [0..current_base-1] + candidate tree nodes [current_base..current_base+m]
    int max_t = current_base + m;
    if (max_t >= max_seq_len) max_t = max_seq_len - 1;

    int tid = threadIdx.x;
    int kv_head = q_head / (n_q_heads / n_kv_heads);

    __shared__ float s_q[256];
    __shared__ float s_q_sum;
    __shared__ float s_tile_scores[128];
    __shared__ float s_warp_reduce[4];
    __shared__ float s_new_max;
    __shared__ float s_alpha;
    __shared__ float s_reduce_acc[64][4];
    __shared__ alignas(16) uint8_t s_v_tile[32768];

    size_t q_offset = (size_t)m * (2 * n_q_heads * head_dim) + (size_t)q_head * (2 * head_dim);
    const __nv_bfloat16* q_in = q_and_gate + q_offset;
    const __nv_bfloat16* gate_in = q_in + head_dim;

    size_t out_offset = (size_t)m * (n_q_heads * head_dim) + (size_t)q_head * head_dim;
    __nv_bfloat16* out_vec = out + out_offset;

    if (tid == 0) s_q_sum = 0.0f;
    __syncthreads();

    float q_sq = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float val = __bfloat162float(q_in[i]);
        s_q[i] = val;
        q_sq += val * val;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        q_sq += __shfl_down_sync(0xFFFFFFFF, q_sq, offset);
    }
    if (tid % 32 == 0) atomicAdd(&s_q_sum, q_sq);
    __syncthreads();

    float q_rrms = rsqrtf(s_q_sum / (float)head_dim + eps);
    for (int i = tid; i < head_dim; i += blockDim.x) {
        s_q[i] = s_q[i] * q_rrms * (1.0f + __bfloat162float(q_norm_w[i]));
    }
    __syncthreads();

    int rotary_dim = 64;
    int half_rotary = rotary_dim / 2;
    for (int i = tid; i < half_rotary; i += blockDim.x) {
        float freq = 1.0f / powf(rope_theta, (float)(2 * i) / (float)rotary_dim);
        float angle = (float)pos_rope * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        float q0 = s_q[i];
        float q1 = s_q[i + half_rotary];

        s_q[i] = q0 * cos_a - q1 * sin_a;
        s_q[i + half_rotary] = q0 * sin_a + q1 * cos_a;
    }
    __syncthreads();

    float scale = 1.0f / sqrtf((float)head_dim);
    float running_max = -1e38f;
    float running_sum = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    int w = tid % 64;
    int half = tid / 64;
    int words_per_tok = head_dim >> 2;

    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int u4_shift = (head_dim == 256) ? 4 : 3;
    int u4_mask = (1 << u4_shift) - 1;

    for (int t_block = 0; t_block <= max_t; t_block += 128) {
        int chunk_len = min(128, max_t + 1 - t_block);

        float dot = -1e38f;
        if (tid < chunk_len) {
            int t = t_block + tid;
            bool valid = true;
            if (d_ancestor_mask && t >= current_base) {
                int cand_idx = t - current_base;
                if (cand_idx > m || (cand_idx != m && !(d_ancestor_mask[m] & (1ULL << cand_idx)))) {
                    valid = false;
                }
            }
            if (valid) {
                size_t k_offset = ((size_t)t * n_kv_heads + kv_head) * head_dim;
                const uint4* k_ptr16 = reinterpret_cast<const uint4*>(k_cache + k_offset);
                float d = 0.0f;
                int num_vec16 = head_dim / 16;
                #pragma unroll 4
                for (int v_idx = 0; v_idx < num_vec16; v_idx++) {
                    uint4 kv4 = k_ptr16[v_idx];
                    int base = v_idx * 16;

                    uint32_t w0 = kv4.x;
                    d += s_q[base + 0] * fp8_e4m3_to_float_v2(w0 & 0xFF);
                    d += s_q[base + 1] * fp8_e4m3_to_float_v2((w0 >> 8) & 0xFF);
                    d += s_q[base + 2] * fp8_e4m3_to_float_v2((w0 >> 16) & 0xFF);
                    d += s_q[base + 3] * fp8_e4m3_to_float_v2((w0 >> 24) & 0xFF);

                    uint32_t w1 = kv4.y;
                    d += s_q[base + 4] * fp8_e4m3_to_float_v2(w1 & 0xFF);
                    d += s_q[base + 5] * fp8_e4m3_to_float_v2((w1 >> 8) & 0xFF);
                    d += s_q[base + 6] * fp8_e4m3_to_float_v2((w1 >> 16) & 0xFF);
                    d += s_q[base + 7] * fp8_e4m3_to_float_v2((w1 >> 24) & 0xFF);

                    uint32_t w2 = kv4.z;
                    d += s_q[base + 8] * fp8_e4m3_to_float_v2(w2 & 0xFF);
                    d += s_q[base + 9] * fp8_e4m3_to_float_v2((w2 >> 8) & 0xFF);
                    d += s_q[base + 10] * fp8_e4m3_to_float_v2((w2 >> 16) & 0xFF);
                    d += s_q[base + 11] * fp8_e4m3_to_float_v2((w2 >> 24) & 0xFF);

                    uint32_t w3 = kv4.w;
                    d += s_q[base + 12] * fp8_e4m3_to_float_v2(w3 & 0xFF);
                    d += s_q[base + 13] * fp8_e4m3_to_float_v2((w3 >> 8) & 0xFF);
                    d += s_q[base + 14] * fp8_e4m3_to_float_v2((w3 >> 16) & 0xFF);
                    d += s_q[base + 15] * fp8_e4m3_to_float_v2((w3 >> 24) & 0xFF);
                }
                dot = d * scale;
            }
        }
        s_tile_scores[tid] = dot;
        __syncthreads();

        float thread_val = (tid < chunk_len) ? s_tile_scores[tid] : -1e38f;
        float chunk_max = thread_val;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            chunk_max = fmaxf(chunk_max, __shfl_down_sync(0xFFFFFFFF, chunk_max, offset));
        }
        if (lane_id == 0) s_warp_reduce[warp_id] = chunk_max;
        __syncthreads();

        if (tid == 0) {
            float m_val = fmaxf(fmaxf(s_warp_reduce[0], s_warp_reduce[1]), fmaxf(s_warp_reduce[2], s_warp_reduce[3]));
            float new_m = fmaxf(running_max, m_val);
            float a = (running_max <= -1e37f) ? 0.0f : __expf(running_max - new_m);
            s_new_max = new_m;
            s_alpha = a;
        }
        __syncthreads();

        float new_max = s_new_max;
        float alpha = s_alpha;
        running_max = new_max;
        running_sum = running_sum * alpha;
        if (half == 0) {
            acc0 = acc0 * alpha;
            acc1 = acc1 * alpha;
            acc2 = acc2 * alpha;
            acc3 = acc3 * alpha;
        }

        float exp_s = (tid < chunk_len) ? __expf(s_tile_scores[tid] - new_max) : 0.0f;
        s_tile_scores[tid] = exp_s;
        __syncthreads();

        float sum_chunk = exp_s;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_chunk += __shfl_down_sync(0xFFFFFFFF, sum_chunk, offset);
        }
        if (lane_id == 0) s_warp_reduce[warp_id] = sum_chunk;
        __syncthreads();
        if (tid == 0) {
            s_warp_reduce[0] += s_warp_reduce[1] + s_warp_reduce[2] + s_warp_reduce[3];
        }
        __syncthreads();
        running_sum += s_warp_reduce[0];

        int total_u4 = (chunk_len * head_dim) >> 4;
        uint4* s_v_u4 = reinterpret_cast<uint4*>(s_v_tile);

        #pragma unroll 2
        for (int idx = tid; idx < total_u4; idx += blockDim.x) {
            int t_offset = idx >> u4_shift;
            int v_idx = idx & u4_mask;
            int t = t_block + t_offset;
            size_t v_off = ((size_t)t * n_kv_heads + kv_head) * head_dim + (v_idx << 4);
            s_v_u4[idx] = *reinterpret_cast<const uint4*>(v_cache + v_off);
        }
        __syncthreads();

        const uint32_t* s_v_u32 = reinterpret_cast<const uint32_t*>(s_v_tile);
        int j_start = (half == 0) ? 0 : 64;
        int j_end   = (half == 0) ? min(64, chunk_len) : chunk_len;

        float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
        if (w < words_per_tok && j_start < j_end) {
            #pragma unroll 4
            for (int j = j_start; j < j_end; j++) {
                float weight = s_tile_scores[j];
                uint32_t v4 = s_v_u32[j * words_per_tok + w];
                c0 += weight * fp8_e4m3_to_float_v2(v4 & 0xFF);
                c1 += weight * fp8_e4m3_to_float_v2((v4 >> 8) & 0xFF);
                c2 += weight * fp8_e4m3_to_float_v2((v4 >> 16) & 0xFF);
                c3 += weight * fp8_e4m3_to_float_v2((v4 >> 24) & 0xFF);
            }
        }

        if (half == 1 && w < words_per_tok) {
            s_reduce_acc[w][0] = c0;
            s_reduce_acc[w][1] = c1;
            s_reduce_acc[w][2] = c2;
            s_reduce_acc[w][3] = c3;
        }
        __syncthreads();

        if (half == 0 && w < words_per_tok) {
            if (chunk_len > 64) {
                c0 += s_reduce_acc[w][0];
                c1 += s_reduce_acc[w][1];
                c2 += s_reduce_acc[w][2];
                c3 += s_reduce_acc[w][3];
            }
            acc0 += c0;
            acc1 += c1;
            acc2 += c2;
            acc3 += c3;
        }
        __syncthreads();
    }

    float inv_sum = 1.0f / (running_sum + 1e-8f);
    if (half == 0 && w < words_per_tok) {
        int base_d = w * 4;
        if (base_d < head_dim) {
            float g0 = __bfloat162float(gate_in[base_d]);
            float sig0 = 1.0f / (1.0f + __expf(-g0));
            out_vec[base_d] = __float2bfloat16(acc0 * inv_sum * sig0);
        }
        if (base_d + 1 < head_dim) {
            float g1 = __bfloat162float(gate_in[base_d + 1]);
            float sig1 = 1.0f / (1.0f + __expf(-g1));
            out_vec[base_d + 1] = __float2bfloat16(acc1 * inv_sum * sig1);
        }
        if (base_d + 2 < head_dim) {
            float g2 = __bfloat162float(gate_in[base_d + 2]);
            float sig2 = 1.0f / (1.0f + __expf(-g2));
            out_vec[base_d + 2] = __float2bfloat16(acc2 * inv_sum * sig2);
        }
        if (base_d + 3 < head_dim) {
            float g3 = __bfloat162float(gate_in[base_d + 3]);
            float sig3 = 1.0f / (1.0f + __expf(-g3));
            out_vec[base_d + 3] = __float2bfloat16(acc3 * inv_sum * sig3);
        }
    }
}

void qwen_gqa_decode_tree_fp8_batch_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* q_and_gate,
    __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const __nv_bfloat16* q_norm_w,
    const __nv_bfloat16* k_norm_w,
    uint8_t* k_cache,
    uint8_t* v_cache,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    const int32_t* d_pos,
    const uint64_t* d_ancestor_mask,
    int pos_base,
    int M,
    int max_seq_len,
    float rope_theta,
    float eps,
    cudaStream_t stream)
{
    if (M <= 0) return;
    int threads = 128;

    dim3 grid_kv(n_kv_heads, M);
    qwen_gqa_write_kv_tree_fp8_batch_kernel<<<grid_kv, threads, 0, stream>>>(
        k, v, k_norm_w, k_cache, v_cache, n_kv_heads, head_dim, d_pos, pos_base, M, max_seq_len, rope_theta, eps);

    dim3 grid_q(n_q_heads, M);
    qwen_gqa_compute_attn_tree_fp8_batch_kernel<<<grid_q, threads, 0, stream>>>(
        out, q_and_gate, q_norm_w, k_cache, v_cache, n_q_heads, n_kv_heads, head_dim, d_pos, d_ancestor_mask, pos_base, M, max_seq_len, rope_theta, eps);
}

// ── KV Cache Compaction Kernel ────────────────────────────────────────────────

__global__ void compact_tree_kv_cache_kernel(
    uint8_t* __restrict__ k_cache,
    uint8_t* __restrict__ v_cache,
    const int32_t* __restrict__ d_winning_slots,
    int pos_base,
    int L,
    int bytes_per_tok)
{
    int tid = threadIdx.x;
    int k = blockIdx.x;
    if (k >= L) return;

    int src_slot = d_winning_slots[k];
    int src_pos = pos_base + src_slot;
    int dst_pos = pos_base + 1 + k;

    if (src_pos == dst_pos) return;

    size_t src_off = (size_t)src_pos * bytes_per_tok;
    size_t dst_off = (size_t)dst_pos * bytes_per_tok;

    int num_u4 = bytes_per_tok >> 4;
    for (int i = tid; i < num_u4; i += blockDim.x) {
        uint4 k_val = *reinterpret_cast<const uint4*>(k_cache + src_off + (i << 4));
        *reinterpret_cast<uint4*>(k_cache + dst_off + (i << 4)) = k_val;

        uint4 v_val = *reinterpret_cast<const uint4*>(v_cache + src_off + (i << 4));
        *reinterpret_cast<uint4*>(v_cache + dst_off + (i << 4)) = v_val;
    }
}

void compact_tree_kv_cache_cuda(
    uint8_t* k_cache,
    uint8_t* v_cache,
    const int32_t* d_winning_slots,
    int pos_base,
    int L,
    int n_kv_heads,
    int head_dim,
    cudaStream_t stream)
{
    if (L <= 0) return;
    int bytes_per_tok = n_kv_heads * head_dim;
    compact_tree_kv_cache_kernel<<<L, 128, 0, stream>>>(k_cache, v_cache, d_winning_slots, pos_base, L, bytes_per_tok);
}

// ── DeltaNet Linear Attention Tree Recurrence ─────────────────────────────────

__global__ void deltanet_conv_tree_kernel(
    __nv_bfloat16* __restrict__ conv_out,
    const __nv_bfloat16* __restrict__ in_qkv,
    const __nv_bfloat16* __restrict__ conv1d_w,
    const __nv_bfloat16* __restrict__ in_conv_state,
    __nv_bfloat16* __restrict__ out_conv_state,
    __nv_bfloat16* slot_conv_0,
    __nv_bfloat16* slot_conv_1,
    __nv_bfloat16* slot_conv_2,
    __nv_bfloat16* slot_conv_3,
    __nv_bfloat16* slot_conv_4,
    __nv_bfloat16* slot_conv_5,
    __nv_bfloat16* slot_conv_6,
    __nv_bfloat16* slot_conv_7,
    const int32_t* __restrict__ d_parent_indices,
    int channels, int M)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;

    const __nv_bfloat16* in_cs = in_conv_state + (size_t)c * 4;
    const __nv_bfloat16* cw = conv1d_w + (size_t)c * 4;

    float w0 = __bfloat162float(cw[0]);
    float w1 = __bfloat162float(cw[1]);
    float w2 = __bfloat162float(cw[2]);
    float w3 = __bfloat162float(cw[3]);

    __nv_bfloat16* slots[8] = {
        slot_conv_0, slot_conv_1, slot_conv_2, slot_conv_3,
        slot_conv_4, slot_conv_5, slot_conv_6, slot_conv_7
    };

    for (int m = 0; m < M; m++) {
        int parent_idx = d_parent_indices ? d_parent_indices[m] : (m - 1);
        float s0, s1, s2;
        if (parent_idx < 0) {
            s0 = __bfloat162float(in_cs[1]);
            s1 = __bfloat162float(in_cs[2]);
            s2 = __bfloat162float(in_cs[3]);
        } else if (parent_idx < 8 && slots[parent_idx]) {
            const __nv_bfloat16* p_cs = slots[parent_idx] + (size_t)c * 4;
            s0 = __bfloat162float(p_cs[1]);
            s1 = __bfloat162float(p_cs[2]);
            s2 = __bfloat162float(p_cs[3]);
        } else {
            s0 = 0.0f; s1 = 0.0f; s2 = 0.0f;
        }

        float s3 = __bfloat162float(in_qkv[(size_t)m * channels + c]);
        float val = s0 * w0 + s1 * w1 + s2 * w2 + s3 * w3;
        float silu_val = val / (1.0f + __expf(-val));
        conv_out[(size_t)m * channels + c] = __float2bfloat16(silu_val);

        if (m < 8 && slots[m]) {
            __nv_bfloat16* slot_cs = slots[m] + (size_t)c * 4;
            slot_cs[0] = __float2bfloat16(s0);
            slot_cs[1] = __float2bfloat16(s1);
            slot_cs[2] = __float2bfloat16(s2);
            slot_cs[3] = __float2bfloat16(s3);
        }
    }
}

__global__ void deltanet_ssm_tree_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ conv_out,
    const __nv_bfloat16* __restrict__ in_z,
    const __nv_bfloat16* __restrict__ in_a,
    const __nv_bfloat16* __restrict__ in_b,
    const __nv_bfloat16* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ norm_w,
    const __nv_bfloat16* __restrict__ in_ssm_state,
    __nv_bfloat16* __restrict__ out_ssm_state,
    __nv_bfloat16* slot_ssm_0,
    __nv_bfloat16* slot_ssm_1,
    __nv_bfloat16* slot_ssm_2,
    __nv_bfloat16* slot_ssm_3,
    __nv_bfloat16* slot_ssm_4,
    __nv_bfloat16* slot_ssm_5,
    __nv_bfloat16* slot_ssm_6,
    __nv_bfloat16* slot_ssm_7,
    const int32_t* __restrict__ d_parent_indices,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int M)
{
    int h = blockIdx.x;
    if (h >= num_v_heads) return;

    int tid = threadIdx.x;
    int k_h = h / (num_v_heads / num_k_heads);

    __shared__ __nv_bfloat16 s_state[128][132];
    __shared__ float s_q[128];
    __shared__ float s_k[128];
    __shared__ float s_v[128];
    __shared__ float s_z[128];
    __shared__ float s_warp_q[4];
    __shared__ float s_warp_k[4];
    __shared__ float s_warp_out[4];
    __shared__ float s_beta;
    __shared__ float s_decay;

    __nv_bfloat16* slots[8] = {
        slot_ssm_0, slot_ssm_1, slot_ssm_2, slot_ssm_3,
        slot_ssm_4, slot_ssm_5, slot_ssm_6, slot_ssm_7
    };

    const __nv_bfloat16* in_state_h = in_ssm_state + (size_t)h * head_dim * head_dim;

    float dt_val = __bfloat162float(dt_bias[h]);
    float a_log_val = __bfloat162float(A_log[h]);
    float nw_val = (tid < head_dim) ? __bfloat162float(norm_w[tid]) : 0.0f;

    int q_offset = k_h * head_dim;
    int k_offset = (num_k_heads * head_dim) + k_h * head_dim;
    int v_offset = (2 * num_k_heads * head_dim) + h * head_dim;
    int qkv_stride = (2 * num_k_heads + num_v_heads) * head_dim;
    int z_stride = num_v_heads * head_dim;

    int current_owner = -999;

    for (int m = 0; m < M; m++) {
        int parent_idx = d_parent_indices ? d_parent_indices[m] : (m - 1);

        if (parent_idx < 0) {
            #pragma unroll 4
            for (int r = 0; r < head_dim; r++) {
                s_state[r][tid] = in_state_h[r * head_dim + tid];
            }
            __syncthreads();
        } else if (parent_idx != current_owner) {
            if (parent_idx < 8 && slots[parent_idx]) {
                const __nv_bfloat16* p_state_h = slots[parent_idx] + (size_t)h * head_dim * head_dim;
                #pragma unroll 4
                for (int r = 0; r < head_dim; r++) {
                    s_state[r][tid] = p_state_h[r * head_dim + tid];
                }
            }
            __syncthreads();
        }

        const __nv_bfloat16* m_conv = conv_out + (size_t)m * qkv_stride;
        const __nv_bfloat16* m_z = in_z + (size_t)m * z_stride;
        const __nv_bfloat16* m_a = in_a + (size_t)m * num_v_heads;
        const __nv_bfloat16* m_b = in_b + (size_t)m * num_v_heads;

        if (tid == 0) {
            float a_val = __bfloat162float(m_a[h]);
            float b_val = __bfloat162float(m_b[h]);
            s_beta = 1.0f / (1.0f + __expf(-b_val));
            float val_a = a_val + dt_val;
            float softplus_a = (val_a > 20.0f) ? val_a : log1pf(__expf(val_a));
            float g = -__expf(a_log_val) * softplus_a;
            s_decay = __expf(g);
        }

        float q_val = (tid < head_dim) ? __bfloat162float(m_conv[q_offset + tid]) : 0.0f;
        float k_val = (tid < head_dim) ? __bfloat162float(m_conv[k_offset + tid]) : 0.0f;
        float v_val = (tid < head_dim) ? __bfloat162float(m_conv[v_offset + tid]) : 0.0f;
        float z_val = (tid < head_dim) ? __bfloat162float(m_z[h * head_dim + tid]) : 0.0f;

        float q_sq = q_val * q_val;
        float k_sq = k_val * k_val;
        for (int offset = 16; offset > 0; offset /= 2) {
            q_sq += __shfl_down_sync(0xFFFFFFFF, q_sq, offset);
            k_sq += __shfl_down_sync(0xFFFFFFFF, k_sq, offset);
        }
        if ((tid & 31) == 0) {
            s_warp_q[tid >> 5] = q_sq;
            s_warp_k[tid >> 5] = k_sq;
        }
        __syncthreads();

        if (tid < head_dim) {
            float q_norm_sq = s_warp_q[0] + s_warp_q[1] + s_warp_q[2] + s_warp_q[3];
            float k_norm_sq = s_warp_k[0] + s_warp_k[1] + s_warp_k[2] + s_warp_k[3];
            float r_q = rsqrtf(q_norm_sq + 1e-6f) * (1.0f / sqrtf((float)head_dim));
            float r_k = rsqrtf(k_norm_sq + 1e-6f);
            s_q[tid] = q_val * r_q;
            s_k[tid] = k_val * r_k;
            s_v[tid] = v_val;
            s_z[tid] = z_val;
        }
        __syncthreads();

        float decay = s_decay;
        float beta = s_beta;
        float out_c = 0.0f;
        if (tid < head_dim) {
            float dot_k = 0.0f;
            #pragma unroll 16
            for (int r = 0; r < head_dim; r++) {
                dot_k = fmaf(__bfloat162float(s_state[r][tid]), s_k[r], dot_k);
            }
            float mem = dot_k * decay;
            float delta_c = (s_v[tid] - mem) * beta;
            #pragma unroll 16
            for (int r = 0; r < head_dim; r++) {
                float s_val = __bfloat162float(s_state[r][tid]);
                float new_s = fmaf(s_k[r], delta_c, decay * s_val);
                s_state[r][tid] = __float2bfloat16(new_s);
                out_c = fmaf(new_s, s_q[r], out_c);
            }
        }

        if (m < 8 && slots[m]) {
            __nv_bfloat16* slot_state_h = slots[m] + (size_t)h * head_dim * head_dim;
            #pragma unroll 4
            for (int r = 0; r < head_dim; r++) {
                slot_state_h[r * head_dim + tid] = s_state[r][tid];
            }
        }

        current_owner = m;

        float out_sq = (tid < head_dim) ? (out_c * out_c) : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            out_sq += __shfl_down_sync(0xFFFFFFFF, out_sq, offset);
        }
        if ((tid & 31) == 0) {
            s_warp_out[tid >> 5] = out_sq;
        }
        __syncthreads();

        if (tid < head_dim) {
            float norm_sq = s_warp_out[0] + s_warp_out[1] + s_warp_out[2] + s_warp_out[3];
            float r_out = rsqrtf(norm_sq / (float)head_dim + 1e-6f);
            float normed = out_c * r_out * nw_val;
            float z_val = s_z[tid];
            float silu_z = z_val / (1.0f + __expf(-z_val));
            out[(size_t)m * z_stride + h * head_dim + tid] = __float2bfloat16(normed * silu_z);
        }
        __syncthreads();
    }
}

void deltanet_linear_attention_tree_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* in_qkv,
    const __nv_bfloat16* in_z,
    const __nv_bfloat16* in_a,
    const __nv_bfloat16* in_b,
    const __nv_bfloat16* conv1d_w,
    const __nv_bfloat16* in_conv_state,
    __nv_bfloat16* out_conv_state,
    __nv_bfloat16* slot_conv_0,
    __nv_bfloat16* slot_conv_1,
    __nv_bfloat16* slot_conv_2,
    __nv_bfloat16* slot_conv_3,
    __nv_bfloat16* slot_conv_4,
    __nv_bfloat16* slot_conv_5,
    __nv_bfloat16* slot_conv_6,
    __nv_bfloat16* slot_conv_7,
    const __nv_bfloat16* A_log,
    const __nv_bfloat16* dt_bias,
    const __nv_bfloat16* norm_w,
    const __nv_bfloat16* in_ssm_state,
    __nv_bfloat16* out_ssm_state,
    __nv_bfloat16* slot_ssm_0,
    __nv_bfloat16* slot_ssm_1,
    __nv_bfloat16* slot_ssm_2,
    __nv_bfloat16* slot_ssm_3,
    __nv_bfloat16* slot_ssm_4,
    __nv_bfloat16* slot_ssm_5,
    __nv_bfloat16* slot_ssm_6,
    __nv_bfloat16* slot_ssm_7,
    const int32_t* d_parent_indices,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int M,
    cudaStream_t stream)
{
    int channels = (2 * num_k_heads + num_v_heads) * head_dim;
    int threads = 256;
    int blocks = (channels + threads - 1) / threads;

    __nv_bfloat16* conv_out = const_cast<__nv_bfloat16*>(in_qkv);
    deltanet_conv_tree_kernel<<<blocks, threads, 0, stream>>>(
        conv_out, in_qkv, conv1d_w, in_conv_state, out_conv_state,
        slot_conv_0, slot_conv_1, slot_conv_2, slot_conv_3,
        slot_conv_4, slot_conv_5, slot_conv_6, slot_conv_7,
        d_parent_indices, channels, M);

    deltanet_ssm_tree_kernel<<<num_v_heads, 128, 0, stream>>>(
        out, conv_out, in_z, in_a, in_b, A_log, dt_bias, norm_w,
        in_ssm_state, out_ssm_state,
        slot_ssm_0, slot_ssm_1, slot_ssm_2, slot_ssm_3,
        slot_ssm_4, slot_ssm_5, slot_ssm_6, slot_ssm_7,
        d_parent_indices,
        num_k_heads, num_v_heads, head_dim, M);
}

// ════════════════════════════════════════════════════════════════════════════════
//  NVIDIA Nemotron 3.5 Lightning: NVFP4, Squared ReLU, Scalar FP8, Mamba-2
// ════════════════════════════════════════════════════════════════════════════════

template<typename TOut, bool Residual = false, bool Scaled = false, bool ReLU2 = false>
__global__ void gemv_nvfp4_block16_kernel(
    TOut* __restrict__ out,
    const __nv_bfloat16* __restrict__ vec,
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scale_fp8,
    float scale_2,
    float alpha,
    int N, int K)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;

    int lane = threadIdx.x; // 0..31
    int num_groups = K / 16;

    const uint8_t* row_w = weight + (size_t)row * (K / 2);
    const uint8_t* row_s = scale_fp8 + (size_t)row * num_groups;

    float acc = 0.0f;

    for (int g = lane; g < num_groups; g += 32) {
        float s_val = fp8_e4m3_to_float_v2(row_s[g]) * scale_2;
        const uint8_t* g_w = row_w + g * 8;
        const __nv_bfloat16* g_v = vec + g * 16;

        #pragma unroll
        for (int b = 0; b < 8; b++) {
            float2 w2 = fp4_e2m1_to_float2(g_w[b]);
            float v0 = __bfloat162float(g_v[b * 2]);
            float v1 = __bfloat162float(g_v[b * 2 + 1]);
            acc += (w2.x * v0 + w2.y * v1) * s_val;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane == 0) {
        if constexpr (Scaled) {
            acc *= alpha;
        }
        if constexpr (ReLU2) {
            acc = (acc > 0.0f) ? (acc * acc) : 0.0f;
        }
        if constexpr (Residual) {
            if constexpr (std::is_same_v<TOut, __nv_bfloat16>) {
                float cur = __bfloat162float(out[row]);
                out[row] = __float2bfloat16(cur + acc);
            } else {
                out[row] = (TOut)(out[row] + acc);
            }
        } else {
            if constexpr (std::is_same_v<TOut, __nv_bfloat16>) {
                out[row] = __float2bfloat16(acc);
            } else {
                out[row] = (TOut)acc;
            }
        }
    }
}

void gemv_nvfp4_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* vec,
    const uint8_t* weight,
    const uint8_t* scale_fp8,
    float scale_2,
    int N, int K,
    cudaStream_t stream)
{
    dim3 threads(32, 4);
    dim3 blocks((N + 3) / 4);
    gemv_nvfp4_block16_kernel<__nv_bfloat16, false, false, false><<<blocks, threads, 0, stream>>>(
        out, vec, weight, scale_fp8, scale_2, 1.0f, N, K);
}

void gemv_nvfp4_relu2_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* vec,
    const uint8_t* weight,
    const uint8_t* scale_fp8,
    float scale_2,
    int N, int K,
    cudaStream_t stream)
{
    dim3 threads(32, 4);
    dim3 blocks((N + 3) / 4);
    gemv_nvfp4_block16_kernel<__nv_bfloat16, false, false, true><<<blocks, threads, 0, stream>>>(
        out, vec, weight, scale_fp8, scale_2, 1.0f, N, K);
}

void gemv_nvfp4_f32_cuda(
    float* out,
    const __nv_bfloat16* vec,
    const uint8_t* weight,
    const uint8_t* scale_fp8,
    float scale_2,
    int N, int K,
    cudaStream_t stream)
{
    dim3 threads(32, 4);
    dim3 blocks((N + 3) / 4);
    gemv_nvfp4_block16_kernel<float, false, false><<<blocks, threads, 0, stream>>>(
        out, vec, weight, scale_fp8, scale_2, 1.0f, N, K);
}

void gemv_nvfp4_residual_cuda(
    __nv_bfloat16* inout,
    const __nv_bfloat16* vec,
    const uint8_t* weight,
    const uint8_t* scale_fp8,
    float scale_2,
    int N, int K,
    cudaStream_t stream)
{
    dim3 threads(32, 4);
    dim3 blocks((N + 3) / 4);
    gemv_nvfp4_block16_kernel<__nv_bfloat16, true, false><<<blocks, threads, 0, stream>>>(
        inout, vec, weight, scale_fp8, scale_2, 1.0f, N, K);
}

void gemv_nvfp4_scaled_residual_cuda(
    __nv_bfloat16* inout,
    const __nv_bfloat16* vec,
    const uint8_t* weight,
    const uint8_t* scale_fp8,
    float scale_2,
    float alpha,
    int N, int K,
    cudaStream_t stream)
{
    dim3 threads(32, 4);
    dim3 blocks((N + 3) / 4);
    gemv_nvfp4_block16_kernel<__nv_bfloat16, true, true><<<blocks, threads, 0, stream>>>(
        inout, vec, weight, scale_fp8, scale_2, alpha, N, K);
}

__global__ void relu2_kernel(__nv_bfloat16* out, const __nv_bfloat16* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float f = __bfloat162float(in[idx]);
        f = (f > 0.0f) ? (f * f) : 0.0f;
        out[idx] = __float2bfloat16(f);
    }
}

void relu2_cuda(__nv_bfloat16* out, const __nv_bfloat16* in, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu2_kernel<<<blocks, threads, 0, stream>>>(out, in, n);
}

template<bool Residual = false>
__global__ void gemv_fp8_scalar_scale_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ vec,
    const uint8_t* __restrict__ weight,
    float scale_val,
    int N, int K)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;

    int lane = threadIdx.x; // 0..31
    const uint8_t* row_w = weight + (size_t)row * K;

    float acc = 0.0f;
    for (int k = lane; k < K; k += 32) {
        float w = fp8_e4m3_to_float_v2(row_w[k]);
        float v = __bfloat162float(vec[k]);
        acc += w * v;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane == 0) {
        acc *= scale_val;
        if constexpr (Residual) {
            float cur = __bfloat162float(out[row]);
            out[row] = __float2bfloat16(cur + acc);
        } else {
            out[row] = __float2bfloat16(acc);
        }
    }
}

void gemv_fp8_scalar_scale_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* vec,
    const uint8_t* weight,
    float scale_val,
    int N, int K,
    cudaStream_t stream)
{
    dim3 threads(32, 4);
    dim3 blocks((N + 3) / 4);
    gemv_fp8_scalar_scale_kernel<false><<<blocks, threads, 0, stream>>>(out, vec, weight, scale_val, N, K);
}

void gemv_fp8_scalar_scale_residual_cuda(
    __nv_bfloat16* inout,
    const __nv_bfloat16* vec,
    const uint8_t* weight,
    float scale_val,
    int N, int K,
    cudaStream_t stream)
{
    dim3 threads(32, 4);
    dim3 blocks((N + 3) / 4);
    gemv_fp8_scalar_scale_kernel<true><<<blocks, threads, 0, stream>>>(inout, vec, weight, scale_val, N, K);
}

__global__ void mamba2_nemotron_conv_kernel(
    __nv_bfloat16* __restrict__ conv_out,
    const __nv_bfloat16* __restrict__ in_proj,
    const __nv_bfloat16* __restrict__ conv1d_w,
    const __nv_bfloat16* __restrict__ conv1d_b,
    __nv_bfloat16* __restrict__ conv_state,
    int channels,
    int in_proj_offset = 0)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;

    __nv_bfloat16* cs = conv_state + c * 4;
    const __nv_bfloat16* cw = conv1d_w + c * 4;

    float s0 = __bfloat162float(cs[1]);
    float s1 = __bfloat162float(cs[2]);
    float s2 = __bfloat162float(cs[3]);
    float s3 = __bfloat162float(in_proj[in_proj_offset + c]);

    cs[0] = __float2bfloat16(s0);
    cs[1] = __float2bfloat16(s1);
    cs[2] = __float2bfloat16(s2);
    cs[3] = __float2bfloat16(s3);

    float w0 = __bfloat162float(cw[0]);
    float w1 = __bfloat162float(cw[1]);
    float w2 = __bfloat162float(cw[2]);
    float w3 = __bfloat162float(cw[3]);
    float b  = conv1d_b ? __bfloat162float(conv1d_b[c]) : 0.0f;

    float val = s0 * w0 + s1 * w1 + s2 * w2 + s3 * w3 + b;
    float silu_val = val / (1.0f + __expf(-val));
    conv_out[c] = __float2bfloat16(silu_val);
}

__global__ void mamba2_nemotron_ssm_step_kernel(
    float* __restrict__ temp_out,
    const __nv_bfloat16* __restrict__ conv_out,
    const __nv_bfloat16* __restrict__ in_proj,
    const __nv_bfloat16* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ D,
    const __nv_bfloat16* __restrict__ dt_bias,
    float* __restrict__ ssm_state,
    int num_heads, int intermediate_size, int dt_offset)
{
    int h = blockIdx.x;
    if (h >= num_heads) return;

    int p = threadIdx.x;

    __shared__ float s_B[128];
    __shared__ float s_C[128];
    __shared__ float s_dt_val;
    __shared__ float s_dA;
    __shared__ float s_D;

    int heads_per_group = num_heads / 8;
    int group = h / heads_per_group;
    s_B[p] = __bfloat162float(conv_out[intermediate_size + group * 128 + p]);
    s_B[p + 64] = __bfloat162float(conv_out[intermediate_size + group * 128 + p + 64]);
    s_C[p] = __bfloat162float(conv_out[intermediate_size + 1024 + group * 128 + p]);
    s_C[p + 64] = __bfloat162float(conv_out[intermediate_size + 1024 + group * 128 + p + 64]);

    if (p == 0) {
        float raw_dt = __bfloat162float(in_proj[dt_offset + h]) + (dt_bias ? __bfloat162float(dt_bias[h]) : 0.0f);
        float dt_val = (raw_dt > 20.0f) ? raw_dt : ((raw_dt < -20.0f) ? __expf(raw_dt) : log1pf(__expf(raw_dt)));
        s_dt_val = dt_val;
        float a_log = __bfloat162float(A_log[h]);
        s_dA = __expf(-__expf(a_log) * dt_val);
        s_D = D ? __bfloat162float(D[h]) : 0.0f;
    }
    __syncthreads();

    float x_p = __bfloat162float(conv_out[h * 64 + p]);
    float* state_row = ssm_state + (size_t)(h * 64 + p) * 128;

    float dt_val = s_dt_val;
    float dA = s_dA;
    float y_p = x_p * s_D;
    float x_dt = x_p * dt_val;

    #pragma unroll
    for (int n = 0; n < 128; n++) {
        float s = state_row[n];
        s = s * dA + (x_dt * s_B[n]);
        state_row[n] = s;
        y_p += s * s_C[n];
    }

    float z_p = __bfloat162float(in_proj[h * 64 + p]);
    float silu_z = z_p / (1.0f + __expf(-z_p));
    temp_out[h * 64 + p] = y_p * silu_z;
}

__global__ void mamba2_nemotron_group_norm_kernel(
    __nv_bfloat16* __restrict__ ssm_out,
    const float* __restrict__ temp_out,
    const __nv_bfloat16* __restrict__ norm_w,
    float eps,
    int group_size)
{
    int g = blockIdx.x;
    if (g >= 8) return;

    int tid = threadIdx.x;
    const float* g_in = temp_out + g * group_size;

    float sum_sq = 0.0f;
    for (int i = tid; i < group_size; i += blockDim.x) {
        float v = g_in[i];
        sum_sq += v * v;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    __shared__ float s_warp_sums[8];
    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) s_warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (tid == 0) {
        float total_sq = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; w++) total_sq += s_warp_sums[w];
        float var = total_sq / (float)group_size;
        s_warp_sums[0] = rsqrtf(var + eps);
    }
    __syncthreads();

    float rscale = s_warp_sums[0];
    for (int i = tid; i < group_size; i += blockDim.x) {
        int idx = g * group_size + i;
        float v = g_in[i];
        float w = norm_w ? __bfloat162float(norm_w[idx]) : 1.0f;
        ssm_out[idx] = __float2bfloat16(v * rscale * w);
    }
}

void mamba2_nemotron_step_cuda(
    __nv_bfloat16* ssm_out,
    const __nv_bfloat16* in_proj,
    const __nv_bfloat16* conv1d_w,
    const __nv_bfloat16* conv1d_b,
    __nv_bfloat16* conv_state,
    const __nv_bfloat16* A_log,
    const __nv_bfloat16* D,
    const __nv_bfloat16* dt_bias,
    const __nv_bfloat16* norm_w,
    float* ssm_state,
    __nv_bfloat16* conv_buf,
    float* temp_out,
    float rms_norm_eps,
    cudaStream_t stream,
    int num_heads,
    int head_dim)
{
    int intermediate_size = num_heads * head_dim;
    int conv_channels = intermediate_size + 2048; // 2 * 8 * 128
    int dt_offset = intermediate_size + conv_channels;
    int group_size = intermediate_size / 8;

    int conv_blocks = (conv_channels + 255) / 256;
    mamba2_nemotron_conv_kernel<<<conv_blocks, 256, 0, stream>>>(
        conv_buf, in_proj, conv1d_w, conv1d_b, conv_state, conv_channels, intermediate_size);

    mamba2_nemotron_ssm_step_kernel<<<num_heads, 64, 0, stream>>>(
        temp_out, conv_buf, in_proj, A_log, D, dt_bias, ssm_state, num_heads, intermediate_size, dt_offset);

    mamba2_nemotron_group_norm_kernel<<<8, 256, 0, stream>>>(
        ssm_out, temp_out, norm_w, rms_norm_eps, group_size);
}

__global__ void mamba2_nemotron_conv_batch_kernel(
    __nv_bfloat16* __restrict__ conv_out_batch,
    const __nv_bfloat16* __restrict__ in_proj_batch,
    const __nv_bfloat16* __restrict__ conv1d_w,
    const __nv_bfloat16* __restrict__ conv1d_b,
    __nv_bfloat16* __restrict__ conv_state,
    int channels, int in_proj_stride, int in_proj_offset, int M,
    __nv_bfloat16* __restrict__ conv_state_slot0 = nullptr,
    __nv_bfloat16* __restrict__ conv_state_slot1 = nullptr,
    __nv_bfloat16* __restrict__ conv_state_slot2 = nullptr)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;

    __nv_bfloat16* cs = conv_state + c * 4;
    const __nv_bfloat16* cw = conv1d_w + c * 4;

    float s0 = __bfloat162float(cs[0]);
    float s1 = __bfloat162float(cs[1]);
    float s2 = __bfloat162float(cs[2]);
    float s3 = __bfloat162float(cs[3]);

    float w0 = __bfloat162float(cw[0]);
    float w1 = __bfloat162float(cw[1]);
    float w2 = __bfloat162float(cw[2]);
    float w3 = __bfloat162float(cw[3]);
    float b  = conv1d_b ? __bfloat162float(conv1d_b[c]) : 0.0f;

    for (int m = 0; m < M; m++) {
        const __nv_bfloat16* in_p = in_proj_batch + (size_t)m * in_proj_stride;
        s0 = s1;
        s1 = s2;
        s2 = s3;
        s3 = __bfloat162float(in_p[in_proj_offset + c]);

        float val = s0 * w0 + s1 * w1 + s2 * w2 + s3 * w3 + b;
        float silu_val = val / (1.0f + __expf(-val));
        conv_out_batch[(size_t)m * channels + c] = __float2bfloat16(silu_val);

        if (m == 0 && conv_state_slot0) {
            __nv_bfloat16* cs0 = conv_state_slot0 + c * 4;
            cs0[0] = __float2bfloat16(s0);
            cs0[1] = __float2bfloat16(s1);
            cs0[2] = __float2bfloat16(s2);
            cs0[3] = __float2bfloat16(s3);
        }
        if (m == 1 && conv_state_slot1) {
            __nv_bfloat16* cs1 = conv_state_slot1 + c * 4;
            cs1[0] = __float2bfloat16(s0);
            cs1[1] = __float2bfloat16(s1);
            cs1[2] = __float2bfloat16(s2);
            cs1[3] = __float2bfloat16(s3);
        }
        if (m == 2 && conv_state_slot2) {
            __nv_bfloat16* cs2 = conv_state_slot2 + c * 4;
            cs2[0] = __float2bfloat16(s0);
            cs2[1] = __float2bfloat16(s1);
            cs2[2] = __float2bfloat16(s2);
            cs2[3] = __float2bfloat16(s3);
        }
    }

    cs[0] = __float2bfloat16(s0);
    cs[1] = __float2bfloat16(s1);
    cs[2] = __float2bfloat16(s2);
    cs[3] = __float2bfloat16(s3);
}

__global__ void mamba2_nemotron_ssm_step_batch_kernel(
    float* __restrict__ temp_out_batch,
    const __nv_bfloat16* __restrict__ conv_out_batch,
    const __nv_bfloat16* __restrict__ in_proj_batch,
    const __nv_bfloat16* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ D,
    const __nv_bfloat16* __restrict__ dt_bias,
    float* __restrict__ ssm_state,
    int num_heads, int intermediate_size, int conv_channels, int in_proj_stride, int dt_offset, int M,
    float* __restrict__ ssm_state_slot0 = nullptr,
    float* __restrict__ ssm_state_slot1 = nullptr,
    float* __restrict__ ssm_state_slot2 = nullptr)
{
    int h = blockIdx.x;
    if (h >= num_heads) return;

    int p = threadIdx.x;

    __shared__ float s_B[128];
    __shared__ float s_C[128];
    __shared__ float s_dt_val;
    __shared__ float s_dA;
    __shared__ float s_D;

    float* state_row = ssm_state + (size_t)(h * 64 + p) * 128;
    int heads_per_group = num_heads / 8;
    int group = h / heads_per_group;

    float r_state[128];
    #pragma unroll
    for (int n = 0; n < 128; n++) {
        r_state[n] = state_row[n];
    }

    for (int m = 0; m < M; m++) {
        const __nv_bfloat16* conv_out = conv_out_batch + (size_t)m * conv_channels;
        const __nv_bfloat16* in_proj = in_proj_batch + (size_t)m * in_proj_stride;
        float* temp_out = temp_out_batch + (size_t)m * intermediate_size;

        s_B[p] = __bfloat162float(conv_out[intermediate_size + group * 128 + p]);
        s_B[p + 64] = __bfloat162float(conv_out[intermediate_size + group * 128 + p + 64]);
        s_C[p] = __bfloat162float(conv_out[intermediate_size + 1024 + group * 128 + p]);
        s_C[p + 64] = __bfloat162float(conv_out[intermediate_size + 1024 + group * 128 + p + 64]);

        if (p == 0) {
            float raw_dt = __bfloat162float(in_proj[dt_offset + h]) + (dt_bias ? __bfloat162float(dt_bias[h]) : 0.0f);
            float dt_val = (raw_dt > 20.0f) ? raw_dt : ((raw_dt < -20.0f) ? __expf(raw_dt) : log1pf(__expf(raw_dt)));
            s_dt_val = dt_val;
            float a_log = __bfloat162float(A_log[h]);
            s_dA = __expf(-__expf(a_log) * dt_val);
            s_D = D ? __bfloat162float(D[h]) : 0.0f;
        }
        __syncthreads();

        float x_p = __bfloat162float(conv_out[h * 64 + p]);
        float dt_val = s_dt_val;
        float dA = s_dA;
        float y_p = x_p * s_D;
        float x_dt = x_p * dt_val;

        #pragma unroll
        for (int n = 0; n < 128; n++) {
            float s = r_state[n] * dA + (x_dt * s_B[n]);
            r_state[n] = s;
            y_p += s * s_C[n];
        }

        if (m == 0 && ssm_state_slot0) {
            float* s0_row = ssm_state_slot0 + (size_t)(h * 64 + p) * 128;
            #pragma unroll
            for (int n = 0; n < 128; n++) {
                s0_row[n] = r_state[n];
            }
        }
        if (m == 1 && ssm_state_slot1) {
            float* s1_row = ssm_state_slot1 + (size_t)(h * 64 + p) * 128;
            #pragma unroll
            for (int n = 0; n < 128; n++) {
                s1_row[n] = r_state[n];
            }
        }
        if (m == 2 && ssm_state_slot2) {
            float* s2_row = ssm_state_slot2 + (size_t)(h * 64 + p) * 128;
            #pragma unroll
            for (int n = 0; n < 128; n++) {
                s2_row[n] = r_state[n];
            }
        }

        float z_p = __bfloat162float(in_proj[h * 64 + p]);
        float silu_z = z_p / (1.0f + __expf(-z_p));
        temp_out[h * 64 + p] = y_p * silu_z;
        __syncthreads();
    }

    #pragma unroll
    for (int n = 0; n < 128; n++) {
        state_row[n] = r_state[n];
    }
}

__global__ void mamba2_nemotron_group_norm_batch_kernel(
    __nv_bfloat16* __restrict__ ssm_out_batch,
    const float* __restrict__ temp_out_batch,
    const __nv_bfloat16* __restrict__ norm_w,
    int intermediate_size, int group_size,
    int M, float eps)
{
    int g = blockIdx.x;
    int m = blockIdx.y;
    if (g >= 8 || m >= M) return;

    int tid = threadIdx.x;
    const float* g_in = temp_out_batch + (size_t)m * intermediate_size + g * group_size;

    float sum_sq = 0.0f;
    for (int i = tid; i < group_size; i += blockDim.x) {
        float v = g_in[i];
        sum_sq += v * v;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    __shared__ float s_warp_sums[8];
    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) s_warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (tid == 0) {
        float total_sq = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; w++) total_sq += s_warp_sums[w];
        float var = total_sq / (float)group_size;
        s_warp_sums[0] = rsqrtf(var + eps);
    }
    __syncthreads();

    float rscale = s_warp_sums[0];
    for (int i = tid; i < group_size; i += blockDim.x) {
        int idx = g * group_size + i;
        float v = g_in[i];
        float w = norm_w ? __bfloat162float(norm_w[idx]) : 1.0f;
        ssm_out_batch[(size_t)m * intermediate_size + idx] = __float2bfloat16(v * rscale * w);
    }
}

void mamba2_nemotron_step_batch_cuda(
    __nv_bfloat16* ssm_out_batch,
    const __nv_bfloat16* in_proj_batch,
    const __nv_bfloat16* conv1d_w,
    const __nv_bfloat16* conv1d_b,
    __nv_bfloat16* conv_state,
    const __nv_bfloat16* A_log,
    const __nv_bfloat16* D,
    const __nv_bfloat16* dt_bias,
    const __nv_bfloat16* norm_w,
    float* ssm_state,
    __nv_bfloat16* conv_buf_batch,
    float* temp_out_batch,
    int M,
    float rms_norm_eps,
    cudaStream_t stream,
    int num_heads,
    int head_dim,
    __nv_bfloat16* conv_state_slot0,
    float* ssm_state_slot0,
    __nv_bfloat16* conv_state_slot1,
    float* ssm_state_slot1,
    __nv_bfloat16* conv_state_slot2,
    float* ssm_state_slot2)
{
    int intermediate_size = num_heads * head_dim;
    int conv_channels = intermediate_size + 2048; // 2 * 8 * 128
    int in_proj_stride = intermediate_size + conv_channels + num_heads; // 10304 or 18560
    int dt_offset = intermediate_size + conv_channels;
    int group_size = intermediate_size / 8;

    int conv_blocks = (conv_channels + 255) / 256;
    mamba2_nemotron_conv_batch_kernel<<<conv_blocks, 256, 0, stream>>>(
        conv_buf_batch, in_proj_batch, conv1d_w, conv1d_b, conv_state, conv_channels, in_proj_stride, intermediate_size, M,
        conv_state_slot0, conv_state_slot1, conv_state_slot2);

    mamba2_nemotron_ssm_step_batch_kernel<<<num_heads, 64, 0, stream>>>(
        temp_out_batch, conv_buf_batch, in_proj_batch, A_log, D, dt_bias, ssm_state,
        num_heads, intermediate_size, conv_channels, in_proj_stride, dt_offset, M,
        ssm_state_slot0, ssm_state_slot1, ssm_state_slot2);

    dim3 grid_norm(8, M);
    mamba2_nemotron_group_norm_batch_kernel<<<grid_norm, 256, 0, stream>>>(
        ssm_out_batch, temp_out_batch, norm_w, intermediate_size, group_size, M, rms_norm_eps);
}

// ════════════════════════════════════════════════════════════════════════════════
//  Batched Kernels for High-Throughput Prefill (Nemotron 3.5 Lightning)
// ════════════════════════════════════════════════════════════════════════════════

template<bool Residual = false>
__global__ void gemm_fp8_scalar_scale_batch_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ mat,
    const uint8_t* __restrict__ weight,
    float scale_val,
    int M, int N, int K)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int m = blockIdx.y;
    if (row >= N || m >= M) return;

    int lane = threadIdx.x; // 0..31
    const uint8_t* row_w = weight + (size_t)row * K;
    const __nv_bfloat16* vec = mat + (size_t)m * K;

    float acc = 0.0f;
    for (int k = lane; k < K; k += 32) {
        float w = fp8_e4m3_to_float_v2(row_w[k]);
        float v = __bfloat162float(vec[k]);
        acc += w * v;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane == 0) {
        acc *= scale_val;
        size_t out_idx = (size_t)m * N + row;
        if constexpr (Residual) {
            float cur = __bfloat162float(out[out_idx]);
            out[out_idx] = __float2bfloat16(cur + acc);
        } else {
            out[out_idx] = __float2bfloat16(acc);
        }
    }
}

void gemm_fp8_scalar_scale_batch_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* mat,
    const uint8_t* weight,
    float scale_val,
    int M, int N, int K,
    cudaStream_t stream)
{
    dim3 threads(32, 4);
    dim3 blocks((N + 3) / 4, M);
    gemm_fp8_scalar_scale_batch_kernel<false><<<blocks, threads, 0, stream>>>(
        out, mat, weight, scale_val, M, N, K);
}

void gemm_fp8_scalar_scale_residual_batch_cuda(
    __nv_bfloat16* inout,
    const __nv_bfloat16* mat,
    const uint8_t* weight,
    float scale_val,
    int M, int N, int K,
    cudaStream_t stream)
{
    dim3 threads(32, 4);
    dim3 blocks((N + 3) / 4, M);
    gemm_fp8_scalar_scale_batch_kernel<true><<<blocks, threads, 0, stream>>>(
        inout, mat, weight, scale_val, M, N, K);
}

template<typename TOut, bool Residual = false, bool Scaled = false, bool Relu2 = false>
__launch_bounds__(256)
__global__ void gemm_nvfp4_block16_batch_kernel(
    TOut* __restrict__ out,
    const __nv_bfloat16* __restrict__ mat,
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scale_fp8,
    float scale_2,
    float alpha,
    int M, int N, int K)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int m_base = blockIdx.y * 16;
    if (row >= N || m_base >= M) return;

    int lane = threadIdx.x; // 0..31
    int num_groups = K / 16;

    const uint8_t* row_w = weight + (size_t)row * (K / 2);
    const uint8_t* row_s = scale_fp8 + (size_t)row * num_groups;

    int active_m = min(16, M - m_base);
    const __nv_bfloat16* vecs[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        vecs[i] = (i < active_m) ? (mat + (size_t)(m_base + i) * K) : nullptr;
    }

    float acc[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) acc[i] = 0.0f;

    for (int g = lane; g < num_groups; g += 32) {
        float s_val = fp8_e4m3_to_float_v2(row_s[g]);
        uint64_t w_packed = *(const uint64_t*)(row_w + g * 8);

        float sw[16];
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            uint8_t byte_val = (uint8_t)(w_packed >> (b * 8));
            float2 w2 = fp4_e2m1_to_float2(byte_val);
            sw[b * 2 + 0] = w2.x * s_val;
            sw[b * 2 + 1] = w2.y * s_val;
        }
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            uint8_t byte_val = (uint8_t)(w_packed >> ((b + 4) * 8));
            float2 w2 = fp4_e2m1_to_float2(byte_val);
            sw[8 + b * 2 + 0] = w2.x * s_val;
            sw[8 + b * 2 + 1] = w2.y * s_val;
        }

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            if (i >= active_m) break;
            uint4 v_p0 = *(const uint4*)(vecs[i] + g * 16);
            uint4 v_p1 = *(const uint4*)(vecs[i] + g * 16 + 8);
            const __nv_bfloat16* vp0 = (const __nv_bfloat16*)&v_p0;
            const __nv_bfloat16* vp1 = (const __nv_bfloat16*)&v_p1;
            #pragma unroll
            for (int b = 0; b < 8; b++) acc[i] = __fmaf_rn(sw[b], __bfloat162float(vp0[b]), acc[i]);
            #pragma unroll
            for (int b = 0; b < 8; b++) acc[i] = __fmaf_rn(sw[8 + b], __bfloat162float(vp1[b]), acc[i]);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            acc[i] += __shfl_down_sync(0xFFFFFFFF, acc[i], offset);
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            if (i >= active_m) break;
            float val = acc[i] * scale_2;
            if constexpr (Scaled) val *= alpha;
            if constexpr (Relu2) {
                val = (val > 0.0f) ? (val * val) : 0.0f;
            }
            size_t out_idx = (size_t)(m_base + i) * N + row;
            if constexpr (Residual) {
                if constexpr (std::is_same_v<TOut, __nv_bfloat16>) {
                    float cur = __bfloat162float(out[out_idx]);
                    out[out_idx] = __float2bfloat16(cur + val);
                } else {
                    out[out_idx] = (TOut)(out[out_idx] + val);
                }
            } else {
                if constexpr (std::is_same_v<TOut, __nv_bfloat16>) {
                    out[out_idx] = __float2bfloat16(val);
                } else {
                    out[out_idx] = (TOut)val;
                }
            }
        }
    }
}

void gemm_nvfp4_batch_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* mat,
    const uint8_t* weight,
    const uint8_t* scale_fp8,
    float scale_2,
    int M, int N, int K,
    cudaStream_t stream)
{
    dim3 threads(32, 4);
    dim3 blocks((N + 3) / 4, (M + 15) / 16);
    gemm_nvfp4_block16_batch_kernel<__nv_bfloat16, false, false, false><<<blocks, threads, 0, stream>>>(
        out, mat, weight, scale_fp8, scale_2, 1.0f, M, N, K);
}

void gemm_nvfp4_relu2_batch_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* mat,
    const uint8_t* weight,
    const uint8_t* scale_fp8,
    float scale_2,
    int M, int N, int K,
    cudaStream_t stream)
{
    dim3 threads(32, 4);
    dim3 blocks((N + 3) / 4, (M + 15) / 16);
    gemm_nvfp4_block16_batch_kernel<__nv_bfloat16, false, false, true><<<blocks, threads, 0, stream>>>(
        out, mat, weight, scale_fp8, scale_2, 1.0f, M, N, K);
}

void gemm_nvfp4_residual_batch_cuda(
    __nv_bfloat16* inout,
    const __nv_bfloat16* mat,
    const uint8_t* weight,
    const uint8_t* scale_fp8,
    float scale_2,
    int M, int N, int K,
    cudaStream_t stream)
{
    dim3 threads(32, 4);
    dim3 blocks((N + 3) / 4, (M + 15) / 16);
    gemm_nvfp4_block16_batch_kernel<__nv_bfloat16, true, false><<<blocks, threads, 0, stream>>>(
        inout, mat, weight, scale_fp8, scale_2, 1.0f, M, N, K);
}

// ════════════════════════════════════════════════════════════════════════════════
//  Nemotron 3.5 Lightning Fused MoE Batch Kernels
// ════════════════════════════════════════════════════════════════════════════════

__global__ void moe_nemotron_up_relu2_batch_kernel(
    __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ hidden_in,
    const int32_t* __restrict__ topk_ids,
    const void* const* __restrict__ expert_ptrs,
    const float* __restrict__ expert_scales,
    int M, int N, int K,
    int top_k = 6,
    int n_experts = 128)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int k = blockIdx.y;
    int m = blockIdx.z;

    if (row >= N || k >= top_k || m >= M) return;

    size_t out_idx = ((size_t)m * top_k + k) * N + row;
    int eid = topk_ids[m * top_k + k];

    if (eid < 0 || eid >= n_experts || !expert_ptrs || !expert_ptrs[eid]) {
        if (threadIdx.x == 0) {
            gate_out[out_idx] = __float2bfloat16(0.0f);
        }
        return;
    }

    const uint8_t* exp_ptr = (const uint8_t*)expert_ptrs[eid];
    const uint8_t* up_w = exp_ptr;
    const uint8_t* up_s = exp_ptr + (size_t)N * (K / 2);
    float scale_2 = expert_scales ? expert_scales[eid * 2 + 0] : 1.0f;

    int lane = threadIdx.x; // 0..31
    int num_groups = K / 16;

    const uint8_t* row_w = up_w + (size_t)row * (K / 2);
    const uint8_t* row_s = up_s + (size_t)row * num_groups;
    const __nv_bfloat16* vec = hidden_in + (size_t)m * K;

    float acc = 0.0f;

    for (int g = lane; g < num_groups; g += 32) {
        float s_val = fp8_e4m3_to_float_v2(row_s[g]);
        uint64_t w_packed = *(const uint64_t*)(row_w + g * 8);
        uint4 v_packed0 = *(const uint4*)(vec + g * 16);
        uint4 v_packed1 = *(const uint4*)(vec + g * 16 + 8);
        const __nv_bfloat16* v_ptr0 = (const __nv_bfloat16*)&v_packed0;
        const __nv_bfloat16* v_ptr1 = (const __nv_bfloat16*)&v_packed1;

        float group_dot = 0.0f;
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            uint8_t byte_val = (uint8_t)(w_packed >> (b * 8));
            float2 w2 = fp4_e2m1_to_float2(byte_val);
            float v0 = __bfloat162float(v_ptr0[b * 2]);
            float v1 = __bfloat162float(v_ptr0[b * 2 + 1]);
            group_dot += w2.x * v0 + w2.y * v1;
        }
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            uint8_t byte_val = (uint8_t)(w_packed >> ((b + 4) * 8));
            float2 w2 = fp4_e2m1_to_float2(byte_val);
            float v0 = __bfloat162float(v_ptr1[b * 2]);
            float v1 = __bfloat162float(v_ptr1[b * 2 + 1]);
            group_dot += w2.x * v0 + w2.y * v1;
        }
        acc += group_dot * s_val;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane == 0) {
        acc *= scale_2;
        float r2 = (acc > 0.0f) ? (acc * acc) : 0.0f;
        gate_out[out_idx] = __float2bfloat16(r2);
    }
}

__global__ void moe_nemotron_down_scaled_residual_batch_kernel(
    __nv_bfloat16* __restrict__ hidden_out,
    const __nv_bfloat16* __restrict__ gate_in,
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    const void* const* __restrict__ expert_ptrs,
    const float* __restrict__ expert_scales,
    int M, int N, int K,
    int top_k = 6,
    int n_experts = 128,
    int down_offset_padding = 0)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int m = blockIdx.y;

    if (row >= N || m >= M) return;

    int lane = threadIdx.x; // 0..31
    int num_groups = K / 16;

    size_t up_w_bytes = (size_t)K * (N / 2);
    size_t up_s_bytes = (size_t)K * (N / 16);
    size_t down_w_bytes = (size_t)N * (K / 2);

    float total_down_acc = 0.0f;

    for (int k = 0; k < top_k; k++) {
        int eid = topk_ids[m * top_k + k];
        float weight = topk_weights[m * top_k + k];
        if (eid < 0 || eid >= n_experts || weight == 0.0f || !expert_ptrs) continue;

        const void* ptr = expert_ptrs[eid];
        if (!ptr) continue;

        const uint8_t* exp_ptr = (const uint8_t*)ptr;
        const uint8_t* down_w = exp_ptr + up_w_bytes + up_s_bytes + down_offset_padding;
        const uint8_t* down_s = down_w + down_w_bytes;
        float scale_2 = expert_scales ? expert_scales[eid * 2 + 1] : 1.0f;

        const uint8_t* row_w = down_w + (size_t)row * (K / 2);
        const uint8_t* row_s = down_s + (size_t)row * num_groups;
        const __nv_bfloat16* vec = gate_in + ((size_t)m * top_k + k) * K;

        float acc = 0.0f;

        for (int g = lane; g < num_groups; g += 32) {
            float s_val = fp8_e4m3_to_float_v2(row_s[g]);
            uint64_t w_packed = *(const uint64_t*)(row_w + g * 8);
            uint4 v_packed0 = *(const uint4*)(vec + g * 16);
            uint4 v_packed1 = *(const uint4*)(vec + g * 16 + 8);
            const __nv_bfloat16* v_ptr0 = (const __nv_bfloat16*)&v_packed0;
            const __nv_bfloat16* v_ptr1 = (const __nv_bfloat16*)&v_packed1;

            float group_dot = 0.0f;
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                uint8_t byte_val = (uint8_t)(w_packed >> (b * 8));
                float2 w2 = fp4_e2m1_to_float2(byte_val);
                float v0 = __bfloat162float(v_ptr0[b * 2]);
                float v1 = __bfloat162float(v_ptr0[b * 2 + 1]);
                group_dot += w2.x * v0 + w2.y * v1;
            }
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                uint8_t byte_val = (uint8_t)(w_packed >> ((b + 4) * 8));
                float2 w2 = fp4_e2m1_to_float2(byte_val);
                float v0 = __bfloat162float(v_ptr1[b * 2]);
                float v1 = __bfloat162float(v_ptr1[b * 2 + 1]);
                group_dot += w2.x * v0 + w2.y * v1;
            }
            acc += group_dot * s_val;
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        }

        if (lane == 0) {
            total_down_acc += acc * (scale_2 * weight);
        }
    }

    if (lane == 0) {
        size_t hidden_idx = (size_t)m * N + row;
        float cur = __bfloat162float(hidden_out[hidden_idx]);
        hidden_out[hidden_idx] = __float2bfloat16(cur + total_down_acc);
    }
}

__global__ void moe_bin_tokens_kernel(
    int32_t* __restrict__ expert_offsets,       // [n_experts + 1]
    int32_t* __restrict__ expert_token_indices, // [total_selections]
    int32_t* __restrict__ expert_token_slots,   // [total_selections]
    const int32_t* __restrict__ topk_ids,       // [total_selections]
    int total_selections,
    int top_k,
    int n_experts)
{
    __shared__ int counts[512];
    __shared__ int offsets[513];
    __shared__ int cur_pos[512];

    int tid = threadIdx.x;

    // 1. Initialize counts
    for (int i = tid; i < n_experts; i += blockDim.x) {
        counts[i] = 0;
    }
    __syncthreads();

    // 2. Count selections per expert
    for (int i = tid; i < total_selections; i += blockDim.x) {
        int eid = topk_ids[i];
        if (eid >= 0 && eid < n_experts) {
            atomicAdd(&counts[eid], 1);
        }
    }
    __syncthreads();

    // 3. Single thread prefix sum
    if (tid == 0) {
        int sum = 0;
        offsets[0] = 0;
        for (int e = 0; e < n_experts; e++) {
            sum += counts[e];
            offsets[e + 1] = sum;
            cur_pos[e] = offsets[e];
        }
    }
    __syncthreads();

    // Write expert_offsets to global memory
    for (int i = tid; i <= n_experts; i += blockDim.x) {
        expert_offsets[i] = offsets[i];
    }

    // 4. Scatter token and slot indices into contiguous expert bins
    for (int i = tid; i < total_selections; i += blockDim.x) {
        int eid = topk_ids[i];
        if (eid >= 0 && eid < n_experts) {
            int pos = atomicAdd(&cur_pos[eid], 1);
            expert_token_indices[pos] = i / top_k;
            expert_token_slots[pos] = i % top_k;
        }
    }
}

void moe_bin_tokens_cuda(
    int32_t* expert_offsets,
    int32_t* expert_token_indices,
    int32_t* expert_token_slots,
    const int32_t* topk_ids,
    int total_selections,
    int top_k,
    int n_experts,
    cudaStream_t stream)
{
    int threads = (n_experts > 256) ? 512 : ((n_experts > 128) ? 256 : 128);
    moe_bin_tokens_kernel<<<1, threads, 0, stream>>>(
        expert_offsets, expert_token_indices, expert_token_slots, topk_ids, total_selections, top_k, n_experts);
}

__launch_bounds__(256)
__global__ void moe_nemotron_binned_up_relu2_batch_kernel(
    __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ hidden_in,
    const int32_t* __restrict__ expert_offsets,
    const int32_t* __restrict__ expert_token_indices,
    const int32_t* __restrict__ expert_token_slots,
    const void* const* __restrict__ expert_ptrs,
    const float* expert_scales,
    int M, int N, int K,
    int top_k = 6,
    int n_experts = 128)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int eid = blockIdx.y;

    if (row >= N || eid >= n_experts) return;

    int start_idx = expert_offsets[eid];
    int end_idx = expert_offsets[eid + 1];
    int num_toks = end_idx - start_idx;

    if (num_toks <= 0) return;
    if (!expert_ptrs || !expert_ptrs[eid]) return;

    const uint8_t* exp_ptr = (const uint8_t*)expert_ptrs[eid];
    const uint8_t* up_w = exp_ptr;
    const uint8_t* up_s = exp_ptr + (size_t)N * (K / 2);
    float scale_2 = expert_scales ? expert_scales[eid * 2 + 0] : 1.0f;

    int lane = threadIdx.x; // 0..31
    int num_groups = K / 16;

    const uint8_t* row_w = up_w + (size_t)row * (K / 2);
    const uint8_t* row_s = up_s + (size_t)row * num_groups;

    for (int t_base = 0; t_base < num_toks; t_base += 16) {
        int active_t = min(16, num_toks - t_base);
        const __nv_bfloat16* vecs[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            if (i < active_t) {
                int m = expert_token_indices[start_idx + t_base + i];
                vecs[i] = hidden_in + (size_t)m * K;
            } else {
                vecs[i] = nullptr;
            }
        }

        float acc[16] = {0.0f};

        for (int g = lane; g < num_groups; g += 32) {
            float s_val = fp8_e4m3_to_float_v2(row_s[g]);
            uint64_t w_packed = *(const uint64_t*)(row_w + g * 8);

            float sw[16];
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                uint8_t byte_val = (uint8_t)(w_packed >> (b * 8));
                float2 w2 = fp4_e2m1_to_float2(byte_val);
                sw[b * 2 + 0] = w2.x * s_val;
                sw[b * 2 + 1] = w2.y * s_val;
            }
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                uint8_t byte_val = (uint8_t)(w_packed >> ((b + 4) * 8));
                float2 w2 = fp4_e2m1_to_float2(byte_val);
                sw[8 + b * 2 + 0] = w2.x * s_val;
                sw[8 + b * 2 + 1] = w2.y * s_val;
            }

            #pragma unroll
            for (int t = 0; t < 16; t++) {
                if (t >= active_t) break;
                uint4 v_p0 = *(const uint4*)(vecs[t] + g * 16);
                uint4 v_p1 = *(const uint4*)(vecs[t] + g * 16 + 8);
                const __nv_bfloat16* vp0 = (const __nv_bfloat16*)&v_p0;
                const __nv_bfloat16* vp1 = (const __nv_bfloat16*)&v_p1;
                #pragma unroll
                for (int b = 0; b < 8; b++) acc[t] = __fmaf_rn(sw[b], __bfloat162float(vp0[b]), acc[t]);
                #pragma unroll
                for (int b = 0; b < 8; b++) acc[t] = __fmaf_rn(sw[8 + b], __bfloat162float(vp1[b]), acc[t]);
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            #pragma unroll
            for (int i = 0; i < 16; i++) acc[i] += __shfl_down_sync(0xFFFFFFFF, acc[i], offset);
        }

        if (lane == 0) {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                if (i >= active_t) break;
                int m = expert_token_indices[start_idx + t_base + i];
                int k = expert_token_slots[start_idx + t_base + i];
                float r = acc[i] * scale_2;
                float r2 = (r > 0.0f) ? (r * r) : 0.0f;
                gate_out[((size_t)m * top_k + k) * N + row] = __float2bfloat16(r2);
            }
        }
    }
}

__launch_bounds__(128, 7)
__global__ void moe_nemotron_binned_down_scaled_residual_batch_kernel(
    __nv_bfloat16* __restrict__ hidden_out,
    const __nv_bfloat16* __restrict__ gate_in,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ expert_offsets,
    const int32_t* __restrict__ expert_token_indices,
    const int32_t* __restrict__ expert_token_slots,
    const void* const* __restrict__ expert_ptrs,
    const float* expert_scales,
    int M, int N, int K,
    int top_k = 6,
    int n_experts = 128,
    int down_offset_padding = 0)
{
    int r = blockIdx.x * blockDim.y + threadIdx.y;
    int row0 = r * 2;
    int row1 = row0 + 1;
    int eid = blockIdx.y;

    if (row0 >= N || eid >= n_experts) return;

    int start_idx = expert_offsets[eid];
    int end_idx = expert_offsets[eid + 1];
    int num_toks = end_idx - start_idx;

    if (num_toks <= 0) return;
    if (!expert_ptrs || !expert_ptrs[eid]) return;

    size_t up_w_bytes = (size_t)K * (N / 2);
    size_t up_s_bytes = (size_t)K * (N / 16);
    size_t down_w_bytes = (size_t)N * (K / 2);

    const uint8_t* exp_ptr = (const uint8_t*)expert_ptrs[eid];
    const uint8_t* down_w = exp_ptr + up_w_bytes + up_s_bytes + down_offset_padding;
    const uint8_t* down_s = down_w + down_w_bytes;
    float scale_2 = expert_scales ? expert_scales[eid * 2 + 1] : 1.0f;

    int lane = threadIdx.x; // 0..31
    int num_groups = K / 16;

    const uint8_t* row0_w = down_w + (size_t)row0 * (K / 2);
    const uint8_t* row0_s = down_s + (size_t)row0 * num_groups;
    const uint8_t* row1_w = down_w + (size_t)row1 * (K / 2);
    const uint8_t* row1_s = down_s + (size_t)row1 * num_groups;

    for (int t_base = 0; t_base < num_toks; t_base += 8) {
        int active_t = min(8, num_toks - t_base);

        const __nv_bfloat16* vecs[8];
        float scaled_weights[8];
        int tok_m[8];

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (i < active_t) {
                int m = expert_token_indices[start_idx + t_base + i];
                int k = expert_token_slots[start_idx + t_base + i];
                tok_m[i] = m;
                float w = topk_weights ? topk_weights[m * top_k + k] : 1.0f;
                scaled_weights[i] = w * scale_2;
                vecs[i] = gate_in + ((size_t)m * top_k + k) * K;
            } else {
                scaled_weights[i] = 0.0f;
                tok_m[i] = 0;
                vecs[i] = nullptr;
            }
        }

        float acc0[8] = {0.0f};
        float acc1[8] = {0.0f};

        for (int g = lane; g < num_groups; g += 32) {
            float s_val0 = fp8_e4m3_to_float_v2(row0_s[g]);
            uint64_t w_packed0 = *(const uint64_t*)(row0_w + g * 8);

            float sw0[16];
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                uint8_t byte_val = (uint8_t)(w_packed0 >> (b * 8));
                float2 w2 = fp4_e2m1_to_float2(byte_val);
                sw0[b * 2 + 0] = w2.x * s_val0;
                sw0[b * 2 + 1] = w2.y * s_val0;
            }
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                uint8_t byte_val = (uint8_t)(w_packed0 >> ((b + 4) * 8));
                float2 w2 = fp4_e2m1_to_float2(byte_val);
                sw0[8 + b * 2 + 0] = w2.x * s_val0;
                sw0[8 + b * 2 + 1] = w2.y * s_val0;
            }

            float s_val1 = fp8_e4m3_to_float_v2(row1_s[g]);
            uint64_t w_packed1 = *(const uint64_t*)(row1_w + g * 8);

            float sw1[16];
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                uint8_t byte_val = (uint8_t)(w_packed1 >> (b * 8));
                float2 w2 = fp4_e2m1_to_float2(byte_val);
                sw1[b * 2 + 0] = w2.x * s_val1;
                sw1[b * 2 + 1] = w2.y * s_val1;
            }
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                uint8_t byte_val = (uint8_t)(w_packed1 >> ((b + 4) * 8));
                float2 w2 = fp4_e2m1_to_float2(byte_val);
                sw1[8 + b * 2 + 0] = w2.x * s_val1;
                sw1[8 + b * 2 + 1] = w2.y * s_val1;
            }

            #pragma unroll
            for (int t = 0; t < 8; t++) {
                if (t >= active_t) break;
                uint4 v_p0 = *(const uint4*)(vecs[t] + g * 16);
                uint4 v_p1 = *(const uint4*)(vecs[t] + g * 16 + 8);
                const __nv_bfloat16* vp0 = (const __nv_bfloat16*)&v_p0;
                const __nv_bfloat16* vp1 = (const __nv_bfloat16*)&v_p1;
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    float f0 = __bfloat162float(vp0[b]);
                    acc0[t] = __fmaf_rn(sw0[b], f0, acc0[t]);
                    acc1[t] = __fmaf_rn(sw1[b], f0, acc1[t]);
                }
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    float f1 = __bfloat162float(vp1[b]);
                    acc0[t] = __fmaf_rn(sw0[8 + b], f1, acc0[t]);
                    acc1[t] = __fmaf_rn(sw1[8 + b], f1, acc1[t]);
                }
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                acc0[i] += __shfl_down_sync(0xFFFFFFFF, acc0[i], offset);
                acc1[i] += __shfl_down_sync(0xFFFFFFFF, acc1[i], offset);
            }
        }

        if (lane == 0) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                if (i >= active_t) break;
                float sw = scaled_weights[i];
                if (sw != 0.0f) {
                    float val0 = acc0[i] * sw;
                    float val1 = acc1[i] * sw;
                    if (row1 < N) {
                        __nv_bfloat162 v2 = __floats2bfloat162_rn(val0, val1);
                        atomicAdd((__nv_bfloat162*)&hidden_out[(size_t)tok_m[i] * N + row0], v2);
                    } else {
                        atomicAdd((__nv_bfloat16*)&hidden_out[(size_t)tok_m[i] * N + row0], __float2bfloat16(val0));
                    }
                }
            }
        }
    }
}

void moe_nemotron_up_relu2_batch_cuda(
    __nv_bfloat16* gate_out,
    const __nv_bfloat16* hidden_in,
    const int32_t* topk_ids,
    const int32_t* expert_offsets,
    const int32_t* expert_token_indices,
    const int32_t* expert_token_slots,
    const void* const* expert_ptrs,
    const float* expert_scales,
    int M, int N, int K,
    cudaStream_t stream,
    int top_k,
    int n_experts)
{
    if (M > 1 && expert_offsets != nullptr) {
        dim3 threads(32, 4);
        dim3 blocks((N + 3) / 4, n_experts);
        moe_nemotron_binned_up_relu2_batch_kernel<<<blocks, threads, 0, stream>>>(
            gate_out, hidden_in, expert_offsets, expert_token_indices, expert_token_slots,
            expert_ptrs, expert_scales, M, N, K, top_k, n_experts);
    } else {
        dim3 threads(32, 4);
        dim3 blocks((N + 3) / 4, top_k, M);
        moe_nemotron_up_relu2_batch_kernel<<<blocks, threads, 0, stream>>>(
            gate_out, hidden_in, topk_ids, expert_ptrs, expert_scales, M, N, K, top_k, n_experts);
    }
}

void moe_nemotron_down_scaled_residual_batch_cuda(
    __nv_bfloat16* hidden_out,
    const __nv_bfloat16* gate_in,
    const int32_t* topk_ids,
    const float* topk_weights,
    const int32_t* expert_offsets,
    const int32_t* expert_token_indices,
    const int32_t* expert_token_slots,
    const void* const* expert_ptrs,
    const float* expert_scales,
    int M, int N, int K,
    cudaStream_t stream,
    int top_k,
    int n_experts,
    int down_offset_padding)
{
    if (M > 1 && expert_offsets != nullptr) {
        dim3 threads(32, 4);
        dim3 blocks(((N / 2) + 3) / 4, n_experts);
        moe_nemotron_binned_down_scaled_residual_batch_kernel<<<blocks, threads, 0, stream>>>(
            hidden_out, gate_in, topk_weights, expert_offsets, expert_token_indices, expert_token_slots,
            expert_ptrs, expert_scales, M, N, K, top_k, n_experts, down_offset_padding);
    } else {
        dim3 threads(32, 4);
        dim3 blocks((N + 3) / 4, M);
        moe_nemotron_down_scaled_residual_batch_kernel<<<blocks, threads, 0, stream>>>(
            hidden_out, gate_in, topk_ids, topk_weights, expert_ptrs, expert_scales, M, N, K, top_k, n_experts, down_offset_padding);
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// BATCHED IQ2_XXS GEMM & MOE ROUTING KERNELS (Nemotron Bolt 24GB Support)
// ════════════════════════════════════════════════════════════════════════════════

template<bool Relu2, bool Residual>
__global__ void gemm_iq2_xxs_batch_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ vec,
    const block_iq2_xxs* __restrict__ weight,
    int M, int N, int K)
{
    __shared__ uint64_t s_grid[256];
    __shared__ uint8_t s_signs[128];

    int tid = threadIdx.y * 32 + threadIdx.x;
    if (tid < 128) {
        s_signs[tid] = c_ksigns_iq2xs[tid];
    }
    s_grid[tid] = c_iq2xxs_grid[tid];
    if (blockDim.y * 32 < 256) {
        s_grid[tid + 128] = c_iq2xxs_grid[tid + 128];
    }
    __syncthreads();

    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int m_base = blockIdx.y * 8;

    if (row >= N || m_base >= M) return;

    int lane = threadIdx.x;
    int n_blocks = K / 256;
    const block_iq2_xxs* row_w = weight + (size_t)row * n_blocks;

    int active_m = min(8, M - m_base);
    const __nv_bfloat16* token_vecs[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        token_vecs[i] = (i < active_m) ? (vec + (size_t)(m_base + i) * K) : nullptr;
    }

    int l = lane / 8;
    int j = lane % 8;
    uint8_t kmask = 1 << j;

    float sums[8] = {0.0f};

    for (int b = 0; b < n_blocks; b++) {
        const block_iq2_xxs* blk = &row_w[b];
        float d = __half2float(blk->d);
        const uint16_t* qs = blk->qs;

        #pragma unroll 4
        for (int ib32 = 0; ib32 < 8; ib32++) {
            uint32_t q0 = qs[ib32 * 4 + 0];
            uint32_t q1 = qs[ib32 * 4 + 1];
            uint32_t q2 = qs[ib32 * 4 + 2];
            uint32_t q3 = qs[ib32 * 4 + 3];

            uint32_t aux0 = q0 | (q1 << 16);
            uint32_t aux1 = q2 | (q3 << 16);
            float db = d * (0.5f + (aux1 >> 28)) * 0.25f;

            uint8_t grid_idx = (aux0 >> (8 * l)) & 0xFF;
            uint8_t sign_idx = (aux1 >> (7 * l)) & 0x7F;

            uint64_t grid_val = s_grid[grid_idx];
            uint8_t sign_val = s_signs[sign_idx];

            uint8_t byte_val = (grid_val >> (8 * j)) & 0xFF;
            float sign_mult = (sign_val & kmask) ? -1.0f : 1.0f;
            float w = db * (float)byte_val * sign_mult;

            int col_idx = (b << 8) + (ib32 << 5) + lane;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                if (i < active_m) {
                    float a = __bfloat162float(token_vecs[i][col_idx]);
                    sums[i] = __fmaf_rn(w, a, sums[i]);
                }
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            sums[i] += __shfl_down_sync(0xffffffff, sums[i], offset);
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (i >= active_m) break;
            int m = m_base + i;
            float sum = sums[i];
            if constexpr (Relu2) {
                sum = (sum > 0.0f) ? (sum * sum) : 0.0f;
                out[(size_t)m * N + row] = __float2bfloat16(sum);
            } else if constexpr (Residual) {
                float cur = __bfloat162float(out[(size_t)m * N + row]);
                out[(size_t)m * N + row] = __float2bfloat16(cur + sum);
            } else {
                out[(size_t)m * N + row] = __float2bfloat16(sum);
            }
        }
    }
}

void gemm_iq2_xxs_relu2_batch_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* vec,
    const block_iq2_xxs* weight,
    int M, int N, int K,
    cudaStream_t stream)
{
    dim3 threads(32, 4);
    dim3 blocks((N + 3) / 4, (M + 7) / 8);
    gemm_iq2_xxs_batch_kernel<true, false><<<blocks, threads, 0, stream>>>(out, vec, weight, M, N, K);
}

void gemm_iq2_xxs_residual_batch_cuda(
    __nv_bfloat16* inout,
    const __nv_bfloat16* vec,
    const block_iq2_xxs* weight,
    int M, int N, int K,
    cudaStream_t stream)
{
    dim3 threads(32, 4);
    dim3 blocks((N + 3) / 4, (M + 7) / 8);
    gemm_iq2_xxs_batch_kernel<false, true><<<blocks, threads, 0, stream>>>(inout, vec, weight, M, N, K);
}

__launch_bounds__(128)
__global__ void moe_iq2_xxs_binned_up_relu2_batch_kernel(
    __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ hidden_in,
    const int32_t* __restrict__ expert_offsets,
    const int32_t* __restrict__ expert_token_indices,
    const int32_t* __restrict__ expert_token_slots,
    const void* const* __restrict__ expert_ptrs,
    const float* expert_scales,
    int M, int N, int K,
    int top_k,
    int n_experts)
{
    __shared__ uint64_t s_grid[256];
    __shared__ uint8_t s_signs[128];

    int tid = threadIdx.y * 32 + threadIdx.x;
    if (tid < 128) {
        s_signs[tid] = c_ksigns_iq2xs[tid];
    }
    s_grid[tid] = c_iq2xxs_grid[tid];
    if (blockDim.y * 32 < 256) {
        s_grid[tid + 128] = c_iq2xxs_grid[tid + 128];
    }
    __syncthreads();

    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int eid = blockIdx.y;

    if (row >= N || eid >= n_experts) return;

    int start_idx = expert_offsets[eid];
    int end_idx = expert_offsets[eid + 1];
    int num_toks = end_idx - start_idx;

    if (num_toks <= 0) return;
    if (!expert_ptrs || !expert_ptrs[eid]) return;

    const block_iq2_xxs* up_w = (const block_iq2_xxs*)expert_ptrs[eid];
    float scale_2 = expert_scales ? expert_scales[eid * 2 + 0] : 1.0f;

    int lane = threadIdx.x;
    int n_blocks = K / 256;
    const block_iq2_xxs* row_w = up_w + (size_t)row * n_blocks;

    int l = lane / 8;
    int j = lane % 8;
    uint8_t kmask = 1 << j;

    for (int t_base = 0; t_base < num_toks; t_base += 8) {
        int active_t = min(8, num_toks - t_base);
        const __nv_bfloat16* vecs[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (i < active_t) {
                int m = expert_token_indices[start_idx + t_base + i];
                vecs[i] = hidden_in + (size_t)m * K;
            } else {
                vecs[i] = nullptr;
            }
        }

        float acc[8] = {0.0f};

        for (int b = 0; b < n_blocks; b++) {
            const block_iq2_xxs* blk = &row_w[b];
            float d = __half2float(blk->d);
            const uint16_t* qs = blk->qs;

            #pragma unroll 4
            for (int ib32 = 0; ib32 < 8; ib32++) {
                uint32_t q0 = qs[ib32 * 4 + 0];
                uint32_t q1 = qs[ib32 * 4 + 1];
                uint32_t q2 = qs[ib32 * 4 + 2];
                uint32_t q3 = qs[ib32 * 4 + 3];

                uint32_t aux0 = q0 | (q1 << 16);
                uint32_t aux1 = q2 | (q3 << 16);
                float db = d * (0.5f + (aux1 >> 28)) * 0.25f;

                uint8_t grid_idx = (aux0 >> (8 * l)) & 0xFF;
                uint8_t sign_idx = (aux1 >> (7 * l)) & 0x7F;

                uint64_t grid_val = s_grid[grid_idx];
                uint8_t sign_val = s_signs[sign_idx];

                uint8_t byte_val = (grid_val >> (8 * j)) & 0xFF;
                float sign_mult = (sign_val & kmask) ? -1.0f : 1.0f;
                float w = db * (float)byte_val * sign_mult;

                int col_idx = (b << 8) + (ib32 << 5) + lane;
                #pragma unroll
                for (int t = 0; t < 8; t++) {
                    if (t < active_t) {
                        float a = __bfloat162float(vecs[t][col_idx]);
                        acc[t] = __fmaf_rn(w, a, acc[t]);
                    }
                }
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            #pragma unroll
            for (int t = 0; t < 8; t++) {
                acc[t] += __shfl_down_sync(0xFFFFFFFF, acc[t], offset);
            }
        }

        if (lane == 0) {
            #pragma unroll
            for (int t = 0; t < 8; t++) {
                if (t >= active_t) break;
                int m = expert_token_indices[start_idx + t_base + t];
                int k = expert_token_slots[start_idx + t_base + t];
                float r = acc[t] * scale_2;
                float r2 = (r > 0.0f) ? (r * r) : 0.0f;
                gate_out[((size_t)m * top_k + k) * N + row] = __float2bfloat16(r2);
            }
        }
    }
}

void moe_iq2_xxs_binned_up_relu2_batch_cuda(
    __nv_bfloat16* gate_out,
    const __nv_bfloat16* hidden_in,
    const int32_t* expert_offsets,
    const int32_t* expert_token_indices,
    const int32_t* expert_token_slots,
    const void* const* expert_ptrs,
    const float* expert_scales,
    int M, int N, int K,
    cudaStream_t stream,
    int top_k,
    int n_experts)
{
    dim3 threads(32, 4);
    dim3 blocks((N + 3) / 4, n_experts);
    moe_iq2_xxs_binned_up_relu2_batch_kernel<<<blocks, threads, 0, stream>>>(
        gate_out, hidden_in, expert_offsets, expert_token_indices, expert_token_slots,
        expert_ptrs, expert_scales, M, N, K, top_k, n_experts);
}

__launch_bounds__(128)
__global__ void moe_iq2_xxs_binned_down_scaled_residual_batch_kernel(
    __nv_bfloat16* __restrict__ hidden_out,
    const __nv_bfloat16* __restrict__ gate_in,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ expert_offsets,
    const int32_t* __restrict__ expert_token_indices,
    const int32_t* __restrict__ expert_token_slots,
    const void* const* __restrict__ expert_ptrs,
    const float* expert_scales,
    int M, int N, int K,
    int top_k,
    int n_experts,
    size_t down_offset_bytes)
{
    __shared__ uint64_t s_grid[256];
    __shared__ uint8_t s_signs[128];

    int tid = threadIdx.y * 32 + threadIdx.x;
    if (tid < 128) {
        s_signs[tid] = c_ksigns_iq2xs[tid];
    }
    s_grid[tid] = c_iq2xxs_grid[tid];
    if (blockDim.y * 32 < 256) {
        s_grid[tid + 128] = c_iq2xxs_grid[tid + 128];
    }
    __syncthreads();

    int r = blockIdx.x * blockDim.y + threadIdx.y;
    int row0 = r * 2;
    int row1 = row0 + 1;
    int eid = blockIdx.y;

    if (row0 >= N || eid >= n_experts) return;

    int start_idx = expert_offsets[eid];
    int end_idx = expert_offsets[eid + 1];
    int num_toks = end_idx - start_idx;

    if (num_toks <= 0) return;
    if (!expert_ptrs || !expert_ptrs[eid]) return;

    const block_iq2_xxs* down_w = (const block_iq2_xxs*)((const uint8_t*)expert_ptrs[eid] + down_offset_bytes);
    float scale_2 = expert_scales ? expert_scales[eid * 2 + 1] : 1.0f;

    int lane = threadIdx.x;
    int n_blocks = K / 256;
    const block_iq2_xxs* row0_w = down_w + (size_t)row0 * n_blocks;
    const block_iq2_xxs* row1_w = down_w + (size_t)row1 * n_blocks;

    int l = lane / 8;
    int j = lane % 8;
    uint8_t kmask = 1 << j;

    for (int t_base = 0; t_base < num_toks; t_base += 8) {
        int active_t = min(8, num_toks - t_base);
        const __nv_bfloat16* vecs[8];
        float scaled_weights[8];
        int tok_m[8];

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (i < active_t) {
                int m = expert_token_indices[start_idx + t_base + i];
                int k = expert_token_slots[start_idx + t_base + i];
                tok_m[i] = m;
                float w = topk_weights ? topk_weights[m * top_k + k] : 1.0f;
                scaled_weights[i] = w * scale_2;
                vecs[i] = gate_in + ((size_t)m * top_k + k) * K;
            } else {
                scaled_weights[i] = 0.0f;
                tok_m[i] = 0;
                vecs[i] = nullptr;
            }
        }

        float acc0[8] = {0.0f};
        float acc1[8] = {0.0f};

        for (int b = 0; b < n_blocks; b++) {
            const block_iq2_xxs* blk0 = &row0_w[b];
            float d0 = __half2float(blk0->d);
            const uint16_t* qs0 = blk0->qs;

            const block_iq2_xxs* blk1 = &row1_w[b];
            float d1 = __half2float(blk1->d);
            const uint16_t* qs1 = blk1->qs;

            #pragma unroll 4
            for (int ib32 = 0; ib32 < 8; ib32++) {
                uint32_t q0_0 = qs0[ib32 * 4 + 0];
                uint32_t q1_0 = qs0[ib32 * 4 + 1];
                uint32_t q2_0 = qs0[ib32 * 4 + 2];
                uint32_t q3_0 = qs0[ib32 * 4 + 3];

                uint32_t aux0_0 = q0_0 | (q1_0 << 16);
                uint32_t aux1_0 = q2_0 | (q3_0 << 16);
                float db0 = d0 * (0.5f + (aux1_0 >> 28)) * 0.25f;

                uint8_t grid_idx0 = (aux0_0 >> (8 * l)) & 0xFF;
                uint8_t sign_idx0 = (aux1_0 >> (7 * l)) & 0x7F;

                uint64_t grid_val0 = s_grid[grid_idx0];
                uint8_t sign_val0 = s_signs[sign_idx0];

                uint8_t byte_val0 = (grid_val0 >> (8 * j)) & 0xFF;
                float sign_mult0 = (sign_val0 & kmask) ? -1.0f : 1.0f;
                float w0 = db0 * (float)byte_val0 * sign_mult0;

                uint32_t q0_1 = qs1[ib32 * 4 + 0];
                uint32_t q1_1 = qs1[ib32 * 4 + 1];
                uint32_t q2_1 = qs1[ib32 * 4 + 2];
                uint32_t q3_1 = qs1[ib32 * 4 + 3];

                uint32_t aux0_1 = q0_1 | (q1_1 << 16);
                uint32_t aux1_1 = q2_1 | (q3_1 << 16);
                float db1 = d1 * (0.5f + (aux1_1 >> 28)) * 0.25f;

                uint8_t grid_idx1 = (aux0_1 >> (8 * l)) & 0xFF;
                uint8_t sign_idx1 = (aux1_1 >> (7 * l)) & 0x7F;

                uint64_t grid_val1 = s_grid[grid_idx1];
                uint8_t sign_val1 = s_signs[sign_idx1];

                uint8_t byte_val1 = (grid_val1 >> (8 * j)) & 0xFF;
                float sign_mult1 = (sign_val1 & kmask) ? -1.0f : 1.0f;
                float w1 = db1 * (float)byte_val1 * sign_mult1;

                int col_idx = (b << 8) + (ib32 << 5) + lane;
                #pragma unroll
                for (int t = 0; t < 8; t++) {
                    if (t < active_t) {
                        float a = __bfloat162float(vecs[t][col_idx]);
                        acc0[t] = __fmaf_rn(w0, a, acc0[t]);
                        acc1[t] = __fmaf_rn(w1, a, acc1[t]);
                    }
                }
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            #pragma unroll
            for (int t = 0; t < 8; t++) {
                acc0[t] += __shfl_down_sync(0xFFFFFFFF, acc0[t], offset);
                acc1[t] += __shfl_down_sync(0xFFFFFFFF, acc1[t], offset);
            }
        }

        if (lane == 0) {
            #pragma unroll
            for (int t = 0; t < 8; t++) {
                if (t >= active_t) break;
                float sw = scaled_weights[t];
                if (sw != 0.0f) {
                    float val0 = acc0[t] * sw;
                    float val1 = acc1[t] * sw;
                    if (row1 < N) {
                        __nv_bfloat162 v2 = __floats2bfloat162_rn(val0, val1);
                        atomicAdd((__nv_bfloat162*)&hidden_out[(size_t)tok_m[t] * N + row0], v2);
                    } else {
                        atomicAdd((__nv_bfloat16*)&hidden_out[(size_t)tok_m[t] * N + row0], __float2bfloat16(val0));
                    }
                }
            }
        }
    }
}

void moe_iq2_xxs_binned_down_scaled_residual_batch_cuda(
    __nv_bfloat16* hidden_out,
    const __nv_bfloat16* gate_in,
    const float* topk_weights,
    const int32_t* expert_offsets,
    const int32_t* expert_token_indices,
    const int32_t* expert_token_slots,
    const void* const* expert_ptrs,
    const float* expert_scales,
    int M, int N, int K,
    cudaStream_t stream,
    int top_k,
    int n_experts,
    size_t down_offset_bytes)
{
    dim3 threads(32, 4);
    dim3 blocks(((N / 2) + 3) / 4, n_experts);
    moe_iq2_xxs_binned_down_scaled_residual_batch_kernel<<<blocks, threads, 0, stream>>>(
        hidden_out, gate_in, topk_weights, expert_offsets, expert_token_indices, expert_token_slots,
        expert_ptrs, expert_scales, M, N, K, top_k, n_experts, down_offset_bytes);
}



