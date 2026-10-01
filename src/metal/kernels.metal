#include <metal_stdlib>
using namespace metal;

// ════════════════════════════════════════════════════════════════════════════════
//  Helpers & Conversions
// ════════════════════════════════════════════════════════════════════════════════

inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

// ════════════════════════════════════════════════════════════════════════════════
//  RMSNorm Kernels
// ════════════════════════════════════════════════════════════════════════════════

kernel void rms_norm_kernel(
    device bfloat* out [[buffer(0)]],
    device const bfloat* x [[buffer(1)]],
    device const bfloat* weight [[buffer(2)]],
    constant int& dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float sdata[32];

    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += threads_per_group) {
        float v = float(x[i]);
        sum_sq += v * v;
    }

    sum_sq = simd_sum(sum_sq);
    if (simd_lane == 0) {
        sdata[simd_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float total = (simd_lane < (threads_per_group >> 5)) ? sdata[simd_lane] : 0.0f;
        total = simd_sum(total);
        if (simd_lane == 0) {
            sdata[0] = rsqrt(total / float(dim) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rsqrt_val = sdata[0];
    for (int i = tid; i < dim; i += threads_per_group) {
        float v = float(x[i]) * rsqrt_val;
        float w = float(weight[i]);
        out[i] = bfloat(v * w);
    }
}

kernel void rms_norm_one_centered_kernel(
    device bfloat* out [[buffer(0)]],
    device const bfloat* x [[buffer(1)]],
    device const bfloat* weight [[buffer(2)]],
    constant int& dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float sdata[32];

    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += threads_per_group) {
        float v = float(x[i]);
        sum_sq += v * v;
    }

    sum_sq = simd_sum(sum_sq);
    if (simd_lane == 0) {
        sdata[simd_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float total = (simd_lane < (threads_per_group >> 5)) ? sdata[simd_lane] : 0.0f;
        total = simd_sum(total);
        if (simd_lane == 0) {
            sdata[0] = rsqrt(total / float(dim) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rsqrt_val = sdata[0];
    for (int i = tid; i < dim; i += threads_per_group) {
        float v = float(x[i]) * rsqrt_val;
        float w = 1.0f + float(weight[i]);
        out[i] = bfloat(v * w);
    }
}

kernel void rms_norm_one_centered_batched_kernel(
    device bfloat* out [[buffer(0)]],
    device const bfloat* x [[buffer(1)]],
    device const bfloat* weight [[buffer(2)]],
    constant int& n_rows [[buffer(3)]],
    constant int& dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    if (row >= uint(n_rows)) return;
    threadgroup float sdata[32];

    device const bfloat* x_row = x + row * dim;
    device bfloat* out_row = out + row * dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += threads_per_group) {
        float v = float(x_row[i]);
        sum_sq += v * v;
    }

    sum_sq = simd_sum(sum_sq);
    if (simd_lane == 0) {
        sdata[simd_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float total = (simd_lane < (threads_per_group >> 5)) ? sdata[simd_lane] : 0.0f;
        total = simd_sum(total);
        if (simd_lane == 0) {
            sdata[0] = rsqrt(total / float(dim) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rsqrt_val = sdata[0];
    for (int i = tid; i < dim; i += threads_per_group) {
        float v = float(x_row[i]) * rsqrt_val;
        float w = 1.0f + float(weight[i]);
        out_row[i] = bfloat(v * w);
    }
}

kernel void rms_norm_batched_kernel(
    device bfloat* out [[buffer(0)]],
    device const bfloat* x [[buffer(1)]],
    device const bfloat* weight [[buffer(2)]],
    constant int& n_rows [[buffer(3)]],
    constant int& dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    if (row >= uint(n_rows)) return;
    threadgroup float sdata[32];

    device const bfloat* x_row = x + row * dim;
    device bfloat* out_row = out + row * dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += threads_per_group) {
        float v = float(x_row[i]);
        sum_sq += v * v;
    }

    sum_sq = simd_sum(sum_sq);
    if (simd_lane == 0) {
        sdata[simd_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float total = (simd_lane < (threads_per_group >> 5)) ? sdata[simd_lane] : 0.0f;
        total = simd_sum(total);
        if (simd_lane == 0) {
            sdata[0] = rsqrt(total / float(dim) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rsqrt_val = sdata[0];
    for (int i = tid; i < dim; i += threads_per_group) {
        float v = float(x_row[i]) * rsqrt_val;
        float w = float(weight[i]);
        out_row[i] = bfloat(v * w);
    }
}

kernel void rms_norm_unweighted_batched_kernel(
    device bfloat* out [[buffer(0)]],
    device const bfloat* x [[buffer(1)]],
    constant int& n_rows [[buffer(2)]],
    constant int& dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    if (row >= uint(n_rows)) return;
    threadgroup float sdata[32];

    device const bfloat* x_row = x + row * dim;
    device bfloat* out_row = out + row * dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += threads_per_group) {
        float v = float(x_row[i]);
        sum_sq += v * v;
    }

    sum_sq = simd_sum(sum_sq);
    if (simd_lane == 0) {
        sdata[simd_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float total = (simd_lane < (threads_per_group >> 5)) ? sdata[simd_lane] : 0.0f;
        total = simd_sum(total);
        if (simd_lane == 0) {
            sdata[0] = rsqrt(total / float(dim) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rsqrt_val = sdata[0];
    for (int i = tid; i < dim; i += threads_per_group) {
        out_row[i] = bfloat(float(x_row[i]) * rsqrt_val);
    }
}

// ════════════════════════════════════════════════════════════════════════════════
//  SiLU * mul (SwiGLU) & Vector Add
// ════════════════════════════════════════════════════════════════════════════════

kernel void silu_mul_kernel(
    device bfloat* out [[buffer(0)]],
    device const bfloat* gate [[buffer(1)]],
    device const bfloat* up [[buffer(2)]],
    constant int& n [[buffer(3)]],
    constant float& swiglu_limit [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= uint(n)) return;
    float g = float(gate[idx]);
    float u = float(up[idx]);
    if (swiglu_limit > 0.0f) {
        g = clamp(g, -swiglu_limit, swiglu_limit);
        u = clamp(u, -swiglu_limit, swiglu_limit);
    }
    out[idx] = bfloat(silu(g) * u);
}

kernel void vector_add_bf16_kernel(
    device bfloat* a [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    constant int& n [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= uint(n)) return;
    a[idx] = bfloat(float(a[idx]) + float(b[idx]));
}

kernel void weighted_add_kernel(
    device bfloat* out [[buffer(0)]],
    device const bfloat* x [[buffer(1)]],
    constant float& weight [[buffer(2)]],
    constant int& dim [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= uint(dim)) return;
    out[idx] = bfloat(float(out[idx]) + weight * float(x[idx]));
}

kernel void add_bias_kernel(
    device bfloat* x [[buffer(0)]],
    device const bfloat* bias [[buffer(1)]],
    constant int& n_rows [[buffer(2)]],
    constant int& dim [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    int total = n_rows * dim;
    if (idx >= uint(total)) return;
    int d = idx % dim;
    x[idx] = bfloat(float(x[idx]) + float(bias[d]));
}

// ════════════════════════════════════════════════════════════════════════════════
//  RoPE (Rotary Position Embeddings)
// ════════════════════════════════════════════════════════════════════════════════

kernel void rope_kernel(
    device bfloat* x [[buffer(0)]],
    constant int& n_vectors [[buffer(1)]],
    constant int& head_dim [[buffer(2)]],
    constant int& rope_dim [[buffer(3)]],
    constant int& position [[buffer(4)]],
    device const float* freq_table [[buffer(5)]],
    constant bool& inverse [[buffer(6)]],
    uint vec_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    if (vec_idx >= uint(n_vectors)) return;
    int half_rope = rope_dim / 2;
    if (tid >= uint(half_rope)) return;

    device bfloat* head = x + vec_idx * head_dim;
    int rope_start = head_dim - rope_dim;

    float cos_th = freq_table[position * rope_dim + tid * 2];
    float sin_th = freq_table[position * rope_dim + tid * 2 + 1];
    if (inverse) sin_th = -sin_th;

    int i0 = rope_start + tid;
    int i1 = rope_start + half_rope + tid;

    float x0 = float(head[i0]);
    float x1 = float(head[i1]);

    head[i0] = bfloat(x0 * cos_th - x1 * sin_th);
    head[i1] = bfloat(x0 * sin_th + x1 * cos_th);
}

kernel void rope_standard_kernel(
    device bfloat* q [[buffer(0)]],
    device bfloat* k [[buffer(1)]],
    constant int& n_q_heads [[buffer(2)]],
    constant int& n_kv_heads [[buffer(3)]],
    constant int& head_dim [[buffer(4)]],
    constant int& pos [[buffer(5)]],
    constant float& rope_theta [[buffer(6)]],
    uint head_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    int half_d = head_dim / 2;
    if (tid >= uint(half_d)) return;

    float freq = 1.0f / pow(rope_theta, float(tid * 2) / float(head_dim));
    float angle = float(pos) * freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    if (head_idx < uint(n_q_heads)) {
        device bfloat* q_head = q + head_idx * head_dim;
        float x0 = float(q_head[tid]);
        float x1 = float(q_head[tid + half_d]);
        q_head[tid] = bfloat(x0 * cos_val - x1 * sin_val);
        q_head[tid + half_d] = bfloat(x0 * sin_val + x1 * cos_val);
    }
    if (head_idx < uint(n_kv_heads)) {
        device bfloat* k_head = k + head_idx * head_dim;
        float x0 = float(k_head[tid]);
        float x1 = float(k_head[tid + half_d]);
        k_head[tid] = bfloat(x0 * cos_val - x1 * sin_val);
        k_head[tid + half_d] = bfloat(x0 * sin_val + x1 * cos_val);
    }
}

// ════════════════════════════════════════════════════════════════════════════════
//  Embedding Lookup
// ════════════════════════════════════════════════════════════════════════════════

kernel void embedding_kernel(
    device bfloat* out [[buffer(0)]],
    device const bfloat* table [[buffer(1)]],
    device const int32_t* ids [[buffer(2)]],
    constant int& seq_len [[buffer(3)]],
    constant int& dim [[buffer(4)]],
    uint seq_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    if (seq_idx >= uint(seq_len)) return;
    int32_t token = ids[seq_idx];
    device const bfloat* src = table + size_t(token) * dim;
    device bfloat* dst = out + size_t(seq_idx) * dim;

    for (int i = tid; i < dim; i += threads_per_group) {
        dst[i] = src[i];
    }
}

kernel void embedding_broadcast_kernel(
    device bfloat* hidden [[buffer(0)]],
    device bfloat* hc_state [[buffer(1)]],
    device const bfloat* table [[buffer(2)]],
    constant int& token_id [[buffer(3)]],
    constant int& dim [[buffer(4)]],
    constant int& hc [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    device const bfloat* src = table + size_t(token_id) * dim;
    for (int i = tid; i < dim; i += threads_per_group) {
        bfloat val = src[i];
        hidden[i] = val;
        for (int h = 0; h < hc; h++) {
            hc_state[h * dim + i] = val;
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════
//  INT4 Symmetric Block-32 GEMV & SwiGLU Fused
// ════════════════════════════════════════════════════════════════════════════════

kernel void gemv_int4_kernel(
    device bfloat* out [[buffer(0)]],
    device const bfloat* vec [[buffer(1)]],
    device const uint8_t* weight [[buffer(2)]],
    device const bfloat* scale [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant bool& is_residual [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    if (row >= uint(N)) return;
    int num_blocks = K / 32;

    device const bfloat* row_scales = scale + row * num_blocks;
    device const uint8_t* row_w = weight + row * (K / 2);

    float sum = 0.0f;
    for (int block = tid; block < num_blocks; block += threads_per_group) {
        float s = float(row_scales[block]);
        int w_offset = block * 16;
        int a_offset = block * 32;

        float block_sum = 0.0f;
        for (int i = 0; i < 16; i++) {
            uint8_t byte_val = row_w[w_offset + i];
            float q0 = float(byte_val & 0x0F) - 8.0f;
            float q1 = float(byte_val >> 4) - 8.0f;
            float a0 = float(vec[a_offset + i * 2]);
            float a1 = float(vec[a_offset + i * 2 + 1]);
            block_sum += q0 * a0 + q1 * a1;
        }
        sum += block_sum * s;
    }

    sum = simd_sum(sum);
    threadgroup float sdata[32];
    uint simd_lane = tid & 31;
    uint simd_id = tid >> 5;
    if (simd_lane == 0) {
        sdata[simd_id] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float total = (simd_lane < (threads_per_group >> 5)) ? sdata[simd_lane] : 0.0f;
        total = simd_sum(total);
        if (simd_lane == 0) {
            if (is_residual) {
                out[row] = bfloat(float(out[row]) + total);
            } else {
                out[row] = bfloat(total);
            }
        }
    }
}

kernel void gemv_int4_swiglu_fused_kernel(
    device bfloat* out [[buffer(0)]],
    device const bfloat* vec [[buffer(1)]],
    device const uint8_t* gate_w [[buffer(2)]],
    device const bfloat* gate_s [[buffer(3)]],
    device const uint8_t* up_w [[buffer(4)]],
    device const bfloat* up_s [[buffer(5)]],
    constant int& N [[buffer(6)]],
    constant int& K [[buffer(7)]],
    constant float& swiglu_limit [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    if (row >= uint(N)) return;
    int num_blocks = K / 32;

    device const bfloat* g_scales = gate_s + row * num_blocks;
    device const uint8_t* g_row_w = gate_w + row * (K / 2);
    device const bfloat* u_scales = up_s + row * num_blocks;
    device const uint8_t* u_row_w = up_w + row * (K / 2);

    float sum_g = 0.0f;
    float sum_u = 0.0f;

    for (int block = tid; block < num_blocks; block += threads_per_group) {
        float s_g = float(g_scales[block]);
        float s_u = float(u_scales[block]);
        int w_offset = block * 16;
        int a_offset = block * 32;

        float b_sum_g = 0.0f;
        float b_sum_u = 0.0f;
        for (int i = 0; i < 16; i++) {
            uint8_t byte_g = g_row_w[w_offset + i];
            uint8_t byte_u = u_row_w[w_offset + i];
            float qg0 = float(byte_g & 0x0F) - 8.0f;
            float qg1 = float(byte_g >> 4) - 8.0f;
            float qu0 = float(byte_u & 0x0F) - 8.0f;
            float qu1 = float(byte_u >> 4) - 8.0f;
            float a0 = float(vec[a_offset + i * 2]);
            float a1 = float(vec[a_offset + i * 2 + 1]);
            b_sum_g += qg0 * a0 + qg1 * a1;
            b_sum_u += qu0 * a0 + qu1 * a1;
        }
        sum_g += b_sum_g * s_g;
        sum_u += b_sum_u * s_u;
    }

    sum_g = simd_sum(sum_g);
    sum_u = simd_sum(sum_u);
    threadgroup float sdata_g[32];
    threadgroup float sdata_u[32];
    uint simd_lane = tid & 31;
    uint simd_id = tid >> 5;
    if (simd_lane == 0) {
        sdata_g[simd_id] = sum_g;
        sdata_u[simd_id] = sum_u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float total_g = (simd_lane < (threads_per_group >> 5)) ? sdata_g[simd_lane] : 0.0f;
        float total_u = (simd_lane < (threads_per_group >> 5)) ? sdata_u[simd_lane] : 0.0f;
        total_g = simd_sum(total_g);
        total_u = simd_sum(total_u);
        if (simd_lane == 0) {
            if (swiglu_limit > 0.0f) {
                total_g = clamp(total_g, -swiglu_limit, swiglu_limit);
                total_u = clamp(total_u, -swiglu_limit, swiglu_limit);
            }
            out[row] = bfloat(silu(total_g) * total_u);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════
//  Qwen 3.8 DeltaNet Linear Attention Recurrence
// ════════════════════════════════════════════════════════════════════════════════

kernel void deltanet_decode_kernel(
    device bfloat* out [[buffer(0)]],               // [num_v_heads * head_dim]
    device const bfloat* in_qkv [[buffer(1)]],      // [Q: 2048, K: 2048, V: 6144]
    device const bfloat* in_z [[buffer(2)]],        // [6144]
    device const bfloat* in_a [[buffer(3)]],        // [48]
    device const bfloat* in_b [[buffer(4)]],        // [48]
    device const bfloat* conv1d_w [[buffer(5)]],    // [10240, 4]
    device const bfloat* in_conv_state [[buffer(6)]],
    device bfloat* out_conv_state [[buffer(7)]],
    device const bfloat* A_log [[buffer(8)]],       // [48]
    device const bfloat* dt_bias [[buffer(9)]],     // [48]
    device const bfloat* norm_w [[buffer(10)]],     // [128]
    device const bfloat* in_ssm_state [[buffer(11)]],// [48, 128, 128]
    device bfloat* out_ssm_state [[buffer(12)]],    // [48, 128, 128]
    constant int& num_k_heads [[buffer(13)]],
    constant int& num_v_heads [[buffer(14)]],
    constant int& head_dim [[buffer(15)]],
    uint h [[threadgroup_position_in_grid]],        // 0..num_v_heads-1 (48 heads)
    uint tid [[thread_position_in_threadgroup]],    // 0..127
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    if (h >= uint(num_v_heads)) return;
    int k_h = h / (num_v_heads / num_k_heads); // 48 / 16 = 3

    threadgroup float s_q[128];
    threadgroup float s_k[128];
    threadgroup float s_v[128];
    threadgroup float s_z[128];
    threadgroup float s_state_col[128];
    threadgroup float sdata[32];

    int q_offset = k_h * head_dim;
    int k_offset = num_k_heads * head_dim + k_h * head_dim;
    int v_offset = 2 * num_k_heads * head_dim + h * head_dim;

    // Load inputs with conv1d update
    if (tid < uint(head_dim)) {
        // Conv1D 1x4 depthwise convolution step for Q, K, V
        float q_val = 0.0f, k_val = 0.0f, v_val = 0.0f;
        int q_ch = q_offset + tid;
        int k_ch = k_offset + tid;
        int v_ch = v_offset + tid;

        // Shift conv state and apply weights
        q_val = float(in_conv_state[q_ch * 4 + 1]) * float(conv1d_w[q_ch * 4 + 0]) +
                float(in_conv_state[q_ch * 4 + 2]) * float(conv1d_w[q_ch * 4 + 1]) +
                float(in_conv_state[q_ch * 4 + 3]) * float(conv1d_w[q_ch * 4 + 2]) +
                float(in_qkv[q_ch]) * float(conv1d_w[q_ch * 4 + 3]);
        out_conv_state[q_ch * 4 + 0] = in_conv_state[q_ch * 4 + 1];
        out_conv_state[q_ch * 4 + 1] = in_conv_state[q_ch * 4 + 2];
        out_conv_state[q_ch * 4 + 2] = in_conv_state[q_ch * 4 + 3];
        out_conv_state[q_ch * 4 + 3] = in_qkv[q_ch];

        k_val = float(in_conv_state[k_ch * 4 + 1]) * float(conv1d_w[k_ch * 4 + 0]) +
                float(in_conv_state[k_ch * 4 + 2]) * float(conv1d_w[k_ch * 4 + 1]) +
                float(in_conv_state[k_ch * 4 + 3]) * float(conv1d_w[k_ch * 4 + 2]) +
                float(in_qkv[k_ch]) * float(conv1d_w[k_ch * 4 + 3]);
        out_conv_state[k_ch * 4 + 0] = in_conv_state[k_ch * 4 + 1];
        out_conv_state[k_ch * 4 + 1] = in_conv_state[k_ch * 4 + 2];
        out_conv_state[k_ch * 4 + 2] = in_conv_state[k_ch * 4 + 3];
        out_conv_state[k_ch * 4 + 3] = in_qkv[k_ch];

        v_val = float(in_conv_state[v_ch * 4 + 1]) * float(conv1d_w[v_ch * 4 + 0]) +
                float(in_conv_state[v_ch * 4 + 2]) * float(conv1d_w[v_ch * 4 + 1]) +
                float(in_conv_state[v_ch * 4 + 3]) * float(conv1d_w[v_ch * 4 + 2]) +
                float(in_qkv[v_ch]) * float(conv1d_w[v_ch * 4 + 3]);
        out_conv_state[v_ch * 4 + 0] = in_conv_state[v_ch * 4 + 1];
        out_conv_state[v_ch * 4 + 1] = in_conv_state[v_ch * 4 + 2];
        out_conv_state[v_ch * 4 + 2] = in_conv_state[v_ch * 4 + 3];
        out_conv_state[v_ch * 4 + 3] = in_qkv[v_ch];

        s_q[tid] = silu(q_val);
        s_k[tid] = silu(k_val);
        s_v[tid] = silu(v_val);
        s_z[tid] = float(in_z[h * head_dim + tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute decay rate
    float a_val = float(in_a[h]);
    float b_val = float(in_b[h]);
    float dt_val = float(dt_bias[h]);
    float a_log_val = float(A_log[h]);
    float beta = 1.0f / (1.0f + exp(-b_val));
    float val_a = a_val + dt_val;
    float softplus_a = (val_a > 20.0f) ? val_a : log(1.0f + exp(val_a));
    float decay = exp(-exp(a_log_val) * softplus_a);

    // L2 Normalize Q and K
    float q_sq = (tid < uint(head_dim)) ? (s_q[tid] * s_q[tid]) : 0.0f;
    float k_sq = (tid < uint(head_dim)) ? (s_k[tid] * s_k[tid]) : 0.0f;
    q_sq = simd_sum(q_sq);
    k_sq = simd_sum(k_sq);

    if (simd_lane == 0) {
        sdata[simd_id * 2] = q_sq;
        sdata[simd_id * 2 + 1] = k_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float t_q = (simd_lane < 4) ? sdata[simd_lane * 2] : 0.0f;
        float t_k = (simd_lane < 4) ? sdata[simd_lane * 2 + 1] : 0.0f;
        t_q = simd_sum(t_q);
        t_k = simd_sum(t_k);
        if (simd_lane == 0) {
            sdata[0] = rsqrt(t_q + 1e-6f);
            sdata[1] = rsqrt(t_k + 1e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(head_dim)) {
        s_q[tid] *= sdata[0];
        s_k[tid] *= sdata[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Recurrence update in state matrix [128, 128]
    device const bfloat* in_state_h = in_ssm_state + size_t(h) * head_dim * head_dim;
    device bfloat* out_state_h = out_ssm_state + size_t(h) * head_dim * head_dim;

    // First compute memory retrieval: v_diff = beta * (v - S^T @ k)
    // S^T @ k: for each j in 0..127: sum_i(S[i, j] * k[i])
    float s_retrieved = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        s_retrieved += float(in_state_h[i * head_dim + tid]) * s_k[i];
    }
    float v_diff = beta * (s_v[tid] - s_retrieved);

    // State update: S_new[i, j] = S_old[i, j] * decay + k[i] * v_diff[j]
    // And output projection: out[j] = sum_i(S_new[i, j] * q[i])
    float out_acc = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        float old_s = float(in_state_h[i * head_dim + tid]);
        float new_s = old_s * decay + s_k[i] * v_diff;
        out_state_h[i * head_dim + tid] = bfloat(new_s);
        out_acc += new_s * s_q[i];
    }

    // Apply norm_w and SiLU(z)
    float normed_out = out_acc * float(norm_w[tid]);
    float gated_out = normed_out * silu(s_z[tid]);
    out[h * head_dim + tid] = bfloat(gated_out);
}

// ════════════════════════════════════════════════════════════════════════════════
//  Softmax, Argmax, Multinomial Sampling
// ════════════════════════════════════════════════════════════════════════════════

kernel void softmax_kernel(
    device float* out [[buffer(0)]],
    device const float* x [[buffer(1)]],
    constant int& rows [[buffer(2)]],
    constant int& cols [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    if (row >= uint(rows)) return;
    device const float* x_row = x + row * cols;
    device float* out_row = out + row * cols;
    threadgroup float sdata[32];

    float max_val = -INFINITY;
    for (int i = tid; i < cols; i += threads_per_group) {
        max_val = max(max_val, x_row[i]);
    }
    max_val = simd_max(max_val);
    if (simd_lane == 0) sdata[simd_id] = max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float m = (simd_lane < (threads_per_group >> 5)) ? sdata[simd_lane] : -INFINITY;
        m = simd_max(m);
        if (simd_lane == 0) sdata[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_val = sdata[0];

    float sum_exp = 0.0f;
    for (int i = tid; i < cols; i += threads_per_group) {
        float e = exp(x_row[i] - max_val);
        out_row[i] = e;
        sum_exp += e;
    }
    sum_exp = simd_sum(sum_exp);
    if (simd_lane == 0) sdata[simd_id] = sum_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float s = (simd_lane < (threads_per_group >> 5)) ? sdata[simd_lane] : 0.0f;
        s = simd_sum(s);
        if (simd_lane == 0) sdata[0] = 1.0f / (s + 1e-9f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = sdata[0];

    for (int i = tid; i < cols; i += threads_per_group) {
        out_row[i] *= inv_sum;
    }
}

kernel void argmax_f32_kernel(
    device int32_t* out [[buffer(0)]],
    device const float* logits [[buffer(1)]],
    constant int& n [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float s_max[32];
    threadgroup int32_t s_idx[32];

    float local_max = -INFINITY;
    int32_t local_idx = 0;

    for (int i = tid; i < n; i += threads_per_group) {
        float v = logits[i];
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_max = simd_shuffle_down(local_max, offset);
        int32_t other_idx = simd_shuffle_down(local_idx, offset);
        if (other_max > local_max) {
            local_max = other_max;
            local_idx = other_idx;
        }
    }

    if (simd_lane == 0) {
        s_max[simd_id] = local_max;
        s_idx[simd_id] = local_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float m = (simd_lane < (threads_per_group >> 5)) ? s_max[simd_lane] : -INFINITY;
        int32_t idx = (simd_lane < (threads_per_group >> 5)) ? s_idx[simd_lane] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float om = simd_shuffle_down(m, offset);
            int32_t oi = simd_shuffle_down(idx, offset);
            if (om > m) {
                m = om;
                idx = oi;
            }
        }
        if (simd_lane == 0) {
            out[0] = idx;
        }
    }
}
