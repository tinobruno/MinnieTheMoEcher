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
        g = min(g, swiglu_limit);
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

kernel void embedding_int4_kernel(
    device bfloat* out [[buffer(0)]],
    device const uint8_t* table_w [[buffer(1)]],
    device const bfloat* table_s [[buffer(2)]],
    device const int32_t* ids [[buffer(3)]],
    constant int& seq_len [[buffer(4)]],
    constant int& dim [[buffer(5)]],
    uint seq_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    if (seq_idx >= uint(seq_len)) return;
    int32_t token = ids[seq_idx];
    int num_blocks = dim / 32;

    device const uint8_t* row_w = table_w + size_t(token) * (size_t(dim) / 2);
    device const bfloat* row_s = table_s + size_t(token) * num_blocks;
    device bfloat* row_out = out + size_t(seq_idx) * dim;

    for (int b = tid; b < num_blocks; b += threads_per_group) {
        float s = float(row_s[b]);
        int w_off = b * 16;
        int a_off = b * 32;
        for (int i = 0; i < 16; i++) {
            uint8_t byte_val = row_w[w_off + i];
            float q0 = (float(byte_val & 0x0F) - 8.0f) * s;
            float q1 = (float(byte_val >> 4) - 8.0f) * s;
            row_out[a_off + i * 2] = bfloat(q0);
            row_out[a_off + i * 2 + 1] = bfloat(q1);
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
                total_g = min(total_g, swiglu_limit);
                total_u = clamp(total_u, -swiglu_limit, swiglu_limit);
            }
            out[row] = bfloat(silu(total_g) * total_u);
        }
    }
}

kernel void gemv_int4_f32_kernel(
    device float* out [[buffer(0)]],
    device const bfloat* vec [[buffer(1)]],
    device const uint8_t* weight [[buffer(2)]],
    device const bfloat* scale [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    if (row >= uint(N)) return;
    int num_blocks = K / 32;

    device const bfloat* row_scales = scale + row * num_blocks;
    device const uint8_t* row_w = weight + size_t(row) * (size_t(K) / 2);

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
            out[row] = total;
        }
    }
}

// ============================================================================
//  INT3 Symmetric Block-32 GEMV & SwiGLU Fused
// ============================================================================

kernel void gemv_int3_kernel(
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
    int blocks_per_row = K / 32;

    device const bfloat* row_s = scale + row * blocks_per_row;
    device const uint8_t* row_w = weight + size_t(row) * (size_t(K) * 3 / 8);

    float sum = 0.0f;
    for (int b = tid; b < blocks_per_row; b += threads_per_group) {
        float s = float(row_s[b]);
        device const uint8_t* blk_w = row_w + b * 12;
        int in_idx = b * 32;

        for (int i = 0; i < 4; i++) {
            uint8_t b0 = blk_w[i * 3 + 0];
            uint8_t b1 = blk_w[i * 3 + 1];
            uint8_t b2 = blk_w[i * 3 + 2];

            float w0 = (float(b0 & 0x07) - 4.0f) * s;
            float w1 = (float((b0 >> 3) & 0x07) - 4.0f) * s;
            float w2 = (float((b0 >> 6) | ((b1 & 0x01) << 2)) - 4.0f) * s;
            float w3 = (float((b1 >> 1) & 0x07) - 4.0f) * s;
            float w4 = (float((b1 >> 4) & 0x07) - 4.0f) * s;
            float w5 = (float((b1 >> 7) | ((b2 & 0x03) << 1)) - 4.0f) * s;
            float w6 = (float((b2 >> 2) & 0x07) - 4.0f) * s;
            float w7 = (float((b2 >> 5) & 0x07) - 4.0f) * s;

            sum += w0 * float(vec[in_idx + i * 8 + 0]) +
                   w1 * float(vec[in_idx + i * 8 + 1]) +
                   w2 * float(vec[in_idx + i * 8 + 2]) +
                   w3 * float(vec[in_idx + i * 8 + 3]) +
                   w4 * float(vec[in_idx + i * 8 + 4]) +
                   w5 * float(vec[in_idx + i * 8 + 5]) +
                   w6 * float(vec[in_idx + i * 8 + 6]) +
                   w7 * float(vec[in_idx + i * 8 + 7]);
        }
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

kernel void gemv_int3_swiglu_fused_kernel(
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
    int blocks_per_row = K / 32;

    device const bfloat* g_scales = gate_s + row * blocks_per_row;
    device const uint8_t* g_row_w = gate_w + size_t(row) * (size_t(K) * 3 / 8);
    device const bfloat* u_scales = up_s + row * blocks_per_row;
    device const uint8_t* u_row_w = up_w + size_t(row) * (size_t(K) * 3 / 8);

    float sum_g = 0.0f;
    float sum_u = 0.0f;

    for (int b = tid; b < blocks_per_row; b += threads_per_group) {
        float sg = float(g_scales[b]);
        float su = float(u_scales[b]);
        device const uint8_t* gb = g_row_w + b * 12;
        device const uint8_t* ub = u_row_w + b * 12;
        int in_idx = b * 32;

        for (int i = 0; i < 4; i++) {
            uint8_t gb0 = gb[i * 3 + 0], gb1 = gb[i * 3 + 1], gb2 = gb[i * 3 + 2];
            uint8_t ub0 = ub[i * 3 + 0], ub1 = ub[i * 3 + 1], ub2 = ub[i * 3 + 2];

            float gw0 = (float(gb0 & 0x07) - 4.0f) * sg;
            float gw1 = (float((gb0 >> 3) & 0x07) - 4.0f) * sg;
            float gw2 = (float((gb0 >> 6) | ((gb1 & 0x01) << 2)) - 4.0f) * sg;
            float gw3 = (float((gb1 >> 1) & 0x07) - 4.0f) * sg;
            float gw4 = (float((gb1 >> 4) & 0x07) - 4.0f) * sg;
            float gw5 = (float((gb1 >> 7) | ((gb2 & 0x03) << 1)) - 4.0f) * sg;
            float gw6 = (float((gb2 >> 2) & 0x07) - 4.0f) * sg;
            float gw7 = (float((gb2 >> 5) & 0x07) - 4.0f) * sg;

            float uw0 = (float(ub0 & 0x07) - 4.0f) * su;
            float uw1 = (float((ub0 >> 3) & 0x07) - 4.0f) * su;
            float uw2 = (float((ub0 >> 6) | ((ub1 & 0x01) << 2)) - 4.0f) * su;
            float uw3 = (float((ub1 >> 1) & 0x07) - 4.0f) * su;
            float uw4 = (float((ub1 >> 4) & 0x07) - 4.0f) * su;
            float uw5 = (float((ub1 >> 7) | ((ub2 & 0x03) << 1)) - 4.0f) * su;
            float uw6 = (float((ub2 >> 2) & 0x07) - 4.0f) * su;
            float uw7 = (float((ub2 >> 5) & 0x07) - 4.0f) * su;

            float v0 = float(vec[in_idx + i * 8 + 0]);
            float v1 = float(vec[in_idx + i * 8 + 1]);
            float v2 = float(vec[in_idx + i * 8 + 2]);
            float v3 = float(vec[in_idx + i * 8 + 3]);
            float v4 = float(vec[in_idx + i * 8 + 4]);
            float v5 = float(vec[in_idx + i * 8 + 5]);
            float v6 = float(vec[in_idx + i * 8 + 6]);
            float v7 = float(vec[in_idx + i * 8 + 7]);

            sum_g += gw0 * v0 + gw1 * v1 + gw2 * v2 + gw3 * v3 + gw4 * v4 + gw5 * v5 + gw6 * v6 + gw7 * v7;
            sum_u += uw0 * v0 + uw1 * v1 + uw2 * v2 + uw3 * v3 + uw4 * v4 + uw5 * v5 + uw6 * v6 + uw7 * v7;
        }
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
                total_g = min(total_g, swiglu_limit);
                total_u = clamp(total_u, -swiglu_limit, swiglu_limit);
            }
            out[row] = bfloat(silu(total_g) * total_u);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════
//  Qwen 3.8 DeltaNet Linear Attention Recurrence
// ════════════════════════════════════════════════════════════════════════════════

kernel void deltanet_conv_kernel(
    device bfloat* conv_out [[buffer(0)]],
    device const bfloat* in_qkv [[buffer(1)]],
    device const bfloat* conv1d_w [[buffer(2)]],
    device const bfloat* in_conv_state [[buffer(3)]],
    device bfloat* out_conv_state [[buffer(4)]],
    constant int& channels [[buffer(5)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= uint(channels)) return;

    device const bfloat* in_cs = in_conv_state + c * 4;
    device bfloat* out_cs = out_conv_state + c * 4;
    device const bfloat* cw = conv1d_w + c * 4;

    float s0 = float(in_cs[1]);
    float s1 = float(in_cs[2]);
    float s2 = float(in_cs[3]);
    float s3 = float(in_qkv[c]);

    out_cs[0] = bfloat(s0);
    out_cs[1] = bfloat(s1);
    out_cs[2] = bfloat(s2);
    out_cs[3] = bfloat(s3);

    float w0 = float(cw[0]);
    float w1 = float(cw[1]);
    float w2 = float(cw[2]);
    float w3 = float(cw[3]);

    float val = s0 * w0 + s1 * w1 + s2 * w2 + s3 * w3;
    float silu_val = val / (1.0f + exp(-val));
    conv_out[c] = bfloat(silu_val);
}

kernel void deltanet_ssm_step_kernel(
    device bfloat* out [[buffer(0)]],               // [num_v_heads * head_dim]
    device const bfloat* conv_out [[buffer(1)]],    // [channels = (2*num_k + num_v)*head_dim]
    device const bfloat* in_z [[buffer(2)]],        // [num_v_heads * head_dim]
    device const bfloat* in_a [[buffer(3)]],        // [num_v_heads]
    device const bfloat* in_b [[buffer(4)]],        // [num_v_heads]
    device const bfloat* A_log [[buffer(5)]],       // [num_v_heads]
    device const bfloat* dt_bias [[buffer(6)]],     // [num_v_heads]
    device const bfloat* norm_w [[buffer(7)]],      // [head_dim]
    device const bfloat* in_ssm_state [[buffer(8)]],// [num_v_heads, head_dim, head_dim]
    device bfloat* out_ssm_state [[buffer(9)]],     // [num_v_heads, head_dim, head_dim]
    constant int& num_k_heads [[buffer(10)]],
    constant int& num_v_heads [[buffer(11)]],
    constant int& head_dim [[buffer(12)]],
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
    threadgroup float s_kv_mem[128];
    threadgroup float s_out[128];
    threadgroup float s_sum_q[4];
    threadgroup float s_sum_k[4];
    threadgroup float s_sum_out[4];

    int q_offset = k_h * head_dim;
    int k_offset = (num_k_heads * head_dim) + k_h * head_dim;
    int v_offset = (2 * num_k_heads * head_dim) + h * head_dim;

    device const bfloat* q_vec = conv_out + q_offset;
    device const bfloat* k_vec = conv_out + k_offset;
    device const bfloat* v_vec = conv_out + v_offset;
    device const bfloat* z_vec = in_z + h * head_dim;

    device const bfloat* in_state_h = in_ssm_state + size_t(h) * head_dim * head_dim;
    device bfloat* out_state_h = out_ssm_state + size_t(h) * head_dim * head_dim;

    float a_val = float(in_a[h]);
    float b_val = float(in_b[h]);
    float dt_val = float(dt_bias[h]);
    float a_log_val = float(A_log[h]);

    float beta = 1.0f / (1.0f + exp(-b_val));
    float val_a = a_val + dt_val;
    float softplus_a = (val_a > 20.0f) ? val_a : log(1.0f + exp(val_a));
    float g = -exp(a_log_val) * softplus_a;
    float decay = exp(g);

    if (tid < uint(head_dim)) {
        s_q[tid] = float(q_vec[tid]);
        s_k[tid] = float(k_vec[tid]);
        s_v[tid] = float(v_vec[tid]);
        s_z[tid] = float(z_vec[tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 1. L2 Normalize Q and K
    float q_sq = (tid < uint(head_dim)) ? (s_q[tid] * s_q[tid]) : 0.0f;
    float k_sq = (tid < uint(head_dim)) ? (s_k[tid] * s_k[tid]) : 0.0f;

    q_sq = simd_sum(q_sq);
    k_sq = simd_sum(k_sq);

    if (simd_lane == 0) {
        s_sum_q[simd_id] = q_sq;
        s_sum_k[simd_id] = k_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float t_q = (simd_lane < 4) ? s_sum_q[simd_lane] : 0.0f;
        float t_k = (simd_lane < 4) ? s_sum_k[simd_lane] : 0.0f;
        t_q = simd_sum(t_q);
        t_k = simd_sum(t_k);
        if (simd_lane == 0) {
            s_sum_q[0] = rsqrt(t_q + 1e-6f) * (1.0f / sqrt(float(head_dim)));
            s_sum_k[0] = rsqrt(t_k + 1e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float r_q = s_sum_q[0];
    float r_k = s_sum_k[0];
    if (tid < uint(head_dim)) {
        s_q[tid] *= r_q;
        s_k[tid] *= r_k;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Compute kv_mem[col] = sum_row (decay * S[row, col]) * k[row]
    float col_s[128];
    if (tid < uint(head_dim)) {
        float mem = 0.0f;
        for (int r = 0; r < head_dim; r++) {
            float s_val = float(in_state_h[r * head_dim + tid]);
            col_s[r] = s_val;
            mem += (decay * s_val) * s_k[r];
        }
        s_kv_mem[tid] = mem;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. State delta update and single-pass VRAM write: S[r, c] = decay * S[r, c] + k[r] * delta[c]
    // and compute out[c] = sum_r S[r, c] * q[r]
    if (tid < uint(head_dim)) {
        float delta_c = (s_v[tid] - s_kv_mem[tid]) * beta;
        float out_c = 0.0f;
        for (int r = 0; r < head_dim; r++) {
            float new_s = decay * col_s[r] + s_k[r] * delta_c;
            out_state_h[r * head_dim + tid] = bfloat(new_s);
            out_c += new_s * s_q[r];
        }
        s_out[tid] = out_c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4. Output RMSNorm + Z-gating
    float out_sq = (tid < uint(head_dim)) ? (s_out[tid] * s_out[tid]) : 0.0f;
    out_sq = simd_sum(out_sq);
    if (simd_lane == 0) {
        s_sum_out[simd_id] = out_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float t_out = (simd_lane < 4) ? s_sum_out[simd_lane] : 0.0f;
        t_out = simd_sum(t_out);
        if (simd_lane == 0) {
            s_sum_out[0] = rsqrt(t_out / float(head_dim) + 1e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float r_out = s_sum_out[0];
    if (tid < uint(head_dim)) {
        float normed = s_out[tid] * r_out * float(norm_w[tid]);
        float z = s_z[tid];
        float silu_z = z / (1.0f + exp(-z));
        out[h * head_dim + tid] = bfloat(normed * silu_z);
    }
}

kernel void gemv_bf16_out_bf16_kernel(
    device bfloat* out [[buffer(0)]],
    device const bfloat* W [[buffer(1)]],
    device const bfloat* x [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    if (row >= uint(N)) return;
    device const bfloat* row_w = W + size_t(row) * size_t(K);

    float sum = 0.0f;
    for (int c = tid; c < K; c += threads_per_group) {
        sum += float(row_w[c]) * float(x[c]);
    }

    sum = simd_sum(sum);
    threadgroup float sdata[32];
    if (simd_lane == 0) {
        sdata[simd_id] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float total = (simd_lane < (threads_per_group >> 5)) ? sdata[simd_lane] : 0.0f;
        total = simd_sum(total);
        if (simd_lane == 0) {
            out[row] = bfloat(total);
        }
    }
}

kernel void gemv_bf16_f32_kernel(
    device float* out [[buffer(0)]],
    device const bfloat* W [[buffer(1)]],
    device const bfloat* x [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    if (row >= uint(N)) return;
    device const bfloat* row_w = W + size_t(row) * size_t(K);

    float sum = 0.0f;
    for (int c = tid; c < K; c += threads_per_group) {
        sum += float(row_w[c]) * float(x[c]);
    }

    sum = simd_sum(sum);
    threadgroup float sdata[32];
    if (simd_lane == 0) {
        sdata[simd_id] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float total = (simd_lane < (threads_per_group >> 5)) ? sdata[simd_lane] : 0.0f;
        total = simd_sum(total);
        if (simd_lane == 0) {
            out[row] = total;
        }
    }
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

constant float fp8_e4m3_lut[256] = {
    0.00000000f, 0.00195312f, 0.00390625f, 0.00585938f, 0.00781250f, 0.00976562f, 0.01171875f, 0.01367188f,
    0.01562500f, 0.01757812f, 0.01953125f, 0.02148438f, 0.02343750f, 0.02539062f, 0.02734375f, 0.02929688f,
    0.03125000f, 0.03515625f, 0.03906250f, 0.04296875f, 0.04687500f, 0.05078125f, 0.05468750f, 0.05859375f,
    0.06250000f, 0.07031250f, 0.07812500f, 0.08593750f, 0.09375000f, 0.10156250f, 0.10937500f, 0.11718750f,
    0.12500000f, 0.14062500f, 0.15625000f, 0.17187500f, 0.18750000f, 0.20312500f, 0.21875000f, 0.23437500f,
    0.25000000f, 0.28125000f, 0.31250000f, 0.34375000f, 0.37500000f, 0.40625000f, 0.43750000f, 0.46875000f,
    0.50000000f, 0.56250000f, 0.62500000f, 0.68750000f, 0.75000000f, 0.81250000f, 0.87500000f, 0.93750000f,
    1.00000000f, 1.12500000f, 1.25000000f, 1.37500000f, 1.50000000f, 1.62500000f, 1.75000000f, 1.87500000f,
    2.00000000f, 2.25000000f, 2.50000000f, 2.75000000f, 3.00000000f, 3.25000000f, 3.50000000f, 3.75000000f,
    4.00000000f, 4.50000000f, 5.00000000f, 5.50000000f, 6.00000000f, 6.50000000f, 7.00000000f, 7.50000000f,
    8.00000000f, 9.00000000f, 10.00000000f, 11.00000000f, 12.00000000f, 13.00000000f, 14.00000000f, 15.00000000f,
    16.00000000f, 18.00000000f, 20.00000000f, 22.00000000f, 24.00000000f, 26.00000000f, 28.00000000f, 30.00000000f,
    32.00000000f, 36.00000000f, 40.00000000f, 44.00000000f, 48.00000000f, 52.00000000f, 56.00000000f, 60.00000000f,
    64.00000000f, 72.00000000f, 80.00000000f, 88.00000000f, 96.00000000f, 104.00000000f, 112.00000000f, 120.00000000f,
    128.00000000f, 144.00000000f, 160.00000000f, 176.00000000f, 192.00000000f, 208.00000000f, 224.00000000f, 240.00000000f,
    256.00000000f, 288.00000000f, 320.00000000f, 352.00000000f, 384.00000000f, 416.00000000f, 448.00000000f, 0.00000000f,
    -0.00000000f, -0.00195312f, -0.00390625f, -0.00585938f, -0.00781250f, -0.00976562f, -0.01171875f, -0.01367188f,
    -0.01562500f, -0.01757812f, -0.01953125f, -0.02148438f, -0.02343750f, -0.02539062f, -0.02734375f, -0.02929688f,
    -0.03125000f, -0.03515625f, -0.03906250f, -0.04296875f, -0.04687500f, -0.05078125f, -0.05468750f, -0.05859375f,
    -0.06250000f, -0.07031250f, -0.07812500f, -0.08593750f, -0.09375000f, -0.10156250f, -0.10937500f, -0.11718750f,
    -0.12500000f, -0.14062500f, -0.15625000f, -0.17187500f, -0.18750000f, -0.20312500f, -0.21875000f, -0.23437500f,
    -0.25000000f, -0.28125000f, -0.31250000f, -0.34375000f, -0.37500000f, -0.40625000f, -0.43750000f, -0.46875000f,
    -0.50000000f, -0.56250000f, -0.62500000f, -0.68750000f, -0.75000000f, -0.81250000f, -0.87500000f, -0.93750000f,
    -1.00000000f, -1.12500000f, -1.25000000f, -1.37500000f, -1.50000000f, -1.62500000f, -1.75000000f, -1.87500000f,
    -2.00000000f, -2.25000000f, -2.50000000f, -2.75000000f, -3.00000000f, -3.25000000f, -3.50000000f, -3.75000000f,
    -4.00000000f, -4.50000000f, -5.00000000f, -5.50000000f, -6.00000000f, -6.50000000f, -7.00000000f, -7.50000000f,
    -8.00000000f, -9.00000000f, -10.00000000f, -11.00000000f, -12.00000000f, -13.00000000f, -14.00000000f, -15.00000000f,
    -16.00000000f, -18.00000000f, -20.00000000f, -22.00000000f, -24.00000000f, -26.00000000f, -28.00000000f, -30.00000000f,
    -32.00000000f, -36.00000000f, -40.00000000f, -44.00000000f, -48.00000000f, -52.00000000f, -56.00000000f, -60.00000000f,
    -64.00000000f, -72.00000000f, -80.00000000f, -88.00000000f, -96.00000000f, -104.00000000f, -112.00000000f, -120.00000000f,
    -128.00000000f, -144.00000000f, -160.00000000f, -176.00000000f, -192.00000000f, -208.00000000f, -224.00000000f, -240.00000000f,
    -256.00000000f, -288.00000000f, -320.00000000f, -352.00000000f, -384.00000000f, -416.00000000f, -448.00000000f, 0.00000000f,
};


inline uint8_t float_to_fp8_e4m3_dev(float val) {
    uint32_t u = as_type<uint32_t>(val);
    uint32_t sign = (u >> 24) & 0x80;
    u &= 0x7FFFFFFF;
    if (u == 0) return (uint8_t)sign;

    u += (1U << 19);
    int exp = ((u >> 23) & 0xFF) - 127 + 7;
    uint32_t mant = (u >> 20) & 0x7;

    if (exp >= 15) {
        return (uint8_t)(sign | 0x7E);
    } else if (exp <= 0) {
        if (exp < -3) return (uint8_t)sign;
        mant = ((u & 0x007FFFFF) | 0x00800000) >> (21 - exp);
        return (uint8_t)(sign | mant);
    }
    return (uint8_t)(sign | (exp << 3) | mant);
}

kernel void qwen_gqa_write_kv_fp8_batch_kernel(
    device bfloat* k [[buffer(0)]],
    device const bfloat* v [[buffer(1)]],
    device const bfloat* k_norm_w [[buffer(2)]],
    device uint8_t* k_cache [[buffer(3)]],
    device uint8_t* v_cache [[buffer(4)]],
    constant int& n_kv_heads [[buffer(5)]],
    constant int& head_dim [[buffer(6)]],
    device const int32_t* d_pos [[buffer(7)]],
    constant int& pos_scalar [[buffer(8)]],
    constant int& M [[buffer(9)]],
    constant int& max_seq_len [[buffer(10)]],
    constant float& rope_theta [[buffer(11)]],
    constant float& eps [[buffer(12)]],
    constant int& has_kn [[buffer(13)]],
    constant int& has_d_pos [[buffer(14)]],
    uint2 group_pos [[threadgroup_position_in_grid]],
    uint2 thread_pos [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    uint tid = thread_pos.x;
    int kv_head = group_pos.x;
    int m = group_pos.y;
    if (kv_head >= n_kv_heads || m >= M) return;

    int pos = (has_d_pos && d_pos) ? d_pos[m] : (pos_scalar + m);
    if (pos >= max_seq_len) return;

    size_t in_offset = (size_t)m * (n_kv_heads * head_dim) + (size_t)kv_head * head_dim;
    device bfloat* k_vec = k + in_offset;
    device const bfloat* v_vec = v + in_offset;

    threadgroup float s_k_sum;
    threadgroup float s_warp_reduce[4];
    if (tid == 0) s_k_sum = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float k_sq = 0.0f;
    for (int i = tid; i < head_dim; i += 128) {
        float val = float(k_vec[i]);
        k_sq += val * val;
    }
    k_sq = simd_sum(k_sq);
    if (simd_lane == 0) s_warp_reduce[simd_id] = k_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float total = (simd_lane < 4) ? s_warp_reduce[simd_lane] : 0.0f;
        total = simd_sum(total);
        if (simd_lane == 0) s_k_sum = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float k_rrms = rsqrt(s_k_sum / float(head_dim) + eps);
    for (int i = tid; i < head_dim; i += 128) {
        float k_normed = float(k_vec[i]) * k_rrms * (1.0f + (has_kn && k_norm_w ? float(k_norm_w[i]) : 0.0f));
        k_vec[i] = bfloat(k_normed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        float freq = 1.0f / pow(rope_theta, float(2 * tid) / 64.0f);
        float angle = float(pos) * freq;
        float cos_a = cos(angle);
        float sin_a = sin(angle);

        float k0 = float(k_vec[tid]);
        float k1 = float(k_vec[tid + 32]);

        k_vec[tid] = bfloat(k0 * cos_a - k1 * sin_a);
        k_vec[tid + 32] = bfloat(k0 * sin_a + k1 * cos_a);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    size_t cache_offset = ((size_t)pos * n_kv_heads + kv_head) * head_dim;
    for (int i = tid; i < head_dim; i += 128) {
        k_cache[cache_offset + i] = float_to_fp8_e4m3_dev(float(k_vec[i]));
        v_cache[cache_offset + i] = float_to_fp8_e4m3_dev(float(v_vec[i]));
    }
}

kernel void qwen_gqa_compute_attn_fp8_batch_kernel(
    device bfloat* out [[buffer(0)]],
    device const bfloat* q_and_gate [[buffer(1)]],
    device const bfloat* q_norm_w [[buffer(2)]],
    device const uint8_t* k_cache [[buffer(3)]],
    device const uint8_t* v_cache [[buffer(4)]],
    constant int& n_q_heads [[buffer(5)]],
    constant int& n_kv_heads [[buffer(6)]],
    constant int& head_dim [[buffer(7)]],
    device const int32_t* d_pos [[buffer(8)]],
    constant int& pos_scalar [[buffer(9)]],
    constant int& M [[buffer(10)]],
    constant int& max_seq_len [[buffer(11)]],
    constant float& rope_theta [[buffer(12)]],
    constant float& eps [[buffer(13)]],
    constant int& has_qn [[buffer(14)]],
    constant int& has_d_pos [[buffer(15)]],
    uint2 group_pos [[threadgroup_position_in_grid]],
    uint2 thread_pos [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    uint tid = thread_pos.x;
    int q_head = group_pos.x;
    int m = group_pos.y;
    if (q_head >= n_q_heads || m >= M) return;

    int pos = (has_d_pos && d_pos) ? d_pos[m] : (pos_scalar + m);
    if (pos >= max_seq_len) pos = max_seq_len - 1;

    int kv_head = q_head / (n_q_heads / n_kv_heads);

    threadgroup float s_q[256];
    threadgroup float s_q_sum;
    threadgroup float s_tile_scores[128];
    threadgroup float s_warp_reduce[4];
    threadgroup float s_new_max;
    threadgroup float s_alpha;
    threadgroup float s_reduce_acc[64][4];

    size_t q_offset = (size_t)m * (2 * n_q_heads * head_dim) + (size_t)q_head * (2 * head_dim);
    device const bfloat* q_in = q_and_gate + q_offset;
    device const bfloat* gate_in = q_in + head_dim;

    size_t out_offset = (size_t)m * (n_q_heads * head_dim) + (size_t)q_head * head_dim;
    device bfloat* out_vec = out + out_offset;

    if (tid == 0) s_q_sum = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float q_sq = 0.0f;
    for (int i = tid; i < head_dim; i += 128) {
        float val = float(q_in[i]);
        s_q[i] = val;
        q_sq += val * val;
    }
    q_sq = simd_sum(q_sq);
    if (simd_lane == 0) s_warp_reduce[simd_id] = q_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float total = (simd_lane < 4) ? s_warp_reduce[simd_lane] : 0.0f;
        total = simd_sum(total);
        if (simd_lane == 0) s_q_sum = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float q_rrms = rsqrt(s_q_sum / float(head_dim) + eps);
    for (int i = tid; i < head_dim; i += 128) {
        s_q[i] = s_q[i] * q_rrms * (1.0f + (has_qn && q_norm_w ? float(q_norm_w[i]) : 0.0f));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        float freq = 1.0f / pow(rope_theta, float(2 * tid) / 64.0f);
        float angle = float(pos) * freq;
        float cos_a = cos(angle);
        float sin_a = sin(angle);

        float q0 = s_q[tid];
        float q1 = s_q[tid + 32];

        s_q[tid] = q0 * cos_a - q1 * sin_a;
        s_q[tid + 32] = q0 * sin_a + q1 * cos_a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = 1.0f / sqrt(float(head_dim));
    float running_max = -1e38f;
    float running_sum = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    int w = tid % 64;
    int half_idx = tid / 64;
    int words_per_tok = head_dim >> 2;

    for (int t_block = 0; t_block <= pos; t_block += 128) {
        int chunk_len = min(128, pos + 1 - t_block);

        float dot = -1e38f;
        if (tid < uint(chunk_len)) {
            int t = t_block + tid;
            size_t k_offset = ((size_t)t * n_kv_heads + kv_head) * head_dim;
            device const uint8_t* k_ptr = k_cache + k_offset;
            float d = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                d += s_q[i] * fp8_e4m3_lut[k_ptr[i]];
            }
            dot = d * scale;
        }
        s_tile_scores[tid] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float thread_val = (tid < uint(chunk_len)) ? s_tile_scores[tid] : -1e38f;
        float chunk_max = simd_max(thread_val);
        if (simd_lane == 0) s_warp_reduce[simd_id] = chunk_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float m_val = max(max(s_warp_reduce[0], s_warp_reduce[1]), max(s_warp_reduce[2], s_warp_reduce[3]));
            float new_m = max(running_max, m_val);
            float a = (running_max <= -1e37f) ? 0.0f : exp(running_max - new_m);
            s_new_max = new_m;
            s_alpha = a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float new_max = s_new_max;
        float alpha = s_alpha;
        running_max = new_max;
        running_sum = running_sum * alpha;
        if (half_idx == 0) {
            acc0 *= alpha;
            acc1 *= alpha;
            acc2 *= alpha;
            acc3 *= alpha;
        }

        float exp_s = (tid < uint(chunk_len)) ? exp(s_tile_scores[tid] - new_max) : 0.0f;
        s_tile_scores[tid] = exp_s;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float sum_chunk = simd_sum(exp_s);
        if (simd_lane == 0) s_warp_reduce[simd_id] = sum_chunk;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            s_warp_reduce[0] += s_warp_reduce[1] + s_warp_reduce[2] + s_warp_reduce[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        running_sum += s_warp_reduce[0];

        int j_start = (half_idx == 0) ? 0 : 64;
        int j_end   = (half_idx == 0) ? min(64, chunk_len) : chunk_len;

        float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
        if (w < words_per_tok && j_start < j_end) {
            for (int j = j_start; j < j_end; j++) {
                float weight = s_tile_scores[j];
                int t = t_block + j;
                size_t v_off = ((size_t)t * n_kv_heads + kv_head) * head_dim + (w << 2);
                device const uint8_t* v_ptr = v_cache + v_off;
                c0 += weight * fp8_e4m3_lut[v_ptr[0]];
                c1 += weight * fp8_e4m3_lut[v_ptr[1]];
                c2 += weight * fp8_e4m3_lut[v_ptr[2]];
                c3 += weight * fp8_e4m3_lut[v_ptr[3]];
            }
        }

        if (half_idx == 1 && w < words_per_tok) {
            s_reduce_acc[w][0] = c0;
            s_reduce_acc[w][1] = c1;
            s_reduce_acc[w][2] = c2;
            s_reduce_acc[w][3] = c3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (half_idx == 0 && w < words_per_tok) {
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
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = 1.0f / (running_sum + 1e-8f);
    if (half_idx == 0 && w < words_per_tok) {
        int base_d = w * 4;
        if (base_d < head_dim) {
            float g0 = float(gate_in[base_d]);
            float sig0 = 1.0f / (1.0f + exp(-g0));
            out_vec[base_d] = bfloat(acc0 * inv_sum * sig0);
        }
        if (base_d + 1 < head_dim) {
            float g1 = float(gate_in[base_d + 1]);
            float sig1 = 1.0f / (1.0f + exp(-g1));
            out_vec[base_d + 1] = bfloat(acc1 * inv_sum * sig1);
        }
        if (base_d + 2 < head_dim) {
            float g2 = float(gate_in[base_d + 2]);
            float sig2 = 1.0f / (1.0f + exp(-g2));
            out_vec[base_d + 2] = bfloat(acc2 * inv_sum * sig2);
        }
        if (base_d + 3 < head_dim) {
            float g3 = float(gate_in[base_d + 3]);
            float sig3 = 1.0f / (1.0f + exp(-g3));
            out_vec[base_d + 3] = bfloat(acc3 * inv_sum * sig3);
        }
    }
}

// ============================================================================
//  BATCНED INT4, INT3, SWIGLU & BF16 GEMM FOR PREFILL ACCELERATION
// ============================================================================

kernel void gemm_int4_batch_kernel(
    device bfloat* out [[buffer(0)]],
    device const bfloat* A [[buffer(1)]],
    device const uint8_t* weight [[buffer(2)]],
    device const bfloat* scale [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant int& M [[buffer(6)]],
    constant bool& is_residual [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    if (row >= uint(N)) return;
    int num_blocks = K / 32;
    int eff_M = min(M, 8);

    device const bfloat* row_scales = scale + row * num_blocks;
    device const uint8_t* row_w = weight + row * (K / 2);

    float sum[8] = {0.0f};

    for (int block = tid; block < num_blocks; block += threads_per_group) {
        float s = float(row_scales[block]);
        int w_offset = block * 16;
        int a_offset = block * 32;

        for (int i = 0; i < 16; i++) {
            uint8_t byte_val = row_w[w_offset + i];
            float q0 = (float(byte_val & 0x0F) - 8.0f) * s;
            float q1 = (float(byte_val >> 4) - 8.0f) * s;
            int a_idx = a_offset + i * 2;

            for (int m = 0; m < eff_M; m++) {
                device const bfloat* vec_m = A + (size_t)m * K + a_idx;
                sum[m] += q0 * float(vec_m[0]) + q1 * float(vec_m[1]);
            }
        }
    }

    threadgroup float sdata[8][32];
    uint simd_lane = tid & 31;
    uint simd_id = tid >> 5;

    for (int m = 0; m < eff_M; m++) {
        float s_val = simd_sum(sum[m]);
        if (simd_lane == 0) {
            sdata[m][simd_id] = s_val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        uint num_simd = threads_per_group >> 5;
        for (int m = 0; m < eff_M; m++) {
            float total = (simd_lane < num_simd) ? sdata[m][simd_lane] : 0.0f;
            total = simd_sum(total);
            if (simd_lane == 0) {
                if (is_residual) {
                    out[(size_t)m * N + row] = bfloat(float(out[(size_t)m * N + row]) + total);
                } else {
                    out[(size_t)m * N + row] = bfloat(total);
                }
            }
        }
    }
}

kernel void gemm_int4_f32_batch_kernel(
    device float* out [[buffer(0)]],
    device const bfloat* A [[buffer(1)]],
    device const uint8_t* weight [[buffer(2)]],
    device const bfloat* scale [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant int& M [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    if (row >= uint(N)) return;
    int num_blocks = K / 32;
    int eff_M = min(M, 8);

    device const bfloat* row_scales = scale + row * num_blocks;
    device const uint8_t* row_w = weight + row * (K / 2);

    float sum[8] = {0.0f};

    for (int block = tid; block < num_blocks; block += threads_per_group) {
        float s = float(row_scales[block]);
        int w_offset = block * 16;
        int a_offset = block * 32;

        for (int i = 0; i < 16; i++) {
            uint8_t byte_val = row_w[w_offset + i];
            float q0 = (float(byte_val & 0x0F) - 8.0f) * s;
            float q1 = (float(byte_val >> 4) - 8.0f) * s;
            int a_idx = a_offset + i * 2;

            for (int m = 0; m < eff_M; m++) {
                device const bfloat* vec_m = A + (size_t)m * K + a_idx;
                sum[m] += q0 * float(vec_m[0]) + q1 * float(vec_m[1]);
            }
        }
    }

    threadgroup float sdata[8][32];
    uint simd_lane = tid & 31;
    uint simd_id = tid >> 5;

    for (int m = 0; m < eff_M; m++) {
        float s_val = simd_sum(sum[m]);
        if (simd_lane == 0) {
            sdata[m][simd_id] = s_val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        uint num_simd = threads_per_group >> 5;
        for (int m = 0; m < eff_M; m++) {
            float total = (simd_lane < num_simd) ? sdata[m][simd_lane] : 0.0f;
            total = simd_sum(total);
            if (simd_lane == 0) {
                out[(size_t)m * N + row] = total;
            }
        }
    }
}

kernel void gemm_int3_batch_kernel(
    device bfloat* out [[buffer(0)]],
    device const bfloat* A [[buffer(1)]],
    device const uint8_t* weight [[buffer(2)]],
    device const bfloat* scale [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant int& M [[buffer(6)]],
    constant bool& is_residual [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    if (row >= uint(N)) return;
    int blocks_per_row = K / 32;
    int eff_M = min(M, 8);

    device const bfloat* row_s = scale + row * blocks_per_row;
    device const uint8_t* row_w = weight + size_t(row) * (size_t(K) * 3 / 8);

    float sum[8] = {0.0f};

    for (int b = tid; b < blocks_per_row; b += threads_per_group) {
        float s = float(row_s[b]);
        device const uint8_t* blk_w = row_w + b * 12;
        int a_offset = b * 32;

        for (int i = 0; i < 4; i++) {
            uint8_t b0 = blk_w[i * 3 + 0];
            uint8_t b1 = blk_w[i * 3 + 1];
            uint8_t b2 = blk_w[i * 3 + 2];

            float w0 = (float(b0 & 0x07) - 4.0f) * s;
            float w1 = (float((b0 >> 3) & 0x07) - 4.0f) * s;
            float w2 = (float((b0 >> 6) | ((b1 & 0x01) << 2)) - 4.0f) * s;
            float w3 = (float((b1 >> 1) & 0x07) - 4.0f) * s;
            float w4 = (float((b1 >> 4) & 0x07) - 4.0f) * s;
            float w5 = (float((b1 >> 7) | ((b2 & 0x03) << 1)) - 4.0f) * s;
            float w6 = (float((b2 >> 2) & 0x07) - 4.0f) * s;
            float w7 = (float((b2 >> 5) & 0x07) - 4.0f) * s;

            int a_sub = a_offset + i * 8;
            for (int m = 0; m < eff_M; m++) {
                device const bfloat* v = A + (size_t)m * K + a_sub;
                sum[m] += w0 * float(v[0]) + w1 * float(v[1]) + w2 * float(v[2]) + w3 * float(v[3])
                        + w4 * float(v[4]) + w5 * float(v[5]) + w6 * float(v[6]) + w7 * float(v[7]);
            }
        }
    }

    threadgroup float sdata[8][32];
    uint simd_lane = tid & 31;
    uint simd_id = tid >> 5;

    for (int m = 0; m < eff_M; m++) {
        float s_val = simd_sum(sum[m]);
        if (simd_lane == 0) {
            sdata[m][simd_id] = s_val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        uint num_simd = threads_per_group >> 5;
        for (int m = 0; m < eff_M; m++) {
            float total = (simd_lane < num_simd) ? sdata[m][simd_lane] : 0.0f;
            total = simd_sum(total);
            if (simd_lane == 0) {
                if (is_residual) {
                    out[(size_t)m * N + row] = bfloat(float(out[(size_t)m * N + row]) + total);
                } else {
                    out[(size_t)m * N + row] = bfloat(total);
                }
            }
        }
    }
}

kernel void gemm_int3_swiglu_fused_batch_kernel(
    device bfloat* out [[buffer(0)]],
    device const bfloat* A [[buffer(1)]],
    device const uint8_t* gate_w [[buffer(2)]],
    device const bfloat* gate_s [[buffer(3)]],
    device const uint8_t* up_w [[buffer(4)]],
    device const bfloat* up_s [[buffer(5)]],
    constant int& N [[buffer(6)]],
    constant int& K [[buffer(7)]],
    constant int& M [[buffer(8)]],
    constant float& swiglu_limit [[buffer(9)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    if (row >= uint(N)) return;
    int blocks_per_row = K / 32;
    int eff_M = min(M, 8);

    device const bfloat* g_scales = gate_s + row * blocks_per_row;
    device const uint8_t* g_row_w = gate_w + size_t(row) * (size_t(K) * 3 / 8);
    device const bfloat* u_scales = up_s + row * blocks_per_row;
    device const uint8_t* u_row_w = up_w + size_t(row) * (size_t(K) * 3 / 8);

    float sum_g[8] = {0.0f};
    float sum_u[8] = {0.0f};

    for (int b = tid; b < blocks_per_row; b += threads_per_group) {
        float sg = float(g_scales[b]);
        float su = float(u_scales[b]);
        device const uint8_t* blk_gw = g_row_w + b * 12;
        device const uint8_t* blk_uw = u_row_w + b * 12;
        int a_offset = b * 32;

        for (int i = 0; i < 4; i++) {
            uint8_t gb0 = blk_gw[i * 3 + 0], gb1 = blk_gw[i * 3 + 1], gb2 = blk_gw[i * 3 + 2];
            uint8_t ub0 = blk_uw[i * 3 + 0], ub1 = blk_uw[i * 3 + 1], ub2 = blk_uw[i * 3 + 2];

            float gw0 = (float(gb0 & 0x07) - 4.0f) * sg;
            float gw1 = (float((gb0 >> 3) & 0x07) - 4.0f) * sg;
            float gw2 = (float((gb0 >> 6) | ((gb1 & 0x01) << 2)) - 4.0f) * sg;
            float gw3 = (float((gb1 >> 1) & 0x07) - 4.0f) * sg;
            float gw4 = (float((gb1 >> 4) & 0x07) - 4.0f) * sg;
            float gw5 = (float((gb1 >> 7) | ((gb2 & 0x03) << 1)) - 4.0f) * sg;
            float gw6 = (float((gb2 >> 2) & 0x07) - 4.0f) * sg;
            float gw7 = (float((gb2 >> 5) & 0x07) - 4.0f) * sg;

            float uw0 = (float(ub0 & 0x07) - 4.0f) * su;
            float uw1 = (float((ub0 >> 3) & 0x07) - 4.0f) * su;
            float uw2 = (float((ub0 >> 6) | ((ub1 & 0x01) << 2)) - 4.0f) * su;
            float uw3 = (float((ub1 >> 1) & 0x07) - 4.0f) * su;
            float uw4 = (float((ub1 >> 4) & 0x07) - 4.0f) * su;
            float uw5 = (float((ub1 >> 7) | ((ub2 & 0x03) << 1)) - 4.0f) * su;
            float uw6 = (float((ub2 >> 2) & 0x07) - 4.0f) * su;
            float uw7 = (float((ub2 >> 5) & 0x07) - 4.0f) * su;

            int a_sub = a_offset + i * 8;
            for (int m = 0; m < eff_M; m++) {
                device const bfloat* v = A + (size_t)m * K + a_sub;
                float v0 = float(v[0]), v1 = float(v[1]), v2 = float(v[2]), v3 = float(v[3]);
                float v4 = float(v[4]), v5 = float(v[5]), v6 = float(v[6]), v7 = float(v[7]);

                sum_g[m] += gw0 * v0 + gw1 * v1 + gw2 * v2 + gw3 * v3 + gw4 * v4 + gw5 * v5 + gw6 * v6 + gw7 * v7;
                sum_u[m] += uw0 * v0 + uw1 * v1 + uw2 * v2 + uw3 * v3 + uw4 * v4 + uw5 * v5 + uw6 * v6 + uw7 * v7;
            }
        }
    }

    threadgroup float sdata_g[8][32];
    threadgroup float sdata_u[8][32];
    uint simd_lane = tid & 31;
    uint simd_id = tid >> 5;

    for (int m = 0; m < eff_M; m++) {
        float sg_val = simd_sum(sum_g[m]);
        float su_val = simd_sum(sum_u[m]);
        if (simd_lane == 0) {
            sdata_g[m][simd_id] = sg_val;
            sdata_u[m][simd_id] = su_val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        uint num_simd = threads_per_group >> 5;
        for (int m = 0; m < eff_M; m++) {
            float total_g = (simd_lane < num_simd) ? sdata_g[m][simd_lane] : 0.0f;
            float total_u = (simd_lane < num_simd) ? sdata_u[m][simd_lane] : 0.0f;
            total_g = simd_sum(total_g);
            total_u = simd_sum(total_u);
            if (simd_lane == 0) {
                if (swiglu_limit > 0.0f) {
                    total_g = min(total_g, swiglu_limit);
                    total_u = clamp(total_u, -swiglu_limit, swiglu_limit);
                }
                out[(size_t)m * N + row] = bfloat(silu(total_g) * total_u);
            }
        }
    }
}

kernel void gemv_bf16_out_bf16_batch_kernel(
    device bfloat* out [[buffer(0)]],
    device const bfloat* W [[buffer(1)]],
    device const bfloat* X [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    constant int& M [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    if (row >= uint(N)) return;
    int eff_M = min(M, 8);

    device const bfloat* row_w = W + (size_t)row * K;
    float sum[8] = {0.0f};

    for (int col = tid; col < K; col += threads_per_group) {
        float w_val = float(row_w[col]);
        for (int m = 0; m < eff_M; m++) {
            sum[m] += w_val * float(X[(size_t)m * K + col]);
        }
    }

    threadgroup float sdata[8][32];
    uint simd_lane = tid & 31;
    uint simd_id = tid >> 5;

    for (int m = 0; m < eff_M; m++) {
        float s_val = simd_sum(sum[m]);
        if (simd_lane == 0) {
            sdata[m][simd_id] = s_val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        uint num_simd = threads_per_group >> 5;
        for (int m = 0; m < eff_M; m++) {
            float total = (simd_lane < num_simd) ? sdata[m][simd_lane] : 0.0f;
            total = simd_sum(total);
            if (simd_lane == 0) {
                out[(size_t)m * N + row] = bfloat(total);
            }
        }
    }
}

kernel void gemm_int4_swiglu_fused_batch_kernel(
    device bfloat* out [[buffer(0)]],
    device const bfloat* A [[buffer(1)]],
    device const uint8_t* gate_w [[buffer(2)]],
    device const bfloat* gate_s [[buffer(3)]],
    device const uint8_t* up_w [[buffer(4)]],
    device const bfloat* up_s [[buffer(5)]],
    constant int& N [[buffer(6)]],
    constant int& K [[buffer(7)]],
    constant int& M [[buffer(8)]],
    constant float& swiglu_limit [[buffer(9)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    if (row >= uint(N)) return;
    int num_blocks = K / 32;
    int eff_M = min(M, 8);

    device const bfloat* g_scales = gate_s + row * num_blocks;
    device const uint8_t* g_row_w = gate_w + row * (K / 2);
    device const bfloat* u_scales = up_s + row * num_blocks;
    device const uint8_t* u_row_w = up_w + row * (K / 2);

    float sum_g[8] = {0.0f};
    float sum_u[8] = {0.0f};

    for (int block = tid; block < num_blocks; block += threads_per_group) {
        float s_g = float(g_scales[block]);
        float s_u = float(u_scales[block]);
        int w_offset = block * 16;
        int a_offset = block * 32;

        for (int i = 0; i < 16; i++) {
            uint8_t byte_g = g_row_w[w_offset + i];
            uint8_t byte_u = u_row_w[w_offset + i];
            float qg0 = (float(byte_g & 0x0F) - 8.0f) * s_g;
            float qg1 = (float(byte_g >> 4) - 8.0f) * s_g;
            float qu0 = (float(byte_u & 0x0F) - 8.0f) * s_u;
            float qu1 = (float(byte_u >> 4) - 8.0f) * s_u;
            int a_idx = a_offset + i * 2;

            for (int m = 0; m < eff_M; m++) {
                device const bfloat* vec_m = A + (size_t)m * K + a_idx;
                float v0 = float(vec_m[0]), v1 = float(vec_m[1]);
                sum_g[m] += qg0 * v0 + qg1 * v1;
                sum_u[m] += qu0 * v0 + qu1 * v1;
            }
        }
    }

    threadgroup float sdata_g[8][32];
    threadgroup float sdata_u[8][32];
    uint simd_lane = tid & 31;
    uint simd_id = tid >> 5;

    for (int m = 0; m < eff_M; m++) {
        float sg_val = simd_sum(sum_g[m]);
        float su_val = simd_sum(sum_u[m]);
        if (simd_lane == 0) {
            sdata_g[m][simd_id] = sg_val;
            sdata_u[m][simd_id] = su_val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        uint num_simd = threads_per_group >> 5;
        for (int m = 0; m < eff_M; m++) {
            float total_g = (simd_lane < num_simd) ? sdata_g[m][simd_lane] : 0.0f;
            float total_u = (simd_lane < num_simd) ? sdata_u[m][simd_lane] : 0.0f;
            total_g = simd_sum(total_g);
            total_u = simd_sum(total_u);
            if (simd_lane == 0) {
                if (swiglu_limit > 0.0f) {
                    total_g = min(total_g, swiglu_limit);
                    total_u = clamp(total_u, -swiglu_limit, swiglu_limit);
                }
                out[(size_t)m * N + row] = bfloat(silu(total_g) * total_u);
            }
        }
    }
}
