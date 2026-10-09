#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

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
    uint pair_id [[thread_position_in_threadgroup]])
{
    if (vec_idx >= uint(n_vectors)) return;
    int half_rope = rope_dim / 2;
    if (pair_id >= uint(half_rope)) return;

    int base_idx = vec_idx * head_dim + (head_dim - rope_dim) + 2 * pair_id;

    float x0 = float(x[base_idx]);
    float x1 = float(x[base_idx + 1]);

    float cos_val = freq_table[position * half_rope * 2 + pair_id * 2];
    float sin_val = freq_table[position * half_rope * 2 + pair_id * 2 + 1];
    if (inverse) sin_val = -sin_val;

    float y0 = x0 * cos_val - x1 * sin_val;
    float y1 = x0 * sin_val + x1 * cos_val;

    x[base_idx]     = bfloat(y0);
    x[base_idx + 1] = bfloat(y1);
}

kernel void store_kv_device_pos_kernel(
    device bfloat* kv_cache [[buffer(0)]],
    device const bfloat* kv_val [[buffer(1)]],
    device const int32_t* d_position [[buffer(2)]],
    constant int& window [[buffer(3)]],
    constant int& head_dim [[buffer(4)]],
    uint d [[thread_position_in_grid]])
{
    if (d >= uint(head_dim)) return;
    int position = d_position ? *d_position : 0;
    int cache_pos = position % window;
    kv_cache[(size_t)cache_pos * head_dim + d] = kv_val[d];
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
    device bfloat* slot_conv_state [[buffer(6)]],
    constant int& has_slot [[buffer(7)]],
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

    if (has_slot == 1) {
        device bfloat* slot_cs = slot_conv_state + c * 4;
        slot_cs[0] = bfloat(s0);
        slot_cs[1] = bfloat(s1);
        slot_cs[2] = bfloat(s2);
        slot_cs[3] = bfloat(s3);
    }

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
    device bfloat* slot_ssm_state [[buffer(13)]],   // [num_v_heads, head_dim, head_dim]
    constant int& has_slot [[buffer(14)]],
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
    device bfloat* slot_state_h = (has_slot == 1) ? (slot_ssm_state + size_t(h) * head_dim * head_dim) : nullptr;

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
            if (has_slot == 1) {
                slot_state_h[r * head_dim + tid] = bfloat(new_s);
            }
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

    if ((K & 3) == 0) {
        int K4 = K >> 2;
        device const bfloat4* row_w4 = (device const bfloat4*)(W + size_t(row) * size_t(K));
        device const bfloat4* x4 = (device const bfloat4*)x;

        float sum = 0.0f;
        for (int c = tid; c < K4; c += threads_per_group) {
            float4 wf = float4(row_w4[c]);
            float4 xf = float4(x4[c]);
            sum += dot(wf, xf);
        }

        if (threads_per_group <= 32) {
            float total = simd_sum(sum);
            if (simd_lane == 0) {
                out[row] = bfloat(total);
            }
            return;
        }

        threadgroup float sdata[32];
        sum = simd_sum(sum);
        if (simd_lane == 0) sdata[simd_id] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_id == 0) {
            float total = (simd_lane < (threads_per_group >> 5)) ? sdata[simd_lane] : 0.0f;
            total = simd_sum(total);
            if (simd_lane == 0) {
                out[row] = bfloat(total);
            }
        }
        return;
    }

    device const bfloat* row_w = W + size_t(row) * size_t(K);
    float sum = 0.0f;
    for (int c = tid; c < K; c += threads_per_group) {
        sum += float(row_w[c]) * float(x[c]);
    }

    sum = simd_sum(sum);
    threadgroup float sdata[32];
    if (simd_lane == 0) sdata[simd_id] = sum;
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

    if ((K & 7) == 0) {
        int K8 = K >> 3;
        device const bfloat4* row_w4 = (device const bfloat4*)(W + size_t(row) * size_t(K));
        device const bfloat4* x4 = (device const bfloat4*)x;

        float sum = 0.0f;
        for (int c = tid; c < K8; c += threads_per_group) {
            int idx = c << 1;
            float4 wf0 = float4(row_w4[idx]);
            float4 xf0 = float4(x4[idx]);
            float4 wf1 = float4(row_w4[idx + 1]);
            float4 xf1 = float4(x4[idx + 1]);
            sum += dot(wf0, xf0) + dot(wf1, xf1);
        }

        if (threads_per_group <= 32) {
            float total = simd_sum(sum);
            if (simd_lane == 0) {
                out[row] = total;
            }
            return;
        }

        threadgroup float sdata[32];
        sum = simd_sum(sum);
        if (simd_lane == 0) sdata[simd_id] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_id == 0) {
            float total = (simd_lane < (threads_per_group >> 5)) ? sdata[simd_lane] : 0.0f;
            total = simd_sum(total);
            if (simd_lane == 0) {
                out[row] = total;
            }
        }
        return;
    }

    if ((K & 3) == 0) {
        int K4 = K >> 2;
        device const bfloat4* row_w4 = (device const bfloat4*)(W + size_t(row) * size_t(K));
        device const bfloat4* x4 = (device const bfloat4*)x;

        float sum = 0.0f;
        for (int c = tid; c < K4; c += threads_per_group) {
            float4 wf = float4(row_w4[c]);
            float4 xf = float4(x4[c]);
            sum += dot(wf, xf);
        }

        if (threads_per_group <= 32) {
            float total = simd_sum(sum);
            if (simd_lane == 0) {
                out[row] = total;
            }
            return;
        }

        threadgroup float sdata[32];
        sum = simd_sum(sum);
        if (simd_lane == 0) sdata[simd_id] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_id == 0) {
            float total = (simd_lane < (threads_per_group >> 5)) ? sdata[simd_lane] : 0.0f;
            total = simd_sum(total);
            if (simd_lane == 0) {
                out[row] = total;
            }
        }
        return;
    }

    device const bfloat* row_w = W + size_t(row) * size_t(K);
    float sum = 0.0f;
    for (int c = tid; c < K; c += threads_per_group) {
        sum += float(row_w[c]) * float(x[c]);
    }

    sum = simd_sum(sum);
    threadgroup float sdata[32];
    if (simd_lane == 0) sdata[simd_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float total = (simd_lane < (threads_per_group >> 5)) ? sdata[simd_lane] : 0.0f;
        total = simd_sum(total);
        if (simd_lane == 0) {
            out[row] = total;
        }
    }
}

kernel void gemv_bf16_f32_batch_kernel(
    device float* out [[buffer(0)]],
    device const bfloat* W [[buffer(1)]],
    device const bfloat* X [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    constant int& M [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    if (row >= uint(N)) return;

    if ((K & 7) == 0) {
        int K8 = K >> 3;
        device const bfloat4* row_w4 = (device const bfloat4*)(W + size_t(row) * size_t(K));
        device const bfloat4* x4_0 = (device const bfloat4*)X;
        device const bfloat4* x4_1 = (device const bfloat4*)(X + (size_t)1 * K);
        device const bfloat4* x4_2 = (device const bfloat4*)(X + (size_t)2 * K);
        device const bfloat4* x4_3 = (device const bfloat4*)(X + (size_t)3 * K);

        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        for (int c = tid; c < K8; c += threads_per_group) {
            int idx = c << 1;
            float4 wf0 = float4(row_w4[idx]);
            float4 wf1 = float4(row_w4[idx + 1]);

            sum0 += dot(wf0, float4(x4_0[idx])) + dot(wf1, float4(x4_0[idx + 1]));
            if (M > 1) sum1 += dot(wf0, float4(x4_1[idx])) + dot(wf1, float4(x4_1[idx + 1]));
            if (M > 2) sum2 += dot(wf0, float4(x4_2[idx])) + dot(wf1, float4(x4_2[idx + 1]));
            if (M > 3) sum3 += dot(wf0, float4(x4_3[idx])) + dot(wf1, float4(x4_3[idx + 1]));
        }

        if (threads_per_group <= 32) {
            float tot0 = simd_sum(sum0);
            if (simd_lane == 0) out[row] = tot0;
            if (M > 1) {
                float tot1 = simd_sum(sum1);
                if (simd_lane == 0) out[(size_t)1 * N + row] = tot1;
            }
            if (M > 2) {
                float tot2 = simd_sum(sum2);
                if (simd_lane == 0) out[(size_t)2 * N + row] = tot2;
            }
            if (M > 3) {
                float tot3 = simd_sum(sum3);
                if (simd_lane == 0) out[(size_t)3 * N + row] = tot3;
            }
            return;
        }

        threadgroup float sdata0[32];
        threadgroup float sdata1[32];
        sum0 = simd_sum(sum0);
        if (simd_lane == 0) sdata0[simd_id] = sum0;
        if (M > 1) {
            sum1 = simd_sum(sum1);
            if (simd_lane == 0) sdata1[simd_id] = sum1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_id == 0) {
            float total0 = (simd_lane < (threads_per_group >> 5)) ? sdata0[simd_lane] : 0.0f;
            total0 = simd_sum(total0);
            if (simd_lane == 0) out[row] = total0;
            if (M > 1) {
                float total1 = (simd_lane < (threads_per_group >> 5)) ? sdata1[simd_lane] : 0.0f;
                total1 = simd_sum(total1);
                if (simd_lane == 0) out[(size_t)1 * N + row] = total1;
            }
        }
        return;
    }

    device const bfloat* row_w = W + size_t(row) * size_t(K);
    for (int m = 0; m < M; m++) {
        device const bfloat* xm = X + (size_t)m * K;
        float sum = 0.0f;
        for (int c = tid; c < K; c += threads_per_group) {
            sum += float(row_w[c]) * float(xm[c]);
        }
        sum = simd_sum(sum);
        if (simd_lane == 0) out[(size_t)m * N + row] = sum;
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
    device bfloat* C [[buffer(0)]],            // [M, N]
    device const bfloat* A [[buffer(1)]],      // [M, K]
    device const uint8_t* W [[buffer(2)]],     // [N, K/2]
    device const bfloat* scales [[buffer(3)]], // [N, K/32]
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant int& M [[buffer(6)]],
    constant bool& is_residual [[buffer(7)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tid_in_tg [[thread_position_in_threadgroup]])
{
    uint tid = tid_in_tg.x;
    uint n_block = tg_pos.x; // tile of 32 output channels
    uint m_block = tg_pos.y; // tile of 8 tokens

    uint n_base = n_block * 32;
    uint m_base = m_block * 8;

    uint col = tid & 31;      // output channel (0..31) in tile
    uint simd_id = tid >> 5;  // 0 or 1 (splits K reduction across 2 SIMD-groups)

    uint global_n = n_base + col;
    if (global_n >= uint(N) || m_base >= uint(M)) return;

    int num_blocks = K / 32;
    device const uint8_t* row_w = W + global_n * (K / 2);
    device const bfloat* row_s = scales + global_n * num_blocks;

    float acc[8] = {0.0f};

    for (int b = simd_id; b < num_blocks; b += 2) {
        float s = float(row_s[b]);
        int w_offset = b * 16;
        int a_offset = b * 32;

        float qw[32];
        for (int i = 0; i < 16; i++) {
            uint8_t byte_val = row_w[w_offset + i];
            qw[i * 2 + 0] = (float(byte_val & 0x0F) - 8.0f) * s;
            qw[i * 2 + 1] = (float(byte_val >> 4) - 8.0f) * s;
        }

        #pragma unroll
        for (int m = 0; m < 8; m++) {
            uint cur_m = m_base + m;
            if (cur_m < uint(M)) {
                device const bfloat* a_ptr = A + (size_t)cur_m * K + a_offset;
                float sum = 0.0f;
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    sum += qw[i] * float(a_ptr[i]);
                }
                acc[m] += sum;
            }
        }
    }

    threadgroup float s_acc[8][64];
    for (int m = 0; m < 8; m++) {
        s_acc[m][tid] = acc[m];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        for (int m = 0; m < 8; m++) {
            uint cur_m = m_base + m;
            if (cur_m < uint(M)) {
                float total = s_acc[m][col] + s_acc[m][col + 32];
                if (is_residual) {
                    C[(size_t)cur_m * N + global_n] = bfloat(float(C[(size_t)cur_m * N + global_n]) + total);
                } else {
                    C[(size_t)cur_m * N + global_n] = bfloat(total);
                }
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════
//  Hardware MPP (MetalPerformancePrimitives) Tiled INT4 & INT3 GEMM Kernels
// ════════════════════════════════════════════════════════════════════════════════

kernel void gemm_int4_mpp_kernel(
    device bfloat* C [[buffer(0)]],            // [M, N]
    device const bfloat* A [[buffer(1)]],      // [M, K]
    device const uint8_t* W [[buffer(2)]],     // [N, K/2]
    device const bfloat* scales [[buffer(3)]], // [N, K/32]
    constant int& M [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& K [[buffer(6)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr int NRA = 64; // N tile (weights)
    constexpr int NRB = 32; // M tile (tokens)
    constexpr int NK  = 32; // K tile (one scale block of 32 weights)

    const int ra = tgpig.y * NRA; // N offset
    const int rb = tgpig.x * NRB; // M offset

    if (ra >= N || rb >= M) return;

    // Threadgroup memory for dequantized W tile: [NRA, NK] = [64, 32] bfloats = 4096 bytes
    threadgroup bfloat* s_w = (threadgroup bfloat*)shmem;
    auto tW = tensor(s_w, dextents<int32_t, 2>(NK, NRA));

    device bfloat* ptrA = (device bfloat*)(A + rb * K);
    auto tA = tensor(ptrA, dextents<int32_t, 2>(K, M - rb), array<int, 2>({1, K}));

    matmul2d<
        matmul2d_descriptor(NRB, NRA, dynamic_extent, false, true, true,
                            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tW), bfloat>();

    // 128 threads in threadgroup (4 simdgroups)
    // Dequantize 64 rows of W x 32 elements = 2048 bfloats = 16 bfloats per thread
    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        int row_idx = tiitg >> 1;  // 0..63
        int sub_block = tiitg & 1; // 0 or 1 (16 elements each = 8 bytes of INT4)

        int global_row = ra + row_idx;
        if (global_row < N) {
            float s = float(scales[global_row * (K / 32) + (loop_k / 32)]);
            device const uint8_t* src = W + (size_t)global_row * (K / 2) + (loop_k / 2) + sub_block * 8;
            threadgroup bfloat* dst = s_w + row_idx * NK + sub_block * 16;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                uint8_t byte_val = src[i];
                dst[i * 2 + 0] = bfloat((float(byte_val & 0x0F) - 8.0f) * s);
                dst[i * 2 + 1] = bfloat((float(byte_val >> 4) - 8.0f) * s);
            }
        } else {
            threadgroup bfloat* dst = s_w + row_idx * NK + sub_block * 16;
            #pragma unroll
            for (int i = 0; i < 16; i++) dst[i] = 0.0bf;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int kExt = min(NK, K - loop_k);
        auto tWv = tensor(s_w, dextents<int32_t, 2>(kExt, NRA), array<int, 2>({1, NK}));
        auto tAv = tensor(ptrA + loop_k, dextents<int32_t, 2>(kExt, M - rb), array<int, 2>({1, K}));

        mm.run(tAv, tWv, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device bfloat* dstC = C + rb * N + ra;
    auto tC = tensor(dstC, dextents<int32_t, 2>(N - ra, M - rb), array<int, 2>({1, N}));
    cT.store(tC);
}

kernel void gemm_int3_mpp_kernel(
    device bfloat* C [[buffer(0)]],            // [M, N]
    device const bfloat* A [[buffer(1)]],      // [M, K]
    device const uint8_t* W [[buffer(2)]],     // [N, K*3/8]
    device const bfloat* scales [[buffer(3)]], // [N, K/32]
    constant int& M [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& K [[buffer(6)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr int NRA = 64; // N tile (weights)
    constexpr int NRB = 32; // M tile (tokens)
    constexpr int NK  = 32; // K tile (one scale block of 32 weights)

    const int ra = tgpig.y * NRA; // N offset
    const int rb = tgpig.x * NRB; // M offset

    if (ra >= N || rb >= M) return;

    // Threadgroup memory for dequantized W tile: [NRA, NK] = [64, 32] bfloats = 4096 bytes
    threadgroup bfloat* s_w = (threadgroup bfloat*)shmem;
    auto tW = tensor(s_w, dextents<int32_t, 2>(NK, NRA));

    device bfloat* ptrA = (device bfloat*)(A + rb * K);
    auto tA = tensor(ptrA, dextents<int32_t, 2>(K, M - rb), array<int, 2>({1, K}));

    matmul2d<
        matmul2d_descriptor(NRB, NRA, dynamic_extent, false, true, true,
                            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tW), bfloat>();

    // 128 threads in threadgroup
    // Dequantize 64 rows of W x 32 elements = 2048 bfloats = 16 bfloats per thread
    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        int row_idx = tiitg >> 1;  // 0..63
        int sub_block = tiitg & 1; // 0 or 1 (16 elements each = 6 bytes of INT3)

        int global_row = ra + row_idx;
        if (global_row < N) {
            float s = float(scales[global_row * (K / 32) + (loop_k / 32)]);
            device const uint8_t* src = W + (size_t)global_row * (K * 3 / 8) + (loop_k * 3 / 8) + sub_block * 6;
            threadgroup bfloat* dst = s_w + row_idx * NK + sub_block * 16;

            #pragma unroll
            for (int i = 0; i < 2; i++) {
                uint8_t b0 = src[i * 3 + 0];
                uint8_t b1 = src[i * 3 + 1];
                uint8_t b2 = src[i * 3 + 2];

                dst[i * 8 + 0] = bfloat((float(b0 & 0x07) - 4.0f) * s);
                dst[i * 8 + 1] = bfloat((float((b0 >> 3) & 0x07) - 4.0f) * s);
                dst[i * 8 + 2] = bfloat((float((b0 >> 6) | ((b1 & 0x01) << 2)) - 4.0f) * s);
                dst[i * 8 + 3] = bfloat((float((b1 >> 1) & 0x07) - 4.0f) * s);

                dst[i * 8 + 4] = bfloat((float((b1 >> 4) & 0x07) - 4.0f) * s);
                dst[i * 8 + 5] = bfloat((float((b1 >> 7) | ((b2 & 0x03) << 1)) - 4.0f) * s);
                dst[i * 8 + 6] = bfloat((float((b2 >> 2) & 0x07) - 4.0f) * s);
                dst[i * 8 + 7] = bfloat((float((b2 >> 5) & 0x07) - 4.0f) * s);
            }
        } else {
            threadgroup bfloat* dst = s_w + row_idx * NK + sub_block * 16;
            #pragma unroll
            for (int i = 0; i < 16; i++) dst[i] = 0.0bf;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int kExt = min(NK, K - loop_k);
        auto tWv = tensor(s_w, dextents<int32_t, 2>(kExt, NRA), array<int, 2>({1, NK}));
        auto tAv = tensor(ptrA + loop_k, dextents<int32_t, 2>(kExt, M - rb), array<int, 2>({1, K}));

        mm.run(tAv, tWv, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device bfloat* dstC = C + rb * N + ra;
    auto tC = tensor(dstC, dextents<int32_t, 2>(N - ra, M - rb), array<int, 2>({1, N}));
    cT.store(tC);
}

kernel void gemm_int4_f32_batch_kernel(
    device float* C [[buffer(0)]],             // [M, N]
    device const bfloat* A [[buffer(1)]],      // [M, K]
    device const uint8_t* W [[buffer(2)]],     // [N, K/2]
    device const bfloat* scales [[buffer(3)]], // [N, K/32]
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant int& M [[buffer(6)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tid_in_tg [[thread_position_in_threadgroup]])
{
    uint tid = tid_in_tg.x;
    uint n_block = tg_pos.x; // tile of 32 output channels
    uint m_block = tg_pos.y; // tile of 8 tokens

    uint n_base = n_block * 32;
    uint m_base = m_block * 8;

    uint col = tid & 31;
    uint simd_id = tid >> 5;

    uint global_n = n_base + col;
    if (global_n >= uint(N) || m_base >= uint(M)) return;

    int num_blocks = K / 32;
    device const uint8_t* row_w = W + global_n * (K / 2);
    device const bfloat* row_s = scales + global_n * num_blocks;

    float acc[8] = {0.0f};

    for (int b = simd_id; b < num_blocks; b += 2) {
        float s = float(row_s[b]);
        int w_offset = b * 16;
        int a_offset = b * 32;

        float qw[32];
        for (int i = 0; i < 16; i++) {
            uint8_t byte_val = row_w[w_offset + i];
            qw[i * 2 + 0] = (float(byte_val & 0x0F) - 8.0f) * s;
            qw[i * 2 + 1] = (float(byte_val >> 4) - 8.0f) * s;
        }

        #pragma unroll
        for (int m = 0; m < 8; m++) {
            uint cur_m = m_base + m;
            if (cur_m < uint(M)) {
                device const bfloat* a_ptr = A + (size_t)cur_m * K + a_offset;
                float sum = 0.0f;
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    sum += qw[i] * float(a_ptr[i]);
                }
                acc[m] += sum;
            }
        }
    }

    threadgroup float s_acc[8][64];
    for (int m = 0; m < 8; m++) {
        s_acc[m][tid] = acc[m];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        for (int m = 0; m < 8; m++) {
            uint cur_m = m_base + m;
            if (cur_m < uint(M)) {
                float total = s_acc[m][col] + s_acc[m][col + 32];
                C[(size_t)cur_m * N + global_n] = total;
            }
        }
    }
}

kernel void gemm_int3_batch_kernel(
    device bfloat* C [[buffer(0)]],            // [M, N]
    device const bfloat* A [[buffer(1)]],      // [M, K]
    device const uint8_t* W [[buffer(2)]],     // [N, K * 3 / 8]
    device const bfloat* scales [[buffer(3)]], // [N, K / 32]
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant int& M [[buffer(6)]],
    constant bool& is_residual [[buffer(7)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tid_in_tg [[thread_position_in_threadgroup]])
{
    uint tid = tid_in_tg.x;
    uint n_block = tg_pos.x; // tile of 32 output channels
    uint m_block = tg_pos.y; // tile of 8 tokens

    uint n_base = n_block * 32;
    uint m_base = m_block * 8;

    uint col = tid & 31;      // output channel (0..31) in tile
    uint simd_id = tid >> 5;  // 0 or 1 (splits K reduction across 2 SIMD-groups)

    uint global_n = n_base + col;
    if (global_n >= uint(N) || m_base >= uint(M)) return;

    int num_blocks = K / 32;
    device const uint8_t* row_w = W + (size_t)global_n * ((size_t)K * 3 / 8);
    device const bfloat* row_s = scales + global_n * num_blocks;

    float acc[8] = {0.0f};

    for (int b = simd_id; b < num_blocks; b += 2) {
        float s = float(row_s[b]);
        int w_offset = b * 12;
        int a_offset = b * 32;

        float qw[32];
        for (int i = 0; i < 4; i++) {
            uint8_t b0 = row_w[w_offset + i * 3 + 0];
            uint8_t b1 = row_w[w_offset + i * 3 + 1];
            uint8_t b2 = row_w[w_offset + i * 3 + 2];

            qw[i * 8 + 0] = ((float)(b0 & 0x07) - 4.0f) * s;
            qw[i * 8 + 1] = ((float)((b0 >> 3) & 0x07) - 4.0f) * s;
            qw[i * 8 + 2] = ((float)((b0 >> 6) | ((b1 & 0x01) << 2)) - 4.0f) * s;
            qw[i * 8 + 3] = ((float)((b1 >> 1) & 0x07) - 4.0f) * s;
            qw[i * 8 + 4] = ((float)((b1 >> 4) & 0x07) - 4.0f) * s;
            qw[i * 8 + 5] = ((float)((b1 >> 7) | ((b2 & 0x03) << 1)) - 4.0f) * s;
            qw[i * 8 + 6] = ((float)((b2 >> 2) & 0x07) - 4.0f) * s;
            qw[i * 8 + 7] = ((float)((b2 >> 5) & 0x07) - 4.0f) * s;
        }

        #pragma unroll
        for (int m = 0; m < 8; m++) {
            uint cur_m = m_base + m;
            if (cur_m < uint(M)) {
                device const bfloat* a_ptr = A + (size_t)cur_m * K + a_offset;
                float sum = 0.0f;
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    sum += qw[i] * float(a_ptr[i]);
                }
                acc[m] += sum;
            }
        }
    }

    threadgroup float s_acc[8][64];
    for (int m = 0; m < 8; m++) {
        s_acc[m][tid] = acc[m];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        for (int m = 0; m < 8; m++) {
            uint cur_m = m_base + m;
            if (cur_m < uint(M)) {
                float total = s_acc[m][col] + s_acc[m][col + 32];
                if (is_residual) {
                    C[(size_t)cur_m * N + global_n] = bfloat(float(C[(size_t)cur_m * N + global_n]) + total);
                } else {
                    C[(size_t)cur_m * N + global_n] = bfloat(total);
                }
            }
        }
    }
}

kernel void gemm_int3_swiglu_fused_batch_kernel(
    device bfloat* out [[buffer(0)]],                  // [M, N]
    device const bfloat* A [[buffer(1)]],              // [M, K]
    device const uint8_t* gate_weight [[buffer(2)]],   // [N, K * 3 / 8]
    device const bfloat* gate_scale [[buffer(3)]],     // [N, K / 32]
    device const uint8_t* up_weight [[buffer(4)]],     // [N, K * 3 / 8]
    device const bfloat* up_scale [[buffer(5)]],       // [N, K / 32]
    constant int& N [[buffer(6)]],
    constant int& K [[buffer(7)]],
    constant int& M [[buffer(8)]],
    constant float& swiglu_limit [[buffer(9)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tid_in_tg [[thread_position_in_threadgroup]])
{
    uint tid = tid_in_tg.x;
    uint n_block = tg_pos.x; // tile of 32 output channels
    uint m_block = tg_pos.y; // tile of 8 tokens

    uint n_base = n_block * 32;
    uint m_base = m_block * 8;

    uint col = tid & 31;
    uint simd_id = tid >> 5;

    uint global_n = n_base + col;
    if (global_n >= uint(N) || m_base >= uint(M)) return;

    int num_blocks = K / 32;
    device const uint8_t* gw = gate_weight + (size_t)global_n * ((size_t)K * 3 / 8);
    device const bfloat* gs = gate_scale + global_n * num_blocks;
    device const uint8_t* uw = up_weight + (size_t)global_n * ((size_t)K * 3 / 8);
    device const bfloat* us = up_scale + global_n * num_blocks;

    float acc_g[8] = {0.0f};
    float acc_u[8] = {0.0f};

    for (int b = simd_id; b < num_blocks; b += 2) {
        float sg = float(gs[b]);
        float su = float(us[b]);
        int w_offset = b * 12;
        int a_offset = b * 32;

        float qg[32], qu[32];
        for (int i = 0; i < 4; i++) {
            uint8_t gb0 = gw[w_offset + i * 3 + 0];
            uint8_t gb1 = gw[w_offset + i * 3 + 1];
            uint8_t gb2 = gw[w_offset + i * 3 + 2];

            qg[i * 8 + 0] = ((float)(gb0 & 0x07) - 4.0f) * sg;
            qg[i * 8 + 1] = ((float)((gb0 >> 3) & 0x07) - 4.0f) * sg;
            qg[i * 8 + 2] = ((float)((gb0 >> 6) | ((gb1 & 0x01) << 2)) - 4.0f) * sg;
            qg[i * 8 + 3] = ((float)((gb1 >> 1) & 0x07) - 4.0f) * sg;
            qg[i * 8 + 4] = ((float)((gb1 >> 4) & 0x07) - 4.0f) * sg;
            qg[i * 8 + 5] = ((float)((gb1 >> 7) | ((gb2 & 0x03) << 1)) - 4.0f) * sg;
            qg[i * 8 + 6] = ((float)((gb2 >> 2) & 0x07) - 4.0f) * sg;
            qg[i * 8 + 7] = ((float)((gb2 >> 5) & 0x07) - 4.0f) * sg;

            uint8_t ub0 = uw[w_offset + i * 3 + 0];
            uint8_t ub1 = uw[w_offset + i * 3 + 1];
            uint8_t ub2 = uw[w_offset + i * 3 + 2];

            qu[i * 8 + 0] = ((float)(ub0 & 0x07) - 4.0f) * su;
            qu[i * 8 + 1] = ((float)((ub0 >> 3) & 0x07) - 4.0f) * su;
            qu[i * 8 + 2] = ((float)((ub0 >> 6) | ((ub1 & 0x01) << 2)) - 4.0f) * su;
            qu[i * 8 + 3] = ((float)((ub1 >> 1) & 0x07) - 4.0f) * su;
            qu[i * 8 + 4] = ((float)((ub1 >> 4) & 0x07) - 4.0f) * su;
            qu[i * 8 + 5] = ((float)((ub1 >> 7) | ((gb2 & 0x03) << 1)) - 4.0f) * su;
            qu[i * 8 + 6] = ((float)((ub2 >> 2) & 0x07) - 4.0f) * su;
            qu[i * 8 + 7] = ((float)((ub2 >> 5) & 0x07) - 4.0f) * su;
        }

        #pragma unroll
        for (int m = 0; m < 8; m++) {
            uint cur_m = m_base + m;
            if (cur_m < uint(M)) {
                device const bfloat* a_ptr = A + (size_t)cur_m * K + a_offset;
                float sum_g = 0.0f;
                float sum_u = 0.0f;
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    float a_val = float(a_ptr[i]);
                    sum_g += qg[i] * a_val;
                    sum_u += qu[i] * a_val;
                }
                acc_g[m] += sum_g;
                acc_u[m] += sum_u;
            }
        }
    }

    threadgroup float s_acc_g[8][64];
    threadgroup float s_acc_u[8][64];
    for (int m = 0; m < 8; m++) {
        s_acc_g[m][tid] = acc_g[m];
        s_acc_u[m][tid] = acc_u[m];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        for (int m = 0; m < 8; m++) {
            uint cur_m = m_base + m;
            if (cur_m < uint(M)) {
                float total_g = s_acc_g[m][col] + s_acc_g[m][col + 32];
                float total_u = s_acc_u[m][col] + s_acc_u[m][col + 32];
                if (swiglu_limit > 0.0f) {
                    total_g = min(total_g, swiglu_limit);
                    total_u = clamp(total_u, -swiglu_limit, swiglu_limit);
                }
                float silu_g = total_g / (1.0f + exp(-total_g));
                out[(size_t)cur_m * N + global_n] = bfloat(silu_g * total_u);
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
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tid_in_tg [[thread_position_in_threadgroup]])
{
    uint tid = tid_in_tg.x;
    uint n_block = tg_pos.x; // tile of 32 output channels
    uint m_block = tg_pos.y; // tile of 8 tokens

    uint n_base = n_block * 32;
    uint m_base = m_block * 8;

    uint col = tid & 31;
    uint simd_id = tid >> 5;

    uint global_n = n_base + col;
    if (global_n >= uint(N) || m_base >= uint(M)) return;

    device const bfloat* row_w = W + (size_t)global_n * K;
    float acc[8] = {0.0f};

    for (int k_blk = simd_id * 16; k_blk < K; k_blk += 32) {
        int k_end = min(k_blk + 16, K);
        float w_val[16];
        for (int k = k_blk; k < k_end; k++) {
            w_val[k - k_blk] = float(row_w[k]);
        }
        #pragma unroll
        for (int m = 0; m < 8; m++) {
            uint cur_m = m_base + m;
            if (cur_m < uint(M)) {
                device const bfloat* x_ptr = X + (size_t)cur_m * K;
                float sum = 0.0f;
                for (int k = k_blk; k < k_end; k++) {
                    sum += w_val[k - k_blk] * float(x_ptr[k]);
                }
                acc[m] += sum;
            }
        }
    }

    threadgroup float s_acc[8][64];
    for (int m = 0; m < 8; m++) {
        s_acc[m][tid] = acc[m];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        for (int m = 0; m < 8; m++) {
            uint cur_m = m_base + m;
            if (cur_m < uint(M)) {
                float total = s_acc[m][col] + s_acc[m][col + 32];
                out[(size_t)cur_m * N + global_n] = bfloat(total);
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
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tid_in_tg [[thread_position_in_threadgroup]])
{
    uint tid = tid_in_tg.x;
    uint n_block = tg_pos.x; // tile of 32 output channels
    uint m_block = tg_pos.y; // tile of 8 tokens

    uint n_base = n_block * 32;
    uint m_base = m_block * 8;

    uint col = tid & 31;
    uint simd_id = tid >> 5;

    uint global_n = n_base + col;
    if (global_n >= uint(N) || m_base >= uint(M)) return;

    int num_blocks = K / 32;
    device const uint8_t* gw = gate_w + global_n * (K / 2);
    device const bfloat* gs = gate_s + global_n * num_blocks;
    device const uint8_t* uw = up_w + global_n * (K / 2);
    device const bfloat* us = up_s + global_n * num_blocks;

    float acc_g[8] = {0.0f};
    float acc_u[8] = {0.0f};

    for (int b = simd_id; b < num_blocks; b += 2) {
        float sg = float(gs[b]);
        float su = float(us[b]);
        int w_offset = b * 16;
        int a_offset = b * 32;

        float qg[32], qu[32];
        for (int i = 0; i < 16; i++) {
            uint8_t gb = gw[w_offset + i];
            uint8_t ub = uw[w_offset + i];
            qg[i * 2 + 0] = (float(gb & 0x0F) - 8.0f) * sg;
            qg[i * 2 + 1] = (float(gb >> 4) - 8.0f) * sg;
            qu[i * 2 + 0] = (float(ub & 0x0F) - 8.0f) * su;
            qu[i * 2 + 1] = (float(ub >> 4) - 8.0f) * su;
        }

        #pragma unroll
        for (int m = 0; m < 8; m++) {
            uint cur_m = m_base + m;
            if (cur_m < uint(M)) {
                device const bfloat* a_ptr = A + (size_t)cur_m * K + a_offset;
                float sum_g = 0.0f;
                float sum_u = 0.0f;
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    float a_val = float(a_ptr[i]);
                    sum_g += qg[i] * a_val;
                    sum_u += qu[i] * a_val;
                }
                acc_g[m] += sum_g;
                acc_u[m] += sum_u;
            }
        }
    }

    threadgroup float s_acc_g[8][64];
    threadgroup float s_acc_u[8][64];
    for (int m = 0; m < 8; m++) {
        s_acc_g[m][tid] = acc_g[m];
        s_acc_u[m][tid] = acc_u[m];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        for (int m = 0; m < 8; m++) {
            uint cur_m = m_base + m;
            if (cur_m < uint(M)) {
                float total_g = s_acc_g[m][col] + s_acc_g[m][col + 32];
                float total_u = s_acc_u[m][col] + s_acc_u[m][col + 32];
                if (swiglu_limit > 0.0f) {
                    total_g = min(total_g, swiglu_limit);
                    total_u = clamp(total_u, -swiglu_limit, swiglu_limit);
                }
                float silu_g = total_g / (1.0f + exp(-total_g));
                out[(size_t)cur_m * N + global_n] = bfloat(silu_g * total_u);
            }
        }
    }
}

// =====================================================================
// High-Performance Prefill MPS / Dequantization Kernels
// =====================================================================

kernel void dequant_int4_to_fp16_kernel(
    device half* out [[buffer(0)]],            // [N, K]
    device const uint8_t* W [[buffer(1)]],     // [N, K/2]
    device const bfloat* scales [[buffer(2)]], // [N, K/32]
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    uint3 id [[thread_position_in_grid]])
{
    uint row = id.y;
    uint block2 = id.x; // pair of 32-element blocks
    if (row >= uint(N) || block2 >= uint(K / 64)) return;

    uint b0_idx = block2 * 2;
    uint b1_idx = b0_idx + 1;

    half s0 = half(float(scales[row * (K / 32) + b0_idx]));
    half s1 = half(float(scales[row * (K / 32) + b1_idx]));

    device const uint8_t* src = W + (size_t)row * (K / 2) + b0_idx * 16;
    device half4* dst = (device half4*)(out + (size_t)row * K + b0_idx * 32);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint8_t byte0 = src[i * 4 + 0];
        uint8_t byte1 = src[i * 4 + 1];
        uint8_t byte2 = src[i * 4 + 2];
        uint8_t byte3 = src[i * 4 + 3];

        half4 v0, v1;
        v0[0] = (half(float(byte0 & 0x0F)) - 8.0h) * s0;
        v0[1] = (half(float(byte0 >> 4)) - 8.0h) * s0;
        v0[2] = (half(float(byte1 & 0x0F)) - 8.0h) * s0;
        v0[3] = (half(float(byte1 >> 4)) - 8.0h) * s0;

        v1[0] = (half(float(byte2 & 0x0F)) - 8.0h) * s0;
        v1[1] = (half(float(byte2 >> 4)) - 8.0h) * s0;
        v1[2] = (half(float(byte3 & 0x0F)) - 8.0h) * s0;
        v1[3] = (half(float(byte3 >> 4)) - 8.0h) * s0;

        dst[i * 2 + 0] = v0;
        dst[i * 2 + 1] = v1;
    }

    src += 16;
    dst += 8;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint8_t byte0 = src[i * 4 + 0];
        uint8_t byte1 = src[i * 4 + 1];
        uint8_t byte2 = src[i * 4 + 2];
        uint8_t byte3 = src[i * 4 + 3];

        half4 v0, v1;
        v0[0] = (half(float(byte0 & 0x0F)) - 8.0h) * s1;
        v0[1] = (half(float(byte0 >> 4)) - 8.0h) * s1;
        v0[2] = (half(float(byte1 & 0x0F)) - 8.0h) * s1;
        v0[3] = (half(float(byte1 >> 4)) - 8.0h) * s1;

        v1[0] = (half(float(byte2 & 0x0F)) - 8.0h) * s1;
        v1[1] = (half(float(byte2 >> 4)) - 8.0h) * s1;
        v1[2] = (half(float(byte3 & 0x0F)) - 8.0h) * s1;
        v1[3] = (half(float(byte3 >> 4)) - 8.0h) * s1;

        dst[i * 2 + 0] = v0;
        dst[i * 2 + 1] = v1;
    }
}

kernel void dequant_int3_to_fp16_kernel(
    device half* out [[buffer(0)]],            // [N, K]
    device const uint8_t* W [[buffer(1)]],     // [N, K * 3 / 8]
    device const bfloat* scales [[buffer(2)]], // [N, K / 32]
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    uint3 id [[thread_position_in_grid]])
{
    uint row = id.y;
    uint block = id.x; // block of 32 elements
    if (row >= uint(N) || block >= uint(K / 32)) return;

    half s = half(float(scales[row * (K / 32) + block]));
    device const uint8_t* src = W + (size_t)row * (K * 3 / 8) + block * 12;
    device half4* dst = (device half4*)(out + (size_t)row * K + block * 32);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint8_t b0 = src[i * 3 + 0];
        uint8_t b1 = src[i * 3 + 1];
        uint8_t b2 = src[i * 3 + 2];

        half4 v0, v1;
        v0[0] = (half(float(b0 & 0x07)) - 4.0h) * s;
        v0[1] = (half(float((b0 >> 3) & 0x07)) - 4.0h) * s;
        v0[2] = (half(float((b0 >> 6) | ((b1 & 0x01) << 2))) - 4.0h) * s;
        v0[3] = (half(float((b1 >> 1) & 0x07)) - 4.0h) * s;

        v1[0] = (half(float((b1 >> 4) & 0x07)) - 4.0h) * s;
        v1[1] = (half(float((b1 >> 7) | ((b2 & 0x03) << 1))) - 4.0h) * s;
        v1[2] = (half(float((b2 >> 2) & 0x07)) - 4.0h) * s;
        v1[3] = (half(float((b2 >> 5) & 0x07)) - 4.0h) * s;

        dst[i * 2 + 0] = v0;
        dst[i * 2 + 1] = v1;
    }
}

kernel void bf16_to_fp16_kernel(
    device half* out [[buffer(0)]],
    device const bfloat* in [[buffer(1)]],
    constant int& count [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id < uint(count)) {
        out[id] = half(float(in[id]));
    }
}

kernel void fp16_to_bf16_kernel(
    device bfloat* out [[buffer(0)]],
    device const half* in [[buffer(1)]],
    constant int& count [[buffer(2)]],
    constant bool& is_residual [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id < uint(count)) {
        if (is_residual) {
            out[id] = bfloat(float(out[id]) + float(in[id]));
        } else {
            out[id] = bfloat(float(in[id]));
        }
    }
}

kernel void swiglu_fp16_to_bf16_kernel(
    device bfloat* out [[buffer(0)]],
    device const half* gate [[buffer(1)]],
    device const half* up [[buffer(2)]],
    constant int& count [[buffer(3)]],
    constant float& swiglu_limit [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id < uint(count)) {
        float g = float(gate[id]);
        float u = float(up[id]);
        if (swiglu_limit > 0.0f) {
            g = min(g, swiglu_limit);
            u = clamp(u, -swiglu_limit, swiglu_limit);
        }
        float silu_g = g / (1.0f + exp(-g));
        out[id] = bfloat(silu_g * u);
    }
}

// Fast SIMD-reduced fused A and B linear attention input projections
kernel void deltanet_in_proj_ab_batch_kernel(
    device bfloat* out_a [[buffer(0)]],
    device bfloat* out_b [[buffer(1)]],
    device const bfloat* Wa [[buffer(2)]],
    device const bfloat* Wb [[buffer(3)]],
    device const bfloat* X [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& K [[buffer(6)]],
    constant int& M [[buffer(7)]],
    uint3 tid_in_tg [[thread_position_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]])
{
    uint n = tg_pos.x;
    uint m = tg_pos.y;
    if (n >= uint(N) || m >= uint(M)) return;

    uint lane = tid_in_tg.x; // 0..31
    device const bfloat* row_wa = Wa + (size_t)n * K;
    device const bfloat* row_wb = Wb + (size_t)n * K;
    device const bfloat* row_x  = X + (size_t)m * K;

    device const bfloat4* vec_wa = (device const bfloat4*)row_wa;
    device const bfloat4* vec_wb = (device const bfloat4*)row_wb;
    device const bfloat4* vec_x  = (device const bfloat4*)row_x;
    int k_vec = K / 4;

    float sum_a = 0.0f;
    float sum_b = 0.0f;
    for (int k = lane; k < k_vec; k += 32) {
        bfloat4 wa = vec_wa[k];
        bfloat4 wb = vec_wb[k];
        bfloat4 x  = vec_x[k];
        sum_a += float(wa[0]) * float(x[0]) + float(wa[1]) * float(x[1]) +
                 float(wa[2]) * float(x[2]) + float(wa[3]) * float(x[3]);
        sum_b += float(wb[0]) * float(x[0]) + float(wb[1]) * float(x[1]) +
                 float(wb[2]) * float(x[2]) + float(wb[3]) * float(x[3]);
    }

    sum_a = simd_sum(sum_a);
    sum_b = simd_sum(sum_b);

    if (lane == 0) {
        out_a[(size_t)m * N + n] = bfloat(sum_a);
        out_b[(size_t)m * N + n] = bfloat(sum_b);
    }
}

// ════════════════════════════════════════════════════════════════════════════════
//  DeepSeek V4 MLA, Hyper-Connections & IQ2_XXS / Q2_K Kernels
// ════════════════════════════════════════════════════════════════════════════════

// ── FP8 Conversions ─────────────────────────────────────────────────────────

inline float fp8_e4m3_to_float_v2(uint8_t val) {
    if (val == 0) return 0.0f;
    uint32_t sign = (uint32_t)(val & 0x80) << 24;
    uint32_t body = ((uint32_t)(val & 0x7F) << 20) + 0x3C000000U;
    return as_type<float>(sign | body);
}

inline float e8m0_to_float_v2(uint8_t val) {
    return as_type<float>((uint32_t)val << 23);
}

inline float4 fp8_e4m3_to_float4(uchar4 val) {
    return float4(
        fp8_e4m3_to_float_v2(val.x),
        fp8_e4m3_to_float_v2(val.y),
        fp8_e4m3_to_float_v2(val.z),
        fp8_e4m3_to_float_v2(val.w)
    );
}

kernel void fp8_dequant_kernel(
    device bfloat* out [[buffer(0)]],
    device const uint8_t* weight [[buffer(1)]],
    device const uint8_t* scale [[buffer(2)]],
    constant int& rows [[buffer(3)]],
    constant int& cols [[buffer(4)]],
    constant int& block_size [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    int r = gid.y;
    int c = gid.x;
    if (r >= rows || c >= cols) return;

    int scale_cols = (cols + block_size - 1) / block_size;
    int br = r / block_size;
    int bc = c / block_size;

    float s_val = e8m0_to_float_v2(scale[br * scale_cols + bc]);
    float w_val = fp8_e4m3_to_float_v2(weight[r * cols + c]) * s_val;
    out[r * cols + c] = bfloat(w_val);
}

kernel void gemv_fp8_kernel(
    device bfloat* out [[buffer(0)]],
    device const bfloat* vec [[buffer(1)]],
    device const uint8_t* weight [[buffer(2)]],
    device const uint8_t* scale [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant int& block_size [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    if (row >= (uint)N) return;

    if (block_size == 128 && (K & 127) == 0) {
        int num_blocks = K >> 7;
        int scale_cols = num_blocks;
        int br = row >> 7;
        int scale_row_offset = br * scale_cols;
        device const uchar4* weight_u4 = (device const uchar4*)(weight + (size_t)row * K);
        device const bfloat4* vec_bf4 = (device const bfloat4*)vec;

        float local_sum = 0.0f;
        for (int b = 0; b < num_blocks; b++) {
            float s_val = e8m0_to_float_v2(scale[scale_row_offset + b]);
            float block_acc = 0.0f;
            int base_u4 = b * 32;
            for (int i = tid; i < 32; i += threads_per_group) {
                uchar4 w4 = weight_u4[base_u4 + i];
                bfloat4 v4 = vec_bf4[base_u4 + i];
                float4 wf = fp8_e4m3_to_float4(w4);
                float4 vf = float4(v4);
                block_acc += dot(wf, vf);
            }
            local_sum += block_acc * s_val;
        }

        if (threads_per_group <= 32) {
            float sum = simd_sum(local_sum);
            if (simd_lane == 0) {
                out[row] = bfloat(sum);
            }
            return;
        }

        threadgroup float s_red[32];
        float warp_sum = simd_sum(local_sum);
        if (simd_lane == 0) s_red[simd_id] = warp_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_id == 0) {
            float b_sum = (simd_lane < (threads_per_group >> 5)) ? s_red[simd_lane] : 0.0f;
            b_sum = simd_sum(b_sum);
            if (simd_lane == 0) {
                out[row] = bfloat(b_sum);
            }
        }
        return;
    }

    threadgroup float s_red[32];
    int scale_cols = (K + block_size - 1) / block_size;
    int br = row / block_size;

    float local_sum = 0.0f;
    for (int k = tid; k < K; k += threads_per_group) {
        int bc = k / block_size;
        float s_val = e8m0_to_float_v2(scale[br * scale_cols + bc]);
        float w_val = fp8_e4m3_to_float_v2(weight[row * K + k]) * s_val;
        local_sum += w_val * float(vec[k]);
    }

    float warp_sum = simd_sum(local_sum);
    if (simd_lane == 0) s_red[simd_id] = warp_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float b_sum = (simd_lane < (threads_per_group >> 5)) ? s_red[simd_lane] : 0.0f;
        b_sum = simd_sum(b_sum);
        if (simd_lane == 0) {
            out[row] = bfloat(b_sum);
        }
    }
}

kernel void gemv_fp8_grouped_kernel(
    device bfloat* out [[buffer(0)]],
    device const bfloat* vec [[buffer(1)]],
    device const uint8_t* weight [[buffer(2)]],
    device const uint8_t* scale [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant int& groups [[buffer(6)]],
    constant int& block_size [[buffer(7)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint2 thread_pos [[thread_position_in_threadgroup]],
    uint2 tpg [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    uint tid = thread_pos.x;
    uint threads_per_group = tpg.x;
    uint row = tg_pos.x;
    uint group = tg_pos.y;
    if (row >= (uint)N || group >= (uint)groups) return;

    if (block_size == 128 && (K & 127) == 0) {
        int num_blocks = K >> 7;
        int scale_cols = num_blocks;
        int scale_rows_per_group = (N + 127) >> 7;
        int br = row >> 7;

        device const uchar4* g_weight_u4 = (device const uchar4*)(weight + (size_t)group * N * K + (size_t)row * K);
        device const uint8_t* g_scale = scale + (size_t)group * scale_rows_per_group * scale_cols;
        device const bfloat4* g_vec_bf4 = (device const bfloat4*)(vec + (size_t)group * K);
        device bfloat* g_out = out + (size_t)group * N;

        int scale_row_offset = br * scale_cols;
        float local_sum = 0.0f;
        for (int b = 0; b < num_blocks; b++) {
            float s_val = e8m0_to_float_v2(g_scale[scale_row_offset + b]);
            float block_acc = 0.0f;
            int base_u4 = b * 32;
            for (int i = tid; i < 32; i += threads_per_group) {
                uchar4 w4 = g_weight_u4[base_u4 + i];
                bfloat4 v4 = g_vec_bf4[base_u4 + i];
                float4 wf = fp8_e4m3_to_float4(w4);
                float4 vf = float4(v4);
                block_acc += dot(wf, vf);
            }
            local_sum += block_acc * s_val;
        }

        if (threads_per_group <= 32) {
            float sum = simd_sum(local_sum);
            if (simd_lane == 0) {
                g_out[row] = bfloat(sum);
            }
            return;
        }

        threadgroup float s_red[32];
        float warp_sum = simd_sum(local_sum);
        if (simd_lane == 0) s_red[simd_id] = warp_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_id == 0) {
            float b_sum = (simd_lane < (threads_per_group >> 5)) ? s_red[simd_lane] : 0.0f;
            b_sum = simd_sum(b_sum);
            if (simd_lane == 0) {
                g_out[row] = bfloat(b_sum);
            }
        }
        return;
    }

    threadgroup float s_red[32];
    int scale_cols = (K + block_size - 1) / block_size;
    int scale_rows_per_group = (N + block_size - 1) / block_size;
    int br = row / block_size;

    device const uint8_t* g_weight = weight + (size_t)group * N * K;
    device const uint8_t* g_scale = scale + (size_t)group * scale_rows_per_group * scale_cols;
    device const bfloat* g_vec = vec + (size_t)group * K;
    device bfloat* g_out = out + (size_t)group * N;

    float local_sum = 0.0f;
    for (int k = tid; k < K; k += threads_per_group) {
        int bc = k / block_size;
        float s_val = e8m0_to_float_v2(g_scale[br * scale_cols + bc]);
        float w_val = fp8_e4m3_to_float_v2(g_weight[row * K + k]) * s_val;
        local_sum += w_val * float(g_vec[k]);
    }

    float warp_sum = simd_sum(local_sum);
    if (simd_lane == 0) s_red[simd_id] = warp_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float b_sum = (simd_lane < (threads_per_group >> 5)) ? s_red[simd_lane] : 0.0f;
        b_sum = simd_sum(b_sum);
        if (simd_lane == 0) {
            g_out[row] = bfloat(b_sum);
        }
    }
}

// ── Hyper-Connections (HC) Metal Kernels ────────────────────────────────────

kernel void gemv_hc_pre_norm_kernel(
    device float* mixes [[buffer(0)]],
    device const bfloat* hc_state [[buffer(1)]],
    device const float* hc_fn [[buffer(2)]],
    constant int& mix_size [[buffer(3)]],
    constant int& hc_dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    if (row >= (uint)mix_size) return;
    threadgroup float s_red[32];

    // Step 1: Compute sum of squares across hc_state in parallel
    float local_sq = 0.0f;
    for (int i = tid; i < hc_dim; i += threads_per_group) {
        float v = float(hc_state[i]);
        local_sq += v * v;
    }
    float warp_sq = simd_sum(local_sq);
    if (simd_lane == 0) s_red[simd_id] = warp_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float b_sq = (simd_lane < (threads_per_group >> 5)) ? s_red[simd_lane] : 0.0f;
        b_sq = simd_sum(b_sq);
        if (simd_lane == 0) s_red[0] = rsqrt(b_sq / float(hc_dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rsqrt_val = s_red[0];

    // Step 2: Compute dot product with row of hc_fn
    device const float* fn_row = hc_fn + (size_t)row * hc_dim;
    float local_dot = 0.0f;
    for (int i = tid; i < hc_dim; i += threads_per_group) {
        local_dot += fn_row[i] * (float(hc_state[i]) * rsqrt_val);
    }
    float warp_dot = simd_sum(local_dot);
    if (simd_lane == 0) s_red[simd_id] = warp_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float b_dot = (simd_lane < (threads_per_group >> 5)) ? s_red[simd_lane] : 0.0f;
        b_dot = simd_sum(b_dot);
        if (simd_lane == 0) mixes[row] = b_dot;
    }
}

kernel void hc_split_sinkhorn_kernel(
    device float* pre [[buffer(0)]],
    device float* post [[buffer(1)]],
    device float* comb [[buffer(2)]],
    device const float* mixes [[buffer(3)]],
    device const float* scale [[buffer(4)]],
    device const float* base [[buffer(5)]],
    constant int& hc_mult [[buffer(6)]],
    constant int& sinkhorn_iters [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    uint tid [[thread_position_in_threadgroup]])
{
    int hc = hc_mult;

    // Pre
    if (tid < (uint)hc) {
        float v = mixes[tid] * scale[0] + base[tid];
        pre[tid] = 1.0f / (1.0f + exp(-v)) + eps;
    }
    // Post
    if (tid >= (uint)hc && tid < (uint)(2 * hc)) {
        int i = tid - hc;
        float v = mixes[hc + i] * scale[1] + base[hc + i];
        post[i] = 2.0f / (1.0f + exp(-v));
    }

    threadgroup float s_comb[64];
    threadgroup float s_row_sum[8];
    threadgroup float s_col_sum[8];

    // Comb logits
    if (tid < (uint)(hc * hc)) {
        int r = tid / hc;
        int c = tid % hc;
        float v = mixes[2 * hc + r * hc + c] * scale[2] + base[2 * hc + r * hc + c];
        s_comb[r * hc + c] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Row max & softmax
    if (tid < (uint)hc) {
        int r = tid;
        float max_val = -1e38f;
        for (int c = 0; c < hc; c++) {
            max_val = max(max_val, s_comb[r * hc + c]);
        }
        float row_sum = 0.0f;
        for (int c = 0; c < hc; c++) {
            float e = exp(s_comb[r * hc + c] - max_val);
            s_comb[r * hc + c] = e;
            row_sum += e;
        }
        for (int c = 0; c < hc; c++) {
            s_comb[r * hc + c] = (s_comb[r * hc + c] / row_sum) + eps;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Col normalize
    if (tid < (uint)hc) {
        int c = tid;
        float col_sum = 0.0f;
        for (int r = 0; r < hc; r++) {
            col_sum += s_comb[r * hc + c];
        }
        float inv = 1.0f / (col_sum + eps);
        for (int r = 0; r < hc; r++) {
            s_comb[r * hc + c] *= inv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Sinkhorn loop
    for (int iter = 0; iter < sinkhorn_iters - 1; iter++) {
        // Row norm
        if (tid < (uint)hc) {
            int r = tid;
            float row_sum = 0.0f;
            for (int c = 0; c < hc; c++) {
                row_sum += s_comb[r * hc + c];
            }
            s_row_sum[r] = 1.0f / (row_sum + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < (uint)(hc * hc)) {
            int r = tid / hc;
            s_comb[tid] *= s_row_sum[r];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Col norm
        if (tid < (uint)hc) {
            int c = tid;
            float col_sum = 0.0f;
            for (int r = 0; r < hc; r++) {
                col_sum += s_comb[r * hc + c];
            }
            s_col_sum[c] = 1.0f / (col_sum + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < (uint)(hc * hc)) {
            int c = tid % hc;
            s_comb[tid] *= s_col_sum[c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < (uint)(hc * hc)) {
        comb[tid] = s_comb[tid];
    }
}

kernel void hc_pre_weighted_add_kernel(
    device bfloat* hidden [[buffer(0)]],
    device const bfloat* hc_state [[buffer(1)]],
    device const float* pre_weights [[buffer(2)]],
    constant int& dim [[buffer(3)]],
    constant int& hc [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)dim) return;
    float sum = 0.0f;
    for (int h = 0; h < hc; h++) {
        sum += pre_weights[h] * float(hc_state[(size_t)h * dim + gid]);
    }
    hidden[gid] = bfloat(sum);
}

kernel void hc_pre_weighted_add_norm_kernel(
    device bfloat* out [[buffer(0)]],
    device const bfloat* hc_state [[buffer(1)]],
    device const float* pre_weights [[buffer(2)]],
    device const bfloat* norm_weight [[buffer(3)]],
    constant int& dim [[buffer(4)]],
    constant int& hc [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float s_red[32];

    float w0 = pre_weights[0];
    float w1 = pre_weights[1];
    float w2 = pre_weights[2];
    float w3 = pre_weights[3];

    device const bfloat* s0 = hc_state + 0 * dim;
    device const bfloat* s1 = hc_state + 1 * dim;
    device const bfloat* s2 = hc_state + 2 * dim;
    device const bfloat* s3 = hc_state + 3 * dim;

    float local_sq = 0.0f;
    for (int i = tid; i < dim; i += threads_per_group) {
        float f0 = float(s0[i]);
        float f1 = float(s1[i]);
        float f2 = float(s2[i]);
        float f3 = float(s3[i]);
        float y = w0 * f0 + w1 * f1 + w2 * f2 + w3 * f3;
        local_sq += y * y;
    }

    float warp_sq = simd_sum(local_sq);
    if (simd_lane == 0) s_red[simd_id] = warp_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float b_sq = (simd_lane < (threads_per_group >> 5)) ? s_red[simd_lane] : 0.0f;
        b_sq = simd_sum(b_sq);
        if (simd_lane == 0) s_red[0] = rsqrt(b_sq / float(dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rsqrt_val = s_red[0];

    for (int i = tid; i < dim; i += threads_per_group) {
        float f0 = float(s0[i]);
        float f1 = float(s1[i]);
        float f2 = float(s2[i]);
        float f3 = float(s3[i]);
        float y = w0 * f0 + w1 * f1 + w2 * f2 + w3 * f3;
        float nw = float(norm_weight[i]);
        out[i] = bfloat(y * rsqrt_val * nw);
    }
}

kernel void hc_post_update_kernel(
    device bfloat* hc_state [[buffer(0)]],
    device const bfloat* hidden [[buffer(1)]],
    device const bfloat* hc_residual [[buffer(2)]],
    device const float* post_weights [[buffer(3)]],
    device const float* comb_weights [[buffer(4)]],
    constant int& dim [[buffer(5)]],
    constant int& hc [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= (uint)dim) return;

    float p0 = post_weights[0], p1 = post_weights[1], p2 = post_weights[2], p3 = post_weights[3];
    float c00 = comb_weights[0], c01 = comb_weights[1], c02 = comb_weights[2], c03 = comb_weights[3];
    float c10 = comb_weights[4], c11 = comb_weights[5], c12 = comb_weights[6], c13 = comb_weights[7];
    float c20 = comb_weights[8], c21 = comb_weights[9], c22 = comb_weights[10], c23 = comb_weights[11];
    float c30 = comb_weights[12], c31 = comb_weights[13], c32 = comb_weights[14], c33 = comb_weights[15];

    float h_val = float(hidden[idx]);
    float r0 = float(hc_residual[0 * dim + idx]);
    float r1 = float(hc_residual[1 * dim + idx]);
    float r2 = float(hc_residual[2 * dim + idx]);
    float r3 = float(hc_residual[3 * dim + idx]);

    hc_state[0 * dim + idx] = bfloat(p0 * h_val + c00 * r0 + c10 * r1 + c20 * r2 + c30 * r3);
    hc_state[1 * dim + idx] = bfloat(p1 * h_val + c01 * r0 + c11 * r1 + c21 * r2 + c31 * r3);
    hc_state[2 * dim + idx] = bfloat(p2 * h_val + c02 * r0 + c12 * r1 + c22 * r2 + c32 * r3);
    hc_state[3 * dim + idx] = bfloat(p3 * h_val + c03 * r0 + c13 * r1 + c23 * r2 + c33 * r3);
}

kernel void hc_head_reduce_kernel(
    device bfloat* hidden [[buffer(0)]],
    device const bfloat* hc_state [[buffer(1)]],
    device const float* mixes [[buffer(2)]],
    device const float* scale [[buffer(3)]],
    device const float* base [[buffer(4)]],
    constant int& dim [[buffer(5)]],
    constant int& hc [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= (uint)dim) return;

    float s = scale[0];
    float sum = 0.0f;
    for (int h = 0; h < hc; h++) {
        float mix = mixes[h];
        float b = base[h];
        float arg = mix * s + b;
        float w = 1.0f / (1.0f + exp(-arg)) + 1e-6f;
        sum += w * float(hc_state[h * dim + idx]);
    }
    hidden[idx] = bfloat(sum);
}

// ── Multi-Head Latent Attention (MLA) Fused Metal Kernel ────────────────────

kernel void mla_attention_fused_kernel(
    device const bfloat* raw_q [[buffer(0)]],
    device const bfloat* raw_kv [[buffer(1)]],
    device const bfloat* comp_kv [[buffer(2)]],
    device const float* attn_sink [[buffer(3)]],
    device bfloat* out [[buffer(4)]],
    device const int32_t* d_position [[buffer(5)]],
    device const int32_t* d_comp_count [[buffer(6)]],
    device const float* freq_table [[buffer(7)]],
    constant int& max_cache_len [[buffer(8)]],
    constant int& head_dim [[buffer(9)]],
    constant int& rope_dim [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    constant float& q_norm_eps [[buffer(12)]],
    device const uint8_t* comp_mask [[buffer(13)]],
    constant int& window [[buffer(14)]],
    threadgroup float* s_scores [[threadgroup(0)]],
    uint h [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float s_q[512];
    threadgroup float s_out[512];
    threadgroup float s_red[32];

    // Step 1: Load raw Q & compute unweighted RMSNorm in shared memory
    float local_sum_sq = 0.0f;
    for (int d = tid; d < head_dim; d += threads_per_group) {
        float val = float(raw_q[h * head_dim + d]);
        s_q[d] = val;
        local_sum_sq += val * val;
    }
    float warp_sum_sq = simd_sum(local_sum_sq);
    if (simd_lane == 0) s_red[simd_id] = warp_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float b_sq = (simd_lane < (threads_per_group >> 5)) ? s_red[simd_lane] : 0.0f;
        b_sq = simd_sum(b_sq);
        if (simd_lane == 0) s_red[0] = rsqrt(b_sq / float(head_dim) + q_norm_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rsqrt_val = s_red[0];

    for (int d = tid; d < head_dim; d += threads_per_group) {
        s_q[d] *= rsqrt_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Forward RoPE rotation on Q
    int pos = d_position ? *d_position : 0;
    int half_rope = rope_dim / 2;
    if (tid < (uint)half_rope) {
        int pair_id = tid;
        int base_idx = (head_dim - rope_dim) + 2 * pair_id;
        float x0 = s_q[base_idx];
        float x1 = s_q[base_idx + 1];

        float cos_val = freq_table[pos * half_rope * 2 + pair_id * 2];
        float sin_val = freq_table[pos * half_rope * 2 + pair_id * 2 + 1];

        s_q[base_idx]     = x0 * cos_val - x1 * sin_val;
        s_q[base_idx + 1] = x0 * sin_val + x1 * cos_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Compute attention dot products
    int n_raw = (pos + 1 < window) ? (pos + 1) : window;
    int n_comp = (pos + 1 > window && d_comp_count != nullptr && comp_kv != nullptr) ? *d_comp_count : 0;
    int cache_len = n_raw + n_comp;
    if (cache_len > max_cache_len) cache_len = max_cache_len;
    if (cache_len < 1) cache_len = 1;

    int raw_start = (pos + 1 > window) ? ((pos + 1) % window) : 0;

    for (int t = tid; t < cache_len; t += threads_per_group) {
        device const bfloat* kv_t = nullptr;
        if (t < n_raw) {
            int raw_slot = (raw_start + t) % window;
            kv_t = raw_kv + (size_t)raw_slot * head_dim;
        } else {
            int comp_slot = t - n_raw;
            if (comp_mask != nullptr && comp_mask[comp_slot] == 0) {
                s_scores[t] = -1e38f;
                continue;
            }
            kv_t = comp_kv + (size_t)comp_slot * head_dim;
        }

        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += s_q[d] * float(kv_t[d]);
        }
        s_scores[t] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Softmax Max + Exp + Sum
    float local_max = -1e38f;
    for (int t = tid; t < cache_len; t += threads_per_group) {
        local_max = max(local_max, s_scores[t]);
    }
    if (attn_sink != nullptr) {
        local_max = max(local_max, attn_sink[h]);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = max(local_max, simd_shuffle_down(local_max, offset));
    }
    if (simd_lane == 0) s_red[simd_id] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float b_max = (simd_lane < (threads_per_group >> 5)) ? s_red[simd_lane] : -1e38f;
        for (int offset = 16; offset > 0; offset /= 2) {
            b_max = max(b_max, simd_shuffle_down(b_max, offset));
        }
        if (simd_lane == 0) s_red[0] = b_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float block_max = s_red[0];

    float local_sum = 0.0f;
    for (int t = tid; t < cache_len; t += threads_per_group) {
        float e = exp(s_scores[t] - block_max);
        s_scores[t] = e;
        local_sum += e;
    }
    if (tid == 0 && attn_sink != nullptr) {
        local_sum += exp(attn_sink[h] - block_max);
    }

    float warp_sum = simd_sum(local_sum);
    if (simd_lane == 0) s_red[simd_id] = warp_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float b_sum = (simd_lane < (threads_per_group >> 5)) ? s_red[simd_lane] : 0.0f;
        b_sum = simd_sum(b_sum);
        if (simd_lane == 0) s_red[0] = 1.0f / b_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = s_red[0];

    for (int t = tid; t < cache_len; t += threads_per_group) {
        s_scores[t] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 5: Weighted Value Reduction (scores @ KV)
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    int d0 = tid * 2;
    int d1 = tid * 2 + 1;

    for (int t = 0; t < n_raw; t++) {
        int raw_slot = (raw_start + t) % window;
        float sc = s_scores[t];
        sum0 += sc * float(raw_kv[(size_t)raw_slot * head_dim + d0]);
        sum1 += sc * float(raw_kv[(size_t)raw_slot * head_dim + d1]);
    }
    int safe_comp = n_comp;
    if (n_raw + safe_comp > max_cache_len) safe_comp = max_cache_len - n_raw;
    if (safe_comp < 0) safe_comp = 0;
    for (int t = 0; t < safe_comp; t++) {
        float sc = s_scores[n_raw + t];
        sum0 += sc * float(comp_kv[(size_t)t * head_dim + d0]);
        sum1 += sc * float(comp_kv[(size_t)t * head_dim + d1]);
    }
    s_out[d0] = sum0;
    s_out[d1] = sum1;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 6: Inverse RoPE Rotation & Store
    if (tid < (uint)half_rope) {
        int pair_id = tid;
        int base_idx = (head_dim - rope_dim) + 2 * pair_id;
        float y0 = s_out[base_idx];
        float y1 = s_out[base_idx + 1];

        float cos_val = freq_table[pos * half_rope * 2 + pair_id * 2];
        float sin_val = -freq_table[pos * half_rope * 2 + pair_id * 2 + 1];

        s_out[base_idx]     = y0 * cos_val - y1 * sin_val;
        s_out[base_idx + 1] = y0 * sin_val + y1 * cos_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    out[h * head_dim + d0] = bfloat(s_out[d0]);
    out[h * head_dim + d1] = bfloat(s_out[d1]);
}

// ── KV Compressor Step Metal Kernel ─────────────────────────────────────────

kernel void compressor_device_step_kernel(
    device const int32_t* d_position [[buffer(0)]],
    device int32_t* d_comp_count [[buffer(1)]],
    device const float* proj_kv [[buffer(2)]],
    device const float* proj_gate [[buffer(3)]],
    device float* comp_kv_state [[buffer(4)]],
    device float* comp_score_state [[buffer(5)]],
    device const float* comp_ape [[buffer(6)]],
    device const bfloat* comp_norm [[buffer(7)]],
    device bfloat* comp_kv_cache [[buffer(8)]],
    device const float* rope_freqs_compressed [[buffer(9)]],
    constant int& ratio [[buffer(10)]],
    constant int& head_dim [[buffer(11)]],
    constant int& rope_dim [[buffer(12)]],
    constant float& rms_norm_eps [[buffer(13)]],
    device const float* idx_proj_kv [[buffer(14)]],
    device const float* idx_proj_gate [[buffer(15)]],
    device float* idx_kv_state [[buffer(16)]],
    device float* idx_score_state [[buffer(17)]],
    device const float* idx_ape [[buffer(18)]],
    device const bfloat* idx_norm [[buffer(19)]],
    device bfloat* idx_comp_kv_cache [[buffer(20)]],
    constant int& max_comp [[buffer(21)]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    int pos = *d_position;
    int pos_mod = pos % ratio;
    bool overlap = (ratio == 4);
    int coff = overlap ? 2 : 1;
    int proj_dim = coff * head_dim;
    int state_idx = overlap ? (ratio + pos_mod) : pos_mod;

    // 1. Copy projection to state & add APE bias
    for (int i = tid; i < proj_dim; i += threads_per_group) {
        comp_kv_state[(size_t)state_idx * proj_dim + i] = proj_kv[i];
        comp_score_state[(size_t)state_idx * proj_dim + i] = proj_gate[i] + comp_ape[(size_t)pos_mod * proj_dim + i];
    }
    if (idx_proj_kv != nullptr) {
        int idx_proj_dim = 256;
        for (int i = tid; i < idx_proj_dim; i += threads_per_group) {
            idx_kv_state[(size_t)state_idx * idx_proj_dim + i] = idx_proj_kv[i];
            idx_score_state[(size_t)state_idx * idx_proj_dim + i] = idx_proj_gate[i] + idx_ape[(size_t)pos_mod * idx_proj_dim + i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Check boundary
    if ((pos + 1) % ratio != 0) return;

    threadgroup float s_pooled[512];
    threadgroup float s_normed[512];
    threadgroup float s_warp_sq[32];
    threadgroup float s_sq_sum;

    // 3. Softmax-gated pooling
    for (int j = tid; j < head_dim; j += threads_per_group) {
        float max_score = -1e38f;
        if (overlap) {
            for (int r = 0; r < ratio; r++) {
                float sp = comp_score_state[(size_t)r * proj_dim + j];
                float sc = comp_score_state[(size_t)(ratio + r) * proj_dim + head_dim + j];
                max_score = max(max_score, max(sp, sc));
            }
            float denom = 0.0f;
            float sum = 0.0f;
            for (int r = 0; r < ratio; r++) {
                float wp = exp(comp_score_state[(size_t)r * proj_dim + j] - max_score);
                float wc = exp(comp_score_state[(size_t)(ratio + r) * proj_dim + head_dim + j] - max_score);
                denom += wp + wc;
                sum += wp * comp_kv_state[(size_t)r * proj_dim + j] + wc * comp_kv_state[(size_t)(ratio + r) * proj_dim + head_dim + j];
            }
            s_pooled[j] = denom > 0.0f ? (sum / denom) : 0.0f;
        } else {
            for (int r = 0; r < ratio; r++) {
                max_score = max(max_score, comp_score_state[(size_t)r * proj_dim + j]);
            }
            float denom = 0.0f;
            float sum = 0.0f;
            for (int r = 0; r < ratio; r++) {
                float w = exp(comp_score_state[(size_t)r * proj_dim + j] - max_score);
                denom += w;
                sum += w * comp_kv_state[(size_t)r * proj_dim + j];
            }
            s_pooled[j] = denom > 0.0f ? (sum / denom) : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4. RMSNorm
    float thread_sq = 0.0f;
    for (int j = tid; j < head_dim; j += threads_per_group) {
        thread_sq += s_pooled[j] * s_pooled[j];
    }
    float warp_sq = simd_sum(thread_sq);
    if (simd_lane == 0) s_warp_sq[simd_id] = warp_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float b_sq = (simd_lane < (threads_per_group >> 5)) ? s_warp_sq[simd_lane] : 0.0f;
        b_sq = simd_sum(b_sq);
        if (simd_lane == 0) s_sq_sum = b_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_scale = rsqrt(s_sq_sum / float(head_dim) + rms_norm_eps);
    for (int j = tid; j < head_dim; j += threads_per_group) {
        s_normed[j] = s_pooled[j] * rms_scale * float(comp_norm[j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 5. RoPE on compressed entry
    int comp_pos = pos + 1 - ratio;
    if (comp_pos >= 65536) comp_pos = 65535;
    if (comp_pos < 0) comp_pos = 0;
    int half_rope = rope_dim / 2;
    if (tid < (uint)half_rope) {
        int pair_id = tid;
        int base_idx = (head_dim - rope_dim) + 2 * pair_id;
        float x0 = s_normed[base_idx];
        float x1 = s_normed[base_idx + 1];
        float cos_val = rope_freqs_compressed[comp_pos * half_rope * 2 + pair_id * 2];
        float sin_val = rope_freqs_compressed[comp_pos * half_rope * 2 + pair_id * 2 + 1];
        s_normed[base_idx]     = x0 * cos_val - x1 * sin_val;
        s_normed[base_idx + 1] = x0 * sin_val + x1 * cos_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 6. Store to cache & shift state
    int comp_idx = *d_comp_count;
    if (max_comp <= 0 || comp_idx < max_comp) {
        for (int j = tid; j < head_dim; j += threads_per_group) {
            comp_kv_cache[(size_t)comp_idx * head_dim + j] = bfloat(s_normed[j]);
        }
    }
    if (overlap) {
        int half_elements = ratio * proj_dim;
        for (int i = tid; i < half_elements; i += threads_per_group) {
            comp_kv_state[i] = comp_kv_state[half_elements + i];
            comp_score_state[i] = comp_score_state[half_elements + i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 7. If ratio == 4, Indexer Compressor
    if (idx_proj_kv != nullptr) {
        int idx_head_dim = 128;
        int idx_proj_dim = 256;
        threadgroup float s_idx_pooled[128];
        threadgroup float s_idx_normed[128];
        threadgroup float s_idx_sq_sum;

        if (tid < (uint)idx_head_dim) {
            int j = tid;
            float max_score = -1e38f;
            for (int r = 0; r < ratio; r++) {
                float sp = idx_score_state[(size_t)r * idx_proj_dim + j];
                float sc = idx_score_state[(size_t)(ratio + r) * idx_proj_dim + idx_head_dim + j];
                max_score = max(max_score, max(sp, sc));
            }
            float denom = 0.0f;
            float sum = 0.0f;
            for (int r = 0; r < ratio; r++) {
                float wp = exp(idx_score_state[(size_t)r * idx_proj_dim + j] - max_score);
                float wc = exp(idx_score_state[(size_t)(ratio + r) * idx_proj_dim + idx_head_dim + j] - max_score);
                denom += wp + wc;
                sum += wp * idx_kv_state[(size_t)r * idx_proj_dim + j] + wc * idx_kv_state[(size_t)(ratio + r) * idx_proj_dim + idx_head_dim + j];
            }
            s_idx_pooled[j] = denom > 0.0f ? (sum / denom) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float idx_thread_sq = (tid < (uint)idx_head_dim) ? (s_idx_pooled[tid] * s_idx_pooled[tid]) : 0.0f;
        float idx_warp_sq = simd_sum(idx_thread_sq);
        if (simd_lane == 0) s_warp_sq[simd_id] = idx_warp_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_id == 0) {
            float b_sq = (simd_lane < (threads_per_group >> 5)) ? s_warp_sq[simd_lane] : 0.0f;
            b_sq = simd_sum(b_sq);
            if (simd_lane == 0) s_idx_sq_sum = b_sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float idx_rms_scale = rsqrt(s_idx_sq_sum / float(idx_head_dim) + rms_norm_eps);
        if (tid < (uint)idx_head_dim) {
            s_idx_normed[tid] = s_idx_pooled[tid] * idx_rms_scale * float(idx_norm[tid]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < (uint)half_rope) {
            int pair_id = tid;
            int base_idx = (idx_head_dim - rope_dim) + 2 * pair_id;
            float x0 = s_idx_normed[base_idx];
            float x1 = s_idx_normed[base_idx + 1];
            float cos_val = rope_freqs_compressed[comp_pos * half_rope * 2 + pair_id * 2];
            float sin_val = rope_freqs_compressed[comp_pos * half_rope * 2 + pair_id * 2 + 1];
            s_idx_normed[base_idx]     = x0 * cos_val - x1 * sin_val;
            s_idx_normed[base_idx + 1] = x0 * sin_val + x1 * cos_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < (uint)idx_head_dim) {
            if (max_comp <= 0 || comp_idx < max_comp) {
                idx_comp_kv_cache[(size_t)comp_idx * idx_head_dim + tid] = bfloat(s_idx_normed[tid]);
            }
        }
        int idx_half_elements = ratio * idx_proj_dim;
        for (int i = tid; i < idx_half_elements; i += threads_per_group) {
            idx_kv_state[i] = idx_kv_state[idx_half_elements + i];
            idx_score_state[i] = idx_score_state[idx_half_elements + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        if (max_comp <= 0 || *d_comp_count < max_comp) {
            *d_comp_count += 1;
        }
    }
}

// ── Indexer Scoring & Top-K Masking Metal Kernels ───────────────────────────

kernel void indexer_score_kernel(
    device float* out_scores [[buffer(0)]],
    device const bfloat* index_comp [[buffer(1)]],
    device const bfloat* q [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    device const int32_t* d_comp_count [[buffer(4)]],
    constant int& max_comp [[buffer(5)]],
    uint c [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    int n_comp = *d_comp_count;
    if (c >= (uint)n_comp || c >= (uint)max_comp) return;

    device const bfloat* kv_c = index_comp + (size_t)c * 128;
    threadgroup float s_kv[128];
    threadgroup float s_warp_scores[32];

    if (tid < 128) {
        s_kv[tid] = float(kv_c[tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_total = 0.0f;
    for (int h = tid; h < 64; h += threads_per_group) {
        device const bfloat* q_h = q + (size_t)h * 128;
        float dot = 0.0f;
        for (int d = 0; d < 128; d++) {
            dot += s_kv[d] * float(q_h[d]);
        }
        if (dot > 0.0f) {
            local_total += dot * weights[h];
        }
    }

    float warp_total = simd_sum(local_total);
    if (simd_lane == 0) s_warp_scores[simd_id] = warp_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float b_total = (simd_lane < (threads_per_group >> 5)) ? s_warp_scores[simd_lane] : 0.0f;
        b_total = simd_sum(b_total);
        if (simd_lane == 0) out_scores[c] = b_total;
    }
}

kernel void indexer_mask_topk_kernel(
    device uint8_t* out_mask [[buffer(0)]],
    device const float* scores [[buffer(1)]],
    device const int32_t* d_comp_count [[buffer(2)]],
    constant int& max_comp [[buffer(3)]],
    constant int& top_k [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    int n_comp = *d_comp_count;
    if (n_comp <= 0) return;
    if (n_comp > max_comp) n_comp = max_comp;

    // Fast path: all fit
    if (n_comp <= top_k) {
        for (int i = tid; i < n_comp; i += threads_per_group) {
            out_mask[i] = 1;
        }
        return;
    }

    threadgroup float s_min[32];
    threadgroup float s_max[32];

    float local_min = 1e38f;
    float local_max = -1e38f;
    for (int i = tid; i < n_comp; i += threads_per_group) {
        float v = scores[i];
        local_min = min(local_min, v);
        local_max = max(local_max, v);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_min = min(local_min, simd_shuffle_down(local_min, offset));
        local_max = max(local_max, simd_shuffle_down(local_max, offset));
    }

    if (simd_lane == 0) {
        s_min[simd_id] = local_min;
        s_max[simd_id] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float b_min = (simd_lane < (threads_per_group >> 5)) ? s_min[simd_lane] : 1e38f;
        float b_max = (simd_lane < (threads_per_group >> 5)) ? s_max[simd_lane] : -1e38f;
        for (int offset = 16; offset > 0; offset /= 2) {
            b_min = min(b_min, simd_shuffle_down(b_min, offset));
            b_max = max(b_max, simd_shuffle_down(b_max, offset));
        }
        if (simd_lane == 0) {
            s_min[0] = b_min;
            s_max[0] = b_max;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_min = s_min[0];
    float global_max = s_max[0];
    float low = global_min;
    float high = global_max;
    threadgroup int s_count;

    for (int iter = 0; iter < 16; iter++) {
        float mid = 0.5f * (low + high);
        int local_cnt = 0;
        for (int i = tid; i < n_comp; i += threads_per_group) {
            if (scores[i] >= mid) local_cnt++;
        }
        int warp_cnt = simd_sum(local_cnt);
        if (simd_lane == 0) s_min[simd_id] = float(warp_cnt);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_id == 0) {
            float b_cnt = (simd_lane < (threads_per_group >> 5)) ? s_min[simd_lane] : 0.0f;
            b_cnt = simd_sum(b_cnt);
            if (simd_lane == 0) s_count = int(b_cnt);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (s_count >= top_k) {
            low = mid;
        } else {
            high = mid;
        }
    }

    float thresh = low;
    for (int i = tid; i < n_comp; i += threads_per_group) {
        out_mask[i] = (scores[i] >= thresh) ? 1 : 0;
    }
}

// ── IQ2_XXS & Q2_K Quantization MSL Compute Shaders ─────────────────────────

constant uint8_t c_ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

constant uint64_t c_iq2xxs_grid[256] = {
    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL,
    0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x08080808082b0808ULL,
    0x08080808082b082bULL, 0x08080808082b2b08ULL, 0x08080808082b2b2bULL, 0x0808080819080819ULL,
    0x0808080819081908ULL, 0x0808080819190808ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL,
    0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b082b2bULL,
    0x080808082b2b082bULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x0808081908190808ULL,
    0x0808081908191919ULL, 0x0808081919080808ULL, 0x080808192b081908ULL, 0x080808192b192b08ULL,
    0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b082b082bULL, 0x0808082b2b08082bULL,
    0x0808190808080819ULL, 0x0808190808081908ULL, 0x0808190808190808ULL, 0x08081908082b0819ULL,
    0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819082b08ULL,
    0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL,
    0x080819082b2b1908ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908082b08ULL,
    0x08081919082b0808ULL, 0x080819191908192bULL, 0x08081919192b2b19ULL, 0x080819192b080808ULL,
    0x080819192b190819ULL, 0x0808192b08082b19ULL, 0x0808192b08190808ULL, 0x0808192b19080808ULL,
    0x0808192b2b081908ULL, 0x0808192b2b2b1908ULL, 0x08082b0808080808ULL, 0x08082b0808081919ULL,
    0x08082b0808082b08ULL, 0x08082b0808191908ULL, 0x08082b08082b2b08ULL, 0x08082b0819080819ULL,
    0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b081919082bULL, 0x08082b082b082b08ULL,
    0x08082b1908081908ULL, 0x08082b1919080808ULL, 0x08082b2b0808082bULL, 0x08082b2b08191908ULL,
    0x0819080808080819ULL, 0x0819080808081908ULL, 0x0819080808190808ULL, 0x08190808082b0819ULL,
    0x0819080819080808ULL, 0x08190808192b0808ULL, 0x081908082b081908ULL, 0x081908082b190808ULL,
    0x081908082b191919ULL, 0x0819081908080808ULL, 0x0819081908082b08ULL, 0x08190819082b0808ULL,
    0x0819081919190808ULL, 0x0819081919192b2bULL, 0x081908192b080808ULL, 0x0819082b082b1908ULL,
    0x0819082b19081919ULL, 0x0819190808080808ULL, 0x0819190808082b08ULL, 0x08191908082b0808ULL,
    0x08191908082b1919ULL, 0x0819190819082b19ULL, 0x081919082b080808ULL, 0x0819191908192b08ULL,
    0x08191919192b082bULL, 0x0819192b08080808ULL, 0x0819192b0819192bULL, 0x08192b0808080819ULL,
    0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b0819080808ULL, 0x08192b082b080819ULL,
    0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b192b2b0808ULL, 0x08192b2b19190819ULL,
    0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808082b2bULL, 0x082b080819081908ULL,
    0x082b0808192b0819ULL, 0x082b08082b080808ULL, 0x082b08082b08082bULL, 0x082b0819082b2b19ULL,
    0x082b081919082b08ULL, 0x082b082b08080808ULL, 0x082b082b0808082bULL, 0x082b190808080819ULL,
    0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b190819080808ULL, 0x082b19081919192bULL,
    0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b1919192b1908ULL, 0x082b192b2b190808ULL,
    0x082b2b0808082b08ULL, 0x082b2b08082b0808ULL, 0x082b2b082b191908ULL, 0x082b2b2b19081908ULL,
    0x1908080808080819ULL, 0x1908080808081908ULL, 0x1908080808190808ULL, 0x1908080808192b08ULL,
    0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL, 0x1908080819082b08ULL,
    0x190808081919192bULL, 0x19080808192b0808ULL, 0x190808082b080819ULL, 0x190808082b081908ULL,
    0x190808082b190808ULL, 0x1908081908080808ULL, 0x19080819082b0808ULL, 0x19080819192b0819ULL,
    0x190808192b080808ULL, 0x190808192b081919ULL, 0x1908082b08080819ULL, 0x1908082b08190808ULL,
    0x1908082b19082b08ULL, 0x1908082b1919192bULL, 0x1908082b192b2b08ULL, 0x1908190808080808ULL,
    0x1908190808082b08ULL, 0x19081908082b0808ULL, 0x190819082b080808ULL, 0x190819082b192b19ULL,
    0x190819190819082bULL, 0x19081919082b1908ULL, 0x1908192b08080808ULL, 0x19082b0808080819ULL,
    0x19082b0808081908ULL, 0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL,
    0x19082b1908080808ULL, 0x19082b1919192b08ULL, 0x19082b19192b0819ULL, 0x19082b192b08082bULL,
    0x19082b2b19081919ULL, 0x19082b2b2b190808ULL, 0x1919080808080808ULL, 0x1919080808082b08ULL,
    0x1919080808190819ULL, 0x1919080808192b19ULL, 0x19190808082b0808ULL, 0x191908082b080808ULL,
    0x191908082b082b08ULL, 0x1919081908081908ULL, 0x191908191908082bULL, 0x191908192b2b1908ULL,
    0x1919082b2b190819ULL, 0x191919082b190808ULL, 0x191919082b19082bULL, 0x1919191908082b2bULL,
    0x1919192b08080819ULL, 0x1919192b19191908ULL, 0x19192b0808080808ULL, 0x19192b0808190819ULL,
    0x19192b0808192b19ULL, 0x19192b08192b1908ULL, 0x19192b1919080808ULL, 0x19192b2b08082b08ULL,
    0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b0808192b2b08ULL,
    0x192b081908080808ULL, 0x192b081919191919ULL, 0x192b082b08192b08ULL, 0x192b082b192b0808ULL,
    0x192b190808080808ULL, 0x192b190808081919ULL, 0x192b191908190808ULL, 0x192b19190819082bULL,
    0x192b19192b081908ULL, 0x192b2b081908082bULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL,
    0x2b08080808082b2bULL, 0x2b08080819080819ULL, 0x2b0808082b08082bULL, 0x2b08081908081908ULL,
    0x2b08081908192b08ULL, 0x2b08081919080808ULL, 0x2b08082b08190819ULL, 0x2b08190808080819ULL,
    0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL,
    0x2b081908192b0808ULL, 0x2b08191908080808ULL, 0x2b0819191908192bULL, 0x2b0819192b191908ULL,
    0x2b08192b08082b19ULL, 0x2b08192b19080808ULL, 0x2b08192b192b0808ULL, 0x2b082b080808082bULL,
    0x2b082b1908081908ULL, 0x2b082b2b08190819ULL, 0x2b19080808081908ULL, 0x2b19080808190808ULL,
    0x2b190808082b1908ULL, 0x2b19080819080808ULL, 0x2b1908082b2b0819ULL, 0x2b1908190819192bULL,
    0x2b1908192b080808ULL, 0x2b19082b19081919ULL, 0x2b19190808080808ULL, 0x2b191908082b082bULL,
    0x2b19190819081908ULL, 0x2b19191919190819ULL, 0x2b192b082b080819ULL, 0x2b192b19082b0808ULL,
    0x2b2b08080808082bULL, 0x2b2b080819190808ULL, 0x2b2b08082b081919ULL, 0x2b2b081908082b19ULL,
    0x2b2b082b08080808ULL, 0x2b2b190808192b08ULL, 0x2b2b2b0819190808ULL, 0x2b2b2b1908081908ULL,
};

struct block_iq2_xxs {
    half d;
    uint16_t qs[32];
};

struct block_q2_K {
    uint8_t scales[16];
    uint8_t qs[64];
    half d;
    half dmin;
};

kernel void gemv_iq2_xxs_swiglu_fused_kernel(
    device bfloat* out [[buffer(0)]],
    device const bfloat* vec [[buffer(1)]],
    device const block_iq2_xxs* w1 [[buffer(2)]],
    device const block_iq2_xxs* w3 [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant float& swiglu_limit [[buffer(6)]],
    uint tg_x [[threadgroup_position_in_grid]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    uint row = tg_x * 8 + simd_id;
    if (row >= (uint)N) return;

    int n_blocks = K / 256;
    device const block_iq2_xxs* row_w1 = w1 + row * n_blocks;
    device const block_iq2_xxs* row_w3 = w3 + row * n_blocks;

    int l = simd_lane / 8;
    int j = simd_lane % 8;
    uint8_t kmask = 1 << j;

    float sum1 = 0.0f;
    float sum3 = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        device const block_iq2_xxs& blk1 = row_w1[b];
        device const block_iq2_xxs& blk3 = row_w3[b];

        float d1 = float(blk1.d);
        float d3 = float(blk3.d);

        #pragma unroll
        for (int ib32 = 0; ib32 < 8; ib32++) {
            uint32_t aux0_1 = uint32_t(blk1.qs[ib32 * 4 + 0]) | (uint32_t(blk1.qs[ib32 * 4 + 1]) << 16);
            uint32_t aux1_1 = uint32_t(blk1.qs[ib32 * 4 + 2]) | (uint32_t(blk1.qs[ib32 * 4 + 3]) << 16);
            float db1 = d1 * (0.5f + float(aux1_1 >> 28)) * 0.25f;

            uint8_t g_idx1 = (aux0_1 >> (8 * l)) & 0xFF;
            uint8_t s_idx1 = (aux1_1 >> (7 * l)) & 0x7F;
            uint64_t g_val1 = c_iq2xxs_grid[g_idx1];
            uint8_t s_val1 = c_ksigns_iq2xs[s_idx1];
            uint8_t byte1 = (g_val1 >> (8 * j)) & 0xFF;
            float sign1 = (s_val1 & kmask) ? -1.0f : 1.0f;
            float weight1 = db1 * float(byte1) * sign1;

            uint32_t aux0_3 = uint32_t(blk3.qs[ib32 * 4 + 0]) | (uint32_t(blk3.qs[ib32 * 4 + 1]) << 16);
            uint32_t aux1_3 = uint32_t(blk3.qs[ib32 * 4 + 2]) | (uint32_t(blk3.qs[ib32 * 4 + 3]) << 16);
            float db3 = d3 * (0.5f + float(aux1_3 >> 28)) * 0.25f;

            uint8_t g_idx3 = (aux0_3 >> (8 * l)) & 0xFF;
            uint8_t s_idx3 = (aux1_3 >> (7 * l)) & 0x7F;
            uint64_t g_val3 = c_iq2xxs_grid[g_idx3];
            uint8_t s_val3 = c_ksigns_iq2xs[s_idx3];
            uint8_t byte3 = (g_val3 >> (8 * j)) & 0xFF;
            float sign3 = (s_val3 & kmask) ? -1.0f : 1.0f;
            float weight3 = db3 * float(byte3) * sign3;

            int col = b * 256 + (ib32 << 5) + simd_lane;
            float a = float(vec[col]);
            sum1 += weight1 * a;
            sum3 += weight3 * a;
        }
    }

    sum1 = simd_sum(sum1);
    sum3 = simd_sum(sum3);

    if (simd_lane == 0) {
        float g = sum1;
        float u = sum3;
        if (swiglu_limit > 0.0f) {
            g = min(g, swiglu_limit);
            u = min(max(u, -swiglu_limit), swiglu_limit);
        }
        float silu_g = g / (1.0f + exp(-g));
        out[row] = bfloat(silu_g * u);
    }
}

kernel void gemv_iq2_xxs_kernel(
    device bfloat* out [[buffer(0)]],
    device const bfloat* vec [[buffer(1)]],
    device const block_iq2_xxs* weight [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    uint tg_x [[threadgroup_position_in_grid]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    uint row = tg_x * 8 + simd_id;
    if (row >= (uint)N) return;

    int n_blocks = K / 256;
    device const block_iq2_xxs* row_w = weight + row * n_blocks;

    int l = simd_lane / 8;
    int j = simd_lane % 8;
    uint8_t kmask = 1 << j;

    float sum = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        device const block_iq2_xxs& blk = row_w[b];
        float d = float(blk.d);

        #pragma unroll
        for (int ib32 = 0; ib32 < 8; ib32++) {
            uint32_t aux0 = uint32_t(blk.qs[ib32 * 4 + 0]) | (uint32_t(blk.qs[ib32 * 4 + 1]) << 16);
            uint32_t aux1 = uint32_t(blk.qs[ib32 * 4 + 2]) | (uint32_t(blk.qs[ib32 * 4 + 3]) << 16);
            float db = d * (0.5f + float(aux1 >> 28)) * 0.25f;

            uint8_t g_idx = (aux0 >> (8 * l)) & 0xFF;
            uint8_t s_idx = (aux1 >> (7 * l)) & 0x7F;
            uint64_t g_val = c_iq2xxs_grid[g_idx];
            uint8_t s_val = c_ksigns_iq2xs[s_idx];
            uint8_t byte_val = (g_val >> (8 * j)) & 0xFF;
            float sign_val = (s_val & kmask) ? -1.0f : 1.0f;
            float weight_val = db * float(byte_val) * sign_val;

            int col = b * 256 + (ib32 << 5) + simd_lane;
            sum += weight_val * float(vec[col]);
        }
    }

    sum = simd_sum(sum);
    if (simd_lane == 0) {
        out[row] = bfloat(sum);
    }
}

kernel void gemv_q2_k_kernel(
    device bfloat* out [[buffer(0)]],
    device const bfloat* vec [[buffer(1)]],
    device const block_q2_K* weight [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    uint tg_x [[threadgroup_position_in_grid]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    uint row = tg_x * 8 + simd_id;
    if (row >= (uint)N) return;

    int n_blocks = K / 256;
    device const block_q2_K* row_w2 = weight + row * n_blocks;

    float sum = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        device const block_q2_K& blk = row_w2[b];
        float d = float(blk.d);
        float min = float(blk.dmin);

        #pragma unroll 4
        for (int iter = 0; iter < 8; iter++) {
            int idx = (iter << 5) + simd_lane; // 0..255
            int group = idx >> 4;
            int l = idx & 15;
            int q_base = ((group >> 3) << 5) + ((group & 1) << 4);
            int shift = ((group >> 1) & 3) << 1;
            uint8_t q = (blk.qs[q_base + l] >> shift) & 0x03;
            uint8_t sc = blk.scales[group];
            float dl = d * float(sc & 0x0F);
            float ml = min * float(sc >> 4);
            float w = dl * float(q) - ml;

            int col_idx = (b << 8) + idx;
            float a = float(vec[col_idx]);
            sum += w * a;
        }
    }

    sum = simd_sum(sum);
    if (simd_lane == 0) {
        out[row] = bfloat(sum);
    }
}

kernel void fused_moe_accum_dynamic_kernel(
    device bfloat* accum [[buffer(0)]],
    device const bfloat* down_buf [[buffer(1)]],
    device const float* topk_weights [[buffer(2)]],
    device const bfloat* shared_down [[buffer(3)]],
    constant int& dim [[buffer(4)]],
    constant int& has_shared [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= (uint)dim) return;

    float sum = (has_shared != 0) ? float(shared_down[idx]) : 0.0f;
    #pragma unroll
    for (int k = 0; k < 6; k++) {
        float w = topk_weights[k];
        sum += float(down_buf[k * dim + idx]) * w;
    }
    accum[idx] = bfloat(sum);
}

kernel void moe_route_top6_kernel(
    device int32_t* topk_ids [[buffer(0)]],
    device float* topk_weights [[buffer(1)]],
    device const bfloat* scores_bf16 [[buffer(2)]],
    device const float* gate_bias [[buffer(3)]],
    constant int& n_experts [[buffer(4)]],
    constant int& top_k [[buffer(5)]],
    constant float& routed_scaling_factor [[buffer(6)]],
    constant int& has_bias [[buffer(7)]],
    uint m [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float tg_scores[512];
    threadgroup float tg_probs[512];

    if (tid < (uint)n_experts && tid < 512) {
        float raw = float(scores_bf16[m * n_experts + tid]);
        float sp = (raw > 20.0f) ? raw : ((raw < -20.0f) ? exp(raw) : log(1.0f + exp(raw)));
        float prob = sqrt(sp);
        tg_probs[tid] = prob;
        float b = (has_bias != 0) ? gate_bias[tid] : 0.0f;
        tg_scores[tid] = prob + b;
    } else if (tid < 512) {
        tg_scores[tid] = -1e30f;
        tg_probs[tid] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        int best_idx[8];
        float best_prob[8];
        float sum_p = 0.0f;
        int k_max = (top_k <= 8) ? top_k : 8;

        for (int k = 0; k < k_max; k++) {
            float max_s = -1e30f;
            int max_i = 0;
            for (int i = 0; i < n_experts; i++) {
                float s = tg_scores[i];
                if (s > max_s) {
                    max_s = s;
                    max_i = i;
                }
            }
            best_idx[k] = max_i;
            best_prob[k] = tg_probs[max_i];
            sum_p += best_prob[k];
            tg_scores[max_i] = -1e30f;
        }

        if (sum_p < 1e-6f) sum_p = 1e-6f;

        for (int k = 0; k < k_max; k++) {
            topk_ids[m * top_k + k] = best_idx[k];
            topk_weights[m * top_k + k] = (best_prob[k] / sum_p) * routed_scaling_factor;
        }
    }
}

// ── DeepSeek V4 MTP Markov Head Predict Kernels ──────────────────────────────

kernel void markov_head_block_kernel(
    device float* block_max_vals [[buffer(0)]],
    device int32_t* block_max_indices [[buffer(1)]],
    device const bfloat* w1 [[buffer(2)]],
    device const bfloat* w2 [[buffer(3)]],
    constant int32_t& input_token [[buffer(4)]],
    constant int32_t& vocab_size [[buffer(5)]],
    constant int32_t& hidden_dim [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint num_blocks [[threadgroups_per_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float4 s_u4[64];
    threadgroup float s_max[8];
    threadgroup int32_t s_idx[8];

    // Load 256-element input embedding (64 bfloat4 vectors) into shared memory
    if (tid < 64) {
        if (input_token >= 0 && input_token < vocab_size) {
            device const bfloat4* u_b4 = (device const bfloat4*)(w1 + (size_t)input_token * 256);
            s_u4[tid] = float4(u_b4[tid]);
        } else {
            s_u4[tid] = float4(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int total_simdgroups = num_blocks * 8;
    int global_simd_id = bid * 8 + simd_id;

    float best_val = -1e30f;
    int32_t best_idx = -1;

    for (int y = global_simd_id; y < vocab_size; y += total_simdgroups) {
        if (y == input_token) continue;
        device const bfloat4* row2_b4 = (device const bfloat4*)(w2 + (size_t)y * 256);
        float acc = dot(float4(row2_b4[simd_lane]), s_u4[simd_lane]) +
                    dot(float4(row2_b4[simd_lane + 32]), s_u4[simd_lane + 32]);
        float row_dot = simd_sum(acc);

        if (simd_lane == 0) {
            if (row_dot > best_val) {
                best_val = row_dot;
                best_idx = y;
            }
        }
    }

    if (simd_lane == 0) {
        s_max[simd_id] = best_val;
        s_idx[simd_id] = best_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float m = (simd_lane < 8) ? s_max[simd_lane] : -1e30f;
        int32_t idx = (simd_lane < 8) ? s_idx[simd_lane] : -1;
        for (int offset = 4; offset > 0; offset >>= 1) {
            float om = simd_shuffle_down(m, offset);
            int32_t oi = simd_shuffle_down(idx, offset);
            if (om > m) {
                m = om;
                idx = oi;
            }
        }
        if (simd_lane == 0) {
            block_max_vals[bid] = m;
            block_max_indices[bid] = idx;
        }
    }
}

kernel void markov_head_reduce_kernel(
    device int32_t* d_out_pred [[buffer(0)]],
    device const float* block_max_vals [[buffer(1)]],
    device const int32_t* block_max_indices [[buffer(2)]],
    constant int32_t& num_blocks [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float s_max[8];
    threadgroup int32_t s_idx[8];

    float local_max = -1e30f;
    int32_t local_idx = -1;

    for (int i = tid; i < num_blocks; i += 256) {
        float v = block_max_vals[i];
        if (v > local_max) {
            local_max = v;
            local_idx = block_max_indices[i];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        float om = simd_shuffle_down(local_max, offset);
        int32_t oi = simd_shuffle_down(local_idx, offset);
        if (om > local_max) {
            local_max = om;
            local_idx = oi;
        }
    }

    if (simd_lane == 0) {
        s_max[simd_id] = local_max;
        s_idx[simd_id] = local_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float m = (simd_lane < 8) ? s_max[simd_lane] : -1e30f;
        int32_t idx = (simd_lane < 8) ? s_idx[simd_lane] : -1;
        for (int offset = 4; offset > 0; offset >>= 1) {
            float om = simd_shuffle_down(m, offset);
            int32_t oi = simd_shuffle_down(idx, offset);
            if (om > m) {
                m = om;
                idx = oi;
            }
        }
        if (simd_lane == 0) {
            d_out_pred[0] = idx;
        }
    }
}


