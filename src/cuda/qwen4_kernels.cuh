#pragma once

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

// GGUF tensor type codes used by Qwen3.8-Flash-Next (qwen4exp)
enum Qwen4GgmlType : int32_t {
    QWEN4_GGML_TYPE_F32     = 0,
    QWEN4_GGML_TYPE_F16     = 1,
    QWEN4_GGML_TYPE_Q4_0    = 2,
    QWEN4_GGML_TYPE_Q4_1    = 3,
    QWEN4_GGML_TYPE_Q5_0    = 6,
    QWEN4_GGML_TYPE_Q5_1    = 7,
    QWEN4_GGML_TYPE_Q8_0    = 8,
    QWEN4_GGML_TYPE_Q2_K    = 10,
    QWEN4_GGML_TYPE_Q3_K    = 11,
    QWEN4_GGML_TYPE_Q4_K    = 12,
    QWEN4_GGML_TYPE_Q5_K    = 13,
    QWEN4_GGML_TYPE_Q6_K    = 14,
    QWEN4_GGML_TYPE_IQ2_XXS = 16,
    QWEN4_GGML_TYPE_IQ2_XS  = 17,
    QWEN4_GGML_TYPE_IQ3_XXS = 18,
    QWEN4_GGML_TYPE_IQ1_S   = 19,
    QWEN4_GGML_TYPE_IQ4_NL  = 20,
    QWEN4_GGML_TYPE_IQ3_S   = 21,
    QWEN4_GGML_TYPE_IQ2_S   = 22,
    QWEN4_GGML_TYPE_IQ4_XS  = 23,
    QWEN4_GGML_TYPE_IQ1_M   = 29,
    QWEN4_GGML_TYPE_BF16    = 30,
    QWEN4_GGML_TYPE_Q1_0    = 40,
    QWEN4_GGML_TYPE_Q2_0    = 42,
};

// Host helpers for GGUF row byte sizing and type names
size_t qwen4_ggml_row_bytes(int32_t ggml_type, int64_t n_cols);
const char* qwen4_ggml_type_name(int32_t ggml_type);
size_t qwen4_q8_1_bytes(int64_t n_cols);

// Quantize FP32 activation vector x[n_cols] -> block_q8_1[n_cols / 32]
void qwen4_quantize_q8_1_cuda(const float* x, void* vx_q8_1, int n_cols, cudaStream_t stream = 0);

// Quantized matrix-vector multiply y[n_rows] = W[n_rows, n_cols] @ vx_q8_1[n_cols / 32]
void qwen4_mmvq_cuda(
    int32_t ggml_type,
    const void* W,
    const void* vx_q8_1,
    float* y,
    int n_rows,
    int n_cols,
    cudaStream_t stream = 0);

// BF16 matrix-vector multiply y[n_rows] = W_bf16[n_rows, n_cols] @ bf16_round(x[n_cols])
void qwen4_bf16_gemv_cuda(
    const uint16_t* W_bf16,
    const float* x,
    float* y,
    int n_rows,
    int n_cols,
    cudaStream_t stream = 0);

// Unified projection dispatcher (handles BF16, F16, F32, and all Q/K/IQ quantized types)
void qwen4_project_cuda(
    int32_t ggml_type,
    const void* W,
    const float* x_f32,
    const void* x_q8_1,
    float* y,
    int n_rows,
    int n_cols,
    cudaStream_t stream = 0);

// Token embedding lookup + broadcast into 4-stream hyper-connection residual R[4, 2560]
void qwen4_embed_to_hc_cuda(
    int32_t embd_type,
    const void* token_embd,
    int token_id,
    float* d_R,
    int n_embd,
    int n_streams,
    cudaStream_t stream = 0);

// Broadcast an existing FP32 embedding vector d_embd[n_embd] into R[n_streams, n_embd]
void qwen4_vec_to_hc_cuda(
    const float* d_embd,
    float* d_R,
    int n_embd,
    int n_streams,
    cudaStream_t stream = 0);

// 4-stream Gated Residual Read:
//   normed = rms_norm_per_stream(R, norm_w, eps)
//   h = down_bf16[320, 10240] @ bf16(normed)
//   delta = up_bf16[10240, 320] @ bf16(h)
//   x_out[2560] = mean_{s=0..3}(R[s] + delta[s])
// tmp_buf must hold at least (10240 + 320) floats.
void qwen4_gr_read_cuda(
    const float* R,
    const float* norm_w,
    const uint16_t* down_bf16,
    const uint16_t* up_bf16,
    float* x_out,
    float* tmp_buf,
    float eps,
    cudaStream_t stream = 0);

// 4-stream Gated Residual Write:
//   normed = rms_norm_per_stream(R, norm_w, eps)
//   gates[4] = inject_bf16[4, 10240] @ bf16(normed)
//   R[s, j] += gates[s] * sub_out[j]
// tmp_buf must hold at least (10240 + 4) floats.
void qwen4_gr_write_cuda(
    float* R,
    const float* norm_w,
    const uint16_t* inject_bf16,
    const float* sub_out,
    float* tmp_buf,
    float eps,
    cudaStream_t stream = 0);

// Dequantize 16 IQ4_NL N-gram head rows (16 * 90 = 1440 bytes) -> dst[2560] floats
void qwen4_ple_iq4nl_dequant_cuda(
    const void* src16_iq4nl,
    float* dst_f32,
    cudaStream_t stream = 0);

// Layer 1 PLE N-gram gated conv block (modifies R[10240] in-place before blk.1.hc_attn_*):
//   conv_state must hold [9 * 10240] floats.
//   scratch must hold at least (10240 + 2560 + 10240 + 10240) = 33280 floats.
//   q8_scratch must hold at least qwen4_q8_1_bytes(2560) = 2880 bytes.
void qwen4_ple_forward_cuda(
    float* R,
    const float* ple_norm_q,
    const float* ple_norm_k,
    const float* ple_norm_c,
    const void* ple_key,
    int32_t ple_key_type,
    const void* ple_val,
    int32_t ple_val_type,
    const uint16_t* ple_conv_f16,
    const float* emb_flat,
    float* conv_state,
    float* scratch,
    void* q8_scratch,
    float eps,
    cudaStream_t stream = 0);

// 36 Gated DeltaNet (GDN) layers (l % 4 != 3):
//   conv_state: [3 * 10240] floats
//   ssm_state:  [48 * 128 * 128] floats
//   scratch:    at least (10240 + 10240 + 6144 + 6144 + 48 * 4) = 32960 floats
//   q8_scratch: at least qwen4_q8_1_bytes(6144) = 6912 bytes
void qwen4_gdn_decode_cuda(
    const float* x_f32,
    const void* x_q8_1,
    const void* w_qkv, int32_t qkv_type,
    const void* w_gate, int32_t gate_type,
    const uint16_t* w_alpha_bf16,
    const uint16_t* w_beta_bf16,
    const float* w_conv_f32,
    const float* ssm_a_f32,
    const float* dt_bias_f32,
    const float* ssm_norm_f32,
    const void* w_out, int32_t out_type,
    float* conv_state,
    float* ssm_state,
    float* y_out,
    float* scratch,
    void* q8_scratch,
    float eps,
    cudaStream_t stream = 0);

// 12 Query-Sparse Attention (QSA) layers (l % 4 == 3):
//   k_cache, v_cache: [max_seq_len, 512] BF16 (uint16_t)
//   idx_pooled:       [max_seq_len / 4, 128] floats
//   idx_tail:         [4, 128] floats
//   idx_dead:         [128] floats
//   scratch:          at least (12288 + 6144 + 6144 + 512 + 512 + 6144 + 512 + 128 + max_seq_len/4) floats
//   d_sel_indices:    [2056] int32_t
//   q8_scratch:       at least qwen4_q8_1_bytes(6144) = 6912 bytes
void qwen4_qsa_decode_cuda(
    const float* x_f32,
    const void* x_q8_1,
    const void* w_q, int32_t q_type,
    const void* w_k, int32_t k_type,
    const void* w_v, int32_t v_type,
    const float* q_norm_f32,
    const float* k_norm_f32,
    const void* w_out, int32_t out_type,
    const uint16_t* w_idx_k_bf16,
    const uint16_t* w_idx_q_bf16,
    const float* idx_k_norm_f32,
    const float* idx_q_norm_f32,
    uint16_t* k_cache,
    uint16_t* v_cache,
    float* idx_pooled,
    float* idx_tail,
    float* idx_dead,
    int pos,
    int max_seq_len,
    float* y_out,
    float* scratch,
    int32_t* d_sel_indices,
    void* q8_scratch,
    float eps,
    cudaStream_t stream = 0);

// MoE Top-10 of 256 Router:
//   Computes 256 BF16 router logits, full-256 softmax, selects top-10, and renormalizes with 2^-14 floor.
//   scratch_logits must hold 256 floats.
void qwen4_router_top10_cuda(
    const uint16_t* gate_inp_bf16,
    const float* x_f32,
    float* scratch_logits,
    int32_t* d_top_idx,
    float* d_top_w,
    cudaStream_t stream = 0);

// Shared expert (SwiGLU + sigmoid scalar gate):
//   Computes y_moe[2560] = sigmoid(dot(bf16(x_f32), gate_inp_shexp_bf16)) * down_shexp(silu(gate_shexp(x)) * up_shexp(x))
//   scratch must hold at least (640 + 640 + 2560) = 3840 floats.
//   q8_scratch must hold at least qwen4_q8_1_bytes(640) = 720 bytes.
void qwen4_shared_expert_cuda(
    const float* x_f32,
    const void* x_q8_1,
    const void* w_gate_shexp, int32_t gate_shexp_type,
    const void* w_up_shexp, int32_t up_shexp_type,
    const void* w_down_shexp, int32_t down_shexp_type,
    const uint16_t* w_gate_inp_shexp_bf16,
    float* y_moe,
    float* scratch,
    void* q8_scratch,
    cudaStream_t stream = 0);

// Device pointer bundle for 10 active routed experts (used when experts are resolved via tiered cache)
struct Qwen4ActiveExpertPtrs {
    const void* gate[10];
    const void* up[10];
    const void* down[10];
    int32_t     slot[10];
};

// Batched range [start_k, start_k + count_k) of routed experts SwiGLU + optional weighted accumulation into y_moe[2560]:
//   Enables overlapping L1-Hit expert compute [0, n_hits) on main_stream with L2->L1 DMA of [n_hits, 10) on copy_stream.
void qwen4_routed_experts_range_cuda(
    const void* x_q8_1,
    const void* base_gate, size_t gate_stride_bytes, int32_t gate_type,
    const void* base_up, size_t up_stride_bytes, int32_t up_type,
    const void* base_down, size_t down_stride_bytes, int32_t down_type,
    const Qwen4ActiveExpertPtrs* d_ptrs,
    const int32_t* d_top_idx,
    const float* d_top_w,
    int start_k,
    int count_k,
    bool do_accum,
    float* y_moe,
    float* scratch,
    void* q8_scratch,
    cudaStream_t stream = 0,
    const float* d_cpu_moe_out = nullptr);

// Batched 10-expert SwiGLU + weighted accumulation into y_moe[2560]:
//   Supports EITHER contiguous resident expert tables (base_gate/up/down != nullptr, zero CPU sync!)
//   OR resolved per-expert device pointers (d_ptrs != nullptr, uploaded from tiered cache).
//   scratch must hold at least (10 * 640 + 10 * 640 + 10 * 2560) = 38400 floats.
//   q8_scratch must hold at least 10 * qwen4_q8_1_bytes(640) = 7200 bytes.
void qwen4_routed_experts_batch10_cuda(
    const void* x_q8_1,
    const void* base_gate, size_t gate_stride_bytes, int32_t gate_type,
    const void* base_up, size_t up_stride_bytes, int32_t up_type,
    const void* base_down, size_t down_stride_bytes, int32_t down_type,
    const Qwen4ActiveExpertPtrs* d_ptrs,
    const int32_t* d_top_idx,
    const float* d_top_w,
    float* y_moe,
    float* scratch,
    void* q8_scratch,
    cudaStream_t stream = 0);

