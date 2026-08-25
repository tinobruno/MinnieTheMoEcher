#pragma once
// activations.cuh — CUDA kernel declarations for moecher

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// ── RMSNorm ────────────────────────────────────────────────────────────────────
// out[i] = weight[i] * x[i] * rsqrt(mean(x^2) + eps)
void rms_norm_cuda(
    __nv_bfloat16* out,            // [dim]
    const __nv_bfloat16* x,        // [dim]
    const __nv_bfloat16* weight,   // [dim]
    int dim,
    float eps,
    cudaStream_t stream = 0);

// Batched: process multiple rows
void rms_norm_cuda_batched(
    __nv_bfloat16* out,            // [n, dim]
    const __nv_bfloat16* x,        // [n, dim]
    const __nv_bfloat16* weight,   // [dim]
    int n, int dim,
    float eps,
    cudaStream_t stream = 0);

void rms_norm_f32_cuda(float* x, int dim, float eps, cudaStream_t stream = 0);

void rms_norm_weighted_f32_cuda(
    float* out,
    const float* x,
    const __nv_bfloat16* weight,
    int dim, float eps,
    cudaStream_t stream = 0);

// Unweighted batched: x * rsqrt(mean(x^2) + eps) per row (no learnable weight)
// Used for per-head Q normalization (DeepseekV4UnweightedRMSNorm)
void rms_norm_unweighted_batched_cuda(
    __nv_bfloat16* out,            // [n, dim]
    const __nv_bfloat16* x,        // [n, dim]
    int n, int dim,
    float eps,
    cudaStream_t stream = 0);

// ── SiLU * mul (SwiGLU) ────────────────────────────────────────────────────────
// out[i] = silu(gate[i]) * up[i], with optional clamping
void silu_mul_cuda(
    __nv_bfloat16* out,            // [n]
    const __nv_bfloat16* gate,     // [n]
    const __nv_bfloat16* up,       // [n]
    int n,
    float swiglu_limit,            // 0 = no clamp
    cudaStream_t stream = 0);

// ── RoPE (Rotary Position Embeddings) ──────────────────────────────────────────
// Apply RoPE to the last rope_dim elements of each head vector
// x shape: [n_heads, head_dim], operates on x[..., head_dim-rope_dim:]
void rope_cuda(
    __nv_bfloat16* x,              // [n_heads, head_dim] or [head_dim] for KV
    int n_vectors,                 // number of vectors (n_heads for Q, 1 for KV)
    int head_dim,
    int rope_dim,
    int position,                  // absolute position in sequence
    const float* freq_table,       // precomputed freqs [max_seq_len, rope_dim/2]
    bool inverse,                  // true for de-rotation on output
    cudaStream_t stream = 0);

// ── FP8 E4M3 dequantization ───────────────────────────────────────────────────
// Dequantize FP8 E4M3 weights to BF16 using E8M0 block scales
// weight: [rows, cols] in F8_E4M3
// scale:  [ceil(rows/block_size), ceil(cols/block_size)] in F8_E8M0
// out:    [rows, cols] in BF16
void fp8_dequant_cuda(
    __nv_bfloat16* out,
    const uint8_t* weight,         // F8_E4M3 raw bytes
    const uint8_t* scale,          // F8_E8M0 raw bytes
    int rows, int cols,
    int block_size,                // 128 for this model
    cudaStream_t stream = 0);

void gemv_fp8_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* vec,
    const uint8_t* weight,
    const uint8_t* scale,
    int rows, int cols,
    int block_size,
    cudaStream_t stream = 0);

void gemv_fp8_grouped_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* vec,
    const uint8_t* weight,
    const uint8_t* scale,
    int rows, int cols,
    int groups, int block_size,
    cudaStream_t stream = 0);

void gemv_hc_pre_norm_cuda(
    float* mixes,
    const __nv_bfloat16* hc_state,
    const float* hc_fn,
    int mix_size, int hc_dim,
    float eps,
    cudaStream_t stream = 0);


// ── FP4 dequantization ────────────────────────────────────────────────────────
// Dequantize FP4 (packed as I8, 2 values per byte) to BF16 using E8M0 scales
// weight: [rows, cols_packed] in I8 (cols_packed = logical_cols / 2)
// scale:  [rows, logical_cols/32] in F8_E8M0  (one scale per 32 fp4 values)
// out:    [rows, logical_cols] in BF16
void fp4_dequant_cuda(
    __nv_bfloat16* out,
    const uint8_t* weight,         // packed FP4 bytes
    const uint8_t* scale,          // E8M0 scales
    int rows, int cols_packed,     // cols_packed = logical_cols/2
    int scale_cols,                // number of scale columns
    cudaStream_t stream = 0);

// ── INT2 asymmetric dequantization ────────────────────────────────────────────
// weight: [rows, cols_packed] in I8 (cols_packed = logical_cols / 4)
// scale_min: [rows, logical_cols/64, 2] in BF16 (scale then min contiguous per block)
// out:    [rows, logical_cols] in BF16
void int2_dequant_cuda(
    __nv_bfloat16* out,
    const uint8_t* weight,
    const __nv_bfloat16* scale_min,
    int rows, int cols_packed,
    int block_size,
    cudaStream_t stream = 0);

void gemv_int2_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* vec,
    const uint8_t* weight,
    const __nv_bfloat16* scale_min,
    int rows, int cols_packed,
    int block_size,
    cudaStream_t stream = 0);
void gemm_int2_cuda(__nv_bfloat16* out, const __nv_bfloat16* A,
                    const uint8_t* weight, const __nv_bfloat16* scale_min,
                    int M, int N, int K_packed, int block_size,
                    cudaStream_t stream = 0);

// ── IQ2_XXS quantization ──────────────────────────────────────────────────────
#define QK_IQ2_XXS 256

struct __align__(2) block_iq2_xxs {
    half d;              // 16-bit scale
    uint16_t qs[32];     // 32 x 16-bit = 64 bytes (256 weights)
};
static_assert(sizeof(block_iq2_xxs) == 66, "wrong iq2_xxs block size");

void iq2_xxs_dequant_cuda(
    __nv_bfloat16* out,
    const block_iq2_xxs* weight,
    int rows, int cols,
    cudaStream_t stream = 0);

void gemv_iq2_xxs_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* vec,
    const block_iq2_xxs* weight,
    int rows, int cols,
    cudaStream_t stream = 0);

// ── Q2_K quantization ─────────────────────────────────────────────────────────
#define QK_K 256

struct __align__(4) block_q2_K {
    uint8_t scales[16];  // 16 x 8-bit scales (scale and min for 16 sub-blocks of 16 weights)
    uint8_t qs[64];      // 64 x 8-bit = 256 2-bit quants
    half d;              // super-block scale
    half dmin;           // super-block min
};
static_assert(sizeof(block_q2_K) == 84, "wrong q2_K block size");

void q2_k_dequant_cuda(
    __nv_bfloat16* out,
    const block_q2_K* weight,
    int rows, int cols,
    cudaStream_t stream = 0);

void gemv_q2_k_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* vec,
    const block_q2_K* weight,
    int rows, int cols,
    cudaStream_t stream = 0);

// ── Embedding lookup ───────────────────────────────────────────────────────────
void embedding_cuda(
    __nv_bfloat16* out,            // [seq_len, dim]
    const __nv_bfloat16* table,    // [vocab_size, dim]
    const int32_t* ids,            // [seq_len]
    int seq_len, int dim,
    cudaStream_t stream = 0);

void embedding_broadcast_cuda(
    __nv_bfloat16* hidden,         // [dim]
    __nv_bfloat16* hc_state,       // [hc, dim]
    const __nv_bfloat16* table,    // [vocab_size, dim]
    int token_id, int dim, int hc,
    cudaStream_t stream = 0);

// ── Softmax ────────────────────────────────────────────────────────────────────
// Row-wise softmax: out[row][col] = exp(x[row][col]) / sum_col(exp(x[row][col]))
void softmax_cuda(
    float* out,                    // [rows, cols]
    const float* x,                // [rows, cols]
    int rows, int cols,
    cudaStream_t stream = 0);

// ── Residual add ───────────────────────────────────────────────────────────────
void add_cuda(
    __nv_bfloat16* out,            // [n]
    const __nv_bfloat16* a,        // [n]
    const __nv_bfloat16* b,        // [n]
    int n,
    cudaStream_t stream = 0);

void add_f32_sigmoid_cuda(
    float* out,
    const float* a,
    const float* bias,
    int n,
    bool apply_sigmoid,
    cudaStream_t stream = 0);

// ── Weighted add (for expert combination) ──────────────────────────────────────
// out += weight * x
void weighted_add_cuda(
    __nv_bfloat16* out,            // [dim]
    const __nv_bfloat16* x,        // [dim]
    float weight,
    int dim,
    cudaStream_t stream = 0);

// ── Top-k selection ────────────────────────────────────────────────────────────
// Find top-k indices and values from scores
void topk_cuda(
    float* out_vals,               // [k]
    int32_t* out_idx,              // [k]
    const float* scores,           // [n]
    int n, int k,
    cudaStream_t stream = 0);

// ── sqrt(softplus(x)) scoring ──────────────────────────────────────────────────
void sqrtsoftplus_cuda(
    float* out,                    // [n]
    const float* x,                // [n]
    int n,
    cudaStream_t stream = 0);

// ── HC Sinkhorn split ──────────────────────────────────────────────────────────
// Implements the Hyper-Connection pre/post mixing
// mixes: [(2+hc)*hc] raw values from the linear projection
// Returns pre[hc], post[hc], comb[hc, hc] after sigmoid + Sinkhorn
void hc_split_sinkhorn_cuda(
    float* pre,                    // [hc_mult]
    float* post,                   // [hc_mult]
    float* comb,                   // [hc_mult * hc_mult]
    const float* mixes,            // [(2+hc_mult)*hc_mult]
    const float* scale,            // [3]
    const float* base,             // [(2+hc_mult)*hc_mult]
    int hc_mult,
    int sinkhorn_iters,
    float eps,
    cudaStream_t stream = 0);

// ── Precompute RoPE frequency table ────────────────────────────────────────────
void precompute_freqs_cuda(
    float* freqs,                  // [max_seq_len, rope_dim/2]  stores cos,sin interleaved
    int max_seq_len,
    int rope_dim,
    float base,
    float factor,
    int original_seq_len,
    int beta_fast,
    int beta_slow,
    cudaStream_t stream = 0);

// ── BF16 <-> F32 conversions ───────────────────────────────────────────────────
void bf16_to_f32_cuda(float* out, const __nv_bfloat16* in, int n, cudaStream_t stream = 0);
void f32_to_bf16_cuda(__nv_bfloat16* out, const float* in, int n, cudaStream_t stream = 0);

// ── Custom MLA Attention Kernel ─────────────────────────────────────────────
void mla_attention_cuda(
    const __nv_bfloat16* q,
    const __nv_bfloat16* kv,
    const float* attn_sink,
    __nv_bfloat16* out,
    int n_heads,
    int cache_len,
    int head_dim,
    float scale,
    cudaStream_t stream = 0);

// ── Compressor Kernels ─────────────────────────────────────────────────────

// BF16 matrix-vector product: out[N] = W[N,K] @ x[K] (row-major W)
// Used for compressor projections (wkv, wgate) which are BF16, not FP8
void gemv_bf16_cuda(
    float* out,                    // [N] F32 output (for accumulation precision)
    const __nv_bfloat16* W,        // [N, K] BF16 weight matrix (row-major)
    const __nv_bfloat16* x,        // [K] BF16 input vector
    int N, int K,
    cudaStream_t stream = 0);

// Softmax-gated weighted sum for KV compression (non-overlapping)
// Computes: out[d] = sum_i( kv[i][d] * softmax(score[i][d]) ) for d in [0, dim)
// kv:    [window, dim] F32 — accumulated KV entries in the compression window
// score: [window, dim] F32 — gate scores (with APE already added)
// out:   [dim] F32 — compressed KV entry
void compressor_pool_cuda(
    float* out,                    // [dim]
    const float* kv,               // [window, dim]
    const float* score,            // [window, dim]
    int window,                    // compression window size (ratio or 2*ratio for overlap)
    int dim,                       // output dimension (head_dim for non-overlap, coff*head_dim)
    cudaStream_t stream = 0);

// Combine raw sliding-window KV and compressed KV into a contiguous buffer
// for attention. Copies raw_kv[raw_len, head_dim] then comp_kv[comp_len, head_dim]
void combine_kv_cuda(
    __nv_bfloat16* out,            // [raw_len + comp_len, head_dim]
    const __nv_bfloat16* raw_kv,   // [raw_len, head_dim]
    int raw_len,
    const __nv_bfloat16* comp_kv,  // [comp_len, head_dim]
    int comp_len,
    int head_dim,
    cudaStream_t stream = 0);


void gemv_bf16_out_bf16_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* W,
    const __nv_bfloat16* x,
    int N, int K,
    cudaStream_t stream);

void hc_pre_weighted_add_cuda(
    __nv_bfloat16* hidden, const __nv_bfloat16* hc_state, const float* pre_weights,
    int dim, int hc, cudaStream_t stream = 0);

void hc_post_update_cuda(
    __nv_bfloat16* hc_state, const __nv_bfloat16* hidden, const __nv_bfloat16* hc_residual,
    const float* post_weights, const float* comb_weights,
    int dim, int hc, cudaStream_t stream = 0);

void hc_head_reduce_cuda(
    __nv_bfloat16* hidden, const __nv_bfloat16* hc_state,
    const float* mixes, const float* scale, const float* base,
    int dim, int hc, cudaStream_t stream = 0);

void gemv_iq2_xxs_swiglu_fused_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* vec,
    const block_iq2_xxs* w1,
    const block_iq2_xxs* w3,
    int N, int K, float swiglu_limit,
    cudaStream_t stream = 0);

void populate_active_expert_ptrs_cuda(
    const void** active_ptrs,
    const int32_t* topk_ids,
    const void* const* flat_expert_ptrs,
    int layer_id, int n_experts, int top_k,
    cudaStream_t stream = 0);

void gemv_iq2_xxs_moe_swiglu_fused_cuda(
    __nv_bfloat16* gate_buf,
    const __nv_bfloat16* vec,
    const void* const* active_expert_ptrs,
    int w1_offset, int w3_offset,
    int N, int K, float swiglu_limit,
    cudaStream_t stream = 0);

void gemv_q2_k_moe_cuda(
    __nv_bfloat16* down_buf,
    const __nv_bfloat16* gate_buf,
    const void* const* active_expert_ptrs,
    int w2_offset,
    int N, int K,
    cudaStream_t stream = 0);

void moe_route_top6_cuda(
    int32_t* topk_ids,
    float* topk_weights,
    const float* scores_f32,
    int n_experts, int top_k, float routed_scaling_factor,
    cudaStream_t stream = 0);

void moe_route_hash_cuda(
    int32_t* topk_ids,
    float* topk_weights,
    const int64_t* tid2eid_table,
    int token_id, int top_k, float routed_scaling_factor,
    cudaStream_t stream = 0);

void moe_route_hash_device_id_cuda(
    int32_t* topk_ids,
    float* topk_weights,
    const int64_t* tid2eid_table,
    const int32_t* d_token_id, int top_k, float routed_scaling_factor,
    cudaStream_t stream = 0);

void moe_route_top6_from_bf16_cuda(
    int32_t* topk_ids,
    float* topk_weights,
    const __nv_bfloat16* scores_bf16,
    const float* gate_bias,
    int n_experts, int top_k, float routed_scaling_factor,
    cudaStream_t stream = 0);

void gemv_f32_cuda(
    float* out,
    const float* vec,
    const float* matrix,
    int M, int K,
    cudaStream_t stream = 0);

void fused_moe_accum_dynamic_cuda(
    __nv_bfloat16* accum,
    const __nv_bfloat16* down_buf,
    const float* topk_weights,
    const __nv_bfloat16* shared_down,
    int dim,
    cudaStream_t stream = 0);

void fused_moe_accum_6_cuda(
    __nv_bfloat16* accum,
    const __nv_bfloat16* down_ptrs,
    float w0, float w1, float w2, float w3, float w4, float w5,
    int dim, cudaStream_t stream = 0);

// Device-driven kernels for CUDA Graph capture
void embedding_broadcast_device_id_cuda(
    __nv_bfloat16* hidden, __nv_bfloat16* hc_state,
    const __nv_bfloat16* table, const int32_t* d_token_id, int dim, int hc,
    cudaStream_t stream = 0);

void rope_device_pos_cuda(
    __nv_bfloat16* x, int n_vectors, int head_dim, int rope_dim,
    const int32_t* d_position, const float* freq_table, bool inverse,
    cudaStream_t stream = 0);

void store_kv_device_pos_cuda(
    __nv_bfloat16* kv_cache, const __nv_bfloat16* kv_val,
    const int32_t* d_position, int window, int head_dim,
    cudaStream_t stream = 0);

void prepare_combined_kv_cuda(
    __nv_bfloat16* combined_kv,
    int32_t* d_cache_len,
    const __nv_bfloat16* raw_kv_cache,
    const __nv_bfloat16* comp_kv_cache,
    const int32_t* d_position,
    const int32_t* d_comp_count,
    int window, int head_dim, int ratio,
    cudaStream_t stream = 0);

void mla_attention_device_len_cuda(
    const __nv_bfloat16* q,
    const __nv_bfloat16* kv,
    const float* attn_sink,
    __nv_bfloat16* out,
    const int32_t* d_cache_len,
    int max_cache_len,
    int head_dim,
    float scale,
    cudaStream_t stream = 0);

void accumulate_expert_imatrix_cuda(
    float* gate_accum,             // [n_experts, hidden_dim]
    float* down_accum,             // [n_experts, moe_intermediate]
    uint32_t* expert_counts,       // [n_experts]
    const __nv_bfloat16* h_norm,   // [num_tokens, hidden_dim]
    const int32_t* topk_indices,   // [num_tokens, top_k]
    int num_tokens,
    int top_k,
    int n_experts,
    int hidden_dim,
    int moe_intermediate,
    cudaStream_t stream = 0);

