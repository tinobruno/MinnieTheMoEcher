#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// LayerNorm with affine transform: out = (x - mean) / sqrt(var + eps) * gamma + beta
void layernorm_affine_bf16_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* x,
    const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta,
    int n_rows,
    int dim,
    float eps,
    cudaStream_t stream = 0
);

// Fast GELU activation with optional bias addition: out = GELU(x + bias)
void gelu_bias_bf16_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* x,
    const __nv_bfloat16* bias,
    int n_rows,
    int dim,
    cudaStream_t stream = 0
);

// Residual connection + bias: x = x + residual + bias
void add_residual_bias_bf16_cuda(
    __nv_bfloat16* x,
    const __nv_bfloat16* residual,
    const __nv_bfloat16* bias,
    int n_rows,
    int dim,
    cudaStream_t stream = 0
);

// Add bias to tensor: x[row, d] += bias[d]
void add_bias_bf16_cuda(
    __nv_bfloat16* x,
    const __nv_bfloat16* bias,
    int n_rows,
    int dim,
    cudaStream_t stream = 0
);

// Elementwise addition of two tensors: dst[i] += src[i]
void add_tensors_bf16_cuda(
    __nv_bfloat16* dst,
    const __nv_bfloat16* src,
    size_t count,
    cudaStream_t stream = 0
);

// Split QKV buffer [N, 3456] into Q [H, N, D], K [H, N, D], V [H, N, D] adding bias and applying 2D RoPE
void vit_split_qkv_bias_bf16_cuda(
    __nv_bfloat16* Q_out,  // [16, N, 72]
    __nv_bfloat16* K_out,  // [16, N, 72]
    __nv_bfloat16* V_out,  // [16, N, 72]
    const __nv_bfloat16* qkv_in, // [N, 3456]
    const __nv_bfloat16* bias,   // [3456]
    const float* cos_table,      // [N, 72] 2D RoPE cos
    const float* sin_table,      // [N, 72] 2D RoPE sin
    int N,
    int n_heads,
    int head_dim,
    cudaStream_t stream = 0
);

// Softmax over rows of attention matrix [16 * N, N] with scaling factor
void vit_softmax_bf16_cuda(
    __nv_bfloat16* scores, // [16 * N, N]
    int total_rows,        // 16 * N
    int seq_len,           // N
    float scale,           // 1.0 / sqrt(head_dim)
    cudaStream_t stream = 0
);

// Merge attention heads output [16, N, 72] -> [N, 1152]
void vit_merge_heads_bf16_cuda(
    __nv_bfloat16* out,          // [N, 1152]
    const __nv_bfloat16* heads,  // [16, N, 72]
    int N,
    int n_heads,
    int head_dim,
    cudaStream_t stream = 0
);

// Spatial merge gather: groups 2x2 adjacent patches into 4x dim features
// Input: [H_patches, W_patches, dim]
// Output: [H_patches/2, W_patches/2, 4 * dim]
void spatial_merge_gather_bf16_cuda(
    __nv_bfloat16* out,
    const __nv_bfloat16* in,
    int H_patches,
    int W_patches,
    int dim,
    cudaStream_t stream = 0
);

// Im2Col extraction for Conv3D patch embedding: [2, 3, H, W] -> [P=2304, 1536] in BF16
void im2col_patch_embed_bf16_cuda(
    __nv_bfloat16* im2col_out, // [P=2304, 1536]
    const float* img_norm,     // [T=2, C=3, H=768, W=768]
    int H, int W,
    int patch_size,
    int temporal_size,
    int in_channels,
    cudaStream_t stream = 0
);

// Add bias and positional embedding: out[p, d] += bias[d] + pos_embed[p, d]
void add_bias_and_posemb_bf16_cuda(
    __nv_bfloat16* out, // [P, dim]
    const __nv_bfloat16* bias, // [dim]
    const __nv_bfloat16* pos_embed, // [P, dim]
    int num_patches,
    int dim,
    cudaStream_t stream = 0
);
