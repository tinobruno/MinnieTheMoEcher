#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include "../src/cuda/activations.cuh"

int main() {
    int dim = 4096;
    int moe_inter = 2048;
    int top_k = 6;
    int n_heads = 64;
    int head_dim = 512;
    int cache_len = 128;
    int vocab = 129280;

    __nv_bfloat16 *d_hidden, *d_q, *d_kv, *d_attn_out, *d_gate, *d_down, *d_moe_accum, *d_head_weight;
    float *d_sink, *d_logits, *d_scores;
    int32_t *d_topk_idx;
    void **d_ptrs;

    cudaMalloc(&d_hidden, dim * 2);
    cudaMalloc(&d_q, n_heads * head_dim * 2);
    cudaMalloc(&d_kv, cache_len * head_dim * 2);
    cudaMalloc(&d_attn_out, n_heads * head_dim * 2);
    cudaMalloc(&d_sink, n_heads * 4);
    cudaMalloc(&d_gate, (top_k + 1) * moe_inter * 2);
    cudaMalloc(&d_down, (top_k + 1) * dim * 2);
    cudaMalloc(&d_moe_accum, dim * 2);
    cudaMalloc(&d_scores, 256 * 4);
    cudaMalloc(&d_topk_idx, top_k * 4);
    cudaMalloc(&d_ptrs, 32 * sizeof(void*));
    cudaMalloc(&d_head_weight, (size_t)vocab * dim * 2);
    cudaMalloc(&d_logits, vocab * 4);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iters = 50;

    // 1. Attention: 43 layers of mla_attention
    cudaEventRecord(start, stream);
    for (int it = 0; it < iters; it++) {
        for (int l = 0; l < 43; l++) {
            mla_attention_cuda(d_q, d_kv, d_sink, d_attn_out, n_heads, cache_len, head_dim, 0.044f, stream);
        }
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms_attn = 0;
    cudaEventElapsedTime(&ms_attn, start, stop);
    std::cout << "43 layers MLA Attention: " << ms_attn / iters << " ms" << std::endl;

    // 2. cuBLAS Logits: 1 x 129280 x 4096
    cudaEventRecord(start, stream);
    for (int it = 0; it < iters; it++) {
        float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     vocab, 1, dim,
                     &alpha,
                     d_head_weight, CUDA_R_16BF, dim,
                     d_hidden, CUDA_R_16BF, dim,
                     &beta,
                     d_hidden, CUDA_R_16BF, vocab,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms_logits = 0;
    cudaEventElapsedTime(&ms_logits, start, stop);
    std::cout << "1x Logits cuBLAS GEMM (head.weight 1.06 GB): " << ms_logits / iters << " ms" << std::endl;

    return 0;
}
