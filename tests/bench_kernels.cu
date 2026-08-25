#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include <chrono>
#include "../src/cuda/activations.cuh"

int main() {
    int dim = 4096;
    int q_lora = 1536;
    int n_heads = 64;
    int head_dim = 512;
    int n_heads_head_dim = n_heads * head_dim; // 32768
    int o_lora = 1024;
    int o_groups = 8;
    int vocab = 129280;

    __nv_bfloat16 *d_hidden, *d_lora, *d_q, *d_out;
    uint8_t *d_w, *d_s;
    float *d_logits;
    cudaMalloc(&d_hidden, dim * 2);
    cudaMalloc(&d_lora, n_heads_head_dim * 2);
    cudaMalloc(&d_q, n_heads_head_dim * 2);
    cudaMalloc(&d_out, n_heads_head_dim * 2);
    cudaMalloc(&d_w, (size_t)n_heads_head_dim * dim);
    cudaMalloc(&d_s, (size_t)n_heads_head_dim * (dim / 128));
    cudaMalloc(&d_logits, vocab * 4);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warmup
    for (int i = 0; i < 10; i++) {
        gemv_fp8_cuda(d_q, d_lora, d_w, d_s, n_heads_head_dim, q_lora, 128, stream);
    }
    cudaStreamSynchronize(stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iters = 100;
    // Benchmark wq_b (1536 -> 32768)
    cudaEventRecord(start, stream);
    for (int i = 0; i < iters; i++) {
        gemv_fp8_cuda(d_q, d_lora, d_w, d_s, n_heads_head_dim, q_lora, 128, stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms_wq_b = 0;
    cudaEventElapsedTime(&ms_wq_b, start, stop);
    std::cout << "wq_b (1536 -> 32768, 50 MB): " << ms_wq_b / iters << " ms" << std::endl;

    // Benchmark wo_a grouped (32768 -> 1024 per group, 8 groups)
    cudaEventRecord(start, stream);
    for (int i = 0; i < iters; i++) {
        gemv_fp8_grouped_cuda(d_lora, d_out, d_w, d_s, o_lora, (n_heads / o_groups) * head_dim, o_groups, 128, stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms_wo_a = 0;
    cudaEventElapsedTime(&ms_wo_a, start, stop);
    std::cout << "wo_a grouped (32768 -> 1024*8, 33.5 MB): " << ms_wo_a / iters << " ms" << std::endl;

    cudaFree(d_hidden); cudaFree(d_lora); cudaFree(d_q); cudaFree(d_out);
    cudaFree(d_w); cudaFree(d_s); cudaFree(d_logits);
    return 0;
}
