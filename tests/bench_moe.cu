#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include <vector>
#include "../src/cuda/activations.cuh"

int main() {
    int dim = 4096;
    int moe_inter = 2048;
    int top_k = 6;

    __nv_bfloat16 *d_hidden, *d_gate, *d_down, *d_moe_accum, *d_shared_down;
    float *d_topk_weights;
    int32_t *d_topk_idx;
    void **d_ptrs;

    cudaMalloc(&d_hidden, dim * 2);
    cudaMalloc(&d_gate, (top_k + 1) * moe_inter * 2);
    cudaMalloc(&d_down, (top_k + 1) * dim * 2);
    cudaMalloc(&d_moe_accum, dim * 2);
    cudaMalloc(&d_shared_down, dim * 2);
    cudaMalloc(&d_topk_weights, top_k * 4);
    cudaMalloc(&d_topk_idx, top_k * 4);
    
    size_t expert_bytes = 8 * 1024 * 1024;
    void *d_expert_block;
    cudaMalloc(&d_expert_block, expert_bytes);
    std::vector<void*> h_ptrs(32, d_expert_block);
    cudaMalloc(&d_ptrs, 32 * sizeof(void*));
    cudaMemcpy(d_ptrs, h_ptrs.data(), 32 * sizeof(void*), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iters = 50;

    // Benchmark 43 layers of MoE
    cudaEventRecord(start, stream);
    for (int it = 0; it < iters; it++) {
        for (int l = 0; l < 43; l++) {
            gemv_iq2_xxs_moe_swiglu_fused_cuda(d_gate, d_hidden, (const void* const*)d_ptrs,
                                               0, 2048 * 4096 / 4, moe_inter, dim, 0.0f, stream);
            gemv_q2_k_moe_cuda(d_down, d_gate, (const void* const*)d_ptrs,
                               4096 * 4096 / 4, dim, moe_inter, stream);
            fused_moe_accum_dynamic_cuda(d_moe_accum, d_down, d_topk_weights, d_shared_down, dim, stream);
        }
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms_moe = 0;
    cudaEventElapsedTime(&ms_moe, start, stop);
    std::cout << "43 layers MoE (SwiGLU + Down + Accum): " << ms_moe / iters << " ms" << std::endl;

    return 0;
}
