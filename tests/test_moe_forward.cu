#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../src/cuda/activations.cuh"

int main() {
    int dim = 4096;
    int moe_inter = 2048;
    int expert_size = 7077888; // 2162688 (w1) + 2162688 (w3) + 2752512 (w2)

    std::ifstream f("moe_test_expert.bin", std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Failed to open moe_test_expert.bin" << std::endl;
        return 1;
    }

    std::vector<uint8_t> h_expert(expert_size);
    f.read((char*)h_expert.data(), expert_size);
    f.close();

    uint8_t* d_expert;
    cudaMalloc(&d_expert, expert_size);
    cudaMemcpy(d_expert, h_expert.data(), expert_size, cudaMemcpyHostToDevice);

    uint8_t* w1_data = d_expert;
    uint8_t* w3_data = d_expert + 2162688;
    uint8_t* w2_data = d_expert + 2162688 + 2162688;

    // Check d scale of w1[0]
    block_iq2_xxs* b_w1 = (block_iq2_xxs*)h_expert.data();
    std::cout << "w1 block 0 d: " << __half2float(b_w1[0].d) << std::endl;
    std::cout << "w1 block 0 qs[0]: " << b_w1[0].qs[0] << std::endl;

    block_q2_K* b_w2 = (block_q2_K*)(h_expert.data() + 2162688 + 2162688);
    std::cout << "w2 block 0 d: " << __half2float(b_w2[0].d) << " dmin: " << __half2float(b_w2[0].dmin) << std::endl;

    // Input activation x (normalized random or ones)
    std::vector<__nv_bfloat16> h_x(dim);
    for (int i = 0; i < dim; i++) {
        h_x[i] = __float2bfloat16(0.01f * std::sin(i * 0.1f));
    }
    __nv_bfloat16 *d_x, *d_gate, *d_up, *d_down;
    cudaMalloc(&d_x, dim * sizeof(__nv_bfloat16));
    cudaMalloc(&d_gate, moe_inter * sizeof(__nv_bfloat16));
    cudaMalloc(&d_up, moe_inter * sizeof(__nv_bfloat16));
    cudaMalloc(&d_down, dim * sizeof(__nv_bfloat16));

    cudaMemcpy(d_x, h_x.data(), dim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    // 1. Gate: [moe_inter, dim] x [dim] -> [moe_inter]
    gemv_iq2_xxs_cuda(d_gate, d_x, (const block_iq2_xxs*)w1_data, moe_inter, dim);

    // 2. Up: [moe_inter, dim] x [dim] -> [moe_inter]
    gemv_iq2_xxs_cuda(d_up, d_x, (const block_iq2_xxs*)w3_data, moe_inter, dim);

    // 3. SwiGLU:
    silu_mul_cuda(d_gate, d_gate, d_up, moe_inter, 10.0f);

    // 4. Down: [dim, moe_inter] x [moe_inter] -> [dim]
    gemv_q2_k_cuda(d_down, d_gate, (const block_q2_K*)w2_data, dim, moe_inter);

    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> h_gate(moe_inter), h_up(moe_inter), h_down(dim);
    cudaMemcpy(h_gate.data(), d_gate, moe_inter * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_down.data(), d_down, dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    float gate_norm = 0.0f, down_norm = 0.0f;
    for (int i = 0; i < moe_inter; i++) {
        float g = __bfloat162float(h_gate[i]);
        gate_norm += g * g;
    }
    for (int i = 0; i < dim; i++) {
        float d = __bfloat162float(h_down[i]);
        down_norm += d * d;
    }

    std::cout << "SwiGLU output norm: " << std::sqrt(gate_norm) << std::endl;
    std::cout << "Down output norm: " << std::sqrt(down_norm) << std::endl;
    std::cout << "First 5 down elements: ";
    for (int i = 0; i < 5; i++) {
        std::cout << __bfloat162float(h_down[i]) << " ";
    }
    std::cout << std::endl;

    cudaFree(d_expert);
    cudaFree(d_x);
    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_down);
    return 0;
}
