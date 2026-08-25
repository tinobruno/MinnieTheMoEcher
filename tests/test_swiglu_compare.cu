#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "src/cuda/activations.cuh"

int main() {
    int dim = 4096;
    int moe_inter = 2048;

    std::cout << "Testing SwiGLU comparison on Expert 0 Layer 0..." << std::endl;

    // Load FP4 Expert 0
    std::ifstream f_fp4("moe_experts.bin", std::ios::binary);
    if (!f_fp4) {
        std::cerr << "Cannot open moe_experts.bin" << std::endl;
        return 1;
    }
    // FP4 block size from manifest: 13631488 (or whatever block size)
    // Let's read w1: 4194304 bytes, w1_scale: 65536 bytes, w3: 4194304 bytes, w3_scale: 65536 bytes, w2: 4194304 bytes, w2_scale: 131072 bytes
    size_t fp4_block_size = 12845056; // let's check exact offsets from moecher_manifest.json
    std::vector<uint8_t> h_fp4(15000000);
    f_fp4.read((char*)h_fp4.data(), h_fp4.size());

    // Load IQ2 Expert 0
    std::ifstream f_iq2("moe_experts_iq2.bin", std::ios::binary);
    if (!f_iq2) {
        std::cerr << "Cannot open moe_experts_iq2.bin" << std::endl;
        return 1;
    }
    size_t w1_size = 2162688;
    size_t w3_size = 2162688;
    size_t w2_size = 2752512;
    size_t iq2_block_size = w1_size + w3_size + w2_size;
    std::vector<uint8_t> h_iq2(iq2_block_size);
    f_iq2.read((char*)h_iq2.data(), iq2_block_size);

    // Input hidden vector
    std::vector<__nv_bfloat16> h_x(dim);
    for (int i = 0; i < dim; i++) {
        h_x[i] = __float2bfloat16(0.01f * sinf((float)i));
    }

    __nv_bfloat16 *d_x, *d_gate_fp4, *d_up_fp4, *d_down_fp4;
    __nv_bfloat16 *d_gate_iq2, *d_up_iq2, *d_down_iq2;
    cudaMalloc(&d_x, dim * sizeof(__nv_bfloat16));
    cudaMalloc(&d_gate_fp4, moe_inter * sizeof(__nv_bfloat16));
    cudaMalloc(&d_up_fp4, moe_inter * sizeof(__nv_bfloat16));
    cudaMalloc(&d_down_fp4, dim * sizeof(__nv_bfloat16));
    cudaMalloc(&d_gate_iq2, moe_inter * sizeof(__nv_bfloat16));
    cudaMalloc(&d_up_iq2, moe_inter * sizeof(__nv_bfloat16));
    cudaMalloc(&d_down_iq2, dim * sizeof(__nv_bfloat16));

    cudaMemcpy(d_x, h_x.data(), dim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    // Run IQ2 SwiGLU
    uint8_t* d_iq2_mem;
    cudaMalloc(&d_iq2_mem, iq2_block_size);
    cudaMemcpy(d_iq2_mem, h_iq2.data(), iq2_block_size, cudaMemcpyHostToDevice);

    gemv_iq2_xxs_cuda(d_gate_iq2, d_x, (const block_iq2_xxs*)d_iq2_mem, moe_inter, dim);
    gemv_iq2_xxs_cuda(d_up_iq2, d_x, (const block_iq2_xxs*)(d_iq2_mem + w1_size), moe_inter, dim);
    silu_mul_cuda(d_gate_iq2, d_gate_iq2, d_up_iq2, moe_inter, 10.0f);
    gemv_q2_k_cuda(d_down_iq2, d_gate_iq2, (const block_q2_K*)(d_iq2_mem + w1_size + w3_size), dim, moe_inter);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> h_down_iq2(dim);
    cudaMemcpy(h_down_iq2.data(), d_down_iq2, dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    float iq2_norm = 0;
    for (int i = 0; i < dim; i++) {
        float v = __bfloat162float(h_down_iq2[i]);
        iq2_norm += v * v;
    }
    std::cout << "IQ2 SwiGLU output norm: " << sqrtf(iq2_norm) << std::endl;
    std::cout << "IQ2 SwiGLU first 5 values: ";
    for (int i = 0; i < 5; i++) std::cout << __bfloat162float(h_down_iq2[i]) << " ";
    std::cout << std::endl;

    // Run FP4 SwiGLU
    size_t fp4_blk_size = 13369344;
    uint8_t* d_fp4_mem;
    cudaMalloc(&d_fp4_mem, fp4_blk_size);
    cudaMemcpy(d_fp4_mem, h_fp4.data(), fp4_blk_size, cudaMemcpyHostToDevice);

    uint8_t* fp4_w1_data = d_fp4_mem + 0;
    uint8_t* fp4_w1_scale = d_fp4_mem + 4194304;
    uint8_t* fp4_w3_data = d_fp4_mem + 4456448;
    uint8_t* fp4_w3_scale = d_fp4_mem + 8650752;
    uint8_t* fp4_w2_data = d_fp4_mem + 8912896;
    uint8_t* fp4_w2_scale = d_fp4_mem + 13107200;

    __nv_bfloat16 *d_dequant_w1, *d_dequant_w3, *d_dequant_w2;
    cudaMalloc(&d_dequant_w1, moe_inter * dim * sizeof(__nv_bfloat16));
    cudaMalloc(&d_dequant_w3, moe_inter * dim * sizeof(__nv_bfloat16));
    cudaMalloc(&d_dequant_w2, dim * moe_inter * sizeof(__nv_bfloat16));

    fp4_dequant_cuda(d_dequant_w1, fp4_w1_data, fp4_w1_scale, moe_inter, dim / 2, 128);
    fp4_dequant_cuda(d_dequant_w3, fp4_w3_data, fp4_w3_scale, moe_inter, dim / 2, 128);
    fp4_dequant_cuda(d_dequant_w2, fp4_w2_data, fp4_w2_scale, dim, moe_inter / 2, 64);

    // GEMV for w1, w3, w2
    std::vector<__nv_bfloat16> h_w1_deq(moe_inter * dim);
    std::vector<__nv_bfloat16> h_w3_deq(moe_inter * dim);
    std::vector<__nv_bfloat16> h_w2_deq(dim * moe_inter);
    cudaMemcpy(h_w1_deq.data(), d_dequant_w1, moe_inter * dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w3_deq.data(), d_dequant_w3, moe_inter * dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w2_deq.data(), d_dequant_w2, dim * moe_inter * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    std::vector<float> gate_fp4_f32(moe_inter, 0.0f);
    std::vector<float> up_fp4_f32(moe_inter, 0.0f);
    for (int r = 0; r < moe_inter; r++) {
        for (int c = 0; c < dim; c++) {
            float a = __bfloat162float(h_x[c]);
            gate_fp4_f32[r] += __bfloat162float(h_w1_deq[r * dim + c]) * a;
            up_fp4_f32[r] += __bfloat162float(h_w3_deq[r * dim + c]) * a;
        }
    }

    std::vector<float> silu_fp4(moe_inter);
    for (int i = 0; i < moe_inter; i++) {
        float g = gate_fp4_f32[i];
        float silu_g = g / (1.0f + expf(-g));
        silu_fp4[i] = silu_g * up_fp4_f32[i];
    }

    std::vector<float> down_fp4_f32(dim, 0.0f);
    for (int r = 0; r < dim; r++) {
        for (int c = 0; c < moe_inter; c++) {
            down_fp4_f32[r] += __bfloat162float(h_w2_deq[r * moe_inter + c]) * silu_fp4[c];
        }
    }

    float fp4_norm = 0;
    for (int i = 0; i < dim; i++) {
        fp4_norm += down_fp4_f32[i] * down_fp4_f32[i];
    }
    std::cout << "FP4 SwiGLU output norm: " << sqrtf(fp4_norm) << std::endl;
    std::cout << "FP4 SwiGLU first 5 values: ";
    for (int i = 0; i < 5; i++) std::cout << down_fp4_f32[i] << " ";
    std::cout << std::endl;

    // Cosine similarity
    float dot = 0;
    for (int i = 0; i < dim; i++) {
        dot += down_fp4_f32[i] * __bfloat162float(h_down_iq2[i]);
    }
    float cos_sim = dot / (sqrtf(fp4_norm) * sqrtf(iq2_norm));
    std::cout << "Cosine similarity between FP4 and IQ2 output: " << cos_sim << std::endl;

    return 0;
}
