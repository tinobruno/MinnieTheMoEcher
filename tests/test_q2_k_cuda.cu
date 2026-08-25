#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../src/cuda/activations.cuh"

// Reference CPU dequantize for Q2_K
void dequant_q2_k_cpu(const block_q2_K* x, float* y, int K) {
    int nb = K / 256;
    for (int b = 0; b < nb; b++) {
        const block_q2_K* xb = &x[b];
        float d = __half2float(xb->d);
        float min = __half2float(xb->dmin);

        for (int group = 0; group < 16; group++) {
            uint8_t sc = xb->scales[group];
            float dl = d * (float)(sc & 0x0F);
            float ml = min * (float)(sc >> 4);
            int q_base = 32 * (group / 8) + 16 * (group & 1);
            int shift = ((group / 2) & 3) * 2;

            for (int l = 0; l < 16; l++) {
                uint8_t q = (xb->qs[q_base + l] >> shift) & 0x03;
                y[group * 16 + l] = dl * (float)q - ml;
            }
        }
        y += 256;
    }
}

int main() {
    std::cout << "Testing Q2_K CUDA Dequant and GEMV..." << std::endl;

    int N = 128;
    int K = 256 * 4; // 1024
    int blocks_per_row = K / 256;
    int total_blocks = N * blocks_per_row;

    // Create synthetic Q2_K data
    std::vector<block_q2_K> h_weight(total_blocks);
    for (int i = 0; i < total_blocks; i++) {
        h_weight[i].d = __float2half(0.02f + 0.001f * (i % 10));
        h_weight[i].dmin = __float2half(0.01f + 0.0005f * (i % 8));
        for (int j = 0; j < 16; j++) {
            h_weight[i].scales[j] = (uint8_t)((j * 17 + i * 3) & 0xFF);
        }
        for (int j = 0; j < 64; j++) {
            h_weight[i].qs[j] = (uint8_t)((j * 29 + i * 7) & 0xFF);
        }
    }

    // CPU dequantize
    std::vector<float> h_deq_ref(N * K);
    for (int r = 0; r < N; r++) {
        dequant_q2_k_cpu(&h_weight[r * blocks_per_row], &h_deq_ref[r * K], K);
    }

    // Input activation vector
    std::vector<__nv_bfloat16> h_vec(K);
    std::vector<float> h_vec_f32(K);
    for (int k = 0; k < K; k++) {
        float val = std::cos((float)k * 0.15f);
        h_vec[k] = __float2bfloat16(val);
        h_vec_f32[k] = __bfloat162float(h_vec[k]);
    }

    // Reference CPU GEMV output
    std::vector<float> h_gemv_ref(N, 0.0f);
    for (int r = 0; r < N; r++) {
        for (int k = 0; k < K; k++) {
            h_gemv_ref[r] += h_deq_ref[r * K + k] * h_vec_f32[k];
        }
    }

    // Allocate GPU memory
    block_q2_K* d_weight;
    __nv_bfloat16* d_vec;
    __nv_bfloat16* d_out_deq;
    __nv_bfloat16* d_out_gemv;

    cudaMalloc(&d_weight, total_blocks * sizeof(block_q2_K));
    cudaMalloc(&d_vec, K * sizeof(__nv_bfloat16));
    cudaMalloc(&d_out_deq, N * K * sizeof(__nv_bfloat16));
    cudaMalloc(&d_out_gemv, N * sizeof(__nv_bfloat16));

    cudaMemcpy(d_weight, h_weight.data(), total_blocks * sizeof(block_q2_K), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, h_vec.data(), K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    // Run GPU Dequant
    q2_k_dequant_cuda(d_out_deq, d_weight, N, K);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> h_out_deq(N * K);
    cudaMemcpy(h_out_deq.data(), d_out_deq, N * K * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    float max_deq_err = 0.0f;
    for (size_t i = 0; i < h_out_deq.size(); i++) {
        float gpu_val = __bfloat162float(h_out_deq[i]);
        float ref_val = h_deq_ref[i];
        float err = std::fabs(gpu_val - ref_val);
        if (err > max_deq_err) max_deq_err = err;
    }
    std::cout << "  Max Dequantization error vs CPU reference: " << max_deq_err << std::endl;
    if (max_deq_err > 1e-2f) {
        std::cerr << "Q2_K Dequant verification failed!" << std::endl;
        return 1;
    }

    // Run GPU GEMV
    gemv_q2_k_cuda(d_out_gemv, d_vec, d_weight, N, K);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> h_out_gemv(N);
    cudaMemcpy(h_out_gemv.data(), d_out_gemv, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    float max_gemv_err = 0.0f;
    for (int r = 0; r < N; r++) {
        float gpu_val = __bfloat162float(h_out_gemv[r]);
        float ref_val = h_gemv_ref[r];
        float err = std::fabs(gpu_val - ref_val);
        if (err > max_gemv_err) max_gemv_err = err;
    }
    std::cout << "  Max GEMV error vs CPU reference: " << max_gemv_err << std::endl;
    if (max_gemv_err > 0.05f) {
        std::cerr << "Q2_K GEMV verification failed!" << std::endl;
        return 1;
    }

    std::cout << "✓ Q2_K CUDA Kernels verified successfully!" << std::endl;

    cudaFree(d_weight);
    cudaFree(d_vec);
    cudaFree(d_out_deq);
    cudaFree(d_out_gemv);
    return 0;
}
