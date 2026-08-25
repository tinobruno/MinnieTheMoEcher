#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../src/cuda/activations.cuh"

// Host table for reference CPU verification
namespace iq2_host {
static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};
static const uint8_t ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};
static const uint64_t iq2xxs_grid[256] = {
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
} // namespace iq2_host

void dequant_cpu(const block_iq2_xxs* x, float* y, int K) {
    int nb = K / 256;
    uint32_t aux32[2];
    const uint8_t* aux8 = (const uint8_t*)aux32;
    for (int i = 0; i < nb; i++) {
        float d = __half2float(x[i].d);
        for (int ib32 = 0; ib32 < 8; ib32++) {
            std::memcpy(aux32, x[i].qs + 4 * ib32, 2 * sizeof(uint32_t));
            float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; l++) {
                const uint8_t* grid = (const uint8_t*)(iq2_host::iq2xxs_grid + aux8[l]);
                const uint8_t signs = iq2_host::ksigns_iq2xs[(aux32[1] >> 7 * l) & 127];
                for (int j = 0; j < 8; j++) {
                    y[j] = db * grid[j] * ((signs & iq2_host::kmask_iq2xs[j]) ? -1.0f : 1.0f);
                }
                y += 8;
            }
        }
    }
}

int main() {
    std::cout << "Testing IQ2_XXS CUDA Dequant and GEMV..." << std::endl;

    int N = 128;
    int K = 256 * 4; // 1024
    int blocks_per_row = K / 256;
    int total_blocks = N * blocks_per_row;

    // Create synthetic IQ2_XXS data
    std::vector<block_iq2_xxs> h_weight(total_blocks);
    for (int i = 0; i < total_blocks; i++) {
        h_weight[i].d = __float2half(0.015f + 0.001f * (i % 10));
        for (int j = 0; j < 32; j++) {
            h_weight[i].qs[j] = (uint16_t)((i * 37 + j * 13) & 0xFFFF);
        }
    }

    // CPU dequantize
    std::vector<float> h_deq_ref(N * K);
    for (int r = 0; r < N; r++) {
        dequant_cpu(&h_weight[r * blocks_per_row], &h_deq_ref[r * K], K);
    }

    // Input activation vector
    std::vector<__nv_bfloat16> h_vec(K);
    std::vector<float> h_vec_f32(K);
    for (int k = 0; k < K; k++) {
        float val = std::sin((float)k * 0.1f);
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
    block_iq2_xxs* d_weight;
    __nv_bfloat16* d_vec;
    __nv_bfloat16* d_out_deq;
    __nv_bfloat16* d_out_gemv;

    cudaMalloc(&d_weight, total_blocks * sizeof(block_iq2_xxs));
    cudaMalloc(&d_vec, K * sizeof(__nv_bfloat16));
    cudaMalloc(&d_out_deq, N * K * sizeof(__nv_bfloat16));
    cudaMalloc(&d_out_gemv, N * sizeof(__nv_bfloat16));

    cudaMemcpy(d_weight, h_weight.data(), total_blocks * sizeof(block_iq2_xxs), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, h_vec.data(), K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    // Run GPU Dequant
    iq2_xxs_dequant_cuda(d_out_deq, d_weight, N, K);
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
        std::cerr << "Dequant verification failed!" << std::endl;
        return 1;
    }

    // Run GPU GEMV
    gemv_iq2_xxs_cuda(d_out_gemv, d_vec, d_weight, N, K);
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
        std::cerr << "GEMV verification failed!" << std::endl;
        return 1;
    }

    std::cout << "✓ IQ2_XXS CUDA Kernels verified successfully!" << std::endl;

    cudaFree(d_weight);
    cudaFree(d_vec);
    cudaFree(d_out_deq);
    cudaFree(d_out_gemv);
    return 0;
}
