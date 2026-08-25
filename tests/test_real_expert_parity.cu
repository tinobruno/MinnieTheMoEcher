#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "src/cuda/activations.cuh"

namespace ref {
static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};
static const uint8_t ksigns_iq2xs[128] = {
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
    144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159,
    160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
    80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
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

void dequantize_row_iq2_xxs(const block_iq2_xxs* x, float* y, int K) {
    const int nb = K / 256;
    uint32_t aux32[2];
    const uint8_t* aux8 = (const uint8_t*)aux32;
    for (int i = 0; i < nb; i++) {
        const float d = __half2float(x[i].d);
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            std::memcpy(aux32, x[i].qs + 4*ib32, 2*sizeof(uint32_t));
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[l]);
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db * (float)grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}

void dequantize_row_q2_k(const block_q2_K* x, float* y, int K) {
    const int nb = K / 256;
    for (int i = 0; i < nb; i++) {
        const float d   = __half2float(x[i].d);
        const float min = __half2float(x[i].dmin);
        const uint8_t* q = x[i].qs;
        int is = 0;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = x[i].scales[is++];
                float dl = d * (float)(sc & 0xF);
                float ml = min * (float)(sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * (float)((q[l] >> shift) & 3) - ml;
                sc = x[i].scales[is++];
                dl = d * (float)(sc & 0xF);
                ml = min * (float)(sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * (float)((q[l+16] >> shift) & 3) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}
} // namespace ref

int main() {
    std::cout << "Loading Expert 0 from moe_experts_iq2.bin..." << std::endl;
    std::ifstream fin("moe_experts_iq2.bin", std::ios::binary);
    if (!fin) {
        std::cerr << "Failed to open moe_experts_iq2.bin" << std::endl;
        return 1;
    }

    size_t w1_size = 2162688;
    size_t w3_size = 2162688;
    size_t w2_size = 2752512;
    std::vector<uint8_t> h_w1(w1_size);
    std::vector<uint8_t> h_w3(w3_size);
    std::vector<uint8_t> h_w2(w2_size);

    fin.read((char*)h_w1.data(), w1_size);
    fin.read((char*)h_w3.data(), w3_size);
    fin.read((char*)h_w2.data(), w2_size);

    int rows_gate = 2048, cols_gate = 4096;
    int rows_down = 4096, cols_down = 2048;

    // CPU Reference dequant
    std::vector<float> cpu_w1(rows_gate * cols_gate);
    std::vector<float> cpu_w2(rows_down * cols_down);
    for (int r = 0; r < rows_gate; r++) {
        ref::dequantize_row_iq2_xxs((const block_iq2_xxs*)(h_w1.data() + r * (cols_gate / 256) * sizeof(block_iq2_xxs)),
                                   cpu_w1.data() + r * cols_gate, cols_gate);
    }
    for (int r = 0; r < rows_down; r++) {
        ref::dequantize_row_q2_k((const block_q2_K*)(h_w2.data() + r * (cols_down / 256) * sizeof(block_q2_K)),
                                 cpu_w2.data() + r * cols_down, cols_down);
    }

    // GPU Dequant
    block_iq2_xxs* d_w1;
    block_q2_K* d_w2;
    __nv_bfloat16* d_out_w1;
    __nv_bfloat16* d_out_w2;

    cudaMalloc(&d_w1, w1_size);
    cudaMalloc(&d_w2, w2_size);
    cudaMalloc(&d_out_w1, rows_gate * cols_gate * sizeof(__nv_bfloat16));
    cudaMalloc(&d_out_w2, rows_down * cols_down * sizeof(__nv_bfloat16));

    cudaMemcpy(d_w1, h_w1.data(), w1_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, h_w2.data(), w2_size, cudaMemcpyHostToDevice);

    iq2_xxs_dequant_cuda(d_out_w1, d_w1, rows_gate, cols_gate);
    q2_k_dequant_cuda(d_out_w2, d_w2, rows_down, cols_down);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> gpu_w1_bf16(rows_gate * cols_gate);
    std::vector<__nv_bfloat16> gpu_w2_bf16(rows_down * cols_down);
    cudaMemcpy(gpu_w1_bf16.data(), d_out_w1, rows_gate * cols_gate * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_w2_bf16.data(), d_out_w2, rows_down * cols_down * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    float max_err_w1 = 0;
    int mismatches = 0;
    for (size_t i = 0; i < cpu_w1.size(); i++) {
        float g = __bfloat162float(gpu_w1_bf16[i]);
        float c = cpu_w1[i];
        float err = std::abs(g - c);
        if (err > 0.01f && mismatches < 10) {
            std::cout << "Mismatch at index " << i << " (r=" << i / cols_gate << ", c=" << i % cols_gate 
                      << "): GPU=" << g << ", CPU=" << c << ", err=" << err << std::endl;
            mismatches++;
        }
        max_err_w1 = std::max(max_err_w1, err);
    }
    std::cout << "IQ2_XXS (Gate) Dequant Max Error between GPU and CPU ref: " << max_err_w1 << std::endl;

    float max_err_w2 = 0;
    for (size_t i = 0; i < cpu_w2.size(); i++) {
        float g = __bfloat162float(gpu_w2_bf16[i]);
        float c = cpu_w2[i];
        max_err_w2 = std::max(max_err_w2, std::abs(g - c));
    }
    std::cout << "Q2_K (Down) Dequant Max Error between GPU and CPU ref: " << max_err_w2 << std::endl;

    // Test GEMV on GPU vs CPU
    std::vector<__nv_bfloat16> h_x(cols_gate);
    std::vector<float> h_x_f32(cols_gate);
    for (int i = 0; i < cols_gate; i++) {
        float val = 0.01f * sinf((float)i);
        h_x[i] = __float2bfloat16(val);
        h_x_f32[i] = val;
    }

    __nv_bfloat16* d_x;
    __nv_bfloat16* d_gemv_out_w1;
    cudaMalloc(&d_x, cols_gate * sizeof(__nv_bfloat16));
    cudaMalloc(&d_gemv_out_w1, rows_gate * sizeof(__nv_bfloat16));
    cudaMemcpy(d_x, h_x.data(), cols_gate * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    gemv_iq2_xxs_cuda(d_gemv_out_w1, d_x, d_w1, rows_gate, cols_gate);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> gpu_gemv_w1(rows_gate);
    cudaMemcpy(gpu_gemv_w1.data(), d_gemv_out_w1, rows_gate * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    float max_gemv_err = 0;
    for (int r = 0; r < rows_gate; r++) {
        float ref_val = 0;
        for (int c = 0; c < cols_gate; c++) {
            ref_val += cpu_w1[r * cols_gate + c] * h_x_f32[c];
        }
        float g_val = __bfloat162float(gpu_gemv_w1[r]);
        max_gemv_err = std::max(max_gemv_err, std::abs(g_val - ref_val));
    }
    std::cout << "IQ2_XXS (w1) GEMV Max Error vs CPU Reference GEMV: " << max_gemv_err << std::endl;

    // Test Q2_K GEMV (w2)
    std::vector<__nv_bfloat16> h_h(cols_down);
    std::vector<float> h_h_f32(cols_down);
    for (int i = 0; i < cols_down; i++) {
        float val = 0.01f * cosf((float)i);
        h_h[i] = __float2bfloat16(val);
        h_h_f32[i] = val;
    }

    __nv_bfloat16* d_h;
    __nv_bfloat16* d_gemv_out_w2;
    cudaMalloc(&d_h, cols_down * sizeof(__nv_bfloat16));
    cudaMalloc(&d_gemv_out_w2, rows_down * sizeof(__nv_bfloat16));
    cudaMemcpy(d_h, h_h.data(), cols_down * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    gemv_q2_k_cuda(d_gemv_out_w2, d_h, d_w2, rows_down, cols_down);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> gpu_gemv_w2(rows_down);
    cudaMemcpy(gpu_gemv_w2.data(), d_gemv_out_w2, rows_down * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    float max_gemv_err_w2 = 0;
    for (int r = 0; r < rows_down; r++) {
        float ref_val = 0;
        for (int c = 0; c < cols_down; c++) {
            ref_val += cpu_w2[r * cols_down + c] * h_h_f32[c];
        }
        float g_val = __bfloat162float(gpu_gemv_w2[r]);
        max_gemv_err_w2 = std::max(max_gemv_err_w2, std::abs(g_val - ref_val));
    }
    std::cout << "Q2_K (w2) GEMV Max Error vs CPU Reference GEMV: " << max_gemv_err_w2 << std::endl;

    return 0;
}
