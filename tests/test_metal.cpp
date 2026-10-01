// test_metal.cpp — Comprehensive verification suite for Metal backend on Apple Silicon
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>

#include "metal/metal_backend.h"
#include "cuda/activations.cuh"

#define TEST_CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAIL: " << msg << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            return false; \
        } \
    } while (0)

bool test_device_and_memory() {
    std::cout << "[TEST] 1. Device detection and unified memory..." << std::endl;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    TEST_CHECK(err == cudaSuccess, "cudaGetDeviceProperties failed");
    std::cout << "       Device: " << prop.name << std::endl;
    std::cout << "       Total VRAM: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "       Compute: " << prop.major << "." << prop.minor << std::endl;
    TEST_CHECK(prop.totalGlobalMem > 0, "VRAM must be > 0");

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "       Free Mem: " << (free_mem / (1024 * 1024)) << " MB / " << (total_mem / (1024 * 1024)) << " MB" << std::endl;

    // Test UMA allocation and copy
    const size_t test_size = 1024 * 1024; // 1 MB
    float* d_buf = nullptr;
    err = cudaMalloc((void**)&d_buf, test_size * sizeof(float));
    TEST_CHECK(err == cudaSuccess && d_buf != nullptr, "cudaMalloc failed");

    std::vector<float> h_in(test_size);
    for (size_t i = 0; i < test_size; i++) h_in[i] = float(i) * 0.5f;

    cudaMemcpy(d_buf, h_in.data(), test_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    std::vector<float> h_out(test_size, 0.0f);
    cudaMemcpy(h_out.data(), d_buf, test_size * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < 100; i++) {
        TEST_CHECK(std::fabs(h_in[i] - h_out[i]) < 1e-5f, "Data mismatch in memory copy");
    }

    cudaFree(d_buf);
    std::cout << "       PASS: Device & Memory" << std::endl;
    return true;
}

bool test_rmsnorm() {
    std::cout << "[TEST] 2. RMSNorm Metal compute kernel..." << std::endl;
    const int dim = 4096;
    const float eps = 1e-6f;

    std::vector<__nv_bfloat16> h_x(dim);
    std::vector<__nv_bfloat16> h_w(dim);
    std::vector<float> ref_out(dim);

    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        float x_val = std::sin(float(i)) * 2.0f;
        float w_val = 1.0f + 0.1f * std::cos(float(i));
        h_x[i] = __nv_bfloat16::from_float(x_val);
        h_w[i] = __nv_bfloat16::from_float(w_val);
        sum_sq += x_val * x_val;
    }
    float rms = 1.0f / std::sqrt(sum_sq / float(dim) + eps);
    for (int i = 0; i < dim; i++) {
        ref_out[i] = h_x[i].to_float() * rms * h_w[i].to_float();
    }

    __nv_bfloat16 *d_x = nullptr, *d_w = nullptr, *d_out = nullptr;
    cudaMalloc((void**)&d_x, dim * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_w, dim * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_out, dim * sizeof(__nv_bfloat16));

    cudaMemcpy(d_x, h_x.data(), dim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w.data(), dim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    rms_norm_cuda(d_out, d_x, d_w, dim, eps, 0);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> gpu_out(dim);
    cudaMemcpy(gpu_out.data(), d_out, dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i = 0; i < dim; i++) {
        float err = std::fabs(gpu_out[i].to_float() - ref_out[i]);
        if (err > max_err) max_err = err;
    }
    std::cout << "       Max RMSNorm error: " << max_err << std::endl;
    TEST_CHECK(max_err < 0.05f, "RMSNorm error too large");

    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_out);
    std::cout << "       PASS: RMSNorm Metal Kernel" << std::endl;
    return true;
}

bool test_softmax_argmax() {
    std::cout << "[TEST] 3. Softmax & Argmax Metal compute kernel..." << std::endl;
    const int rows = 1;
    const int cols = 8192;

    std::vector<float> h_x(cols);
    for (int i = 0; i < cols; i++) {
        h_x[i] = std::sin(float(i) * 0.1f) * 5.0f;
    }
    h_x[4321] = 50.0f; // Known peak

    float *d_x = nullptr, *d_sm = nullptr;
    int32_t *d_idx = nullptr;

    cudaMalloc((void**)&d_x, cols * sizeof(float));
    cudaMalloc((void**)&d_sm, cols * sizeof(float));
    cudaMalloc((void**)&d_idx, sizeof(int32_t));

    cudaMemcpy(d_x, h_x.data(), cols * sizeof(float), cudaMemcpyHostToDevice);

    softmax_cuda(d_sm, d_x, rows, cols, 0);
    argmax_f32_cuda(d_idx, d_sm, cols, 0);
    cudaDeviceSynchronize();

    int32_t best_idx = -1;
    float best_val = 0.0f;
    cudaMemcpy(&best_idx, d_idx, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&best_val, d_sm + 4321, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "       Argmax detected index: " << best_idx << ", max prob: " << best_val << std::endl;
    TEST_CHECK(best_idx == 4321, "Argmax failed to find peak token");
    TEST_CHECK(best_val > 0.99f, "Peak token prob should be near 1.0");

    cudaFree(d_x);
    cudaFree(d_sm);
    cudaFree(d_idx);
    std::cout << "       PASS: Softmax & Argmax Metal Kernel" << std::endl;
    return true;
}

bool test_accelerate_gemm() {
    std::cout << "[TEST] 4. Accelerate BLAS GEMM (cublasGemmEx)..." << std::endl;
    const int M = 4, N = 8, K = 16;
    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(N * K, 1.0f); // Transposed layout for weight [N, K]
    std::vector<float> h_C(M * N, 0.0f);

    for (int i = 0; i < M * K; i++) h_A[i] = float(i % 5);
    for (int i = 0; i < N * K; i++) h_B[i] = float((i + 1) % 3);

    // CPU ref: C = A * B^T
    std::vector<float> ref_C(M * N, 0.0f);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += h_A[m * K + k] * h_B[n * K + k];
            }
            ref_C[m * N + n] = sum;
        }
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, N * K * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    // In cuBLAS col-major: C_col(N, M) = B_col(N, K) * A_col(K, M)
    // where B is N x K (lda=K, trans=T), A is K x M (ldb=K, trans=N)
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle = nullptr;
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                 &alpha, d_B, CUDA_R_32F, K, d_A, CUDA_R_32F, K,
                 &beta, d_C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M * N; i++) {
        TEST_CHECK(std::fabs(h_C[i] - ref_C[i]) < 1e-4f, "GEMM calculation mismatch");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    std::cout << "       PASS: Accelerate BLAS GEMM" << std::endl;
    return true;
}

bool test_silu_mul() {
    std::cout << "[TEST] 5. Fused SwiGLU / silu_mul Metal compute kernel..." << std::endl;
    const int n = 2048;
    std::vector<__nv_bfloat16> h_gate(n), h_up(n), h_out(n);
    std::vector<float> ref_out(n);

    for (int i = 0; i < n; i++) {
        float g = std::sin(float(i) * 0.05f) * 3.0f;
        float u = std::cos(float(i) * 0.05f) * 2.0f;
        h_gate[i] = __nv_bfloat16::from_float(g);
        h_up[i] = __nv_bfloat16::from_float(u);

        float silu_g = g / (1.0f + std::exp(-g));
        ref_out[i] = silu_g * u;
    }

    __nv_bfloat16 *d_gate = nullptr, *d_up = nullptr, *d_out = nullptr;
    cudaMalloc((void**)&d_gate, n * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_up, n * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_out, n * sizeof(__nv_bfloat16));

    cudaMemcpy(d_gate, h_gate.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_up, h_up.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    silu_mul_cuda(d_out, d_gate, d_up, n, 0.0f, 0);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out.data(), d_out, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = std::fabs(h_out[i].to_float() - ref_out[i]);
        if (err > max_err) max_err = err;
    }
    std::cout << "       Max SwiGLU error: " << max_err << std::endl;
    TEST_CHECK(max_err < 0.05f, "SwiGLU error too large");

    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_out);
    std::cout << "       PASS: SwiGLU Metal Kernel" << std::endl;
    return true;
}

bool test_rope() {
    std::cout << "[TEST] 6. RoPE Metal compute kernel..." << std::endl;
    const int n_heads = 16, head_dim = 128, rope_dim = 64, pos = 42;
    const int max_seq_len = 128;
    const int total_dim = n_heads * head_dim;

    std::vector<__nv_bfloat16> h_x(total_dim);
    for (int i = 0; i < total_dim; i++) h_x[i] = __nv_bfloat16::from_float(std::sin(float(i)));

    __nv_bfloat16 *d_x = nullptr;
    float *d_freq = nullptr;
    cudaMalloc((void**)&d_x, total_dim * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_freq, max_seq_len * rope_dim * sizeof(float));

    precompute_freqs_cuda(d_freq, max_seq_len, rope_dim, 10000.0f, 1.0f, 0, 32, 1, 0);
    cudaMemcpy(d_x, h_x.data(), total_dim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    rope_cuda(d_x, n_heads, head_dim, rope_dim, pos, d_freq, false, 0);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> out_x(total_dim);
    cudaMemcpy(out_x.data(), d_x, total_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    // Verify RoPE preserved norms across rotary dimensions
    float orig_norm = 0.0f, new_norm = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        orig_norm += h_x[i].to_float() * h_x[i].to_float();
        new_norm += out_x[i].to_float() * out_x[i].to_float();
    }
    float norm_diff = std::fabs(std::sqrt(orig_norm) - std::sqrt(new_norm));
    std::cout << "       RoPE head norm preservation diff: " << norm_diff << std::endl;
    TEST_CHECK(norm_diff < 0.05f, "RoPE should preserve L2 norm of head vectors");

    cudaFree(d_x);
    cudaFree(d_freq);
    std::cout << "       PASS: RoPE Metal Kernel" << std::endl;
    return true;
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  MinnieTheMoEcher — Metal Backend Test Suite (Apple M6)  " << std::endl;
    std::cout << "==========================================================" << std::endl;

    if (!test_device_and_memory()) return 1;
    if (!test_rmsnorm()) return 1;
    if (!test_softmax_argmax()) return 1;
    if (!test_accelerate_gemm()) return 1;
    if (!test_silu_mul()) return 1;
    if (!test_rope()) return 1;

    std::cout << "==========================================================" << std::endl;
    std::cout << "  ALL TESTS PASSED ON APPLE SILICON METAL GPU!            " << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}
