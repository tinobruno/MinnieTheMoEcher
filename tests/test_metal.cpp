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


bool test_gemv_int4_f32() {
    std::cout << "[TEST] 7. INT4 GEMV f32 (LM Head)..." << std::endl;
    const int N = 64;
    const int K = 128;
    const int num_blocks = K / 32;

    std::vector<__nv_bfloat16> h_vec(K);
    for (int i = 0; i < K; i++) h_vec[i] = __nv_bfloat16::from_float(0.1f * ((i % 5) - 2));

    std::vector<uint8_t> h_weight(N * (K / 2));
    std::vector<__nv_bfloat16> h_scale(N * num_blocks);
    for (size_t i = 0; i < h_weight.size(); i++) h_weight[i] = (uint8_t)(i & 0xFF);
    for (size_t i = 0; i < h_scale.size(); i++) h_scale[i] = __nv_bfloat16::from_float(0.05f * ((i % 3) + 1));

    std::vector<float> ref_out(N, 0.0f);
    for (int r = 0; r < N; r++) {
        float sum = 0.0f;
        for (int b = 0; b < num_blocks; b++) {
            float s = h_scale[r * num_blocks + b].to_float();
            int w_off = r * (K / 2) + b * 16;
            int a_off = b * 32;
            float b_sum = 0.0f;
            for (int i = 0; i < 16; i++) {
                uint8_t byte_val = h_weight[w_off + i];
                float q0 = float(byte_val & 0x0F) - 8.0f;
                float q1 = float(byte_val >> 4) - 8.0f;
                b_sum += q0 * h_vec[a_off + i * 2].to_float() + q1 * h_vec[a_off + i * 2 + 1].to_float();
            }
            sum += b_sum * s;
        }
        ref_out[r] = sum;
    }

    std::vector<float> gpu_out(N, 0.0f);
    gemv_int4_f32_cuda(gpu_out.data(), h_vec.data(), h_weight.data(), h_scale.data(), N, K, 0);

    float max_err = 0.0f;
    for (int r = 0; r < N; r++) {
        float err = std::fabs(gpu_out[r] - ref_out[r]);
        if (err > max_err) max_err = err;
    }
    std::cout << "       Max INT4 GEMV f32 error: " << max_err << std::endl;
    TEST_CHECK(max_err < 1e-4f, "INT4 GEMV f32 error mismatch");
    std::cout << "       PASS: INT4 GEMV f32" << std::endl;
    return true;
}

bool test_gemv_int3_and_dequant() {
    std::cout << "[TEST] 8. INT3 GEMV & Dequantization..." << std::endl;
    const int N = 32;
    const int K = 64;
    const int num_blocks = K / 32; // 2 blocks
    const size_t bytes_per_row = (K * 3) / 8; // 24 bytes

    std::vector<__nv_bfloat16> h_vec(K);
    for (int i = 0; i < K; i++) h_vec[i] = __nv_bfloat16::from_float(0.2f * ((i % 7) - 3));

    std::vector<uint8_t> h_weight(N * bytes_per_row);
    std::vector<__nv_bfloat16> h_scale(N * num_blocks);
    for (size_t i = 0; i < h_weight.size(); i++) h_weight[i] = (uint8_t)((i * 37) & 0xFF);
    for (size_t i = 0; i < h_scale.size(); i++) h_scale[i] = __nv_bfloat16::from_float(0.1f * ((i % 4) + 1));

    // Dequantize to BF16
    std::vector<__nv_bfloat16> dequant_out(N * K);
    dequant_int3_block_cuda(dequant_out.data(), h_weight.data(), h_scale.data(), N, K, 32, 0);

    // Compute GEMV via gemv_int3_cuda
    std::vector<__nv_bfloat16> gemv_out(N);
    gemv_int3_cuda(gemv_out.data(), h_vec.data(), h_weight.data(), h_scale.data(), N, K, 0);

    // Compute expected result from dequantized weights
    float max_err = 0.0f;
    for (int r = 0; r < N; r++) {
        float sum = 0.0f;
        for (int c = 0; c < K; c++) {
            sum += dequant_out[r * K + c].to_float() * h_vec[c].to_float();
        }
        float err = std::fabs(gemv_out[r].to_float() - sum);
        if (err > max_err) max_err = err;
    }
    std::cout << "       Max INT3 GEMV vs Dequant error: " << max_err << std::endl;
    TEST_CHECK(max_err < 0.05f, "INT3 GEMV vs Dequant mismatch");
    std::cout << "       PASS: INT3 GEMV & Dequant" << std::endl;
    return true;
}

bool test_deltanet_recurrence() {
    std::cout << "[TEST] 9. DeltaNet Linear Attention recurrence..." << std::endl;
    const int num_k_heads = 2, num_v_heads = 6, head_dim = 16;
    const int channels = (2 * num_k_heads + num_v_heads) * head_dim; // 10 * 16 = 160
    const int M = 3;

    std::vector<__nv_bfloat16> in_qkv(M * channels);
    for (size_t i = 0; i < in_qkv.size(); i++) in_qkv[i] = __nv_bfloat16::from_float(std::sin(float(i)));

    std::vector<__nv_bfloat16> in_z(M * num_v_heads * head_dim);
    for (size_t i = 0; i < in_z.size(); i++) in_z[i] = __nv_bfloat16::from_float(0.5f);

    std::vector<__nv_bfloat16> in_a(M * num_v_heads, __nv_bfloat16::from_float(-1.0f));
    std::vector<__nv_bfloat16> in_b(M * num_v_heads, __nv_bfloat16::from_float(0.0f));
    std::vector<__nv_bfloat16> conv1d_w(channels * 4, __nv_bfloat16::from_float(0.25f));
    std::vector<__nv_bfloat16> in_conv_state(channels * 4, __nv_bfloat16::from_float(0.0f));
    std::vector<__nv_bfloat16> out_conv_state(channels * 4, __nv_bfloat16::from_float(0.0f));
    std::vector<__nv_bfloat16> A_log(num_v_heads, __nv_bfloat16::from_float(-0.5f));
    std::vector<__nv_bfloat16> dt_bias(num_v_heads, __nv_bfloat16::from_float(0.1f));
    std::vector<__nv_bfloat16> norm_w(head_dim, __nv_bfloat16::from_float(1.0f));
    std::vector<__nv_bfloat16> in_ssm_state(num_v_heads * head_dim * head_dim, __nv_bfloat16::from_float(0.0f));
    std::vector<__nv_bfloat16> out_ssm_state(num_v_heads * head_dim * head_dim, __nv_bfloat16::from_float(0.0f));
    std::vector<__nv_bfloat16> out(M * num_v_heads * head_dim, __nv_bfloat16::from_float(0.0f));

    deltanet_linear_attention_decode_batch_cuda(
        out.data(), in_qkv.data(), in_z.data(), in_a.data(), in_b.data(),
        conv1d_w.data(), in_conv_state.data(), out_conv_state.data(),
        nullptr, nullptr, nullptr, nullptr,
        A_log.data(), dt_bias.data(), norm_w.data(),
        in_ssm_state.data(), out_ssm_state.data(),
        nullptr, nullptr, nullptr, nullptr,
        num_k_heads, num_v_heads, head_dim, M, 0);

    // Verify non-zero output and reasonable bounded values
    float norm_sum = 0.0f;
    for (size_t i = 0; i < out.size(); i++) {
        float v = out[i].to_float();
        TEST_CHECK(!std::isnan(v) && !std::isinf(v), "DeltaNet output is NaN or Inf");
        norm_sum += v * v;
    }
    std::cout << "       DeltaNet output energy norm: " << std::sqrt(norm_sum) << std::endl;
    TEST_CHECK(norm_sum > 0.01f, "DeltaNet output is unexpectedly zero");
    std::cout << "       PASS: DeltaNet Recurrence" << std::endl;
    return true;
}


bool test_gemv_int4_gpu() {
    std::cout << "[TEST] 10. INT4 GEMV Metal GPU kernel (gemv_int4_cuda)..." << std::endl;
    const int N = 64;
    const int K = 128;
    const int num_blocks = K / 32;

    std::vector<__nv_bfloat16> h_vec(K);
    for (int i = 0; i < K; i++) h_vec[i] = __nv_bfloat16::from_float(0.1f * ((i % 5) - 2));

    std::vector<uint8_t> h_weight(N * (K / 2));
    std::vector<__nv_bfloat16> h_scale(N * num_blocks);
    for (size_t i = 0; i < h_weight.size(); i++) h_weight[i] = (uint8_t)(i & 0xFF);
    for (size_t i = 0; i < h_scale.size(); i++) h_scale[i] = __nv_bfloat16::from_float(0.05f * ((i % 3) + 1));

    std::vector<float> ref_out(N, 0.0f);
    for (int r = 0; r < N; r++) {
        float sum = 0.0f;
        for (int b = 0; b < num_blocks; b++) {
            float sc = h_scale[r * num_blocks + b].to_float();
            int w_off = r * (K / 2) + b * 16;
            int a_off = b * 32;
            float b_sum = 0.0f;
            for (int i = 0; i < 16; i++) {
                uint8_t byte_val = h_weight[w_off + i];
                float q0 = float(byte_val & 0x0F) - 8.0f;
                float q1 = float(byte_val >> 4) - 8.0f;
                b_sum += q0 * h_vec[a_off + i * 2].to_float() + q1 * h_vec[a_off + i * 2 + 1].to_float();
            }
            sum += b_sum * sc;
        }
        ref_out[r] = sum;
    }

    __nv_bfloat16 *d_out, *d_vec, *d_scale;
    uint8_t *d_weight;
    cudaMalloc((void**)&d_out, N * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_vec, K * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_weight, N * (K / 2));
    cudaMalloc((void**)&d_scale, N * num_blocks * sizeof(__nv_bfloat16));

    memcpy(d_vec, h_vec.data(), K * sizeof(__nv_bfloat16));
    memcpy(d_weight, h_weight.data(), N * (K / 2));
    memcpy(d_scale, h_scale.data(), N * num_blocks * sizeof(__nv_bfloat16));

    gemv_int4_cuda(d_out, d_vec, d_weight, d_scale, N, K, 0);

    float max_err = 0.0f;
    for (int r = 0; r < N; r++) {
        float gpu_val = d_out[r].to_float();
        float err = std::fabs(gpu_val - ref_out[r]);
        if (err > max_err) max_err = err;
    }
    std::cout << "       Max INT4 GPU kernel error: " << max_err << std::endl;
    TEST_CHECK(max_err < 0.1f, "INT4 GPU kernel mismatch");
    std::cout << "       PASS: INT4 GEMV Metal GPU kernel" << std::endl;

    // Now test residual version
    gemv_int4_residual_cuda(d_out, d_vec, d_weight, d_scale, N, K, 0);
    float max_res_err = 0.0f;
    for (int r = 0; r < N; r++) {
        float gpu_val = d_out[r].to_float();
        float err = std::fabs(gpu_val - 2.0f * ref_out[r]);
        if (err > max_res_err) max_res_err = err;
    }
    std::cout << "       Max INT4 GPU residual error: " << max_res_err << std::endl;
    TEST_CHECK(max_res_err < 0.2f, "INT4 GPU residual mismatch");
    std::cout << "       PASS: INT4 GEMV Metal GPU residual kernel" << std::endl;

    cudaFree(d_out); cudaFree(d_vec); cudaFree(d_weight); cudaFree(d_scale);
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
    if (!test_gemv_int4_f32()) return 1;
    if (!test_gemv_int3_and_dequant()) return 1;
    if (!test_deltanet_recurrence()) return 1;
    if (!test_gemv_int4_gpu()) return 1;

    std::cout << "==========================================================" << std::endl;
    std::cout << "  ALL TESTS PASSED ON APPLE SILICON METAL GPU!            " << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}
