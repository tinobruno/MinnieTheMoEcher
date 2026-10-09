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
    std::cout << "       gpu[0]=" << gpu_out[0].to_float() << " ref[0]=" << ref_out[0] << " gpu[1]=" << gpu_out[1].to_float() << " ref[1]=" << ref_out[1] << std::endl;
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

    float* d_out;
    __nv_bfloat16* d_vec;
    uint8_t* d_weight;
    __nv_bfloat16* d_scale;
    cudaMalloc((void**)&d_out, N * sizeof(float));
    cudaMalloc((void**)&d_vec, K * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_weight, N * (K / 2));
    cudaMalloc((void**)&d_scale, N * num_blocks * sizeof(__nv_bfloat16));

    memcpy(d_vec, h_vec.data(), K * sizeof(__nv_bfloat16));
    memcpy(d_weight, h_weight.data(), N * (K / 2));
    memcpy(d_scale, h_scale.data(), N * num_blocks * sizeof(__nv_bfloat16));

    gemv_int4_f32_cuda(d_out, d_vec, d_weight, d_scale, N, K, 0);
    cudaDeviceSynchronize();

    float max_err = 0.0f;
    for (int r = 0; r < N; r++) {
        float err = std::fabs(d_out[r] - ref_out[r]);
        if (err > max_err) max_err = err;
    }
    std::cout << "       Max INT4 GEMV f32 (GPU) error: " << max_err << std::endl;
    cudaFree(d_out); cudaFree(d_vec); cudaFree(d_weight); cudaFree(d_scale);
    TEST_CHECK(max_err < 1e-3f, "INT4 GEMV f32 GPU error mismatch");
    std::cout << "       PASS: INT4 GEMV f32 (GPU)" << std::endl;
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
    std::cout << "[TEST] 9. DeltaNet Linear Attention recurrence (Metal GPU + rollback slots)..." << std::endl;
    const int num_k_heads = 4, num_v_heads = 12, head_dim = 128;
    const int channels = (2 * num_k_heads + num_v_heads) * head_dim; // 20 * 128 = 2560
    const int M = 3;

    __nv_bfloat16 *d_qkv = nullptr, *d_z = nullptr, *d_a = nullptr, *d_b = nullptr, *d_cw = nullptr;
    __nv_bfloat16 *d_ics = nullptr, *d_ocs = nullptr, *d_alog = nullptr, *d_dt = nullptr, *d_nw = nullptr;
    __nv_bfloat16 *d_issm = nullptr, *d_ossm = nullptr, *d_out = nullptr;
    __nv_bfloat16 *d_sconv[4] = {nullptr, nullptr, nullptr, nullptr};
    __nv_bfloat16 *d_sssm[4] = {nullptr, nullptr, nullptr, nullptr};

    cudaMalloc((void**)&d_qkv, M * channels * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_z, M * num_v_heads * head_dim * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_a, M * num_v_heads * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_b, M * num_v_heads * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_cw, channels * 4 * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_ics, channels * 4 * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_ocs, channels * 4 * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_alog, num_v_heads * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_dt, num_v_heads * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_nw, head_dim * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_issm, num_v_heads * head_dim * head_dim * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_ossm, num_v_heads * head_dim * head_dim * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_out, M * num_v_heads * head_dim * sizeof(__nv_bfloat16));

    for (int k = 0; k < 3; k++) {
        cudaMalloc((void**)&d_sconv[k], channels * 4 * sizeof(__nv_bfloat16));
        cudaMalloc((void**)&d_sssm[k], num_v_heads * head_dim * head_dim * sizeof(__nv_bfloat16));
    }

    for (int i = 0; i < M * channels; i++) d_qkv[i] = __nv_bfloat16::from_float(std::sin(float(i)));
    for (int i = 0; i < M * num_v_heads * head_dim; i++) d_z[i] = __nv_bfloat16::from_float(0.5f);
    for (int i = 0; i < M * num_v_heads; i++) d_a[i] = __nv_bfloat16::from_float(-1.0f);
    for (int i = 0; i < M * num_v_heads; i++) d_b[i] = __nv_bfloat16::from_float(0.0f);
    for (int i = 0; i < channels * 4; i++) d_cw[i] = __nv_bfloat16::from_float(0.25f);
    for (int i = 0; i < channels * 4; i++) d_ics[i] = __nv_bfloat16::from_float(0.0f);
    for (int i = 0; i < num_v_heads; i++) d_alog[i] = __nv_bfloat16::from_float(-0.5f);
    for (int i = 0; i < num_v_heads; i++) d_dt[i] = __nv_bfloat16::from_float(0.1f);
    for (int i = 0; i < head_dim; i++) d_nw[i] = __nv_bfloat16::from_float(1.0f);
    for (int i = 0; i < num_v_heads * head_dim * head_dim; i++) d_issm[i] = __nv_bfloat16::from_float(0.0f);

    deltanet_linear_attention_decode_batch_cuda(
        d_out, d_qkv, d_z, d_a, d_b,
        d_cw, d_ics, d_ocs,
        d_sconv[0], d_sconv[1], d_sconv[2], nullptr,
        d_alog, d_dt, d_nw,
        d_issm, d_ossm,
        d_sssm[0], d_sssm[1], d_sssm[2], nullptr,
        num_k_heads, num_v_heads, head_dim, M, 0);
    cudaStreamSynchronize(0);

    // Verify GPU output and reasonable bounded values
    float norm_sum = 0.0f;
    for (int i = 0; i < M * num_v_heads * head_dim; i++) {
        float v = d_out[i].to_float();
        TEST_CHECK(!std::isnan(v) && !std::isinf(v), "DeltaNet output is NaN or Inf");
        norm_sum += v * v;
    }
    std::cout << "       DeltaNet GPU output energy norm: " << std::sqrt(norm_sum) << std::endl;
    TEST_CHECK(norm_sum > 0.01f, "DeltaNet GPU output is unexpectedly zero");

    // Verify rollback slots were written
    float sconv0_norm = 0.0f, sssm0_norm = 0.0f;
    for (int i = 0; i < channels * 4; i++) {
        float v = d_sconv[0][i].to_float();
        sconv0_norm += v * v;
    }
    for (int i = 0; i < num_v_heads * head_dim * head_dim; i++) {
        float v = d_sssm[0][i].to_float();
        sssm0_norm += v * v;
    }
    std::cout << "       DeltaNet slot_conv_0 energy norm: " << std::sqrt(sconv0_norm) << std::endl;
    std::cout << "       DeltaNet slot_ssm_0 energy norm: " << std::sqrt(sssm0_norm) << std::endl;
    TEST_CHECK(sconv0_norm > 0.01f, "DeltaNet slot_conv_0 is unexpectedly zero");
    TEST_CHECK(sssm0_norm > 0.01f, "DeltaNet slot_ssm_0 is unexpectedly zero");

    cudaFree(d_qkv); cudaFree(d_z); cudaFree(d_a); cudaFree(d_b); cudaFree(d_cw);
    cudaFree(d_ics); cudaFree(d_ocs); cudaFree(d_alog); cudaFree(d_dt); cudaFree(d_nw);
    cudaFree(d_issm); cudaFree(d_ossm); cudaFree(d_out);
    for (int k = 0; k < 3; k++) {
        cudaFree(d_sconv[k]);
        cudaFree(d_sssm[k]);
    }

    std::cout << "       PASS: DeltaNet Recurrence & Rollback Slots on Metal GPU" << std::endl;
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
    cudaDeviceSynchronize();

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
    cudaDeviceSynchronize();
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


bool test_gemv_int3_gpu() {
    std::cout << "[TEST] 11. INT3 GEMV Metal GPU kernel (gemv_int3_cuda)..." << std::endl;
    const int N = 32;
    const int K = 64;
    const int num_blocks = K / 32;
    const size_t bytes_per_row = (K * 3) / 8;

    std::vector<__nv_bfloat16> h_vec(K);
    for (int i = 0; i < K; i++) h_vec[i] = __nv_bfloat16::from_float(0.2f * ((i % 7) - 3));

    std::vector<uint8_t> h_weight(N * bytes_per_row);
    std::vector<__nv_bfloat16> h_scale(N * num_blocks);
    for (size_t i = 0; i < h_weight.size(); i++) h_weight[i] = (uint8_t)((i * 37) & 0xFF);
    for (size_t i = 0; i < h_scale.size(); i++) h_scale[i] = __nv_bfloat16::from_float(0.1f * ((i % 4) + 1));

    std::vector<float> ref_out(N, 0.0f);
    for (int r = 0; r < N; r++) {
        float sum = 0.0f;
        for (int b = 0; b < num_blocks; b++) {
            float s = h_scale[r * num_blocks + b].to_float();
            const uint8_t* blk_w = h_weight.data() + r * bytes_per_row + b * 12;
            int in_idx = b * 32;
            for (int i = 0; i < 4; i++) {
                uint8_t b0 = blk_w[i * 3 + 0], b1 = blk_w[i * 3 + 1], b2 = blk_w[i * 3 + 2];
                float w0 = ((float)(b0 & 0x07) - 4.0f) * s;
                float w1 = ((float)((b0 >> 3) & 0x07) - 4.0f) * s;
                float w2 = ((float)((b0 >> 6) | ((b1 & 0x01) << 2)) - 4.0f) * s;
                float w3 = ((float)((b1 >> 1) & 0x07) - 4.0f) * s;
                float w4 = ((float)((b1 >> 4) & 0x07) - 4.0f) * s;
                float w5 = ((float)((b1 >> 7) | ((b2 & 0x03) << 1)) - 4.0f) * s;
                float w6 = ((float)((b2 >> 2) & 0x07) - 4.0f) * s;
                float w7 = ((float)((b2 >> 5) & 0x07) - 4.0f) * s;
                sum += w0 * h_vec[in_idx + i * 8 + 0].to_float() +
                       w1 * h_vec[in_idx + i * 8 + 1].to_float() +
                       w2 * h_vec[in_idx + i * 8 + 2].to_float() +
                       w3 * h_vec[in_idx + i * 8 + 3].to_float() +
                       w4 * h_vec[in_idx + i * 8 + 4].to_float() +
                       w5 * h_vec[in_idx + i * 8 + 5].to_float() +
                       w6 * h_vec[in_idx + i * 8 + 6].to_float() +
                       w7 * h_vec[in_idx + i * 8 + 7].to_float();
            }
        }
        ref_out[r] = sum;
    }

    __nv_bfloat16 *d_out, *d_vec, *d_scale;
    uint8_t *d_weight;
    cudaMalloc((void**)&d_out, N * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_vec, K * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_weight, N * bytes_per_row);
    cudaMalloc((void**)&d_scale, N * num_blocks * sizeof(__nv_bfloat16));

    memcpy(d_vec, h_vec.data(), K * sizeof(__nv_bfloat16));
    memcpy(d_weight, h_weight.data(), N * bytes_per_row);
    memcpy(d_scale, h_scale.data(), N * num_blocks * sizeof(__nv_bfloat16));

    gemv_int3_cuda(d_out, d_vec, d_weight, d_scale, N, K, 0);
    cudaDeviceSynchronize();

    float max_err = 0.0f;
    for (int r = 0; r < N; r++) {
        float gpu_val = d_out[r].to_float();
        float err = std::fabs(gpu_val - ref_out[r]);
        if (err > max_err) max_err = err;
    }
    std::cout << "       Max INT3 GPU kernel error: " << max_err << std::endl;
    TEST_CHECK(max_err < 0.05f, "INT3 GPU kernel mismatch");
    std::cout << "       PASS: INT3 GEMV Metal GPU kernel" << std::endl;

    gemv_int3_residual_cuda(d_out, d_vec, d_weight, d_scale, N, K, 0);
    cudaDeviceSynchronize();

    float max_res_err = 0.0f;
    for (int r = 0; r < N; r++) {
        float gpu_val = d_out[r].to_float();
        float err = std::fabs(gpu_val - 2.0f * ref_out[r]);
        if (err > max_res_err) max_res_err = err;
    }
    std::cout << "       Max INT3 GPU residual error: " << max_res_err << std::endl;
    TEST_CHECK(max_res_err < 0.1f, "INT3 GPU residual mismatch");
    std::cout << "       PASS: INT3 GEMV Metal GPU residual kernel" << std::endl;

    cudaFree(d_out); cudaFree(d_vec); cudaFree(d_weight); cudaFree(d_scale);
    return true;
}


bool test_gqa_attention() {
    std::cout << "[TEST] 12. GQA Attention Metal compute kernel..." << std::endl;
    int n_q_heads = 24, n_kv_heads = 4, head_dim = 256, max_seq = 64;
    size_t q_bytes = 2 * n_q_heads * head_dim * sizeof(__nv_bfloat16);
    size_t kv_bytes = n_kv_heads * head_dim * sizeof(__nv_bfloat16);
    size_t out_bytes = n_q_heads * head_dim * sizeof(__nv_bfloat16);
    size_t cache_bytes = max_seq * n_kv_heads * head_dim;

    __nv_bfloat16* qg = (__nv_bfloat16*)nullptr; cudaMalloc((void**)&qg, q_bytes);
    __nv_bfloat16* k = (__nv_bfloat16*)nullptr; cudaMalloc((void**)&k, kv_bytes);
    __nv_bfloat16* v = (__nv_bfloat16*)nullptr; cudaMalloc((void**)&v, kv_bytes);
    __nv_bfloat16* qn = (__nv_bfloat16*)nullptr; cudaMalloc((void**)&qn, head_dim * sizeof(__nv_bfloat16));
    __nv_bfloat16* kn = (__nv_bfloat16*)nullptr; cudaMalloc((void**)&kn, head_dim * sizeof(__nv_bfloat16));
    uint8_t* k_cache = (uint8_t*)nullptr; cudaMalloc((void**)&k_cache, cache_bytes);
    uint8_t* v_cache = (uint8_t*)nullptr; cudaMalloc((void**)&v_cache, cache_bytes);
    __nv_bfloat16* out = (__nv_bfloat16*)nullptr; cudaMalloc((void**)&out, out_bytes);

    for (size_t i = 0; i < 2 * n_q_heads * head_dim; i++) qg[i] = __nv_bfloat16::from_float(0.01f * (i % 10));
    for (size_t i = 0; i < n_kv_heads * head_dim; i++) {
        k[i] = __nv_bfloat16::from_float(0.02f * (i % 7));
        v[i] = __nv_bfloat16::from_float(0.03f * (i % 5));
    }
    for (int i = 0; i < head_dim; i++) {
        qn[i] = __nv_bfloat16::from_float(0.0f);
        kn[i] = __nv_bfloat16::from_float(0.0f);
    }
    cudaMemset(k_cache, 0, cache_bytes);
    cudaMemset(v_cache, 0, cache_bytes);
    cudaMemset(out, 0, out_bytes);

    qwen_gqa_decode_gated_fp8_batch_cuda(
        out, qg, k, v, qn, kn, k_cache, v_cache,
        n_q_heads, n_kv_heads, head_dim, nullptr, 0, 1, max_seq, 10000.0f, 1e-6f, nullptr);
    metal_stream_synchronize(nullptr);

    float norm = 0.0f;
    for (size_t i = 0; i < n_q_heads * head_dim; i++) {
        float val = out[i].to_float();
        TEST_CHECK(!std::isnan(val) && !std::isinf(val), "GQA output is NaN or Inf");
        norm += val * val;
    }
    std::cout << "       GQA Attention output norm: " << std::sqrt(norm) << std::endl;
    TEST_CHECK(norm > 0.001f, "GQA output norm unexpectedly zero");
    std::cout << "       PASS: GQA Attention Metal Kernel" << std::endl;
    cudaFree(qg); cudaFree(k); cudaFree(v); cudaFree(qn); cudaFree(kn); cudaFree(k_cache); cudaFree(v_cache); cudaFree(out);
    return true;
}

bool test_hc_and_sinkhorn() {
    std::cout << "[TEST] 13. Hyper-Connections (HC) Pre-Norm, Sinkhorn, and Update..." << std::endl;
    const int hc = 4;
    const int mix_size = 4;
    const int hc_dim = 256;
    const float eps = 1e-6f;

    float* d_mixes;
    __nv_bfloat16* d_hc_state;
    float* d_hc_fn;
    cudaMalloc((void**)&d_mixes, mix_size * sizeof(float));
    cudaMalloc((void**)&d_hc_state, hc * hc_dim * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_hc_fn, mix_size * hc_dim * sizeof(float));

    for (int i = 0; i < hc * hc_dim; i++) {
        d_hc_state[i] = __nv_bfloat16::from_float(0.1f * ((i % 11) - 5));
    }
    for (int i = 0; i < mix_size * hc_dim; i++) {
        d_hc_fn[i] = 0.05f * ((i % 7) - 3);
    }

    gemv_hc_pre_norm_cuda(d_mixes, d_hc_state, d_hc_fn, mix_size, hc_dim, eps, nullptr);
    metal_stream_synchronize(nullptr);

    for (int i = 0; i < mix_size; i++) {
        float m = d_mixes[i];
        TEST_CHECK(!std::isnan(m) && !std::isinf(m), "HC pre-norm produced NaN or Inf");
    }

    // Split Sinkhorn
    const int mix_total = (2 + hc) * hc; // 24
    float *d_sink_mixes, *d_pre, *d_post, *d_comb, *d_scale, *d_base;
    cudaMalloc((void**)&d_sink_mixes, mix_total * sizeof(float));
    cudaMalloc((void**)&d_pre, hc * sizeof(float));
    cudaMalloc((void**)&d_post, hc * sizeof(float));
    cudaMalloc((void**)&d_comb, hc * hc * sizeof(float));
    cudaMalloc((void**)&d_scale, 3 * sizeof(float));
    cudaMalloc((void**)&d_base, mix_total * sizeof(float));
    for (int i = 0; i < mix_total; i++) {
        d_sink_mixes[i] = 0.1f * ((i % 5) - 2);
        d_base[i] = 0.0f;
    }
    d_scale[0] = 1.0f; d_scale[1] = 1.0f; d_scale[2] = 1.0f;

    hc_split_sinkhorn_cuda(d_pre, d_post, d_comb, d_sink_mixes, d_scale, d_base, hc, 20, eps, nullptr);
    metal_stream_synchronize(nullptr);

    for (int i = 0; i < hc; i++) {
        TEST_CHECK(!std::isnan(d_pre[i]) && !std::isinf(d_pre[i]), "Sinkhorn pre NaN/Inf");
        TEST_CHECK(!std::isnan(d_post[i]) && !std::isinf(d_post[i]), "Sinkhorn post NaN/Inf");
        TEST_CHECK(d_pre[i] > 0.0f && d_pre[i] < 1.0f + eps, "Sinkhorn pre weights out of sigmoid range");
    }

    // Verify comb row sums = 1.0 (doubly stochastic)
    for (int r = 0; r < hc; r++) {
        float r_sum = 0.0f;
        for (int c = 0; c < hc; c++) r_sum += d_comb[r * hc + c];
        TEST_CHECK(std::fabs(r_sum - 1.0f) < 0.05f, "Sinkhorn comb row sum must be ~1.0");
    }
    std::cout << "       Sinkhorn comb matrix is doubly-stochastic (row sum ~ 1.0)" << std::endl;

    // Pre-weighted add + norm
    __nv_bfloat16 *d_out, *d_norm_w;
    cudaMalloc((void**)&d_out, hc_dim * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_norm_w, hc_dim * sizeof(__nv_bfloat16));
    for (int i = 0; i < hc_dim; i++) d_norm_w[i] = __nv_bfloat16::from_float(1.0f);

    hc_pre_weighted_add_norm_cuda(d_out, d_hc_state, d_pre, d_norm_w, hc_dim, hc, eps, nullptr);
    metal_stream_synchronize(nullptr);

    float norm_sq = 0.0f;
    for (int i = 0; i < hc_dim; i++) {
        float val = d_out[i].to_float();
        TEST_CHECK(!std::isnan(val) && !std::isinf(val), "HC norm produced NaN or Inf");
        norm_sq += val * val;
    }
    float rms = std::sqrt(norm_sq / float(hc_dim));
    std::cout << "       HC pre-weighted add RMS: " << rms << std::endl;
    TEST_CHECK(std::fabs(rms - 1.0f) < 0.15f, "RMSNorm of HC output should be close to 1.0");
    std::cout << "       PASS: Hyper-Connections (HC) Pre-Norm & Sinkhorn" << std::endl;

    cudaFree(d_mixes); cudaFree(d_sink_mixes); cudaFree(d_hc_state); cudaFree(d_hc_fn);
    cudaFree(d_pre); cudaFree(d_post); cudaFree(d_comb); cudaFree(d_scale); cudaFree(d_base);
    cudaFree(d_out); cudaFree(d_norm_w);
    return true;
}

bool test_mla_attention_fused() {
    std::cout << "[TEST] 14. MLA Attention Fused (64 heads, 512 head_dim)..." << std::endl;
    const int n_heads = 64;
    const int head_dim = 512;
    const int rope_dim = 64;
    const int window = 128;
    const int max_comp = 64;
    const int max_cache_len = window + max_comp; // 192

    __nv_bfloat16 *d_q, *d_raw_kv, *d_comp_kv, *d_out;
    float *d_sink, *d_freqs;
    int32_t *d_pos, *d_comp_cnt;
    cudaMalloc((void**)&d_q, n_heads * head_dim * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_raw_kv, window * head_dim * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_comp_kv, max_comp * head_dim * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_sink, n_heads * sizeof(float));
    cudaMalloc((void**)&d_out, n_heads * head_dim * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_pos, sizeof(int32_t));
    cudaMalloc((void**)&d_comp_cnt, sizeof(int32_t));
    cudaMalloc((void**)&d_freqs, 65536 * (rope_dim / 2) * 2 * sizeof(float));

    *d_pos = 10;
    *d_comp_cnt = 0;

    for (int i = 0; i < n_heads * head_dim; i++) {
        d_q[i] = __nv_bfloat16::from_float(0.01f * ((i % 13) - 6));
    }
    for (int i = 0; i < window * head_dim; i++) {
        d_raw_kv[i] = __nv_bfloat16::from_float(0.02f * ((i % 17) - 8));
    }
    for (int i = 0; i < max_comp * head_dim; i++) {
        d_comp_kv[i] = __nv_bfloat16::from_float(0.0f);
    }
    for (int i = 0; i < n_heads; i++) {
        d_sink[i] = -10.0f;
    }
    // Simple identity RoPE freqs (cos=1, sin=0)
    for (int i = 0; i < 65536 * (rope_dim / 2) * 2; i += 2) {
        d_freqs[i] = 1.0f;
        d_freqs[i + 1] = 0.0f;
    }

    float scale = 1.0f / std::sqrt((float)head_dim);
    mla_attention_fused_cuda(
        d_q, d_raw_kv, d_comp_kv, d_sink, d_out,
        d_pos, d_comp_cnt, d_freqs, max_cache_len,
        head_dim, rope_dim, scale, 1e-6f,
        nullptr, window, nullptr);
    metal_stream_synchronize(nullptr);

    float norm = 0.0f;
    for (int i = 0; i < n_heads * head_dim; i++) {
        float v = d_out[i].to_float();
        TEST_CHECK(!std::isnan(v) && !std::isinf(v), "MLA attention produced NaN or Inf");
        norm += v * v;
    }
    std::cout << "       MLA Attention output norm: " << std::sqrt(norm) << std::endl;
    TEST_CHECK(norm > 0.01f, "MLA attention output norm unexpectedly zero");
    std::cout << "       PASS: MLA Attention Fused Kernel" << std::endl;

    cudaFree(d_q); cudaFree(d_raw_kv); cudaFree(d_comp_kv); cudaFree(d_sink);
    cudaFree(d_out); cudaFree(d_pos); cudaFree(d_comp_cnt); cudaFree(d_freqs);
    return true;
}

bool test_gemv_iq2_xxs_and_swiglu() {
    std::cout << "[TEST] 15. IQ2_XXS SwiGLU Fused GEMV..." << std::endl;
    const int N = 64;
    const int K = 256;
    const int n_blocks = K / 256;

    __nv_bfloat16 *d_out, *d_vec;
    block_iq2_xxs *d_w1, *d_w3;
    cudaMalloc((void**)&d_out, N * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_vec, K * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_w1, N * n_blocks * sizeof(block_iq2_xxs));
    cudaMalloc((void**)&d_w3, N * n_blocks * sizeof(block_iq2_xxs));

    for (int i = 0; i < K; i++) {
        d_vec[i] = __nv_bfloat16::from_float(0.1f * ((i % 5) - 2));
    }
    for (int r = 0; r < N; r++) {
        d_w1[r].d = half::from_float(0.25f);
        d_w3[r].d = half::from_float(0.25f);
        for (int q = 0; q < 32; q++) {
            d_w1[r].qs[q] = (uint16_t)((r * 31 + q * 17) & 0xFFFF);
            d_w3[r].qs[q] = (uint16_t)((r * 43 + q * 23) & 0xFFFF);
        }
    }

    gemv_iq2_xxs_swiglu_fused_cuda(d_out, d_vec, d_w1, d_w3, N, K, 0.0f, nullptr);
    metal_stream_synchronize(nullptr);

    float norm = 0.0f;
    for (int i = 0; i < N; i++) {
        float v = d_out[i].to_float();
        TEST_CHECK(!std::isnan(v) && !std::isinf(v), "IQ2_XXS SwiGLU produced NaN or Inf");
        norm += v * v;
    }
    std::cout << "       IQ2_XXS SwiGLU output energy: " << std::sqrt(norm) << std::endl;
    TEST_CHECK(norm > 0.001f, "IQ2_XXS SwiGLU output unexpectedly zero");
    std::cout << "       PASS: IQ2_XXS SwiGLU Fused Kernel" << std::endl;

    cudaFree(d_out); cudaFree(d_vec); cudaFree(d_w1); cudaFree(d_w3);
    return true;
}

bool test_gemv_q2_k() {
    std::cout << "[TEST] 16. Q2_K GEMV Down-Projection..." << std::endl;
    const int N = 64;
    const int K = 256;
    const int n_blocks = K / 256;

    __nv_bfloat16 *d_out, *d_vec;
    block_q2_K *d_weight;
    cudaMalloc((void**)&d_out, N * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_vec, K * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_weight, N * n_blocks * sizeof(block_q2_K));

    for (int i = 0; i < K; i++) {
        d_vec[i] = __nv_bfloat16::from_float(0.1f * ((i % 7) - 3));
    }
    for (int r = 0; r < N; r++) {
        d_weight[r].d = half::from_float(0.5f);
        d_weight[r].dmin = half::from_float(0.1f);
        for (int s = 0; s < 16; s++) d_weight[r].scales[s] = 0x22;
        for (int q = 0; q < 64; q++) d_weight[r].qs[q] = (uint8_t)((r * 19 + q) & 0xFF);
    }

    gemv_q2_k_cuda(d_out, d_vec, d_weight, N, K, nullptr);
    metal_stream_synchronize(nullptr);

    // Compute CPU reference GEMV
    float max_err = 0.0f;
    for (int r = 0; r < N; r++) {
        float ref_sum = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            const block_q2_K& blk = d_weight[r * n_blocks + b];
            float d = blk.d.to_float();
            float min = blk.dmin.to_float();
            for (int group = 0; group < 16; group++) {
                uint8_t sc = blk.scales[group];
                float dl = d * (float)(sc & 0x0F);
                float ml = min * (float)(sc >> 4);
                int q_base = 32 * (group / 8) + 16 * (group & 1);
                int shift = ((group / 2) & 3) * 2;
                for (int l = 0; l < 16; l++) {
                    uint8_t q = (blk.qs[q_base + l] >> shift) & 0x03;
                    float w = dl * (float)q - ml;
                    int col = b * 256 + group * 16 + l;
                    ref_sum += w * d_vec[col].to_float();
                }
            }
        }
        float gpu_val = d_out[r].to_float();
        float err = std::abs(gpu_val - ref_sum);
        if (err > max_err) max_err = err;
    }
    std::cout << "       Max Q2_K GEMV vs CPU error: " << max_err << std::endl;
    TEST_CHECK(max_err < 0.1f, "Q2_K GEMV Metal kernel produced excessive error vs reference");
    std::cout << "       PASS: Q2_K GEMV Metal Kernel" << std::endl;

    cudaFree(d_out); cudaFree(d_vec); cudaFree(d_weight);
    return true;
}

bool test_fp8_dequant_and_gemv() {
    std::cout << "[TEST] 17. FP8 GEMV & Dequantization..." << std::endl;
    const int rows = 64;
    const int cols = 128;
    const int block_size = 128;

    __nv_bfloat16 *d_dequant, *d_out, *d_vec;
    uint8_t *d_weight, *d_scale;
    cudaMalloc((void**)&d_dequant, rows * cols * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_out, rows * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_vec, cols * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_weight, rows * cols);
    cudaMalloc((void**)&d_scale, rows * (cols / block_size));

    for (int i = 0; i < cols; i++) d_vec[i] = __nv_bfloat16::from_float(0.1f * ((i % 5) - 2));
    for (int i = 0; i < rows * cols; i++) d_weight[i] = (uint8_t)(0x38 + (i % 8)); // Valid non-zero FP8 E4M3 values
    for (int i = 0; i < rows; i++) d_scale[i] = 0x7F; // 2^0 = 1.0f in E8M0

    fp8_dequant_cuda(d_dequant, d_weight, d_scale, rows, cols, block_size, nullptr);
    gemv_fp8_cuda(d_out, d_vec, d_weight, d_scale, rows, cols, block_size, nullptr);
    metal_stream_synchronize(nullptr);

    float dequant_norm = 0.0f;
    for (int i = 0; i < rows * cols; i++) {
        float v = d_dequant[i].to_float();
        TEST_CHECK(!std::isnan(v) && !std::isinf(v), "FP8 dequant produced NaN or Inf");
        dequant_norm += v * v;
    }
    std::cout << "       FP8 Dequant matrix norm: " << std::sqrt(dequant_norm) << std::endl;
    TEST_CHECK(dequant_norm > 0.01f, "FP8 dequant norm unexpectedly zero");

    float gemv_norm = 0.0f;
    for (int i = 0; i < rows; i++) {
        float v = d_out[i].to_float();
        TEST_CHECK(!std::isnan(v) && !std::isinf(v), "FP8 GEMV produced NaN or Inf");
        gemv_norm += v * v;
    }
    std::cout << "       FP8 GEMV vector norm: " << std::sqrt(gemv_norm) << std::endl;
    TEST_CHECK(gemv_norm > 0.01f, "FP8 GEMV norm unexpectedly zero");

    // Microbenchmark with DeepSeek-V4 realistic dimensions: 2048 x 4096
    const int b_rows = 2048, b_cols = 4096;
    __nv_bfloat16 *b_out, *b_vec;
    uint8_t *b_weight, *b_scale;
    cudaMalloc((void**)&b_out, b_rows * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&b_vec, b_cols * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&b_weight, b_rows * b_cols);
    cudaMalloc((void**)&b_scale, (b_rows / block_size) * (b_cols / block_size));
    for (int i = 0; i < 5; i++) {
        gemv_fp8_cuda(b_out, b_vec, b_weight, b_scale, b_rows, b_cols, block_size, nullptr);
    }
    metal_stream_synchronize(nullptr);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 50; i++) {
        gemv_fp8_cuda(b_out, b_vec, b_weight, b_scale, b_rows, b_cols, block_size, nullptr);
    }
    metal_stream_synchronize(nullptr);
    // Benchmark wq_b dimensions: 32768 x 1536
    const int wq_rows = 32768, wq_cols = 1536;
    __nv_bfloat16 *wq_out, *wq_vec;
    uint8_t *wq_weight, *wq_scale;
    cudaMalloc((void**)&wq_out, wq_rows * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&wq_vec, wq_cols * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&wq_weight, wq_rows * wq_cols);
    cudaMalloc((void**)&wq_scale, (wq_rows / block_size) * (wq_cols / block_size));
    metal_stream_synchronize(nullptr);
    auto t_wq0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        gemv_fp8_cuda(wq_out, wq_vec, wq_weight, wq_scale, wq_rows, wq_cols, block_size, nullptr);
    }
    metal_stream_synchronize(nullptr);
    auto t_wq1 = std::chrono::high_resolution_clock::now();
    double wq_ms = std::chrono::duration<double, std::milli>(t_wq1 - t_wq0).count() / 10.0;
    std::cout << "       FP8 GEMV wq_b (32768x1536) benchmark: " << wq_ms << " ms" << std::endl;
    cudaFree(wq_out); cudaFree(wq_vec); cudaFree(wq_weight); cudaFree(wq_scale);

    // Benchmark wo_b dimensions: 4096 x 8192
    const int wob_rows = 4096, wob_cols = 8192;
    __nv_bfloat16 *wob_out, *wob_vec;
    uint8_t *wob_weight, *wob_scale;
    cudaMalloc((void**)&wob_out, wob_rows * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&wob_vec, wob_cols * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&wob_weight, wob_rows * wob_cols);
    cudaMalloc((void**)&wob_scale, (wob_rows / block_size) * (wob_cols / block_size));
    metal_stream_synchronize(nullptr);
    auto t_wob0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        gemv_fp8_cuda(wob_out, wob_vec, wob_weight, wob_scale, wob_rows, wob_cols, block_size, nullptr);
    }
    metal_stream_synchronize(nullptr);
    auto t_wob1 = std::chrono::high_resolution_clock::now();
    double wob_ms = std::chrono::duration<double, std::milli>(t_wob1 - t_wob0).count() / 10.0;
    std::cout << "       FP8 GEMV wo_b (4096x8192) benchmark: " << wob_ms << " ms" << std::endl;
    cudaFree(wob_out); cudaFree(wob_vec); cudaFree(wob_weight); cudaFree(wob_scale);
    std::cout << "       PASS: FP8 Dequant & GEMV Metal Kernels" << std::endl;

    cudaFree(d_dequant); cudaFree(d_out); cudaFree(d_vec); cudaFree(d_weight); cudaFree(d_scale);
    return true;
}

bool test_fused_moe_accum() {
    std::cout << "[TEST] 18. Fused 6-way MoE Dynamic Accumulation..." << std::endl;
    const int dim = 4096;
    __nv_bfloat16 *d_accum, *d_down_buf, *d_shared_down;
    float *d_topk_weights;
    cudaMalloc((void**)&d_accum, dim * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_down_buf, 6 * dim * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_shared_down, dim * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_topk_weights, 6 * sizeof(float));

    float weights[6] = {0.25f, 0.20f, 0.15f, 0.15f, 0.15f, 0.10f};
    for (int k = 0; k < 6; k++) d_topk_weights[k] = weights[k];
    for (int i = 0; i < dim; i++) {
        d_shared_down[i] = __nv_bfloat16::from_float(0.05f * ((i % 5) - 2));
        for (int k = 0; k < 6; k++) {
            d_down_buf[k * dim + i] = __nv_bfloat16::from_float(0.1f * ((i + k) % 7 - 3));
        }
    }

    fused_moe_accum_dynamic_cuda(d_accum, d_down_buf, d_topk_weights, d_shared_down, dim, nullptr);
    metal_stream_synchronize(nullptr);

    float max_err = 0.0f;
    for (int i = 0; i < dim; i++) {
        float ref = d_shared_down[i].to_float();
        for (int k = 0; k < 6; k++) {
            ref += d_down_buf[k * dim + i].to_float() * weights[k];
        }
        float gpu_val = d_accum[i].to_float();
        float err = std::abs(gpu_val - ref);
        if (err > max_err) max_err = err;
    }
    std::cout << "       Max MoE Accumulation vs CPU error: " << max_err << std::endl;
    TEST_CHECK(max_err < 0.05f, "Fused MoE accumulation error exceeds tolerance");
    std::cout << "       PASS: Fused MoE Dynamic Accumulation Metal Kernel" << std::endl;

    cudaFree(d_accum); cudaFree(d_down_buf); cudaFree(d_shared_down); cudaFree(d_topk_weights);
    return true;
}

bool test_moe_top6_routing() {
    std::cout << "[TEST] 19. GPU-Accelerated MoE Top-6 Routing Kernel..." << std::endl;
    const int M = 8;
    const int n_experts = 256;
    const int top_k = 6;
    const float routed_scaling = 2.5f;

    __nv_bfloat16* d_scores;
    float* d_bias;
    int32_t* d_topk_ids;
    float* d_topk_weights;

    cudaMalloc((void**)&d_scores, M * n_experts * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_bias, n_experts * sizeof(float));
    cudaMalloc((void**)&d_topk_ids, M * top_k * sizeof(int32_t));
    cudaMalloc((void**)&d_topk_weights, M * top_k * sizeof(float));

    std::vector<__nv_bfloat16> h_scores(M * n_experts);
    std::vector<float> h_bias(n_experts);
    for (int i = 0; i < n_experts; i++) {
        h_bias[i] = 0.05f * std::sin((float)i * 0.1f);
    }
    for (int m = 0; m < M; m++) {
        for (int i = 0; i < n_experts; i++) {
            float val = std::cos((float)(m * 256 + i) * 0.07f) * 4.0f;
            h_scores[m * n_experts + i] = __nv_bfloat16::from_float(val);
        }
    }

    cudaMemcpy(d_scores, h_scores.data(), M * n_experts * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), n_experts * sizeof(float), cudaMemcpyHostToDevice);

    // Test 1: Single token routing
    moe_route_top6_from_bf16_cuda(d_topk_ids, d_topk_weights, d_scores, d_bias, n_experts, top_k, routed_scaling, nullptr);
    metal_stream_synchronize(nullptr);

    // CPU reference for token 0
    std::vector<std::pair<float, int>> ref_exp_scores(n_experts);
    std::vector<float> ref_exp_probs(n_experts);
    for (int i = 0; i < n_experts; i++) {
        float raw = h_scores[i].to_float();
        float sp = (raw > 20.0f) ? raw : ((raw < -20.0f) ? expf(raw) : log1pf(expf(raw)));
        float prob = sqrtf(sp);
        ref_exp_probs[i] = prob;
        ref_exp_scores[i] = {prob + h_bias[i], i};
    }
    std::partial_sort(ref_exp_scores.begin(), ref_exp_scores.begin() + top_k, ref_exp_scores.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    float ref_sum_p = 0.0f;
    for (int k = 0; k < top_k; k++) ref_sum_p += ref_exp_probs[ref_exp_scores[k].second];
    if (ref_sum_p < 1e-6f) ref_sum_p = 1e-6f;

    std::vector<int32_t> out_ids(M * top_k);
    std::vector<float> out_weights(M * top_k);
    cudaMemcpy(out_ids.data(), d_topk_ids, top_k * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_weights.data(), d_topk_weights, top_k * sizeof(float), cudaMemcpyDeviceToHost);

    for (int k = 0; k < top_k; k++) {
        int expected_id = ref_exp_scores[k].second;
        float expected_w = (ref_exp_probs[expected_id] / ref_sum_p) * routed_scaling;
        TEST_CHECK(out_ids[k] == expected_id, "Single-token routing ID mismatch");
        TEST_CHECK(std::abs(out_weights[k] - expected_w) < 1e-4f, "Single-token routing weight mismatch");
    }

    // Test 2: Batched routing for M tokens
    moe_route_top6_from_bf16_batch_cuda(d_topk_ids, d_topk_weights, d_scores, d_bias, M, n_experts, top_k, routed_scaling, nullptr);
    metal_stream_synchronize(nullptr);
    cudaMemcpy(out_ids.data(), d_topk_ids, M * top_k * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_weights.data(), d_topk_weights, M * top_k * sizeof(float), cudaMemcpyDeviceToHost);

    for (int m = 0; m < M; m++) {
        for (int i = 0; i < n_experts; i++) {
            float raw = h_scores[m * n_experts + i].to_float();
            float sp = (raw > 20.0f) ? raw : ((raw < -20.0f) ? expf(raw) : log1pf(expf(raw)));
            float prob = sqrtf(sp);
            ref_exp_probs[i] = prob;
            ref_exp_scores[i] = {prob + h_bias[i], i};
        }
        std::partial_sort(ref_exp_scores.begin(), ref_exp_scores.begin() + top_k, ref_exp_scores.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        float m_sum_p = 0.0f;
        for (int k = 0; k < top_k; k++) m_sum_p += ref_exp_probs[ref_exp_scores[k].second];
        if (m_sum_p < 1e-6f) m_sum_p = 1e-6f;

        for (int k = 0; k < top_k; k++) {
            int expected_id = ref_exp_scores[k].second;
            float expected_w = (ref_exp_probs[expected_id] / m_sum_p) * routed_scaling;
            TEST_CHECK(out_ids[m * top_k + k] == expected_id, "Batched routing ID mismatch");
            TEST_CHECK(std::abs(out_weights[m * top_k + k] - expected_w) < 1e-4f, "Batched routing weight mismatch");
        }
    }

    std::cout << "       PASS: GPU MoE Top-6 Routing (Single & Batched M=" << M << ")" << std::endl;
    cudaFree(d_scores); cudaFree(d_bias); cudaFree(d_topk_ids); cudaFree(d_topk_weights);
    return true;
}

bool test_markov_head_predict() {
    std::cout << "[TEST] 20. DeepSeek V4 MTP Markov Head Predict Kernel..." << std::endl;
    const int vocab_size = 1024;
    const int hidden_dim = 256;
    const int input_token = 42;
    const int expected_target = 314;

    __nv_bfloat16* d_w1;
    __nv_bfloat16* d_w2;
    int32_t* d_pred;
    float* d_temp_vals;
    int32_t* d_temp_idx;

    cudaMalloc((void**)&d_w1, (size_t)vocab_size * hidden_dim * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_w2, (size_t)vocab_size * hidden_dim * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_pred, sizeof(int32_t));
    cudaMalloc((void**)&d_temp_vals, 256 * sizeof(float));
    cudaMalloc((void**)&d_temp_idx, 256 * sizeof(int32_t));

    std::vector<__nv_bfloat16> h_w1((size_t)vocab_size * hidden_dim, __nv_bfloat16::from_float(0.01f));
    std::vector<__nv_bfloat16> h_w2((size_t)vocab_size * hidden_dim, __nv_bfloat16::from_float(0.01f));

    // Make input_token vector distinctive
    for (int d = 0; d < hidden_dim; d++) {
        h_w1[(size_t)input_token * hidden_dim + d] = __nv_bfloat16::from_float(0.1f * (d % 5 + 1));
        // Make expected_target have the highest dot product
        h_w2[(size_t)expected_target * hidden_dim + d] = __nv_bfloat16::from_float(0.5f * (d % 5 + 1));
    }

    cudaMemcpy(d_w1, h_w1.data(), h_w1.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, h_w2.data(), h_w2.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    markov_head_predict_cuda(d_pred, d_temp_vals, d_temp_idx, d_w1, d_w2, input_token, vocab_size, hidden_dim, nullptr);
    metal_stream_synchronize(nullptr);

    int32_t pred_result = -1;
    cudaMemcpy(&pred_result, d_pred, sizeof(int32_t), cudaMemcpyDeviceToHost);

    std::cout << "       Predicted next token: " << pred_result << " (expected: " << expected_target << ")" << std::endl;
    TEST_CHECK(pred_result == expected_target, "Markov head predicted token does not match expected target");
    std::cout << "       PASS: DeepSeek V4 MTP Markov Head Predict Kernel" << std::endl;

    cudaFree(d_w1); cudaFree(d_w2); cudaFree(d_pred); cudaFree(d_temp_vals); cudaFree(d_temp_idx);
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
    if (!test_gemv_int3_gpu()) return 1;
    if (!test_gqa_attention()) return 1;
    if (!test_hc_and_sinkhorn()) return 1;
    if (!test_mla_attention_fused()) return 1;
    if (!test_gemv_iq2_xxs_and_swiglu()) return 1;
    if (!test_gemv_q2_k()) return 1;
    if (!test_fp8_dequant_and_gemv()) return 1;
    if (!test_fused_moe_accum()) return 1;
    if (!test_moe_top6_routing()) return 1;
    if (!test_markov_head_predict()) return 1;

    std::cout << "==========================================================" << std::endl;
    std::cout << "  ALL 20 TESTS PASSED ON APPLE SILICON METAL GPU!         " << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}

