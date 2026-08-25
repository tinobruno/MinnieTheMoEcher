import sys

content = open('src/server_single.cpp').read()

target_hc_pre = """        // We need F32 GEMM here. Use cuBLAS with F32.
        {
            float alpha = 1.0f, beta = 0.0f;

            // First compute rsqrt of RMS
            // For simplicity, compute on CPU (small data transfer)
            std::vector<float> x_flat(hc_dim);
            CUDA_CHECK(cudaMemcpy(x_flat.data(), buf_hc_input_.f32(),
                                   hc_dim * sizeof(float), cudaMemcpyDeviceToHost));

            float sum_sq = 0;
            for (int i = 0; i < hc_dim; i++) sum_sq += x_flat[i] * x_flat[i];
            float rsqrt_val = 1.0f / sqrtf(sum_sq / hc_dim + cfg_.hc_eps);

            // Scale x by rsqrt
            for (auto& v : x_flat) v *= rsqrt_val;
            CUDA_CHECK(cudaMemcpy(buf_hc_input_.f32(), x_flat.data(),
                                   hc_dim * sizeof(float), cudaMemcpyHostToDevice));

            // mixes = x_scaled @ hc_fn.T
            CUBLAS_CHECK(cublasGemmEx(
                cublas_handle_,
                CUBLAS_OP_T, CUBLAS_OP_N,
                mix_size, 1, hc_dim,
                &alpha,
                hc_fn.f32(), CUDA_R_32F, hc_dim,
                buf_hc_input_.f32(), CUDA_R_32F, hc_dim,
                &beta,
                buf_hc_mixes_.f32(), CUDA_R_32F, mix_size,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT));
        }

        // Split mixes into pre, post, comb via Sinkhorn
        hc_split_sinkhorn_cuda(
            buf_hc_pre_.f32(), buf_hc_post_.f32(), buf_hc_comb_.f32(),
            buf_hc_mixes_.f32(), hc_scale.f32(), hc_base.f32(),
            hc, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, main_stream_);
        CUDA_CHECK(cudaStreamSynchronize(main_stream_));

        // Compute weighted sum: y = sum(pre[i] * hc_state[i]) for i in 0..hc-1
        // Result in buf_hidden_
        {
            float pre_host[8];
            CUDA_CHECK(cudaMemcpy(pre_host, buf_hc_pre_.f32(),
                                   hc * sizeof(float), cudaMemcpyDeviceToHost));

            if (dbg_hc_pre_call_ < 2) {
                float post_host2[8], comb_host2[64];
                CUDA_CHECK(cudaMemcpy(post_host2, buf_hc_post_.f32(), hc * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(comb_host2, buf_hc_comb_.f32(), hc * hc * sizeof(float), cudaMemcpyDeviceToHost));
                LOG_INFO("HC pre call #%d:", dbg_hc_pre_call_);
                for (int i = 0; i < hc; i++)
                    LOG_INFO("  pre[%d]=%.6f post[%d]=%.6f", i, pre_host[i], i, post_host2[i]);
                for (int i = 0; i < hc; i++) {
                    std::string s;
                    for (int j = 0; j < hc; j++) {
                        char buf[32]; snprintf(buf, sizeof(buf), "%.4f ", comb_host2[i*hc+j]);
                        s += buf;
                    }
                    LOG_INFO("  comb[%d] = [%s]", i, s.c_str());
                }
                dbg_hc_pre_call_++;
            }

            // Zero output
            CUDA_CHECK(cudaMemset(buf_hidden_.data, 0, cfg_.hidden_size * sizeof(__nv_bfloat16)));

            for (int h = 0; h < hc; h++) {
                weighted_add_cuda(buf_hidden_.bf16(),
                                  buf_hc_state_.bf16() + (size_t)h * cfg_.hidden_size,
                                  pre_host[h], cfg_.hidden_size, main_stream_);
            }
        }"""

repl_hc_pre = """        {
            float alpha = 1.0f, beta = 0.0f;
            rms_norm_f32_cuda(buf_hc_input_.f32(), hc_dim, cfg_.hc_eps, main_stream_);

            CUBLAS_CHECK(cublasGemmEx(
                cublas_handle_,
                CUBLAS_OP_T, CUBLAS_OP_N,
                mix_size, 1, hc_dim,
                &alpha,
                hc_fn.f32(), CUDA_R_32F, hc_dim,
                buf_hc_input_.f32(), CUDA_R_32F, hc_dim,
                &beta,
                buf_hc_mixes_.f32(), CUDA_R_32F, mix_size,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }

        hc_split_sinkhorn_cuda(
            buf_hc_pre_.f32(), buf_hc_post_.f32(), buf_hc_comb_.f32(),
            buf_hc_mixes_.f32(), hc_scale.f32(), hc_base.f32(),
            hc, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, main_stream_);

        hc_pre_weighted_add_cuda(
            buf_hidden_.bf16(),
            buf_hc_state_.bf16(),
            buf_hc_pre_.f32(),
            dim, hc, main_stream_);"""


target_hc_post = """        CUDA_CHECK(cudaStreamSynchronize(main_stream_));

        float post_host[8], comb_host[64];
        CUDA_CHECK(cudaMemcpy(post_host, buf_hc_post_.f32(),
                               hc * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(comb_host, buf_hc_comb_.f32(),
                               hc * hc * sizeof(float), cudaMemcpyDeviceToHost));

        // Save current hc_state to persistent temp buffer
        CUDA_CHECK(cudaMemcpyAsync(buf_hc_residual_.data, buf_hc_state_.data,
                                    (size_t)hc * dim * sizeof(__nv_bfloat16),
                                    cudaMemcpyDeviceToDevice, main_stream_));

        for (int i = 0; i < hc; i++) {
            // Start with post[i] * sublayer_out
            __nv_bfloat16* dst = buf_hc_state_.bf16() + (size_t)i * dim;
            // Scale buf_hidden_ by post[i] and write to dst
            CUDA_CHECK(cudaMemset(dst, 0, dim * sizeof(__nv_bfloat16)));
            weighted_add_cuda(dst, buf_hidden_.bf16(), post_host[i], dim, main_stream_);

            // Add sum_j(comb[j][i] * residual[j]) - Transposed per reference
            for (int j = 0; j < hc; j++) {
                float c = comb_host[j * hc + i];
                if (fabsf(c) < 1e-10f) continue;
                weighted_add_cuda(dst, buf_hc_residual_.bf16() + (size_t)j * dim,
                                  c, dim, main_stream_);
            }
        }"""

repl_hc_post = """        CUDA_CHECK(cudaMemcpyAsync(buf_hc_residual_.data, buf_hc_state_.data,
                                    (size_t)hc * dim * sizeof(__nv_bfloat16),
                                    cudaMemcpyDeviceToDevice, main_stream_));

        hc_post_update_cuda(
            buf_hc_state_.bf16(),
            buf_hidden_.bf16(),
            buf_hc_residual_.bf16(),
            buf_hc_post_.f32(),
            buf_hc_comb_.f32(),
            dim, hc, main_stream_);"""

target_hc_head = """        // Compute rsqrt
        std::vector<float> x_flat(hc_dim);
        CUDA_CHECK(cudaMemcpy(x_flat.data(), buf_hc_input_.f32(),
                               hc_dim * sizeof(float), cudaMemcpyDeviceToHost));
        float sum_sq = 0;
        for (int i = 0; i < hc_dim; i++) sum_sq += x_flat[i] * x_flat[i];
        float rsqrt_val = 1.0f / sqrtf(sum_sq / hc_dim + cfg_.rms_norm_eps);
        for (auto& v : x_flat) v *= rsqrt_val;
        CUDA_CHECK(cudaMemcpy(buf_hc_input_.f32(), x_flat.data(),
                               hc_dim * sizeof(float), cudaMemcpyHostToDevice));

        // mixes = x @ hc_head_fn.T  -> [hc]
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasGemmEx(
            cublas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            hc, 1, hc_dim,
            &alpha,
            hc_head_fn_.f32(), CUDA_R_32F, hc_dim,
            buf_hc_input_.f32(), CUDA_R_32F, hc_dim,
            &beta,
            buf_hc_mixes_.f32(), CUDA_R_32F, hc,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

        // pre = sigmoid(mix * scale + base) + eps
        float mixes_host[8], scale_host[8], base_host[8];
        CUDA_CHECK(cudaMemcpy(mixes_host, buf_hc_mixes_.f32(), hc * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(scale_host, hc_head_scale_.f32(), sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(base_host, hc_head_base_.f32(), hc * sizeof(float), cudaMemcpyDeviceToHost));

        float pre_host[8];
        for (int i = 0; i < hc; i++) {
            float v = mixes_host[i] * scale_host[0] + base_host[i];
            pre_host[i] = 1.0f / (1.0f + expf(-v)) + cfg_.hc_eps;
        }

        if (dbg_head_) {
            LOG_INFO("HC head reduce: hc=%d", hc);
            for (int i = 0; i < hc; i++)
                LOG_INFO("  mix[%d]=%.6f scale=%.6f base[%d]=%.6f -> pre[%d]=%.6f",
                         i, mixes_host[i], scale_host[0], i, base_host[i], i, pre_host[i]);
            // Also dump norms of each hc_state copy
            std::vector<__nv_bfloat16> tmp(dim);
            for (int h = 0; h < hc; h++) {
                CUDA_CHECK(cudaMemcpy(tmp.data(), buf_hc_state_.bf16() + (size_t)h * dim,
                                       dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
                float norm = 0;
                for (int i = 0; i < dim; i++) { float v2 = __bfloat162float(tmp[i]); norm += v2*v2; }
                LOG_INFO("  hc_state[%d] norm=%.6f", h, sqrtf(norm));
            }
            dbg_head_ = false;
        }

        // Weighted sum
        CUDA_CHECK(cudaMemset(buf_hidden_.data, 0, dim * sizeof(__nv_bfloat16)));
        for (int h = 0; h < hc; h++) {
            weighted_add_cuda(buf_hidden_.bf16(),
                              buf_hc_state_.bf16() + (size_t)h * dim,
                              pre_host[h], dim, main_stream_);
        }"""

repl_hc_head = """        rms_norm_f32_cuda(buf_hc_input_.f32(), hc_dim, cfg_.rms_norm_eps, main_stream_);

        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasGemmEx(
            cublas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            hc, 1, hc_dim,
            &alpha,
            hc_head_fn_.f32(), CUDA_R_32F, hc_dim,
            buf_hc_input_.f32(), CUDA_R_32F, hc_dim,
            &beta,
            buf_hc_mixes_.f32(), CUDA_R_32F, hc,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

        hc_head_reduce_cuda(
            buf_hidden_.bf16(),
            buf_hc_state_.bf16(),
            buf_hc_mixes_.f32(),
            hc_head_scale_.f32(),
            hc_head_base_.f32(),
            dim, hc, main_stream_);"""


if target_hc_pre not in content:
    print("Could not find target_hc_pre")
    sys.exit(1)
content = content.replace(target_hc_pre, repl_hc_pre)

if target_hc_post not in content:
    print("Could not find target_hc_post")
    sys.exit(1)
content = content.replace(target_hc_post, repl_hc_post)

if target_hc_head not in content:
    print("Could not find target_hc_head")
    sys.exit(1)
content = content.replace(target_hc_head, repl_hc_head)

open('src/server_single.cpp', 'w').write(content)
print("Patch HC done.")
