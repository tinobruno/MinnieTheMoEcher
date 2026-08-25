import sys
content = open('src/server_single.cpp').read()

idx_pre = content.find("void hc_pre(")
idx_post = content.find("void hc_post()")
idx_head = content.find("void hc_head_reduce()")
idx_after_head = content.find("int run()", idx_head)

new_hc_pre = """    void hc_pre(GPUTensor& hc_fn, GPUTensor& hc_scale, GPUTensor& hc_base) {
        int dim = cfg_.hidden_size;
        int hc = cfg_.hc_mult;
        int mix_size = (2 + hc) * hc;
        int hc_dim = hc * dim;

        bf16_to_f32_cuda(buf_hc_input_.f32(), buf_hc_state_.bf16(), hc_dim, main_stream_);

        {
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
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT));
        }

        hc_split_sinkhorn_cuda(
            buf_hc_pre_.f32(), buf_hc_post_.f32(), buf_hc_comb_.f32(),
            buf_hc_mixes_.f32(), hc_scale.f32(), hc_base.f32(),
            hc, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, main_stream_);

        CUDA_CHECK(cudaMemsetAsync(buf_hidden_.data, 0, cfg_.hidden_size * sizeof(__nv_bfloat16), main_stream_));

        hc_pre_weighted_add_cuda(
            buf_hidden_.bf16(),
            buf_hc_state_.bf16(),
            buf_hc_pre_.f32(),
            dim, hc, main_stream_);
    }
"""

new_hc_post = """    void hc_post() {
        int dim = cfg_.hidden_size;
        int hc = cfg_.hc_mult;

        CUDA_CHECK(cudaMemcpyAsync(buf_hc_residual_.data, buf_hc_state_.data,
                                    (size_t)hc * dim * sizeof(__nv_bfloat16),
                                    cudaMemcpyDeviceToDevice, main_stream_));

        hc_post_update_cuda(
            buf_hc_state_.bf16(),
            buf_hidden_.bf16(),
            buf_hc_residual_.bf16(),
            buf_hc_post_.f32(),
            buf_hc_comb_.f32(),
            dim, hc, main_stream_);
    }
"""

new_hc_head = """    void hc_head_reduce() {
        int dim = cfg_.hidden_size;
        int hc = cfg_.hc_mult;
        int hc_dim = hc * dim;

        bf16_to_f32_cuda(buf_hc_input_.f32(), buf_hc_state_.bf16(), hc_dim, main_stream_);

        rms_norm_f32_cuda(buf_hc_input_.f32(), hc_dim, cfg_.rms_norm_eps, main_stream_);

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
            dim, hc, main_stream_);
    }
"""

new_content = content[:idx_pre] + new_hc_pre + "\n" + new_hc_post + "\n" + new_hc_head + "\n" + content[idx_after_head:]

open('src/server_single.cpp', 'w').write(new_content)
