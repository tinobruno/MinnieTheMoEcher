import sys
content = open('src/server_single.cpp').read()
old_code = """        // We need F32 GEMM here. Use cuBLAS with F32.
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
                                   hc_dim * sizeof(float), cudaMemcpyHostToDevice));"""
new_code = """        // We need F32 GEMM here. Use cuBLAS with F32.
        {
            float alpha = 1.0f, beta = 0.0f;

            // Optimized with GPU kernel
            rms_norm_f32_cuda(buf_hc_input_.f32(), hc_dim, cfg_.hc_eps, main_stream_);"""
content = content.replace(old_code, new_code)
open('src/server_single.cpp', 'w').write(content)
