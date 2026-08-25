import sys
content = open('src/server_single.cpp').read()

target_pre = """            float alpha = 1.0f, beta = 0.0f;
            rms_norm_f32_cuda(buf_hc_input_.f32(), hc_dim, cfg_.hc_eps, main_stream_);"""
repl_pre = """            float alpha = 1.0f, beta = 0.0f;
            std::vector<float> x_flat(hc_dim);
            CUDA_CHECK(cudaMemcpyAsync(x_flat.data(), buf_hc_input_.f32(),
                                   hc_dim * sizeof(float), cudaMemcpyDeviceToHost, main_stream_));
            CUDA_CHECK(cudaStreamSynchronize(main_stream_));
            float sum_sq = 0;
            for (int i = 0; i < hc_dim; i++) sum_sq += x_flat[i] * x_flat[i];
            float rsqrt_val = 1.0f / sqrtf(sum_sq / hc_dim + cfg_.hc_eps);
            for (auto& v : x_flat) v *= rsqrt_val;
            CUDA_CHECK(cudaMemcpyAsync(buf_hc_input_.f32(), x_flat.data(),
                                   hc_dim * sizeof(float), cudaMemcpyHostToDevice, main_stream_));"""

target_head = """            float alpha = 1.0f, beta = 0.0f;
            rms_norm_f32_cuda(buf_hc_input_.f32(), hc_dim, cfg_.rms_norm_eps, main_stream_);"""
repl_head = """            float alpha = 1.0f, beta = 0.0f;
            std::vector<float> x_flat(hc_dim);
            CUDA_CHECK(cudaMemcpyAsync(x_flat.data(), buf_hc_input_.f32(),
                                   hc_dim * sizeof(float), cudaMemcpyDeviceToHost, main_stream_));
            CUDA_CHECK(cudaStreamSynchronize(main_stream_));
            float sum_sq = 0;
            for (int i = 0; i < hc_dim; i++) sum_sq += x_flat[i] * x_flat[i];
            float rsqrt_val = 1.0f / sqrtf(sum_sq / hc_dim + cfg_.rms_norm_eps);
            for (auto& v : x_flat) v *= rsqrt_val;
            CUDA_CHECK(cudaMemcpyAsync(buf_hc_input_.f32(), x_flat.data(),
                                   hc_dim * sizeof(float), cudaMemcpyHostToDevice, main_stream_));"""

if target_pre not in content:
    print("target_pre not found")
    sys.exit(1)
content = content.replace(target_pre, repl_pre)

if target_head not in content:
    print("target_head not found")
    sys.exit(1)
content = content.replace(target_head, repl_head)

open('src/server_single.cpp', 'w').write(content)
print("Patched RMS")
