import sys
content = open('src/server_single.cpp').read()

old_code = """        // So let's use buf_dequant_ which is 128MB.
        auto t3 = std::chrono::high_resolution_clock::now();
        gemm_bf16(buf_dequant_.bf16(), 1, vocab, dim,
                  buf_hidden_.bf16(), head_weight_.bf16());
        bf16_to_f32_cuda(buf_logits_.f32(), buf_dequant_.bf16(), vocab, main_stream_);
        cudaStreamSynchronize(main_stream_);
        auto t4 = std::chrono::high_resolution_clock::now();
        if (!is_prefill) {
            float t_lm_head = std::chrono::duration<float, std::milli>(t4 - t3).count();
            printf("[PROFILE] t_lm_head=%.3fms\\n", t_lm_head);
        }"""

new_code = """        // So let's use buf_dequant_ which is 128MB.
        auto t3 = std::chrono::high_resolution_clock::now();
        gemm_bf16(buf_dequant_.bf16(), 1, vocab, dim,
                  buf_hidden_.bf16(), head_weight_.bf16());
        bf16_to_f32_cuda(buf_logits_.f32(), buf_dequant_.bf16(), vocab, main_stream_);
        cudaStreamSynchronize(main_stream_);
        auto t4 = std::chrono::high_resolution_clock::now();
        float t_lm_head = std::chrono::duration<float, std::milli>(t4 - t3).count();
        printf("[PROFILE] t_lm_head=%.3fms\\n", t_lm_head);"""
        
content = content.replace(old_code, new_code)
open('src/server_single.cpp', 'w').write(content)
