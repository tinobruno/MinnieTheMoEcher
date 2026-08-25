import sys
content = open('src/server_single.cpp').read()

old_code = """        if (M == 1 && K % 8 == 0 && (size_t)B % 16 == 0 && (size_t)A % 16 == 0) {
            gemv_bf16_out_bf16_cuda(C, B, A, N, K, main_stream_);
            return;
        }"""
        
new_code = """        if (M == 1) {
            if (K % 8 == 0 && (size_t)B % 16 == 0 && (size_t)A % 16 == 0) {
                gemv_bf16_out_bf16_cuda(C, B, A, N, K, main_stream_);
                return;
            } else {
                printf("gemm_bf16 fallback to cuBLAS for M=1! K=%d, B_align=%lu, A_align=%lu\\n", K, (size_t)B % 16, (size_t)A % 16);
            }
        }"""
content = content.replace(old_code, new_code)

# Add t_lm_head print just in case I missed it!
# Wait, I never added t_lm_head timing! Let me add it.
old_logits = """        // Compute logits
        gemm_bf16(buf_dequant_.bf16(), 1, vocab, dim,
                  buf_hidden_.bf16(), head_weight_.bf16());
        bf16_to_f32_cuda(buf_logits_.f32(), buf_dequant_.bf16(), vocab, main_stream_);"""

new_logits = """        // Compute logits
        auto t3 = std::chrono::high_resolution_clock::now();
        gemm_bf16(buf_dequant_.bf16(), 1, vocab, dim,
                  buf_hidden_.bf16(), head_weight_.bf16());
        bf16_to_f32_cuda(buf_logits_.f32(), buf_dequant_.bf16(), vocab, main_stream_);
        cudaStreamSynchronize(main_stream_);
        auto t4 = std::chrono::high_resolution_clock::now();
        float t_lm_head = std::chrono::duration<float, std::milli>(t4 - t3).count();
        printf("[PROFILE] t_lm_head=%.3fms\\n", t_lm_head);"""

content = content.replace(old_logits, new_logits)

open('src/server_single.cpp', 'w').write(content)
