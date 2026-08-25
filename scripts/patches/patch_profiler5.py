import sys
content = open('src/server_single.cpp').read()
content = content.replace("gemm_fp8_dequant(buf_logits_.f32(), 1, cfg_.vocab_size, cfg_.hidden_size,", """
auto t_lm_1 = std::chrono::high_resolution_clock::now();
gemm_fp8_dequant(buf_logits_.f32(), 1, cfg_.vocab_size, cfg_.hidden_size,
""")
content = content.replace("head_w_.u8(), head_s_.u8());", """
head_w_.u8(), head_s_.u8());
cudaStreamSynchronize(main_stream_);
auto t_lm_2 = std::chrono::high_resolution_clock::now();
fprintf(stderr, "[PROFILE] t_lm_head=%.3fms\\n", std::chrono::duration<float, std::milli>(t_lm_2 - t_lm_1).count());
""")
open('src/server_single.cpp', 'w').write(content)
