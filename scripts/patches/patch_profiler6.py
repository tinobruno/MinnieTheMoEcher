import sys
content = open('src/server_single.cpp').read()
content = content.replace("gemv_bf16_cuda(buf_logits_.f32(), head_weight_.bf16(), buf_hidden_.bf16(),", """
auto t_lm_1 = std::chrono::high_resolution_clock::now();
gemv_bf16_cuda(buf_logits_.f32(), head_weight_.bf16(), buf_hidden_.bf16(),
""")
content = content.replace("cfg_.vocab_size, cfg_.hidden_size, main_stream_);", """
cfg_.vocab_size, cfg_.hidden_size, main_stream_);
cudaStreamSynchronize(main_stream_);
auto t_lm_2 = std::chrono::high_resolution_clock::now();
fprintf(stderr, "[PROFILE] t_lm_head=%.3fms\\n", std::chrono::duration<float, std::milli>(t_lm_2 - t_lm_1).count());
""")
open('src/server_single.cpp', 'w').write(content)
