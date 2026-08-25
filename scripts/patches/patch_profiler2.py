import sys
content = open('src/server_single.cpp').read()
content = content.replace("forward_attention(i, pos);", """
auto t_att_1 = std::chrono::high_resolution_clock::now();
forward_attention(i, pos);
cudaStreamSynchronize(main_stream_);
auto t_att_2 = std::chrono::high_resolution_clock::now();
if (i < 2 && pos < 5) fprintf(stderr, "[PROFILE] L%d: t_att=%.3fms\\n", i, std::chrono::duration<float, std::milli>(t_att_2 - t_att_1).count());
""")
open('src/server_single.cpp', 'w').write(content)
