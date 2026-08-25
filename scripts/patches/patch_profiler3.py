import sys
content = open('src/server_single.cpp').read()
content = content.replace("forward_attention(layer_id, position);", """
auto t_att_1 = std::chrono::high_resolution_clock::now();
forward_attention(layer_id, position);
cudaStreamSynchronize(main_stream_);
auto t_att_2 = std::chrono::high_resolution_clock::now();
if (layer_id < 2 && position < 5) fprintf(stderr, "[PROFILE] L%d: t_att=%.3fms\\n", layer_id, std::chrono::duration<float, std::milli>(t_att_2 - t_att_1).count());
""")
open('src/server_single.cpp', 'w').write(content)
