import sys
content = open('src/server_single.cpp').read()
content = content.replace('auto t_att_1 = std::chrono::high_resolution_clock::now();\nforward_attention(layer_id, position);\ncudaStreamSynchronize(main_stream_);\nauto t_att_2 = std::chrono::high_resolution_clock::now();\nif (layer_id < 2) fprintf(stderr, "[PROFILE] L%d: t_att=%.3fms\\n", layer_id, std::chrono::duration<float, std::milli>(t_att_2 - t_att_1).count());', 'forward_attention(layer_id, position);')
content = content.replace('auto t_moe_1 = std::chrono::high_resolution_clock::now();\n        forward_moe(layer_id, token_id);\ncudaStreamSynchronize(main_stream_);\nauto t_moe_2 = std::chrono::high_resolution_clock::now();\nif (layer_id < 2) fprintf(stderr, "[PROFILE] L%d: t_moe=%.3fms\\n", layer_id, std::chrono::duration<float, std::milli>(t_moe_2 - t_moe_1).count());', '        forward_moe(layer_id, token_id);')
content = content.replace('auto t_hc_1 = std::chrono::high_resolution_clock::now();\n        hc_post();\ncudaStreamSynchronize(main_stream_);\nauto t_hc_2 = std::chrono::high_resolution_clock::now();\nif (layer_id < 2) fprintf(stderr, "[PROFILE] L%d: t_hc=%.3fms\\n", layer_id, std::chrono::duration<float, std::milli>(t_hc_2 - t_hc_1).count());', '        hc_post();')
content = content.replace("""        auto loop_start = std::chrono::high_resolution_clock::now();
        for (int layer = 0; layer < cfg_.num_hidden_layers; layer++) {
            forward_layer(layer, token_id, position);
        }
        cudaStreamSynchronize(main_stream_);
        auto loop_end = std::chrono::high_resolution_clock::now();
        if (position > 0) {
            float t_loop = std::chrono::duration<float, std::milli>(loop_end - loop_start).count();
            printf("[PROFILE] t_entire_loop=%.3fms\\n", t_loop);
        }""", """        for (int layer = 0; layer < cfg_.num_hidden_layers; layer++) {
            forward_layer(layer, token_id, position);
        }""")
open('src/server_single.cpp', 'w').write(content)
