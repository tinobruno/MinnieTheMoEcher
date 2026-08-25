import sys
content = open('src/server_single.cpp').read()
old_code = """        // ── MoE FFN ──
        forward_moe(layer_id, token_id);
        if (dbg) dump_bf16("L0 moe_out", buf_hidden_.bf16(), dim);

        // ── HC post for FFN ──
        hc_post();
        if (dbg) dump_bf16("L0 hc_post_ffn[0]", buf_hc_state_.bf16(), dim);"""
new_code = """        // ── MoE FFN ──
auto t_moe_1 = std::chrono::high_resolution_clock::now();
        forward_moe(layer_id, token_id);
cudaStreamSynchronize(main_stream_);
auto t_moe_2 = std::chrono::high_resolution_clock::now();
if (layer_id < 2) fprintf(stderr, "[PROFILE] L%d: t_moe=%.3fms\\n", layer_id, std::chrono::duration<float, std::milli>(t_moe_2 - t_moe_1).count());
        if (dbg) dump_bf16("L0 moe_out", buf_hidden_.bf16(), dim);

        // ── HC post for FFN ──
auto t_hc_1 = std::chrono::high_resolution_clock::now();
        hc_post();
cudaStreamSynchronize(main_stream_);
auto t_hc_2 = std::chrono::high_resolution_clock::now();
if (layer_id < 2) fprintf(stderr, "[PROFILE] L%d: t_hc=%.3fms\\n", layer_id, std::chrono::duration<float, std::milli>(t_hc_2 - t_hc_1).count());
        if (dbg) dump_bf16("L0 hc_post_ffn[0]", buf_hc_state_.bf16(), dim);"""
content = content.replace(old_code, new_code)
open('src/server_single.cpp', 'w').write(content)
