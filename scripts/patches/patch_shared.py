import sys
content = open('src/server_single.cpp').read()

old_code = """        if (cfg_.n_shared_experts > 0)
        {
            // w1(x) -> gate [moe_inter]
            gemm_fp8_dequant(buf_gate_.bf16(), 1, moe_inter, dim,
                             buf_hidden_.bf16(),
                             lw.shared_w1_w.u8(), lw.shared_w1_s.u8());
            // w3(x) -> up [moe_inter]
            gemm_fp8_dequant(buf_up_.bf16(), 1, moe_inter, dim,
                             buf_hidden_.bf16(),
                             lw.shared_w3_w.u8(), lw.shared_w3_s.u8());
            // SiLU * mul
            silu_mul_cuda(buf_gate_.bf16(), buf_gate_.bf16(), buf_up_.bf16(),
                          moe_inter, cfg_.swiglu_limit, main_stream_);
            // w2(h) -> [dim]
            gemm_fp8_dequant(buf_down_.bf16(), 1, dim, moe_inter,
                             buf_gate_.bf16(),
                             lw.shared_w2_w.u8(), lw.shared_w2_s.u8());
            // Accumulate shared expert output (no routing weight — always added)
            add_cuda(buf_moe_accum_.bf16(), buf_moe_accum_.bf16(),
                     buf_down_.bf16(), dim, main_stream_);
        }"""

new_code = """        if (cfg_.n_shared_experts > 0)
        {
            auto ts1 = std::chrono::high_resolution_clock::now();
            // w1(x) -> gate [moe_inter]
            gemm_fp8_dequant(buf_gate_.bf16(), 1, moe_inter, dim,
                             buf_hidden_.bf16(),
                             lw.shared_w1_w.u8(), lw.shared_w1_s.u8());
            // w3(x) -> up [moe_inter]
            gemm_fp8_dequant(buf_up_.bf16(), 1, moe_inter, dim,
                             buf_hidden_.bf16(),
                             lw.shared_w3_w.u8(), lw.shared_w3_s.u8());
            // SiLU * mul
            silu_mul_cuda(buf_gate_.bf16(), buf_gate_.bf16(), buf_up_.bf16(),
                          moe_inter, cfg_.swiglu_limit, main_stream_);
            // w2(h) -> [dim]
            gemm_fp8_dequant(buf_down_.bf16(), 1, dim, moe_inter,
                             buf_gate_.bf16(),
                             lw.shared_w2_w.u8(), lw.shared_w2_s.u8());
            // Accumulate shared expert output (no routing weight — always added)
            add_cuda(buf_moe_accum_.bf16(), buf_moe_accum_.bf16(),
                     buf_down_.bf16(), dim, main_stream_);
            cudaStreamSynchronize(main_stream_);
            auto ts2 = std::chrono::high_resolution_clock::now();
            if (layer_id < 2) {
                float t_shared = std::chrono::duration<float, std::milli>(ts2 - ts1).count();
                fprintf(stderr, "[PROFILE] L%d: t_shared=%.3fms\\n", layer_id, t_shared);
            }
        }"""

content = content.replace(old_code, new_code)
open('src/server_single.cpp', 'w').write(content)
