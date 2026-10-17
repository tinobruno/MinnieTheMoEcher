    // ── Forward one layer (Nemotron 3.5 Lightning Hybrid Mamba-2 SSM + MoE) ─

    void forward_layer_nemotron(int layer_id, int token_id, int position) {
        auto& lw = layers_[layer_id];
        int dim = cfg_.hidden_size;
        int n_q_heads = cfg_.num_attention_heads;
        int n_kv_heads = cfg_.num_key_value_heads;
        int head_dim = cfg_.head_dim;
        int inter_size = cfg_.intermediate_size > 0 ? cfg_.intermediate_size : cfg_.moe_intermediate_size;

        auto matmul_proj = [&](GPUTensor& out, GPUTensor& in_vec, GPUTensor& weight, GPUTensor& scale, int N, int K) {
            if (weight.dtype == "fp4") {
                gemv_fp4_cuda(out.bf16(), in_vec.bf16(), (const uint8_t*)weight.data, (const uint8_t*)scale.data, N, K, main_stream_);
            } else if (weight.dtype == "int4") {
                gemv_int4_cuda(out.bf16(), in_vec.bf16(), (const uint8_t*)weight.data, scale.bf16(), N, K, main_stream_);
            } else if (weight.dtype == "int3") {
                gemv_int3_cuda(out.bf16(), in_vec.bf16(), (const uint8_t*)weight.data, scale.bf16(), N, K, main_stream_);
            } else {
                gemv_bf16_out_bf16_cuda(out.bf16(), weight.bf16(), in_vec.bf16(), N, K, main_stream_);
            }
        };

        // ════════════════════════════════════════════════════════════════════════
        // 1. SEQUENCE MIXER (Mamba-2 SSM or Attention)
        // ════════════════════════════════════════════════════════════════════════

        // Pre-RMSNorm: buf_hidden_ -> buf_hidden2_
        rms_norm_cuda(buf_hidden2_.bf16(), buf_hidden_.bf16(),
                      lw.attn_norm_w.bf16(), dim, cfg_.rms_norm_eps, main_stream_);

        if (is_blackwell_tc_active_) {
            quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                           buf_hidden2_.bf16(), 1, dim, main_stream_);
        }

        if (lw.is_linear_attn) {
            // ── Mamba-2 SSM Recurrence Step ─────────────────────────────────
            int num_k_heads = 16;
            int num_v_heads = 48;
            int ssm_head_dim = cfg_.ssm_d_state > 0 ? cfg_.ssm_d_state : 128;
            int qkv_dim = (2 * num_k_heads + num_v_heads) * ssm_head_dim;
            int z_dim = num_v_heads * ssm_head_dim;

            // Input Projections
            if (is_blackwell_tc_active_ && lw.w_in_qkv.dtype == "fp4") {
                gemm_fp4_blackwell_tensorcore_cuda(buf_q_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                   (const uint8_t*)lw.w_in_qkv.data, (const uint8_t*)lw.w_in_qkv_scale.data,
                                                   qkv_dim, dim, 1, main_stream_);
            } else {
                matmul_proj(buf_q_, buf_hidden2_, lw.w_in_qkv, lw.w_in_qkv_scale, qkv_dim, dim);
            }

            if (is_blackwell_tc_active_ && lw.w_in_z.dtype == "fp4") {
                gemm_fp4_blackwell_tensorcore_cuda(buf_up_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                   (const uint8_t*)lw.w_in_z.data, (const uint8_t*)lw.w_in_z_scale.data,
                                                   z_dim, dim, 1, main_stream_);
            } else {
                matmul_proj(buf_up_, buf_hidden2_, lw.w_in_z, lw.w_in_z_scale, z_dim, dim);
            }

            gemv_bf16_out_bf16_cuda(buf_linear_a_.bf16(), lw.w_in_a.bf16(), buf_hidden2_.bf16(), num_v_heads, dim, main_stream_);
            gemv_bf16_out_bf16_cuda(buf_linear_b_.bf16(), lw.w_in_b.bf16(), buf_hidden2_.bf16(), num_v_heads, dim, main_stream_);

            // Fused 1D Causal Conv + Recurrent SSM Decode Step
            deltanet_linear_attention_decode_cuda(
                buf_attn_out_.bf16(),
                buf_q_.bf16(),
                buf_up_.bf16(),
                buf_linear_a_.bf16(),
                buf_linear_b_.bf16(),
                lw.conv1d_w.bf16(),
                lw.conv_state.bf16(),
                lw.A_log.bf16(),
                lw.dt_bias.bf16(),
                lw.linear_norm_w.bf16(),
                lw.ssm_state.bf16(),
                num_k_heads, num_v_heads, ssm_head_dim, main_stream_);

            // Output Projection + in-place residual addition into buf_hidden_
            if (lw.linear_out_proj.dtype == "fp4") {
                if (is_blackwell_tc_active_) {
                    quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                   buf_attn_out_.bf16(), 1, z_dim, main_stream_);
                    gemm_fp4_residual_blackwell_tensorcore_cuda(buf_hidden_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                                (const uint8_t*)lw.linear_out_proj.data, (const uint8_t*)lw.linear_out_proj_scale.data,
                                                                dim, z_dim, 1, main_stream_);
                } else {
                    gemv_fp4_residual_cuda(buf_hidden_.bf16(), buf_attn_out_.bf16(),
                                           (const uint8_t*)lw.linear_out_proj.data, (const uint8_t*)lw.linear_out_proj_scale.data,
                                           dim, z_dim, main_stream_);
                }
            } else if (lw.linear_out_proj.dtype == "int4") {
                gemv_int4_residual_cuda(buf_hidden_.bf16(), buf_attn_out_.bf16(),
                                        (const uint8_t*)lw.linear_out_proj.data, lw.linear_out_proj_scale.bf16(),
                                        dim, z_dim, main_stream_);
            } else {
                matmul_proj(buf_hidden2_, buf_attn_out_, lw.linear_out_proj, lw.linear_out_proj_scale, dim, z_dim);
                vector_add_bf16_cuda(buf_hidden_.bf16(), buf_hidden2_.bf16(), dim, main_stream_);
            }
        } else {
            // ── Standard GQA Attention Step ─────────────────────────────────
            int q_dim = n_q_heads * head_dim;
            int kv_dim = n_kv_heads * head_dim;

            matmul_proj(buf_q_, buf_hidden2_, lw.w_q, lw.w_q_scale, q_dim, dim);
            matmul_proj(buf_gate_, buf_hidden2_, lw.w_k, lw.w_k_scale, kv_dim, dim);
            matmul_proj(buf_up_, buf_hidden2_, lw.w_v, lw.w_v_scale, kv_dim, dim);

            // RoPE on Q & K
            rope_freqs_step_direct_cuda(buf_q_.bf16(), buf_q_.bf16(), position,
                                        n_q_heads, head_dim, cfg_.rope_theta, main_stream_);
            rope_freqs_step_direct_cuda(buf_gate_.bf16(), buf_gate_.bf16(), position,
                                        n_kv_heads, head_dim, cfg_.rope_theta, main_stream_);

            // Insert into KV ring buffer
            size_t kv_offset = (size_t)(position % cfg_.max_seq_len) * kv_dim;
            CUDA_CHECK(cudaMemcpyAsync(lw.k_cache_gqa.bf16() + kv_offset, buf_gate_.bf16(),
                                       kv_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, main_stream_));
            CUDA_CHECK(cudaMemcpyAsync(lw.v_cache_gqa.bf16() + kv_offset, buf_up_.bf16(),
                                       kv_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, main_stream_));

            // GQA Attention Decode
            int ctx_len = std::min(position + 1, cfg_.max_seq_len);
            qwen2_gqa_decode_fp8_batch_cuda(
                buf_attn_out_.bf16(), buf_q_.bf16(),
                lw.k_cache_gqa.bf16(), lw.v_cache_gqa.bf16(),
                nullptr, nullptr,
                1, ctx_len, n_q_heads, n_kv_heads, head_dim,
                1.0f / sqrtf((float)head_dim),
                cfg_.rope_theta, cfg_.rms_norm_eps,
                nullptr, main_stream_);

            // Attention Output Projection + residual accumulation
            if (lw.w_o.dtype == "int4") {
                gemv_int4_residual_cuda(buf_hidden_.bf16(), buf_attn_out_.bf16(),
                                        (const uint8_t*)lw.w_o.data, lw.w_o_scale.bf16(),
                                        dim, q_dim, main_stream_);
            } else {
                matmul_proj(buf_hidden2_, buf_attn_out_, lw.w_o, lw.w_o_scale, dim, q_dim);
                vector_add_bf16_cuda(buf_hidden_.bf16(), buf_hidden2_.bf16(), dim, main_stream_);
            }
        }

        // ════════════════════════════════════════════════════════════════════════
        // 2. FEED-FORWARD / MoE MIXER
        // ════════════════════════════════════════════════════════════════════════

        // Pre-RMSNorm: buf_hidden_ -> buf_hidden2_
        rms_norm_cuda(buf_hidden2_.bf16(), buf_hidden_.bf16(),
                      lw.ffn_norm_w.bf16(), dim, cfg_.rms_norm_eps, main_stream_);

        if (cfg_.n_routed_experts > 0 && lw.gate_w.data) {
            // ── Sparse MoE Feed-Forward Layer ───────────────────────────────
            int n_experts = cfg_.n_routed_experts;
            int top_k = cfg_.num_experts_per_tok > 0 ? cfg_.num_experts_per_tok : 6;
            int moe_inter = cfg_.moe_intermediate_size > 0 ? cfg_.moe_intermediate_size : inter_size;

            // Route to Top-K Experts
            gemm_bf16(buf_scores_bf16_.bf16(), 1, n_experts, dim,
                      buf_hidden2_.bf16(), lw.gate_w.bf16());

            moe_route_top6_from_bf16_cuda(buf_topk_idx_.i32(), buf_topk_vals_.f32(),
                                          buf_scores_bf16_.bf16(),
                                          lw.gate_bias.data ? lw.gate_bias.f32() : nullptr,
                                          n_experts, top_k, cfg_.routed_scaling_factor, main_stream_);

            // Copy active expert indices to host for dispatch
            CUDA_CHECK(cudaMemcpyAsync(topk_ids_host_, buf_topk_idx_.i32(),
                                       top_k * sizeof(int32_t), cudaMemcpyDeviceToHost, main_stream_));
            CUDA_CHECK(cudaStreamSynchronize(main_stream_));

            // Execute Active Experts
            for (int k = 0; k < top_k; k++) {
                int eid = topk_ids_host_[k];
                if (eid < 0 || eid >= n_experts) continue;

                // Expert Gate + Up Projection & SwiGLU
                // In Minnie, expert weights are addressed via expert_loader_ flat pointers or layer offsets
                gemv_int4_swiglu_fused_cuda(buf_gate_.bf16(), buf_hidden2_.bf16(),
                                            (const uint8_t*)lw.w_gate.data, lw.w_gate_scale.bf16(),
                                            (const uint8_t*)lw.w_up.data, lw.w_up_scale.bf16(),
                                            moe_inter, dim, cfg_.swiglu_limit, main_stream_);

                // Down Projection
                gemv_int4_residual_cuda(buf_hidden_.bf16(), buf_gate_.bf16(),
                                        (const uint8_t*)lw.w_down.data, lw.w_down_scale.bf16(),
                                        dim, moe_inter, main_stream_);
            }
        } else {
            // ── Dense SwiGLU Layer ──────────────────────────────────────────
            if (lw.w_gate.dtype == "int4") {
                gemv_int4_swiglu_fused_cuda(buf_gate_.bf16(), buf_hidden2_.bf16(),
                                            (const uint8_t*)lw.w_gate.data, lw.w_gate_scale.bf16(),
                                            (const uint8_t*)lw.w_up.data, lw.w_up_scale.bf16(),
                                            inter_size, dim, cfg_.swiglu_limit, main_stream_);
                gemv_int4_residual_cuda(buf_hidden_.bf16(), buf_gate_.bf16(),
                                        (const uint8_t*)lw.w_down.data, lw.w_down_scale.bf16(),
                                        dim, inter_size, main_stream_);
            } else if (lw.w_gate.dtype == "int3") {
                gemv_int3_swiglu_fused_cuda(buf_gate_.bf16(), buf_hidden2_.bf16(),
                                            (const uint8_t*)lw.w_gate.data, lw.w_gate_scale.bf16(),
                                            (const uint8_t*)lw.w_up.data, lw.w_up_scale.bf16(),
                                            inter_size, dim, cfg_.swiglu_limit, main_stream_);
                gemv_int3_residual_cuda(buf_hidden_.bf16(), buf_gate_.bf16(),
                                        (const uint8_t*)lw.w_down.data, lw.w_down_scale.bf16(),
                                        dim, inter_size, main_stream_);
            } else {
                matmul_proj(buf_gate_, buf_hidden2_, lw.w_gate, lw.w_gate_scale, inter_size, dim);
                matmul_proj(buf_up_, buf_hidden2_, lw.w_up, lw.w_up_scale, inter_size, dim);
                silu_mul_cuda(buf_gate_.bf16(), buf_gate_.bf16(), buf_up_.bf16(), inter_size, cfg_.swiglu_limit, main_stream_);

                matmul_proj(buf_hidden2_, buf_gate_, lw.w_down, lw.w_down_scale, dim, inter_size);
                vector_add_bf16_cuda(buf_hidden_.bf16(), buf_hidden2_.bf16(), dim, main_stream_);
            }
        }
    }
