    // ── Forward one layer (Qwen / Llama GQA + SwiGLU) ───────────────────────

    void forward_layer_qwen(int layer_id, int position) {
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

        // 1. Attention Pre-RMSNorm: buf_hidden_ -> buf_hidden2_
        if (cfg_.architecture == ModelArch::QWEN2) {
            rms_norm_cuda(buf_hidden2_.bf16(), buf_hidden_.bf16(),
                          lw.attn_norm_w.bf16(), dim, cfg_.rms_norm_eps, main_stream_);
        } else {
            rms_norm_one_centered_cuda(buf_hidden2_.bf16(), buf_hidden_.bf16(),
                                       lw.attn_norm_w.bf16(), dim, cfg_.rms_norm_eps, main_stream_);
        }

        if (is_blackwell_tc_active_) {
            quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                           buf_hidden2_.bf16(), 1, dim, main_stream_);
        }

        if (lw.is_linear_attn) {
            // Qwen 3.8 Gated DeltaNet Linear Attention Projections
            if (is_blackwell_tc_active_ && lw.w_in_qkv.dtype == "fp4") {
                gemm_fp4_blackwell_tensorcore_cuda(buf_q_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                    (const uint8_t*)lw.w_in_qkv.data, (const uint8_t*)lw.w_in_qkv_scale.data,
                                                    10240, dim, 1, main_stream_);
            } else {
                matmul_proj(buf_q_, buf_hidden2_, lw.w_in_qkv, lw.w_in_qkv_scale, 10240, dim);
            }
            if (is_blackwell_tc_active_ && lw.w_in_z.dtype == "fp4") {
                gemm_fp4_blackwell_tensorcore_cuda(buf_up_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                    (const uint8_t*)lw.w_in_z.data, (const uint8_t*)lw.w_in_z_scale.data,
                                                    6144, dim, 1, main_stream_);
            } else {
                matmul_proj(buf_up_, buf_hidden2_, lw.w_in_z, lw.w_in_z_scale, 6144, dim);
            }
            gemv_bf16_out_bf16_cuda(buf_linear_a_.bf16(), lw.w_in_a.bf16(), buf_hidden2_.bf16(), 48, dim, main_stream_);
            gemv_bf16_out_bf16_cuda(buf_linear_b_.bf16(), lw.w_in_b.bf16(), buf_hidden2_.bf16(), 48, dim, main_stream_);

            // 3. Fused 1D Causal Conv + Recurrent SSM Decode Step
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
                16, 48, 128, main_stream_);

            // 4. Output Projection: in-place residual accumulation into buf_hidden_
            if (lw.linear_out_proj.dtype == "fp4") {
                if (is_blackwell_tc_active_) {
                    quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                   buf_attn_out_.bf16(), 1, 6144, main_stream_);
                    gemm_fp4_residual_blackwell_tensorcore_cuda(buf_hidden_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                                (const uint8_t*)lw.linear_out_proj.data, (const uint8_t*)lw.linear_out_proj_scale.data,
                                                                dim, 6144, 1, main_stream_);
                } else {
                    gemv_fp4_residual_cuda(buf_hidden_.bf16(), buf_attn_out_.bf16(),
                                            (const uint8_t*)lw.linear_out_proj.data, (const uint8_t*)lw.linear_out_proj_scale.data,
                                            dim, 6144, main_stream_);
                }
            } else if (lw.linear_out_proj.dtype == "int4") {
                gemv_int4_residual_cuda(buf_hidden_.bf16(), buf_attn_out_.bf16(),
                                        (const uint8_t*)lw.linear_out_proj.data, lw.linear_out_proj_scale.bf16(),
                                        dim, 6144, main_stream_);
            } else {
                matmul_proj(buf_hidden2_, buf_attn_out_, lw.linear_out_proj, lw.linear_out_proj_scale, dim, 6144);
                vector_add_bf16_cuda(buf_hidden_.bf16(), buf_hidden2_.bf16(), dim, main_stream_);
            }
        } else if (cfg_.architecture == ModelArch::QWEN2) {
            // Standard Full GQA Attention (Ungated) Projections
            int q_dim = n_q_heads * head_dim;
            int kv_dim = n_kv_heads * head_dim;
            if (is_blackwell_tc_active_ && lw.w_q.dtype == "fp4") {
                gemm_fp4_blackwell_tensorcore_cuda(buf_q_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                    (const uint8_t*)lw.w_q.data, (const uint8_t*)lw.w_q_scale.data,
                                                    q_dim, dim, 1, main_stream_);
            } else {
                matmul_proj(buf_q_, buf_hidden2_, lw.w_q, lw.w_q_scale, q_dim, dim);
            }
            if (is_blackwell_tc_active_ && lw.w_k.dtype == "fp4") {
                gemm_fp4_blackwell_tensorcore_cuda(buf_gate_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                    (const uint8_t*)lw.w_k.data, (const uint8_t*)lw.w_k_scale.data,
                                                    kv_dim, dim, 1, main_stream_);
            } else {
                matmul_proj(buf_gate_, buf_hidden2_, lw.w_k, lw.w_k_scale, kv_dim, dim);
            }
            if (is_blackwell_tc_active_ && lw.w_v.dtype == "fp4") {
                gemm_fp4_blackwell_tensorcore_cuda(buf_up_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                    (const uint8_t*)lw.w_v.data, (const uint8_t*)lw.w_v_scale.data,
                                                    kv_dim, dim, 1, main_stream_);
            } else {
                matmul_proj(buf_up_, buf_hidden2_, lw.w_v, lw.w_v_scale, kv_dim, dim);
            }

            if (lw.gqa_q_bias.data) {
                add_bias_bf16_cuda(buf_q_.bf16(), lw.gqa_q_bias.bf16(), 1, q_dim, main_stream_);
            }
            if (lw.gqa_k_bias.data) {
                add_bias_bf16_cuda(buf_gate_.bf16(), lw.gqa_k_bias.bf16(), 1, kv_dim, main_stream_);
            }
            if (lw.gqa_v_bias.data) {
                add_bias_bf16_cuda(buf_up_.bf16(), lw.gqa_v_bias.bf16(), 1, kv_dim, main_stream_);
            }

            // 3 & 4. QK Norm + RoPE + GQA FP8 Decode (Ungated)
            qwen2_gqa_decode_fp8_cuda(
                buf_attn_out_.bf16(),
                buf_q_.bf16(),
                buf_gate_.bf16(),
                buf_up_.bf16(),
                lw.gqa_q_norm_w.data ? lw.gqa_q_norm_w.bf16() : nullptr,
                lw.gqa_k_norm_w.data ? lw.gqa_k_norm_w.bf16() : nullptr,
                lw.k_cache_gqa.u8(),
                lw.v_cache_gqa.u8(),
                n_q_heads, n_kv_heads, head_dim,
                buf_input_pos_.i32(), position, cfg_.max_seq_len,
                cfg_.rope_theta, cfg_.rms_norm_eps,
                buf_mrope_pos_.data ? buf_mrope_pos_.i32() : nullptr,
                main_stream_);

            // 5. Output Projection: in-place residual accumulation into buf_hidden_
            if (lw.w_o.dtype == "fp4") {
                if (is_blackwell_tc_active_) {
                    quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                   buf_attn_out_.bf16(), 1, q_dim, main_stream_);
                    gemm_fp4_residual_blackwell_tensorcore_cuda(buf_hidden_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                                (const uint8_t*)lw.w_o.data, (const uint8_t*)lw.w_o_scale.data,
                                                                dim, q_dim, 1, main_stream_);
                } else {
                    gemv_fp4_residual_cuda(buf_hidden_.bf16(), buf_attn_out_.bf16(),
                                            (const uint8_t*)lw.w_o.data, (const uint8_t*)lw.w_o_scale.data,
                                            dim, q_dim, main_stream_);
                }
            } else if (lw.w_o.dtype == "int4") {
                gemv_int4_residual_cuda(buf_hidden_.bf16(), buf_attn_out_.bf16(),
                                        (const uint8_t*)lw.w_o.data, lw.w_o_scale.bf16(),
                                        dim, q_dim, main_stream_);
            } else {
                matmul_proj(buf_hidden2_, buf_attn_out_, lw.w_o, lw.w_o_scale, dim, q_dim);
                vector_add_bf16_cuda(buf_hidden_.bf16(), buf_hidden2_.bf16(), dim, main_stream_);
            }
        } else {
            // Standard Full GQA Attention (Gated) Projections
            if (is_blackwell_tc_active_ && lw.w_q.dtype == "fp4") {
                gemm_fp4_blackwell_tensorcore_cuda(buf_q_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                    (const uint8_t*)lw.w_q.data, (const uint8_t*)lw.w_q_scale.data,
                                                    2 * n_q_heads * head_dim, dim, 1, main_stream_);
            } else {
                matmul_proj(buf_q_, buf_hidden2_, lw.w_q, lw.w_q_scale, 2 * n_q_heads * head_dim, dim);
            }
            if (is_blackwell_tc_active_ && lw.w_k.dtype == "fp4") {
                gemm_fp4_blackwell_tensorcore_cuda(buf_gate_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                    (const uint8_t*)lw.w_k.data, (const uint8_t*)lw.w_k_scale.data,
                                                    n_kv_heads * head_dim, dim, 1, main_stream_);
            } else {
                matmul_proj(buf_gate_, buf_hidden2_, lw.w_k, lw.w_k_scale, n_kv_heads * head_dim, dim);
            }
            if (is_blackwell_tc_active_ && lw.w_v.dtype == "fp4") {
                gemm_fp4_blackwell_tensorcore_cuda(buf_up_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                    (const uint8_t*)lw.w_v.data, (const uint8_t*)lw.w_v_scale.data,
                                                    n_kv_heads * head_dim, dim, 1, main_stream_);
            } else {
                matmul_proj(buf_up_, buf_hidden2_, lw.w_v, lw.w_v_scale, n_kv_heads * head_dim, dim);
            }

            // 3 & 4. QK Norm + RoPE + GQA FP8 Decode + Sigmoid Gate
            qwen_gqa_decode_gated_fp8_cuda(
                buf_attn_out_.bf16(),
                buf_q_.bf16(),
                buf_gate_.bf16(),
                buf_up_.bf16(),
                lw.gqa_q_norm_w.bf16(),
                lw.gqa_k_norm_w.bf16(),
                lw.k_cache_gqa.u8(),
                lw.v_cache_gqa.u8(),
                n_q_heads, n_kv_heads, head_dim,
                buf_input_pos_.i32(), position, cfg_.max_seq_len,
                cfg_.rope_theta, cfg_.rms_norm_eps, main_stream_);

            // 5. Output Projection: in-place residual accumulation into buf_hidden_
            if (lw.w_o.dtype == "fp4") {
                if (is_blackwell_tc_active_) {
                    quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                   buf_attn_out_.bf16(), 1, n_q_heads * head_dim, main_stream_);
                    gemm_fp4_residual_blackwell_tensorcore_cuda(buf_hidden_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                                (const uint8_t*)lw.w_o.data, (const uint8_t*)lw.w_o_scale.data,
                                                                dim, n_q_heads * head_dim, 1, main_stream_);
                } else {
                    gemv_fp4_residual_cuda(buf_hidden_.bf16(), buf_attn_out_.bf16(),
                                            (const uint8_t*)lw.w_o.data, (const uint8_t*)lw.w_o_scale.data,
                                            dim, n_q_heads * head_dim, main_stream_);
                }
            } else if (lw.w_o.dtype == "int4") {
                gemv_int4_residual_cuda(buf_hidden_.bf16(), buf_attn_out_.bf16(),
                                        (const uint8_t*)lw.w_o.data, lw.w_o_scale.bf16(),
                                        dim, n_q_heads * head_dim, main_stream_);
            } else {
                matmul_proj(buf_hidden2_, buf_attn_out_, lw.w_o, lw.w_o_scale, dim, n_q_heads * head_dim);
                vector_add_bf16_cuda(buf_hidden_.bf16(), buf_hidden2_.bf16(), dim, main_stream_);
            }
        }

        // 7. FFN Pre-RMSNorm: buf_hidden_ -> buf_hidden2_
        if (cfg_.architecture == ModelArch::QWEN2) {
            rms_norm_cuda(buf_hidden2_.bf16(), buf_hidden_.bf16(),
                          lw.ffn_norm_w.bf16(), dim, cfg_.rms_norm_eps, main_stream_);
        } else {
            rms_norm_one_centered_cuda(buf_hidden2_.bf16(), buf_hidden_.bf16(),
                                       lw.ffn_norm_w.bf16(), dim, cfg_.rms_norm_eps, main_stream_);
        }

        // 8 & 9. Gate & Up projections + Fused SiLU(Gate) * Up
        if (lw.w_gate.dtype == "fp4") {
            if (is_blackwell_tc_active_) {
                quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                               buf_hidden2_.bf16(), 1, dim, main_stream_);
                gemm_fp4_swiglu_blackwell_tensorcore_cuda(buf_gate_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                          (const uint8_t*)lw.w_gate.data, (const uint8_t*)lw.w_gate_scale.data,
                                                          (const uint8_t*)lw.w_up.data, (const uint8_t*)lw.w_up_scale.data,
                                                          inter_size, dim, 1, cfg_.swiglu_limit, main_stream_);
            } else {
                gemv_fp4_swiglu_fused_cuda(buf_gate_.bf16(), buf_hidden2_.bf16(),
                                            (const uint8_t*)lw.w_gate.data, (const uint8_t*)lw.w_gate_scale.data,
                                            (const uint8_t*)lw.w_up.data, (const uint8_t*)lw.w_up_scale.data,
                                            inter_size, dim, cfg_.swiglu_limit, main_stream_);
            }
        } else if (lw.w_gate.dtype == "int4") {
            gemv_int4_swiglu_fused_cuda(buf_gate_.bf16(), buf_hidden2_.bf16(),
                                        (const uint8_t*)lw.w_gate.data, lw.w_gate_scale.bf16(),
                                        (const uint8_t*)lw.w_up.data, lw.w_up_scale.bf16(),
                                        inter_size, dim, cfg_.swiglu_limit, main_stream_);
        } else if (lw.w_gate.dtype == "int3") {
            gemv_int3_swiglu_fused_cuda(buf_gate_.bf16(), buf_hidden2_.bf16(),
                                        (const uint8_t*)lw.w_gate.data, lw.w_gate_scale.bf16(),
                                        (const uint8_t*)lw.w_up.data, lw.w_up_scale.bf16(),
                                        inter_size, dim, cfg_.swiglu_limit, main_stream_);
        } else {
            matmul_proj(buf_gate_, buf_hidden2_, lw.w_gate, lw.w_gate_scale, inter_size, dim);
            matmul_proj(buf_up_, buf_hidden2_, lw.w_up, lw.w_up_scale, inter_size, dim);
            silu_mul_cuda(buf_gate_.bf16(), buf_gate_.bf16(), buf_up_.bf16(), inter_size, cfg_.swiglu_limit, main_stream_);
        }

        // 10. Down projection: in-place residual accumulation into buf_hidden_
        if (lw.w_down.dtype == "fp4") {
            if (is_blackwell_tc_active_) {
                quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                               buf_gate_.bf16(), 1, inter_size, main_stream_);
                gemm_fp4_residual_blackwell_tensorcore_cuda(buf_hidden_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                            (const uint8_t*)lw.w_down.data, (const uint8_t*)lw.w_down_scale.data,
                                                            dim, inter_size, 1, main_stream_);
            } else {
                gemv_fp4_residual_cuda(buf_hidden_.bf16(), buf_gate_.bf16(),
                                        (const uint8_t*)lw.w_down.data, (const uint8_t*)lw.w_down_scale.data,
                                        dim, inter_size, main_stream_);
            }
        } else if (lw.w_down.dtype == "int4") {
            gemv_int4_residual_cuda(buf_hidden_.bf16(), buf_gate_.bf16(),
                                    (const uint8_t*)lw.w_down.data, lw.w_down_scale.bf16(),
                                    dim, inter_size, main_stream_);
        } else if (lw.w_down.dtype == "int3") {
            gemv_int3_residual_cuda(buf_hidden_.bf16(), buf_gate_.bf16(),
                                    (const uint8_t*)lw.w_down.data, lw.w_down_scale.bf16(),
                                    dim, inter_size, main_stream_);
        } else {
            matmul_proj(buf_hidden2_, buf_gate_, lw.w_down, lw.w_down_scale, dim, inter_size);
            vector_add_bf16_cuda(buf_hidden_.bf16(), buf_hidden2_.bf16(), dim, main_stream_);
        }
    }

    void forward_layer_qwen_batch(int layer_id, int position, int M, bool is_tree = false) {
        auto& lw = layers_[layer_id];
        int dim = cfg_.hidden_size;
        int n_q_heads = cfg_.num_attention_heads;
        int n_kv_heads = cfg_.num_key_value_heads;
        int head_dim = cfg_.head_dim;
        int inter_size = cfg_.intermediate_size > 0 ? cfg_.intermediate_size : cfg_.moe_intermediate_size;

        static bool s_layer_prof_done[64] = {false};
        bool prof = (M > 8 && !s_layer_prof_done[layer_id] && (layer_id == 0 || layer_id == 3));
        std::chrono::high_resolution_clock::time_point t0, t1, t2, t3, t4, t5, t6, t7;
        if (prof) {
            cudaStreamSynchronize(main_stream_);
            t0 = std::chrono::high_resolution_clock::now();
        }

        auto matmul_proj_batch = [&](GPUTensor& out, GPUTensor& in_vec, GPUTensor& weight, GPUTensor& scale, int N, int K, float beta = 0.0f) {
            if (weight.dtype == "fp4") {
                if (M == 1) {
                    gemv_fp4_cuda(out.bf16(), in_vec.bf16(), (const uint8_t*)weight.data, (const uint8_t*)scale.data, N, K, main_stream_);
                } else {
                    gemm_fp4_batch_cuda(out.bf16(), in_vec.bf16(), (const uint8_t*)weight.data, (const uint8_t*)scale.data, N, K, M, main_stream_);
                }
            } else if (weight.dtype == "int4") {
                gemm_int4_dequant(out.bf16(), M, N, K, in_vec.bf16(), (const uint8_t*)weight.data, scale.bf16(), 32, main_stream_, beta, &buf_dequant_, &buf_dequant_2_);
            } else if (weight.dtype == "int3") {
                gemm_int3_dequant(out.bf16(), M, N, K, in_vec.bf16(), (const uint8_t*)weight.data, scale.bf16(), 32, main_stream_, beta, &buf_dequant_, &buf_dequant_2_);
            } else {
                if (M <= 8) {
                    gemv_bf16_out_bf16_batch_cuda(out.bf16(), weight.bf16(), in_vec.bf16(), N, K, M, main_stream_);
                } else {
                    gemm_bf16(out.bf16(), M, N, K, in_vec.bf16(), weight.bf16(), 1.0f, beta);
                }
            }
        };

        // 1. Attention Pre-RMSNorm: [M, dim] -> [M, dim]
        if (cfg_.architecture == ModelArch::QWEN2) {
            rms_norm_cuda_batched(buf_hidden2_batch_.bf16(), buf_hidden_batch_.bf16(),
                                  lw.attn_norm_w.bf16(), M, dim, cfg_.rms_norm_eps, main_stream_);
        } else {
            rms_norm_one_centered_cuda_batched(buf_hidden2_batch_.bf16(), buf_hidden_batch_.bf16(),
                                               lw.attn_norm_w.bf16(), M, dim, cfg_.rms_norm_eps, main_stream_);
        }

        if (prof) {
            cudaStreamSynchronize(main_stream_);
            t1 = std::chrono::high_resolution_clock::now();
        }

        if (is_blackwell_tc_active_ && (lw.w_in_qkv.dtype == "fp4" || lw.w_q.dtype == "fp4")) {
            quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                           buf_hidden2_batch_.bf16(), M, dim, main_stream_);
        }

        if (lw.is_linear_attn) {
            // Qwen 3.8 Gated DeltaNet Linear Attention Projections for M tokens
            if (M > 8 && (lw.w_in_qkv.dtype == "int3" || lw.w_in_qkv.dtype == "int4") && (lw.w_in_z.dtype == "int3" || lw.w_in_z.dtype == "int4")) {
                // Dual-stream concurrent projections:
                // main_stream_: w_in_qkv -> buf_q_batch_ using buf_dequant_
                // side_stream_: w_in_z   -> buf_up_batch_ using buf_dequant_2_
                CUDA_CHECK(cudaEventRecord(main_event_, main_stream_));
                CUDA_CHECK(cudaStreamWaitEvent(side_stream_, main_event_, 0));

                if (lw.w_in_z.dtype == "int3") {
                    gemm_int3_dequant(buf_up_batch_.bf16(), M, 6144, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_in_z.data, lw.w_in_z_scale.bf16(), 32,
                                      side_stream_, 0.0f, &buf_dequant_2_, cublas_side_handle_);
                } else {
                    gemm_int4_dequant(buf_up_batch_.bf16(), M, 6144, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_in_z.data, lw.w_in_z_scale.bf16(), 32,
                                      side_stream_, 0.0f, &buf_dequant_2_, cublas_side_handle_);
                }

                gemm_bf16(buf_linear_a_batch_.bf16(), M, 48, dim, buf_hidden2_batch_.bf16(), lw.w_in_a.bf16(), 1.0f, 0.0f, side_stream_, cublas_side_handle_);
                gemm_bf16(buf_linear_b_batch_.bf16(), M, 48, dim, buf_hidden2_batch_.bf16(), lw.w_in_b.bf16(), 1.0f, 0.0f, side_stream_, cublas_side_handle_);
                CUDA_CHECK(cudaEventRecord(side_event_, side_stream_));

                if (lw.w_in_qkv.dtype == "int3") {
                    gemm_int3_dequant(buf_q_batch_.bf16(), M, 10240, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_in_qkv.data, lw.w_in_qkv_scale.bf16(), 32,
                                      main_stream_, 0.0f, &buf_dequant_, cublas_handle_);
                } else {
                    gemm_int4_dequant(buf_q_batch_.bf16(), M, 10240, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_in_qkv.data, lw.w_in_qkv_scale.bf16(), 32,
                                      main_stream_, 0.0f, &buf_dequant_, cublas_handle_);
                }

                CUDA_CHECK(cudaStreamWaitEvent(main_stream_, side_event_, 0));
            } else {
                if (is_blackwell_tc_active_ && lw.w_in_qkv.dtype == "fp4") {
                    gemm_fp4_blackwell_tensorcore_cuda(buf_q_batch_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                        (const uint8_t*)lw.w_in_qkv.data, (const uint8_t*)lw.w_in_qkv_scale.data,
                                                        10240, dim, M, main_stream_);
                } else {
                    matmul_proj_batch(buf_q_batch_, buf_hidden2_batch_, lw.w_in_qkv, lw.w_in_qkv_scale, 10240, dim);
                }
                if (is_blackwell_tc_active_ && lw.w_in_z.dtype == "fp4") {
                    gemm_fp4_blackwell_tensorcore_cuda(buf_up_batch_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                        (const uint8_t*)lw.w_in_z.data, (const uint8_t*)lw.w_in_z_scale.data,
                                                        6144, dim, M, main_stream_);
                } else {
                    matmul_proj_batch(buf_up_batch_, buf_hidden2_batch_, lw.w_in_z, lw.w_in_z_scale, 6144, dim);
                }
#if defined(__APPLE__)
                deltanet_in_proj_ab_batch_cuda(buf_linear_a_batch_.bf16(), buf_linear_b_batch_.bf16(),
                                               lw.w_in_a.bf16(), lw.w_in_b.bf16(), buf_hidden2_batch_.bf16(),
                                               48, dim, M, main_stream_);
#else
                if (M <= 8) {
                    gemv_bf16_out_bf16_batch_cuda(buf_linear_a_batch_.bf16(), lw.w_in_a.bf16(), buf_hidden2_batch_.bf16(), 48, dim, M, main_stream_);
                    gemv_bf16_out_bf16_batch_cuda(buf_linear_b_batch_.bf16(), lw.w_in_b.bf16(), buf_hidden2_batch_.bf16(), 48, dim, M, main_stream_);
                } else {
                    gemm_bf16(buf_linear_a_batch_.bf16(), M, 48, dim, buf_hidden2_batch_.bf16(), lw.w_in_a.bf16());
                    gemm_bf16(buf_linear_b_batch_.bf16(), M, 48, dim, buf_hidden2_batch_.bf16(), lw.w_in_b.bf16());
                }
#endif
            }

            if (prof) {
                cudaStreamSynchronize(main_stream_);
                t2 = std::chrono::high_resolution_clock::now();
            }

            if (is_tree) {
                deltanet_linear_attention_tree_cuda(
                    buf_attn_out_batch_.bf16(),
                    buf_q_batch_.bf16(),
                    buf_up_batch_.bf16(),
                    buf_linear_a_batch_.bf16(),
                    buf_linear_b_batch_.bf16(),
                    lw.conv1d_w.bf16(),
                    lw.conv_state.bf16(),
                    lw.conv_state.bf16(),
                    lw.conv_state_slots[0].bf16(),
                    lw.conv_state_slots[1].bf16(),
                    lw.conv_state_slots[2].bf16(),
                    lw.conv_state_slots[3].bf16(),
                    lw.conv_state_slots[4].bf16(),
                    lw.conv_state_slots[5].bf16(),
                    lw.conv_state_slots[6].bf16(),
                    lw.conv_state_slots[7].bf16(),
                    lw.A_log.bf16(),
                    lw.dt_bias.bf16(),
                    lw.linear_norm_w.bf16(),
                    lw.ssm_state.bf16(),
                    lw.ssm_state.bf16(),
                    lw.ssm_state_slots[0].bf16(),
                    lw.ssm_state_slots[1].bf16(),
                    lw.ssm_state_slots[2].bf16(),
                    lw.ssm_state_slots[3].bf16(),
                    lw.ssm_state_slots[4].bf16(),
                    lw.ssm_state_slots[5].bf16(),
                    lw.ssm_state_slots[6].bf16(),
                    lw.ssm_state_slots[7].bf16(),
                    buf_tree_parent_indices_.i32(),
                    16, 48, 128, M, main_stream_);
            } else {
                // DeltaNet SSM recurrence across M tokens (fully batched, zero intermediate VRAM roundtrips)
                deltanet_linear_attention_decode_batch_cuda(
                    buf_attn_out_batch_.bf16(),
                    buf_q_batch_.bf16(),
                    buf_up_batch_.bf16(),
                    buf_linear_a_batch_.bf16(),
                    buf_linear_b_batch_.bf16(),
                    lw.conv1d_w.bf16(),
                    lw.conv_state.bf16(),
                    lw.conv_state.bf16(),
                    (M > 1 && M <= 4) ? lw.conv_state_slots[0].bf16() : nullptr,
                    (M > 2 && M <= 4) ? lw.conv_state_slots[1].bf16() : nullptr,
                    (M > 3 && M <= 4) ? lw.conv_state_slots[2].bf16() : nullptr,
                    (M > 4 && M <= 4) ? lw.conv_state_slots[3].bf16() : nullptr,
                    lw.A_log.bf16(),
                    lw.dt_bias.bf16(),
                    lw.linear_norm_w.bf16(),
                    lw.ssm_state.bf16(),
                    lw.ssm_state.bf16(),
                    (M > 1 && M <= 4) ? lw.ssm_state_slots[0].bf16() : nullptr,
                    (M > 2 && M <= 4) ? lw.ssm_state_slots[1].bf16() : nullptr,
                    (M > 3 && M <= 4) ? lw.ssm_state_slots[2].bf16() : nullptr,
                    (M > 4 && M <= 4) ? lw.ssm_state_slots[3].bf16() : nullptr,
                    16, 48, 128, M, main_stream_);
            }

            if (prof) {
                cudaStreamSynchronize(main_stream_);
                t3 = std::chrono::high_resolution_clock::now();
            }

            // Output projection: [M, 6144] -> [M, 5120]
            if (lw.linear_out_proj.dtype == "fp4") {
                if (is_blackwell_tc_active_) {
                    quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                   buf_attn_out_batch_.bf16(), M, 6144, main_stream_);
                    gemm_fp4_residual_blackwell_tensorcore_cuda(buf_hidden_batch_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                                (const uint8_t*)lw.linear_out_proj.data, (const uint8_t*)lw.linear_out_proj_scale.data,
                                                                dim, 6144, M, main_stream_);
                } else if (M == 1) {
                    gemv_fp4_residual_cuda(buf_hidden_batch_.bf16(), buf_attn_out_batch_.bf16(),
                                           (const uint8_t*)lw.linear_out_proj.data, (const uint8_t*)lw.linear_out_proj_scale.data,
                                           dim, 6144, main_stream_);
                } else {
                    gemm_fp4_residual_batch_cuda(buf_hidden_batch_.bf16(), buf_attn_out_batch_.bf16(),
                                                 (const uint8_t*)lw.linear_out_proj.data, (const uint8_t*)lw.linear_out_proj_scale.data,
                                                 dim, 6144, M, main_stream_);
                }
            } else {
                if (M <= 8) {
                    matmul_proj_batch(buf_hidden2_batch_, buf_attn_out_batch_, lw.linear_out_proj, lw.linear_out_proj_scale, dim, 6144);
                    vector_add_bf16_cuda(buf_hidden_batch_.bf16(), buf_hidden2_batch_.bf16(), M * dim, main_stream_);
                } else {
                    matmul_proj_batch(buf_hidden_batch_, buf_attn_out_batch_, lw.linear_out_proj, lw.linear_out_proj_scale, dim, 6144, 1.0f);
                }
            }
        } else if (cfg_.architecture == ModelArch::QWEN2) {
            // Full GQA Attention Projections for M tokens (Ungated)
            int q_dim = n_q_heads * head_dim;
            int kv_dim = n_kv_heads * head_dim;
            if (M > 8 && (lw.w_q.dtype == "int3" || lw.w_q.dtype == "int4") && (lw.w_k.dtype == "int3" || lw.w_k.dtype == "int4") && (lw.w_v.dtype == "int3" || lw.w_v.dtype == "int4")) {
                CUDA_CHECK(cudaEventRecord(main_event_, main_stream_));
                CUDA_CHECK(cudaStreamWaitEvent(side_stream_, main_event_, 0));

                if (lw.w_k.dtype == "int3") {
                    gemm_int3_dequant(buf_gate_batch_.bf16(), M, kv_dim, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_k.data, lw.w_k_scale.bf16(), 32,
                                      side_stream_, 0.0f, &buf_dequant_2_, cublas_side_handle_);
                } else {
                    gemm_int4_dequant(buf_gate_batch_.bf16(), M, kv_dim, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_k.data, lw.w_k_scale.bf16(), 32,
                                      side_stream_, 0.0f, &buf_dequant_2_, cublas_side_handle_);
                }
                if (lw.w_v.dtype == "int3") {
                    gemm_int3_dequant(buf_up_batch_.bf16(), M, kv_dim, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_v.data, lw.w_v_scale.bf16(), 32,
                                      side_stream_, 0.0f, &buf_dequant_2_, cublas_side_handle_);
                } else {
                    gemm_int4_dequant(buf_up_batch_.bf16(), M, kv_dim, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_v.data, lw.w_v_scale.bf16(), 32,
                                      side_stream_, 0.0f, &buf_dequant_2_, cublas_side_handle_);
                }
                CUDA_CHECK(cudaEventRecord(side_event_, side_stream_));

                if (lw.w_q.dtype == "int3") {
                    gemm_int3_dequant(buf_q_batch_.bf16(), M, q_dim, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_q.data, lw.w_q_scale.bf16(), 32,
                                      main_stream_, 0.0f, &buf_dequant_, cublas_handle_);
                } else {
                    gemm_int4_dequant(buf_q_batch_.bf16(), M, q_dim, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_q.data, lw.w_q_scale.bf16(), 32,
                                      main_stream_, 0.0f, &buf_dequant_, cublas_handle_);
                }

                CUDA_CHECK(cudaStreamWaitEvent(main_stream_, side_event_, 0));
            } else {
                if (is_blackwell_tc_active_ && lw.w_q.dtype == "fp4") {
                    gemm_fp4_blackwell_tensorcore_cuda(buf_q_batch_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                        (const uint8_t*)lw.w_q.data, (const uint8_t*)lw.w_q_scale.data,
                                                        q_dim, dim, M, main_stream_);
                } else {
                    matmul_proj_batch(buf_q_batch_, buf_hidden2_batch_, lw.w_q, lw.w_q_scale, q_dim, dim);
                }
                if (is_blackwell_tc_active_ && lw.w_k.dtype == "fp4") {
                    gemm_fp4_blackwell_tensorcore_cuda(buf_gate_batch_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                        (const uint8_t*)lw.w_k.data, (const uint8_t*)lw.w_k_scale.data,
                                                        kv_dim, dim, M, main_stream_);
                } else {
                    matmul_proj_batch(buf_gate_batch_, buf_hidden2_batch_, lw.w_k, lw.w_k_scale, kv_dim, dim);
                }
                if (is_blackwell_tc_active_ && lw.w_v.dtype == "fp4") {
                    gemm_fp4_blackwell_tensorcore_cuda(buf_up_batch_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                        (const uint8_t*)lw.w_v.data, (const uint8_t*)lw.w_v_scale.data,
                                                        kv_dim, dim, M, main_stream_);
                } else {
                    matmul_proj_batch(buf_up_batch_, buf_hidden2_batch_, lw.w_v, lw.w_v_scale, kv_dim, dim);
                }
            }

            if (lw.gqa_q_bias.data) {
                add_bias_bf16_cuda(buf_q_batch_.bf16(), lw.gqa_q_bias.bf16(), M, q_dim, main_stream_);
            }
            if (lw.gqa_k_bias.data) {
                add_bias_bf16_cuda(buf_gate_batch_.bf16(), lw.gqa_k_bias.bf16(), M, kv_dim, main_stream_);
            }
            if (lw.gqa_v_bias.data) {
                add_bias_bf16_cuda(buf_up_batch_.bf16(), lw.gqa_v_bias.bf16(), M, kv_dim, main_stream_);
            }

            qwen2_gqa_decode_fp8_batch_cuda(
                buf_attn_out_batch_.bf16(),
                buf_q_batch_.bf16(),
                buf_gate_batch_.bf16(),
                buf_up_batch_.bf16(),
                lw.gqa_q_norm_w.data ? lw.gqa_q_norm_w.bf16() : nullptr,
                lw.gqa_k_norm_w.data ? lw.gqa_k_norm_w.bf16() : nullptr,
                lw.k_cache_gqa.u8(),
                lw.v_cache_gqa.u8(),
                n_q_heads, n_kv_heads, head_dim,
                buf_input_pos_batch_.data ? buf_input_pos_batch_.i32() : nullptr,
                position,
                M,
                cfg_.max_seq_len,
                cfg_.rope_theta, cfg_.rms_norm_eps,
                buf_mrope_pos_.data ? buf_mrope_pos_.i32() : nullptr,
                main_stream_);

            if (lw.w_o.dtype == "fp4") {
                if (is_blackwell_tc_active_) {
                    quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                   buf_attn_out_batch_.bf16(), M, q_dim, main_stream_);
                    gemm_fp4_residual_blackwell_tensorcore_cuda(buf_hidden_batch_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                                (const uint8_t*)lw.w_o.data, (const uint8_t*)lw.w_o_scale.data,
                                                                dim, q_dim, M, main_stream_);
                } else if (M == 1) {
                    gemv_fp4_residual_cuda(buf_hidden_batch_.bf16(), buf_attn_out_batch_.bf16(),
                                           (const uint8_t*)lw.w_o.data, (const uint8_t*)lw.w_o_scale.data,
                                           dim, q_dim, main_stream_);
                } else {
                    gemm_fp4_residual_batch_cuda(buf_hidden_batch_.bf16(), buf_attn_out_batch_.bf16(),
                                                 (const uint8_t*)lw.w_o.data, (const uint8_t*)lw.w_o_scale.data,
                                                 dim, q_dim, M, main_stream_);
                }
            } else {
                if (M <= 8) {
                    matmul_proj_batch(buf_hidden2_batch_, buf_attn_out_batch_, lw.w_o, lw.w_o_scale, dim, q_dim);
                    vector_add_bf16_cuda(buf_hidden_batch_.bf16(), buf_hidden2_batch_.bf16(), M * dim, main_stream_);
                } else {
                    matmul_proj_batch(buf_hidden_batch_, buf_attn_out_batch_, lw.w_o, lw.w_o_scale, dim, q_dim, 1.0f);
                }
            }
        } else {
            // Full GQA Attention Projections for M tokens
            if (M > 8 && (lw.w_q.dtype == "int3" || lw.w_q.dtype == "int4") && (lw.w_k.dtype == "int3" || lw.w_k.dtype == "int4") && (lw.w_v.dtype == "int3" || lw.w_v.dtype == "int4")) {
                int q_dim = 2 * n_q_heads * head_dim;
                int kv_dim = n_kv_heads * head_dim;
                CUDA_CHECK(cudaEventRecord(main_event_, main_stream_));
                CUDA_CHECK(cudaStreamWaitEvent(side_stream_, main_event_, 0));

                if (lw.w_k.dtype == "int3") {
                    gemm_int3_dequant(buf_gate_batch_.bf16(), M, kv_dim, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_k.data, lw.w_k_scale.bf16(), 32,
                                      side_stream_, 0.0f, &buf_dequant_2_, cublas_side_handle_);
                } else {
                    gemm_int4_dequant(buf_gate_batch_.bf16(), M, kv_dim, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_k.data, lw.w_k_scale.bf16(), 32,
                                      side_stream_, 0.0f, &buf_dequant_2_, cublas_side_handle_);
                }
                if (lw.w_v.dtype == "int3") {
                    gemm_int3_dequant(buf_up_batch_.bf16(), M, kv_dim, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_v.data, lw.w_v_scale.bf16(), 32,
                                      side_stream_, 0.0f, &buf_dequant_2_, cublas_side_handle_);
                } else {
                    gemm_int4_dequant(buf_up_batch_.bf16(), M, kv_dim, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_v.data, lw.w_v_scale.bf16(), 32,
                                      side_stream_, 0.0f, &buf_dequant_2_, cublas_side_handle_);
                }
                CUDA_CHECK(cudaEventRecord(side_event_, side_stream_));

                if (lw.w_q.dtype == "int3") {
                    gemm_int3_dequant(buf_q_batch_.bf16(), M, q_dim, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_q.data, lw.w_q_scale.bf16(), 32,
                                      main_stream_, 0.0f, &buf_dequant_, cublas_handle_);
                } else {
                    gemm_int4_dequant(buf_q_batch_.bf16(), M, q_dim, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_q.data, lw.w_q_scale.bf16(), 32,
                                      main_stream_, 0.0f, &buf_dequant_, cublas_handle_);
                }

                CUDA_CHECK(cudaStreamWaitEvent(main_stream_, side_event_, 0));
            } else {
                if (is_blackwell_tc_active_ && lw.w_q.dtype == "fp4") {
                    gemm_fp4_blackwell_tensorcore_cuda(buf_q_batch_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                        (const uint8_t*)lw.w_q.data, (const uint8_t*)lw.w_q_scale.data,
                                                        2 * n_q_heads * head_dim, dim, M, main_stream_);
                } else {
                    matmul_proj_batch(buf_q_batch_, buf_hidden2_batch_, lw.w_q, lw.w_q_scale, 2 * n_q_heads * head_dim, dim);
                }
                if (is_blackwell_tc_active_ && lw.w_k.dtype == "fp4") {
                    gemm_fp4_blackwell_tensorcore_cuda(buf_gate_batch_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                        (const uint8_t*)lw.w_k.data, (const uint8_t*)lw.w_k_scale.data,
                                                        n_kv_heads * head_dim, dim, M, main_stream_);
                } else {
                    matmul_proj_batch(buf_gate_batch_, buf_hidden2_batch_, lw.w_k, lw.w_k_scale, n_kv_heads * head_dim, dim);
                }
                if (is_blackwell_tc_active_ && lw.w_v.dtype == "fp4") {
                    gemm_fp4_blackwell_tensorcore_cuda(buf_up_batch_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                        (const uint8_t*)lw.w_v.data, (const uint8_t*)lw.w_v_scale.data,
                                                        n_kv_heads * head_dim, dim, M, main_stream_);
                } else {
                    matmul_proj_batch(buf_up_batch_, buf_hidden2_batch_, lw.w_v, lw.w_v_scale, n_kv_heads * head_dim, dim);
                }
            }

            if (prof) {
                cudaStreamSynchronize(main_stream_);
                t2 = std::chrono::high_resolution_clock::now();
            }

            if (is_tree) {
                qwen_gqa_decode_tree_fp8_batch_cuda(
                    buf_attn_out_batch_.bf16(),
                    buf_q_batch_.bf16(),
                    buf_gate_batch_.bf16(),
                    buf_up_batch_.bf16(),
                    lw.gqa_q_norm_w.bf16(),
                    lw.gqa_k_norm_w.bf16(),
                    lw.k_cache_gqa.u8(),
                    lw.v_cache_gqa.u8(),
                    n_q_heads, n_kv_heads, head_dim,
                    buf_input_pos_batch_.data ? buf_input_pos_batch_.i32() : nullptr,
                    buf_tree_ancestor_mask_.u64(),
                    position,
                    M,
                    cfg_.max_seq_len,
                    cfg_.rope_theta, cfg_.rms_norm_eps, main_stream_);
            } else {
                qwen_gqa_decode_gated_fp8_batch_cuda(
                    buf_attn_out_batch_.bf16(),
                    buf_q_batch_.bf16(),
                    buf_gate_batch_.bf16(),
                    buf_up_batch_.bf16(),
                    lw.gqa_q_norm_w.bf16(),
                    lw.gqa_k_norm_w.bf16(),
                    lw.k_cache_gqa.u8(),
                    lw.v_cache_gqa.u8(),
                    n_q_heads, n_kv_heads, head_dim,
                    buf_input_pos_batch_.data ? buf_input_pos_batch_.i32() : nullptr,
                    position,
                    M,
                    cfg_.max_seq_len,
                    cfg_.rope_theta, cfg_.rms_norm_eps, main_stream_);
            }

            if (prof) {
                cudaStreamSynchronize(main_stream_);
                t3 = std::chrono::high_resolution_clock::now();
            }

            if (lw.w_o.dtype == "fp4") {
                if (is_blackwell_tc_active_) {
                    quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                   buf_attn_out_batch_.bf16(), M, n_q_heads * head_dim, main_stream_);
                    gemm_fp4_residual_blackwell_tensorcore_cuda(buf_hidden_batch_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                                (const uint8_t*)lw.w_o.data, (const uint8_t*)lw.w_o_scale.data,
                                                                dim, n_q_heads * head_dim, M, main_stream_);
                } else if (M == 1) {
                    gemv_fp4_residual_cuda(buf_hidden_batch_.bf16(), buf_attn_out_batch_.bf16(),
                                           (const uint8_t*)lw.w_o.data, (const uint8_t*)lw.w_o_scale.data,
                                           dim, n_q_heads * head_dim, main_stream_);
                } else {
                    gemm_fp4_residual_batch_cuda(buf_hidden_batch_.bf16(), buf_attn_out_batch_.bf16(),
                                                 (const uint8_t*)lw.w_o.data, (const uint8_t*)lw.w_o_scale.data,
                                                 dim, n_q_heads * head_dim, M, main_stream_);
                }
            } else {
                if (M <= 8) {
                    matmul_proj_batch(buf_hidden2_batch_, buf_attn_out_batch_, lw.w_o, lw.w_o_scale, dim, n_q_heads * head_dim);
                    vector_add_bf16_cuda(buf_hidden_batch_.bf16(), buf_hidden2_batch_.bf16(), M * dim, main_stream_);
                } else {
                    matmul_proj_batch(buf_hidden_batch_, buf_attn_out_batch_, lw.w_o, lw.w_o_scale, dim, n_q_heads * head_dim, 1.0f);
                }
            }
        }

        if (prof) {
            cudaStreamSynchronize(main_stream_);
            t4 = std::chrono::high_resolution_clock::now();
        }

        if (cfg_.architecture == ModelArch::QWEN2) {
            rms_norm_cuda_batched(buf_hidden2_batch_.bf16(), buf_hidden_batch_.bf16(),
                                  lw.ffn_norm_w.bf16(), M, dim, cfg_.rms_norm_eps, main_stream_);
        } else {
            rms_norm_one_centered_cuda_batched(buf_hidden2_batch_.bf16(), buf_hidden_batch_.bf16(),
                                               lw.ffn_norm_w.bf16(), M, dim, cfg_.rms_norm_eps, main_stream_);
        }

        // FFN SwiGLU for M tokens simultaneously:
        if (lw.w_gate.dtype == "fp4") {
            if (is_blackwell_tc_active_) {
                quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                               buf_hidden2_batch_.bf16(), M, dim, main_stream_);
                gemm_fp4_swiglu_blackwell_tensorcore_cuda(buf_gate_batch_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                          (const uint8_t*)lw.w_gate.data, (const uint8_t*)lw.w_gate_scale.data,
                                                          (const uint8_t*)lw.w_up.data, (const uint8_t*)lw.w_up_scale.data,
                                                          inter_size, dim, M, cfg_.swiglu_limit, main_stream_);
            } else if (M == 1) {
                gemv_fp4_swiglu_fused_cuda(buf_gate_batch_.bf16(), buf_hidden2_batch_.bf16(),
                                           (const uint8_t*)lw.w_gate.data, (const uint8_t*)lw.w_gate_scale.data,
                                           (const uint8_t*)lw.w_up.data, (const uint8_t*)lw.w_up_scale.data,
                                           inter_size, dim, cfg_.swiglu_limit, main_stream_);
            } else {
                gemm_fp4_swiglu_fused_batch_cuda(buf_gate_batch_.bf16(), buf_hidden2_batch_.bf16(),
                                                  (const uint8_t*)lw.w_gate.data, (const uint8_t*)lw.w_gate_scale.data,
                                                  (const uint8_t*)lw.w_up.data, (const uint8_t*)lw.w_up_scale.data,
                                                  inter_size, dim, M, cfg_.swiglu_limit, main_stream_);
            }
        } else if (lw.w_gate.dtype == "int4" && M <= 8) {
            gemm_int4_swiglu_fused_batch_cuda(buf_gate_batch_.bf16(), buf_hidden2_batch_.bf16(),
                                              (const uint8_t*)lw.w_gate.data, lw.w_gate_scale.bf16(),
                                              (const uint8_t*)lw.w_up.data, lw.w_up_scale.bf16(),
                                              inter_size, dim, M, cfg_.swiglu_limit, main_stream_);
        } else if (lw.w_gate.dtype == "int3" && M <= 8) {
            gemm_int3_swiglu_fused_batch_cuda(buf_gate_batch_.bf16(), buf_hidden2_batch_.bf16(),
                                              (const uint8_t*)lw.w_gate.data, lw.w_gate_scale.bf16(),
                                              (const uint8_t*)lw.w_up.data, lw.w_up_scale.bf16(),
                                              inter_size, dim, M, cfg_.swiglu_limit, main_stream_);
        } else if (M > 8 && lw.w_gate.dtype == "int3" && lw.w_up.dtype == "int3") {
            // High-occupancy double-buffered pipelined MLP projections across buf_dequant_ & buf_dequant_2_ (48 MB tiles, 136 cuBLAS blocks)
            gemm_int3_dequant(buf_gate_batch_.bf16(), M, inter_size, dim, buf_hidden2_batch_.bf16(),
                              (const uint8_t*)lw.w_gate.data, lw.w_gate_scale.bf16(), 32,
                              main_stream_, 0.0f, &buf_dequant_, &buf_dequant_2_, cublas_handle_);

            gemm_int3_dequant(buf_up_batch_.bf16(), M, inter_size, dim, buf_hidden2_batch_.bf16(),
                              (const uint8_t*)lw.w_up.data, lw.w_up_scale.bf16(), 32,
                              main_stream_, 0.0f, &buf_dequant_, &buf_dequant_2_, cublas_handle_);

            if (prof) {
                cudaStreamSynchronize(main_stream_);
                t5 = std::chrono::high_resolution_clock::now();
            }
            silu_mul_cuda(buf_gate_batch_.bf16(), buf_gate_batch_.bf16(),
                          buf_up_batch_.bf16(), M * inter_size, cfg_.swiglu_limit, main_stream_);
        } else if (M > 8 && lw.w_gate.dtype == "int4" && lw.w_up.dtype == "int4") {
            gemm_int4_dequant(buf_gate_batch_.bf16(), M, inter_size, dim, buf_hidden2_batch_.bf16(),
                              (const uint8_t*)lw.w_gate.data, lw.w_gate_scale.bf16(), 32,
                              main_stream_, 0.0f, &buf_dequant_, &buf_dequant_2_, cublas_handle_);

            gemm_int4_dequant(buf_up_batch_.bf16(), M, inter_size, dim, buf_hidden2_batch_.bf16(),
                              (const uint8_t*)lw.w_up.data, lw.w_up_scale.bf16(), 32,
                              main_stream_, 0.0f, &buf_dequant_, &buf_dequant_2_, cublas_handle_);

            if (prof) {
                cudaStreamSynchronize(main_stream_);
                t5 = std::chrono::high_resolution_clock::now();
            }
            silu_mul_cuda(buf_gate_batch_.bf16(), buf_gate_batch_.bf16(),
                          buf_up_batch_.bf16(), M * inter_size, cfg_.swiglu_limit, main_stream_);
        } else {
            matmul_proj_batch(buf_gate_batch_, buf_hidden2_batch_, lw.w_gate, lw.w_gate_scale, inter_size, dim);
            matmul_proj_batch(buf_up_batch_, buf_hidden2_batch_, lw.w_up, lw.w_up_scale, inter_size, dim);
            if (prof) {
                cudaStreamSynchronize(main_stream_);
                t5 = std::chrono::high_resolution_clock::now();
            }
            silu_mul_cuda(buf_gate_batch_.bf16(), buf_gate_batch_.bf16(),
                          buf_up_batch_.bf16(), M * inter_size, cfg_.swiglu_limit, main_stream_);
        }

        if (prof) {
            cudaStreamSynchronize(main_stream_);
            t6 = std::chrono::high_resolution_clock::now();
        }

        // Down projection: [M, inter_size] -> [M, dim]
        if (lw.w_down.dtype == "fp4") {
            if (is_blackwell_tc_active_) {
                quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                               buf_gate_batch_.bf16(), M, inter_size, main_stream_);
                gemm_fp4_residual_blackwell_tensorcore_cuda(buf_hidden_batch_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                            (const uint8_t*)lw.w_down.data, (const uint8_t*)lw.w_down_scale.data,
                                                            dim, inter_size, M, main_stream_);
            } else if (M == 1) {
                gemv_fp4_residual_cuda(buf_hidden_batch_.bf16(), buf_gate_batch_.bf16(),
                                       (const uint8_t*)lw.w_down.data, (const uint8_t*)lw.w_down_scale.data,
                                       dim, inter_size, main_stream_);
            } else {
                gemm_fp4_residual_batch_cuda(buf_hidden_batch_.bf16(), buf_gate_batch_.bf16(),
                                             (const uint8_t*)lw.w_down.data, (const uint8_t*)lw.w_down_scale.data,
                                             dim, inter_size, M, main_stream_);
            }
        } else {
            if (M <= 8) {
                matmul_proj_batch(buf_hidden2_batch_, buf_gate_batch_, lw.w_down, lw.w_down_scale, dim, inter_size);
                vector_add_bf16_cuda(buf_hidden_batch_.bf16(), buf_hidden2_batch_.bf16(), M * dim, main_stream_);
            } else {
                matmul_proj_batch(buf_hidden_batch_, buf_gate_batch_, lw.w_down, lw.w_down_scale, dim, inter_size, 1.0f);
            }
        }

        if (prof) {
            cudaStreamSynchronize(main_stream_);
            t7 = std::chrono::high_resolution_clock::now();
            s_layer_prof_done[layer_id] = true;
            float ms_norm = std::chrono::duration<float, std::milli>(t1 - t0).count();
            float ms_ain  = std::chrono::duration<float, std::milli>(t2 - t1).count();
            float ms_core = std::chrono::duration<float, std::milli>(t3 - t2).count();
            float ms_aout = std::chrono::duration<float, std::milli>(t4 - t3).count();
            float ms_min  = std::chrono::duration<float, std::milli>(t5 - t4).count();
            float ms_act  = std::chrono::duration<float, std::milli>(t6 - t5).count();
            float ms_down = std::chrono::duration<float, std::milli>(t7 - t6).count();
            float ms_tot  = std::chrono::duration<float, std::milli>(t7 - t0).count();
            LOG_INFO("[LAYER PROF L%d %s M=%d] total: %.2f ms | norm: %.2f | attn_in: %.2f | attn_core: %.2f | attn_out: %.2f | mlp_in: %.2f | swiglu: %.2f | mlp_down: %.2f",
                     layer_id, lw.is_linear_attn ? "DeltaNet" : "GQA", M, ms_tot, ms_norm, ms_ain, ms_core, ms_aout, ms_min, ms_act, ms_down);
        }
    }

    bool batch_graph_captured_[9] = {false};
    cudaGraph_t batch_graph_[9] = {nullptr};
    cudaGraphExec_t batch_graph_exec_[9] = {nullptr};

    void forward_token_batch_qwen_device_body(int position, int M, bool compute_logits = true, bool is_tree = false) {
        int dim = cfg_.hidden_size;
        // 1. Embedding lookup for M tokens
        if (embed_weight_.dtype == "fp4") {
            embedding_fp4_cuda(buf_hidden_batch_.bf16(), (const uint8_t*)embed_weight_.data, (const uint8_t*)embed_weight_scale_.data, buf_input_tokens_batch_.i32(), M, dim, main_stream_);
        } else if (embed_weight_.dtype == "int4") {
            embedding_int4_cuda(buf_hidden_batch_.bf16(), (uint8_t*)embed_weight_.data, embed_weight_scale_.bf16(), buf_input_tokens_batch_.i32(), M, dim, main_stream_);
        } else {
            embedding_cuda(buf_hidden_batch_.bf16(), embed_weight_.bf16(), buf_input_tokens_batch_.i32(), M, dim, main_stream_);
        }

        // Override with vision tower embedding for image tokens
        if (active_visual_embeddings_) {
            for (int m = 0; m < M; m++) {
                int pos = position + m;
                if (pos >= visual_embed_start_pos_ && pos < visual_embed_start_pos_ + visual_embed_num_tokens_) {
                    int v_idx = pos - visual_embed_start_pos_;
                    CUDA_CHECK(cudaMemcpyAsync(buf_hidden_batch_.bf16() + (size_t)m * dim,
                                               active_visual_embeddings_ + (size_t)v_idx * dim,
                                               dim * sizeof(__nv_bfloat16),
                                               cudaMemcpyDeviceToDevice, main_stream_));
                }
            }
        }

        // 2. Process each layer
        static int s_prof_pass = 0;
        bool do_prof = (M > 8 && s_prof_pass++ == 1);
        if (do_prof) {
            cudaStreamSynchronize(main_stream_);
            auto t_start = std::chrono::high_resolution_clock::now();
            float t_delta_net = 0.0f;
            float t_gqa = 0.0f;
            int n_delta_net = 0;
            int n_gqa = 0;
            for (int layer = 0; layer < cfg_.num_hidden_layers; layer++) {
                auto tl0 = std::chrono::high_resolution_clock::now();
                forward_layer_qwen_batch(layer, position, M, is_tree);
                cudaStreamSynchronize(main_stream_);
                auto tl1 = std::chrono::high_resolution_clock::now();
                float ms = std::chrono::duration<float, std::milli>(tl1 - tl0).count();
                if (layers_[layer].is_linear_attn) {
                    t_delta_net += ms;
                    n_delta_net++;
                } else {
                    t_gqa += ms;
                    n_gqa++;
                }
            }
            auto t_end = std::chrono::high_resolution_clock::now();
            float total_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
            LOG_INFO("[PREFILL PROFILE M=%d] Total 64 layers: %.2f ms (%.1f tok/s) | DeltaNet (%d layers): avg %.2f ms (sum %.1f ms) | GQA (%d layers): avg %.2f ms (sum %.1f ms)",
                     M, total_ms, M / (total_ms / 1000.0f), n_delta_net, t_delta_net / std::max(1, n_delta_net), t_delta_net, n_gqa, t_gqa / std::max(1, n_gqa), t_gqa);
        } else {
            for (int layer = 0; layer < cfg_.num_hidden_layers; layer++) {
                forward_layer_qwen_batch(layer, position, M, is_tree);
            }
        }

        if (!compute_logits) return;

        // 3. Final norm
        if (cfg_.architecture == ModelArch::QWEN2) {
            rms_norm_cuda_batched(buf_hidden_batch_.bf16(), buf_hidden_batch_.bf16(),
                                  norm_weight_.bf16(), M, dim, cfg_.rms_norm_eps, main_stream_);
        } else {
            rms_norm_one_centered_cuda_batched(buf_hidden_batch_.bf16(), buf_hidden_batch_.bf16(),
                                               norm_weight_.bf16(), M, dim, cfg_.rms_norm_eps, main_stream_);
        }

        // 4. Logits: hidden @ head_weight.T -> [M, vocab_size] in F32
        if (head_weight_.dtype == "fp4") {
            if (is_blackwell_tc_active_) {
                quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                               buf_hidden_batch_.bf16(), M, dim, main_stream_);
                gemm_fp4_f32_blackwell_tensorcore_cuda(buf_logits_batch_.f32(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                       (const uint8_t*)head_weight_.data, (const uint8_t*)head_weight_scale_.data,
                                                       cfg_.vocab_size, dim, M, main_stream_);
            } else {
                gemm_fp4_f32_batch_cuda(buf_logits_batch_.f32(), buf_hidden_batch_.bf16(),
                                         (const uint8_t*)head_weight_.data, (const uint8_t*)head_weight_scale_.data,
                                         cfg_.vocab_size, dim, M, main_stream_);
            }
        } else if (head_weight_scale_.data) {
            gemm_int4_f32_batch_cuda(buf_logits_batch_.f32(), buf_hidden_batch_.bf16(),
                                     (const uint8_t*)head_weight_.data, head_weight_scale_.bf16(),
                                     cfg_.vocab_size, dim, M, main_stream_);
        } else {
            gemm_bf16_f32(buf_logits_batch_.f32(), M, cfg_.vocab_size, dim,
                          buf_hidden_batch_.bf16(), head_weight_.bf16());
        }
    }

    void forward_token_batch_qwen(const int* tokens_host, int position, int M) {
        if (!h_target_batch_tok_) {
            CUDA_CHECK(cudaMallocHost(&h_target_batch_tok_, 64 * sizeof(int32_t)));
            CUDA_CHECK(cudaMallocHost(&h_target_batch_pos_, 64 * sizeof(int32_t)));
        }
        for (int m = 0; m < M; m++) {
            h_target_batch_pos_[m] = position + m;
            h_target_batch_tok_[m] = tokens_host[m];
        }
        CUDA_CHECK(cudaMemcpyAsync(buf_input_pos_batch_.i32(), h_target_batch_pos_, M * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
        CUDA_CHECK(cudaMemcpyAsync(buf_input_tokens_batch_.i32(), h_target_batch_tok_, M * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));

        if (M >= 2 && M <= 8 && batch_graph_captured_[M]) {
            CUDA_CHECK(cudaGraphLaunch(batch_graph_exec_[M], main_stream_));
        } else {
            forward_token_batch_qwen_device_body(position, M);
        }
    }


    void forward_token_tree_qwen(const TreeTopology& tree, int position) {
        int M = tree.num_nodes;
        if (!h_target_batch_tok_) {
            CUDA_CHECK(cudaMallocHost(&h_target_batch_tok_, 64 * sizeof(int32_t)));
            CUDA_CHECK(cudaMallocHost(&h_target_batch_pos_, 64 * sizeof(int32_t)));
            CUDA_CHECK(cudaMallocHost(&h_tree_ancestor_mask_, 64 * sizeof(uint64_t)));
            CUDA_CHECK(cudaMallocHost(&h_tree_parent_, 64 * sizeof(int32_t)));
        }
        for (int m = 0; m < M; m++) {
            h_target_batch_pos_[m] = position + tree.depth[m];
            h_target_batch_tok_[m] = tree.tokens[m];
            h_tree_ancestor_mask_[m] = tree.ancestor_mask[m];
            h_tree_parent_[m] = tree.parent[m];
        }
        CUDA_CHECK(cudaMemcpyAsync(buf_input_pos_batch_.i32(), h_target_batch_pos_, M * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
        CUDA_CHECK(cudaMemcpyAsync(buf_input_tokens_batch_.i32(), h_target_batch_tok_, M * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
        CUDA_CHECK(cudaMemcpyAsync(buf_tree_ancestor_mask_.u64(), h_tree_ancestor_mask_, M * sizeof(uint64_t), cudaMemcpyHostToDevice, main_stream_));
        CUDA_CHECK(cudaMemcpyAsync(buf_tree_parent_indices_.i32(), h_tree_parent_, M * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));

        if (M == tree_graph_M_ && tree_graph_captured_) {
            CUDA_CHECK(cudaGraphLaunch(tree_graph_exec_, main_stream_));
        } else {
            forward_token_batch_qwen_device_body(position, M, /*compute_logits=*/true, /*is_tree=*/true);
        }
    }

    void init_batch_cuda_graphs() {
#if defined(__APPLE__)
        for (int m = 0; m <= 8; ++m) batch_graph_captured_[m] = false;
        return;
#endif
        if (cfg_.architecture != ModelArch::QWEN) return;
        std::vector<int> target_batch_sizes;
        if (gpu_caps_.is_vram_constrained) {
            int max_k = (mtp_k_ > 0) ? mtp_k_ : 1;
            int max_m = std::min(max_k + 1, 5);
            for (int m = 2; m <= max_m; ++m) {
                target_batch_sizes.push_back(m);
            }
        } else {
            int max_k = (mtp_k_ > 0) ? mtp_k_ : 4;
            int max_m = std::min(max_k + 1, 8);
            for (int m = 2; m <= max_m; ++m) {
                target_batch_sizes.push_back(m);
            }
        }
        for (int M : target_batch_sizes) {
            LOG_INFO("Warming up and capturing Batched Target CUDA Graph (M=%d)...", M);

            std::vector<int32_t> dummy_toks(M, 0);
            std::vector<int32_t> dummy_pos(M, 0);
            CUDA_CHECK(cudaMemcpy(buf_input_tokens_batch_.i32(), dummy_toks.data(), M * sizeof(int32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(buf_input_pos_batch_.i32(), dummy_pos.data(), M * sizeof(int32_t), cudaMemcpyHostToDevice));

            // Warmup
            for (int i = 0; i < 3; i++) {
                forward_token_batch_qwen_device_body(0, M);
            }
            CUDA_CHECK(cudaStreamSynchronize(main_stream_));

            cudaError_t cap_err = cudaStreamBeginCapture(main_stream_, cudaStreamCaptureModeGlobal);
            if (cap_err != cudaSuccess) {
                batch_graph_captured_[M] = false;
                continue;
            }

            forward_token_batch_qwen_device_body(0, M);

            cudaError_t end_err = cudaStreamEndCapture(main_stream_, &batch_graph_[M]);
            if (end_err != cudaSuccess) {
                batch_graph_captured_[M] = false;
                continue;
            }

            cudaError_t inst_err = cudaGraphInstantiate(&batch_graph_exec_[M], batch_graph_[M], nullptr, nullptr, 0);
            if (inst_err != cudaSuccess) {
                batch_graph_captured_[M] = false;
                continue;
            }

            batch_graph_captured_[M] = true;
            LOG_INFO("Batched Target CUDA Graph (M=%d) instantiated successfully!", M);
        }

        if (enable_tree_mtp_) {
            int M = std::max(2, std::min(4, tree_nodes_ + 1));
            tree_graph_M_ = M;
            LOG_INFO("Warming up and capturing Tree Attention CUDA Graph (nodes=%d, M=%d)...", tree_nodes_, M);
            std::vector<int32_t> dummy_toks(M, 0);
            std::vector<int32_t> dummy_pos(M, 0);
            std::vector<uint64_t> dummy_ancestors(M, 0ULL);
            std::vector<int32_t> dummy_parents(M, 0);

            for (int m = 0; m < M; m++) {
                dummy_parents[m] = (m == 0) ? -1 : (m == 3 ? 1 : 0);
                if (m == 0) dummy_ancestors[m] = 1ULL << 0;
                else if (m == 1) dummy_ancestors[m] = (1ULL << 0) | (1ULL << 1);
                else if (m == 2) dummy_ancestors[m] = (1ULL << 0) | (1ULL << 2);
                else if (m == 3) dummy_ancestors[m] = (1ULL << 0) | (1ULL << 1) | (1ULL << 3);
            }

            CUDA_CHECK(cudaMemcpy(buf_input_tokens_batch_.i32(), dummy_toks.data(), M * sizeof(int32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(buf_input_pos_batch_.i32(), dummy_pos.data(), M * sizeof(int32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(buf_tree_ancestor_mask_.u64(), dummy_ancestors.data(), M * sizeof(uint64_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(buf_tree_parent_indices_.i32(), dummy_parents.data(), M * sizeof(int32_t), cudaMemcpyHostToDevice));

            // Warmup
            for (int i = 0; i < 3; i++) {
                forward_token_batch_qwen_device_body(0, M, /*compute_logits=*/true, /*is_tree=*/true);
            }
            CUDA_CHECK(cudaStreamSynchronize(main_stream_));

            cudaError_t cap_err = cudaStreamBeginCapture(main_stream_, cudaStreamCaptureModeGlobal);
            if (cap_err != cudaSuccess) {
                tree_graph_captured_ = false;
                LOG_WARN("cudaStreamBeginCapture failed for tree graph: %s", cudaGetErrorString(cap_err));
            } else {
                forward_token_batch_qwen_device_body(0, M, /*compute_logits=*/true, /*is_tree=*/true);

                cudaError_t end_err = cudaStreamEndCapture(main_stream_, &tree_graph_);
                if (end_err != cudaSuccess) {
                    tree_graph_captured_ = false;
                    LOG_WARN("cudaStreamEndCapture failed for tree graph: %s", cudaGetErrorString(end_err));
                } else {
                    cudaError_t inst_err = cudaGraphInstantiate(&tree_graph_exec_, tree_graph_, nullptr, nullptr, 0);
                    if (inst_err != cudaSuccess) {
                        tree_graph_captured_ = false;
                        LOG_WARN("cudaGraphInstantiate failed for tree graph: %s", cudaGetErrorString(inst_err));
                    } else {
                        tree_graph_captured_ = true;
                        LOG_INFO("Tree Attention CUDA Graph (nodes=%d, M=%d) instantiated successfully!", tree_nodes_, M);
                    }
                }
            }
        }
    }

    inline void checkpoint_target_state() {
        if (target_ssm_pool_.data && target_ssm_slots_[0].data) {
            CUDA_CHECK(cudaMemcpyAsync(target_ssm_slots_[0].data, target_ssm_pool_.data, target_ssm_pool_.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
            CUDA_CHECK(cudaMemcpyAsync(target_conv_slots_[0].data, target_conv_pool_.data, target_conv_pool_.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
        }
    }

    inline void restore_target_state() {
        if (target_ssm_pool_.data && target_ssm_slots_[0].data) {
            CUDA_CHECK(cudaMemcpyAsync(target_ssm_pool_.data, target_ssm_slots_[0].data, target_ssm_pool_.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
            CUDA_CHECK(cudaMemcpyAsync(target_conv_pool_.data, target_conv_slots_[0].data, target_conv_pool_.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
        }
    }

    inline void commit_target_state_slot(int slot_idx) {
        if (slot_idx < 0 || slot_idx >= 8) return;
        if (!enable_tree_mtp_ && slot_idx >= 4) return;
        if (target_ssm_pool_.data && target_ssm_slots_[slot_idx].data) {
            CUDA_CHECK(cudaMemcpyAsync(target_ssm_pool_.data, target_ssm_slots_[slot_idx].data, target_ssm_pool_.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
            CUDA_CHECK(cudaMemcpyAsync(target_conv_pool_.data, target_conv_slots_[slot_idx].data, target_conv_pool_.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
        }
    }

    inline void compact_tree_kv_all_gqa_layers(const int* winning_slots, int pos_base, int L) {
        if (L <= 0 || !winning_slots) return;
        bool needs_compact = false;
        for (int k = 0; k < L; k++) {
            if (winning_slots[k] != (k + 1)) {
                needs_compact = true;
                break;
            }
        }
        if (!needs_compact) return;

        CUDA_CHECK(cudaMemcpyAsync(buf_tree_winning_slots_.i32(), winning_slots, L * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
        int n_kv_heads = cfg_.num_key_value_heads;
        int head_dim = cfg_.head_dim;
        for (int layer = 0; layer < cfg_.num_hidden_layers; layer++) {
            auto& lw = layers_[layer];
            if (!lw.is_linear_attn) {
                compact_tree_kv_cache_cuda(
                    lw.k_cache_gqa.u8(),
                    lw.v_cache_gqa.u8(),
                    buf_tree_winning_slots_.i32(),
                    pos_base,
                    L,
                    n_kv_heads,
                    head_dim,
                    main_stream_);
            }
        }
    }
