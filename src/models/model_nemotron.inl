    // ── Forward one layer (Nemotron 3.5 Lightning Hybrid Mamba-2 SSM + MoE) ─
    inline static cudaEvent_t s_moe_sub_ev[6] = {nullptr};
    inline static bool s_moe_sub_init = false;

    void forward_layer_nemotron(int layer_id, int token_id, int position) {
        auto& lw = layers_[layer_id];
        int dim = cfg_.hidden_size;
        int n_q_heads = cfg_.num_attention_heads;
        int n_kv_heads = cfg_.num_key_value_heads;
        int head_dim = cfg_.head_dim;
        int inter_size = cfg_.intermediate_size > 0 ? cfg_.intermediate_size : cfg_.moe_intermediate_size;
        int ltype = (layer_id < (int)cfg_.layer_types.size()) ? cfg_.layer_types[layer_id] : (lw.is_linear_attn ? 1 : (lw.gate_w.data ? 2 : 0));

        if (ltype == 1) {
            // ════════════════════════════════════════════════════════════════════════
            // 1. MAMBA-2 SSM MIXER LAYER (Nemotron 3.5 Lightning / Bolt)
            // ════════════════════════════════════════════════════════════════════════
            int mamba_heads = cfg_.is_nemotron_bolt() ? 128 : 64;
            int mamba_head_dim = 64;
            int mamba_inter = mamba_heads * mamba_head_dim;
            int in_proj_dim = cfg_.is_nemotron_bolt() ? 18568 : 10304;

            rms_norm_cuda(buf_hidden2_.bf16(), buf_hidden_.bf16(),
                          lw.attn_norm_w.bf16(), dim, cfg_.rms_norm_eps, main_stream_);

            // In-projection: FP8 [in_proj_dim, dim] with scalar scale -> buf_q_ [in_proj_dim]
            gemv_fp8_scalar_scale_cuda(buf_q_.bf16(), buf_hidden2_.bf16(),
                                       (const uint8_t*)lw.w_in_qkv.data, lw.w_in_qkv_scale_val,
                                       in_proj_dim, dim, main_stream_);

            // Mamba-2 SSM Decode Step (Causal Conv1d + SSM recurrent update + Gated RMSNorm) -> buf_attn_out_ [mamba_inter]
            mamba2_nemotron_step_cuda(
                buf_attn_out_.bf16(),
                buf_q_.bf16(),
                lw.conv1d_w.bf16(),
                lw.conv1d_bias.bf16(),
                lw.conv_state.bf16(),
                lw.A_log.bf16(),
                lw.mamba_d.bf16(),
                lw.dt_bias.bf16(),
                lw.linear_norm_w.bf16(),
                lw.ssm_state.f32(),
                buf_mamba_conv_.bf16(),
                buf_mamba_temp_.f32(),
                cfg_.rms_norm_eps,
                main_stream_,
                mamba_heads,
                mamba_head_dim);

            // Out-projection: FP8 [dim, mamba_inter] with scalar scale + residual accumulation -> buf_hidden_ [dim]
            gemv_fp8_scalar_scale_residual_cuda(
                buf_hidden_.bf16(), buf_attn_out_.bf16(),
                (const uint8_t*)lw.linear_out_proj.data, lw.linear_out_proj_scale_val,
                dim, mamba_inter, main_stream_);
            return;
        } else if (ltype == 0) {
            // ════════════════════════════════════════════════════════════════════════
            // 2. STANDARD GQA ATTENTION MIXER LAYER (Nemotron 3.5 Lightning / Bolt)
            // ════════════════════════════════════════════════════════════════════════
            rms_norm_cuda(buf_hidden2_.bf16(), buf_hidden_.bf16(),
                          lw.attn_norm_w.bf16(), dim, cfg_.rms_norm_eps, main_stream_);

            int q_dim = n_q_heads * head_dim; // 32 * 128 = 4096
            int kv_dim = n_kv_heads * head_dim;

            gemv_bf16_out_bf16_cuda(buf_q_.bf16(), lw.w_q.bf16(), buf_hidden2_.bf16(), q_dim, dim, main_stream_);
            gemv_bf16_out_bf16_cuda(buf_gate_.bf16(), lw.w_k.bf16(), buf_hidden2_.bf16(), kv_dim, dim, main_stream_);
            gemv_bf16_out_bf16_cuda(buf_up_.bf16(), lw.w_v.bf16(), buf_hidden2_.bf16(), kv_dim, dim, main_stream_);

            // Ungated GQA FP8 Decode (Nemotron-H has NoPE - rope_theta = 0.0f)
            qwen2_gqa_decode_fp8_cuda(
                buf_attn_out_.bf16(),
                buf_q_.bf16(),
                buf_gate_.bf16(),
                buf_up_.bf16(),
                nullptr,
                nullptr,
                lw.k_cache_gqa.u8(),
                lw.v_cache_gqa.u8(),
                n_q_heads, n_kv_heads, head_dim,
                buf_input_pos_.i32(), position, cfg_.max_seq_len,
                0.0f, cfg_.rms_norm_eps,
                nullptr,
                main_stream_);

            // Attention Output Projection: BF16 [dim, q_dim] + residual accumulation into buf_hidden_
            gemv_bf16_out_bf16_cuda(buf_hidden2_.bf16(), lw.w_o.bf16(), buf_attn_out_.bf16(), dim, q_dim, main_stream_);
            vector_add_bf16_cuda(buf_hidden_.bf16(), buf_hidden2_.bf16(), dim, main_stream_);
            return;
        } else {
            // ════════════════════════════════════════════════════════════════════════
            // 3. SPARSE MoE FEED-FORWARD LAYER (Nemotron 3.5 Lightning / Bolt)
            // ════════════════════════════════════════════════════════════════════════
            rms_norm_cuda(buf_hidden2_.bf16(), buf_hidden_.bf16(),
                          lw.ffn_norm_w.bf16(), dim, cfg_.rms_norm_eps, main_stream_);

            // Find MoE layer index
            int moe_layer_idx = 0;
            for (int i = 0; i < layer_id; i++) {
                if (cfg_.layer_types[i] == 2) moe_layer_idx++;
            }

            // 3a. Shared Expert (NVFP4, Squared ReLU, residual)
            if (lw.shared_w1_w.data && lw.shared_w2_w.data) {
                int shared_inter = cfg_.moe_shared_expert_intermediate_size > 0 ? cfg_.moe_shared_expert_intermediate_size : 3712;
                gemv_nvfp4_cuda(buf_gate_.bf16(), buf_hidden2_.bf16(),
                                (const uint8_t*)lw.shared_w1_w.data, (const uint8_t*)lw.shared_w1_s.data,
                                lw.shared_w1_s2, shared_inter, dim, main_stream_);
                relu2_cuda(buf_gate_.bf16(), buf_gate_.bf16(), shared_inter, main_stream_);
                gemv_nvfp4_residual_cuda(buf_hidden_.bf16(), buf_gate_.bf16(),
                                         (const uint8_t*)lw.shared_w2_w.data, (const uint8_t*)lw.shared_w2_s.data,
                                         lw.shared_w2_s2, dim, shared_inter, main_stream_);
            }

            // 3b. Routed Experts
            if (cfg_.is_nemotron_bolt()) {
                // ── Nemotron Bolt LatentMoE Forward Pass ─────────────────────────────
                int n_experts = cfg_.get_layer_n_routed_experts(layer_id);
                int top_k = cfg_.get_layer_num_experts_per_tok(layer_id);
                int moe_inter = cfg_.get_layer_moe_intermediate_size(layer_id);
                int latent_dim = cfg_.moe_latent_size > 0 ? cfg_.moe_latent_size : 1024;

                if (n_experts > 0 && lw.gate_w.data) {
                    gemm_bf16(buf_scores_bf16_.bf16(), 1, n_experts, dim,
                              buf_hidden2_.bf16(), lw.gate_w.bf16());

                    moe_route_topk_from_bf16_cuda(buf_topk_idx_.i32(), buf_topk_vals_.f32(),
                                                  buf_scores_bf16_.bf16(),
                                                  lw.gate_bias.data ? lw.gate_bias.f32() : nullptr,
                                                  n_experts, top_k, cfg_.routed_scaling_factor, main_stream_);

                    CUDA_CHECK(cudaMemcpyAsync(topk_ids_host_, buf_topk_idx_.i32(),
                                               top_k * sizeof(int32_t), cudaMemcpyDeviceToHost, main_stream_));
                    CUDA_CHECK(cudaMemcpyAsync(topk_weights_host_, buf_topk_vals_.f32(),
                                               top_k * sizeof(float), cudaMemcpyDeviceToHost, main_stream_));
                    CUDA_CHECK(cudaEventRecord(dma_event_, main_stream_));
                    CUDA_CHECK(cudaEventSynchronize(dma_event_));

                    // Latent Down-Projection: buf_hidden2_ [4096] -> buf_latent_down_ [1024]
                    if (lw.fc1_latent_proj.dtype == "fp8") {
                        gemv_fp8_scalar_scale_cuda(buf_latent_down_.bf16(), buf_hidden2_.bf16(),
                                                   (const uint8_t*)lw.fc1_latent_proj.data, lw.fc1_scale_val,
                                                   latent_dim, dim, main_stream_);
                    } else {
                        gemv_bf16_out_bf16_cuda(buf_latent_down_.bf16(), lw.fc1_latent_proj.bf16(),
                                                buf_hidden2_.bf16(), latent_dim, dim, main_stream_);
                    }

                    // Resolve pointers / DMA for all active top-k experts
                    void* exp_ptrs[16] = {nullptr};
                    for (int k = 0; k < top_k; k++) {
                        int eid = topk_ids_host_[k];
                        if (eid < 0 || eid >= n_experts || topk_weights_host_[k] == 0.0f) continue;
                        exp_ptrs[k] = expert_loader_.try_get_expert_cached(moe_layer_idx, eid);
                        if (!exp_ptrs[k]) {
                            void* dram_src = expert_loader_.get_dram_expert_ptr(moe_layer_idx, eid);
                            void* scratch_gpu = (char*)buf_dram_moe_scratch_.data + (size_t)k * expert_loader_.expert_block_size();
                            if (dram_src && scratch_gpu) {
                                CUDA_CHECK(cudaMemcpyAsync(scratch_gpu, dram_src, expert_loader_.expert_block_size(),
                                                           cudaMemcpyHostToDevice, main_stream_));
                                exp_ptrs[k] = scratch_gpu;
                            }
                        }
                    }

                    size_t up_w_bytes = (size_t)moe_inter * (latent_dim / 2);
                    size_t up_s_bytes = (size_t)moe_inter * (latent_dim / 16);
                    size_t down_w_bytes = (size_t)latent_dim * (moe_inter / 2);
                    size_t down_s_bytes = (size_t)latent_dim * (moe_inter / 16);

                    for (int k = 0; k < top_k; k++) {
                        int eid = topk_ids_host_[k];
                        float weight = topk_weights_host_[k];
                        if (eid < 0 || eid >= n_experts || weight == 0.0f) continue;

                        void* exp_ptr = exp_ptrs[k];
                        if (!exp_ptr) continue;

                        const uint8_t* up_w = (const uint8_t*)exp_ptr;
                        const uint8_t* up_s = up_w + up_w_bytes;
                        const uint8_t* down_w = up_s + up_s_bytes;
                        const uint8_t* down_s = down_w + down_w_bytes;

                        float up_s2 = 1.0f;
                        float down_s2 = 1.0f;
                        int stride_exp = std::max({cfg_.n_routed_experts, cfg_.max_n_routed_experts(), 128});
                        size_t s_idx = ((size_t)moe_layer_idx * stride_exp + eid) * 2;
                        if (s_idx + 1 < nemotron_expert_scales_.size()) {
                            up_s2 = nemotron_expert_scales_[s_idx];
                            down_s2 = nemotron_expert_scales_[s_idx + 1];
                        }

                        // 1. Up-projection on latent input vector: [moe_inter, latent_dim]
                        gemv_nvfp4_cuda(buf_gate_.bf16(), buf_latent_down_.bf16(),
                                        up_w, up_s, up_s2, moe_inter, latent_dim, main_stream_);

                        // 2. Squared ReLU:
                        relu2_cuda(buf_gate_.bf16(), buf_gate_.bf16(), moe_inter, main_stream_);

                        // 3. Down-projection into buf_down_ slot k: [latent_dim, moe_inter]
                        gemv_nvfp4_cuda(buf_down_.bf16() + (size_t)k * latent_dim, buf_gate_.bf16(),
                                        down_w, down_s, down_s2, latent_dim, moe_inter, main_stream_);
                    }

                    // 4. Accumulate active top-k experts in latent space [latent_dim]
                    fused_moe_accum_dynamic_cuda(buf_latent_accum_.bf16(), buf_down_.bf16(),
                                                 buf_topk_vals_.f32(), nullptr, latent_dim, top_k, main_stream_);

                    // 5. Latent Up-Projection: buf_latent_accum_ [1024] -> residual addition into buf_hidden_ [4096]
                    if (lw.fc2_latent_proj.dtype == "fp8") {
                        gemv_fp8_scalar_scale_residual_cuda(buf_hidden_.bf16(), buf_latent_accum_.bf16(),
                                                            (const uint8_t*)lw.fc2_latent_proj.data, lw.fc2_scale_val,
                                                            dim, latent_dim, main_stream_);
                    } else {
                        gemv_bf16_out_bf16_cuda(buf_hidden2_.bf16(), lw.fc2_latent_proj.bf16(),
                                                buf_latent_accum_.bf16(), dim, latent_dim, main_stream_);
                        vector_add_bf16_cuda(buf_hidden_.bf16(), buf_hidden2_.bf16(), dim, main_stream_);
                    }
                }
            } else if (cfg_.n_routed_experts > 0 && lw.gate_w.data) {
                // ── Nemotron Lightning Top-6 Routed Experts ──────────────────────────
                int n_experts = cfg_.n_routed_experts;
                int top_k = cfg_.num_experts_per_tok > 0 ? cfg_.num_experts_per_tok : 6;
                int moe_inter = cfg_.moe_intermediate_size > 0 ? cfg_.moe_intermediate_size : inter_size;

                gemm_bf16(buf_scores_bf16_.bf16(), 1, n_experts, dim,
                          buf_hidden2_.bf16(), lw.gate_w.bf16());

                moe_route_top6_from_bf16_cuda(buf_topk_idx_.i32(), buf_topk_vals_.f32(),
                                              buf_scores_bf16_.bf16(),
                                              lw.gate_bias.data ? lw.gate_bias.f32() : nullptr,
                                              n_experts, top_k, cfg_.routed_scaling_factor, main_stream_);

                CUDA_CHECK(cudaMemcpyAsync(topk_ids_host_, buf_topk_idx_.i32(),
                                           top_k * sizeof(int32_t), cudaMemcpyDeviceToHost, main_stream_));
                CUDA_CHECK(cudaMemcpyAsync(topk_weights_host_, buf_topk_vals_.f32(),
                                           top_k * sizeof(float), cudaMemcpyDeviceToHost, main_stream_));
                CUDA_CHECK(cudaEventRecord(dma_event_, main_stream_));
                CUDA_CHECK(cudaEventSynchronize(dma_event_));

                // Batch resolve / pipeline DMA transfers for all top-k experts
                void* exp_ptrs[8] = {nullptr};
                for (int k = 0; k < top_k; k++) {
                    int eid = topk_ids_host_[k];
                    if (eid < 0 || eid >= n_experts || topk_weights_host_[k] == 0.0f) continue;
                    exp_ptrs[k] = expert_loader_.try_get_expert_cached(moe_layer_idx, eid);
                    if (!exp_ptrs[k]) {
                        void* dram_src = expert_loader_.get_dram_expert_ptr(moe_layer_idx, eid);
                        void* scratch_gpu = (char*)buf_dram_moe_scratch_.data + (size_t)k * expert_loader_.expert_block_size();
                        if (dram_src && scratch_gpu) {
                            CUDA_CHECK(cudaMemcpyAsync(scratch_gpu, dram_src, expert_loader_.expert_block_size(),
                                                       cudaMemcpyHostToDevice, main_stream_));
                            exp_ptrs[k] = scratch_gpu;
                        }
                    }
                }

                for (int k = 0; k < top_k; k++) {
                    int eid = topk_ids_host_[k];
                    float weight = topk_weights_host_[k];
                    if (eid < 0 || eid >= n_experts || weight == 0.0f) continue;

                    void* exp_ptr = exp_ptrs[k];
                    if (!exp_ptr) continue;

                    // Expert block layout:
                    // up_proj.weight: offset 0 [moe_inter, dim/2] nvfp4
                    // up_proj.weight_scale: offset 2494464 [moe_inter, dim/16] F8_E4M3
                    // down_proj.weight: offset 2806288 [dim, moe_inter/2] nvfp4
                    // down_proj.weight_scale: offset 5300752 [dim, moe_inter/16] F8_E4M3
                    const uint8_t* up_w = (const uint8_t*)exp_ptr;
                    const uint8_t* up_s = (const uint8_t*)exp_ptr + 2494464;
                    const uint8_t* down_w = (const uint8_t*)exp_ptr + 2806288;
                    const uint8_t* down_s = (const uint8_t*)exp_ptr + 5300752;

                    float up_s2 = 1.0f;
                    float down_s2 = 1.0f;
                    int stride_exp = std::max({cfg_.n_routed_experts, cfg_.max_n_routed_experts(), 128});
                    size_t s_idx = ((size_t)moe_layer_idx * stride_exp + eid) * 2;
                    if (s_idx + 1 < nemotron_expert_scales_.size()) {
                        up_s2 = nemotron_expert_scales_[s_idx];
                        down_s2 = nemotron_expert_scales_[s_idx + 1];
                    }

                    // 1. Up proj: [moe_inter, dim]
                    gemv_nvfp4_cuda(buf_gate_.bf16(), buf_hidden2_.bf16(),
                                    up_w, up_s, up_s2, moe_inter, dim, main_stream_);

                    // 2. Squared ReLU activation: ReLU(x)^2
                    relu2_cuda(buf_gate_.bf16(), buf_gate_.bf16(), moe_inter, main_stream_);

                    // 3. Down proj with scaling weight & residual accumulation: [dim, moe_inter]
                    gemv_nvfp4_scaled_residual_cuda(buf_hidden_.bf16(), buf_gate_.bf16(),
                                                    down_w, down_s, down_s2, weight,
                                                    dim, moe_inter, main_stream_);
                }
            }
            return;
        }
    }

    // ── Forward one layer batched (Nemotron 3.5 Lightning Hybrid Mamba-2 SSM + MoE) ─

    void forward_layer_nemotron_batch(int layer_id, int position, int M) {
        auto& lw = layers_[layer_id];
        int dim = cfg_.hidden_size;
        int n_q_heads = cfg_.num_attention_heads;
        int n_kv_heads = cfg_.num_key_value_heads;
        int head_dim = cfg_.head_dim;
        int inter_size = cfg_.intermediate_size > 0 ? cfg_.intermediate_size : cfg_.moe_intermediate_size;
        int ltype = (layer_id < (int)cfg_.layer_types.size()) ? cfg_.layer_types[layer_id] : (lw.is_linear_attn ? 1 : (lw.gate_w.data ? 2 : 0));

        if (ltype == 1) {
            // ════════════════════════════════════════════════════════════════════════
            // 1. MAMBA-2 SSM MIXER LAYER (Nemotron 3.5 Lightning / Bolt)
            // ════════════════════════════════════════════════════════════════════════
            int mamba_heads = cfg_.is_nemotron_bolt() ? 128 : 64;
            int mamba_head_dim = 64;
            int mamba_inter = mamba_heads * mamba_head_dim;
            int in_proj_dim = cfg_.is_nemotron_bolt() ? 18568 : 10304;

            rms_norm_cuda_batched(buf_hidden2_batch_.bf16(), buf_hidden_batch_.bf16(),
                                  lw.attn_norm_w.bf16(), M, dim, cfg_.rms_norm_eps, main_stream_);

            // In-projection: Batched FP8 [M, in_proj_dim] with scalar scale -> buf_q_batch_
            gemm_fp8_scalar_scale_cublaslt(buf_q_batch_.bf16(), buf_hidden2_batch_.bf16(),
                                           (const uint8_t*)lw.w_in_qkv.data, lw.w_in_qkv_scale_val,
                                           M, in_proj_dim, dim, 0.0f, main_stream_);

            // Mamba-2 SSM Step (Single batched launch across all M tokens)
            mamba2_nemotron_step_batch_cuda(
                buf_attn_out_batch_.bf16(),
                buf_q_batch_.bf16(),
                lw.conv1d_w.bf16(),
                lw.conv1d_bias.bf16(),
                lw.conv_state.bf16(),
                lw.A_log.bf16(),
                lw.mamba_d.bf16(),
                lw.dt_bias.bf16(),
                lw.linear_norm_w.bf16(),
                lw.ssm_state.f32(),
                buf_mamba_conv_batch_.bf16(),
                buf_mamba_temp_batch_.f32(),
                M,
                cfg_.rms_norm_eps,
                main_stream_,
                mamba_heads,
                mamba_head_dim);

            // Out-projection: Batched FP8 [M, dim] with scalar scale + residual accumulation -> buf_hidden_batch_
            gemm_fp8_scalar_scale_cublaslt(
                buf_hidden_batch_.bf16(), buf_attn_out_batch_.bf16(),
                (const uint8_t*)lw.linear_out_proj.data, lw.linear_out_proj_scale_val,
                M, dim, mamba_inter, 1.0f, main_stream_);
            return;
        } else if (ltype == 0) {
            // ════════════════════════════════════════════════════════════════════════
            // 2. STANDARD GQA ATTENTION MIXER LAYER (Nemotron 3.5 Lightning / Bolt)
            // ════════════════════════════════════════════════════════════════════════
            rms_norm_cuda_batched(buf_hidden2_batch_.bf16(), buf_hidden_batch_.bf16(),
                                  lw.attn_norm_w.bf16(), M, dim, cfg_.rms_norm_eps, main_stream_);

            int q_dim = n_q_heads * head_dim; // 32 * 128 = 4096
            int kv_dim = n_kv_heads * head_dim; // 2 * 128 = 256

            // Batched Q, K, V Projections
            gemm_bf16(buf_q_batch_.bf16(), M, q_dim, dim, buf_hidden2_batch_.bf16(), lw.w_q.bf16());
            gemm_bf16(buf_gate_batch_.bf16(), M, kv_dim, dim, buf_hidden2_batch_.bf16(), lw.w_k.bf16());
            gemm_bf16(buf_up_batch_.bf16(), M, kv_dim, dim, buf_hidden2_batch_.bf16(), lw.w_v.bf16());

            // Batched Ungated GQA FP8 Decode (Nemotron-H has NoPE - rope_theta = 0.0f)
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
                0.0f, cfg_.rms_norm_eps,
                buf_mrope_pos_.data ? buf_mrope_pos_.i32() : nullptr,
                main_stream_);

            // Attention Output Projection: BF16 [M, dim] + residual accumulation into buf_hidden_batch_
            gemm_bf16(buf_hidden2_batch_.bf16(), M, dim, q_dim, buf_attn_out_batch_.bf16(), lw.w_o.bf16());
            vector_add_bf16_cuda(buf_hidden_batch_.bf16(), buf_hidden2_batch_.bf16(), M * dim, main_stream_);
            return;
        } else {
            // ════════════════════════════════════════════════════════════════════════
            // 3. SPARSE MoE FEED-FORWARD LAYER (Nemotron 3.5 Lightning / Bolt)
            // ════════════════════════════════════════════════════════════════════════
            rms_norm_cuda_batched(buf_hidden2_batch_.bf16(), buf_hidden_batch_.bf16(),
                                  lw.ffn_norm_w.bf16(), M, dim, cfg_.rms_norm_eps, main_stream_);

            // Find MoE layer index (0..39 for Bolt, 0..22 for Lightning)
            int moe_layer_idx = 0;
            for (int i = 0; i < layer_id; i++) {
                if (cfg_.layer_types[i] == 2) moe_layer_idx++;
            }

            int stride_exp = std::max({cfg_.n_routed_experts, cfg_.max_n_routed_experts(), 128});
            const float* layer_scales = (const float*)buf_expert_scales_gpu_.data + (size_t)moe_layer_idx * stride_exp * 2;

            if (!s_moe_sub_init) {
                s_moe_sub_init = true;
                for (int i = 0; i < 6; i++) CUDA_CHECK(cudaEventCreate(&s_moe_sub_ev[i]));
            }
            if (layer_id == 1) CUDA_CHECK(cudaEventRecord(s_moe_sub_ev[0], main_stream_));

            // Shared Expert: Run synchronously on main_stream_ with direct residual accumulation
            bool has_shared = (lw.shared_w1_w.data && lw.shared_w2_w.data);
            if (has_shared) {
                int shared_inter = cfg_.moe_shared_expert_intermediate_size > 0 ? cfg_.moe_shared_expert_intermediate_size : (cfg_.is_nemotron_bolt() ? 5376 : 3712);
                gemm_nvfp4_relu2_batch_cuda(buf_gate_batch_.bf16(), buf_hidden2_batch_.bf16(),
                                            (const uint8_t*)lw.shared_w1_w.data, (const uint8_t*)lw.shared_w1_s.data,
                                            lw.shared_w1_s2, M, shared_inter, dim, main_stream_);
                gemm_nvfp4_residual_batch_cuda(buf_hidden_batch_.bf16(), buf_gate_batch_.bf16(),
                                               (const uint8_t*)lw.shared_w2_w.data, (const uint8_t*)lw.shared_w2_s.data,
                                               lw.shared_w2_s2, M, dim, shared_inter, main_stream_);
            }

            if (layer_id == 1) CUDA_CHECK(cudaEventRecord(s_moe_sub_ev[1], main_stream_));

            if (cfg_.is_nemotron_bolt()) {
                // ── Nemotron Bolt Batched LatentMoE Forward Pass ─────────────────────
                int n_experts = cfg_.get_layer_n_routed_experts(layer_id);
                int top_k = cfg_.get_layer_num_experts_per_tok(layer_id);
                int moe_inter = cfg_.get_layer_moe_intermediate_size(layer_id);
                int latent_dim = cfg_.moe_latent_size > 0 ? cfg_.moe_latent_size : 1024;
                int total_moe_layers = 40;

                if (n_experts > 0 && lw.gate_w.data) {
                    gemm_bf16(buf_scores_bf16_batch_.bf16(), M, n_experts, dim,
                              buf_hidden2_batch_.bf16(), lw.gate_w.bf16());

                    moe_route_top6_from_bf16_batch_cuda(
                        buf_topk_idx_batch_.i32(), buf_topk_vals_batch_.f32(),
                        buf_scores_bf16_batch_.bf16(),
                        lw.gate_bias.data ? lw.gate_bias.f32() : nullptr,
                        M, n_experts, top_k, cfg_.routed_scaling_factor, main_stream_);

                    moe_bin_tokens_cuda(buf_moe_offsets_.i32(), buf_moe_token_indices_.i32(), buf_moe_token_slots_.i32(),
                                        buf_topk_idx_batch_.i32(), M * top_k, top_k, n_experts, main_stream_);

                    // Latent Down-Projection: buf_hidden2_batch_ [M, 4096] -> buf_latent_down_batch_ [M, 1024]
                    if (lw.fc1_latent_proj.dtype == "fp8") {
                        gemm_fp8_scalar_scale_cublaslt(buf_latent_down_batch_.bf16(), buf_hidden2_batch_.bf16(),
                                                       (const uint8_t*)lw.fc1_latent_proj.data, lw.fc1_scale_val,
                                                       M, latent_dim, dim, 0.0f, main_stream_);
                    } else {
                        gemm_bf16(buf_latent_down_batch_.bf16(), M, latent_dim, dim,
                                  buf_hidden2_batch_.bf16(), lw.fc1_latent_proj.bf16());
                    }

                    // Zero latent accumulation buffer: [M, latent_dim]
                    CUDA_CHECK(cudaMemsetAsync(buf_latent_accum_batch_.data, 0,
                                               (size_t)M * latent_dim * sizeof(__nv_bfloat16), main_stream_));

                    const void* const* layer_ptrs = nullptr;
                    if (layer_fully_resident_[moe_layer_idx]) {
                        // Lookahead DRAM prefetch if needed (for 16GB mode: 18 layers in VRAM, 22 in DRAM)
                        int vram_threshold = 18;
                        if (dma_stream_ && moe_layer_idx >= (vram_threshold - NUM_DRAM_MOE_BUFFERS) && moe_layer_idx < vram_threshold) {
                            int prefetch_layer = moe_layer_idx + NUM_DRAM_MOE_BUFFERS;
                            int buf_idx = prefetch_layer - vram_threshold;
                            prefetch_dram_layer_async(prefetch_layer, buf_idx);
                        }
                        layer_ptrs = (const void* const*)buf_all_expert_ptrs_.data + (size_t)moe_layer_idx * stride_exp;
                    } else {
                        int vram_threshold = 18;
                        int buf_idx = (moe_layer_idx - vram_threshold) % NUM_DRAM_MOE_BUFFERS;
                        if (dma_stream_) {
                            CUDA_CHECK(cudaStreamWaitEvent(main_stream_, dma_ready_event_[buf_idx], 0));
                        }
                        layer_ptrs = (const void* const*)buf_dram_expert_ptrs_gpu_[buf_idx].data;
                    }

                    // Routed Up on latent down: [M, latent_dim] -> buf_moe_gate_batch_ [M * top_k, moe_inter]
                    moe_nemotron_up_relu2_batch_cuda(
                        buf_moe_gate_batch_.bf16(), buf_latent_down_batch_.bf16(),
                        buf_topk_idx_batch_.i32(),
                        buf_moe_offsets_.i32(), buf_moe_token_indices_.i32(), buf_moe_token_slots_.i32(),
                        layer_ptrs,
                        layer_scales,
                        M, moe_inter, latent_dim, main_stream_, top_k, n_experts);

                    // Routed Down onto latent accumulation buffer: [M * top_k, moe_inter] -> buf_latent_accum_batch_ [M, latent_dim]
                    moe_nemotron_down_scaled_residual_batch_cuda(
                        buf_latent_accum_batch_.bf16(), buf_moe_gate_batch_.bf16(),
                        buf_topk_idx_batch_.i32(), buf_topk_vals_batch_.f32(),
                        buf_moe_offsets_.i32(), buf_moe_token_indices_.i32(), buf_moe_token_slots_.i32(),
                        layer_ptrs,
                        layer_scales,
                        M, latent_dim, moe_inter, main_stream_, top_k, n_experts);

                    // If DRAM layer, trigger prefetch for next DRAM layer
                    if (!layer_fully_resident_[moe_layer_idx] && dma_stream_) {
                        int vram_threshold = 18;
                        int buf_idx = (moe_layer_idx - vram_threshold) % NUM_DRAM_MOE_BUFFERS;
                        CUDA_CHECK(cudaEventRecord(compute_done_event_[buf_idx], main_stream_));
                        int next_layer = moe_layer_idx + NUM_DRAM_MOE_BUFFERS;
                        if (next_layer < total_moe_layers) {
                            CUDA_CHECK(cudaStreamWaitEvent(dma_stream_, compute_done_event_[buf_idx], 0));
                            prefetch_dram_layer_async(next_layer, buf_idx);
                        }
                    }

                    // Latent Up-Projection: buf_latent_accum_batch_ [M, 1024] -> residual add into buf_hidden_batch_ [M, 4096]
                    if (lw.fc2_latent_proj.dtype == "fp8") {
                        gemm_fp8_scalar_scale_cublaslt(buf_hidden_batch_.bf16(), buf_latent_accum_batch_.bf16(),
                                                       (const uint8_t*)lw.fc2_latent_proj.data, lw.fc2_scale_val,
                                                       M, dim, latent_dim, 1.0f, main_stream_);
                    } else {
                        gemm_bf16(buf_hidden2_batch_.bf16(), M, dim, latent_dim,
                                  buf_latent_accum_batch_.bf16(), lw.fc2_latent_proj.bf16());
                        vector_add_bf16_cuda(buf_hidden_batch_.bf16(), buf_hidden2_batch_.bf16(), M * dim, main_stream_);
                    }
                }
                return;
            }

            // ── Nemotron Lightning Routed MoE ────────────────────────────────────────
            int n_experts = cfg_.n_routed_experts;
            int top_k = cfg_.num_experts_per_tok > 0 ? cfg_.num_experts_per_tok : 6;
            int moe_inter = cfg_.moe_intermediate_size > 0 ? cfg_.moe_intermediate_size : inter_size;

            if (cfg_.n_routed_experts > 0 && lw.gate_w.data) {
                gemm_bf16(buf_scores_bf16_batch_.bf16(), M, n_experts, dim,
                          buf_hidden2_batch_.bf16(), lw.gate_w.bf16());

                moe_route_top6_from_bf16_batch_cuda(
                    buf_topk_idx_batch_.i32(), buf_topk_vals_batch_.f32(),
                    buf_scores_bf16_batch_.bf16(),
                    lw.gate_bias.data ? lw.gate_bias.f32() : nullptr,
                    M, n_experts, top_k, cfg_.routed_scaling_factor, main_stream_);

                // Bin tokens into expert bins for accelerated batched execution
                moe_bin_tokens_cuda(buf_moe_offsets_.i32(), buf_moe_token_indices_.i32(), buf_moe_token_slots_.i32(),
                                    buf_topk_idx_batch_.i32(), M * top_k, top_k, n_experts, main_stream_);
            }

            if (layer_fully_resident_[moe_layer_idx]) {
                // Staged Lookahead Prefetch: Stream one DRAM layer at a time during late VRAM MoE layers (7..10)
                // so initial DRAM layers (11..14) arrive smoothly without saturating PCIe during prompt start.
                if (dma_stream_ && moe_layer_idx >= (11 - NUM_DRAM_MOE_BUFFERS) && moe_layer_idx < 11) {
                    int prefetch_layer = moe_layer_idx + NUM_DRAM_MOE_BUFFERS;
                    int buf_idx = prefetch_layer - 11;
                    prefetch_dram_layer_async(prefetch_layer, buf_idx);
                }

                // Routed Experts (Top-6, Squared ReLU, scaled residual)
                if (cfg_.n_routed_experts > 0 && lw.gate_w.data) {
                    const void* const* layer_ptrs = (const void* const*)buf_all_expert_ptrs_.data + (size_t)moe_layer_idx * stride_exp;

                    moe_nemotron_up_relu2_batch_cuda(
                        buf_moe_gate_batch_.bf16(), buf_hidden2_batch_.bf16(),
                        buf_topk_idx_batch_.i32(),
                        buf_moe_offsets_.i32(), buf_moe_token_indices_.i32(), buf_moe_token_slots_.i32(),
                        layer_ptrs,
                        layer_scales,
                        M, moe_inter, dim, main_stream_, top_k, n_experts);

                    if (layer_id == 1) CUDA_CHECK(cudaEventRecord(s_moe_sub_ev[3], main_stream_));

                    moe_nemotron_down_scaled_residual_batch_cuda(
                        buf_hidden_batch_.bf16(), buf_moe_gate_batch_.bf16(),
                        buf_topk_idx_batch_.i32(), buf_topk_vals_batch_.f32(),
                        buf_moe_offsets_.i32(), buf_moe_token_indices_.i32(), buf_moe_token_slots_.i32(),
                        layer_ptrs,
                        layer_scales,
                        M, dim, moe_inter, main_stream_, top_k, n_experts);

                    if (layer_id == 1) CUDA_CHECK(cudaEventRecord(s_moe_sub_ev[4], main_stream_));
                }

                if (layer_id == 1) CUDA_CHECK(cudaEventRecord(s_moe_sub_ev[5], main_stream_));
                return;
            }

            // Path for DRAM layers (layers 11..22):
            int buf_idx = (moe_layer_idx - 11) % NUM_DRAM_MOE_BUFFERS;

            // Wait for prefetched layer in buf_idx (transferred asynchronously in background)
            if (dma_stream_) {
                CUDA_CHECK(cudaStreamWaitEvent(main_stream_, dma_ready_event_[buf_idx], 0));
            }

            const void* const* layer_ptrs = (const void* const*)buf_dram_expert_ptrs_gpu_[buf_idx].data;

            // Launch Routed Up and Down on main_stream_
            moe_nemotron_up_relu2_batch_cuda(
                buf_moe_gate_batch_.bf16(), buf_hidden2_batch_.bf16(),
                buf_topk_idx_batch_.i32(),
                buf_moe_offsets_.i32(), buf_moe_token_indices_.i32(), buf_moe_token_slots_.i32(),
                layer_ptrs,
                layer_scales,
                M, moe_inter, dim, main_stream_, top_k, n_experts);

            moe_nemotron_down_scaled_residual_batch_cuda(
                buf_hidden_batch_.bf16(), buf_moe_gate_batch_.bf16(),
                buf_topk_idx_batch_.i32(), buf_topk_vals_batch_.f32(),
                buf_moe_offsets_.i32(), buf_moe_token_indices_.i32(), buf_moe_token_slots_.i32(),
                layer_ptrs,
                layer_scales,
                M, dim, moe_inter, main_stream_, top_k, n_experts);

            // Signal compute done on buf_idx and trigger prefetch for next layer using this buffer (Lookahead = NUM_DRAM_MOE_BUFFERS)
            if (dma_stream_) {
                CUDA_CHECK(cudaEventRecord(compute_done_event_[buf_idx], main_stream_));
                int next_layer = moe_layer_idx + NUM_DRAM_MOE_BUFFERS;
                if (next_layer <= 22) {
                    CUDA_CHECK(cudaStreamWaitEvent(dma_stream_, compute_done_event_[buf_idx], 0));
                    prefetch_dram_layer_async(next_layer, buf_idx);
                }
            }
            return;
        }
    }

    void forward_token_batch_nemotron_device_body(int position, int M, bool do_compute_logits = false) {
        int dim = cfg_.hidden_size;

        // 1. Embedding lookup for M tokens
        if (embed_weight_.dtype == "fp4") {
            embedding_fp4_cuda(buf_hidden_batch_.bf16(), (const uint8_t*)embed_weight_.data, (const uint8_t*)embed_weight_scale_.data, buf_input_tokens_batch_.i32(), M, dim, main_stream_);
        } else if (embed_weight_.dtype == "int4") {
            embedding_int4_cuda(buf_hidden_batch_.bf16(), (uint8_t*)embed_weight_.data, embed_weight_scale_.bf16(), buf_input_tokens_batch_.i32(), M, dim, main_stream_);
        } else {
            embedding_cuda(buf_hidden_batch_.bf16(), embed_weight_.bf16(), buf_input_tokens_batch_.i32(), M, dim, main_stream_);
        }

        // Vision tower embedding for image tokens
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

        // 2. Process all layers with profiling
        cudaStreamSynchronize(main_stream_);
        auto t_layers_start = std::chrono::high_resolution_clock::now();
        static cudaEvent_t s_layer_events[128] = {nullptr};
        static bool s_events_init = false;
        if (!s_events_init) {
            s_events_init = true;
            for (int i = 0; i < 128; i++) {
                CUDA_CHECK(cudaEventCreate(&s_layer_events[i]));
            }
        }


        CUDA_CHECK(cudaEventRecord(s_layer_events[0], main_stream_));
        for (int layer = 0; layer < cfg_.num_hidden_layers; layer++) {
            forward_layer_nemotron_batch(layer, position, M);
            CUDA_CHECK(cudaEventRecord(s_layer_events[layer + 1], main_stream_));
        }

        cudaStreamSynchronize(main_stream_);
        auto t_layers_end = std::chrono::high_resolution_clock::now();
        float total_layers_ms = std::chrono::duration<float, std::milli>(t_layers_end - t_layers_start).count();

        float ms_ssm = 0.0f, ms_gqa = 0.0f, ms_vram_moe = 0.0f, ms_dram_moe = 0.0f;
        for (int layer = 0; layer < cfg_.num_hidden_layers; layer++) {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, s_layer_events[layer], s_layer_events[layer + 1]);
            int ltype = (layer < (int)cfg_.layer_types.size()) ? cfg_.layer_types[layer] : 0;
            if (ltype == 2) {
                int moe_idx = 0;
                for (int i = 0; i < layer; i++) if (cfg_.layer_types[i] == 2) moe_idx++;
                LOG_INFO("  [LAYER %2d] MoE %2d (%s) ms=%.2f", layer, moe_idx, layer_fully_resident_[moe_idx] ? "VRAM" : "DRAM", ms);
            }
            if (ltype == 1) ms_ssm += ms;
            else if (ltype == 0) ms_gqa += ms;
            else {
                int moe_idx = 0;
                for (int i = 0; i < layer; i++) if (cfg_.layer_types[i] == 2) moe_idx++;
                if (layer_fully_resident_[moe_idx]) {
                    ms_vram_moe += ms;
                } else {
                    ms_dram_moe += ms;
                }
            }
        }

        int n_vram_moe = 0, n_dram_moe = 0;
        int n_moe_total = 0;
        for (int i = 0; i < cfg_.num_hidden_layers; i++) {
            if (cfg_.layer_types[i] == 2) {
                if (layer_fully_resident_[n_moe_total]) n_vram_moe++;
                else n_dram_moe++;
                n_moe_total++;
            }
        }

        LOG_INFO("[NEMOTRON BREAKDOWN M=%d] Total: %.2f ms (%.1f tok/s) | SSM: %.2f ms | GQA: %.2f ms | VRAM MoE (%d layers): %.2f ms (%.2f ms/l) | DRAM MoE (%d layers): %.2f ms (%.2f ms/l)",
                 M, total_layers_ms, (float)M / (total_layers_ms / 1000.0f),
                 ms_ssm, ms_gqa, ms_vram_moe, n_vram_moe > 0 ? (ms_vram_moe / n_vram_moe) : 0.0f,
                 n_dram_moe, ms_dram_moe, n_dram_moe > 0 ? (ms_dram_moe / n_dram_moe) : 0.0f);

        if (s_moe_sub_init) {
            float t_router = 0, t_shared = 0, t_up = 0, t_down = 0, t_end = 0;
            cudaEventElapsedTime(&t_router, s_moe_sub_ev[0], s_moe_sub_ev[1]);
            cudaEventElapsedTime(&t_shared, s_moe_sub_ev[1], s_moe_sub_ev[2]);
            cudaEventElapsedTime(&t_up, s_moe_sub_ev[2], s_moe_sub_ev[3]);
            cudaEventElapsedTime(&t_down, s_moe_sub_ev[3], s_moe_sub_ev[4]);
            cudaEventElapsedTime(&t_end, s_moe_sub_ev[4], s_moe_sub_ev[5]);
            LOG_INFO("  [MOE SUB-STEP Layer 1 M=%d] Router+Bin: %.2f ms | SharedExpert: %.2f ms | Up: %.2f ms | Down: %.2f ms | LayerEnd: %.2f ms | Sum: %.2f ms",
                     M, t_router, t_shared, t_up, t_down, t_end, t_router + t_shared + t_up + t_down + t_end);
        }

        // Always copy the last token's hidden state into buf_hidden_ (for generation / continuation)
        CUDA_CHECK(cudaMemcpyAsync(buf_hidden_.bf16(),
                                   buf_hidden_batch_.bf16() + (size_t)(M - 1) * dim,
                                   dim * sizeof(__nv_bfloat16),
                                   cudaMemcpyDeviceToDevice, main_stream_));

        if (do_compute_logits) {
            // 3. Final RMSNorm on the last token (in buf_hidden_)
            rms_norm_cuda(buf_hidden_.bf16(), buf_hidden_.bf16(),
                          norm_weight_.bf16(), dim, cfg_.rms_norm_eps, main_stream_);

            // 4. Compute logits for the last token
            compute_logits();
        }
    }

    void prefill_prompt_batched_nemotron(const std::vector<int>& prompt, int start_offset = 0, bool compute_last_logits = true) {
        if (prompt.empty()) return;
        int remaining_tokens = (int)prompt.size() - start_offset;
        if (remaining_tokens <= 0) {
            if (compute_last_logits) {
                rms_norm_cuda(buf_hidden_.bf16(), buf_hidden_.bf16(),
                              norm_weight_.bf16(), cfg_.hidden_size, cfg_.rms_norm_eps, main_stream_);
                compute_logits();
                CUDA_CHECK(cudaStreamSynchronize(main_stream_));
            }
            return;
        }
        if (remaining_tokens == 1) {
            forward_token_eager(prompt[start_offset], start_offset, compute_last_logits);
            CUDA_CHECK(cudaStreamSynchronize(main_stream_));
            return;
        }

        int total_tokens = (int)prompt.size();
        int chunk_size = 1024; // M=1024 hardware prefill chunk size

        ensure_prefill_host_capacity(total_tokens);

        static bool s_warmed_up = false;
        if (!s_warmed_up) {
            s_warmed_up = true;
            int warm_M = 2;
            int32_t warm_tok[2] = {1, 128803};
            int32_t warm_pos[2] = {0, 1};
            CUDA_CHECK(cudaMemcpy(buf_input_tokens_batch_.i32(), warm_tok, 2 * sizeof(int32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(buf_input_pos_batch_.i32(), warm_pos, 2 * sizeof(int32_t), cudaMemcpyHostToDevice));
            forward_token_batch_nemotron_device_body(0, warm_M, false);
            CUDA_CHECK(cudaStreamSynchronize(main_stream_));
            reset_all_kv_caches();
            CUDA_CHECK(cudaStreamSynchronize(main_stream_));
        }

        auto t_start = std::chrono::high_resolution_clock::now();
        LOG_INFO("Nemotron Batched Prefill: %d prompt tokens (start_offset=%d, chunk_size=%d)...",
                 total_tokens, start_offset, chunk_size);

        expert_loader_.set_demotion_enabled(false);

        for (int start = start_offset; start < total_tokens; start += chunk_size) {
            int M = std::min(chunk_size, total_tokens - start);
            int position = start;
            bool is_last_chunk = (start + M == total_tokens);

            for (int m = 0; m < M; m++) {
                h_prefill_tok_[m] = prompt[start + m];
                h_prefill_pos_[m] = position + m;
            }

            CUDA_CHECK(cudaMemcpyAsync(buf_input_pos_batch_.i32(), h_prefill_pos_,
                                       M * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
            CUDA_CHECK(cudaMemcpyAsync(buf_input_tokens_batch_.i32(), h_prefill_tok_,
                                       M * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));

            bool need_logits = is_last_chunk && compute_last_logits;
            forward_token_batch_nemotron_device_body(position, M, need_logits);
        }

        expert_loader_.set_demotion_enabled(true);

        CUDA_CHECK(cudaStreamSynchronize(main_stream_));
        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        double tok_per_sec = (remaining_tokens / (ms / 1000.0));
        LOG_INFO("Nemotron Batched Prefill completed: %d tokens in %.2f ms (%.1f tok/s)",
                 remaining_tokens, ms, tok_per_sec);
    }
