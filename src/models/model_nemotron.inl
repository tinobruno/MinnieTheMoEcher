    // ── Forward one layer (Nemotron 3.5 Lightning Hybrid Mamba-2 SSM + MoE) ─
    inline static cudaEvent_t s_moe_sub_ev[6] = {nullptr};
    inline static bool s_moe_sub_init = false;
    inline static cudaEvent_t s_dram_sub_ev[6] = {nullptr};
    inline static bool s_dram_sub_init = false;

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
            int in_proj_dim = cfg_.is_nemotron_bolt() ? 18560 : 10304;

            rms_norm_cuda(buf_hidden2_.bf16(), buf_hidden_.bf16(),
                          lw.attn_norm_w.bf16(), dim, cfg_.rms_norm_eps, main_stream_);

            // In-projection: NVFP4 or FP8 [in_proj_dim, dim] with scalar scale -> buf_q_ [in_proj_dim]
            if (lw.w_in_qkv.dtype == "nvfp4") {
                gemv_nvfp4_cuda(buf_q_.bf16(), buf_hidden2_.bf16(),
                                (const uint8_t*)lw.w_in_qkv.data, (const uint8_t*)lw.w_in_qkv_scale.data, lw.w_in_qkv_scale_val,
                                in_proj_dim, dim, main_stream_);
            } else {
                gemv_fp8_scalar_scale_cuda(buf_q_.bf16(), buf_hidden2_.bf16(),
                                           (const uint8_t*)lw.w_in_qkv.data, lw.w_in_qkv_scale_val,
                                           in_proj_dim, dim, main_stream_);
            }

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

            // Out-projection: NVFP4 or FP8 [dim, mamba_inter] with scalar scale + residual accumulation -> buf_hidden_ [dim]
            if (lw.linear_out_proj.dtype == "nvfp4") {
                gemv_nvfp4_residual_cuda(buf_hidden_.bf16(), buf_attn_out_.bf16(),
                                         (const uint8_t*)lw.linear_out_proj.data, (const uint8_t*)lw.linear_out_proj_scale.data, lw.linear_out_proj_scale_val,
                                         dim, mamba_inter, main_stream_);
            } else {
                gemv_fp8_scalar_scale_residual_cuda(
                    buf_hidden_.bf16(), buf_attn_out_.bf16(),
                    (const uint8_t*)lw.linear_out_proj.data, lw.linear_out_proj_scale_val,
                    dim, mamba_inter, main_stream_);
            }
            return;
        } else if (ltype == 0) {
            // ════════════════════════════════════════════════════════════════════════
            // 2. STANDARD GQA ATTENTION MIXER LAYER (Nemotron 3.5 Lightning / Bolt)
            // ════════════════════════════════════════════════════════════════════════
            rms_norm_cuda(buf_hidden2_.bf16(), buf_hidden_.bf16(),
                          lw.attn_norm_w.bf16(), dim, cfg_.rms_norm_eps, main_stream_);

            int q_dim = n_q_heads * head_dim; // 32 * 128 = 4096
            int kv_dim = n_kv_heads * head_dim;

            auto gemv_gqa_proj = [&](__nv_bfloat16* out, const GPUTensor& w, const __nv_bfloat16* in, int out_dim, int in_dim) {
                if (w.dtype == "q8_0" || w.dtype == "Q8_0" || w.dtype == "U8") {
                    gemv_q8_0_cuda(out, in, (const block_q8_0*)w.data, out_dim, in_dim, main_stream_);
                } else {
                    gemv_bf16_out_bf16_cuda(out, w.bf16(), in, out_dim, in_dim, main_stream_);
                }
            };

            gemv_gqa_proj(buf_q_.bf16(), lw.w_q, buf_hidden2_.bf16(), q_dim, dim);
            gemv_gqa_proj(buf_gate_.bf16(), lw.w_k, buf_hidden2_.bf16(), kv_dim, dim);
            gemv_gqa_proj(buf_up_.bf16(), lw.w_v, buf_hidden2_.bf16(), kv_dim, dim);

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

            // Attention Output Projection: Q8_0/U8 or BF16 [dim, q_dim] + residual accumulation into buf_hidden_
            gemv_gqa_proj(buf_hidden2_.bf16(), lw.w_o, buf_attn_out_.bf16(), dim, q_dim);
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

            // 3a. Shared Expert (NVFP4 or IQ2_XXS, Squared ReLU, residual) for non-Bolt models
            if (!cfg_.is_nemotron_bolt() && lw.shared_w1_w.data && lw.shared_w2_w.data) {
                int shared_inter = cfg_.moe_shared_expert_intermediate_size > 0 ? cfg_.moe_shared_expert_intermediate_size : 3712;
                if (lw.shared_w1_w.dtype == "iq2_xxs" || lw.shared_w1_w.dtype == "iq2") {
                    gemv_iq2_xxs_cuda(buf_gate_.bf16(), buf_hidden2_.bf16(),
                                      (const block_iq2_xxs*)lw.shared_w1_w.data, shared_inter, dim, main_stream_);
                    relu2_cuda(buf_gate_.bf16(), buf_gate_.bf16(), shared_inter, main_stream_);
                    gemv_iq2_xxs_residual_cuda(buf_hidden_.bf16(), buf_gate_.bf16(),
                                               (const block_iq2_xxs*)lw.shared_w2_w.data, dim, shared_inter, main_stream_);
                } else {
                    gemv_nvfp4_cuda(buf_gate_.bf16(), buf_hidden2_.bf16(),
                                    (const uint8_t*)lw.shared_w1_w.data, (const uint8_t*)lw.shared_w1_s.data,
                                    lw.shared_w1_s2, shared_inter, dim, main_stream_);
                    relu2_cuda(buf_gate_.bf16(), buf_gate_.bf16(), shared_inter, main_stream_);
                    gemv_nvfp4_residual_cuda(buf_hidden_.bf16(), buf_gate_.bf16(),
                                             (const uint8_t*)lw.shared_w2_w.data, (const uint8_t*)lw.shared_w2_s.data,
                                             lw.shared_w2_s2, dim, shared_inter, main_stream_);
                }
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

                    // Track expert activation frequency (GPU-native, 100% CUDA Graph compatible)
                    int stride = cfg_.n_routed_experts;
                    if (d_expert_counts_.data) {
                        accumulate_expert_freq_cuda(
                            (uint32_t*)d_expert_counts_.data,
                            (int32_t*)d_step_topk_.data,
                            buf_topk_idx_.i32(),
                            buf_track_flag_.i32(),
                            layer_id,
                            stride,
                            n_experts,
                            top_k,
                            main_stream_);
                    }

                    // Early D2H dispatch: launch immediately so it runs concurrently with GPU compute
                    CUDA_CHECK(cudaMemcpyAsync(topk_ids_host_, buf_topk_idx_.i32(),
                                               top_k * sizeof(int32_t), cudaMemcpyDeviceToHost, main_stream_));
                    CUDA_CHECK(cudaMemcpyAsync(topk_weights_host_, buf_topk_vals_.f32(),
                                               top_k * sizeof(float), cudaMemcpyDeviceToHost, main_stream_));
                    CUDA_CHECK(cudaEventRecord(dma_event_, main_stream_));

                    // Latent Down-Projection: buf_hidden2_ [4096] -> buf_latent_down_ [1024]
                    if (lw.fc1_latent_proj.dtype == "fp8") {
                        gemv_fp8_scalar_scale_cuda(buf_latent_down_.bf16(), buf_hidden2_.bf16(),
                                                   (const uint8_t*)lw.fc1_latent_proj.data, lw.fc1_scale_val,
                                                   latent_dim, dim, main_stream_);
                    } else if (lw.fc1_latent_proj.dtype == "q8_0" || lw.fc1_latent_proj.dtype == "Q8_0" || lw.fc1_latent_proj.dtype == "U8") {
                        gemv_q8_0_cuda(buf_latent_down_.bf16(), buf_hidden2_.bf16(),
                                       (const block_q8_0*)lw.fc1_latent_proj.data,
                                       latent_dim, dim, main_stream_);
                    } else {
                        gemv_bf16_out_bf16_cuda(buf_latent_down_.bf16(), lw.fc1_latent_proj.bf16(),
                                                buf_hidden2_.bf16(), latent_dim, dim, main_stream_);
                    }

                    // Shared Expert for Bolt (5376 dim, runs on GPU concurrently while D2H travels to CPU)
                    if (lw.shared_w1_w.data && lw.shared_w2_w.data) {
                        int shared_inter = cfg_.moe_shared_expert_intermediate_size > 0 ? cfg_.moe_shared_expert_intermediate_size : 5376;
                        if (lw.shared_w1_w.dtype == "iq2_xxs" || lw.shared_w1_w.dtype == "iq2") {
                            gemv_iq2_xxs_relu2_cuda(buf_gate_.bf16(), buf_hidden2_.bf16(),
                                                    (const block_iq2_xxs*)lw.shared_w1_w.data, shared_inter, dim, main_stream_);
                            gemv_iq2_xxs_residual_cuda(buf_hidden_.bf16(), buf_gate_.bf16(),
                                                       (const block_iq2_xxs*)lw.shared_w2_w.data, dim, shared_inter, main_stream_);
                        } else {
                            gemv_nvfp4_cuda(buf_gate_.bf16(), buf_hidden2_.bf16(),
                                            (const uint8_t*)lw.shared_w1_w.data, (const uint8_t*)lw.shared_w1_s.data,
                                            lw.shared_w1_s2, shared_inter, dim, main_stream_);
                            relu2_cuda(buf_gate_.bf16(), buf_gate_.bf16(), shared_inter, main_stream_);
                            gemv_nvfp4_residual_cuda(buf_hidden_.bf16(), buf_gate_.bf16(),
                                                     (const uint8_t*)lw.shared_w2_w.data, (const uint8_t*)lw.shared_w2_s.data,
                                                     lw.shared_w2_s2, dim, shared_inter, main_stream_);
                        }
                    }

                    // Synchronize D2H copy (which already finished during Latent Down + Shared Expert!)
                    CUDA_CHECK(cudaEventSynchronize(dma_event_));

                    // Resolve pointers / DMA for all active top-k experts with static VRAM L1 residency
                    void* exp_ptrs[16] = {nullptr};
                    bool needs_dma[16] = {false};
                    void* dram_srcs[16] = {nullptr};
                    int stream_buf = moe_layer_idx % 2;

                    int n_miss = 0;
                    for (int k = 0; k < top_k; k++) {
                        int eid = topk_ids_host_[k];
                        if (eid < 0 || eid >= n_experts || topk_weights_host_[k] == 0.0f) continue;
                        int s_idx = expert_loader_.try_get_expert_cached_slot(moe_layer_idx, eid, exp_ptrs[k]);
                        if (s_idx >= 0 && exp_ptrs[k] != nullptr) {
                            needs_dma[k] = false;
                        } else {
                            exp_ptrs[k] = streaming_slots_gpu_[stream_buf][k];
                            needs_dma[k] = true;
                            void* p = expert_loader_.get_flat_dram_ptr(moe_layer_idx, eid);
                            if (!p) p = expert_loader_.try_get_dram_expert_ptr(moe_layer_idx, eid);
                            dram_srcs[k] = p;
                            if (!p) n_miss++;
                        }
                    }

                    if (n_miss > 0) {
                        for (int k = 0; k < top_k; k++) {
                            if (needs_dma[k] && !dram_srcs[k]) {
                                int eid = topk_ids_host_[k];
                                dram_srcs[k] = expert_loader_.get_dram_expert_ptr(moe_layer_idx, eid);
                            }
                        }
                    }

                    // Asynchronously queue PCIe transfers on dma_stream_ for any uncached experts
                    bool dma_launched[16] = {false};
                    for (int k = 0; k < top_k; k++) {
                        if (needs_dma[k] && exp_ptrs[k] && dram_srcs[k]) {
                            CUDA_CHECK(cudaMemcpyAsync(exp_ptrs[k], dram_srcs[k], expert_loader_.expert_block_size(),
                                                       cudaMemcpyHostToDevice, dma_stream_));
                            CUDA_CHECK(cudaEventRecord(expert_events_[k], dma_stream_));
                            dma_launched[k] = true;
                        }
                    }

                    size_t max_moe_inter = cfg_.moe_intermediate_size > 0 ? (size_t)cfg_.moe_intermediate_size : 2560;
                    size_t up_w_off = 0;
                    size_t up_s_off = max_moe_inter * (latent_dim / 2);
                    size_t up_s2_off = up_s_off + max_moe_inter * (latent_dim / 16);
                    size_t down_w_off = up_s2_off + 16;
                    size_t down_s_off = down_w_off + (size_t)latent_dim * (max_moe_inter / 2);

                    if (expert_parts_.count("up_proj.weight")) {
                        up_w_off = expert_parts_["up_proj.weight"].offset_in_block;
                    }
                    if (expert_parts_.count("up_proj.weight_scale")) {
                        up_s_off = expert_parts_["up_proj.weight_scale"].offset_in_block;
                    }
                    if (expert_parts_.count("down_proj.weight")) {
                        down_w_off = expert_parts_["down_proj.weight"].offset_in_block;
                    }
                    if (expert_parts_.count("down_proj.weight_scale")) {
                        down_s_off = expert_parts_["down_proj.weight_scale"].offset_in_block;
                    }

                    auto compute_expert_k = [&](int k) {
                        int eid = topk_ids_host_[k];
                        float weight = topk_weights_host_[k];
                        if (eid < 0 || eid >= n_experts || weight == 0.0f) return;

                        void* exp_ptr = exp_ptrs[k];
                        if (!exp_ptr) return;

                        // GPU main stream waits for expert k DMA to complete before executing kernels
                        if (dma_launched[k]) {
                            CUDA_CHECK(cudaStreamWaitEvent(main_stream_, expert_events_[k], 0));
                        } else if (needs_dma[k]) {
                            return; // Unresolved DMA source, skip safely
                        }

                        const uint8_t* up_w = (const uint8_t*)exp_ptr + up_w_off;
                        const uint8_t* up_s = (const uint8_t*)exp_ptr + up_s_off;
                        const uint8_t* down_w = (const uint8_t*)exp_ptr + down_w_off;
                        const uint8_t* down_s = (const uint8_t*)exp_ptr + down_s_off;

                        float up_s2 = 1.0f;
                        float down_s2 = 1.0f;
                        int stride_exp = std::max({cfg_.n_routed_experts, cfg_.max_n_routed_experts(), 128});
                        size_t s_idx = ((size_t)moe_layer_idx * stride_exp + eid) * 2;
                        if (s_idx + 1 < nemotron_expert_scales_.size()) {
                            up_s2 = nemotron_expert_scales_[s_idx];
                            down_s2 = nemotron_expert_scales_[s_idx + 1];
                        }

                        if (cfg_.expert_dtype == "iq2_xxs" || cfg_.expert_dtype == "iq2") {
                            // 1. Fused Up-projection + Squared ReLU on latent input vector: [moe_inter, latent_dim]
                            gemv_iq2_xxs_relu2_cuda(buf_gate_.bf16(), buf_latent_down_.bf16(),
                                                    (const block_iq2_xxs*)up_w, moe_inter, latent_dim, main_stream_);

                            // 2. Fused Down-projection + Weight Scale + Residual Accumulation directly into buf_latent_accum_: [latent_dim, moe_inter]
                            gemv_iq2_xxs_scaled_residual_cuda(buf_latent_accum_.bf16(), buf_gate_.bf16(),
                                                              (const block_iq2_xxs*)down_w, weight,
                                                              latent_dim, moe_inter, main_stream_);
                        } else {
                            // 1. Fused Up-projection + Squared ReLU on latent input vector: [moe_inter, latent_dim]
                            gemv_nvfp4_relu2_cuda(buf_gate_.bf16(), buf_latent_down_.bf16(),
                                                  up_w, up_s, up_s2, moe_inter, latent_dim, main_stream_);

                            // 2. Fused Down-projection + Weight Scale + Residual Accumulation directly into buf_latent_accum_: [latent_dim, moe_inter]
                            gemv_nvfp4_scaled_residual_cuda(buf_latent_accum_.bf16(), buf_gate_.bf16(),
                                                            down_w, down_s, down_s2, weight,
                                                            latent_dim, moe_inter, main_stream_);
                        }
                    };

                    CUDA_CHECK(cudaMemsetAsync(buf_latent_accum_.data, 0, (size_t)latent_dim * sizeof(__nv_bfloat16), main_stream_));

                    // Pass 1: Compute all resident VRAM L1 experts first with zero DMA wait.
                    // This keeps sm_120 Tensor Cores busy while PCIe transfers uncached L2 experts concurrently!
                    for (int k = 0; k < top_k; k++) {
                        if (!needs_dma[k]) {
                            compute_expert_k(k);
                        }
                    }

                    // Pass 2: Compute uncached experts (waits for their individual DMA events)
                    for (int k = 0; k < top_k; k++) {
                        if (needs_dma[k]) {
                            compute_expert_k(k);
                        }
                    }

                    // 5. Latent Up-Projection: buf_latent_accum_ [1024] -> residual addition into buf_hidden_ [4096]
                    if (lw.fc2_latent_proj.dtype == "fp8") {
                        gemv_fp8_scalar_scale_residual_cuda(buf_hidden_.bf16(), buf_latent_accum_.bf16(),
                                                            (const uint8_t*)lw.fc2_latent_proj.data, lw.fc2_scale_val,
                                                            dim, latent_dim, main_stream_);
                    } else if (lw.fc2_latent_proj.dtype == "q8_0" || lw.fc2_latent_proj.dtype == "Q8_0" || lw.fc2_latent_proj.dtype == "U8") {
                        gemv_q8_0_residual_cuda(buf_hidden_.bf16(), buf_latent_accum_.bf16(),
                                                (const block_q8_0*)lw.fc2_latent_proj.data,
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
                void* exp_ptrs[16] = {nullptr};
                size_t exp_block_sz = expert_loader_.expert_block_size() > 0 ? (size_t)expert_loader_.expert_block_size() : (size_t)5615616;
                for (int k = 0; k < top_k; k++) {
                    int eid = topk_ids_host_[k];
                    if (eid < 0 || eid >= n_experts || topk_weights_host_[k] == 0.0f) continue;
                    exp_ptrs[k] = expert_loader_.try_get_expert_cached(moe_layer_idx, eid);
                    if (!exp_ptrs[k]) {
                        void* dram_src = expert_loader_.get_flat_dram_ptr(moe_layer_idx, eid);
                        if (!dram_src) dram_src = expert_loader_.get_dram_expert_ptr(moe_layer_idx, eid);
                        void* scratch_gpu = (char*)buf_dram_moe_scratch_.data + (size_t)k * exp_block_sz;
                        void* src_to_dma = dram_src;
                        if (!expert_loader_.dram_is_pinned() && pinned_host_moe_scratch_) {
                            void* pinned_dst = (char*)pinned_host_moe_scratch_ + (size_t)k * exp_block_sz;
                            if (dram_src && pinned_dst) {
                                memcpy(pinned_dst, dram_src, exp_block_sz);
                                src_to_dma = pinned_dst;
                            }
                        }
                        if (src_to_dma && scratch_gpu) {
                            CUDA_CHECK(cudaMemcpyAsync(scratch_gpu, src_to_dma, exp_block_sz,
                                                       cudaMemcpyHostToDevice, main_stream_));
                            exp_ptrs[k] = scratch_gpu;
                        }
                    }
                }

                size_t up_w_sz = (size_t)moe_inter * (dim / 2);
                size_t up_s_sz = (size_t)moe_inter * (dim / 16);
                size_t down_w_sz = (size_t)dim * (moe_inter / 2);

                for (int k = 0; k < top_k; k++) {
                    int eid = topk_ids_host_[k];
                    float weight = topk_weights_host_[k];
                    if (eid < 0 || eid >= n_experts || weight == 0.0f) continue;

                    void* exp_ptr = exp_ptrs[k];
                    if (!exp_ptr) continue;

                    const uint8_t* up_w = (const uint8_t*)exp_ptr;
                    const uint8_t* up_s = (const uint8_t*)exp_ptr + up_w_sz;
                    const uint8_t* down_w = (const uint8_t*)exp_ptr + up_w_sz + up_s_sz + 16;
                    const uint8_t* down_s = down_w + down_w_sz;

                    float up_s2 = 1.0f;
                    float down_s2 = 1.0f;
                    int stride_exp = std::max({cfg_.n_routed_experts, cfg_.max_n_routed_experts(), 128});
                    size_t s_idx = ((size_t)moe_layer_idx * stride_exp + eid) * 2;
                    if (s_idx + 1 < nemotron_expert_scales_.size()) {
                        up_s2 = nemotron_expert_scales_[s_idx];
                        down_s2 = nemotron_expert_scales_[s_idx + 1];
                    }

                    // 1. Fused Up proj + Squared ReLU: [moe_inter, dim]
                    gemv_nvfp4_relu2_cuda(buf_gate_.bf16(), buf_hidden2_.bf16(),
                                          up_w, up_s, up_s2, moe_inter, dim, main_stream_);

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
            int in_proj_dim = cfg_.is_nemotron_bolt() ? 18560 : 10304;

            rms_norm_cuda_batched(buf_hidden2_batch_.bf16(), buf_hidden_batch_.bf16(),
                                  lw.attn_norm_w.bf16(), M, dim, cfg_.rms_norm_eps, main_stream_);

            // In-projection: Batched NVFP4 or FP8 [M, in_proj_dim] with scalar scale -> buf_q_batch_
            if (lw.w_in_qkv.dtype == "nvfp4") {
                gemm_nvfp4_batch_cuda(buf_q_batch_.bf16(), buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_in_qkv.data, (const uint8_t*)lw.w_in_qkv_scale.data, lw.w_in_qkv_scale_val,
                                      M, in_proj_dim, dim, main_stream_);
            } else {
                gemm_fp8_scalar_scale_cublaslt(buf_q_batch_.bf16(), buf_hidden2_batch_.bf16(),
                                               (const uint8_t*)lw.w_in_qkv.data, lw.w_in_qkv_scale_val,
                                               M, in_proj_dim, dim, 0.0f, main_stream_);
            }

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
                mamba_head_dim,
                (M > 1 && M <= 4) ? lw.conv_state_slots[0].bf16() : nullptr,
                (M > 1 && M <= 4) ? lw.ssm_state_slots[0].f32() : nullptr,
                (M > 2 && M <= 4) ? lw.conv_state_slots[1].bf16() : nullptr,
                (M > 2 && M <= 4) ? lw.ssm_state_slots[1].f32() : nullptr,
                (M > 3 && M <= 4) ? lw.conv_state_slots[2].bf16() : nullptr,
                (M > 3 && M <= 4) ? lw.ssm_state_slots[2].f32() : nullptr);

            // Out-projection: Batched NVFP4 or FP8 [M, dim] with scalar scale + residual accumulation -> buf_hidden_batch_
            if (lw.linear_out_proj.dtype == "nvfp4") {
                gemm_nvfp4_residual_batch_cuda(
                    buf_hidden_batch_.bf16(), buf_attn_out_batch_.bf16(),
                    (const uint8_t*)lw.linear_out_proj.data, (const uint8_t*)lw.linear_out_proj_scale.data, lw.linear_out_proj_scale_val,
                    M, dim, mamba_inter, main_stream_);
            } else {
                gemm_fp8_scalar_scale_cublaslt(
                    buf_hidden_batch_.bf16(), buf_attn_out_batch_.bf16(),
                    (const uint8_t*)lw.linear_out_proj.data, lw.linear_out_proj_scale_val,
                    M, dim, mamba_inter, 1.0f, main_stream_);
            }
            return;
        } else if (ltype == 0) {
            // ════════════════════════════════════════════════════════════════════════
            // 2. STANDARD GQA ATTENTION MIXER LAYER (Nemotron 3.5 Lightning / Bolt)
            // ════════════════════════════════════════════════════════════════════════
            rms_norm_cuda_batched(buf_hidden2_batch_.bf16(), buf_hidden_batch_.bf16(),
                                  lw.attn_norm_w.bf16(), M, dim, cfg_.rms_norm_eps, main_stream_);

            int q_dim = n_q_heads * head_dim; // 32 * 128 = 4096
            int kv_dim = n_kv_heads * head_dim; // 2 * 128 = 256

            auto gemm_gqa_proj_batch = [&](__nv_bfloat16* out, const GPUTensor& w, const __nv_bfloat16* in, int out_dim, int in_dim) {
                if (w.dtype == "q8_0" || w.dtype == "Q8_0" || w.dtype == "U8") {
                    for (int m = 0; m < M; m++) {
                        gemv_q8_0_cuda(out + (size_t)m * out_dim,
                                       in + (size_t)m * in_dim,
                                       (const block_q8_0*)w.data,
                                       out_dim, in_dim, main_stream_);
                    }
                } else {
                    gemm_bf16(out, M, out_dim, in_dim, in, w.bf16());
                }
            };

            // Batched Q, K, V Projections
            gemm_gqa_proj_batch(buf_q_batch_.bf16(), lw.w_q, buf_hidden2_batch_.bf16(), q_dim, dim);
            gemm_gqa_proj_batch(buf_gate_batch_.bf16(), lw.w_k, buf_hidden2_batch_.bf16(), kv_dim, dim);
            gemm_gqa_proj_batch(buf_up_batch_.bf16(), lw.w_v, buf_hidden2_batch_.bf16(), kv_dim, dim);

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

            // Attention Output Projection: Q8_0/U8 or BF16 [M, dim] + residual accumulation into buf_hidden_batch_
            gemm_gqa_proj_batch(buf_hidden2_batch_.bf16(), lw.w_o, buf_attn_out_batch_.bf16(), dim, q_dim);
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
                for (int i = 0; i < 6; i++) CUDA_CHECK(cudaEventCreate(&s_dram_sub_ev[i]));
                s_dram_sub_init = true;
            }
            if (layer_id == 1) CUDA_CHECK(cudaEventRecord(s_moe_sub_ev[0], main_stream_));
            if (moe_layer_idx == 10) CUDA_CHECK(cudaEventRecord(s_dram_sub_ev[0], main_stream_));

            // Shared Expert for non-Bolt Nemotron: Run synchronously on main_stream_ with direct residual accumulation
            bool has_shared = (lw.shared_w1_w.data && lw.shared_w2_w.data);
            if (has_shared && !cfg_.is_nemotron_bolt()) {
                int shared_inter = cfg_.moe_shared_expert_intermediate_size > 0 ? cfg_.moe_shared_expert_intermediate_size : 3712;
                if (lw.shared_w1_w.dtype == "iq2_xxs" || lw.shared_w1_w.dtype == "iq2") {
                    gemm_iq2_xxs_relu2_batch_cuda(buf_gate_batch_.bf16(), buf_hidden2_batch_.bf16(),
                                                  (const block_iq2_xxs*)lw.shared_w1_w.data,
                                                  M, shared_inter, dim, main_stream_);
                    gemm_iq2_xxs_residual_batch_cuda(buf_hidden_batch_.bf16(), buf_gate_batch_.bf16(),
                                                     (const block_iq2_xxs*)lw.shared_w2_w.data,
                                                     M, dim, shared_inter, main_stream_);
                } else {
                    gemm_nvfp4_relu2_batch_cuda(buf_gate_batch_.bf16(), buf_hidden2_batch_.bf16(),
                                                (const uint8_t*)lw.shared_w1_w.data, (const uint8_t*)lw.shared_w1_s.data,
                                                lw.shared_w1_s2, M, shared_inter, dim, main_stream_);
                    gemm_nvfp4_residual_batch_cuda(buf_hidden_batch_.bf16(), buf_gate_batch_.bf16(),
                                                   (const uint8_t*)lw.shared_w2_w.data, (const uint8_t*)lw.shared_w2_s.data,
                                                   lw.shared_w2_s2, M, dim, shared_inter, main_stream_);
                }
            }

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

                    // Bin tokens into expert bins for accelerated batched execution
                    moe_bin_tokens_cuda(buf_moe_offsets_.i32(), buf_moe_token_indices_.i32(), buf_moe_token_slots_.i32(),
                                        buf_topk_idx_batch_.i32(), M * top_k, top_k, n_experts, main_stream_);

                    // Early D2H dispatch: Copy topk_ids to host concurrently while Latent Down-Projection computes on GPU
                    if (!layer_fully_resident_[moe_layer_idx]) {
                        CUDA_CHECK(cudaMemcpyAsync(topk_ids_batch_host_, buf_topk_idx_batch_.i32(),
                                                   (size_t)M * top_k * sizeof(int32_t), cudaMemcpyDeviceToHost, main_stream_));
                        CUDA_CHECK(cudaEventRecord(dma_event_, main_stream_));
                    }

                    int stride = cfg_.n_routed_experts;
                    if (d_expert_counts_.data && track_expert_freq_) {
                        accumulate_expert_freq_batch_cuda(
                            (uint32_t*)d_expert_counts_.data,
                            buf_topk_idx_batch_.i32(),
                            M,
                            layer_id,
                            stride,
                            n_experts,
                            top_k,
                            main_stream_);
                    }

                    // Latent Down-Projection: buf_hidden2_batch_ [M, 4096] -> buf_latent_down_batch_ [M, 1024]
                    if (lw.fc1_latent_proj.dtype == "fp8") {
                        gemm_fp8_scalar_scale_cublaslt(buf_latent_down_batch_.bf16(), buf_hidden2_batch_.bf16(),
                                                       (const uint8_t*)lw.fc1_latent_proj.data, lw.fc1_scale_val,
                                                       M, latent_dim, dim, 0.0f, main_stream_);
                    } else if (lw.fc1_latent_proj.dtype == "q8_0" || lw.fc1_latent_proj.dtype == "Q8_0" || lw.fc1_latent_proj.dtype == "U8") {
                        if (M == 1) {
                            gemv_q8_0_cuda(buf_latent_down_batch_.bf16(),
                                           buf_hidden2_batch_.bf16(),
                                           (const block_q8_0*)lw.fc1_latent_proj.data,
                                           latent_dim, dim, main_stream_);
                        } else {
                            gemm_q8_0_batch_cuda(buf_latent_down_batch_.bf16(),
                                                 buf_hidden2_batch_.bf16(),
                                                 (const block_q8_0*)lw.fc1_latent_proj.data,
                                                 M, latent_dim, dim, main_stream_);
                        }
                    } else {
                        gemm_bf16(buf_latent_down_batch_.bf16(), M, latent_dim, dim,
                                  buf_hidden2_batch_.bf16(), lw.fc1_latent_proj.bf16());
                    }

                    // Zero latent accumulation buffer: [M, latent_dim]
                    CUDA_CHECK(cudaMemsetAsync(buf_latent_accum_batch_.data, 0,
                                               (size_t)M * latent_dim * sizeof(__nv_bfloat16), main_stream_));

                    // 1. Batched Up-Projection + Squared ReLU
                    const void* const* layer_ptrs = nullptr;
                    size_t down_w_off = expert_parts_.count("down_proj.weight") ? expert_parts_["down_proj.weight"].offset_in_block : (size_t)675840;
                    size_t exp_block_sz = expert_loader_.expert_block_size() > 0 ? expert_loader_.expert_block_size() : 1351680;

                    if (layer_fully_resident_[moe_layer_idx]) {
                        layer_ptrs = (const void* const*)buf_all_expert_ptrs_.data + (size_t)moe_layer_idx * stride_exp;
                    } else {
                        // Wait for early D2H copy dispatched concurrently during routing/latent-down
                        CUDA_CHECK(cudaEventSynchronize(dma_event_));

                        static std::vector<void*> s_layer_ptrs;
                        if ((int)s_layer_ptrs.size() < n_experts) {
                            s_layer_ptrs.resize(n_experts, nullptr);
                        }
                        std::fill(s_layer_ptrs.begin(), s_layer_ptrs.end(), nullptr);

                        static std::vector<int> s_active_uncached;
                        s_active_uncached.clear();
                        static std::vector<bool> s_expert_seen;
                        if ((int)s_expert_seen.size() < n_experts) s_expert_seen.resize(n_experts, false);
                        std::fill(s_expert_seen.begin(), s_expert_seen.end(), false);

                        for (int i = 0; i < M * top_k; i++) {
                            int eid = topk_ids_batch_host_[i];
                            if (eid < 0 || eid >= n_experts || s_expert_seen[eid]) continue;
                            s_expert_seen[eid] = true;
                            void* p = expert_loader_.try_get_expert_cached(moe_layer_idx, eid);
                            if (p) {
                                s_layer_ptrs[eid] = p;
                            } else {
                                s_active_uncached.push_back(eid);
                            }
                        }

                        int n_uncached = (int)s_active_uncached.size();
                        int max_scratch_slots = 256;

                        // 1. Parallel CPU copy to pinned host scratch buffer using expert_pool_
                        int copy_count = std::min(n_uncached, max_scratch_slots);
                        if (pinned_host_moe_scratch_ && copy_count > 0) {
                            if (expert_pool_ && copy_count > 16) {
                                int n_threads = std::min((int)expert_pool_->size(), 16);
                                int chunk = (copy_count + n_threads - 1) / n_threads;
                                std::vector<std::future<void>> futs;
                                for (int t = 0; t < n_threads; t++) {
                                    int s = t * chunk;
                                    int e = std::min(s + chunk, copy_count);
                                    if (s >= e) continue;
                                    futs.push_back(expert_pool_->enqueue([this, moe_layer_idx, s, e, exp_block_sz]() {
                                        for (int u = s; u < e; u++) {
                                            int eid = s_active_uncached[u];
                                            void* dram_src = expert_loader_.get_flat_dram_ptr(moe_layer_idx, eid);
                                            if (!dram_src) dram_src = expert_loader_.get_dram_expert_ptr(moe_layer_idx, eid);
                                            void* pinned_dst = (char*)pinned_host_moe_scratch_ + (size_t)u * exp_block_sz;
                                            if (dram_src && pinned_dst) {
                                                memcpy(pinned_dst, dram_src, exp_block_sz);
                                            }
                                        }
                                    }));
                                }
                                for (auto& f : futs) f.wait();
                                for (int u = 0; u < copy_count; u++) {
                                    int eid = s_active_uncached[u];
                                    s_layer_ptrs[eid] = (char*)buf_dram_moe_scratch_.data + (size_t)u * exp_block_sz;
                                }
                            } else {
                                for (int u = 0; u < copy_count; u++) {
                                    int eid = s_active_uncached[u];
                                    void* dram_src = expert_loader_.get_flat_dram_ptr(moe_layer_idx, eid);
                                    if (!dram_src) dram_src = expert_loader_.get_dram_expert_ptr(moe_layer_idx, eid);
                                    void* pinned_dst = (char*)pinned_host_moe_scratch_ + (size_t)u * exp_block_sz;
                                    void* scratch_gpu = (char*)buf_dram_moe_scratch_.data + (size_t)u * exp_block_sz;
                                    if (dram_src && pinned_dst) {
                                        memcpy(pinned_dst, dram_src, exp_block_sz);
                                    }
                                    s_layer_ptrs[eid] = scratch_gpu;
                                }
                            }

                            // 2. High-speed contiguous PCIe DMA burst (14 GB/s in ~14 ms)
                            size_t dma_bytes = (size_t)copy_count * exp_block_sz;
                            CUDA_CHECK(cudaMemcpyAsync(buf_dram_moe_scratch_.data, pinned_host_moe_scratch_,
                                                       dma_bytes, cudaMemcpyHostToDevice, dma_stream_));
                            CUDA_CHECK(cudaEventRecord(dma_event_, dma_stream_));
                        }

                        CUDA_CHECK(cudaMemcpyAsync(buf_layer_expert_ptrs_.data, s_layer_ptrs.data(),
                                                   (size_t)n_experts * sizeof(void*), cudaMemcpyHostToDevice, main_stream_));
                        layer_ptrs = (const void* const*)buf_layer_expert_ptrs_.data;
                    }

                    if (layer_id == 1) CUDA_CHECK(cudaEventRecord(s_moe_sub_ev[1], main_stream_));
                    if (moe_layer_idx == 10) CUDA_CHECK(cudaEventRecord(s_dram_sub_ev[1], main_stream_));

                    // Shared Expert: Runs concurrently on main_stream_ while DMA transfers active uncached experts on dma_stream_
                    bool has_shared = (lw.shared_w1_w.data && lw.shared_w2_w.data);
                    if (has_shared) {
                        int shared_inter = cfg_.moe_shared_expert_intermediate_size > 0 ? cfg_.moe_shared_expert_intermediate_size : 5376;
                        if (lw.shared_w1_w.dtype == "iq2_xxs" || lw.shared_w1_w.dtype == "iq2") {
                            gemm_iq2_xxs_relu2_batch_cuda(buf_gate_batch_.bf16(), buf_hidden2_batch_.bf16(),
                                                          (const block_iq2_xxs*)lw.shared_w1_w.data,
                                                          M, shared_inter, dim, main_stream_);
                            gemm_iq2_xxs_residual_batch_cuda(buf_hidden_batch_.bf16(), buf_gate_batch_.bf16(),
                                                             (const block_iq2_xxs*)lw.shared_w2_w.data,
                                                             M, dim, shared_inter, main_stream_);
                        }
                    }

                    if (!layer_fully_resident_[moe_layer_idx]) {
                        CUDA_CHECK(cudaStreamWaitEvent(main_stream_, dma_event_, 0));
                    }
                    if (layer_id == 1) CUDA_CHECK(cudaEventRecord(s_moe_sub_ev[2], main_stream_));
                    if (moe_layer_idx == 10) CUDA_CHECK(cudaEventRecord(s_dram_sub_ev[2], main_stream_));

                    if (cfg_.expert_dtype == "iq2_xxs" || cfg_.expert_dtype == "iq2") {
                        moe_iq2_xxs_binned_up_relu2_batch_cuda(
                            buf_moe_gate_batch_.bf16(),
                            buf_latent_down_batch_.bf16(),
                            buf_moe_offsets_.i32(),
                            buf_moe_token_indices_.i32(),
                            buf_moe_token_slots_.i32(),
                            layer_ptrs,
                            layer_scales,
                            M, moe_inter, latent_dim, main_stream_, top_k, n_experts);
                        if (layer_id == 1) CUDA_CHECK(cudaEventRecord(s_moe_sub_ev[3], main_stream_));
                        if (moe_layer_idx == 10) CUDA_CHECK(cudaEventRecord(s_dram_sub_ev[3], main_stream_));

                        // 2. Batched Down-Projection + Top-k Scaled Residual Accumulation into buf_latent_accum_batch_
                        moe_iq2_xxs_binned_down_scaled_residual_batch_cuda(
                            buf_latent_accum_batch_.bf16(),
                            buf_moe_gate_batch_.bf16(),
                            buf_topk_vals_batch_.f32(),
                            buf_moe_offsets_.i32(),
                            buf_moe_token_indices_.i32(),
                            buf_moe_token_slots_.i32(),
                            layer_ptrs,
                            layer_scales,
                            M, latent_dim, moe_inter, main_stream_, top_k, n_experts,
                            down_w_off);
                        if (layer_id == 1) CUDA_CHECK(cudaEventRecord(s_moe_sub_ev[4], main_stream_));
                        if (moe_layer_idx == 10) CUDA_CHECK(cudaEventRecord(s_dram_sub_ev[4], main_stream_));
                    } else {
                        moe_nemotron_up_relu2_batch_cuda(
                            buf_moe_gate_batch_.bf16(),
                            buf_latent_down_batch_.bf16(),
                            buf_topk_idx_batch_.i32(),
                            buf_moe_offsets_.i32(),
                            buf_moe_token_indices_.i32(),
                            buf_moe_token_slots_.i32(),
                            layer_ptrs,
                            layer_scales,
                            M, moe_inter, latent_dim, main_stream_, top_k, n_experts);
                        if (layer_id == 1) CUDA_CHECK(cudaEventRecord(s_moe_sub_ev[3], main_stream_));

                        moe_nemotron_down_scaled_residual_batch_cuda(
                            buf_latent_accum_batch_.bf16(),
                            buf_moe_gate_batch_.bf16(),
                            buf_topk_idx_batch_.i32(),
                            buf_topk_vals_batch_.f32(),
                            buf_moe_offsets_.i32(),
                            buf_moe_token_indices_.i32(),
                            buf_moe_token_slots_.i32(),
                            layer_ptrs,
                            layer_scales,
                            M, latent_dim, moe_inter, main_stream_, top_k, n_experts,
                            /*down_offset_padding=*/16);
                        if (layer_id == 1) CUDA_CHECK(cudaEventRecord(s_moe_sub_ev[4], main_stream_));
                    }

                    // 5. Latent Up-Projection: buf_latent_accum_batch_ [M, 1024] -> residual add into buf_hidden_batch_ [M, 4096]
                    if (lw.fc2_latent_proj.dtype == "fp8") {
                        gemm_fp8_scalar_scale_cublaslt(buf_hidden_batch_.bf16(), buf_latent_accum_batch_.bf16(),
                                                       (const uint8_t*)lw.fc2_latent_proj.data, lw.fc2_scale_val,
                                                       M, dim, latent_dim, 1.0f, main_stream_);
                    } else if (lw.fc2_latent_proj.dtype == "q8_0" || lw.fc2_latent_proj.dtype == "Q8_0" || lw.fc2_latent_proj.dtype == "U8") {
                        if (M == 1) {
                            gemv_q8_0_residual_cuda(buf_hidden_batch_.bf16(),
                                                    buf_latent_accum_batch_.bf16(),
                                                    (const block_q8_0*)lw.fc2_latent_proj.data,
                                                    dim, latent_dim, main_stream_);
                        } else {
                            gemm_q8_0_residual_batch_cuda(buf_hidden_batch_.bf16(),
                                                          buf_latent_accum_batch_.bf16(),
                                                          (const block_q8_0*)lw.fc2_latent_proj.data,
                                                          M, dim, latent_dim, main_stream_);
                        }
                    } else {
                        gemm_bf16(buf_hidden2_batch_.bf16(), M, dim, latent_dim,
                                  buf_latent_accum_batch_.bf16(), lw.fc2_latent_proj.bf16());
                        vector_add_bf16_cuda(buf_hidden_batch_.bf16(), buf_hidden2_batch_.bf16(), M * dim, main_stream_);
                    }
                    if (layer_id == 1) CUDA_CHECK(cudaEventRecord(s_moe_sub_ev[5], main_stream_));
                    if (moe_layer_idx == 10) CUDA_CHECK(cudaEventRecord(s_dram_sub_ev[5], main_stream_));
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

                if (!layer_fully_resident_[moe_layer_idx]) {
                    CUDA_CHECK(cudaMemcpyAsync(topk_ids_batch_host_, buf_topk_idx_batch_.i32(),
                                               (size_t)M * top_k * sizeof(int32_t), cudaMemcpyDeviceToHost, main_stream_));
                    CUDA_CHECK(cudaEventRecord(dma_event_, main_stream_));
                }

                // Bin tokens into expert bins for accelerated batched execution
                moe_bin_tokens_cuda(buf_moe_offsets_.i32(), buf_moe_token_indices_.i32(), buf_moe_token_slots_.i32(),
                                    buf_topk_idx_batch_.i32(), M * top_k, top_k, n_experts, main_stream_);
            }

            if (layer_id == 1) CUDA_CHECK(cudaEventRecord(s_moe_sub_ev[1], main_stream_));

            // Routed Experts (Top-6, Squared ReLU, scaled residual)
            if (cfg_.n_routed_experts > 0 && lw.gate_w.data) {
                const void* const* layer_ptrs = nullptr;
                size_t exp_block_sz = expert_loader_.expert_block_size() > 0 ? (size_t)expert_loader_.expert_block_size() : (size_t)5615616;

                if (layer_fully_resident_[moe_layer_idx]) {
                    layer_ptrs = (const void* const*)buf_all_expert_ptrs_.data + (size_t)moe_layer_idx * stride_exp;
                } else {
                    CUDA_CHECK(cudaEventSynchronize(dma_event_));

                    static std::vector<void*> s_lightning_layer_ptrs;
                    if ((int)s_lightning_layer_ptrs.size() < n_experts) {
                        s_lightning_layer_ptrs.resize(n_experts, nullptr);
                    }
                    std::fill(s_lightning_layer_ptrs.begin(), s_lightning_layer_ptrs.end(), nullptr);

                    static std::vector<int> s_lightning_active_uncached;
                    s_lightning_active_uncached.clear();
                    static std::vector<bool> s_lightning_expert_seen;
                    if ((int)s_lightning_expert_seen.size() < n_experts) s_lightning_expert_seen.resize(n_experts, false);
                    std::fill(s_lightning_expert_seen.begin(), s_lightning_expert_seen.end(), false);

                    for (int i = 0; i < M * top_k; i++) {
                        int eid = topk_ids_batch_host_[i];
                        if (eid < 0 || eid >= n_experts || s_lightning_expert_seen[eid]) continue;
                        s_lightning_expert_seen[eid] = true;
                        void* p = expert_loader_.try_get_expert_cached(moe_layer_idx, eid);
                        if (p) {
                            s_lightning_layer_ptrs[eid] = p;
                        } else {
                            s_lightning_active_uncached.push_back(eid);
                        }
                    }

                    int n_uncached = (int)s_lightning_active_uncached.size();
                    int max_scratch_slots = 128;
                    int copy_count = std::min(n_uncached, max_scratch_slots);

                    if (pinned_host_moe_scratch_ && copy_count > 0) {
                        if (expert_pool_ && copy_count > 8) {
                            int n_threads = std::min((int)expert_pool_->size(), 16);
                            int chunk = (copy_count + n_threads - 1) / n_threads;
                            std::vector<std::future<void>> futs;
                            for (int t = 0; t < n_threads; t++) {
                                int s = t * chunk;
                                int e = std::min(s + chunk, copy_count);
                                if (s >= e) continue;
                                futs.push_back(expert_pool_->enqueue([this, moe_layer_idx, s, e, exp_block_sz]() {
                                    for (int u = s; u < e; u++) {
                                        int eid = s_lightning_active_uncached[u];
                                        void* dram_src = expert_loader_.get_flat_dram_ptr(moe_layer_idx, eid);
                                        if (!dram_src) dram_src = expert_loader_.get_dram_expert_ptr(moe_layer_idx, eid);
                                        void* pinned_dst = (char*)pinned_host_moe_scratch_ + (size_t)u * exp_block_sz;
                                        if (dram_src && pinned_dst) {
                                            memcpy(pinned_dst, dram_src, exp_block_sz);
                                        }
                                    }
                                }));
                            }
                            for (auto& f : futs) f.wait();
                            for (int u = 0; u < copy_count; u++) {
                                int eid = s_lightning_active_uncached[u];
                                s_lightning_layer_ptrs[eid] = (char*)buf_dram_moe_scratch_.data + (size_t)u * exp_block_sz;
                            }
                        } else {
                            for (int u = 0; u < copy_count; u++) {
                                int eid = s_lightning_active_uncached[u];
                                void* dram_src = expert_loader_.get_flat_dram_ptr(moe_layer_idx, eid);
                                if (!dram_src) dram_src = expert_loader_.get_dram_expert_ptr(moe_layer_idx, eid);
                                void* pinned_dst = (char*)pinned_host_moe_scratch_ + (size_t)u * exp_block_sz;
                                void* scratch_gpu = (char*)buf_dram_moe_scratch_.data + (size_t)u * exp_block_sz;
                                if (dram_src && pinned_dst) {
                                    memcpy(pinned_dst, dram_src, exp_block_sz);
                                }
                                s_lightning_layer_ptrs[eid] = scratch_gpu;
                            }
                        }

                        // High-speed contiguous PCIe DMA burst on dma_stream_
                        size_t dma_bytes = (size_t)copy_count * exp_block_sz;
                        CUDA_CHECK(cudaMemcpyAsync(buf_dram_moe_scratch_.data, pinned_host_moe_scratch_,
                                                   dma_bytes, cudaMemcpyHostToDevice, dma_stream_));
                        CUDA_CHECK(cudaEventRecord(dma_event_, dma_stream_));
                    }

                    CUDA_CHECK(cudaMemcpyAsync(buf_layer_expert_ptrs_.data, s_lightning_layer_ptrs.data(),
                                               (size_t)n_experts * sizeof(void*), cudaMemcpyHostToDevice, main_stream_));
                    layer_ptrs = (const void* const*)buf_layer_expert_ptrs_.data;

                    if (copy_count > 0) {
                        CUDA_CHECK(cudaStreamWaitEvent(main_stream_, dma_event_, 0));
                    }
                }

                if (layer_id == 1) CUDA_CHECK(cudaEventRecord(s_moe_sub_ev[2], main_stream_));

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
                    M, dim, moe_inter, main_stream_, top_k, n_experts,
                    /*down_offset_padding=*/16);

                if (layer_id == 1) CUDA_CHECK(cudaEventRecord(s_moe_sub_ev[4], main_stream_));
            }
            if (layer_id == 1) CUDA_CHECK(cudaEventRecord(s_moe_sub_ev[5], main_stream_));
            return;
        }
    }

    void forward_token_batch_nemotron_device_body(int position, int M, bool do_compute_logits = false, bool compute_all_logits = false) {
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

        // 2. Process all layers with profiling (only for prefill M > 4)
        bool is_prefill = (M > 4);
        if (is_prefill) expert_loader_.set_in_prefill(true);
        if (M > 4) {
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
                     ms_ssm, ms_gqa, n_vram_moe, ms_vram_moe, n_vram_moe > 0 ? (ms_vram_moe / n_vram_moe) : 0.0f,
                     n_dram_moe, ms_dram_moe, n_dram_moe > 0 ? (ms_dram_moe / n_dram_moe) : 0.0f);
        } else {
            for (int layer = 0; layer < cfg_.num_hidden_layers; layer++) {
                forward_layer_nemotron_batch(layer, position, M);
            }
        }

        if (s_moe_sub_init) {
            float t_p0 = 0, t_p1 = 0, t_p2 = 0, t_p3 = 0, t_p4 = 0;
            cudaEventElapsedTime(&t_p0, s_moe_sub_ev[0], s_moe_sub_ev[1]);
            cudaEventElapsedTime(&t_p1, s_moe_sub_ev[1], s_moe_sub_ev[2]);
            cudaEventElapsedTime(&t_p2, s_moe_sub_ev[2], s_moe_sub_ev[3]);
            cudaEventElapsedTime(&t_p3, s_moe_sub_ev[3], s_moe_sub_ev[4]);
            cudaEventElapsedTime(&t_p4, s_moe_sub_ev[4], s_moe_sub_ev[5]);
            if (cfg_.is_nemotron_bolt()) {
                LOG_INFO("  [BOLT MOE SUB-STEP Layer 1 (VRAM) M=%d] Router+LatentDown: %.2f ms | Shared+DMA: %.2f ms | UpProj: %.2f ms | DownProj: %.2f ms | LatentUp: %.2f ms | Sum: %.2f ms",
                         M, t_p0, t_p1, t_p2, t_p3, t_p4, t_p0 + t_p1 + t_p2 + t_p3 + t_p4);
                if (s_dram_sub_init) {
                    float d_p0 = 0, d_p1 = 0, d_p2 = 0, d_p3 = 0, d_p4 = 0;
                    cudaEventElapsedTime(&d_p0, s_dram_sub_ev[0], s_dram_sub_ev[1]);
                    cudaEventElapsedTime(&d_p1, s_dram_sub_ev[1], s_dram_sub_ev[2]);
                    cudaEventElapsedTime(&d_p2, s_dram_sub_ev[2], s_dram_sub_ev[3]);
                    cudaEventElapsedTime(&d_p3, s_dram_sub_ev[3], s_dram_sub_ev[4]);
                    cudaEventElapsedTime(&d_p4, s_dram_sub_ev[4], s_dram_sub_ev[5]);
                    LOG_INFO("  [BOLT MOE SUB-STEP Layer 10 (DRAM) M=%d] Discover+Copy: %.2f ms | Shared+DMA: %.2f ms | UpProj: %.2f ms | DownProj: %.2f ms | LatentUp: %.2f ms | Sum: %.2f ms",
                             M, d_p0, d_p1, d_p2, d_p3, d_p4, d_p0 + d_p1 + d_p2 + d_p3 + d_p4);
                }
            } else {
                LOG_INFO("  [MOE SUB-STEP Layer 1 M=%d] Router+Bin: %.2f ms | SharedExpert: %.2f ms | Up: %.2f ms | Down: %.2f ms | LayerEnd: %.2f ms | Sum: %.2f ms",
                         M, t_p0, t_p1, t_p2, t_p3, t_p4, t_p0 + t_p1 + t_p2 + t_p3 + t_p4);
            }
        }

        // Always copy the last token's hidden state into buf_hidden_ (for generation / continuation)
        CUDA_CHECK(cudaMemcpyAsync(buf_hidden_.bf16(),
                                   buf_hidden_batch_.bf16() + (size_t)(M - 1) * dim,
                                   dim * sizeof(__nv_bfloat16),
                                   cudaMemcpyDeviceToDevice, main_stream_));

        if (compute_all_logits) {
            rms_norm_cuda_batched(buf_hidden_batch_.bf16(), buf_hidden_batch_.bf16(),
                                  norm_weight_.bf16(), M, dim, cfg_.rms_norm_eps, main_stream_);
            if (head_weight_.dtype == "nvfp4") {
                for (int m = 0; m < M; m++) {
                    gemv_nvfp4_f32_cuda(buf_logits_batch_.f32() + (size_t)m * cfg_.vocab_size,
                                        buf_hidden_batch_.bf16() + (size_t)m * dim,
                                        (const uint8_t*)head_weight_.data,
                                        (const uint8_t*)head_weight_scale_.data,
                                        head_weight_scale_2_,
                                        cfg_.vocab_size, dim, main_stream_);
                }
            } else if (head_weight_.dtype == "fp4") {
                quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                               buf_hidden_batch_.bf16(), M, dim, main_stream_);
                gemm_fp4_f32_blackwell_tensorcore_cuda(buf_logits_batch_.f32(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                       (const uint8_t*)head_weight_.data, (const uint8_t*)head_weight_scale_.data,
                                                       cfg_.vocab_size, dim, M, main_stream_);
            } else if (head_weight_.dtype == "q8_0" || head_weight_.dtype == "Q8_0" || head_weight_.dtype == "U8") {
                for (int m = 0; m < M; m++) {
                    gemv_q8_0_f32_cuda(buf_logits_batch_.f32() + (size_t)m * cfg_.vocab_size,
                                       buf_hidden_batch_.bf16() + (size_t)m * dim,
                                       (const block_q8_0*)head_weight_.data,
                                       cfg_.vocab_size, dim, main_stream_);
                }
            } else {
                gemm_bf16_f32(buf_logits_batch_.f32(), M, cfg_.vocab_size, dim,
                              buf_hidden_batch_.bf16(), head_weight_.bf16());
            }
        } else if (do_compute_logits) {
            // 3. Final RMSNorm on the last token (in buf_hidden_)
            rms_norm_cuda(buf_hidden_.bf16(), buf_hidden_.bf16(),
                          norm_weight_.bf16(), dim, cfg_.rms_norm_eps, main_stream_);

            // 4. Compute logits for the last token
            compute_logits();
        }
        if (is_prefill) expert_loader_.set_in_prefill(false);
    }

    void forward_token_batch_nemotron(const int* tokens_host, int position, int M) {
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

        forward_token_batch_nemotron_device_body(position, M, /*do_compute_logits=*/false, /*compute_all_logits=*/true);
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

        expert_loader_.set_in_prefill(true);
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

        if (track_expert_freq_) {
            expert_freq_total_tokens_ += remaining_tokens;
        }

        expert_loader_.set_demotion_enabled(true);
        expert_loader_.set_in_prefill(false);

        CUDA_CHECK(cudaStreamSynchronize(main_stream_));
        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        double tok_per_sec = (remaining_tokens / (ms / 1000.0));
        LOG_INFO("Nemotron Batched Prefill completed: %d tokens in %.2f ms (%.1f tok/s, cold disk reads: %llu)",
                 remaining_tokens, ms, tok_per_sec, (unsigned long long)expert_loader_.cold_disk_reads_.load());
    }
