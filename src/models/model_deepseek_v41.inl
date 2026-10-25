// ════════════════════════════════════════════════════════════════════════════════
//  model_deepseek_v41.inl — DeepSeek-V4.1-Flash Native C++/CUDA Engine
//  Architecture: DeepseekV41ForCausalLM (552B total, 14B active / token)
//  Hidden Dim: 5120 | Layers: 40 | Routed Experts: 384 | Top-K: 6 | Shared: 1
//  Hyper-Connections: hc_mult = 4 (Sinkhorn doubly stochastic pre/post/comb mixing)
//  Hierarchical 3-Level Cache: L1 VRAM (96GB) <-> L2 DRAM <-> L3 NVMe SSD (Direct I/O)
// ════════════════════════════════════════════════════════════════════════════════

    // ── Forward Layer DeepSeek V4.1 ──────────────────────────────────────────

    void forward_layer_deepseek_v41(int layer_id, int token_id, int position) {
        auto& lw = layers_[layer_id];
        int dim = cfg_.hidden_size > 0 ? cfg_.hidden_size : DEEPSEEK_V41_HIDDEN_DIM;
        int hc = cfg_.hc_mult;
        int mix_size = (2 + hc) * hc;
        int hc_dim = hc * dim;

        // ── 1. Attention Hyper-Connection Mixes from buf_hc_state_ ───────────
        // mixes = x_norm @ hc_attn_fn.T -> [mix_size]
        gemv_hc_pre_norm_cuda(buf_hc_mixes_.f32(), buf_hc_state_.bf16(), lw.hc_attn_fn.f32(),
                              mix_size, hc_dim, cfg_.hc_eps, main_stream_);

        // Split mixes into attn_pre, attn_post, attn_comb via Sinkhorn
        hc_split_sinkhorn_cuda(
            buf_hc_pre_.f32(), buf_hc_post_.f32(), buf_hc_comb_.f32(),
            buf_hc_mixes_.f32(), lw.hc_attn_scale.f32(), lw.hc_attn_base.f32(),
            hc, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, main_stream_);

        // ── 2. Collapse HC copies using pre_mix & Attention Pre-Norm ─────────
        // In layer 0, pre_mix is identity [1,0,0,0]. In layer > 0, pre_mix is ffn_pre from prev layer.
        hc_pre_weighted_add_norm_cuda(buf_hidden_.bf16(), buf_hc_state_.bf16(),
                                      buf_hc_pre_mix_.f32(), lw.attn_norm_w.bf16(),
                                      dim, hc, cfg_.rms_norm_eps, main_stream_);
        CUDA_CHECK(cudaEventRecord(main_event_, main_stream_));

        // ── 3. Single-Pass Mega-mHC Latent Attention (MLA / CED) ─────────────
        // Input: buf_hidden_ (normalized). Output written back into buf_hidden_.
        forward_attention_deepseek_v41(layer_id, position);

        // ── 4. Expand attention output back to HC residual stream ────────────
        // buf_hc_after_attn_ = post * attn_out + comb @ buf_hc_state_
        hc_post_update_cuda(buf_hc_after_attn_.bf16(), buf_hidden_.bf16(), buf_hc_state_.bf16(),
                            buf_hc_post_.f32(), buf_hc_comb_.f32(),
                            dim, hc, main_stream_);

        // ── 5. Collapse HC copies using attn_pre & FFN Pre-Norm ──────────────
        // FFN uses attn_pre produced by step 1 in buf_hc_pre_
        hc_pre_weighted_add_norm_cuda(buf_hidden_.bf16(), buf_hc_after_attn_.bf16(),
                                      buf_hc_pre_.f32(), lw.ffn_norm_w.bf16(),
                                      dim, hc, cfg_.rms_norm_eps, main_stream_);
        CUDA_CHECK(cudaEventRecord(main_event_, main_stream_));

        // ── 6. FFN Hyper-Connection Mixes from buf_hc_after_attn_ ────────────
        gemv_hc_pre_norm_cuda(buf_hc_mixes_.f32(), buf_hc_after_attn_.bf16(), lw.hc_ffn_fn.f32(),
                              mix_size, hc_dim, cfg_.hc_eps, main_stream_);

        // Split into ffn_pre (saved into buf_hc_pre_mix_ for next layer), ffn_post, ffn_comb
        hc_split_sinkhorn_cuda(
            buf_hc_pre_mix_.f32(), buf_hc_post_.f32(), buf_hc_comb_.f32(),
            buf_hc_mixes_.f32(), lw.hc_ffn_scale.f32(), lw.hc_ffn_base.f32(),
            hc, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, main_stream_);

        // ── 7. MoE FFN (384 Experts, Top-6 Routing + 1 Shared Expert) ────────
        // Input: buf_hidden_ (normalized). Output accumulated back into buf_hidden_.
        forward_moe_deepseek_v41(layer_id, token_id);

        // ── 8. Expand FFN output back to HC residual stream ──────────────────
        // buf_hc_state_ = post * ffn_out + comb @ buf_hc_after_attn_
        hc_post_update_cuda(buf_hc_state_.bf16(), buf_hidden_.bf16(), buf_hc_after_attn_.bf16(),
                            buf_hc_post_.f32(), buf_hc_comb_.f32(),
                            dim, hc, main_stream_);
    }

    // ── Mega-mHC Latent Attention Forward ─────────────────────────────────────

    void forward_attention_deepseek_v41(int layer_id, int position) {
        auto& lw = layers_[layer_id];
        int dim = cfg_.hidden_size > 0 ? cfg_.hidden_size : DEEPSEEK_V41_HIDDEN_DIM;
        int n_heads = cfg_.num_attention_heads;
        int head_dim_val = cfg_.head_dim;
        int q_lora = cfg_.q_lora_rank > 0 ? cfg_.q_lora_rank : 1280;
        int rope_dim = cfg_.qk_rope_head_dim > 0 ? cfg_.qk_rope_head_dim : 64;
        int window = cfg_.sliding_window > 0 ? cfg_.sliding_window : (cfg_.window_size > 0 ? cfg_.window_size : 4096);

        // RoPE frequencies
        float* layer_rope_freqs = rope_freqs_.f32();

        // ── KV Compression Projection on side_stream_ (CED dual-mode) ────────
        CUDA_CHECK(cudaStreamWaitEvent(side_stream_, main_event_, 0));

        if (lw.wkv_w.dtype == "fp4") {
            gemv_fp4_cuda(buf_kv_.bf16(), buf_hidden_.bf16(),
                          (const uint8_t*)lw.wkv_w.data, (const uint8_t*)lw.wkv_s.data,
                          head_dim_val, dim, side_stream_);
        } else if (lw.wkv_w.dtype == "int4") {
            gemv_int4_cuda(buf_kv_.bf16(), buf_hidden_.bf16(),
                           (const uint8_t*)lw.wkv_w.data, lw.wkv_s.bf16(),
                           head_dim_val, dim, side_stream_);
        } else {
            gemm_fp8_dequant(buf_kv_.bf16(), 1, head_dim_val, dim,
                             buf_hidden_.bf16(),
                             lw.wkv_w.u8(), lw.wkv_s.u8(), 32, side_stream_);
        }

        rms_norm_cuda(buf_kv_.bf16(), buf_kv_.bf16(),
                      lw.kv_norm_w.bf16(), head_dim_val, cfg_.rms_norm_eps, side_stream_);

        rope_device_pos_cuda(buf_kv_.bf16(), 1, head_dim_val, rope_dim,
                             buf_input_pos_.i32(), layer_rope_freqs, false, side_stream_);

        store_kv_device_pos_cuda(lw.kv_cache.bf16(), buf_kv_.bf16(),
                                 buf_input_pos_.i32(), window, head_dim_val, side_stream_);
        CUDA_CHECK(cudaEventRecord(side_event_, side_stream_));

        // ── Query Projection on main_stream_ ─────────────────────────────────
        // 1. q_lora = wq_a(x) -> [q_lora_rank]
        if (lw.wq_a_w.dtype == "fp4") {
            gemv_fp4_cuda(buf_lora_.bf16(), buf_hidden_.bf16(),
                          (const uint8_t*)lw.wq_a_w.data, (const uint8_t*)lw.wq_a_s.data,
                          q_lora, dim, main_stream_);
        } else if (lw.wq_a_w.dtype == "int4") {
            gemv_int4_cuda(buf_lora_.bf16(), buf_hidden_.bf16(),
                           (const uint8_t*)lw.wq_a_w.data, lw.wq_a_s.bf16(),
                           q_lora, dim, main_stream_);
        } else {
            gemm_fp8_dequant(buf_lora_.bf16(), 1, q_lora, dim,
                             buf_hidden_.bf16(),
                             lw.wq_a_w.u8(), lw.wq_a_s.u8(), 32, main_stream_);
        }

        // 2. q_normed = q_norm(q_lora)
        rms_norm_cuda(buf_lora_.bf16(), buf_lora_.bf16(),
                      lw.q_norm_w.bf16(), q_lora, cfg_.rms_norm_eps, main_stream_);

        // 3. q = wq_b(q_normed) -> [n_heads * head_dim]
        int q_total_dim = n_heads * head_dim_val;
        if (lw.wq_b_w.dtype == "fp4") {
            gemv_fp4_cuda(buf_q_.bf16(), buf_lora_.bf16(),
                          (const uint8_t*)lw.wq_b_w.data, (const uint8_t*)lw.wq_b_s.data,
                          q_total_dim, q_lora, main_stream_);
        } else if (lw.wq_b_w.dtype == "int4") {
            gemv_int4_cuda(buf_q_.bf16(), buf_lora_.bf16(),
                           (const uint8_t*)lw.wq_b_w.data, lw.wq_b_s.bf16(),
                           q_total_dim, q_lora, main_stream_);
        } else {
            gemm_fp8_dequant(buf_q_.bf16(), 1, q_total_dim, q_lora,
                             buf_lora_.bf16(),
                             lw.wq_b_w.u8(), lw.wq_b_s.u8(), 32, main_stream_);
        }

        // Wait for KV projection completion on side_stream_
        CUDA_CHECK(cudaStreamWaitEvent(main_stream_, side_event_, 0));

        // ── Single-Pass Mega-mHC Multi-Head Context Attention ────────────────
        float qk_scale = 1.0f / sqrtf((float)head_dim_val);
        int max_combined = window + cfg_.max_compressed_entries;
        if (max_combined < 1) max_combined = 128;
        mla_attention_fused_cuda(
            buf_q_.bf16(), lw.kv_cache.bf16(), lw.comp_kv_cache.bf16(), lw.attn_sink.f32(),
            buf_attn_out_.bf16(), buf_input_pos_.i32(), lw.d_comp_kv_count.i32(),
            layer_rope_freqs, max_combined, head_dim_val, rope_dim, qk_scale,
            0.0f, nullptr, window, main_stream_
        );

        // ── Grouped Output Projection wo_a + wo_b ─────────────────────────────
        int o_groups = cfg_.o_groups > 0 ? cfg_.o_groups : 8;
        int o_lora = cfg_.o_lora_rank > 0 ? cfg_.o_lora_rank : 1024;
        int heads_per_group = n_heads / o_groups;
        int hpg_dim = heads_per_group * head_dim_val;

        if (lw.wo_a_w.dtype == "int4") {
            for (int g = 0; g < o_groups; g++) {
                const uint8_t* w_g = (const uint8_t*)lw.wo_a_w.data + (size_t)g * o_lora * (hpg_dim / 2);
                const __nv_bfloat16* s_g = lw.wo_a_s.bf16() + (size_t)g * o_lora * (hpg_dim / 32);
                const __nv_bfloat16* a_g = buf_attn_out_.bf16() + (size_t)g * hpg_dim;
                __nv_bfloat16* out_g = buf_lora_.bf16() + (size_t)g * o_lora;
                gemv_int4_cuda(out_g, a_g, w_g, s_g, o_lora, hpg_dim, main_stream_);
            }
        } else {
            gemv_fp8_grouped_cuda(buf_lora_.bf16(), buf_attn_out_.bf16(),
                                  lw.wo_a_w.u8(), lw.wo_a_s.u8(),
                                  o_lora, hpg_dim, o_groups, 32, main_stream_);
        }

        // wo_b: [dim, o_groups * o_lora] -> written directly into buf_hidden_
        if (lw.wo_b_w.dtype == "int4") {
            gemv_int4_cuda(buf_hidden_.bf16(), buf_lora_.bf16(), (const uint8_t*)lw.wo_b_w.data, lw.wo_b_s.bf16(), dim, o_groups * o_lora, main_stream_);
        } else {
            gemm_fp8_dequant(buf_hidden_.bf16(), 1, dim, o_groups * o_lora,
                             buf_lora_.bf16(),
                             lw.wo_b_w.u8(), lw.wo_b_s.u8(), 32, main_stream_);
        }
    }

    // ── DeepSeek-V4.1 MoE Forward with 3-Level Caching ────────────────────────

    void forward_moe_deepseek_v41(int layer_id, int token_id) {
        auto& lw = layers_[layer_id];
        int dim = cfg_.hidden_size > 0 ? cfg_.hidden_size : DEEPSEEK_V41_HIDDEN_DIM;
        int n_experts = cfg_.n_routed_experts > 0 ? cfg_.n_routed_experts : DEEPSEEK_V41_EXPERTS;
        int top_k = cfg_.num_experts_per_tok > 0 ? cfg_.num_experts_per_tok : DEEPSEEK_V41_ACTIVE_EXPERTS;
        int moe_inter = cfg_.moe_intermediate_size > 0 ? cfg_.moe_intermediate_size : DEEPSEEK_V41_MOE_INTERMEDIATE;
        int shared_inter = (cfg_.moe_shared_expert_intermediate_size > 0 && cfg_.moe_shared_expert_intermediate_size != 3712)
                           ? cfg_.moe_shared_expert_intermediate_size : moe_inter;

        // ── 1. Router Gating: W_gate @ hidden -> [384] logits ─────────────
        gemm_bf16(buf_scores_bf16_.bf16(), 1, n_experts, dim,
                  buf_hidden_.bf16(), lw.gate_w.bf16());

        // ── 2. Top-6 Gating with SqrtSoftplus / Softmax & Temperature ─────────
        deepseek_v41_gating_top6_cuda(
            buf_topk_idx_.i32(),
            buf_topk_vals_.f32(),
            buf_scores_bf16_.bf16(),
            lw.gate_bias.f32(),
            n_experts,
            top_k,
            cfg_.routed_scaling_factor,
            1.0f,
            main_stream_);

        // ── 3. Asynchronously Copy Top-6 Expert IDs to Host ──────────────────
        CUDA_CHECK(cudaMemcpyAsync(topk_ids_host_, buf_topk_idx_.i32(),
                                   top_k * sizeof(int32_t), cudaMemcpyDeviceToHost, main_stream_));
        CUDA_CHECK(cudaEventRecord(dma_event_, main_stream_));

        // ── 4. Side-Stream Compute Overlap: 1 Shared Expert ───────────────────
        CUDA_CHECK(cudaStreamWaitEvent(side_stream_, main_event_, 0));
        __nv_bfloat16* shared_gate = buf_gate_.bf16() + top_k * moe_inter;
        __nv_bfloat16* shared_up   = buf_up_.bf16()   + top_k * moe_inter;
        __nv_bfloat16* shared_down = buf_down_.bf16() + top_k * dim;

        if (lw.shared_w1_w.dtype == "int4") {
            gemv_int4_cuda(shared_gate, buf_hidden_.bf16(), (const uint8_t*)lw.shared_w1_w.data, lw.shared_w1_s.bf16(), shared_inter, dim, side_stream_);
            gemv_int4_cuda(shared_up, buf_hidden_.bf16(), (const uint8_t*)lw.shared_w3_w.data, lw.shared_w3_s.bf16(), shared_inter, dim, side_stream_);
            silu_mul_cuda(shared_gate, shared_gate, shared_up, shared_inter, cfg_.swiglu_limit, side_stream_);
            gemv_int4_cuda(shared_down, shared_gate, (const uint8_t*)lw.shared_w2_w.data, lw.shared_w2_s.bf16(), dim, shared_inter, side_stream_);
        } else {
            gemm_fp8_dequant(shared_gate, 1, shared_inter, dim,
                             buf_hidden_.bf16(),
                             lw.shared_w1_w.u8(), lw.shared_w1_s.u8(), 32, side_stream_);
            gemm_fp8_dequant(shared_up, 1, shared_inter, dim,
                             buf_hidden_.bf16(),
                             lw.shared_w3_w.u8(), lw.shared_w3_s.u8(), 32, side_stream_);
            silu_mul_cuda(shared_gate, shared_gate, shared_up, shared_inter, cfg_.swiglu_limit, side_stream_);
            gemm_fp8_dequant(shared_down, 1, dim, shared_inter,
                             shared_gate,
                             lw.shared_w2_w.u8(), lw.shared_w2_s.u8(), 32, side_stream_);
        }
        CUDA_CHECK(cudaEventRecord(side_event_, side_stream_));

        // ── 5. Resolve Top-6 Routed Experts across 3-Tier Cache Hierarchy ──────
        CUDA_CHECK(cudaEventSynchronize(dma_event_));

        void* exp_ptrs[6] = {nullptr};
        bool is_cold_expert[6] = {false};
        size_t exp_block_sz = expert_loader_.expert_block_size() > 0 ? (size_t)expert_loader_.expert_block_size() : (size_t)18800640;

        for (int k = 0; k < top_k; k++) {
            int eid = topk_ids_host_[k];
            if (eid < 0 || eid >= n_experts) continue;

            if (expert_loader_.is_mixed()) {
                if (expert_loader_.is_hot(layer_id, eid)) {
                    // Hot Expert: 100% VRAM Resident (Zero-latency hit)
                    exp_ptrs[k] = expert_loader_.get_hot_vram_ptr(layer_id, eid);
                    is_cold_expert[k] = false;
                    continue;
                } else {
                    is_cold_expert[k] = true;
                    // Cold Expert: Check L2 Host DRAM Cache
                    void* dram_src = expert_loader_.get_flat_dram_ptr(layer_id, eid);
                    if (!dram_src) dram_src = expert_loader_.try_get_dram_expert_ptr(layer_id, eid);

                    // Tier 3: Fetch Cold Expert from NVMe SSD (Direct I/O)
                    if (!dram_src) {
                        int64_t key = (int64_t)layer_id * n_experts + eid;
                        dram_src = expert_loader_.fetch_cold_expert_from_disk(layer_id, eid, key);
                    }

                    // DMA Transfer from DRAM -> VRAM Streaming Slot (only 9.95 MB!)
                    void* vram_slot = streaming_slots_gpu_[0][k];
                    if (dram_src && vram_slot) {
                        CUDA_CHECK(cudaMemcpyAsync(vram_slot, dram_src, expert_loader_.cold_block_size(), cudaMemcpyHostToDevice, main_stream_));
                        exp_ptrs[k] = vram_slot;
                    }
                }
            } else {
                // Tier 1: Check L1 VRAM Cache (Zero-latency hit)
                void* p_l1 = expert_loader_.try_get_expert_cached(layer_id, eid);
                if (p_l1) {
                    exp_ptrs[k] = p_l1;
                    continue;
                }

                // Tier 2: Check L2 Host DRAM Cache (Microsecond PCIe DMA)
                void* dram_src = expert_loader_.get_flat_dram_ptr(layer_id, eid);
                if (!dram_src) dram_src = expert_loader_.try_get_dram_expert_ptr(layer_id, eid);

                // Tier 3: Fetch Cold Expert from NVMe SSD (Direct I/O)
                if (!dram_src) {
                    int64_t key = (int64_t)layer_id * n_experts + eid;
                    dram_src = expert_loader_.fetch_cold_expert_from_disk(layer_id, eid, key);
                }

                // DMA Transfer from DRAM -> VRAM Streaming Slot
                void* vram_slot = streaming_slots_gpu_[0][k];
                if (dram_src && vram_slot) {
                    CUDA_CHECK(cudaMemcpyAsync(vram_slot, dram_src, exp_block_sz, cudaMemcpyHostToDevice, main_stream_));
                    exp_ptrs[k] = vram_slot;
                }
            }
        }

        // ── 6. Compute 6 Routed Experts with Native SwiGLU ─────────────────────
        for (int k = 0; k < top_k; k++) {
            if (exp_ptrs[k]) {
                execute_expert_swiglu(exp_ptrs[k], 1.0f, k, buf_hidden_.bf16(), is_cold_expert[k]);
            }
        }

        // ── 7. Join Side Stream (Shared Expert) + Fused 6-way Dynamic Accumulation
        CUDA_CHECK(cudaStreamWaitEvent(main_stream_, side_event_, 0));
        __nv_bfloat16* shared_down_ptr = buf_down_.bf16() + top_k * dim;
        fused_moe_accum_dynamic_cuda(buf_hidden_.bf16(), buf_down_.bf16(),
                                     buf_topk_vals_.f32(), shared_down_ptr, dim, main_stream_);
    }

    // ── Batched Prefill for DeepSeek-V4.1 ─────────────────────────────────────

    void forward_layer_deepseek_v41_batch(int layer_id, int position, int M) {
        for (int m = 0; m < M; m++) {
            forward_layer_deepseek_v41(layer_id, 0, position + m);
        }
    }
