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
        int window = cfg_.window_size > 0 ? cfg_.window_size : (cfg_.sliding_window > 0 ? cfg_.sliding_window : 128);

        // Select per-layer RoPE frequencies based on compress_ratio (matches DeepSeek V4.1 reference architecture)
        bool is_compressed = (layer_id < (int)cfg_.compress_ratios.size() &&
                              cfg_.compress_ratios[layer_id] > 0);
        float* layer_rope_freqs = is_compressed ? rope_freqs_compressed_.f32()
                                                : rope_freqs_.f32();

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

        // ── Device-Driven KV Compressor (for kv_source_layers 2, 8, 14, 20) ──
        if (lw.comp_wkv.data) {
            forward_compressor(layer_id);
        }

        // ── Single-Pass Mega-mHC Multi-Head Context Attention ────────────────
        float qk_scale = 1.0f / sqrtf((float)head_dim_val);
        int max_combined = 4096;

        const __nv_bfloat16* p_comp_kv = nullptr;
        const int32_t* p_d_comp_count = nullptr;
        if (is_compressed) {
            int src_layer = (layer_id >= 20) ? 20 : ((layer_id >= 14) ? 14 : ((layer_id >= 8) ? 8 : 2));
            p_comp_kv = layers_[src_layer].comp_kv_cache.bf16();
            p_d_comp_count = layers_[src_layer].d_comp_kv_count.i32();
        }

        mla_attention_fused_cuda(
            buf_q_.bf16(), lw.kv_cache.bf16(), p_comp_kv, lw.attn_sink.f32(),
            buf_attn_out_.bf16(), buf_input_pos_.i32(), p_d_comp_count,
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

        // ── 2b. Collect imatrix activations during calibration pass ───────────
        if (collect_imatrix_) {
            float* g_accum = (float*)d_gate_accum_.data + (size_t)layer_id * n_experts * dim;
            float* d_accum = (float*)d_down_accum_.data + (size_t)layer_id * n_experts * moe_inter;
            uint32_t* e_cnt = (uint32_t*)d_expert_counts_.data + (size_t)layer_id * n_experts;
            accumulate_expert_imatrix_cuda(
                g_accum, d_accum, e_cnt,
                buf_hidden_.bf16(), buf_topk_idx_.i32(),
                1, top_k, n_experts, dim, moe_inter, main_stream_);
        }

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
        expert_loader_.begin_request_epoch();

        void* exp_ptrs[6] = {nullptr};
        bool is_cold_expert[6] = {false};
        void* dram_srcs[6] = {nullptr};
        int disk_miss_k[6];
        int n_disk_miss = 0;
        size_t exp_block_sz = expert_loader_.expert_block_size() > 0 ? (size_t)expert_loader_.expert_block_size() : (size_t)18800640;
        size_t copy_sz = expert_loader_.is_mixed() ? (size_t)expert_loader_.cold_block_size() : exp_block_sz;

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
                    debug_cold_routed_token_++;
                    // Tier 1 (VRAM L1 Cold Pool): Zero-latency hit if preloaded or dynamically cached in cold_pool_gpu_
                    void* vram_cold = expert_loader_.get_cold_vram_ptr(layer_id, eid);
                    if (vram_cold) {
                        exp_ptrs[k] = vram_cold;
                        continue;
                    }
                    // Tier 2: Check L2 Host DRAM Cache (updates LRU & active_req)
                    void* dram_src = expert_loader_.try_get_dram_expert_ptr(layer_id, eid);
                    if (dram_src) {
                        dram_srcs[k] = dram_src;
                    } else {
                        disk_miss_k[n_disk_miss++] = k;
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

                if (dram_src) {
                    dram_srcs[k] = dram_src;
                } else {
                    disk_miss_k[n_disk_miss++] = k;
                }
            }
        }

        // If any experts missed L2 DRAM and require NVMe SSD I/O, launch VRAM-resident
        // and DRAM-hit experts on main_stream_ FIRST so GPU compute overlaps with SSD I/O!
        bool expert_launched[6] = {false};
        if (n_disk_miss > 0) {
            for (int k = 0; k < top_k; k++) {
                if (exp_ptrs[k] != nullptr) {
                    execute_expert_swiglu(exp_ptrs[k], 1.0f, k, buf_hidden_.bf16(), is_cold_expert[k]);
                    expert_launched[k] = true;
                } else if (dram_srcs[k] != nullptr) {
                    int eid = topk_ids_host_[k];
                    void* vram_slot = expert_loader_.is_mixed()
                        ? expert_loader_.allocate_cold_vram_l1_slot(layer_id, eid)
                        : nullptr;
                    if (!vram_slot) vram_slot = streaming_slots_gpu_[0][k];
                    CUDA_CHECK(cudaMemcpyAsync(vram_slot, dram_srcs[k], copy_sz, cudaMemcpyHostToDevice, main_stream_));
                    exp_ptrs[k] = vram_slot;
                    execute_expert_swiglu(exp_ptrs[k], 1.0f, k, buf_hidden_.bf16(), is_cold_expert[k]);
                    expert_launched[k] = true;
                }
            }
        }

        // Tier 3: Fetch any NVMe SSD misses (in parallel across expert_pool_ if multiple misses occur)
        if (n_disk_miss == 1 || !expert_pool_) {
            for (int m = 0; m < n_disk_miss; m++) {
                int k = disk_miss_k[m];
                int eid = topk_ids_host_[k];
                int64_t key = (int64_t)layer_id * n_experts + eid;
                dram_srcs[k] = expert_loader_.fetch_cold_expert_from_disk(layer_id, eid, key);
            }
        } else if (n_disk_miss > 1) {
            std::future<void*> futs[6];
            for (int m = 0; m < n_disk_miss; m++) {
                int k = disk_miss_k[m];
                int eid = topk_ids_host_[k];
                int64_t key = (int64_t)layer_id * n_experts + eid;
                futs[m] = expert_pool_->enqueue([this, layer_id, eid, key]() {
                    return expert_loader_.fetch_cold_expert_from_disk(layer_id, eid, key);
                });
            }
            for (int m = 0; m < n_disk_miss; m++) {
                int k = disk_miss_k[m];
                dram_srcs[k] = futs[m].get();
            }
        }

        // Queue async H2D DMA transfers into Dynamic VRAM L1 Cold Cache (or streaming slot fallback)
        for (int k = 0; k < top_k; k++) {
            if (exp_ptrs[k] != nullptr) continue;
            int eid = topk_ids_host_[k];
            if (eid < 0 || eid >= n_experts) continue;
            void* dram_src = dram_srcs[k];
            void* vram_slot = expert_loader_.is_mixed()
                ? expert_loader_.allocate_cold_vram_l1_slot(layer_id, eid)
                : nullptr;
            if (!vram_slot) vram_slot = streaming_slots_gpu_[0][k];
            if (dram_src && vram_slot) {
                CUDA_CHECK(cudaMemcpyAsync(vram_slot, dram_src, copy_sz, cudaMemcpyHostToDevice, main_stream_));
                exp_ptrs[k] = vram_slot;
            } else {
                LOG_ERROR("Step %d L%d: Cold expert %d FAILED to load! (dram_src=%p, vram_slot=%p)",
                          decode_step_idx_, layer_id, eid, dram_src, vram_slot);
            }
        }

        // ── 6. Compute remaining Routed Experts with Native SwiGLU ───────────
        for (int k = 0; k < top_k; k++) {
            if (expert_launched[k]) continue;
            if (exp_ptrs[k]) {
                execute_expert_swiglu(exp_ptrs[k], 1.0f, k, buf_hidden_.bf16(), is_cold_expert[k]);
            } else {
                CUDA_CHECK(cudaMemsetAsync(buf_down_.bf16() + (size_t)k * dim, 0, dim * sizeof(__nv_bfloat16), main_stream_));
            }
        }

        // ── 7. Join Side Stream (Shared Expert) + Fused 6-way Dynamic Accumulation
        CUDA_CHECK(cudaStreamWaitEvent(main_stream_, side_event_, 0));
        __nv_bfloat16* shared_down_ptr = buf_down_.bf16() + top_k * dim;
        fused_moe_accum_dynamic_cuda(buf_hidden_.bf16(), buf_down_.bf16(),
                                     buf_topk_vals_.f32(), shared_down_ptr, dim, main_stream_);
    }

    // ── Batched Prefill for DeepSeek-V4.1 (All-Resident / Mixed NVFP4+IQ2) ────

    void prefill_prompt_batched_deepseek_v41(const std::vector<int>& prompt, int start_offset = 0, bool compute_last_logits = true) {
        if (prompt.empty()) return;
        int remaining_tokens = (int)prompt.size() - start_offset;
        if (remaining_tokens <= 0) return;
        if (remaining_tokens <= 1) {
            for (size_t i = (size_t)start_offset; i < prompt.size(); i++) {
                forward_token_eager(prompt[i], (int)i, (i + 1 == prompt.size() && compute_last_logits));
            }
            CUDA_CHECK(cudaStreamSynchronize(main_stream_));
            return;
        }

        int total_tokens = (int)prompt.size();
        int chunk_size = prefill_chunk_size_;
        auto t_start = std::chrono::high_resolution_clock::now();

        int dim = cfg_.hidden_size > 0 ? cfg_.hidden_size : DEEPSEEK_V41_HIDDEN_DIM;
        int hc = cfg_.hc_mult;
        int hc_dim = hc * dim;
        int mix_size = (2 + hc) * hc;
        int q_lora = cfg_.q_lora_rank > 0 ? cfg_.q_lora_rank : 1280;
        int n_heads = cfg_.num_attention_heads;
        int head_dim_val = cfg_.head_dim;
        int rope_dim = cfg_.qk_rope_head_dim > 0 ? cfg_.qk_rope_head_dim : 64;
        int o_lora = cfg_.o_lora_rank > 0 ? cfg_.o_lora_rank : 1024;
        int o_groups = cfg_.o_groups > 0 ? cfg_.o_groups : 8;
        int window = cfg_.window_size > 0 ? cfg_.window_size : (cfg_.sliding_window > 0 ? cfg_.sliding_window : 128);
        int heads_per_group = n_heads / o_groups; // 64 / 8 = 8
        int hpg_dim = heads_per_group * head_dim_val; // 8 * 512 = 4096
        int moe_inter = cfg_.moe_intermediate_size > 0 ? cfg_.moe_intermediate_size : DEEPSEEK_V41_MOE_INTERMEDIATE;
        int shared_inter = (cfg_.moe_shared_expert_intermediate_size > 0 && cfg_.moe_shared_expert_intermediate_size != 3712)
                           ? cfg_.moe_shared_expert_intermediate_size : moe_inter;
        int n_experts = cfg_.n_routed_experts > 0 ? cfg_.n_routed_experts : DEEPSEEK_V41_EXPERTS;
        int top_k = cfg_.num_experts_per_tok > 0 ? cfg_.num_experts_per_tok : DEEPSEEK_V41_ACTIVE_EXPERTS;
        float scale = 1.0f / sqrtf((float)head_dim_val);
        int max_combined = 4096;

        int max_super_m = std::max(chunk_size, 4096);
        std::vector<float> h_pre_mix_init((size_t)max_super_m * hc, 0.0f);
        for (int m = 0; m < max_super_m; m++) {
            h_pre_mix_init[(size_t)m * hc] = 1.0f;
        }

        static int32_t* s_h_comp_counts = nullptr;
        static int32_t* s_d_comp_counts = nullptr;
        static int s_comp_counts_cap = 0;
        if (!s_d_comp_counts || s_comp_counts_cap < 2 * max_super_m) {
            if (s_h_comp_counts) cudaFreeHost(s_h_comp_counts);
            if (s_d_comp_counts) cudaFree(s_d_comp_counts);
            s_comp_counts_cap = 2 * max_super_m;
            CUDA_CHECK(cudaMallocHost(&s_h_comp_counts, (size_t)s_comp_counts_cap * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&s_d_comp_counts, (size_t)s_comp_counts_cap * sizeof(int32_t)));
        }

        static cudaEvent_t s_pfev[14] = {nullptr};
        static bool s_pfev_init = false;
        if (!s_pfev_init) {
            s_pfev_init = true;
            for (int i = 0; i < 14; i++) CUDA_CHECK(cudaEventCreate(&s_pfev[i]));
        }

        static bool s_cublas_warmup = false;
        if (!s_cublas_warmup) {
            s_cublas_warmup = true;
            gemm_bf16(prefill_bufs_.buf_hidden.bf16(), 1, 16, 16,
                      prefill_bufs_.buf_hidden.bf16(), prefill_bufs_.buf_hidden.bf16(),
                      1.0f, 0.0f, main_stream_);
            CUDA_CHECK(cudaStreamSynchronize(main_stream_));
        }

        for (int super_start = start_offset; super_start < total_tokens; super_start += max_super_m) {
            int super_end = std::min(total_tokens, super_start + max_super_m);
            int super_M = super_end - super_start;
            bool is_last_super_chunk = (super_end == total_tokens);

            // 1. Copy all tokens, positions, and causal compressor counts for the super-chunk
            for (int m = 0; m < super_M; m++) {
                int pos = super_start + m;
                h_prefill_tok_[m] = prompt[pos];
                h_prefill_pos_[m] = pos;
                s_h_comp_counts[m] = pos + 1;                   // causal count for ratio=1
                s_h_comp_counts[max_super_m + m] = (pos + 1) / 2; // causal count for ratio=2
            }
            CUDA_CHECK(cudaMemcpyAsync(prefill_bufs_.buf_tokens.i32(), h_prefill_tok_,
                                       (size_t)super_M * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
            CUDA_CHECK(cudaMemcpyAsync(prefill_bufs_.buf_pos.i32(), h_prefill_pos_,
                                       (size_t)super_M * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
            CUDA_CHECK(cudaMemcpyAsync(s_d_comp_counts, s_h_comp_counts,
                                       (size_t)(2 * max_super_m) * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));

            // Initialize HC pre-mix for layer 0 to [1, 0, 0, 0] for all super_M tokens
            CUDA_CHECK(cudaMemcpyAsync(prefill_bufs_.buf_hc_pre_mix.f32(), h_pre_mix_init.data(),
                                       (size_t)super_M * hc * sizeof(float), cudaMemcpyHostToDevice, main_stream_));

            // 2. Embedding lookup and broadcast to HC copies across 512-token sub-chunks
            for (int start = super_start; start < super_end; start += chunk_size) {
                int M = std::min(chunk_size, super_end - start);
                int sub_off = start - super_start;
                int position = start;
                const int32_t* chunk_tokens_d = prefill_bufs_.buf_tokens.i32() + sub_off;
                __nv_bfloat16* chunk_hc_state = prefill_bufs_.buf_hc_state.bf16() + (size_t)sub_off * hc_dim;

                if (embed_weight_.dtype == "int4") {
                    embedding_int4_broadcast_device_id_cuda(
                        prefill_bufs_.buf_hidden.bf16(),
                        chunk_hc_state,
                        (uint8_t*)embed_weight_.data,
                        embed_weight_scale_.bf16(),
                        chunk_tokens_d,
                        dim, hc, main_stream_);
                } else {
                    embedding_broadcast_batch_cuda(
                        prefill_bufs_.buf_hidden.bf16(),
                        chunk_hc_state,
                        embed_weight_.bf16(),
                        chunk_tokens_d,
                        M, dim, hc, main_stream_);
                }

                // Override with vision tower embedding for image tokens
                if (active_visual_embeddings_) {
                    for (int m = 0; m < M; m++) {
                        int pos = position + m;
                        if (pos >= visual_embed_start_pos_ && pos < visual_embed_start_pos_ + visual_embed_num_tokens_) {
                            int v_idx = pos - visual_embed_start_pos_;
                            const __nv_bfloat16* src_vis = active_visual_embeddings_ + (size_t)v_idx * dim;
                            for (int h = 0; h < hc; h++) {
                                CUDA_CHECK(cudaMemcpyAsync(chunk_hc_state + (size_t)m * hc_dim + (size_t)h * dim,
                                                           src_vis,
                                                           dim * sizeof(__nv_bfloat16),
                                                           cudaMemcpyDeviceToDevice, main_stream_));
                            }
                        }
                    }
                }
            }

            // 3. Layer-Outer, Sub-Chunk-Inner loop:
            //    Each layer stages its cold experts into buf_prefill_cold_scratch ONCE and reuses
            //    them across all 512-token sub-chunks of the 4,096-token super-chunk!
            for (int layer_id = 0; layer_id < cfg_.num_hidden_layers; layer_id++) {
                auto& lw = layers_[layer_id];
                int cold_bs = expert_loader_.cold_block_size();
                bool use_staged_cold = (cfg_.expert_dtype == "mixed_nvfp4_iq2" && prefill_bufs_.buf_prefill_cold_scratch.data != nullptr);
                int max_stage_slots = (use_staged_cold && cold_bs > 0)
                    ? (int)(prefill_bufs_.buf_prefill_cold_scratch.size_bytes / (size_t)cold_bs)
                    : 0;

                int stage_slot_for_eid[384];
                void* dram_ptr_for_eid[384];
                int layer_eid_total_counts[384] = {0};
                for (int eid = 0; eid < n_experts; eid++) {
                    stage_slot_for_eid[eid] = -1;
                    dram_ptr_for_eid[eid] = nullptr;
                }
                int next_stage_slot = 0;

                for (int start = super_start; start < super_end; start += chunk_size) {
                    int M = std::min(chunk_size, super_end - start);
                    int sub_off = start - super_start;
                    const int32_t* chunk_pos_d = prefill_bufs_.buf_pos.i32() + sub_off;
                    __nv_bfloat16* chunk_hc_state = prefill_bufs_.buf_hc_state.bf16() + (size_t)sub_off * hc_dim;
                    float* chunk_hc_pre_mix = prefill_bufs_.buf_hc_pre_mix.f32() + (size_t)sub_off * hc;

                    bool do_prof = (start == super_start) && (layer_id == 0 || layer_id == 20 || layer_id == 39);
                    if (do_prof) CUDA_CHECK(cudaEventRecord(s_pfev[0], main_stream_));

                    // ── 1. Attention Hyper-Connection Mixes from chunk_hc_state ──────────
                    gemv_hc_pre_norm_batch_cuda(prefill_bufs_.buf_hc_mixes.f32(),
                                                chunk_hc_state,
                                                lw.hc_attn_fn.f32(),
                                                M, mix_size, hc_dim, cfg_.hc_eps, main_stream_);

                    // Split into attn_pre (buf_hc_pre), attn_post (buf_hc_post), attn_comb (buf_hc_comb)
                    hc_split_sinkhorn_batch_cuda(prefill_bufs_.buf_hc_pre.f32(),
                                                 prefill_bufs_.buf_hc_post.f32(),
                                                 prefill_bufs_.buf_hc_comb.f32(),
                                                 prefill_bufs_.buf_hc_mixes.f32(),
                                                 lw.hc_attn_scale.f32(),
                                                 lw.hc_attn_base.f32(),
                                                 M, hc, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, main_stream_);

                    // ── 2. Collapse HC copies using chunk_hc_pre_mix & Attention Pre-Norm
                    hc_pre_weighted_add_norm_batch_cuda(prefill_bufs_.buf_hidden.bf16(),
                                                        chunk_hc_state,
                                                        chunk_hc_pre_mix,
                                                        lw.attn_norm_w.bf16(),
                                                        M, dim, hc, cfg_.rms_norm_eps, main_stream_);

                    if (do_prof) CUDA_CHECK(cudaEventRecord(s_pfev[1], main_stream_));

                    // ── 3. Attention Q & KV Projections ──────────────────────────────────
                    if (lw.wq_a_w.dtype == "fp4") {
                        gemm_fp4_batch_cuda(prefill_bufs_.buf_lora.bf16(), prefill_bufs_.buf_hidden.bf16(),
                                            (const uint8_t*)lw.wq_a_w.data, (const uint8_t*)lw.wq_a_s.data,
                                            q_lora, dim, M, main_stream_);
                    } else if (lw.wq_a_w.dtype == "int4") {
                        gemm_int4_dequant(prefill_bufs_.buf_lora.bf16(), M, q_lora, dim,
                                         prefill_bufs_.buf_hidden.bf16(),
                                         (const uint8_t*)lw.wq_a_w.data, lw.wq_a_s.bf16(), 32, main_stream_);
                    } else {
                        gemm_fp8_dequant(prefill_bufs_.buf_lora.bf16(), M, q_lora, dim,
                                         prefill_bufs_.buf_hidden.bf16(),
                                         lw.wq_a_w.u8(), lw.wq_a_s.u8(), 32, main_stream_);
                    }

                    rms_norm_cuda_batched(prefill_bufs_.buf_lora.bf16(), prefill_bufs_.buf_lora.bf16(),
                                          lw.q_norm_w.bf16(), M, q_lora, cfg_.rms_norm_eps, main_stream_);

                    if (lw.wq_b_w.dtype == "fp4") {
                        gemm_fp4_batch_cuda(prefill_bufs_.buf_q.bf16(), prefill_bufs_.buf_lora.bf16(),
                                            (const uint8_t*)lw.wq_b_w.data, (const uint8_t*)lw.wq_b_s.data,
                                            n_heads * head_dim_val, q_lora, M, main_stream_);
                    } else if (lw.wq_b_w.dtype == "int4") {
                        gemm_int4_dequant(prefill_bufs_.buf_q.bf16(), M, n_heads * head_dim_val, q_lora,
                                         prefill_bufs_.buf_lora.bf16(),
                                         (const uint8_t*)lw.wq_b_w.data, lw.wq_b_s.bf16(), 32, main_stream_);
                    } else {
                        gemm_fp8_dequant(prefill_bufs_.buf_q.bf16(), M, n_heads * head_dim_val, q_lora,
                                         prefill_bufs_.buf_lora.bf16(),
                                         lw.wq_b_w.u8(), lw.wq_b_s.u8(), 32, main_stream_);
                    }

                    if (lw.wkv_w.dtype == "fp4") {
                        gemm_fp4_batch_cuda(prefill_bufs_.buf_kv.bf16(), prefill_bufs_.buf_hidden.bf16(),
                                            (const uint8_t*)lw.wkv_w.data, (const uint8_t*)lw.wkv_s.data,
                                            head_dim_val, dim, M, main_stream_);
                    } else if (lw.wkv_w.dtype == "int4") {
                        gemm_int4_dequant(prefill_bufs_.buf_kv.bf16(), M, head_dim_val, dim,
                                         prefill_bufs_.buf_hidden.bf16(),
                                         (const uint8_t*)lw.wkv_w.data, lw.wkv_s.bf16(), 32, main_stream_);
                    } else {
                        gemm_fp8_dequant(prefill_bufs_.buf_kv.bf16(), M, head_dim_val, dim,
                                         prefill_bufs_.buf_hidden.bf16(),
                                         lw.wkv_w.u8(), lw.wkv_s.u8(), 32, main_stream_);
                    }

                    rms_norm_cuda_batched(prefill_bufs_.buf_kv.bf16(), prefill_bufs_.buf_kv.bf16(),
                                          lw.kv_norm_w.bf16(), M, head_dim_val, cfg_.rms_norm_eps, main_stream_);

                    if (do_prof) CUDA_CHECK(cudaEventRecord(s_pfev[2], main_stream_));

                    // ── 4. Causal Attention loop across the M tokens ─────────────────────
                    bool is_compressed = (layer_id < (int)cfg_.compress_ratios.size() &&
                                          cfg_.compress_ratios[layer_id] > 0);
                    float* layer_rope_freqs = is_compressed ? rope_freqs_compressed_.f32()
                                                            : rope_freqs_.f32();
                    int ratio = cfg_.layer_compress_ratio(layer_id);

                    const __nv_bfloat16* p_comp_kv = nullptr;
                    if (is_compressed) {
                        int src_layer = (layer_id >= 20) ? 20 : ((layer_id >= 14) ? 14 : ((layer_id >= 8) ? 8 : 2));
                        p_comp_kv = layers_[src_layer].comp_kv_cache.bf16();
                    }

                    // Batch RoPE for all M tokens in one launch (in-place on distinct rows of buf_kv):
                    rope_device_pos_batch_cuda(prefill_bufs_.buf_kv.bf16(), M, head_dim_val, rope_dim,
                                               chunk_pos_d, layer_rope_freqs, false, main_stream_);

                    // 1. Batched KV Compressor step (on source layers 2, 8, 14, 20)
                    if (ratio > 0 && lw.comp_wkv.data) {
                        int coff = (ratio == 4) ? 2 : 1;
                        int proj_dim = coff * head_dim_val;
                        float* proj_kv_batch = prefill_bufs_.buf_shared_gate.f32();
                        float* proj_gate_batch = lw.comp_wgate.data ? prefill_bufs_.buf_shared_up.f32() : nullptr;
                        const float alpha_one = 1.0f;
                        const float beta_zero = 0.0f;
                        CUBLAS_CHECK(cublasGemmEx(
                            cublas_handle_,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            proj_dim, M, dim,
                            &alpha_one,
                            lw.comp_wkv.bf16(), CUDA_R_16BF, dim,
                            prefill_bufs_.buf_hidden.bf16(), CUDA_R_16BF, dim,
                            &beta_zero,
                            proj_kv_batch, CUDA_R_32F, proj_dim,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                        if (proj_gate_batch) {
                            CUBLAS_CHECK(cublasGemmEx(
                                cublas_handle_,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                proj_dim, M, dim,
                                &alpha_one,
                                lw.comp_wgate.bf16(), CUDA_R_16BF, dim,
                                prefill_bufs_.buf_hidden.bf16(), CUDA_R_16BF, dim,
                                &beta_zero,
                                proj_gate_batch, CUDA_R_32F, proj_dim,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                        }
                        compressor_device_batch_step_cuda(
                            chunk_pos_d,
                            lw.d_comp_kv_count.i32(),
                            proj_kv_batch,
                            proj_gate_batch,
                            M,
                            lw.comp_kv_state.f32(),
                            lw.comp_score_state.f32(),
                            lw.comp_ape.data ? lw.comp_ape.f32() : nullptr,
                            lw.comp_norm.bf16(),
                            lw.comp_kv_cache.bf16(),
                            rope_freqs_compressed_.f32(),
                            ratio,
                            head_dim_val,
                            rope_dim,
                            cfg_.rms_norm_eps,
                            (cfg_.max_seq_len / ratio + 2),
                            main_stream_);
                    }

                    // 2. Batched Flash-MLA Attention across all M tokens in parallel
                    const int32_t* p_d_comp_counts = is_compressed
                        ? ((ratio == 2) ? (s_d_comp_counts + max_super_m + sub_off) : (s_d_comp_counts + sub_off))
                        : nullptr;
                    mla_attention_fused_batch_cuda(
                        prefill_bufs_.buf_q.bf16(),
                        lw.kv_cache.bf16(),
                        prefill_bufs_.buf_kv.bf16(),
                        p_comp_kv,
                        lw.attn_sink.f32(),
                        prefill_bufs_.buf_attn_out.bf16(),
                        chunk_pos_d,
                        p_d_comp_counts,
                        M,
                        layer_rope_freqs,
                        max_combined,
                        head_dim_val,
                        rope_dim,
                        scale,
                        0.0f,
                        window,
                        main_stream_);

                    // 3. Store the trailing min(M, window) KV vectors into the 128-token sliding window ring buffer
                    int m_store_start = std::max(0, M - window);
                    int m_store_count = M - m_store_start;
                    store_kv_batch_cuda(
                        lw.kv_cache.bf16(),
                        prefill_bufs_.buf_kv.bf16() + (size_t)m_store_start * head_dim_val,
                        chunk_pos_d + m_store_start,
                        m_store_count,
                        window,
                        head_dim_val,
                        main_stream_);

                    if (do_prof) CUDA_CHECK(cudaEventRecord(s_pfev[3], main_stream_));

                    // ── 5. Output projection (wo_a and wo_b) ──────────────────────────────
                    if (lw.wo_a_w.dtype == "int4") {
                        gemv_int4_grouped_batch_cuda(prefill_bufs_.buf_lora.bf16(),
                                                     prefill_bufs_.buf_attn_out.bf16(),
                                                     (const uint8_t*)lw.wo_a_w.data, lw.wo_a_s.bf16(),
                                                     M, o_lora, hpg_dim, o_groups, main_stream_);
                    } else if (M >= 128) {
                        fp8_dequant_cuda(buf_dequant_.bf16(), lw.wo_a_w.u8(), lw.wo_a_s.u8(),
                                         o_groups * o_lora, hpg_dim, 32, main_stream_);
                        const float alpha_one = 1.0f;
                        const float beta_zero = 0.0f;
                        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                            cublas_handle_,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            o_lora, M, hpg_dim,
                            &alpha_one,
                            buf_dequant_.bf16(), CUDA_R_16BF, hpg_dim, (long long)o_lora * hpg_dim,
                            prefill_bufs_.buf_attn_out.bf16(), CUDA_R_16BF, o_groups * hpg_dim, (long long)hpg_dim,
                            &beta_zero,
                            prefill_bufs_.buf_lora.bf16(), CUDA_R_16BF, o_groups * o_lora, (long long)o_lora,
                            o_groups,
                            CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
                    } else {
                        gemv_fp8_grouped_batch_cuda(prefill_bufs_.buf_lora.bf16(),
                                                    prefill_bufs_.buf_attn_out.bf16(),
                                                    lw.wo_a_w.u8(), lw.wo_a_s.u8(),
                                                    M, o_lora, hpg_dim, o_groups, 32, main_stream_);
                    }

                    if (lw.wo_b_w.dtype == "int4") {
                        gemm_int4_dequant(prefill_bufs_.buf_hidden.bf16(), M, dim, o_groups * o_lora,
                                         prefill_bufs_.buf_lora.bf16(),
                                         (const uint8_t*)lw.wo_b_w.data, lw.wo_b_s.bf16(), 32, main_stream_);
                    } else {
                        gemm_fp8_dequant(prefill_bufs_.buf_hidden.bf16(), M, dim, o_groups * o_lora,
                                         prefill_bufs_.buf_lora.bf16(),
                                         lw.wo_b_w.u8(), lw.wo_b_s.u8(), 32, main_stream_);
                    }

                    // ── 6. HC post for attention: reads chunk_hc_state, writes to buf_hc_after_attn
                    hc_post_update_batch_cuda(prefill_bufs_.buf_hc_after_attn.bf16(),
                                              prefill_bufs_.buf_hidden.bf16(),
                                              chunk_hc_state,
                                              prefill_bufs_.buf_hc_post.f32(),
                                              prefill_bufs_.buf_hc_comb.f32(),
                                              M, dim, hc, main_stream_);

                    if (do_prof) CUDA_CHECK(cudaEventRecord(s_pfev[4], main_stream_));

                    // ── 7. HC pre for FFN + FFN norm (reads buf_hc_after_attn, uses attn_pre in buf_hc_pre)
                    hc_pre_weighted_add_norm_batch_cuda(prefill_bufs_.buf_hidden.bf16(),
                                                        prefill_bufs_.buf_hc_after_attn.bf16(),
                                                        prefill_bufs_.buf_hc_pre.f32(),
                                                        lw.ffn_norm_w.bf16(),
                                                        M, dim, hc, cfg_.rms_norm_eps, main_stream_);

                    // ── 8. FFN HC Mixes & Sinkhorn (saves ffn_pre into chunk_hc_pre_mix for next layer)
                    gemv_hc_pre_norm_batch_cuda(prefill_bufs_.buf_hc_mixes.f32(),
                                                prefill_bufs_.buf_hc_after_attn.bf16(),
                                                lw.hc_ffn_fn.f32(),
                                                M, mix_size, hc_dim, cfg_.hc_eps, main_stream_);

                    hc_split_sinkhorn_batch_cuda(chunk_hc_pre_mix,
                                                 prefill_bufs_.buf_hc_post.f32(),
                                                 prefill_bufs_.buf_hc_comb.f32(),
                                                 prefill_bufs_.buf_hc_mixes.f32(),
                                                 lw.hc_ffn_scale.f32(),
                                                 lw.hc_ffn_base.f32(),
                                                 M, hc, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, main_stream_);

                    if (do_prof) CUDA_CHECK(cudaEventRecord(s_pfev[5], main_stream_));

                    // ── 9. Router Gating + DeepSeek-V4.1 Top-6 SqrtSoftplus across M tokens
                    gemm_bf16(prefill_bufs_.buf_scores_bf16.bf16(), M, n_experts, dim,
                              prefill_bufs_.buf_hidden.bf16(), lw.gate_w.bf16());

                    deepseek_v41_gating_top6_batch_cuda(
                        prefill_bufs_.buf_topk_idx.i32(),
                        prefill_bufs_.buf_topk_vals.f32(),
                        prefill_bufs_.buf_scores_bf16.bf16(),
                        lw.gate_bias.f32(),
                        M, n_experts, top_k, cfg_.routed_scaling_factor,
                        1.0f,
                        main_stream_);

                    static cudaEvent_t s_topk_d2h_ev = nullptr;
                    if (!s_topk_d2h_ev) CUDA_CHECK(cudaEventCreateWithFlags(&s_topk_d2h_ev, cudaEventDisableTiming));

                    // ── 9b. Active Cold Expert VRAM Staging & Grouped Work Tile Build ────
                    const uint8_t* layer_type_map = expert_loader_.expert_type_map_gpu() + (size_t)layer_id * n_experts;
                    const void* const* flat_ptrs = expert_loader_.flat_vram_ptrs_gpu();

                    if (use_staged_cold) {
                        // Enqueue fast D2H copy of router selections
                        CUDA_CHECK(cudaMemcpyAsync(h_topk_batch_, prefill_bufs_.buf_topk_idx.i32(),
                                                   (size_t)M * top_k * sizeof(int32_t),
                                                   cudaMemcpyDeviceToHost, main_stream_));
                        CUDA_CHECK(cudaEventRecord(s_topk_d2h_ev, main_stream_));

                        // Build expert-grouped work tiles (up to 8 tokens/tile) for Hot and Cold passes
                        build_moe_work_tiles_cuda(prefill_bufs_.buf_topk_idx.i32(),
                                                  layer_type_map, 0, n_experts, M, main_stream_);
                    }

                    if (do_prof) CUDA_CHECK(cudaEventRecord(s_pfev[6], main_stream_));

                    // ── 10. Shared Expert + Hot Routed Experts (overlapped with Cold DMA) ─
                    if (lw.shared_w1_w.dtype == "int4") {
                        gemm_int4_dequant(prefill_bufs_.buf_shared_gate.bf16(), M, shared_inter, dim,
                                         prefill_bufs_.buf_hidden.bf16(),
                                         (const uint8_t*)lw.shared_w1_w.data, lw.shared_w1_s.bf16(), 32, main_stream_);

                        gemm_int4_dequant(prefill_bufs_.buf_shared_up.bf16(), M, shared_inter, dim,
                                         prefill_bufs_.buf_hidden.bf16(),
                                         (const uint8_t*)lw.shared_w3_w.data, lw.shared_w3_s.bf16(), 32, main_stream_);

                        silu_mul_cuda(prefill_bufs_.buf_shared_gate.bf16(),
                                      prefill_bufs_.buf_shared_gate.bf16(),
                                      prefill_bufs_.buf_shared_up.bf16(),
                                      M * shared_inter, cfg_.swiglu_limit, main_stream_);

                        gemm_int4_dequant(prefill_bufs_.buf_shared_down.bf16(), M, dim, shared_inter,
                                         prefill_bufs_.buf_shared_gate.bf16(),
                                         (const uint8_t*)lw.shared_w2_w.data, lw.shared_w2_s.bf16(), 32, main_stream_);
                    } else {
                        gemm_fp8_dequant(prefill_bufs_.buf_shared_gate.bf16(), M, shared_inter, dim,
                                         prefill_bufs_.buf_hidden.bf16(),
                                         lw.shared_w1_w.u8(), lw.shared_w1_s.u8(), 32, main_stream_);

                        gemm_fp8_dequant(prefill_bufs_.buf_shared_up.bf16(), M, shared_inter, dim,
                                         prefill_bufs_.buf_hidden.bf16(),
                                         lw.shared_w3_w.u8(), lw.shared_w3_s.u8(), 32, main_stream_);

                        silu_mul_cuda(prefill_bufs_.buf_shared_gate.bf16(),
                                      prefill_bufs_.buf_shared_gate.bf16(),
                                      prefill_bufs_.buf_shared_up.bf16(),
                                      M * shared_inter, cfg_.swiglu_limit, main_stream_);

                        gemm_fp8_dequant(prefill_bufs_.buf_shared_down.bf16(), M, dim, shared_inter,
                                         prefill_bufs_.buf_shared_gate.bf16(),
                                         lw.shared_w2_w.u8(), lw.shared_w2_s.u8(), 32, main_stream_);
                    }

                    size_t w1_cold_off = 0;
                    size_t w3_cold_off = (size_t)moe_inter * (dim / 256) * sizeof(block_iq2_xxs);
                    size_t w2_cold_off = w3_cold_off + (size_t)moe_inter * (dim / 256) * sizeof(block_iq2_xxs);

                    // Launch Hot Routed Experts IMMEDIATELY on main_stream_ before waiting for PCIe DMA!
                    if (use_staged_cold) {
                        const void* const* layer_flat_ptrs = flat_ptrs + (size_t)layer_id * n_experts;
                        gemv_mixed_moe_swiglu_grouped_pass_cuda(
                            prefill_bufs_.buf_gate.bf16(),
                            prefill_bufs_.buf_hidden.bf16(),
                            (int)w1_cold_off, (int)w3_cold_off,
                            moe_inter, dim, cfg_.swiglu_limit,
                            layer_flat_ptrs, 0, n_experts, M, true, main_stream_);

                        gemv_mixed_moe_down_grouped_pass_cuda(
                            prefill_bufs_.buf_down.bf16(),
                            prefill_bufs_.buf_gate.bf16(),
                            layer_flat_ptrs, 0, n_experts,
                            (int)w2_cold_off,
                            dim, moe_inter, M, true, main_stream_);
                    }

                    if (do_prof) CUDA_CHECK(cudaEventRecord(s_pfev[7], main_stream_));

                    // Process cold expert DMA transfers concurrently with Shared + Hot Routed Experts
                    int exact_cold_tiles = -1;
                    bool need_dma_wait = false;
                    if (use_staged_cold) {
                        CUDA_CHECK(cudaEventSynchronize(s_topk_d2h_ev));

                        // Scan unique cold experts activated by any of the M tokens that are NOT already
                        // in VRAM L1 cold_pool_gpu_ AND NOT already staged in buf_prefill_cold_scratch
                        // from an earlier sub-chunk of this same layer!
                        uint8_t cold_needed[384] = {0};
                        int eid_counts[384] = {0};
                        int needed_count = 0;
                        int total_picks = M * top_k;
                        for (int i = 0; i < total_picks; i++) {
                            int eid = h_topk_batch_[i];
                            if (eid >= 0 && eid < n_experts) {
                                eid_counts[eid]++;
                                layer_eid_total_counts[eid]++;
                                if (!expert_loader_.is_hot(layer_id, eid) &&
                                    !expert_loader_.get_cold_vram_ptr(layer_id, eid) &&
                                    stage_slot_for_eid[eid] < 0 &&
                                    !cold_needed[eid]) {
                                    cold_needed[eid] = 1;
                                    needed_count++;
                                }
                            }
                        }
                        exact_cold_tiles = 0;
                        for (int eid = 0; eid < n_experts; eid++) {
                            if (eid_counts[eid] > 0 && !expert_loader_.is_hot(layer_id, eid)) {
                                exact_cold_tiles += (eid_counts[eid] + 7) / 8;
                            }
                        }

                        if (needed_count > 0) {
                            expert_loader_.begin_request_epoch();

                            // 1. Fast path: resolve DRAM L2 hits and enqueue async H2D copies into buf_prefill_cold_scratch
                            std::vector<int> miss_eids;
                            miss_eids.reserve(32);
                            int dram_hits_count = 0;
                            for (int eid = 0; eid < n_experts; eid++) {
                                if (!cold_needed[eid]) continue;
                                void* src = expert_loader_.try_get_dram_expert_ptr(layer_id, eid);
                                if (src) {
                                    dram_hits_count++;
                                    dram_ptr_for_eid[eid] = src;
                                    if (next_stage_slot < max_stage_slots) {
                                        int slot = next_stage_slot++;
                                        stage_slot_for_eid[eid] = slot;
                                        void* dst = (char*)prefill_bufs_.buf_prefill_cold_scratch.data + (size_t)slot * cold_bs;
                                        CUDA_CHECK(cudaMemcpyAsync(dst, src, cold_bs, cudaMemcpyHostToDevice, dma_stream_));
                                    }
                                } else {
                                    miss_eids.push_back(eid);
                                }
                            }

                            // 2. If any cold experts missed L2 DRAM, wait for DRAM hit DMAs to finish BEFORE advancing epoch & evicting DRAM slots
                            if (!miss_eids.empty()) {
                                if (dram_hits_count > 0) {
                                    CUDA_CHECK(cudaStreamSynchronize(dma_stream_));
                                    expert_loader_.begin_request_epoch();
                                }
                                for (size_t m_off = 0; m_off < miss_eids.size(); m_off += 32) {
                                    size_t m_end = std::min(miss_eids.size(), m_off + 32);
                                    size_t chunk_len = m_end - m_off;
                                    if (m_off > 0) {
                                        CUDA_CHECK(cudaStreamSynchronize(dma_stream_));
                                        expert_loader_.begin_request_epoch();
                                    }
                                    if (chunk_len > 1 && expert_pool_) {
                                        std::vector<std::future<void*>> futs;
                                        futs.reserve(chunk_len);
                                        for (size_t idx = m_off; idx < m_end; idx++) {
                                            int eid = miss_eids[idx];
                                            futs.push_back(expert_pool_->enqueue([this, layer_id, eid]() {
                                                return expert_loader_.get_dram_expert_ptr(layer_id, eid);
                                            }));
                                        }
                                        for (size_t idx = 0; idx < chunk_len; idx++) {
                                            int eid = miss_eids[m_off + idx];
                                            void* src = futs[idx].get();
                                            if (src) {
                                                dram_ptr_for_eid[eid] = src;
                                                if (next_stage_slot < max_stage_slots) {
                                                    int slot = next_stage_slot++;
                                                    stage_slot_for_eid[eid] = slot;
                                                    void* dst = (char*)prefill_bufs_.buf_prefill_cold_scratch.data + (size_t)slot * cold_bs;
                                                    CUDA_CHECK(cudaMemcpyAsync(dst, src, cold_bs, cudaMemcpyHostToDevice, dma_stream_));
                                                }
                                            }
                                        }
                                    } else {
                                        for (size_t idx = m_off; idx < m_end; idx++) {
                                            int eid = miss_eids[idx];
                                            void* src = expert_loader_.get_dram_expert_ptr(layer_id, eid);
                                            if (src) {
                                                dram_ptr_for_eid[eid] = src;
                                                if (next_stage_slot < max_stage_slots) {
                                                    int slot = next_stage_slot++;
                                                    stage_slot_for_eid[eid] = slot;
                                                    void* dst = (char*)prefill_bufs_.buf_prefill_cold_scratch.data + (size_t)slot * cold_bs;
                                                    CUDA_CHECK(cudaMemcpyAsync(dst, src, cold_bs, cudaMemcpyHostToDevice, dma_stream_));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if (start == super_start || needed_count > 0) {
                            // Populate fast device pointer table (Hot -> resident VRAM, Cold VRAM -> cold_pool_gpu_, Staged Cold -> scratch, Overflow -> UVA DRAM)
                            for (int eid = 0; eid < n_experts; eid++) {
                                if (expert_loader_.is_hot(layer_id, eid)) {
                                    h_layer_vram_ptrs_pinned_[eid] = expert_loader_.get_hot_vram_ptr(layer_id, eid);
                                } else if (void* vc = expert_loader_.get_cold_vram_ptr(layer_id, eid)) {
                                    h_layer_vram_ptrs_pinned_[eid] = vc;
                                } else if (stage_slot_for_eid[eid] >= 0) {
                                    h_layer_vram_ptrs_pinned_[eid] = (char*)prefill_bufs_.buf_prefill_cold_scratch.data + (size_t)stage_slot_for_eid[eid] * cold_bs;
                                } else if (dram_ptr_for_eid[eid] != nullptr) {
                                    h_layer_vram_ptrs_pinned_[eid] = dram_ptr_for_eid[eid];
                                } else {
                                    h_layer_vram_ptrs_pinned_[eid] = nullptr;
                                }
                            }
                            CUDA_CHECK(cudaMemcpyAsync(prefill_bufs_.buf_layer_vram_ptrs.data,
                                                       h_layer_vram_ptrs_pinned_,
                                                       (size_t)n_experts * sizeof(void*),
                                                       cudaMemcpyHostToDevice, dma_stream_));
                            CUDA_CHECK(cudaEventRecord(dma_event_, dma_stream_));
                            need_dma_wait = true;
                        }
                    }

                    // ── 11. Cold Routed Experts (after PCIe DMA completes) ───────────────
                    auto& w1_info = expert_parts_["w1.weight"];
                    auto& w3_info = expert_parts_["w3.weight"];
                    auto& w2_info = expert_parts_["w2.weight"];

                    if (use_staged_cold) {
                        if (need_dma_wait) {
                            CUDA_CHECK(cudaStreamWaitEvent(main_stream_, dma_event_, 0));
                        }
                        if (do_prof) CUDA_CHECK(cudaEventRecord(s_pfev[10], main_stream_));

                        const void* const* staged_ptrs = (const void* const*)prefill_bufs_.buf_layer_vram_ptrs.data;

                        gemv_mixed_moe_swiglu_grouped_pass_cuda(
                            prefill_bufs_.buf_gate.bf16(),
                            prefill_bufs_.buf_hidden.bf16(),
                            (int)w1_cold_off, (int)w3_cold_off,
                            moe_inter, dim, cfg_.swiglu_limit,
                            staged_ptrs, 0, n_experts, M, false, main_stream_,
                            exact_cold_tiles);

                        if (do_prof) CUDA_CHECK(cudaEventRecord(s_pfev[11], main_stream_));

                        gemv_mixed_moe_down_grouped_pass_cuda(
                            prefill_bufs_.buf_down.bf16(),
                            prefill_bufs_.buf_gate.bf16(),
                            staged_ptrs, 0, n_experts,
                            (int)w2_cold_off,
                            dim, moe_inter, M, false, main_stream_,
                            exact_cold_tiles);
                    } else if (cfg_.expert_dtype == "mixed_nvfp4_iq2") {
                        gemv_mixed_moe_swiglu_fused_batch_cuda(
                            prefill_bufs_.buf_gate.bf16(),
                            prefill_bufs_.buf_hidden.bf16(),
                            (int)w1_cold_off, (int)w3_cold_off,
                            moe_inter, dim, cfg_.swiglu_limit,
                            prefill_bufs_.buf_topk_idx.i32(),
                            flat_ptrs, expert_loader_.expert_type_map_gpu(),
                            layer_id, n_experts, M, main_stream_);

                        gemv_mixed_moe_down_batch_cuda(
                            prefill_bufs_.buf_down.bf16(),
                            prefill_bufs_.buf_gate.bf16(),
                            prefill_bufs_.buf_topk_idx.i32(),
                            flat_ptrs, expert_loader_.expert_type_map_gpu(),
                            layer_id, n_experts,
                            (int)w2_cold_off,
                            dim, moe_inter, M, main_stream_);
                    } else {
                        gemv_iq2_xxs_moe_swiglu_fused_batch_cuda(
                            prefill_bufs_.buf_gate.bf16(),
                            prefill_bufs_.buf_hidden.bf16(),
                            w1_info.offset_in_block, w3_info.offset_in_block,
                            moe_inter, dim, cfg_.swiglu_limit,
                            prefill_bufs_.buf_topk_idx.i32(),
                            flat_ptrs, layer_id, n_experts, M, main_stream_);

                        gemv_q2_k_moe_batch_cuda(
                            prefill_bufs_.buf_down.bf16(),
                            prefill_bufs_.buf_gate.bf16(),
                            prefill_bufs_.buf_topk_idx.i32(),
                            flat_ptrs, layer_id, n_experts,
                            w2_info.offset_in_block,
                            dim, moe_inter, M, main_stream_);
                    }

                    if (do_prof) CUDA_CHECK(cudaEventRecord(s_pfev[8], main_stream_));

                    // ── 12. Fused MoE Accumulation (routed experts + shared expert) ───────
                    fused_moe_accum_dynamic_batch_cuda(
                        prefill_bufs_.buf_hidden.bf16(),
                        prefill_bufs_.buf_down.bf16(),
                        prefill_bufs_.buf_topk_vals.f32(),
                        prefill_bufs_.buf_shared_down.bf16(),
                        dim, M, main_stream_);

                    // ── 13. HC post for FFN: reads buf_hc_after_attn, writes to chunk_hc_state
                    hc_post_update_batch_cuda(chunk_hc_state,
                                              prefill_bufs_.buf_hidden.bf16(),
                                              prefill_bufs_.buf_hc_after_attn.bf16(),
                                              prefill_bufs_.buf_hc_post.f32(),
                                              prefill_bufs_.buf_hc_comb.f32(),
                                              M, dim, hc, main_stream_);

                    if (do_prof) {
                        CUDA_CHECK(cudaEventRecord(s_pfev[9], main_stream_));
                        CUDA_CHECK(cudaStreamSynchronize(main_stream_));
                        float t_attn_hc=0, t_qkv=0, t_attn=0, t_out=0, t_ffn_hc=0, t_gate=0, t_shared=0, t_moe=0, t_accum=0;
                        float t_dma=0, t_swiglu=0, t_down=0;
                        cudaEventElapsedTime(&t_attn_hc, s_pfev[0], s_pfev[1]);
                        cudaEventElapsedTime(&t_qkv, s_pfev[1], s_pfev[2]);
                        cudaEventElapsedTime(&t_attn, s_pfev[2], s_pfev[3]);
                        cudaEventElapsedTime(&t_out, s_pfev[3], s_pfev[4]);
                        cudaEventElapsedTime(&t_ffn_hc, s_pfev[4], s_pfev[5]);
                        cudaEventElapsedTime(&t_gate, s_pfev[5], s_pfev[6]);
                        cudaEventElapsedTime(&t_shared, s_pfev[6], s_pfev[7]);
                        cudaEventElapsedTime(&t_moe, s_pfev[7], s_pfev[8]);
                        cudaEventElapsedTime(&t_accum, s_pfev[8], s_pfev[9]);
                        cudaEventElapsedTime(&t_dma, s_pfev[7], s_pfev[10]);
                        cudaEventElapsedTime(&t_swiglu, s_pfev[10], s_pfev[11]);
                        cudaEventElapsedTime(&t_down, s_pfev[11], s_pfev[8]);
                        float t_tot = t_attn_hc + t_qkv + t_attn + t_out + t_ffn_hc + t_gate + t_shared + t_moe + t_accum;
                        LOG_INFO("[V4.1 LAYER %d PROFILE M=%d] Total: %.2f ms | Attn_HC: %.2f ms | QKV: %.2f ms | Attn_Loop: %.2f ms | Out: %.2f ms | FFN_HC: %.2f ms | Gate: %.2f ms | Shared: %.2f ms | Routed_MoE: %.2f ms (DMA: %.2f, SwiGLU: %.2f, Down: %.2f) | Accum: %.2f ms",
                                 layer_id, M, t_tot, t_attn_hc, t_qkv, t_attn, t_out, t_ffn_hc, t_gate, t_shared, t_moe, t_dma, t_swiglu, t_down, t_accum);
                    }
                } // end sub-chunk loop

                // Pre-warm Dynamic VRAM L1 Cold Cache (cold_pool_gpu_) with the top most-activated
                // staged cold experts of this layer via ultra-fast GPU-to-GPU D2D copy (1.8 TB/s)!
                if (use_staged_cold && next_stage_slot > 0) {
                    std::pair<int, int> cand_eids[384];
                    int n_cands = 0;
                    for (int eid = 0; eid < n_experts; eid++) {
                        if (stage_slot_for_eid[eid] >= 0 && layer_eid_total_counts[eid] > 0) {
                            cand_eids[n_cands++] = {layer_eid_total_counts[eid], eid};
                        }
                    }
                    std::sort(cand_eids, cand_eids + n_cands, [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                        return a.first > b.first;
                    });
                    int max_promote = std::min(n_cands, 12);
                    for (int p = 0; p < max_promote; p++) {
                        int eid = cand_eids[p].second;
                        void* vram_l1_dst = expert_loader_.allocate_cold_vram_l1_slot(layer_id, eid);
                        if (vram_l1_dst) {
                            const void* scratch_src = (const char*)prefill_bufs_.buf_prefill_cold_scratch.data + (size_t)stage_slot_for_eid[eid] * cold_bs;
                            CUDA_CHECK(cudaMemcpyAsync(vram_l1_dst, scratch_src, cold_bs, cudaMemcpyDeviceToDevice, main_stream_));
                        }
                    }
                }
            } // end layer loop

            // 4. If this is the last super-chunk, prepare logits for the last token
            if (is_last_super_chunk) {
                CUDA_CHECK(cudaMemcpyAsync(buf_hc_state_.bf16(),
                                           prefill_bufs_.buf_hc_state.bf16() + (size_t)(super_M - 1) * hc_dim,
                                           hc_dim * sizeof(__nv_bfloat16),
                                           cudaMemcpyDeviceToDevice, main_stream_));
                CUDA_CHECK(cudaMemcpyAsync(buf_hc_pre_mix_.f32(),
                                           prefill_bufs_.buf_hc_pre_mix.f32() + (size_t)(super_M - 1) * hc,
                                           hc * sizeof(float),
                                           cudaMemcpyDeviceToDevice, main_stream_));
                if (compute_last_logits) {
                    hc_pre_weighted_add_cuda(buf_hidden_.bf16(), buf_hc_state_.bf16(), buf_hc_pre_mix_.f32(), dim, hc, main_stream_);
                    rms_norm_cuda(buf_hidden_.bf16(), buf_hidden_.bf16(),
                                  norm_weight_.bf16(), dim, cfg_.rms_norm_eps, main_stream_);
                    compute_logits();
                }
            }
        } // end super-chunk loop

        CUDA_CHECK(cudaStreamSynchronize(main_stream_));
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();
        size_t evaluated_tokens = (total_tokens > start_offset) ? (size_t)(total_tokens - start_offset) : 0;
        double tok_per_sec = (elapsed_sec > 0.0 && evaluated_tokens > 0) ? ((double)evaluated_tokens / elapsed_sec) : 0.0;
        LOG_INFO("[PREFILL STATS] %zu prompt tokens prefilled in %.3fs -> %.2f tok/s (V4.1 batched chunk_size=%d, offset=%d)",
                 evaluated_tokens, elapsed_sec, tok_per_sec, chunk_size, start_offset);
        LOG_INFO("[PREFILL I/O] Total cold expert disk reads: %llu", (unsigned long long)expert_loader_.cold_disk_reads_.load());
    }

    // ── DeepSeek-V4.1 Protocol, Parameters & Chat Templating ──────────────────

    std::string get_default_system_prompt_deepseek_v41() const {
        return "";
    }

    nlohmann::json get_recommended_sampling_deepseek_v41() const {
        return {
            {"temperature", 0.0f},
            {"top_p", 1.0f},
            {"repetition_penalty", 1.0f},
            {"reasoning_effort", "high"},
            {"thinking_budget", 75},
            {"anti_loop", false}
        };
    }

    MultimodalPrompt format_chat_prompt_deepseek_v41(
        const nlohmann::json& messages,
        bool enable_thinking = true,
        const std::string& reasoning_effort = "high",
        const nlohmann::json& tools = nlohmann::json(),
        bool model_has_vision = false) const
    {
        MultimodalPrompt out;
        std::vector<int> result;

        int BOS = tokenizer_.get_token_id("<｜begin of sentence｜>");
        if (BOS < 0) BOS = tokenizer_.get_token_id("<｜begin\xe2\x96\x81of\xe2\x96\x81sentence｜>");
        if (BOS < 0) BOS = 0;

        int EOS = tokenizer_.get_token_id("<｜end of sentence｜>");
        if (EOS < 0) EOS = tokenizer_.get_token_id("<｜end\xe2\x96\x81of\xe2\x96\x81sentence｜>");
        if (EOS < 1) EOS = 1;

        int SYSTEM = tokenizer_.get_token_id("<｜System｜>");
        if (SYSTEM < 0) SYSTEM = tokenizer_.get_token_id("<｜System\xe2\x96\x81｜>");
        if (SYSTEM < 0) SYSTEM = 128799;

        int USER = tokenizer_.get_token_id("<｜User｜>");
        if (USER < 0) USER = 128803;

        int ASSISTANT = tokenizer_.get_token_id("<｜Assistant｜>");
        if (ASSISTANT < 0) ASSISTANT = 128804;

        int THINK_BEGIN = tokenizer_.get_token_id("<think>");
        if (THINK_BEGIN < 0) THINK_BEGIN = 128821;

        int THINK_END = tokenizer_.get_token_id("</think>");
        if (THINK_END < 0) THINK_END = 128822;

        bool has_image = false;
        for (const auto& msg : messages) {
            if (message_has_image(msg)) {
                has_image = true;
                break;
            }
        }
        bool actual_thinking = enable_thinking && !has_image;

        // V4.1 official numeric reasoning budget: low=50, high=75, max=100
        int budget = 75;
        std::string effort_lower = reasoning_effort;
        for (char& c : effort_lower) c = (char)std::tolower((unsigned char)c);
        if (effort_lower == "low") {
            budget = 50;
        } else if (effort_lower == "high" || effort_lower.empty()) {
            budget = 75;
        } else if (effort_lower == "max") {
            budget = 100;
        } else {
            try {
                int parsed = std::stoi(reasoning_effort);
                if (parsed >= 1 && parsed <= 100) budget = parsed;
            } catch (...) {}
        }

        result.push_back(BOS);

        // Check if messages contains an initial system prompt
        bool has_system = (!messages.empty() && messages[0].value("role", "") == "system");
        std::string sys_content = "";
        if (has_system) {
            sys_content = get_message_content_string(messages[0]);
        }

        // Render tools using official V4.1 DSML format (<｜DSML｜ calls>)
        std::string tools_prompt = "";
        if (!tools.is_null() && !tools.empty()) {
            tools_prompt = "## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜ calls>\" block like the following:\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"$TOOL_NAME\">\n<｜DSML｜ parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜ parameter>\n...\n</｜DSML｜ invoke>\n</｜DSML｜ calls>\n\n### Available Tool Schemas\n\n";
            for (const auto& t : tools) {
                tools_prompt += t.dump() + "\n";
            }
            tools_prompt += "\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.";
        }

        if (actual_thinking) {
            // In thinking mode: <｜System｜>Reasoning Effort: {budget} ...
            result.push_back(SYSTEM);
            std::string leading_sys = "Reasoning Effort: " + std::to_string(budget) +
                                      " (range 1-100, the higher the value, the more thorough the reasoning)\n\n";
            if (!sys_content.empty()) {
                leading_sys += sys_content;
                if (!tools_prompt.empty()) leading_sys += "\n\n" + tools_prompt;
            } else if (!tools_prompt.empty()) {
                leading_sys += tools_prompt;
            }
            auto enc = tokenizer_.encode(leading_sys);
            result.insert(result.end(), enc.begin(), enc.end());
        } else {
            // In chat mode (non-thinking): only emit <｜System｜> if a system prompt or tools exist
            if (!sys_content.empty() || !tools_prompt.empty()) {
                result.push_back(SYSTEM);
                std::string leading_sys = sys_content;
                if (!tools_prompt.empty()) {
                    if (!leading_sys.empty()) leading_sys += "\n\n";
                    leading_sys += tools_prompt;
                }
                auto enc = tokenizer_.encode(leading_sys);
                result.insert(result.end(), enc.begin(), enc.end());
            }
        }

        // Process message turns
        size_t start_idx = has_system ? 1 : 0;
        for (size_t i = start_idx; i < messages.size(); i++) {
            std::string role = messages[i].value("role", "user");
            std::string content = get_message_content_string(messages[i]);

            if (role == "system") {
                // Mid-conversation system message (V4.1 feature)
                result.push_back(SYSTEM);
                auto enc = tokenizer_.encode(content);
                result.insert(result.end(), enc.begin(), enc.end());
            } else if (role == "user") {
                result.push_back(USER);
                if (model_has_vision && message_has_image(messages[i])) {
                    int IMAGE_TOKEN = tokenizer_.get_token_id("<｜image｜>");
                    if (IMAGE_TOKEN < 0) IMAGE_TOKEN = 129279;
                    out.image_b64 = extract_image_b64_from_message(messages[i]);
                    out.visual_start_pos = (int)result.size();
                    out.visual_num_tokens = 576;
                    result.insert(result.end(), 576, IMAGE_TOKEN);
                    auto newline_enc = tokenizer_.encode("\n");
                    result.insert(result.end(), newline_enc.begin(), newline_enc.end());
                }
                auto enc = tokenizer_.encode(content);
                result.insert(result.end(), enc.begin(), enc.end());
            } else if (role == "assistant") {
                result.push_back(ASSISTANT);
                bool has_reasoning = false;
                if (messages[i].contains("reasoning_content") && messages[i]["reasoning_content"].is_string()) {
                    std::string r_content = messages[i]["reasoning_content"].get<std::string>();
                    if (!r_content.empty()) {
                        result.push_back(THINK_BEGIN);
                        auto r_enc = tokenizer_.encode(r_content);
                        result.insert(result.end(), r_enc.begin(), r_enc.end());
                        result.push_back(THINK_END);
                        has_reasoning = true;
                    }
                }
                if (!has_reasoning) {
                    result.push_back(THINK_END);
                }
                auto enc = tokenizer_.encode(content);
                result.insert(result.end(), enc.begin(), enc.end());
                result.push_back(EOS);
            } else if (role == "tool" || role == "function") {
                result.push_back(USER);
                auto enc = tokenizer_.encode("<tool_result>\n" + content + "\n</tool_result>");
                result.insert(result.end(), enc.begin(), enc.end());
            }
        }

        // Final assistant generation header
        if (!messages.empty()) {
            std::string last_role = messages.back().value("role", "user");
            if (last_role == "user" || last_role == "tool" || last_role == "function" || (last_role == "system" && messages.size() > 1)) {
                result.push_back(ASSISTANT);
                if (actual_thinking) {
                    result.push_back(THINK_BEGIN);
                } else {
                    result.push_back(THINK_END);
                }
            }
        }

        out.prompt = result;
        return out;
    }
