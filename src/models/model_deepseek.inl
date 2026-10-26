    // ── Forward one layer ───────────────────────────────────────────────────

    void forward_layer(int layer_id, int token_id, int position) {
        if (cfg_.is_qwen()) {
            forward_layer_qwen(layer_id, position);
            return;
        }
        if (cfg_.is_nemotron()) {
            forward_layer_nemotron(layer_id, token_id, position);
            return;
        }
        if (cfg_.is_deepseek_v41()) {
            forward_layer_deepseek_v41(layer_id, token_id, position);
            return;
        }

        auto& lw = layers_[layer_id];
        int dim = cfg_.hidden_size;

        // ── HC pre for attention + Attention norm (fused into 1 pass) ──
        hc_pre_norm(buf_hc_state_, lw.hc_attn_fn, lw.hc_attn_scale, lw.hc_attn_base,
                    lw.attn_norm_w.bf16());
        CUDA_CHECK(cudaEventRecord(main_event_, main_stream_));

        // ── Attention ──
        forward_attention(layer_id, position);

        // ── HC post for attention: reads buf_hc_state_, writes to buf_hc_after_attn_ ──
        hc_post(buf_hc_after_attn_, buf_hc_state_);

        // ── HC pre for FFN + FFN norm (fused into 1 pass) ──
        hc_pre_norm(buf_hc_after_attn_, lw.hc_ffn_fn, lw.hc_ffn_scale, lw.hc_ffn_base,
                    lw.ffn_norm_w.bf16());
        CUDA_CHECK(cudaEventRecord(main_event_, main_stream_));

        // ── MoE FFN ──
        forward_moe(layer_id, token_id);

        // ── HC post for FFN: reads buf_hc_after_attn_, writes to buf_hc_state_ ──
        hc_post(buf_hc_state_, buf_hc_after_attn_);
    }

    // ── HC pre + Norm: reduce [hc, dim] -> [dim] and RMSNorm directly ────────
    void hc_pre_norm(GPUTensor& in_hc, GPUTensor& hc_fn, GPUTensor& hc_scale, GPUTensor& hc_base,
                     const __nv_bfloat16* norm_weight) {
        int dim = cfg_.hidden_size;
        int hc = cfg_.hc_mult;
        int mix_size = (2 + hc) * hc;
        int hc_dim = hc * dim;

        // Compute mixes from normalized hc_state directly across 24 distributed SM blocks
        gemv_hc_pre_norm_cuda(buf_hc_mixes_.f32(), in_hc.bf16(), hc_fn.f32(),
                              mix_size, hc_dim, cfg_.hc_eps, main_stream_);

        // Split mixes into pre, post, comb via Sinkhorn
        hc_split_sinkhorn_cuda(
            buf_hc_pre_.f32(), buf_hc_post_.f32(), buf_hc_comb_.f32(),
            buf_hc_mixes_.f32(), hc_scale.f32(), hc_base.f32(),
            hc, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, main_stream_);

        // Fused: weighted sum + RMSNorm directly into buf_hidden_
        hc_pre_weighted_add_norm_cuda(buf_hidden_.bf16(), in_hc.bf16(),
                                      buf_hc_pre_.f32(), norm_weight,
                                      dim, hc, cfg_.rms_norm_eps, main_stream_);
    }

    // ── HC pre: reduce [hc, dim] -> [dim] ───────────────────────────────────

    void hc_pre(GPUTensor& in_hc, GPUTensor& hc_fn, GPUTensor& hc_scale, GPUTensor& hc_base) {
        int dim = cfg_.hidden_size;
        int hc = cfg_.hc_mult;
        int mix_size = (2 + hc) * hc;
        int hc_dim = hc * dim;

        // Compute mixes from normalized hc_state directly in a single fused pass
        gemv_hc_pre_norm_cuda(buf_hc_mixes_.f32(), in_hc.bf16(), hc_fn.f32(),
                              mix_size, hc_dim, cfg_.hc_eps, main_stream_);

        // Split mixes into pre, post, comb via Sinkhorn
        hc_split_sinkhorn_cuda(
            buf_hc_pre_.f32(), buf_hc_post_.f32(), buf_hc_comb_.f32(),
            buf_hc_mixes_.f32(), hc_scale.f32(), hc_base.f32(),
            hc, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, main_stream_);

        // Compute weighted sum: y = sum(pre[i] * in_hc[i]) for i in 0..hc-1
        // Result in buf_hidden_
        hc_pre_weighted_add_cuda(buf_hidden_.bf16(), in_hc.bf16(),
                                 buf_hc_pre_.f32(), dim, hc, main_stream_);
    }

    // ── HC post: expand [dim] -> [hc, dim] (static: reads in_hc, writes out_hc) ──

    void hc_post(GPUTensor& out_hc, GPUTensor& in_hc) {
        int dim = cfg_.hidden_size;
        int hc = cfg_.hc_mult;

        hc_post_update_cuda(out_hc.bf16(), buf_hidden_.bf16(), in_hc.bf16(),
                            buf_hc_post_.f32(), buf_hc_comb_.f32(),
                            dim, hc, main_stream_);
    }

    // ── HC head: reduce [hc, dim] -> [dim] for final logits ─────────────────

    void hc_head_reduce() {
        int dim = cfg_.hidden_size;
        int hc = cfg_.hc_mult;
        int hc_dim = hc * dim;

        // mixes = x_norm @ hc_head_fn.T -> [hc]
        gemv_hc_pre_norm_cuda(buf_hc_mixes_.f32(), buf_hc_state_.bf16(),
                              hc_head_fn_.f32(), hc, hc_dim, cfg_.rms_norm_eps, main_stream_);

        // Use fused GPU kernel for head reduce
        // pre = sigmoid(mix * scale + base) + eps
        hc_head_reduce_cuda(buf_hidden_.bf16(), buf_hc_state_.bf16(),
                            buf_hc_mixes_.f32(), hc_head_scale_.f32(), hc_head_base_.f32(),
                            dim, hc, main_stream_);
    }

    // ── Attention forward ───────────────────────────────────────────────────

    void forward_attention(int layer_id, int position) {
        auto& lw = layers_[layer_id];
        int dim = cfg_.hidden_size;
        int n_heads = cfg_.num_attention_heads;
        int head_dim_val = cfg_.head_dim;
        int rope_dim = cfg_.qk_rope_head_dim;
        int q_lora = cfg_.q_lora_rank;
        int o_lora = cfg_.o_lora_rank;
        int o_groups = cfg_.o_groups;
        int window = cfg_.sliding_window;
        bool dbg = (layer_id == 0 && position == 0);

        // Select per-layer RoPE frequencies based on compress_ratio
        bool is_compressed = (layer_id < (int)cfg_.compress_ratios.size() &&
                             cfg_.compress_ratios[layer_id] > 0);
        float* layer_rope_freqs = is_compressed ? rope_freqs_compressed_.f32()
                                                : rope_freqs_.f32();

        // ── Q projection & KV projection (executed concurrently) ───────────
        // KV projection on side_stream_ (must wait for attn_norm to finish on main_stream_)
        CUDA_CHECK(cudaStreamWaitEvent(side_stream_, main_event_, 0));
        if (lw.wkv_w.dtype == "int4") {
            gemv_int4_cuda(buf_kv_.bf16(), buf_hidden_.bf16(), (const uint8_t*)lw.wkv_w.data, lw.wkv_s.bf16(), head_dim_val, dim, side_stream_);
        } else {
            gemm_fp8_dequant(buf_kv_.bf16(), 1, head_dim_val, dim,
                             buf_hidden_.bf16(),
                             lw.wkv_w.u8(), lw.wkv_s.u8(), 128, side_stream_);
        }
        rms_norm_cuda(buf_kv_.bf16(), buf_kv_.bf16(),
                      lw.kv_norm_w.bf16(), head_dim_val, cfg_.rms_norm_eps, side_stream_);
        rope_device_pos_cuda(buf_kv_.bf16(), 1, head_dim_val, rope_dim,
                             buf_input_pos_.i32(), layer_rope_freqs, false, side_stream_);
        store_kv_device_pos_cuda(lw.kv_cache.bf16(), buf_kv_.bf16(),
                                 buf_input_pos_.i32(), window, head_dim_val, side_stream_);
        CUDA_CHECK(cudaEventRecord(side_event_, side_stream_));

        // Q projection on main_stream_
        // q_raw = wq_a(x) -> [q_lora_rank]
        if (lw.wq_a_w.dtype == "int4") {
            gemv_int4_cuda(buf_lora_.bf16(), buf_hidden_.bf16(), (const uint8_t*)lw.wq_a_w.data, lw.wq_a_s.bf16(), q_lora, dim, main_stream_);
        } else {
            gemm_fp8_dequant(buf_lora_.bf16(), 1, q_lora, dim,
                             buf_hidden_.bf16(),
                             lw.wq_a_w.u8(), lw.wq_a_s.u8(), 128, main_stream_);
        }

        // q_normed = q_norm(q_raw)
        rms_norm_cuda(buf_lora_.bf16(), buf_lora_.bf16(),
                      lw.q_norm_w.bf16(), q_lora, cfg_.rms_norm_eps, main_stream_);

        // q = wq_b(q_normed) -> [n_heads * head_dim]
        if (lw.wq_b_w.dtype == "int4") {
            gemv_int4_cuda(buf_q_.bf16(), buf_lora_.bf16(), (const uint8_t*)lw.wq_b_w.data, lw.wq_b_s.bf16(), n_heads * head_dim_val, q_lora, main_stream_);
        } else {
            gemm_fp8_dequant(buf_q_.bf16(), 1, n_heads * head_dim_val, q_lora,
                             buf_lora_.bf16(),
                             lw.wq_b_w.u8(), lw.wq_b_s.u8(), 128, main_stream_);
        }
        
        // Synchronize main_stream_ with side_stream_ before compressor and attention
        CUDA_CHECK(cudaStreamWaitEvent(main_stream_, side_event_, 0));

        // ── Run compressor to accumulate/emit compressed KV entries ──────────
        // ── Device-driven KV Compressor ──────────────────────────────────────
        int ratio = cfg_.layer_compress_ratio(layer_id);
        if (ratio > 0) {
            forward_compressor(layer_id);
        }

        // ── DeepSeek V4 Indexer (for CSA ratio=4 layers) ────────────────────
        uint8_t* comp_mask_ptr = nullptr;
        if (ratio == 4 && lw.indexer_wq_b_w.data) {
            int idx_head_dim = 128;
            int idx_heads = 64;
            int idx_q_dim = idx_heads * idx_head_dim; // 8192

            // 1. Indexer Q projection: q = indexer_wq_b @ qr_norm
            if (lw.indexer_wq_b_w.dtype == "int4") {
                gemv_int4_cuda(buf_indexer_q_.bf16(), buf_lora_.bf16(),
                               (const uint8_t*)lw.indexer_wq_b_w.data, lw.indexer_wq_b_s.bf16(),
                               idx_q_dim, q_lora, main_stream_);
            } else {
                gemm_fp8_dequant(buf_indexer_q_.bf16(), 1, idx_q_dim, q_lora,
                                 buf_lora_.bf16(),
                                 lw.indexer_wq_b_w.u8(), lw.indexer_wq_b_s.u8(), 128, main_stream_);
            }

            // 2. RoPE on Indexer Q
            rope_device_pos_cuda(buf_indexer_q_.bf16(), idx_heads, idx_head_dim, rope_dim,
                                 buf_input_pos_.i32(), layer_rope_freqs, false, main_stream_);

            // 3. Weight projection: weights = indexer_weights_proj @ hidden
            gemv_bf16_cuda(buf_indexer_weights_f32_.f32(), lw.indexer_weights_proj.bf16(),
                           buf_hidden_.bf16(), idx_heads, dim, main_stream_);

            // 4. Score compressed blocks & generate Top-K mask (100% device-driven)
            int max_comp_entries = cfg_.max_compressed_entries > 0 ? cfg_.max_compressed_entries : 2048;
            indexer_score_and_mask_cuda(
                buf_indexer_mask_.u8(),
                buf_indexer_scores_.f32(),
                lw.indexer_comp_kv_cache.bf16(),
                buf_indexer_q_.bf16(),
                buf_indexer_weights_f32_.f32(),
                lw.d_comp_kv_count.i32(),
                max_comp_entries,
                512,
                main_stream_);

            comp_mask_ptr = buf_indexer_mask_.u8();
        }

        // Ensure KV store on side_stream_ is complete before reading KV cache on main_stream_
        CUDA_CHECK(cudaStreamWaitEvent(main_stream_, side_event_, 0));

        // ── Attention computation ───────────────────────────────────────────
        float scale = 1.0f / sqrtf((float)head_dim_val);
        int max_combined = window + cfg_.max_compressed_entries;

        // Fused Direct-Addressing Flash-MLA: Q-Norm + Forward RoPE + Attention Dot Products + Softmax + Value Reduction + Inverse RoPE
        mla_attention_fused_cuda(
            buf_q_.bf16(), lw.kv_cache.bf16(), lw.comp_kv_cache.bf16(), lw.attn_sink.f32(),
            buf_attn_out_.bf16(), buf_input_pos_.i32(), lw.d_comp_kv_count.i32(),
            layer_rope_freqs, max_combined, head_dim_val, rope_dim, scale,
            cfg_.rms_norm_eps, comp_mask_ptr, window, main_stream_
        );

        // ── Output projection (grouped low-rank MLA) ─────────────────────
        // Attention output: [n_heads * head_dim] = [32768]
        // wo_a: [o_groups * o_lora_rank, heads_per_group * head_dim] = [8192, 4096]
        //   Block-diagonal: group g's rows [g*o_lora : (g+1)*o_lora] operate on
        //   attn_out[g*hpg_dim : (g+1)*hpg_dim] where hpg_dim = heads_per_group * head_dim
        // wo_b: [dim, o_groups * o_lora_rank] = [4096, 8192]

        int heads_per_group = n_heads / o_groups;  // 64 / 8 = 8
        int hpg_dim = heads_per_group * head_dim_val;  // 8 * 512 = 4096
        
        // Apply wo_a per group (block-diagonal GEMM)
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
                                  o_lora, hpg_dim, o_groups, 128, main_stream_);
        }

        // wo_b: [dim, o_groups * o_lora] FP8/INT4 → output is [dim]
        if (lw.wo_b_w.dtype == "int4") {
            gemv_int4_cuda(buf_hidden_.bf16(), buf_lora_.bf16(), (const uint8_t*)lw.wo_b_w.data, lw.wo_b_s.bf16(), dim, o_groups * o_lora, main_stream_);
        } else {
            gemm_fp8_dequant(buf_hidden_.bf16(), 1, dim, o_groups * o_lora,
                             buf_lora_.bf16(),
                             lw.wo_b_w.u8(), lw.wo_b_s.u8(), 128, main_stream_);
        }
    }

    // ── MoE forward ─────────────────────────────────────────────────────────

    void forward_moe(int layer_id, int token_id) {
        auto& lw = layers_[layer_id];
        int dim = cfg_.hidden_size;
        int moe_inter = cfg_.moe_intermediate_size;
        int n_experts = cfg_.n_routed_experts;
        int top_k = cfg_.num_experts_per_tok;

        // ── Routing ─────────────────────────────────────────────────────────
        if (layer_id < cfg_.n_hash_layers) {
            // Hash layers 0..2: lookup tid2eid table with constant weight (routed_scaling_factor / top_k)
            moe_route_hash_device_id_cuda(buf_topk_idx_.i32(), buf_topk_vals_.f32(),
                                          lw.tid2eid.i64(), buf_input_token_.i32(), top_k,
                                          cfg_.routed_scaling_factor, main_stream_);
        } else {
            // Score-based layers 3..42: gate_w -> fused (bias + SqrtSoftplus + top-6 routing)
            gemm_bf16(buf_scores_bf16_.bf16(), 1, n_experts, dim,
                      buf_hidden_.bf16(), lw.gate_w.bf16());

            moe_route_top6_from_bf16_cuda(buf_topk_idx_.i32(), buf_topk_vals_.f32(),
                                          buf_scores_bf16_.bf16(), lw.gate_bias.f32(),
                                          n_experts, top_k, cfg_.routed_scaling_factor, main_stream_);
        }

        // 3b. Collect imatrix activations during calibration pass
        if (collect_imatrix_) {
            float* g_accum = (float*)d_gate_accum_.data + (size_t)layer_id * n_experts * dim;
            float* d_accum = (float*)d_down_accum_.data + (size_t)layer_id * n_experts * moe_inter;
            uint32_t* e_cnt = (uint32_t*)d_expert_counts_.data + (size_t)layer_id * n_experts;
            accumulate_expert_imatrix_cuda(
                g_accum, d_accum, e_cnt,
                buf_hidden_.bf16(), buf_topk_idx_.i32(),
                1, top_k, n_experts, dim, moe_inter, main_stream_);
        }

        // 3c. Track expert activation frequency (GPU-native, 100% CUDA Graph compatible, zero CPU latency)
        if (d_expert_counts_.data) {
            accumulate_expert_freq_cuda(
                (uint32_t*)d_expert_counts_.data,
                (int32_t*)d_step_topk_.data,
                buf_topk_idx_.i32(),
                buf_track_flag_.i32(),
                layer_id,
                n_experts,
                n_experts,
                top_k,
                main_stream_);
        }

        // 4. Populate active expert pointers
        const void* const* flat_ptrs = nullptr;
        if (expert_loader_.all_resident(cfg_.num_hidden_layers)) {
            flat_ptrs = expert_loader_.flat_vram_ptrs_gpu();
        } else {
            // ── TIMING INSTRUMENTATION for offloading path ──
            static thread_local int64_t t_token_count = 0;
            static thread_local double t_cache_resolve_us = 0;
            static thread_local double t_io_submit_us = 0;
            static thread_local double t_io_wait_us = 0;
            static thread_local double t_compute_us = 0;
            static thread_local int t_l1_hits = 0;
            static thread_local int t_l2_hits = 0;
            static thread_local int t_ssd_reads = 0;
            if (layer_id == 3) {  // Reset per-token at first MoE layer
                t_cache_resolve_us = 0; t_io_submit_us = 0; t_io_wait_us = 0; t_compute_us = 0;
                t_l1_hits = 0; t_l2_hits = 0; t_ssd_reads = 0;
            }
            auto _tp0 = std::chrono::high_resolution_clock::now();
            // Async copy top-k IDs to host with event-based sync (no full pipeline flush)
            CUDA_CHECK(cudaMemcpyAsync(topk_ids_host_, buf_topk_idx_.i32(),
                                       top_k * sizeof(int32_t), cudaMemcpyDeviceToHost, main_stream_));
            CUDA_CHECK(cudaEventRecord(dma_event_, main_stream_));

            // ── Launch shared expert FIRST on side_stream_ while we wait for top-k IDs ──
            // This overlaps shared expert GPU compute with the CPU-side expert loading below.
            // The side_stream_ waits for main_event_ (ffn_norm) which was recorded before forward_moe.
            CUDA_CHECK(cudaStreamWaitEvent(side_stream_, main_event_, 0));
            __nv_bfloat16* shared_gate = buf_gate_.bf16() + top_k * moe_inter;
            __nv_bfloat16* shared_up   = buf_up_.bf16()   + top_k * moe_inter;
            __nv_bfloat16* shared_down = buf_down_.bf16() + top_k * dim;

            if (lw.shared_w1_w.dtype == "int4") {
                gemv_int4_cuda(shared_gate, buf_hidden_.bf16(), (const uint8_t*)lw.shared_w1_w.data, lw.shared_w1_s.bf16(), moe_inter, dim, side_stream_);
                gemv_int4_cuda(shared_up, buf_hidden_.bf16(), (const uint8_t*)lw.shared_w3_w.data, lw.shared_w3_s.bf16(), moe_inter, dim, side_stream_);
                silu_mul_cuda(shared_gate, shared_gate, shared_up,
                              moe_inter, cfg_.swiglu_limit, side_stream_);
                gemv_int4_cuda(shared_down, shared_gate, (const uint8_t*)lw.shared_w2_w.data, lw.shared_w2_s.bf16(), dim, moe_inter, side_stream_);
            } else {
                gemm_fp8_dequant(shared_gate, 1, moe_inter, dim,
                                 buf_hidden_.bf16(),
                                 lw.shared_w1_w.u8(), lw.shared_w1_s.u8(), 128, side_stream_);
                gemm_fp8_dequant(shared_up, 1, moe_inter, dim,
                                 buf_hidden_.bf16(),
                                 lw.shared_w3_w.u8(), lw.shared_w3_s.u8(), 128, side_stream_);
                silu_mul_cuda(shared_gate, shared_gate, shared_up,
                              moe_inter, cfg_.swiglu_limit, side_stream_);
                gemm_fp8_dequant(shared_down, 1, dim, moe_inter,
                                 shared_gate,
                                 lw.shared_w2_w.u8(), lw.shared_w2_s.u8(), 128, side_stream_);
            }
            CUDA_CHECK(cudaEventRecord(side_event_, side_stream_));

            // ── Now wait only for the D→H copy to complete (NOT full pipeline flush) ──
            CUDA_CHECK(cudaEventSynchronize(dma_event_));

            // ── Resolve all 6 experts' cache in a single lock (no I/O blocking) ──
            void* batch_ptrs[32] = {nullptr};
            bool batch_needs_dma[32] = {false};
            // batch_get_experts resolves cache + starts parallel I/O + waits for all
            // But we need TRUE interleaving, so we'll do it ourselves:

            // Phase 1: Resolve cache under single lock (fast, no I/O)
            struct ExpertReq {
                int64_t key;
                void* gpu_dst;
                void* host_src;
                int stage_idx;
                bool needs_disk;
                bool l1_hit;
            } reqs[32];

            {
                std::unique_lock<std::mutex> lock(expert_loader_.cache_mutex_);
                expert_loader_.access_counter_++;

                for (int k = 0; k < top_k; k++) {
                    int eid = topk_ids_host_[k];
                    reqs[k].l1_hit = false;
                    reqs[k].stage_idx = -1;
                    reqs[k].needs_disk = false;

                    if (eid < 0 || eid >= n_experts) {
                        batch_ptrs[k] = nullptr;
                        batch_needs_dma[k] = false;
                        reqs[k].l1_hit = true;
                        continue;
                    }

                    int64_t key = (int64_t)layer_id * expert_loader_.n_experts_ + eid;
                    reqs[k].key = key;

                    // Check L1 (VRAM)
                    auto it = expert_loader_.key_to_slot_.find(key);
                    if (it != expert_loader_.key_to_slot_.end()) {
                        auto& slot = expert_loader_.cache_slots_[it->second];
                        slot.last_used = expert_loader_.access_counter_;
                        batch_ptrs[k] = slot.gpu_data;
                        batch_needs_dma[k] = false;
                        reqs[k].l1_hit = true;
                        continue;
                    }

                    // L1 miss — find eviction candidate
                    int evict_slot = -1;
                    int64_t oldest = INT64_MAX;
                    for (int i = 0; i < expert_loader_.cache_capacity_; i++) {
                        if (expert_loader_.cache_slots_[i].layer_id < 0) { evict_slot = i; break; }
                        if (expert_loader_.cache_slots_[i].last_used < oldest) {
                            oldest = expert_loader_.cache_slots_[i].last_used;
                            evict_slot = i;
                        }
                    }

                    auto& slot = expert_loader_.cache_slots_[evict_slot];
                    if (slot.layer_id >= 0) {
                        int64_t old_key = (int64_t)slot.layer_id * expert_loader_.n_experts_ + slot.expert_id;
                        expert_loader_.key_to_slot_.erase(old_key);
                        expert_loader_.flat_vram_ptrs_[old_key] = nullptr;
                        // Demote evicted expert to L2 DRAM (victim cache)
                        expert_loader_.demote_to_l2(slot.layer_id, slot.expert_id, old_key, slot.gpu_data);
                    }
                    slot.layer_id = layer_id;
                    slot.expert_id = eid;
                    slot.last_used = expert_loader_.access_counter_;
                    expert_loader_.key_to_slot_[key] = evict_slot;
                    expert_loader_.flat_vram_ptrs_[key] = slot.gpu_data;

                    reqs[k].gpu_dst = slot.gpu_data;
                    batch_ptrs[k] = slot.gpu_data;
                    batch_needs_dma[k] = true;

                    // Check L2 (DRAM)
                    if (expert_loader_.dram_cache_capacity_ > 0) {
                        auto dram_it = expert_loader_.dram_key_to_slot_.find(key);
                        if (dram_it != expert_loader_.dram_key_to_slot_.end()) {
                            auto& ds = expert_loader_.dram_cache_slots_[dram_it->second];
                            ds.last_used = expert_loader_.access_counter_;
                            reqs[k].host_src = ds.gpu_data;
                        } else {
                            // L2 miss — use staging ring buffer for SSD read
                            // Do NOT evict L2 entries: L2 is a stable fast cache, not SSD staging
                            reqs[k].stage_idx = expert_loader_.staging_idx_;
                            expert_loader_.staging_idx_ = (expert_loader_.staging_idx_ + 1) % ExpertLoader::NUM_STAGING_BUFFERS;
                            reqs[k].needs_disk = true;
                        }
                    } else {
                        reqs[k].stage_idx = expert_loader_.staging_idx_;
                        expert_loader_.staging_idx_ = (expert_loader_.staging_idx_ + 1) % ExpertLoader::NUM_STAGING_BUFFERS;
                        reqs[k].needs_disk = true;
                    }
                }
            }
            // ── Mutex released ──

            auto _tp1 = std::chrono::high_resolution_clock::now();
            t_cache_resolve_us += std::chrono::duration<double, std::micro>(_tp1 - _tp0).count();

            // Count hits/misses
            for (int k = 0; k < top_k; k++) {
                if (reqs[k].l1_hit) { t_l1_hits++; }
                else if (!reqs[k].needs_disk) { t_l2_hits++; }
                else { t_ssd_reads++; }
            }

            // Phase 2: Submit ALL I/O to thread pool (non-blocking, returns futures)
            std::future<bool> io_futures[32];
            for (int k = 0; k < top_k; k++) {
                if (reqs[k].l1_hit) continue;
                io_futures[k] = expert_pool_->enqueue([this, &reqs, k]() -> bool {
                    auto& r = reqs[k];
                    void* host_ptr = r.host_src;
                    if (r.stage_idx >= 0) {
                        // SSD path: 2.5ms SSD read provides enough time for writeback (0.27ms) to complete
                        auto& stage = expert_loader_.staging_ring_[r.stage_idx];
                        CUDA_CHECK(cudaEventSynchronize(stage.event));
                        int64_t bytes = expert_loader_.expert_file_.pread_exact(
                            stage.ptr, expert_loader_.expert_block_size_, r.key * expert_loader_.expert_block_size_);
                        if (bytes != expert_loader_.expert_block_size_) return false;
                        host_ptr = stage.ptr;
                        CUDA_CHECK(cudaMemcpyAsync(r.gpu_dst, host_ptr, expert_loader_.expert_block_size_,
                                                    cudaMemcpyHostToDevice, expert_streams_[k]));
                        CUDA_CHECK(cudaEventRecord(stage.event, expert_streams_[k]));
                    } else {
                        // L2 hit path: DMA is immediate, must wait for writeback to complete first
                        expert_loader_.wait_for_writeback(expert_streams_[k]);
                        if (r.needs_disk) {
                            int64_t bytes = expert_loader_.expert_file_.pread_exact(
                                host_ptr, expert_loader_.expert_block_size_, r.key * expert_loader_.expert_block_size_);
                            if (bytes != expert_loader_.expert_block_size_) return false;
                        }
                        // Route through pinned staging if DRAM cache is unpinned
                        // cudaMemcpyAsync from pageable memory is SYNCHRONOUS — blocks the stream!
                        // Instead: memcpy(DRAM→pinned) + cudaMemcpyAsync(pinned→VRAM) = true async DMA
                        if (!expert_loader_.dram_is_pinned_ && k < ExpertLoader::NUM_DMA_STAGING && expert_loader_.dma_staging_[k]) {
                            memcpy(expert_loader_.dma_staging_[k], host_ptr, expert_loader_.expert_block_size_);
                            CUDA_CHECK(cudaMemcpyAsync(r.gpu_dst, expert_loader_.dma_staging_[k], expert_loader_.expert_block_size_,
                                                        cudaMemcpyHostToDevice, expert_streams_[k]));
                        } else {
                            CUDA_CHECK(cudaMemcpyAsync(r.gpu_dst, host_ptr, expert_loader_.expert_block_size_,
                                                        cudaMemcpyHostToDevice, expert_streams_[k]));
                        }
                    }
                    return true;
                });
            }

            auto _tp2 = std::chrono::high_resolution_clock::now();
            t_io_submit_us += std::chrono::duration<double, std::micro>(_tp2 - _tp1).count();

            // Phase 3: Wait for ALL I/O, then batch compute with fused kernel
            // This eliminates ~30 CUDA API calls per layer (cudaEventRecord + cudaStreamWaitEvent
            // + 2-4 kernel launches per expert) by using a single fused kernel call.
            // The lost compute-I/O overlap (~6ms) is outweighed by saved API overhead (~8ms).
            for (int k = 0; k < top_k; k++) {
                if (reqs[k].l1_hit) {
                    active_expert_ptrs_host_[k] = batch_ptrs[k];
                    continue;
                }
                // Wait for this expert's I/O future
                if (!io_futures[k].get()) {
                    active_expert_ptrs_host_[k] = nullptr;
                    continue;
                }
                active_expert_ptrs_host_[k] = batch_ptrs[k];
            }

            // Make main_stream_ wait for ALL expert DMA streams to complete
            for (int k = 0; k < top_k; k++) {
                if (reqs[k].l1_hit || !active_expert_ptrs_host_[k]) continue;
                CUDA_CHECK(cudaEventRecord(expert_events_[k], expert_streams_[k]));
                CUDA_CHECK(cudaStreamWaitEvent(main_stream_, expert_events_[k], 0));
            }

            // Copy expert pointers to GPU for fused kernel
            CUDA_CHECK(cudaMemcpyAsync(buf_active_expert_ptrs_.data, active_expert_ptrs_host_,
                                       top_k * sizeof(void*), cudaMemcpyHostToDevice, main_stream_));

            // Launch fused kernel: processes all 6 experts in 2 kernel calls instead of 12-24
            auto& w1_info = expert_parts_["w1.weight"];
            auto& w3_info = expert_parts_["w3.weight"];
            auto& w2_info = expert_parts_["w2.weight"];
            if (cfg_.expert_dtype == "iq2_xxs") {
                gemv_iq2_xxs_moe_swiglu_fused_cuda(
                    buf_gate_.bf16(), buf_hidden_.bf16(),
                    (const void* const*)buf_active_expert_ptrs_.data,
                    w1_info.offset_in_block, w3_info.offset_in_block,
                    moe_inter, dim, cfg_.swiglu_limit,
                    nullptr, nullptr, 0, 0, main_stream_);

                gemv_q2_k_moe_cuda(
                    buf_down_.bf16(), buf_gate_.bf16(),
                    (const void* const*)buf_active_expert_ptrs_.data,
                    w2_info.offset_in_block,
                    dim, moe_inter,
                    nullptr, nullptr, 0, 0, main_stream_);
            } else {
                // Fallback: per-expert compute for non-iq2_xxs dtypes
                for (int k = 0; k < top_k; k++) {
                    if (active_expert_ptrs_host_[k]) {
                        execute_expert_swiglu(active_expert_ptrs_host_[k], 1.0f, k);
                    }
                }
            }

            auto _tp3 = std::chrono::high_resolution_clock::now();
            t_io_wait_us += std::chrono::duration<double, std::micro>(_tp3 - _tp2).count();

            // Log per-token stats at last MoE layer
            if (layer_id == cfg_.num_hidden_layers - 1) {
                t_token_count++;
                if (t_token_count % 10 == 1) {
                    LOG_INFO("[MoE Perf] Token #%lld: cache_resolve=%.1fms io_submit=%.1fms io_wait+compute=%.1fms | L1=%d L2=%d SSD=%d",
                             t_token_count, t_cache_resolve_us/1000.0, t_io_submit_us/1000.0,
                             t_io_wait_us/1000.0, t_l1_hits, t_l2_hits, t_ssd_reads);
                }
            }
        }

        // 5. Shared expert (only for all-resident path; offloading path launched it above)
        if (expert_loader_.all_resident(cfg_.num_hidden_layers)) {
            CUDA_CHECK(cudaStreamWaitEvent(side_stream_, main_event_, 0));
            __nv_bfloat16* shared_gate = buf_gate_.bf16() + top_k * moe_inter;
            __nv_bfloat16* shared_up   = buf_up_.bf16()   + top_k * moe_inter;
            __nv_bfloat16* shared_down = buf_down_.bf16() + top_k * dim;

            if (lw.shared_w1_w.dtype == "int4") {
                gemv_int4_cuda(shared_gate, buf_hidden_.bf16(), (const uint8_t*)lw.shared_w1_w.data, lw.shared_w1_s.bf16(), moe_inter, dim, side_stream_);
                gemv_int4_cuda(shared_up, buf_hidden_.bf16(), (const uint8_t*)lw.shared_w3_w.data, lw.shared_w3_s.bf16(), moe_inter, dim, side_stream_);
                silu_mul_cuda(shared_gate, shared_gate, shared_up,
                              moe_inter, cfg_.swiglu_limit, side_stream_);
                gemv_int4_cuda(shared_down, shared_gate, (const uint8_t*)lw.shared_w2_w.data, lw.shared_w2_s.bf16(), dim, moe_inter, side_stream_);
            } else {
                gemm_fp8_dequant(shared_gate, 1, moe_inter, dim,
                                 buf_hidden_.bf16(),
                                 lw.shared_w1_w.u8(), lw.shared_w1_s.u8(), 128, side_stream_);
                gemm_fp8_dequant(shared_up, 1, moe_inter, dim,
                                 buf_hidden_.bf16(),
                                 lw.shared_w3_w.u8(), lw.shared_w3_s.u8(), 128, side_stream_);
                silu_mul_cuda(shared_gate, shared_gate, shared_up,
                              moe_inter, cfg_.swiglu_limit, side_stream_);
                gemm_fp8_dequant(shared_down, 1, dim, moe_inter,
                                 shared_gate,
                                 lw.shared_w2_w.u8(), lw.shared_w2_s.u8(), 128, side_stream_);
            }
            CUDA_CHECK(cudaEventRecord(side_event_, side_stream_));
        }

        // 6. Launch 6 routed experts (all-resident path uses fused kernel)
        auto& w1_info = expert_parts_["w1.weight"];
        auto& w3_info = expert_parts_["w3.weight"];
        auto& w2_info = expert_parts_["w2.weight"];

        if (expert_loader_.all_resident(cfg_.num_hidden_layers)) {
            if (cfg_.expert_dtype == "mixed_nvfp4_iq2") {
                gemv_mixed_moe_swiglu_fused_batch_cuda(
                    buf_gate_.bf16(), buf_hidden_.bf16(),
                    w1_info.offset_in_block, w3_info.offset_in_block,
                    moe_inter, dim, cfg_.swiglu_limit,
                    buf_topk_idx_.i32(), flat_ptrs,
                    expert_loader_.expert_type_map_gpu(),
                    layer_id, n_experts, 1, main_stream_);

                gemv_mixed_moe_down_batch_cuda(
                    buf_down_.bf16(), buf_gate_.bf16(),
                    buf_topk_idx_.i32(), flat_ptrs,
                    expert_loader_.expert_type_map_gpu(),
                    layer_id, n_experts,
                    w2_info.offset_in_block,
                    dim, moe_inter, 1, main_stream_);
            } else if (cfg_.expert_dtype == "iq2_xxs") {
                gemv_iq2_xxs_moe_swiglu_fused_cuda(
                    buf_gate_.bf16(), buf_hidden_.bf16(),
                    (const void* const*)buf_active_expert_ptrs_.data,
                    w1_info.offset_in_block, w3_info.offset_in_block,
                    moe_inter, dim, cfg_.swiglu_limit,
                    buf_topk_idx_.i32(), flat_ptrs, layer_id, n_experts, main_stream_);

                gemv_q2_k_moe_cuda(
                    buf_down_.bf16(), buf_gate_.bf16(),
                    (const void* const*)buf_active_expert_ptrs_.data,
                    w2_info.offset_in_block,
                    dim, moe_inter,
                    buf_topk_idx_.i32(), flat_ptrs, layer_id, n_experts, main_stream_);
            } else {
                for (int k = 0; k < top_k; k++) {
                    void* ptr = active_expert_ptrs_host_[k];
                    if (ptr) {
                        execute_expert_swiglu(ptr, 1.0f, k);
                    }
                }
            }
        }
        // Note: offloading path already computed experts individually above

        // Wait for shared expert on side_stream_ before accumulating
        CUDA_CHECK(cudaStreamWaitEvent(main_stream_, side_event_, 0));

        // 7. Fused 6-way dynamic accumulation + shared expert directly into buf_hidden_
        __nv_bfloat16* shared_down_ptr = buf_down_.bf16() + top_k * dim;
        fused_moe_accum_dynamic_cuda(buf_hidden_.bf16(), buf_down_.bf16(),
                                     buf_topk_vals_.f32(), shared_down_ptr, dim, main_stream_);
    }

    // ── Execute a single expert SwiGLU ──────────────────────────────────────

    void execute_expert_swiglu(void* expert_block, float routing_weight, int k, const __nv_bfloat16* in_vec = nullptr, bool is_cold_iq2 = false) {
        int dim = cfg_.hidden_size > 0 ? cfg_.hidden_size : DEEPSEEK_V41_HIDDEN_DIM;
        int moe_inter = cfg_.moe_intermediate_size > 0 ? cfg_.moe_intermediate_size : DEEPSEEK_V41_MOE_INTERMEDIATE;
        const __nv_bfloat16* in_hidden = in_vec ? in_vec : buf_hidden_.bf16();

        uint8_t* block = (uint8_t*)expert_block;
        auto& w1_info = expert_parts_["w1.weight"];
        auto& w3_info = expert_parts_["w3.weight"];
        auto& w2_info = expert_parts_["w2.weight"];

        uint8_t* w1_data = block + w1_info.offset_in_block;
        uint8_t* w3_data = block + w3_info.offset_in_block;
        uint8_t* w2_data = block + w2_info.offset_in_block;

        cudaStream_t stream = main_stream_;
        __nv_bfloat16* my_gate = buf_gate_.bf16() + k * moe_inter;
        __nv_bfloat16* my_up   = buf_up_.bf16()   + k * moe_inter;
        __nv_bfloat16* my_down = buf_down_.bf16() + k * dim;

        if (cfg_.expert_dtype == "iq2_xxs" || is_cold_iq2) {
            if (is_cold_iq2) {
                size_t w1_sz = (size_t)moe_inter * (dim / 256) * sizeof(block_iq2_xxs);
                size_t w3_sz = (size_t)moe_inter * (dim / 256) * sizeof(block_iq2_xxs);
                w1_data = block;
                w3_data = block + w1_sz;
                w2_data = block + w1_sz + w3_sz;
            }
            gemv_iq2_xxs_swiglu_fused_cuda(my_gate, in_hidden,
                                           (const block_iq2_xxs*)w1_data,
                                           (const block_iq2_xxs*)w3_data,
                                           moe_inter, dim, cfg_.swiglu_limit, stream);
            gemm_q2_k_dequant(my_down, 1, dim, moe_inter, my_gate, (const block_q2_K*)w2_data, stream);
        } else if (cfg_.expert_dtype == "int2") {
            auto& w1s_info = expert_parts_["w1.scale"];
            auto& w3s_info = expert_parts_["w3.scale"];
            auto& w2s_info = expert_parts_["w2.scale"];
            uint8_t* w1_scale = block + w1s_info.offset_in_block;
            uint8_t* w3_scale = block + w3s_info.offset_in_block;
            uint8_t* w2_scale = block + w2s_info.offset_in_block;
            int block_size = 256; 
            gemm_int2_dequant(my_gate, 1, moe_inter, dim, in_hidden, w1_data, (__nv_bfloat16*)w1_scale, block_size, stream);
            gemm_int2_dequant(my_up,   1, moe_inter, dim, in_hidden, w3_data, (__nv_bfloat16*)w3_scale, block_size, stream);
            silu_mul_cuda(my_gate, my_gate, my_up, moe_inter, cfg_.swiglu_limit, stream);
            gemm_int2_dequant(my_down, 1, dim, moe_inter, my_gate, w2_data, (__nv_bfloat16*)w2_scale, block_size, stream);
        } else if (cfg_.expert_dtype == "sparse_nvfp4") {
            auto& w1m_info = expert_parts_["w1.meta"];
            auto& w1s_info = expert_parts_["w1.scale"];
            auto& w3m_info = expert_parts_["w3.meta"];
            auto& w3s_info = expert_parts_["w3.scale"];
            auto& w2m_info = expert_parts_["w2.meta"];
            auto& w2s_info = expert_parts_["w2.scale"];

            uint8_t* w1_meta  = block + w1m_info.offset_in_block;
            uint8_t* w1_scale = block + w1s_info.offset_in_block;
            uint8_t* w3_meta  = block + w3m_info.offset_in_block;
            uint8_t* w3_scale = block + w3s_info.offset_in_block;
            uint8_t* w2_meta  = block + w2m_info.offset_in_block;
            uint8_t* w2_scale = block + w2s_info.offset_in_block;

            gemv_sparse_nvfp4_swiglu_fused_cuda(
                my_gate, in_hidden,
                w1_data, w1_meta, w1_scale,
                w3_data, w3_meta, w3_scale,
                moe_inter, dim, cfg_.swiglu_limit, stream);

            gemv_sparse_nvfp4_cuda(
                my_down, my_gate,
                w2_data, w2_meta, w2_scale,
                dim, moe_inter, stream);
        } else {
            auto& w1s_info = expert_parts_["w1.scale"];
            auto& w3s_info = expert_parts_["w3.scale"];
            auto& w2s_info = expert_parts_["w2.scale"];
            uint8_t* w1_scale = block + w1s_info.offset_in_block;
            uint8_t* w3_scale = block + w3s_info.offset_in_block;
            uint8_t* w2_scale = block + w2s_info.offset_in_block;
            int w1_scale_cols = w1s_info.shape.size() > 1 ? w1s_info.shape[1] : 1;
            int w2_scale_cols = w2s_info.shape.size() > 1 ? w2s_info.shape[1] : 1;
            gemm_fp4_dequant(my_gate, 1, moe_inter, dim, in_hidden, w1_data, w1_scale, w1_scale_cols, stream);
            gemm_fp4_dequant(my_up,   1, moe_inter, dim, in_hidden, w3_data, w3_scale, w1_scale_cols, stream);
            silu_mul_cuda(my_gate, my_gate, my_up, moe_inter, cfg_.swiglu_limit, stream);
            gemm_fp4_dequant(my_down, 1, dim, moe_inter, my_gate, w2_data, w2_scale, w2_scale_cols, stream);
        }
    }

    void compute_logits() {
        int dim = cfg_.hidden_size;
        int vocab = cfg_.vocab_size;

        if (head_weight_.dtype == "nvfp4") {
            gemv_nvfp4_f32_cuda(buf_logits_.f32(), buf_hidden_.bf16(),
                                (const uint8_t*)head_weight_.data, (const uint8_t*)head_weight_scale_.data,
                                head_weight_scale_2_, vocab, dim, main_stream_);
        } else if (head_weight_.dtype == "fp4") {
            if (is_blackwell_tc_active_) {
                quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                               buf_hidden_.bf16(), 1, dim, main_stream_);
                gemm_fp4_f32_blackwell_tensorcore_cuda(buf_logits_.f32(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                       (const uint8_t*)head_weight_.data, (const uint8_t*)head_weight_scale_.data,
                                                       vocab, dim, 1, main_stream_);
            } else {
                gemv_fp4_f32_cuda(buf_logits_.f32(), buf_hidden_.bf16(),
                                   (const uint8_t*)head_weight_.data, (const uint8_t*)head_weight_scale_.data,
                                   vocab, dim, main_stream_);
            }
        } else if (head_weight_.dtype == "int4") {
            gemv_int4_f32_cuda(buf_logits_.f32(), buf_hidden_.bf16(),
                               (const uint8_t*)head_weight_.data, head_weight_scale_.bf16(),
                               vocab, dim, main_stream_);
        } else if (head_weight_.dtype == "q8_0" || head_weight_.dtype == "Q8_0" || head_weight_.dtype == "U8") {
            gemv_q8_0_f32_cuda(buf_logits_.f32(), buf_hidden_.bf16(),
                               (const block_q8_0*)head_weight_.data,
                               vocab, dim, main_stream_);
        } else {
            // Fast CUDA graph compatible matrix-vector multiplication directly to float32 logits
            gemv_bf16_cuda(buf_logits_.f32(), head_weight_.bf16(), buf_hidden_.bf16(), vocab, dim, main_stream_);
        }
        argmax_f32_cuda(buf_argmax_out_.i32(), buf_logits_.f32(), vocab, main_stream_);
        argmax_cache_valid_ = true;
    }

    // ── Batched Prefill for DeepSeek V4 Flash MoE (All-Resident Mode) ────────
    void prefill_prompt_batched_deepseek(const std::vector<int>& prompt, int start_offset = 0) {
        if (prompt.empty()) return;
        int remaining_tokens = (int)prompt.size() - start_offset;
        if (remaining_tokens <= 0) return;
        if (remaining_tokens <= 1) {
            for (size_t i = (size_t)start_offset; i < prompt.size(); i++) {
                forward_token(prompt[i], (int)i);
            }
            CUDA_CHECK(cudaStreamSynchronize(main_stream_));
            return;
        }

        int total_tokens = (int)prompt.size();
        int chunk_size = prefill_chunk_size_;
        auto t_start = std::chrono::high_resolution_clock::now();

        int dim = cfg_.hidden_size;
        int hc = cfg_.hc_mult;
        int hc_dim = hc * dim;
        int mix_size = (2 + hc) * hc;
        int q_lora = cfg_.q_lora_rank;
        int n_heads = cfg_.num_attention_heads;
        int head_dim_val = cfg_.head_dim;
        int rope_dim = cfg_.qk_rope_head_dim;
        int o_lora = cfg_.o_lora_rank;
        int o_groups = cfg_.o_groups;
        int window = cfg_.sliding_window;
        int heads_per_group = n_heads / o_groups; // 64 / 8 = 8
        int hpg_dim = heads_per_group * head_dim_val; // 8 * 512 = 4096
        int moe_inter = cfg_.moe_intermediate_size;
        int n_experts = cfg_.n_routed_experts;
        int top_k = cfg_.num_experts_per_tok;
        float scale = 1.0f / sqrtf((float)head_dim_val);
        int max_combined = window + cfg_.max_compressed_entries;

        for (int start = start_offset; start < total_tokens; start += chunk_size) {
            int M = std::min(chunk_size, total_tokens - start);
            int position = start;
            bool is_last_chunk = (start + M == total_tokens);

            // 1. Copy tokens and positions to pinned host buffers, then async DMA to device
            for (int m = 0; m < M; m++) {
                h_prefill_tok_[m] = prompt[start + m];
                h_prefill_pos_[m] = position + m;
            }
            CUDA_CHECK(cudaMemcpyAsync(prefill_bufs_.buf_tokens.i32(), h_prefill_tok_,
                                       M * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
            CUDA_CHECK(cudaMemcpyAsync(prefill_bufs_.buf_pos.i32(), h_prefill_pos_,
                                       M * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));

            // 2. Embedding lookup and broadcast to HC copies: [M, dim] and [M, hc * dim]
            embedding_broadcast_batch_cuda(
                prefill_bufs_.buf_hidden.bf16(),
                prefill_bufs_.buf_hc_state.bf16(),
                embed_weight_.bf16(),
                prefill_bufs_.buf_tokens.i32(),
                M, dim, hc, main_stream_);

            // Override with vision tower embedding for image tokens
            if (active_visual_embeddings_) {
                for (int m = 0; m < M; m++) {
                    int pos = position + m;
                    if (pos >= visual_embed_start_pos_ && pos < visual_embed_start_pos_ + visual_embed_num_tokens_) {
                        int v_idx = pos - visual_embed_start_pos_;
                        const __nv_bfloat16* src_vis = active_visual_embeddings_ + (size_t)v_idx * dim;
                        CUDA_CHECK(cudaMemcpyAsync(prefill_bufs_.buf_hidden.bf16() + (size_t)m * dim,
                                                   src_vis,
                                                   dim * sizeof(__nv_bfloat16),
                                                   cudaMemcpyDeviceToDevice, main_stream_));
                        for (int h = 0; h < hc; h++) {
                            CUDA_CHECK(cudaMemcpyAsync(prefill_bufs_.buf_hc_state.bf16() + (size_t)m * hc_dim + (size_t)h * dim,
                                                       src_vis,
                                                       dim * sizeof(__nv_bfloat16),
                                                       cudaMemcpyDeviceToDevice, main_stream_));
                        }
                    }
                }
            }

            // 3. Process each layer
            for (int layer_id = 0; layer_id < cfg_.num_hidden_layers; layer_id++) {
                auto& lw = layers_[layer_id];

                // ── HC pre for attention + Attention norm (fused into 1 pass) ──
                gemv_hc_pre_norm_batch_cuda(prefill_bufs_.buf_hc_mixes.f32(),
                                            prefill_bufs_.buf_hc_state.bf16(),
                                            lw.hc_attn_fn.f32(),
                                            M, mix_size, hc_dim, cfg_.hc_eps, main_stream_);

                hc_split_sinkhorn_batch_cuda(prefill_bufs_.buf_hc_pre.f32(),
                                             prefill_bufs_.buf_hc_post.f32(),
                                             prefill_bufs_.buf_hc_comb.f32(),
                                             prefill_bufs_.buf_hc_mixes.f32(),
                                             lw.hc_attn_scale.f32(),
                                             lw.hc_attn_base.f32(),
                                             M, hc, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, main_stream_);

                hc_pre_weighted_add_norm_batch_cuda(prefill_bufs_.buf_hidden.bf16(),
                                                    prefill_bufs_.buf_hc_state.bf16(),
                                                    prefill_bufs_.buf_hc_pre.f32(),
                                                    lw.attn_norm_w.bf16(),
                                                    M, dim, hc, cfg_.rms_norm_eps, main_stream_);

                // ── Attention Q & KV Projections (batched via cuBLAS Tensor Cores) ──
                if (lw.wq_a_w.dtype == "int4") {
                    gemm_int4_dequant(prefill_bufs_.buf_lora.bf16(), M, q_lora, dim,
                                     prefill_bufs_.buf_hidden.bf16(),
                                     (const uint8_t*)lw.wq_a_w.data, lw.wq_a_s.bf16(), 32, main_stream_);
                } else {
                    gemm_fp8_dequant(prefill_bufs_.buf_lora.bf16(), M, q_lora, dim,
                                     prefill_bufs_.buf_hidden.bf16(),
                                     lw.wq_a_w.u8(), lw.wq_a_s.u8(), 128, main_stream_);
                }

                rms_norm_cuda_batched(prefill_bufs_.buf_lora.bf16(), prefill_bufs_.buf_lora.bf16(),
                                      lw.q_norm_w.bf16(), M, q_lora, cfg_.rms_norm_eps, main_stream_);

                if (lw.wq_b_w.dtype == "int4") {
                    gemm_int4_dequant(prefill_bufs_.buf_q.bf16(), M, n_heads * head_dim_val, q_lora,
                                     prefill_bufs_.buf_lora.bf16(),
                                     (const uint8_t*)lw.wq_b_w.data, lw.wq_b_s.bf16(), 32, main_stream_);
                } else {
                    gemm_fp8_dequant(prefill_bufs_.buf_q.bf16(), M, n_heads * head_dim_val, q_lora,
                                     prefill_bufs_.buf_lora.bf16(),
                                     lw.wq_b_w.u8(), lw.wq_b_s.u8(), 128, main_stream_);
                }

                if (lw.wkv_w.dtype == "int4") {
                    gemm_int4_dequant(prefill_bufs_.buf_kv.bf16(), M, head_dim_val, dim,
                                     prefill_bufs_.buf_hidden.bf16(),
                                     (const uint8_t*)lw.wkv_w.data, lw.wkv_s.bf16(), 32, main_stream_);
                } else {
                    gemm_fp8_dequant(prefill_bufs_.buf_kv.bf16(), M, head_dim_val, dim,
                                     prefill_bufs_.buf_hidden.bf16(),
                                     lw.wkv_w.u8(), lw.wkv_s.u8(), 128, main_stream_);
                }

                rms_norm_cuda_batched(prefill_bufs_.buf_kv.bf16(), prefill_bufs_.buf_kv.bf16(),
                                      lw.kv_norm_w.bf16(), M, head_dim_val, cfg_.rms_norm_eps, main_stream_);

                // ── Causal Attention loop across the M tokens ──
                bool is_compressed = (layer_id < (int)cfg_.compress_ratios.size() &&
                                      cfg_.compress_ratios[layer_id] > 0);
                float* layer_rope_freqs = is_compressed ? rope_freqs_compressed_.f32()
                                                        : rope_freqs_.f32();
                int ratio = cfg_.layer_compress_ratio(layer_id);

                for (int m = 0; m < M; m++) {
                    const int32_t* d_pos = prefill_bufs_.buf_pos.i32() + m;
                    __nv_bfloat16* tok_kv = prefill_bufs_.buf_kv.bf16() + (size_t)m * head_dim_val;
                    const __nv_bfloat16* tok_hidden = prefill_bufs_.buf_hidden.bf16() + (size_t)m * dim;
                    const __nv_bfloat16* tok_lora = prefill_bufs_.buf_lora.bf16() + (size_t)m * q_lora;
                    const __nv_bfloat16* tok_q = prefill_bufs_.buf_q.bf16() + (size_t)m * (n_heads * head_dim_val);
                    __nv_bfloat16* tok_attn_out = prefill_bufs_.buf_attn_out.bf16() + (size_t)m * (n_heads * head_dim_val);

                    // 1. Store KV into ring-buffer (rotated latent value vectors)
                    rope_device_pos_cuda(tok_kv, 1, head_dim_val, rope_dim,
                                         d_pos, layer_rope_freqs, false, main_stream_);
                    store_kv_device_pos_cuda(lw.kv_cache.bf16(), tok_kv,
                                             d_pos, window, head_dim_val, main_stream_);

                    // 2. KV Compressor step
                    if (ratio > 0) {
                        forward_compressor_step(layer_id, tok_hidden, d_pos);
                    }

                    // 3. Indexer (for CSA ratio=4 layers)
                    uint8_t* comp_mask_ptr = nullptr;
                    if (ratio == 4 && lw.indexer_wq_b_w.data) {
                        int idx_head_dim = 128;
                        int idx_heads = 64;
                        int idx_q_dim = idx_heads * idx_head_dim; // 8192

                        if (lw.indexer_wq_b_w.dtype == "int4") {
                            gemv_int4_cuda(buf_indexer_q_.bf16(), tok_lora,
                                           (const uint8_t*)lw.indexer_wq_b_w.data, lw.indexer_wq_b_s.bf16(),
                                           idx_q_dim, q_lora, main_stream_);
                        } else {
                            gemv_fp8_cuda(buf_indexer_q_.bf16(), tok_lora,
                                          lw.indexer_wq_b_w.u8(), lw.indexer_wq_b_s.u8(),
                                          idx_q_dim, q_lora, 128, main_stream_);
                        }

                        rope_device_pos_cuda(buf_indexer_q_.bf16(), idx_heads, idx_head_dim, rope_dim,
                                             d_pos, layer_rope_freqs, false, main_stream_);

                        gemv_bf16_cuda(buf_indexer_weights_f32_.f32(), lw.indexer_weights_proj.bf16(),
                                       tok_hidden, idx_heads, dim, main_stream_);

                        int max_comp_entries = cfg_.max_compressed_entries > 0 ? cfg_.max_compressed_entries : 2048;
                        indexer_score_and_mask_cuda(
                            buf_indexer_mask_.u8(),
                            buf_indexer_scores_.f32(),
                            lw.indexer_comp_kv_cache.bf16(),
                            buf_indexer_q_.bf16(),
                            buf_indexer_weights_f32_.f32(),
                            lw.d_comp_kv_count.i32(),
                            max_comp_entries,
                            512,
                            main_stream_);

                        comp_mask_ptr = buf_indexer_mask_.u8();
                    }

                    // 4. Flash-MLA Attention
                    mla_attention_fused_cuda(
                        tok_q, lw.kv_cache.bf16(), lw.comp_kv_cache.bf16(), lw.attn_sink.f32(),
                        tok_attn_out, d_pos, lw.d_comp_kv_count.i32(),
                        layer_rope_freqs, max_combined, head_dim_val, rope_dim, scale,
                        cfg_.rms_norm_eps, comp_mask_ptr, window, main_stream_);
                }

                // ── Output projection (wo_a and wo_b) ──
                if (lw.wo_a_w.dtype == "int4") {
                    gemv_int4_grouped_batch_cuda(prefill_bufs_.buf_lora.bf16(),
                                                 prefill_bufs_.buf_attn_out.bf16(),
                                                 (const uint8_t*)lw.wo_a_w.data, lw.wo_a_s.bf16(),
                                                 M, o_lora, hpg_dim, o_groups, main_stream_);
                } else {
                    gemv_fp8_grouped_batch_cuda(prefill_bufs_.buf_lora.bf16(),
                                                prefill_bufs_.buf_attn_out.bf16(),
                                                lw.wo_a_w.u8(), lw.wo_a_s.u8(),
                                                M, o_lora, hpg_dim, o_groups, 128, main_stream_);
                }

                if (lw.wo_b_w.dtype == "int4") {
                    gemm_int4_dequant(prefill_bufs_.buf_hidden.bf16(), M, dim, o_groups * o_lora,
                                     prefill_bufs_.buf_lora.bf16(),
                                     (const uint8_t*)lw.wo_b_w.data, lw.wo_b_s.bf16(), 32, main_stream_);
                } else {
                    gemm_fp8_dequant(prefill_bufs_.buf_hidden.bf16(), M, dim, o_groups * o_lora,
                                     prefill_bufs_.buf_lora.bf16(),
                                     lw.wo_b_w.u8(), lw.wo_b_s.u8(), 128, main_stream_);
                }

                // ── HC post for attention: reads buf_hc_state, writes to buf_hc_after_attn ──
                hc_post_update_batch_cuda(prefill_bufs_.buf_hc_after_attn.bf16(),
                                          prefill_bufs_.buf_hidden.bf16(),
                                          prefill_bufs_.buf_hc_state.bf16(),
                                          prefill_bufs_.buf_hc_post.f32(),
                                          prefill_bufs_.buf_hc_comb.f32(),
                                          M, dim, hc, main_stream_);

                // ── HC pre for FFN + FFN norm (reads buf_hc_after_attn, writes to buf_hidden) ──
                gemv_hc_pre_norm_batch_cuda(prefill_bufs_.buf_hc_mixes.f32(),
                                            prefill_bufs_.buf_hc_after_attn.bf16(),
                                            lw.hc_ffn_fn.f32(),
                                            M, mix_size, hc_dim, cfg_.hc_eps, main_stream_);

                hc_split_sinkhorn_batch_cuda(prefill_bufs_.buf_hc_pre.f32(),
                                             prefill_bufs_.buf_hc_post.f32(),
                                             prefill_bufs_.buf_hc_comb.f32(),
                                             prefill_bufs_.buf_hc_mixes.f32(),
                                             lw.hc_ffn_scale.f32(),
                                             lw.hc_ffn_base.f32(),
                                             M, hc, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, main_stream_);

                hc_pre_weighted_add_norm_batch_cuda(prefill_bufs_.buf_hidden.bf16(),
                                                    prefill_bufs_.buf_hc_after_attn.bf16(),
                                                    prefill_bufs_.buf_hc_pre.f32(),
                                                    lw.ffn_norm_w.bf16(),
                                                    M, dim, hc, cfg_.rms_norm_eps, main_stream_);

                // ── MoE Routing across M tokens ──
                if (layer_id < cfg_.n_hash_layers) {
                    moe_route_hash_device_id_batch_cuda(
                        prefill_bufs_.buf_topk_idx.i32(),
                        prefill_bufs_.buf_topk_vals.f32(),
                        lw.tid2eid.i64(),
                        prefill_bufs_.buf_tokens.i32(),
                        M, top_k, cfg_.routed_scaling_factor,
                        main_stream_);
                } else {
                    gemm_bf16(prefill_bufs_.buf_scores_bf16.bf16(), M, n_experts, dim,
                              prefill_bufs_.buf_hidden.bf16(), lw.gate_w.bf16());

                    moe_route_top6_from_bf16_batch_cuda(
                        prefill_bufs_.buf_topk_idx.i32(),
                        prefill_bufs_.buf_topk_vals.f32(),
                        prefill_bufs_.buf_scores_bf16.bf16(),
                        lw.gate_bias.f32(),
                        M, n_experts, top_k, cfg_.routed_scaling_factor,
                        main_stream_);
                }

                // ── Shared Expert (batched Tensor Cores) ──
                if (lw.shared_w1_w.dtype == "int4") {
                    gemm_int4_dequant(prefill_bufs_.buf_shared_gate.bf16(), M, moe_inter, dim,
                                     prefill_bufs_.buf_hidden.bf16(),
                                     (const uint8_t*)lw.shared_w1_w.data, lw.shared_w1_s.bf16(), 32, main_stream_);

                    gemm_int4_dequant(prefill_bufs_.buf_shared_up.bf16(), M, moe_inter, dim,
                                     prefill_bufs_.buf_hidden.bf16(),
                                     (const uint8_t*)lw.shared_w3_w.data, lw.shared_w3_s.bf16(), 32, main_stream_);

                    silu_mul_cuda(prefill_bufs_.buf_shared_gate.bf16(),
                                  prefill_bufs_.buf_shared_gate.bf16(),
                                  prefill_bufs_.buf_shared_up.bf16(),
                                  M * moe_inter, cfg_.swiglu_limit, main_stream_);

                    gemm_int4_dequant(prefill_bufs_.buf_shared_down.bf16(), M, dim, moe_inter,
                                     prefill_bufs_.buf_shared_gate.bf16(),
                                     (const uint8_t*)lw.shared_w2_w.data, lw.shared_w2_s.bf16(), 32, main_stream_);
                } else {
                    gemm_fp8_dequant(prefill_bufs_.buf_shared_gate.bf16(), M, moe_inter, dim,
                                     prefill_bufs_.buf_hidden.bf16(),
                                     lw.shared_w1_w.u8(), lw.shared_w1_s.u8(), 128, main_stream_);

                    gemm_fp8_dequant(prefill_bufs_.buf_shared_up.bf16(), M, moe_inter, dim,
                                     prefill_bufs_.buf_hidden.bf16(),
                                     lw.shared_w3_w.u8(), lw.shared_w3_s.u8(), 128, main_stream_);

                    silu_mul_cuda(prefill_bufs_.buf_shared_gate.bf16(),
                                  prefill_bufs_.buf_shared_gate.bf16(),
                                  prefill_bufs_.buf_shared_up.bf16(),
                                  M * moe_inter, cfg_.swiglu_limit, main_stream_);

                    gemm_fp8_dequant(prefill_bufs_.buf_shared_down.bf16(), M, dim, moe_inter,
                                     prefill_bufs_.buf_shared_gate.bf16(),
                                     lw.shared_w2_w.u8(), lw.shared_w2_s.u8(), 128, main_stream_);
                }

                // ── Routed Experts (Resident MoE) ──
                const void* const* flat_ptrs = expert_loader_.flat_vram_ptrs_gpu();
                auto& w1_info = expert_parts_["w1.weight"];
                auto& w3_info = expert_parts_["w3.weight"];
                auto& w2_info = expert_parts_["w2.weight"];

                if (cfg_.expert_dtype == "mixed_nvfp4_iq2") {
                    gemv_mixed_moe_swiglu_fused_batch_cuda(
                        prefill_bufs_.buf_gate.bf16(),
                        prefill_bufs_.buf_hidden.bf16(),
                        w1_info.offset_in_block, w3_info.offset_in_block,
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
                        w2_info.offset_in_block,
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

                // ── Fused Accumulation (routed experts + shared expert) ──
                fused_moe_accum_dynamic_batch_cuda(
                    prefill_bufs_.buf_hidden.bf16(),
                    prefill_bufs_.buf_down.bf16(),
                    prefill_bufs_.buf_topk_vals.f32(),
                    prefill_bufs_.buf_shared_down.bf16(),
                    dim, M, main_stream_);

                // ── HC post for FFN: reads buf_hc_after_attn, writes to buf_hc_state ──
                hc_post_update_batch_cuda(prefill_bufs_.buf_hc_state.bf16(),
                                          prefill_bufs_.buf_hidden.bf16(),
                                          prefill_bufs_.buf_hc_after_attn.bf16(),
                                          prefill_bufs_.buf_hc_post.f32(),
                                          prefill_bufs_.buf_hc_comb.f32(),
                                          M, dim, hc, main_stream_);
            } // end layer loop

            // 4. If this is the last chunk, prepare logits for the last token
            if (is_last_chunk) {
                CUDA_CHECK(cudaMemcpyAsync(buf_hc_state_.bf16(),
                                           prefill_bufs_.buf_hc_state.bf16() + (size_t)(M - 1) * (hc * dim),
                                           hc * dim * sizeof(__nv_bfloat16),
                                           cudaMemcpyDeviceToDevice, main_stream_));
                if (!cfg_.is_deepseek_v41()) {
                    hc_head_reduce();
                }
                rms_norm_cuda(buf_hidden_.bf16(), buf_hidden_.bf16(),
                              norm_weight_.bf16(), dim, cfg_.rms_norm_eps, main_stream_);
                compute_logits();
            }
        } // end chunk loop

        CUDA_CHECK(cudaStreamSynchronize(main_stream_));
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();
        size_t evaluated_tokens = (total_tokens > start_offset) ? (size_t)(total_tokens - start_offset) : 0;
        double tok_per_sec = (elapsed_sec > 0.0 && evaluated_tokens > 0) ? ((double)evaluated_tokens / elapsed_sec) : 0.0;
        LOG_INFO("[PREFILL STATS] %zu prompt tokens prefilled in %.3fs -> %.2f tok/s (batched chunk_size=%d, offset=%d)",
                 evaluated_tokens, elapsed_sec, tok_per_sec, chunk_size, start_offset);
    }
