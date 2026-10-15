#pragma once
#include "common.hpp"
#include "logging.hpp"
#include "activations.cuh"

using json = nlohmann::json;

// ════════════════════════════════════════════════════════════════════════════════
//  Prompt-Lookup Drafting (PLD) Speculative Engine
// ════════════════════════════════════════════════════════════════════════════════

class PromptLookupDrafter {
public:
    static std::vector<int> draft(const std::vector<int>& history, int max_draft = 4, int max_ngram = 3, int min_ngram = 2) {
        int hist_len = (int)history.size();
        if (hist_len <= max_ngram + 1) return {};

        for (int n = max_ngram; n >= min_ngram; n--) {
            if (hist_len <= n + 1) continue;

            const int* match_pattern = &history[hist_len - n];

            // Search backward in history for prior match
            for (int i = hist_len - n - 2; i >= 0; i--) {
                bool matched = true;
                for (int j = 0; j < n; j++) {
                    if (history[i + j] != match_pattern[j]) {
                        matched = false;
                        break;
                    }
                }
                if (matched) {
                    // Match found! Candidate tokens start right after the match
                    int start_cand = i + n;
                    int avail = hist_len - n - start_cand;
                    int cand_len = std::min(max_draft, avail);
                    if (cand_len > 0) {
                        std::vector<int> candidates;
                        candidates.reserve(cand_len);
                        for (int k = 0; k < cand_len; k++) {
                            candidates.push_back(history[start_cand + k]);
                        }
                        return candidates;
                    }
                }
            }
        }
        return {};
    }
};

// ════════════════════════════════════════════════════════════════════════════════
//  Qwen 3.8 2B Neural Speculative Decoding Draft Engine
// ════════════════════════════════════════════════════════════════════════════════

struct QwenDraftLayer {
    bool is_linear_attn = true;
    // Linear Attention (DeltaNet)
    GPUTensor w_in_qkv, w_in_qkv_scale; // [6144, 2048]
    GPUTensor w_in_z, w_in_z_scale;     // [2048, 2048]
    GPUTensor w_in_a, w_in_a_scale;     // [16, 2048]
    GPUTensor w_in_b, w_in_b_scale;     // [16, 2048]
    GPUTensor conv1d_w;                 // [6144, 1, 4]
    GPUTensor A_log, dt_bias;           // [16]
    GPUTensor linear_norm_w;            // [128]
    GPUTensor linear_out_proj, linear_out_proj_scale; // [2048, 2048]
    GPUTensor ssm_state, ssm_state_chk;                // [16, 128, 128]
    GPUTensor conv_state, conv_state_chk;              // [6144, 4]

    // Full GQA Attention
    GPUTensor w_q, w_q_scale;           // [4096, 2048]
    GPUTensor w_k, w_k_scale;           // [512, 2048]
    GPUTensor w_v, w_v_scale;           // [512, 2048]
    GPUTensor q_norm_w, k_norm_w;       // [256]
    GPUTensor w_o, w_o_scale;           // [2048, 2048]
    GPUTensor k_cache, v_cache;         // [2048, 2, 256]

    // Shared layer norms & MLP
    GPUTensor input_norm_w, post_attn_norm_w;
    GPUTensor w_gate, w_gate_scale;     // [6144, 2048]
    GPUTensor w_up, w_up_scale;         // [6144, 2048]
    GPUTensor w_down, w_down_scale;     // [2048, 6144]
};

class QwenDraftEngine2B {
public:
    bool loaded_ = false;
    int vocab_size_ = 248320;
    int hidden_size_ = 2048;
    int num_layers_ = 24;
    int intermediate_size_ = 6144;
    int num_heads_ = 8;
    int num_kv_heads_ = 2;
    int head_dim_ = 256;
    int linear_num_heads_ = 16;
    int linear_head_dim_ = 128;

    GPUTensor embed_w_;
    GPUTensor embed_w_int4_;
    GPUTensor embed_w_scale_;
    GPUTensor final_norm_w_;
    std::vector<QwenDraftLayer> layers_;

    // Working activations
    GPUTensor buf_input_token_;
    GPUTensor buf_hidden_;
    GPUTensor buf_hidden2_;
    GPUTensor buf_qkv_;
    GPUTensor buf_z_;
    GPUTensor buf_a_;
    GPUTensor buf_b_;
    GPUTensor buf_q_;
    GPUTensor buf_k_;
    GPUTensor buf_v_;
    GPUTensor buf_attn_out_;
    GPUTensor buf_gate_;
    GPUTensor buf_up_;
    GPUTensor buf_logits_;
    GPUTensor buf_argmax_out_;

    GPUTensor draft_ssm_pool_;
    GPUTensor draft_ssm_slots_[8];
    GPUTensor draft_conv_pool_;
    GPUTensor draft_conv_slots_[8];
    int32_t* h_draft_tok_ = nullptr;
    int32_t* h_draft_pos_ = nullptr;

    bool load(const std::string& draft_manifest_path, cudaStream_t stream = nullptr) {
        std::ifstream mf(draft_manifest_path);
        if (!mf.is_open()) return false;
        json manifest;
        try { mf >> manifest; } catch (...) { return false; }
        mf.close();

        std::string manifest_dir = "";
        size_t last_slash = draft_manifest_path.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            manifest_dir = draft_manifest_path.substr(0, last_slash + 1);
        }

        std::string bin_name = manifest.value("dense_bin", "qwen3_8_draft_2b.bin");
        std::string bin_path = manifest_dir + bin_name;

        std::ifstream bf(bin_path, std::ios::binary);
        if (!bf.is_open()) {
            LOG_WARN("Cannot open draft binary: %s", bin_path.c_str());
            return false;
        }

        auto& dense_tensors = manifest["dense_tensors"];

        auto load_t = [&](GPUTensor& t, const std::string& name) -> bool {
            if (!dense_tensors.contains(name)) return false;
            auto& meta = dense_tensors[name];
            size_t offset = meta["offset"].get<size_t>();
            size_t nbytes = meta["nbytes"].get<size_t>();
            t.alloc(nbytes);
            std::vector<uint8_t> host_buf(nbytes);
            bf.seekg(offset);
            bf.read((char*)host_buf.data(), nbytes);
            CUDA_CHECK(cudaMemcpy(t.data, host_buf.data(), nbytes, cudaMemcpyHostToDevice));
            return true;
        };

        auto load_quant_t = [&](GPUTensor& w, GPUTensor& s, const std::string& name) -> bool {
            if (!dense_tensors.contains(name)) return false;
            auto& meta = dense_tensors[name];
            size_t offset = meta["offset"].get<size_t>();
            size_t nbytes = meta["nbytes"].get<size_t>();
            w.alloc(nbytes);
            w.dtype = meta.value("dtype", "int4");
            std::vector<uint8_t> host_buf(nbytes);
            bf.seekg(offset);
            bf.read((char*)host_buf.data(), nbytes);
            CUDA_CHECK(cudaMemcpy(w.data, host_buf.data(), nbytes, cudaMemcpyHostToDevice));

            if (meta.contains("scale_offset")) {
                size_t s_off = meta["scale_offset"].get<size_t>();
                size_t s_nb = meta["scale_nbytes"].get<size_t>();
                s.alloc(s_nb);
                std::vector<uint8_t> s_buf(s_nb);
                bf.seekg(s_off);
                bf.read((char*)s_buf.data(), s_nb);
                CUDA_CHECK(cudaMemcpy(s.data, s_buf.data(), s_nb, cudaMemcpyHostToDevice));
            }
            return true;
        };

        if (!load_quant_t(embed_w_, embed_w_scale_, "model.language_model.embed_tokens.weight")) {
            load_t(embed_w_, "model.language_model.embed_tokens.weight");
        }
        load_t(final_norm_w_, "model.language_model.norm.weight");

        layers_.resize(num_layers_);
        int n_linear = 0;
        for (int l = 0; l < num_layers_; l++) {
            std::string prefix = "model.language_model.layers." + std::to_string(l);
            auto& lw = layers_[l];
            load_t(lw.input_norm_w, prefix + ".input_layernorm.weight");
            load_t(lw.post_attn_norm_w, prefix + ".post_attention_layernorm.weight");
            load_quant_t(lw.w_gate, lw.w_gate_scale, prefix + ".mlp.gate_proj.weight");
            load_quant_t(lw.w_up, lw.w_up_scale, prefix + ".mlp.up_proj.weight");
            load_quant_t(lw.w_down, lw.w_down_scale, prefix + ".mlp.down_proj.weight");

            if (dense_tensors.contains(prefix + ".linear_attn.in_proj_qkv.weight")) {
                lw.is_linear_attn = true;
                n_linear++;
                load_quant_t(lw.w_in_qkv, lw.w_in_qkv_scale, prefix + ".linear_attn.in_proj_qkv.weight");
                load_quant_t(lw.w_in_z, lw.w_in_z_scale, prefix + ".linear_attn.in_proj_z.weight");
                load_quant_t(lw.w_in_a, lw.w_in_a_scale, prefix + ".linear_attn.in_proj_a.weight");
                load_quant_t(lw.w_in_b, lw.w_in_b_scale, prefix + ".linear_attn.in_proj_b.weight");
                load_t(lw.conv1d_w, prefix + ".linear_attn.conv1d.weight");
                load_t(lw.A_log, prefix + ".linear_attn.A_log");
                load_t(lw.dt_bias, prefix + ".linear_attn.dt_bias");
                load_t(lw.linear_norm_w, prefix + ".linear_attn.norm.weight");
                load_quant_t(lw.linear_out_proj, lw.linear_out_proj_scale, prefix + ".linear_attn.out_proj.weight");
            } else {
                lw.is_linear_attn = false;
                load_quant_t(lw.w_q, lw.w_q_scale, prefix + ".self_attn.q_proj.weight");
                load_quant_t(lw.w_k, lw.w_k_scale, prefix + ".self_attn.k_proj.weight");
                load_quant_t(lw.w_v, lw.w_v_scale, prefix + ".self_attn.v_proj.weight");
                load_t(lw.q_norm_w, prefix + ".self_attn.q_norm.weight");
                load_t(lw.k_norm_w, prefix + ".self_attn.k_norm.weight");
                load_quant_t(lw.w_o, lw.w_o_scale, prefix + ".self_attn.o_proj.weight");

                lw.k_cache.alloc(2048 * num_kv_heads_ * head_dim_ * sizeof(__nv_bfloat16));
                lw.v_cache.alloc(2048 * num_kv_heads_ * head_dim_ * sizeof(__nv_bfloat16));
                CUDA_CHECK(cudaMemset(lw.k_cache.data, 0, lw.k_cache.size_bytes));
                CUDA_CHECK(cudaMemset(lw.v_cache.data, 0, lw.v_cache.size_bytes));
            }
        }

        if (n_linear > 0) {
            size_t ssm_bytes_per_layer = 16 * 128 * 128 * sizeof(__nv_bfloat16);
            size_t conv_bytes_per_layer = 6144 * 4 * sizeof(__nv_bfloat16);
            draft_ssm_pool_.alloc(n_linear * ssm_bytes_per_layer);
            draft_conv_pool_.alloc(n_linear * conv_bytes_per_layer);
            CUDA_CHECK(cudaMemset(draft_ssm_pool_.data, 0, draft_ssm_pool_.size_bytes));
            CUDA_CHECK(cudaMemset(draft_conv_pool_.data, 0, draft_conv_pool_.size_bytes));

            for (int s = 0; s < 8; s++) {
                draft_ssm_slots_[s].alloc(n_linear * ssm_bytes_per_layer);
                draft_conv_slots_[s].alloc(n_linear * conv_bytes_per_layer);
                CUDA_CHECK(cudaMemset(draft_ssm_slots_[s].data, 0, draft_ssm_slots_[s].size_bytes));
                CUDA_CHECK(cudaMemset(draft_conv_slots_[s].data, 0, draft_conv_slots_[s].size_bytes));
            }

            int lin_idx = 0;
            for (int l = 0; l < num_layers_; l++) {
                if (layers_[l].is_linear_attn) {
                    layers_[l].ssm_state.data = (uint8_t*)draft_ssm_pool_.data + lin_idx * ssm_bytes_per_layer;
                    layers_[l].ssm_state.size_bytes = ssm_bytes_per_layer;

                    layers_[l].conv_state.data = (uint8_t*)draft_conv_pool_.data + lin_idx * conv_bytes_per_layer;
                    layers_[l].conv_state.size_bytes = conv_bytes_per_layer;
                    lin_idx++;
                }
            }
        }

        buf_input_token_.alloc(sizeof(int32_t));
        buf_hidden_.alloc(hidden_size_ * sizeof(__nv_bfloat16));
        buf_hidden2_.alloc(hidden_size_ * sizeof(__nv_bfloat16));
        buf_qkv_.alloc(6144 * sizeof(__nv_bfloat16));
        buf_z_.alloc(2048 * sizeof(__nv_bfloat16));
        buf_a_.alloc(16 * sizeof(__nv_bfloat16));
        buf_b_.alloc(16 * sizeof(__nv_bfloat16));
        buf_q_.alloc(2 * num_heads_ * head_dim_ * sizeof(__nv_bfloat16));
        buf_k_.alloc(num_kv_heads_ * head_dim_ * sizeof(__nv_bfloat16));
        buf_v_.alloc(num_kv_heads_ * head_dim_ * sizeof(__nv_bfloat16));
        buf_attn_out_.alloc(hidden_size_ * sizeof(__nv_bfloat16));
        buf_gate_.alloc(intermediate_size_ * sizeof(__nv_bfloat16));
        buf_up_.alloc(intermediate_size_ * sizeof(__nv_bfloat16));
        buf_logits_.alloc(vocab_size_ * sizeof(float));
        buf_argmax_out_.alloc(sizeof(int32_t));

        loaded_ = true;
        LOG_INFO("Qwen 3.8 2B Neural Speculative Draft Model loaded successfully (24 layers, 1.7GB VRAM)!");
        return true;
    }

    void reset_state(cudaStream_t stream = nullptr) {
        if (!loaded_) return;
        if (draft_ssm_pool_.data) CUDA_CHECK(cudaMemsetAsync(draft_ssm_pool_.data, 0, draft_ssm_pool_.size_bytes, stream));
        if (draft_conv_pool_.data) CUDA_CHECK(cudaMemsetAsync(draft_conv_pool_.data, 0, draft_conv_pool_.size_bytes, stream));
        for (int s = 0; s < 8; s++) {
            if (draft_ssm_slots_[s].data) CUDA_CHECK(cudaMemsetAsync(draft_ssm_slots_[s].data, 0, draft_ssm_slots_[s].size_bytes, stream));
            if (draft_conv_slots_[s].data) CUDA_CHECK(cudaMemsetAsync(draft_conv_slots_[s].data, 0, draft_conv_slots_[s].size_bytes, stream));
        }
        for (auto& lw : layers_) {
            if (!lw.is_linear_attn) {
                if (lw.k_cache.data) CUDA_CHECK(cudaMemsetAsync(lw.k_cache.data, 0, lw.k_cache.size_bytes, stream));
                if (lw.v_cache.data) CUDA_CHECK(cudaMemsetAsync(lw.v_cache.data, 0, lw.v_cache.size_bytes, stream));
            }
        }
    }

    inline void checkpoint_state(cudaStream_t stream = nullptr) {
        if (!loaded_) return;
        if (draft_ssm_pool_.data && draft_ssm_slots_[0].data) {
            CUDA_CHECK(cudaMemcpyAsync(draft_ssm_slots_[0].data, draft_ssm_pool_.data, draft_ssm_pool_.size_bytes, cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(draft_conv_slots_[0].data, draft_conv_pool_.data, draft_conv_pool_.size_bytes, cudaMemcpyDeviceToDevice, stream));
        }
    }

    inline void restore_state(cudaStream_t stream = nullptr) {
        if (!loaded_) return;
        if (draft_ssm_pool_.data && draft_ssm_slots_[0].data) {
            CUDA_CHECK(cudaMemcpyAsync(draft_ssm_pool_.data, draft_ssm_slots_[0].data, draft_ssm_pool_.size_bytes, cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(draft_conv_pool_.data, draft_conv_slots_[0].data, draft_conv_pool_.size_bytes, cudaMemcpyDeviceToDevice, stream));
        }
    }

    inline void commit_state_slot(int slot_idx, cudaStream_t stream = nullptr) {
        if (!loaded_ || slot_idx < 0 || slot_idx >= 8) return;
        if (draft_ssm_pool_.data && draft_ssm_slots_[slot_idx].data) {
            CUDA_CHECK(cudaMemcpyAsync(draft_ssm_pool_.data, draft_ssm_slots_[slot_idx].data, draft_ssm_pool_.size_bytes, cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(draft_conv_pool_.data, draft_conv_slots_[slot_idx].data, draft_conv_pool_.size_bytes, cudaMemcpyDeviceToDevice, stream));
        }
    }

    bool graph_captured_ = false;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    bool draft_k_graph_captured_[9] = {false};
    cudaGraph_t draft_k_graph_[9] = {nullptr};
    cudaGraphExec_t draft_k_graph_exec_[9] = {nullptr};
    GPUTensor buf_input_pos_;
    int32_t* draft_best_host_ = nullptr;
    GPUTensor buf_draft_k_tokens_;
    int32_t* h_draft_k_cands_ = nullptr;
    int32_t* h_draft_k_pos_ = nullptr;

    void forward_token_device_body(int position, cudaStream_t stream) {
        // 1. Embedding lookup
        if (embed_w_.dtype == "int4") {
            embedding_int4_cuda(buf_hidden_.bf16(), (uint8_t*)embed_w_.data, embed_w_scale_.bf16(), buf_input_token_.i32(), 1, hidden_size_, stream);
        } else {
            embedding_cuda(buf_hidden_.bf16(), embed_w_.bf16(), buf_input_token_.i32(), 1, hidden_size_, stream);
        }

        auto matmul_proj = [&](GPUTensor& out, GPUTensor& in_vec, GPUTensor& weight, GPUTensor& scale, int N, int K) {
            if (weight.dtype == "int4") {
                gemv_int4_cuda(out.bf16(), in_vec.bf16(), (const uint8_t*)weight.data, scale.bf16(), N, K, stream);
            } else if (weight.dtype == "int3") {
                gemv_int3_cuda(out.bf16(), in_vec.bf16(), (const uint8_t*)weight.data, scale.bf16(), N, K, stream);
            } else {
                gemv_bf16_out_bf16_cuda(out.bf16(), weight.bf16(), in_vec.bf16(), N, K, stream);
            }
        };

        // 2. 24 Layers
        for (int l = 0; l < num_layers_; l++) {
            auto& lw = layers_[l];

            // Attention RMSNorm (1-centered for Qwen 3.8)
            rms_norm_one_centered_cuda(buf_hidden2_.bf16(), buf_hidden_.bf16(),
                                       lw.input_norm_w.bf16(), hidden_size_, 1e-6f, stream);

            if (lw.is_linear_attn) {
                matmul_proj(buf_qkv_, buf_hidden2_, lw.w_in_qkv, lw.w_in_qkv_scale, 6144, hidden_size_);
                matmul_proj(buf_z_, buf_hidden2_, lw.w_in_z, lw.w_in_z_scale, 2048, hidden_size_);
                matmul_proj(buf_a_, buf_hidden2_, lw.w_in_a, lw.w_in_a_scale, 16, hidden_size_);
                matmul_proj(buf_b_, buf_hidden2_, lw.w_in_b, lw.w_in_b_scale, 16, hidden_size_);

                deltanet_linear_attention_decode_cuda(
                    buf_attn_out_.bf16(), buf_qkv_.bf16(), buf_z_.bf16(),
                    buf_a_.bf16(), buf_b_.bf16(),
                    lw.conv1d_w.bf16(), lw.conv_state.bf16(),
                    lw.A_log.bf16(), lw.dt_bias.bf16(),
                    lw.linear_norm_w.bf16(), lw.ssm_state.bf16(),
                    16, 16, 128, stream);

                if (lw.linear_out_proj.dtype == "int4") {
                    gemv_int4_residual_cuda(buf_hidden_.bf16(), buf_attn_out_.bf16(),
                                            (const uint8_t*)lw.linear_out_proj.data, lw.linear_out_proj_scale.bf16(),
                                            hidden_size_, 2048, stream);
                } else {
                    matmul_proj(buf_hidden2_, buf_attn_out_, lw.linear_out_proj, lw.linear_out_proj_scale, hidden_size_, 2048);
                    vector_add_bf16_cuda(buf_hidden_.bf16(), buf_hidden2_.bf16(), hidden_size_, stream);
                }
            } else {
                matmul_proj(buf_q_, buf_hidden2_, lw.w_q, lw.w_q_scale, 2 * num_heads_ * head_dim_, hidden_size_);
                matmul_proj(buf_k_, buf_hidden2_, lw.w_k, lw.w_k_scale, num_kv_heads_ * head_dim_, hidden_size_);
                matmul_proj(buf_v_, buf_hidden2_, lw.w_v, lw.w_v_scale, num_kv_heads_ * head_dim_, hidden_size_);

                qwen_gqa_decode_gated_cuda(
                    buf_attn_out_.bf16(), buf_q_.bf16(), buf_k_.bf16(), buf_v_.bf16(),
                    lw.q_norm_w.bf16(), lw.k_norm_w.bf16(),
                    lw.k_cache.bf16(), lw.v_cache.bf16(),
                    num_heads_, num_kv_heads_, head_dim_,
                    buf_input_pos_.data ? buf_input_pos_.i32() : nullptr, position, 2048, 10000000.0f, 1e-6f, stream);

                if (lw.w_o.dtype == "int4") {
                    gemv_int4_residual_cuda(buf_hidden_.bf16(), buf_attn_out_.bf16(),
                                            (const uint8_t*)lw.w_o.data, lw.w_o_scale.bf16(),
                                            hidden_size_, num_heads_ * head_dim_, stream);
                } else {
                    matmul_proj(buf_hidden2_, buf_attn_out_, lw.w_o, lw.w_o_scale, hidden_size_, num_heads_ * head_dim_);
                    vector_add_bf16_cuda(buf_hidden_.bf16(), buf_hidden2_.bf16(), hidden_size_, stream);
                }
            }

            // FFN RMSNorm (1-centered for Qwen 3.8)
            rms_norm_one_centered_cuda(buf_hidden2_.bf16(), buf_hidden_.bf16(),
                                       lw.post_attn_norm_w.bf16(), hidden_size_, 1e-6f, stream);

            // Fused SwiGLU & Down Projection
            if (lw.w_gate.dtype == "int4") {
                gemv_int4_swiglu_fused_cuda(buf_gate_.bf16(), buf_hidden2_.bf16(),
                                            (const uint8_t*)lw.w_gate.data, lw.w_gate_scale.bf16(),
                                            (const uint8_t*)lw.w_up.data, lw.w_up_scale.bf16(),
                                            intermediate_size_, hidden_size_, 0.0f, stream);
            } else {
                matmul_proj(buf_gate_, buf_hidden2_, lw.w_gate, lw.w_gate_scale, intermediate_size_, hidden_size_);
                matmul_proj(buf_up_, buf_hidden2_, lw.w_up, lw.w_up_scale, intermediate_size_, hidden_size_);
                silu_mul_cuda(buf_gate_.bf16(), buf_gate_.bf16(), buf_up_.bf16(), intermediate_size_, 0.0f, stream);
            }

            if (lw.w_down.dtype == "int4") {
                gemv_int4_residual_cuda(buf_hidden_.bf16(), buf_gate_.bf16(),
                                        (const uint8_t*)lw.w_down.data, lw.w_down_scale.bf16(),
                                        hidden_size_, intermediate_size_, stream);
            } else {
                matmul_proj(buf_hidden2_, buf_gate_, lw.w_down, lw.w_down_scale, hidden_size_, intermediate_size_);
                vector_add_bf16_cuda(buf_hidden_.bf16(), buf_hidden2_.bf16(), hidden_size_, stream);
            }
        }

        // 3. Final RMSNorm
        rms_norm_cuda(buf_hidden_.bf16(), buf_hidden_.bf16(),
                      final_norm_w_.bf16(), hidden_size_, 1e-6f, stream);

        // 4. Logits
        gemv_bf16_cuda(buf_logits_.f32(), embed_w_.bf16(), buf_hidden_.bf16(), vocab_size_, hidden_size_, stream);

        // 5. Argmax
        argmax_f32_cuda(buf_argmax_out_.i32(), buf_logits_.f32(), vocab_size_, stream);
    }

    void draft_k_tokens_device_body(int K, cudaStream_t stream) {
        for (int k = 0; k < K; k++) {
            forward_token_device_body(0, stream);

            // Save draft state after step k (after processing cand_tokens[k]) into slot k
            if (k < 8 && draft_ssm_pool_.data && draft_ssm_slots_[k].data) {
                CUDA_CHECK(cudaMemcpyAsync(draft_ssm_slots_[k].data, draft_ssm_pool_.data, draft_ssm_pool_.size_bytes, cudaMemcpyDeviceToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(draft_conv_slots_[k].data, draft_conv_pool_.data, draft_conv_pool_.size_bytes, cudaMemcpyDeviceToDevice, stream));
            }

            // Save argmax output into device candidate slot (k+1) and feed directly to input for step k+1
            CUDA_CHECK(cudaMemcpyAsync(buf_draft_k_tokens_.i32() + (k + 1), buf_argmax_out_.i32(), sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
            if (k + 1 < K) {
                CUDA_CHECK(cudaMemcpyAsync(buf_input_token_.i32(), buf_argmax_out_.i32(), sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
                increment_i32_cuda(buf_input_pos_.i32(), 1, stream);
            }
        }
    }

    void init_cuda_graph(cudaStream_t stream = nullptr) {
#if defined(__APPLE__)
        graph_captured_ = false;
        for (int k = 0; k <= 8; ++k) draft_k_graph_captured_[k] = false;
        return;
#endif
        if (!loaded_) return;
        buf_input_pos_.alloc(sizeof(int32_t));
        buf_draft_k_tokens_.alloc(16 * sizeof(int32_t));
        if (!h_draft_k_cands_) {
            CUDA_CHECK(cudaMallocHost(&h_draft_k_cands_, 16 * sizeof(int32_t)));
            CUDA_CHECK(cudaMallocHost(&h_draft_k_pos_, 16 * sizeof(int32_t)));
        }

        int dummy_tok = 0;
        int dummy_pos = 0;
        CUDA_CHECK(cudaMemcpy(buf_input_token_.i32(), &dummy_tok, sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(buf_input_pos_.i32(), &dummy_pos, sizeof(int32_t), cudaMemcpyHostToDevice));

        // Warmup single step
        for (int i = 0; i < 3; i++) {
            forward_token_device_body(0, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        cudaError_t cap_err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        if (cap_err == cudaSuccess) {
            forward_token_device_body(0, stream);
            cudaError_t end_err = cudaStreamEndCapture(stream, &graph_);
            if (end_err == cudaSuccess) {
                if (cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0) == cudaSuccess) {
                    graph_captured_ = true;
                    LOG_INFO("Qwen 3.8 2B Single-Step Draft CUDA Graph instantiated successfully!");
                }
            }
        }

        // Multi-Step Draft CUDA Graph capture for K in {1, 2, 3, 4, 5, 6, 7}
        for (int K : {1, 2, 3, 4, 5, 6, 7}) {
            for (int i = 0; i < 2; i++) {
                draft_k_tokens_device_body(K, stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));

            cudaError_t k_cap = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            if (k_cap != cudaSuccess) continue;

            draft_k_tokens_device_body(K, stream);

            cudaError_t k_end = cudaStreamEndCapture(stream, &draft_k_graph_[K]);
            if (k_end != cudaSuccess) continue;

            cudaError_t k_inst = cudaGraphInstantiate(&draft_k_graph_exec_[K], draft_k_graph_[K], nullptr, nullptr, 0);
            if (k_inst == cudaSuccess) {
                draft_k_graph_captured_[K] = true;
                LOG_INFO("Qwen 3.8 2B Multi-Step Draft CUDA Graph (K=%d) instantiated successfully!", K);
            }
        }
    }

    void draft_k_tokens(int start_token, int start_pos, int K, std::vector<int>& cands, cudaStream_t stream = nullptr) {
        cands.clear();
        cands.push_back(start_token);
        if (!loaded_ || K <= 0) return;

        if (!h_draft_k_cands_) {
            CUDA_CHECK(cudaMallocHost(&h_draft_k_cands_, 16 * sizeof(int32_t)));
            CUDA_CHECK(cudaMallocHost(&h_draft_k_pos_, 16 * sizeof(int32_t)));
            buf_draft_k_tokens_.alloc(16 * sizeof(int32_t));
        }

        h_draft_k_cands_[0] = start_token;
        CUDA_CHECK(cudaMemcpyAsync(buf_input_token_.i32(), &h_draft_k_cands_[0], sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(buf_input_pos_.i32(), &start_pos, sizeof(int32_t), cudaMemcpyHostToDevice, stream));

        if (K >= 1 && K <= 7 && draft_k_graph_captured_[K]) {
            CUDA_CHECK(cudaGraphLaunch(draft_k_graph_exec_[K], stream));
        } else {
            draft_k_tokens_device_body(K, stream);
        }

        // Copy all K drafted tokens to host in ONE single DMA transfer
        CUDA_CHECK(cudaMemcpyAsync(h_draft_k_cands_ + 1, buf_draft_k_tokens_.i32() + 1, K * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for (int k = 0; k < K; k++) {
            cands.push_back(h_draft_k_cands_[k + 1]);
        }
    }

    void replay_tokens(const int* tokens, int start_pos, int count, cudaStream_t stream = nullptr) {
        if (!loaded_ || count <= 0) return;
        for (int j = 0; j < count; j++) {
            forward_token_async(tokens[j], start_pos + j, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }

    void forward_token_async(int token_id, int position, cudaStream_t stream = nullptr) {
        if (!loaded_) return;
        if (!h_draft_tok_) {
            CUDA_CHECK(cudaMallocHost(&h_draft_tok_, sizeof(int32_t)));
            CUDA_CHECK(cudaMallocHost(&h_draft_pos_, sizeof(int32_t)));
        }
        *h_draft_tok_ = token_id;
        *h_draft_pos_ = position;
        CUDA_CHECK(cudaMemcpyAsync(buf_input_token_.i32(), h_draft_tok_, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        if (buf_input_pos_.data) {
            CUDA_CHECK(cudaMemcpyAsync(buf_input_pos_.i32(), h_draft_pos_, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        }

        if (graph_captured_) {
            CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream));
        } else {
            forward_token_device_body(position, stream);
        }
    }

    void forward_token_async_pinned(const int32_t* h_tok, const int32_t* h_pos, int position, cudaStream_t stream = nullptr) {
        if (!loaded_) return;
        CUDA_CHECK(cudaMemcpyAsync(buf_input_token_.i32(), h_tok, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        if (buf_input_pos_.data) {
            CUDA_CHECK(cudaMemcpyAsync(buf_input_pos_.i32(), h_pos, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        }

        if (graph_captured_) {
            CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream));
        } else {
            forward_token_device_body(position, stream);
        }
    }

    int forward_token(int token_id, int position, cudaStream_t stream = nullptr) {
        forward_token_async(token_id, position, stream);

        if (!draft_best_host_) {
            CUDA_CHECK(cudaMallocHost(&draft_best_host_, sizeof(int32_t)));
        }
        CUDA_CHECK(cudaMemcpyAsync(draft_best_host_, buf_argmax_out_.i32(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        static int dbg_cnt = 0;
        if (dbg_cnt++ < 6) {
            std::vector<float> all_logits(vocab_size_);
            CUDA_CHECK(cudaMemcpy(all_logits.data(), buf_logits_.f32(), vocab_size_ * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<int> indices(vocab_size_);
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(), indices.begin() + 3, indices.end(), [&](int a, int b) {
                return all_logits[a] > all_logits[b];
            });
            LOG_INFO("[DRAFT TOP3] in=%d pos=%d -> #1=%d(%.2f) #2=%d(%.2f) #3=%d(%.2f)",
                     token_id, position,
                     indices[0], all_logits[indices[0]],
                     indices[1], all_logits[indices[1]],
                     indices[2], all_logits[indices[2]]);
        }

        return *draft_best_host_;
    }
};

// ════════════════════════════════════════════════════════════════════════════════
//  MTP Self-Drafter: Multi-Token Prediction using the target model's own MTP module
//  Reference: syv-ai/qwen38-27b-rtx3090 — MTP speculation with draft vocabulary
// ════════════════════════════════════════════════════════════════════════════════

class MTPSelfDrafter {
public:
    bool loaded_ = false;
    int hidden_size_ = 5120;
    int intermediate_size_ = 17408;
    int num_heads_ = 24;       // q heads
    int num_kv_heads_ = 4;     // kv heads
    int head_dim_ = 256;
    int full_vocab_size_ = 0;   // Size of full vocabulary (248320)
    float rms_eps_ = 1e-6f;

    // MTP module weights (loaded from main model checkpoint)
    GPUTensor mtp_fc_w_, mtp_fc_s_;        // [5120, 10240] BF16 or INT4
    GPUTensor mtp_pre_fc_norm_hidden_w_;   // [5120] BF16
    GPUTensor mtp_pre_fc_norm_embed_w_;    // [5120] BF16
    GPUTensor mtp_norm_w_;                 // [5120] BF16

    // MTP transformer layer weights
    GPUTensor mtp_layer_attn_norm_w_;      // [5120] BF16
    GPUTensor mtp_layer_post_attn_norm_w_; // [5120] BF16
    GPUTensor mtp_layer_q_proj_w_, mtp_layer_q_proj_s_;   // [12288, 5120] INT4
    GPUTensor mtp_layer_k_proj_w_, mtp_layer_k_proj_s_;   // [1024, 5120] INT4
    GPUTensor mtp_layer_v_proj_w_, mtp_layer_v_proj_s_;   // [1024, 5120] INT4
    GPUTensor mtp_layer_o_proj_w_, mtp_layer_o_proj_s_;   // [5120, 6144] INT4
    GPUTensor mtp_layer_q_norm_w_;          // [256] BF16
    GPUTensor mtp_layer_k_norm_w_;          // [256] BF16
    GPUTensor mtp_layer_gate_w_, mtp_layer_gate_s_;  // [17408, 5120] INT4
    GPUTensor mtp_layer_up_w_, mtp_layer_up_s_;      // [17408, 5120] INT4
    GPUTensor mtp_layer_down_w_, mtp_layer_down_s_;  // [5120, 17408] INT4

    // MTP KV cache (small - only 1 layer, limited context)
    int max_seq_len_ = 32768;
    GPUTensor mtp_k_cache_;  // [MAX_SEQ_LEN, num_kv_heads, head_dim] BF16
    GPUTensor mtp_v_cache_;  // [MAX_SEQ_LEN, num_kv_heads, head_dim] BF16

    // Full model lm_head (shared pointer, not owned — points to MoecherEngine::head_weight_)
    __nv_bfloat16* full_lm_head_w_ = nullptr;  // [full_vocab_size, hidden_size] BF16

    // Draft vocabulary and compact draft lm_head (40k x 5120 BF16 = ~400MB vs 2.54GB full)
    bool use_draft_vocab_ = false;
    int draft_vocab_size_ = 0;
    GPUTensor draft_lm_head_w_;                // [draft_vocab_size, hidden_size] BF16
    std::vector<int32_t> draft_vocab_ids_;     // [draft_vocab_size] host table mapping draft_id -> full_vocab_id

    // Working buffers
    GPUTensor buf_mtp_hidden_;      // [5120] BF16
    GPUTensor buf_mtp_hidden2_;     // [5120] BF16
    GPUTensor buf_mtp_concat_;      // [10240] BF16
    GPUTensor buf_mtp_embed_;       // [5120] BF16
    GPUTensor buf_mtp_q_;           // [24*256*2] BF16 (gated)
    GPUTensor buf_mtp_k_;           // [4*256] BF16
    GPUTensor buf_mtp_v_;           // [4*256] BF16
    GPUTensor buf_mtp_attn_out_;    // [24*256] BF16
    GPUTensor buf_mtp_gate_;        // [17408] BF16
    GPUTensor buf_mtp_up_;          // [17408] BF16
    GPUTensor buf_mtp_logits_;      // [full_vocab_size] F32
    GPUTensor buf_mtp_argmax_;      // [1] INT32
    GPUTensor buf_mtp_input_pos_;   // [1] INT32
    GPUTensor buf_draft_k_tokens_;  // [16] INT32 — output candidate tokens
    GPUTensor buf_mtp_top2_idx_;    // [2] INT32
    GPUTensor buf_mtp_top2_val_;    // [2] FLOAT
    GPUTensor buf_mtp_hidden_saved_;   // [5120] BF16
    GPUTensor buf_mtp_hidden_saved_2_; // [5120] BF16

    int32_t* h_draft_best_ = nullptr;
    int32_t* h_draft_k_cands_ = nullptr;
    int32_t* h_draft_pos_ = nullptr;
    int32_t* h_draft_top2_ = nullptr;


    // Reference to target model's embedding table (shared, not owned)
    __nv_bfloat16* target_embed_w_ = nullptr;
    uint8_t* target_embed_w_int4_ = nullptr;
    const uint8_t* target_embed_w_fp4_ = nullptr;
    __nv_bfloat16* target_embed_s_ = nullptr;
    const uint8_t* target_embed_s_fp4_ = nullptr;
    bool is_embed_int4_ = false;
    bool is_embed_fp4_ = false;

    bool load_mtp_weights(const void* mapped_data, const json& tensor_map,
                          __nv_bfloat16* embed_w, uint8_t* embed_w_int4, void* embed_s, bool is_embed_int4, bool is_embed_fp4,
                          __nv_bfloat16* full_lm_head, int full_vocab,
                          const std::string& model_dir, cudaStream_t stream,
                          int max_seq_len = 32768) {
        max_seq_len_ = max_seq_len > 0 ? max_seq_len : 32768;
        full_vocab_size_ = full_vocab;
        target_embed_w_ = embed_w;
        target_embed_w_int4_ = embed_w_int4;
        target_embed_w_fp4_ = is_embed_fp4 ? (const uint8_t*)embed_w_int4 : nullptr;
        target_embed_s_ = is_embed_int4 ? (__nv_bfloat16*)embed_s : nullptr;
        target_embed_s_fp4_ = is_embed_fp4 ? (const uint8_t*)embed_s : nullptr;
        is_embed_int4_ = is_embed_int4;
        is_embed_fp4_ = is_embed_fp4;
        full_lm_head_w_ = full_lm_head;

        auto load_t = [&](GPUTensor& gpu, const std::string& name) -> bool {
            if (!tensor_map.contains(name)) return false;
            auto& info = tensor_map[name];
            int64_t offset = info["offset"].get<int64_t>();
            int64_t nbytes = info["nbytes"].get<int64_t>();
            gpu.dtype = info["dtype"].get<std::string>();
            gpu.shape.clear();
            for (auto& s : info["shape"]) gpu.shape.push_back(s.get<int>());
            gpu.alloc(nbytes);
            CUDA_CHECK(cudaMemcpyAsync(gpu.data, (const char*)mapped_data + offset, nbytes,
                                       cudaMemcpyHostToDevice, stream));
            return true;
        };

        auto load_layer_weight = [&](GPUTensor& gpu_w, GPUTensor& gpu_s, const std::string& name) -> bool {
            if (!tensor_map.contains(name)) return false;
            auto& info = tensor_map[name];
            std::string dtype = info.value("dtype", "");
            int64_t offset = info["offset"].get<int64_t>();
            int64_t nbytes = info["nbytes"].get<int64_t>();
            gpu_w.dtype = dtype;
            gpu_w.shape.clear();
            for (auto& s : info["shape"]) gpu_w.shape.push_back(s.get<int>());
            gpu_w.alloc(nbytes);
            CUDA_CHECK(cudaMemcpyAsync(gpu_w.data, (const char*)mapped_data + offset, nbytes,
                                       cudaMemcpyHostToDevice, stream));
            if (dtype == "int4" || dtype == "fp4") {
                if (info.contains("scale_offset")) {
                    int64_t scale_offset = info["scale_offset"].get<int64_t>();
                    int64_t scale_nbytes = info["scale_nbytes"].get<int64_t>();
                    gpu_s.dtype = info.value("scale_dtype", dtype == "fp4" ? "F8_E8M0" : "bfloat16");
                    gpu_s.alloc(scale_nbytes);
                    CUDA_CHECK(cudaMemcpyAsync(gpu_s.data, (const char*)mapped_data + scale_offset, scale_nbytes,
                                               cudaMemcpyHostToDevice, stream));
                }
            }
            return true;
        };

        // Load MTP module weights
        bool fc_loaded = load_layer_weight(mtp_fc_w_, mtp_fc_s_, "mtp.fc.weight");
        if (!fc_loaded) {
            LOG_WARN("MTP: mtp.fc.weight not found or failed to load, MTP self-drafter disabled");
            return false;
        }
        load_t(mtp_pre_fc_norm_hidden_w_, "mtp.pre_fc_norm_hidden.weight");
        load_t(mtp_pre_fc_norm_embed_w_, "mtp.pre_fc_norm_embedding.weight");
        load_t(mtp_norm_w_, "mtp.norm.weight");

        // Load MTP transformer layer
        load_t(mtp_layer_attn_norm_w_, "mtp.layers.0.input_layernorm.weight");
        load_t(mtp_layer_post_attn_norm_w_, "mtp.layers.0.post_attention_layernorm.weight");
        load_layer_weight(mtp_layer_q_proj_w_, mtp_layer_q_proj_s_, "mtp.layers.0.self_attn.q_proj.weight");
        load_layer_weight(mtp_layer_k_proj_w_, mtp_layer_k_proj_s_, "mtp.layers.0.self_attn.k_proj.weight");
        load_layer_weight(mtp_layer_v_proj_w_, mtp_layer_v_proj_s_, "mtp.layers.0.self_attn.v_proj.weight");
        load_layer_weight(mtp_layer_o_proj_w_, mtp_layer_o_proj_s_, "mtp.layers.0.self_attn.o_proj.weight");
        load_t(mtp_layer_q_norm_w_, "mtp.layers.0.self_attn.q_norm.weight");
        load_t(mtp_layer_k_norm_w_, "mtp.layers.0.self_attn.k_norm.weight");
        load_layer_weight(mtp_layer_gate_w_, mtp_layer_gate_s_, "mtp.layers.0.mlp.gate_proj.weight");
        load_layer_weight(mtp_layer_up_w_, mtp_layer_up_s_, "mtp.layers.0.mlp.up_proj.weight");
        load_layer_weight(mtp_layer_down_w_, mtp_layer_down_s_, "mtp.layers.0.mlp.down_proj.weight");

        if (!mtp_layer_attn_norm_w_.data || !mtp_layer_post_attn_norm_w_.data ||
            !mtp_layer_q_proj_w_.data || !mtp_layer_k_proj_w_.data ||
            !mtp_layer_v_proj_w_.data || !mtp_layer_o_proj_w_.data ||
            !mtp_layer_gate_w_.data || !mtp_layer_up_w_.data || !mtp_layer_down_w_.data) {
            LOG_WARN("MTP: One or more MTP transformer layer weights could not be loaded, MTP self-drafter disabled");
            return false;
        }

        // Allocate MTP KV cache
        int max_seq = max_seq_len_ > 0 ? max_seq_len_ : 32768;
        mtp_k_cache_.alloc((size_t)max_seq * num_kv_heads_ * head_dim_ * sizeof(__nv_bfloat16));
        mtp_v_cache_.alloc((size_t)max_seq * num_kv_heads_ * head_dim_ * sizeof(__nv_bfloat16));
        CUDA_CHECK(cudaMemsetAsync(mtp_k_cache_.data, 0, mtp_k_cache_.size_bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(mtp_v_cache_.data, 0, mtp_v_cache_.size_bytes, stream));

        // Try loading compact draft vocabulary and draft lm_head (BF16)
        std::string draft_ids_path = model_dir + "/draft_vocab_ids.bin";
        std::string draft_head_path = model_dir + "/draft_lm_head_int8_bf16.bin";

        FILE* f_ids = fopen(draft_ids_path.c_str(), "rb");
        FILE* f_head = fopen(draft_head_path.c_str(), "rb");
        if (f_ids && f_head) {
            uint32_t n_ids = 0;
            if (fread(&n_ids, sizeof(uint32_t), 1, f_ids) == 1 && n_ids > 0) {
                draft_vocab_ids_.resize(n_ids);
                if (fread(draft_vocab_ids_.data(), sizeof(int32_t), n_ids, f_ids) == n_ids) {
                    uint32_t rows = 0, cols = 0;
                    if (fread(&rows, sizeof(uint32_t), 1, f_head) == 1 &&
                        fread(&cols, sizeof(uint32_t), 1, f_head) == 1 &&
                        rows == n_ids && (int)cols == hidden_size_) {
                        size_t head_bytes = (size_t)rows * cols * sizeof(__nv_bfloat16);
                        std::vector<char> head_buf(head_bytes);
                        if (fread(head_buf.data(), 1, head_bytes, f_head) == head_bytes) {
                            draft_lm_head_w_.alloc(head_bytes);
                            CUDA_CHECK(cudaMemcpyAsync(draft_lm_head_w_.data, head_buf.data(), head_bytes, cudaMemcpyHostToDevice, stream));
                            use_draft_vocab_ = true;
                            draft_vocab_size_ = (int)n_ids;
                            LOG_INFO("MTP: Loaded draft vocabulary (%d tokens) and draft lm_head (%.1f MB BF16, 6.2x faster drafting!)",
                                     draft_vocab_size_, head_bytes / (1024.0 * 1024.0));
                        }
                    }
                }
            }
        }
        if (f_ids) fclose(f_ids);
        if (f_head) fclose(f_head);

        // Validate lm_head pointer if draft vocabulary is not available
        if (!use_draft_vocab_) {
            if (!full_lm_head_w_) {
                LOG_WARN("MTP: full lm_head pointer is null and no draft lm_head found, MTP drafter disabled");
                return false;
            }
            LOG_INFO("MTP: Using full model lm_head [%d, %d] BF16 (shared, no extra VRAM)",
                     full_vocab_size_, hidden_size_);
        }

        // Allocate working buffers
        buf_mtp_hidden_.alloc(hidden_size_ * sizeof(__nv_bfloat16));
        buf_mtp_hidden2_.alloc(hidden_size_ * sizeof(__nv_bfloat16));
        buf_mtp_concat_.alloc(2 * hidden_size_ * sizeof(__nv_bfloat16));
        buf_mtp_embed_.alloc(hidden_size_ * sizeof(__nv_bfloat16));
        buf_mtp_q_.alloc(2 * num_heads_ * head_dim_ * sizeof(__nv_bfloat16));
        buf_mtp_k_.alloc(num_kv_heads_ * head_dim_ * sizeof(__nv_bfloat16));
        buf_mtp_v_.alloc(num_kv_heads_ * head_dim_ * sizeof(__nv_bfloat16));
        buf_mtp_attn_out_.alloc(num_heads_ * head_dim_ * sizeof(__nv_bfloat16));
        buf_mtp_gate_.alloc(intermediate_size_ * sizeof(__nv_bfloat16));
        buf_mtp_up_.alloc(intermediate_size_ * sizeof(__nv_bfloat16));
        buf_mtp_logits_.alloc(full_vocab_size_ * sizeof(float));
        buf_mtp_argmax_.alloc(sizeof(int32_t));
        buf_mtp_input_pos_.alloc(sizeof(int32_t));
        buf_draft_k_tokens_.alloc(16 * sizeof(int32_t));
        buf_mtp_top2_idx_.alloc(2 * sizeof(int32_t));
        buf_mtp_top2_val_.alloc(2 * sizeof(float));
        buf_mtp_hidden_saved_.alloc(hidden_size_ * sizeof(__nv_bfloat16));
        buf_mtp_hidden_saved_2_.alloc(hidden_size_ * sizeof(__nv_bfloat16));

        CUDA_CHECK(cudaMallocHost(&h_draft_best_, sizeof(int32_t)));
        CUDA_CHECK(cudaMallocHost(&h_draft_k_cands_, 16 * sizeof(int32_t)));
        CUDA_CHECK(cudaMallocHost(&h_draft_pos_, sizeof(int32_t)));
        CUDA_CHECK(cudaMallocHost(&h_draft_top2_, 2 * sizeof(int32_t)));


        CUDA_CHECK(cudaStreamSynchronize(stream));

        loaded_ = true;
        LOG_INFO("MTP Self-Drafter loaded: 1 transformer layer + full %dk vocab (shared lm_head) = %.1f MB own VRAM",
                 full_vocab_size_ / 1000,
                 (mtp_fc_w_.size_bytes + mtp_fc_s_.size_bytes + mtp_layer_q_proj_w_.size_bytes + mtp_layer_k_proj_w_.size_bytes +
                  mtp_layer_v_proj_w_.size_bytes + mtp_layer_o_proj_w_.size_bytes +
                  mtp_layer_gate_w_.size_bytes + mtp_layer_up_w_.size_bytes + mtp_layer_down_w_.size_bytes +
                  mtp_k_cache_.size_bytes + mtp_v_cache_.size_bytes) / (1024.0 * 1024.0));
        return true;
    }

    void reset_kv_cache(cudaStream_t stream = nullptr) {
        if (mtp_k_cache_.data) CUDA_CHECK(cudaMemsetAsync(mtp_k_cache_.data, 0, mtp_k_cache_.size_bytes, stream));
        if (mtp_v_cache_.data) CUDA_CHECK(cudaMemsetAsync(mtp_v_cache_.data, 0, mtp_v_cache_.size_bytes, stream));
    }

    // Forward one MTP step: given target hidden state and last token, produce draft token
    // target_hidden: [hidden_size] BF16 — the hidden state from the target model's last layer (pre-norm)
    // last_token_id: the last token that was predicted (full vocab id)
    // position: current sequence position
    // Returns: draft token id (in full vocabulary space), or -1 if not in draft vocab
    int forward_one_step(const __nv_bfloat16* target_hidden, int last_token_id,
                         int position, cudaStream_t stream) {
        if (!loaded_) return -1;

        auto matmul_proj = [&](GPUTensor& out, __nv_bfloat16* in_vec, GPUTensor& weight, GPUTensor& scale, int N, int K) {
            if (weight.dtype == "int4") {
                gemv_int4_cuda(out.bf16(), in_vec, (const uint8_t*)weight.data, scale.bf16(), N, K, stream);
            } else if (weight.dtype == "fp4") {
                gemv_fp4_cuda(out.bf16(), in_vec, (const uint8_t*)weight.data, (const uint8_t*)scale.data, N, K, stream);
            } else {
                gemv_bf16_out_bf16_cuda(out.bf16(), weight.bf16(), in_vec, N, K, stream);
            }
        };

        // 1. Norm the target hidden state
        rms_norm_one_centered_cuda(buf_mtp_hidden_.bf16(), target_hidden,
                                   mtp_pre_fc_norm_hidden_w_.bf16(), hidden_size_, rms_eps_, stream);

        // 2. Embed the last token and norm it
        int32_t tok = last_token_id;
        CUDA_CHECK(cudaMemcpyAsync(buf_mtp_argmax_.i32(), &tok, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        if (is_embed_fp4_) {
            embedding_fp4_cuda(buf_mtp_embed_.bf16(), target_embed_w_fp4_, target_embed_s_fp4_, buf_mtp_argmax_.i32(), 1, hidden_size_, stream);
        } else if (is_embed_int4_) {
            embedding_int4_cuda(buf_mtp_embed_.bf16(), target_embed_w_int4_, target_embed_s_, buf_mtp_argmax_.i32(), 1, hidden_size_, stream);
        } else {
            embedding_cuda(buf_mtp_embed_.bf16(), target_embed_w_, buf_mtp_argmax_.i32(), 1, hidden_size_, stream);
        }
        rms_norm_one_centered_cuda(buf_mtp_embed_.bf16(), buf_mtp_embed_.bf16(),
                                   mtp_pre_fc_norm_embed_w_.bf16(), hidden_size_, rms_eps_, stream);

        // 3. Concatenate: [normed_embed, normed_hidden] → [10240]
        //    Order matters! FC weight trained with embed first, hidden second (per reference impl)
        CUDA_CHECK(cudaMemcpyAsync(buf_mtp_concat_.bf16(), buf_mtp_embed_.bf16(),
                                   hidden_size_ * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(buf_mtp_concat_.bf16() + hidden_size_, buf_mtp_hidden_.bf16(),
                                   hidden_size_ * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));

        // 4. FC projection: [10240] → [5120]
        if (mtp_fc_w_.dtype == "int4") {
            gemv_int4_cuda(buf_mtp_hidden_.bf16(), buf_mtp_concat_.bf16(),
                           (const uint8_t*)mtp_fc_w_.data, mtp_fc_s_.bf16(),
                           hidden_size_, 2 * hidden_size_, stream);
        } else if (mtp_fc_w_.dtype == "fp4") {
            gemv_fp4_cuda(buf_mtp_hidden_.bf16(), buf_mtp_concat_.bf16(),
                          (const uint8_t*)mtp_fc_w_.data, (const uint8_t*)mtp_fc_s_.data,
                          hidden_size_, 2 * hidden_size_, stream);
        } else {
            gemv_bf16_out_bf16_cuda(buf_mtp_hidden_.bf16(), mtp_fc_w_.bf16(), buf_mtp_concat_.bf16(),
                                    hidden_size_, 2 * hidden_size_, stream);
        }

        // 5. MTP Transformer layer
        // 5a. Attention pre-norm
        rms_norm_one_centered_cuda(buf_mtp_hidden2_.bf16(), buf_mtp_hidden_.bf16(),
                                   mtp_layer_attn_norm_w_.bf16(), hidden_size_, rms_eps_, stream);

        // 5b. Q/K/V projections
        matmul_proj(buf_mtp_q_, buf_mtp_hidden2_.bf16(), mtp_layer_q_proj_w_, mtp_layer_q_proj_s_,
                    2 * num_heads_ * head_dim_, hidden_size_);
        matmul_proj(buf_mtp_k_, buf_mtp_hidden2_.bf16(), mtp_layer_k_proj_w_, mtp_layer_k_proj_s_,
                    num_kv_heads_ * head_dim_, hidden_size_);
        matmul_proj(buf_mtp_v_, buf_mtp_hidden2_.bf16(), mtp_layer_v_proj_w_, mtp_layer_v_proj_s_,
                    num_kv_heads_ * head_dim_, hidden_size_);

        // 5c. GQA decode attention with gated Q (same as main model)
        *h_draft_pos_ = position;
        CUDA_CHECK(cudaMemcpyAsync(buf_mtp_input_pos_.i32(), h_draft_pos_, sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream));
#if defined(__APPLE__)
        qwen_gqa_decode_gated_cuda(
            buf_mtp_attn_out_.bf16(), buf_mtp_q_.bf16(), buf_mtp_k_.bf16(), buf_mtp_v_.bf16(),
            mtp_layer_q_norm_w_.bf16(), mtp_layer_k_norm_w_.bf16(),
            mtp_k_cache_.bf16(), mtp_v_cache_.bf16(),
            num_heads_, num_kv_heads_, head_dim_,
            buf_mtp_input_pos_.i32(), position, max_seq_len_ > 0 ? max_seq_len_ : 32768,
            10000000.0f, rms_eps_, stream, /*cache_pos=*/-1);
#else
        qwen_gqa_decode_gated_cuda(
            buf_mtp_attn_out_.bf16(), buf_mtp_q_.bf16(), buf_mtp_k_.bf16(), buf_mtp_v_.bf16(),
            mtp_layer_q_norm_w_.bf16(), mtp_layer_k_norm_w_.bf16(),
            mtp_k_cache_.bf16(), mtp_v_cache_.bf16(),
            num_heads_, num_kv_heads_, head_dim_,
            buf_mtp_input_pos_.i32(), position, max_seq_len_ > 0 ? max_seq_len_ : 32768,
            10000000.0f, rms_eps_, stream);
#endif

        // 5d. Output projection + residual
        if (mtp_layer_o_proj_w_.dtype == "int4") {
            gemv_int4_residual_cuda(buf_mtp_hidden_.bf16(), buf_mtp_attn_out_.bf16(),
                                    (const uint8_t*)mtp_layer_o_proj_w_.data, mtp_layer_o_proj_s_.bf16(),
                                    hidden_size_, num_heads_ * head_dim_, stream);
        } else if (mtp_layer_o_proj_w_.dtype == "fp4") {
            gemv_fp4_residual_cuda(buf_mtp_hidden_.bf16(), buf_mtp_attn_out_.bf16(),
                                   (const uint8_t*)mtp_layer_o_proj_w_.data, (const uint8_t*)mtp_layer_o_proj_s_.data,
                                   hidden_size_, num_heads_ * head_dim_, stream);
        } else {
            gemv_bf16_out_bf16_cuda(buf_mtp_hidden2_.bf16(), mtp_layer_o_proj_w_.bf16(),
                                    buf_mtp_attn_out_.bf16(), hidden_size_, num_heads_ * head_dim_, stream);
            vector_add_bf16_cuda(buf_mtp_hidden_.bf16(), buf_mtp_hidden2_.bf16(), hidden_size_, stream);
        }

        // 5e. FFN pre-norm
        rms_norm_one_centered_cuda(buf_mtp_hidden2_.bf16(), buf_mtp_hidden_.bf16(),
                                   mtp_layer_post_attn_norm_w_.bf16(), hidden_size_, rms_eps_, stream);

        // 5f. FFN: SwiGLU + Down projection
        if (mtp_layer_gate_w_.dtype == "int4") {
            gemv_int4_swiglu_fused_cuda(buf_mtp_gate_.bf16(), buf_mtp_hidden2_.bf16(),
                                        (const uint8_t*)mtp_layer_gate_w_.data, mtp_layer_gate_s_.bf16(),
                                        (const uint8_t*)mtp_layer_up_w_.data, mtp_layer_up_s_.bf16(),
                                        intermediate_size_, hidden_size_, 0.0f, stream);
        } else if (mtp_layer_gate_w_.dtype == "fp4") {
            gemv_fp4_swiglu_fused_cuda(buf_mtp_gate_.bf16(), buf_mtp_hidden2_.bf16(),
                                       (const uint8_t*)mtp_layer_gate_w_.data, (const uint8_t*)mtp_layer_gate_s_.data,
                                       (const uint8_t*)mtp_layer_up_w_.data, (const uint8_t*)mtp_layer_up_s_.data,
                                       intermediate_size_, hidden_size_, 0.0f, stream);
        } else {
            gemv_bf16_out_bf16_cuda(buf_mtp_gate_.bf16(), mtp_layer_gate_w_.bf16(),
                                    buf_mtp_hidden2_.bf16(), intermediate_size_, hidden_size_, stream);
            gemv_bf16_out_bf16_cuda(buf_mtp_up_.bf16(), mtp_layer_up_w_.bf16(),
                                    buf_mtp_hidden2_.bf16(), intermediate_size_, hidden_size_, stream);
            silu_mul_cuda(buf_mtp_gate_.bf16(), buf_mtp_gate_.bf16(), buf_mtp_up_.bf16(), intermediate_size_, 0.0f, stream);
        }

        if (mtp_layer_down_w_.dtype == "int4") {
            gemv_int4_residual_cuda(buf_mtp_hidden_.bf16(), buf_mtp_gate_.bf16(),
                                    (const uint8_t*)mtp_layer_down_w_.data, mtp_layer_down_s_.bf16(),
                                    hidden_size_, intermediate_size_, stream);
        } else if (mtp_layer_down_w_.dtype == "fp4") {
            gemv_fp4_residual_cuda(buf_mtp_hidden_.bf16(), buf_mtp_gate_.bf16(),
                                   (const uint8_t*)mtp_layer_down_w_.data, (const uint8_t*)mtp_layer_down_s_.data,
                                   hidden_size_, intermediate_size_, stream);
        } else {
            gemv_bf16_out_bf16_cuda(buf_mtp_hidden2_.bf16(), mtp_layer_down_w_.bf16(),
                                    buf_mtp_gate_.bf16(), hidden_size_, intermediate_size_, stream);
            vector_add_bf16_cuda(buf_mtp_hidden_.bf16(), buf_mtp_hidden2_.bf16(), hidden_size_, stream);
        }

        // 6. Final norm
        rms_norm_one_centered_cuda(buf_mtp_hidden_.bf16(), buf_mtp_hidden_.bf16(),
                                   mtp_norm_w_.bf16(), hidden_size_, rms_eps_, stream);

        int active_vocab = use_draft_vocab_ ? draft_vocab_size_ : full_vocab_size_;
        const __nv_bfloat16* active_head = use_draft_vocab_ ? draft_lm_head_w_.bf16() : full_lm_head_w_;

        // 7. lm_head: hidden @ lm_head.T → [active_vocab] F32
        gemv_bf16_cuda(buf_mtp_logits_.f32(), active_head, buf_mtp_hidden_.bf16(),
                       active_vocab, hidden_size_, stream);

        // 8. Argmax over active vocab
        argmax_f32_cuda(buf_mtp_argmax_.i32(), buf_mtp_logits_.f32(), active_vocab, stream);

        // 9. Copy argmax to host
        CUDA_CHECK(cudaMemcpyAsync(h_draft_best_, buf_mtp_argmax_.i32(), sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // 10. Return full_vocab_id
        int best_idx = *h_draft_best_;
        if (best_idx < 0 || best_idx >= active_vocab) return -1;
        int predicted_id = use_draft_vocab_ ? draft_vocab_ids_[best_idx] : best_idx;
        if (predicted_id < 0 || predicted_id >= full_vocab_size_) return -1;
        return predicted_id;
    }

    // Draft K tokens using the MTP module
    // target_hidden: [hidden_size] BF16 — hidden state from target model after final norm
    // first_token: the token to start drafting from (the last sampled token)
    // position: current sequence position (where first_token will be placed)
    // K: number of draft tokens to generate
    // cands: output vector of candidate tokens [first_token, draft_1, draft_2, ..., draft_K]
    // target_embed: pointer to target model's embedding table
    void draft_k_tokens(const __nv_bfloat16* target_hidden, int first_token,
                        int position, int K, std::vector<int>& cands,
                        cudaStream_t stream) {
        cands.clear();
        cands.push_back(first_token);
        if (!loaded_ || K <= 0) return;

        // The MTP module takes:
        // - The target model's hidden state (after the final norm of the last layer)
        // - The embedding of the previously predicted token
        // And produces a draft token prediction

        // For chained drafting (K > 1):
        // Step 0: hidden = target_hidden, token = first_token → draft_1
        // Step 1: hidden = mtp_hidden (from step 0), token = draft_1 → draft_2
        // Step 2: hidden = mtp_hidden (from step 1), token = draft_2 → draft_3
        // ...

        const __nv_bfloat16* current_hidden = target_hidden;
        int current_token = first_token;

        for (int k = 0; k < K; k++) {
            int draft_token = forward_one_step(current_hidden, current_token, position + 1 + k, stream);
            if (draft_token < 0) break;

            cands.push_back(draft_token);
            current_hidden = buf_mtp_hidden_.bf16();  // Use the MTP's own hidden state for next step
            current_token = draft_token;
        }
    }

    // Forward one step returning top-2 candidate tokens
    std::pair<int, int> forward_one_step_top2(const __nv_bfloat16* target_hidden, int last_token_id,
                                              int position, cudaStream_t stream) {
        if (!loaded_) return {-1, -1};

        auto matmul_proj = [&](GPUTensor& out, __nv_bfloat16* in_vec, GPUTensor& weight, GPUTensor& scale, int N, int K) {
            if (weight.dtype == "int4") {
                gemv_int4_cuda(out.bf16(), in_vec, (const uint8_t*)weight.data, scale.bf16(), N, K, stream);
            } else if (weight.dtype == "fp4") {
                gemv_fp4_cuda(out.bf16(), in_vec, (const uint8_t*)weight.data, (const uint8_t*)scale.data, N, K, stream);
            } else {
                gemv_bf16_out_bf16_cuda(out.bf16(), weight.bf16(), in_vec, N, K, stream);
            }
        };

        // 1. Norm the target hidden state
        rms_norm_one_centered_cuda(buf_mtp_hidden_.bf16(), target_hidden,
                                   mtp_pre_fc_norm_hidden_w_.bf16(), hidden_size_, rms_eps_, stream);

        // 2. Embed the last token and norm it
        int32_t tok = last_token_id;
        CUDA_CHECK(cudaMemcpyAsync(buf_mtp_argmax_.i32(), &tok, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        if (is_embed_fp4_) {
            embedding_fp4_cuda(buf_mtp_embed_.bf16(), target_embed_w_fp4_, target_embed_s_fp4_, buf_mtp_argmax_.i32(), 1, hidden_size_, stream);
        } else if (is_embed_int4_) {
            embedding_int4_cuda(buf_mtp_embed_.bf16(), target_embed_w_int4_, target_embed_s_, buf_mtp_argmax_.i32(), 1, hidden_size_, stream);
        } else {
            embedding_cuda(buf_mtp_embed_.bf16(), target_embed_w_, buf_mtp_argmax_.i32(), 1, hidden_size_, stream);
        }
        rms_norm_one_centered_cuda(buf_mtp_embed_.bf16(), buf_mtp_embed_.bf16(),
                                   mtp_pre_fc_norm_embed_w_.bf16(), hidden_size_, rms_eps_, stream);

        // 3. Concatenate: [normed_embed, normed_hidden] -> [10240]
        CUDA_CHECK(cudaMemcpyAsync(buf_mtp_concat_.bf16(), buf_mtp_embed_.bf16(),
                                   hidden_size_ * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(buf_mtp_concat_.bf16() + hidden_size_, buf_mtp_hidden_.bf16(),
                                   hidden_size_ * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));

        // 4. FC projection
        if (mtp_fc_w_.dtype == "int4") {
            gemv_int4_cuda(buf_mtp_hidden_.bf16(), buf_mtp_concat_.bf16(),
                           (const uint8_t*)mtp_fc_w_.data, mtp_fc_s_.bf16(),
                           hidden_size_, 2 * hidden_size_, stream);
        } else if (mtp_fc_w_.dtype == "fp4") {
            gemv_fp4_cuda(buf_mtp_hidden_.bf16(), buf_mtp_concat_.bf16(),
                          (const uint8_t*)mtp_fc_w_.data, (const uint8_t*)mtp_fc_s_.data,
                          hidden_size_, 2 * hidden_size_, stream);
        } else {
            gemv_bf16_out_bf16_cuda(buf_mtp_hidden_.bf16(), mtp_fc_w_.bf16(), buf_mtp_concat_.bf16(),
                                    hidden_size_, 2 * hidden_size_, stream);
        }

        // 5. MTP Transformer layer
        rms_norm_one_centered_cuda(buf_mtp_hidden2_.bf16(), buf_mtp_hidden_.bf16(),
                                   mtp_layer_attn_norm_w_.bf16(), hidden_size_, rms_eps_, stream);

        matmul_proj(buf_mtp_q_, buf_mtp_hidden2_.bf16(), mtp_layer_q_proj_w_, mtp_layer_q_proj_s_,
                    2 * num_heads_ * head_dim_, hidden_size_);
        matmul_proj(buf_mtp_k_, buf_mtp_hidden2_.bf16(), mtp_layer_k_proj_w_, mtp_layer_k_proj_s_,
                    num_kv_heads_ * head_dim_, hidden_size_);
        matmul_proj(buf_mtp_v_, buf_mtp_hidden2_.bf16(), mtp_layer_v_proj_w_, mtp_layer_v_proj_s_,
                    num_kv_heads_ * head_dim_, hidden_size_);

        *h_draft_pos_ = position;
        CUDA_CHECK(cudaMemcpyAsync(buf_mtp_input_pos_.i32(), h_draft_pos_, sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream));
#if defined(__APPLE__)
        qwen_gqa_decode_gated_cuda(
            buf_mtp_attn_out_.bf16(), buf_mtp_q_.bf16(), buf_mtp_k_.bf16(), buf_mtp_v_.bf16(),
            mtp_layer_q_norm_w_.bf16(), mtp_layer_k_norm_w_.bf16(),
            mtp_k_cache_.bf16(), mtp_v_cache_.bf16(),
            num_heads_, num_kv_heads_, head_dim_,
            buf_mtp_input_pos_.i32(), position, max_seq_len_ > 0 ? max_seq_len_ : 32768,
            10000000.0f, rms_eps_, stream, /*cache_pos=*/-1);
#else
        qwen_gqa_decode_gated_cuda(
            buf_mtp_attn_out_.bf16(), buf_mtp_q_.bf16(), buf_mtp_k_.bf16(), buf_mtp_v_.bf16(),
            mtp_layer_q_norm_w_.bf16(), mtp_layer_k_norm_w_.bf16(),
            mtp_k_cache_.bf16(), mtp_v_cache_.bf16(),
            num_heads_, num_kv_heads_, head_dim_,
            buf_mtp_input_pos_.i32(), position, max_seq_len_ > 0 ? max_seq_len_ : 32768,
            10000000.0f, rms_eps_, stream);
#endif

        if (mtp_layer_o_proj_w_.dtype == "int4") {
            gemv_int4_residual_cuda(buf_mtp_hidden_.bf16(), buf_mtp_attn_out_.bf16(),
                                    (const uint8_t*)mtp_layer_o_proj_w_.data, mtp_layer_o_proj_s_.bf16(),
                                    hidden_size_, num_heads_ * head_dim_, stream);
        } else if (mtp_layer_o_proj_w_.dtype == "fp4") {
            gemv_fp4_residual_cuda(buf_mtp_hidden_.bf16(), buf_mtp_attn_out_.bf16(),
                                   (const uint8_t*)mtp_layer_o_proj_w_.data, (const uint8_t*)mtp_layer_o_proj_s_.data,
                                   hidden_size_, num_heads_ * head_dim_, stream);
        } else {
            gemv_bf16_out_bf16_cuda(buf_mtp_hidden2_.bf16(), mtp_layer_o_proj_w_.bf16(),
                                    buf_mtp_attn_out_.bf16(), hidden_size_, num_heads_ * head_dim_, stream);
            vector_add_bf16_cuda(buf_mtp_hidden_.bf16(), buf_mtp_hidden2_.bf16(), hidden_size_, stream);
        }

        rms_norm_one_centered_cuda(buf_mtp_hidden2_.bf16(), buf_mtp_hidden_.bf16(),
                                   mtp_layer_post_attn_norm_w_.bf16(), hidden_size_, rms_eps_, stream);

        if (mtp_layer_gate_w_.dtype == "int4") {
            gemv_int4_swiglu_fused_cuda(buf_mtp_gate_.bf16(), buf_mtp_hidden2_.bf16(),
                                        (const uint8_t*)mtp_layer_gate_w_.data, mtp_layer_gate_s_.bf16(),
                                        (const uint8_t*)mtp_layer_up_w_.data, mtp_layer_up_s_.bf16(),
                                        intermediate_size_, hidden_size_, 0.0f, stream);
        } else if (mtp_layer_gate_w_.dtype == "fp4") {
            gemv_fp4_swiglu_fused_cuda(buf_mtp_gate_.bf16(), buf_mtp_hidden2_.bf16(),
                                        (const uint8_t*)mtp_layer_gate_w_.data, (const uint8_t*)mtp_layer_gate_s_.data,
                                        (const uint8_t*)mtp_layer_up_w_.data, (const uint8_t*)mtp_layer_up_s_.data,
                                        intermediate_size_, hidden_size_, 0.0f, stream);
        } else {
            gemv_bf16_out_bf16_cuda(buf_mtp_gate_.bf16(), mtp_layer_gate_w_.bf16(),
                                    buf_mtp_hidden2_.bf16(), intermediate_size_, hidden_size_, stream);
            gemv_bf16_out_bf16_cuda(buf_mtp_up_.bf16(), mtp_layer_up_w_.bf16(),
                                    buf_mtp_hidden2_.bf16(), intermediate_size_, hidden_size_, stream);
            silu_mul_cuda(buf_mtp_gate_.bf16(), buf_mtp_gate_.bf16(), buf_mtp_up_.bf16(), intermediate_size_, 0.0f, stream);
        }

        if (mtp_layer_down_w_.dtype == "int4") {
            gemv_int4_residual_cuda(buf_mtp_hidden_.bf16(), buf_mtp_gate_.bf16(),
                                    (const uint8_t*)mtp_layer_down_w_.data, mtp_layer_down_s_.bf16(),
                                    hidden_size_, intermediate_size_, stream);
        } else if (mtp_layer_down_w_.dtype == "fp4") {
            gemv_fp4_residual_cuda(buf_mtp_hidden_.bf16(), buf_mtp_gate_.bf16(),
                                   (const uint8_t*)mtp_layer_down_w_.data, (const uint8_t*)mtp_layer_down_s_.data,
                                   hidden_size_, intermediate_size_, stream);
        } else {
            gemv_bf16_out_bf16_cuda(buf_mtp_hidden2_.bf16(), mtp_layer_down_w_.bf16(),
                                    buf_mtp_gate_.bf16(), hidden_size_, intermediate_size_, stream);
            vector_add_bf16_cuda(buf_mtp_hidden_.bf16(), buf_mtp_hidden2_.bf16(), hidden_size_, stream);
        }

        // 6. Final norm
        rms_norm_one_centered_cuda(buf_mtp_hidden_.bf16(), buf_mtp_hidden_.bf16(),
                                   mtp_norm_w_.bf16(), hidden_size_, rms_eps_, stream);

        int active_vocab = use_draft_vocab_ ? draft_vocab_size_ : full_vocab_size_;
        const __nv_bfloat16* active_head = use_draft_vocab_ ? draft_lm_head_w_.bf16() : full_lm_head_w_;

        // 7. lm_head
        gemv_bf16_cuda(buf_mtp_logits_.f32(), active_head, buf_mtp_hidden_.bf16(),
                       active_vocab, hidden_size_, stream);

        // 8. Top-2 over active vocab
        top2_f32_cuda(buf_mtp_top2_idx_.i32(), buf_mtp_top2_val_.f32(), buf_mtp_logits_.f32(), active_vocab, stream);

        // 9. Copy top-2 to host
        CUDA_CHECK(cudaMemcpyAsync(h_draft_top2_, buf_mtp_top2_idx_.i32(), 2 * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        int idx1 = h_draft_top2_[0];
        int idx2 = h_draft_top2_[1];

        int tok1 = (idx1 >= 0 && idx1 < active_vocab) ? (use_draft_vocab_ ? draft_vocab_ids_[idx1] : idx1) : -1;
        int tok2 = (idx2 >= 0 && idx2 < active_vocab) ? (use_draft_vocab_ ? draft_vocab_ids_[idx2] : idx2) : -1;

        return {tok1, tok2};
    }

    // Forward tree drafting for candidate nodes (configurable: 1, 2, or 3 candidate nodes)
    bool draft_tree_tokens(const __nv_bfloat16* target_hidden, int first_token,
                           int position, struct TreeTopology& tree, cudaStream_t stream, int tree_nodes = 2);
};

// ════════════════════════════════════════════════════════════════════════════════
//  Speculative Decoding Tree Topology & Longest Path Resolver
// ════════════════════════════════════════════════════════════════════════════════

struct TreeTopology {
    int num_nodes = 0;                  // Total candidate nodes (e.g. 3 or 4)
    int depth[16] = {0};                // Depth in tree (0 = root)
    int parent[16] = {-1};              // Parent index in candidate array (-1 for root)
    uint64_t ancestor_mask[16] = {0};   // Bit j is 1 if candidate j is an ancestor of candidate i
    int tokens[16] = {0};               // Predicted candidate token IDs
};

struct TreeVerificationResult {
    int accepted_count = 0;             // Number of speculative tokens accepted (0..3)
    std::vector<int> accepted_tokens;   // The accepted tokens
    std::vector<int> winning_slots;     // Slot indices of the winning path (for KV compaction)
    int bonus_token = -1;               // The next token predicted by the base model
    int last_accepted_node = 0;         // Node index of deepest accepted node (for SSM state commit)
};

inline bool MTPSelfDrafter::draft_tree_tokens(const __nv_bfloat16* target_hidden, int first_token,
                                              int position, TreeTopology& tree, cudaStream_t stream, int tree_nodes) {
    if (!loaded_) return false;

    // Node 0: Root (already accepted first_token)
    tree.depth[0] = 0;
    tree.parent[0] = -1;
    tree.ancestor_mask[0] = (1ULL << 0);
    tree.tokens[0] = first_token;

    if (tree_nodes <= 1) {
        int t1 = forward_one_step(target_hidden, first_token, position + 1, stream);
        if (t1 < 0) return false;
        tree.num_nodes = 2;
        tree.depth[1] = 1; tree.parent[1] = 0; tree.ancestor_mask[1] = (1ULL << 0) | (1ULL << 1); tree.tokens[1] = t1;
        return true;
    }

    // Step 1: from root (target_hidden, first_token) -> top-2 (Node 1, Node 2)
    auto pair1 = forward_one_step_top2(target_hidden, first_token, position + 1, stream);
    if (pair1.first < 0) return false;
    int t1 = pair1.first;
    int t2 = (pair1.second >= 0 && pair1.second != t1) ? pair1.second : pair1.first;

    tree.depth[1] = 1; tree.parent[1] = 0; tree.ancestor_mask[1] = (1ULL << 0) | (1ULL << 1); tree.tokens[1] = t1;
    tree.depth[2] = 1; tree.parent[2] = 0; tree.ancestor_mask[2] = (1ULL << 0) | (1ULL << 2); tree.tokens[2] = t2;

    if (tree_nodes == 2) {
        // 2 alternative branches: Top-1 and Top-2 (no depth-2 continuation)
        tree.num_nodes = 3;
        return true;
    }

    // Step 2: from Node 1 (buf_mtp_hidden_, t1) -> top-1 (Node 3)
    int t3 = forward_one_step(buf_mtp_hidden_.bf16(), t1, position + 2, stream);
    if (t3 < 0) t3 = t1;

    tree.num_nodes = 4;
    tree.depth[3] = 2; tree.parent[3] = 1; tree.ancestor_mask[3] = (1ULL << 0) | (1ULL << 1) | (1ULL << 3); tree.tokens[3] = t3;

    return true;
}

static inline TreeVerificationResult resolve_tree_longest_path(
    const TreeTopology& tree,
    const int32_t* host_batch_preds,
    int eos_id,
    int eos2_id,
    int think_start_id,
    int think_end_id,
    bool enable_thinking)
{
    TreeVerificationResult res;
    int p0 = host_batch_preds[0];

    auto is_valid_token = [&](int tok) {
        if (tok < 0) return false;
        if (tok == eos_id || (eos2_id >= 0 && tok == eos2_id)) return false;
        if (!enable_thinking && (tok == think_start_id || tok == think_end_id)) return false;
        return true;
    };

    if (p0 == tree.tokens[1] && is_valid_token(tree.tokens[1])) {
        // Path A: Node 1 accepted
        res.accepted_count = 1;
        res.accepted_tokens.push_back(tree.tokens[1]);
        res.winning_slots.push_back(1);
        res.last_accepted_node = 1;
        res.bonus_token = host_batch_preds[1];

        int p1 = host_batch_preds[1];
        if (tree.num_nodes >= 4 && p1 == tree.tokens[3] && is_valid_token(tree.tokens[3])) {
            // Path A extended: Node 3 accepted
            res.accepted_count = 2;
            res.accepted_tokens.push_back(tree.tokens[3]);
            res.winning_slots.push_back(3);
            res.last_accepted_node = 3;
            res.bonus_token = host_batch_preds[3];
        }
    } else if (tree.num_nodes > 2 && p0 == tree.tokens[2] && is_valid_token(tree.tokens[2])) {
        // Path B: Node 2 accepted
        res.accepted_count = 1;
        res.accepted_tokens.push_back(tree.tokens[2]);
        res.winning_slots.push_back(2);
        res.last_accepted_node = 2;
        res.bonus_token = host_batch_preds[2];
    } else {
        // Fallback: accept root only
        res.accepted_count = 0;
        res.bonus_token = p0;
        res.last_accepted_node = 0;
    }

    return res;
}



