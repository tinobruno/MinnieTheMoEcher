#pragma once

#include "common.hpp"
#include "logging.hpp"
#include "activations.cuh"
#include "expert_loader.hpp"
#include "speculative.hpp"
#include "tokenizer.hpp"
#include "multimodal.hpp"
#include "zdb.hpp"
#include "vision_tower.hpp"
#include "image_loader.hpp"
#include "tool_exec.hpp"
#include "thread_pool.h"
#include "platform/platform_io.hpp"

#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <unordered_map>
#include <map>
#include <mutex>
#include <atomic>
#include <random>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ════════════════════════════════════════════════════════════════════════════════
//  Model Engine
// ════════════════════════════════════════════════════════════════════════════════

class MoecherEngine {
public:
    ModelConfig cfg_;
    BPETokenizer tokenizer_;
    std::string model_dir_;
    std::string manifest_path_;

    std::string get_model_id() const {
        if (!cfg_.model_id.empty()) return cfg_.model_id;
        if (cfg_.architecture == ModelArch::QWEN2) {
            return "qwen2.5-vl-3b";
        }
        if (cfg_.architecture == ModelArch::QWEN) {
            if (model_dir_.find("q4") != std::string::npos || manifest_path_.find("q4") != std::string::npos) {
                return "qwen3.8-27b-q4";
            }
            return "qwen3.8-27b";
        }
        if (model_dir_.find("coder") != std::string::npos || manifest_path_.find("coder") != std::string::npos) {
            return "deepseek-coder-v2-lite";
        }
        if (model_dir_.find("q4") != std::string::npos || manifest_path_.find("q4") != std::string::npos) {
            return "deepseek-v4-flash-q4";
        }
        return "deepseek-v4-flash";
    }

    std::string get_model_display_name() const {
        if (!cfg_.model_name.empty()) return cfg_.model_name;
        if (cfg_.architecture == ModelArch::QWEN2) {
            return "Qwen 2.5 VL 3B";
        }
        if (cfg_.architecture == ModelArch::QWEN) {
            if (model_dir_.find("q4") != std::string::npos || manifest_path_.find("q4") != std::string::npos) {
                return "Qwen 3.8 27B Q4";
            }
            return "Qwen 3.8 27B";
        }
        if (model_dir_.find("coder") != std::string::npos || manifest_path_.find("coder") != std::string::npos) {
            return "DeepSeek Coder V2 Lite";
        }
        if (model_dir_.find("q4") != std::string::npos || manifest_path_.find("q4") != std::string::npos) {
            return "DeepSeek V4-Flash Q4";
        }
        return "DeepSeek V4-Flash";
    }
    
    // Qwen 3.8 2B Neural Speculative Draft Engine (legacy)
    QwenDraftEngine2B qwen_draft_;

    // MTP Self-Drafter (new, faster)
    MTPSelfDrafter mtp_drafter_;

    // Qwen Vision Tower (Multimodal)
    bool enable_vision_ = true;
    bool has_vision_ = false;
    std::unique_ptr<moecher::vision::QwenVisionTower> vision_tower_;
    const __nv_bfloat16* active_visual_embeddings_ = nullptr;
    int visual_embed_start_pos_ = -1;
    int visual_embed_num_tokens_ = 0;

    // On-demand Vision Tower loader state (zero VRAM consumption during text chat)
    struct VisionTowerConfig {
        bool available = false;
        std::string dense_path;
        json tensor_map;
        std::string vision_bin;
        std::string bridge_bin;
        std::string embed_mean_bin;
        int output_dim = 0;
    } vision_tower_cfg_;

    bool ensure_vision_loaded() {
        if (vision_tower_ && vision_tower_->is_loaded()) return true;
        if (!vision_tower_cfg_.available) return false;
        LOG_INFO("[Vision] Multimodal image detected. Loading Vision Tower on-demand (~990 MB)...");
        vision_tower_ = std::make_unique<moecher::vision::QwenVisionTower>();
        if (vision_tower_->load_from_manifest(
                vision_tower_cfg_.dense_path,
                vision_tower_cfg_.tensor_map,
                cublas_handle_,
                main_stream_,
                vision_tower_cfg_.output_dim,
                vision_tower_cfg_.vision_bin,
                vision_tower_cfg_.bridge_bin,
                vision_tower_cfg_.embed_mean_bin)) {
            LOG_INFO("[Vision] Vision Tower successfully loaded on-demand! (num_blocks=%d, 576 tokens output x %d dim)",
                     vision_tower_->num_blocks_, vision_tower_cfg_.output_dim);
            return true;
        } else {
            LOG_WARN("[Vision] Failed to load Vision Tower weights on-demand.");
            vision_tower_.reset();
            return false;
        }
    }

    void unload_vision_tower() {
        if (vision_tower_) {
            LOG_INFO("[Vision] Unloading Vision Tower from VRAM to preserve full headroom for LLM decode...");
            vision_tower_.reset();
        }
    }

    // In-process Vision Delegate Engine (Multimodal Delegation)
    std::unique_ptr<MoecherEngine> vision_delegate_engine_;
    std::string delegate_manifest_path_;
    bool has_vision_delegate() const { return !delegate_manifest_path_.empty() || vision_delegate_engine_ != nullptr; }

    void refresh_resident_pointers() {
        if (cfg_.is_nemotron() && buf_all_expert_ptrs_.data) {
            int total_moe_layers = 0;
            for (auto t : cfg_.layer_types) if (t == 2) total_moe_layers++;
            if (total_moe_layers == 0) total_moe_layers = cfg_.is_nemotron_bolt() ? 40 : 23;
            int max_routed_exp = std::max({cfg_.n_routed_experts, cfg_.max_n_routed_experts(), 128});

            std::vector<void*> h_all_ptrs((size_t)total_moe_layers * max_routed_exp, nullptr);
            for (int l = 0; l < total_moe_layers && l < 64; l++) {
                bool all_in = true;
                int n_e = cfg_.get_moe_layer_n_routed_experts(l);
                if (n_e <= 0) n_e = 128;
                for (int e = 0; e < n_e && e < max_routed_exp; e++) {
                    h_all_ptrs[(size_t)l * max_routed_exp + e] = expert_loader_.try_get_expert_cached(l, e);
                    if (!h_all_ptrs[(size_t)l * max_routed_exp + e]) all_in = false;
                }
                layer_fully_resident_[l] = all_in;
            }
            CUDA_CHECK(cudaMemcpy(buf_all_expert_ptrs_.data, h_all_ptrs.data(),
                                  (size_t)total_moe_layers * max_routed_exp * sizeof(void*), cudaMemcpyHostToDevice));
        }
    }

    bool has_vision() const { return has_vision_; }
    bool is_vision_loaded() const { return vision_tower_ != nullptr; }
    bool supports_vision() const { return has_vision() || has_vision_delegate(); }

    // Prompt-Lookup Drafting (PLD) Speculative Decoding
    bool enable_pld_ = true;
    int pld_draft_tokens_ = 4;
    bool enable_mtp_ = true;
    bool default_enable_mtp_ = true;
    bool enable_tree_mtp_ = false;
    bool default_enable_tree_mtp_ = false;
    int tree_nodes_ = 2; // 2 = 2 alternative branches (Top-1 and Top-2, M=3 total nodes)
    int default_tree_nodes_ = 2;
    int tree_graph_M_ = 3;
    int mtp_k_ = 0; // 0 = auto (1 for 16GB GPUs, 4 for >= 24GB GPUs)
    int default_mtp_k_ = 0;
    bool enable_spec_ = true;
    bool default_enable_spec_ = true;
    
    // Prompt Prefix KV Cache Tracking
    std::vector<int> cached_tokens_;

    // Hardware Detection & Feature Gating
    GpuCapabilities gpu_caps_;
    bool is_blackwell_tc_active_ = false;
    const GpuCapabilities& gpu_capabilities() const { return gpu_caps_; }

    void reset_all_kv_caches() {
        for (int l = 0; l < cfg_.num_hidden_layers; l++) {
            if (layers_[l].kv_cache.data) {
                CUDA_CHECK(cudaMemset(layers_[l].kv_cache.data, 0,
                                       layers_[l].kv_cache.size_bytes));
            }
            if (layers_[l].k_cache_gqa.data) {
                CUDA_CHECK(cudaMemset(layers_[l].k_cache_gqa.data, 0, layers_[l].k_cache_gqa.size_bytes));
                CUDA_CHECK(cudaMemset(layers_[l].v_cache_gqa.data, 0, layers_[l].v_cache_gqa.size_bytes));
            }
            if (layers_[l].ssm_state.data) {
                CUDA_CHECK(cudaMemset(layers_[l].ssm_state.data, 0, layers_[l].ssm_state.size_bytes));
            }
            if (layers_[l].conv_state.data) {
                CUDA_CHECK(cudaMemset(layers_[l].conv_state.data, 0, layers_[l].conv_state.size_bytes));
            }
            if (layers_[l].d_comp_kv_count.data) {
                CUDA_CHECK(cudaMemset(layers_[l].d_comp_kv_count.data, 0, sizeof(int32_t)));
            }
            if (layers_[l].d_attn_cache_len.data) {
                CUDA_CHECK(cudaMemset(layers_[l].d_attn_cache_len.data, 0, sizeof(int32_t)));
            }
            int ratio = cfg_.layer_compress_ratio(l);
            if (ratio > 0) {
                layers_[l].comp_kv_count = 0;
                if (layers_[l].comp_kv_cache.data)
                    CUDA_CHECK(cudaMemset(layers_[l].comp_kv_cache.data, 0,
                                           layers_[l].comp_kv_cache.size_bytes));
                if (layers_[l].comp_kv_state.data)
                    CUDA_CHECK(cudaMemset(layers_[l].comp_kv_state.data, 0,
                                           layers_[l].comp_kv_state.size_bytes));
                if (layers_[l].comp_score_state.data) {
                    int coff = (ratio == 4) ? 2 : 1;
                    int state_rows = coff * ratio;
                    int state_cols = coff * cfg_.head_dim;
                    std::vector<float> neg_inf(state_rows * state_cols,
                                               -std::numeric_limits<float>::infinity());
                    CUDA_CHECK(cudaMemcpy(layers_[l].comp_score_state.data, neg_inf.data(),
                                           neg_inf.size() * sizeof(float), cudaMemcpyHostToDevice));
                }
                if (ratio == 4) {
                    if (layers_[l].indexer_comp_kv_cache.data)
                        CUDA_CHECK(cudaMemset(layers_[l].indexer_comp_kv_cache.data, 0,
                                               layers_[l].indexer_comp_kv_cache.size_bytes));
                    if (layers_[l].indexer_comp_kv_state.data)
                        CUDA_CHECK(cudaMemset(layers_[l].indexer_comp_kv_state.data, 0,
                                               layers_[l].indexer_comp_kv_state.size_bytes));
                    if (layers_[l].indexer_comp_score_state.data) {
                        int indexer_head_dim = 128;
                        int indexer_proj_dim = 2 * indexer_head_dim;
                        int indexer_state_rows = 2 * ratio;
                        std::vector<float> neg_inf_idx(indexer_state_rows * indexer_proj_dim,
                                                       -std::numeric_limits<float>::infinity());
                        CUDA_CHECK(cudaMemcpy(layers_[l].indexer_comp_score_state.data, neg_inf_idx.data(),
                                               neg_inf_idx.size() * sizeof(float), cudaMemcpyHostToDevice));
                    }
                }
            }
        }

        if (buf_hc_state_.data) {
            CUDA_CHECK(cudaMemset(buf_hc_state_.data, 0, buf_hc_state_.size_bytes));
        }
        if (buf_hc_after_attn_.data) {
            CUDA_CHECK(cudaMemset(buf_hc_after_attn_.data, 0, buf_hc_after_attn_.size_bytes));
        }
        if (target_ssm_pool_.data) {
            CUDA_CHECK(cudaMemsetAsync(target_ssm_pool_.data, 0, target_ssm_pool_.size_bytes, main_stream_));
            CUDA_CHECK(cudaMemsetAsync(target_conv_pool_.data, 0, target_conv_pool_.size_bytes, main_stream_));
            for (int s = 0; s < 8; s++) {
                if (target_ssm_slots_[s].data) CUDA_CHECK(cudaMemsetAsync(target_ssm_slots_[s].data, 0, target_ssm_slots_[s].size_bytes, main_stream_));
                if (target_conv_slots_[s].data) CUDA_CHECK(cudaMemsetAsync(target_conv_slots_[s].data, 0, target_conv_slots_[s].size_bytes, main_stream_));
            }
        }
        if (qwen_draft_.loaded_) {
            qwen_draft_.reset_state(main_stream_);
        }
        if (mtp_drafter_.loaded_) {
            mtp_drafter_.reset_kv_cache(main_stream_);
        }
        cached_tokens_.clear();
        argmax_cache_valid_ = false;
    }

    // Thread pool for loading experts (max 16 concurrent reads)
    std::unique_ptr<ThreadPool> expert_pool_;
    bool dbg_first_token_ = false;
    int dbg_hc_pre_call_ = 0;
    bool dbg_head_ = true;
    int dbg_sample_count_ = 0;

    cudaStream_t main_stream_ = nullptr;
    cudaStream_t side_stream_ = nullptr;
    cudaStream_t dma_stream_ = nullptr;
    cudaEvent_t side_event_ = nullptr;
    cudaEvent_t main_event_ = nullptr;
    cudaEvent_t dma_event_ = nullptr;
    cudaStream_t expert_streams_[64] = {nullptr};
    cudaEvent_t expert_events_[64] = {nullptr};
    cudaStream_t dequant_stream_main_ = nullptr;
    cudaStream_t dequant_stream_side_ = nullptr;
    cudaEvent_t dequant_ready_event_[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
    cudaEvent_t gemm_done_event_[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};

    void* streaming_slots_gpu_[2][32] = {{nullptr}};
    int streaming_buf_idx_ = 0;

    cublasHandle_t cublas_handle_ = nullptr;
    cublasHandle_t cublas_side_handle_ = nullptr;
    cublasLtHandle_t cublaslt_handle_ = nullptr;
    cublasLtHandle_t cublaslt_side_handle_ = nullptr;
    void* cublaslt_workspace_ = nullptr;
    void* cublaslt_workspace_side_ = nullptr;
    size_t cublaslt_workspace_size_ = 4 * 1024 * 1024;
    bool is_blackwell_fp8_active_ = false;
    ExpertLoader expert_loader_;

    struct CublasLtCachedPlan {
        int M = 0;
        int N = 0;
        int K = 0;
        cublasLtMatmulDesc_t op_desc = nullptr;
        cublasLtMatrixLayout_t a_desc = nullptr;
        cublasLtMatrixLayout_t b_desc = nullptr;
        cublasLtMatrixLayout_t c_desc = nullptr;
        cublasLtMatmulPreference_t pref = nullptr;
        cublasLtMatmulHeuristicResult_t heuristic{};
        bool has_heuristic = false;

        void destroy() {
            if (a_desc) { cublasLtMatrixLayoutDestroy(a_desc); a_desc = nullptr; }
            if (b_desc) { cublasLtMatrixLayoutDestroy(b_desc); b_desc = nullptr; }
            if (c_desc) { cublasLtMatrixLayoutDestroy(c_desc); c_desc = nullptr; }
            if (op_desc) { cublasLtMatmulDescDestroy(op_desc); op_desc = nullptr; }
            if (pref) { cublasLtMatmulPreferenceDestroy(pref); pref = nullptr; }
            has_heuristic = false;
            M = N = K = 0;
        }
    };
    static constexpr int NUM_CUBLASLT_CACHED_PLANS = 8;
    CublasLtCachedPlan cublaslt_plans_[NUM_CUBLASLT_CACHED_PLANS];
    int cublaslt_plan_next_evict_ = 0;

    // ── Resident GPU tensors (loaded at startup) ────────────────────────────

    // Embedding & head
    GPUTensor embed_weight_, embed_weight_scale_;   // [vocab_size, hidden_size] BF16 or INT4
    GPUTensor head_weight_, head_weight_scale_; // [vocab_size, hidden_size] (BF16 or INT4 for logits)
    GPUTensor norm_weight_;    // [hidden_size] BF16

    // Per-layer resident tensors
    struct LayerWeights {
        // Attention
        GPUTensor wq_a_w, wq_a_s;     // Low-rank Q down: [q_lora_rank, hidden]
        GPUTensor wq_b_w, wq_b_s;     // Low-rank Q up: [n_heads*head_dim, q_lora_rank]
        GPUTensor wkv_w, wkv_s;       // KV projection: [head_dim, hidden]
        GPUTensor wo_a_w, wo_a_s;     // Low-rank O down: [o_groups*o_lora_rank, n_heads*head_dim/o_groups]
        GPUTensor wo_b_w, wo_b_s;     // Low-rank O up: [hidden, o_groups*o_lora_rank]
        GPUTensor q_norm_w;            // [q_lora_rank]
        GPUTensor kv_norm_w;           // [head_dim]
        GPUTensor attn_norm_w;         // [hidden]
        GPUTensor attn_sink;           // [n_heads] F32
        GPUTensor ffn_norm_w;          // [hidden]

        // Compressor weights (for layers with compress_ratio > 0)
        GPUTensor comp_wkv;       // [coff*head_dim, hidden_size] BF16
        GPUTensor comp_wgate;     // [coff*head_dim, hidden_size] BF16
        GPUTensor comp_ape;       // [ratio, coff*head_dim] F32
        GPUTensor comp_norm;      // [head_dim] BF16

        // Compressed KV cache (for layers with compress_ratio > 0)
        GPUTensor comp_kv_cache;  // [max_compressed_entries, head_dim] BF16
        int comp_kv_count = 0;    // number of compressed entries written so far
        GPUTensor d_comp_kv_count;   // [1] int32_t on GPU for graph/device kernels
        GPUTensor d_attn_cache_len;  // [1] int32_t on GPU for dynamic MLA attention

        // Compressor state buffers (GPU, F32) for incremental decode compression
        GPUTensor comp_kv_state;     // [coff * ratio, coff * head_dim] F32
        GPUTensor comp_score_state;  // [coff * ratio, coff * head_dim] F32

        // Indexer (for layers with compress_ratio == 4)
        GPUTensor indexer_comp_wkv;
        GPUTensor indexer_comp_wgate;
        GPUTensor indexer_comp_ape;
        GPUTensor indexer_comp_norm;
        GPUTensor indexer_weights_proj;
        GPUTensor indexer_wq_b_w, indexer_wq_b_s;
        GPUTensor indexer_comp_kv_cache;  // [max_compressed_entries, 128] BF16
        GPUTensor indexer_comp_kv_state;     // [2 * ratio, 2 * 128] F32
        GPUTensor indexer_comp_score_state;  // [2 * ratio, 2 * 128] F32

        // Gate
        GPUTensor gate_w;              // [n_experts, hidden] BF16
        GPUTensor gate_bias;           // [n_experts] F32 (null for hash layers)
        std::vector<float> gate_bias_host;
        GPUTensor tid2eid;             // [vocab_size, top_k] I64 (only hash layers)
        std::vector<int64_t> tid2eid_host; // Host cached for 0-latency CPU lookup

        // Shared expert (FP8)
        GPUTensor shared_w1_w, shared_w1_s;
        GPUTensor shared_w2_w, shared_w2_s;
        GPUTensor shared_w3_w, shared_w3_s;

        // HC (Hyper-Connection) parameters
        GPUTensor hc_attn_fn;     // [(2+hc)*hc, hc*hidden] F32
        GPUTensor hc_attn_base;   // [(2+hc)*hc] F32
        GPUTensor hc_attn_scale;  // [3] F32
        GPUTensor hc_ffn_fn;
        GPUTensor hc_ffn_base;
        GPUTensor hc_ffn_scale;

        // KV cache (sliding window)
        GPUTensor kv_cache;       // [window_size, head_dim] BF16

        // Standard GQA & Dense SwiGLU (Qwen / Llama)
        bool is_linear_attn = false;
        GPUTensor w_q, w_q_scale;           // [n_heads * head_dim * 2, hidden]
        GPUTensor w_k, w_k_scale;           // [n_kv_heads * head_dim, hidden]
        GPUTensor w_v, w_v_scale;           // [n_kv_heads * head_dim, hidden]
        GPUTensor w_o, w_o_scale;           // [hidden, n_heads * head_dim]
        GPUTensor gqa_q_norm_w;  // [head_dim] BF16
        GPUTensor gqa_k_norm_w;  // [head_dim] BF16
        GPUTensor gqa_q_bias, gqa_k_bias, gqa_v_bias; // [q_dim], [kv_dim], [kv_dim] BF16
        GPUTensor w_gate, w_gate_scale;        // [intermediate_size, hidden]
        GPUTensor w_up, w_up_scale;          // [intermediate_size, hidden]
        GPUTensor w_down, w_down_scale;        // [hidden, intermediate_size]
        GPUTensor k_cache_gqa;   // [max_seq_len, n_kv_heads, head_dim] BF16
        GPUTensor v_cache_gqa;   // [max_seq_len, n_kv_heads, head_dim] BF16

        // Qwen 3.8 Linear Attention (Gated DeltaNet)
        GPUTensor w_in_qkv, w_in_qkv_scale;       // [10240, 5120]
        GPUTensor w_in_z, w_in_z_scale;         // [6144, 5120]
        GPUTensor w_in_a;         // [48, 5120] BF16
        GPUTensor w_in_b;         // [48, 5120] BF16
        GPUTensor conv1d_w;       // [10240, 1, 4] BF16
        GPUTensor conv1d_bias;    // Conv bias for Nemotron SSM
        GPUTensor mamba_d;        // D skip vector for Nemotron SSM
        GPUTensor A_log;          // [48] BF16
        GPUTensor dt_bias;        // [48] BF16
        GPUTensor linear_norm_w;  // [128] BF16
        GPUTensor linear_out_proj, linear_out_proj_scale;// [5120, 6144]
        float w_in_qkv_scale_val = 1.0f;
        float linear_out_proj_scale_val = 1.0f;
        float shared_w1_s2 = 1.0f;
        float shared_w2_s2 = 1.0f;

        // Nemotron Bolt LatentMoE projections
        GPUTensor fc1_latent_proj, fc1_latent_proj_scale; // [moe_latent_size, hidden_size]
        GPUTensor fc2_latent_proj, fc2_latent_proj_scale; // [hidden_size, moe_latent_size]
        float fc1_scale_val = 1.0f;
        float fc2_scale_val = 1.0f;

        GPUTensor ssm_state;      // [48, 128, 128] BF16 (50% memory cut)
        GPUTensor conv_state;     // [10240, 4] BF16
        GPUTensor ssm_state_slots[8];
        GPUTensor conv_state_slots[8];
    };
    std::vector<LayerWeights> layers_;

    // Head HC parameters
    GPUTensor hc_head_fn_;
    GPUTensor hc_head_base_;
    GPUTensor hc_head_scale_;

    // RoPE frequency tables
    GPUTensor rope_freqs_;              // [MAX_SEQ_LEN, rope_dim/2, 2] F32 — non-compressed layers
    GPUTensor rope_freqs_compressed_;   // [MAX_SEQ_LEN, rope_dim/2, 2] F32 — compressed layers

    // Expert layout info (from manifest)
    struct ExpertPartInfo {
        int offset_in_block;
        int nbytes;
        std::string dtype;
        std::vector<int> shape;
        int offset_data;
        int offset_scales;
    };
    std::map<std::string, ExpertPartInfo> expert_parts_;

    // Working buffers (reused across forward passes)
    GPUTensor buf_active_expert_ptrs_; // [32] void* GPU pointers
    void* active_expert_ptrs_host_[32] = {nullptr};
    int32_t* topk_ids_host_ = nullptr;
    float* topk_weights_host_ = nullptr;
    float head_weight_scale_2_ = 1.0f;
    std::vector<float> nemotron_expert_scales_;
    GPUTensor buf_mamba_conv_; // [6144] BF16 for Nemotron Mamba-2 conv output
    GPUTensor buf_mamba_temp_; // [4096] F32 for Nemotron Mamba-2 intermediate gated output
    GPUTensor buf_hidden_;       // [MAX_SEQ_LEN, hidden_size] BF16
    GPUTensor buf_hidden2_;
    GPUTensor buf_q_;            // [MAX_SEQ_LEN, n_heads*head_dim] BF16
    GPUTensor buf_kv_;           // [MAX_SEQ_LEN, head_dim] BF16
    GPUTensor buf_attn_out_;     // [MAX_SEQ_LEN, n_heads*head_dim] BF16
    GPUTensor buf_lora_;         // temp for lora intermediate
    GPUTensor buf_gate_;         // [MAX_SEQ_LEN, moe_inter] BF16
    GPUTensor buf_up_;           // same
    GPUTensor buf_down_;         // same
    GPUTensor buf_expert_out_;   // [hidden_size] BF16
    GPUTensor buf_moe_accum_;    // [hidden_size] BF16 (accumulated expert output)
    GPUTensor buf_latent_down_;  // [moe_latent_size] BF16 for Nemotron Bolt LatentMoE
    GPUTensor buf_latent_accum_; // [moe_latent_size] BF16 for Nemotron Bolt LatentMoE
    GPUTensor buf_dequant_;      // temp buffer for dequantized weights
    GPUTensor buf_dequant_2_;    // second buffer for concurrent dual-stream prefill
    GPUTensor buf_logits_;       // [vocab_size] F32
    GPUTensor buf_scores_f32_;   // [n_experts] F32
    GPUTensor buf_scores_bf16_;  // [n_experts] BF16
    GPUTensor buf_topk_vals_;    // [top_k] F32
    GPUTensor buf_topk_idx_;     // [top_k] I32
    GPUTensor buf_linear_a_;     // [48] BF16 for Qwen 3.8 DeltaNet decay A
    GPUTensor buf_linear_b_;     // [48] BF16 for Qwen 3.8 DeltaNet decay B
    GPUTensor buf_input_ids_;    // [MAX_SEQ_LEN] I32

    // Batched working buffers for Multi-Token Speculative Decoding (M=4..8)
    GPUTensor buf_input_tokens_batch_; // [8] I32
    GPUTensor buf_input_pos_batch_;    // [8] I32
    GPUTensor buf_hidden_batch_;       // [8, hidden_size] BF16
    GPUTensor buf_hidden2_batch_;      // [8, hidden_size] BF16
    GPUTensor buf_activation_fp4_;     // [16, max_dim / 2] FP4 bytes for Blackwell Tensor Core GEMM
    GPUTensor buf_activation_scale_;   // [16, max_dim / 32] E8M0 scale bytes for Blackwell Tensor Core GEMM
    GPUTensor buf_activation_fp8_;      // [max_batch_m, max_dim] FP8 bytes for Blackwell Tensor Core GEMM
    GPUTensor buf_activation_fp8_side_; // [max_batch_m, max_dim] FP8 bytes for concurrent side-stream GEMM
    GPUTensor buf_q_batch_;            // [8, 10240] BF16
    GPUTensor buf_gate_batch_;         // [8, intermediate_size] BF16
    GPUTensor buf_up_batch_;           // [8, intermediate_size] BF16
    GPUTensor buf_attn_out_batch_;     // [8, 6144] BF16
    GPUTensor buf_linear_a_batch_;     // [8, 48] BF16
    GPUTensor buf_linear_b_batch_;     // [8, 48] BF16
    GPUTensor buf_logits_batch_;       // [8, vocab_size] F32
    GPUTensor buf_argmax_out_batch_;   // [8] I32
    GPUTensor buf_tree_ancestor_mask_;  // [8] uint64_t
    GPUTensor buf_tree_parent_indices_; // [8] int32_t
    GPUTensor buf_tree_winning_slots_;  // [8] int32_t
    GPUTensor buf_scores_bf16_batch_;   // [max_batch_m, n_experts] BF16 for Nemotron / MoE batched routing
    GPUTensor buf_topk_idx_batch_;      // [max_batch_m, top_k] I32
    GPUTensor buf_topk_vals_batch_;     // [max_batch_m, top_k] F32
    int32_t* topk_ids_batch_host_ = nullptr;
    float* topk_weights_batch_host_ = nullptr;
    GPUTensor buf_moe_offsets_;        // [513] int32_t for MoE token binning
    GPUTensor buf_moe_token_indices_;  // [max_batch_m * top_k] int32_t for MoE token binning
    GPUTensor buf_moe_token_slots_;    // [max_batch_m * top_k] int32_t for MoE token binning
    GPUTensor buf_moe_gate_batch_;      // [max_batch_m, top_k, moe_inter] BF16 for Nemotron batched MoE
    GPUTensor buf_latent_down_batch_;  // [max_batch_m, moe_latent_size] BF16 for Bolt
    GPUTensor buf_latent_accum_batch_; // [max_batch_m, moe_latent_size] BF16 for Bolt
    GPUTensor buf_layer_expert_ptrs_;  // [512] void* GPU pointers
    GPUTensor buf_all_expert_ptrs_;    // [64 * 512] void* GPU pointers for resident MoE layers
    bool layer_fully_resident_[64] = {false};
    void* h_layer_expert_ptrs_[512] = {nullptr};
    GPUTensor buf_expert_scales_gpu_;  // GPU scales
    static constexpr int NUM_DRAM_MOE_BUFFERS = 4;
    GPUTensor buf_dram_moe_scratch_;   // [NUM_DRAM_MOE_BUFFERS * max_experts * expert_block_size] dedicated VRAM scratch for DRAM MoE layers
    void* pinned_host_moe_scratch_ = nullptr;
    size_t pinned_host_moe_scratch_size_ = 0;
    GPUTensor buf_dram_expert_ptrs_gpu_[NUM_DRAM_MOE_BUFFERS]; // [512] void* GPU pointers for 4-buffer circular ring
    cudaEvent_t dma_ready_event_[NUM_DRAM_MOE_BUFFERS] = {nullptr};
    cudaEvent_t compute_done_event_[NUM_DRAM_MOE_BUFFERS] = {nullptr};

    void prefetch_dram_layer_async(int moe_layer_idx, int buf_idx) {
        if (!dma_stream_ || buf_idx < 0 || buf_idx >= NUM_DRAM_MOE_BUFFERS) return;
        if (moe_layer_idx < 0 || moe_layer_idx >= 64 || layer_fully_resident_[moe_layer_idx]) return;
        size_t exp_bytes = expert_loader_.expert_block_size();
        void* dram_src = expert_loader_.get_dram_expert_ptr(moe_layer_idx, 0);
        int n_exp = cfg_.get_moe_layer_n_routed_experts(moe_layer_idx);
        if (n_exp <= 0) n_exp = 128;
        void* scratch_dst = (char*)buf_dram_moe_scratch_.data + (size_t)buf_idx * n_exp * exp_bytes;
        if (dram_src && scratch_dst) {
            CUDA_CHECK(cudaMemcpyAsync(scratch_dst, dram_src, (size_t)n_exp * exp_bytes, cudaMemcpyHostToDevice, dma_stream_));
            CUDA_CHECK(cudaEventRecord(dma_ready_event_[buf_idx], dma_stream_));
        }
    }
    GPUTensor buf_mamba_conv_batch_;   // [max_batch_m, 6144] BF16 for Nemotron batched conv
    GPUTensor buf_mamba_temp_batch_;   // [max_batch_m, 4096] F32 for Nemotron batched SSM intermediate
    cudaGraph_t tree_graph_ = nullptr;
    cudaGraphExec_t tree_graph_exec_ = nullptr;
    bool tree_graph_captured_ = false;

    GPUTensor target_ssm_pool_;
    GPUTensor target_ssm_slots_[8];
    GPUTensor target_conv_pool_;
    GPUTensor target_conv_slots_[8];
    int32_t* h_target_batch_tok_ = nullptr;
    int32_t* h_target_batch_pos_ = nullptr;
    uint64_t* h_tree_ancestor_mask_ = nullptr;
    int32_t* h_tree_parent_ = nullptr;
    int32_t* h_single_tok_ = nullptr;
    int32_t* h_single_pos_ = nullptr;
    int32_t* h_single_flag_ = nullptr;
    int32_t* h_prefill_tok_ = nullptr;
    int32_t* h_prefill_pos_ = nullptr;
    size_t h_prefill_cap_ = 0;

    void ensure_prefill_host_capacity(size_t n) {
        if (n <= h_prefill_cap_) return;
        if (h_prefill_tok_) CUDA_CHECK(cudaFreeHost(h_prefill_tok_));
        if (h_prefill_pos_) CUDA_CHECK(cudaFreeHost(h_prefill_pos_));
        size_t cap = std::max(n, (size_t)cfg_.max_seq_len);
        if (cap < 2048) cap = 2048;
        CUDA_CHECK(cudaMallocHost(&h_prefill_tok_, cap * sizeof(int32_t)));
        CUDA_CHECK(cudaMallocHost(&h_prefill_pos_, cap * sizeof(int32_t)));
        h_prefill_cap_ = cap;
    }

    GPUTensor buf_hc_state_;      // [hc_mult, hidden_size] BF16 — active HC hidden state
    GPUTensor buf_hc_after_attn_; // [hc_mult, hidden_size] BF16 — intermediate HC state after attention
    GPUTensor buf_hc_pre_;       // [hc_mult] F32
    GPUTensor buf_hc_post_;      // [hc_mult] F32
    GPUTensor buf_hc_comb_;      // [hc_mult*hc_mult] F32
    GPUTensor buf_hc_mixes_;     // [(2+hc)*hc] F32
    GPUTensor buf_hc_input_;     // [hc_mult*hidden_size] F32 — flattened HC state

    // Compressor working buffers
    GPUTensor buf_comp_proj_;    // F32 working buffer for compressor projections
    GPUTensor buf_comp_out_;     // F32 working buffer for compressor pooling output
    GPUTensor buf_comp_bf16_;    // BF16 working buffer for compressed entry (head_dim)
    GPUTensor buf_indexer_q_;    // [64 * 128] BF16 for indexer query heads
    GPUTensor buf_indexer_weights_bf16_; // [64] BF16
    GPUTensor buf_indexer_weights_f32_;  // [64] F32
    GPUTensor buf_indexer_scores_;       // [max_compressed_entries] F32
    GPUTensor buf_indexer_mask_;         // [max_compressed_entries] uint8_t
    GPUTensor buf_indexer_proj_kv_;      // [256] F32
    GPUTensor buf_indexer_proj_gate_;    // [256] F32

    // Device-driven inputs for CUDA Graph & Device ArgMax
    GPUTensor buf_input_token_;  // [1] int32_t on GPU
    GPUTensor buf_input_pos_;    // [1] int32_t on GPU
    GPUTensor buf_mrope_pos_;    // [2] int32_t on GPU [v_start, v_num]
    GPUTensor buf_track_flag_;   // [1] int32_t on GPU (1=track, 0=skip)
    GPUTensor buf_argmax_out_;   // [1] int32_t on GPU for 4-byte sampling
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    bool graph_captured_ = false;
    bool argmax_cache_valid_ = false;

    // Imatrix calibration accumulators
    bool collect_imatrix_ = false;
    GPUTensor d_gate_accum_;
    GPUTensor d_down_accum_;
    GPUTensor d_expert_counts_;

    // Expert frequency & semantic specialization tracking (--track mode)
    bool track_expert_freq_ = false;
    bool track_current_token_ = false; // Only true for generated content tokens (excludes prompt prefill, reasoning block, & control tokens)
    GPUTensor d_step_topk_;                     // [n_layers * top_k] I32 on GPU
    int32_t* step_topk_host_ = nullptr;        // [2 * n_layers * top_k] I32 pinned double-buffer
    int decode_step_idx_ = 0;
    std::vector<uint32_t> expert_freq_counts_;  // [n_layers * n_experts]
    int64_t expert_freq_total_tokens_ = 0;
    std::vector<std::vector<std::unordered_map<int, uint32_t>>> expert_token_counts_; // [n_layers][n_experts][token_id -> count]
    std::vector<std::vector<std::vector<std::pair<std::string, uint32_t>>>> expert_loaded_top_tokens_; // [n_layers][n_experts] -> list of top tokens

    // Pre-allocated host buffers (zero runtime heap allocations)
    std::vector<float> router_probs_host_;
    std::vector<float> router_selection_host_;
    std::vector<int> router_indices_host_;
    std::vector<float> logits_host_;
    std::vector<float> probs_host_;
    __nv_bfloat16* logits_bf16_host_ = nullptr;

    // Batched Prefill Buffers & State for DeepSeek V4
    struct PrefillBatchBuffers {
        GPUTensor buf_tokens;
        GPUTensor buf_pos;
        GPUTensor buf_hidden;
        GPUTensor buf_hc_state;
        GPUTensor buf_hc_after_attn;
        GPUTensor buf_hc_mixes;
        GPUTensor buf_hc_pre;
        GPUTensor buf_hc_post;
        GPUTensor buf_hc_comb;
        GPUTensor buf_lora;
        GPUTensor buf_q;
        GPUTensor buf_kv;
        GPUTensor buf_attn_out;
        GPUTensor buf_scores_bf16;
        GPUTensor buf_topk_idx;
        GPUTensor buf_topk_vals;
        GPUTensor buf_gate;
        GPUTensor buf_down;
        GPUTensor buf_shared_gate;
        GPUTensor buf_shared_up;
        GPUTensor buf_shared_down;

        void alloc(int M, int dim, int hc, int q_lora, int o_lora, int o_groups, int n_heads, int head_dim_val, int top_k, int moe_inter, int n_experts) {
            int max_lora = std::max(q_lora, o_groups * o_lora);
            buf_tokens.alloc(M * sizeof(int32_t));
            buf_pos.alloc(M * sizeof(int32_t));
            buf_hidden.alloc((size_t)M * dim * sizeof(__nv_bfloat16));
            buf_hc_state.alloc((size_t)M * (hc * dim) * sizeof(__nv_bfloat16));
            buf_hc_after_attn.alloc((size_t)M * (hc * dim) * sizeof(__nv_bfloat16));
            buf_hc_mixes.alloc((size_t)M * ((2 + hc) * hc) * sizeof(float));
            buf_hc_pre.alloc((size_t)M * hc * sizeof(float));
            buf_hc_post.alloc((size_t)M * hc * sizeof(float));
            buf_hc_comb.alloc((size_t)M * (hc * hc) * sizeof(float));
            buf_lora.alloc((size_t)M * max_lora * sizeof(__nv_bfloat16));
            buf_q.alloc((size_t)M * (n_heads * head_dim_val) * sizeof(__nv_bfloat16));
            buf_kv.alloc((size_t)M * head_dim_val * sizeof(__nv_bfloat16));
            buf_attn_out.alloc((size_t)M * (n_heads * head_dim_val) * sizeof(__nv_bfloat16));
            buf_scores_bf16.alloc((size_t)M * n_experts * sizeof(__nv_bfloat16));
            buf_topk_idx.alloc((size_t)M * top_k * sizeof(int32_t));
            buf_topk_vals.alloc((size_t)M * top_k * sizeof(float));
            buf_gate.alloc((size_t)M * (top_k * moe_inter) * sizeof(__nv_bfloat16));
            buf_down.alloc((size_t)M * (top_k * dim) * sizeof(__nv_bfloat16));
            buf_shared_gate.alloc((size_t)M * moe_inter * sizeof(__nv_bfloat16));
            buf_shared_up.alloc((size_t)M * moe_inter * sizeof(__nv_bfloat16));
            buf_shared_down.alloc((size_t)M * dim * sizeof(__nv_bfloat16));
        }
    };

    PrefillBatchBuffers prefill_bufs_;
    std::string batched_prefill_mode_ = "auto";
    bool enable_batched_prefill_ = false;
    int prefill_chunk_size_ = 512;

    ~MoecherEngine() {
        if (h_prefill_tok_) {
            cudaFreeHost(h_prefill_tok_);
            h_prefill_tok_ = nullptr;
        }
        if (h_prefill_pos_) {
            cudaFreeHost(h_prefill_pos_);
            h_prefill_pos_ = nullptr;
        }
        if (h_single_tok_) {
            cudaFreeHost(h_single_tok_);
            h_single_tok_ = nullptr;
        }
        if (h_single_pos_) {
            cudaFreeHost(h_single_pos_);
            h_single_pos_ = nullptr;
        }
        if (h_single_flag_) {
            cudaFreeHost(h_single_flag_);
            h_single_flag_ = nullptr;
        }
        if (topk_ids_host_) {
            cudaFreeHost(topk_ids_host_);
            topk_ids_host_ = nullptr;
        }
        if (topk_weights_host_) {
            cudaFreeHost(topk_weights_host_);
            topk_weights_host_ = nullptr;
        }
        if (h_target_batch_tok_) {
            cudaFreeHost(h_target_batch_tok_);
            h_target_batch_tok_ = nullptr;
        }
        if (h_target_batch_pos_) {
            cudaFreeHost(h_target_batch_pos_);
            h_target_batch_pos_ = nullptr;
        }
        if (h_target_batch_tok_) {
            cudaFreeHost(h_target_batch_tok_);
            h_target_batch_tok_ = nullptr;
        }
        if (h_tree_ancestor_mask_) {
            cudaFreeHost(h_tree_ancestor_mask_);
            h_tree_ancestor_mask_ = nullptr;
        }
        if (h_tree_parent_) {
            cudaFreeHost(h_tree_parent_);
            h_tree_parent_ = nullptr;
        }
        if (pinned_host_moe_scratch_) {
            cudaFreeHost(pinned_host_moe_scratch_);
            pinned_host_moe_scratch_ = nullptr;
        }
        for (int m = 0; m <= 8; m++) {
            if (batch_graph_exec_[m]) {
                cudaGraphExecDestroy(batch_graph_exec_[m]);
                batch_graph_exec_[m] = nullptr;
            }
            if (batch_graph_[m]) {
                cudaGraphDestroy(batch_graph_[m]);
                batch_graph_[m] = nullptr;
            }
        }
        if (tree_graph_exec_) {
            cudaGraphExecDestroy(tree_graph_exec_);
            tree_graph_exec_ = nullptr;
        }
        if (tree_graph_) {
            cudaGraphDestroy(tree_graph_);
            tree_graph_ = nullptr;
        }
        if (step_topk_host_) {
            cudaFreeHost(step_topk_host_);
            step_topk_host_ = nullptr;
        }
        if (topk_ids_batch_host_) {
            cudaFreeHost(topk_ids_batch_host_);
            topk_ids_batch_host_ = nullptr;
        }
        if (topk_weights_batch_host_) {
            cudaFreeHost(topk_weights_batch_host_);
            topk_weights_batch_host_ = nullptr;
        }
        if (graph_exec_) {
            cudaGraphExecDestroy(graph_exec_);
            graph_exec_ = nullptr;
        }
        if (graph_) {
            cudaGraphDestroy(graph_);
            graph_ = nullptr;
        }
        if (logits_bf16_host_) {
            cudaFreeHost(logits_bf16_host_);
            logits_bf16_host_ = nullptr;
        }
        if (main_event_) {
            cudaEventDestroy(main_event_);
            main_event_ = nullptr;
        }
        if (side_event_) {
            cudaEventDestroy(side_event_);
            side_event_ = nullptr;
        }
        if (dma_event_) {
            cudaEventDestroy(dma_event_);
            dma_event_ = nullptr;
        }
        for (int b = 0; b < NUM_DRAM_MOE_BUFFERS; b++) {
            if (dma_ready_event_[b]) { cudaEventDestroy(dma_ready_event_[b]); dma_ready_event_[b] = nullptr; }
            if (compute_done_event_[b]) { cudaEventDestroy(compute_done_event_[b]); compute_done_event_[b] = nullptr; }
        }
        if (side_stream_) {
            cudaStreamDestroy(side_stream_);
            side_stream_ = nullptr;
        }
        if (dma_stream_) {
            cudaStreamDestroy(dma_stream_);
            dma_stream_ = nullptr;
        }
        if (cublas_handle_) {
            cublasDestroy(cublas_handle_);
            cublas_handle_ = nullptr;
        }
        if (cublas_side_handle_) {
            cublasDestroy(cublas_side_handle_);
            cublas_side_handle_ = nullptr;
        }
        if (cublaslt_workspace_) {
            cudaFree(cublaslt_workspace_);
            cublaslt_workspace_ = nullptr;
        }
        if (cublaslt_workspace_side_) {
            cudaFree(cublaslt_workspace_side_);
            cublaslt_workspace_side_ = nullptr;
        }
        if (cublaslt_handle_) {
            cublasLtDestroy(cublaslt_handle_);
            cublaslt_handle_ = nullptr;
        }
        if (cublaslt_side_handle_) {
            cublasLtDestroy(cublaslt_side_handle_);
            cublaslt_side_handle_ = nullptr;
        }
        if (dequant_stream_main_) {
            cudaStreamDestroy(dequant_stream_main_);
            dequant_stream_main_ = nullptr;
        }
        if (dequant_stream_side_) {
            cudaStreamDestroy(dequant_stream_side_);
            dequant_stream_side_ = nullptr;
        }
        for (int s = 0; s < 2; s++) {
            for (int i = 0; i < 2; i++) {
                if (dequant_ready_event_[s][i]) {
                    cudaEventDestroy(dequant_ready_event_[s][i]);
                    dequant_ready_event_[s][i] = nullptr;
                }
                if (gemm_done_event_[s][i]) {
                    cudaEventDestroy(gemm_done_event_[s][i]);
                    gemm_done_event_[s][i] = nullptr;
                }
            }
        }
        for (int b = 0; b < 2; b++) {
            for (int k = 0; k < 32; k++) {
                if (streaming_slots_gpu_[b][k]) {
                    cudaFree(streaming_slots_gpu_[b][k]);
                    streaming_slots_gpu_[b][k] = nullptr;
                }
            }
        }
    }

    // Dequant buffer — large enough for the biggest weight matrix
    static constexpr size_t DEQUANT_BUF_SIZE = 64 * 1024 * 1024;  // 64 MB

    // ── Load model from manifest ────────────────────────────────────────────

    bool load(const std::string& manifest_path, float max_vram_gb = 0.0f, float dram_cache_gb = 0.0f,
              const std::string& expert_dtype_override = "", bool buffered_io = false,
              int max_seq_len_override = 0) {
        manifest_path_ = manifest_path;
        LOG_INFO("Loading manifest: %s", manifest_path.c_str());

        std::ifstream f(manifest_path);
        if (!f.is_open()) { LOG_ERROR("Cannot open manifest"); return false; }
        json manifest;
        f >> manifest;

        // Parse config
        cfg_.from_json(manifest["model_config"]);
        if (max_seq_len_override > 0) {
            cfg_.max_seq_len = max_seq_len_override;
            cfg_.max_compressed_entries = 0;
            for (int r : cfg_.compress_ratios) {
                if (r > 0) {
                    int entries = cfg_.max_seq_len / r + 2;
                    if (entries > cfg_.max_compressed_entries) cfg_.max_compressed_entries = entries;
                }
            }
        }
        if (cfg_.model_name.empty() && manifest.contains("model_name")) {
            cfg_.model_name = manifest["model_name"].get<std::string>();
        }
        if (cfg_.model_id.empty() && manifest.contains("model_id")) {
            cfg_.model_id = manifest["model_id"].get<std::string>();
        }
        if (!expert_dtype_override.empty()) {
            cfg_.expert_dtype = expert_dtype_override;
        }
        LOG_INFO("Model: %d layers, %d experts, %d active, hidden=%d, dtype=%s, max_seq_len=%d, sliding_window=%d",
                 cfg_.num_hidden_layers, cfg_.n_routed_experts,
                 cfg_.num_experts_per_tok, cfg_.hidden_size, cfg_.expert_dtype.c_str(), cfg_.max_seq_len, cfg_.sliding_window);

        std::filesystem::path manifest_p(manifest_path);
        std::filesystem::path base_dir = manifest_p.parent_path();
        model_dir_ = base_dir.string();

        // Load tokenizer
        std::string tok_path = "tokenizer.json";
        if (manifest.contains("tokenizer")) {
            if (manifest["tokenizer"].is_object()) {
                tok_path = manifest["tokenizer"].value("tokenizer_json", "tokenizer.json");
            } else if (manifest["tokenizer"].is_string()) {
                tok_path = manifest["tokenizer"].get<std::string>();
            }
        }
        std::string tok_full = (base_dir / tok_path).string();
        if (!tokenizer_.load(tok_full) && !tokenizer_.load(tok_path) &&
            !tokenizer_.load("tokenizer.json") && !tokenizer_.load((base_dir / "tokenizer.json").string())) {
            LOG_ERROR("Failed to load tokenizer from %s or %s", tok_full.c_str(), tok_path.c_str());
            return false;
        }

        // Init CUDA
        CUDA_CHECK(cudaStreamCreateWithFlags(&main_stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&side_stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&dma_stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&side_event_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&main_event_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&dma_event_, cudaEventDisableTiming));
        for (int b = 0; b < NUM_DRAM_MOE_BUFFERS; b++) {
            CUDA_CHECK(cudaEventCreateWithFlags(&dma_ready_event_[b], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&compute_done_event_[b], cudaEventDisableTiming));
        }
        for (int i = 0; i < 64; i++) {
            CUDA_CHECK(cudaStreamCreate(&expert_streams_[i]));
            CUDA_CHECK(cudaEventCreate(&expert_events_[i]));
        }
        if (!topk_ids_host_) {
            CUDA_CHECK(cudaMallocHost(&topk_ids_host_, 32 * sizeof(int32_t)));
            CUDA_CHECK(cudaMallocHost(&topk_weights_host_, 32 * sizeof(float)));
        }
        CUDA_CHECK(cudaStreamCreate(&dequant_stream_main_));
        CUDA_CHECK(cudaStreamCreate(&dequant_stream_side_));
        for (int s = 0; s < 2; s++) {
            for (int i = 0; i < 2; i++) {
                CUDA_CHECK(cudaEventCreateWithFlags(&dequant_ready_event_[s][i], cudaEventDisableTiming));
                CUDA_CHECK(cudaEventCreateWithFlags(&gemm_done_event_[s][i], cudaEventDisableTiming));
                CUDA_CHECK(cudaEventRecord(gemm_done_event_[s][i], (s == 0) ? main_stream_ : side_stream_));
            }
        }
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
        CUBLAS_CHECK(cublasSetStream(cublas_handle_, main_stream_));
        cublasSetMathMode(cublas_handle_, CUBLAS_DEFAULT_MATH);

        CUBLAS_CHECK(cublasCreate(&cublas_side_handle_));
        CUBLAS_CHECK(cublasSetStream(cublas_side_handle_, side_stream_));
        cublasSetMathMode(cublas_side_handle_, CUBLAS_DEFAULT_MATH);

        CUBLAS_CHECK(cublasLtCreate(&cublaslt_handle_));
        CUBLAS_CHECK(cublasLtCreate(&cublaslt_side_handle_));
        CUDA_CHECK(cudaMalloc(&cublaslt_workspace_, cublaslt_workspace_size_));
        CUDA_CHECK(cudaMalloc(&cublaslt_workspace_side_, cublaslt_workspace_size_));

        // ── Hardware Capabilities & Persistent L2 Cache Configuration ─────────────
        gpu_caps_ = detect_gpu_capabilities(0);
        LOG_INFO("Hardware: %s (Compute %d.%d, %.1f GB VRAM)",
                 gpu_caps_.device_name.c_str(), gpu_caps_.major, gpu_caps_.minor,
                 gpu_caps_.total_vram_bytes / (1024.0 * 1024.0 * 1024.0));
        LOG_INFO("Hardware Capabilities: Ampere=%d, Ada+=%d, Blackwell+=%d, FP8_TC=%d, FP4=%d, VRAM-Constrained=%d",
                 (int)gpu_caps_.is_ampere, (int)gpu_caps_.is_ada_or_newer,
                 (int)gpu_caps_.is_blackwell_or_newer, (int)gpu_caps_.supports_fp8,
                 (int)gpu_caps_.supports_fp4, (int)gpu_caps_.is_vram_constrained);
        // Check if model requires NVIDIA Blackwell FP4 (NVFP4)
        bool model_requires_fp4 = false;
        if (manifest.contains("target_vram_profile")) {
            model_requires_fp4 = true;
        }
        if (manifest.contains("dense_tensors") && manifest["dense_tensors"].is_object()) {
            for (auto& [tname, tinfo] : manifest["dense_tensors"].items()) {
                if (tinfo.value("dtype", "") == "fp4") {
                    model_requires_fp4 = true;
                    break;
                }
            }
        }
        if (manifest.contains("model_config")) {
            std::string edtype = manifest["model_config"].value("expert_dtype", "");
            if (edtype == "fp4" || edtype == "sparse_nvfp4" || edtype == "mixed_nvfp4_iq2") {
                model_requires_fp4 = true;
            }
        }

        is_blackwell_tc_active_ = gpu_caps_.supports_fp4 && (gpu_caps_.major >= 12) && model_requires_fp4;
        LOG_INFO("Blackwell Tensor Core Acceleration: %s",
                 is_blackwell_tc_active_ ? "ENABLED (Native sm_120a hardware FP4 mma.sync)" : "DISABLED (SIMT Fallback / Non-FP4 Model)");

        is_blackwell_fp8_active_ = gpu_caps_.supports_fp8 && (gpu_caps_.major >= 12);
        LOG_INFO("Blackwell sm_120a Native FP8 Tensor Cores: %s",
                 is_blackwell_fp8_active_ ? "ENABLED (cuBLASLt FP8_E4M3 Tensor Core Pipeline)" : "DISABLED (BF16 Fallback)");

        if (max_seq_len_override == 0 && gpu_caps_.total_vram_bytes > 0 &&
            gpu_caps_.total_vram_bytes <= 17ULL * 1024 * 1024 * 1024 && cfg_.max_seq_len > 8192) {
            LOG_INFO("Hardware: 16GB GPU detected (%.1f GB VRAM). Auto-tuning max_seq_len from %d to 8192 to prevent Windows shared DRAM paging (use --ctx <n> to override).",
                     gpu_caps_.total_vram_bytes / (1024.0 * 1024.0 * 1024.0), cfg_.max_seq_len);
            cfg_.max_seq_len = 8192;
        }

        if (model_requires_fp4 && !gpu_caps_.supports_fp4) {
            LOG_ERROR("================================================================================");
            LOG_ERROR("FATAL: Hardware Unsupported! Model requires NVIDIA Blackwell NVFP4 (SM 10.0+ / 12.0+)");
            LOG_ERROR("Current GPU is '%s' (Compute %d.%d), which does not support hardware FP4.",
                      gpu_caps_.device_name.c_str(), gpu_caps_.major, gpu_caps_.minor);
            LOG_ERROR("Please use an INT4 or BF16 model on this GPU, or run on Blackwell hardware.");
            LOG_ERROR("================================================================================");
            return false;
        }

        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            size_t max_persisting_l2 = prop.persistingL2CacheMaxSize;
            if (max_persisting_l2 > 0) {
                size_t l2_limit = (max_persisting_l2 * 3) / 4; // Use 75% for persistent window
                cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2_limit);
                LOG_INFO("Persistent L2 Cache configured: %.1f MB / %.1f MB",
                         l2_limit / (1024.0 * 1024.0), max_persisting_l2 / (1024.0 * 1024.0));
            }
        }

        expert_pool_ = std::make_unique<ThreadPool>(16);

        // Determine available VRAM for expert cache
        size_t initial_vram_free, vram_total;
        CUDA_CHECK(cudaMemGetInfo(&initial_vram_free, &vram_total));
        LOG_INFO("VRAM: %.1f GB free / %.1f GB total",
                 initial_vram_free / (1024.0 * 1024.0 * 1024.0), vram_total / (1024.0 * 1024.0 * 1024.0));

        // Load dense tensors from attention_dense_layers.bin
        std::string dense_path = manifest["dense_bin"].get<std::string>();
        std::string dense_full = (base_dir / dense_path).string();
        if (!load_dense_tensors(dense_full, manifest["dense_tensors"]) &&
            !load_dense_tensors(dense_path, manifest["dense_tensors"])) {
            LOG_ERROR("Failed to load dense tensors from %s", dense_full.c_str());
            return false;
        }

        // Load expert layout info safely (optional for dense models like Qwen)
        int expert_block_size = 0;
        int expert_n_layers = 0;
        int expert_n_experts = 0;
        if (manifest.contains("expert_layout") && !manifest["expert_layout"].is_null()) {
            auto& el = manifest["expert_layout"];
            expert_block_size = el.value("block_size", 0);
            expert_n_layers = el.value("n_layers", 0);
            if (expert_n_layers > cfg_.num_hidden_layers && cfg_.num_hidden_layers > 0) {
                expert_n_layers = cfg_.num_hidden_layers;
            }
            expert_n_experts = el.value("n_experts", 0);

            if (el.contains("parts") && el["parts"].is_object()) {
                for (auto& [part_name, part_info] : el["parts"].items()) {
                    ExpertPartInfo epi;
                    epi.offset_in_block = part_info.value("offset_in_block", 0);
                    epi.nbytes = part_info.value("nbytes", 0);
                    epi.dtype = part_info.value("dtype", "fp4");
                    if (part_info.contains("shape")) {
                        for (auto& s : part_info["shape"]) epi.shape.push_back(s.get<int>());
                    }
                    expert_parts_[part_name] = epi;
                }
            }
        }

        // Init expert loader with O_DIRECT
        std::string expert_path = manifest.value("expert_bin", "");
        std::string expert_full = expert_path.empty() ? "" : (base_dir / expert_path).string();
        if (!expert_path.empty() && !std::filesystem::exists(expert_full) && std::filesystem::exists(expert_path)) {
            expert_full = expert_path;
        }

        // Allocate working buffers first
        alloc_buffers();

        // Reserve VRAM: rest goes to expert cache
        size_t vram_free_after_dense;
        CUDA_CHECK(cudaMemGetInfo(&vram_free_after_dense, &vram_total));
        size_t total_experts_bytes = (size_t)expert_n_layers * expert_n_experts * expert_block_size;
        size_t vram_reserve = cfg_.is_nemotron() ? (600ULL * 1024 * 1024) : (4ULL * 1024 * 1024 * 1024); // reserve 600MB for Nemotron
        size_t cache_budget = vram_free_after_dense > vram_reserve ? (vram_free_after_dense - vram_reserve) : vram_free_after_dense;
        
        if (max_vram_gb > 0.0f) {
            size_t max_vram_bytes = (size_t)(max_vram_gb * 1024.0 * 1024.0 * 1024.0);
            size_t process_dense_usage = (initial_vram_free > vram_free_after_dense) ? (initial_vram_free - vram_free_after_dense) : 0;
            if (max_vram_bytes > process_dense_usage + 1ULL * 1024 * 1024 * 1024) {
                size_t user_budget = max_vram_bytes - process_dense_usage - 1ULL * 1024 * 1024 * 1024;
                if (user_budget < cache_budget) {
                    cache_budget = user_budget;
                }
            }
        }

        // Cap cache budget to the exact size of all routed experts if it fits
        if (cache_budget > total_experts_bytes) {
            cache_budget = total_experts_bytes;
        }

        if (cfg_.is_nemotron() && !cfg_.is_nemotron_bolt()) {
            // Cap resident VRAM cache budget to exactly 11 complete layers (0..10) = 1,408 experts (7.73 GiB).
            // This leaves layers 11..22 (1,536 experts = 8.44 GiB) in pinned DRAM L2 cache.
            // With 4-buffer scratch buffer (512 slots = 2.81 GB) + dense weights (2.80 GB), total VRAM is 14.32 GB,
            // leaving 1.67 GB of safety headroom on RTX 5060 Ti 16GB, with ZERO disk reads and zero OOM.
            size_t nemotron_11_layers = (size_t)11 * 128 * expert_block_size;
            if (cache_budget > nemotron_11_layers) {
                cache_budget = nemotron_11_layers;
            }
        }
        
        size_t dram_cache_budget = (size_t)(dram_cache_gb * 1024.0 * 1024.0 * 1024.0);
        if (cfg_.is_nemotron() && dram_cache_budget == 0) {
            size_t avail = 0;
#if defined(_WIN32)
            MEMORYSTATUSEX memInfo;
            memInfo.dwLength = sizeof(MEMORYSTATUSEX);
            if (GlobalMemoryStatusEx(&memInfo)) {
                avail = memInfo.ullAvailPhys;
            }
#elif defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGE_SIZE)
            size_t pages = sysconf(_SC_AVPHYS_PAGES);
            size_t page_size = sysconf(_SC_PAGE_SIZE);
            avail = pages * page_size;
#endif
            if (avail > 4ULL * 1024 * 1024 * 1024) {
                if (cfg_.is_nemotron_bolt()) {
                    size_t safe_avail = avail > 4ULL * 1024 * 1024 * 1024 ? (avail - 4ULL * 1024 * 1024 * 1024) : 0;
                    size_t remaining_experts_bytes = (size_t)(expert_n_layers * expert_n_experts - 4978) * expert_block_size;
                    if (safe_avail > remaining_experts_bytes) safe_avail = remaining_experts_bytes;
                    dram_cache_budget = safe_avail;
                    LOG_INFO("[Memory] Auto-allocated %.1f GB DRAM L2 cache for Nemotron Bolt (out of %.1f GB available physical RAM)",
                             (double)dram_cache_budget / (1024.0 * 1024.0 * 1024.0),
                             (double)avail / (1024.0 * 1024.0 * 1024.0));
                } else {
                    dram_cache_budget = (size_t)12 * 128 * expert_block_size;
                    LOG_INFO("[Memory] Auto-allocated %.1f GB DRAM L2 cache for Nemotron: 100%% of all 2,944 experts resident in memory (zero SSD disk reads)!",
                             (double)dram_cache_budget / (1024.0 * 1024.0 * 1024.0));
                }
            }
        }
        LOG_INFO("Expert L1 (VRAM) cache budget: %.1f GB", cache_budget / (1024.0 * 1024.0 * 1024.0));
        LOG_INFO("Expert L2 (DRAM) cache budget: %.1f GB", dram_cache_budget / (1024.0 * 1024.0 * 1024.0));
        
        if (cfg_.n_routed_experts > 0 && cfg_.expert_dtype == "mixed_nvfp4_iq2") {
            std::string cold_path = "";
            int cold_block_size = 7077888;
            std::string hot_path = "";
            int hot_block_size = 10223616;
            if (manifest.contains("expert_bins")) {
                if (manifest["expert_bins"].contains("cold")) {
                    cold_path = manifest["expert_bins"]["cold"].value("path", "");
                    cold_block_size = manifest["expert_bins"]["cold"].value("block_size", 7077888);
                }
                if (manifest["expert_bins"].contains("hot")) {
                    hot_path = manifest["expert_bins"]["hot"].value("path", "");
                    hot_block_size = manifest["expert_bins"]["hot"].value("block_size", 10223616);
                }
            }
            std::string map_path = manifest.value("expert_map_file", "");

            auto resolve_p = [&](const std::string& p) -> std::string {
                if (p.empty()) return "";
                if (std::filesystem::exists(p)) return p;
                auto full = (base_dir / p).string();
                if (std::filesystem::exists(full)) return full;
                auto rel_root = (base_dir / ".." / ".." / p).string();
                if (std::filesystem::exists(rel_root)) return rel_root;
                return p;
            };

            cold_path = resolve_p(cold_path);
            hot_path = resolve_p(hot_path);
            map_path = resolve_p(map_path);

            if (!expert_loader_.init_mixed(cold_path, cold_block_size,
                                           hot_path, hot_block_size,
                                           map_path,
                                           expert_n_layers, expert_n_experts,
                                           buffered_io)) {
                return false;
            }

            expert_loader_.preload_all_mixed(16);

            expert_token_counts_.resize(cfg_.num_hidden_layers);
            for (int l = 0; l < cfg_.num_hidden_layers; l++) {
                expert_token_counts_[l].resize(cfg_.n_routed_experts);
            }
            std::string json_profile_path = (base_dir / "expert_profile.json").string();
            load_expert_profile_json(json_profile_path);
        } else if (cfg_.n_routed_experts > 0 && !expert_full.empty()) {
            if (!expert_loader_.init(expert_full, expert_block_size,
                                      expert_n_layers, expert_n_experts,
                                      cache_budget, dram_cache_budget, buffered_io)) return false;

            // Allocate double-buffered streaming slots in VRAM for offloading
            for (int b = 0; b < 2; b++) {
                for (int k = 0; k < 32; k++) {
                    CUDA_CHECK(cudaMalloc(&streaming_slots_gpu_[b][k], expert_loader_.expert_block_size_));
                }
            }

            // Preload experts into VRAM & DRAM (frequency-guided if expert_freq.bin exists)
            std::string freq_path = (base_dir / "expert_freq.bin").string();
            expert_freq_counts_ = load_expert_freq(freq_path, cfg_.num_hidden_layers, cfg_.n_routed_experts, &expert_freq_total_tokens_);
            if (expert_freq_counts_.empty()) {
                expert_freq_counts_.assign((size_t)cfg_.num_hidden_layers * cfg_.n_routed_experts, 0);
            }
            expert_loader_.preload_all(16, get_loader_freq_counts());

            expert_token_counts_.resize(cfg_.num_hidden_layers);
            for (int l = 0; l < cfg_.num_hidden_layers; l++) {
                expert_token_counts_[l].resize(cfg_.n_routed_experts);
            }
            std::string json_profile_path = (base_dir / "expert_profile.json").string();
            load_expert_profile_json(json_profile_path);

            if (cfg_.is_nemotron()) {
                int total_moe_layers = 0;
                for (auto t : cfg_.layer_types) if (t == 2) total_moe_layers++;
                if (total_moe_layers == 0) total_moe_layers = cfg_.is_nemotron_bolt() ? 40 : 23;
                int max_routed_exp = std::max({cfg_.n_routed_experts, cfg_.max_n_routed_experts(), 128});

                std::string scales_bin = (base_dir / "expert_scales.bin").string();
                if (!std::filesystem::exists(scales_bin) && std::filesystem::exists("models/nemotron_3.5_lightning/expert_scales.bin")) {
                    scales_bin = "models/nemotron_3.5_lightning/expert_scales.bin";
                }
                nemotron_expert_scales_.resize((size_t)total_moe_layers * max_routed_exp * 2, 1.0f);
                FILE* fp = fopen(scales_bin.c_str(), "rb");
                if (fp) {
                    size_t nread = fread(nemotron_expert_scales_.data(), sizeof(float), nemotron_expert_scales_.size(), fp);
                    fclose(fp);
                    LOG_INFO("Loaded %zu Nemotron expert scales from %s", nread, scales_bin.c_str());
                } else {
                    LOG_WARN("Could not open %s, falling back to 1.0f scales", scales_bin.c_str());
                }
                buf_expert_scales_gpu_.alloc(nemotron_expert_scales_.size() * sizeof(float));
                CUDA_CHECK(cudaMemcpy(buf_expert_scales_gpu_.data, nemotron_expert_scales_.data(),
                                      nemotron_expert_scales_.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));

                buf_all_expert_ptrs_.alloc((size_t)total_moe_layers * max_routed_exp * sizeof(void*));
                std::vector<void*> h_all_ptrs((size_t)total_moe_layers * max_routed_exp, nullptr);
                for (int l = 0; l < total_moe_layers && l < 64; l++) {
                    bool all_in = true;
                    int n_e = cfg_.get_moe_layer_n_routed_experts(l);
                    if (n_e <= 0) n_e = 128;
                    for (int e = 0; e < n_e && e < max_routed_exp; e++) {
                        void* p = expert_loader_.try_get_expert_cached(l, e);
                        h_all_ptrs[(size_t)l * max_routed_exp + e] = p;
                        if (!p) all_in = false;
                    }
                    layer_fully_resident_[l] = all_in;
                }
                CUDA_CHECK(cudaMemcpy(buf_all_expert_ptrs_.data, h_all_ptrs.data(),
                                      (size_t)total_moe_layers * max_routed_exp * sizeof(void*), cudaMemcpyHostToDevice));

                if (cfg_.is_nemotron_bolt() || cfg_.is_nemotron()) {
                    int max_slots = cfg_.is_nemotron_bolt() ? 256 : 128;
                    size_t exp_block_sz = expert_loader_.expert_block_size() > 0 ? (size_t)expert_loader_.expert_block_size()
                                        : (cfg_.is_nemotron_bolt() ? (size_t)1351680 : (size_t)5615616);
                    buf_dram_moe_scratch_.alloc(max_slots * exp_block_sz);
                    size_t staging_sz = (size_t)max_slots * exp_block_sz;
                    if (!pinned_host_moe_scratch_ || pinned_host_moe_scratch_size_ < staging_sz) {
                        if (pinned_host_moe_scratch_) cudaFreeHost(pinned_host_moe_scratch_);
                        CUDA_CHECK(cudaMallocHost(&pinned_host_moe_scratch_, staging_sz));
                        pinned_host_moe_scratch_size_ = staging_sz;
                    }
                } else {
                    buf_dram_moe_scratch_.alloc((size_t)NUM_DRAM_MOE_BUFFERS * max_routed_exp * expert_block_size);
                    for (int b = 0; b < NUM_DRAM_MOE_BUFFERS; b++) {
                        buf_dram_expert_ptrs_gpu_[b].alloc((size_t)max_routed_exp * sizeof(void*));
                        std::vector<void*> h_ptrs(max_routed_exp);
                        char* base = (char*)buf_dram_moe_scratch_.data + (size_t)b * max_routed_exp * expert_block_size;
                        for (int e = 0; e < max_routed_exp; e++) {
                            h_ptrs[e] = base + (size_t)e * expert_block_size;
                        }
                        CUDA_CHECK(cudaMemcpy(buf_dram_expert_ptrs_gpu_[b].data, h_ptrs.data(),
                                              (size_t)max_routed_exp * sizeof(void*), cudaMemcpyHostToDevice));
                    }
                }
            }

            if (d_expert_counts_.data && !expert_freq_counts_.empty()) {
                CUDA_CHECK(cudaMemcpy(d_expert_counts_.data, expert_freq_counts_.data(),
                                      expert_freq_counts_.size() * sizeof(uint32_t),
                                      cudaMemcpyHostToDevice));
            }
        } else {
            LOG_INFO("Dense architecture active (0 routed experts). Bypassing expert cache.");
        }

        // Precompute RoPE frequencies — two tables for non-compressed vs compressed layers
        // DeepSeek-V4 reference: non-compressed layers use base freq without YaRN interpolation;
        // compressed layers use compress_rope_theta (160000) with full YaRN.
        int rope_dim = cfg_.qk_rope_head_dim;
        size_t freq_bytes = MAX_SEQ_LEN * (rope_dim / 2) * 2 * sizeof(float);

        // Non-compressed layers: base=10000, NO YaRN (original_seq_len=0 disables it)
        rope_freqs_.alloc(freq_bytes);
        precompute_freqs_cuda(rope_freqs_.f32(), MAX_SEQ_LEN, rope_dim,
                              cfg_.rope_theta, 1.0f,
                              0, cfg_.rope_beta_fast,
                              cfg_.rope_beta_slow, main_stream_);

        // Compressed layers: base=160000, with YaRN (factor=16)
        rope_freqs_compressed_.alloc(freq_bytes);
        precompute_freqs_cuda(rope_freqs_compressed_.f32(), MAX_SEQ_LEN, rope_dim,
                              cfg_.compress_rope_theta, cfg_.rope_factor,
                              cfg_.original_seq_len, cfg_.rope_beta_fast,
                              cfg_.rope_beta_slow, main_stream_);
        CUDA_CHECK(cudaStreamSynchronize(main_stream_));
        LOG_INFO("RoPE: non-compressed base=%.0f (no YaRN), compressed base=%.0f (YaRN factor=%.0f)",
                 cfg_.rope_theta, cfg_.compress_rope_theta, cfg_.rope_factor);

        if (!cfg_.compress_ratios.empty()) {
            LOG_INFO("Compress ratios loaded: %zu layers", cfg_.compress_ratios.size());
            LOG_INFO("  Layer 0: ratio=%d", cfg_.layer_compress_ratio(0));
            LOG_INFO("  Layer 2: ratio=%d", cfg_.layer_compress_ratio(2));
            LOG_INFO("  Layer 3: ratio=%d", cfg_.layer_compress_ratio(3));
        } else {
            LOG_INFO("No compress_ratios found, using full window for all layers");
        }

        // Auto-load Qwen 3.8 2B Neural Speculative Draft Model if available
        if (cfg_.architecture == ModelArch::QWEN) {
            std::string draft_path = model_dir_.empty() ? "draft_2b/draft_manifest.json" : (model_dir_ + "/draft_2b/draft_manifest.json");
            if (!std::ifstream(draft_path).good()) {
                draft_path = "f:/Moecher/models/qwen3_8_27b_q4/draft_2b/draft_manifest.json";
            }
            if (std::ifstream(draft_path).good()) {
                qwen_draft_.load(draft_path, main_stream_);
            }
        }

        // Auto-configure In-Process Vision Delegate if specified in manifest
        std::string delegate_manifest_path;
        if (manifest.contains("vision_delegate")) {
            if (manifest["vision_delegate"].is_string()) {
                delegate_manifest_path = manifest["vision_delegate"].get<std::string>();
            } else if (manifest["vision_delegate"].is_object()) {
                delegate_manifest_path = manifest["vision_delegate"].value("manifest", "");
            }
        } else if (manifest.contains("model_config") && manifest["model_config"].contains("vision_delegate")) {
            if (manifest["model_config"]["vision_delegate"].is_string()) {
                delegate_manifest_path = manifest["model_config"]["vision_delegate"].get<std::string>();
            } else if (manifest["model_config"]["vision_delegate"].is_object()) {
                delegate_manifest_path = manifest["model_config"]["vision_delegate"].value("manifest", "");
            }
        }
        if (!delegate_manifest_path.empty()) {
            std::filesystem::path full_p = base_dir / delegate_manifest_path;
            if (std::ifstream(full_p).good()) {
                delegate_manifest_path = full_p.string();
            }
            if (std::ifstream(delegate_manifest_path).good()) {
                delegate_manifest_path_ = delegate_manifest_path;
                LOG_INFO("[Vision Delegate] Found vision delegate manifest at: %s", delegate_manifest_path_.c_str());

                // Check available VRAM to determine if we should eagerly load and keep resident (large GPU / UMA)
                // or use dynamic swapping (VRAM-constrained GPU like 16GB)
                size_t free_vram = 0, total_vram = 0;
                cudaMemGetInfo(&free_vram, &total_vram);
                size_t required_for_delegate = 4800ULL * 1024 * 1024; // ~4.8 GB

                if (free_vram >= required_for_delegate && !gpu_caps_.is_vram_constrained) {
                    LOG_INFO("[Vision Delegate] Sufficient VRAM available (%.1f GB free / %.1f GB total). Loading vision delegate permanently into VRAM...",
                             (double)free_vram / (1024.0 * 1024.0 * 1024.0), (double)total_vram / (1024.0 * 1024.0 * 1024.0));
                    vision_delegate_engine_ = std::make_unique<MoecherEngine>();
                    int delegate_seq_len = (max_seq_len_override > 0) ? std::min(max_seq_len_override, 2048) : 2048;
                    if (vision_delegate_engine_->load(delegate_manifest_path_, max_vram_gb, dram_cache_gb, expert_dtype_override, buffered_io, delegate_seq_len)) {
                        LOG_INFO("[Vision Delegate] Vision delegate permanently resident in VRAM!");
                    } else {
                        LOG_ERROR("[Vision Delegate] Failed to load vision delegate engine!");
                        vision_delegate_engine_.reset();
                    }
                } else {
                    LOG_INFO("[Vision Delegate] VRAM-constrained hardware detected (%.1f GB free / %.1f GB total). Configuring dynamic on-demand VRAM swap (zero idle VRAM overhead)!",
                             (double)free_vram / (1024.0 * 1024.0 * 1024.0), (double)total_vram / (1024.0 * 1024.0 * 1024.0));
                }
            } else {
                LOG_WARN("[Vision Delegate] Vision delegate manifest file not found: %s", delegate_manifest_path.c_str());
            }
        }

        // Auto-configure Direct Vision Tower if available in manifest and no delegate is configured
        has_vision_ = false;
        vision_tower_cfg_.available = false;
        bool vision_flag = (manifest.contains("model_config") && manifest["model_config"].value("has_vision", false)) ||
                           manifest.contains("visual_config") ||
                           manifest.contains("vision_bin") ||
                           (manifest.contains("dense_tensors") && manifest["dense_tensors"].contains("model.visual.patch_embed.proj.weight"));
        if (enable_vision_ && !has_vision_delegate() && (cfg_.is_qwen() || cfg_.architecture == ModelArch::DEEPSEEK_V4) && vision_flag) {
            if (manifest.contains("dense_tensors") && manifest["dense_tensors"].contains("model.visual.patch_embed.proj.weight")) {
                std::string dense_path = manifest.value("dense_bin", "");
                std::string dense_full = dense_path.empty() ? "" : (base_dir / dense_path).string();
                if (!dense_full.empty() && !std::ifstream(dense_full).good()) {
                    dense_full = dense_path;
                }

                std::string vision_bin = manifest.value("vision_bin", "");
                std::string vision_full = vision_bin.empty() ? "" : (base_dir / vision_bin).string();
                if (!vision_full.empty() && !std::ifstream(vision_full).good()) {
                    vision_full = vision_bin;
                }

                std::string bridge_bin = manifest.value("bridge_bin", "");
                std::string bridge_full = bridge_bin.empty() ? "" : (base_dir / bridge_bin).string();
                if (!bridge_full.empty() && !std::ifstream(bridge_full).good()) {
                    bridge_full = bridge_bin;
                }

                std::string embed_mean_bin = manifest.value("embed_mean_bin", "deepseek_embed_mean.bin");
                std::string embed_mean_full = embed_mean_bin.empty() ? "" : (base_dir / embed_mean_bin).string();
                if (!embed_mean_full.empty() && !std::ifstream(embed_mean_full).good()) {
                    embed_mean_full = embed_mean_bin;
                }

                int output_dim = cfg_.hidden_size; // 4096 for DeepSeek, 5120 for Qwen, 2048 for Qwen2.5-VL
                
                // Save configuration for on-demand lazy loading (zero VRAM overhead during text chat)
                vision_tower_cfg_.available = true;
                vision_tower_cfg_.dense_path = dense_full;
                vision_tower_cfg_.tensor_map = manifest["dense_tensors"];
                vision_tower_cfg_.vision_bin = vision_full;
                vision_tower_cfg_.bridge_bin = bridge_full;
                vision_tower_cfg_.embed_mean_bin = embed_mean_full;
                vision_tower_cfg_.output_dim = output_dim;
                has_vision_ = true;

                LOG_INFO("[Vision] Vision configuration detected in manifest. Enabled on-demand Vision Tower (zero VRAM at startup; loads ~990 MB only when an image is processed).");
            }
        } else if (!enable_vision_ && vision_flag) {
            LOG_INFO("[Vision] Vision Tower explicitly disabled via --no-vision flag (saves ~990 MB VRAM).");
        }

        apply_l2_cache_persistence();
        init_cuda_graph();
        init_batched_prefill();

        LOG_INFO("Model loaded successfully");
        return true;
    }

    void apply_l2_cache_persistence() {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess || prop.persistingL2CacheMaxSize == 0) {
            return;
        }

        size_t l2_limit = (prop.persistingL2CacheMaxSize * 3) / 4;
        size_t total_pinned_bytes = 0;

        auto pin_buffer = [this, l2_limit, &total_pinned_bytes](void* ptr, size_t num_bytes) -> bool {
            if (!ptr || num_bytes == 0) return false;
            if (total_pinned_bytes + num_bytes > l2_limit) return false;

            cudaStreamAttrValue attr;
            std::memset(&attr, 0, sizeof(attr));
            attr.accessPolicyWindow.base_ptr = ptr;
            attr.accessPolicyWindow.num_bytes = num_bytes;
            attr.accessPolicyWindow.hitRatio = 1.0f;
            attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

            cudaStreamSetAttribute(main_stream_, cudaStreamAttributeAccessPolicyWindow, &attr);
            cudaStreamSetAttribute(side_stream_, cudaStreamAttributeAccessPolicyWindow, &attr);
            total_pinned_bytes += num_bytes;
            return true;
        };

        // Priority 1: High-frequency normalization weights, sinks, router biases, RoPE tables
        pin_buffer(norm_weight_.data, norm_weight_.size_bytes);
        pin_buffer(rope_freqs_.data, rope_freqs_.size_bytes);
        pin_buffer(rope_freqs_compressed_.data, rope_freqs_compressed_.size_bytes);
        for (auto& lw : layers_) {
            pin_buffer(lw.attn_norm_w.data, lw.attn_norm_w.size_bytes);
            pin_buffer(lw.ffn_norm_w.data, lw.ffn_norm_w.size_bytes);
            pin_buffer(lw.q_norm_w.data, lw.q_norm_w.size_bytes);
            pin_buffer(lw.kv_norm_w.data, lw.kv_norm_w.size_bytes);
            pin_buffer(lw.attn_sink.data, lw.attn_sink.size_bytes);
            pin_buffer(lw.gate_bias.data, lw.gate_bias.size_bytes);
        }

        // Priority 2: Hot active sliding window KV caches (128 tokens per layer = ~5.5 MB)
        for (auto& lw : layers_) {
            pin_buffer(lw.kv_cache.data, lw.kv_cache.size_bytes);
        }

        // Priority 3: Highway Connection projections and scale/base (all 43 layers + head reduce = ~22.6 MB)
        pin_buffer(hc_head_fn_.data, hc_head_fn_.size_bytes);
        pin_buffer(hc_head_scale_.data, hc_head_scale_.size_bytes);
        pin_buffer(hc_head_base_.data, hc_head_base_.size_bytes);
        for (auto& lw : layers_) {
            pin_buffer(lw.hc_attn_fn.data, lw.hc_attn_fn.size_bytes);
            pin_buffer(lw.hc_attn_scale.data, lw.hc_attn_scale.size_bytes);
            pin_buffer(lw.hc_attn_base.data, lw.hc_attn_base.size_bytes);
            pin_buffer(lw.hc_ffn_fn.data, lw.hc_ffn_fn.size_bytes);
            pin_buffer(lw.hc_ffn_scale.data, lw.hc_ffn_scale.size_bytes);
            pin_buffer(lw.hc_ffn_base.data, lw.hc_ffn_base.size_bytes);
        }

        // Priority 4: MoE Router Linear Projection Weights (all 43 layers)
        for (auto& lw : layers_) {
            pin_buffer(lw.gate_w.data, lw.gate_w.size_bytes);
        }

        // Priority 5: Compressor weights
        for (auto& lw : layers_) {
            pin_buffer(lw.comp_wkv.data, lw.comp_wkv.size_bytes);
            pin_buffer(lw.comp_wgate.data, lw.comp_wgate.size_bytes);
            pin_buffer(lw.comp_ape.data, lw.comp_ape.size_bytes);
            pin_buffer(lw.comp_norm.data, lw.comp_norm.size_bytes);
        }

        // Priority 6: Shared expert & Attention Low-Rank Projections (up to remaining budget)
        for (auto& lw : layers_) {
            pin_buffer(lw.shared_w1_w.data, lw.shared_w1_w.size_bytes);
            pin_buffer(lw.shared_w3_w.data, lw.shared_w3_w.size_bytes);
            pin_buffer(lw.shared_w2_w.data, lw.shared_w2_w.size_bytes);
            pin_buffer(lw.wq_a_w.data, lw.wq_a_w.size_bytes);
            pin_buffer(lw.wkv_w.data, lw.wkv_w.size_bytes);
            pin_buffer(lw.wo_a_w.data, lw.wo_a_w.size_bytes);
        }

        // Priority 7: Compressed KV caches (up to remaining persistent budget)
        for (auto& lw : layers_) {
            pin_buffer(lw.comp_kv_cache.data, lw.comp_kv_cache.size_bytes);
        }

        LOG_INFO("Persistent L2 Cache: Pinned %.2f MB / %.2f MB budget (zero thrashing, 100%% utilization)",
                 total_pinned_bytes / (1024.0 * 1024.0), l2_limit / (1024.0 * 1024.0));
    }

    void init_cuda_graph() {
#if defined(__APPLE__)
        graph_captured_ = false;
        return;
#endif
        init_mla_dynamic_shared_memory();
        if (cfg_.is_nemotron() || (!cfg_.is_qwen() && !expert_loader_.all_resident(cfg_.num_hidden_layers))) {
            LOG_INFO("Running in eager mode for decode verification.");
            graph_captured_ = false;
            return;
        }
        LOG_INFO("Warming up and capturing CUDA Graph for decode acceleration...");
        int dummy_tok = 0;
        int dummy_pos = 0;
        int dummy_flag = 0;
        CUDA_CHECK(cudaMemcpy(buf_input_token_.i32(), &dummy_tok, sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(buf_input_pos_.i32(), &dummy_pos, sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(buf_track_flag_.i32(), &dummy_flag, sizeof(int32_t), cudaMemcpyHostToDevice));

        // 3 warmup iterations
        for (int i = 0; i < 3; i++) {
            forward_token_eager(dummy_tok, dummy_pos);
        }
        CUDA_CHECK(cudaStreamSynchronize(main_stream_));

        CUDA_CHECK(cudaMemcpy(buf_input_token_.i32(), &dummy_tok, sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(buf_input_pos_.i32(), &dummy_pos, sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(buf_track_flag_.i32(), &dummy_flag, sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaStreamSynchronize(main_stream_));

        // Capture forward graph
        cudaError_t cap_err = cudaStreamBeginCapture(main_stream_, cudaStreamCaptureModeGlobal);
        if (cap_err != cudaSuccess) {
            LOG_WARN("cudaStreamBeginCapture failed: %s. Falling back to eager mode.", cudaGetErrorString(cap_err));
            graph_captured_ = false;
            return;
        }

        forward_token_device_body(dummy_tok, dummy_pos);

        cudaError_t end_err = cudaStreamEndCapture(main_stream_, &graph_);
        if (end_err != cudaSuccess) {
            LOG_WARN("cudaStreamEndCapture failed: %s. Falling back to eager mode.", cudaGetErrorString(end_err));
            graph_captured_ = false;
            return;
        }

        cudaError_t inst_err = cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0);
        if (inst_err != cudaSuccess) {
            LOG_WARN("cudaGraphInstantiate failed: %s. Falling back to eager mode.", cudaGetErrorString(inst_err));
            graph_captured_ = false;
            return;
        }

        graph_captured_ = true;
        LOG_INFO("CUDA Graph instantiated successfully! Decode speed accelerated.");

        if (cfg_.architecture == ModelArch::QWEN) {
            init_batch_cuda_graphs();
        }
        if (qwen_draft_.loaded_) {
            qwen_draft_.init_cuda_graph(main_stream_);
        }
    }

    void init_batched_prefill() {
        if (cfg_.architecture != ModelArch::DEEPSEEK_V4) {
            enable_batched_prefill_ = false;
            return;
        }
        if (batched_prefill_mode_ == "off") {
            enable_batched_prefill_ = false;
            LOG_INFO("[Prefill] Batched prefill explicitly disabled via CLI (--batched-prefill off).");
            return;
        }

        // Tier 0: Embedding format check (batched embedding lookup currently expects BF16 embeddings)
        if (embed_weight_.dtype == "int4") {
            LOG_INFO("[Prefill] INT4 embedding weights detected. Using sequential prefill.");
            enable_batched_prefill_ = false;
            return;
        }

        // Tier 1: Check all-resident MoE topology
        bool all_res = expert_loader_.all_resident(cfg_.num_hidden_layers);
        if (!all_res) {
            if (batched_prefill_mode_ == "on") {
                LOG_WARN("[Prefill] Warning: --batched-prefill on requested but experts are not all resident in VRAM. Forcing safe sequential prefill to prevent cache thrashing.");
            } else {
                LOG_INFO("[Prefill] Constrained VRAM detected (experts are offloaded). Batched prefill disabled (using safe sequential prefill).");
            }
            enable_batched_prefill_ = false;
            return;
        }

        // Tier 2: Check available VRAM headroom
        size_t free_bytes = 0, total_bytes = 0;
        cudaError_t mem_err = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (mem_err != cudaSuccess) {
            enable_batched_prefill_ = false;
            LOG_WARN("[Prefill] cudaMemGetInfo failed. Disabling batched prefill.");
            return;
        }

        int M = prefill_chunk_size_;
        int dim = cfg_.hidden_size;
        int hc = cfg_.hc_mult;
        int q_lora = cfg_.q_lora_rank;
        int o_lora = cfg_.o_lora_rank;
        int o_groups = cfg_.o_groups;
        int max_lora = std::max(q_lora, o_groups * o_lora);
        int n_heads = cfg_.num_attention_heads;
        int head_dim_val = cfg_.head_dim;
        int top_k = cfg_.num_experts_per_tok;
        int moe_inter = cfg_.moe_intermediate_size;
        int n_experts = cfg_.n_routed_experts;

        size_t req_bytes = 0;
        req_bytes += (size_t)M * sizeof(int32_t); // tokens
        req_bytes += (size_t)M * sizeof(int32_t); // pos
        req_bytes += (size_t)M * dim * sizeof(__nv_bfloat16); // hidden
        req_bytes += (size_t)M * (hc * dim) * sizeof(__nv_bfloat16); // hc_state
        req_bytes += (size_t)M * (hc * dim) * sizeof(__nv_bfloat16); // hc_after_attn
        req_bytes += (size_t)M * ((2 + hc) * hc) * sizeof(float); // hc_mixes
        req_bytes += (size_t)M * hc * sizeof(float); // hc_pre
        req_bytes += (size_t)M * hc * sizeof(float); // hc_post
        req_bytes += (size_t)M * (hc * hc) * sizeof(float); // hc_comb
        req_bytes += (size_t)M * max_lora * sizeof(__nv_bfloat16); // lora
        req_bytes += (size_t)M * (n_heads * head_dim_val) * sizeof(__nv_bfloat16); // q
        req_bytes += (size_t)M * head_dim_val * sizeof(__nv_bfloat16); // kv
        req_bytes += (size_t)M * (n_heads * head_dim_val) * sizeof(__nv_bfloat16); // attn_out
        req_bytes += (size_t)M * n_experts * sizeof(__nv_bfloat16); // scores_bf16
        req_bytes += (size_t)M * top_k * sizeof(int32_t); // topk_idx
        req_bytes += (size_t)M * top_k * sizeof(float); // topk_vals
        req_bytes += (size_t)M * (top_k * moe_inter) * sizeof(__nv_bfloat16); // gate
        req_bytes += (size_t)M * (top_k * dim) * sizeof(__nv_bfloat16); // down
        req_bytes += (size_t)M * moe_inter * sizeof(__nv_bfloat16); // shared_gate
        req_bytes += (size_t)M * moe_inter * sizeof(__nv_bfloat16); // shared_up
        req_bytes += (size_t)M * dim * sizeof(__nv_bfloat16); // shared_down

        constexpr size_t SAFETY_MARGIN_BYTES = 512 * 1024 * 1024ULL; // 512 MB
        if (free_bytes < req_bytes + SAFETY_MARGIN_BYTES) {
            LOG_WARN("[Prefill] Insufficient free VRAM for batched prefill (required: %.1f MB, free: %.1f MB). Using safe sequential prefill.",
                     (double)req_bytes / (1024.0 * 1024.0), (double)free_bytes / (1024.0 * 1024.0));
            enable_batched_prefill_ = false;
            return;
        }

        // Allocate GPU buffers
        prefill_bufs_.alloc(M, dim, hc, q_lora, o_lora, o_groups, n_heads, head_dim_val, top_k, moe_inter, n_experts);

        // Allocate pinned host buffers
        if (!h_prefill_tok_) {
            CUDA_CHECK(cudaMallocHost(&h_prefill_tok_, M * sizeof(int32_t)));
            CUDA_CHECK(cudaMallocHost(&h_prefill_pos_, M * sizeof(int32_t)));
        }

        enable_batched_prefill_ = true;
        LOG_INFO("[Prefill] Batched prefill enabled (chunk_size=%d, allocated=%.1f MB, all-resident mode).",
                 M, (double)req_bytes / (1024.0 * 1024.0));
    }

    // ── Forward pass for a single token (decode mode) ───────────────────────
    // Returns logits on GPU [vocab_size] in F32

    // Persistent RNG for sampling (seeded once)
    std::mt19937 rng_{std::random_device{}()};

    void forward_token(int token_id, int position) {
        if (!h_single_tok_) {
            CUDA_CHECK(cudaMallocHost(&h_single_tok_, sizeof(int32_t)));
            CUDA_CHECK(cudaMallocHost(&h_single_pos_, sizeof(int32_t)));
            CUDA_CHECK(cudaMallocHost(&h_single_flag_, sizeof(int32_t)));
        }
        int track_flag = (track_expert_freq_ && track_current_token_) ? 1 : 0;
        *h_single_flag_ = track_flag;
        *h_single_tok_ = token_id;
        *h_single_pos_ = position;
        if (graph_captured_) {
            CUDA_CHECK(cudaMemcpyAsync(buf_track_flag_.i32(), h_single_flag_, sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
            CUDA_CHECK(cudaMemcpyAsync(buf_input_token_.i32(), h_single_tok_, sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
            CUDA_CHECK(cudaMemcpyAsync(buf_input_pos_.i32(), h_single_pos_, sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
            CUDA_CHECK(cudaGraphLaunch(graph_exec_, main_stream_));
        } else {
            forward_token_eager(token_id, position);
        }
        if (track_expert_freq_ && step_topk_host_ && d_step_topk_.data) {
            int slot = decode_step_idx_ % 2;
            int n_layers = cfg_.num_hidden_layers;
            int top_k = cfg_.num_experts_per_tok;
            int32_t* dst_ptr = step_topk_host_ + slot * (n_layers * top_k);
            CUDA_CHECK(cudaMemcpyAsync(dst_ptr, d_step_topk_.data,
                                       (size_t)n_layers * top_k * sizeof(int32_t),
                                       cudaMemcpyDeviceToHost, main_stream_));
        }
    }

    void forward_token_eager(int token_id, int position, bool do_compute_logits = true) {
        if (!h_single_tok_) {
            CUDA_CHECK(cudaMallocHost(&h_single_tok_, sizeof(int32_t)));
            CUDA_CHECK(cudaMallocHost(&h_single_pos_, sizeof(int32_t)));
            CUDA_CHECK(cudaMallocHost(&h_single_flag_, sizeof(int32_t)));
        }
        int track_flag = (track_expert_freq_ && track_current_token_) ? 1 : 0;
        *h_single_flag_ = track_flag;
        *h_single_tok_ = token_id;
        *h_single_pos_ = position;
        CUDA_CHECK(cudaMemcpyAsync(buf_track_flag_.i32(), h_single_flag_, sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
        CUDA_CHECK(cudaMemcpyAsync(buf_input_token_.i32(), h_single_tok_, sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
        CUDA_CHECK(cudaMemcpyAsync(buf_input_pos_.i32(), h_single_pos_, sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
        forward_token_device_body(token_id, position, do_compute_logits);
#if defined(__APPLE__)
        CUDA_CHECK(cudaStreamSynchronize(main_stream_));
#endif
        if (track_expert_freq_ && step_topk_host_ && d_step_topk_.data) {
            int slot = decode_step_idx_ % 2;
            int n_layers = cfg_.num_hidden_layers;
            int top_k = cfg_.num_experts_per_tok;
            int32_t* dst_ptr = step_topk_host_ + slot * (n_layers * top_k);
            CUDA_CHECK(cudaMemcpyAsync(dst_ptr, d_step_topk_.data,
                                       (size_t)n_layers * top_k * sizeof(int32_t),
                                       cudaMemcpyDeviceToHost, main_stream_));
        }
    }

    void forward_token_device_body(int token_id, int position, bool do_compute_logits = true) {
        int dim = cfg_.hidden_size;
        int hc = cfg_.hc_mult;

        if (cfg_.is_nemotron()) {
            // 1. Embedding lookup
            if (embed_weight_.dtype == "fp4") {
                embedding_fp4_cuda(buf_hidden_.bf16(), (const uint8_t*)embed_weight_.data, (const uint8_t*)embed_weight_scale_.data, buf_input_token_.i32(), 1, dim, main_stream_);
            } else if (embed_weight_.dtype == "int4") {
                embedding_int4_cuda(buf_hidden_.bf16(), (uint8_t*)embed_weight_.data, embed_weight_scale_.bf16(), buf_input_token_.i32(), 1, dim, main_stream_);
            } else {
                embedding_cuda(buf_hidden_.bf16(), embed_weight_.bf16(), buf_input_token_.i32(), 1, dim, main_stream_);
            }

            // 2. Process each layer
            for (int layer = 0; layer < cfg_.num_hidden_layers; layer++) {
                forward_layer_nemotron(layer, token_id, position);
            }

            if (do_compute_logits) {
                // 3. Final RMSNorm
                rms_norm_cuda(buf_hidden_.bf16(), buf_hidden_.bf16(),
                              norm_weight_.bf16(), dim, cfg_.rms_norm_eps, main_stream_);

                // 4. Compute logits
                compute_logits();
            }
            return;
        }

        if (cfg_.is_qwen()) {
            // 1. Standard Embedding lookup
            if (embed_weight_.dtype == "fp4") {
                embedding_fp4_cuda(buf_hidden_.bf16(), (const uint8_t*)embed_weight_.data, (const uint8_t*)embed_weight_scale_.data, buf_input_token_.i32(), 1, dim, main_stream_);
            } else if (embed_weight_.dtype == "int4") {
                embedding_int4_cuda(buf_hidden_.bf16(), (uint8_t*)embed_weight_.data, embed_weight_scale_.bf16(), buf_input_token_.i32(), 1, dim, main_stream_);
            } else {
                embedding_cuda(buf_hidden_.bf16(), embed_weight_.bf16(), buf_input_token_.i32(), 1, dim, main_stream_);
            }

            // Override with vision tower embedding for image tokens
            if (active_visual_embeddings_ && position >= visual_embed_start_pos_ && position < visual_embed_start_pos_ + visual_embed_num_tokens_) {
                int v_idx = position - visual_embed_start_pos_;
                CUDA_CHECK(cudaMemcpyAsync(buf_hidden_.bf16(),
                                           active_visual_embeddings_ + (size_t)v_idx * dim,
                                           dim * sizeof(__nv_bfloat16),
                                           cudaMemcpyDeviceToDevice, main_stream_));
            }

            // 2. Process each layer
            for (int layer = 0; layer < cfg_.num_hidden_layers; layer++) {
                forward_layer_qwen(layer, position);
            }

            // 3. Final norm
            if (cfg_.architecture == ModelArch::QWEN) {
                rms_norm_one_centered_cuda(buf_hidden_.bf16(), buf_hidden_.bf16(),
                                           norm_weight_.bf16(), dim, cfg_.rms_norm_eps, main_stream_);
            } else {
                rms_norm_cuda(buf_hidden_.bf16(), buf_hidden_.bf16(),
                              norm_weight_.bf16(), dim, cfg_.rms_norm_eps, main_stream_);
            }

            // 4. Save POST-norm hidden state for MTP drafter
            //    (MTP expects the output of the final RMSNorm, per Qwen3 MTP spec)
            if (mtp_drafter_.loaded_) {
                CUDA_CHECK(cudaMemcpyAsync(buf_hidden2_.bf16(), buf_hidden_.bf16(),
                                           dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, main_stream_));
            }

            // 5. Logits: hidden @ head_weight.T -> [vocab_size]
            if (do_compute_logits) {
                compute_logits();
            }
            return;
        }

        // 1 & 2. Embedding lookup and broadcast to HC copies: [1, dim] -> [hc, dim]
        if (embed_weight_.dtype == "int4") {
            embedding_int4_broadcast_device_id_cuda(buf_hidden_.bf16(), buf_hc_state_.bf16(),
                                               (uint8_t*)embed_weight_.data, embed_weight_scale_.bf16(), buf_input_token_.i32(), dim, hc, main_stream_);
        } else {
            embedding_broadcast_device_id_cuda(buf_hidden_.bf16(), buf_hc_state_.bf16(),
                                               embed_weight_.bf16(), buf_input_token_.i32(), dim, hc, main_stream_);
        }

        // Override with vision tower embedding for image tokens
        if (active_visual_embeddings_ && position >= visual_embed_start_pos_ && position < visual_embed_start_pos_ + visual_embed_num_tokens_) {
            int v_idx = position - visual_embed_start_pos_;
            const __nv_bfloat16* src_vis = active_visual_embeddings_ + (size_t)v_idx * dim;
            CUDA_CHECK(cudaMemcpyAsync(buf_hidden_.bf16(),
                                       src_vis,
                                       dim * sizeof(__nv_bfloat16),
                                       cudaMemcpyDeviceToDevice, main_stream_));
            for (int h = 0; h < hc; h++) {
                CUDA_CHECK(cudaMemcpyAsync(buf_hc_state_.bf16() + (size_t)h * dim,
                                           src_vis,
                                           dim * sizeof(__nv_bfloat16),
                                           cudaMemcpyDeviceToDevice, main_stream_));
            }
        }

        // 3. Process each layer
        for (int layer = 0; layer < cfg_.num_hidden_layers; layer++) {
            forward_layer(layer, token_id, position);
        }

        // 4. Head HC: reduce [hc, dim] -> [dim]
        hc_head_reduce();

        // 5. Final norm
        rms_norm_cuda(buf_hidden_.bf16(), buf_hidden_.bf16(),
                      norm_weight_.bf16(), dim, cfg_.rms_norm_eps, main_stream_);

        // 6. Logits: hidden @ head_weight.T -> [vocab_size]
        if (do_compute_logits) {
            compute_logits();
        }

        if (track_expert_freq_ && track_current_token_ && !is_control_token(token_id)) {
            expert_freq_total_tokens_++;
        }
    }

    // ── Pinned System and Rolling Turn KV Cache Snapshots ───────────────────
    struct KVSnapshot {
        bool valid = false;
        std::vector<int> tokens;

        struct LayerSnapshot {
            // DeepSeek V4
            GPUTensor snap_kv_cache;
            GPUTensor snap_comp_kv_cache;
            GPUTensor snap_comp_kv_state;
            GPUTensor snap_comp_score_state;
            GPUTensor snap_indexer_comp_kv_cache;
            GPUTensor snap_indexer_comp_kv_state;
            GPUTensor snap_indexer_comp_score_state;
            int comp_kv_count = 0;

            // Qwen
            GPUTensor snap_k_cache_gqa;
            GPUTensor snap_v_cache_gqa;
            GPUTensor snap_ssm_state;
            GPUTensor snap_conv_state;
        };
        std::vector<LayerSnapshot> layers;

        GPUTensor snap_hc_state;
        GPUTensor snap_hc_after_attn;
    };

    KVSnapshot system_kv_snapshot_;
    KVSnapshot turn_kv_snapshot_;

    void snapshot_kv(KVSnapshot& snap, const std::vector<int>& tokens) {
        if (tokens.empty()) return;
        snap.tokens = tokens;
        snap.layers.resize(cfg_.num_hidden_layers);

        for (int l = 0; l < cfg_.num_hidden_layers; l++) {
            auto& lw = layers_[l];
            auto& l_snap = snap.layers[l];

            if (lw.kv_cache.data) {
                if (!l_snap.snap_kv_cache.data) l_snap.snap_kv_cache.alloc(lw.kv_cache.size_bytes);
                CUDA_CHECK(cudaMemcpyAsync(l_snap.snap_kv_cache.data, lw.kv_cache.data, lw.kv_cache.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
            }
            if (lw.comp_kv_cache.data) {
                if (!l_snap.snap_comp_kv_cache.data) l_snap.snap_comp_kv_cache.alloc(lw.comp_kv_cache.size_bytes);
                CUDA_CHECK(cudaMemcpyAsync(l_snap.snap_comp_kv_cache.data, lw.comp_kv_cache.data, lw.comp_kv_cache.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
                if (lw.d_comp_kv_count.data) {
                    CUDA_CHECK(cudaMemcpyAsync(&l_snap.comp_kv_count, lw.d_comp_kv_count.data, sizeof(int32_t), cudaMemcpyDeviceToHost, main_stream_));
                } else {
                    l_snap.comp_kv_count = lw.comp_kv_count;
                }
            }
            if (lw.comp_kv_state.data) {
                if (!l_snap.snap_comp_kv_state.data) l_snap.snap_comp_kv_state.alloc(lw.comp_kv_state.size_bytes);
                CUDA_CHECK(cudaMemcpyAsync(l_snap.snap_comp_kv_state.data, lw.comp_kv_state.data, lw.comp_kv_state.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
            }
            if (lw.comp_score_state.data) {
                if (!l_snap.snap_comp_score_state.data) l_snap.snap_comp_score_state.alloc(lw.comp_score_state.size_bytes);
                CUDA_CHECK(cudaMemcpyAsync(l_snap.snap_comp_score_state.data, lw.comp_score_state.data, lw.comp_score_state.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
            }
            if (lw.indexer_comp_kv_cache.data) {
                if (!l_snap.snap_indexer_comp_kv_cache.data) l_snap.snap_indexer_comp_kv_cache.alloc(lw.indexer_comp_kv_cache.size_bytes);
                CUDA_CHECK(cudaMemcpyAsync(l_snap.snap_indexer_comp_kv_cache.data, lw.indexer_comp_kv_cache.data, lw.indexer_comp_kv_cache.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
            }
            if (lw.indexer_comp_kv_state.data) {
                if (!l_snap.snap_indexer_comp_kv_state.data) l_snap.snap_indexer_comp_kv_state.alloc(lw.indexer_comp_kv_state.size_bytes);
                CUDA_CHECK(cudaMemcpyAsync(l_snap.snap_indexer_comp_kv_state.data, lw.indexer_comp_kv_state.data, lw.indexer_comp_kv_state.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
            }
            if (lw.indexer_comp_score_state.data) {
                if (!l_snap.snap_indexer_comp_score_state.data) l_snap.snap_indexer_comp_score_state.alloc(lw.indexer_comp_score_state.size_bytes);
                CUDA_CHECK(cudaMemcpyAsync(l_snap.snap_indexer_comp_score_state.data, lw.indexer_comp_score_state.data, lw.indexer_comp_score_state.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
            }

            // Qwen
            if (lw.k_cache_gqa.data) {
                int n_kv = cfg_.num_key_value_heads > 0 ? cfg_.num_key_value_heads : 4;
                int h_dim = cfg_.head_dim > 0 ? cfg_.head_dim : 256;
                size_t active_gqa_bytes = (size_t)tokens.size() * n_kv * h_dim * sizeof(uint8_t);
                if (active_gqa_bytes > lw.k_cache_gqa.size_bytes || active_gqa_bytes == 0) active_gqa_bytes = lw.k_cache_gqa.size_bytes;
                if (!l_snap.snap_k_cache_gqa.data || l_snap.snap_k_cache_gqa.size_bytes < active_gqa_bytes) {
                    if (gpu_caps_.is_vram_constrained) l_snap.snap_k_cache_gqa.alloc_host(active_gqa_bytes);
                    else l_snap.snap_k_cache_gqa.alloc(active_gqa_bytes);
                }
                auto kind = l_snap.snap_k_cache_gqa.is_host ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice;
                CUDA_CHECK(cudaMemcpyAsync(l_snap.snap_k_cache_gqa.data, lw.k_cache_gqa.data, active_gqa_bytes, kind, main_stream_));
            }
            if (lw.v_cache_gqa.data) {
                int n_kv = cfg_.num_key_value_heads > 0 ? cfg_.num_key_value_heads : 4;
                int h_dim = cfg_.head_dim > 0 ? cfg_.head_dim : 256;
                size_t active_gqa_bytes = (size_t)tokens.size() * n_kv * h_dim * sizeof(uint8_t);
                if (active_gqa_bytes > lw.v_cache_gqa.size_bytes || active_gqa_bytes == 0) active_gqa_bytes = lw.v_cache_gqa.size_bytes;
                if (!l_snap.snap_v_cache_gqa.data || l_snap.snap_v_cache_gqa.size_bytes < active_gqa_bytes) {
                    if (gpu_caps_.is_vram_constrained) l_snap.snap_v_cache_gqa.alloc_host(active_gqa_bytes);
                    else l_snap.snap_v_cache_gqa.alloc(active_gqa_bytes);
                }
                auto kind = l_snap.snap_v_cache_gqa.is_host ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice;
                CUDA_CHECK(cudaMemcpyAsync(l_snap.snap_v_cache_gqa.data, lw.v_cache_gqa.data, active_gqa_bytes, kind, main_stream_));
            }
            if (lw.ssm_state.data) {
                if (!l_snap.snap_ssm_state.data) {
                    if (gpu_caps_.is_vram_constrained) l_snap.snap_ssm_state.alloc_host(lw.ssm_state.size_bytes);
                    else l_snap.snap_ssm_state.alloc(lw.ssm_state.size_bytes);
                }
                auto kind = l_snap.snap_ssm_state.is_host ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice;
                CUDA_CHECK(cudaMemcpyAsync(l_snap.snap_ssm_state.data, lw.ssm_state.data, lw.ssm_state.size_bytes, kind, main_stream_));
            }
            if (lw.conv_state.data) {
                if (!l_snap.snap_conv_state.data) {
                    if (gpu_caps_.is_vram_constrained) l_snap.snap_conv_state.alloc_host(lw.conv_state.size_bytes);
                    else l_snap.snap_conv_state.alloc(lw.conv_state.size_bytes);
                }
                auto kind = l_snap.snap_conv_state.is_host ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice;
                CUDA_CHECK(cudaMemcpyAsync(l_snap.snap_conv_state.data, lw.conv_state.data, lw.conv_state.size_bytes, kind, main_stream_));
            }
        }

        if (buf_hc_state_.data) {
            if (!snap.snap_hc_state.data) snap.snap_hc_state.alloc(buf_hc_state_.size_bytes);
            CUDA_CHECK(cudaMemcpyAsync(snap.snap_hc_state.data, buf_hc_state_.data, buf_hc_state_.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
        }
        if (buf_hc_after_attn_.data) {
            if (!snap.snap_hc_after_attn.data) snap.snap_hc_after_attn.alloc(buf_hc_after_attn_.size_bytes);
            CUDA_CHECK(cudaMemcpyAsync(snap.snap_hc_after_attn.data, buf_hc_after_attn_.data, buf_hc_after_attn_.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
        }

        CUDA_CHECK(cudaStreamSynchronize(main_stream_));
        snap.valid = true;
    }

    void snapshot_system_kv(const std::vector<int>& tokens) {
        snapshot_kv(system_kv_snapshot_, tokens);
        LOG_INFO("Pinned System KV Cache snapshot created (%zu tokens across %d layers).",
                 tokens.size(), cfg_.num_hidden_layers);
    }

    void restore_kv(const KVSnapshot& snap) {
        if (!snap.valid) return;

        for (int l = 0; l < cfg_.num_hidden_layers; l++) {
            auto& lw = layers_[l];
            const auto& l_snap = snap.layers[l];

            if (lw.kv_cache.data && l_snap.snap_kv_cache.data) {
                CUDA_CHECK(cudaMemcpyAsync(lw.kv_cache.data, l_snap.snap_kv_cache.data, lw.kv_cache.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
            }
            if (lw.comp_kv_cache.data && l_snap.snap_comp_kv_cache.data) {
                CUDA_CHECK(cudaMemcpyAsync(lw.comp_kv_cache.data, l_snap.snap_comp_kv_cache.data, lw.comp_kv_cache.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
                lw.comp_kv_count = l_snap.comp_kv_count;
                if (lw.d_comp_kv_count.data) {
                    int32_t val = l_snap.comp_kv_count;
                    CUDA_CHECK(cudaMemcpyAsync(lw.d_comp_kv_count.data, &val, sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                }
                if (lw.d_attn_cache_len.data) {
                    int32_t val = (int32_t)snap.tokens.size();
                    CUDA_CHECK(cudaMemcpyAsync(lw.d_attn_cache_len.data, &val, sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                }
            }
            if (lw.comp_kv_state.data && l_snap.snap_comp_kv_state.data) {
                CUDA_CHECK(cudaMemcpyAsync(lw.comp_kv_state.data, l_snap.snap_comp_kv_state.data, lw.comp_kv_state.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
            }
            if (lw.comp_score_state.data && l_snap.snap_comp_score_state.data) {
                CUDA_CHECK(cudaMemcpyAsync(lw.comp_score_state.data, l_snap.snap_comp_score_state.data, lw.comp_score_state.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
            }
            if (lw.indexer_comp_kv_cache.data && l_snap.snap_indexer_comp_kv_cache.data) {
                CUDA_CHECK(cudaMemcpyAsync(lw.indexer_comp_kv_cache.data, l_snap.snap_indexer_comp_kv_cache.data, lw.indexer_comp_kv_cache.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
            }
            if (lw.indexer_comp_kv_state.data && l_snap.snap_indexer_comp_kv_state.data) {
                CUDA_CHECK(cudaMemcpyAsync(lw.indexer_comp_kv_state.data, l_snap.snap_indexer_comp_kv_state.data, lw.indexer_comp_kv_state.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
            }
            if (lw.indexer_comp_score_state.data && l_snap.snap_indexer_comp_score_state.data) {
                CUDA_CHECK(cudaMemcpyAsync(lw.indexer_comp_score_state.data, l_snap.snap_indexer_comp_score_state.data, lw.indexer_comp_score_state.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
            }

            // Qwen
            if (lw.k_cache_gqa.data && l_snap.snap_k_cache_gqa.data) {
                int n_kv = cfg_.num_key_value_heads > 0 ? cfg_.num_key_value_heads : 4;
                int h_dim = cfg_.head_dim > 0 ? cfg_.head_dim : 256;
                size_t active_gqa_bytes = (size_t)snap.tokens.size() * n_kv * h_dim * sizeof(uint8_t);
                if (active_gqa_bytes > l_snap.snap_k_cache_gqa.size_bytes || active_gqa_bytes == 0) {
                    active_gqa_bytes = l_snap.snap_k_cache_gqa.size_bytes;
                }
                auto kind = l_snap.snap_k_cache_gqa.is_host ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
                CUDA_CHECK(cudaMemcpyAsync(lw.k_cache_gqa.data, l_snap.snap_k_cache_gqa.data, active_gqa_bytes, kind, main_stream_));
            }
            if (lw.v_cache_gqa.data && l_snap.snap_v_cache_gqa.data) {
                int n_kv = cfg_.num_key_value_heads > 0 ? cfg_.num_key_value_heads : 4;
                int h_dim = cfg_.head_dim > 0 ? cfg_.head_dim : 256;
                size_t active_gqa_bytes = (size_t)snap.tokens.size() * n_kv * h_dim * sizeof(uint8_t);
                if (active_gqa_bytes > l_snap.snap_v_cache_gqa.size_bytes || active_gqa_bytes == 0) {
                    active_gqa_bytes = l_snap.snap_v_cache_gqa.size_bytes;
                }
                auto kind = l_snap.snap_v_cache_gqa.is_host ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
                CUDA_CHECK(cudaMemcpyAsync(lw.v_cache_gqa.data, l_snap.snap_v_cache_gqa.data, active_gqa_bytes, kind, main_stream_));
            }
            if (lw.ssm_state.data && l_snap.snap_ssm_state.data) {
                auto kind = l_snap.snap_ssm_state.is_host ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
                CUDA_CHECK(cudaMemcpyAsync(lw.ssm_state.data, l_snap.snap_ssm_state.data, lw.ssm_state.size_bytes, kind, main_stream_));
            }
            if (lw.conv_state.data && l_snap.snap_conv_state.data) {
                auto kind = l_snap.snap_conv_state.is_host ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
                CUDA_CHECK(cudaMemcpyAsync(lw.conv_state.data, l_snap.snap_conv_state.data, lw.conv_state.size_bytes, kind, main_stream_));
            }
        }

        if (buf_hc_state_.data && snap.snap_hc_state.data) {
            CUDA_CHECK(cudaMemcpyAsync(buf_hc_state_.data, snap.snap_hc_state.data, buf_hc_state_.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
        }
        if (buf_hc_after_attn_.data && snap.snap_hc_after_attn.data) {
            CUDA_CHECK(cudaMemcpyAsync(buf_hc_after_attn_.data, snap.snap_hc_after_attn.data, buf_hc_after_attn_.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
        }

        if (qwen_draft_.loaded_) {
            qwen_draft_.reset_state(main_stream_);
        }
        if (mtp_drafter_.loaded_) {
            mtp_drafter_.reset_kv_cache(main_stream_);
        }

        cached_tokens_ = snap.tokens;
    }

    void restore_system_kv() {
        restore_kv(system_kv_snapshot_);
    }

    // ── Prefill Prompt Prefix without Generation (for Startup KV Cache Pre-warming) ──
    void prefill_prefix(const std::vector<int>& prefix_tokens) {
        if (prefix_tokens.empty()) return;
        reset_all_kv_caches();
        cached_tokens_.clear();

        track_current_token_ = false;
        ensure_prefill_host_capacity(prefix_tokens.size());
        for (size_t i = 0; i < prefix_tokens.size(); i++) {
            h_prefill_tok_[i] = prefix_tokens[i];
            h_prefill_pos_[i] = (int32_t)i;
        }
        if (!h_single_flag_) {
            CUDA_CHECK(cudaMallocHost(&h_single_flag_, sizeof(int32_t)));
        }
        *h_single_flag_ = 0;

        auto prefill_start_time = std::chrono::steady_clock::now();

        if (cfg_.is_qwen()) {
            size_t curr = 0;
            while (curr < prefix_tokens.size()) {
                size_t remaining = prefix_tokens.size() - curr;
                int chunk_m = (remaining >= 512) ? 512 : (int)remaining;
                CUDA_CHECK(cudaMemcpyAsync(buf_input_pos_batch_.i32(), &h_prefill_pos_[curr],
                                           chunk_m * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                CUDA_CHECK(cudaMemcpyAsync(buf_input_tokens_batch_.i32(), &h_prefill_tok_[curr],
                                           chunk_m * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                if (curr % 128 == 0 || curr + chunk_m >= prefix_tokens.size()) {
                    LOG_INFO("Prefill prefix progress: %zu / %zu tokens (chunk_m=%d)...", curr, prefix_tokens.size(), chunk_m);
                }
                forward_token_batch_qwen_device_body((int)curr, chunk_m, /*compute_logits=*/false);
                curr += chunk_m;
            }
        } else if (cfg_.is_nemotron()) {
            prefill_prompt_batched_nemotron(prefix_tokens, 0, /*compute_last_logits=*/false);
        } else {
            // ModelArch::DEEPSEEK_V4
            if (!cfg_.is_nemotron() && enable_batched_prefill_ && prefix_tokens.size() > 1) {
                prefill_prompt_batched_deepseek(prefix_tokens);
            } else {
                for (size_t i = 0; i < prefix_tokens.size(); i++) {
                    if (graph_captured_) {
                        CUDA_CHECK(cudaMemcpyAsync(buf_track_flag_.i32(), h_single_flag_, sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                        CUDA_CHECK(cudaMemcpyAsync(buf_input_token_.i32(), &h_prefill_tok_[i], sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                        CUDA_CHECK(cudaMemcpyAsync(buf_input_pos_.i32(), &h_prefill_pos_[i], sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                        CUDA_CHECK(cudaGraphLaunch(graph_exec_, main_stream_));
                    } else {
                        forward_token_eager(prefix_tokens[i], (int)i, (i + 1 == prefix_tokens.size()));
                    }
                }
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(main_stream_));

        auto prefill_end_time = std::chrono::steady_clock::now();
        double prefill_sec = std::chrono::duration<double>(prefill_end_time - prefill_start_time).count();
        double prefill_tps = (prefill_sec > 0.0) ? (double)prefix_tokens.size() / prefill_sec : 0.0;
        LOG_INFO("[PREFILL STATS] %zu prefix tokens pre-warmed in %.3fs -> %.2f tok/s",
                 prefix_tokens.size(), prefill_sec, prefill_tps);

        cached_tokens_ = prefix_tokens;
        snapshot_system_kv(prefix_tokens);
    }

    // ── Zero Decode Bypass (ZDB) Structures & Method ────────────────────────

    using ZeroDecodeCandidate = moecher::zdb::ZeroDecodeCandidate;
    using ZeroDecodeResult = moecher::zdb::ZeroDecodeResult;


    ZeroDecodeResult execute_zero_decode_bypass(
        const std::vector<int>& prompt_in,
        const std::vector<std::string>& candidate_choices,
        float temperature = 1.0f,
        const std::string& output_format = "json",
        bool bypass_prefix_cache = false,
        float min_confidence = 0.70f,
        float max_top1_gap = 0.5f
    ) {
        ZeroDecodeResult result;
        result.decode_passes = 0;
        if (prompt_in.empty() || candidate_choices.empty()) {
            return result;
        }

        auto start_time = std::chrono::steady_clock::now();

        std::vector<int> prompt = prompt_in;
        if (prompt.size() > (size_t)cfg_.max_seq_len) {
            LOG_WARN("[ZDB] Prompt size (%zu) exceeds max_seq_len (%d). Clamping prompt.",
                     prompt.size(), cfg_.max_seq_len);
            size_t excess = prompt.size() - (size_t)cfg_.max_seq_len;
            prompt.erase(prompt.begin(), prompt.begin() + excess);
        }
        result.prompt_tokens = prompt.size();

        // ── Prefix Matching against existing KV cache ────────────────────────────
        size_t prefix_len = 0;
        if (bypass_prefix_cache) {
            reset_all_kv_caches();
            cached_tokens_.clear();
            turn_kv_snapshot_.valid = false;
            turn_kv_snapshot_.tokens.clear();
        } else {
            while (prefix_len < prompt.size() && prefix_len < cached_tokens_.size() &&
                   prompt[prefix_len] == cached_tokens_[prefix_len]) {
                prefix_len++;
            }
        }

        bool can_reuse_prefix = (prefix_len > 0);
        if ((cfg_.architecture == ModelArch::DEEPSEEK_V4 || cfg_.is_nemotron()) && prefix_len < cached_tokens_.size()) {
            can_reuse_prefix = false;
            prefix_len = 0;
        }

        if (!bypass_prefix_cache && !can_reuse_prefix && turn_kv_snapshot_.valid && !turn_kv_snapshot_.tokens.empty()) {
            size_t turn_len = turn_kv_snapshot_.tokens.size();
            if (prompt.size() >= turn_len) {
                bool turn_matches = true;
                for (size_t i = 0; i < turn_len; i++) {
                    if (prompt[i] != turn_kv_snapshot_.tokens[i]) {
                        turn_matches = false;
                        break;
                    }
                }
                if (turn_matches) {
                    restore_kv(turn_kv_snapshot_);
                    prefix_len = turn_len;
                    can_reuse_prefix = true;
                }
            }
        }

        if (!bypass_prefix_cache && !can_reuse_prefix && system_kv_snapshot_.valid && !system_kv_snapshot_.tokens.empty()) {
            size_t sys_len = system_kv_snapshot_.tokens.size();
            if (prompt.size() >= sys_len) {
                bool sys_matches = true;
                for (size_t i = 0; i < sys_len; i++) {
                    if (prompt[i] != system_kv_snapshot_.tokens[i]) {
                        sys_matches = false;
                        break;
                    }
                }
                if (sys_matches) {
                    restore_system_kv();
                    prefix_len = sys_len;
                    can_reuse_prefix = true;
                }
            }
        }

        // Ensure at least the last token is forwarded so buf_logits_ is correctly populated
        if (prefix_len >= prompt.size() && !prompt.empty()) {
            prefix_len = prompt.size() - 1;
        }

        if (!can_reuse_prefix || prefix_len == 0) {
            reset_all_kv_caches();
            prefix_len = 0;
        }

        track_current_token_ = false;
        auto prefill_start_time = std::chrono::steady_clock::now();

        if (prefix_len < prompt.size()) {
            ensure_prefill_host_capacity(prompt.size());
            for (size_t i = prefix_len; i < prompt.size(); i++) {
                h_prefill_tok_[i] = prompt[i];
                h_prefill_pos_[i] = (int32_t)i;
            }

            if (!h_single_flag_) {
                CUDA_CHECK(cudaMallocHost(&h_single_flag_, sizeof(int32_t)));
            }
            *h_single_flag_ = 0;

            bool draft_model_active = (!mtp_drafter_.loaded_ && qwen_draft_.loaded_);

            if (cfg_.is_qwen()) {
                size_t curr = prefix_len;
                while (curr + 1 < prompt.size()) {
                    size_t remaining = (prompt.size() - 1) - curr;
                    int chunk_m = (remaining >= 512) ? 512 : (int)remaining;
                    CUDA_CHECK(cudaMemcpyAsync(buf_input_pos_batch_.i32(), &h_prefill_pos_[curr],
                                               chunk_m * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                    CUDA_CHECK(cudaMemcpyAsync(buf_input_tokens_batch_.i32(), &h_prefill_tok_[curr],
                                               chunk_m * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                    forward_token_batch_qwen_device_body((int)curr, chunk_m, /*compute_logits=*/false);
                    if (draft_model_active) {
                        for (int m = 0; m < chunk_m; m++) {
                            qwen_draft_.forward_token_async_pinned(&h_prefill_tok_[curr + m],
                                                                   &h_prefill_pos_[curr + m],
                                                                   (int)(curr + m), main_stream_);
                        }
                    }
                    curr += chunk_m;
                }

                if (curr < prompt.size()) {
                    if (graph_captured_) {
                        CUDA_CHECK(cudaMemcpyAsync(buf_track_flag_.i32(), h_single_flag_, sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                        CUDA_CHECK(cudaMemcpyAsync(buf_input_token_.i32(), &h_prefill_tok_[curr], sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                        CUDA_CHECK(cudaMemcpyAsync(buf_input_pos_.i32(), &h_prefill_pos_[curr], sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                        CUDA_CHECK(cudaGraphLaunch(graph_exec_, main_stream_));
                    } else {
                        forward_token_eager(prompt[curr], (int)curr);
                    }
                    if (draft_model_active) {
                        qwen_draft_.forward_token_async_pinned(&h_prefill_tok_[curr],
                                                               &h_prefill_pos_[curr],
                                                               (int)curr, main_stream_);
                    }
                }
            } else if (cfg_.is_nemotron()) {
                prefill_prompt_batched_nemotron(prompt, (int)prefix_len, /*compute_last_logits=*/true);
            } else {
                // ModelArch::DEEPSEEK_V4
                size_t remaining = (prompt.size() > prefix_len) ? (prompt.size() - prefix_len) : 0;
                if (!cfg_.is_nemotron() && enable_batched_prefill_ && remaining > 1) {
                    prefill_prompt_batched_deepseek(prompt, (int)prefix_len);
                } else {
                    for (size_t i = prefix_len; i < prompt.size(); i++) {
                        if (graph_captured_) {
                            CUDA_CHECK(cudaMemcpyAsync(buf_track_flag_.i32(), h_single_flag_, sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                            CUDA_CHECK(cudaMemcpyAsync(buf_input_token_.i32(), &h_prefill_tok_[i], sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                            CUDA_CHECK(cudaMemcpyAsync(buf_input_pos_.i32(), &h_prefill_pos_[i], sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                            CUDA_CHECK(cudaGraphLaunch(graph_exec_, main_stream_));
                        } else {
                            forward_token_eager(prompt[i], (int)i, (i + 1 == prompt.size()));
                        }
                    }
                }
            }
            CUDA_CHECK(cudaStreamSynchronize(main_stream_));
        }

        auto prefill_end_time = std::chrono::steady_clock::now();
        result.prefill_sec = std::chrono::duration<double>(prefill_end_time - prefill_start_time).count();
        cached_tokens_ = prompt;

        // ── Step-Zero Logit Slicing ──────────────────────────────────────────────
        // Retrieve global unconstrained argmax
        if (buf_argmax_out_.data) {
            CUDA_CHECK(cudaMemcpy(&result.global_argmax_token_id, buf_argmax_out_.i32(), sizeof(int32_t), cudaMemcpyDeviceToHost));
            result.global_argmax_token_str = tokenizer_.decode({result.global_argmax_token_id});
            if (result.global_argmax_token_id >= 0 && result.global_argmax_token_id < cfg_.vocab_size) {
                CUDA_CHECK(cudaMemcpy(&result.global_argmax_logit, buf_logits_.f32() + result.global_argmax_token_id, sizeof(float), cudaMemcpyDeviceToHost));
            }
        }

        // Map each candidate choice to its token IDs and collect all unique tokens to read
        std::vector<std::vector<int>> choice_tokens;
        std::vector<int> all_tokens_to_read = moecher::zdb::collect_candidate_tokens(candidate_choices, tokenizer_, cfg_.vocab_size, choice_tokens);

        // Asynchronously copy candidate logits from GPU to CPU
        std::vector<float> host_logits(all_tokens_to_read.size(), -1e9f);
        for (size_t i = 0; i < all_tokens_to_read.size(); ++i) {
            int tid = all_tokens_to_read[i];
            CUDA_CHECK(cudaMemcpyAsync(&host_logits[i], buf_logits_.f32() + tid, sizeof(float), cudaMemcpyDeviceToHost, main_stream_));
        }
        CUDA_CHECK(cudaStreamSynchronize(main_stream_));

        std::unordered_map<int, float> logit_lookup;
        for (size_t i = 0; i < all_tokens_to_read.size(); ++i) {
            logit_lookup[all_tokens_to_read[i]] = host_logits[i];
        }

        double prefill_sec = result.prefill_sec;
        result = moecher::zdb::score_and_underwrite_candidates(
            candidate_choices, choice_tokens, logit_lookup,
            result.global_argmax_token_id, result.global_argmax_token_str, result.global_argmax_logit,
            temperature, min_confidence, max_top1_gap, output_format
        );
        result.prefill_sec = prefill_sec;
        auto total_end_time = std::chrono::steady_clock::now();
        result.total_sec = std::chrono::duration<double>(total_end_time - start_time).count();
        result.prompt_tokens = prompt.size();

        LOG_INFO("[ZDB] Zero Decode Bypass: winner='%s' (conf=%.4f) | choices=%zu | prefill=%.2fms total=%.2fms | decode passes=0",
                 result.winning_choice.c_str(), result.confidence, candidate_choices.size(),
                 result.prefill_sec * 1000.0, result.total_sec * 1000.0);

        return result;
    }

    // ── Generate tokens ─────────────────────────────────────────────────────

    std::string generate(const std::vector<int>& prompt_in, int max_tokens = 20000,
                         float temperature = 1.0f,
                         std::function<bool(const std::string&,bool)> on_token = nullptr,
                         float repetition_penalty = 1.0f,
                         bool enable_thinking = true,
                         int max_thinking_tokens = 2048,
                         float top_p = 0.95f,
                         float min_p = 0.0f,
                         int top_k = 1024,
                         const __nv_bfloat16* visual_embeddings = nullptr,
                         int visual_start_pos = -1,
                         int visual_num_tokens = 0,
                         bool bypass_prefix_cache = false) {
        expert_loader_.begin_request_epoch();
        // Ensure prompt fits within max_seq_len (defense-in-depth safety clamp)
        std::vector<int> prompt = prompt_in;
        if (prompt.size() > (size_t)cfg_.max_seq_len) {
            LOG_WARN("[Engine] Prompt size (%zu) exceeds max_seq_len (%d). Clamping prompt to context limit.",
                     prompt.size(), cfg_.max_seq_len);
            size_t excess = prompt.size() - (size_t)cfg_.max_seq_len;
            prompt.erase(prompt.begin(), prompt.begin() + excess);
        }

        // Configure visual embeddings for prompt prefill if present
        bool had_multimodal = (visual_embeddings && visual_start_pos >= 0 && visual_num_tokens > 0);
        if (had_multimodal) {
            active_visual_embeddings_ = visual_embeddings;
            visual_embed_start_pos_ = visual_start_pos;
            visual_embed_num_tokens_ = visual_num_tokens;
            if (buf_mrope_pos_.data) {
                int h_mrope[2] = {visual_start_pos, visual_num_tokens};
                CUDA_CHECK(cudaMemcpyAsync(buf_mrope_pos_.i32(), h_mrope, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
            }
            reset_all_kv_caches();
            cached_tokens_.clear();
            LOG_INFO("[Engine] Multimodal input active: visual tokens injected at pos %d..%d (%d tokens)",
                     visual_start_pos, visual_start_pos + visual_num_tokens - 1, visual_num_tokens);
        } else {
            active_visual_embeddings_ = nullptr;
            visual_embed_start_pos_ = -1;
            visual_embed_num_tokens_ = 0;
            if (buf_mrope_pos_.data) {
                int h_mrope[2] = {-1, 0};
                CUDA_CHECK(cudaMemcpyAsync(buf_mrope_pos_.i32(), h_mrope, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
            }
        }

        // Reset debug flags for this request
        dbg_first_token_ = false;
        dbg_hc_pre_call_ = 0;
        dbg_head_ = true;
        dbg_sample_count_ = 0;

        // ── Prefix Matching against existing KV cache ────────────────────────────
        size_t prefix_len = 0;
        if (bypass_prefix_cache) {
            reset_all_kv_caches();
            cached_tokens_.clear();
            turn_kv_snapshot_.valid = false;
            turn_kv_snapshot_.tokens.clear();
            LOG_INFO("[KV CACHE] Bypass prefix cache requested: forcing cold full prefill of %zu tokens", prompt.size());
        } else {
            while (prefix_len < prompt.size() && prefix_len < cached_tokens_.size() &&
                   prompt[prefix_len] == cached_tokens_[prefix_len]) {
                prefix_len++;
            }
        }

        // For architectures with recurrent/stateful compression (DeepSeek V4 MLA / Linear Attn),
        // we can safely reuse the prefix when prefix_len == cached_tokens_.size() (exact continuation)
        // or for pure positional GQA.
        bool can_reuse_prefix = (prefix_len > 0);
        if (active_visual_embeddings_) {
            can_reuse_prefix = false;
            prefix_len = 0;
        } else if ((cfg_.architecture == ModelArch::DEEPSEEK_V4 || cfg_.is_nemotron()) && prefix_len < cached_tokens_.size()) {
            can_reuse_prefix = false;
            prefix_len = 0;
        }

        // If continuous prefix matching failed, check if prompt starts with the rolling Turn KV snapshot!
        if (!bypass_prefix_cache && !active_visual_embeddings_ && !can_reuse_prefix && turn_kv_snapshot_.valid && !turn_kv_snapshot_.tokens.empty()) {
            size_t turn_len = turn_kv_snapshot_.tokens.size();
            if (prompt.size() >= turn_len) {
                bool turn_matches = true;
                size_t first_diff = turn_len;
                for (size_t i = 0; i < turn_len; i++) {
                    if (prompt[i] != turn_kv_snapshot_.tokens[i]) {
                        turn_matches = false;
                        first_diff = i;
                        break;
                    }
                }
                if (turn_matches) {
                    restore_kv(turn_kv_snapshot_);
                    prefix_len = turn_len;
                    can_reuse_prefix = true;
                    LOG_INFO("Restored rolling Turn KV Cache snapshot of %zu tokens (<0.1ms). Skipping prefill 0..%zu, evaluating %zu..%zu",
                             turn_len, turn_len > 0 ? turn_len - 1 : 0, turn_len, prompt.size() - 1);
                } else {
                    LOG_INFO("Turn KV snapshot mismatch: prompt size %zu, snapshot size %zu, first diff at index %zu (prompt[%zu]=%d, snapshot[%zu]=%d)",
                             prompt.size(), turn_len, first_diff, first_diff, prompt[first_diff], first_diff, turn_kv_snapshot_.tokens[first_diff]);
                }
            } else {
                LOG_INFO("Turn KV snapshot skipped: prompt size %zu < snapshot size %zu", prompt.size(), turn_len);
            }
        }

        // If turn snapshot matching failed, check if prompt starts with the pinned System KV snapshot!
        if (!bypass_prefix_cache && !active_visual_embeddings_ && !can_reuse_prefix && system_kv_snapshot_.valid && !system_kv_snapshot_.tokens.empty()) {
            size_t sys_len = system_kv_snapshot_.tokens.size();
            if (prompt.size() >= sys_len) {
                bool sys_matches = true;
                size_t first_diff = sys_len;
                for (size_t i = 0; i < sys_len; i++) {
                    if (prompt[i] != system_kv_snapshot_.tokens[i]) {
                        sys_matches = false;
                        first_diff = i;
                        break;
                    }
                }
                if (sys_matches) {
                    restore_system_kv();
                    prefix_len = sys_len;
                    can_reuse_prefix = true;
                    LOG_INFO("Restored pinned System KV Cache snapshot of %zu tokens (<0.1ms). Skipping system prefill 0..%zu, evaluating %zu..%zu",
                             sys_len, sys_len > 0 ? sys_len - 1 : 0, sys_len, prompt.size() - 1);
                } else {
                    LOG_INFO("System KV snapshot mismatch: prompt size %zu, snapshot size %zu, first diff at index %zu (prompt[%zu]=%d, snapshot[%zu]=%d)",
                             prompt.size(), sys_len, first_diff, first_diff, prompt[first_diff], first_diff, system_kv_snapshot_.tokens[first_diff]);
                }
            } else {
                LOG_INFO("System KV snapshot skipped: prompt size %zu < snapshot size %zu", prompt.size(), sys_len);
            }
        }

        // Ensure at least the last token is forwarded so buf_logits_ is correctly populated
        if (prefix_len >= prompt.size() && !prompt.empty()) {
            prefix_len = prompt.size() - 1;
        }

        if (!can_reuse_prefix || prefix_len == 0) {
            reset_all_kv_caches();
            prefix_len = 0;
        } else {
            LOG_INFO("Reusing KV Cache prefix of %zu tokens (skipping prefill 0..%zu, evaluating %zu..%zu)",
                     prefix_len, prefix_len > 0 ? prefix_len - 1 : 0, prefix_len, prompt.size() - 1);
        }

        // Disable tracking during prompt prefill (prevents system prompt boilerplate from skewing stats)
        track_current_token_ = false;

        auto prefill_start_time = std::chrono::steady_clock::now();

        if (prefix_len < prompt.size()) {
            ensure_prefill_host_capacity(prompt.size());
            for (size_t i = prefix_len; i < prompt.size(); i++) {
                h_prefill_tok_[i] = prompt[i];
                h_prefill_pos_[i] = (int32_t)i;
            }

            if (!h_single_flag_) {
                CUDA_CHECK(cudaMallocHost(&h_single_flag_, sizeof(int32_t)));
            }
            *h_single_flag_ = 0;

            bool draft_model_active = (!mtp_drafter_.loaded_ && qwen_draft_.loaded_);

            if (cfg_.is_qwen()) {
                size_t curr = prefix_len;
                // Prefill intermediate prompt tokens in chunks of up to 8 (skipping intermediate LM-head computation)
                while (curr + 1 < prompt.size()) {
                    size_t remaining = (prompt.size() - 1) - curr;
                    int chunk_m = (remaining >= 512) ? 512 : (int)remaining;
                    CUDA_CHECK(cudaMemcpyAsync(buf_input_pos_batch_.i32(), &h_prefill_pos_[curr],
                                               chunk_m * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                    CUDA_CHECK(cudaMemcpyAsync(buf_input_tokens_batch_.i32(), &h_prefill_tok_[curr],
                                               chunk_m * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                    forward_token_batch_qwen_device_body((int)curr, chunk_m, /*compute_logits=*/false);
                    if (draft_model_active) {
                        for (int m = 0; m < chunk_m; m++) {
                            qwen_draft_.forward_token_async_pinned(&h_prefill_tok_[curr + m],
                                                                   &h_prefill_pos_[curr + m],
                                                                   (int)(curr + m), main_stream_);
                        }
                    }
                    curr += chunk_m;
                }

                // Forward the very last token individually so buf_logits_ and buf_hidden2_ (for MTP) are populated
                if (curr < prompt.size()) {
                    if (graph_captured_) {
                        CUDA_CHECK(cudaMemcpyAsync(buf_track_flag_.i32(), h_single_flag_, sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                        CUDA_CHECK(cudaMemcpyAsync(buf_input_token_.i32(), &h_prefill_tok_[curr], sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                        CUDA_CHECK(cudaMemcpyAsync(buf_input_pos_.i32(), &h_prefill_pos_[curr], sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                        CUDA_CHECK(cudaGraphLaunch(graph_exec_, main_stream_));
                    } else {
                        forward_token_eager(prompt[curr], (int)curr);
                    }
                    if (draft_model_active) {
                        qwen_draft_.forward_token_async_pinned(&h_prefill_tok_[curr],
                                                               &h_prefill_pos_[curr],
                                                               (int)curr, main_stream_);
                    }
                }
            } else if (cfg_.is_nemotron()) {
                prefill_prompt_batched_nemotron(prompt, (int)prefix_len, /*compute_last_logits=*/true);
            } else {
                // ModelArch::DEEPSEEK_V4
                size_t remaining = (prompt.size() > prefix_len) ? (prompt.size() - prefix_len) : 0;
                if (!cfg_.is_nemotron() && enable_batched_prefill_ && remaining > 1) {
                    prefill_prompt_batched_deepseek(prompt, (int)prefix_len);
                } else {
                    for (size_t i = prefix_len; i < prompt.size(); i++) {
                        if (graph_captured_) {
                            CUDA_CHECK(cudaMemcpyAsync(buf_track_flag_.i32(), h_single_flag_, sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                            CUDA_CHECK(cudaMemcpyAsync(buf_input_token_.i32(), &h_prefill_tok_[i], sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                            CUDA_CHECK(cudaMemcpyAsync(buf_input_pos_.i32(), &h_prefill_pos_[i], sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                            CUDA_CHECK(cudaGraphLaunch(graph_exec_, main_stream_));
                        } else {
                            forward_token_eager(prompt[i], (int)i, (i + 1 == prompt.size()));
                        }
                    }
                }
            }

            // Synchronize main_stream_ ONCE at the end of the entire prefill pipeline
            CUDA_CHECK(cudaStreamSynchronize(main_stream_));
        }

        // Visual tokens are fully prefilled into KV cache; disable active override for generation
        active_visual_embeddings_ = nullptr;
        visual_embed_start_pos_ = -1;
        visual_embed_num_tokens_ = 0;

        auto prefill_end_time = std::chrono::steady_clock::now();
        double prefill_sec = std::chrono::duration<double>(prefill_end_time - prefill_start_time).count();
        size_t evaluated_tokens = (prompt.size() > prefix_len) ? (prompt.size() - prefix_len) : 0;
        double prefill_tps = (prefill_sec > 0.0 && evaluated_tokens > 0) ? (double)evaluated_tokens / prefill_sec : 0.0;
        LOG_INFO("[PREFILL STATS] %zu tokens evaluated in %.3fs -> %.2f tok/s (prefix cached: %zu / %zu, %.1f%%)",
                 evaluated_tokens, prefill_sec, prefill_tps,
                 prefix_len, prompt.size(),
                 prompt.empty() ? 0.0 : (100.0 * prefix_len / prompt.size()));

        last_prefill_tokens_ = evaluated_tokens;
        last_cached_tokens_ = prefix_len;
        last_prefill_sec_ = prefill_sec;
        last_prefill_tps_ = prefill_tps;

        // Token generation loop
        std::vector<int> output_ids;
        std::vector<int> history(prompt.begin(), prompt.end());
        int position = (int)prompt.size();
        std::string generated_text;
        std::string token_buffer;

        // Think token IDs
        int think_start_id = tokenizer_.get_token_id("<think>");
        int think_end_id = tokenizer_.get_token_id("</think>");
        if (think_start_id < 0) think_start_id = 128821;
        if (think_end_id < 0) think_end_id = 128822;

        int eos2_id = tokenizer_.get_token_id("<|end_of_sentence|>");
        if (eos2_id < 0) eos2_id = tokenizer_.get_token_id("<｜end of sentence｜>");
        int im_end_id = tokenizer_.get_token_id("<|im_end|>");
        if (im_end_id >= 0 && eos2_id < 0) eos2_id = im_end_id;

        int user_id = tokenizer_.get_token_id("<｜User｜>");
        if (user_id < 0) user_id = 128803;
        int asst_id = tokenizer_.get_token_id("<｜Assistant｜>");
        if (asst_id < 0) asst_id = 128804;

        LOG_WARN("Think tokens: start=%d end=%d, EOS=%d eos2=%d user=%d asst=%d, think_budget=%d",
                 think_start_id, think_end_id, cfg_.eos_token_id, eos2_id, user_id, asst_id, max_thinking_tokens);

        // Refusal opening tokens to suppress at the start of content generation
        std::vector<int> refusal_start_tokens;
        auto add_refusal_phrase = [&](const std::string& phrase) {
            auto enc = tokenizer_.encode(phrase);
            if (!enc.empty()) refusal_start_tokens.push_back(enc[0]);
            auto enc_sp = tokenizer_.encode(" " + phrase);
            if (!enc_sp.empty()) refusal_start_tokens.push_back(enc_sp[0]);
        };
        add_refusal_phrase("I'm sorry");
        add_refusal_phrase("I cannot");
        add_refusal_phrase("I can't");
        add_refusal_phrase("Sorry");
        add_refusal_phrase("Unfortunately");
        add_refusal_phrase("As an AI");
        add_refusal_phrase("I am unable");
        add_refusal_phrase("I apologize");
        std::sort(refusal_start_tokens.begin(), refusal_start_tokens.end());
        refusal_start_tokens.erase(std::unique(refusal_start_tokens.begin(), refusal_start_tokens.end()), refusal_start_tokens.end());

        // Track whether we are inside a <think> block
        // DeepSeek V3/V4 and Nemotron thinking mode: the chat template appends <think> to the prompt,
        // so the model starts generation INSIDE the think block.
        bool in_think_block = enable_thinking && !had_multimodal;
        if (had_multimodal) {
            enable_thinking = false;
            max_thinking_tokens = 0;
            in_think_block = false;
        }
        bool think_block_ended = !in_think_block;
        int content_start_history_idx = in_think_block ? -1 : (int)history.size();
        std::string finish_reason = "length";

        // Set per-request repetition penalty
        current_rep_penalty_ = repetition_penalty;

        int content_tokens_generated = 0;
        int thinking_tokens_generated = 0;

        decode_step_idx_ = 0;
        int n_layers = cfg_.num_hidden_layers;
        int moe_top_k = cfg_.num_experts_per_tok;
        int n_experts = cfg_.n_routed_experts;
        std::vector<std::pair<int, std::vector<int32_t>>> per_token_topk;
        int prev_tracked_token = -1;
        int prev_slot = -1;
        int draft_streak = 0;

        std::string last_think_token_str;

        auto gen_start_time = std::chrono::steady_clock::now();
        int spec_cycles = 0;
        int spec_drafted = 0;
        int spec_accepted = 0;
        double total_draft_ms = 0.0;
        double total_verify_ms = 0.0;

        auto emit_token_cpu = [&](int next_token, int step_t) -> bool {
            // Handle think block filtering
            if (think_start_id >= 0 && next_token == think_start_id) {
                if (!think_block_ended) {
                    in_think_block = true;
                    think_block_ended = false;
                } else {
                    return true; // Already past think block, do not push or emit think_start_id
                }
            }
            bool just_ended_think = false;
            if (think_end_id >= 0 && next_token == think_end_id) {
                if (in_think_block) {
                    in_think_block = false;
                    think_block_ended = true;
                    just_ended_think = true;
                    if (content_start_history_idx < 0) {
                        content_start_history_idx = (int)history.size();
                    }
                } else {
                    return true; // Already past think block, do not push duplicate think_end_id
                }
            }

            if (think_block_ended && !just_ended_think && (next_token == think_start_id || next_token == think_end_id)) {
                return true; // Never emit or track think tokens once think block has ended
            }

            output_ids.push_back(next_token);
            history.push_back(next_token);

            // Enable tracking for all generated tokens produced by the model (excluding control & special tokens)
            track_current_token_ = !is_control_token(next_token);
            if (track_expert_freq_ && track_current_token_) {
                expert_freq_total_tokens_++;
            }

            if (prev_slot >= 0 && prev_tracked_token >= 0 && track_expert_freq_ && step_topk_host_) {
                int32_t* src_ptr = step_topk_host_ + prev_slot * (n_layers * moe_top_k);
                std::vector<int32_t> topk_copy(src_ptr, src_ptr + (n_layers * moe_top_k));
                per_token_topk.push_back({prev_tracked_token, std::move(topk_copy)});
            }
            if (track_current_token_) {
                prev_tracked_token = next_token;
                prev_slot = decode_step_idx_ % 2;
            } else {
                prev_tracked_token = -1;
                prev_slot = -1;
            }

            if (next_token == think_start_id || next_token == think_end_id) {
                if (next_token == think_end_id && on_token) {
                    on_token("", false); // Signal end of reasoning to stream consumer
                }
                return true;
            }

            if (in_think_block) {
                thinking_tokens_generated++;
            } else {
                content_tokens_generated++;
            }

            std::string token_text = tokenizer_.decode({next_token});
            int min_thinking_tokens = std::min(48, max_thinking_tokens > 0 ? max_thinking_tokens : 48);
            if (in_think_block && thinking_tokens_generated >= min_thinking_tokens && (token_text.find("</think>") != std::string::npos || (token_buffer + token_text).find("</think>") != std::string::npos)) {
                in_think_block = false;
                think_block_ended = true;
                if (content_start_history_idx < 0) {
                    content_start_history_idx = (int)history.size();
                }
            }
            if (in_think_block) {
                last_think_token_str = token_text;
            }

            token_buffer += token_text;
            generated_text += token_text;

            // Check if token_buffer ends with a complete UTF-8 character.
            bool is_complete = true;
            if (!token_buffer.empty()) {
                int m = std::min((int)token_buffer.size(), 4);
                for (int j = 1; j <= m; j++) {
                    unsigned char b = token_buffer[token_buffer.size() - j];
                    if ((b & 0xC0) != 0x80) { // Found leading byte or ASCII
                        if ((b & 0x80) == 0) { // ASCII
                            is_complete = true;
                        } else if ((b & 0xE0) == 0xC0) {
                            is_complete = (j >= 2);
                        } else if ((b & 0xF0) == 0xE0) {
                            is_complete = (j >= 3);
                        } else if ((b & 0xF8) == 0xF0) {
                            is_complete = (j >= 4);
                        } else {
                            is_complete = true; // invalid utf8 leading byte, just flush
                        }
                        break;
                    }
                }
            }

            if (is_complete) {
                if (next_token != cfg_.bos_token_id) {
                    if (!g_quiet) {
                        printf("%s", token_buffer.c_str());
                        fflush(stdout);
                    }
                }
                if (on_token) {
                    if (!on_token(token_buffer, in_think_block)) {
                        LOG_WARN("Generation aborted by token callback at step %d", step_t);
                        token_buffer.clear();
                        return false;
                    }
                }
                token_buffer.clear();
            }
            return true;
        };

        auto emit_token = [&](int next_token, int step_t) -> bool {
            if (!emit_token_cpu(next_token, step_t)) return false;
            forward_token(next_token, position);
            if (qwen_draft_.loaded_) {
                qwen_draft_.forward_token_async(next_token, position, main_stream_);
            }
            position++;
            decode_step_idx_++;
            return true;
        };

        int next_token = -1;
        for (int t = 0; content_tokens_generated < max_tokens; t++) {
            if (g_stop_requested.load()) {
                LOG_WARN("Generation stopped by client stop request at step %d", t);
                finish_reason = "stop";
                break;
            }

            // N-Gram Repetition Loop Detector & Breaker (breaks attractor loops in reasoning & content)
            int loop_len = 0;
            int n_out = (int)output_ids.size();
            if (in_think_block) {
                // Reasoning phase: detect exact token repetition loops.
                // To avoid false positives on legitimate formatting, markdown lists, bullet points,
                // table dividers, equations, or code planning, require a robust minimum number of cycles
                // and a substantial total repeated token span (at least 40+ tokens).
                for (int L = 1; L <= 512; L++) {
                    int min_cycles = (L == 1) ? 40 :
                                     ((L == 2) ? 24 :
                                     ((L <= 4) ? 16 :
                                     ((L <= 8) ? 10 :
                                     ((L <= 16) ? 6 :
                                     ((L <= 32) ? 4 :
                                     ((L <= 64) ? 4 : 3))))));
                    if (min_cycles * L > n_out) continue;
                    if (min_cycles * L < 40) continue; // Safety floor: never trip on less than 40 repeated tokens
                    bool match = true;
                    const int* p_last = output_ids.data() + (n_out - L);
                    for (int c = 1; c < min_cycles; c++) {
                        const int* p_prev = output_ids.data() + (n_out - (c + 1) * L);
                        for (int i = 0; i < L; i++) {
                            if (p_last[i] != p_prev[i]) {
                                match = false;
                                break;
                            }
                        }
                        if (!match) break;
                    }
                    if (match) {
                        loop_len = L;
                        break;
                    }
                }
                // Structural repetition matching is disabled: normal reasoning routinely lists
                // parallel items (buttons, steps, features, cases) which falsely trigger fuzzy pattern matchers.
                // Exact repetition detection robustly catches true attractor loops.
            } else {
                // Content generation phase:
                // Only inspect content tokens (NEVER cross back into reasoning scratchpad!).
                // In code/HTML/CSS generation, structural repetitions (buttons, rows, list items) are NORMAL.
                // Exact token loops require substantial cycle count and at least 48 tokens.
                int n_content = content_tokens_generated;
                for (int L = 1; L <= 512; L++) {
                    int min_cycles = (L == 1) ? 40 :
                                     ((L == 2) ? 24 :
                                     ((L <= 4) ? 16 :
                                     ((L <= 8) ? 10 :
                                     ((L <= 16) ? 6 :
                                     ((L <= 32) ? 4 :
                                     ((L <= 64) ? 4 : 3))))));
                    if (min_cycles * L > n_content) continue;
                    if (min_cycles * L < 48) continue; // Content safety floor: at least 48 identical repeated tokens
                    bool match = true;
                    const int* p_last = output_ids.data() + (n_out - L);
                    for (int c = 1; c < min_cycles; c++) {
                        const int* p_prev = output_ids.data() + (n_out - (c + 1) * L);
                        for (int i = 0; i < L; i++) {
                            if (p_last[i] != p_prev[i]) {
                                match = false;
                                break;
                            }
                        }
                        if (!match) break;
                    }
                    if (match) {
                        loop_len = L;
                        break;
                    }
                }
            }

            if (loop_len > 0) {
                if (in_think_block) {
                    LOG_WARN("[Anti-Loop] Repetition loop detected during reasoning (cycle length %d). Forcing </think> and starting content.", loop_len);
                    in_think_block = false;
                    think_block_ended = true;
                    if (content_start_history_idx < 0) {
                        content_start_history_idx = (int)history.size();
                    }
                    track_current_token_ = false;
                    if (think_end_id >= 0) {
                        forward_token(think_end_id, position);
                        if (qwen_draft_.loaded_) {
                            qwen_draft_.forward_token_async(think_end_id, position, main_stream_);
                        }
                        position++;
                        output_ids.push_back(think_end_id);
                        history.push_back(think_end_id);
                    }
                    if (on_token) {
                        on_token("", false); // Signal end of reasoning to stream consumer
                    }
                    std::vector<int> transition_tokens = tokenizer_.encode("\n\n");
                    for (int tok_id : transition_tokens) {
                        track_current_token_ = false;
                        forward_token(tok_id, position);
                        if (qwen_draft_.loaded_) {
                            qwen_draft_.forward_token_async(tok_id, position, main_stream_);
                        }
                        position++;
                        output_ids.push_back(tok_id);
                        history.push_back(tok_id);
                        content_tokens_generated++;
                        std::string tok_str = tokenizer_.decode({tok_id});
                        generated_text += tok_str;
                        if (on_token) {
                            on_token(tok_str, false);
                        }
                    }
                    // Mask think tokens in logits so content generation cannot immediately re-open a think block
                    float neg_inf = -1e9f;
                    if (think_start_id >= 0) {
                        CUDA_CHECK(cudaMemcpyAsync(buf_logits_.f32() + think_start_id, &neg_inf, sizeof(float), cudaMemcpyHostToDevice, main_stream_));
                    }
                    if (think_end_id >= 0) {
                        CUDA_CHECK(cudaMemcpyAsync(buf_logits_.f32() + think_end_id, &neg_inf, sizeof(float), cudaMemcpyHostToDevice, main_stream_));
                    }
                    argmax_cache_valid_ = false;
                    next_token = -1;
                    continue;
                } else {
                    LOG_WARN("[Anti-Loop] Repetition loop detected during content generation (cycle length %d). Halting generation cleanly.", loop_len);
                    // If inside an unclosed code block, cleanly close it before halting
                    size_t code_blocks = 0;
                    size_t pos = 0;
                    while ((pos = generated_text.find("```", pos)) != std::string::npos) {
                        code_blocks++;
                        pos += 3;
                    }
                    if (code_blocks % 2 != 0) {
                        if (on_token) on_token("\n```\n", false);
                        generated_text += "\n```\n";
                    }
                    finish_reason = "stop";
                    break;
                }
            }

            // Graceful sentence boundary transition when thinking budget is reached
            if (in_think_block && max_thinking_tokens > 0 && thinking_tokens_generated >= max_thinking_tokens) {
                int grace_limit = max_thinking_tokens + (std::min)(24, (std::max)(4, max_thinking_tokens / 8));
                bool is_clean_boundary = false;
                if (!last_think_token_str.empty()) {
                    for (char c : last_think_token_str) {
                        if (c == '\n' || c == '.' || c == ':' || c == '?' || c == '!') {
                            is_clean_boundary = true;
                            break;
                        }
                    }
                }
                if (is_clean_boundary || thinking_tokens_generated >= grace_limit) {
                    LOG_WARN("Thinking budget reached (%d/%d tokens, clean_boundary=%d). Closing </think> and starting content.",
                             thinking_tokens_generated, max_thinking_tokens, is_clean_boundary ? 1 : 0);

                    // 1. Inject </think> to close thinking block
                    in_think_block = false;
                    think_block_ended = true;
                    if (content_start_history_idx < 0) {
                        content_start_history_idx = (int)history.size();
                    }
                    track_current_token_ = false;
                    if (think_end_id >= 0) {
                        forward_token(think_end_id, position);
                        if (qwen_draft_.loaded_) {
                            qwen_draft_.forward_token_async(think_end_id, position, main_stream_);
                        }
                        position++;
                        output_ids.push_back(think_end_id);
                        history.push_back(think_end_id);
                    }

                    // 2. Inject and stream "\n\n" as the beginning of CONTENT
                    std::vector<int> transition_tokens = tokenizer_.encode("\n\n");
                    for (int tok_id : transition_tokens) {
                        track_current_token_ = false; // Synthetic transition tokens not tracked
                        forward_token(tok_id, position);
                        if (qwen_draft_.loaded_) {
                            qwen_draft_.forward_token_async(tok_id, position, main_stream_);
                        }
                        position++;
                        output_ids.push_back(tok_id);
                        history.push_back(tok_id);
                        content_tokens_generated++;
                        std::string tok_str = tokenizer_.decode({tok_id});
                        generated_text += tok_str;
                        if (on_token) {
                            on_token(tok_str, false); // Streamed directly as content
                        }
                    }
                    next_token = -1;
                    continue;
                }
            }

            // Sample from logits if next_token is not pre-set
            if (next_token < 0) {
                if (content_tokens_generated == 0 && had_multimodal) {
                    float neg_inf = -1e9f;
                    if (cfg_.eos_token_id >= 0) {
                        CUDA_CHECK(cudaMemcpyAsync(buf_logits_.f32() + cfg_.eos_token_id, &neg_inf, sizeof(float), cudaMemcpyHostToDevice, main_stream_));
                    }
                    if (eos2_id >= 0) {
                        CUDA_CHECK(cudaMemcpyAsync(buf_logits_.f32() + eos2_id, &neg_inf, sizeof(float), cudaMemcpyHostToDevice, main_stream_));
                    }
                    argmax_cache_valid_ = false;
                }
                if (!in_think_block || think_block_ended) {
                    float neg_inf = -1e9f;
                    if (think_start_id >= 0) {
                        CUDA_CHECK(cudaMemcpyAsync(buf_logits_.f32() + think_start_id, &neg_inf, sizeof(float), cudaMemcpyHostToDevice, main_stream_));
                    }
                    if (think_end_id >= 0) {
                        CUDA_CHECK(cudaMemcpyAsync(buf_logits_.f32() + think_end_id, &neg_inf, sizeof(float), cudaMemcpyHostToDevice, main_stream_));
                    }
                    argmax_cache_valid_ = false;
                }
                int min_thinking_tokens = std::min(48, max_thinking_tokens > 0 ? max_thinking_tokens : 48);
                if (in_think_block && !think_block_ended && thinking_tokens_generated < min_thinking_tokens) {
                    float neg_inf = -1e9f;
                    if (think_end_id >= 0) {
                        CUDA_CHECK(cudaMemcpyAsync(buf_logits_.f32() + think_end_id, &neg_inf, sizeof(float), cudaMemcpyHostToDevice, main_stream_));
                    }
                    if (cfg_.eos_token_id >= 0) {
                        CUDA_CHECK(cudaMemcpyAsync(buf_logits_.f32() + cfg_.eos_token_id, &neg_inf, sizeof(float), cudaMemcpyHostToDevice, main_stream_));
                    }
                    if (eos2_id >= 0) {
                        CUDA_CHECK(cudaMemcpyAsync(buf_logits_.f32() + eos2_id, &neg_inf, sizeof(float), cudaMemcpyHostToDevice, main_stream_));
                    }
                    if (im_end_id >= 0) {
                        CUDA_CHECK(cudaMemcpyAsync(buf_logits_.f32() + im_end_id, &neg_inf, sizeof(float), cudaMemcpyHostToDevice, main_stream_));
                    }
                    argmax_cache_valid_ = false;
                }
                // Suppress refusal opening tokens at the start of content generation
                if (content_tokens_generated == 0 && think_block_ended) {
                    float neg_inf = -1e9f;
                    for (int r_tok : refusal_start_tokens) {
                        if (r_tok >= 0) {
                            CUDA_CHECK(cudaMemcpyAsync(buf_logits_.f32() + r_tok, &neg_inf, sizeof(float), cudaMemcpyHostToDevice, main_stream_));
                        }
                    }
                    argmax_cache_valid_ = false;
                }
                next_token = sample_token(temperature, history, content_tokens_generated, in_think_block,
                                          top_k, top_p, min_p, content_start_history_idx);
            }
            
            // If EOS is sampled while inside think block, transition to </think> and continue
            if (in_think_block && (next_token == cfg_.eos_token_id || (eos2_id >= 0 && next_token == eos2_id))) {
                LOG_WARN("EOS sampled during thinking at step %d (%d thinking tokens). Transitioning to </think> and starting content.",
                         t, thinking_tokens_generated);
                in_think_block = false;
                think_block_ended = true;
                if (content_start_history_idx < 0) {
                    content_start_history_idx = (int)history.size();
                }
                track_current_token_ = false;
                if (think_end_id >= 0) {
                    forward_token(think_end_id, position);
                    if (qwen_draft_.loaded_) {
                        qwen_draft_.forward_token_async(think_end_id, position, main_stream_);
                    }
                    position++;
                    output_ids.push_back(think_end_id);
                    history.push_back(think_end_id);
                }
                if (on_token) {
                    on_token("", false); // Signal end of reasoning to stream consumer
                }
                std::vector<int> transition_tokens = tokenizer_.encode("\n\n");
                for (int tok_id : transition_tokens) {
                    track_current_token_ = false;
                    forward_token(tok_id, position);
                    if (qwen_draft_.loaded_) {
                        qwen_draft_.forward_token_async(tok_id, position, main_stream_);
                    }
                    position++;
                    output_ids.push_back(tok_id);
                    history.push_back(tok_id);
                    content_tokens_generated++;
                    std::string tok_str = tokenizer_.decode({tok_id});
                    generated_text += tok_str;
                    if (on_token) {
                        on_token(tok_str, false);
                    }
                }
                float neg_inf = -1e9f;
                if (think_start_id >= 0) {
                    CUDA_CHECK(cudaMemcpyAsync(buf_logits_.f32() + think_start_id, &neg_inf, sizeof(float), cudaMemcpyHostToDevice, main_stream_));
                }
                if (think_end_id >= 0) {
                    CUDA_CHECK(cudaMemcpyAsync(buf_logits_.f32() + think_end_id, &neg_inf, sizeof(float), cudaMemcpyHostToDevice, main_stream_));
                }
                argmax_cache_valid_ = false;
                next_token = -1;
                continue;
            }

            // Check all EOS and stop conditions in content mode
            if (next_token == cfg_.eos_token_id || (eos2_id >= 0 && next_token == eos2_id) || (im_end_id >= 0 && next_token == im_end_id)) {
                LOG_WARN("Stop token hit: token=%d (cfg_eos=%d, eos2=%d, im_end=%d) at step %d (content: %d/%d, think: %d/%d)",
                         next_token, cfg_.eos_token_id, eos2_id, im_end_id, t,
                         content_tokens_generated, max_tokens,
                         thinking_tokens_generated, max_thinking_tokens);
                finish_reason = "stop";
                break;
            }

            if ((!in_think_block || think_block_ended || !enable_thinking) && (next_token == think_start_id || next_token == think_end_id)) {
                if (!enable_thinking && content_tokens_generated > 0) {
                    LOG_WARN("Stop on think token in non-thinking mode: token=%d at step %d (content: %d)",
                             next_token, t, content_tokens_generated);
                    finish_reason = "stop";
                    break;
                }
                float neg_inf = -1e9f;
                CUDA_CHECK(cudaMemcpyAsync(buf_logits_.f32() + next_token, &neg_inf, sizeof(float), cudaMemcpyHostToDevice, main_stream_));
                argmax_cache_valid_ = false;
                next_token = -1;
                continue;
            }

            // 2. Speculative Decoding Verification Loop (MTP Self-Drafter + PLD + 2B Neural Drafter fallback)
            bool allow_draft = (cfg_.architecture == ModelArch::QWEN || cfg_.is_nemotron()) && enable_spec_ && (draft_streak >= 0) && !in_think_block &&
                               (content_tokens_generated + 4 < max_tokens) && !g_stop_requested.load();

            if (allow_draft) {
                if (!emit_token_cpu(next_token, t)) {
                    finish_reason = "stop";
                    break;
                }

                std::vector<int> cand_tokens;
                cand_tokens.reserve(8);
                cand_tokens.push_back(next_token);

                bool is_pld = false;
                bool is_mtp = false;
                bool is_tree = false;
                TreeTopology tree_top;

                // Priority 1: MTP Tree Attention Self-Drafter (candidate tree: tree_nodes_ branches)
                if (cfg_.architecture == ModelArch::QWEN && enable_mtp_ && enable_tree_mtp_ && mtp_drafter_.loaded_) {
                    auto t0 = std::chrono::steady_clock::now();
                    is_tree = mtp_drafter_.draft_tree_tokens(buf_hidden2_.bf16(), next_token, position, tree_top, main_stream_, tree_nodes_);
                    auto t1 = std::chrono::steady_clock::now();
                    total_draft_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
                }

                // If Tree drafting was successful, verify candidate tree on Target model
                int tree_M = tree_top.num_nodes;
                if (is_tree && tree_M >= 2) {
                    auto tv0 = std::chrono::steady_clock::now();
                    forward_token_tree_qwen(tree_top, position);

                    // Compute argmax across all tree nodes
                    argmax_f32_batch_cuda(buf_argmax_out_batch_.i32(), buf_logits_batch_.f32(), cfg_.vocab_size, tree_M, main_stream_);

                    static int32_t* host_tree_preds = nullptr;
                    if (!host_tree_preds) {
                        CUDA_CHECK(cudaMallocHost(&host_tree_preds, 16 * sizeof(int32_t)));
                    }
                    CUDA_CHECK(cudaMemcpyAsync(host_tree_preds, buf_argmax_out_batch_.i32(), tree_M * sizeof(int32_t), cudaMemcpyDeviceToHost, main_stream_));
                    CUDA_CHECK(cudaStreamSynchronize(main_stream_));
                    auto tv1 = std::chrono::steady_clock::now();
                    double ver_ms = std::chrono::duration<double, std::milli>(tv1 - tv0).count();
                    total_verify_ms += ver_ms;
                    if (spec_cycles < 5) {
                        LOG_INFO("TREE VERIFY TIMING: cycle=%d nodes=%d took %.2fms (graph=%d)",
                                 spec_cycles, tree_M, ver_ms, (int)tree_graph_captured_);
                    }

                    // Resolve longest verified path
                    TreeVerificationResult vres = resolve_tree_longest_path(
                        tree_top, host_tree_preds,
                        cfg_.eos_token_id, eos2_id,
                        think_start_id, think_end_id, enable_thinking);

                    if (spec_cycles < 5) {
                        LOG_INFO("TREE DEBUG: cycle=%d accepted=%d last_node=%d bonus=%d winning_slots=%zu",
                                 spec_cycles, vres.accepted_count, vres.last_accepted_node,
                                 vres.bonus_token, vres.winning_slots.size());
                    }

                    // Emit accepted draft tokens to user output
                    for (int k = 0; k < vres.accepted_count; k++) {
                        if (!emit_token_cpu(vres.accepted_tokens[k], t)) {
                            finish_reason = "stop";
                            break;
                        }
                    }

                    if (finish_reason == "stop") break;

                    spec_cycles++;
                    spec_drafted += (tree_M - 1);
                    spec_accepted += vres.accepted_count;

                    // 1. KV cache compaction for GQA layers (moves winning path KV entries to contiguous slots)
                    if (!vres.winning_slots.empty()) {
                        compact_tree_kv_all_gqa_layers(vres.winning_slots.data(), position, (int)vres.winning_slots.size());
                    }

                    // 2. Commit DeltaNet SSM state of winning path
                    commit_target_state_slot(vres.last_accepted_node);

                    // 3. Save post-norm hidden state from last accepted node for next MTP draft cycle
                    CUDA_CHECK(cudaMemcpyAsync(buf_hidden2_.bf16(),
                                               buf_hidden_batch_.bf16() + (size_t)vres.last_accepted_node * cfg_.hidden_size,
                                               cfg_.hidden_size * sizeof(__nv_bfloat16),
                                               cudaMemcpyDeviceToDevice, main_stream_));

                    // 4. Bonus token becomes next_token for next cycle (or sample if temperature > 0 and at root)
                    if (vres.bonus_token >= 0 && (temperature <= 0.0f || vres.accepted_count > 0)) {
                        next_token = vres.bonus_token;
                    } else {
                        CUDA_CHECK(cudaMemcpyAsync(buf_logits_.f32(),
                                                   buf_logits_batch_.f32() + (size_t)vres.last_accepted_node * cfg_.vocab_size,
                                                   cfg_.vocab_size * sizeof(float),
                                                   cudaMemcpyDeviceToDevice, main_stream_));
                        if (!in_think_block || think_block_ended) {
                            float neg_inf = -1e9f;
                            if (think_start_id >= 0) {
                                CUDA_CHECK(cudaMemcpyAsync(buf_logits_.f32() + think_start_id, &neg_inf, sizeof(float), cudaMemcpyHostToDevice, main_stream_));
                            }
                            if (think_end_id >= 0) {
                                CUDA_CHECK(cudaMemcpyAsync(buf_logits_.f32() + think_end_id, &neg_inf, sizeof(float), cudaMemcpyHostToDevice, main_stream_));
                            }
                        }
                        CUDA_CHECK(cudaStreamSynchronize(main_stream_));
                        argmax_cache_valid_ = false;
                        next_token = sample_token(temperature, history, content_tokens_generated, in_think_block,
                                                  top_k, top_p, min_p, content_start_history_idx);
                    }

                    int advanced = 1 + vres.accepted_count;
                    position += advanced;
                    decode_step_idx_ += advanced;

                    if (vres.accepted_count == 0) {
                        draft_streak = 0;
                    } else {
                        draft_streak = std::min(draft_streak + vres.accepted_count, 6);
                    }
                    continue;
                }

                // Priority 1b: Linear MTP Self-Drafter (fallback if tree drafting disabled/unloaded)
                if (cfg_.architecture == ModelArch::QWEN && enable_mtp_ && mtp_drafter_.loaded_) {
                    bool in_tool_call = false;
                    size_t last_tc_open = generated_text.rfind("<tool_call>");
                    if (last_tc_open != std::string::npos) {
                        size_t last_tc_close = generated_text.rfind("</tool_call>");
                        if (last_tc_close == std::string::npos || last_tc_close < last_tc_open) {
                            in_tool_call = true;
                        }
                    }
                    if (in_tool_call || draft_streak >= 0) {
                        int max_k = (mtp_k_ > 0) ? mtp_k_ : (gpu_caps_.is_vram_constrained ? 1 : 4);
                        int K = (in_tool_call || draft_streak >= 3) ? 4 : ((draft_streak >= 1) ? 2 : 1);
                        if (K > max_k) K = max_k;
                        auto t0 = std::chrono::steady_clock::now();
                        mtp_drafter_.draft_k_tokens(buf_hidden2_.bf16(), next_token, position, K, cand_tokens, main_stream_);
                        auto t1 = std::chrono::steady_clock::now();
                        total_draft_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
                        is_mtp = (cand_tokens.size() > 1);
                    }
                }

                // Priority 2: Prompt-Lookup Drafting (only as fallback when MTP is not loaded / disabled)
                if ((cfg_.architecture == ModelArch::QWEN || cfg_.is_nemotron()) && !is_mtp && (!enable_mtp_ || !mtp_drafter_.loaded_) && enable_pld_ && !history.empty()) {
                    int max_pld = cfg_.is_nemotron() ? 3 : std::min(pld_draft_tokens_, 3);
                    int pld_min_ngram = cfg_.is_nemotron() ? 2 : 1;
                    std::vector<int> pld_cands = PromptLookupDrafter::draft(history, max_pld, 4, pld_min_ngram);
                    if (!pld_cands.empty()) {
                        for (int tok : pld_cands) {
                            if ((!in_think_block || think_block_ended) && (tok == think_start_id || tok == think_end_id)) {
                                break;
                            }
                            cand_tokens.push_back(tok);
                        }
                        if (cand_tokens.size() > 1) {
                            is_pld = true;
                        }
                    }
                }

                // Priority 3: Legacy 2B Neural Drafter (fallback on confirmed streak)
                if (cfg_.architecture == ModelArch::QWEN && !is_mtp && !is_pld && qwen_draft_.loaded_ && draft_streak >= 2) {
                    int K = (draft_streak >= 3) ? 4 : 2;
                    qwen_draft_.draft_k_tokens(next_token, position, K, cand_tokens, main_stream_);
                }

                int M = (int)cand_tokens.size();
                if ((cfg_.architecture == ModelArch::QWEN || cfg_.is_nemotron()) && M >= 2) {
                    auto tv0 = std::chrono::steady_clock::now();
                    if (cfg_.is_nemotron()) {
                        forward_token_batch_nemotron(cand_tokens.data(), position, M);
                    } else {
                        forward_token_batch_qwen(cand_tokens.data(), position, M);
                    }

                    int num_verify = M - 1;
                    apply_repetition_penalty_to_batch(buf_logits_batch_.f32(), M, history, in_think_block, content_start_history_idx);
                    // Verify candidate positions simultaneously in ONE GPU pass (computing all M predictions)
                    argmax_f32_batch_cuda(buf_argmax_out_batch_.i32(), buf_logits_batch_.f32(), cfg_.vocab_size, M, main_stream_);

                    static int32_t* host_batch_preds = nullptr;
                    if (!host_batch_preds) {
                        CUDA_CHECK(cudaMallocHost(&host_batch_preds, 16 * sizeof(int32_t)));
                    }
                    CUDA_CHECK(cudaMemcpyAsync(host_batch_preds, buf_argmax_out_batch_.i32(), M * sizeof(int32_t), cudaMemcpyDeviceToHost, main_stream_));
                    CUDA_CHECK(cudaStreamSynchronize(main_stream_));
                    auto tv1 = std::chrono::steady_clock::now();
                    double ver_ms = std::chrono::duration<double, std::milli>(tv1 - tv0).count();
                    total_verify_ms += ver_ms;
                    if (spec_cycles < 5) {
                        LOG_INFO("VERIFY TIMING: cycle=%d M=%d took %.2fms (graph=%d)", spec_cycles, M, ver_ms, (int)batch_graph_captured_[M]);
                    }

                    int accepted = 0;
                    int bonus_token = -1;

                    for (int k = 0; k < num_verify; k++) {
                        int pred_k = host_batch_preds[k];
                        int draft_cand = cand_tokens[k + 1];

                        // Debug: log first 5 MTP spec cycles
                        if (is_mtp && spec_cycles < 5) {
                            LOG_INFO("MTP DEBUG: cycle=%d k=%d pred=%d draft=%d %s",
                                     spec_cycles, k, pred_k, draft_cand,
                                     pred_k == draft_cand ? "MATCH" : "MISS");
                        }

                        bool allow_think_tok = in_think_block && !think_block_ended;
                        if (pred_k == draft_cand &&
                            pred_k != cfg_.eos_token_id && (eos2_id < 0 || pred_k != eos2_id) &&
                            (allow_think_tok || (pred_k != think_start_id && pred_k != think_end_id))) {
                            accepted++;
                            if (!emit_token_cpu(draft_cand, t)) {
                                finish_reason = "stop";
                                break;
                            }
                        } else {
                            bonus_token = pred_k;
                            break;
                        }
                    }

                    if (finish_reason == "stop") break;

                    spec_cycles++;
                    spec_drafted += num_verify;
                    spec_accepted += accepted;

                    if (accepted == num_verify) {
                        // All draft tokens accepted!
                        // For MTP: no drafter state management needed (it reads hidden state each time)
                        // For 2B drafter: need to keep it in sync
                        if (!is_mtp && qwen_draft_.loaded_) {
                            if (is_pld) {
                                qwen_draft_.replay_tokens(cand_tokens.data() + 1, position + 1, num_verify, main_stream_);
                            } else {
                                qwen_draft_.forward_token_async(cand_tokens[M - 1], position + (M - 1), main_stream_);
                            }
                        }

                        // Save post-norm hidden state from last batch slot for MTP drafter
                        if (mtp_drafter_.loaded_) {
                            CUDA_CHECK(cudaMemcpyAsync(buf_hidden2_.bf16(),
                                                       buf_hidden_batch_.bf16() + (size_t)(M - 1) * cfg_.hidden_size,
                                                       cfg_.hidden_size * sizeof(__nv_bfloat16),
                                                       cudaMemcpyDeviceToDevice, main_stream_));
                        }

                        if (temperature <= 0.0f) {
                            // Fast path: host_batch_preds[M - 1] already holds the exact greedy argmax of position M - 1!
                            next_token = host_batch_preds[M - 1];
                        } else {
                            // Copy latest logits from position M - 1 for multinomial sampling
                            CUDA_CHECK(cudaMemcpyAsync(buf_logits_.f32(),
                                                       buf_logits_batch_.f32() + (size_t)(M - 1) * cfg_.vocab_size,
                                                       cfg_.vocab_size * sizeof(float),
                                                       cudaMemcpyDeviceToDevice, main_stream_));
                            if (!in_think_block || think_block_ended) {
                                float neg_inf = -1e9f;
                                if (think_start_id >= 0) {
                                    CUDA_CHECK(cudaMemcpyAsync(buf_logits_.f32() + think_start_id, &neg_inf, sizeof(float), cudaMemcpyHostToDevice, main_stream_));
                                }
                                if (think_end_id >= 0) {
                                    CUDA_CHECK(cudaMemcpyAsync(buf_logits_.f32() + think_end_id, &neg_inf, sizeof(float), cudaMemcpyHostToDevice, main_stream_));
                                }
                            }
                            CUDA_CHECK(cudaStreamSynchronize(main_stream_));
                            argmax_cache_valid_ = false;
                            next_token = sample_token(temperature, history, content_tokens_generated, in_think_block,
                                                      top_k, top_p, min_p, content_start_history_idx);
                        }

                        position += M;
                        decode_step_idx_ += M;
                        draft_streak = std::min(draft_streak + 2, 6);
                    } else {
                        // Mismatch at index 'accepted' (0 <= accepted < num_verify)
                        // Valid candidate sequence: cand_tokens[0 .. accepted]
                        // 1. Commit Target model state to slot 'accepted' (0-overhead D2D copy)
                        commit_target_state_slot(accepted);
                        // Save post-norm hidden state from accepted batch slot for MTP drafter
                        if (mtp_drafter_.loaded_) {
                            CUDA_CHECK(cudaMemcpyAsync(buf_hidden2_.bf16(),
                                                       buf_hidden_batch_.bf16() + (size_t)accepted * cfg_.hidden_size,
                                                       cfg_.hidden_size * sizeof(__nv_bfloat16),
                                                       cudaMemcpyDeviceToDevice, main_stream_));
                        }

                        // 2. Commit Draft model state to slot 'accepted' (only for 2B drafter)
                        if (!is_mtp && qwen_draft_.loaded_) {
                            if (is_pld) {
                                if (accepted > 0) {
                                    qwen_draft_.replay_tokens(cand_tokens.data() + 1, position + 1, accepted, main_stream_);
                                }
                            } else {
                                qwen_draft_.commit_state_slot(accepted, main_stream_);
                            }
                        }

                        // 3. bonus_token becomes next_token for next cycle
                        next_token = bonus_token;

                        position += (accepted + 1);
                        decode_step_idx_ += (accepted + 1);

                        if (accepted == 0) {
                            draft_streak = (is_mtp || cfg_.is_nemotron() || is_pld) ? 0 : -8;
                        } else {
                            draft_streak = std::min(draft_streak + accepted, 6);
                        }
                    }
                    continue;
                } else {
                    // Fallback to single token forward if drafting failed
                    forward_token(next_token, position);
                    if (qwen_draft_.loaded_) {
                        qwen_draft_.forward_token_async(next_token, position, main_stream_);
                    }
                    position++;
                    decode_step_idx_++;
                    next_token = -1;
                    draft_streak = (cfg_.is_nemotron() || enable_pld_) ? 0 : -2;
                    continue;
                }
            }

            if (draft_streak < 0) draft_streak++;

            if (!emit_token(next_token, t)) {
                finish_reason = "stop";
                break;
            }
            next_token = -1;
        }

        // Check if we hit max_tokens
        if ((int)output_ids.size() >= max_tokens) {
            finish_reason = "length";
        }

        // Flush anything remaining in the buffer
        if (!token_buffer.empty()) {
            if (!g_quiet) {
                printf("%s", token_buffer.c_str());
                fflush(stdout);
            }
            if (on_token) {
                on_token(token_buffer, in_think_block);
            }
        }

        if (g_log_tokens) {
            printf("\n");
        }

        // Separate reasoning tokens and content tokens
        std::vector<int> reasoning_ids;
        std::vector<int> content_ids;
        bool is_in_think = (enable_thinking && !had_multimodal);
        bool think_already_ended = false;
        for (int id : output_ids) {
            if (think_start_id >= 0 && id == think_start_id) {
                if (!think_already_ended) {
                    is_in_think = true;
                }
                continue;
            }
            if (think_end_id >= 0 && id == think_end_id) {
                is_in_think = false;
                think_already_ended = true;
                continue;
            }
            if (is_in_think && !think_already_ended) {
                reasoning_ids.push_back(id);
            } else {
                content_ids.push_back(id);
            }
        }
        last_reasoning_text_ = tokenizer_.decode(reasoning_ids);
        last_content_text_ = tokenizer_.decode(content_ids);
        std::string result = last_content_text_;

        prompt_token_count_ = (int)prompt.size();
        completion_token_count_ = (int)output_ids.size();
        last_finish_reason_ = finish_reason;

        auto gen_end_time = std::chrono::steady_clock::now();
        double total_sec = std::chrono::duration<double>(gen_end_time - gen_start_time).count();
        double tok_per_sec = (total_sec > 0.0) ? (double)output_ids.size() / total_sec : 0.0;
        last_decode_sec_ = total_sec;
        last_decode_tps_ = tok_per_sec;
        LOG_INFO("[GENERATION STATS] %zu tokens in %.3fs -> %.2f tok/s (Content: %d, Thinking: %d)",
                 output_ids.size(), total_sec, tok_per_sec, content_tokens_generated, thinking_tokens_generated);
        if (spec_cycles > 0) {
            LOG_INFO("[SPECULATIVE STATS] %d cycles | %d drafted | %d accepted (%.1f%% acceptance rate) | draft: %.1fms total (%.2fms/c) | verify: %.1fms total (%.2fms/c)",
                     spec_cycles, spec_drafted, spec_accepted,
                     spec_drafted > 0 ? (100.0 * spec_accepted / spec_drafted) : 0.0,
                     total_draft_ms, total_draft_ms / spec_cycles,
                     total_verify_ms, total_verify_ms / spec_cycles);
        }

        CUDA_CHECK(cudaStreamSynchronize(main_stream_));
        if (prev_slot >= 0 && prev_tracked_token >= 0 && track_expert_freq_ && step_topk_host_) {
            int32_t* src_ptr = step_topk_host_ + prev_slot * (n_layers * moe_top_k);
            std::vector<int32_t> topk_copy(src_ptr, src_ptr + (n_layers * moe_top_k));
            per_token_topk.push_back({prev_tracked_token, std::move(topk_copy)});
        }

        sync_expert_freq_from_gpu();

        if (track_expert_freq_ && !per_token_topk.empty()) {
            std::lock_guard<std::mutex> lock(expert_profile_mutex_);
            if ((int)expert_token_counts_.size() != n_layers) {
                expert_token_counts_.resize(n_layers);
                for (int l = 0; l < n_layers; l++) expert_token_counts_[l].resize(n_experts);
            }
            for (auto& [tok, topk_vec] : per_token_topk) {
                if (!is_control_token(tok)) {
                    for (int l = 0; l < n_layers && l < (int)expert_token_counts_.size(); l++) {
                        for (int k = 0; k < moe_top_k; k++) {
                            int eid = topk_vec[l * moe_top_k + k];
                            if (eid >= 0 && eid < (int)expert_token_counts_[l].size()) {
                                expert_token_counts_[l][eid][tok]++;
                            }
                        }
                    }
                }
            }
        }

        cached_tokens_ = history;
        snapshot_kv(turn_kv_snapshot_, history);
        if (buf_mrope_pos_.data) {
            int h_mrope[2] = {-1, 0};
            CUDA_CHECK(cudaMemcpyAsync(buf_mrope_pos_.i32(), h_mrope, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
        }
        return result;
    }

    // Per-request stats (set by generate())
    int prompt_token_count_ = 0;
    int completion_token_count_ = 0;
    std::string last_finish_reason_ = "stop";
    size_t last_prefill_tokens_ = 0;
    size_t last_cached_tokens_ = 0;
    double last_prefill_sec_ = 0.0;
    double last_prefill_tps_ = 0.0;
    double last_decode_sec_ = 0.0;
    double last_decode_tps_ = 0.0;
    std::string last_reasoning_text_;
    std::string last_content_text_;

    // ── Vision Perception Delegate Execution ─────────────────────────────────
    std::string describe_image(const std::string& image_b64, const std::string& user_query = "") {
        // 1. If an in-process delegate engine is already resident (e.g. 96GB GPU or sufficient VRAM), run directly!
        if (vision_delegate_engine_) {
            return vision_delegate_engine_->describe_image(image_b64, user_query);
        }

        // 2. On-demand dynamic swap path (for VRAM-constrained systems or when delegate was not eagerly loaded)
        if (!delegate_manifest_path_.empty()) {
            size_t free_vram = 0, total_vram = 0;
            cudaMemGetInfo(&free_vram, &total_vram);
            size_t required_for_delegate = 4800ULL * 1024 * 1024; // ~4.8 GB

            bool needs_eviction = (free_vram < required_for_delegate);
            bool evicted = false;

            auto t_start = std::chrono::steady_clock::now();
            if (needs_eviction) {
                LOG_INFO("[Vision Delegate] Available VRAM (%.2f GB) is below requirement (%.2f GB). Evicting Nemotron L1 expert cache to host DRAM...",
                         (double)free_vram / (1024.0 * 1024.0 * 1024.0),
                         (double)required_for_delegate / (1024.0 * 1024.0 * 1024.0));
                evicted = expert_loader_.evict_l1_to_host();
            } else {
                LOG_INFO("[Vision Delegate] Sufficient VRAM available (%.2f GB free). Loading delegate directly without unloading L1 layers...",
                         (double)free_vram / (1024.0 * 1024.0 * 1024.0));
            }

            auto delegate_engine = std::make_unique<MoecherEngine>();
            bool loaded = delegate_engine->load(delegate_manifest_path_, 0, 0, "", false, 2048);
            if (!loaded) {
                LOG_ERROR("[Vision Delegate] Failed to load on-demand vision delegate engine!");
                if (evicted) {
                    expert_loader_.restore_l1_from_host(get_loader_freq_counts());
                    refresh_resident_pointers();
                }
                return "";
            }

            // Run vision inference in clean VRAM
            std::string perception = delegate_engine->describe_image(image_b64, user_query);

            // Unload vision delegate from VRAM
            delegate_engine.reset();
            cudaDeviceSynchronize();
            LOG_INFO("[Vision Delegate] Delegate unloaded from VRAM.");

            // Restore Nemotron L1 expert cache if it was evicted
            if (evicted) {
                expert_loader_.restore_l1_from_host(get_loader_freq_counts());
                refresh_resident_pointers();
            }

            auto t_end = std::chrono::steady_clock::now();
            double total_s = std::chrono::duration<double>(t_end - t_start).count();
            LOG_INFO("[Vision Delegate] Dynamic vision turn completed in %.2f s", total_s);
            return perception;
        }

        // 3. Direct vision tower execution (when running inside Qwen delegate itself)
        if (!has_vision_ || !ensure_vision_loaded()) {
            LOG_WARN("[Vision Delegate] describe_image called but vision tower could not be loaded!");
            return "";
        }

        moecher::vision::ProcessedImage proc_img;
        int img_sz = vision_tower_->image_size_;
        if (!moecher::vision::preprocess_image_from_base64(image_b64, proc_img, img_sz)) {
            LOG_WARN("[Vision Delegate] Failed to preprocess image from base64 (target size %d)!", img_sz);
            return "";
        }

        const __nv_bfloat16* d_vis_out = vision_tower_->forward(proc_img, main_stream_);
        if (!d_vis_out) {
            LOG_WARN("[Vision Delegate] Vision tower forward returned null!");
            return "";
        }

        int IM_START = tokenizer_.get_token_id("<|im_start|>");
        int IM_END = tokenizer_.get_token_id("<|im_end|>");
        int VIS_START = tokenizer_.get_token_id("<|vision_start|>");
        int VIS_END = tokenizer_.get_token_id("<|vision_end|>");
        int IMG_PAD = tokenizer_.get_token_id("<|image_pad|>");

        if (IM_START < 0) IM_START = 151644;
        if (IM_END < 0) IM_END = 151645;
        if (VIS_START < 0) VIS_START = 151652;
        if (VIS_END < 0) VIS_END = 151653;
        if (IMG_PAD < 0) IMG_PAD = 151655;

        std::vector<int> prompt;
        auto append_str = [&](const std::string& s) {
            auto enc = tokenizer_.encode(s);
            prompt.insert(prompt.end(), enc.begin(), enc.end());
        };

        prompt.push_back(IM_START);
        append_str("system\nYou are a helpful assistant.");
        prompt.push_back(IM_END);
        append_str("\n");

        prompt.push_back(IM_START);
        append_str("user\n");
        prompt.push_back(VIS_START);
        int visual_start_pos = (int)prompt.size();
        int visual_num_tokens = vision_tower_->merged_patches_; // 576
        prompt.insert(prompt.end(), visual_num_tokens, IMG_PAD);
        prompt.push_back(VIS_END);

        std::string qwen_instruction = "Describe this image in detail. Identify any vehicles (make, model, color, year), read exact text and license plates, and detail structural parts and environment.";
        if (!user_query.empty() && user_query.size() < 250) {
            qwen_instruction += "\nUser inquiry to address: " + user_query;
        }
        append_str(qwen_instruction);
        prompt.push_back(IM_END);
        append_str("\n");

        prompt.push_back(IM_START);
        append_str("assistant\n");

        reset_all_kv_caches();
        turn_kv_snapshot_.valid = false;
        turn_kv_snapshot_.tokens.clear();

        std::string perception = generate(
            prompt,
            /*max_tokens=*/160,
            /*temperature=*/0.0f,
            /*token_callback=*/nullptr,
            /*repetition_penalty=*/1.10f,
            /*enable_thinking=*/false,
            /*max_thinking_tokens=*/0,
            /*top_p=*/0.9f,
            /*min_p=*/0.05f,
            /*top_k=*/20,
            d_vis_out,
            visual_start_pos,
            visual_num_tokens
        );

        if (gpu_caps_.is_vram_constrained) {
            unload_vision_tower();
        }

        while (!perception.empty() && (perception.front() == ' ' || perception.front() == '\n' || perception.front() == '\r')) {
            perception.erase(perception.begin());
        }
        while (!perception.empty() && (perception.back() == ' ' || perception.back() == '\n' || perception.back() == '\r')) {
            perception.pop_back();
        }

        LOG_INFO("[Vision Delegate] Perception output (%zu chars): \"%s\"", perception.size(), perception.c_str());
        return perception;
    }

private:
    // ── Dense tensor loading ────────────────────────────────────────────────

    bool load_dense_tensors(const std::string& dense_path,
                            const json& tensor_map) {
        LOG_INFO("Loading dense tensors from %s", dense_path.c_str());

        moecher::platform::MemoryMappedFile dense_mmap;
        if (!dense_mmap.open_read(dense_path)) {
            LOG_ERROR("Cannot open or mmap dense bin: %s", dense_path.c_str());
            return false;
        }
        void* mapped = dense_mmap.data();

        auto load_tensor = [&](GPUTensor& gpu, const std::string& name) -> bool {
            if (!tensor_map.contains(name)) {
                return false;
            }
            auto& info = tensor_map[name];
            int64_t offset = info["offset"].get<int64_t>();
            int64_t nbytes = info["nbytes"].get<int64_t>();
            gpu.dtype = info["dtype"].get<std::string>();
            gpu.shape.clear();
            for (auto& s : info["shape"]) gpu.shape.push_back(s.get<int>());
            gpu.alloc(nbytes);
            CUDA_CHECK(cudaMemcpy(gpu.data, (char*)mapped + offset, nbytes,
                                   cudaMemcpyHostToDevice));
            return true;
        };

        auto load_quant_tensor = [&](GPUTensor& gpu_w, GPUTensor& gpu_s, const std::string& name) -> bool {
            if (!tensor_map.contains(name)) {
                return false;
            }
            auto& info = tensor_map[name];
            std::string dtype = info.value("dtype", "");
            std::string scale_name = name + "_scale";
            if (!tensor_map.contains(scale_name)) {
                size_t w_pos = name.rfind(".weight");
                if (w_pos != std::string::npos) {
                    scale_name = name.substr(0, w_pos) + ".weight_scale";
                }
            }
            bool has_separate_scale = tensor_map.contains(scale_name);

            if (dtype != "int4" && dtype != "int3" && dtype != "fp4" && dtype != "nvfp4" &&
                !(dtype == "U8" && has_separate_scale)) {
                return false;
            }
            int64_t offset = info["offset"].get<int64_t>();
            int64_t nbytes = info["nbytes"].get<int64_t>();
            gpu_w.dtype = dtype;
            gpu_w.shape.clear();
            for (auto& s : info["shape"]) gpu_w.shape.push_back(s.get<int>());
            gpu_w.alloc(nbytes);
            CUDA_CHECK(cudaMemcpy(gpu_w.data, (char*)mapped + offset, nbytes, cudaMemcpyHostToDevice));

            if (name == "lm_head.weight" && info.contains("scale_2")) {
                head_weight_scale_2_ = info["scale_2"].get<float>();
                LOG_INFO("Loaded lm_head with NVFP4 and scale_2 = %e", head_weight_scale_2_);
            }

            if (info.contains("scale_offset")) {
                int64_t scale_offset = info["scale_offset"].get<int64_t>();
                int64_t scale_nbytes = info["scale_nbytes"].get<int64_t>();
                gpu_s.dtype = info.value("scale_dtype", (dtype == "fp4" || dtype == "nvfp4") ? "F8_E4M3" : "bfloat16");
                gpu_s.alloc(scale_nbytes);
                CUDA_CHECK(cudaMemcpy(gpu_s.data, (char*)mapped + scale_offset, scale_nbytes, cudaMemcpyHostToDevice));
            } else if (has_separate_scale) {
                load_tensor(gpu_s, scale_name);
            }
            return true;
        };

        auto read_scalar_f32 = [&](const std::string& name, float default_val = 1.0f) -> float {
            if (!tensor_map.contains(name)) return default_val;
            auto& info = tensor_map[name];
            if (info.contains("scalar_scale")) return info["scalar_scale"].get<float>();
            if (info.contains("scale_2")) return info["scale_2"].get<float>();
            if (info.value("nbytes", 0) == 4) {
                int64_t offset = info["offset"].get<int64_t>();
                return *(const float*)((const char*)mapped + offset);
            }
            return default_val;
        };

        auto load_layer_weight = [&](GPUTensor& gpu_w, GPUTensor& gpu_s, const std::string& name) -> bool {
            if (load_quant_tensor(gpu_w, gpu_s, name)) return true;
            return load_tensor(gpu_w, name);
        };

        // Load global tensors
        if (!load_quant_tensor(embed_weight_, embed_weight_scale_, "embed.weight")) {
            if (!load_quant_tensor(embed_weight_, embed_weight_scale_, "backbone.embeddings.weight")) {
                if (!load_quant_tensor(embed_weight_, embed_weight_scale_, "model.embed_tokens.weight")) {
                    if (!load_quant_tensor(embed_weight_, embed_weight_scale_, "model.language_model.embed_tokens.weight")) {
                        if (!load_tensor(embed_weight_, "embed.weight")) {
                            if (!load_tensor(embed_weight_, "backbone.embeddings.weight")) {
                                if (!load_tensor(embed_weight_, "model.embed_tokens.weight")) {
                                    load_tensor(embed_weight_, "model.language_model.embed_tokens.weight");
                                }
                            }
                        }
                    }
                }
            }
        }
        if (!load_quant_tensor(head_weight_, head_weight_scale_, "head.weight")) {
            if (!load_quant_tensor(head_weight_, head_weight_scale_, "lm_head.weight")) {
                if (!load_quant_tensor(head_weight_, head_weight_scale_, "model.language_model.lm_head.weight")) {
                    if (!load_tensor(head_weight_, "head.weight")) {
                        if (!load_tensor(head_weight_, "lm_head.weight")) {
                            if (!load_tensor(head_weight_, "model.language_model.lm_head.weight")) {
                                // Tie embeddings if lm_head is shared
                                if (embed_weight_.data) {
                                    head_weight_.dtype = embed_weight_.dtype;
                                    head_weight_.shape = embed_weight_.shape;
                                    head_weight_.alloc(embed_weight_.size_bytes);
                                    CUDA_CHECK(cudaMemcpy(head_weight_.data, embed_weight_.data, embed_weight_.size_bytes, cudaMemcpyDeviceToDevice));
                                    if (embed_weight_scale_.data) {
                                        head_weight_scale_.dtype = embed_weight_scale_.dtype;
                                        head_weight_scale_.shape = embed_weight_scale_.shape;
                                        head_weight_scale_.alloc(embed_weight_scale_.size_bytes);
                                        CUDA_CHECK(cudaMemcpy(head_weight_scale_.data, embed_weight_scale_.data, embed_weight_scale_.size_bytes, cudaMemcpyDeviceToDevice));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (!load_tensor(norm_weight_, "norm.weight")) {
            if (!load_tensor(norm_weight_, "backbone.norm_f.weight")) {
                if (!load_tensor(norm_weight_, "model.norm.weight")) {
                    load_tensor(norm_weight_, "model.language_model.norm.weight");
                }
            }
        }

        if (cfg_.architecture == ModelArch::DEEPSEEK_V4) {
            // Head HC
            load_tensor(hc_head_fn_, "hc_head_fn");
            load_tensor(hc_head_base_, "hc_head_base");
            load_tensor(hc_head_scale_, "hc_head_scale");
        }

        // Load per-layer tensors
        layers_.resize(cfg_.num_hidden_layers);
        for (int l = 0; l < cfg_.num_hidden_layers; l++) {
            auto& lw = layers_[l];
            lw.d_comp_kv_count.alloc(sizeof(int32_t));
            lw.d_attn_cache_len.alloc(sizeof(int32_t));
            CUDA_CHECK(cudaMemset(lw.d_comp_kv_count.data, 0, sizeof(int32_t)));
            CUDA_CHECK(cudaMemset(lw.d_attn_cache_len.data, 0, sizeof(int32_t)));

            if (cfg_.is_qwen()) {
                std::string hf_prefix = "model.layers." + std::to_string(l);
                std::string lm_prefix = "model.language_model.layers." + std::to_string(l);
                std::string alt_prefix = "layers." + std::to_string(l);

                // RMSNorms
                if (!load_tensor(lw.attn_norm_w, lm_prefix + ".input_layernorm.weight")) {
                    if (!load_tensor(lw.attn_norm_w, hf_prefix + ".input_layernorm.weight")) {
                        load_tensor(lw.attn_norm_w, alt_prefix + ".attn_norm.weight");
                    }
                }
                if (!load_tensor(lw.ffn_norm_w, lm_prefix + ".post_attention_layernorm.weight")) {
                    if (!load_tensor(lw.ffn_norm_w, hf_prefix + ".post_attention_layernorm.weight")) {
                        load_tensor(lw.ffn_norm_w, alt_prefix + ".ffn_norm.weight");
                    }
                }

                // Check for Linear Attention (Gated DeltaNet) vs Full GQA Attention
                if (load_layer_weight(lw.w_in_qkv, lw.w_in_qkv_scale, lm_prefix + ".linear_attn.in_proj_qkv.weight") ||
                    load_layer_weight(lw.w_in_qkv, lw.w_in_qkv_scale, hf_prefix + ".linear_attn.in_proj_qkv.weight")) {
                    lw.is_linear_attn = true;
                    if (!load_layer_weight(lw.w_in_z, lw.w_in_z_scale, lm_prefix + ".linear_attn.in_proj_z.weight")) {
                        load_layer_weight(lw.w_in_z, lw.w_in_z_scale, hf_prefix + ".linear_attn.in_proj_z.weight");
                    }
                    if (!load_tensor(lw.w_in_a, lm_prefix + ".linear_attn.in_proj_a.weight")) {
                        load_tensor(lw.w_in_a, hf_prefix + ".linear_attn.in_proj_a.weight");
                    }
                    if (!load_tensor(lw.w_in_b, lm_prefix + ".linear_attn.in_proj_b.weight")) {
                        load_tensor(lw.w_in_b, hf_prefix + ".linear_attn.in_proj_b.weight");
                    }
                    if (!load_tensor(lw.conv1d_w, lm_prefix + ".linear_attn.conv1d.weight")) {
                        load_tensor(lw.conv1d_w, hf_prefix + ".linear_attn.conv1d.weight");
                    }
                    if (!load_tensor(lw.A_log, lm_prefix + ".linear_attn.A_log")) {
                        load_tensor(lw.A_log, hf_prefix + ".linear_attn.A_log");
                    }
                    if (!load_tensor(lw.dt_bias, lm_prefix + ".linear_attn.dt_bias")) {
                        load_tensor(lw.dt_bias, hf_prefix + ".linear_attn.dt_bias");
                    }
                    if (!load_tensor(lw.linear_norm_w, lm_prefix + ".linear_attn.norm.weight")) {
                        load_tensor(lw.linear_norm_w, hf_prefix + ".linear_attn.norm.weight");
                    }
                    if (!load_layer_weight(lw.linear_out_proj, lw.linear_out_proj_scale, lm_prefix + ".linear_attn.out_proj.weight")) {
                        load_layer_weight(lw.linear_out_proj, lw.linear_out_proj_scale, hf_prefix + ".linear_attn.out_proj.weight");
                    }

                    // Linear attention recurrent states managed in contiguous target_ssm_pool_
                } else {
                    // Full GQA Projections
                    lw.is_linear_attn = false;
                    if (!load_layer_weight(lw.w_q, lw.w_q_scale, lm_prefix + ".self_attn.q_proj.weight")) {
                        if (!load_layer_weight(lw.w_q, lw.w_q_scale, hf_prefix + ".self_attn.q_proj.weight")) {
                            load_layer_weight(lw.w_q, lw.w_q_scale, alt_prefix + ".attn.q.weight");
                        }
                    }
                    if (!load_layer_weight(lw.w_k, lw.w_k_scale, lm_prefix + ".self_attn.k_proj.weight")) {
                        if (!load_layer_weight(lw.w_k, lw.w_k_scale, hf_prefix + ".self_attn.k_proj.weight")) {
                            load_layer_weight(lw.w_k, lw.w_k_scale, alt_prefix + ".attn.k.weight");
                        }
                    }
                    if (!load_layer_weight(lw.w_v, lw.w_v_scale, lm_prefix + ".self_attn.v_proj.weight")) {
                        if (!load_layer_weight(lw.w_v, lw.w_v_scale, hf_prefix + ".self_attn.v_proj.weight")) {
                            load_layer_weight(lw.w_v, lw.w_v_scale, alt_prefix + ".attn.v.weight");
                        }
                    }
                    if (!load_layer_weight(lw.w_o, lw.w_o_scale, lm_prefix + ".self_attn.o_proj.weight")) {
                        if (!load_layer_weight(lw.w_o, lw.w_o_scale, hf_prefix + ".self_attn.o_proj.weight")) {
                            load_layer_weight(lw.w_o, lw.w_o_scale, alt_prefix + ".attn.o.weight");
                        }
                    }
                    if (!load_tensor(lw.gqa_q_norm_w, lm_prefix + ".self_attn.q_norm.weight")) {
                        load_tensor(lw.gqa_q_norm_w, hf_prefix + ".self_attn.q_norm.weight");
                    }
                    if (!load_tensor(lw.gqa_k_norm_w, lm_prefix + ".self_attn.k_norm.weight")) {
                        load_tensor(lw.gqa_k_norm_w, hf_prefix + ".self_attn.k_norm.weight");
                    }
                    if (!load_tensor(lw.gqa_q_bias, lm_prefix + ".self_attn.q_proj.bias")) {
                        load_tensor(lw.gqa_q_bias, hf_prefix + ".self_attn.q_proj.bias");
                    }
                    if (!load_tensor(lw.gqa_k_bias, lm_prefix + ".self_attn.k_proj.bias")) {
                        load_tensor(lw.gqa_k_bias, hf_prefix + ".self_attn.k_proj.bias");
                    }
                    if (!load_tensor(lw.gqa_v_bias, lm_prefix + ".self_attn.v_proj.bias")) {
                        load_tensor(lw.gqa_v_bias, hf_prefix + ".self_attn.v_proj.bias");
                    }

                    // Allocate GQA FP8 KV cache (1 byte per element)
                    int max_seq = cfg_.max_seq_len > 0 ? cfg_.max_seq_len : 32768;
                    int n_kv = cfg_.num_key_value_heads > 0 ? cfg_.num_key_value_heads : 4;
                    int h_dim = cfg_.head_dim > 0 ? cfg_.head_dim : 256;
                    lw.k_cache_gqa.alloc((size_t)max_seq * n_kv * h_dim * sizeof(uint8_t));
                    lw.v_cache_gqa.alloc((size_t)max_seq * n_kv * h_dim * sizeof(uint8_t));
                    CUDA_CHECK(cudaMemset(lw.k_cache_gqa.data, 0, lw.k_cache_gqa.size_bytes));
                    CUDA_CHECK(cudaMemset(lw.v_cache_gqa.data, 0, lw.v_cache_gqa.size_bytes));
                }

                // SwiGLU FFN Projections
                if (!load_layer_weight(lw.w_gate, lw.w_gate_scale, lm_prefix + ".mlp.gate_proj.weight")) {
                    if (!load_layer_weight(lw.w_gate, lw.w_gate_scale, hf_prefix + ".mlp.gate_proj.weight")) {
                        load_layer_weight(lw.w_gate, lw.w_gate_scale, alt_prefix + ".ffn.gate.weight");
                    }
                }
                if (!load_layer_weight(lw.w_up, lw.w_up_scale, lm_prefix + ".mlp.up_proj.weight")) {
                    if (!load_layer_weight(lw.w_up, lw.w_up_scale, hf_prefix + ".mlp.up_proj.weight")) {
                        load_layer_weight(lw.w_up, lw.w_up_scale, alt_prefix + ".ffn.up.weight");
                    }
                }
                if (!load_layer_weight(lw.w_down, lw.w_down_scale, lm_prefix + ".mlp.down_proj.weight")) {
                    if (!load_layer_weight(lw.w_down, lw.w_down_scale, hf_prefix + ".mlp.down_proj.weight")) {
                        load_layer_weight(lw.w_down, lw.w_down_scale, alt_prefix + ".ffn.down.weight");
                    }
                }

                if ((l + 1) % 10 == 0 || l == cfg_.num_hidden_layers - 1) {
                    LOG_INFO("  Loaded Qwen layer %d/%d (%s%s)",
                             l + 1, cfg_.num_hidden_layers,
                             lw.is_linear_attn ? "Linear Attention DeltaNet" : "Full GQA Attention",
                             lw.w_gate.dtype == "fp4" ? " [FP4]" : (lw.w_gate.dtype == "int4" ? " [INT4]" : (lw.w_gate.dtype == "int3" ? " [INT3]" : (lw.w_gate.dtype == "BF16" || lw.w_gate.dtype == "bfloat16" ? " [BF16]" : ""))));
                }
                continue;
            }

            if (cfg_.is_nemotron()) {
                std::string prefix = "backbone.layers." + std::to_string(l);
                int ltype = (l < (int)cfg_.layer_types.size()) ? cfg_.layer_types[l] : 1;

                if (ltype == 1) {
                    // Mamba-2 SSM layer
                    lw.is_linear_attn = true;
                    load_tensor(lw.attn_norm_w, prefix + ".norm.weight");
                    load_tensor(lw.linear_norm_w, prefix + ".mixer.norm.weight");
                    load_tensor(lw.A_log, prefix + ".mixer.A_log");
                    load_tensor(lw.mamba_d, prefix + ".mixer.D");
                    load_tensor(lw.conv1d_w, prefix + ".mixer.conv1d.weight");
                    load_tensor(lw.conv1d_bias, prefix + ".mixer.conv1d.bias");
                    load_tensor(lw.dt_bias, prefix + ".mixer.dt_bias");

                    // Projections (NVFP4, FP8, or BF16)
                    load_layer_weight(lw.w_in_qkv, lw.w_in_qkv_scale, prefix + ".mixer.in_proj.weight");
                    load_layer_weight(lw.linear_out_proj, lw.linear_out_proj_scale, prefix + ".mixer.out_proj.weight");

                    std::string in_proj_name = prefix + ".mixer.in_proj.weight";
                    lw.w_in_qkv_scale_val = read_scalar_f32(prefix + ".mixer.in_proj.weight_scale_2",
                                                            read_scalar_f32(prefix + ".mixer.in_proj.weight_scale",
                                                                            read_scalar_f32(in_proj_name, 1.0f)));
                    std::string out_proj_name = prefix + ".mixer.out_proj.weight";
                    lw.linear_out_proj_scale_val = read_scalar_f32(prefix + ".mixer.out_proj.weight_scale_2",
                                                                   read_scalar_f32(prefix + ".mixer.out_proj.weight_scale",
                                                                                   read_scalar_f32(out_proj_name, 1.0f)));
                } else if (ltype == 0) {
                    // GQA Attention layer
                    lw.is_linear_attn = false;
                    load_tensor(lw.attn_norm_w, prefix + ".norm.weight");
                    load_layer_weight(lw.w_q, lw.w_q_scale, prefix + ".mixer.q_proj.weight");
                    load_layer_weight(lw.w_k, lw.w_k_scale, prefix + ".mixer.k_proj.weight");
                    load_layer_weight(lw.w_v, lw.w_v_scale, prefix + ".mixer.v_proj.weight");
                    load_layer_weight(lw.w_o, lw.w_o_scale, prefix + ".mixer.o_proj.weight");

                    int max_seq = cfg_.max_seq_len > 0 ? cfg_.max_seq_len : 65536;
                    int n_kv = cfg_.num_key_value_heads > 0 ? cfg_.num_key_value_heads : 2;
                    int h_dim = cfg_.head_dim > 0 ? cfg_.head_dim : 128;
                    lw.k_cache_gqa.alloc((size_t)max_seq * n_kv * h_dim * sizeof(uint8_t));
                    lw.v_cache_gqa.alloc((size_t)max_seq * n_kv * h_dim * sizeof(uint8_t));
                    CUDA_CHECK(cudaMemset(lw.k_cache_gqa.data, 0, lw.k_cache_gqa.size_bytes));
                    CUDA_CHECK(cudaMemset(lw.v_cache_gqa.data, 0, lw.v_cache_gqa.size_bytes));
                } else {
                    // MoE Layer (ltype == 2)
                    lw.is_linear_attn = false;
                    load_tensor(lw.ffn_norm_w, prefix + ".norm.weight");
                    load_tensor(lw.gate_w, prefix + ".mixer.gate.weight");
                    load_tensor(lw.gate_bias, prefix + ".mixer.gate.e_score_correction_bias");

                    // Convert router gate weights from F32 to BF16 for fast matrix multiplication
                    if (lw.gate_w.dtype == "F32" || lw.gate_w.dtype == "float32") {
                        GPUTensor f32_gate;
                        f32_gate.dtype = "F32";
                        f32_gate.alloc(lw.gate_w.size_bytes);
                        CUDA_CHECK(cudaMemcpy(f32_gate.data, lw.gate_w.data, lw.gate_w.size_bytes, cudaMemcpyDeviceToDevice));
                        lw.gate_w.free();
                        lw.gate_w.dtype = "bfloat16";
                        int n_exp = cfg_.layer_n_routed_experts.empty() ? cfg_.n_routed_experts : cfg_.get_layer_n_routed_experts(l);
                        if (n_exp <= 0) n_exp = 128;
                        int gate_elements = n_exp * cfg_.hidden_size;
                        lw.gate_w.alloc(gate_elements * sizeof(__nv_bfloat16));
                        f32_to_bf16_cuda(lw.gate_w.bf16(), f32_gate.f32(), gate_elements, main_stream_);
                        CUDA_CHECK(cudaStreamSynchronize(main_stream_));
                        f32_gate.free();
                    }

                    // Shared experts
                    load_layer_weight(lw.shared_w1_w, lw.shared_w1_s, prefix + ".mixer.shared_experts.up_proj.weight");
                    load_layer_weight(lw.shared_w2_w, lw.shared_w2_s, prefix + ".mixer.shared_experts.down_proj.weight");

                    lw.shared_w1_s2 = read_scalar_f32(prefix + ".mixer.shared_experts.up_proj.weight_scale_2",
                                                      read_scalar_f32(prefix + ".mixer.shared_experts.up_proj.weight", 1.0f));
                    lw.shared_w2_s2 = read_scalar_f32(prefix + ".mixer.shared_experts.down_proj.weight_scale_2",
                                                      read_scalar_f32(prefix + ".mixer.shared_experts.down_proj.weight", 1.0f));

                    if (l == 1) {
                        LOG_INFO("Nemotron Layer 1 shared experts: up_w=%p up_s=%p (s2=%f), down_w=%p down_s=%p (s2=%f)",
                                 lw.shared_w1_w.data, lw.shared_w1_s.data, lw.shared_w1_s2,
                                 lw.shared_w2_w.data, lw.shared_w2_s.data, lw.shared_w2_s2);
                    }

                    // LatentMoE Projections (Nemotron Bolt)
                    if (cfg_.is_nemotron_bolt() || tensor_map.contains(prefix + ".mixer.fc1_latent_proj.weight")) {
                        load_layer_weight(lw.fc1_latent_proj, lw.fc1_latent_proj_scale, prefix + ".mixer.fc1_latent_proj.weight");
                        load_layer_weight(lw.fc2_latent_proj, lw.fc2_latent_proj_scale, prefix + ".mixer.fc2_latent_proj.weight");
                        lw.fc1_scale_val = read_scalar_f32(prefix + ".mixer.fc1_latent_proj.weight_scale",
                                                           read_scalar_f32(prefix + ".mixer.fc1_latent_proj.weight", 1.0f));
                        lw.fc2_scale_val = read_scalar_f32(prefix + ".mixer.fc2_latent_proj.weight_scale",
                                                           read_scalar_f32(prefix + ".mixer.fc2_latent_proj.weight", 1.0f));
                    }
                }

                if ((l + 1) % 10 == 0 || l == cfg_.num_hidden_layers - 1) {
                    LOG_INFO("  Loaded Nemotron layer %d/%d (type=%s)",
                             l + 1, cfg_.num_hidden_layers,
                             ltype == 1 ? "Mamba-2 SSM" : (ltype == 0 ? "GQA Attention" : "MoE"));
                }
                continue;
            }

            std::string prefix = "layers." + std::to_string(l);

            if (!load_quant_tensor(lw.wq_a_w, lw.wq_a_s, prefix + ".attn.wq_a.weight")) {
                load_tensor(lw.wq_a_w, prefix + ".attn.wq_a.weight");
                load_tensor(lw.wq_a_s, prefix + ".attn.wq_a.scale");
            }
            if (!load_quant_tensor(lw.wq_b_w, lw.wq_b_s, prefix + ".attn.wq_b.weight")) {
                load_tensor(lw.wq_b_w, prefix + ".attn.wq_b.weight");
                load_tensor(lw.wq_b_s, prefix + ".attn.wq_b.scale");
            }
            if (!load_quant_tensor(lw.wkv_w, lw.wkv_s, prefix + ".attn.wkv.weight")) {
                load_tensor(lw.wkv_w, prefix + ".attn.wkv.weight");
                load_tensor(lw.wkv_s, prefix + ".attn.wkv.scale");
            }
            if (!load_quant_tensor(lw.wo_a_w, lw.wo_a_s, prefix + ".attn.wo_a.weight")) {
                load_tensor(lw.wo_a_w, prefix + ".attn.wo_a.weight");
                load_tensor(lw.wo_a_s, prefix + ".attn.wo_a.scale");
            }
            if (!load_quant_tensor(lw.wo_b_w, lw.wo_b_s, prefix + ".attn.wo_b.weight")) {
                load_tensor(lw.wo_b_w, prefix + ".attn.wo_b.weight");
                load_tensor(lw.wo_b_s, prefix + ".attn.wo_b.scale");
            }
            load_tensor(lw.q_norm_w, prefix + ".attn.q_norm.weight");
            load_tensor(lw.kv_norm_w, prefix + ".attn.kv_norm.weight");
            load_tensor(lw.attn_norm_w, prefix + ".attn_norm.weight");
            load_tensor(lw.attn_sink, prefix + ".attn.attn_sink");
            load_tensor(lw.ffn_norm_w, prefix + ".ffn_norm.weight");

            // Compressor weights (for layers with compress_ratio > 0)
            int ratio = cfg_.layer_compress_ratio(l);
            if (ratio > 0) {
                load_tensor(lw.comp_wkv, prefix + ".attn.compressor.wkv.weight");
                load_tensor(lw.comp_wgate, prefix + ".attn.compressor.wgate.weight");
                load_tensor(lw.comp_ape, prefix + ".attn.compressor.ape");
                load_tensor(lw.comp_norm, prefix + ".attn.compressor.norm.weight");

                // Allocate compressed KV cache for the full context window
                int max_comp = cfg_.max_seq_len / ratio + 2;
                lw.comp_kv_cache.alloc((size_t)max_comp * cfg_.head_dim * sizeof(__nv_bfloat16));
                CUDA_CHECK(cudaMemset(lw.comp_kv_cache.data, 0, lw.comp_kv_cache.size_bytes));
                lw.comp_kv_count = 0;

                // Allocate compressor state buffers
                // coff = 1 + (ratio == 4 ? 1 : 0)  — overlap only for CSA (ratio=4)
                int coff = (ratio == 4) ? 2 : 1;
                int state_rows = coff * ratio;
                int state_cols = coff * cfg_.head_dim;
                lw.comp_kv_state.alloc((size_t)state_rows * state_cols * sizeof(float));
                lw.comp_score_state.alloc((size_t)state_rows * state_cols * sizeof(float));
                CUDA_CHECK(cudaMemset(lw.comp_kv_state.data, 0, lw.comp_kv_state.size_bytes));
                // Initialize score_state to -inf so softmax ignores unfilled slots
                std::vector<float> neg_inf(state_rows * state_cols, -std::numeric_limits<float>::infinity());
                CUDA_CHECK(cudaMemcpy(lw.comp_score_state.data, neg_inf.data(),
                                       neg_inf.size() * sizeof(float), cudaMemcpyHostToDevice));

                LOG_INFO("  Layer %d: compressor loaded (ratio=%d, coff=%d, max_comp=%d)", l, ratio, coff, max_comp);

                if (ratio == 4) {
                    load_tensor(lw.indexer_comp_wkv, prefix + ".attn.indexer.compressor.wkv.weight");
                    load_tensor(lw.indexer_comp_wgate, prefix + ".attn.indexer.compressor.wgate.weight");
                    load_tensor(lw.indexer_comp_ape, prefix + ".attn.indexer.compressor.ape");
                    load_tensor(lw.indexer_comp_norm, prefix + ".attn.indexer.compressor.norm.weight");
                    load_tensor(lw.indexer_weights_proj, prefix + ".attn.indexer.weights_proj.weight");

                    if (!load_quant_tensor(lw.indexer_wq_b_w, lw.indexer_wq_b_s, prefix + ".attn.indexer.wq_b.weight")) {
                        load_tensor(lw.indexer_wq_b_w, prefix + ".attn.indexer.wq_b.weight");
                        load_tensor(lw.indexer_wq_b_s, prefix + ".attn.indexer.wq_b.scale");
                    }

                    int indexer_head_dim = 128;
                    lw.indexer_comp_kv_cache.alloc((size_t)max_comp * indexer_head_dim * sizeof(__nv_bfloat16));
                    CUDA_CHECK(cudaMemset(lw.indexer_comp_kv_cache.data, 0, lw.indexer_comp_kv_cache.size_bytes));

                    int indexer_proj_dim = 2 * indexer_head_dim;
                    int indexer_state_rows = 2 * ratio;
                    lw.indexer_comp_kv_state.alloc((size_t)indexer_state_rows * indexer_proj_dim * sizeof(float));
                    lw.indexer_comp_score_state.alloc((size_t)indexer_state_rows * indexer_proj_dim * sizeof(float));
                    CUDA_CHECK(cudaMemset(lw.indexer_comp_kv_state.data, 0, lw.indexer_comp_kv_state.size_bytes));
                    std::vector<float> neg_inf_idx(indexer_state_rows * indexer_proj_dim, -std::numeric_limits<float>::infinity());
                    CUDA_CHECK(cudaMemcpy(lw.indexer_comp_score_state.data, neg_inf_idx.data(),
                                           neg_inf_idx.size() * sizeof(float), cudaMemcpyHostToDevice));
                }
            }

            // Gate
            load_tensor(lw.gate_w, prefix + ".ffn.gate.weight");
            if (l < cfg_.n_hash_layers) {
                load_tensor(lw.tid2eid, prefix + ".ffn.gate.tid2eid");
                if (lw.tid2eid.data) {
                    size_t count = lw.tid2eid.size_bytes / sizeof(int64_t);
                    lw.tid2eid_host.resize(count);
                    CUDA_CHECK(cudaMemcpy(lw.tid2eid_host.data(), lw.tid2eid.i64(),
                                           lw.tid2eid.size_bytes, cudaMemcpyDeviceToHost));
                }
            } else {
                load_tensor(lw.gate_bias, prefix + ".ffn.gate.bias");
                if (lw.gate_bias.data) {
                    lw.gate_bias_host.resize(cfg_.n_routed_experts);
                    CUDA_CHECK(cudaMemcpy(lw.gate_bias_host.data(), lw.gate_bias.f32(),
                                           cfg_.n_routed_experts * sizeof(float), cudaMemcpyDeviceToHost));
                }
            }

            // Shared expert (FP8 or INT4)
            if (!load_quant_tensor(lw.shared_w1_w, lw.shared_w1_s, prefix + ".ffn.shared_experts.w1.weight")) {
                load_tensor(lw.shared_w1_w, prefix + ".ffn.shared_experts.w1.weight");
                load_tensor(lw.shared_w1_s, prefix + ".ffn.shared_experts.w1.scale");
            }
            if (!load_quant_tensor(lw.shared_w2_w, lw.shared_w2_s, prefix + ".ffn.shared_experts.w2.weight")) {
                load_tensor(lw.shared_w2_w, prefix + ".ffn.shared_experts.w2.weight");
                load_tensor(lw.shared_w2_s, prefix + ".ffn.shared_experts.w2.scale");
            }
            if (!load_quant_tensor(lw.shared_w3_w, lw.shared_w3_s, prefix + ".ffn.shared_experts.w3.weight")) {
                load_tensor(lw.shared_w3_w, prefix + ".ffn.shared_experts.w3.weight");
                load_tensor(lw.shared_w3_s, prefix + ".ffn.shared_experts.w3.scale");
            }

            // HC parameters
            load_tensor(lw.hc_attn_fn, prefix + ".hc_attn_fn");
            load_tensor(lw.hc_attn_base, prefix + ".hc_attn_base");
            load_tensor(lw.hc_attn_scale, prefix + ".hc_attn_scale");
            load_tensor(lw.hc_ffn_fn, prefix + ".hc_ffn_fn");
            load_tensor(lw.hc_ffn_base, prefix + ".hc_ffn_base");
            load_tensor(lw.hc_ffn_scale, prefix + ".hc_ffn_scale");

            // KV cache
            lw.kv_cache.alloc((size_t)cfg_.sliding_window * cfg_.head_dim * sizeof(__nv_bfloat16));
            CUDA_CHECK(cudaMemset(lw.kv_cache.data, 0, lw.kv_cache.size_bytes));

            if ((l + 1) % 10 == 0 || l == cfg_.num_hidden_layers - 1) {
                LOG_INFO("  Loaded layer %d/%d", l + 1, cfg_.num_hidden_layers);
            }
        }

        if (cfg_.architecture == ModelArch::QWEN || cfg_.architecture == ModelArch::NEMOTRON_35) {
            int n_linear = 0;
            for (auto& lw : layers_) if (lw.is_linear_attn) n_linear++;
            if (n_linear > 0) {
                size_t ssm_heads = cfg_.ssm_dt_rank > 0 ? cfg_.ssm_dt_rank : (cfg_.is_nemotron_bolt() ? 128 : (cfg_.is_nemotron() ? 64 : 48));
                size_t ssm_d = cfg_.ssm_d_state > 0 ? cfg_.ssm_d_state : 128;
                size_t ssm_head_dim = cfg_.is_nemotron() ? 64 : ssm_d;
                size_t ssm_bytes_per_layer = ssm_heads * ssm_head_dim * ssm_d * (cfg_.is_nemotron() ? sizeof(float) : sizeof(__nv_bfloat16));
                size_t conv_dim = cfg_.ssm_d_conv > 0 ? cfg_.ssm_d_conv : 4;
                size_t conv_channels = cfg_.is_nemotron_bolt() ? 10240 : (cfg_.is_nemotron() ? 6144 : 10240);
                size_t conv_bytes_per_layer = conv_channels * conv_dim * sizeof(__nv_bfloat16);
                target_ssm_pool_.alloc(n_linear * ssm_bytes_per_layer);
                target_conv_pool_.alloc(n_linear * conv_bytes_per_layer);
                CUDA_CHECK(cudaMemset(target_ssm_pool_.data, 0, target_ssm_pool_.size_bytes));
                CUDA_CHECK(cudaMemset(target_conv_pool_.data, 0, target_conv_pool_.size_bytes));

                int num_slots;
                if (enable_tree_mtp_ && enable_mtp_) {
                    num_slots = gpu_caps_.is_vram_constrained ? std::min(8, tree_nodes_ + 1) : 8;
                } else if (cfg_.is_nemotron()) {
                    num_slots = 3; // Slots 0, 1, 2 for speculative verification up to M=4
                } else if (mtp_k_ == 1 || (mtp_k_ == 0 && gpu_caps_.is_vram_constrained)) {
                    num_slots = 1;
                } else if (mtp_k_ > 1) {
                    num_slots = std::min(8, mtp_k_);
                } else {
                    num_slots = 4;
                }
                for (int s = 0; s < num_slots; s++) {
                    target_ssm_slots_[s].alloc(n_linear * ssm_bytes_per_layer);
                    target_conv_slots_[s].alloc(n_linear * conv_bytes_per_layer);
                    CUDA_CHECK(cudaMemset(target_ssm_slots_[s].data, 0, target_ssm_slots_[s].size_bytes));
                    CUDA_CHECK(cudaMemset(target_conv_slots_[s].data, 0, target_conv_slots_[s].size_bytes));
                }
                for (int s = num_slots; s < 8; s++) {
                    target_ssm_slots_[s].data = nullptr;
                    target_ssm_slots_[s].size_bytes = 0;
                    target_conv_slots_[s].data = nullptr;
                    target_conv_slots_[s].size_bytes = 0;
                }

                int lin_idx = 0;
                for (int l = 0; l < cfg_.num_hidden_layers; l++) {
                    if (layers_[l].is_linear_attn) {
                        layers_[l].ssm_state.data = (uint8_t*)target_ssm_pool_.data + lin_idx * ssm_bytes_per_layer;
                        layers_[l].ssm_state.size_bytes = ssm_bytes_per_layer;
                        layers_[l].conv_state.data = (uint8_t*)target_conv_pool_.data + lin_idx * conv_bytes_per_layer;
                        layers_[l].conv_state.size_bytes = conv_bytes_per_layer;

                        for (int s = 0; s < 8; s++) {
                            if (s < num_slots) {
                                layers_[l].ssm_state_slots[s].data = (uint8_t*)target_ssm_slots_[s].data + lin_idx * ssm_bytes_per_layer;
                                layers_[l].ssm_state_slots[s].size_bytes = ssm_bytes_per_layer;
                                layers_[l].conv_state_slots[s].data = (uint8_t*)target_conv_slots_[s].data + lin_idx * conv_bytes_per_layer;
                                layers_[l].conv_state_slots[s].size_bytes = conv_bytes_per_layer;
                            } else {
                                layers_[l].ssm_state_slots[s].data = nullptr;
                                layers_[l].ssm_state_slots[s].size_bytes = 0;
                                layers_[l].conv_state_slots[s].data = nullptr;
                                layers_[l].conv_state_slots[s].size_bytes = 0;
                            }
                        }
                        lin_idx++;
                    }
                }
            }
        }

        // Load MTP Self-Drafter weights from the same checkpoint
        if (enable_mtp_ && cfg_.architecture == ModelArch::QWEN && embed_weight_.data) {
            std::string mtp_model_dir = model_dir_;
            if (mtp_model_dir.empty()) {
                mtp_model_dir = "f:/Moecher/models/qwen3_8_27b_q4";
            }
            if (tensor_map.contains("mtp.fc.weight")) {
                std::string draft_ids_path = mtp_model_dir + "/draft_vocab_ids.bin";
                std::string draft_head_path = mtp_model_dir + "/draft_lm_head_int8_bf16.bin";
                bool has_draft_head = (std::filesystem::exists(draft_ids_path) && std::filesystem::exists(draft_head_path));
                if ((head_weight_.dtype == "int4" || head_weight_.dtype == "fp4") && !has_draft_head) {
                    LOG_WARN("MTP: Full lm_head is %s and no draft_lm_head_int8_bf16.bin found — skipping MTP", head_weight_.dtype.c_str());
                } else {
                    mtp_drafter_.load_mtp_weights(mapped, tensor_map, 
                                                  (embed_weight_.dtype == "int4" || embed_weight_.dtype == "fp4") ? nullptr : embed_weight_.bf16(),
                                                  (embed_weight_.dtype == "int4" || embed_weight_.dtype == "fp4") ? (uint8_t*)embed_weight_.data : nullptr,
                                                  embed_weight_scale_.data,
                                                  embed_weight_.dtype == "int4",
                                                  embed_weight_.dtype == "fp4",
                                                  (head_weight_.dtype == "int4" || head_weight_.dtype == "fp4") ? nullptr : head_weight_.bf16(),
                                                  cfg_.vocab_size, mtp_model_dir, main_stream_,
                                                  cfg_.max_seq_len);
                }
            }
        } else if (!enable_mtp_) {
            LOG_INFO("MTP: Self-drafter explicitly disabled via --no-mtp flag");
        }

        dense_mmap.close();
        LOG_INFO("Dense tensors loaded");
        return true;
    }

    // ── Allocate working buffers ────────────────────────────────────────────

    void alloc_buffers() {
        int dim = cfg_.hidden_size;
        int n_heads = cfg_.num_attention_heads;
        int head_dim_val = cfg_.head_dim;
        int hc = cfg_.hc_mult;
        int moe_inter = cfg_.moe_intermediate_size;
        int top_k = cfg_.num_experts_per_tok;

        size_t max_inter = std::max({
            (size_t)(top_k + 1) * moe_inter,
            (size_t)cfg_.intermediate_size,
            (size_t)17408
        });
        size_t max_down = std::max({
            (size_t)(top_k + 1) * dim,
            (size_t)cfg_.intermediate_size,
            (size_t)dim
        });
        size_t max_q_dim = std::max({
            (size_t)n_heads * head_dim_val,
            (size_t)18560,
            (size_t)12288,
            (size_t)dim
        });

        buf_hidden_.alloc(dim * sizeof(__nv_bfloat16));
        buf_hidden2_.alloc(dim * sizeof(__nv_bfloat16));
        buf_q_.alloc(max_q_dim * sizeof(__nv_bfloat16));
        buf_kv_.alloc(head_dim_val * sizeof(__nv_bfloat16));
        buf_attn_out_.alloc(max_q_dim * sizeof(__nv_bfloat16));
        buf_lora_.alloc(std::max({
            (size_t)cfg_.q_lora_rank,
            (size_t)cfg_.o_lora_rank * cfg_.o_groups,
            (size_t)n_heads * head_dim_val
        }) * sizeof(__nv_bfloat16));
        buf_gate_.alloc(max_inter * sizeof(__nv_bfloat16));
        buf_up_.alloc(max_inter * sizeof(__nv_bfloat16));
        buf_down_.alloc(max_down * sizeof(__nv_bfloat16));
        buf_expert_out_.alloc(dim * sizeof(__nv_bfloat16));
        if (cfg_.n_routed_experts > 0) {
            buf_moe_accum_.alloc(dim * sizeof(__nv_bfloat16));
            int latent_size = cfg_.moe_latent_size > 0 ? cfg_.moe_latent_size : 1024;
            buf_latent_down_.alloc(latent_size * sizeof(__nv_bfloat16));
            buf_latent_accum_.alloc(latent_size * sizeof(__nv_bfloat16));
        }
        size_t dequant_needed;
        if (cfg_.n_routed_experts > 0) {
            dequant_needed = std::max((size_t)256 * 1024 * 1024, (size_t)max_inter * dim * sizeof(__nv_bfloat16));
        } else {
            dequant_needed = 48 * 1024 * 1024;
        }
        buf_dequant_.alloc(dequant_needed);
        buf_dequant_2_.alloc(dequant_needed);
        buf_logits_.alloc((size_t)cfg_.vocab_size * sizeof(float));
        int max_n_routed = std::max({cfg_.n_routed_experts, cfg_.max_n_routed_experts(), 512});
        int max_k_val = std::max({cfg_.num_experts_per_tok, cfg_.max_num_experts_per_tok(), 64});
        if (cfg_.n_routed_experts > 0) {
            buf_scores_f32_.alloc(max_n_routed * sizeof(float));
            buf_scores_bf16_.alloc(max_n_routed * sizeof(__nv_bfloat16));
            buf_topk_vals_.alloc(max_k_val * sizeof(float));
            buf_topk_idx_.alloc(max_k_val * sizeof(int32_t));
        }
        buf_linear_a_.alloc(64 * sizeof(__nv_bfloat16));
        buf_linear_b_.alloc(64 * sizeof(__nv_bfloat16));
        if (cfg_.is_nemotron()) {
            size_t conv_channels = cfg_.is_nemotron_bolt() ? 10240 : (cfg_.is_nemotron() ? 6144 : 10240);
            size_t temp_channels = cfg_.is_nemotron_bolt() ? 8192 : 4096;
            buf_mamba_conv_.alloc(conv_channels * sizeof(__nv_bfloat16));
            buf_mamba_temp_.alloc(temp_channels * sizeof(float));
        }
        buf_input_ids_.alloc(sizeof(int32_t));
        buf_input_token_.alloc(sizeof(int32_t));
        buf_input_pos_.alloc(sizeof(int32_t));
        buf_mrope_pos_.alloc(2 * sizeof(int32_t));
        int dummy_mrope[2] = {-1, 0};
        CUDA_CHECK(cudaMemcpy(buf_mrope_pos_.i32(), dummy_mrope, 2 * sizeof(int32_t), cudaMemcpyHostToDevice));
        buf_hc_state_.alloc((size_t)hc * dim * sizeof(__nv_bfloat16));
        buf_hc_after_attn_.alloc((size_t)hc * dim * sizeof(__nv_bfloat16));  // static intermediate after attention
        buf_hc_pre_.alloc(hc * sizeof(float));
        buf_hc_post_.alloc(hc * sizeof(float));
        buf_hc_comb_.alloc(hc * hc * sizeof(float));
        buf_hc_mixes_.alloc((2 + hc) * hc * sizeof(float));
        buf_hc_input_.alloc((size_t)hc * dim * sizeof(float));
        buf_active_expert_ptrs_.alloc(32 * sizeof(void*));

        // Speculative decoding and parallel prefill batch buffers (M=1024 capacity, hardware Tensor Core prefill)
        int max_batch_m = 1024;
        buf_input_tokens_batch_.alloc(max_batch_m * sizeof(int32_t));
        buf_input_pos_batch_.alloc(max_batch_m * sizeof(int32_t));
        buf_hidden_batch_.alloc(max_batch_m * dim * sizeof(__nv_bfloat16));
        buf_hidden2_batch_.alloc(max_batch_m * dim * sizeof(__nv_bfloat16));
        int max_act_k = (int)std::max({(size_t)dim, (size_t)max_inter, (size_t)18560, (size_t)10304, (size_t)10240, (size_t)6144});
        int max_act_m = 1024;
        buf_activation_fp4_.alloc((size_t)max_act_m * (max_act_k / 2));
        buf_activation_scale_.alloc((size_t)max_act_m * (max_act_k / 32));
        buf_activation_fp8_.alloc((size_t)max_act_m * max_act_k);
        buf_activation_fp8_side_.alloc((size_t)max_act_m * max_act_k);
        buf_q_batch_.alloc(max_batch_m * std::max({(size_t)18560, (size_t)10304, (size_t)10240, (size_t)(2 * n_heads * head_dim_val)}) * sizeof(__nv_bfloat16));
        buf_gate_batch_.alloc(max_batch_m * max_inter * sizeof(__nv_bfloat16));
        buf_up_batch_.alloc(max_batch_m * max_inter * sizeof(__nv_bfloat16));
        buf_attn_out_batch_.alloc(max_batch_m * std::max((size_t)8192, (size_t)n_heads * head_dim_val) * sizeof(__nv_bfloat16));
        buf_linear_a_batch_.alloc(max_batch_m * 64 * sizeof(__nv_bfloat16));
        buf_linear_b_batch_.alloc(max_batch_m * 64 * sizeof(__nv_bfloat16));
        buf_logits_batch_.alloc((size_t)std::min(max_batch_m, 8) * cfg_.vocab_size * sizeof(float));
        buf_argmax_out_batch_.alloc(max_batch_m * sizeof(int32_t));
        buf_tree_ancestor_mask_.alloc(8 * sizeof(uint64_t));
        buf_tree_parent_indices_.alloc(8 * sizeof(int32_t));
        buf_tree_winning_slots_.alloc(8 * sizeof(int32_t));

        if (cfg_.n_routed_experts > 0) {
            int max_routed_exp = std::max({cfg_.n_routed_experts, cfg_.max_n_routed_experts(), 512});
            int max_top_k = std::max({cfg_.num_experts_per_tok, cfg_.max_num_experts_per_tok(), 16});
            buf_scores_bf16_batch_.alloc(max_batch_m * max_routed_exp * sizeof(__nv_bfloat16));
            buf_topk_idx_batch_.alloc(max_batch_m * max_top_k * sizeof(int32_t));
            buf_topk_vals_batch_.alloc(max_batch_m * max_top_k * sizeof(float));
            if (!topk_ids_batch_host_) {
                CUDA_CHECK(cudaMallocHost(&topk_ids_batch_host_, max_batch_m * max_top_k * sizeof(int32_t)));
            }
            if (!topk_weights_batch_host_) {
                CUDA_CHECK(cudaMallocHost(&topk_weights_batch_host_, max_batch_m * max_top_k * sizeof(float)));
            }
            int moe_inter_val = cfg_.moe_intermediate_size > 0 ? cfg_.moe_intermediate_size : 2560;
            int latent_size = cfg_.moe_latent_size > 0 ? cfg_.moe_latent_size : 1024;
            buf_moe_offsets_.alloc(513 * sizeof(int32_t));
            buf_moe_token_indices_.alloc((size_t)max_batch_m * max_top_k * sizeof(int32_t));
            buf_moe_token_slots_.alloc((size_t)max_batch_m * max_top_k * sizeof(int32_t));
            buf_moe_gate_batch_.alloc((size_t)max_batch_m * max_top_k * moe_inter_val * sizeof(__nv_bfloat16));
            buf_latent_down_batch_.alloc((size_t)max_batch_m * latent_size * sizeof(__nv_bfloat16));
            buf_latent_accum_batch_.alloc((size_t)max_batch_m * latent_size * sizeof(__nv_bfloat16));
            buf_layer_expert_ptrs_.alloc(max_routed_exp * sizeof(void*));
            int n_moe_layers = 0;
            for (auto t : cfg_.layer_types) if (t == 2) n_moe_layers++;
            if (n_moe_layers == 0) n_moe_layers = cfg_.is_nemotron_bolt() ? 40 : 23;
            buf_all_expert_ptrs_.alloc((size_t)n_moe_layers * max_routed_exp * sizeof(void*));
        }

        if (cfg_.is_nemotron()) {
            size_t conv_channels = cfg_.is_nemotron_bolt() ? 10240 : (cfg_.is_nemotron() ? 6144 : 10240);
            size_t temp_channels = cfg_.is_nemotron_bolt() ? 8192 : 4096;
            buf_mamba_conv_batch_.alloc((size_t)max_batch_m * conv_channels * sizeof(__nv_bfloat16));
            buf_mamba_temp_batch_.alloc((size_t)max_batch_m * temp_channels * sizeof(float));
        }

        if (cfg_.is_nemotron_bolt() || cfg_.is_nemotron()) {
            int max_slots = cfg_.is_nemotron_bolt() ? 256 : 128;
            size_t exp_block_sz = expert_loader_.expert_block_size() > 0 ? (size_t)expert_loader_.expert_block_size()
                                : (cfg_.is_nemotron_bolt() ? (size_t)1351680 : (size_t)5615616);
            buf_dram_moe_scratch_.alloc(max_slots * exp_block_sz);
            size_t staging_sz = (size_t)max_slots * exp_block_sz;
            if (!pinned_host_moe_scratch_ || pinned_host_moe_scratch_size_ < staging_sz) {
                if (pinned_host_moe_scratch_) cudaFreeHost(pinned_host_moe_scratch_);
                CUDA_CHECK(cudaMallocHost(&pinned_host_moe_scratch_, staging_sz));
                pinned_host_moe_scratch_size_ = staging_sz;
            }
        }

        // Compressor working buffers
        // Max projection output size: coff=2 for ratio=4, head_dim=512 -> 1024
        int max_comp_dim = 2 * head_dim_val;  // coff=2
        buf_comp_proj_.alloc(max_comp_dim * sizeof(float));  // for wkv or wgate output
        buf_comp_out_.alloc(max_comp_dim * sizeof(float));   // for pooling output
        buf_comp_bf16_.alloc(head_dim_val * sizeof(__nv_bfloat16));  // compressed entry in BF16

        // Indexer buffers
        int max_comp_entries = cfg_.max_compressed_entries > 0 ? cfg_.max_compressed_entries : 2048;
        buf_indexer_q_.alloc(64 * 128 * sizeof(__nv_bfloat16));
        buf_indexer_weights_bf16_.alloc(64 * sizeof(__nv_bfloat16));
        buf_indexer_weights_f32_.alloc(64 * sizeof(float));
        buf_indexer_scores_.alloc(max_comp_entries * sizeof(float));
        buf_indexer_mask_.alloc(max_comp_entries * sizeof(uint8_t));
        buf_indexer_proj_kv_.alloc(256 * sizeof(float));
        buf_indexer_proj_gate_.alloc(256 * sizeof(float));

        buf_argmax_out_.alloc(sizeof(int32_t));
        buf_track_flag_.alloc(sizeof(int32_t));
        CUDA_CHECK(cudaMemset(buf_track_flag_.data, 0, sizeof(int32_t)));
        d_expert_counts_.alloc((size_t)cfg_.num_hidden_layers * std::max(cfg_.n_routed_experts, 256) * sizeof(uint32_t));
        CUDA_CHECK(cudaMemset(d_expert_counts_.data, 0, d_expert_counts_.size_bytes));
        
        size_t step_topk_sz = (size_t)cfg_.num_hidden_layers * std::max(cfg_.num_experts_per_tok, 6) * sizeof(int32_t);
        d_step_topk_.alloc(step_topk_sz);
        CUDA_CHECK(cudaMemset(d_step_topk_.data, 0, d_step_topk_.size_bytes));
        if (step_topk_host_) cudaFreeHost(step_topk_host_);
        CUDA_CHECK(cudaMallocHost(&step_topk_host_, 2 * step_topk_sz));
        std::memset(step_topk_host_, 0, 2 * step_topk_sz);

        router_indices_host_.resize(cfg_.n_routed_experts);
        logits_host_.resize(cfg_.vocab_size);
        probs_host_.resize(cfg_.vocab_size);

        if (logits_bf16_host_) cudaFreeHost(logits_bf16_host_);
        CUDA_CHECK(cudaMallocHost(&logits_bf16_host_, cfg_.vocab_size * sizeof(__nv_bfloat16)));
    }

    // ── cuBLAS GEMM helper (BF16) ──────────────────────────────────────────
    // C = A @ B.T  where A is [M, K], B is [N, K] (row-major), C is [M, N]
    // cuBLAS uses column-major, so we compute B.T @ A.T = (A @ B.T).T

    void gemm_bf16(
        __nv_bfloat16* C, int M, int N, int K,
        const __nv_bfloat16* A,  // [M, K]
        const __nv_bfloat16* B,  // [N, K] — stored as weight[out, in]
        float alpha = 1.0f, float beta = 0.0f,
        cudaStream_t stream = nullptr,
        cublasHandle_t handle = nullptr)
    {
        if (!stream) stream = main_stream_;
        if (!handle) handle = cublas_handle_;
        if (M == 1) {
            gemv_bf16_out_bf16_cuda(C, B, A, N, K, stream);
            return;
        }
        if (M <= 8 && alpha == 1.0f && beta == 0.0f && (K % 8 == 0)) {
            gemv_bf16_out_bf16_batch_cuda(C, B, A, N, K, M, stream);
            return;
        }
        if (handle == cublas_handle_ && stream != main_stream_) cublasSetStream(handle, stream);
        cudaError_t pre_err = cudaGetLastError();
        if (pre_err != cudaSuccess) {
            LOG_ERROR("gemm_bf16 pre-launch CUDA error: %s (%d)", cudaGetErrorString(pre_err), (int)pre_err);
        }
        cublasStatus_t stat_code = cublasGemmEx(
            handle,
            CUBLAS_OP_T,    // B transposed (K x N -> N x K)
            CUBLAS_OP_N,    // A not transposed
            N, M, K,        // m=N, n=M, k=K in cuBLAS terms
            &alpha,
            B, CUDA_R_16BF, K,    // lda = K
            A, CUDA_R_16BF, K,    // ldb = K
            &beta,
            C, CUDA_R_16BF, N,    // ldc = N
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);
        if (stat_code != CUBLAS_STATUS_SUCCESS) {
            LOG_ERROR("gemm_bf16 failed: stat=%d M=%d N=%d K=%d A=%p B=%p C=%p", (int)stat_code, M, N, K, (const void*)A, (const void*)B, (void*)C);
            CUBLAS_CHECK(stat_code);
        }
        if (handle == cublas_handle_ && stream != main_stream_) cublasSetStream(handle, main_stream_);
    }

    void gemm_bf16_f32(
        float* C, int M, int N, int K,
        const __nv_bfloat16* A,  // [M, K]
        const __nv_bfloat16* B,  // [N, K] — stored as weight[out, in]
        float alpha = 1.0f, float beta = 0.0f)
    {
        if (M == 1) {
            gemv_bf16_cuda(C, B, A, N, K, main_stream_);
            return;
        }
        if (alpha == 1.0f && beta == 0.0f && (K % 8 == 0)) {
            for (int m = 0; m < M; m += 32) {
                int cur_m = std::min(32, M - m);
                gemv_bf16_batch_cuda(C + (size_t)m * N, B, A + (size_t)m * K, N, K, cur_m, main_stream_);
            }
            return;
        }
        CUBLAS_CHECK(cublasGemmEx(
            cublas_handle_,
            CUBLAS_OP_T,    // B transposed (K x N -> N x K)
            CUBLAS_OP_N,    // A not transposed
            N, M, K,        // m=N, n=M, k=K in cuBLAS terms
            &alpha,
            B, CUDA_R_16BF, K,    // lda = K
            A, CUDA_R_16BF, K,    // ldb = K
            &beta,
            C, CUDA_R_32F, N,     // ldc = N
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT));
    }

    void gemm_fp8_scalar_scale_cublaslt(
        __nv_bfloat16* out,
        const __nv_bfloat16* in_hidden,
        const uint8_t* weight_fp8,
        float scale_val,
        int M, int N, int K,
        float beta = 0.0f,
        cudaStream_t stream = nullptr)
    {
        if (!stream) stream = main_stream_;
        if (M <= 0 || N <= 0 || K <= 0) return;

        if (is_blackwell_fp8_active_ && cublaslt_handle_ && M >= 2) {
            uint8_t* a_fp8 = (stream == side_stream_) ? buf_activation_fp8_side_.u8() : buf_activation_fp8_.u8();
            quantize_bf16_to_fp8_e4m3_cuda(a_fp8, in_hidden, M * K, stream);

            cublasLtHandle_t lt_handle = (stream == side_stream_) ? cublaslt_side_handle_ : cublaslt_handle_;
            void* ws = (stream == side_stream_) ? cublaslt_workspace_side_ : cublaslt_workspace_;

            CublasLtCachedPlan* plan = nullptr;
            for (int i = 0; i < NUM_CUBLASLT_CACHED_PLANS; i++) {
                if (cublaslt_plans_[i].M == M && cublaslt_plans_[i].N == N && cublaslt_plans_[i].K == K) {
                    plan = &cublaslt_plans_[i];
                    break;
                }
            }
            if (!plan) {
                plan = &cublaslt_plans_[cublaslt_plan_next_evict_ % NUM_CUBLASLT_CACHED_PLANS];
                cublaslt_plan_next_evict_++;
                plan->destroy();
                plan->M = M;
                plan->N = N;
                plan->K = K;

                CUBLAS_CHECK(cublasLtMatmulDescCreate(&plan->op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
                cublasOperation_t transa = CUBLAS_OP_T;
                cublasOperation_t transb = CUBLAS_OP_N;
                CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
                CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

                CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan->a_desc, CUDA_R_8F_E4M3, K, N, K));
                CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan->b_desc, CUDA_R_8F_E4M3, K, M, K));
                CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan->c_desc, CUDA_R_16BF, N, M, N));

                CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&plan->pref));
                CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(plan->pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &cublaslt_workspace_size_, sizeof(cublaslt_workspace_size_)));

                int returnedResults = 0;
                CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(lt_handle, plan->op_desc, plan->a_desc, plan->b_desc, plan->c_desc, plan->c_desc, plan->pref, 1, &plan->heuristic, &returnedResults));
                plan->has_heuristic = (returnedResults > 0);
                LOG_INFO("[cuBLASLt FP8 Plan Created] M=%d N=%d K=%d has_heuristic=%d (ret=%d)",
                         M, N, K, (int)plan->has_heuristic, returnedResults);
            }

            float alpha = scale_val;
            if (plan->has_heuristic) {
                CUBLAS_CHECK(cublasLtMatmul(
                    lt_handle,
                    plan->op_desc,
                    &alpha,
                    weight_fp8, plan->a_desc,
                    a_fp8, plan->b_desc,
                    &beta,
                    out, plan->c_desc,
                    out, plan->c_desc,
                    &plan->heuristic.algo,
                    ws, cublaslt_workspace_size_,
                    stream));
            } else {
                if (beta == 0.0f) {
                    gemm_fp8_scalar_scale_batch_cuda(out, in_hidden, weight_fp8, scale_val, M, N, K, stream);
                } else {
                    gemm_fp8_scalar_scale_residual_batch_cuda(out, in_hidden, weight_fp8, scale_val, M, N, K, stream);
                }
            }
        } else {
            if (beta == 0.0f) {
                gemm_fp8_scalar_scale_batch_cuda(out, in_hidden, weight_fp8, scale_val, M, N, K, stream);
            } else {
                gemm_fp8_scalar_scale_residual_batch_cuda(out, in_hidden, weight_fp8, scale_val, M, N, K, stream);
            }
        }
    }

    // ── Dequant + GEMM: dequantize FP8 weight to BF16 temp, then GEMM ──────

    void gemm_fp8_dequant(
        __nv_bfloat16* C, int M, int N, int K,
        const __nv_bfloat16* A,      // [M, K] BF16 input
        const uint8_t* weight,       // [N, K] FP8 E4M3
        const uint8_t* scale,        // [ceil(N/128), ceil(K/128)] E8M0
        int block_size = 128,
        cudaStream_t stream = nullptr)
    {
        if (!stream) stream = main_stream_;
        if (M == 1) {
            // Fused gemv for decoding
            gemv_fp8_cuda(C, A, weight, scale, N, K, block_size, stream);
        } else {
            // Dequantize weight to BF16 in buf_dequant_
            fp8_dequant_cuda(buf_dequant_.bf16(), weight, scale, N, K, block_size, stream);
            // GEMM with dequantized weight
            gemm_bf16(C, M, N, K, A, buf_dequant_.bf16());
        }
    }

    // ── Dequant + GEMM: dequantize INT4 weight to BF16 temp, then GEMM ──────

    void gemm_int4_dequant(
        __nv_bfloat16* C, int M, int N, int K,
        const __nv_bfloat16* A,              // [M, K] BF16 input
        const uint8_t* weight,               // [N, K/2] packed INT4
        const __nv_bfloat16* scale,          // [N, K/32] BF16 block scales
        int block_size,
        cudaStream_t stream,
        float beta,
        GPUTensor* dequant_buf,
        GPUTensor* dequant_buf_2,
        cublasHandle_t handle = nullptr)
    {
        if (!stream) stream = main_stream_;
        if (!dequant_buf) dequant_buf = &buf_dequant_;
        if (!handle) handle = cublas_handle_;
        if (M == 1) {
            gemv_int4_cuda(C, A, weight, scale, N, K, stream);
            return;
        }
        if (M <= 8) {
            gemm_int4_batch_cuda(C, A, weight, scale, N, K, M, stream);
            return;
        }

        if (is_blackwell_fp8_active_ && cublaslt_handle_ && M >= 32) {
            uint8_t* sub_buf[2];
            size_t tile_bytes;
            if (dequant_buf_2 && dequant_buf_2 != dequant_buf) {
                sub_buf[0] = dequant_buf->u8();
                sub_buf[1] = dequant_buf_2->u8();
                tile_bytes = std::min(dequant_buf->size_bytes, dequant_buf_2->size_bytes);
            } else {
                size_t half_bytes = dequant_buf->size_bytes / 2;
                sub_buf[0] = dequant_buf->u8();
                sub_buf[1] = dequant_buf->u8() + half_bytes;
                tile_bytes = half_bytes;
            }

            int max_rows = (int)(tile_bytes / (sizeof(uint8_t) * (size_t)K));
            int tile_n = (max_rows / 128) * 128;
            if (tile_n <= 0) tile_n = (max_rows / 32) * 32;
            if (tile_n <= 0) tile_n = std::max(32, max_rows);

            int stream_idx = (stream == side_stream_) ? 1 : 0;
            cudaStream_t deq_stream = (stream == side_stream_) ? dequant_stream_side_ : dequant_stream_main_;
            if (!deq_stream) deq_stream = stream;
            cublasLtHandle_t lt_handle = (stream == side_stream_) ? cublaslt_side_handle_ : cublaslt_handle_;
            void* ws = (stream == side_stream_) ? cublaslt_workspace_side_ : cublaslt_workspace_;
            uint8_t* a_fp8 = (stream == side_stream_) ? buf_activation_fp8_side_.u8() : buf_activation_fp8_.u8();

            quantize_bf16_to_fp8_e4m3_cuda(a_fp8, A, M * K, stream);

            int blocks_per_row = K / 32;
            int num_tiles = (N + tile_n - 1) / tile_n;
            const float alpha = 1.0f;

            cublasLtMatmulDesc_t operationDesc = NULL;
            CUBLAS_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
            cublasOperation_t transa = CUBLAS_OP_T;
            cublasOperation_t transb = CUBLAS_OP_N;
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

            cublasLtMatmulPreference_t preference = NULL;
            CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
            CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &cublaslt_workspace_size_, sizeof(cublaslt_workspace_size_)));

            cublasLtMatrixLayout_t Bdesc = NULL;
            CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, K, M, K));

            CUDA_CHECK(cudaEventRecord(gemm_done_event_[stream_idx][0], stream));
            CUDA_CHECK(cudaEventRecord(gemm_done_event_[stream_idx][1], stream));

            for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
                int n_start = tile_idx * tile_n;
                int cur_n = std::min(tile_n, N - n_start);
                size_t start_block = (size_t)n_start * blocks_per_row;
                int buf_idx = tile_idx % 2;

                CUDA_CHECK(cudaStreamWaitEvent(deq_stream, gemm_done_event_[stream_idx][buf_idx], 0));
                dequant_int4_to_fp8_block_cuda(sub_buf[buf_idx], weight + start_block * 16, scale + start_block, cur_n, K, block_size, deq_stream);
                CUDA_CHECK(cudaEventRecord(dequant_ready_event_[stream_idx][buf_idx], deq_stream));

                cublasLtMatrixLayout_t Adesc = NULL, Cdesc = NULL;
                CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, K, cur_n, K));
                CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, cur_n, M, N));

                cublasLtMatmulHeuristicResult_t heuristicResult;
                int returnedResults = 0;
                CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(lt_handle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

                CUDA_CHECK(cudaStreamWaitEvent(stream, dequant_ready_event_[stream_idx][buf_idx], 0));
                if (returnedResults > 0) {
                    CUBLAS_CHECK(cublasLtMatmul(
                        lt_handle,
                        operationDesc,
                        &alpha,
                        sub_buf[buf_idx], Adesc,
                        a_fp8, Bdesc,
                        &beta,
                        C + n_start, Cdesc,
                        C + n_start, Cdesc,
                        &heuristicResult.algo,
                        ws, cublaslt_workspace_size_,
                        stream));
                }
                CUDA_CHECK(cudaEventRecord(gemm_done_event_[stream_idx][buf_idx], stream));

                CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
                CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
            }

            CUBLAS_CHECK(cublasLtMatmulDescDestroy(operationDesc));
            CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
            CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
            return;
        }

        __nv_bfloat16* sub_buf[2];
        size_t tile_bytes;
        if (dequant_buf_2 && dequant_buf_2 != dequant_buf) {
            sub_buf[0] = dequant_buf->bf16();
            sub_buf[1] = dequant_buf_2->bf16();
            tile_bytes = std::min(dequant_buf->size_bytes, dequant_buf_2->size_bytes);
        } else {
            size_t half_bytes = dequant_buf->size_bytes / 2;
            sub_buf[0] = dequant_buf->bf16();
            sub_buf[1] = dequant_buf->bf16() + (half_bytes / sizeof(__nv_bfloat16));
            tile_bytes = half_bytes;
        }

        int max_rows = (int)(tile_bytes / (sizeof(__nv_bfloat16) * (size_t)K));
        int tile_n = (max_rows / 128) * 128;
        if (tile_n <= 0) tile_n = (max_rows / 32) * 32;
        if (tile_n <= 0) tile_n = std::max(32, max_rows);

        int stream_idx = (stream == side_stream_) ? 1 : 0;
        cudaStream_t deq_stream = (stream == side_stream_) ? dequant_stream_side_ : dequant_stream_main_;
        if (!deq_stream) deq_stream = stream;

        int blocks_per_row = K / 32;
        int num_tiles = (N + tile_n - 1) / tile_n;
        const float alpha = 1.0f;

        if (handle == cublas_handle_ && stream != main_stream_) cublasSetStream(handle, stream);

        CUDA_CHECK(cudaEventRecord(gemm_done_event_[stream_idx][0], stream));
        CUDA_CHECK(cudaEventRecord(gemm_done_event_[stream_idx][1], stream));

        for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
            int n_start = tile_idx * tile_n;
            int cur_n = std::min(tile_n, N - n_start);
            size_t start_block = (size_t)n_start * blocks_per_row;
            int buf_idx = tile_idx % 2;

            CUDA_CHECK(cudaStreamWaitEvent(deq_stream, gemm_done_event_[stream_idx][buf_idx], 0));
            dequant_int4_block_cuda(sub_buf[buf_idx], weight + start_block * 16, scale + start_block, cur_n, K, block_size, deq_stream);
            CUDA_CHECK(cudaEventRecord(dequant_ready_event_[stream_idx][buf_idx], deq_stream));

            CUDA_CHECK(cudaStreamWaitEvent(stream, dequant_ready_event_[stream_idx][buf_idx], 0));
            CUBLAS_CHECK(cublasGemmEx(
                handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                cur_n, M, K,
                &alpha,
                sub_buf[buf_idx], CUDA_R_16BF, K,
                A, CUDA_R_16BF, K,
                &beta,
                C + n_start, CUDA_R_16BF, N,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT));
            CUDA_CHECK(cudaEventRecord(gemm_done_event_[stream_idx][buf_idx], stream));
        }

        if (handle == cublas_handle_ && stream != main_stream_) cublasSetStream(handle, main_stream_);
    }

    void gemm_int4_dequant(
        __nv_bfloat16* C, int M, int N, int K,
        const __nv_bfloat16* A,              // [M, K] BF16 input
        const uint8_t* weight,               // [N, K/2] packed INT4
        const __nv_bfloat16* scale,          // [N, K/32] BF16 block scales
        int block_size = 32,
        cudaStream_t stream = nullptr,
        float beta = 0.0f,
        GPUTensor* dequant_buf = nullptr,
        cublasHandle_t handle = nullptr)
    {
        gemm_int4_dequant(C, M, N, K, A, weight, scale, block_size, stream, beta, dequant_buf, nullptr, handle);
    }

    void gemm_int3_dequant(
        __nv_bfloat16* C, int M, int N, int K,
        const __nv_bfloat16* A,              // [M, K] BF16 input
        const uint8_t* weight,               // packed INT3
        const __nv_bfloat16* scale,          // [N, K/32] BF16 block scales
        int block_size,
        cudaStream_t stream,
        float beta,
        GPUTensor* dequant_buf,
        GPUTensor* dequant_buf_2,
        cublasHandle_t handle = nullptr)
    {
        if (!stream) stream = main_stream_;
        if (!dequant_buf) dequant_buf = &buf_dequant_;
        if (!handle) handle = cublas_handle_;
        if (M == 1) {
            gemv_int3_cuda(C, A, weight, scale, N, K, stream);
            return;
        }
        if (M <= 8) {
            gemm_int3_batch_cuda(C, A, weight, scale, N, K, M, stream);
            return;
        }

        if (is_blackwell_fp8_active_ && cublaslt_handle_ && M >= 32) {
            uint8_t* sub_buf[2];
            size_t tile_bytes;
            if (dequant_buf_2 && dequant_buf_2 != dequant_buf) {
                sub_buf[0] = dequant_buf->u8();
                sub_buf[1] = dequant_buf_2->u8();
                tile_bytes = std::min(dequant_buf->size_bytes, dequant_buf_2->size_bytes);
            } else {
                size_t half_bytes = dequant_buf->size_bytes / 2;
                sub_buf[0] = dequant_buf->u8();
                sub_buf[1] = dequant_buf->u8() + half_bytes;
                tile_bytes = half_bytes;
            }

            int max_rows = (int)(tile_bytes / (sizeof(uint8_t) * (size_t)K));
            int tile_n = (max_rows / 128) * 128;
            if (tile_n <= 0) tile_n = (max_rows / 32) * 32;
            if (tile_n <= 0) tile_n = std::max(32, max_rows);

            int stream_idx = (stream == side_stream_) ? 1 : 0;
            cudaStream_t deq_stream = (stream == side_stream_) ? dequant_stream_side_ : dequant_stream_main_;
            if (!deq_stream) deq_stream = stream;
            cublasLtHandle_t lt_handle = (stream == side_stream_) ? cublaslt_side_handle_ : cublaslt_handle_;
            void* ws = (stream == side_stream_) ? cublaslt_workspace_side_ : cublaslt_workspace_;
            uint8_t* a_fp8 = (stream == side_stream_) ? buf_activation_fp8_side_.u8() : buf_activation_fp8_.u8();

            quantize_bf16_to_fp8_e4m3_cuda(a_fp8, A, M * K, stream);

            int blocks_per_row = K / 32;
            int num_tiles = (N + tile_n - 1) / tile_n;
            const float alpha = 1.0f;

            cublasLtMatmulDesc_t operationDesc = NULL;
            CUBLAS_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
            cublasOperation_t transa = CUBLAS_OP_T;
            cublasOperation_t transb = CUBLAS_OP_N;
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

            cublasLtMatmulPreference_t preference = NULL;
            CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
            CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &cublaslt_workspace_size_, sizeof(cublaslt_workspace_size_)));

            cublasLtMatrixLayout_t Bdesc = NULL;
            CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, K, M, K));

            CUDA_CHECK(cudaEventRecord(gemm_done_event_[stream_idx][0], stream));
            CUDA_CHECK(cudaEventRecord(gemm_done_event_[stream_idx][1], stream));

            for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
                int n_start = tile_idx * tile_n;
                int cur_n = std::min(tile_n, N - n_start);
                size_t start_block = (size_t)n_start * blocks_per_row;
                int buf_idx = tile_idx % 2;

                CUDA_CHECK(cudaStreamWaitEvent(deq_stream, gemm_done_event_[stream_idx][buf_idx], 0));
                dequant_int3_to_fp8_block_cuda(sub_buf[buf_idx], weight + start_block * 12, scale + start_block, cur_n, K, block_size, deq_stream);
                CUDA_CHECK(cudaEventRecord(dequant_ready_event_[stream_idx][buf_idx], deq_stream));

                cublasLtMatrixLayout_t Adesc = NULL, Cdesc = NULL;
                CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, K, cur_n, K));
                CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, cur_n, M, N));

                cublasLtMatmulHeuristicResult_t heuristicResult;
                int returnedResults = 0;
                CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(lt_handle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

                CUDA_CHECK(cudaStreamWaitEvent(stream, dequant_ready_event_[stream_idx][buf_idx], 0));
                if (returnedResults > 0) {
                    CUBLAS_CHECK(cublasLtMatmul(
                        lt_handle,
                        operationDesc,
                        &alpha,
                        sub_buf[buf_idx], Adesc,
                        a_fp8, Bdesc,
                        &beta,
                        C + n_start, Cdesc,
                        C + n_start, Cdesc,
                        &heuristicResult.algo,
                        ws, cublaslt_workspace_size_,
                        stream));
                }
                CUDA_CHECK(cudaEventRecord(gemm_done_event_[stream_idx][buf_idx], stream));

                CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
                CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
            }

            CUBLAS_CHECK(cublasLtMatmulDescDestroy(operationDesc));
            CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
            CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
            return;
        }

        __nv_bfloat16* sub_buf[2];
        size_t tile_bytes;
        if (dequant_buf_2 && dequant_buf_2 != dequant_buf) {
            sub_buf[0] = dequant_buf->bf16();
            sub_buf[1] = dequant_buf_2->bf16();
            tile_bytes = std::min(dequant_buf->size_bytes, dequant_buf_2->size_bytes);
        } else {
            size_t half_bytes = dequant_buf->size_bytes / 2;
            sub_buf[0] = dequant_buf->bf16();
            sub_buf[1] = dequant_buf->bf16() + (half_bytes / sizeof(__nv_bfloat16));
            tile_bytes = half_bytes;
        }

        int max_rows = (int)(tile_bytes / (sizeof(__nv_bfloat16) * (size_t)K));
        int tile_n = (max_rows / 128) * 128;
        if (tile_n <= 0) tile_n = (max_rows / 32) * 32;
        if (tile_n <= 0) tile_n = std::max(32, max_rows);

        int stream_idx = (stream == side_stream_) ? 1 : 0;
        cudaStream_t deq_stream = (stream == side_stream_) ? dequant_stream_side_ : dequant_stream_main_;
        if (!deq_stream) deq_stream = stream;

        int blocks_per_row = K / 32;
        int num_tiles = (N + tile_n - 1) / tile_n;
        const float alpha = 1.0f;

        if (handle == cublas_handle_ && stream != main_stream_) cublasSetStream(handle, stream);

        CUDA_CHECK(cudaEventRecord(gemm_done_event_[stream_idx][0], stream));
        CUDA_CHECK(cudaEventRecord(gemm_done_event_[stream_idx][1], stream));

        for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
            int n_start = tile_idx * tile_n;
            int cur_n = std::min(tile_n, N - n_start);
            size_t start_block = (size_t)n_start * blocks_per_row;
            int buf_idx = tile_idx % 2;

            CUDA_CHECK(cudaStreamWaitEvent(deq_stream, gemm_done_event_[stream_idx][buf_idx], 0));
            dequant_int3_block_cuda(sub_buf[buf_idx], weight + start_block * 12, scale + start_block, cur_n, K, block_size, deq_stream);
            CUDA_CHECK(cudaEventRecord(dequant_ready_event_[stream_idx][buf_idx], deq_stream));

            CUDA_CHECK(cudaStreamWaitEvent(stream, dequant_ready_event_[stream_idx][buf_idx], 0));
            CUBLAS_CHECK(cublasGemmEx(
                handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                cur_n, M, K,
                &alpha,
                sub_buf[buf_idx], CUDA_R_16BF, K,
                A, CUDA_R_16BF, K,
                &beta,
                C + n_start, CUDA_R_16BF, N,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT));
            CUDA_CHECK(cudaEventRecord(gemm_done_event_[stream_idx][buf_idx], stream));
        }

        if (handle == cublas_handle_ && stream != main_stream_) cublasSetStream(handle, main_stream_);
    }

    void gemm_int3_dequant(
        __nv_bfloat16* C, int M, int N, int K,
        const __nv_bfloat16* A,              // [M, K] BF16 input
        const uint8_t* weight,               // packed INT3
        const __nv_bfloat16* scale,          // [N, K/32] BF16 block scales
        int block_size = 32,
        cudaStream_t stream = nullptr,
        float beta = 0.0f,
        GPUTensor* dequant_buf = nullptr,
        cublasHandle_t handle = nullptr)
    {
        gemm_int3_dequant(C, M, N, K, A, weight, scale, block_size, stream, beta, dequant_buf, nullptr, handle);
    }

    // ── Dequant + GEMM for FP4 experts ──────────────────────────────────────

    void gemm_fp4_dequant(
        __nv_bfloat16* C, int M, int N, int K_logical,
        const __nv_bfloat16* A,      // [M, K_logical] BF16
        const uint8_t* weight,       // [N, K_logical/2] packed FP4
        const uint8_t* scale,        // [N, K_logical/32] E8M0
        int scale_cols, cudaStream_t stream = nullptr)
    {
        if (!stream) stream = main_stream_;
        int K_packed = K_logical / 2;
        fp4_dequant_cuda(buf_dequant_.bf16(), weight, scale, N, K_packed, scale_cols, stream);
        gemm_bf16(C, M, N, K_logical, A, buf_dequant_.bf16());
    }

    void gemm_int2_dequant(
        __nv_bfloat16* C, int M, int N, int K_logical,
        const __nv_bfloat16* A,
        const uint8_t* weight,       // [N, K_logical/4] packed INT2
        const __nv_bfloat16* scale_min, // [N, K_logical/block_size, 2] BF16
        int block_size, cudaStream_t stream)
    {
        int K_packed = K_logical / 4;
        if (M == 1) {
            // Fused gemv for decoding
            gemv_int2_cuda(C, A, weight, scale_min, N, K_packed, block_size, stream);
        } else {
            // Prefill: dequantize then gemm
            for (int m = 0; m < M; m++) {
                gemv_int2_cuda(C + m * N, A + m * K_logical, weight, scale_min, N, K_packed, block_size, stream);
            }
        }
    }

    void gemm_iq2_xxs_dequant(
        __nv_bfloat16* C, int M, int N, int K,
        const __nv_bfloat16* A,
        const block_iq2_xxs* weight,
        cudaStream_t stream = nullptr)
    {
        if (!stream) stream = main_stream_;
        if (M == 1) {
            gemv_iq2_xxs_cuda(C, A, weight, N, K, stream);
        } else {
            for (int m = 0; m < M; m++) {
                gemv_iq2_xxs_cuda(C + m * N, A + m * K, weight, N, K, stream);
            }
        }
    }

    void gemm_q2_k_dequant(
        __nv_bfloat16* C, int M, int N, int K,
        const __nv_bfloat16* A,
        const block_q2_K* weight,
        cudaStream_t stream = nullptr)
    {
        if (!stream) stream = main_stream_;
        if (M == 1) {
            gemv_q2_k_cuda(C, A, weight, N, K, stream);
        } else {
            for (int m = 0; m < M; m++) {
                gemv_q2_k_cuda(C + m * N, A + m * K, weight, N, K, stream);
            }
        }
    }

    // ── KV Compressor Forward (Device-Driven for CUDA Graph) ────────────────
    // Implements gated pooling compression for CSA (ratio=4, overlap) and
    // HCA (ratio=128, non-overlap) layers following DeepSeek V4 architecture.
    //
    // All state updates, pooling, RMSNorm, RoPE, and counter increments are
    // executed inside GPU kernels, enabling full CUDA Graph capture and replay.

    void forward_compressor_step(int layer_id, const __nv_bfloat16* tok_hidden, const int32_t* d_pos) {
        auto& lw = layers_[layer_id];
        int ratio = cfg_.layer_compress_ratio(layer_id);
        if (ratio <= 0) return;

        int dim = cfg_.hidden_size;
        int head_dim_val = cfg_.head_dim;
        int rope_dim = cfg_.qk_rope_head_dim;
        bool overlap = (ratio == 4);
        int coff = overlap ? 2 : 1;
        int proj_dim = coff * head_dim_val; // 1024 for CSA

        // 1. Attention Compressor Projections: [proj_dim] = wkv @ hidden, wgate @ hidden
        gemv_bf16_cuda(buf_comp_proj_.f32(), lw.comp_wkv.bf16(),
                       tok_hidden, proj_dim, dim, main_stream_);
        gemv_bf16_cuda(buf_comp_out_.f32(), lw.comp_wgate.bf16(),
                       tok_hidden, proj_dim, dim, main_stream_);

        // 2. Indexer Compressor Projections: [256] = indexer_comp_wkv @ hidden, indexer_comp_wgate @ hidden
        const float* idx_proj_kv = nullptr;
        const float* idx_proj_gate = nullptr;
        if (ratio == 4 && lw.indexer_comp_wkv.data) {
            int idx_proj_dim = 256;
            gemv_bf16_cuda(buf_indexer_proj_kv_.f32(), lw.indexer_comp_wkv.bf16(),
                           tok_hidden, idx_proj_dim, dim, main_stream_);
            gemv_bf16_cuda(buf_indexer_proj_gate_.f32(), lw.indexer_comp_wgate.bf16(),
                           tok_hidden, idx_proj_dim, dim, main_stream_);
            idx_proj_kv = buf_indexer_proj_kv_.f32();
            idx_proj_gate = buf_indexer_proj_gate_.f32();
        }

        // 3. Launch 100% device-driven compressor step kernel
        compressor_device_step_cuda(
            d_pos,
            lw.d_comp_kv_count.i32(),
            buf_comp_proj_.f32(),
            buf_comp_out_.f32(),
            lw.comp_kv_state.f32(),
            lw.comp_score_state.f32(),
            lw.comp_ape.f32(),
            lw.comp_norm.bf16(),
            lw.comp_kv_cache.bf16(),
            rope_freqs_compressed_.f32(),
            ratio,
            head_dim_val,
            rope_dim,
            cfg_.rms_norm_eps,
            idx_proj_kv,
            idx_proj_gate,
            lw.indexer_comp_kv_state.f32(),
            lw.indexer_comp_score_state.f32(),
            lw.indexer_comp_ape.f32(),
            lw.indexer_comp_norm.bf16(),
            lw.indexer_comp_kv_cache.bf16(),
            (cfg_.max_seq_len / ratio + 2),
            main_stream_);
    }

    void forward_compressor(int layer_id) {
        forward_compressor_step(layer_id, buf_hidden_.bf16(), buf_input_pos_.i32());
    }

#include "models/model_qwen.inl"

#include "models/model_nemotron.inl"

#include "models/model_deepseek.inl"

    // ── Sample from logits ──────────────────────────────────────────────────

    float current_rep_penalty_ = 1.0f;  // Set per-request by generate()

    int sample_token(float temperature, const std::vector<int>& history, int step = 0, bool is_reasoning = true,
                     int top_k = 1024, float top_p = 0.95f, float min_p = 0.0f, int content_start_idx = 0) {
        return sample_from_logits_ptr(buf_logits_.f32(), temperature, history, step, is_reasoning, top_k, top_p, min_p, content_start_idx);
    }

    void apply_repetition_penalty_to_batch(float* logits_batch, int M, const std::vector<int>& history, bool is_reasoning = true, int content_start_idx = 0) {
        float eff_penalty = current_rep_penalty_;
        if (eff_penalty <= 1.0f && cfg_.is_nemotron()) {
            eff_penalty = 1.05f;
        }
        if (eff_penalty <= 1.0f || history.empty()) return;
        static int32_t* d_rep_hist = nullptr;
        static const int MAX_REP_HIST = 2048;
        if (!d_rep_hist) {
            CUDA_CHECK(cudaMalloc(&d_rep_hist, MAX_REP_HIST * sizeof(int32_t)));
        }
        int hist_len = (int)history.size();
        int start_idx = is_reasoning ? 0 : std::max(0, content_start_idx);
        int available = std::max(0, hist_len - start_idx);
        int n_penalize = std::min(available, MAX_REP_HIST);
        if (n_penalize > 0) {
            thread_local std::vector<int32_t> unique_tokens;
            unique_tokens.clear();
            thread_local std::unordered_set<int32_t> seen;
            seen.clear();
            const int* src_ptr = history.data() + (hist_len - n_penalize);
            for (int i = 0; i < n_penalize; i++) {
                int32_t tok = src_ptr[i];
                if (tok >= 0 && seen.insert(tok).second) {
                    unique_tokens.push_back(tok);
                }
            }
            int n_unique = (int)unique_tokens.size();
            if (n_unique > 0) {
                CUDA_CHECK(cudaMemcpyAsync(d_rep_hist, unique_tokens.data(), n_unique * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                for (int m = 0; m < M; m++) {
                    apply_repetition_penalty_cuda(logits_batch + (size_t)m * cfg_.vocab_size, d_rep_hist, n_unique, eff_penalty, main_stream_);
                }
            }
        }
    }

    int sample_from_logits_ptr(float* logits_ptr, float temperature, const std::vector<int>& history, int step = 0, bool is_reasoning = true,
                               int top_k = 1024, float top_p = 0.95f, float min_p = 0.0f, int content_start_idx = 0) {
        int vocab = cfg_.vocab_size;
        static int32_t* h_sample_pin = nullptr;
        if (!h_sample_pin) {
            CUDA_CHECK(cudaMallocHost(&h_sample_pin, sizeof(int32_t)));
        }

        apply_repetition_penalty_to_batch(logits_ptr, 1, history, is_reasoning, content_start_idx);
        float eff_penalty = current_rep_penalty_;
        if (eff_penalty <= 1.0f && cfg_.is_nemotron()) {
            eff_penalty = 1.05f;
        }
        if (eff_penalty > 1.0f) {
            argmax_cache_valid_ = false;
        }

        if (temperature <= 0.0f) {
            if (logits_ptr != buf_logits_.f32() || !graph_captured_ || !argmax_cache_valid_) {
                argmax_f32_cuda(buf_argmax_out_.i32(), logits_ptr, vocab, main_stream_);
            }
            argmax_cache_valid_ = false;
            CUDA_CHECK(cudaMemcpyAsync(h_sample_pin, buf_argmax_out_.i32(), sizeof(int32_t), cudaMemcpyDeviceToHost, main_stream_));
            CUDA_CHECK(cudaStreamSynchronize(main_stream_));
            return *h_sample_pin;
        }

        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng_);
        float eff_min_p = (min_p > 0.0f) ? min_p : 0.05f;
#if defined(__APPLE__)
        sample_multinomial_f32_cuda(buf_argmax_out_.i32(), logits_ptr, vocab, temperature, r, eff_min_p, main_stream_, top_k, top_p);
#else
        sample_multinomial_f32_cuda(buf_argmax_out_.i32(), logits_ptr, vocab, temperature, r, eff_min_p, main_stream_);
#endif
        CUDA_CHECK(cudaMemcpyAsync(h_sample_pin, buf_argmax_out_.i32(), sizeof(int32_t), cudaMemcpyDeviceToHost, main_stream_));
        CUDA_CHECK(cudaStreamSynchronize(main_stream_));
        return *h_sample_pin;
    }

public:
    bool collect_imatrix(const std::string& dataset_path, const std::string& out_dat_path, int max_tokens = -1) {
        LOG_INFO("=== Starting Importance Matrix Calibration ===");
        LOG_INFO("Calibration dataset: %s", dataset_path.c_str());
        LOG_INFO("Output .dat path:    %s", out_dat_path.c_str());

        std::ifstream file(dataset_path);
        if (!file.is_open()) {
            LOG_ERROR("Cannot open calibration dataset: %s", dataset_path.c_str());
            return false;
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string raw_text = buffer.str();
        file.close();

        LOG_INFO("Dataset text loaded (%zu bytes). Tokenizing...", raw_text.size());
        std::vector<int> tokens = tokenizer_.encode(raw_text);
        if (tokens.empty()) {
            LOG_ERROR("Tokenization produced 0 tokens");
            return false;
        }

        if (max_tokens > 0 && (int)tokens.size() > max_tokens) {
            tokens.resize(max_tokens);
        }
        size_t total_tokens = tokens.size();
        LOG_INFO("Total calibration tokens to stream: %zu", total_tokens);

        int n_layers = cfg_.num_hidden_layers;
        int n_experts = cfg_.n_routed_experts;
        int dim = cfg_.hidden_size;
        int moe_inter = cfg_.moe_intermediate_size;

        size_t gate_bytes = (size_t)n_layers * n_experts * dim * sizeof(float);
        size_t down_bytes = (size_t)n_layers * n_experts * moe_inter * sizeof(float);
        size_t counts_bytes = (size_t)n_layers * n_experts * sizeof(uint32_t);

        d_gate_accum_.alloc(gate_bytes);
        d_down_accum_.alloc(down_bytes);
        d_expert_counts_.alloc(counts_bytes);

        CUDA_CHECK(cudaMemset(d_gate_accum_.data, 0, gate_bytes));
        CUDA_CHECK(cudaMemset(d_down_accum_.data, 0, down_bytes));
        CUDA_CHECK(cudaMemset(d_expert_counts_.data, 0, counts_bytes));

        collect_imatrix_ = true;
        auto start_time = std::chrono::steady_clock::now();

        // Reset HC and cache state
        if (buf_hc_state_.data) CUDA_CHECK(cudaMemset(buf_hc_state_.data, 0, buf_hc_state_.size_bytes));
        if (buf_hc_after_attn_.data) CUDA_CHECK(cudaMemset(buf_hc_after_attn_.data, 0, buf_hc_after_attn_.size_bytes));

        LOG_INFO("Streaming %zu tokens through all %d transformer layers on GPU...", total_tokens, n_layers);
        for (size_t i = 0; i < total_tokens; i++) {
            forward_token_eager(tokens[i], (int)(i % MAX_SEQ_LEN));

            if ((i + 1) % 1000 == 0 || i + 1 == total_tokens) {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                double tps = (i + 1) / (elapsed > 0.0 ? elapsed : 1.0);
                double pct = ((i + 1) * 100.0) / total_tokens;
                printf("\r[Calibrating Experts] %zu/%zu tokens (%.1f%%) | %.1f tok/s | Elapsed: %.1fs",
                       i + 1, total_tokens, pct, tps, elapsed);
                fflush(stdout);
            }
        }
        printf("\n");
        CUDA_CHECK(cudaStreamSynchronize(main_stream_));
        collect_imatrix_ = false;

        LOG_INFO("Copying activation statistics to host...");
        std::vector<float> h_gate(n_layers * n_experts * dim);
        std::vector<float> h_down(n_layers * n_experts * moe_inter);
        std::vector<uint32_t> h_counts(n_layers * n_experts);

        CUDA_CHECK(cudaMemcpy(h_gate.data(), d_gate_accum_.data, gate_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_down.data(), d_down_accum_.data, down_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_counts.data(), d_expert_counts_.data, counts_bytes, cudaMemcpyDeviceToHost));

        LOG_INFO("Normalizing per-expert activations and writing %s...", out_dat_path.c_str());
        std::ofstream out(out_dat_path, std::ios::binary);
        if (!out.is_open()) {
            LOG_ERROR("Cannot open output imatrix file: %s", out_dat_path.c_str());
            return false;
        }

        int32_t num_entries = n_layers * 3;
        out.write((const char*)&num_entries, sizeof(num_entries));

        for (int L = 0; L < n_layers; L++) {
            std::vector<double> layer_gate_sum(dim, 0.0);
            std::vector<double> layer_down_sum(moe_inter, 0.0);
            uint64_t total_layer_calls = 0;

            for (int e = 0; e < n_experts; e++) {
                uint32_t count = h_counts[L * n_experts + e];
                total_layer_calls += count;
                float* g_ptr = &h_gate[(L * n_experts + e) * dim];
                float* d_ptr = &h_down[(L * n_experts + e) * moe_inter];
                for (int j = 0; j < dim; j++) layer_gate_sum[j] += g_ptr[j];
                for (int j = 0; j < moe_inter; j++) layer_down_sum[j] += d_ptr[j];
            }

            double total_denom = std::max(1.0, (double)total_layer_calls);
            std::vector<float> avg_gate(dim);
            std::vector<float> avg_down(moe_inter);
            for (int j = 0; j < dim; j++) avg_gate[j] = std::max(1e-4f, (float)(layer_gate_sum[j] / total_denom));
            for (int j = 0; j < moe_inter; j++) avg_down[j] = std::max(1e-4f, (float)(layer_down_sum[j] / total_denom));

            for (int e = 0; e < n_experts; e++) {
                uint32_t count = h_counts[L * n_experts + e];
                float* g_ptr = &h_gate[(L * n_experts + e) * dim];
                float* d_ptr = &h_down[(L * n_experts + e) * moe_inter];
                if (count >= 10) {
                    float inv_cnt = 1.0f / (float)count;
                    for (int j = 0; j < dim; j++) g_ptr[j] = std::max(1e-4f, g_ptr[j] * inv_cnt);
                    for (int j = 0; j < moe_inter; j++) d_ptr[j] = std::max(1e-4f, d_ptr[j] * inv_cnt);
                } else {
                    for (int j = 0; j < dim; j++) g_ptr[j] = avg_gate[j];
                    for (int j = 0; j < moe_inter; j++) d_ptr[j] = avg_down[j];
                }
            }

            auto write_entry = [&](const std::string& name, const float* data, int32_t nval) {
                int32_t name_len = (int32_t)name.size();
                int32_t ncall = (int32_t)total_tokens;
                out.write((const char*)&name_len, sizeof(name_len));
                out.write(name.data(), name_len);
                out.write((const char*)&ncall, sizeof(ncall));
                out.write((const char*)&nval, sizeof(nval));
                out.write((const char*)data, nval * sizeof(float));
            };

            std::string gate_name = "blk." + std::to_string(L) + ".ffn_gate_exps.weight";
            std::string up_name   = "blk." + std::to_string(L) + ".ffn_up_exps.weight";
            std::string down_name = "blk." + std::to_string(L) + ".ffn_down_exps.weight";

            write_entry(gate_name, &h_gate[L * n_experts * dim], n_experts * dim);
            write_entry(up_name,   &h_gate[L * n_experts * dim], n_experts * dim);
            write_entry(down_name, &h_down[L * n_experts * moe_inter], n_experts * moe_inter);
        }

        out.close();
        LOG_INFO("=== Importance matrix calibration complete: %s ===", out_dat_path.c_str());
        return true;
    }

    // ── Expert frequency tracking ────────────────────────────────────────────

    // ── Expert frequency & specialization tracking ──────────────────────────

    bool is_control_token(int token_id) const {
        if (token_id < 0) return true;
        if (token_id == cfg_.bos_token_id || token_id == cfg_.eos_token_id) return true;

        std::string tok_str = tokenizer_.decode_token_str(token_id);
        if (tok_str.empty()) return true;

        if (tok_str == "<think>" || tok_str == "</think>" ||
            tok_str == "<|im_start|>" || tok_str == "<|im_end|>" ||
            tok_str == "<|end_of_sentence|>" || tok_str == "<｜end of sentence｜>" ||
            tok_str == "<｜User｜>" || tok_str == "<｜Assistant｜>" ||
            tok_str == "<｜begin of sentence｜>" || tok_str == "<|endoftext|>") {
            return true;
        }

        // Catch any special tag formatted like <|...|> or <｜...｜> or <..._...>
        if (tok_str.size() >= 4) {
            if (tok_str.rfind("<|", 0) == 0 || tok_str.rfind("<｜", 0) == 0) return true;
            if (tok_str.front() == '<' && tok_str.back() == '>') return true;
        }
        return false;
    }

    std::mutex expert_profile_mutex_;

    std::vector<uint32_t> get_loader_freq_counts() const {
        if (expert_freq_counts_.empty()) return {};
        if ((int)expert_freq_counts_.size() == cfg_.num_hidden_layers * cfg_.n_routed_experts && cfg_.is_nemotron()) {
            std::vector<uint32_t> sub;
            sub.reserve(40 * cfg_.n_routed_experts);
            for (int l = 0; l < cfg_.num_hidden_layers; l++) {
                if (l < (int)cfg_.layer_types.size() && cfg_.layer_types[l] == 2) {
                    const uint32_t* src = &expert_freq_counts_[(size_t)l * cfg_.n_routed_experts];
                    sub.insert(sub.end(), src, src + cfg_.n_routed_experts);
                }
            }
            return sub;
        }
        return expert_freq_counts_;
    }

    void sync_expert_freq_from_gpu() {
        if (track_expert_freq_ && d_expert_counts_.data && !expert_freq_counts_.empty()) {
            std::lock_guard<std::mutex> lock(expert_profile_mutex_);
            CUDA_CHECK(cudaMemcpy(expert_freq_counts_.data(), d_expert_counts_.data,
                                  expert_freq_counts_.size() * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));
        }
    }

    void enable_expert_tracking(const std::string& path = "") {
        int n_layers = cfg_.num_hidden_layers;
        int n_experts = cfg_.n_routed_experts;
        
        expert_freq_counts_ = load_expert_freq(path, n_layers, n_experts, &expert_freq_total_tokens_);
        if (expert_freq_counts_.empty()) {
            expert_freq_counts_.assign((size_t)n_layers * n_experts, 0);
            expert_freq_total_tokens_ = 0;
            if (d_expert_counts_.data) {
                CUDA_CHECK(cudaMemset(d_expert_counts_.data, 0, d_expert_counts_.size_bytes));
            }
            LOG_INFO("Expert frequency tracking enabled (%d layers x %d experts, new profile)", n_layers, n_experts);
        } else {
            if (d_expert_counts_.data) {
                CUDA_CHECK(cudaMemcpy(d_expert_counts_.data, expert_freq_counts_.data(),
                                      expert_freq_counts_.size() * sizeof(uint32_t),
                                      cudaMemcpyHostToDevice));
            }
            LOG_INFO("Expert frequency tracking enabled (%d layers x %d experts, resuming from %lld tokens)", n_layers, n_experts, expert_freq_total_tokens_);
        }

        expert_token_counts_.resize(n_layers);
        for (int l = 0; l < n_layers; l++) {
            expert_token_counts_[l].resize(n_experts);
        }

        if (!path.empty()) {
            std::string json_profile_path = (std::filesystem::path(path).parent_path() / "expert_profile.json").string();
            load_expert_profile_json(json_profile_path);
        }

        track_expert_freq_ = true;
    }

    static std::string infer_expert_category(const std::vector<std::pair<std::string, uint32_t>>& top_tokens) {
        if (top_tokens.empty()) return "General Prose";

        double code_score = 0;
        double math_score = 0;
        double reasoning_score = 0;
        double format_score = 0;
        double multi_score = 0;
        double prose_score = 0;

        for (auto& [tok, count] : top_tokens) {
            double w = (double)count;
            std::string t = tok;

            size_t start = t.find_first_not_of(" \t\r\n");
            std::string trimmed = (start == std::string::npos) ? t : t.substr(start);
            size_t end = trimmed.find_last_not_of(" \t\r\n");
            if (end != std::string::npos) trimmed = trimmed.substr(0, end + 1);

            std::string lower = trimmed;
            for (char& c : lower) c = (char)::tolower((unsigned char)c);

            // 1. Format / Punctuation check
            if (t == "\n" || t == "\n\n" || t == "\r\n" || t == "  " || t == "    " ||
                t == "#" || t == "##" || t == "###" || t == "|" || t == "---" ||
                t == "." || t == "," || t == ";" || t == ":" || t == "!" || t == "?" ||
                t == "\"" || t == "'" || t == "`" || t == "*" || t == "**" || t == "-") {
                format_score += w * 1.5;
                continue;
            }

            // 2. HTML / CSS / JS / Programming check
            if (lower == "html" || lower == "<!doctype" || lower == "<html" || lower == "</html>" ||
                lower == "<head" || lower == "</head>" || lower == "<body" || lower == "</body>" ||
                lower == "<div" || lower == "</div>" || lower == "<span" || lower == "</span>" ||
                lower == "<script" || lower == "</script>" || lower == "<style" || lower == "</style>" ||
                lower == "<canvas" || lower == "</canvas>" || lower == "<button" || lower == "</button>" ||
                lower == "<input" || lower == "<p" || lower == "<h1" || lower == "<h2" || lower == "<h3" ||
                lower == "function" || lower == "const" || lower == "let" || lower == "var" ||
                lower == "document" || lower == "window" || lower == "getelementbyid" || lower == "addeventlistener" ||
                lower == "def" || lower == "class" || lower == "import" || lower == "export" || lower == "return" ||
                lower == "void" || lower == "int" || lower == "float" || lower == "double" || lower == "bool" ||
                lower == "char" || lower == "string" || lower == "auto" || lower == "public" || lower == "private" ||
                lower == "protected" || lower == "virtual" || lower == "override" || lower == "static" ||
                lower == "template" || lower == "typename" || lower == "struct" || lower == "enum" ||
                lower == "namespace" || lower == "using" || lower == "include" || lower == "async" || lower == "await" ||
                lower == "yield" || lower == "lambda" || lower == "try" || lower == "catch" || lower == "throw" ||
                lower == "finally" || lower == "self" || lower == "this" || lower == "null" || lower == "nullptr" ||
                lower == "true" || lower == "false" || lower == "undefined" || lower == "console" || lower == "printf" ||
                lower == "print" || lower == "std" || lower == "sizeof" || lower == "typedef" || lower == "malloc" ||
                lower == "free" || lower == "cuda" || lower == "ptr" || lower == "select" || lower == "from" ||
                lower == "where" || lower == "insert" || lower == "update" || lower == "delete" || lower == "join" ||
                trimmed == "::" || trimmed == "->" || trimmed == "=>" || trimmed == "!=" ||
                trimmed == "==" || trimmed == "===" || trimmed == "!==" || trimmed == "<=" ||
                trimmed == ">=" || trimmed == "&&" || trimmed == "||" || trimmed == "++" ||
                trimmed == "--" || trimmed == "+=" || trimmed == "-=" || trimmed == "*=" ||
                trimmed == "/=" || trimmed == "//" || trimmed == "/*" || trimmed == "*/" ||
                trimmed == "{" || trimmed == "}" || trimmed == "[" || trimmed == "]" ||
                trimmed == "```" || trimmed.find("```") != std::string::npos ||
                trimmed.find("px") != std::string::npos || trimmed.find("rgb") != std::string::npos) {
                code_score += w * 3.0;
                continue;
            }

            // 3. Reasoning / Thought Process markers
            if (lower == "wait" || lower == "let" || lower == "first" || lower == "because" ||
                lower == "therefore" || lower == "however" || lower == "consider" || lower == "analyze" ||
                lower == "step" || lower == "check" || lower == "verify" || lower == "assume" ||
                lower == "hypothesis" || lower == "notice" || lower == "recall" || lower == "indeed" ||
                lower == "clearly" || lower == "specifically" || lower == "alternatively" || lower == "so" ||
                lower == "thus" || lower == "hence" || lower == "since" || lower == "given" ||
                lower == "then" || lower == "now" || lower == "next" || lower == "finally" ||
                lower == "need" || lower == "should" || lower == "must" || lower == "will" ||
                lower == "goal" || lower == "idea" || lower == "approach" || lower == "solution" ||
                lower == "method" || lower == "case" || lower == "problem" || lower == "think" ||
                lower == "thought" || lower == "reasoning" || lower == "logic" ||
                trimmed == "<think>" || trimmed == "</think>") {
                reasoning_score += w * 2.5;
                continue;
            }

            // 4. Math / Logic & LaTeX notation
            if (trimmed.find("\\frac") != std::string::npos || trimmed.find("\\int") != std::string::npos ||
                trimmed.find("\\sum") != std::string::npos || trimmed.find("\\prod") != std::string::npos ||
                trimmed.find("\\sqrt") != std::string::npos || trimmed.find("\\partial") != std::string::npos ||
                trimmed.find("\\alpha") != std::string::npos || trimmed.find("\\beta") != std::string::npos ||
                trimmed.find("\\gamma") != std::string::npos || trimmed.find("\\delta") != std::string::npos ||
                trimmed.find("\\theta") != std::string::npos || trimmed.find("\\lambda") != std::string::npos ||
                trimmed.find("\\sigma") != std::string::npos || trimmed.find("\\pi") != std::string::npos ||
                trimmed.find("\\in") != std::string::npos || trimmed.find("\\approx") != std::string::npos ||
                trimmed.find("\\le") != std::string::npos || trimmed.find("\\ge") != std::string::npos ||
                lower == "matrix" || lower == "vector" || lower == "tensor" || lower == "integral" ||
                lower == "derivative" || lower == "equation" || lower == "theorem" || lower == "lemma" ||
                lower == "proof" || lower == "sin" || lower == "cos" || lower == "tan" ||
                lower == "log" || lower == "exp" || lower == "mod" || lower == "prime" ||
                lower == "eigen" || lower == "polynomial" || trimmed == "+" || trimmed == "=" ||
                trimmed == "<>" || trimmed == "^" || trimmed == "<" || trimmed == ">" ||
                (trimmed.size() == 1 && trimmed[0] >= '0' && trimmed[0] <= '9')) {
                math_score += w * 2.5;
                continue;
            }

            // 5. Non-ASCII multilingual
            bool has_non_ascii = false;
            for (unsigned char c : t) {
                if (c >= 0x80) { has_non_ascii = true; break; }
            }
            if (has_non_ascii) {
                multi_score += w * 2.0;
                continue;
            }

            prose_score += w;
        }

        double max_score = prose_score;
        std::string best_cat = "General Prose";

        if (code_score > max_score) { max_score = code_score; best_cat = "Coding / Syntax"; }
        if (math_score > max_score) { max_score = math_score; best_cat = "Math / Logic"; }
        if (reasoning_score > max_score) { max_score = reasoning_score; best_cat = "Reasoning"; }
        if (format_score > max_score) { max_score = format_score; best_cat = "Format / Syntax"; }
        if (multi_score > max_score) { max_score = multi_score; best_cat = "Multilingual"; }

        return best_cat;
    }

    json generate_expert_profile_json() {
        std::lock_guard<std::mutex> lock(expert_profile_mutex_);
        json root;
        root["status"] = "ok";
        root["tracking_enabled"] = track_expert_freq_;
        root["total_tokens"] = expert_freq_total_tokens_;
        root["n_layers"] = cfg_.num_hidden_layers;
        root["n_experts"] = cfg_.n_routed_experts;
        root["active_experts_per_tok"] = cfg_.num_experts_per_tok;

        int n_layers = cfg_.num_hidden_layers;
        int n_experts = cfg_.n_routed_experts;
        int l1_res_count = expert_loader_.all_resident(n_layers) ? (n_layers * n_experts) : (int)expert_loader_.cache_capacity_;
        int l2_res_count = expert_loader_.dram_cache_capacity_;
        int ssd_count = (n_layers * n_experts) - l1_res_count - l2_res_count;
        if (ssd_count < 0) ssd_count = 0;

        root["l1_capacity"] = expert_loader_.cache_capacity_;
        root["l1_resident"] = l1_res_count;
        root["l2_capacity"] = expert_loader_.dram_cache_capacity_;
        root["l2_resident"] = l2_res_count;
        root["ssd_count"] = ssd_count;

        std::map<std::string, int> cat_counts;
        cat_counts["Coding / Syntax"] = 0;
        cat_counts["Math / Logic"] = 0;
        cat_counts["Reasoning"] = 0;
        cat_counts["General Prose"] = 0;
        cat_counts["Format / Syntax"] = 0;
        cat_counts["Multilingual"] = 0;

        struct ExpertEntry {
            int layer;
            int expert_id;
            uint32_t count;
            float hit_pct;
            std::string category;
            std::string location;
            std::string tier;
            std::vector<std::pair<std::string, uint32_t>> top_tokens;
        };

        std::vector<ExpertEntry> entries;
        double denom_tokens = expert_freq_total_tokens_ > 0 ? (double)expert_freq_total_tokens_ : 1.0;

        for (int l = 0; l < n_layers; l++) {
            for (int e = 0; e < n_experts; e++) {
                uint32_t cnt = 0;
                if (!expert_freq_counts_.empty()) {
                    cnt = expert_freq_counts_[(size_t)l * n_experts + e];
                }
                if (cnt == 0) continue;

                std::vector<std::pair<std::string, uint32_t>> top_toks;
                if (l < (int)expert_token_counts_.size() && e < (int)expert_token_counts_[l].size() && !expert_token_counts_[l][e].empty()) {
                    std::vector<std::pair<int, uint32_t>> raw_toks;
                    for (auto& [tid, tcnt] : expert_token_counts_[l][e]) {
                        if (!is_control_token(tid)) {
                            raw_toks.push_back({tid, tcnt});
                        }
                    }
                    std::sort(raw_toks.begin(), raw_toks.end(), [](const auto& a, const auto& b) {
                        return a.second > b.second;
                    });
                    int take = std::min((int)raw_toks.size(), 8);
                    for (int i = 0; i < take; i++) {
                        std::string decoded = tokenizer_.decode_token_str(raw_toks[i].first);
                        if (!decoded.empty()) {
                            top_toks.push_back({decoded, raw_toks[i].second});
                        }
                    }
                } else if (l < (int)expert_loaded_top_tokens_.size() && e < (int)expert_loaded_top_tokens_[l].size()) {
                    top_toks = expert_loaded_top_tokens_[l][e];
                }

                std::string category = infer_expert_category(top_toks);
                cat_counts[category]++;

                std::string loc = expert_loader_.get_expert_location(l, e);
                std::string tier = expert_loader_.get_expert_tier(l, e);

                float hit_pct = (float)((cnt * 100.0) / denom_tokens);
                entries.push_back({l, e, cnt, hit_pct, category, loc, tier, top_toks});
            }
        }

        std::sort(entries.begin(), entries.end(), [](const ExpertEntry& a, const ExpertEntry& b) {
            return a.count > b.count;
        });

        json exp_arr = json::array();
        int rank = 1;
        for (auto& ent : entries) {
            json item;
            item["rank"] = rank++;
            item["layer"] = ent.layer;
            item["expert_id"] = ent.expert_id;
            item["count"] = ent.count;
            item["hit_pct"] = std::round(ent.hit_pct * 100.0) / 100.0;
            item["category"] = ent.category;
            item["location"] = ent.location;
            item["tier"] = ent.tier;
            json toks_json = json::array();
            for (auto& [tstr, tcnt] : ent.top_tokens) {
                toks_json.push_back({{"token", tstr}, {"count", tcnt}});
            }
            item["top_tokens"] = toks_json;
            exp_arr.push_back(item);
        }

        root["experts"] = exp_arr;
        root["categories_summary"] = cat_counts;
        root["total_active_experts"] = (int)entries.size();
        return root;
    }

    void save_expert_profile(const std::string& base_dir) {
        if (expert_token_counts_.empty() && expert_freq_counts_.empty()) return;
        json profile = generate_expert_profile_json();

        std::string json_path = base_dir.empty() ? "expert_profile.json" : (base_dir + "/expert_profile.json");
        std::string txt_path = base_dir.empty() ? "expert_profile.txt" : (base_dir + "/expert_profile.txt");

        std::ofstream jout(json_path);
        if (jout.is_open()) {
            jout << profile.dump(2);
            jout.close();
        }

        std::ofstream tout(txt_path);
        if (tout.is_open()) {
            tout << "====================================================================================================\n";
            tout << "                        MINNIETHEMOECHER EXPERT SPECIALIZATION PROFILE\n";
            tout << "====================================================================================================\n";
            tout << "Total Tokens Profiled: " << expert_freq_total_tokens_ 
                 << " | Active Experts: " << profile["total_active_experts"] 
                 << " | Model: " << cfg_.num_hidden_layers << " layers x " << cfg_.n_routed_experts << " experts\n\n";
            tout << "LAYER  EXPERT   HITS     HIT %     CATEGORY              TOP TOKENS (TRIGGER VOCABULARY)\n";
            tout << "----------------------------------------------------------------------------------------------------\n";
            for (auto& item : profile["experts"]) {
                char linebuf[512];
                int l = item["layer"].get<int>();
                int e = item["expert_id"].get<int>();
                uint32_t cnt = item["count"].get<uint32_t>();
                double pct = item["hit_pct"].get<double>();
                std::string cat = item["category"].get<std::string>();

                std::string toks_str;
                for (auto& t : item["top_tokens"]) {
                    if (!toks_str.empty()) toks_str += ", ";
                    std::string tok_name = t["token"].get<std::string>();
                    for (char& c : tok_name) { if (c == '\n') c = ' '; if (c == '\r') c = ' '; }
                    toks_str += "\"" + tok_name + "\" (" + std::to_string(t["count"].get<uint32_t>()) + ")";
                }

                snprintf(linebuf, sizeof(linebuf), "L%-4d E%-6d %-8u %-8.2f%% %-20s  %s\n",
                         l, e, cnt, pct, cat.c_str(), toks_str.c_str());
                tout << linebuf;
            }
            tout << "====================================================================================================\n";
            tout.close();
        }
        LOG_INFO("Saved expert specialization profile: %s and %s", json_path.c_str(), txt_path.c_str());
    }

    void load_expert_profile_json(const std::string& path) {
        std::ifstream in(path);
        if (!in.is_open()) return;
        try {
            json root;
            in >> root;
            if (root.contains("experts") && root["experts"].is_array()) {
                int n_layers = cfg_.num_hidden_layers;
                int n_experts = cfg_.n_routed_experts;
                if ((int)expert_token_counts_.size() != n_layers) {
                    expert_token_counts_.resize(n_layers);
                    for (int l = 0; l < n_layers; l++) expert_token_counts_[l].resize(n_experts);
                }
                expert_loaded_top_tokens_.resize(n_layers);
                for (int l = 0; l < n_layers; l++) expert_loaded_top_tokens_[l].resize(n_experts);

                for (auto& exp : root["experts"]) {
                    int l = exp.value("layer", -1);
                    int e = exp.value("expert_id", -1);
                    if (l >= 0 && l < n_layers && e >= 0 && e < n_experts && exp.contains("top_tokens")) {
                        std::vector<std::pair<std::string, uint32_t>> toks;
                        for (auto& t : exp["top_tokens"]) {
                            std::string tstr = t.value("token", "");
                            uint32_t tcnt = t.value("count", (uint32_t)0);
                            int tid = t.value("token_id", -1);
                            if (tid >= 0) {
                                expert_token_counts_[l][e][tid] += tcnt;
                            }
                            if (!tstr.empty()) {
                                toks.push_back({tstr, tcnt});
                            }
                        }
                        expert_loaded_top_tokens_[l][e] = std::move(toks);
                    }
                }
                LOG_INFO("Loaded existing expert specialization profile from %s", path.c_str());
            }
        } catch (...) {
            LOG_WARN("Failed to parse existing expert profile JSON: %s", path.c_str());
        }
    }

    void save_expert_freq(const std::string& path) {
        sync_expert_freq_from_gpu();
        if (expert_freq_counts_.empty()) return;
        std::ofstream out(path, std::ios::binary);
        if (!out.is_open()) {
            LOG_ERROR("Cannot write expert freq file: %s", path.c_str());
            return;
        }
        int32_t n_layers = cfg_.num_hidden_layers;
        int32_t n_experts = cfg_.n_routed_experts;
        out.write((const char*)&n_layers, sizeof(n_layers));
        out.write((const char*)&n_experts, sizeof(n_experts));
        out.write((const char*)&expert_freq_total_tokens_, sizeof(expert_freq_total_tokens_));
        out.write((const char*)expert_freq_counts_.data(), expert_freq_counts_.size() * sizeof(uint32_t));
        out.close();

        uint32_t total_selections = 0;
        for (auto c : expert_freq_counts_) total_selections += c;
        LOG_INFO("Saved expert frequency data: %s (%d layers, %lld tokens, %u total selections)",
                 path.c_str(), n_layers, expert_freq_total_tokens_, total_selections);

        // Also save human-readable and JSON semantic profile
        std::string base_dir = std::filesystem::path(path).parent_path().string();
        save_expert_profile(base_dir);
    }

    static std::vector<uint32_t> load_expert_freq(const std::string& path, int expected_layers, int expected_experts, int64_t* total_tokens_out = nullptr) {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) return {};
        int32_t n_layers, n_experts;
        int64_t total_tokens;
        in.read((char*)&n_layers, sizeof(n_layers));
        in.read((char*)&n_experts, sizeof(n_experts));
        in.read((char*)&total_tokens, sizeof(total_tokens));
        if (n_layers != expected_layers || n_experts != expected_experts) {
            LOG_WARN("Expert freq file mismatch: expected %dx%d, got %dx%d. Ignoring.",
                     expected_layers, expected_experts, n_layers, n_experts);
            return {};
        }
        std::vector<uint32_t> counts((size_t)n_layers * n_experts);
        in.read((char*)counts.data(), counts.size() * sizeof(uint32_t));
        if (!in.good()) {
            LOG_WARN("Expert freq file truncated. Ignoring.");
            return {};
        }
        if (total_tokens_out) *total_tokens_out = total_tokens;
        LOG_INFO("Loaded expert freq data: %s (%d layers, %lld tokens)", path.c_str(), n_layers, total_tokens);
        return counts;
    }
};
