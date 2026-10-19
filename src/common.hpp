#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <chrono>
#include <nlohmann/json.hpp>
#include "version.hpp"

#if defined(__APPLE__)
#include "metal/metal_backend.h"
#else
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#endif

// ════════════════════════════════════════════════════════════════════════════════
//  Macros & Constants
// ════════════════════════════════════════════════════════════════════════════════

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", \
                #x, __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)
#endif

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(x) do { \
    cublasStatus_t stat = (x); \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)stat, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)
#endif

static constexpr int PAGE_SIZE = 4096;
static constexpr int MAX_SEQ_LEN = 65536;
static constexpr const char* MOECHER_VERSION = moecher::VERSION;

// ════════════════════════════════════════════════════════════════════════════════
//  Model Config (from manifest)
// ════════════════════════════════════════════════════════════════════════════════

enum class ModelArch { DEEPSEEK_V4, QWEN, QWEN2, NEMOTRON_35 };

struct ModelConfig {
    ModelArch architecture = ModelArch::DEEPSEEK_V4;
    bool is_qwen() const { return architecture == ModelArch::QWEN || architecture == ModelArch::QWEN2; }
    bool is_nemotron() const { return architecture == ModelArch::NEMOTRON_35; }
    bool is_deepseek() const { return architecture == ModelArch::DEEPSEEK_V4; }
    std::string model_name = "";
    std::string model_id = "";
    int vocab_size = 129280;
    int hidden_size = 4096;
    int num_hidden_layers = 43;
    int num_attention_heads = 64;
    int num_key_value_heads = 8;
    int head_dim = 512;
    int intermediate_size = 13824;
    int qk_rope_head_dim = 64;
    int q_lora_rank = 1024;
    int o_lora_rank = 1024;
    int o_groups = 8;
    int moe_intermediate_size = 2048;
    int moe_shared_expert_intermediate_size = 3712;
    int n_routed_experts = 256;
    int num_experts_per_tok = 6;
    int n_shared_experts = 1;
    int n_hash_layers = 3;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    float rope_factor = 16.0f;
    int rope_beta_fast = 32;
    int rope_beta_slow = 1;
    int original_seq_len = 65536;
    int sliding_window = 128;
    int window_size = 128;              // raw attention window per layer (paper: 128)
    int max_seq_len = 65536;           // max sequence context length (default: 64k)
    std::string scoring_func = "sqrtsoftplus";
    float routed_scaling_factor = 1.5f;
    float swiglu_limit = 10.0f;
    int hc_mult = 4;
    int hc_sinkhorn_iters = 20;
    float hc_eps = 1e-6f;
    int bos_token_id = 0;
    int eos_token_id = 1;
    float compress_rope_theta = 160000.0f;
    std::string expert_dtype = "fp4";
    std::vector<int> compress_ratios;  // per-layer compression ratios
    int max_compressed_entries = 0;    // max compressed KV entries per layer

    // Nemotron 3.5 hybrid Mamba-2 SSM parameters
    int ssm_d_state = 128;
    int ssm_d_conv = 4;
    int ssm_expand = 2;
    int ssm_dt_rank = 256;
    std::vector<int> layer_types;      // 0: Attention, 1: Mamba-2 SSM, 2: MoE (or hybrid)

    // Recommended / default inference parameters per model
    float default_temperature = -1.0f;
    float default_repetition_penalty = -1.0f;
    std::string default_reasoning_effort = "";
    int default_thinking_budget = -1;

    int layer_compress_ratio(int layer_id) const {
        if (layer_id < (int)compress_ratios.size()) return compress_ratios[layer_id];
        return 0;
    }

    void from_json(const nlohmann::json& j) {
        auto get = [&](auto& field, const char* key) {
            if (j.contains(key) && !j.at(key).is_null()) j.at(key).get_to(field);
        };
        get(model_name, "model_name");
        get(model_id, "model_id");
        if (model_name.empty()) get(model_name, "name");
        if (model_id.empty()) get(model_id, "id");
        if (j.contains("architecture")) {
            std::string arch_str = j["architecture"].get<std::string>();
            if (model_id.find("nemotron") != std::string::npos || model_name.find("Nemotron") != std::string::npos || arch_str == "nemotron" || arch_str == "nemotron_35" || arch_str == "nemotron_h") {
                architecture = ModelArch::NEMOTRON_35;
            } else if (model_id.find("qwen3") != std::string::npos || model_name.find("Qwen 3") != std::string::npos || arch_str == "qwen3") {
                architecture = ModelArch::QWEN;
            } else if (arch_str == "qwen2" || arch_str == "qwen2.5") {
                architecture = ModelArch::QWEN2;
            } else if (arch_str == "qwen" || arch_str == "llama") {
                architecture = ModelArch::QWEN;
            } else {
                architecture = ModelArch::DEEPSEEK_V4;
            }
        }
        get(vocab_size, "vocab_size");
        get(hidden_size, "hidden_size");
        get(num_hidden_layers, "num_hidden_layers");
        get(num_attention_heads, "num_attention_heads");
        get(num_key_value_heads, "num_key_value_heads");
        get(head_dim, "head_dim");
        get(intermediate_size, "intermediate_size");
        get(qk_rope_head_dim, "qk_rope_head_dim");
        get(q_lora_rank, "q_lora_rank");
        get(o_lora_rank, "o_lora_rank");
        get(o_groups, "o_groups");
        get(moe_intermediate_size, "moe_intermediate_size");
        get(moe_shared_expert_intermediate_size, "moe_shared_expert_intermediate_size");
        get(n_routed_experts, "n_routed_experts");
        get(num_experts_per_tok, "num_experts_per_tok");
        get(n_shared_experts, "n_shared_experts");
        get(n_hash_layers, "n_hash_layers");
        get(rms_norm_eps, "rms_norm_eps");
        get(rope_theta, "rope_theta");
        get(rope_factor, "rope_factor");
        get(rope_beta_fast, "rope_beta_fast");
        get(rope_beta_slow, "rope_beta_slow");
        get(original_seq_len, "original_seq_len");
        get(sliding_window, "sliding_window");
        get(window_size, "window_size");
        get(max_seq_len, "max_seq_len");
        get(max_seq_len, "max_position_embeddings");
        // If window_size not in manifest, use sliding_window as fallback
        if (!j.contains("window_size")) window_size = sliding_window;
        get(scoring_func, "scoring_func");
        get(routed_scaling_factor, "routed_scaling_factor");
        get(swiglu_limit, "swiglu_limit");
        get(hc_mult, "hc_mult");
        get(hc_sinkhorn_iters, "hc_sinkhorn_iters");
        get(hc_eps, "hc_eps");
        get(bos_token_id, "bos_token_id");
        get(eos_token_id, "eos_token_id");
        get(compress_rope_theta, "compress_rope_theta");
        get(expert_dtype, "expert_dtype");
        get(ssm_d_state, "ssm_d_state");
        get(ssm_d_conv, "ssm_d_conv");
        get(ssm_expand, "ssm_expand");
        get(ssm_dt_rank, "ssm_dt_rank");
        if (j.contains("layer_types") && j["layer_types"].is_array()) {
            layer_types = j["layer_types"].get<std::vector<int>>();
        }
        if (j.contains("compress_ratios") && j["compress_ratios"].is_array()) {
            compress_ratios = j["compress_ratios"].get<std::vector<int>>();
        }
        // Compute max compressed entries: max_seq_len / min_ratio (critical for DeepSeek MLA compressed KV cache)
        max_compressed_entries = 0;
        for (int r : compress_ratios) {
            if (r > 0) {
                int entries = max_seq_len / r + 2;
                if (entries > max_compressed_entries) max_compressed_entries = entries;
            }
        }
        get(default_temperature, "default_temperature");
        get(default_repetition_penalty, "default_repetition_penalty");
        get(default_reasoning_effort, "default_reasoning_effort");
        get(default_thinking_budget, "default_thinking_budget");

        if (default_temperature < 0.0f) {
            default_temperature = is_nemotron() ? 0.6f : 0.7f;
        }
        if (default_repetition_penalty < 0.0f) {
            default_repetition_penalty = 1.00f;
        }
        if (default_reasoning_effort.empty()) {
            default_reasoning_effort = is_nemotron() ? "high" : "medium";
        }
        if (default_thinking_budget < 0) {
            default_thinking_budget = 4096;
        }
    }
};

// ════════════════════════════════════════════════════════════════════════════════
//  GPU Hardware Capabilities & Gating
// ════════════════════════════════════════════════════════════════════════════════

struct GpuCapabilities {
    int major = 0;
    int minor = 0;
    size_t total_vram_bytes = 0;
    std::string device_name = "";
    bool is_ampere = false;           // SM 8.0/8.6 (RTX 3090, A100)
    bool is_ada_or_newer = false;     // SM 8.9+ (RTX 4060 Ti, 4090)
    bool is_blackwell_or_newer = false; // SM 10.0+ (RTX 5060 Ti, 5090)
    bool supports_fp8 = false;        // FP8 Tensor Cores (Ada/Hopper+)
    bool supports_fp4 = false;        // Blackwell+
    bool is_vram_constrained = false; // <= 18GB (e.g. 16GB cards)
};

static inline GpuCapabilities detect_gpu_capabilities(int device_id = 0) {
    GpuCapabilities caps;
#if !defined(__APPLE__)
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) {
        caps.major = prop.major;
        caps.minor = prop.minor;
        caps.total_vram_bytes = prop.totalGlobalMem;
        caps.device_name = prop.name;
        caps.is_ampere = (prop.major == 8 && (prop.minor == 0 || prop.minor == 6));
        caps.is_ada_or_newer = (prop.major > 8 || (prop.major == 8 && prop.minor >= 9));
        caps.is_blackwell_or_newer = (prop.major >= 10);
        caps.supports_fp8 = caps.is_ada_or_newer;
        caps.supports_fp4 = caps.is_blackwell_or_newer;
        caps.is_vram_constrained = (prop.totalGlobalMem <= 18ULL * 1024 * 1024 * 1024);
    }
#endif
    return caps;
}

// ════════════════════════════════════════════════════════════════════════════════
//  GPU Tensor: thin wrapper around a CUDA allocation
// ════════════════════════════════════════════════════════════════════════════════

struct GPUTensor {
    void* data = nullptr;
    size_t size_bytes = 0;
    std::vector<int> shape;
    std::string dtype;  // "BF16", "F32", "F8_E4M3", "I8", etc.

    bool is_host = false;

    GPUTensor() = default;
    GPUTensor(const GPUTensor&) = delete;
    GPUTensor& operator=(const GPUTensor&) = delete;
    GPUTensor(GPUTensor&& other) noexcept {
        data = other.data;
        size_bytes = other.size_bytes;
        shape = std::move(other.shape);
        dtype = std::move(other.dtype);
        is_host = other.is_host;
        other.data = nullptr;
        other.size_bytes = 0;
        other.is_host = false;
    }
    GPUTensor& operator=(GPUTensor&& other) noexcept {
        if (this != &other) {
            free();
            data = other.data;
            size_bytes = other.size_bytes;
            shape = std::move(other.shape);
            dtype = std::move(other.dtype);
            is_host = other.is_host;
            other.data = nullptr;
            other.size_bytes = 0;
            other.is_host = false;
        }
        return *this;
    }

    void alloc(size_t bytes) {
        if (data) free();
#if !defined(__APPLE__)
        CUDA_CHECK(cudaMalloc(&data, bytes));
#endif
        size_bytes = bytes;
        is_host = false;
    }
    void alloc_host(size_t bytes) {
        if (data) free();
#if !defined(__APPLE__)
        CUDA_CHECK(cudaMallocHost(&data, bytes));
#endif
        size_bytes = bytes;
        is_host = true;
    }
    void free() {
        if (data) {
#if !defined(__APPLE__)
            if (is_host) cudaFreeHost(data);
            else cudaFree(data);
#endif
            data = nullptr;
            size_bytes = 0;
            is_host = false;
        }
    }
    ~GPUTensor() { free(); }

#if !defined(__APPLE__)
    __nv_bfloat16* bf16() { return (__nv_bfloat16*)data; }
    const __nv_bfloat16* bf16() const { return (const __nv_bfloat16*)data; }
#endif
    float* f32() { return (float*)data; }
    const float* f32() const { return (const float*)data; }
    uint8_t* u8() { return (uint8_t*)data; }
    const uint8_t* u8() const { return (const uint8_t*)data; }
    int32_t* i32() { return (int32_t*)data; }
    const int32_t* i32() const { return (const int32_t*)data; }
    int64_t* i64() { return (int64_t*)data; }
    const int64_t* i64() const { return (const int64_t*)data; }
    uint64_t* u64() { return (uint64_t*)data; }
    const uint64_t* u64() const { return (const uint64_t*)data; }
};
