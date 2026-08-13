// server_single.cpp — Bare-metal DeepSeek V4-Flash MoE inference engine
// Stack: C++17, CUDA, cuBLAS, cpp-httplib, nlohmann/json
// Features:
//   - O_DIRECT SSD offloading for MoE experts (3090 24GB friendly)
//   - Manifest-driven tensor loading from moecher_manifest.json
//   - OpenAI-compatible /v1/chat/completions API on port 8001
//   - BPE tokenizer from HuggingFace tokenizer.json
//   - Full DeepSeek V4-Flash forward pass: MLA + MoE + HC residuals

#ifndef _GNU_SOURCE
#define _GNU_SOURCE  // for O_DIRECT
#endif
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <map>
#include <vector>
#include <future>
#include <mutex>
#include <memory>
#include "thread_pool.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <cstdarg>
#include <cassert>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <numeric>
#include <future>
#include <mutex>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <chrono>
#include <mutex>
#include <memory>
#include <random>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <nlohmann/json.hpp>
#include <httplib.h>

#include "activations.cuh"

using json = nlohmann::json;

// ════════════════════════════════════════════════════════════════════════════════
//  Macros & Constants
// ════════════════════════════════════════════════════════════════════════════════

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", \
                #x, __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(x) do { \
    cublasStatus_t stat = (x); \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)stat, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

static constexpr int PAGE_SIZE = 4096;
static constexpr int MAX_SEQ_LEN = 65536;

// ════════════════════════════════════════════════════════════════════════════════
//  Logging
// ════════════════════════════════════════════════════════════════════════════════

static std::mutex g_log_mutex;
static std::ofstream g_log_file;
bool g_log_experts = false;
bool g_log_tokens = true;


static void log_msg(const char* level, const char* fmt, ...) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    char timebuf[64];
    strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", localtime(&time));

    char buf[4096];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    fprintf(stderr, "[%s] [%s] %s\n", timebuf, level, buf);
    if (g_log_file.is_open()) {
        g_log_file << "[" << timebuf << "] [" << level << "] " << buf << "\n";
        g_log_file.flush();
    }
}

#define LOG_INFO(...)  log_msg("INFO",  __VA_ARGS__)
#define LOG_WARN(...)  log_msg("WARN",  __VA_ARGS__)
#define LOG_ERROR(...) log_msg("ERROR", __VA_ARGS__)

// ════════════════════════════════════════════════════════════════════════════════
//  Model Config (from manifest)
// ════════════════════════════════════════════════════════════════════════════════

struct ModelConfig {
    int vocab_size = 129280;
    int hidden_size = 4096;
    int num_hidden_layers = 43;
    int num_attention_heads = 64;
    int head_dim = 512;
    int qk_rope_head_dim = 64;
    int q_lora_rank = 1024;
    int o_lora_rank = 1024;
    int o_groups = 8;
    int moe_intermediate_size = 2048;
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

    int layer_compress_ratio(int layer_id) const {
        if (layer_id < (int)compress_ratios.size()) return compress_ratios[layer_id];
        return 0;
    }

    void from_json(const json& j) {
        auto get = [&](auto& field, const char* key) {
            if (j.contains(key)) j.at(key).get_to(field);
        };
        get(vocab_size, "vocab_size");
        get(hidden_size, "hidden_size");
        get(num_hidden_layers, "num_hidden_layers");
        get(num_attention_heads, "num_attention_heads");
        get(head_dim, "head_dim");
        get(qk_rope_head_dim, "qk_rope_head_dim");
        get(q_lora_rank, "q_lora_rank");
        get(o_lora_rank, "o_lora_rank");
        get(o_groups, "o_groups");
        get(moe_intermediate_size, "moe_intermediate_size");
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
        if (j.contains("compress_ratios") && j["compress_ratios"].is_array()) {
            compress_ratios = j["compress_ratios"].get<std::vector<int>>();
        }
        // Compute max compressed entries: sliding_window / min_ratio
        // This determines the size of the compressed KV cache per layer
        max_compressed_entries = 0;
        for (int r : compress_ratios) {
            if (r > 0) {
                int entries = sliding_window / r;
                if (entries > max_compressed_entries) max_compressed_entries = entries;
            }
        }
    }
};

// ════════════════════════════════════════════════════════════════════════════════
//  Minimal BPE Tokenizer (reads HuggingFace tokenizer.json)
// ════════════════════════════════════════════════════════════════════════════════

class BPETokenizer {
public:
    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) { LOG_ERROR("Cannot open tokenizer: %s", path.c_str()); return false; }
        json tok;
        try { f >> tok; } catch (const std::exception& e) {
            LOG_ERROR("JSON parse error in tokenizer: %s", e.what()); return false;
        }

        // Load vocab
        auto& model = tok["model"];
        auto& vocab = model["vocab"];
        for (auto& [k, v] : vocab.items()) {
            int id = v.get<int>();
            token_to_id_[k] = id;
            id_to_token_[id] = k;
        }

        // Load merges
        if (model.contains("merges")) {
            for (auto& m : model["merges"]) {
                std::string merge_str = m.get<std::string>();
                merges_.push_back(merge_str);
                merge_rank_[merge_str] = (int)merges_.size();
            }
        }

        // Load added tokens (special tokens + role tokens like User, Assistant)
        if (tok.contains("added_tokens")) {
            for (auto& at : tok["added_tokens"]) {
                int id = at["id"].get<int>();
                std::string content = at["content"].get<std::string>();
                token_to_id_[content] = id;
                id_to_token_[id] = content;
                // Add ALL added tokens to special_tokens_ for greedy matching
                // DeepSeek marks User/Assistant as special=false but they still
                // need to be matched greedily during tokenization
                special_tokens_.push_back({content, id});
            }
        }

        // Sort special tokens by length (longest first) for greedy matching
        std::sort(special_tokens_.begin(), special_tokens_.end(),
                  [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

        // Build byte-to-unicode mapping (GPT-2 style)
        build_byte_mapping();

        LOG_INFO("Tokenizer loaded: %zu vocab, %zu merges, %zu special tokens",
                 token_to_id_.size(), merges_.size(), special_tokens_.size());
        return true;
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;

        // Split on special tokens first
        std::vector<std::pair<std::string, bool>> segments; // (text, is_special)
        split_on_special(text, segments);

        for (auto& [seg, is_special] : segments) {
            if (is_special) {
                auto it = token_to_id_.find(seg);
                if (it != token_to_id_.end()) ids.push_back(it->second);
                continue;
            }
            // BPE encode non-special text
            bpe_encode_segment(seg, ids);
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string result;
        for (int id : ids) {
            auto it = id_to_token_.find(id);
            if (it != id_to_token_.end()) {
                result += decode_token(it->second);
            }
        }
        return result;
    }

    std::string decode_token_str(int id) const {
        auto it = id_to_token_.find(id);
        if (it != id_to_token_.end()) return decode_token(it->second);
        return "";
    }

    int get_token_id(const std::string& token) const {
        auto it = token_to_id_.find(token);
        return (it != token_to_id_.end()) ? it->second : -1;
    }

private:
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<int, std::string> id_to_token_;
    std::vector<std::string> merges_;
    std::unordered_map<std::string, int> merge_rank_;
    std::vector<std::pair<std::string, int>> special_tokens_;

    // Byte <-> unicode mapping (GPT-2 BPE uses unicode chars for bytes)
    uint32_t byte_to_char_[256];  // Maps byte value -> unicode codepoint
    std::unordered_map<char32_t, uint8_t> char_to_byte_;

    void build_byte_mapping() {
        // GPT-2 byte encoder: maps bytes to printable unicode characters
        // This matches the HuggingFace tokenizers byte_level pre-tokenizer
        int n = 0;
        for (int b = 0; b < 256; b++) {
            // Printable ASCII + extended range map to themselves
            if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
                byte_to_char_[b] = (uint32_t)b;
            } else {
                // Non-printable bytes map to unicode starting at U+0100
                byte_to_char_[b] = 256 + n;
                n++;
            }
        }
    }

    std::string bytes_to_unicode(const std::string& bytes) const {
        std::string result;
        for (unsigned char b : bytes) {
            uint32_t cp = byte_to_char_[b];
            // UTF-8 encode the codepoint
            if (cp < 0x80) {
                result += (char)cp;
            } else if (cp < 0x800) {
                result += (char)(0xC0 | (cp >> 6));
                result += (char)(0x80 | (cp & 0x3F));
            } else {
                result += (char)(0xE0 | (cp >> 12));
                result += (char)(0x80 | ((cp >> 6) & 0x3F));
                result += (char)(0x80 | (cp & 0x3F));
            }
        }
        return result;
    }

    std::string decode_token(const std::string& token) const {
        // Reverse the byte encoding: each unicode char maps back to a byte
        std::string result;
        size_t i = 0;
        while (i < token.size()) {
            unsigned char c = (unsigned char)token[i];
            uint32_t codepoint;
            if (c < 0x80) {
                codepoint = c;
                i += 1;
            } else if ((c & 0xE0) == 0xC0) {
                codepoint = (c & 0x1F) << 6;
                if (i + 1 < token.size()) codepoint |= ((unsigned char)token[i+1] & 0x3F);
                i += 2;
            } else if ((c & 0xF0) == 0xE0) {
                codepoint = (c & 0x0F) << 12;
                if (i + 1 < token.size()) codepoint |= ((unsigned char)token[i+1] & 0x3F) << 6;
                if (i + 2 < token.size()) codepoint |= ((unsigned char)token[i+2] & 0x3F);
                i += 3;
            } else {
                codepoint = (c & 0x07) << 18;
                if (i + 1 < token.size()) codepoint |= ((unsigned char)token[i+1] & 0x3F) << 12;
                if (i + 2 < token.size()) codepoint |= ((unsigned char)token[i+2] & 0x3F) << 6;
                if (i + 3 < token.size()) codepoint |= ((unsigned char)token[i+3] & 0x3F);
                i += 4;
            }

            // Check if this codepoint is in our byte mapping
            if (codepoint < 256) {
                result += (char)codepoint;
            } else if (codepoint >= 256 && codepoint < 256 + 256) {
                // This is a remapped non-printable byte
                // Find which byte maps to this codepoint
                for (int b = 0; b < 256; b++) {
                    if (byte_to_char_[b] == codepoint) {
                        result += (char)b;
                        break;
                    }
                }
            } else {
                // Pass through as UTF-8
                // Re-encode the codepoint
                if (codepoint < 0x80) {
                    result += (char)codepoint;
                } else if (codepoint < 0x800) {
                    result += (char)(0xC0 | (codepoint >> 6));
                    result += (char)(0x80 | (codepoint & 0x3F));
                } else if (codepoint < 0x10000) {
                    result += (char)(0xE0 | (codepoint >> 12));
                    result += (char)(0x80 | ((codepoint >> 6) & 0x3F));
                    result += (char)(0x80 | (codepoint & 0x3F));
                } else {
                    result += (char)(0xF0 | (codepoint >> 18));
                    result += (char)(0x80 | ((codepoint >> 12) & 0x3F));
                    result += (char)(0x80 | ((codepoint >> 6) & 0x3F));
                    result += (char)(0x80 | (codepoint & 0x3F));
                }
            }
        }

        std::string pattern = "\xe2\x96\x81";
        size_t pos = 0;
        while ((pos = result.find(pattern, pos)) != std::string::npos) {
            result.replace(pos, pattern.length(), " ");
            pos += 1;
        }

        return result;
    }

    void split_on_special(const std::string& text,
                          std::vector<std::pair<std::string, bool>>& out) const {
        size_t pos = 0;
        while (pos < text.size()) {
            bool found = false;
            for (auto& [tok, id] : special_tokens_) {
                if (text.compare(pos, tok.size(), tok) == 0) {
                    if (pos > 0) {
                        // Collect any text before this special token
                    }
                    out.push_back({tok, true});
                    pos += tok.size();
                    found = true;
                    break;
                }
            }
            if (!found) {
                // Accumulate non-special character
                if (out.empty() || out.back().second) {
                    out.push_back({"", false});
                }
                out.back().first += text[pos];
                pos++;
            }
        }
    }

    void bpe_encode_segment(const std::string& text, std::vector<int>& ids) const {
        if (text.empty()) return;

        // Convert to unicode representation
        std::string unicode_text = bytes_to_unicode(text);

        // Simple word-level split (split on Ġ = space in unicode BPE, keeping Ġ with following word)
        // After bytes_to_unicode, space (0x20) becomes Ġ (U+0120) = UTF-8 0xC4 0xA0
        std::vector<std::string> words;
        std::string current;
        for (size_t i = 0; i < unicode_text.size(); i++) {
            // Check for Ġ (UTF-8: 0xC4 0xA0) — the unicode representation of space
            if (i + 1 < unicode_text.size() &&
                (unsigned char)unicode_text[i] == 0xC4 &&
                (unsigned char)unicode_text[i+1] == 0xA0 &&
                !current.empty()) {
                words.push_back(current);
                current.clear();
            }
            current += unicode_text[i];
        }
        if (!current.empty()) words.push_back(current);

        // BPE merge each word
        for (auto& word : words) {
            // Start with character-level tokens
            std::vector<std::string> tokens;
            size_t i = 0;
            while (i < word.size()) {
                unsigned char c = (unsigned char)word[i];
                int char_len = 1;
                if ((c & 0xE0) == 0xC0) char_len = 2;
                else if ((c & 0xF0) == 0xE0) char_len = 3;
                else if ((c & 0xF8) == 0xF0) char_len = 4;
                tokens.push_back(word.substr(i, char_len));
                i += char_len;
            }

            // Iteratively merge
            while (tokens.size() > 1) {
                int best_rank = INT_MAX;
                int best_idx = -1;
                for (size_t j = 0; j + 1 < tokens.size(); j++) {
                    std::string pair = tokens[j] + " " + tokens[j + 1];
                    auto it = merge_rank_.find(pair);
                    if (it != merge_rank_.end() && it->second < best_rank) {
                        best_rank = it->second;
                        best_idx = (int)j;
                    }
                }
                if (best_idx < 0) break;
                tokens[best_idx] = tokens[best_idx] + tokens[best_idx + 1];
                tokens.erase(tokens.begin() + best_idx + 1);
            }

            // Look up token IDs
            for (auto& tok : tokens) {
                auto it = token_to_id_.find(tok);
                if (it != token_to_id_.end()) {
                    ids.push_back(it->second);
                } else {
                    // Fallback: encode as individual bytes
                    for (unsigned char b : tok) {
                        // UTF-8 encode the codepoint for this byte
                        uint32_t cp = byte_to_char_[b];
                        std::string byte_tok;
                        if (cp < 0x80) {
                            byte_tok += (char)cp;
                        } else if (cp < 0x800) {
                            byte_tok += (char)(0xC0 | (cp >> 6));
                            byte_tok += (char)(0x80 | (cp & 0x3F));
                        } else {
                            byte_tok += (char)(0xE0 | (cp >> 12));
                            byte_tok += (char)(0x80 | ((cp >> 6) & 0x3F));
                            byte_tok += (char)(0x80 | (cp & 0x3F));
                        }
                        auto bit = token_to_id_.find(byte_tok);
                        if (bit != token_to_id_.end()) {
                            ids.push_back(bit->second);
                        }
                    }
                }
            }
        }
    }
};

// ════════════════════════════════════════════════════════════════════════════════
//  GPU Tensor: thin wrapper around a CUDA allocation
// ════════════════════════════════════════════════════════════════════════════════

struct GPUTensor {
    void* data = nullptr;
    size_t size_bytes = 0;
    std::vector<int> shape;
    std::string dtype;  // "BF16", "F32", "F8_E4M3", "I8", etc.

    void alloc(size_t bytes) {
        if (data) CUDA_CHECK(cudaFree(data));
        CUDA_CHECK(cudaMalloc(&data, bytes));
        size_bytes = bytes;
    }
    void free() {
        if (data) { cudaFree(data); data = nullptr; size_bytes = 0; }
    }
    ~GPUTensor() { free(); }

    __nv_bfloat16* bf16() { return (__nv_bfloat16*)data; }
    float* f32() { return (float*)data; }
    uint8_t* u8() { return (uint8_t*)data; }
    int32_t* i32() { return (int32_t*)data; }
    int64_t* i64() { return (int64_t*)data; }
};

// ════════════════════════════════════════════════════════════════════════════════
//  O_DIRECT Expert Loader with LRU Cache
// ════════════════════════════════════════════════════════════════════════════════

struct ExpertCacheEntry {
    int layer_id = -1;
    int expert_id = -1;
    int64_t last_used = 0;      // for LRU eviction
    void* gpu_data = nullptr;   // pointer into expert cache pool
    int slot_index = -1;
};

class ExpertLoader {
public:
    int expert_block_size_ = 0;
    int n_layers_ = 0;
    int n_experts_ = 0;
    int expert_fd_ = -1;

    // LRU cache
    int cache_capacity_ = 0;
    void* cache_pool_gpu_ = nullptr;

    // Ring buffer for staging
    static constexpr int NUM_STAGING_BUFFERS = 32;
    struct StagingBuffer {
        void* ptr = nullptr;
        cudaEvent_t event = nullptr;
    };
    std::vector<StagingBuffer> staging_ring_;
    int staging_idx_ = 0;

    std::vector<ExpertCacheEntry> cache_slots_;
    std::unordered_map<int64_t, int> key_to_slot_;  // (layer*n_experts+expert) -> slot index
    int64_t access_counter_ = 0;
    std::mutex cache_mutex_;

    bool init(const std::string& expert_bin_path, int block_size,
              int n_layers, int n_experts, size_t cache_budget_bytes) {
        expert_block_size_ = block_size;
        n_layers_ = n_layers;
        n_experts_ = n_experts;

        expert_fd_ = open(expert_bin_path.c_str(), O_RDONLY | O_DIRECT);
        if (expert_fd_ < 0) {
            LOG_WARN("O_DIRECT not supported, falling back to buffered IO");
            expert_fd_ = open(expert_bin_path.c_str(), O_RDONLY);
        }
        if (expert_fd_ < 0) {
            LOG_ERROR("Cannot open expert bin: %s", expert_bin_path.c_str());
            return false;
        }

        cache_capacity_ = (int)(cache_budget_bytes / block_size);
        if (cache_capacity_ < 64) cache_capacity_ = 64;
        LOG_INFO("Expert cache: %d slots (%.1f GB)", cache_capacity_,
                 (double)cache_capacity_ * block_size / 1e9);

        CUDA_CHECK(cudaMalloc(&cache_pool_gpu_, (size_t)cache_capacity_ * block_size));

        // Initialize staging ring buffer
        staging_ring_.resize(NUM_STAGING_BUFFERS);
        for (int i = 0; i < NUM_STAGING_BUFFERS; i++) {
            CUDA_CHECK(cudaMallocHost(&staging_ring_[i].ptr, block_size));
            CUDA_CHECK(cudaEventCreateWithFlags(&staging_ring_[i].event, cudaEventDisableTiming));
            // Record immediately so first wait passes
            CUDA_CHECK(cudaEventRecord(staging_ring_[i].event, 0));
            
            if ((uintptr_t)staging_ring_[i].ptr % PAGE_SIZE != 0) {
                LOG_WARN("Staging buffer %d is not page-aligned!", i);
            }
        }

        cache_slots_.resize(cache_capacity_);
        for (int i = 0; i < cache_capacity_; i++) {
            cache_slots_[i].slot_index = i;
            cache_slots_[i].gpu_data = (char*)cache_pool_gpu_ + (size_t)i * block_size;
        }

        return true;
    }

    void* get_expert(int layer_id, int expert_id, cudaStream_t stream) {
        int64_t key = (int64_t)layer_id * n_experts_ + expert_id;
        
        std::unique_lock<std::mutex> lock(cache_mutex_);
        access_counter_++;

        auto it = key_to_slot_.find(key);
        if (it != key_to_slot_.end()) {
            cache_slots_[it->second].last_used = access_counter_;
            void* ptr = cache_slots_[it->second].gpu_data;
            lock.unlock();
            return ptr;
        }

        int evict_slot = -1;
        int64_t oldest = INT64_MAX;
        for (int i = 0; i < cache_capacity_; i++) {
            if (cache_slots_[i].layer_id < 0) {
                evict_slot = i;
                break;
            }
            if (cache_slots_[i].last_used < oldest) {
                oldest = cache_slots_[i].last_used;
                evict_slot = i;
            }
        }

        auto& slot = cache_slots_[evict_slot];
        if (slot.layer_id >= 0) {
            if (g_log_experts) {
                LOG_INFO("[ExpertCache] Evicted L%d E%d -> Loaded L%d E%d", 
                         slot.layer_id, slot.expert_id, layer_id, expert_id);
            }
            int64_t old_key = (int64_t)slot.layer_id * n_experts_ + slot.expert_id;
            key_to_slot_.erase(old_key);
        } else {
            if (g_log_experts) {
                LOG_INFO("[ExpertCache] Loaded L%d E%d into free slot %d", layer_id, expert_id, evict_slot);
            }
        }

        slot.layer_id = layer_id;
        slot.expert_id = expert_id;
        slot.last_used = access_counter_;
        key_to_slot_[key] = evict_slot;

        int stage_idx = staging_idx_;
        staging_idx_ = (staging_idx_ + 1) % NUM_STAGING_BUFFERS;
        lock.unlock();

        // Wait for the staging buffer to be free
        auto& stage = staging_ring_[stage_idx];
        CUDA_CHECK(cudaEventSynchronize(stage.event));

        int64_t file_offset = (int64_t)key * expert_block_size_;
        ssize_t bytes_read = pread(expert_fd_, stage.ptr, expert_block_size_, file_offset);
        if (bytes_read != expert_block_size_) {
            LOG_ERROR("Expert read failed: layer=%d expert=%d offset=%ld got=%ld",
                      layer_id, expert_id, file_offset, bytes_read);
            return nullptr;
        }

        // Output to console if enabled:
        if (g_log_experts) {
            printf("\033[36m[CACHE] %s Layer %d Expert %d\033[0m\n", 
                   (evict_slot >= 0 && slot.layer_id >= 0) ? "Evicted & Loaded" : "Loaded", 
                   layer_id, expert_id);
        }

        CUDA_CHECK(cudaMemcpyAsync(slot.gpu_data, stage.ptr, expert_block_size_,
                                    cudaMemcpyHostToDevice, stream));
        
        // Record event on the stream so we know when the memcpy finishes
        CUDA_CHECK(cudaEventRecord(stage.event, stream));

        return slot.gpu_data;
    }

    void cleanup() {
        if (expert_fd_ >= 0) close(expert_fd_);
        if (cache_pool_gpu_) cudaFree(cache_pool_gpu_);
        for (auto& stage : staging_ring_) {
            if (stage.event) cudaEventDestroy(stage.event);
            if (stage.ptr) cudaFreeHost(stage.ptr);
        }
    }
};

// ════════════════════════════════════════════════════════════════════════════════
//  Model Engine
// ════════════════════════════════════════════════════════════════════════════════

class MoecherEngine {
public:
    ModelConfig cfg_;
    BPETokenizer tokenizer_;
    std::string model_dir_;
    
    // Thread pool for loading experts (max 16 concurrent reads)
    std::unique_ptr<ThreadPool> expert_pool_;

    cudaStream_t main_stream_;
    cublasHandle_t cublas_handle_ = nullptr;
    ExpertLoader expert_loader_;

    // ── Resident GPU tensors (loaded at startup) ────────────────────────────

    // Embedding & head
    GPUTensor embed_weight_;   // [vocab_size, hidden_size] BF16
    GPUTensor head_weight_;    // [vocab_size, hidden_size] BF16 (for logits)
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

        // Compressor state buffers (GPU, F32) for incremental decode compression
        GPUTensor comp_kv_state;     // [coff * ratio, coff * head_dim] F32
        GPUTensor comp_score_state;  // [coff * ratio, coff * head_dim] F32

        // Gate
        GPUTensor gate_w;              // [n_experts, hidden] BF16
        GPUTensor gate_bias;           // [n_experts] F32 (null for hash layers)
        GPUTensor tid2eid;             // [vocab_size, top_k] I64 (only hash layers)

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
    };
    std::map<std::string, ExpertPartInfo> expert_parts_;

    // Working buffers (reused across forward passes)
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
    GPUTensor buf_dequant_;      // temp buffer for dequantized weights
    GPUTensor buf_logits_;       // [vocab_size] F32
    GPUTensor buf_scores_f32_;   // [n_experts] F32
    GPUTensor buf_scores_bf16_;  // [n_experts] BF16
    GPUTensor buf_topk_vals_;    // [top_k] F32
    GPUTensor buf_topk_idx_;     // [top_k] I32
    GPUTensor buf_input_ids_;    // [MAX_SEQ_LEN] I32
    GPUTensor buf_hc_state_;     // [hc_mult, hidden_size] BF16 — HC hidden state
    GPUTensor buf_hc_residual_;  // [hc_mult, hidden_size] BF16 — temp for hc_post
    GPUTensor buf_hc_pre_;       // [hc_mult] F32
    GPUTensor buf_hc_post_;      // [hc_mult] F32
    GPUTensor buf_hc_comb_;      // [hc_mult*hc_mult] F32
    GPUTensor buf_hc_mixes_;     // [(2+hc)*hc] F32
    GPUTensor buf_hc_input_;     // [hc_mult*hidden_size] F32 — flattened HC state

    // Compressor working buffers
    GPUTensor buf_comp_proj_;    // F32 working buffer for compressor projections
    GPUTensor buf_comp_out_;     // F32 working buffer for compressor pooling output
    GPUTensor buf_comp_bf16_;    // BF16 working buffer for compressed entry (head_dim)
    GPUTensor buf_combined_kv_;  // BF16 buffer for combined raw+compressed KV

    // Dequant buffer — large enough for the biggest weight matrix
    static constexpr size_t DEQUANT_BUF_SIZE = 64 * 1024 * 1024;  // 64 MB

    // ── Load model from manifest ────────────────────────────────────────────

    bool load(const std::string& manifest_path, float max_vram_gb = 0.0f) {
        LOG_INFO("Loading manifest: %s", manifest_path.c_str());

        std::ifstream f(manifest_path);
        if (!f.is_open()) { LOG_ERROR("Cannot open manifest"); return false; }
        json manifest;
        f >> manifest;

        // Parse config
        cfg_.from_json(manifest["model_config"]);
        LOG_INFO("Model: %d layers, %d experts, %d active, hidden=%d",
                 cfg_.num_hidden_layers, cfg_.n_routed_experts,
                 cfg_.num_experts_per_tok, cfg_.hidden_size);

        // Load tokenizer
        std::string tok_path = manifest["tokenizer"]["tokenizer_json"].get<std::string>();
        if (!tokenizer_.load(tok_path)) return false;

        // Init CUDA
        CUDA_CHECK(cudaStreamCreate(&main_stream_));
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
        CUBLAS_CHECK(cublasSetStream(cublas_handle_, main_stream_));
        cublasSetMathMode(cublas_handle_, CUBLAS_DEFAULT_MATH);

        expert_pool_ = std::make_unique<ThreadPool>(16);

        // Determine available VRAM for expert cache
        size_t vram_free, vram_total;
        CUDA_CHECK(cudaMemGetInfo(&vram_free, &vram_total));
        LOG_INFO("VRAM: %.1f GB free / %.1f GB total",
                 vram_free / 1e9, vram_total / 1e9);

        // Load dense tensors from attention_dense_layers.bin
        std::string dense_path = manifest["dense_bin"].get<std::string>();
        if (!load_dense_tensors(dense_path, manifest["dense_tensors"])) return false;

        // Load expert layout info
        auto& el = manifest["expert_layout"];
        int expert_block_size = el["block_size"].get<int>();
        int expert_n_layers = el["n_layers"].get<int>();
        int expert_n_experts = el["n_experts"].get<int>();

        for (auto& [part_name, part_info] : el["parts"].items()) {
            ExpertPartInfo epi;
            epi.offset_in_block = part_info["offset_in_block"].get<int>();
            epi.nbytes = part_info["nbytes"].get<int>();
            epi.dtype = part_info["dtype"].get<std::string>();
            for (auto& s : part_info["shape"]) epi.shape.push_back(s.get<int>());
            expert_parts_[part_name] = epi;
        }

        // Init expert loader with O_DIRECT
        std::string expert_path = manifest["expert_bin"].get<std::string>();

        // Reserve VRAM: estimate resident usage, rest goes to expert cache
        CUDA_CHECK(cudaMemGetInfo(&vram_free, &vram_total));
        size_t cache_budget = vram_free - 2ULL * 1024 * 1024 * 1024;  // leave 2 GB headroom
        
        if (max_vram_gb > 0.0f) {
            size_t max_vram_bytes = (size_t)(max_vram_gb * 1024.0 * 1024.0 * 1024.0);
            size_t used_vram = vram_total - vram_free;
            if (max_vram_bytes > used_vram + 2ULL * 1024 * 1024 * 1024) {
                size_t user_budget = max_vram_bytes - used_vram - 2ULL * 1024 * 1024 * 1024;
                if (user_budget < cache_budget) {
                    cache_budget = user_budget;
                }
            } else {
                cache_budget = 0;
            }
        }
        
        LOG_INFO("Expert cache budget: %.1f GB", cache_budget / 1e9);
        if (!expert_loader_.init(expert_path, expert_block_size,
                                  expert_n_layers, expert_n_experts,
                                  cache_budget)) return false;

        // Allocate working buffers
        alloc_buffers();

        // Precompute RoPE frequencies — two tables for non-compressed vs compressed layers
        // DS4 reference: non-compressed layers use base freq without YaRN interpolation;
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
        LOG_INFO("Model loaded successfully");
        return true;
    }

    // ── Forward pass for a single token (decode mode) ───────────────────────
    // Returns logits on GPU [vocab_size] in F32

    void dump_bf16(const char* label, __nv_bfloat16* gpu_ptr, int count, int show = 8) {
        std::vector<__nv_bfloat16> buf(count);
        CUDA_CHECK(cudaStreamSynchronize(main_stream_));
        CUDA_CHECK(cudaMemcpy(buf.data(), gpu_ptr, count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        std::string s = "  " + std::string(label) + ": [";
        for (int i = 0; i < std::min(show, count); i++) {
            char tmp[32]; snprintf(tmp, sizeof(tmp), "%.6f", __bfloat162float(buf[i]));
            if (i > 0) s += ", ";
            s += tmp;
        }
        s += "]";
        float norm = 0;
        for (int i = 0; i < count; i++) { float v = __bfloat162float(buf[i]); norm += v*v; }
        char tmp[64]; snprintf(tmp, sizeof(tmp), " norm=%.6f", sqrtf(norm));
        s += tmp;
        LOG_INFO("%s", s.c_str());
    }

    // Debug flag — reset before each generate() call
    bool dbg_first_token_ = true;
    int dbg_hc_pre_call_ = 0;
    bool dbg_head_ = true;
    int dbg_sample_count_ = 0;

    // Persistent RNG for sampling (seeded once)
    std::mt19937 rng_{std::random_device{}()};

    void forward_token(int token_id, int position) {
        int dim = cfg_.hidden_size;
        int n_heads = cfg_.num_attention_heads;
        int head_dim_val = cfg_.head_dim;
        int rope_dim = cfg_.qk_rope_head_dim;
        int q_lora = cfg_.q_lora_rank;
        int o_lora = cfg_.o_lora_rank;
        int o_groups = cfg_.o_groups;
        int window = cfg_.sliding_window;
        int hc = cfg_.hc_mult;

        // 1. Embedding lookup
        int32_t tid_host = token_id;
        CUDA_CHECK(cudaMemcpyAsync(buf_input_ids_.i32(), &tid_host, sizeof(int32_t),
                                    cudaMemcpyHostToDevice, main_stream_));
        embedding_cuda(buf_hidden_.bf16(), embed_weight_.bf16(),
                       buf_input_ids_.i32(), 1, dim, main_stream_);

        if (dbg_first_token_) {
            LOG_INFO("DEBUG: token_id=%d position=%d", token_id, position);
            dump_bf16("embed", buf_hidden_.bf16(), dim);
        }

        // 2. Expand to HC copies: [1, dim] -> [hc, dim]
        // Simply replicate the hidden state hc times
        // (HC state accumulates across layers for this token, not across tokens)
        for (int h = 0; h < hc; h++) {
            CUDA_CHECK(cudaMemcpyAsync(
                buf_hc_state_.bf16() + (size_t)h * dim,
                buf_hidden_.bf16(), dim * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToDevice, main_stream_));
        }
        // Sync to ensure HC state is ready before layer processing
        CUDA_CHECK(cudaStreamSynchronize(main_stream_));

        // 3. Process each layer
        for (int layer = 0; layer < cfg_.num_hidden_layers; layer++) {
            forward_layer(layer, token_id, position);
        }

        // 4. Head HC: reduce [hc, dim] -> [dim]
        hc_head_reduce();

        if (dbg_first_token_) dump_bf16("after_hc_head", buf_hidden_.bf16(), dim);

        // 5. Final norm
        rms_norm_cuda(buf_hidden_.bf16(), buf_hidden_.bf16(),
                      norm_weight_.bf16(), dim, cfg_.rms_norm_eps, main_stream_);

        if (dbg_first_token_) dump_bf16("after_final_norm", buf_hidden_.bf16(), dim);

        // 6. Logits: hidden @ head_weight.T -> [vocab_size]
        compute_logits();

        dbg_first_token_ = false;
    }

    // ── Generate tokens ─────────────────────────────────────────────────────

    std::string generate(const std::string& prompt, int max_tokens = 512,
                         float temperature = 0.0f,
                         std::function<void(const std::string&)> on_token = nullptr,
                         float repetition_penalty = 1.1f) {
        // Reset debug flags for this request
        dbg_first_token_ = true;
        dbg_hc_pre_call_ = 0;
        dbg_head_ = true;
        dbg_sample_count_ = 0;

        // Reset KV caches for all layers (critical: stale cache = garbled output)
        for (int l = 0; l < cfg_.num_hidden_layers; l++) {
            CUDA_CHECK(cudaMemset(layers_[l].kv_cache.data, 0,
                                   layers_[l].kv_cache.size_bytes));
            // Reset compressor state for compressed layers
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
            }
        }

        // Reset HC (Hierarchical Compressor) routing state.
        // This is critical: stale HC state from a previous request causes
        // the expert routing to be biased by the old conversation, leading
        // to garbled output on subsequent turns.
        if (buf_hc_state_.data) {
            CUDA_CHECK(cudaMemset(buf_hc_state_.data, 0, buf_hc_state_.size_bytes));
        }

        // Store repetition_penalty for use in sample_token
        current_rep_penalty_ = repetition_penalty;

        // Tokenize
        std::vector<int> input_ids = tokenizer_.encode(prompt);
        LOG_INFO("Prompt tokens: %zu", input_ids.size());
        // Log first 10 token IDs for debugging
        std::string ids_str;
        for (size_t i = 0; i < std::min(input_ids.size(), (size_t)10); i++) {
            if (i > 0) ids_str += ", ";
            ids_str += std::to_string(input_ids[i]);
        }
        LOG_INFO("First tokens: [%s]", ids_str.c_str());

        if (g_log_tokens) {
            printf("\n");
        }

        std::vector<int> output_ids;

        // Prefill: process all prompt tokens
        for (size_t i = 0; i < input_ids.size(); i++) {
            forward_token(input_ids[i], (int)i);
        }

        // Repetition history: only generated tokens (not prompt tokens).
        // Seeding with prompt tokens was tried but it incorrectly penalizes
        // tokens from the user's current query (e.g., "Amiga").
        std::vector<int> history;

        // Decode: generate tokens one by one
        int position = (int)input_ids.size();
        std::string token_buffer;
        std::string generated_text;  // Track full output for sentence-boundary stopping
        std::string finish_reason = "stop";

        // Think-block filtering state
        // DeepSeek V4 generates <think>...</think> before the actual response
        bool in_think_block = false;
        bool think_block_ended = false;
        std::string think_detect_buffer;
        // Get token IDs for think tags
        int think_start_id = tokenizer_.get_token_id("<think>");
        int think_end_id = tokenizer_.get_token_id("</think>");
        // Also check with special token format
        if (think_start_id < 0) think_start_id = tokenizer_.get_token_id("\xef\xbd\x9c" "think" "\xef\xbd\x9c");
        if (think_end_id < 0) think_end_id = tokenizer_.get_token_id("\xef\xbd\x9c" "/think" "\xef\xbd\x9c");

        // Additional EOS tokens for DeepSeek V4
        int eos2_id = tokenizer_.get_token_id(
            "<\xef\xbd\x9c" "end\xe2\x96\x81of\xe2\x96\x81sentence" "\xef\xbd\x9c>");

        LOG_INFO("Think tokens: start=%d end=%d, EOS=%d eos2=%d",
                 think_start_id, think_end_id, cfg_.eos_token_id, eos2_id);

        // Track how many content tokens (non-think) we've generated
        int content_tokens_generated = 0;

        for (int t = 0; t < max_tokens; t++) {
            // Sample from logits, suppressing EOS for the first few tokens
            int next_token = sample_token(temperature, history, content_tokens_generated);

            // Check all EOS conditions
            if (next_token == cfg_.eos_token_id ||
                (eos2_id >= 0 && next_token == eos2_id)) {
                LOG_INFO("EOS hit: token=%d (cfg_eos=%d, eos2=%d) at step %d/%d",
                         next_token, cfg_.eos_token_id, eos2_id, t, max_tokens);
                break;
            }

            output_ids.push_back(next_token);
            history.push_back(next_token);
            content_tokens_generated++;

            // Handle think block filtering
            if (think_start_id >= 0 && next_token == think_start_id) {
                in_think_block = true;
                // Forward the token but don't emit it
                forward_token(next_token, position);
                position++;
                continue;
            }
            if (think_end_id >= 0 && next_token == think_end_id) {
                in_think_block = false;
                think_block_ended = true;
                // Forward the token but don't emit it
                forward_token(next_token, position);
                position++;
                continue;
            }
            if (in_think_block) {
                // Inside think block — forward but don't emit
                forward_token(next_token, position);
                position++;
                continue;
            }

            std::string token_text = tokenizer_.decode({next_token});

            // If we just exited a think block, skip leading newlines
            if (think_block_ended && !token_text.empty()) {
                size_t start = token_text.find_first_not_of("\n\r");
                if (start == std::string::npos) {
                    // All whitespace, skip
                    forward_token(next_token, position);
                    position++;
                    continue;
                }
                if (start > 0) token_text = token_text.substr(start);
                think_block_ended = false;
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
                if (g_log_tokens) {
                    printf("%s", token_buffer.c_str());
                    fflush(stdout);
                }
                if (on_token) {
                    on_token(token_buffer);
                }
                token_buffer.clear();
            }

            // Forward the new token
            forward_token(next_token, position);
            position++;

            // Graceful sentence-boundary stopping:
            // When we're within 80% of max_tokens and just completed a sentence,
            // stop early to avoid truncating mid-thought.
            if (content_tokens_generated >= (max_tokens * 4 / 5)) {
                // Check if the generated text ends at a sentence boundary
                if (!generated_text.empty()) {
                    char last_char = generated_text.back();
                    // Also check for sentence-ending after whitespace
                    size_t last_non_ws = generated_text.find_last_not_of(" \t\n\r");
                    if (last_non_ws != std::string::npos) {
                        last_char = generated_text[last_non_ws];
                    }
                    if (last_char == '.' || last_char == '!' || last_char == '?' ||
                        last_char == '\n') {
                        LOG_INFO("Graceful stop at sentence boundary (token %d/%d)",
                                 content_tokens_generated, max_tokens);
                        finish_reason = "stop";
                        break;
                    }
                }
            }
        }

        // Check if we hit max_tokens
        if ((int)output_ids.size() >= max_tokens) {
            finish_reason = "length";
        }

        // Flush anything remaining in the buffer
        if (!token_buffer.empty()) {
            if (g_log_tokens) {
                printf("%s", token_buffer.c_str());
                fflush(stdout);
            }
            if (on_token) {
                on_token(token_buffer);
            }
        }

        if (g_log_tokens) {
            printf("\n");
        }

        // Decode only non-think output tokens for the result
        // Filter think tokens from output_ids
        std::vector<int> visible_ids;
        bool skip = false;
        for (int id : output_ids) {
            if (think_start_id >= 0 && id == think_start_id) { skip = true; continue; }
            if (think_end_id >= 0 && id == think_end_id) { skip = false; continue; }
            if (!skip) visible_ids.push_back(id);
        }
        std::string result = tokenizer_.decode(visible_ids);
        // Trim leading whitespace from result (after think block)
        size_t first_non_ws = result.find_first_not_of("\n\r ");
        if (first_non_ws != std::string::npos && first_non_ws > 0) {
            result = result.substr(first_non_ws);
        }

        prompt_token_count_ = (int)input_ids.size();
        completion_token_count_ = (int)output_ids.size();
        last_finish_reason_ = finish_reason;

        return result;
    }

    // Per-request stats (set by generate())
    int prompt_token_count_ = 0;
    int completion_token_count_ = 0;
    std::string last_finish_reason_ = "stop";

private:
    // ── Dense tensor loading ────────────────────────────────────────────────

    bool load_dense_tensors(const std::string& dense_path,
                            const json& tensor_map) {
        LOG_INFO("Loading dense tensors from %s", dense_path.c_str());

        int fd = open(dense_path.c_str(), O_RDONLY);
        if (fd < 0) { LOG_ERROR("Cannot open dense bin"); return false; }

        struct stat st;
        fstat(fd, &st);
        size_t file_size = st.st_size;

        // mmap the entire dense bin for easy access
        void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) { LOG_ERROR("mmap failed"); close(fd); return false; }

        auto load_tensor = [&](GPUTensor& gpu, const std::string& name) -> bool {
            if (!tensor_map.contains(name)) {
                LOG_WARN("Tensor not found in manifest: %s", name.c_str());
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

        // Load global tensors
        load_tensor(embed_weight_, "embed.weight");
        load_tensor(head_weight_, "head.weight");
        load_tensor(norm_weight_, "norm.weight");

        // Head HC
        load_tensor(hc_head_fn_, "hc_head_fn");
        load_tensor(hc_head_base_, "hc_head_base");
        load_tensor(hc_head_scale_, "hc_head_scale");

        // Load per-layer tensors
        layers_.resize(cfg_.num_hidden_layers);
        for (int l = 0; l < cfg_.num_hidden_layers; l++) {
            auto& lw = layers_[l];
            std::string prefix = "layers." + std::to_string(l);

            load_tensor(lw.wq_a_w, prefix + ".attn.wq_a.weight");
            load_tensor(lw.wq_a_s, prefix + ".attn.wq_a.scale");
            load_tensor(lw.wq_b_w, prefix + ".attn.wq_b.weight");
            load_tensor(lw.wq_b_s, prefix + ".attn.wq_b.scale");
            load_tensor(lw.wkv_w, prefix + ".attn.wkv.weight");
            load_tensor(lw.wkv_s, prefix + ".attn.wkv.scale");
            load_tensor(lw.wo_a_w, prefix + ".attn.wo_a.weight");
            load_tensor(lw.wo_a_s, prefix + ".attn.wo_a.scale");
            load_tensor(lw.wo_b_w, prefix + ".attn.wo_b.weight");
            load_tensor(lw.wo_b_s, prefix + ".attn.wo_b.scale");
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

                // Allocate compressed KV cache
                int max_comp = cfg_.sliding_window / ratio;
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
            }

            // Gate
            load_tensor(lw.gate_w, prefix + ".ffn.gate.weight");
            if (l < cfg_.n_hash_layers) {
                load_tensor(lw.tid2eid, prefix + ".ffn.gate.tid2eid");
            } else {
                load_tensor(lw.gate_bias, prefix + ".ffn.gate.bias");
            }

            // Shared expert (FP8)
            load_tensor(lw.shared_w1_w, prefix + ".ffn.shared_experts.w1.weight");
            load_tensor(lw.shared_w1_s, prefix + ".ffn.shared_experts.w1.scale");
            load_tensor(lw.shared_w2_w, prefix + ".ffn.shared_experts.w2.weight");
            load_tensor(lw.shared_w2_s, prefix + ".ffn.shared_experts.w2.scale");
            load_tensor(lw.shared_w3_w, prefix + ".ffn.shared_experts.w3.weight");
            load_tensor(lw.shared_w3_s, prefix + ".ffn.shared_experts.w3.scale");

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

        munmap(mapped, file_size);
        close(fd);
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

        buf_hidden_.alloc(dim * sizeof(__nv_bfloat16));
        buf_hidden2_.alloc(dim * sizeof(__nv_bfloat16));
        buf_q_.alloc((size_t)n_heads * head_dim_val * sizeof(__nv_bfloat16));
        buf_kv_.alloc(head_dim_val * sizeof(__nv_bfloat16));
        buf_attn_out_.alloc((size_t)n_heads * head_dim_val * sizeof(__nv_bfloat16));
        buf_lora_.alloc(std::max({
            (size_t)cfg_.q_lora_rank,
            (size_t)cfg_.o_lora_rank * cfg_.o_groups,
            (size_t)n_heads * head_dim_val
        }) * sizeof(__nv_bfloat16));
        buf_gate_.alloc(moe_inter * sizeof(__nv_bfloat16));
        buf_up_.alloc(moe_inter * sizeof(__nv_bfloat16));
        buf_down_.alloc(dim * sizeof(__nv_bfloat16));
        buf_expert_out_.alloc(dim * sizeof(__nv_bfloat16));
        buf_moe_accum_.alloc(dim * sizeof(__nv_bfloat16));
        buf_dequant_.alloc(128 * 1024 * 1024);  // 128 MB for largest dequant
        buf_logits_.alloc((size_t)cfg_.vocab_size * sizeof(float));
        buf_scores_f32_.alloc(cfg_.n_routed_experts * sizeof(float));
        buf_scores_bf16_.alloc(cfg_.n_routed_experts * sizeof(__nv_bfloat16));
        buf_topk_vals_.alloc(cfg_.num_experts_per_tok * sizeof(float));
        buf_topk_idx_.alloc(cfg_.num_experts_per_tok * sizeof(int32_t));
        buf_input_ids_.alloc(sizeof(int32_t));
        buf_hc_state_.alloc((size_t)hc * dim * sizeof(__nv_bfloat16));
        buf_hc_residual_.alloc((size_t)hc * dim * sizeof(__nv_bfloat16));  // persistent temp for hc_post
        buf_hc_pre_.alloc(hc * sizeof(float));
        buf_hc_post_.alloc(hc * sizeof(float));
        buf_hc_comb_.alloc(hc * hc * sizeof(float));
        buf_hc_mixes_.alloc((2 + hc) * hc * sizeof(float));
        buf_hc_input_.alloc((size_t)hc * dim * sizeof(float));

        // Compressor working buffers
        // Max projection output size: coff=2 for ratio=4, head_dim=512 -> 1024
        int max_comp_dim = 2 * head_dim_val;  // coff=2
        buf_comp_proj_.alloc(max_comp_dim * sizeof(float));  // for wkv or wgate output
        buf_comp_out_.alloc(max_comp_dim * sizeof(float));   // for pooling output
        buf_comp_bf16_.alloc(head_dim_val * sizeof(__nv_bfloat16));  // compressed entry in BF16
        // Combined KV buffer: raw window + max compressed entries
        int max_combined = cfg_.sliding_window + cfg_.max_compressed_entries;
        buf_combined_kv_.alloc((size_t)max_combined * head_dim_val * sizeof(__nv_bfloat16));
    }

    // ── cuBLAS GEMM helper (BF16) ──────────────────────────────────────────
    // C = A @ B.T  where A is [M, K], B is [N, K] (row-major), C is [M, N]
    // cuBLAS uses column-major, so we compute B.T @ A.T = (A @ B.T).T

    void gemm_bf16(
        __nv_bfloat16* C, int M, int N, int K,
        const __nv_bfloat16* A,  // [M, K]
        const __nv_bfloat16* B,  // [N, K] — stored as weight[out, in]
        float alpha = 1.0f, float beta = 0.0f)
    {
        // In column-major: A_col = K x M, B_col = K x N
        // cuBLAS: C_col = B_col.T @ A_col = N x K @ K x M = N x M
        // Which is C_row = M x N ✓
        CUBLAS_CHECK(cublasGemmEx(
            cublas_handle_,
            CUBLAS_OP_T,    // B transposed (K x N -> N x K)
            CUBLAS_OP_N,    // A not transposed
            N, M, K,        // m=N, n=M, k=K in cuBLAS terms
            &alpha,
            B, CUDA_R_16BF, K,    // lda = K
            A, CUDA_R_16BF, K,    // ldb = K
            &beta,
            C, CUDA_R_16BF, N,    // ldc = N
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT));
    }

    // ── Dequant + GEMM: dequantize FP8 weight to BF16 temp, then GEMM ──────

    void gemm_fp8_dequant(
        __nv_bfloat16* C, int M, int N, int K,
        const __nv_bfloat16* A,      // [M, K] BF16 input
        const uint8_t* weight,       // [N, K] FP8 E4M3
        const uint8_t* scale,        // [ceil(N/128), ceil(K/128)] E8M0
        int block_size = 128)
    {
        // Dequantize weight to BF16 in buf_dequant_
        fp8_dequant_cuda(buf_dequant_.bf16(), weight, scale, N, K, block_size, main_stream_);
        // GEMM with dequantized weight
        gemm_bf16(C, M, N, K, A, buf_dequant_.bf16());
    }

    // ── Dequant + GEMM for FP4 experts ──────────────────────────────────────

    void gemm_fp4_dequant(
        __nv_bfloat16* C, int M, int N, int K_logical,
        const __nv_bfloat16* A,      // [M, K_logical] BF16
        const uint8_t* weight,       // [N, K_logical/2] packed FP4
        const uint8_t* scale,        // [N, K_logical/32] E8M0
        int scale_cols)
    {
        int K_packed = K_logical / 2;
        fp4_dequant_cuda(buf_dequant_.bf16(), weight, scale, N, K_packed, scale_cols, main_stream_);
        gemm_bf16(C, M, N, K_logical, A, buf_dequant_.bf16());
    }

    // ── KV Compressor Forward ───────────────────────────────────────────────
    // Implements gated pooling compression for CSA (ratio=4, overlap) and
    // HCA (ratio=128, non-overlap) layers following the DeepSeek V4 paper.
    //
    // Called once per token per compressed layer. Accumulates kv/score state
    // and emits a compressed entry every `ratio` tokens.

    void forward_compressor(int layer_id, int position) {
        auto& lw = layers_[layer_id];
        int ratio = cfg_.layer_compress_ratio(layer_id);
        if (ratio <= 0) return;

        int dim = cfg_.hidden_size;
        int head_dim_val = cfg_.head_dim;
        int rope_dim = cfg_.qk_rope_head_dim;
        bool overlap = (ratio == 4);
        int coff = overlap ? 2 : 1;
        int proj_dim = coff * head_dim_val;  // output dim of wkv/wgate
        int state_rows = coff * ratio;  // total rows in state: 4 for HCA, 8 for CSA

        // 1. Project hidden state through compressor wkv: [proj_dim] = wkv @ hidden
        gemv_bf16_cuda(buf_comp_proj_.f32(), lw.comp_wkv.bf16(),
                       buf_hidden_.bf16(), proj_dim, dim, main_stream_);

        // State index: cycles through all state_rows slots via modular arithmetic
        // For CSA (coff=2, ratio=4): position % 8 cycles 0,1,2,3,4,5,6,7,0,...
        //   Block N writes to the first half (0-3) or second half (4-7)
        // For HCA (coff=1, ratio=128): position % 128 cycles 0,1,...,127,0,...
        int state_idx = position % state_rows;

        // Copy kv projection to state slot
        CUDA_CHECK(cudaMemcpyAsync(
            lw.comp_kv_state.f32() + (size_t)state_idx * proj_dim,
            buf_comp_proj_.f32(), proj_dim * sizeof(float),
            cudaMemcpyDeviceToDevice, main_stream_));

        // 2. Project hidden state through compressor wgate: [proj_dim]
        gemv_bf16_cuda(buf_comp_out_.f32(), lw.comp_wgate.bf16(),
                       buf_hidden_.bf16(), proj_dim, dim, main_stream_);

        // 3. Add APE bias to gate scores
        // APE shape: [ratio, proj_dim], row = position % ratio
        int ape_row = position % ratio;
        CUDA_CHECK(cudaMemcpyAsync(
            lw.comp_score_state.f32() + (size_t)state_idx * proj_dim,
            buf_comp_out_.f32(), proj_dim * sizeof(float),
            cudaMemcpyDeviceToDevice, main_stream_));

        // Add APE bias: score_state[state_idx] += ape[ape_row]
        float alpha_one = 1.0f;
        float* score_ptr = lw.comp_score_state.f32() + (size_t)state_idx * proj_dim;
        const float* ape_ptr = lw.comp_ape.f32() + (size_t)ape_row * proj_dim;
        CUBLAS_CHECK(cublasSaxpy(cublas_handle_, proj_dim, &alpha_one,
                                 ape_ptr, 1, score_ptr, 1));

        // 4. Check if we have a complete block to compress
        bool should_compress = ((position + 1) % ratio == 0);
        if (!should_compress) return;

        // 5. Perform softmax-gated pooling
        if (overlap) {
            // CSA overlap: the state naturally has data from two consecutive blocks
            // thanks to modular cycling.
            //
            // At compression time (position 3, 7, 11, ...):
            // - One half of state (rows 0-3 or rows 4-7) has the CURRENT block
            // - The other half has the PREVIOUS block (or is empty/-inf for first block)
            //
            // Reference logic:
            //   first_half  = state[:ratio, :head_dim]    (rows 0-3, first head_dim dims)
            //   second_half = state[ratio:, head_dim:]    (rows 4-7, second head_dim dims)
            //   pool_input  = cat(first_half, second_half) → [2*ratio, head_dim]
            //   pool_score  = cat(score_first_half, score_second_half) → [2*ratio, head_dim]

            // Use buf_dequant_ as temp workspace
            float* tmp_kv = (float*)buf_dequant_.data;
            float* tmp_score = tmp_kv + 2 * ratio * head_dim_val;

            // First half: rows 0..ratio-1, take first head_dim dims
            for (int i = 0; i < ratio; i++) {
                CUDA_CHECK(cudaMemcpyAsync(
                    tmp_kv + (size_t)i * head_dim_val,
                    lw.comp_kv_state.f32() + (size_t)i * proj_dim,
                    head_dim_val * sizeof(float), cudaMemcpyDeviceToDevice, main_stream_));
                CUDA_CHECK(cudaMemcpyAsync(
                    tmp_score + (size_t)i * head_dim_val,
                    lw.comp_score_state.f32() + (size_t)i * proj_dim,
                    head_dim_val * sizeof(float), cudaMemcpyDeviceToDevice, main_stream_));
            }
            // Second half: rows ratio..2*ratio-1, take second head_dim dims (offset by head_dim)
            for (int i = 0; i < ratio; i++) {
                CUDA_CHECK(cudaMemcpyAsync(
                    tmp_kv + (size_t)(ratio + i) * head_dim_val,
                    lw.comp_kv_state.f32() + (size_t)(ratio + i) * proj_dim + head_dim_val,
                    head_dim_val * sizeof(float), cudaMemcpyDeviceToDevice, main_stream_));
                CUDA_CHECK(cudaMemcpyAsync(
                    tmp_score + (size_t)(ratio + i) * head_dim_val,
                    lw.comp_score_state.f32() + (size_t)(ratio + i) * proj_dim + head_dim_val,
                    head_dim_val * sizeof(float), cudaMemcpyDeviceToDevice, main_stream_));
            }

            // Pool: softmax over 2*ratio rows, weighted sum -> [head_dim]
            compressor_pool_cuda(buf_comp_out_.f32(), tmp_kv, tmp_score,
                                 2 * ratio, head_dim_val, main_stream_);

            // No carry-over needed: state naturally cycles via position % (coff*ratio)
        } else {
            // HCA non-overlapping: pool over ratio rows
            compressor_pool_cuda(buf_comp_out_.f32(),
                                 lw.comp_kv_state.f32(),
                                 lw.comp_score_state.f32(),
                                 ratio, head_dim_val, main_stream_);

            // Reset state for next block (HCA has no overlap, clean slate)
            CUDA_CHECK(cudaMemsetAsync(lw.comp_kv_state.data, 0,
                                       lw.comp_kv_state.size_bytes, main_stream_));
            std::vector<float> neg_inf(ratio * head_dim_val, -std::numeric_limits<float>::infinity());
            CUDA_CHECK(cudaMemcpyAsync(lw.comp_score_state.data, neg_inf.data(),
                                       neg_inf.size() * sizeof(float), cudaMemcpyHostToDevice, main_stream_));
        }

        // 6. Convert pooled output to BF16 and apply RMSNorm
        f32_to_bf16_cuda(buf_comp_bf16_.bf16(), buf_comp_out_.f32(),
                         head_dim_val, main_stream_);
        rms_norm_cuda(buf_comp_bf16_.bf16(), buf_comp_bf16_.bf16(),
                      lw.comp_norm.bf16(), head_dim_val, cfg_.rms_norm_eps, main_stream_);

        // 7. Apply RoPE to compressed entry using compressed-layer frequencies
        int comp_pos = position;
        rope_cuda(buf_comp_bf16_.bf16(), 1, head_dim_val, rope_dim,
                  comp_pos, rope_freqs_compressed_.f32(), false, main_stream_);

        // 8. Store in compressed KV cache
        int comp_idx = lw.comp_kv_count;
        int max_comp = cfg_.sliding_window / ratio;
        if (comp_idx < max_comp) {
            CUDA_CHECK(cudaMemcpyAsync(
                lw.comp_kv_cache.bf16() + (size_t)comp_idx * head_dim_val,
                buf_comp_bf16_.bf16(), head_dim_val * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToDevice, main_stream_));
            lw.comp_kv_count = comp_idx + 1;
        }
    }

    // ── Forward one layer ───────────────────────────────────────────────────

    void forward_layer(int layer_id, int token_id, int position) {
        auto& lw = layers_[layer_id];
        int dim = cfg_.hidden_size;
        int hc = cfg_.hc_mult;
        bool dbg = (layer_id == 0 && position == 0);

        // ── HC pre for attention ──
        hc_pre(lw.hc_attn_fn, lw.hc_attn_scale, lw.hc_attn_base);
        if (dbg) dump_bf16("L0 hc_pre_attn", buf_hidden_.bf16(), dim);

        // ── Attention norm ──
        rms_norm_cuda(buf_hidden_.bf16(), buf_hidden_.bf16(),
                      lw.attn_norm_w.bf16(), dim, cfg_.rms_norm_eps, main_stream_);
        if (dbg) dump_bf16("L0 attn_normed", buf_hidden_.bf16(), dim);

        // ── Attention ──
        forward_attention(layer_id, position);
        if (dbg) dump_bf16("L0 attn_out", buf_hidden_.bf16(), dim);

        // ── HC post for attention ──
        hc_post();
        if (dbg) dump_bf16("L0 hc_post_attn[0]", buf_hc_state_.bf16(), dim);

        // ── HC pre for FFN ──
        hc_pre(lw.hc_ffn_fn, lw.hc_ffn_scale, lw.hc_ffn_base);

        // ── FFN norm ──
        rms_norm_cuda(buf_hidden_.bf16(), buf_hidden_.bf16(),
                      lw.ffn_norm_w.bf16(), dim, cfg_.rms_norm_eps, main_stream_);

        // ── MoE FFN ──
        forward_moe(layer_id, token_id);
        if (dbg) dump_bf16("L0 moe_out", buf_hidden_.bf16(), dim);

        // ── HC post for FFN ──
        hc_post();
        if (dbg) dump_bf16("L0 hc_post_ffn[0]", buf_hc_state_.bf16(), dim);
    }

    // ── HC pre: reduce [hc, dim] -> [dim] ───────────────────────────────────

    void hc_pre(GPUTensor& hc_fn, GPUTensor& hc_scale, GPUTensor& hc_base) {
        int dim = cfg_.hidden_size;
        int hc = cfg_.hc_mult;
        int mix_size = (2 + hc) * hc;
        int hc_dim = hc * dim;

        // Flatten HC state to F32: [hc*dim]
        bf16_to_f32_cuda(buf_hc_input_.f32(), buf_hc_state_.bf16(), hc_dim, main_stream_);

        // Compute RMS of flattened state for normalization
        // rsqrt = rsqrt(mean(x^2) + eps)
        // This is done as part of the linear projection

        // mixes = linear(x_flat, hc_fn) * rsqrt
        // hc_fn: [mix_size, hc_dim] F32
        // x_flat: [1, hc_dim] F32
        // mixes: [1, mix_size] F32

        // We need F32 GEMM here. Use cuBLAS with F32.
        {
            float alpha = 1.0f, beta = 0.0f;

            // First compute rsqrt of RMS
            // For simplicity, compute on CPU (small data transfer)
            std::vector<float> x_flat(hc_dim);
            CUDA_CHECK(cudaMemcpy(x_flat.data(), buf_hc_input_.f32(),
                                   hc_dim * sizeof(float), cudaMemcpyDeviceToHost));

            float sum_sq = 0;
            for (int i = 0; i < hc_dim; i++) sum_sq += x_flat[i] * x_flat[i];
            float rsqrt_val = 1.0f / sqrtf(sum_sq / hc_dim + cfg_.hc_eps);

            // Scale x by rsqrt
            for (auto& v : x_flat) v *= rsqrt_val;
            CUDA_CHECK(cudaMemcpy(buf_hc_input_.f32(), x_flat.data(),
                                   hc_dim * sizeof(float), cudaMemcpyHostToDevice));

            // mixes = x_scaled @ hc_fn.T
            CUBLAS_CHECK(cublasGemmEx(
                cublas_handle_,
                CUBLAS_OP_T, CUBLAS_OP_N,
                mix_size, 1, hc_dim,
                &alpha,
                hc_fn.f32(), CUDA_R_32F, hc_dim,
                buf_hc_input_.f32(), CUDA_R_32F, hc_dim,
                &beta,
                buf_hc_mixes_.f32(), CUDA_R_32F, mix_size,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT));
        }

        // Split mixes into pre, post, comb via Sinkhorn
        hc_split_sinkhorn_cuda(
            buf_hc_pre_.f32(), buf_hc_post_.f32(), buf_hc_comb_.f32(),
            buf_hc_mixes_.f32(), hc_scale.f32(), hc_base.f32(),
            hc, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, main_stream_);
        CUDA_CHECK(cudaStreamSynchronize(main_stream_));

        // Compute weighted sum: y = sum(pre[i] * hc_state[i]) for i in 0..hc-1
        // Result in buf_hidden_
        {
            float pre_host[8];
            CUDA_CHECK(cudaMemcpy(pre_host, buf_hc_pre_.f32(),
                                   hc * sizeof(float), cudaMemcpyDeviceToHost));

            if (dbg_hc_pre_call_ < 2) {
                float post_host2[8], comb_host2[64];
                CUDA_CHECK(cudaMemcpy(post_host2, buf_hc_post_.f32(), hc * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(comb_host2, buf_hc_comb_.f32(), hc * hc * sizeof(float), cudaMemcpyDeviceToHost));
                LOG_INFO("HC pre call #%d:", dbg_hc_pre_call_);
                for (int i = 0; i < hc; i++)
                    LOG_INFO("  pre[%d]=%.6f post[%d]=%.6f", i, pre_host[i], i, post_host2[i]);
                for (int i = 0; i < hc; i++) {
                    std::string s;
                    for (int j = 0; j < hc; j++) {
                        char buf[32]; snprintf(buf, sizeof(buf), "%.4f ", comb_host2[i*hc+j]);
                        s += buf;
                    }
                    LOG_INFO("  comb[%d] = [%s]", i, s.c_str());
                }
                dbg_hc_pre_call_++;
            }

            // Zero output
            CUDA_CHECK(cudaMemset(buf_hidden_.data, 0, cfg_.hidden_size * sizeof(__nv_bfloat16)));

            for (int h = 0; h < hc; h++) {
                weighted_add_cuda(buf_hidden_.bf16(),
                                  buf_hc_state_.bf16() + (size_t)h * cfg_.hidden_size,
                                  pre_host[h], cfg_.hidden_size, main_stream_);
            }
        }

        // Save residual (the full HC state before sublayer) — it's already in buf_hc_state_
        // buf_hidden2_ will hold the sublayer output after attention/FFN
    }

    // ── HC post: expand [dim] -> [hc, dim] ──────────────────────────────────

    void hc_post() {
        int dim = cfg_.hidden_size;
        int hc = cfg_.hc_mult;

        // buf_hidden_ contains sublayer output [dim]
        // buf_hc_state_ contains residual [hc, dim]
        // New hc_state[i] = post[i] * sublayer_out + sum_j(comb[i][j] * residual[j])

        CUDA_CHECK(cudaStreamSynchronize(main_stream_));

        float post_host[8], comb_host[64];
        CUDA_CHECK(cudaMemcpy(post_host, buf_hc_post_.f32(),
                               hc * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(comb_host, buf_hc_comb_.f32(),
                               hc * hc * sizeof(float), cudaMemcpyDeviceToHost));

        // Save current hc_state to persistent temp buffer
        CUDA_CHECK(cudaMemcpyAsync(buf_hc_residual_.data, buf_hc_state_.data,
                                    (size_t)hc * dim * sizeof(__nv_bfloat16),
                                    cudaMemcpyDeviceToDevice, main_stream_));

        for (int i = 0; i < hc; i++) {
            // Start with post[i] * sublayer_out
            __nv_bfloat16* dst = buf_hc_state_.bf16() + (size_t)i * dim;
            // Scale buf_hidden_ by post[i] and write to dst
            CUDA_CHECK(cudaMemset(dst, 0, dim * sizeof(__nv_bfloat16)));
            weighted_add_cuda(dst, buf_hidden_.bf16(), post_host[i], dim, main_stream_);

            // Add sum_j(comb[j][i] * residual[j]) - Transposed per reference
            for (int j = 0; j < hc; j++) {
                float c = comb_host[j * hc + i];
                if (fabsf(c) < 1e-10f) continue;
                weighted_add_cuda(dst, buf_hc_residual_.bf16() + (size_t)j * dim,
                                  c, dim, main_stream_);
            }
        }
    }

    // ── HC head: reduce [hc, dim] -> [dim] for final logits ─────────────────

    void hc_head_reduce() {
        int dim = cfg_.hidden_size;
        int hc = cfg_.hc_mult;
        int hc_dim = hc * dim;

        // Similar to hc_pre but simpler: pre = sigmoid(mix * scale + base) + eps
        bf16_to_f32_cuda(buf_hc_input_.f32(), buf_hc_state_.bf16(), hc_dim, main_stream_);

        // Compute rsqrt
        std::vector<float> x_flat(hc_dim);
        CUDA_CHECK(cudaMemcpy(x_flat.data(), buf_hc_input_.f32(),
                               hc_dim * sizeof(float), cudaMemcpyDeviceToHost));
        float sum_sq = 0;
        for (int i = 0; i < hc_dim; i++) sum_sq += x_flat[i] * x_flat[i];
        float rsqrt_val = 1.0f / sqrtf(sum_sq / hc_dim + cfg_.rms_norm_eps);
        for (auto& v : x_flat) v *= rsqrt_val;
        CUDA_CHECK(cudaMemcpy(buf_hc_input_.f32(), x_flat.data(),
                               hc_dim * sizeof(float), cudaMemcpyHostToDevice));

        // mixes = x @ hc_head_fn.T  -> [hc]
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasGemmEx(
            cublas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            hc, 1, hc_dim,
            &alpha,
            hc_head_fn_.f32(), CUDA_R_32F, hc_dim,
            buf_hc_input_.f32(), CUDA_R_32F, hc_dim,
            &beta,
            buf_hc_mixes_.f32(), CUDA_R_32F, hc,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

        // pre = sigmoid(mix * scale + base) + eps
        float mixes_host[8], scale_host[8], base_host[8];
        CUDA_CHECK(cudaMemcpy(mixes_host, buf_hc_mixes_.f32(), hc * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(scale_host, hc_head_scale_.f32(), sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(base_host, hc_head_base_.f32(), hc * sizeof(float), cudaMemcpyDeviceToHost));

        float pre_host[8];
        for (int i = 0; i < hc; i++) {
            float v = mixes_host[i] * scale_host[0] + base_host[i];
            pre_host[i] = 1.0f / (1.0f + expf(-v)) + cfg_.hc_eps;
        }

        if (dbg_head_) {
            LOG_INFO("HC head reduce: hc=%d", hc);
            for (int i = 0; i < hc; i++)
                LOG_INFO("  mix[%d]=%.6f scale=%.6f base[%d]=%.6f -> pre[%d]=%.6f",
                         i, mixes_host[i], scale_host[0], i, base_host[i], i, pre_host[i]);
            // Also dump norms of each hc_state copy
            std::vector<__nv_bfloat16> tmp(dim);
            for (int h = 0; h < hc; h++) {
                CUDA_CHECK(cudaMemcpy(tmp.data(), buf_hc_state_.bf16() + (size_t)h * dim,
                                       dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
                float norm = 0;
                for (int i = 0; i < dim; i++) { float v2 = __bfloat162float(tmp[i]); norm += v2*v2; }
                LOG_INFO("  hc_state[%d] norm=%.6f", h, sqrtf(norm));
            }
            dbg_head_ = false;
        }

        // Weighted sum
        CUDA_CHECK(cudaMemset(buf_hidden_.data, 0, dim * sizeof(__nv_bfloat16)));
        for (int h = 0; h < hc; h++) {
            weighted_add_cuda(buf_hidden_.bf16(),
                              buf_hc_state_.bf16() + (size_t)h * dim,
                              pre_host[h], dim, main_stream_);
        }
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

        // ── Q projection (low-rank) ─────────────────────────────────────────
        // q_raw = wq_a(x) -> [q_lora_rank]
        gemm_fp8_dequant(buf_lora_.bf16(), 1, q_lora, dim,
                         buf_hidden_.bf16(),
                         lw.wq_a_w.u8(), lw.wq_a_s.u8());
                if (dbg) dump_bf16("wq_a_out", buf_lora_.bf16(), q_lora);

        // q_normed = q_norm(q_raw)
        rms_norm_cuda(buf_lora_.bf16(), buf_lora_.bf16(),
                      lw.q_norm_w.bf16(), q_lora, cfg_.rms_norm_eps, main_stream_);
                if (dbg) dump_bf16("q_normed", buf_lora_.bf16(), q_lora);

        // q = wq_b(q_normed) -> [n_heads * head_dim]
        gemm_fp8_dequant(buf_q_.bf16(), 1, n_heads * head_dim_val, q_lora,
                         buf_lora_.bf16(),
                         lw.wq_b_w.u8(), lw.wq_b_s.u8());
        
        // Per-head Q normalization (DeepseekV4UnweightedRMSNorm)
        // q = q * rsqrt(mean(q^2) + eps) per head
        rms_norm_unweighted_batched_cuda(buf_q_.bf16(), buf_q_.bf16(),
                                         n_heads, head_dim_val, cfg_.rms_norm_eps,
                                         main_stream_);
        
        // Apply RoPE to last rope_dim elements of each Q head
        rope_cuda(buf_q_.bf16(), n_heads, head_dim_val, rope_dim,
                  position, layer_rope_freqs, false, main_stream_);
        
        // ── KV projection ───────────────────────────────────────────────────
        // kv = wkv(x) -> [head_dim]
        gemm_fp8_dequant(buf_kv_.bf16(), 1, head_dim_val, dim,
                         buf_hidden_.bf16(),
                         lw.wkv_w.u8(), lw.wkv_s.u8());
                if (dbg) dump_bf16("wkv_out", buf_kv_.bf16(), head_dim_val);

        // kv_norm
        rms_norm_cuda(buf_kv_.bf16(), buf_kv_.bf16(),
                      lw.kv_norm_w.bf16(), head_dim_val, cfg_.rms_norm_eps, main_stream_);
        
        // Apply RoPE to KV
        rope_cuda(buf_kv_.bf16(), 1, head_dim_val, rope_dim,
                  position, layer_rope_freqs, false, main_stream_);
        
        // Store KV in cache at position % window
        int cache_pos = position % window;
        CUDA_CHECK(cudaMemcpyAsync(
            lw.kv_cache.bf16() + (size_t)cache_pos * head_dim_val,
            buf_kv_.bf16(), head_dim_val * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToDevice, main_stream_));
                if (dbg) dump_bf16("kv_after_norm_rope", buf_kv_.bf16(), head_dim_val);

        // ── Run compressor to accumulate/emit compressed KV entries ──────────
        // This must happen AFTER we've stored the raw KV but BEFORE attention,
        // because compressed layers attend to both raw and compressed entries.
        int ratio = cfg_.layer_compress_ratio(layer_id);
        if (ratio > 0) {
            forward_compressor(layer_id, position);
        }

        // ── Attention computation ───────────────────────────────────────────
        // For each head: score = q_head @ kv_cache.T / sqrt(head_dim)
        // Then softmax and weighted sum

        // YaRN mscale correction for attention softmax scale.
        float scale = 1.0f / sqrtf((float)head_dim_val);
        if (cfg_.rope_factor > 1.0f) {
            float mscale = 0.1f * logf((float)cfg_.rope_factor) + 1.0f;
            scale *= mscale * mscale;
        }

        const __nv_bfloat16* attn_kv_ptr;
        int attn_cache_len;

        if (ratio > 0 && lw.comp_kv_count > 0 && position + 1 > window) {
            // Compressed layer: attend to raw sliding window + compressed entries
            int raw_entries = std::min(position + 1, window);
            int comp_entries = lw.comp_kv_count;

            // Build combined KV buffer: [raw_entries | comp_entries]
            // Raw entries start from the oldest in the circular buffer
            int raw_start = 0;
            if (position + 1 > window) {
                // Circular buffer wraps: entries are not contiguous
                // Copy in order: from (cache_pos+1)%window to end, then from 0 to cache_pos
                int after_pos = (cache_pos + 1) % window;
                int tail = window - after_pos;
                CUDA_CHECK(cudaMemcpyAsync(
                    buf_combined_kv_.bf16(),
                    lw.kv_cache.bf16() + (size_t)after_pos * head_dim_val,
                    tail * head_dim_val * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, main_stream_));
                CUDA_CHECK(cudaMemcpyAsync(
                    buf_combined_kv_.bf16() + (size_t)tail * head_dim_val,
                    lw.kv_cache.bf16(),
                    (size_t)after_pos * head_dim_val * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, main_stream_));
            } else {
                // Not wrapped: entries are contiguous from 0
                CUDA_CHECK(cudaMemcpyAsync(
                    buf_combined_kv_.bf16(),
                    lw.kv_cache.bf16(),
                    raw_entries * head_dim_val * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, main_stream_));
            }
            // Append compressed entries
            CUDA_CHECK(cudaMemcpyAsync(
                buf_combined_kv_.bf16() + (size_t)raw_entries * head_dim_val,
                lw.comp_kv_cache.bf16(),
                comp_entries * head_dim_val * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToDevice, main_stream_));

            attn_kv_ptr = buf_combined_kv_.bf16();
            attn_cache_len = raw_entries + comp_entries;
        } else {
            // Non-compressed layer or no compressed entries yet: use raw cache
            int total_entries = std::min(position + 1, window);
            int kv_start = 0;
            if (position + 1 > window) {
                // Circular buffer: need to linearize
                int after_pos = (cache_pos + 1) % window;
                int tail = window - after_pos;
                CUDA_CHECK(cudaMemcpyAsync(
                    buf_combined_kv_.bf16(),
                    lw.kv_cache.bf16() + (size_t)after_pos * head_dim_val,
                    tail * head_dim_val * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, main_stream_));
                CUDA_CHECK(cudaMemcpyAsync(
                    buf_combined_kv_.bf16() + (size_t)tail * head_dim_val,
                    lw.kv_cache.bf16(),
                    (size_t)after_pos * head_dim_val * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, main_stream_));
                attn_kv_ptr = buf_combined_kv_.bf16();
            } else {
                attn_kv_ptr = lw.kv_cache.bf16();
            }
            attn_cache_len = total_entries;
        }

        mla_attention_cuda(
            buf_q_.bf16(), attn_kv_ptr, lw.attn_sink.f32(),
            buf_attn_out_.bf16(), n_heads, attn_cache_len, head_dim_val, scale, main_stream_
        );
        // Inverse RoPE on attention output: the absorbed KV contains RoPE in 
        // the rope dimensions. When computing the value weighted sum, these  
        // rotations are position-mixed. We undo the query position's rotation 
        // before the output projection to align with what the model expects.
        rope_cuda(buf_attn_out_.bf16(), n_heads, head_dim_val, rope_dim,
                  position, layer_rope_freqs, true, main_stream_);

        // ── Output projection (grouped low-rank MLA) ─────────────────────
        // Attention output: [n_heads * head_dim] = [32768]
        // wo_a: [o_groups * o_lora_rank, heads_per_group * head_dim] = [8192, 4096]
        //   Block-diagonal: group g's rows [g*o_lora : (g+1)*o_lora] operate on
        //   attn_out[g*hpg_dim : (g+1)*hpg_dim] where hpg_dim = heads_per_group * head_dim
        // wo_b: [dim, o_groups * o_lora_rank] = [4096, 8192]

        int heads_per_group = n_heads / o_groups;  // 64 / 8 = 8
        int hpg_dim = heads_per_group * head_dim_val;  // 8 * 512 = 4096
        
        // Apply wo_a per group (block-diagonal GEMM)
        // For each group g:
        //   lora_g = attn_out_g @ wo_a_g.T  where
        //   attn_out_g = buf_attn_out_[g * hpg_dim : (g+1) * hpg_dim]  — [1, hpg_dim]
        //   wo_a_g = wo_a[g * o_lora : (g+1) * o_lora, :]  — [o_lora, hpg_dim]
        //   lora_g = [1, o_lora]
        // Output goes into buf_lora_ [o_groups * o_lora]
        
        // Dequantize wo_a weight fully first
        fp8_dequant_cuda(buf_dequant_.bf16(), lw.wo_a_w.u8(), lw.wo_a_s.u8(),
                         o_groups * o_lora, hpg_dim, 128, main_stream_);

        for (int g = 0; g < o_groups; g++) {
            const __nv_bfloat16* A_g = buf_attn_out_.bf16() + (size_t)g * hpg_dim;
            const __nv_bfloat16* B_g = buf_dequant_.bf16() + (size_t)g * o_lora * hpg_dim;
            __nv_bfloat16* C_g = buf_lora_.bf16() + (size_t)g * o_lora;
            gemm_bf16(C_g, 1, o_lora, hpg_dim, A_g, B_g);
        }
                if (dbg) {
            dump_bf16("attn_out_raw", buf_attn_out_.bf16(), n_heads * head_dim_val);
            dump_bf16("wo_a_result", buf_lora_.bf16(), o_groups * o_lora);
        }

        // wo_b: [dim, o_groups * o_lora] FP8 → output is [dim]
        gemm_fp8_dequant(buf_hidden_.bf16(), 1, dim, o_groups * o_lora,
                         buf_lora_.bf16(),
                         lw.wo_b_w.u8(), lw.wo_b_s.u8());
                if (dbg) dump_bf16("wo_b_result", buf_hidden_.bf16(), dim);
    }

    // ── MoE forward ─────────────────────────────────────────────────────────

    void forward_moe(int layer_id, int token_id) {
        auto& lw = layers_[layer_id];
        int dim = cfg_.hidden_size;
        int moe_inter = cfg_.moe_intermediate_size;
        int n_experts = cfg_.n_routed_experts;
        int top_k = cfg_.num_experts_per_tok;

        // ── Routing ─────────────────────────────────────────────────────────
        std::vector<int32_t> expert_ids(top_k);
        std::vector<float> expert_weights(top_k);

        if (layer_id < cfg_.n_hash_layers) {
            // Hash routing: lookup tid2eid table
            std::vector<int64_t> eid_host(top_k);
            CUDA_CHECK(cudaMemcpy(eid_host.data(),
                                   lw.tid2eid.i64() + (size_t)token_id * top_k,
                                   top_k * sizeof(int64_t), cudaMemcpyDeviceToHost));
            for (int k = 0; k < top_k; k++) {
                expert_ids[k] = (int32_t)eid_host[k];
                expert_weights[k] = cfg_.routed_scaling_factor / top_k;
            }
        } else {
            // Score-based routing
            // gate_w is BF16 [n_experts, dim]
            gemm_bf16(buf_scores_bf16_.bf16(), 1, n_experts, dim,
                      buf_hidden_.bf16(), lw.gate_w.bf16());

            // Convert to F32
            bf16_to_f32_cuda(buf_scores_f32_.f32(), buf_scores_bf16_.bf16(),
                             n_experts, main_stream_);

            // Add bias if present, in F32!
            if (lw.gate_bias.data) {
                add_f32_sigmoid_cuda(buf_scores_f32_.f32(), buf_scores_f32_.f32(),
                                     lw.gate_bias.f32(), n_experts, false /*apply_sigmoid*/, main_stream_);
            }

            // Apply sqrtsoftplus (or whatever activation the model uses)
            sqrtsoftplus_cuda(buf_scores_f32_.f32(), buf_scores_f32_.f32(), n_experts, main_stream_);
            CUDA_CHECK(cudaStreamSynchronize(main_stream_));

            // Copy scores to CPU for top-k selection
            std::vector<float> scores(n_experts);
            CUDA_CHECK(cudaMemcpy(scores.data(), buf_scores_f32_.f32(),
                                   n_experts * sizeof(float), cudaMemcpyDeviceToHost));

            // Top-k on CPU
            std::vector<int> indices(n_experts);
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                              [&](int a, int b) { return scores[a] > scores[b]; });

            float weight_sum = 0;
            for (int k = 0; k < top_k; k++) {
                expert_ids[k] = indices[k];
                expert_weights[k] = scores[indices[k]];
                weight_sum += expert_weights[k];
            }
            // Normalize and scale
            for (int k = 0; k < top_k; k++)
                expert_weights[k] = expert_weights[k] / weight_sum * cfg_.routed_scaling_factor;
        }

        // ── Zero accumulator ────────────────────────────────────────────────
        CUDA_CHECK(cudaMemset(buf_moe_accum_.data, 0, dim * sizeof(__nv_bfloat16)));

        // ── Execute routed experts ──────────────────────────────────────────
        std::future<void*> expert_futures[32];
        for (int k = 0; k < top_k; k++) {
            int eid = expert_ids[k];
            if (eid < 0) continue;

            expert_futures[k] = expert_pool_->enqueue([this, layer_id, eid]() {
                return expert_loader_.get_expert(layer_id, eid, main_stream_);
            });
        }

        for (int k = 0; k < top_k; k++) {
            int eid = expert_ids[k];
            float weight = expert_weights[k];
            if (eid < 0) continue;

            void* expert_block = expert_futures[k].get();
            if (!expert_block) { LOG_ERROR("Failed to load expert L%d E%d", layer_id, eid); continue; }

            // Run SwiGLU: out = w2(silu(w1(x)) * w3(x))
            execute_expert_swiglu(expert_block, weight);
        }

        // ── Execute shared expert ───────────────────────────────────────────
        {
            // w1(x) -> gate [moe_inter]
            gemm_fp8_dequant(buf_gate_.bf16(), 1, moe_inter, dim,
                             buf_hidden_.bf16(),
                             lw.shared_w1_w.u8(), lw.shared_w1_s.u8());
            // w3(x) -> up [moe_inter]
            gemm_fp8_dequant(buf_up_.bf16(), 1, moe_inter, dim,
                             buf_hidden_.bf16(),
                             lw.shared_w3_w.u8(), lw.shared_w3_s.u8());
            // SiLU * mul
            silu_mul_cuda(buf_gate_.bf16(), buf_gate_.bf16(), buf_up_.bf16(),
                          moe_inter, cfg_.swiglu_limit, main_stream_);
            // w2(h) -> [dim]
            gemm_fp8_dequant(buf_down_.bf16(), 1, dim, moe_inter,
                             buf_gate_.bf16(),
                             lw.shared_w2_w.u8(), lw.shared_w2_s.u8());
            // Accumulate shared expert output (no routing weight — always added)
            add_cuda(buf_moe_accum_.bf16(), buf_moe_accum_.bf16(),
                     buf_down_.bf16(), dim, main_stream_);
        }

        // Store result in buf_hidden_
        CUDA_CHECK(cudaMemcpyAsync(buf_hidden_.data, buf_moe_accum_.data,
                                    dim * sizeof(__nv_bfloat16),
                                    cudaMemcpyDeviceToDevice, main_stream_));
    }

    // ── Execute a single FP4 expert SwiGLU ──────────────────────────────────

    void execute_expert_swiglu(void* expert_block, float routing_weight) {
        int dim = cfg_.hidden_size;
        int moe_inter = cfg_.moe_intermediate_size;

        auto& w1_info = expert_parts_["w1.weight"];
        auto& w1s_info = expert_parts_["w1.scale"];
        auto& w3_info = expert_parts_["w3.weight"];
        auto& w3s_info = expert_parts_["w3.scale"];
        auto& w2_info = expert_parts_["w2.weight"];
        auto& w2s_info = expert_parts_["w2.scale"];

        uint8_t* block = (uint8_t*)expert_block;

        // w1: FP4 [moe_inter, dim/2] -> [moe_inter, dim]
        uint8_t* w1_data = block + w1_info.offset_in_block;
        uint8_t* w1_scale = block + w1s_info.offset_in_block;

        // w3: FP4 [moe_inter, dim/2] -> [moe_inter, dim]
        uint8_t* w3_data = block + w3_info.offset_in_block;
        uint8_t* w3_scale = block + w3s_info.offset_in_block;

        // w2: FP4 [dim, moe_inter/2] -> [dim, moe_inter]
        uint8_t* w2_data = block + w2_info.offset_in_block;
        uint8_t* w2_scale = block + w2s_info.offset_in_block;

        // Determine FP4 logical dimensions
        // w1/w3: physical [moe_inter, dim/2], logical [moe_inter, dim]
        // scale: [moe_inter, dim/32]
        int w1_scale_cols = w1s_info.shape.size() > 1 ? w1s_info.shape[1] : 1;
        int w2_scale_cols = w2s_info.shape.size() > 1 ? w2s_info.shape[1] : 1;

        // gate = w1(x): [1, dim] x [moe_inter, dim].T -> [1, moe_inter]
        gemm_fp4_dequant(buf_gate_.bf16(), 1, moe_inter, dim,
                         buf_hidden_.bf16(), w1_data, w1_scale, w1_scale_cols);

        // up = w3(x)
        gemm_fp4_dequant(buf_up_.bf16(), 1, moe_inter, dim,
                         buf_hidden_.bf16(), w3_data, w3_scale, w1_scale_cols);

        // SiLU(gate) * up with clamping
        silu_mul_cuda(buf_gate_.bf16(), buf_gate_.bf16(), buf_up_.bf16(),
                      moe_inter, cfg_.swiglu_limit, main_stream_);

        // down = w2(h): [1, moe_inter] x [dim, moe_inter].T -> [1, dim]
        gemm_fp4_dequant(buf_down_.bf16(), 1, dim, moe_inter,
                         buf_gate_.bf16(), w2_data, w2_scale, w2_scale_cols);

        // Accumulate with routing weight
        weighted_add_cuda(buf_moe_accum_.bf16(), buf_down_.bf16(),
                          routing_weight, dim, main_stream_);
    }

    void compute_logits() {
        int dim = cfg_.hidden_size;
        int vocab = cfg_.vocab_size;

        // logits = hidden @ head_weight.T -> [vocab]
        // head_weight: [vocab, dim] BF16
        // Use buf_attn_out_ as temp BF16 logits buffer (it's large enough: n_heads*head_dim >= vocab? No.)
        // Actually we need vocab * sizeof(bf16) = 129280 * 2 = 252 KB. buf_attn_out_ is n_heads*head_dim*2 = 64KB.
        // So let's use buf_dequant_ which is 128MB.
        gemm_bf16(buf_dequant_.bf16(), 1, vocab, dim,
                  buf_hidden_.bf16(), head_weight_.bf16());
        bf16_to_f32_cuda(buf_logits_.f32(), buf_dequant_.bf16(), vocab, main_stream_);
    }

    // ── Sample from logits ──────────────────────────────────────────────────

    float current_rep_penalty_ = 1.1f;  // Set per-request by generate()

    int sample_token(float temperature, const std::vector<int>& history, int step = 0) {
        int vocab = cfg_.vocab_size;
        std::vector<float> logits(vocab);
        CUDA_CHECK(cudaStreamSynchronize(main_stream_));
        CUDA_CHECK(cudaMemcpy(logits.data(), buf_logits_.f32(),
                               vocab * sizeof(float), cudaMemcpyDeviceToHost));

        // EOS suppression: prevent premature termination for the first N tokens.
        // The model sometimes produces competitive EOS logits early in generation,
        // especially in multi-turn conversations. Suppress EOS until we have
        // enough content tokens to form a meaningful response.
        static constexpr int MIN_TOKENS_BEFORE_EOS = 20;
        if (step < MIN_TOKENS_BEFORE_EOS) {
            if (cfg_.eos_token_id >= 0 && cfg_.eos_token_id < vocab) {
                logits[cfg_.eos_token_id] = -1e9f;
            }
            // Also suppress the EOS string token if different
            int eos2 = tokenizer_.get_token_id(
                "<\xef\xbd\x9c" "end\xe2\x96\x81of\xe2\x96\x81sentence" "\xef\xbd\x9c>");
            if (eos2 >= 0 && eos2 < vocab && eos2 != cfg_.eos_token_id) {
                logits[eos2] = -1e9f;
            }
        }

        // Repetition penalty — penalizes already-seen tokens
        float rep_penalty = current_rep_penalty_;
        std::unordered_set<int> seen_tokens;
        for (int token : history) {
            seen_tokens.insert(token);
        }
        for (int token : seen_tokens) {
            if (token >= 0 && token < vocab) {
                if (logits[token] > 0) logits[token] /= rep_penalty;
                else logits[token] *= rep_penalty;
            }
        }

        // Debug: print top-5 logits on first decode
        if (dbg_sample_count_ < 3) {
            std::vector<std::pair<float, int>> scored;
            for (int i = 0; i < vocab; i++) scored.push_back({logits[i], i});
            std::partial_sort(scored.begin(), scored.begin() + 5, scored.end(),
                             [](auto& a, auto& b) { return a.first > b.first; });
            LOG_INFO("  Top-5 logits:");
            for (int i = 0; i < 5; i++) {
                LOG_INFO("    [%d] token=%d logit=%.4f", i, scored[i].second, scored[i].first);
            }
            dbg_sample_count_++;
        }

        if (temperature <= 0.0f) {
            // Greedy
            return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
        }

        // Dynamic temperature cooling: use lower temperature for the first few
        // tokens to make topic selection more deterministic. This prevents the
        // model from picking context-bleed tokens (e.g., "I see the confusion")
        // when the conversation history contains unrelated topics.
        static constexpr int COOLDOWN_TOKENS = 5;
        float effective_temp = temperature;
        if (step < COOLDOWN_TOKENS) {
            effective_temp = std::max(0.2f, temperature * 0.5f);
        }

        // Temperature scaling
        float max_logit = *std::max_element(logits.begin(), logits.end());
        std::vector<float> probs(vocab);
        float sum_exp = 0;
        for (int i = 0; i < vocab; i++) {
            probs[i] = expf((logits[i] - max_logit) / effective_temp);
            sum_exp += probs[i];
        }
        for (auto& p : probs) p /= sum_exp;

        // Min-p filtering: discard tokens with probability < min_p * max_prob
        // This prevents low-probability tokens (like context-bleed from history)
        // from being sampled. Higher values = more focused sampling.
        float min_p = 0.1f;
        float max_prob = *std::max_element(probs.begin(), probs.end());
        float min_p_threshold = min_p * max_prob;
        for (auto& p : probs) {
            if (p < min_p_threshold) p = 0.0f;
        }

        // Top-p (nucleus) sampling with top-k safety net
        float top_p = 0.85f;
        int top_k = 40;
        std::vector<std::pair<float, int>> sorted_probs;
        for (int i = 0; i < vocab; i++) {
            if (probs[i] > 0.0f) sorted_probs.push_back({probs[i], i});
        }
        std::sort(sorted_probs.begin(), sorted_probs.end(),
                  [](auto& a, auto& b) { return a.first > b.first; });

        // Build filtered distribution (top-k AND top-p)
        std::vector<float> final_probs(vocab, 0.0f);
        float cumsum = 0.0f;
        int kept = 0;
        for (auto& sp : sorted_probs) {
            if (kept >= top_k) break;  // top-k cutoff
            final_probs[sp.second] = sp.first;
            cumsum += sp.first;
            kept++;
            if (cumsum > top_p) break;  // top-p cutoff
        }

        // Use persistent RNG (seeded once per engine instance)
        std::discrete_distribution<int> dist(final_probs.begin(), final_probs.end());
        return dist(rng_);
    }
};

// ════════════════════════════════════════════════════════════════════════════════
//  Chat Template
// ════════════════════════════════════════════════════════════════════════════════

static std::string apply_chat_template(const json& messages, const BPETokenizer& tok) {
    // DeepSeek V4 chat format (from official encoding_dsv4.py, thinking_mode="chat"):
    // <｜begin▁of▁sentence｜>{system_content}
    // <｜User｜>{user_content}<｜Assistant｜></think>{assistant_content}<｜end▁of▁sentence｜>
    // <｜User｜>{user_content}<｜Assistant｜></think>
    //
    // Key: </think> after <｜Assistant｜> signals "chat mode" — skip thinking, answer directly.
    // Without this token, the model enters an ambiguous state and produces premature EOS.
    static const std::string BOS = "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence" "\xef\xbd\x9c>";
    static const std::string EOS = "<\xef\xbd\x9c" "end\xe2\x96\x81of\xe2\x96\x81sentence" "\xef\xbd\x9c>";
    static const std::string USER = "<\xef\xbd\x9c" "User" "\xef\xbd\x9c>";
    static const std::string ASSISTANT = "<\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c>";
    static const std::string THINK_END = "</think>";

    std::string result;
    result += BOS;

    for (size_t i = 0; i < messages.size(); i++) {
        std::string role = messages[i]["role"].get<std::string>();
        std::string content = messages[i]["content"].get<std::string>();

        if (role == "system") {
            // System message: raw content, no wrapper tokens (per official encoding)
            result += content;
        } else if (role == "user") {
            result += USER;
            result += content;
        } else if (role == "assistant") {
            // In chat mode: <｜Assistant｜></think>{content}<｜end▁of▁sentence｜>
            result += ASSISTANT;
            result += THINK_END;
            result += content;
            result += EOS;
        }
    }

    // Add assistant prompt for generation
    if (!messages.empty()) {
        std::string last_role = messages.back()["role"].get<std::string>();
        if (last_role == "user") {
            result += ASSISTANT;
            result += THINK_END;  // Signal chat mode: skip thinking, answer directly
        }
    }

    return result;
}

// ════════════════════════════════════════════════════════════════════════════════
//  HTTP Server (OpenAI-compatible)
// ════════════════════════════════════════════════════════════════════════════════

static std::mutex g_engine_mutex;  // serialize inference requests
static int g_request_counter = 0;  // for unique request IDs

static void run_server(MoecherEngine& engine, int port) {
    httplib::Server svr;

    // Health check
    svr.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
        json body = {
            {"object", "list"},
            {"data", {{
                {"id", "deepseek-v4-flash"},
                {"object", "model"},
                {"owned_by", "moecher"}
            }}}
        };
        res.set_content(body.dump(), "application/json");
    });

    // Chat completions
    svr.Post("/v1/chat/completions",
        [&engine](const httplib::Request& req, httplib::Response& res) {
            auto start = std::chrono::steady_clock::now();

            json request;
            try { request = json::parse(req.body); }
            catch (...) {
                res.status = 400;
                res.set_content("{\"error\":\"Invalid JSON\"}", "application/json");
                return;
            }

            // Log request
            LOG_INFO("REQUEST: %s", req.body.c_str());

            auto& messages = request["messages"];
            float temperature = request.value("temperature", 0.0f);
            int max_tokens = request.value("max_tokens", 512);
            bool stream = request.value("stream", false);
            float repetition_penalty = request.value("repetition_penalty", 1.1f);

            // Apply chat template
            std::string prompt = apply_chat_template(messages, engine.tokenizer_);
            LOG_INFO("PROMPT (len=%zu): [%s]", prompt.size(),
                     prompt.substr(0, std::min(prompt.size(), (size_t)500)).c_str());

            std::string req_id = "chatcmpl-moecher-" + std::to_string(++g_request_counter);

            if (stream) {
                // SSE streaming
                res.set_chunked_content_provider(
                    "text/event-stream",
                    [&engine, prompt, max_tokens, temperature, req_id, repetition_penalty](size_t offset, httplib::DataSink &sink) {
                        std::lock_guard<std::mutex> lock(g_engine_mutex);
                        
                        json chunk = {
                            {"id", req_id},
                            {"object", "chat.completion.chunk"},
                            {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                            {"model", "deepseek-v4-flash"},
                            {"choices", {{
                                {"index", 0},
                                {"delta", {{"role", "assistant"}}},
                                {"finish_reason", nullptr}
                            }}}
                        };
                        std::string sse = "data: " + chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                        sink.write(sse.data(), sse.size());

                        engine.generate(prompt, max_tokens, temperature, [&](const std::string& text) {
                            chunk["choices"][0]["delta"] = {{"content", text}};
                            std::string sse_chunk = "data: " + chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                            sink.write(sse_chunk.data(), sse_chunk.size());
                        }, repetition_penalty);

                        json finish = {
                            {"id", req_id},
                            {"object", "chat.completion.chunk"},
                            {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                            {"model", "deepseek-v4-flash"},
                            {"choices", {{
                                {"index", 0},
                                {"delta", json::object()},
                                {"finish_reason", engine.last_finish_reason_}
                            }}}
                        };
                        sse = "data: " + finish.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                        sink.write(sse.data(), sse.size());
                        sink.write("data: [DONE]\n\n", 14);
                        sink.done();
                        return true;
                    }
                );
            } else {
                std::string response_text;
                {
                    std::lock_guard<std::mutex> lock(g_engine_mutex);
                    response_text = engine.generate(prompt, max_tokens, temperature, nullptr, repetition_penalty);
                }
                
                json response = {
                    {"id", req_id},
                    {"object", "chat.completion"},
                    {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                    {"model", "deepseek-v4-flash"},
                    {"choices", {{
                        {"index", 0},
                        {"message", {{"role", "assistant"}, {"content", response_text}}},
                        {"finish_reason", engine.last_finish_reason_}
                    }}},
                    {"usage", {
                        {"prompt_tokens", engine.prompt_token_count_},
                        {"completion_tokens", engine.completion_token_count_},
                        {"total_tokens", engine.prompt_token_count_ + engine.completion_token_count_}
                    }}
                };
                res.set_content(response.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
            }
            auto end = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
            LOG_INFO("RESPONSE (%.0fms)", elapsed_ms);
        });

    LOG_INFO("Server listening on port %d", port);
    svr.listen("0.0.0.0", port);
}

// ════════════════════════════════════════════════════════════════════════════════
//  Main
// ════════════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    std::string manifest_path = "moecher_manifest.json";
    int port = 8001;
    float max_vram_gb = 0.0f;
    std::string log_path = "moecher.log";

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--manifest" && i + 1 < argc) {
            manifest_path = argv[++i];
        } else if (std::string(argv[i]) == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--log" && i + 1 < argc) {
            log_path = argv[++i];
        } else if (std::string(argv[i]) == "--max-vram" && i + 1 < argc) {
            max_vram_gb = std::stof(argv[++i]);
        } else if (std::string(argv[i]) == "--log-experts") {
            g_log_experts = true;
        } else if (std::string(argv[i]) == "--no-log-tokens") {
            g_log_tokens = false;
        }
    }

    // Open log file
    g_log_file.open(log_path, std::ios::app);
    LOG_INFO("═══ moecher starting ═══");

    MoecherEngine engine;
    if (!engine.load(manifest_path, max_vram_gb)) {
        LOG_ERROR("Failed to load model");
        return 1;
    }

    run_server(engine, port);
    return 0;
}
