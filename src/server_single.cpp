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
static constexpr int MAX_SEQ_LEN = 4096;

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
    char byte_to_char_[256];
    std::unordered_map<char32_t, uint8_t> char_to_byte_;

    void build_byte_mapping() {
        // GPT-2 byte encoder: maps bytes to printable unicode characters
        // This matches the HuggingFace tokenizers byte_level pre-tokenizer
        int n = 0;
        for (int b = 0; b < 256; b++) {
            // Printable ASCII + extended range map to themselves
            if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
                byte_to_char_[b] = (char)b;
            } else {
                // Non-printable bytes map to unicode starting at U+0100
                byte_to_char_[b] = (char)(256 + n);
                n++;
            }
        }
    }

    std::string bytes_to_unicode(const std::string& bytes) const {
        std::string result;
        for (unsigned char b : bytes) {
            // Encode each byte as its unicode character representation
            unsigned char c = byte_to_char_[b];
            if (c < 128) {
                result += (char)c;
            } else {
                // UTF-8 encode
                result += (char)(0xC0 | (c >> 6));
                result += (char)(0x80 | (c & 0x3F));
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
                    if ((unsigned char)byte_to_char_[b] == (unsigned char)codepoint) {
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

        // Simple word-level split (split on spaces, keeping space with following word)
        std::vector<std::string> words;
        std::string current;
        for (size_t i = 0; i < unicode_text.size(); i++) {
            if (unicode_text[i] == ' ' && !current.empty()) {
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
                        std::string byte_tok(1, byte_to_char_[b]);
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
    GPUTensor rope_freqs_;         // [MAX_SEQ_LEN, rope_dim/2, 2] F32

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

        // Precompute RoPE frequencies
        int rope_dim = cfg_.qk_rope_head_dim;
        rope_freqs_.alloc(MAX_SEQ_LEN * (rope_dim / 2) * 2 * sizeof(float));
        precompute_freqs_cuda(rope_freqs_.f32(), MAX_SEQ_LEN, rope_dim,
                              cfg_.rope_theta, cfg_.rope_factor,
                              cfg_.original_seq_len, cfg_.rope_beta_fast,
                              cfg_.rope_beta_slow, main_stream_);
        CUDA_CHECK(cudaStreamSynchronize(main_stream_));

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

        static bool first_token = true;

        // 1. Embedding lookup
        int32_t tid_host = token_id;
        CUDA_CHECK(cudaMemcpyAsync(buf_input_ids_.i32(), &tid_host, sizeof(int32_t),
                                    cudaMemcpyHostToDevice, main_stream_));
        embedding_cuda(buf_hidden_.bf16(), embed_weight_.bf16(),
                       buf_input_ids_.i32(), 1, dim, main_stream_);

        if (first_token) {
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

        if (first_token) dump_bf16("after_hc_head", buf_hidden_.bf16(), dim);

        // 5. Final norm
        rms_norm_cuda(buf_hidden_.bf16(), buf_hidden_.bf16(),
                      norm_weight_.bf16(), dim, cfg_.rms_norm_eps, main_stream_);

        if (first_token) dump_bf16("after_final_norm", buf_hidden_.bf16(), dim);

        // 6. Logits: hidden @ head_weight.T -> [vocab_size]
        compute_logits();

        first_token = false;
    }

    // ── Generate tokens ─────────────────────────────────────────────────────

    std::string generate(const std::string& prompt, int max_tokens = 512,
                         float temperature = 0.0f,
                         std::function<void(const std::string&)> on_token = nullptr) {
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

        std::vector<int> history = input_ids;

        // Decode: generate tokens one by one
        int position = (int)input_ids.size();
        for (int t = 0; t < max_tokens; t++) {
            // Sample from logits
            int next_token = sample_token(temperature, history);

            if (next_token == cfg_.eos_token_id) break;
            output_ids.push_back(next_token);
            history.push_back(next_token);

            std::string token_text = tokenizer_.decode({next_token});
            if (g_log_tokens) {
                printf("%s", token_text.c_str());
                fflush(stdout);
            }
            if (on_token) {
                on_token(token_text);
            }

            // Forward the new token
            forward_token(next_token, position);
            position++;
        }

        if (g_log_tokens) {
            printf("\n");
        }

        return tokenizer_.decode(output_ids);
    }

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

            static int hc_pre_call = 0;
            if (hc_pre_call < 2) {
                float post_host2[8], comb_host2[64];
                CUDA_CHECK(cudaMemcpy(post_host2, buf_hc_post_.f32(), hc * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(comb_host2, buf_hc_comb_.f32(), hc * hc * sizeof(float), cudaMemcpyDeviceToHost));
                LOG_INFO("HC pre call #%d:", hc_pre_call);
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
                hc_pre_call++;
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

            // Add sum_j(comb[i][j] * residual[j])
            for (int j = 0; j < hc; j++) {
                float c = comb_host[i * hc + j];
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

        static bool dbg_head = true;
        if (dbg_head) {
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
            dbg_head = false;
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
                  position, rope_freqs_.f32(), false, main_stream_);
        
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
                  position, rope_freqs_.f32(), false, main_stream_);
        
        // Store KV in cache at position % window
        int cache_pos = position % window;
        CUDA_CHECK(cudaMemcpyAsync(
            lw.kv_cache.bf16() + (size_t)cache_pos * head_dim_val,
            buf_kv_.bf16(), head_dim_val * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToDevice, main_stream_));
                if (dbg) dump_bf16("kv_after_norm_rope", buf_kv_.bf16(), head_dim_val);

        // ── Attention computation ───────────────────────────────────────────
        // For each head: score = q_head @ kv_cache.T / sqrt(head_dim)
        // Then softmax and weighted sum

        int cache_len = std::min(position + 1, window);

        
        float scale = 1.0f / sqrtf((float)head_dim_val);
        mla_attention_cuda(
            buf_q_.bf16(), lw.kv_cache.bf16(), lw.attn_sink.f32(),
            buf_attn_out_.bf16(), n_heads, cache_len, head_dim_val, scale, main_stream_
        );
        // Inverse RoPE on attention output: K=V in DeepSeek V4, so the value
        // picked up RoPE rotation. Undo it with inverse rotation (-sin) before
        // the grouped output projection.
        rope_cuda(buf_attn_out_.bf16(), n_heads, head_dim_val, rope_dim,
                  position, rope_freqs_.f32(), true, main_stream_);

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
            // scores = x @ gate_w.T -> [n_experts]
            // gate_w is BF16 [n_experts, dim]
            gemm_bf16((__nv_bfloat16*)buf_scores_f32_.data, 1, n_experts, dim,
                      buf_hidden_.bf16(), lw.gate_w.bf16());

            // Convert to F32 and apply sqrtsoftplus
            bf16_to_f32_cuda(buf_scores_f32_.f32(), (__nv_bfloat16*)buf_scores_f32_.data,
                             n_experts, main_stream_);
            sqrtsoftplus_cuda(buf_scores_f32_.f32(), buf_scores_f32_.f32(), n_experts, main_stream_);

            // Copy original scores for weight computation
            std::vector<float> original_scores(n_experts);
            CUDA_CHECK(cudaMemcpy(original_scores.data(), buf_scores_f32_.f32(),
                                   n_experts * sizeof(float), cudaMemcpyDeviceToHost));

            // Add bias for top-k selection
            if (lw.gate_bias.data) {
                // Add bias on CPU (small vector)
                std::vector<float> biased(n_experts);
                std::vector<float> bias_host(n_experts);
                CUDA_CHECK(cudaMemcpy(bias_host.data(), lw.gate_bias.f32(),
                                       n_experts * sizeof(float), cudaMemcpyDeviceToHost));
                for (int i = 0; i < n_experts; i++)
                    biased[i] = original_scores[i] + bias_host[i];

                // Top-k on CPU
                std::vector<int> indices(n_experts);
                std::iota(indices.begin(), indices.end(), 0);
                std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                                  [&](int a, int b) { return biased[a] > biased[b]; });

                float weight_sum = 0;
                for (int k = 0; k < top_k; k++) {
                    expert_ids[k] = indices[k];
                    expert_weights[k] = original_scores[indices[k]];
                    weight_sum += expert_weights[k];
                }
                // Normalize and scale
                for (int k = 0; k < top_k; k++)
                    expert_weights[k] = expert_weights[k] / weight_sum * cfg_.routed_scaling_factor;
            }
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

    int sample_token(float temperature, const std::vector<int>& history) {
        int vocab = cfg_.vocab_size;
        std::vector<float> logits(vocab);
        CUDA_CHECK(cudaStreamSynchronize(main_stream_));
        CUDA_CHECK(cudaMemcpy(logits.data(), buf_logits_.f32(),
                               vocab * sizeof(float), cudaMemcpyDeviceToHost));

        // Repetition penalty
        float rep_penalty = 1.15f;
        for (int token : history) {
            if (token >= 0 && token < vocab) {
                if (logits[token] > 0) logits[token] /= rep_penalty;
                else logits[token] *= rep_penalty;
            }
        }

        // Debug: print top-5 logits on first decode
        static int dbg_count = 0;
        if (dbg_count < 3) {
            std::vector<std::pair<float, int>> scored;
            for (int i = 0; i < vocab; i++) scored.push_back({logits[i], i});
            std::partial_sort(scored.begin(), scored.begin() + 5, scored.end(),
                             [](auto& a, auto& b) { return a.first > b.first; });
            LOG_INFO("  Top-5 logits:");
            for (int i = 0; i < 5; i++) {
                LOG_INFO("    [%d] token=%d logit=%.4f", i, scored[i].second, scored[i].first);
            }
            dbg_count++;
        }

        if (temperature <= 0.0f) {
            // Greedy
            return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
        }

        // Temperature sampling
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum_exp = 0;
        for (auto& l : logits) {
            l = expf((l - max_logit) / temperature);
            sum_exp += l;
        }
        for (auto& l : logits) l /= sum_exp;

        // Top-p (nucleus) sampling
        float top_p = 0.9f;
        std::vector<std::pair<float, int>> probs;
        for (int i = 0; i < vocab; i++) probs.push_back({logits[i], i});
        std::sort(probs.begin(), probs.end(), [](auto& a, auto& b) { return a.first > b.first; });

        float cumsum = 0.0f;
        std::vector<float> filtered_probs(vocab, 0.0f);
        for (auto& p : probs) {
            filtered_probs[p.second] = p.first;
            cumsum += p.first;
            if (cumsum > top_p) break;
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<int> dist(filtered_probs.begin(), filtered_probs.end());
        return dist(gen);
    }
};

// ════════════════════════════════════════════════════════════════════════════════
//  Chat Template
// ════════════════════════════════════════════════════════════════════════════════

static std::string apply_chat_template(const json& messages, const BPETokenizer& tok) {
    // DeepSeek V4 chat format:
    // <｜begin▁of▁sentence｜><｜User｜>message<｜Assistant｜>
    // Note: uses < > (U+003C/003E) not 〈 〉 (U+3008/3009), with fullwidth ｜ (U+FF5C)
    std::string result;
    // BOS token: <｜begin▁of▁sentence｜>
    result += "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence" "\xef\xbd\x9c>";

    for (auto& msg : messages) {
        std::string role = msg["role"].get<std::string>();
        std::string content = msg["content"].get<std::string>();

        if (role == "system") {
            result += "<\xef\xbd\x9c" "begin\xe2\x96\x81sys" "\xef\xbd\x9c>";
            result += content;
            result += "<\xef\xbd\x9c" "end\xe2\x96\x81sys" "\xef\xbd\x9c>";
        } else if (role == "user") {
            result += "<\xef\xbd\x9c" "User" "\xef\xbd\x9c>";
            result += content;
        } else if (role == "assistant") {
            result += "<\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c>";
            result += content;
        }
    }

    // Add assistant prompt for generation
    if (!messages.empty()) {
        std::string last_role = messages.back()["role"].get<std::string>();
        if (last_role == "user") {
            result += "<\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c>";
        }
    }

    return result;
}

// ════════════════════════════════════════════════════════════════════════════════
//  HTTP Server (OpenAI-compatible)
// ════════════════════════════════════════════════════════════════════════════════

static std::mutex g_engine_mutex;  // serialize inference requests

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

            // Apply chat template
            std::string prompt = apply_chat_template(messages, engine.tokenizer_);

            
            if (stream) {
                // SSE streaming
                res.set_chunked_content_provider(
                    "text/event-stream",
                    [&engine, prompt, max_tokens, temperature](size_t offset, httplib::DataSink &sink) {
                        std::lock_guard<std::mutex> lock(g_engine_mutex);
                        
                        json chunk = {
                            {"id", "chatcmpl-moecher"},
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
                        });

                        json finish = {
                            {"id", "chatcmpl-moecher"},
                            {"object", "chat.completion.chunk"},
                            {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                            {"model", "deepseek-v4-flash"},
                            {"choices", {{
                                {"index", 0},
                                {"delta", json::object()},
                                {"finish_reason", "stop"}
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
                    response_text = engine.generate(prompt, max_tokens, temperature);
                }
                
                json response = {
                    {"id", "chatcmpl-moecher"},
                    {"object", "chat.completion"},
                    {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                    {"model", "deepseek-v4-flash"},
                    {"choices", {{
                        {"index", 0},
                        {"message", {{"role", "assistant"}, {"content", response_text}}},
                        {"finish_reason", "stop"}
                    }}}
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
