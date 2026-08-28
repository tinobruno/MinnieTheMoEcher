// server_single.cpp — Bare-metal DeepSeek V4-Flash MoE inference engine
// Stack: C++17, CUDA, cuBLAS, cpp-httplib, nlohmann/json
// Features:
//   - O_DIRECT SSD offloading for MoE experts (3090 24GB friendly)
//   - Manifest-driven tensor loading from moecher_manifest.json
//   - OpenAI-compatible /v1/chat/completions API on port 8001
//   - BPE tokenizer from HuggingFace tokenizer.json
//   - Full DeepSeek V4-Flash forward pass: MLA + MoE + HC residuals

#include "platform/platform_io.hpp"
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
#include <atomic>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <immintrin.h>
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

using std::string;

static constexpr int PAGE_SIZE = 4096;
static constexpr int MAX_SEQ_LEN = 65536;

// ════════════════════════════════════════════════════════════════════════════════
//  Logging
// ════════════════════════════════════════════════════════════════════════════════

static std::mutex g_log_mutex;
static std::ofstream g_log_file;
bool g_log_experts = false;
bool g_log_tokens = true;
bool g_quiet = false;
bool g_server_ready = false;
static std::atomic<bool> g_stop_requested{false};


static void log_msg(const char* level, const char* fmt, ...) {
    if (g_quiet && g_server_ready && (strcmp(level, "INFO") == 0 || strcmp(level, "WARN") == 0)) {
        return;
    }
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

enum class ModelArch { DEEPSEEK_V4, QWEN };

struct ModelConfig {
    ModelArch architecture = ModelArch::DEEPSEEK_V4;
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
    int max_seq_len = 32768;           // max sequence context length
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
            if (j.contains(key) && !j.at(key).is_null()) j.at(key).get_to(field);
        };
        if (j.contains("architecture")) {
            std::string arch_str = j["architecture"].get<std::string>();
            if (arch_str == "qwen2" || arch_str == "qwen" || arch_str == "qwen3" || arch_str == "llama") {
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
        if (j.contains("compress_ratios") && j["compress_ratios"].is_array()) {
            compress_ratios = j["compress_ratios"].get<std::vector<int>>();
        }
        // Compute max compressed entries: max_seq_len / min_ratio
        // This determines the size of the compressed KV cache per layer across the full context
        max_compressed_entries = 0;
        for (int r : compress_ratios) {
            if (r > 0) {
                int entries = max_seq_len / r + 2;
                if (entries > max_compressed_entries) max_compressed_entries = entries;
            }
        }
    }
};

// ════════════════════════════════════════════════════════════════════════════════
//  JoyAI BPE Tokenizer (reads HuggingFace tokenizer.json)
// ════════════════════════════════════════════════════════════════════════════════

static inline bool ascii_alpha(uint8_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static inline bool ascii_digit(uint8_t c) {
    return c >= '0' && c <= '9';
}

static inline bool ascii_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '\v' || c == '\f';
}

static inline bool ascii_newline(uint8_t c) {
    return c == '\n' || c == '\r';
}

static inline bool joyai_ascii_punct_symbol(uint8_t c) {
    return (c >= '!' && c <= '/') ||
           (c >= ':' && c <= '@') ||
           (c >= '[' && c <= '`') ||
           (c >= '{' && c <= '~');
}

static inline int utf8_len_from_first_byte(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

static inline uint64_t next_utf8_char(const char* s, uint64_t len, uint64_t pos) {
    int n = utf8_len_from_first_byte((uint8_t)s[pos]);
    if (pos + (uint64_t)n > len) n = 1;
    return pos + (uint64_t)n;
}

static inline bool utf8_is_cjk_hira_kata(uint32_t cp) {
    return (cp >= 0x4e00 && cp <= 0x9fa5) ||
           (cp >= 0x3040 && cp <= 0x309f) ||
           (cp >= 0x30a0 && cp <= 0x30ff);
}

static inline uint32_t utf8_peek_one(const char* s, uint64_t len, uint64_t pos, uint64_t* next) {
    const uint8_t c0 = (uint8_t)s[pos];
    int n = utf8_len_from_first_byte(c0);
    if (pos + (uint64_t)n > len) n = 1;
    *next = pos + (uint64_t)n;

    if (n == 1) return c0;
    if (n == 2) {
        return ((uint32_t)(c0 & 0x1f) << 6) |
               ((uint32_t)((uint8_t)s[pos + 1] & 0x3f));
    }
    if (n == 3) {
        return ((uint32_t)(c0 & 0x0f) << 12) |
               ((uint32_t)((uint8_t)s[pos + 1] & 0x3f) << 6) |
               ((uint32_t)((uint8_t)s[pos + 2] & 0x3f));
    }
    return ((uint32_t)(c0 & 0x07) << 18) |
           ((uint32_t)((uint8_t)s[pos + 1] & 0x3f) << 12) |
           ((uint32_t)((uint8_t)s[pos + 2] & 0x3f) << 6) |
           ((uint32_t)((uint8_t)s[pos + 3] & 0x3f));
}

static inline bool joyai_cjk_at(const char* s, uint64_t len, uint64_t pos) {
    if ((uint8_t)s[pos] < 128) return false;
    uint64_t next = pos;
    uint32_t cp = utf8_peek_one(s, len, pos, &next);
    return utf8_is_cjk_hira_kata(cp);
}

static inline bool joyai_letter_like_at(const char* s, uint64_t len, uint64_t pos) {
    (void)len;
    uint8_t c = (uint8_t)s[pos];
    if (c < 128) return ascii_alpha(c);
    return true;
}

static inline uint64_t joyai_consume_letters(const char* s, uint64_t len, uint64_t pos) {
    while (pos < len && joyai_letter_like_at(s, len, pos)) {
        pos = next_utf8_char(s, len, pos);
    }
    return pos;
}

class BPETokenizer {
public:
    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) { LOG_ERROR("Cannot open tokenizer: %s", path.c_str()); return false; }
        json tok;
        try { f >> tok; } catch (const std::exception& e) {
            LOG_ERROR("JSON parse error in tokenizer: %s", e.what()); return false;
        }

        auto& model = tok["model"];
        auto& vocab = model["vocab"];
        for (auto& [k, v] : vocab.items()) {
            int id = v.get<int>();
            token_to_id_[k] = id;
            id_to_token_[id] = k;
        }

        if (model.contains("merges")) {
            for (auto& m : model["merges"]) {
                std::string merge_str = m.get<std::string>();
                merges_.push_back(merge_str);
                merge_rank_[merge_str] = (int)merges_.size();
            }
        }

        if (tok.contains("added_tokens")) {
            for (auto& at : tok["added_tokens"]) {
                int id = at["id"].get<int>();
                std::string content = at["content"].get<std::string>();
                token_to_id_[content] = id;
                id_to_token_[id] = content;
                special_tokens_.push_back({content, id});
            }
        }

        std::sort(special_tokens_.begin(), special_tokens_.end(),
                  [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

        build_byte_mapping();

        LOG_INFO("Tokenizer loaded: %zu vocab, %zu merges, %zu special tokens",
                 token_to_id_.size(), merges_.size(), special_tokens_.size());
        return true;
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        std::vector<std::pair<std::string, bool>> segments;
        split_on_special(text, segments);

        for (auto& [seg, is_special] : segments) {
            if (is_special) {
                auto it = token_to_id_.find(seg);
                if (it != token_to_id_.end()) ids.push_back(it->second);
                continue;
            }
            joyai_tokenize_segment(seg, ids);
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

    uint32_t byte_to_char_[256];
    std::unordered_map<char32_t, uint8_t> char_to_byte_;

    void build_byte_mapping() {
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
                byte_to_char_[b] = (uint32_t)b;
            } else {
                byte_to_char_[b] = 256 + n;
                n++;
            }
        }
    }

    std::string bytes_to_unicode(const std::string& bytes) const {
        std::string result;
        for (unsigned char b : bytes) {
            uint32_t cp = byte_to_char_[b];
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

            if (codepoint < 256) {
                result += (char)codepoint;
            } else if (codepoint >= 256 && codepoint < 256 + 256) {
                for (int b = 0; b < 256; b++) {
                    if (byte_to_char_[b] == codepoint) {
                        result += (char)b;
                        break;
                    }
                }
            } else {
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
                    out.push_back({tok, true});
                    pos += tok.size();
                    found = true;
                    break;
                }
            }
            if (!found) {
                if (out.empty() || out.back().second) {
                    out.push_back({"", false});
                }
                out.back().first += text[pos];
                pos++;
            }
        }
    }

    void bpe_emit_piece(const std::string& raw_piece, std::vector<int>& ids) const {
        if (raw_piece.empty()) return;

        std::string encoded = bytes_to_unicode(raw_piece);

        std::vector<std::string> tokens;
        size_t i = 0;
        while (i < encoded.size()) {
            unsigned char c = (unsigned char)encoded[i];
            int char_len = 1;
            if ((c & 0xE0) == 0xC0) char_len = 2;
            else if ((c & 0xF0) == 0xE0) char_len = 3;
            else if ((c & 0xF8) == 0xF0) char_len = 4;
            tokens.push_back(encoded.substr(i, char_len));
            i += char_len;
        }

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

        for (auto& tok : tokens) {
            auto it = token_to_id_.find(tok);
            if (it != token_to_id_.end()) {
                ids.push_back(it->second);
            } else {
                for (unsigned char b : tok) {
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

    void joyai_tokenize_segment(const std::string& text, std::vector<int>& out) const {
        const uint64_t len = text.size();
        uint64_t pos = 0;

        while (pos < len) {
            uint64_t start = pos;
            uint8_t c = (uint8_t)text[pos];

            if (ascii_digit(c)) {
                int ndigits = 0;
                while (pos < len && ascii_digit((uint8_t)text[pos]) && ndigits < 3) {
                    pos++;
                    ndigits++;
                }
            } else if (joyai_cjk_at(text.c_str(), len, pos)) {
                do {
                    pos = next_utf8_char(text.c_str(), len, pos);
                } while (pos < len && joyai_cjk_at(text.c_str(), len, pos));
            } else if (joyai_ascii_punct_symbol(c) &&
                       pos + 1 < len &&
                       ascii_alpha((uint8_t)text[pos + 1])) {
                pos++;
                while (pos < len && ascii_alpha((uint8_t)text[pos])) pos++;
            } else if (joyai_letter_like_at(text.c_str(), len, pos)) {
                pos = joyai_consume_letters(text.c_str(), len, pos);
            } else if (!ascii_newline(c) &&
                       !joyai_ascii_punct_symbol(c) &&
                       pos + 1 < len &&
                       joyai_letter_like_at(text.c_str(), len, pos + 1)) {
                pos++;
                pos = joyai_consume_letters(text.c_str(), len, pos);
            } else if (c == ' ' &&
                       pos + 1 < len &&
                       joyai_ascii_punct_symbol((uint8_t)text[pos + 1])) {
                pos++;
                while (pos < len && joyai_ascii_punct_symbol((uint8_t)text[pos])) pos++;
                while (pos < len && ascii_newline((uint8_t)text[pos])) pos++;
            } else if (joyai_ascii_punct_symbol(c)) {
                while (pos < len && joyai_ascii_punct_symbol((uint8_t)text[pos])) pos++;
                while (pos < len && ascii_newline((uint8_t)text[pos])) pos++;
            } else if (ascii_space(c)) {
                uint64_t p = pos;
                uint64_t last_newline_end = 0;
                while (p < len && ascii_space((uint8_t)text[p])) {
                    uint8_t sc = (uint8_t)text[p++];
                    if (ascii_newline(sc)) last_newline_end = p;
                }
                if (last_newline_end) {
                    pos = last_newline_end;
                } else if (p < len && p > pos + 1 &&
                           (joyai_letter_like_at(text.c_str(), len, p) ||
                            joyai_ascii_punct_symbol((uint8_t)text[p]))) {
                    pos = p - 1;
                } else {
                    pos = p;
                }
            } else {
                pos = next_utf8_char(text.c_str(), len, pos);
            }

            if (pos == start) pos = next_utf8_char(text.c_str(), len, pos);
            bpe_emit_piece(text.substr(start, pos - start), out);
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

    GPUTensor() = default;
    GPUTensor(const GPUTensor&) = delete;
    GPUTensor& operator=(const GPUTensor&) = delete;
    GPUTensor(GPUTensor&& other) noexcept {
        data = other.data;
        size_bytes = other.size_bytes;
        shape = std::move(other.shape);
        dtype = std::move(other.dtype);
        other.data = nullptr;
        other.size_bytes = 0;
    }
    GPUTensor& operator=(GPUTensor&& other) noexcept {
        if (this != &other) {
            free();
            data = other.data;
            size_bytes = other.size_bytes;
            shape = std::move(other.shape);
            dtype = std::move(other.dtype);
            other.data = nullptr;
            other.size_bytes = 0;
        }
        return *this;
    }

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
    moecher::platform::DirectFileHandle expert_file_;

    // L1 LRU cache (VRAM)
    int cache_capacity_ = 0;
    void* cache_pool_gpu_ = nullptr;
    std::vector<ExpertCacheEntry> cache_slots_;
    std::unordered_map<int64_t, int> key_to_slot_;  // (layer*n_experts+expert) -> L1 slot index
    std::vector<void*> flat_vram_ptrs_;             // Lock-free flat array for 0-latency expert lookups

    // L2 LRU cache (DRAM)
    int dram_cache_capacity_ = 0;
    void* dram_cache_pool_ = nullptr;
    std::vector<ExpertCacheEntry> dram_cache_slots_;
    std::unordered_map<int64_t, int> dram_key_to_slot_; // (layer*n_experts+expert) -> L2 slot index
    std::vector<void*> flat_dram_ptrs_;                 // Lock-free flat array for 0-latency DRAM lookups

    // Ring buffer for staging (used when bypassing DRAM cache)
    static constexpr int NUM_STAGING_BUFFERS = 32;
    struct StagingBuffer {
        void* ptr = nullptr;
        cudaEvent_t event = nullptr;
    };
    std::vector<StagingBuffer> staging_ring_;
    int staging_idx_ = 0;

    int64_t access_counter_ = 0;
    std::mutex cache_mutex_;

    int n_pinned_layers() const {
        return (n_experts_ > 0) ? std::min(n_layers_, cache_capacity_ / n_experts_) : 0;
    }

    bool is_layer_pinned(int layer_id) const {
        return layer_id < n_pinned_layers();
    }

    inline void* get_dram_or_disk_ptr(int layer_id, int expert_id) {
        int64_t key = (int64_t)layer_id * n_experts_ + expert_id;
        if (key >= 0 && key < (int64_t)flat_dram_ptrs_.size()) {
            void* ptr = flat_dram_ptrs_[key];
            if (ptr) return ptr;
        }
        return fetch_cold_expert_from_disk(layer_id, expert_id, key);
    }

    void* fetch_cold_expert_from_disk(int layer_id, int expert_id, int64_t key) {
        std::unique_lock<std::mutex> lock(cache_mutex_);
        if (dram_cache_capacity_ > 0) {
            auto it = dram_key_to_slot_.find(key);
            if (it != dram_key_to_slot_.end()) {
                return dram_cache_slots_[it->second].gpu_data;
            }
            access_counter_++;
            int evict_dram = 0;
            int64_t oldest_dram = INT64_MAX;
            for (int i = 0; i < dram_cache_capacity_; i++) {
                if (dram_cache_slots_[i].layer_id < 0) {
                    evict_dram = i;
                    break;
                }
                if (dram_cache_slots_[i].last_used < oldest_dram) {
                    oldest_dram = dram_cache_slots_[i].last_used;
                    evict_dram = i;
                }
            }
            auto& dslot = dram_cache_slots_[evict_dram];
            if (dslot.layer_id >= 0) {
                int64_t old_dram_key = (int64_t)dslot.layer_id * n_experts_ + dslot.expert_id;
                dram_key_to_slot_.erase(old_dram_key);
                if (old_dram_key >= 0 && old_dram_key < (int64_t)flat_dram_ptrs_.size()) {
                    flat_dram_ptrs_[old_dram_key] = nullptr;
                }
            }
            dslot.layer_id = layer_id;
            dslot.expert_id = expert_id;
            dslot.last_used = access_counter_;
            dram_key_to_slot_[key] = evict_dram;

            int64_t file_offset = (int64_t)key * expert_block_size_;
            expert_file_.pread_exact(dslot.gpu_data, expert_block_size_, file_offset);
            if (key >= 0 && key < (int64_t)flat_dram_ptrs_.size()) {
                flat_dram_ptrs_[key] = dslot.gpu_data;
            }
            return dslot.gpu_data;
        } else {
            int stage_idx = staging_idx_;
            staging_idx_ = (staging_idx_ + 1) % NUM_STAGING_BUFFERS;
            auto& stage = staging_ring_[stage_idx];
            int64_t file_offset = (int64_t)key * expert_block_size_;
            expert_file_.pread_exact(stage.ptr, expert_block_size_, file_offset);
            return stage.ptr;
        }
    }

    bool init(const std::string& expert_bin_path, int block_size,
              int n_layers, int n_experts, size_t cache_budget_bytes, size_t dram_budget_bytes) {
        expert_block_size_ = block_size;
        n_layers_ = n_layers;
        n_experts_ = n_experts;

        if (!expert_file_.open_read(expert_bin_path, true)) {
            LOG_ERROR("Cannot open expert bin: %s", expert_bin_path.c_str());
            return false;
        }

        cache_capacity_ = (int)(cache_budget_bytes / block_size);
        if (cache_capacity_ < 64) cache_capacity_ = 64;
        LOG_INFO("Expert L1 cache: %d slots (%.1f GB)", cache_capacity_,
                 (double)cache_capacity_ * block_size / (1024.0 * 1024.0 * 1024.0));

        CUDA_CHECK(cudaMalloc(&cache_pool_gpu_, (size_t)cache_capacity_ * block_size));

        cache_slots_.resize(cache_capacity_);
        for (int i = 0; i < cache_capacity_; i++) {
            cache_slots_[i].slot_index = i;
            cache_slots_[i].gpu_data = (char*)cache_pool_gpu_ + (size_t)i * block_size;
        }

        dram_cache_capacity_ = (int)(dram_budget_bytes / block_size);
        if (dram_cache_capacity_ > 0) {
            LOG_INFO("Expert L2 cache: %d slots (%.1f GB)", dram_cache_capacity_,
                     (double)dram_cache_capacity_ * block_size / (1024.0 * 1024.0 * 1024.0));
            size_t total_dram_bytes = (size_t)dram_cache_capacity_ * block_size;
            cudaError_t err = cudaMallocHost(&dram_cache_pool_, total_dram_bytes);
            if (err != cudaSuccess) {
                cudaGetLastError(); // Clear error state
#if defined(_WIN32) || defined(_WIN64)
                dram_cache_pool_ = VirtualAlloc(NULL, total_dram_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
                dram_cache_pool_ = aligned_alloc(4096, total_dram_bytes);
#endif
                if (dram_cache_pool_) {
                    cudaError_t reg_err = cudaHostRegister(dram_cache_pool_, total_dram_bytes, cudaHostRegisterDefault);
                    if (reg_err != cudaSuccess) {
                        cudaGetLastError();
                        LOG_WARN("cudaHostRegister could not pin %.1f GB DRAM cache (OS locked memory limit / ulimit). DMA transfers will run with pageable fallback.",
                                 (double)total_dram_bytes / (1024.0 * 1024.0 * 1024.0));
                    } else {
                        LOG_INFO("Pinned %.1f GB DRAM cache into physical RAM with cudaHostRegister",
                                 (double)total_dram_bytes / (1024.0 * 1024.0 * 1024.0));
                    }
                } else {
                    LOG_ERROR("System RAM allocation also failed. Disabling L2 DRAM cache.");
                    dram_cache_capacity_ = 0;
                }
            } else {
                LOG_INFO("Allocated %.1f GB pinned DRAM cache with cudaMallocHost",
                         (double)total_dram_bytes / (1024.0 * 1024.0 * 1024.0));
            }
            if (dram_cache_capacity_ > 0 && dram_cache_pool_) {
                dram_cache_slots_.resize(dram_cache_capacity_);
                for (int i = 0; i < dram_cache_capacity_; i++) {
                    dram_cache_slots_[i].slot_index = i;
                    dram_cache_slots_[i].gpu_data = (char*)dram_cache_pool_ + (size_t)i * block_size;
                }
            }
        }

        // Initialize staging ring buffer
        staging_ring_.resize(NUM_STAGING_BUFFERS);
        for (auto& s : staging_ring_) {
            CUDA_CHECK(cudaMallocHost(&s.ptr, block_size));
            CUDA_CHECK(cudaEventCreateWithFlags(&s.event, cudaEventDisableTiming));
        }

        flat_vram_ptrs_.assign((size_t)n_layers * n_experts, nullptr);
        flat_vram_ptrs_gpu_.alloc((size_t)n_layers * n_experts * sizeof(void*));
        CUDA_CHECK(cudaMemset(flat_vram_ptrs_gpu_.data, 0, (size_t)n_layers * n_experts * sizeof(void*)));
        return true;
    }

    bool all_resident(int active_layers = 0) const {
        int n_active = (active_layers > 0) ? active_layers : n_layers_;
        return cache_capacity_ >= n_active * n_experts_;
    }

    inline void* try_get_expert_cached(int layer_id, int expert_id) {
        int64_t key = (int64_t)layer_id * n_experts_ + expert_id;
        if (key >= 0 && key < (int64_t)flat_vram_ptrs_.size()) {
            return flat_vram_ptrs_[key];
        }
        return nullptr;
    }

    void* touch_expert_cached(int layer_id, int expert_id) {
        int64_t key = (int64_t)layer_id * n_experts_ + expert_id;
        std::unique_lock<std::mutex> lock(cache_mutex_);
        auto it = key_to_slot_.find(key);
        if (it != key_to_slot_.end()) {
            access_counter_++;
            cache_slots_[it->second].last_used = access_counter_;
            return cache_slots_[it->second].gpu_data;
        }
        return nullptr;
    }

    void* get_expert(int layer_id, int expert_id, cudaStream_t stream) {
        return get_expert_async(layer_id, expert_id, stream);
    }

    void* get_expert_async(int layer_id, int expert_id, cudaStream_t stream) {
        std::unique_lock<std::mutex> lock(cache_mutex_);
        access_counter_++;

        int64_t key = (int64_t)layer_id * n_experts_ + expert_id;

        // 1. Check L1 Cache (VRAM)
        auto it = key_to_slot_.find(key);
        if (it != key_to_slot_.end()) {
            auto& slot = cache_slots_[it->second];
            slot.last_used = access_counter_;
            return slot.gpu_data;
        }

        // L1 Miss: find eviction candidate in L1
        int evict_slot = -1;
        int64_t oldest_time = INT64_MAX;
        for (int i = 0; i < cache_capacity_; i++) {
            if (cache_slots_[i].layer_id < 0) {
                evict_slot = i;
                break;
            }
            if (cache_slots_[i].last_used < oldest_time) {
                oldest_time = cache_slots_[i].last_used;
                evict_slot = i;
            }
        }

        auto& slot = cache_slots_[evict_slot];
        if (slot.layer_id >= 0) {
            int64_t old_key = (int64_t)slot.layer_id * n_experts_ + slot.expert_id;
            key_to_slot_.erase(old_key);
            flat_vram_ptrs_[old_key] = nullptr;
        }

        slot.layer_id = layer_id;
        slot.expert_id = expert_id;
        slot.last_used = access_counter_;
        key_to_slot_[key] = evict_slot;
        flat_vram_ptrs_[key] = slot.gpu_data;

        void* host_src_ptr = nullptr;
        int stage_idx = -1;
        bool needs_disk_read = false;

        // 2. Check L2 Cache (DRAM) if available
        if (dram_cache_capacity_ > 0) {
            auto dram_it = dram_key_to_slot_.find(key);
            if (dram_it != dram_key_to_slot_.end()) {
                // L2 Hit!
                auto& dram_slot = dram_cache_slots_[dram_it->second];
                dram_slot.last_used = access_counter_;
                host_src_ptr = dram_slot.gpu_data;
                if (g_log_experts) {
                    LOG_INFO("[ExpertCache] L2 Hit: L%d E%d -> L1 slot %d", layer_id, expert_id, evict_slot);
                }
            } else {
                // L2 Miss: Find L2 eviction candidate and read from disk directly into L2
                int evict_dram = -1;
                int64_t oldest_dram_time = INT64_MAX;
                for (int i = 0; i < dram_cache_capacity_; i++) {
                    if (dram_cache_slots_[i].layer_id < 0) {
                        evict_dram = i;
                        break;
                    }
                    if (dram_cache_slots_[i].last_used < oldest_dram_time) {
                        oldest_dram_time = dram_cache_slots_[i].last_used;
                        evict_dram = i;
                    }
                }

                auto& dram_slot = dram_cache_slots_[evict_dram];
                if (dram_slot.layer_id >= 0) {
                    int64_t old_dram_key = (int64_t)dram_slot.layer_id * n_experts_ + dram_slot.expert_id;
                    dram_key_to_slot_.erase(old_dram_key);
                }

                dram_slot.layer_id = layer_id;
                dram_slot.expert_id = expert_id;
                dram_slot.last_used = access_counter_;
                dram_key_to_slot_[key] = evict_dram;

                host_src_ptr = dram_slot.gpu_data;
                needs_disk_read = true;
                if (g_log_experts) {
                    LOG_INFO("[ExpertCache] L2 Miss (SSD read): L%d E%d -> L2 slot %d -> L1 slot %d", 
                             layer_id, expert_id, evict_dram, evict_slot);
                }
            }
        } else {
            // No L2 cache, fallback to staging buffer
            stage_idx = staging_idx_;
            staging_idx_ = (staging_idx_ + 1) % NUM_STAGING_BUFFERS;
            needs_disk_read = true;
            
            if (g_log_experts) {
                LOG_INFO("[ExpertCache] L1 Miss (SSD read, No L2): L%d E%d -> L1 slot %d", 
                         layer_id, expert_id, evict_slot);
            }
        }

        lock.unlock();

        if (stage_idx >= 0) {
            auto& stage = staging_ring_[stage_idx];
            CUDA_CHECK(cudaEventSynchronize(stage.event));
            
            int64_t file_offset = (int64_t)key * expert_block_size_;
            int64_t bytes_read = expert_file_.pread_exact(stage.ptr, expert_block_size_, file_offset);
            if (bytes_read != expert_block_size_) {
                LOG_ERROR("Expert read failed: layer=%d expert=%d offset=%ld got=%ld",
                          layer_id, expert_id, file_offset, (long)bytes_read);
                return nullptr;
            }
            host_src_ptr = stage.ptr;
            CUDA_CHECK(cudaMemcpyAsync(slot.gpu_data, host_src_ptr, expert_block_size_,
                                        cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaEventRecord(stage.event, stream));
        } else {
            if (needs_disk_read) {
                int64_t file_offset = (int64_t)key * expert_block_size_;
                int64_t bytes_read = expert_file_.pread_exact(host_src_ptr, expert_block_size_, file_offset);
                if (bytes_read != expert_block_size_) {
                    LOG_ERROR("Expert read failed: layer=%d expert=%d offset=%ld got=%ld",
                              layer_id, expert_id, file_offset, (long)bytes_read);
                    return nullptr;
                }
            }
            // Memory in host_src_ptr (L2) is already populated, async copy to L1
            CUDA_CHECK(cudaMemcpyAsync(slot.gpu_data, host_src_ptr, expert_block_size_,
                                        cudaMemcpyHostToDevice, stream));
        }

        return slot.gpu_data;
    }

    bool preload_all(int n_threads = 16) {
        int total = n_layers_ * n_experts_;
        int to_load = std::min(total, cache_capacity_);
        LOG_INFO("Preloading %d/%d experts into VRAM L1 cache...", to_load, total);
        
        auto start = std::chrono::steady_clock::now();
        std::atomic<int> loaded{0};

        std::vector<std::thread> workers;
        int chunk_size = (to_load + n_threads - 1) / n_threads;

        for (int t = 0; t < n_threads; t++) {
            int start_idx = t * chunk_size;
            int end_idx = std::min(start_idx + chunk_size, to_load);
            if (start_idx >= end_idx) continue;

            workers.emplace_back([this, start_idx, end_idx, &loaded, to_load]() {
                void* stage_ptr = nullptr;
                cudaStream_t stream = nullptr;
                CUDA_CHECK(cudaMallocHost(&stage_ptr, expert_block_size_));
                CUDA_CHECK(cudaStreamCreate(&stream));

                for (int i = start_idx; i < end_idx; i++) {
                    int l = i / n_experts_;
                    int e = i % n_experts_;
                    int64_t file_offset = (int64_t)i * expert_block_size_;

                    int64_t bytes_read = expert_file_.pread_exact(stage_ptr, expert_block_size_, file_offset);
                    if (bytes_read == expert_block_size_) {
                        void* dst = cache_slots_[i].gpu_data;
                        cache_slots_[i].layer_id = l;
                        cache_slots_[i].expert_id = e;
                        cache_slots_[i].last_used = 1;
                        flat_vram_ptrs_[i] = dst;
                        CUDA_CHECK(cudaMemcpyAsync(dst, stage_ptr, expert_block_size_, cudaMemcpyHostToDevice, stream));
                        CUDA_CHECK(cudaStreamSynchronize(stream));
                    }
                    int done = ++loaded;
                    if (done % 1000 == 0 || done == to_load) {
                        LOG_INFO("  Preloaded %d/%d experts into VRAM...", done, to_load);
                    }
                }
                CUDA_CHECK(cudaStreamDestroy(stream));
                CUDA_CHECK(cudaFreeHost(stage_ptr));
            });
        }

        for (auto& w : workers) {
            w.join();
        }

        for (int i = 0; i < to_load; i++) {
            if (cache_slots_[i].layer_id >= 0) {
                int64_t key = (int64_t)cache_slots_[i].layer_id * n_experts_ + cache_slots_[i].expert_id;
                key_to_slot_[key] = i;
            }
        }
        access_counter_ = 1;

        if (dram_cache_capacity_ > 0) {
            int dram_to_load = std::min(total - to_load, dram_cache_capacity_);
            int start_layer = to_load / n_experts_;
            int end_layer = (to_load + dram_to_load - 1) / n_experts_;
            LOG_INFO("Preloading %d/%d experts into DRAM L2 cache (layers %d..%d)...",
                     dram_to_load, dram_cache_capacity_, start_layer, end_layer);
            std::vector<std::thread> dram_workers;
            int dram_chunk = (dram_to_load + n_threads - 1) / n_threads;
            for (int t = 0; t < n_threads; t++) {
                int start_i = t * dram_chunk;
                int end_i = std::min(start_i + dram_chunk, dram_to_load);
                if (start_i >= end_i) continue;

                dram_workers.emplace_back([this, start_i, end_i, to_load]() {
                    for (int i = start_i; i < end_i; i++) {
                        int expert_global_idx = to_load + i;
                        int l = expert_global_idx / n_experts_;
                        int e = expert_global_idx % n_experts_;
                        int64_t file_offset = (int64_t)expert_global_idx * expert_block_size_;
                        void* dst = dram_cache_slots_[i].gpu_data;
                        int64_t bytes_read = expert_file_.pread_exact(dst, expert_block_size_, file_offset);
                        if (bytes_read == expert_block_size_) {
                            dram_cache_slots_[i].layer_id = l;
                            dram_cache_slots_[i].expert_id = e;
                            dram_cache_slots_[i].last_used = 1;
                        }
                    }
                });
            }
            for (auto& w : dram_workers) {
                w.join();
            }

            for (int i = 0; i < dram_to_load; i++) {
                if (dram_cache_slots_[i].layer_id >= 0) {
                    int64_t key = (int64_t)dram_cache_slots_[i].layer_id * n_experts_ + dram_cache_slots_[i].expert_id;
                    dram_key_to_slot_[key] = i;
                }
            }
        }

        // Copy flat pointer table to GPU for GPU-native kernel execution
        CUDA_CHECK(cudaMemcpy(flat_vram_ptrs_gpu_.data, flat_vram_ptrs_.data(),
                               flat_vram_ptrs_.size() * sizeof(void*), cudaMemcpyHostToDevice));

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        LOG_INFO("Preload completed in %.2f seconds", elapsed / 1000.0);
        return true;
    }

    const void* const* flat_vram_ptrs_gpu() const {
        return (const void* const*)flat_vram_ptrs_gpu_.data;
    }

    void cleanup() {
        expert_file_.close();
        if (cache_pool_gpu_) cudaFree(cache_pool_gpu_);
        if (dram_cache_pool_) {
            if (cudaFreeHost(dram_cache_pool_) != cudaSuccess) {
                cudaGetLastError();
#if defined(_WIN32) || defined(_WIN64)
                VirtualFree(dram_cache_pool_, 0, MEM_RELEASE);
#else
                free(dram_cache_pool_);
#endif
            }
            dram_cache_pool_ = nullptr;
        }
        flat_vram_ptrs_gpu_.free();
        for (auto& stage : staging_ring_) {
            if (stage.event) cudaEventDestroy(stage.event);
            if (stage.ptr) cudaFreeHost(stage.ptr);
        }
    }
private:
    GPUTensor flat_vram_ptrs_gpu_;
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
    cudaStream_t expert_streams_[32] = {nullptr};
    cudaEvent_t expert_events_[32] = {nullptr};

    void* streaming_slots_gpu_[2][32] = {{nullptr}};
    int streaming_buf_idx_ = 0;

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
        GPUTensor d_comp_kv_count;   // [1] int32_t on GPU for graph/device kernels
        GPUTensor d_attn_cache_len;  // [1] int32_t on GPU for dynamic MLA attention

        // Compressor state buffers (GPU, F32) for incremental decode compression
        GPUTensor comp_kv_state;     // [coff * ratio, coff * head_dim] F32
        GPUTensor comp_score_state;  // [coff * ratio, coff * head_dim] F32

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
        GPUTensor A_log;          // [48] BF16
        GPUTensor dt_bias;        // [48] BF16
        GPUTensor linear_norm_w;  // [128] BF16
        GPUTensor linear_out_proj, linear_out_proj_scale;// [5120, 6144]
        GPUTensor ssm_state;      // [48, 128, 128] F32
        GPUTensor conv_state;     // [10240, 4] BF16
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
    int32_t topk_ids_host_[32] = {0};
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
    GPUTensor buf_linear_a_;     // [48] BF16 for Qwen 3.8 DeltaNet decay A
    GPUTensor buf_linear_b_;     // [48] BF16 for Qwen 3.8 DeltaNet decay B
    GPUTensor buf_input_ids_;    // [MAX_SEQ_LEN] I32
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
    GPUTensor buf_combined_kv_;  // BF16 buffer for combined raw+compressed KV

    // Device-driven inputs for CUDA Graph & Device ArgMax
    GPUTensor buf_input_token_;  // [1] int32_t on GPU
    GPUTensor buf_input_pos_;    // [1] int32_t on GPU
    GPUTensor buf_argmax_out_;   // [1] int32_t on GPU for 4-byte sampling
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    bool graph_captured_ = false;

    // Imatrix calibration accumulators
    bool collect_imatrix_ = false;
    GPUTensor d_gate_accum_;
    GPUTensor d_down_accum_;
    GPUTensor d_expert_counts_;

    // Pre-allocated host buffers (zero runtime heap allocations)
    std::vector<float> router_probs_host_;
    std::vector<float> router_selection_host_;
    std::vector<int> router_indices_host_;
    std::vector<float> logits_host_;
    std::vector<float> probs_host_;
    __nv_bfloat16* logits_bf16_host_ = nullptr;

    ~MoecherEngine() {
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
        if (side_stream_) {
            cudaStreamDestroy(side_stream_);
            side_stream_ = nullptr;
        }
        if (dma_stream_) {
            cudaStreamDestroy(dma_stream_);
            dma_stream_ = nullptr;
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

    bool load(const std::string& manifest_path, float max_vram_gb = 0.0f, float dram_cache_gb = 0.0f, const std::string& expert_dtype_override = "") {
        LOG_INFO("Loading manifest: %s", manifest_path.c_str());

        std::ifstream f(manifest_path);
        if (!f.is_open()) { LOG_ERROR("Cannot open manifest"); return false; }
        json manifest;
        f >> manifest;

        // Parse config
        cfg_.from_json(manifest["model_config"]);
        if (!expert_dtype_override.empty()) {
            cfg_.expert_dtype = expert_dtype_override;
        }
        LOG_INFO("Model: %d layers, %d experts, %d active, hidden=%d, dtype=%s",
                 cfg_.num_hidden_layers, cfg_.n_routed_experts,
                 cfg_.num_experts_per_tok, cfg_.hidden_size, cfg_.expert_dtype.c_str());

        std::filesystem::path manifest_p(manifest_path);
        std::filesystem::path base_dir = manifest_p.parent_path();

        // Load tokenizer
        std::string tok_path = manifest["tokenizer"]["tokenizer_json"].get<std::string>();
        std::string tok_full = (base_dir / tok_path).string();
        if (!tokenizer_.load(tok_full) && !tokenizer_.load(tok_path)) {
            LOG_ERROR("Failed to load tokenizer from %s or %s", tok_full.c_str(), tok_path.c_str());
            return false;
        }

        // Init CUDA
        CUDA_CHECK(cudaStreamCreate(&main_stream_));
        CUDA_CHECK(cudaStreamCreate(&side_stream_));
        CUDA_CHECK(cudaStreamCreate(&dma_stream_));
        CUDA_CHECK(cudaEventCreateWithFlags(&side_event_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&main_event_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&dma_event_, cudaEventDisableTiming));
        for (int i = 0; i < 32; i++) {
            CUDA_CHECK(cudaStreamCreate(&expert_streams_[i]));
            CUDA_CHECK(cudaEventCreate(&expert_events_[i]));
        }
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
        CUBLAS_CHECK(cublasSetStream(cublas_handle_, main_stream_));
        cublasSetMathMode(cublas_handle_, CUBLAS_DEFAULT_MATH);

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

        // Load expert layout info
        auto& el = manifest["expert_layout"];
        int expert_block_size = el["block_size"].get<int>();
        int expert_n_layers = el["n_layers"].get<int>();
        if (expert_n_layers > cfg_.num_hidden_layers && cfg_.num_hidden_layers > 0) {
            expert_n_layers = cfg_.num_hidden_layers;
        }
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
        std::string expert_full = expert_path.empty() ? "" : (base_dir / expert_path).string();

        // Allocate working buffers first
        alloc_buffers();

        // Reserve VRAM: rest goes to expert cache
        size_t vram_free_after_dense;
        CUDA_CHECK(cudaMemGetInfo(&vram_free_after_dense, &vram_total));
        size_t total_experts_bytes = (size_t)expert_n_layers * expert_n_experts * expert_block_size;
        size_t cache_budget = vram_free_after_dense > (1ULL * 1024 * 1024 * 1024) ? (vram_free_after_dense - 1ULL * 1024 * 1024 * 1024) : vram_free_after_dense;
        
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
        
        size_t dram_cache_budget = (size_t)(dram_cache_gb * 1024.0 * 1024.0 * 1024.0);
        LOG_INFO("Expert L1 (VRAM) cache budget: %.1f GB", cache_budget / (1024.0 * 1024.0 * 1024.0));
        LOG_INFO("Expert L2 (DRAM) cache budget: %.1f GB", dram_cache_budget / (1024.0 * 1024.0 * 1024.0));
        
        if (cfg_.n_routed_experts > 0 && !expert_full.empty()) {
            if (!expert_loader_.init(expert_full, expert_block_size,
                                      expert_n_layers, expert_n_experts,
                                      cache_budget, dram_cache_budget)) return false;

            // Allocate double-buffered streaming slots in VRAM for offloading
            for (int b = 0; b < 2; b++) {
                for (int k = 0; k < 32; k++) {
                    CUDA_CHECK(cudaMalloc(&streaming_slots_gpu_[b][k], expert_loader_.expert_block_size_));
                }
            }

            // Preload experts into VRAM
            expert_loader_.preload_all();
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

        init_cuda_graph();

        LOG_INFO("Model loaded successfully");
        return true;
    }

    void init_cuda_graph() {
        if (cfg_.architecture == ModelArch::QWEN || !expert_loader_.all_resident(cfg_.num_hidden_layers)) {
            LOG_INFO("Running in eager mode for decode verification.");
            graph_captured_ = false;
            return;
        }
        LOG_INFO("Warming up and capturing CUDA Graph for decode acceleration...");
        int dummy_tok = 0;
        int dummy_pos = 0;
        CUDA_CHECK(cudaMemcpy(buf_input_token_.i32(), &dummy_tok, sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(buf_input_pos_.i32(), &dummy_pos, sizeof(int32_t), cudaMemcpyHostToDevice));

        // 3 warmup iterations
        for (int i = 0; i < 3; i++) {
            forward_token_eager(dummy_tok, dummy_pos);
        }
        CUDA_CHECK(cudaStreamSynchronize(main_stream_));

        CUDA_CHECK(cudaMemcpy(buf_input_token_.i32(), &dummy_tok, sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(buf_input_pos_.i32(), &dummy_pos, sizeof(int32_t), cudaMemcpyHostToDevice));
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
    }

    // ── Forward pass for a single token (decode mode) ───────────────────────
    // Returns logits on GPU [vocab_size] in F32

    // Persistent RNG for sampling (seeded once)
    std::mt19937 rng_{std::random_device{}()};

    void forward_token(int token_id, int position) {
        if (graph_captured_) {
            CUDA_CHECK(cudaMemcpyAsync(buf_input_token_.i32(), &token_id, sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
            CUDA_CHECK(cudaMemcpyAsync(buf_input_pos_.i32(), &position, sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
            CUDA_CHECK(cudaGraphLaunch(graph_exec_, main_stream_));
        } else {
            forward_token_eager(token_id, position);
        }
    }

    void forward_token_eager(int token_id, int position) {
        CUDA_CHECK(cudaMemcpy(buf_input_token_.i32(), &token_id, sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(buf_input_pos_.i32(), &position, sizeof(int32_t), cudaMemcpyHostToDevice));
        forward_token_device_body(token_id, position);
    }

    void forward_token_device_body(int token_id, int position) {
        int dim = cfg_.hidden_size;
        int hc = cfg_.hc_mult;

        if (cfg_.architecture == ModelArch::QWEN) {
            // 1. Standard Embedding lookup
            embedding_cuda(buf_hidden_.bf16(), embed_weight_.bf16(), buf_input_token_.i32(), 1, dim, main_stream_);

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

            // 4. Logits: hidden @ head_weight.T -> [vocab_size]
            compute_logits();
            return;
        }

        // 1 & 2. Embedding lookup and broadcast to HC copies: [1, dim] -> [hc, dim]
        embedding_broadcast_device_id_cuda(buf_hidden_.bf16(), buf_hc_state_.bf16(),
                                           embed_weight_.bf16(), buf_input_token_.i32(), dim, hc, main_stream_);

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
        compute_logits();
    }

    // ── Generate tokens ─────────────────────────────────────────────────────

    std::string generate(const std::vector<int>& prompt, int max_tokens = 512,
                         float temperature = 1.0f,
                         std::function<bool(const std::string&,bool)> on_token = nullptr,
                         float repetition_penalty = 1.0f,
                         bool enable_thinking = true,
                         int max_thinking_tokens = 2048,
                         float top_p = 0.95f,
                         float min_p = 0.0f,
                         int top_k = 1024) {
        // Reset debug flags for this request
        dbg_first_token_ = false;
        dbg_hc_pre_call_ = 0;
        dbg_head_ = true;
        dbg_sample_count_ = 0;

        // Reset KV caches for all layers (critical: stale cache = garbled output)
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
        if (buf_hc_after_attn_.data) {
            CUDA_CHECK(cudaMemset(buf_hc_after_attn_.data, 0, buf_hc_after_attn_.size_bytes));
        }

        // Prefill prompt
        for (size_t i = 0; i < prompt.size(); i++) {
            forward_token_eager(prompt[i], (int)i);
        }

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

        // Track whether we are inside a <think> block
        // DeepSeek V3/V4 thinking mode: the chat template appends <think> to the prompt,
        // so the model starts generation INSIDE the think block.
        bool in_think_block = enable_thinking;
        bool think_block_ended = false;
        std::string finish_reason = "length";

        // Set per-request repetition penalty
        current_rep_penalty_ = repetition_penalty;

        int content_tokens_generated = 0;
        int thinking_tokens_generated = 0;

        std::string last_think_token_str;

        for (int t = 0; content_tokens_generated < max_tokens; t++) {
            if (g_stop_requested.load()) {
                LOG_WARN("Generation stopped by client stop request at step %d", t);
                finish_reason = "stop";
                break;
            }

            // Graceful sentence boundary transition when thinking budget is reached
            if (in_think_block && max_thinking_tokens > 0 && thinking_tokens_generated >= max_thinking_tokens) {
                int grace_limit = max_thinking_tokens + 255;
                bool is_clean_boundary = false;
                if (!last_think_token_str.empty()) {
                    char last_char = last_think_token_str.back();
                    if (last_char == '\n' || last_char == '.' || last_char == ':') {
                        is_clean_boundary = true;
                    }
                }
                if (is_clean_boundary || thinking_tokens_generated >= grace_limit) {
                    LOG_WARN("Thinking budget reached (%d/%d tokens, clean_boundary=%d). Closing </think> and starting content.",
                             thinking_tokens_generated, max_thinking_tokens, is_clean_boundary ? 1 : 0);

                    // 1. Inject </think> to close thinking block
                    in_think_block = false;
                    think_block_ended = true;
                    if (think_end_id >= 0) {
                        forward_token(think_end_id, position);
                        position++;
                        output_ids.push_back(think_end_id);
                        history.push_back(think_end_id);
                    }

                    // 2. Inject and stream "Allright, here is the solution:\n\n" as the beginning of CONTENT
                    std::vector<int> transition_tokens = tokenizer_.encode("Allright, here is the solution:\n\n");
                    for (int tok_id : transition_tokens) {
                        forward_token(tok_id, position);
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
                    continue;
                }
            }

            // Sample from logits
            int next_token = sample_token(temperature, history, content_tokens_generated, in_think_block,
                                          top_k, top_p, min_p);
            
            // If EOS is sampled while inside think block, transition to </think> and continue
            if (in_think_block && (next_token == cfg_.eos_token_id || (eos2_id >= 0 && next_token == eos2_id))) {
                LOG_WARN("EOS sampled during thinking at step %d (%d thinking tokens). Transitioning to </think> and starting content.",
                         t, thinking_tokens_generated);
                in_think_block = false;
                think_block_ended = true;
                if (think_end_id >= 0) {
                    forward_token(think_end_id, position);
                    position++;
                    output_ids.push_back(think_end_id);
                    history.push_back(think_end_id);
                }
                continue;
            }

            // Check all EOS and stop conditions in content mode
            if (next_token == cfg_.eos_token_id || (eos2_id >= 0 && next_token == eos2_id)) {
                LOG_WARN("Stop token hit: token=%d (cfg_eos=%d, eos2=%d) at step %d (content: %d/%d, think: %d/%d)",
                         next_token, cfg_.eos_token_id, eos2_id, t,
                         content_tokens_generated, max_tokens,
                         thinking_tokens_generated, max_thinking_tokens);
                finish_reason = "stop";
                break;
            }

            if (!enable_thinking && (next_token == think_start_id || next_token == think_end_id)) {
                LOG_WARN("Thinking control token hit in no-think mode: token=%d", next_token);
                finish_reason = "stop";
                break;
            }

            output_ids.push_back(next_token);
            history.push_back(next_token);

            // Pipeline: Launch GPU forward pass for next token immediately so GPU runs concurrently with CPU text decoding
            forward_token(next_token, position);
            position++;
           
            // Handle think block filtering
            if (think_start_id >= 0 && next_token == think_start_id) {
                if (!think_block_ended) {
                    in_think_block = true;
                    think_block_ended = false;
                }
                continue;
            }
            if (think_end_id >= 0 && next_token == think_end_id) {
                in_think_block = false;
                think_block_ended = true;
                continue;
            }

            if (in_think_block) {
                thinking_tokens_generated++;
            } else {
                content_tokens_generated++;
            }

            std::string token_text = tokenizer_.decode({next_token});
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
                if (t > cfg_.bos_token_id) {
                    if (!g_quiet) {
                        printf("%s", token_buffer.c_str());
                        fflush(stdout);
                    }
                }
                if (on_token) {
                    if (!on_token(token_buffer, in_think_block)) {
                        LOG_WARN("Generation aborted by token callback at step %d", t);
                        finish_reason = "stop";
                        token_buffer.clear();
                        break;
                    }
                }
                token_buffer.clear();
            }

            // Graceful sentence-boundary stopping:
            // When we're within 80% of max_tokens and just completed a sentence,
            // stop early to avoid truncating mid-thought.
            if (false) {
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
                                        LOG_WARN("Graceful stop at sentence boundary (token %d/%d)",
                                                content_tokens_generated, max_tokens);
                                        finish_reason = "stop";
                                        break;
                                    }
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

        prompt_token_count_ = (int)prompt.size();
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
            if (dtype != "int4") {
                return false;
            }
            int64_t offset = info["offset"].get<int64_t>();
            int64_t nbytes = info["nbytes"].get<int64_t>();
            gpu_w.dtype = dtype;
            gpu_w.shape.clear();
            for (auto& s : info["shape"]) gpu_w.shape.push_back(s.get<int>());
            gpu_w.alloc(nbytes);
            CUDA_CHECK(cudaMemcpy(gpu_w.data, (char*)mapped + offset, nbytes, cudaMemcpyHostToDevice));

            if (info.contains("scale_offset")) {
                int64_t scale_offset = info["scale_offset"].get<int64_t>();
                int64_t scale_nbytes = info["scale_nbytes"].get<int64_t>();
                gpu_s.dtype = info.value("scale_dtype", "bfloat16");
                gpu_s.alloc(scale_nbytes);
                CUDA_CHECK(cudaMemcpy(gpu_s.data, (char*)mapped + scale_offset, scale_nbytes, cudaMemcpyHostToDevice));
            }
            return true;
        };

        // Load global tensors
        if (!load_tensor(embed_weight_, "embed.weight")) {
            if (!load_tensor(embed_weight_, "model.embed_tokens.weight")) {
                load_tensor(embed_weight_, "model.language_model.embed_tokens.weight");
            }
        }
        if (!load_tensor(head_weight_, "head.weight")) {
            if (!load_tensor(head_weight_, "lm_head.weight")) {
                if (!load_tensor(head_weight_, "model.language_model.lm_head.weight")) {
                    // Tie embeddings if lm_head is shared
                    if (embed_weight_.data) {
                        head_weight_.dtype = embed_weight_.dtype;
                        head_weight_.shape = embed_weight_.shape;
                        head_weight_.alloc(embed_weight_.size_bytes);
                        CUDA_CHECK(cudaMemcpy(head_weight_.data, embed_weight_.data, embed_weight_.size_bytes, cudaMemcpyDeviceToDevice));
                    }
                }
            }
        }
        if (!load_tensor(norm_weight_, "norm.weight")) {
            if (!load_tensor(norm_weight_, "model.norm.weight")) {
                load_tensor(norm_weight_, "model.language_model.norm.weight");
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

            if (cfg_.architecture == ModelArch::QWEN) {
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
                if (load_quant_tensor(lw.w_in_qkv, lw.w_in_qkv_scale, lm_prefix + ".linear_attn.in_proj_qkv.weight") ||
                    load_quant_tensor(lw.w_in_qkv, lw.w_in_qkv_scale, hf_prefix + ".linear_attn.in_proj_qkv.weight")) {
                    lw.is_linear_attn = true;
                    if (!load_quant_tensor(lw.w_in_z, lw.w_in_z_scale, lm_prefix + ".linear_attn.in_proj_z.weight")) {
                        load_quant_tensor(lw.w_in_z, lw.w_in_z_scale, hf_prefix + ".linear_attn.in_proj_z.weight");
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
                    if (!load_quant_tensor(lw.linear_out_proj, lw.linear_out_proj_scale, lm_prefix + ".linear_attn.out_proj.weight")) {
                        load_quant_tensor(lw.linear_out_proj, lw.linear_out_proj_scale, hf_prefix + ".linear_attn.out_proj.weight");
                    }

                    // Allocate linear attention recurrent states:
                    // SSM state: [48, 128, 128] F32 = 3 MB
                    lw.ssm_state.alloc(48 * 128 * 128 * sizeof(float));
                    CUDA_CHECK(cudaMemset(lw.ssm_state.data, 0, lw.ssm_state.size_bytes));
                    // Conv state: [10240, 4] BF16 = 80 KB
                    lw.conv_state.alloc(10240 * 4 * sizeof(__nv_bfloat16));
                    CUDA_CHECK(cudaMemset(lw.conv_state.data, 0, lw.conv_state.size_bytes));
                } else {
                    // Full GQA Projections
                    lw.is_linear_attn = false;
                    if (!load_quant_tensor(lw.w_q, lw.w_q_scale, lm_prefix + ".self_attn.q_proj.weight")) {
                        if (!load_quant_tensor(lw.w_q, lw.w_q_scale, hf_prefix + ".self_attn.q_proj.weight")) {
                            load_quant_tensor(lw.w_q, lw.w_q_scale, alt_prefix + ".attn.q.weight");
                        }
                    }
                    if (!load_quant_tensor(lw.w_k, lw.w_k_scale, lm_prefix + ".self_attn.k_proj.weight")) {
                        if (!load_quant_tensor(lw.w_k, lw.w_k_scale, hf_prefix + ".self_attn.k_proj.weight")) {
                            load_quant_tensor(lw.w_k, lw.w_k_scale, alt_prefix + ".attn.k.weight");
                        }
                    }
                    if (!load_quant_tensor(lw.w_v, lw.w_v_scale, lm_prefix + ".self_attn.v_proj.weight")) {
                        if (!load_quant_tensor(lw.w_v, lw.w_v_scale, hf_prefix + ".self_attn.v_proj.weight")) {
                            load_quant_tensor(lw.w_v, lw.w_v_scale, alt_prefix + ".attn.v.weight");
                        }
                    }
                    if (!load_quant_tensor(lw.w_o, lw.w_o_scale, lm_prefix + ".self_attn.o_proj.weight")) {
                        if (!load_quant_tensor(lw.w_o, lw.w_o_scale, hf_prefix + ".self_attn.o_proj.weight")) {
                            load_quant_tensor(lw.w_o, lw.w_o_scale, alt_prefix + ".attn.o.weight");
                        }
                    }
                    if (!load_tensor(lw.gqa_q_norm_w, lm_prefix + ".self_attn.q_norm.weight")) {
                        load_tensor(lw.gqa_q_norm_w, hf_prefix + ".self_attn.q_norm.weight");
                    }
                    if (!load_tensor(lw.gqa_k_norm_w, lm_prefix + ".self_attn.k_norm.weight")) {
                        load_tensor(lw.gqa_k_norm_w, hf_prefix + ".self_attn.k_norm.weight");
                    }

                    // Allocate GQA KV cache
                    int max_seq = cfg_.max_seq_len > 0 ? cfg_.max_seq_len : 32768;
                    int n_kv = cfg_.num_key_value_heads > 0 ? cfg_.num_key_value_heads : 4;
                    int h_dim = cfg_.head_dim > 0 ? cfg_.head_dim : 256;
                    lw.k_cache_gqa.alloc((size_t)max_seq * n_kv * h_dim * sizeof(__nv_bfloat16));
                    lw.v_cache_gqa.alloc((size_t)max_seq * n_kv * h_dim * sizeof(__nv_bfloat16));
                    CUDA_CHECK(cudaMemset(lw.k_cache_gqa.data, 0, lw.k_cache_gqa.size_bytes));
                    CUDA_CHECK(cudaMemset(lw.v_cache_gqa.data, 0, lw.v_cache_gqa.size_bytes));
                }

                // SwiGLU FFN Projections
                if (!load_quant_tensor(lw.w_gate, lw.w_gate_scale, lm_prefix + ".mlp.gate_proj.weight")) {
                    if (!load_quant_tensor(lw.w_gate, lw.w_gate_scale, hf_prefix + ".mlp.gate_proj.weight")) {
                        load_quant_tensor(lw.w_gate, lw.w_gate_scale, alt_prefix + ".ffn.gate.weight");
                    }
                }
                if (!load_quant_tensor(lw.w_up, lw.w_up_scale, lm_prefix + ".mlp.up_proj.weight")) {
                    if (!load_quant_tensor(lw.w_up, lw.w_up_scale, hf_prefix + ".mlp.up_proj.weight")) {
                        load_quant_tensor(lw.w_up, lw.w_up_scale, alt_prefix + ".ffn.up.weight");
                    }
                }
                if (!load_quant_tensor(lw.w_down, lw.w_down_scale, lm_prefix + ".mlp.down_proj.weight")) {
                    if (!load_quant_tensor(lw.w_down, lw.w_down_scale, hf_prefix + ".mlp.down_proj.weight")) {
                        load_quant_tensor(lw.w_down, lw.w_down_scale, alt_prefix + ".ffn.down.weight");
                    }
                }

                if ((l + 1) % 10 == 0 || l == cfg_.num_hidden_layers - 1) {
                    LOG_INFO("  Loaded Qwen layer %d/%d (%s%s)",
                             l + 1, cfg_.num_hidden_layers,
                             lw.is_linear_attn ? "Linear Attention DeltaNet" : "Full GQA Attention",
                             lw.w_gate.dtype == "int4" ? " [INT4]" : "");
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
        size_t max_q_dim = std::max({
            (size_t)n_heads * head_dim_val,
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
        buf_down_.alloc(max_inter * sizeof(__nv_bfloat16));
        buf_expert_out_.alloc(dim * sizeof(__nv_bfloat16));
        buf_moe_accum_.alloc(dim * sizeof(__nv_bfloat16));
        buf_dequant_.alloc(128 * 1024 * 1024);  // 128 MB for largest dequant
        buf_logits_.alloc((size_t)cfg_.vocab_size * sizeof(float));
        buf_scores_f32_.alloc(std::max(cfg_.n_routed_experts, 64) * sizeof(float));
        buf_scores_bf16_.alloc(std::max(cfg_.n_routed_experts, 64) * sizeof(__nv_bfloat16));
        buf_topk_vals_.alloc(std::max(cfg_.num_experts_per_tok, 64) * sizeof(float));
        buf_topk_idx_.alloc(std::max(cfg_.num_experts_per_tok, 64) * sizeof(int32_t));
        buf_linear_a_.alloc(64 * sizeof(__nv_bfloat16));
        buf_linear_b_.alloc(64 * sizeof(__nv_bfloat16));
        buf_input_ids_.alloc(sizeof(int32_t));
        buf_input_token_.alloc(sizeof(int32_t));
        buf_input_pos_.alloc(sizeof(int32_t));
        buf_hc_state_.alloc((size_t)hc * dim * sizeof(__nv_bfloat16));
        buf_hc_after_attn_.alloc((size_t)hc * dim * sizeof(__nv_bfloat16));  // static intermediate after attention
        buf_hc_pre_.alloc(hc * sizeof(float));
        buf_hc_post_.alloc(hc * sizeof(float));
        buf_hc_comb_.alloc(hc * hc * sizeof(float));
        buf_hc_mixes_.alloc((2 + hc) * hc * sizeof(float));
        buf_hc_input_.alloc((size_t)hc * dim * sizeof(float));
        buf_active_expert_ptrs_.alloc(32 * sizeof(void*));

        // Compressor working buffers
        // Max projection output size: coff=2 for ratio=4, head_dim=512 -> 1024
        int max_comp_dim = 2 * head_dim_val;  // coff=2
        buf_comp_proj_.alloc(max_comp_dim * sizeof(float));  // for wkv or wgate output
        buf_comp_out_.alloc(max_comp_dim * sizeof(float));   // for pooling output
        buf_comp_bf16_.alloc(head_dim_val * sizeof(__nv_bfloat16));  // compressed entry in BF16

        // Combined KV buffer: raw window + max compressed entries
        int max_combined = (cfg_.sliding_window > 0 ? cfg_.sliding_window : 64) + cfg_.max_compressed_entries;
        buf_combined_kv_.alloc((size_t)max_combined * head_dim_val * sizeof(__nv_bfloat16));

        buf_argmax_out_.alloc(sizeof(int32_t));

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
        float alpha = 1.0f, float beta = 0.0f)
    {
        if (M == 1) {
            gemv_bf16_out_bf16_cuda(C, B, A, N, K, main_stream_);
            return;
        }
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
        // State index:
        // For CSA (ratio=4, overlap): always write into second half (rows 4..7)
        // For HCA (ratio=128, non-overlap): write into rows 0..127
        int pos_mod = position % ratio;
        int state_idx = overlap ? (ratio + pos_mod) : pos_mod;

        // Copy kv projection to state slot
        CUDA_CHECK(cudaMemcpyAsync(
            lw.comp_kv_state.f32() + (size_t)state_idx * proj_dim,
            buf_comp_proj_.f32(), proj_dim * sizeof(float),
            cudaMemcpyDeviceToDevice, main_stream_));

        // 2. Project hidden state through compressor wgate: [proj_dim]
        gemv_bf16_cuda(buf_comp_out_.f32(), lw.comp_wgate.bf16(),
                       buf_hidden_.bf16(), proj_dim, dim, main_stream_);

        // 3. Add APE bias to gate scores: row = position % ratio
        CUDA_CHECK(cudaMemcpyAsync(
            lw.comp_score_state.f32() + (size_t)state_idx * proj_dim,
            buf_comp_out_.f32(), proj_dim * sizeof(float),
            cudaMemcpyDeviceToDevice, main_stream_));

        // Add APE bias: score_state[state_idx] += ape[pos_mod]
        float alpha_one = 1.0f;
        float* score_ptr = lw.comp_score_state.f32() + (size_t)state_idx * proj_dim;
        const float* ape_ptr = lw.comp_ape.f32() + (size_t)pos_mod * proj_dim;
        CUBLAS_CHECK(cublasSaxpy(cublas_handle_, proj_dim, &alpha_one,
                                 ape_ptr, 1, score_ptr, 1));

        // 4. Check if we have a complete block to compress
        bool should_compress = ((position + 1) % ratio == 0);
        if (!should_compress) return;

        // 5. Perform softmax-gated pooling
        if (overlap) {
            // CSA overlap:
            // Rows 0..ratio-1 (first half) contain previous block
            // Rows ratio..2*ratio-1 (second half) contain current block
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

            // Shift second half (rows 4..7) into first half (rows 0..3) for next block
            size_t half_bytes = (size_t)ratio * proj_dim * sizeof(float);
            CUDA_CHECK(cudaMemcpyAsync(
                lw.comp_kv_state.f32(),
                lw.comp_kv_state.f32() + (size_t)ratio * proj_dim,
                half_bytes, cudaMemcpyDeviceToDevice, main_stream_));
            CUDA_CHECK(cudaMemcpyAsync(
                lw.comp_score_state.f32(),
                lw.comp_score_state.f32() + (size_t)ratio * proj_dim,
                half_bytes, cudaMemcpyDeviceToDevice, main_stream_));
            CUDA_CHECK(cudaMemcpyAsync(
                lw.comp_kv_state.f32() + (size_t)ratio * proj_dim,
                lw.comp_kv_state.f32(),
                half_bytes, cudaMemcpyDeviceToDevice, main_stream_));
            CUDA_CHECK(cudaMemcpyAsync(
                lw.comp_score_state.f32() + (size_t)ratio * proj_dim,
                lw.comp_score_state.f32(),
                half_bytes, cudaMemcpyDeviceToDevice, main_stream_));
        } else {
            // HCA non-overlapping: pool over ratio rows
            compressor_pool_cuda(buf_comp_out_.f32(),
                                 lw.comp_kv_state.f32(),
                                 lw.comp_score_state.f32(),
                                 ratio, head_dim_val, main_stream_);
        }

        // 6. Apply RMSNorm in FP32 directly to pooled output, then convert to BF16
        rms_norm_weighted_f32_cuda(buf_comp_out_.f32(), buf_comp_out_.f32(),
                                   lw.comp_norm.bf16(), head_dim_val, cfg_.rms_norm_eps, main_stream_);
        f32_to_bf16_cuda(buf_comp_bf16_.bf16(), buf_comp_out_.f32(),
                         head_dim_val, main_stream_);

        // 7. Apply RoPE to compressed entry using compressed-layer frequencies
        int comp_pos = position + 1 - ratio;
        rope_cuda(buf_comp_bf16_.bf16(), 1, head_dim_val, rope_dim,
                  comp_pos, rope_freqs_compressed_.f32(), false, main_stream_);

        // 8. Store in compressed KV cache
        int comp_idx = lw.comp_kv_count;
        int max_comp = cfg_.max_seq_len / ratio + 2;
        if (comp_idx < max_comp) {
            CUDA_CHECK(cudaMemcpyAsync(
                lw.comp_kv_cache.bf16() + (size_t)comp_idx * head_dim_val,
                buf_comp_bf16_.bf16(), head_dim_val * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToDevice, main_stream_));
            lw.comp_kv_count = comp_idx + 1;
            int count_val = lw.comp_kv_count;
            CUDA_CHECK(cudaMemcpy(
                lw.d_comp_kv_count.i32(), &count_val, sizeof(int32_t),
                cudaMemcpyHostToDevice));
        }
    }

    // ── Forward one layer (Qwen / Llama GQA + SwiGLU) ───────────────────────

    void forward_layer_qwen(int layer_id, int position) {
        auto& lw = layers_[layer_id];
        int dim = cfg_.hidden_size;
        int n_q_heads = cfg_.num_attention_heads;
        int n_kv_heads = cfg_.num_key_value_heads;
        int head_dim = cfg_.head_dim;
        int inter_size = cfg_.intermediate_size > 0 ? cfg_.intermediate_size : cfg_.moe_intermediate_size;

        auto matmul_proj = [&](GPUTensor& out, GPUTensor& in_vec, GPUTensor& weight, GPUTensor& scale, int N, int K) {
            if (weight.dtype == "int4") {
                gemv_int4_cuda(out.bf16(), in_vec.bf16(), (const uint8_t*)weight.data, scale.bf16(), N, K, main_stream_);
            } else {
                gemm_bf16(out.bf16(), 1, N, K, in_vec.bf16(), weight.bf16());
            }
        };

        // 1. Attention Pre-RMSNorm: buf_hidden_ -> buf_hidden2_
        rms_norm_one_centered_cuda(buf_hidden2_.bf16(), buf_hidden_.bf16(),
                                   lw.attn_norm_w.bf16(), dim, cfg_.rms_norm_eps, main_stream_);

        if (lw.is_linear_attn) {
            // Qwen 3.8 Gated DeltaNet Linear Attention
            // 2. Projections: in_proj_qkv (10240), in_proj_z (6144), in_proj_a (48), in_proj_b (48)
            matmul_proj(buf_q_, buf_hidden2_, lw.w_in_qkv, lw.w_in_qkv_scale, 10240, dim);
            matmul_proj(buf_up_, buf_hidden2_, lw.w_in_z, lw.w_in_z_scale, 6144, dim);
            gemm_bf16(buf_linear_a_.bf16(), 1, 48, dim, buf_hidden2_.bf16(), lw.w_in_a.bf16());
            gemm_bf16(buf_linear_b_.bf16(), 1, 48, dim, buf_hidden2_.bf16(), lw.w_in_b.bf16());

            // 3. Fused 1D Causal Conv + Recurrent SSM Decode Step
            deltanet_linear_attention_decode_cuda(
                buf_attn_out_.bf16(),
                buf_q_.bf16(),
                buf_up_.bf16(),
                buf_linear_a_.bf16(),
                buf_linear_b_.bf16(),
                lw.conv1d_w.bf16(),
                lw.conv_state.bf16(),
                lw.A_log.bf16(),
                lw.dt_bias.bf16(),
                lw.linear_norm_w.bf16(),
                lw.ssm_state.f32(),
                16, 48, 128, main_stream_);

            // 4. Output Projection: buf_attn_out_ (6144) -> buf_hidden2_ (5120)
            matmul_proj(buf_hidden2_, buf_attn_out_, lw.linear_out_proj, lw.linear_out_proj_scale, dim, 6144);
        } else {
            // Standard Full GQA Attention (Gated)
            // 2. Q projection (2 * n_q_heads * head_dim), K projection, V projection
            matmul_proj(buf_q_, buf_hidden2_, lw.w_q, lw.w_q_scale, 2 * n_q_heads * head_dim, dim);
            matmul_proj(buf_gate_, buf_hidden2_, lw.w_k, lw.w_k_scale, n_kv_heads * head_dim, dim);
            matmul_proj(buf_up_, buf_hidden2_, lw.w_v, lw.w_v_scale, n_kv_heads * head_dim, dim);

            // 3 & 4. QK Norm + RoPE + GQA Decode + Sigmoid Gate
            qwen_gqa_decode_gated_cuda(
                buf_attn_out_.bf16(),
                buf_q_.bf16(),
                buf_gate_.bf16(),
                buf_up_.bf16(),
                lw.gqa_q_norm_w.bf16(),
                lw.gqa_k_norm_w.bf16(),
                lw.k_cache_gqa.bf16(),
                lw.v_cache_gqa.bf16(),
                n_q_heads, n_kv_heads, head_dim, position, cfg_.max_seq_len,
                cfg_.rope_theta, cfg_.rms_norm_eps, main_stream_);

            // 5. Output Projection: buf_attn_out_ -> buf_hidden2_
            matmul_proj(buf_hidden2_, buf_attn_out_, lw.w_o, lw.w_o_scale, dim, n_q_heads * head_dim);
        }

        // 6. Residual connection: buf_hidden_ += buf_hidden2_
        vector_add_bf16_cuda(buf_hidden_.bf16(), buf_hidden2_.bf16(), dim, main_stream_);

        // 7. FFN Pre-RMSNorm: buf_hidden_ -> buf_hidden2_
        rms_norm_one_centered_cuda(buf_hidden2_.bf16(), buf_hidden_.bf16(),
                                   lw.ffn_norm_w.bf16(), dim, cfg_.rms_norm_eps, main_stream_);

        // 8. Gate & Up projections for SwiGLU
        matmul_proj(buf_gate_, buf_hidden2_, lw.w_gate, lw.w_gate_scale, inter_size, dim);
        matmul_proj(buf_up_, buf_hidden2_, lw.w_up, lw.w_up_scale, inter_size, dim);

        // 9. Fused SiLU(Gate) * Up
        silu_mul_cuda(buf_gate_.bf16(), buf_gate_.bf16(), buf_up_.bf16(), inter_size, 0.0f, main_stream_);

        // 10. Down projection: buf_gate_ -> buf_hidden2_
        matmul_proj(buf_hidden2_, buf_gate_, lw.w_down, lw.w_down_scale, dim, inter_size);

        // 11. Residual connection: buf_hidden_ += buf_hidden2_
        vector_add_bf16_cuda(buf_hidden_.bf16(), buf_hidden2_.bf16(), dim, main_stream_);
    }

    // ── Forward one layer ───────────────────────────────────────────────────

    void forward_layer(int layer_id, int token_id, int position) {
        if (cfg_.architecture == ModelArch::QWEN) {
            forward_layer_qwen(layer_id, position);
            return;
        }

        auto& lw = layers_[layer_id];
        int dim = cfg_.hidden_size;

        // ── HC pre for attention: reads buf_hc_state_ ──
        hc_pre(buf_hc_state_, lw.hc_attn_fn, lw.hc_attn_scale, lw.hc_attn_base);

        // ── Attention norm ──
        rms_norm_cuda(buf_hidden_.bf16(), buf_hidden_.bf16(),
                      lw.attn_norm_w.bf16(), dim, cfg_.rms_norm_eps, main_stream_);
        CUDA_CHECK(cudaEventRecord(main_event_, main_stream_));

        // ── Attention ──
        forward_attention(layer_id, position);

        // ── HC post for attention: reads buf_hc_state_, writes to buf_hc_after_attn_ ──
        hc_post(buf_hc_after_attn_, buf_hc_state_);

        // ── HC pre for FFN: reads buf_hc_after_attn_ ──
        hc_pre(buf_hc_after_attn_, lw.hc_ffn_fn, lw.hc_ffn_scale, lw.hc_ffn_base);

        // ── FFN norm ──
        rms_norm_cuda(buf_hidden_.bf16(), buf_hidden_.bf16(),
                      lw.ffn_norm_w.bf16(), dim, cfg_.rms_norm_eps, main_stream_);
        CUDA_CHECK(cudaEventRecord(main_event_, main_stream_));

        // ── MoE FFN ──
        forward_moe(layer_id, token_id);

        // ── HC post for FFN: reads buf_hc_after_attn_, writes to buf_hc_state_ ──
        hc_post(buf_hc_state_, buf_hc_after_attn_);
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
        
        // Per-head Q normalization (DeepseekV4UnweightedRMSNorm)
        rms_norm_unweighted_batched_cuda(buf_q_.bf16(), buf_q_.bf16(),
                                         n_heads, head_dim_val, cfg_.rms_norm_eps,
                                         main_stream_);
        
        // Apply RoPE to last rope_dim elements of each Q head
        rope_device_pos_cuda(buf_q_.bf16(), n_heads, head_dim_val, rope_dim,
                             buf_input_pos_.i32(), layer_rope_freqs, false, main_stream_);

        // Synchronize main_stream_ with side_stream_ before compressor and attention
        CUDA_CHECK(cudaStreamWaitEvent(main_stream_, side_event_, 0));

        // ── Run compressor to accumulate/emit compressed KV entries ──────────
        int ratio = cfg_.layer_compress_ratio(layer_id);
        if (ratio > 0) {
            forward_compressor(layer_id, position);
        }

        // ── Attention computation ───────────────────────────────────────────
        float scale = 1.0f / sqrtf((float)head_dim_val);

        // Prepare combined raw and compressed KV buffer directly on GPU
        prepare_combined_kv_cuda(
            buf_combined_kv_.bf16(),
            lw.d_attn_cache_len.i32(),
            lw.kv_cache.bf16(),
            lw.comp_kv_cache.bf16(),
            buf_input_pos_.i32(),
            lw.d_comp_kv_count.i32(),
            window, head_dim_val, ratio,
            main_stream_);

        int max_combined = window + cfg_.max_compressed_entries;
        mla_attention_device_len_cuda(
            buf_q_.bf16(), buf_combined_kv_.bf16(), lw.attn_sink.f32(),
            buf_attn_out_.bf16(), lw.d_attn_cache_len.i32(), max_combined, head_dim_val, scale, main_stream_
        );

        // Inverse RoPE on attention output
        rope_device_pos_cuda(buf_attn_out_.bf16(), n_heads, head_dim_val, rope_dim,
                             buf_input_pos_.i32(), layer_rope_freqs, true, main_stream_);

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

        // 4. Populate active expert pointers
        const void* const* flat_ptrs = nullptr;
        if (expert_loader_.all_resident(cfg_.num_hidden_layers)) {
            flat_ptrs = expert_loader_.flat_vram_ptrs_gpu();
        } else {
            CUDA_CHECK(cudaMemcpyAsync(topk_ids_host_, buf_topk_idx_.i32(),
                                       top_k * sizeof(int32_t), cudaMemcpyDeviceToHost, main_stream_));
            CUDA_CHECK(cudaStreamSynchronize(main_stream_));

            std::future<void*> expert_futures[32];
            void* cached_blocks[32] = {nullptr};

            for (int k = 0; k < top_k; k++) {
                int eid = topk_ids_host_[k];
                if (eid < 0 || eid >= n_experts) continue;
                cached_blocks[k] = expert_loader_.touch_expert_cached(layer_id, eid);
                if (!cached_blocks[k]) {
                    expert_futures[k] = expert_pool_->enqueue([this, layer_id, eid, k]() {
                        return expert_loader_.get_expert(layer_id, eid, expert_streams_[k]);
                    });
                }
            }

            for (int k = 0; k < top_k; k++) {
                int eid = topk_ids_host_[k];
                if (eid < 0 || eid >= n_experts) {
                    active_expert_ptrs_host_[k] = nullptr;
                    continue;
                }
                void* ptr = cached_blocks[k];
                if (!ptr) {
                    ptr = expert_futures[k].get();
                }
                active_expert_ptrs_host_[k] = ptr;
                if (!ptr) {
                    LOG_ERROR("Failed to fetch expert L%d E%d", layer_id, eid);
                }
            }

            for (int k = 0; k < top_k; k++) {
                if (!cached_blocks[k] && active_expert_ptrs_host_[k]) {
                    CUDA_CHECK(cudaEventRecord(expert_events_[k], expert_streams_[k]));
                    CUDA_CHECK(cudaStreamWaitEvent(main_stream_, expert_events_[k], 0));
                }
            }

            CUDA_CHECK(cudaMemcpyAsync(buf_active_expert_ptrs_.data, active_expert_ptrs_host_,
                                       top_k * sizeof(void*), cudaMemcpyHostToDevice, main_stream_));
        }

        // 5. Launch 6 routed experts
        auto& w1_info = expert_parts_["w1.weight"];
        auto& w3_info = expert_parts_["w3.weight"];
        auto& w2_info = expert_parts_["w2.weight"];

        if (cfg_.expert_dtype == "iq2_xxs") {
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

        // 5. Shared expert executed concurrently on side_stream_ (must wait for ffn_norm to finish on main_stream_)
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

        // Wait for shared expert on side_stream_ before accumulating
        CUDA_CHECK(cudaStreamWaitEvent(main_stream_, side_event_, 0));

        // 6. Fused 6-way dynamic accumulation + shared expert directly into buf_hidden_
        fused_moe_accum_dynamic_cuda(buf_hidden_.bf16(), buf_down_.bf16(),
                                     buf_topk_vals_.f32(), shared_down, dim, main_stream_);
    }

    // ── Execute a single expert SwiGLU ──────────────────────────────────────

    void execute_expert_swiglu(void* expert_block, float routing_weight, int k) {
        int dim = cfg_.hidden_size;
        int moe_inter = cfg_.moe_intermediate_size;

        auto& w1_info = expert_parts_["w1.weight"];
        auto& w3_info = expert_parts_["w3.weight"];
        auto& w2_info = expert_parts_["w2.weight"];

        uint8_t* block = (uint8_t*)expert_block;
        uint8_t* w1_data = block + w1_info.offset_in_block;
        uint8_t* w3_data = block + w3_info.offset_in_block;
        uint8_t* w2_data = block + w2_info.offset_in_block;

        cudaStream_t stream = main_stream_;
        __nv_bfloat16* my_gate = buf_gate_.bf16() + k * moe_inter;
        __nv_bfloat16* my_up   = buf_up_.bf16()   + k * moe_inter;
        __nv_bfloat16* my_down = buf_down_.bf16() + k * dim;

        if (cfg_.expert_dtype == "iq2_xxs") {
            gemv_iq2_xxs_swiglu_fused_cuda(my_gate, buf_hidden_.bf16(),
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
            gemm_int2_dequant(my_gate, 1, moe_inter, dim, buf_hidden_.bf16(), w1_data, (__nv_bfloat16*)w1_scale, block_size, stream);
            gemm_int2_dequant(my_up,   1, moe_inter, dim, buf_hidden_.bf16(), w3_data, (__nv_bfloat16*)w3_scale, block_size, stream);
            silu_mul_cuda(my_gate, my_gate, my_up, moe_inter, cfg_.swiglu_limit, stream);
            gemm_int2_dequant(my_down, 1, dim, moe_inter, my_gate, w2_data, (__nv_bfloat16*)w2_scale, block_size, stream);
        } else {
            auto& w1s_info = expert_parts_["w1.scale"];
            auto& w3s_info = expert_parts_["w3.scale"];
            auto& w2s_info = expert_parts_["w2.scale"];
            uint8_t* w1_scale = block + w1s_info.offset_in_block;
            uint8_t* w3_scale = block + w3s_info.offset_in_block;
            uint8_t* w2_scale = block + w2s_info.offset_in_block;
            int w1_scale_cols = w1s_info.shape.size() > 1 ? w1s_info.shape[1] : 1;
            int w2_scale_cols = w2s_info.shape.size() > 1 ? w2s_info.shape[1] : 1;
            gemm_fp4_dequant(my_gate, 1, moe_inter, dim, buf_hidden_.bf16(), w1_data, w1_scale, w1_scale_cols, stream);
            gemm_fp4_dequant(my_up,   1, moe_inter, dim, buf_hidden_.bf16(), w3_data, w3_scale, w1_scale_cols, stream);
            silu_mul_cuda(my_gate, my_gate, my_up, moe_inter, cfg_.swiglu_limit, stream);
            gemm_fp4_dequant(my_down, 1, dim, moe_inter, my_gate, w2_data, w2_scale, w2_scale_cols, stream);
        }
    }

    void compute_logits() {
        int dim = cfg_.hidden_size;
        int vocab = cfg_.vocab_size;

        // Fast CUDA graph compatible matrix-vector multiplication directly to float32 logits
        gemv_bf16_cuda(buf_logits_.f32(), head_weight_.bf16(), buf_hidden_.bf16(), vocab, dim, main_stream_);
        argmax_f32_cuda(buf_argmax_out_.i32(), buf_logits_.f32(), vocab, main_stream_);
    }

    // ── Sample from logits ──────────────────────────────────────────────────

    float current_rep_penalty_ = 1.0f;  // Set per-request by generate()

    int sample_token(float temperature, const std::vector<int>& history, int step = 0, bool is_reasoning = true,
                     int top_k = 1024, float top_p = 0.95f, float min_p = 0.0f) {
        int vocab = cfg_.vocab_size;

        if (temperature <= 0.0f) {
            int best = 0;
            CUDA_CHECK(cudaMemcpyAsync(&best, buf_argmax_out_.i32(), sizeof(int32_t), cudaMemcpyDeviceToHost, main_stream_));
            CUDA_CHECK(cudaStreamSynchronize(main_stream_));
            return best;
        }

        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng_);
        sample_multinomial_f32_cuda(buf_argmax_out_.i32(), buf_logits_.f32(), vocab, temperature, r, main_stream_);
        int sampled = 0;
        CUDA_CHECK(cudaMemcpyAsync(&sampled, buf_argmax_out_.i32(), sizeof(int32_t), cudaMemcpyDeviceToHost, main_stream_));
        CUDA_CHECK(cudaStreamSynchronize(main_stream_));
        return sampled;
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
};

// ════════════════════════════════════════════════════════════════════════════════
//  Chat Template
// ════════════════════════════════════════════════════════════════════════════════

static const std::string DEEPSEEK_V4_REASONING_EFFORT_MAX_PREFIX =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

static std::vector<int> apply_chat_template(const json& messages, const BPETokenizer& tok, bool enable_thinking = true, const std::string& reasoning_effort = "high") {
    int IM_START = tok.get_token_id("<|im_start|>");
    int IM_END = tok.get_token_id("<|im_end|>");
    if (IM_START >= 0 && IM_END >= 0) {
        // ChatML template (Qwen / Llama / SmolLM)
        std::vector<int> result;
        bool has_system = (!messages.empty() && messages[0]["role"].get<std::string>() == "system");
        if (!has_system && enable_thinking) {
            std::string sys_prompt = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
            result.push_back(IM_START);
            auto sys_role = tok.encode("system\n" + sys_prompt);
            result.insert(result.end(), sys_role.begin(), sys_role.end());
            result.push_back(IM_END);
            auto nl_enc = tok.encode("\n");
            result.insert(result.end(), nl_enc.begin(), nl_enc.end());
        }
        for (size_t i = 0; i < messages.size(); i++) {
            std::string role = messages[i]["role"].get<std::string>();
            std::string content = messages[i]["content"].get<std::string>();

            result.push_back(IM_START);
            auto role_enc = tok.encode(role + "\n");
            result.insert(result.end(), role_enc.begin(), role_enc.end());
            auto content_enc = tok.encode(content);
            result.insert(result.end(), content_enc.begin(), content_enc.end());
            result.push_back(IM_END);
            auto nl_enc = tok.encode("\n");
            result.insert(result.end(), nl_enc.begin(), nl_enc.end());
        }
        if (!messages.empty() && messages.back()["role"].get<std::string>() == "user") {
            result.push_back(IM_START);
            auto asst_enc = tok.encode("assistant\n");
            result.insert(result.end(), asst_enc.begin(), asst_enc.end());
            if (enable_thinking) {
                auto think_enc = tok.encode("<think>\n");
                result.insert(result.end(), think_enc.begin(), think_enc.end());
            }
        }
        return result;
    }

    int BOS = tok.get_token_id("<｜begin of sentence｜>");
    if (BOS < 0) BOS = tok.get_token_id("<｜begin\xe2\x96\x81of\xe2\x96\x81sentence｜>");
    if (BOS < 0) BOS = 0;

    int EOS = tok.get_token_id("<｜end of sentence｜>");
    if (EOS < 0) EOS = tok.get_token_id("<｜end\xe2\x96\x81of\xe2\x96\x81sentence｜>");
    if (EOS < 1) EOS = 1;

    int USER = tok.get_token_id("<｜User｜>");
    if (USER < 0) USER = 128803;

    int ASSISTANT = tok.get_token_id("<｜Assistant｜>");
    if (ASSISTANT < 0) ASSISTANT = 128804;

    int THINK_BEGIN = tok.get_token_id("<think>");
    if (THINK_BEGIN < 0) THINK_BEGIN = 128821;

    int THINK_END = tok.get_token_id("</think>");
    if (THINK_END < 0) THINK_END = 128822;

    std::vector<int> result;
    result.push_back(BOS);

    if (enable_thinking && reasoning_effort == "max") {
        auto prefix_enc = tok.encode(DEEPSEEK_V4_REASONING_EFFORT_MAX_PREFIX);
        result.insert(result.end(), prefix_enc.begin(), prefix_enc.end());
    }
    for (size_t i = 0; i < messages.size(); i++) {
        std::string role = messages[i]["role"].get<std::string>();
        std::string content = messages[i]["content"].get<std::string>();

        if (role == "system") {
            // System message: raw content, no wrapper tokens (per official encoding)
            auto enc = tok.encode(content);
            result.insert(result.end(), enc.begin(), enc.end());
        } else if (role == "user") {
            result.push_back(USER);
            auto enc = tok.encode(content);
            result.insert(result.end(), enc.begin(), enc.end());
        } else if (role == "assistant") {
            result.push_back(ASSISTANT);
            result.push_back(THINK_END);
            auto enc = tok.encode(content);
            result.insert(result.end(), enc.begin(), enc.end());
            result.push_back(EOS); // <｜end▁of▁sentence｜>
        }
    }

    // Add assistant prompt for generation
    if (!messages.empty()) {
        std::string last_role = messages.back()["role"].get<std::string>();
        if (last_role == "user") {
            result.push_back(ASSISTANT);
            if (enable_thinking) {
                result.push_back(THINK_BEGIN);
            } else {
                result.push_back(THINK_END);
            }
        }
    }

    return result;
}

// ════════════════════════════════════════════════════════════════════════════════
//  HTTP Server (OpenAI-compatible)
// ════════════════════════════════════════════════════════════════════════════════

static std::mutex g_engine_mutex;  // serialize inference requests
static int g_request_counter = 0;  // for unique request IDs

static void run_server(MoecherEngine& engine, int port, int default_thinking_budget = 4096) {
    httplib::Server svr;

    // CORS headers and OPTIONS preflight
    svr.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "*");
        if (req.method == "OPTIONS") {
            res.status = 200;
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    // Serve web UI assets directly
    svr.set_mount_point("/", "./web");
    svr.set_mount_point("/", "../web");

    // Health & status check
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\",\"engine\":\"moecher\",\"version\":\"2.05\"}", "application/json");
    });

    svr.Get("/v1/models", [&engine](const httplib::Request&, httplib::Response& res) {
        std::string model_id = (engine.cfg_.architecture == ModelArch::QWEN) ? "qwen3.8-27b-q4" : "deepseek-v4-flash";
        json body = {
            {"object", "list"},
            {"data", {
                {
                    {"id", model_id},
                    {"object", "model"},
                    {"owned_by", "moecher"}
                }
            }}
        };
        res.set_content(body.dump(), "application/json");
    });

    // Chat completions
    svr.Post("/v1/chat/completions",
        [&engine, default_thinking_budget](const httplib::Request& req, httplib::Response& res) {
            g_stop_requested.store(false);
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
            {
                static std::mutex s_log_mutex;
                std::lock_guard<std::mutex> log_lock(s_log_mutex);
                std::ofstream log_file("client_request.log", std::ios::app);
                if (log_file.is_open()) {
                    auto now = std::chrono::system_clock::now();
                    std::time_t t = std::chrono::system_clock::to_time_t(now);
                    char time_buf[64];
                    std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
                    log_file << "[" << time_buf << "] " << req.body << std::endl;
                }
            }

            auto& messages = request["messages"];
            float temperature = request.value("temperature", 1.0f);
            float top_p = request.value("top_p", 0.95f);
            float min_p = request.value("min_p", 0.0f);
            int top_k = request.value("top_k", 1024);
            int max_tokens = request.value("max_tokens", 512);
            bool stream = request.value("stream", false);
            float repetition_penalty = request.value("repetition_penalty", 1.0f);
            std::string reasoning_effort = request.value("reasoning_effort", "high");
            bool enable_thinking = true;
            int max_thinking_tokens = default_thinking_budget;

            if (reasoning_effort == "none") {
                enable_thinking = false;
                max_thinking_tokens = 0;
            }

            if (request.contains("thinking") && request["thinking"].is_object()) {
                if (request["thinking"].contains("type") && request["thinking"]["type"] == "disabled") {
                    enable_thinking = false;
                    max_thinking_tokens = 0;
                }
                if (enable_thinking && request["thinking"].contains("budget_tokens") && request["thinking"]["budget_tokens"].is_number_integer()) {
                    max_thinking_tokens = request["thinking"]["budget_tokens"].get<int>();
                    if (max_thinking_tokens <= 0) enable_thinking = false;
                }
            }
            if (enable_thinking && request.contains("max_thinking_tokens") && request["max_thinking_tokens"].is_number_integer()) {
                max_thinking_tokens = request["max_thinking_tokens"].get<int>();
                if (max_thinking_tokens <= 0) enable_thinking = false;
            }
            if (enable_thinking && request.contains("thinking_budget") && request["thinking_budget"].is_number_integer()) {
                max_thinking_tokens = request["thinking_budget"].get<int>();
                if (max_thinking_tokens <= 0) enable_thinking = false;
            }

            if (!enable_thinking) {
                max_thinking_tokens = 0;
            }

            LOG_INFO("Reasoning effort: %s, Thinking: %s, Thinking budget: %d",
                     reasoning_effort.c_str(), enable_thinking ? "enabled" : "disabled", max_thinking_tokens);

            // Apply chat template
            std::vector<int> prompt = apply_chat_template(messages, engine.tokenizer_, enable_thinking, reasoning_effort);
            LOG_INFO("PROMPT (len=%zu)", prompt.size());

            std::string req_id = "chatcmpl-moecher-" + std::to_string(++g_request_counter);

            if (stream) {
                // SSE streaming
                res.set_chunked_content_provider(
                    "text/event-stream",
                    [&engine, prompt, max_tokens, temperature, req_id, repetition_penalty, enable_thinking, max_thinking_tokens, top_p, min_p, top_k](size_t offset, httplib::DataSink &sink) {
                        if (offset > 0) return false;
                        std::lock_guard<std::mutex> lock(g_engine_mutex);

                        json initial_chunk = {
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
                        std::string sse = "data: " + initial_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                        sink.write(sse.data(), sse.size());
                        
                        engine.generate(prompt, max_tokens, temperature, [&](const std::string& text, bool is_reasoning) -> bool {
                            if (g_stop_requested.load()) return false;
                            if (text.empty()) return true; // Skip empty tokens
          
                            json delta_chunk = {
                                {"id", req_id},
                                {"object", "chat.completion.chunk"},
                                {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                {"model", "deepseek-v4-flash"},
                                {"choices", {{
                                    {"index", 0},
                                    {"delta", is_reasoning ? json{{"reasoning_content", text}} : json{{"content", text}}},
                                    {"finish_reason", nullptr}
                                }}}
                            };

                            std::string sse_chunk = "data: " + delta_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                            bool ok = sink.write(sse_chunk.data(), sse_chunk.size());
                            return ok && !g_stop_requested.load();
                        }, repetition_penalty, enable_thinking, max_thinking_tokens, top_p, min_p, top_k);

                        std::string final_finish_reason = engine.last_finish_reason_.empty() ? "stop" : engine.last_finish_reason_;

                        json finish = {
                            {"id", req_id},
                            {"object", "chat.completion.chunk"},
                            {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                            {"model", "deepseek-v4-flash"},
                            {"choices", {{
                                {"index", 0},
                                {"delta", json::object()},
                                {"finish_reason", final_finish_reason}
                            }}},
                            {"usage", {
                                {"prompt_tokens", engine.prompt_token_count_},
                                {"completion_tokens", engine.completion_token_count_},
                                {"total_tokens", engine.prompt_token_count_ + engine.completion_token_count_}
                            }}
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
                    response_text = engine.generate(prompt, max_tokens, temperature, nullptr, repetition_penalty, enable_thinking, max_thinking_tokens, top_p, min_p, top_k);
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

    // Stop / abort generation endpoints
    svr.Post("/v1/chat/stop", [](const httplib::Request&, httplib::Response& res) {
        LOG_WARN("Received stop request on /v1/chat/stop");
        g_stop_requested.store(true);
        json body = {{"status", "ok"}, {"message", "Generation stopping"}};
        res.set_content(body.dump(), "application/json");
    });
    svr.Post("/v1/stop", [](const httplib::Request&, httplib::Response& res) {
        LOG_WARN("Received stop request on /v1/stop");
        g_stop_requested.store(true);
        json body = {{"status", "ok"}, {"message", "Generation stopping"}};
        res.set_content(body.dump(), "application/json");
    });

    LOG_INFO("Server listening on port %d", port);
    LOG_INFO("version 2.03");
    g_server_ready = true;
    svr.listen("0.0.0.0", port);
}

// ════════════════════════════════════════════════════════════════════════════════
//  Main
// ════════════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
#if defined(_WIN32) || defined(_WIN64)
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    std::string manifest_path = "moecher_manifest.json";
    int port = 8001;
    float max_vram_gb = 0.0f;
    float dram_cache_gb = 0.0f;
    std::string log_path = "moecher.log";
    std::string expert_dtype_override = "";
    int default_thinking_budget = 4096;
    std::string imatrix_dataset = "";
    std::string imatrix_out = "";
    int imatrix_max_tokens = -1;

    for (int i = 1; i < argc; i++) {
        if ((std::string(argv[i]) == "--manifest" || std::string(argv[i]) == "-m") && i + 1 < argc) {
            manifest_path = argv[++i];
        } else if ((std::string(argv[i]) == "--port" || std::string(argv[i]) == "-p") && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--log" && i + 1 < argc) {
            log_path = argv[++i];
        } else if ((std::string(argv[i]) == "--max-vram" || std::string(argv[i]) == "-V") && i + 1 < argc) {
            max_vram_gb = std::stof(argv[++i]);
        } else if (std::string(argv[i]) == "--dram-cache-gb" && i + 1 < argc) {
            dram_cache_gb = std::stof(argv[++i]);
        } else if (std::string(argv[i]) == "--expert-dtype" && i + 1 < argc) {
            expert_dtype_override = argv[++i];
        } else if ((std::string(argv[i]) == "--thinking-budget" || std::string(argv[i]) == "--budget" || std::string(argv[i]) == "--max-thinking-tokens") && i + 1 < argc) {
            default_thinking_budget = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--imatrix-dataset" && i + 1 < argc) {
            imatrix_dataset = argv[++i];
        } else if (std::string(argv[i]) == "--imatrix-out" && i + 1 < argc) {
            imatrix_out = argv[++i];
        } else if (std::string(argv[i]) == "--imatrix-max-tokens" && i + 1 < argc) {
            imatrix_max_tokens = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--log-experts") {
            g_log_experts = true;
        } else if (std::string(argv[i]) == "--no-log-tokens") {
            g_log_tokens = false;
        } else if (std::string(argv[i]) == "--quiet" || std::string(argv[i]) == "-q") {
            g_quiet = true;
            g_log_tokens = false;
        }
    }

    // Open log file
    g_log_file.open(log_path, std::ios::app);
    LOG_INFO("=== moecher starting ===");
    LOG_INFO("=== v2.05 ===");
    LOG_INFO("Default thinking token budget: %d", default_thinking_budget);

    MoecherEngine engine;
    if (!engine.load(manifest_path, max_vram_gb, dram_cache_gb, expert_dtype_override)) {
        LOG_ERROR("Failed to load model");
        return 1;
    }

    if (!imatrix_dataset.empty() && !imatrix_out.empty()) {
        bool ok = engine.collect_imatrix(imatrix_dataset, imatrix_out, imatrix_max_tokens);
        return ok ? 0 : 1;
    }

    run_server(engine, port, default_thinking_budget);
    return 0;
}
