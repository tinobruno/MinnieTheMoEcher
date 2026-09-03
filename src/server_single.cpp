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
#include "embedded_web.hpp"

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
bool g_track_experts = false;
bool g_track_reset = false;
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
    bool dram_is_pinned_ = false;  // True if DRAM cache is pinned (cudaMallocHost or cudaHostRegister succeeded)
    std::vector<ExpertCacheEntry> dram_cache_slots_;
    std::unordered_map<int64_t, int> dram_key_to_slot_; // (layer*n_experts+expert) -> L2 slot index
    std::vector<void*> flat_dram_ptrs_;                 // Lock-free flat array for 0-latency DRAM lookups

    // Pinned staging buffers for DMA when DRAM cache is unpinned
    // cudaMemcpyAsync from pageable memory is SYNCHRONOUS — it blocks the stream.
    // These pinned buffers enable: memcpy(DRAM→pinned) + cudaMemcpyAsync(pinned→VRAM) = true async DMA.
    static constexpr int NUM_DMA_STAGING = 8;  // One per expert stream
    void* dma_staging_[NUM_DMA_STAGING] = {nullptr};

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

    // L2 victim cache: writeback stream for async VRAM→DRAM demotion
    cudaStream_t dram_writeback_stream_ = nullptr;
    cudaEvent_t writeback_event_ = nullptr;  // Ensures writeback completes before slot reuse

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
    // ── L2 Victim Cache: demote evicted L1 expert to L2 DRAM ──────────────
    // Called under cache_mutex_. Queues async VRAM→DRAM copy on writeback stream.
    // Throttled to MAX_DEMOTIONS_PER_TOKEN to avoid saturating PCIe with D2H traffic
    // which competes with NVMe SSD reads on the same bus.
    static constexpr int MAX_DEMOTIONS_PER_TOKEN = 16;
    int demotions_this_token_ = 0;
    int64_t demotion_token_id_ = -1;  // access_counter_ of current token

    void demote_to_l2(int layer_id, int expert_id, int64_t key, void* vram_src) {
        if (dram_cache_capacity_ <= 0 || !dram_writeback_stream_) return;

        // Reset demotion counter on new token
        if (access_counter_ != demotion_token_id_) {
            demotion_token_id_ = access_counter_;
            demotions_this_token_ = 0;
        }

        // Throttle: skip demotion if we've hit the per-token cap
        if (demotions_this_token_ >= MAX_DEMOTIONS_PER_TOKEN) return;

        // Skip if already in L2 (e.g. preloaded and never evicted)
        if (dram_key_to_slot_.count(key)) return;

        // Find LRU eviction candidate in L2
        int evict_dram = -1;
        int64_t oldest_d = INT64_MAX;
        for (int i = 0; i < dram_cache_capacity_; i++) {
            if (dram_cache_slots_[i].layer_id < 0) { evict_dram = i; break; }
            if (dram_cache_slots_[i].last_used < oldest_d) {
                oldest_d = dram_cache_slots_[i].last_used;
                evict_dram = i;
            }
        }
        if (evict_dram < 0) return;

        auto& ds = dram_cache_slots_[evict_dram];
        if (ds.layer_id >= 0) {
            int64_t old_key = (int64_t)ds.layer_id * n_experts_ + ds.expert_id;
            dram_key_to_slot_.erase(old_key);
        }

        ds.layer_id = layer_id;
        ds.expert_id = expert_id;
        ds.last_used = access_counter_;
        dram_key_to_slot_[key] = evict_dram;

        // Async copy VRAM → pinned DRAM (non-blocking, dedicated stream)
        cudaMemcpyAsync(ds.gpu_data, vram_src, expert_block_size_,
                        cudaMemcpyDeviceToHost, dram_writeback_stream_);
        // Record event so L2 hit path can wait for writeback to complete
        cudaEventRecord(writeback_event_, dram_writeback_stream_);
        demotions_this_token_++;
    }

    // Call before submitting new H2D DMA to an evicted L1 slot.
    // Makes the expert stream wait for any pending writeback to complete.
    void wait_for_writeback(cudaStream_t expert_stream) {
        if (dram_writeback_stream_ && writeback_event_) {
            cudaStreamWaitEvent(expert_stream, writeback_event_, 0);
        }
    }

    bool init(const std::string& expert_bin_path, int block_size,
              int n_layers, int n_experts, size_t cache_budget_bytes, size_t dram_budget_bytes,
              bool use_buffered_io = false) {
        expert_block_size_ = block_size;
        n_layers_ = n_layers;
        n_experts_ = n_experts;

        bool use_direct = !use_buffered_io;
        if (!expert_file_.open_read(expert_bin_path, use_direct)) {
            LOG_ERROR("Cannot open expert bin: %s", expert_bin_path.c_str());
            return false;
        }
        if (use_buffered_io) {
            LOG_INFO("Expert I/O: BUFFERED mode (OS file cache enabled — recommended for offloading)");
        } else {
            LOG_INFO("Expert I/O: DIRECT mode (FILE_FLAG_NO_BUFFERING / O_DIRECT)");
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
                        dram_is_pinned_ = false;
                        LOG_WARN("cudaHostRegister could not pin %.1f GB DRAM cache. Using pinned staging buffers for async DMA.",
                                 (double)total_dram_bytes / (1024.0 * 1024.0 * 1024.0));
                    } else {
                        dram_is_pinned_ = true;
                        LOG_INFO("Pinned %.1f GB DRAM cache into physical RAM with cudaHostRegister",
                                 (double)total_dram_bytes / (1024.0 * 1024.0 * 1024.0));
                    }
                } else {
                    LOG_ERROR("System RAM allocation also failed. Disabling L2 DRAM cache.");
                    dram_cache_capacity_ = 0;
                }
            } else {
                dram_is_pinned_ = true;
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

        // Create writeback stream for L2 victim cache demotion
        if (dram_cache_capacity_ > 0) {
            CUDA_CHECK(cudaStreamCreate(&dram_writeback_stream_));
            CUDA_CHECK(cudaEventCreateWithFlags(&writeback_event_, cudaEventDisableTiming));
            LOG_INFO("L2 victim cache enabled: evicted L1 experts will be demoted to L2 DRAM");
        }

        // Allocate pinned DMA staging buffers (used when DRAM cache is unpinned)
        if (!dram_is_pinned_ && dram_cache_capacity_ > 0) {
            LOG_INFO("Allocating %d pinned DMA staging buffers (%.1f MB each) for async DRAM→VRAM transfers",
                     NUM_DMA_STAGING, (double)block_size / (1024.0 * 1024.0));
            for (int i = 0; i < NUM_DMA_STAGING; i++) {
                CUDA_CHECK(cudaMallocHost(&dma_staging_[i], block_size));
            }
        }
        return true;
    }

    bool all_resident(int active_layers = 0) const {
        int n_active = (active_layers > 0) ? active_layers : n_layers_;
        return cache_capacity_ >= n_active * n_experts_;
    }

    std::string get_expert_location(int layer_id, int expert_id) const {
        if (all_resident(n_layers_)) {
            return "L1 (VRAM)";
        }
        int64_t key = (int64_t)layer_id * n_experts_ + expert_id;
        if (key >= 0 && key < (int64_t)flat_vram_ptrs_.size() && flat_vram_ptrs_[key] != nullptr) {
            return "L1 (VRAM)";
        }
        if (key >= 0 && key < (int64_t)flat_dram_ptrs_.size() && flat_dram_ptrs_[key] != nullptr) {
            return "L2 (DRAM)";
        }
        return "SSD (Disk)";
    }

    std::string get_expert_tier(int layer_id, int expert_id) const {
        if (all_resident(n_layers_)) {
            return "l1";
        }
        int64_t key = (int64_t)layer_id * n_experts_ + expert_id;
        if (key >= 0 && key < (int64_t)flat_vram_ptrs_.size() && flat_vram_ptrs_[key] != nullptr) {
            return "l1";
        }
        if (key >= 0 && key < (int64_t)flat_dram_ptrs_.size() && flat_dram_ptrs_[key] != nullptr) {
            return "l2";
        }
        return "ssd";
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
            // Demote evicted expert to L2 DRAM (victim cache)
            demote_to_l2(slot.layer_id, slot.expert_id, old_key, slot.gpu_data);
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
                // L2 miss — use staging ring buffer for SSD read
                // Do NOT evict L2 entries: L2 is a stable fast cache, not SSD staging
                stage_idx = staging_idx_;
                staging_idx_ = (staging_idx_ + 1) % NUM_STAGING_BUFFERS;
                needs_disk_read = true;
                if (g_log_experts) {
                    LOG_INFO("[ExpertCache] L2 Miss (SSD read via staging): L%d E%d -> L1 slot %d", 
                             layer_id, expert_id, evict_slot);
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
            // SSD path: 2.5ms SSD read provides enough time for writeback to complete
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
            // L2 hit path: DMA is immediate, must wait for writeback to complete first
            wait_for_writeback(stream);
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

    // ── Batch expert loading: resolve all experts in ONE lock, parallel I/O outside ──
    // Returns the number of experts that needed DMA (non-cached).
    // gpu_ptrs_out[k] is populated with the VRAM pointer for expert k.
    // needs_dma_out[k] is true if expert k required a DMA transfer (not L1-cached).
    int batch_get_experts(int layer_id, const int32_t* expert_ids, int count,
                          void* gpu_ptrs_out[], bool needs_dma_out[],
                          cudaStream_t* streams, ThreadPool* pool) {
        struct BatchSlot {
            int64_t key;
            int l1_slot;           // L1 eviction slot index
            void* gpu_dst;         // VRAM destination
            void* host_src;        // DRAM source (L2 or staging)
            int stage_idx;         // staging ring index (-1 if L2)
            bool needs_disk;       // true if SSD read needed
            bool l1_hit;           // true if already in L1
        };

        BatchSlot batch[32];
        int n_uncached = 0;

        // ── Phase 1: Single lock — resolve all experts, allocate L1/L2 slots ──
        {
            std::unique_lock<std::mutex> lock(cache_mutex_);
            access_counter_++;

            for (int k = 0; k < count; k++) {
                int eid = expert_ids[k];
                if (eid < 0 || eid >= n_experts_) {
                    gpu_ptrs_out[k] = nullptr;
                    needs_dma_out[k] = false;
                    batch[k].l1_hit = true;
                    continue;
                }

                int64_t key = (int64_t)layer_id * n_experts_ + eid;
                batch[k].key = key;
                batch[k].stage_idx = -1;
                batch[k].needs_disk = false;
                batch[k].l1_hit = false;

                // Check L1 (VRAM)
                auto it = key_to_slot_.find(key);
                if (it != key_to_slot_.end()) {
                    auto& slot = cache_slots_[it->second];
                    slot.last_used = access_counter_;
                    gpu_ptrs_out[k] = slot.gpu_data;
                    needs_dma_out[k] = false;
                    batch[k].l1_hit = true;
                    continue;
                }

                // L1 miss — find eviction candidate
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
                    // Demote evicted expert to L2 DRAM (victim cache)
                    demote_to_l2(slot.layer_id, slot.expert_id, old_key, slot.gpu_data);
                }

                slot.layer_id = layer_id;
                slot.expert_id = eid;
                slot.last_used = access_counter_;
                key_to_slot_[key] = evict_slot;
                flat_vram_ptrs_[key] = slot.gpu_data;

                batch[k].l1_slot = evict_slot;
                batch[k].gpu_dst = slot.gpu_data;
                gpu_ptrs_out[k] = slot.gpu_data;
                needs_dma_out[k] = true;
                n_uncached++;

                // Check L2 (DRAM)
                if (dram_cache_capacity_ > 0) {
                    auto dram_it = dram_key_to_slot_.find(key);
                    if (dram_it != dram_key_to_slot_.end()) {
                        // L2 hit
                        auto& dram_slot = dram_cache_slots_[dram_it->second];
                        dram_slot.last_used = access_counter_;
                        batch[k].host_src = dram_slot.gpu_data;
                    } else {
                        // L2 miss — use staging ring buffer for SSD read
                        // Do NOT evict L2 entries: L2 is a stable fast cache, not SSD staging
                        batch[k].stage_idx = staging_idx_;
                        staging_idx_ = (staging_idx_ + 1) % NUM_STAGING_BUFFERS;
                        batch[k].needs_disk = true;
                    }
                } else {
                    // No L2 — use staging ring
                    batch[k].stage_idx = staging_idx_;
                    staging_idx_ = (staging_idx_ + 1) % NUM_STAGING_BUFFERS;
                    batch[k].needs_disk = true;
                }
            }
        }
        // ── Mutex released ──

        if (n_uncached == 0) return 0;

        // ── Phase 2: Parallel I/O + DMA outside the lock ──
        // Submit all SSD reads and DRAM→VRAM DMAs concurrently
        std::future<bool> io_futures[32];

        for (int k = 0; k < count; k++) {
            if (batch[k].l1_hit) continue;

            io_futures[k] = pool->enqueue([this, &batch, k, &streams]() -> bool {
                auto& b = batch[k];
                void* host_ptr = b.host_src;

                if (b.stage_idx >= 0) {
                    // SSD path: 2.5ms SSD read provides enough time for writeback to complete
                    auto& stage = staging_ring_[b.stage_idx];
                    CUDA_CHECK(cudaEventSynchronize(stage.event));
                    int64_t bytes = expert_file_.pread_exact(stage.ptr, expert_block_size_, b.key * expert_block_size_);
                    if (bytes != expert_block_size_) return false;
                    host_ptr = stage.ptr;
                    CUDA_CHECK(cudaMemcpyAsync(b.gpu_dst, host_ptr, expert_block_size_,
                                                cudaMemcpyHostToDevice, streams[k]));
                    CUDA_CHECK(cudaEventRecord(stage.event, streams[k]));
                } else {
                    // L2 hit path: DMA is immediate, must wait for writeback to complete first
                    wait_for_writeback(streams[k]);
                    if (b.needs_disk) {
                        // L2 miss — read from SSD into L2 DRAM slot
                        int64_t bytes = expert_file_.pread_exact(host_ptr, expert_block_size_, b.key * expert_block_size_);
                        if (bytes != expert_block_size_) return false;
                    }
                    // L2 hit or just-read — DMA from DRAM to VRAM
                    CUDA_CHECK(cudaMemcpyAsync(b.gpu_dst, host_ptr, expert_block_size_,
                                                cudaMemcpyHostToDevice, streams[k]));
                }
                return true;
            });
        }

        // Wait for all I/O to complete
        for (int k = 0; k < count; k++) {
            if (batch[k].l1_hit) continue;
            if (!io_futures[k].get()) {
                LOG_ERROR("Failed to fetch expert L%d E%d", layer_id, expert_ids[k]);
                gpu_ptrs_out[k] = nullptr;
            }
        }

        return n_uncached;
    }

    bool preload_all(int n_threads = 16, const std::vector<uint32_t>& freq_counts = {}) {
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

            if (!freq_counts.empty() && (int)freq_counts.size() == total) {
                // Frequency-guided smart preload: pick the hottest experts across all non-L1 slots
                struct Candidate {
                    int layer_id;
                    int expert_id;
                    uint32_t count;
                };
                std::vector<Candidate> candidates;
                candidates.reserve(total - to_load);
                for (int l = 0; l < n_layers_; l++) {
                    for (int e = 0; e < n_experts_; e++) {
                        int64_t key = (int64_t)l * n_experts_ + e;
                        if (key_to_slot_.find(key) == key_to_slot_.end()) {
                            candidates.push_back({l, e, freq_counts[(size_t)l * n_experts_ + e]});
                        }
                    }
                }
                std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
                    if (a.count != b.count) return a.count > b.count;
                    if (a.layer_id != b.layer_id) return a.layer_id < b.layer_id;
                    return a.expert_id < b.expert_id;
                });

                LOG_INFO("Preloading %d/%d hot experts into DRAM L2 cache across all %d layers (frequency-guided)...",
                         dram_to_load, dram_cache_capacity_, n_layers_);
                std::vector<std::thread> dram_workers;
                int dram_chunk = (dram_to_load + n_threads - 1) / n_threads;
                for (int t = 0; t < n_threads; t++) {
                    int start_i = t * dram_chunk;
                    int end_i = std::min(start_i + dram_chunk, dram_to_load);
                    if (start_i >= end_i) continue;

                    dram_workers.emplace_back([this, start_i, end_i, &candidates]() {
                        for (int i = start_i; i < end_i; i++) {
                            int l = candidates[i].layer_id;
                            int e = candidates[i].expert_id;
                            int expert_global_idx = l * n_experts_ + e;
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
            } else {
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
//  Model Engine
// ════════════════════════════════════════════════════════════════════════════════

class MoecherEngine {
public:
    ModelConfig cfg_;
    BPETokenizer tokenizer_;
    std::string model_dir_;
    
    // Prompt-Lookup Drafting (PLD) Speculative Decoding
    bool enable_pld_ = true;
    int pld_draft_tokens_ = 4;
    
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
        GPUTensor ssm_state;      // [48, 128, 128] BF16 (50% memory cut)
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
    GPUTensor buf_track_flag_;   // [1] int32_t on GPU (1=track, 0=skip)
    GPUTensor buf_argmax_out_;   // [1] int32_t on GPU for 4-byte sampling
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    bool graph_captured_ = false;

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

    ~MoecherEngine() {
        if (step_topk_host_) {
            cudaFreeHost(step_topk_host_);
            step_topk_host_ = nullptr;
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

    bool load(const std::string& manifest_path, float max_vram_gb = 0.0f, float dram_cache_gb = 0.0f,
              const std::string& expert_dtype_override = "", bool buffered_io = false) {
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
        model_dir_ = base_dir.string();

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

        // ── Hardware Persistent L2 Cache Configuration ────────────────────────────
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            size_t max_persisting_l2 = prop.persistingL2CacheMaxSize;
            if (max_persisting_l2 > 0) {
                size_t l2_limit = (max_persisting_l2 * 3) / 4; // Use 75% for persistent window
                cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2_limit);
                LOG_INFO("Hardware: %s (Compute %d.%d)", prop.name, prop.major, prop.minor);
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
            expert_loader_.preload_all(16, expert_freq_counts_);

            expert_token_counts_.resize(cfg_.num_hidden_layers);
            for (int l = 0; l < cfg_.num_hidden_layers; l++) {
                expert_token_counts_[l].resize(cfg_.n_routed_experts);
            }
            std::string json_profile_path = (base_dir / "expert_profile.json").string();
            load_expert_profile_json(json_profile_path);

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

        apply_l2_cache_persistence();
        init_cuda_graph();

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
        if (cfg_.architecture != ModelArch::QWEN && !expert_loader_.all_resident(cfg_.num_hidden_layers)) {
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
    }

    // ── Forward pass for a single token (decode mode) ───────────────────────
    // Returns logits on GPU [vocab_size] in F32

    // Persistent RNG for sampling (seeded once)
    std::mt19937 rng_{std::random_device{}()};

    void forward_token(int token_id, int position) {
        int track_flag = (track_expert_freq_ && track_current_token_) ? 1 : 0;
        if (graph_captured_) {
            CUDA_CHECK(cudaMemcpyAsync(buf_track_flag_.i32(), &track_flag, sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
            CUDA_CHECK(cudaMemcpyAsync(buf_input_token_.i32(), &token_id, sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
            CUDA_CHECK(cudaMemcpyAsync(buf_input_pos_.i32(), &position, sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
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

    void forward_token_eager(int token_id, int position) {
        int track_flag = (track_expert_freq_ && track_current_token_) ? 1 : 0;
        CUDA_CHECK(cudaMemcpy(buf_track_flag_.i32(), &track_flag, sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(buf_input_token_.i32(), &token_id, sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(buf_input_pos_.i32(), &position, sizeof(int32_t), cudaMemcpyHostToDevice));
        forward_token_device_body(token_id, position);
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

        if (track_expert_freq_ && track_current_token_ && !is_control_token(token_id)) {
            expert_freq_total_tokens_++;
        }
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

        // Disable tracking during prompt prefill (prevents system prompt boilerplate from skewing stats)
        track_current_token_ = false;

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

        decode_step_idx_ = 0;
        int n_layers = cfg_.num_hidden_layers;
        int moe_top_k = cfg_.num_experts_per_tok;
        int n_experts = cfg_.n_routed_experts;
        std::vector<std::pair<int, std::vector<int32_t>>> per_token_topk;
        int prev_tracked_token = -1;
        int prev_slot = -1;

        std::string last_think_token_str;

        auto emit_token = [&](int next_token, int step_t) -> bool {
            // Handle think block filtering
            if (think_start_id >= 0 && next_token == think_start_id) {
                if (!think_block_ended) {
                    in_think_block = true;
                    think_block_ended = false;
                }
            }
            if (think_end_id >= 0 && next_token == think_end_id) {
                in_think_block = false;
                think_block_ended = true;
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

            // Pipeline: Launch GPU forward pass for next token immediately so GPU runs concurrently with CPU text decoding
            forward_token(next_token, position);
            position++;
            decode_step_idx_++;

            if (next_token == think_start_id || next_token == think_end_id) {
                return true;
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
                if (step_t > cfg_.bos_token_id) {
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
                    track_current_token_ = false;
                    if (think_end_id >= 0) {
                        forward_token(think_end_id, position);
                        position++;
                        output_ids.push_back(think_end_id);
                        history.push_back(think_end_id);
                    }

                    // 2. Inject and stream "Allright, here is the solution:\n\n" as the beginning of CONTENT
                    std::vector<int> transition_tokens = tokenizer_.encode("Allright, here is the solution:\n\n");
                    for (int tok_id : transition_tokens) {
                        track_current_token_ = false; // Synthetic transition tokens not tracked
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
                track_current_token_ = false;
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

            if (!emit_token(next_token, t)) {
                finish_reason = "stop";
                break;
            }

            // 2. Prompt-Lookup Speculative Verification Loop (PLD)
            if (enable_pld_ && content_tokens_generated < max_tokens) {
                std::vector<int> candidates = PromptLookupDrafter::draft(history, pld_draft_tokens_, 3, 2);
                for (int cand : candidates) {
                    if (content_tokens_generated >= max_tokens) break;
                    if (g_stop_requested.load()) break;

                    // Sample prediction at current position from the forwarded logits
                    int pred = sample_token(temperature, history, content_tokens_generated, in_think_block,
                                            top_k, top_p, min_p);

                    if (pred == cand && pred != cfg_.eos_token_id && (eos2_id < 0 || pred != eos2_id) &&
                        (enable_thinking || (pred != think_start_id && pred != think_end_id))) {
                        // MATCH! Speculative candidate verified and accepted
                        if (!emit_token(pred, t)) {
                            finish_reason = "stop";
                            break;
                        }
                    } else {
                        // Mismatch or stop token: Draft sequence ends.
                        // Emit base model's genuine prediction
                        if (pred == cfg_.eos_token_id || (eos2_id >= 0 && pred == eos2_id)) {
                            finish_reason = "stop";
                            break;
                        }
                        if (!enable_thinking && (pred == think_start_id || pred == think_end_id)) {
                            finish_reason = "stop";
                            break;
                        }
                        emit_token(pred, t);
                        break;
                    }
                }
                if (finish_reason == "stop") break;
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
                                }
                            }
                        }
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
                    // SSM state: [48, 128, 128] BF16 = 1.5 MB (50% memory bandwidth cut)
                    lw.ssm_state.alloc(48 * 128 * 128 * sizeof(__nv_bfloat16));
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

    // ── KV Compressor Forward (Device-Driven for CUDA Graph) ────────────────
    // Implements gated pooling compression for CSA (ratio=4, overlap) and
    // HCA (ratio=128, non-overlap) layers following DeepSeek V4 architecture.
    //
    // All state updates, pooling, RMSNorm, RoPE, and counter increments are
    // executed inside GPU kernels, enabling full CUDA Graph capture and replay.

    void forward_compressor(int layer_id) {
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
                       buf_hidden_.bf16(), proj_dim, dim, main_stream_);
        gemv_bf16_cuda(buf_comp_out_.f32(), lw.comp_wgate.bf16(),
                       buf_hidden_.bf16(), proj_dim, dim, main_stream_);

        // 2. Indexer Compressor Projections: [256] = indexer_comp_wkv @ hidden, indexer_comp_wgate @ hidden
        const float* idx_proj_kv = nullptr;
        const float* idx_proj_gate = nullptr;
        if (ratio == 4 && lw.indexer_comp_wkv.data) {
            int idx_proj_dim = 256;
            gemv_bf16_cuda(buf_indexer_proj_kv_.f32(), lw.indexer_comp_wkv.bf16(),
                           buf_hidden_.bf16(), idx_proj_dim, dim, main_stream_);
            gemv_bf16_cuda(buf_indexer_proj_gate_.f32(), lw.indexer_comp_wgate.bf16(),
                           buf_hidden_.bf16(), idx_proj_dim, dim, main_stream_);
            idx_proj_kv = buf_indexer_proj_kv_.f32();
            idx_proj_gate = buf_indexer_proj_gate_.f32();
        }

        // 3. Launch 100% device-driven compressor step kernel
        compressor_device_step_cuda(
            buf_input_pos_.i32(),
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
            main_stream_);
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
                gemv_bf16_out_bf16_cuda(out.bf16(), weight.bf16(), in_vec.bf16(), N, K, main_stream_);
            }
        };

        // 1. Attention Pre-RMSNorm: buf_hidden_ -> buf_hidden2_
        rms_norm_one_centered_cuda(buf_hidden2_.bf16(), buf_hidden_.bf16(),
                                   lw.attn_norm_w.bf16(), dim, cfg_.rms_norm_eps, main_stream_);

        if (lw.is_linear_attn) {
            // Qwen 3.8 Gated DeltaNet Linear Attention Projections
            matmul_proj(buf_q_, buf_hidden2_, lw.w_in_qkv, lw.w_in_qkv_scale, 10240, dim);
            matmul_proj(buf_up_, buf_hidden2_, lw.w_in_z, lw.w_in_z_scale, 6144, dim);
            gemv_bf16_out_bf16_cuda(buf_linear_a_.bf16(), lw.w_in_a.bf16(), buf_hidden2_.bf16(), 48, dim, main_stream_);
            gemv_bf16_out_bf16_cuda(buf_linear_b_.bf16(), lw.w_in_b.bf16(), buf_hidden2_.bf16(), 48, dim, main_stream_);

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
                lw.ssm_state.bf16(),
                16, 48, 128, main_stream_);

            // 4. Output Projection: buf_attn_out_ (6144) -> buf_hidden2_ (5120)
            matmul_proj(buf_hidden2_, buf_attn_out_, lw.linear_out_proj, lw.linear_out_proj_scale, dim, 6144);
        } else {
            // Standard Full GQA Attention (Gated) Projections
            matmul_proj(buf_q_, buf_hidden2_, lw.w_q, lw.w_q_scale, 2 * n_q_heads * head_dim, dim);
            matmul_proj(buf_gate_, buf_hidden2_, lw.w_k, lw.w_k_scale, n_kv_heads * head_dim, dim);
            matmul_proj(buf_up_, buf_hidden2_, lw.w_v, lw.w_v_scale, n_kv_heads * head_dim, dim);

            // 3 & 4. QK Norm + RoPE + GQA FP8 Decode + Sigmoid Gate
            qwen_gqa_decode_gated_fp8_cuda(
                buf_attn_out_.bf16(),
                buf_q_.bf16(),
                buf_gate_.bf16(),
                buf_up_.bf16(),
                lw.gqa_q_norm_w.bf16(),
                lw.gqa_k_norm_w.bf16(),
                lw.k_cache_gqa.u8(),
                lw.v_cache_gqa.u8(),
                n_q_heads, n_kv_heads, head_dim,
                buf_input_pos_.i32(), position, cfg_.max_seq_len,
                cfg_.rope_theta, cfg_.rms_norm_eps, main_stream_);

            // 5. Output Projection: buf_attn_out_ -> buf_hidden2_
            matmul_proj(buf_hidden2_, buf_attn_out_, lw.w_o, lw.w_o_scale, dim, n_q_heads * head_dim);
        }

        // 6. Residual connection: buf_hidden_ += buf_hidden2_
        vector_add_bf16_cuda(buf_hidden_.bf16(), buf_hidden2_.bf16(), dim, main_stream_);

        // 7. FFN Pre-RMSNorm: buf_hidden_ -> buf_hidden2_
        rms_norm_one_centered_cuda(buf_hidden2_.bf16(), buf_hidden_.bf16(),
                                   lw.ffn_norm_w.bf16(), dim, cfg_.rms_norm_eps, main_stream_);

        // 8 & 9. Gate & Up projections + Fused SiLU(Gate) * Up
        if (lw.w_gate.dtype == "int4") {
            gemv_int4_swiglu_fused_cuda(buf_gate_.bf16(), buf_hidden2_.bf16(),
                                        (const uint8_t*)lw.w_gate.data, lw.w_gate_scale.bf16(),
                                        (const uint8_t*)lw.w_up.data, lw.w_up_scale.bf16(),
                                        inter_size, dim, cfg_.swiglu_limit, main_stream_);
        } else {
            matmul_proj(buf_gate_, buf_hidden2_, lw.w_gate, lw.w_gate_scale, inter_size, dim);
            matmul_proj(buf_up_, buf_hidden2_, lw.w_up, lw.w_up_scale, inter_size, dim);
            silu_mul_cuda(buf_gate_.bf16(), buf_gate_.bf16(), buf_up_.bf16(), inter_size, cfg_.swiglu_limit, main_stream_);
        }

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
            gemm_fp8_dequant(buf_indexer_q_.bf16(), 1, idx_q_dim, q_lora,
                             buf_lora_.bf16(),
                             lw.indexer_wq_b_w.u8(), lw.indexer_wq_b_s.u8(), 128, main_stream_);

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

        if (head_weight_.dtype == "int4") {
            gemv_int4_f32_cuda(buf_logits_.f32(), buf_hidden_.bf16(),
                               (const uint8_t*)head_weight_.data, head_weight_scale_.bf16(),
                               vocab, dim, main_stream_);
        } else {
            // Fast CUDA graph compatible matrix-vector multiplication directly to float32 logits
            gemv_bf16_cuda(buf_logits_.f32(), head_weight_.bf16(), buf_hidden_.bf16(), vocab, dim, main_stream_);
        }
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
        sample_multinomial_f32_cuda(buf_argmax_out_.i32(), buf_logits_.f32(), vocab, temperature, r, min_p, main_stream_);
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

    // Web UI endpoints: serve from disk if found in ./web, otherwise serve embedded compiled-in assets
    auto serve_asset = [](const std::string& disk_path, std::string_view embedded, const char* mime, httplib::Response& res) {
        std::ifstream in(disk_path, std::ios::binary);
        if (in.is_open()) {
            std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            res.set_content(content, mime);
        } else {
            res.set_content(std::string(embedded), mime);
        }
        res.set_header("Cache-Control", "no-cache, no-store, must-revalidate");
        res.set_header("Pragma", "no-cache");
        res.set_header("Expires", "0");
    };

    svr.Get("/", [&serve_asset](const httplib::Request&, httplib::Response& res) {
        serve_asset("web/index.html", moecher::embedded_web::INDEX_HTML(), "text/html; charset=utf-8", res);
    });
    svr.Get("/index.html", [&serve_asset](const httplib::Request&, httplib::Response& res) {
        serve_asset("web/index.html", moecher::embedded_web::INDEX_HTML(), "text/html; charset=utf-8", res);
    });
    svr.Get("/style.css", [&serve_asset](const httplib::Request&, httplib::Response& res) {
        serve_asset("web/style.css", moecher::embedded_web::STYLE_CSS(), "text/css; charset=utf-8", res);
    });
    svr.Get("/script.js", [&serve_asset](const httplib::Request&, httplib::Response& res) {
        serve_asset("web/script.js", moecher::embedded_web::SCRIPT_JS(), "application/javascript; charset=utf-8", res);
    });

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

    // Expert Specialization & Profile endpoints
    svr.Get("/v1/experts/profile", [&engine](const httplib::Request&, httplib::Response& res) {
        json profile = engine.generate_expert_profile_json();
        res.set_content(profile.dump(), "application/json");
    });
    svr.Get("/v1/experts/stats", [&engine](const httplib::Request&, httplib::Response& res) {
        json profile = engine.generate_expert_profile_json();
        res.set_content(profile.dump(), "application/json");
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

                        if (g_track_experts && !engine.expert_freq_counts_.empty()) {
                            std::string fpath = engine.model_dir_.empty() ? "expert_freq.bin" : (engine.model_dir_ + "/expert_freq.bin");
                            engine.save_expert_freq(fpath);
                            engine.save_expert_profile(engine.model_dir_);
                        }
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

            if (g_track_experts && !engine.expert_freq_counts_.empty()) {
                std::string fpath = engine.model_dir_.empty() ? "expert_freq.bin" : (engine.model_dir_ + "/expert_freq.bin");
                engine.save_expert_freq(fpath);
                engine.save_expert_profile(engine.model_dir_);
            }
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
    // Low-latency active spin-wait on CUDA stream completions (eliminates OS sleep latency)
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);

#if defined(_WIN32) || defined(_WIN64)
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    timeBeginPeriod(1); // 1ms high-resolution timer period on Windows
#endif
    std::string manifest_path = "moecher_manifest.json";
    int port = 8001;
    float max_vram_gb = 0.0f;
    float dram_cache_gb = 0.0f;
    bool buffered_io = false;
    std::string log_path = "moecher.log";
    std::string expert_dtype_override = "";
    int default_thinking_budget = 4096;
    std::string imatrix_dataset = "";
    std::string imatrix_out = "";
    int imatrix_max_tokens = -1;
    bool enable_pld = true;
    int pld_draft_tokens = 4;

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
        } else if (std::string(argv[i]) == "--no-pld" || std::string(argv[i]) == "--disable-pld") {
            enable_pld = false;
        } else if ((std::string(argv[i]) == "--pld-tokens" || std::string(argv[i]) == "--pld-draft-tokens") && i + 1 < argc) {
            pld_draft_tokens = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--log-experts") {
            g_log_experts = true;
        } else if (std::string(argv[i]) == "--no-log-tokens") {
            g_log_tokens = false;
        } else if (std::string(argv[i]) == "--quiet" || std::string(argv[i]) == "-q") {
            g_quiet = true;
            g_log_tokens = false;
        } else if (std::string(argv[i]) == "--buffered-io") {
            buffered_io = true;
        } else if (std::string(argv[i]) == "--track" ||
                   std::string(argv[i]) == "--track-all-moe" ||
                   std::string(argv[i]) == "-track-all-moe" ||
                   std::string(argv[i]) == "--track-moe-all" ||
                   std::string(argv[i]) == "-track-moe-all" ||
                   std::string(argv[i]) == "--track-moe" ||
                   std::string(argv[i]) == "-track-moe" ||
                   std::string(argv[i]) == "-track") {
            g_track_experts = true;
        } else if (std::string(argv[i]) == "--track-reset" ||
                   std::string(argv[i]) == "--reset-track" ||
                   std::string(argv[i]) == "-track-reset" ||
                   std::string(argv[i]) == "-reset-track") {
            g_track_reset = true;
            g_track_experts = true;
        }
    }

    // Open log file
    g_log_file.open(log_path, std::ios::app);
    LOG_INFO("=== moecher starting ===");
    LOG_INFO("=== v2.05 ===");
    LOG_INFO("Default thinking token budget: %d", default_thinking_budget);

    MoecherEngine engine;
    engine.enable_pld_ = enable_pld;
    engine.pld_draft_tokens_ = pld_draft_tokens;
    LOG_INFO("Prompt-Lookup Drafting (PLD): %s (draft_tokens=%d)", enable_pld ? "enabled" : "disabled", pld_draft_tokens);

    if (!engine.load(manifest_path, max_vram_gb, dram_cache_gb, expert_dtype_override, buffered_io)) {
        LOG_ERROR("Failed to load model");
        return 1;
    }

    if (!imatrix_dataset.empty() && !imatrix_out.empty()) {
        bool ok = engine.collect_imatrix(imatrix_dataset, imatrix_out, imatrix_max_tokens);
        return ok ? 0 : 1;
    }

    // Enable expert frequency tracking if --track flag is set
    std::string freq_path = (std::filesystem::path(manifest_path).parent_path() / "expert_freq.bin").string();
    if (g_track_experts) {
        if (g_track_reset) {
            LOG_INFO("Resetting expert frequency & specialization tracking profile...");
            engine.enable_expert_tracking(""); // Start with empty profile
        } else {
            engine.enable_expert_tracking(freq_path);
        }
    }

    run_server(engine, port, default_thinking_budget);

    // Save expert frequency & specialization profile on shutdown
    if (g_track_experts && !engine.expert_freq_counts_.empty()) {
        engine.save_expert_freq(freq_path);
        std::string model_dir = std::filesystem::path(manifest_path).parent_path().string();
        engine.save_expert_profile(model_dir);
    }
    return 0;
}
