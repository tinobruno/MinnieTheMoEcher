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


static void log_msg(const char* level, const char* fmt, ...) {
    if (g_quiet && g_server_ready && strcmp(level, "INFO") == 0) {
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
    int expert_fd_ = -1;

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

    bool init(const std::string& expert_bin_path, int block_size,
              int n_layers, int n_experts, size_t cache_budget_bytes, size_t dram_budget_bytes) {
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
            CUDA_CHECK(cudaMallocHost(&dram_cache_pool_, (size_t)dram_cache_capacity_ * block_size));
            dram_cache_slots_.resize(dram_cache_capacity_);
            for (int i = 0; i < dram_cache_capacity_; i++) {
                dram_cache_slots_[i].slot_index = i;
                dram_cache_slots_[i].gpu_data = (char*)dram_cache_pool_ + (size_t)i * block_size;
            }
        }

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

        flat_vram_ptrs_.assign((size_t)n_layers * n_experts, nullptr);
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
        int64_t key = (int64_t)layer_id * n_experts_ + expert_id;
        
        std::unique_lock<std::mutex> lock(cache_mutex_);
        access_counter_++;

        // Check L1 (VRAM)
        auto it = key_to_slot_.find(key);
        if (it != key_to_slot_.end()) {
            cache_slots_[it->second].last_used = access_counter_;
            void* ptr = cache_slots_[it->second].gpu_data;
            if (key >= 0 && key < (int64_t)flat_vram_ptrs_.size()) {
                flat_vram_ptrs_[key] = ptr;
            }
            // Also update L2 last_used if it exists in L2, so it doesn't get evicted randomly
            auto it2 = dram_key_to_slot_.find(key);
            if (it2 != dram_key_to_slot_.end()) {
                dram_cache_slots_[it2->second].last_used = access_counter_;
            }
            lock.unlock();
            return ptr;
        }

        // Need an L1 slot
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
            int64_t old_key = (int64_t)slot.layer_id * n_experts_ + slot.expert_id;
            key_to_slot_.erase(old_key);
            if (old_key >= 0 && old_key < (int64_t)flat_vram_ptrs_.size()) {
                flat_vram_ptrs_[old_key] = nullptr;
            }
        }
        
        slot.layer_id = layer_id;
        slot.expert_id = expert_id;
        slot.last_used = access_counter_;
        key_to_slot_[key] = evict_slot;
        if (key >= 0 && key < (int64_t)flat_vram_ptrs_.size()) {
            flat_vram_ptrs_[key] = slot.gpu_data;
        }

        // Check L2 (DRAM)
        void* host_src_ptr = nullptr;
        int stage_idx = -1;

        if (dram_cache_capacity_ > 0) {
            auto it2 = dram_key_to_slot_.find(key);
            if (it2 != dram_key_to_slot_.end()) {
                // L2 Hit
                dram_cache_slots_[it2->second].last_used = access_counter_;
                host_src_ptr = dram_cache_slots_[it2->second].gpu_data;
                
                if (g_log_experts) {
                    LOG_INFO("[ExpertCache] L2 Hit: L%d E%d -> L1 slot %d", layer_id, expert_id, evict_slot);
                }
            } else {
                // L2 Miss - we must load into L2 first, then L1
                int evict_dram = -1;
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
                
                auto& d_slot = dram_cache_slots_[evict_dram];
                if (d_slot.layer_id >= 0) {
                    int64_t old_key = (int64_t)d_slot.layer_id * n_experts_ + d_slot.expert_id;
                    dram_key_to_slot_.erase(old_key);
                }
                
                d_slot.layer_id = layer_id;
                d_slot.expert_id = expert_id;
                d_slot.last_used = access_counter_;
                dram_key_to_slot_[key] = evict_dram;
                
                host_src_ptr = d_slot.gpu_data;
                
                if (g_log_experts) {
                    LOG_INFO("[ExpertCache] L2 Miss (SSD read): L%d E%d -> L2 slot %d -> L1 slot %d", 
                             layer_id, expert_id, evict_dram, evict_slot);
                }
                
                // Read from disk into L2 directly
                int64_t file_offset = (int64_t)key * expert_block_size_;
                ssize_t bytes_read = pread(expert_fd_, host_src_ptr, expert_block_size_, file_offset);
                if (bytes_read != expert_block_size_) {
                    LOG_ERROR("Expert read failed: layer=%d expert=%d offset=%ld got=%ld",
                              layer_id, expert_id, file_offset, bytes_read);
                    lock.unlock();
                    return nullptr;
                }
            }
        } else {
            // No L2 cache, fallback to staging buffer
            stage_idx = staging_idx_;
            staging_idx_ = (staging_idx_ + 1) % NUM_STAGING_BUFFERS;
            
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
            ssize_t bytes_read = pread(expert_fd_, stage.ptr, expert_block_size_, file_offset);
            if (bytes_read != expert_block_size_) {
                LOG_ERROR("Expert read failed: layer=%d expert=%d offset=%ld got=%ld",
                          layer_id, expert_id, file_offset, bytes_read);
                return nullptr;
            }
            host_src_ptr = stage.ptr;
            CUDA_CHECK(cudaMemcpyAsync(slot.gpu_data, host_src_ptr, expert_block_size_,
                                        cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaEventRecord(stage.event, stream));
        } else {
            // Memory in host_src_ptr (L2) is already populated, async copy to L1
            // Since L2 is just pinned host memory, we can copy from it directly safely.
            CUDA_CHECK(cudaMemcpyAsync(slot.gpu_data, host_src_ptr, expert_block_size_,
                                        cudaMemcpyHostToDevice, stream));
        }

        return slot.gpu_data;
    }

    bool preload_all(int n_threads = 16) {
        int total = n_layers_ * n_experts_;
        int to_load = std::min(total, cache_capacity_);
        LOG_INFO("Preloading %d/%d experts into VRAM...", to_load, total);
        
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

                    ssize_t bytes_read = pread(expert_fd_, stage_ptr, expert_block_size_, file_offset);
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
                        LOG_INFO("  Preloaded %d/%d experts...", done, to_load);
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
            if (dram_to_load > 0) {
                LOG_INFO("Preloading %d/%d experts into DRAM L2 cache...", dram_to_load, dram_cache_capacity_);
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
                            ssize_t bytes_read = pread(expert_fd_, dst, expert_block_size_, file_offset);
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

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        LOG_INFO("All %d experts preloaded in %.2f seconds (%.2f GB/s)",
                 to_load, elapsed / 1000.0,
                 ((double)to_load * expert_block_size_) / (elapsed / 1000.0 * 1024.0 * 1024.0 * 1024.0));

        // Copy flat pointer table to GPU for GPU-native kernel execution
        flat_vram_ptrs_gpu_.alloc(flat_vram_ptrs_.size() * sizeof(void*));
        CUDA_CHECK(cudaMemcpy(flat_vram_ptrs_gpu_.data, flat_vram_ptrs_.data(),
                               flat_vram_ptrs_.size() * sizeof(void*), cudaMemcpyHostToDevice));
        return true;
    }

    const void* const* flat_vram_ptrs_gpu() const {
        return (const void* const*)flat_vram_ptrs_gpu_.data;
    }

    void cleanup() {
        if (expert_fd_ >= 0) close(expert_fd_);
        if (cache_pool_gpu_) cudaFree(cache_pool_gpu_);
        if (dram_cache_pool_) cudaFreeHost(dram_cache_pool_);
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

    cudaStream_t main_stream_;
    cudaStream_t side_stream_;
    cudaEvent_t side_event_;
    cudaEvent_t main_event_;
    cudaStream_t expert_streams_[32];
    cudaEvent_t expert_events_[32];

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
    // Pre-allocated host buffers (zero runtime heap allocations)
    std::vector<float> router_probs_host_;
    std::vector<float> router_selection_host_;
    std::vector<int> router_indices_host_;
    std::vector<float> logits_host_;
    std::vector<float> probs_host_;
    __nv_bfloat16* logits_bf16_host_ = nullptr;

    ~MoecherEngine() {
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
        if (side_stream_) {
            cudaStreamDestroy(side_stream_);
            side_stream_ = nullptr;
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

        // Load tokenizer
        std::string tok_path = manifest["tokenizer"]["tokenizer_json"].get<std::string>();
        if (!tokenizer_.load(tok_path)) return false;

        // Init CUDA
        CUDA_CHECK(cudaStreamCreate(&main_stream_));
        CUDA_CHECK(cudaStreamCreate(&side_stream_));
        CUDA_CHECK(cudaEventCreateWithFlags(&side_event_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&main_event_, cudaEventDisableTiming));
        for (int i = 0; i < 32; i++) {
            CUDA_CHECK(cudaStreamCreate(&expert_streams_[i]));
            CUDA_CHECK(cudaEventCreate(&expert_events_[i]));
        }
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
        CUBLAS_CHECK(cublasSetStream(cublas_handle_, main_stream_));
        cublasSetMathMode(cublas_handle_, CUBLAS_DEFAULT_MATH);

        expert_pool_ = std::make_unique<ThreadPool>(16);

        // Determine available VRAM for expert cache
        size_t vram_free, vram_total;
        CUDA_CHECK(cudaMemGetInfo(&vram_free, &vram_total));
        LOG_INFO("VRAM: %.1f GB free / %.1f GB total",
                 vram_free / (1024.0 * 1024.0 * 1024.0), vram_total / (1024.0 * 1024.0 * 1024.0));

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

        // Allocate working buffers first
        alloc_buffers();

        // Reserve VRAM: rest goes to expert cache
        CUDA_CHECK(cudaMemGetInfo(&vram_free, &vram_total));
        size_t total_experts_bytes = (size_t)expert_n_layers * expert_n_experts * expert_block_size;
        size_t cache_budget = vram_free > (1ULL * 1024 * 1024 * 1024) ? (vram_free - 1ULL * 1024 * 1024 * 1024) : vram_free;
        
        if (max_vram_gb > 0.0f) {
            size_t max_vram_bytes = (size_t)(max_vram_gb * 1024.0 * 1024.0 * 1024.0);
            size_t used_vram = vram_total - vram_free;
            if (max_vram_bytes > used_vram + 1ULL * 1024 * 1024 * 1024) {
                size_t user_budget = max_vram_bytes - used_vram - 1ULL * 1024 * 1024 * 1024;
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
        if (!expert_loader_.init(expert_path, expert_block_size,
                                  expert_n_layers, expert_n_experts,
                                  cache_budget, dram_cache_budget)) return false;

        // Preload experts into VRAM
        expert_loader_.preload_all();

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
        // 1 & 2. Embedding lookup and broadcast to HC copies: [1, dim] -> [hc, dim]
        embedding_broadcast_cuda(buf_hidden_.bf16(), buf_hc_state_.bf16(),
                                 embed_weight_.bf16(), token_id, dim, hc, main_stream_);

        if (dbg_first_token_) {
            LOG_INFO("DEBUG: token_id=%d position=%d", token_id, position);
            dump_bf16("embed", buf_hidden_.bf16(), dim);
        }

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

    std::string generate(const std::vector<int>& prompt, int max_tokens = 512,
                         float temperature = 0.0f,
                         std::function<void(const std::string&,bool)> on_token = nullptr,
                         float repetition_penalty = 1.0f,
                         bool enable_thinking = true) {
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
        std::vector<int> input_ids = prompt;//tokenizer_.encode(prompt);
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
        bool in_think_block = enable_thinking;
        bool think_block_ended = false;
        std::string think_detect_buffer;
        // Get token IDs for think tags
        int think_start_id = tokenizer_.get_token_id("<think>");
        int think_end_id = tokenizer_.get_token_id("</think>");
        // Also check with special token format
        //if (think_start_id < 0) think_start_id = tokenizer_.get_token_id("\xef\xbd\x9c" "think" "\xef\xbd\x9c");
        //if (think_end_id < 0) think_end_id = tokenizer_.get_token_id("\xef\xbd\x9c" "/think" "\xef\xbd\x9c");


        // Additional EOS tokens for DeepSeek V4
        int eos2_id = tokenizer_.get_token_id(
            "<\xef\xbd\x9c" "end\xe2\x96\x81of\xe2\x96\x81sentence" "\xef\xbd\x9c>");

        LOG_WARN("Think tokens: start=%d end=%d, EOS=%d eos2=%d",
                 think_start_id, think_end_id, cfg_.eos_token_id, eos2_id);

        // Track how many content tokens (non-think) we've generated
        int content_tokens_generated = 0;

        //force and inject think start
      //  in_think_block = true ;

       // forward_token(think_start_id, position);
      //  position++;    

        for (int t = 0; content_tokens_generated < max_tokens; t++) {
            // Sample from logits, suppressing EOS for the first few tokens
            int next_token = sample_token(temperature, history, content_tokens_generated,in_think_block);
            // Check all EOS conditions
            if (next_token == cfg_.eos_token_id ||
                (eos2_id >= 0 && next_token == eos2_id)) {
                LOG_WARN("EOS hit: token=%d (cfg_eos=%d, eos2=%d) at step %d/%d",
                         next_token, cfg_.eos_token_id, eos2_id, t, max_tokens);
                if (!in_think_block) break;
                forward_token(think_end_id, position);
                position++;
                in_think_block = false;
                continue;
            }

            output_ids.push_back(next_token);
            history.push_back(next_token);
            if (!in_think_block) content_tokens_generated++;

           
            // Handle think block filtering
            if (think_start_id >= 0 && next_token == think_start_id) {
                in_think_block = true;
                think_block_ended = false;
                // Forward the token but don't emit it
                LOG_WARN("think_start_id hit");
                forward_token(next_token, position);
                position++;
                continue;
            }
            if (think_end_id >= 0 && next_token == think_end_id) {
                in_think_block = false;
                think_block_ended = true;
                LOG_WARN("think_end_id hit");   

                // Forward the token but don't emit it
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
                if (t > cfg_.bos_token_id) {
                    if (!g_quiet) {
                        printf("%s", token_buffer.c_str());
                        fflush(stdout);
                    }
                }
                if (on_token) {
                    on_token(token_buffer, in_think_block);
                }
                token_buffer.clear();
            }

            // Forward the new token
            forward_token(next_token, position);
            position++;

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
        int top_k = cfg_.num_experts_per_tok;

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
        buf_gate_.alloc((size_t)(top_k + 1) * moe_inter * sizeof(__nv_bfloat16));
        buf_up_.alloc((size_t)(top_k + 1) * moe_inter * sizeof(__nv_bfloat16));
        buf_down_.alloc((size_t)(top_k + 1) * dim * sizeof(__nv_bfloat16));
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
        buf_active_expert_ptrs_.alloc(32 * sizeof(void*));

        // Compressor working buffers
        // Max projection output size: coff=2 for ratio=4, head_dim=512 -> 1024
        int max_comp_dim = 2 * head_dim_val;  // coff=2
        buf_comp_proj_.alloc(max_comp_dim * sizeof(float));  // for wkv or wgate output
        buf_comp_out_.alloc(max_comp_dim * sizeof(float));   // for pooling output
        buf_comp_bf16_.alloc(head_dim_val * sizeof(__nv_bfloat16));  // compressed entry in BF16
        // Combined KV buffer: raw window + max compressed entries
        int max_combined = cfg_.sliding_window + cfg_.max_compressed_entries;
        buf_combined_kv_.alloc((size_t)max_combined * head_dim_val * sizeof(__nv_bfloat16));

        router_probs_host_.resize(cfg_.n_routed_experts);
        router_selection_host_.resize(cfg_.n_routed_experts);
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
        CUDA_CHECK(cudaEventRecord(main_event_, main_stream_));

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
        CUDA_CHECK(cudaEventRecord(main_event_, main_stream_));

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

        // Compute mixes from normalized hc_state directly in a single fused pass
        gemv_hc_pre_norm_cuda(buf_hc_mixes_.f32(), buf_hc_state_.bf16(), hc_fn.f32(),
                              mix_size, hc_dim, cfg_.hc_eps, main_stream_);

        // Split mixes into pre, post, comb via Sinkhorn
        hc_split_sinkhorn_cuda(
            buf_hc_pre_.f32(), buf_hc_post_.f32(), buf_hc_comb_.f32(),
            buf_hc_mixes_.f32(), hc_scale.f32(), hc_base.f32(),
            hc, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, main_stream_);

        // Compute weighted sum: y = sum(pre[i] * hc_state[i]) for i in 0..hc-1
        // Result in buf_hidden_
        hc_pre_weighted_add_cuda(buf_hidden_.bf16(), buf_hc_state_.bf16(),
                                 buf_hc_pre_.f32(), dim, hc, main_stream_);

        // Save residual (the full HC state before sublayer) — it's already in buf_hc_state_
        // buf_hidden2_ will hold the sublayer output after attention/FFN
    }

    // ── HC post: expand [dim] -> [hc, dim] ──────────────────────────────────

    void hc_post() {
        int dim = cfg_.hidden_size;
        int hc = cfg_.hc_mult;

        // Ping-pong: read from buf_hc_state_, write new updated state to buf_hc_residual_
        hc_post_update_cuda(buf_hc_residual_.bf16(), buf_hidden_.bf16(), buf_hc_state_.bf16(),
                            buf_hc_post_.f32(), buf_hc_comb_.f32(),
                            dim, hc, main_stream_);

        // Swap tensors (host pointer swap, zero GPU memory copy overhead)
        std::swap(buf_hc_state_, buf_hc_residual_);
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
        int cache_pos = position % window;
        __nv_bfloat16* kv_dst = lw.kv_cache.bf16() + (size_t)cache_pos * head_dim_val;

        // KV projection on side_stream_ (must wait for attn_norm to finish on main_stream_)
        CUDA_CHECK(cudaStreamWaitEvent(side_stream_, main_event_, 0));
        gemm_fp8_dequant(buf_kv_.bf16(), 1, head_dim_val, dim,
                         buf_hidden_.bf16(),
                         lw.wkv_w.u8(), lw.wkv_s.u8(), 128, side_stream_);
        rms_norm_cuda(kv_dst, buf_kv_.bf16(),
                      lw.kv_norm_w.bf16(), head_dim_val, cfg_.rms_norm_eps, side_stream_);
        rope_cuda(kv_dst, 1, head_dim_val, rope_dim,
                  position, layer_rope_freqs, false, side_stream_);
        CUDA_CHECK(cudaEventRecord(side_event_, side_stream_));

        // Q projection on main_stream_
        // q_raw = wq_a(x) -> [q_lora_rank]
        gemm_fp8_dequant(buf_lora_.bf16(), 1, q_lora, dim,
                         buf_hidden_.bf16(),
                         lw.wq_a_w.u8(), lw.wq_a_s.u8(), 128, main_stream_);
        if (dbg) dump_bf16("wq_a_out", buf_lora_.bf16(), q_lora);

        // q_normed = q_norm(q_raw)
        rms_norm_cuda(buf_lora_.bf16(), buf_lora_.bf16(),
                      lw.q_norm_w.bf16(), q_lora, cfg_.rms_norm_eps, main_stream_);
        if (dbg) dump_bf16("q_normed", buf_lora_.bf16(), q_lora);

        // q = wq_b(q_normed) -> [n_heads * head_dim]
        gemm_fp8_dequant(buf_q_.bf16(), 1, n_heads * head_dim_val, q_lora,
                         buf_lora_.bf16(),
                         lw.wq_b_w.u8(), lw.wq_b_s.u8(), 128, main_stream_);
        
        // Per-head Q normalization (DeepseekV4UnweightedRMSNorm)
        rms_norm_unweighted_batched_cuda(buf_q_.bf16(), buf_q_.bf16(),
                                         n_heads, head_dim_val, cfg_.rms_norm_eps,
                                         main_stream_);
        
        // Apply RoPE to last rope_dim elements of each Q head
        rope_cuda(buf_q_.bf16(), n_heads, head_dim_val, rope_dim,
                  position, layer_rope_freqs, false, main_stream_);

        // Synchronize main_stream_ with side_stream_ before compressor and attention
        CUDA_CHECK(cudaStreamWaitEvent(main_stream_, side_event_, 0));
        if (dbg) {
            dump_bf16("wkv_out", buf_kv_.bf16(), head_dim_val);
            dump_bf16("kv_after_norm_rope", kv_dst, head_dim_val);
        }

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
        
        gemv_fp8_grouped_cuda(buf_lora_.bf16(), buf_attn_out_.bf16(),
                              lw.wo_a_w.u8(), lw.wo_a_s.u8(),
                              o_lora, hpg_dim, o_groups, 128, main_stream_);
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
        if (layer_id < cfg_.n_hash_layers) {
            // Hash layers 0..2: lookup tid2eid table with constant weight (routed_scaling_factor / top_k)
            moe_route_hash_cuda(buf_topk_idx_.i32(), buf_topk_vals_.f32(),
                                lw.tid2eid.i64(), token_id, top_k,
                                cfg_.routed_scaling_factor, main_stream_);
        } else {
            // Score-based layers 3..42: gate_w -> fused (bias + SqrtSoftplus + top-6 routing)
            gemm_bf16(buf_scores_bf16_.bf16(), 1, n_experts, dim,
                      buf_hidden_.bf16(), lw.gate_w.bf16());

            moe_route_top6_from_bf16_cuda(buf_topk_idx_.i32(), buf_topk_vals_.f32(),
                                          buf_scores_bf16_.bf16(), lw.gate_bias.f32(),
                                          n_experts, top_k, cfg_.routed_scaling_factor, main_stream_);
        }

        // 4. Populate active expert pointers
        if (expert_loader_.all_resident(cfg_.num_hidden_layers)) {
            // 0-latency GPU-native fast path for full VRAM
            populate_active_expert_ptrs_cuda(
                (const void**)buf_active_expert_ptrs_.data,
                buf_topk_idx_.i32(),
                expert_loader_.flat_vram_ptrs_gpu(),
                layer_id, n_experts, top_k, main_stream_);
        } else {
            // Dynamic fetch path for offload / 24GB VRAM
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

            CUDA_CHECK(cudaMemcpy(buf_active_expert_ptrs_.data, active_expert_ptrs_host_,
                                  top_k * sizeof(void*), cudaMemcpyHostToDevice));
        }

        // 5. Launch 6 routed experts on main_stream_
        auto& w1_info = expert_parts_["w1.weight"];
        auto& w3_info = expert_parts_["w3.weight"];
        auto& w2_info = expert_parts_["w2.weight"];

        gemv_iq2_xxs_moe_swiglu_fused_cuda(
            buf_gate_.bf16(), buf_hidden_.bf16(),
            (const void* const*)buf_active_expert_ptrs_.data,
            w1_info.offset_in_block, w3_info.offset_in_block,
            moe_inter, dim, cfg_.swiglu_limit, main_stream_);

        gemv_q2_k_moe_cuda(
            buf_down_.bf16(), buf_gate_.bf16(),
            (const void* const*)buf_active_expert_ptrs_.data,
            w2_info.offset_in_block,
            dim, moe_inter, main_stream_);

        // 5. Shared expert executed concurrently on side_stream_ (must wait for ffn_norm to finish on main_stream_)
        CUDA_CHECK(cudaStreamWaitEvent(side_stream_, main_event_, 0));
        __nv_bfloat16* shared_gate = buf_gate_.bf16() + top_k * moe_inter;
        __nv_bfloat16* shared_up   = buf_up_.bf16()   + top_k * moe_inter;
        __nv_bfloat16* shared_down = buf_down_.bf16() + top_k * dim;

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

        cudaStream_t stream = expert_streams_[k];
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

        // logits = hidden @ head_weight.T -> [vocab]
        gemm_bf16(buf_dequant_.bf16(), 1, vocab, dim,
                  buf_hidden_.bf16(), head_weight_.bf16());
        bf16_to_f32_cuda(buf_logits_.f32(), buf_dequant_.bf16(), vocab, main_stream_);
    }

    // ── Sample from logits ──────────────────────────────────────────────────

    float current_rep_penalty_ = 1.0f;  // Set per-request by generate()

    int sample_token(float temperature, const std::vector<int>& history, int step = 0, bool is_reasoning = true) {
        int vocab = cfg_.vocab_size;
        float* logits = logits_host_.data();
        CUDA_CHECK(cudaMemcpyAsync(logits, buf_logits_.f32(),
                                   vocab * sizeof(float), cudaMemcpyDeviceToHost, main_stream_));
        CUDA_CHECK(cudaStreamSynchronize(main_stream_));

        // Suppress EOS for the first 2 tokens to avoid empty answers
        if (step < 2) {
            if (cfg_.eos_token_id >= 0 && cfg_.eos_token_id < vocab) {
                logits[cfg_.eos_token_id] = -1e9f;
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

        static constexpr int COOLDOWN_TOKENS = 5;
        float effective_temp = temperature;
        if (is_reasoning && step < COOLDOWN_TOKENS) {
            effective_temp = std::max(temperature, temperature * 0.5f);
        }

        int best_id = 0;
        float max_logit = logits[0];
        for (int i = 1; i < vocab; i++) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                best_id = i;
            }
        }

        if (temperature <= 0.0f) {
            return best_id;
        }

        // Exact Min-p filtering in log-space: prob >= min_p * max_prob <=> logit >= max_logit + temp * ln(min_p)
        float min_p = 0.1f;
        float logit_cutoff = max_logit + effective_temp * -2.302585093f;
        float inv_temp = 1.0f / effective_temp;

        // Collect candidates passing min-p (typically only 5 to 50 tokens out of 129k)
        std::vector<std::pair<float, int>> candidates;
        candidates.reserve(64);
        for (int i = 0; i < vocab; i++) {
            if (logits[i] >= logit_cutoff) {
                float prob = expf((logits[i] - max_logit) * inv_temp);
                candidates.push_back({prob, i});
            }
        }
        if (candidates.empty()) {
            return best_id;
        }

        int top_k = 40;
        int k_keep = std::min((int)candidates.size(), top_k);
        std::partial_sort(candidates.begin(), candidates.begin() + k_keep, candidates.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });

        float top_p = 0.95f;
        float total_prob = 0.0f;
        for (auto& c : candidates) total_prob += c.first;
        float norm_p = 1.0f / total_prob;

        std::vector<float> sample_weights;
        std::vector<int> sample_indices;
        sample_weights.reserve(k_keep);
        sample_indices.reserve(k_keep);

        float cumsum = 0.0f;
        for (int i = 0; i < k_keep; i++) {
            sample_weights.push_back(candidates[i].first);
            sample_indices.push_back(candidates[i].second);
            cumsum += candidates[i].first * norm_p;
            if (cumsum > top_p) break;
        }

        std::discrete_distribution<int> dist(sample_weights.begin(), sample_weights.end());
        return sample_indices[dist(rng_)];
    }
};

// ════════════════════════════════════════════════════════════════════════════════
//  Chat Template
// ════════════════════════════════════════════════════════════════════════════════

static std::vector<int> apply_chat_template(const json& messages, const BPETokenizer& tok, bool enable_thinking = true) {
    // DeepSeek V4 chat format (from official encoding_dsv4.py, thinking_mode="chat"):
    // <｜begin▁of▁sentence｜>{system_content}
    // <｜User｜>{user_content}<｜Assistant｜></think>{assistant_content}<｜end▁of▁sentence｜>
    // <｜User｜>{user_content}<｜Assistant｜></think>
    //
    // Key: </think> after <｜Assistant｜> signals "chat mode" — skip thinking, answer directly.
    // Without this token, the model enters an ambiguous state and produces premature EOS.


    static const int BOS = 0;
    static const int EOS = 1;
    static const int USER = 128803;
    static const int ASSISTANT = 128804;
    static const int THINK_BEGIN = 128821;
    static const int THINK_END = 128822;

    /*static const std::string BOS = "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence" "\xef\xbd\x9c>";
    static const std::string EOS = "<\xef\xbd\x9c" "end\xe2\x96\x81of\xe2\x96\x81sentence" "\xef\xbd\x9c>";
    static const std::string USER = "<\xef\xbd\x9c" "User" "\xef\xbd\x9c>";
    static const std::string ASSISTANT = "<\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c>";
    static const std::string THINK_BEGIN = "<think>";
    static const std::string THINK_END = "</think>";
*/

    std::vector<int> result;
    result.push_back(BOS);

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
            if (messages[i].contains("reasoning_content") && !messages[i]["reasoning_content"].is_null()) {
                string past_thoughts = messages[i]["reasoning_content"].get<std::string>();
                if (!past_thoughts.empty()) {
                    result.push_back(THINK_BEGIN);
                    auto enc = tok.encode(past_thoughts);
                    result.insert(result.end(), enc.begin(), enc.end());
                    result.push_back(THINK_END);
                }
            } else {
                 result.push_back(THINK_END);
            }
           
            auto enc = tok.encode(content);
            result.insert(result.end(), enc.begin(), enc.end());
            result.push_back(EOS);
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

static void run_server(MoecherEngine& engine, int port) {
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
            float repetition_penalty = request.value("repetition_penalty", 1.0f);

            bool enable_thinking = true;
            if (request.contains("thinking") && request["thinking"].contains("type")) {
                if (request["thinking"]["type"] == "disabled") {
                    enable_thinking = false;
                }
            }
            std::string reasoning_effort = request.value("reasoning_effort", "high");
            LOG_INFO("Reasoning effort: %s, Thinking: %s", reasoning_effort.c_str(), enable_thinking ? "enabled" : "disabled");

            // Apply chat template
           // std::string prompt = apply_chat_template(messages, engine.tokenizer_);
            std::vector<int> prompt = apply_chat_template(messages, engine.tokenizer_, enable_thinking);
            //engine.tokenizer_.encode(prompt);
            LOG_INFO("PROMPT (len=%zu)", prompt.size());

            std::string req_id = "chatcmpl-moecher-" + std::to_string(++g_request_counter);

            if (stream) {
                // SSE streaming
                res.set_chunked_content_provider(
                    "text/event-stream",
                    [&engine, prompt, max_tokens, temperature, req_id, repetition_penalty, enable_thinking](size_t offset, httplib::DataSink &sink) {
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
                        
                        engine.generate(prompt, max_tokens, temperature, [&](const std::string& text, bool is_reasoning) {
                            if (text.empty()) return; // Skip empty tokens
          
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
                            sink.write(sse_chunk.data(), sse_chunk.size());
                        }, repetition_penalty, enable_thinking);

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
    LOG_INFO("version 2.03");
    g_server_ready = true;
    svr.listen("0.0.0.0", port);
}

// ════════════════════════════════════════════════════════════════════════════════
//  Main
// ════════════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    std::string manifest_path = "moecher_manifest.json";
    int port = 8001;
    float max_vram_gb = 0.0f;
    float dram_cache_gb = 0.0f;
    std::string log_path = "moecher.log";
    std::string expert_dtype_override = "";

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
    LOG_INFO("═══ moecher starting ═══");
    LOG_INFO("═══ v2.02 ═══");

    MoecherEngine engine;
    if (!engine.load(manifest_path, max_vram_gb, dram_cache_gb, expert_dtype_override)) {
        LOG_ERROR("Failed to load model");
        return 1;
    }

    run_server(engine, port);
    return 0;
}
