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
#include "tool_exec.hpp"
#include "mcp_client.hpp"
#include "version.hpp"
#include "image_loader.hpp"
#include "vision_tower.hpp"
#include "web_ui.hpp"

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
#include <nlohmann/json.hpp>

#ifndef CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH
#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH (64 * 1024 * 1024)
#endif
#ifndef CPPHTTPLIB_PAYLOAD_MAX_LENGTH
#define CPPHTTPLIB_PAYLOAD_MAX_LENGTH (64 * 1024 * 1024)
#endif
#ifndef CPPHTTPLIB_HEADER_MAX_LENGTH
#define CPPHTTPLIB_HEADER_MAX_LENGTH (1024 * 1024)
#endif
#ifndef CPPHTTPLIB_REQUEST_URI_MAX_LENGTH
#define CPPHTTPLIB_REQUEST_URI_MAX_LENGTH (1024 * 1024)
#endif

#include <httplib.h>
#ifdef _WIN32
#include <wininet.h>
#endif

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
static constexpr const char* MOECHER_VERSION = moecher::VERSION;

// ════════════════════════════════════════════════════════════════════════════════
//  Logging & Agentic (statically linked from src/logging.cpp & src/agentic.cpp)
// ════════════════════════════════════════════════════════════════════════════════

#include "logging.hpp"
#include "agentic.hpp"


// ════════════════════════════════════════════════════════════════════════════════
//  Model Config (from manifest)
// ════════════════════════════════════════════════════════════════════════════════

enum class ModelArch { DEEPSEEK_V4, QWEN, QWEN2 };

struct ModelConfig {
    ModelArch architecture = ModelArch::DEEPSEEK_V4;
    bool is_qwen() const { return architecture == ModelArch::QWEN || architecture == ModelArch::QWEN2; }
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

    int layer_compress_ratio(int layer_id) const {
        if (layer_id < (int)compress_ratios.size()) return compress_ratios[layer_id];
        return 0;
    }

    void from_json(const json& j) {
        auto get = [&](auto& field, const char* key) {
            if (j.contains(key) && !j.at(key).is_null()) j.at(key).get_to(field);
        };
        get(model_name, "model_name");
        get(model_id, "model_id");
        if (model_name.empty()) get(model_name, "name");
        if (model_id.empty()) get(model_id, "id");
        if (j.contains("architecture")) {
            std::string arch_str = j["architecture"].get<std::string>();
            if (model_id.find("qwen3") != std::string::npos || model_name.find("Qwen 3") != std::string::npos || arch_str == "qwen3") {
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
    return caps;
}

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
        CUDA_CHECK(cudaMalloc(&data, bytes));
        size_bytes = bytes;
        is_host = false;
    }
    void alloc_host(size_t bytes) {
        if (data) free();
        CUDA_CHECK(cudaMallocHost(&data, bytes));
        size_bytes = bytes;
        is_host = true;
    }
    void free() {
        if (data) {
            if (is_host) cudaFreeHost(data);
            else cudaFree(data);
            data = nullptr;
            size_bytes = 0;
            is_host = false;
        }
    }
    ~GPUTensor() { free(); }

    __nv_bfloat16* bf16() { return (__nv_bfloat16*)data; }
    const __nv_bfloat16* bf16() const { return (const __nv_bfloat16*)data; }
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

    // ── Mixed Quantization Support (Hot: NVFP4, Cold: IQ2_XXS) ──
    bool is_mixed_ = false;
    moecher::platform::DirectFileHandle expert_file_cold_;
    moecher::platform::DirectFileHandle expert_file_hot_;
    int cold_block_size_ = 7077888;
    int hot_block_size_ = 10223616;
    void* hot_pool_gpu_ = nullptr;
    void* cold_pool_gpu_ = nullptr;
    std::vector<uint8_t> expert_type_map_host_;
    GPUTensor expert_type_map_gpu_;

    const uint8_t* expert_type_map_gpu() const {
        return expert_type_map_gpu_.u8();
    }

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

    bool init_mixed(const std::string& cold_bin_path, int cold_block_size,
                    const std::string& hot_bin_path, int hot_block_size,
                    const std::string& map_file_path,
                    int n_layers, int n_experts,
                    bool use_buffered_io = false) {
        is_mixed_ = true;
        cold_block_size_ = cold_block_size;
        hot_block_size_ = hot_block_size;
        expert_block_size_ = cold_block_size; // For fallback compatibility
        n_layers_ = n_layers;
        n_experts_ = n_experts;

        bool use_direct = !use_buffered_io;
        if (!expert_file_cold_.open_read(cold_bin_path, use_direct)) {
            LOG_ERROR("Cannot open cold expert bin: %s", cold_bin_path.c_str());
            return false;
        }
        if (!expert_file_hot_.open_read(hot_bin_path, use_direct)) {
            LOG_ERROR("Cannot open hot expert bin: %s", hot_bin_path.c_str());
            return false;
        }

        // Read mixed expert map
        int total_experts = n_layers * n_experts;
        expert_type_map_host_.resize(total_experts, 0);

        std::ifstream mf(map_file_path, std::ios::binary);
        if (!mf.is_open()) {
            LOG_ERROR("Cannot open mixed expert map file: %s", map_file_path.c_str());
            return false;
        }
        mf.read(reinterpret_cast<char*>(expert_type_map_host_.data()), total_experts);
        if (!mf) {
            LOG_ERROR("Failed to read %d bytes from expert map: %s", total_experts, map_file_path.c_str());
            return false;
        }
        mf.close();

        int total_hot = 0;
        int total_cold = 0;
        for (int i = 0; i < total_experts; i++) {
            if (expert_type_map_host_[i] == 1) total_hot++;
            else total_cold++;
        }

        size_t hot_bytes = (size_t)total_hot * hot_block_size;
        size_t cold_bytes = (size_t)total_cold * cold_block_size;
        size_t total_bytes = hot_bytes + cold_bytes;

        LOG_INFO("══════════════════════════════════════════════════════════════════════");
        LOG_INFO("ExpertLoader: Initializing Mixed Quantization MoE (100%% VRAM Resident)");
        LOG_INFO("  Total Experts : %d (%d layers x %d)", total_experts, n_layers, n_experts);
        LOG_INFO("  Hot Experts   : %d (%.2f GB) in 2:4 Sparse NVFP4", total_hot, (double)hot_bytes / (1024.0 * 1024.0 * 1024.0));
        LOG_INFO("  Cold Experts  : %d (%.2f GB) in IQ2_XXS + Q2_K", total_cold, (double)cold_bytes / (1024.0 * 1024.0 * 1024.0));
        LOG_INFO("  Total MoE VRAM: %.2f GB", (double)total_bytes / (1024.0 * 1024.0 * 1024.0));
        LOG_INFO("══════════════════════════════════════════════════════════════════════");

        // Copy map to GPU
        expert_type_map_gpu_.alloc(total_experts);
        CUDA_CHECK(cudaMemcpy(expert_type_map_gpu_.data, expert_type_map_host_.data(),
                              total_experts, cudaMemcpyHostToDevice));

        // Allocate VRAM pools
        CUDA_CHECK(cudaMalloc(&hot_pool_gpu_, hot_bytes));
        CUDA_CHECK(cudaMalloc(&cold_pool_gpu_, cold_bytes));

        // All resident capacity: 100% of all experts in VRAM
        cache_capacity_ = total_experts;

        flat_vram_ptrs_.assign(total_experts, nullptr);
        flat_vram_ptrs_gpu_.alloc((size_t)total_experts * sizeof(void*));

        int cur_hot = 0;
        int cur_cold = 0;
        for (int i = 0; i < total_experts; i++) {
            if (expert_type_map_host_[i] == 1) {
                flat_vram_ptrs_[i] = (char*)hot_pool_gpu_ + (size_t)cur_hot * hot_block_size;
                cur_hot++;
            } else {
                flat_vram_ptrs_[i] = (char*)cold_pool_gpu_ + (size_t)cur_cold * cold_block_size;
                cur_cold++;
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

    bool preload_all_mixed(int n_threads = 16) {
        int total = n_layers_ * n_experts_;
        LOG_INFO("Preloading %d mixed experts into VRAM (Hot: NVFP4, Cold: IQ2_XXS)...", total);

        auto start = std::chrono::steady_clock::now();
        std::atomic<int> loaded{0};

        std::vector<std::thread> workers;
        int chunk_size = (total + n_threads - 1) / n_threads;

        for (int t = 0; t < n_threads; t++) {
            int start_idx = t * chunk_size;
            int end_idx = std::min(start_idx + chunk_size, total);
            if (start_idx >= end_idx) continue;

            workers.emplace_back([this, start_idx, end_idx, &loaded, total]() {
                size_t max_block = std::max(hot_block_size_, cold_block_size_);
                void* stage_ptr = nullptr;
                cudaStream_t stream = nullptr;
                CUDA_CHECK(cudaMallocHost(&stage_ptr, max_block));
                CUDA_CHECK(cudaStreamCreate(&stream));

                for (int i = start_idx; i < end_idx; i++) {
                    uint8_t is_hot = expert_type_map_host_[i];
                    void* dst = flat_vram_ptrs_[i];

                    if (is_hot) {
                        int64_t file_offset = (int64_t)i * hot_block_size_;
                        int64_t bytes = expert_file_hot_.pread_exact(stage_ptr, hot_block_size_, file_offset);
                        if (bytes == hot_block_size_) {
                            CUDA_CHECK(cudaMemcpyAsync(dst, stage_ptr, hot_block_size_, cudaMemcpyHostToDevice, stream));
                            CUDA_CHECK(cudaStreamSynchronize(stream));
                        } else {
                            LOG_ERROR("Failed reading hot expert %d at offset %lld (got %lld)", i, (long long)file_offset, (long long)bytes);
                        }
                    } else {
                        int64_t file_offset = (int64_t)i * cold_block_size_;
                        int64_t bytes = expert_file_cold_.pread_exact(stage_ptr, cold_block_size_, file_offset);
                        if (bytes == cold_block_size_) {
                            CUDA_CHECK(cudaMemcpyAsync(dst, stage_ptr, cold_block_size_, cudaMemcpyHostToDevice, stream));
                            CUDA_CHECK(cudaStreamSynchronize(stream));
                        } else {
                            LOG_ERROR("Failed reading cold expert %d at offset %lld (got %lld)", i, (long long)file_offset, (long long)bytes);
                        }
                    }

                    int done = ++loaded;
                    if (done % 1000 == 0 || done == total) {
                        LOG_INFO("  Preloaded %d/%d mixed experts into VRAM...", done, total);
                    }
                }
                CUDA_CHECK(cudaStreamDestroy(stream));
                CUDA_CHECK(cudaFreeHost(stage_ptr));
            });
        }

        for (auto& w : workers) {
            w.join();
        }

        // Copy flat pointer table to GPU for GPU-native kernel execution
        CUDA_CHECK(cudaMemcpy(flat_vram_ptrs_gpu_.data, flat_vram_ptrs_.data(),
                              flat_vram_ptrs_.size() * sizeof(void*), cudaMemcpyHostToDevice));

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        LOG_INFO("Preload completed: 100%% VRAM Resident (11008 experts) in %.2f seconds", elapsed / 1000.0);
        return true;
    }

    const void* const* flat_vram_ptrs_gpu() const {
        return (const void* const*)flat_vram_ptrs_gpu_.data;
    }

    void cleanup() {
        expert_file_.close();
        if (is_mixed_) {
            expert_file_cold_.close();
            expert_file_hot_.close();
            if (hot_pool_gpu_) { cudaFree(hot_pool_gpu_); hot_pool_gpu_ = nullptr; }
            if (cold_pool_gpu_) { cudaFree(cold_pool_gpu_); cold_pool_gpu_ = nullptr; }
            expert_type_map_gpu_.free();
        }
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
    bool has_vision_delegate() const { return vision_delegate_engine_ != nullptr; }

    bool has_vision() const { return has_vision_; }
    bool is_vision_loaded() const { return vision_tower_ != nullptr; }
    bool supports_vision() const { return has_vision() || has_vision_delegate(); }

    // Prompt-Lookup Drafting (PLD) Speculative Decoding
    bool enable_pld_ = true;
    int pld_draft_tokens_ = 4;
    bool enable_mtp_ = true;
    bool default_enable_mtp_ = true;
#if defined(__APPLE__)
    bool enable_tree_mtp_ = false;
    bool default_enable_tree_mtp_ = false;
#else
    bool enable_tree_mtp_ = true;
    bool default_enable_tree_mtp_ = true;
#endif
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
    cudaStream_t expert_streams_[32] = {nullptr};
    cudaEvent_t expert_events_[32] = {nullptr};
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
        GPUTensor A_log;          // [48] BF16
        GPUTensor dt_bias;        // [48] BF16
        GPUTensor linear_norm_w;  // [128] BF16
        GPUTensor linear_out_proj, linear_out_proj_scale;// [5120, 6144]
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
        std::string tok_path = manifest["tokenizer"]["tokenizer_json"].get<std::string>();
        std::string tok_full = (base_dir / tok_path).string();
        if (!tokenizer_.load(tok_full) && !tokenizer_.load(tok_path) &&
            !tokenizer_.load("tokenizer.json") && !tokenizer_.load((base_dir / "tokenizer.json").string())) {
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
        size_t vram_reserve = 4ULL * 1024 * 1024 * 1024; // reserve 4GB for KV cache snapshots, activations and CUDA graphs
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
        
        size_t dram_cache_budget = (size_t)(dram_cache_gb * 1024.0 * 1024.0 * 1024.0);
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

        // Auto-load In-Process Vision Delegate Engine if specified in manifest
        json delegate_cfg;
        if (manifest.contains("vision_delegate")) {
            delegate_cfg = manifest["vision_delegate"];
        } else if (manifest.contains("model_config") && manifest["model_config"].contains("vision_delegate")) {
            delegate_cfg = manifest["model_config"]["vision_delegate"];
        }
        if (!delegate_cfg.empty()) {
            std::string delegate_rel = delegate_cfg.value("manifest", "");
            std::string delegate_manifest_path = (base_dir / delegate_rel).string();
            if (!std::ifstream(delegate_manifest_path).good()) {
                delegate_manifest_path = delegate_rel;
            }
            if (std::ifstream(delegate_manifest_path).good()) {
                LOG_INFO("[Vision Delegate] Found vision delegate manifest at: %s. Loading delegate engine...", delegate_manifest_path.c_str());
                vision_delegate_engine_ = std::make_unique<MoecherEngine>();
                int delegate_seq_len = (max_seq_len_override > 0) ? std::min(max_seq_len_override, 2048) : 2048;
                if (vision_delegate_engine_->load(delegate_manifest_path, max_vram_gb, dram_cache_gb, expert_dtype_override, buffered_io, delegate_seq_len)) {
                    LOG_INFO("[Vision Delegate] Vision delegate engine successfully loaded and operational!");
                } else {
                    LOG_ERROR("[Vision Delegate] Failed to load vision delegate engine!");
                    vision_delegate_engine_.reset();
                }
            } else {
                LOG_ERROR("[Vision Delegate] Vision delegate manifest file not found: %s", delegate_manifest_path.c_str());
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
        if (!cfg_.is_qwen() && !expert_loader_.all_resident(cfg_.num_hidden_layers)) {
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

    void forward_token_eager(int token_id, int position) {
        int track_flag = (track_expert_freq_ && track_current_token_) ? 1 : 0;
        CUDA_CHECK(cudaMemcpy(buf_track_flag_.i32(), &track_flag, sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(buf_input_token_.i32(), &token_id, sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(buf_input_pos_.i32(), &position, sizeof(int32_t), cudaMemcpyHostToDevice));
        forward_token_device_body(token_id, position);
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

    void forward_token_device_body(int token_id, int position) {
        int dim = cfg_.hidden_size;
        int hc = cfg_.hc_mult;

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
            compute_logits();
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
        compute_logits();

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
        } else {
            // ModelArch::DEEPSEEK_V4
            if (enable_batched_prefill_ && prefix_tokens.size() > 1) {
                prefill_prompt_batched_deepseek(prefix_tokens);
            } else {
                for (size_t i = 0; i < prefix_tokens.size(); i++) {
                    if (graph_captured_) {
                        CUDA_CHECK(cudaMemcpyAsync(buf_track_flag_.i32(), h_single_flag_, sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                        CUDA_CHECK(cudaMemcpyAsync(buf_input_token_.i32(), &h_prefill_tok_[i], sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                        CUDA_CHECK(cudaMemcpyAsync(buf_input_pos_.i32(), &h_prefill_pos_[i], sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                        CUDA_CHECK(cudaGraphLaunch(graph_exec_, main_stream_));
                    } else {
                        forward_token_eager(prefix_tokens[i], (int)i);
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

    struct ZeroDecodeCandidate {
        std::string choice;
        std::vector<int> token_ids;
        float raw_logit = -1e9f;
        float probability = 0.0f;
    };

    struct ZeroDecodeResult {
        int winning_index = -1;
        std::string winning_choice;
        float confidence = 0.0f;
        std::vector<ZeroDecodeCandidate> candidates;
        std::map<std::string, float> probabilities;
        int global_argmax_token_id = -1;
        std::string global_argmax_token_str;
        float global_argmax_logit = -1e9f;
        bool is_choice_top1 = false;
        float top1_gap = 0.0f;           // distance between global argmax logit and best candidate logit
        float candidate_margin = 0.0f;   // distance between #1 candidate logit and #2 candidate logit
        bool gate_passed = false;        // true if candidate passes System 1 confidence & alignment arbiter
        std::string system_path = "system_1_zdb"; // "system_1_zdb" or "system_2_fallback"
        double prefill_sec = 0.0;
        double total_sec = 0.0;
        size_t prompt_tokens = 0;
        int decode_passes = 0; // strictly 0
        std::string rendered_output;
    };

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
        if (cfg_.architecture == ModelArch::DEEPSEEK_V4 && prefix_len < cached_tokens_.size()) {
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
            } else {
                // ModelArch::DEEPSEEK_V4
                size_t remaining = (prompt.size() > prefix_len) ? (prompt.size() - prefix_len) : 0;
                if (enable_batched_prefill_ && remaining > 1) {
                    prefill_prompt_batched_deepseek(prompt, (int)prefix_len);
                } else {
                    for (size_t i = prefix_len; i < prompt.size(); i++) {
                        if (graph_captured_) {
                            CUDA_CHECK(cudaMemcpyAsync(buf_track_flag_.i32(), h_single_flag_, sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                            CUDA_CHECK(cudaMemcpyAsync(buf_input_token_.i32(), &h_prefill_tok_[i], sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                            CUDA_CHECK(cudaMemcpyAsync(buf_input_pos_.i32(), &h_prefill_pos_[i], sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                            CUDA_CHECK(cudaGraphLaunch(graph_exec_, main_stream_));
                        } else {
                            forward_token_eager(prompt[i], (int)i);
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
        result.candidates.resize(candidate_choices.size());
        std::vector<int> all_tokens_to_read;

        for (size_t c_idx = 0; c_idx < candidate_choices.size(); c_idx++) {
            const std::string& choice = candidate_choices[c_idx];
            auto& cand = result.candidates[c_idx];
            cand.choice = choice;

            auto add_tok = [&](const std::vector<int>& toks) {
                if (!toks.empty()) {
                    int t = toks[0];
                    if (t >= 0 && t < cfg_.vocab_size) {
                        if (std::find(cand.token_ids.begin(), cand.token_ids.end(), t) == cand.token_ids.end()) {
                            cand.token_ids.push_back(t);
                        }
                        if (std::find(all_tokens_to_read.begin(), all_tokens_to_read.end(), t) == all_tokens_to_read.end()) {
                            all_tokens_to_read.push_back(t);
                        }
                    }
                }
            };

            add_tok(tokenizer_.encode(choice));
            add_tok(tokenizer_.encode(" " + choice));

            std::string lower_c = choice;
            for (char& ch : lower_c) ch = (char)::tolower((unsigned char)ch);
            if (lower_c != choice) {
                add_tok(tokenizer_.encode(lower_c));
                add_tok(tokenizer_.encode(" " + lower_c));
            }

            std::string upper_c = choice;
            for (char& ch : upper_c) ch = (char)::toupper((unsigned char)ch);
            if (upper_c != choice) {
                add_tok(tokenizer_.encode(upper_c));
                add_tok(tokenizer_.encode(" " + upper_c));
            }

            std::string title_c = choice;
            if (!title_c.empty()) {
                title_c[0] = (char)::toupper((unsigned char)title_c[0]);
                for (size_t i = 1; i < title_c.size(); i++) title_c[i] = (char)::tolower((unsigned char)title_c[i]);
                if (title_c != choice && title_c != lower_c && title_c != upper_c) {
                    add_tok(tokenizer_.encode(title_c));
                    add_tok(tokenizer_.encode(" " + title_c));
                }
            }
        }

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

        float max_choice_logit = -1e9f;
        for (size_t c_idx = 0; c_idx < result.candidates.size(); c_idx++) {
            auto& cand = result.candidates[c_idx];
            cand.raw_logit = -1e9f;

            for (int tid : cand.token_ids) {
                auto it = logit_lookup.find(tid);
                if (it != logit_lookup.end()) {
                    float l_val = it->second;
                    if (l_val > cand.raw_logit) {
                        cand.raw_logit = l_val;
                    }
                }
                if (tid == result.global_argmax_token_id) {
                    result.is_choice_top1 = true;
                }
            }

            if (cand.raw_logit > max_choice_logit) {
                max_choice_logit = cand.raw_logit;
                result.winning_index = (int)c_idx;
                result.winning_choice = cand.choice;
            }
        }

        // Softmax normalization across candidate choices
        float eff_temp = std::max(temperature, 0.01f);
        float sum_exp = 0.0f;
        for (auto& cand : result.candidates) {
            float exp_val = std::exp((cand.raw_logit - max_choice_logit) / eff_temp);
            cand.probability = exp_val;
            sum_exp += exp_val;
        }

        if (sum_exp > 0.0f) {
            for (auto& cand : result.candidates) {
                cand.probability /= sum_exp;
                result.probabilities[cand.choice] = cand.probability;
            }
        }

        if (result.winning_index >= 0 && result.winning_index < (int)result.candidates.size()) {
            result.confidence = result.candidates[result.winning_index].probability;
        }

        // ── Hybrid Arbiter (System 1 vs System 2 Confidence & Alignment) ───────
        float second_choice_logit = -1e9f;
        for (size_t c_idx = 0; c_idx < result.candidates.size(); c_idx++) {
            if ((int)c_idx != result.winning_index && result.candidates[c_idx].raw_logit > second_choice_logit) {
                second_choice_logit = result.candidates[c_idx].raw_logit;
            }
        }
        result.candidate_margin = max_choice_logit - second_choice_logit;

        if (result.is_choice_top1) {
            result.top1_gap = 0.0f;
        } else {
            result.top1_gap = std::max(0.0f, result.global_argmax_logit - max_choice_logit);
        }

        // Arbiter gating decision:
        // Distinguish surface formatting tokens (e.g. "**", "*", "\"", "#", ":", "-")
        // from genuine hesitation/scratchpad tokens (e.g. "\n\n", "<think>", "Based", "I").
        bool is_formatting_token = (result.global_argmax_token_str == "**" ||
                                    result.global_argmax_token_str == "*" ||
                                    result.global_argmax_token_str == "#" ||
                                    result.global_argmax_token_str == "##" ||
                                    result.global_argmax_token_str == "\"" ||
                                    result.global_argmax_token_str == ":" ||
                                    result.global_argmax_token_str == "-");

        float effective_gap_tolerance = is_formatting_token ? std::max(max_top1_gap, 2.0f) : max_top1_gap;

        // Pass System 1 if candidate was global top-1 OR if top1_gap is within tolerance AND confidence >= min_confidence
        result.gate_passed = (result.is_choice_top1 || result.top1_gap <= effective_gap_tolerance) &&
                             (result.confidence >= min_confidence);
        result.system_path = result.gate_passed ? "system_1_zdb" : "system_2_fallback";

        auto total_end_time = std::chrono::steady_clock::now();
        result.total_sec = std::chrono::duration<double>(total_end_time - start_time).count();

        // Direct CPU Memory Stitch
        if (output_format == "text" || output_format == "choice") {
            result.rendered_output = result.winning_choice;
        } else {
            json out_json = {
                {"decision", result.winning_choice},
                {"winning_index", result.winning_index},
                {"confidence", result.confidence},
                {"probabilities", result.probabilities},
                {"zero_decode_bypass", true},
                {"decode_passes", 0},
                {"arbiter", {
                    {"gate_passed", result.gate_passed},
                    {"system_path", result.system_path},
                    {"top1_gap", result.top1_gap},
                    {"candidate_margin", result.candidate_margin},
                    {"is_choice_top1", result.is_choice_top1}
                }},
                {"global_top1", {
                    {"token_id", result.global_argmax_token_id},
                    {"token", result.global_argmax_token_str},
                    {"logit", result.global_argmax_logit},
                    {"is_candidate", result.is_choice_top1}
                }},
                {"timing", {
                    {"prefill_ms", result.prefill_sec * 1000.0},
                    {"total_ms", result.total_sec * 1000.0}
                }}
            };
            result.rendered_output = out_json.dump(-1, ' ', false, json::error_handler_t::replace);
        }

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
        } else if (cfg_.architecture == ModelArch::DEEPSEEK_V4 && prefix_len < cached_tokens_.size()) {
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
            } else {
                // ModelArch::DEEPSEEK_V4
                size_t remaining = (prompt.size() > prefix_len) ? (prompt.size() - prefix_len) : 0;
                if (enable_batched_prefill_ && remaining > 1) {
                    prefill_prompt_batched_deepseek(prompt, (int)prefix_len);
                } else {
                    for (size_t i = prefix_len; i < prompt.size(); i++) {
                        if (graph_captured_) {
                            CUDA_CHECK(cudaMemcpyAsync(buf_track_flag_.i32(), h_single_flag_, sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                            CUDA_CHECK(cudaMemcpyAsync(buf_input_token_.i32(), &h_prefill_tok_[i], sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                            CUDA_CHECK(cudaMemcpyAsync(buf_input_pos_.i32(), &h_prefill_pos_[i], sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
                            CUDA_CHECK(cudaGraphLaunch(graph_exec_, main_stream_));
                        } else {
                            forward_token_eager(prompt[i], (int)i);
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

        // Track whether we are inside a <think> block
        // DeepSeek V3/V4 thinking mode: the chat template appends <think> to the prompt,
        // so the model starts generation INSIDE the think block.
        bool in_think_block = enable_thinking && !had_multimodal;
        if (had_multimodal) {
            enable_thinking = false;
            max_thinking_tokens = 0;
        }
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
            if (in_think_block && (token_text.find("</think>") != std::string::npos || (token_buffer + token_text).find("</think>") != std::string::npos)) {
                in_think_block = false;
                think_block_ended = true;
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
                next_token = sample_token(temperature, history, content_tokens_generated, in_think_block,
                                          top_k, top_p, min_p);
            }
            
            // If EOS is sampled while inside think block, transition to </think> and continue
            if (in_think_block && (next_token == cfg_.eos_token_id || (eos2_id >= 0 && next_token == eos2_id))) {
                LOG_WARN("EOS sampled during thinking at step %d (%d thinking tokens). Transitioning to </think> and starting content.",
                         t, thinking_tokens_generated);
                in_think_block = false;
                think_block_ended = true;
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

            if (!enable_thinking && (next_token == think_start_id || next_token == think_end_id)) {
                if (content_tokens_generated > 0) {
                    LOG_WARN("Stop on think token in non-thinking mode: token=%d at step %d (content: %d)",
                             next_token, t, content_tokens_generated);
                    finish_reason = "stop";
                    break;
                }
                float neg_inf = -1e9f;
                CUDA_CHECK(cudaMemcpyAsync(buf_logits_.f32() + next_token, &neg_inf, sizeof(float), cudaMemcpyHostToDevice, main_stream_));
                next_token = -1;
                continue;
            }

            // 2. Speculative Decoding Verification Loop (MTP Self-Drafter + PLD + 2B Neural Drafter fallback)
            bool allow_draft = enable_spec_ && (draft_streak >= 0) && !in_think_block &&
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
                        CUDA_CHECK(cudaStreamSynchronize(main_stream_));
                        argmax_cache_valid_ = false;
                        next_token = sample_token(temperature, history, content_tokens_generated, in_think_block,
                                                  top_k, top_p, min_p);
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
                if (cfg_.architecture == ModelArch::QWEN && !is_mtp && (!enable_mtp_ || !mtp_drafter_.loaded_) && enable_pld_ && !history.empty()) {
                    int max_pld = std::min(pld_draft_tokens_, 2);
                    std::vector<int> pld_cands = PromptLookupDrafter::draft(history, max_pld, 6, 6);
                    if (!pld_cands.empty()) {
                        for (int tok : pld_cands) {
                            cand_tokens.push_back(tok);
                        }
                        is_pld = true;
                    }
                }

                // Priority 3: Legacy 2B Neural Drafter (fallback on confirmed streak)
                if (cfg_.architecture == ModelArch::QWEN && !is_mtp && !is_pld && qwen_draft_.loaded_ && draft_streak >= 2) {
                    int K = (draft_streak >= 3) ? 4 : 2;
                    qwen_draft_.draft_k_tokens(next_token, position, K, cand_tokens, main_stream_);
                }

                int M = (int)cand_tokens.size();
                if (cfg_.architecture == ModelArch::QWEN && M >= 2) {
                    auto tv0 = std::chrono::steady_clock::now();
                    forward_token_batch_qwen(cand_tokens.data(), position, M);

                    int num_verify = M - 1;
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

                        if (pred_k == draft_cand &&
                            pred_k != cfg_.eos_token_id && (eos2_id < 0 || pred_k != eos2_id) &&
                            (enable_thinking || (pred_k != think_start_id && pred_k != think_end_id))) {
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
                            CUDA_CHECK(cudaStreamSynchronize(main_stream_));
                            argmax_cache_valid_ = false;
                            next_token = sample_token(temperature, history, content_tokens_generated, in_think_block,
                                                      top_k, top_p, min_p);
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
                            draft_streak = is_mtp ? 0 : -8;
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
                    draft_streak = -2;
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

    // ── Vision Perception Delegate Execution ─────────────────────────────────
    std::string describe_image(const std::string& image_b64, const std::string& user_query = "") {
        if (has_vision_delegate()) {
            return vision_delegate_engine_->describe_image(image_b64, user_query);
        }
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
        if (!user_query.empty()) {
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
            /*max_tokens=*/512,
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
            if (dtype != "int4" && dtype != "int3" && dtype != "fp4") {
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
                gpu_s.dtype = info.value("scale_dtype", dtype == "fp4" ? "F8_E8M0" : "bfloat16");
                gpu_s.alloc(scale_nbytes);
                CUDA_CHECK(cudaMemcpy(gpu_s.data, (char*)mapped + scale_offset, scale_nbytes, cudaMemcpyHostToDevice));
            } else {
                std::string scale_name = name + "_scale";
                if (!tensor_map.contains(scale_name)) {
                    size_t w_pos = name.rfind(".weight");
                    if (w_pos != std::string::npos) {
                        scale_name = name.substr(0, w_pos) + ".weight_scale";
                    }
                }
                if (tensor_map.contains(scale_name)) {
                    load_tensor(gpu_s, scale_name);
                }
            }
            return true;
        };

        auto load_layer_weight = [&](GPUTensor& gpu_w, GPUTensor& gpu_s, const std::string& name) -> bool {
            if (load_quant_tensor(gpu_w, gpu_s, name)) return true;
            return load_tensor(gpu_w, name);
        };

        // Load global tensors
        if (!load_quant_tensor(embed_weight_, embed_weight_scale_, "embed.weight")) {
            if (!load_quant_tensor(embed_weight_, embed_weight_scale_, "model.embed_tokens.weight")) {
                if (!load_quant_tensor(embed_weight_, embed_weight_scale_, "model.language_model.embed_tokens.weight")) {
                    if (!load_tensor(embed_weight_, "embed.weight")) {
                        if (!load_tensor(embed_weight_, "model.embed_tokens.weight")) {
                            load_tensor(embed_weight_, "model.language_model.embed_tokens.weight");
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

        if (cfg_.architecture == ModelArch::QWEN) {
            int n_linear = 0;
            for (auto& lw : layers_) if (lw.is_linear_attn) n_linear++;
            if (n_linear > 0) {
                size_t ssm_bytes_per_layer = 48 * 128 * 128 * sizeof(__nv_bfloat16);
                size_t conv_bytes_per_layer = 10240 * 4 * sizeof(__nv_bfloat16);
                target_ssm_pool_.alloc(n_linear * ssm_bytes_per_layer);
                target_conv_pool_.alloc(n_linear * conv_bytes_per_layer);
                CUDA_CHECK(cudaMemset(target_ssm_pool_.data, 0, target_ssm_pool_.size_bytes));
                CUDA_CHECK(cudaMemset(target_conv_pool_.data, 0, target_conv_pool_.size_bytes));

                int num_slots;
                if (enable_tree_mtp_ && enable_mtp_) {
                    num_slots = gpu_caps_.is_vram_constrained ? std::min(8, tree_nodes_ + 1) : 8;
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
        buf_moe_accum_.alloc(dim * sizeof(__nv_bfloat16));
        size_t dequant_needed;
        if (cfg_.n_routed_experts > 0) {
            dequant_needed = std::max((size_t)256 * 1024 * 1024, (size_t)max_inter * dim * sizeof(__nv_bfloat16));
        } else {
            dequant_needed = 48 * 1024 * 1024;
        }
        buf_dequant_.alloc(dequant_needed);
        buf_dequant_2_.alloc(dequant_needed);
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

        // Speculative decoding and parallel prefill batch buffers (M=512 capacity, hardware Tensor Core prefill)
        int max_batch_m = 512;
        buf_input_tokens_batch_.alloc(max_batch_m * sizeof(int32_t));
        buf_input_pos_batch_.alloc(max_batch_m * sizeof(int32_t));
        buf_hidden_batch_.alloc(max_batch_m * dim * sizeof(__nv_bfloat16));
        buf_hidden2_batch_.alloc(max_batch_m * dim * sizeof(__nv_bfloat16));
        int max_act_k = (int)std::max({(size_t)dim, (size_t)max_inter, (size_t)10240, (size_t)6144});
        int max_act_m = 512;
        buf_activation_fp4_.alloc((size_t)max_act_m * (max_act_k / 2));
        buf_activation_scale_.alloc((size_t)max_act_m * (max_act_k / 32));
        buf_activation_fp8_.alloc((size_t)max_act_m * max_act_k);
        buf_activation_fp8_side_.alloc((size_t)max_act_m * max_act_k);
        buf_q_batch_.alloc(max_batch_m * std::max(10240, 2 * n_heads * head_dim_val) * sizeof(__nv_bfloat16));
        buf_gate_batch_.alloc(max_batch_m * max_inter * sizeof(__nv_bfloat16));
        buf_up_batch_.alloc(max_batch_m * max_inter * sizeof(__nv_bfloat16));
        buf_attn_out_batch_.alloc(max_batch_m * std::max(6144, n_heads * head_dim_val) * sizeof(__nv_bfloat16));
        buf_linear_a_batch_.alloc(max_batch_m * 64 * sizeof(__nv_bfloat16));
        buf_linear_b_batch_.alloc(max_batch_m * 64 * sizeof(__nv_bfloat16));
        buf_logits_batch_.alloc((size_t)std::min(max_batch_m, 8) * cfg_.vocab_size * sizeof(float));
        buf_argmax_out_batch_.alloc(max_batch_m * sizeof(int32_t));
        buf_tree_ancestor_mask_.alloc(8 * sizeof(uint64_t));
        buf_tree_parent_indices_.alloc(8 * sizeof(int32_t));
        buf_tree_winning_slots_.alloc(8 * sizeof(int32_t));

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
        if (handle == cublas_handle_ && stream != main_stream_) cublasSetStream(handle, stream);
        CUBLAS_CHECK(cublasGemmEx(
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
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
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
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
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
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
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
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
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

    // ── Forward one layer (Qwen / Llama GQA + SwiGLU) ───────────────────────

    void forward_layer_qwen(int layer_id, int position) {
        auto& lw = layers_[layer_id];
        int dim = cfg_.hidden_size;
        int n_q_heads = cfg_.num_attention_heads;
        int n_kv_heads = cfg_.num_key_value_heads;
        int head_dim = cfg_.head_dim;
        int inter_size = cfg_.intermediate_size > 0 ? cfg_.intermediate_size : cfg_.moe_intermediate_size;

        auto matmul_proj = [&](GPUTensor& out, GPUTensor& in_vec, GPUTensor& weight, GPUTensor& scale, int N, int K) {
            if (weight.dtype == "fp4") {
                gemv_fp4_cuda(out.bf16(), in_vec.bf16(), (const uint8_t*)weight.data, (const uint8_t*)scale.data, N, K, main_stream_);
            } else if (weight.dtype == "int4") {
                gemv_int4_cuda(out.bf16(), in_vec.bf16(), (const uint8_t*)weight.data, scale.bf16(), N, K, main_stream_);
            } else if (weight.dtype == "int3") {
                gemv_int3_cuda(out.bf16(), in_vec.bf16(), (const uint8_t*)weight.data, scale.bf16(), N, K, main_stream_);
            } else {
                gemv_bf16_out_bf16_cuda(out.bf16(), weight.bf16(), in_vec.bf16(), N, K, main_stream_);
            }
        };

        // 1. Attention Pre-RMSNorm: buf_hidden_ -> buf_hidden2_
        if (cfg_.architecture == ModelArch::QWEN2) {
            rms_norm_cuda(buf_hidden2_.bf16(), buf_hidden_.bf16(),
                          lw.attn_norm_w.bf16(), dim, cfg_.rms_norm_eps, main_stream_);
        } else {
            rms_norm_one_centered_cuda(buf_hidden2_.bf16(), buf_hidden_.bf16(),
                                       lw.attn_norm_w.bf16(), dim, cfg_.rms_norm_eps, main_stream_);
        }

        if (is_blackwell_tc_active_) {
            quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                           buf_hidden2_.bf16(), 1, dim, main_stream_);
        }

        if (lw.is_linear_attn) {
            // Qwen 3.8 Gated DeltaNet Linear Attention Projections
            if (is_blackwell_tc_active_ && lw.w_in_qkv.dtype == "fp4") {
                gemm_fp4_blackwell_tensorcore_cuda(buf_q_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                    (const uint8_t*)lw.w_in_qkv.data, (const uint8_t*)lw.w_in_qkv_scale.data,
                                                    10240, dim, 1, main_stream_);
            } else {
                matmul_proj(buf_q_, buf_hidden2_, lw.w_in_qkv, lw.w_in_qkv_scale, 10240, dim);
            }
            if (is_blackwell_tc_active_ && lw.w_in_z.dtype == "fp4") {
                gemm_fp4_blackwell_tensorcore_cuda(buf_up_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                    (const uint8_t*)lw.w_in_z.data, (const uint8_t*)lw.w_in_z_scale.data,
                                                    6144, dim, 1, main_stream_);
            } else {
                matmul_proj(buf_up_, buf_hidden2_, lw.w_in_z, lw.w_in_z_scale, 6144, dim);
            }
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

            // 4. Output Projection: in-place residual accumulation into buf_hidden_
            if (lw.linear_out_proj.dtype == "fp4") {
                if (is_blackwell_tc_active_) {
                    quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                   buf_attn_out_.bf16(), 1, 6144, main_stream_);
                    gemm_fp4_residual_blackwell_tensorcore_cuda(buf_hidden_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                                (const uint8_t*)lw.linear_out_proj.data, (const uint8_t*)lw.linear_out_proj_scale.data,
                                                                dim, 6144, 1, main_stream_);
                } else {
                    gemv_fp4_residual_cuda(buf_hidden_.bf16(), buf_attn_out_.bf16(),
                                            (const uint8_t*)lw.linear_out_proj.data, (const uint8_t*)lw.linear_out_proj_scale.data,
                                            dim, 6144, main_stream_);
                }
            } else if (lw.linear_out_proj.dtype == "int4") {
                gemv_int4_residual_cuda(buf_hidden_.bf16(), buf_attn_out_.bf16(),
                                        (const uint8_t*)lw.linear_out_proj.data, lw.linear_out_proj_scale.bf16(),
                                        dim, 6144, main_stream_);
            } else {
                matmul_proj(buf_hidden2_, buf_attn_out_, lw.linear_out_proj, lw.linear_out_proj_scale, dim, 6144);
                vector_add_bf16_cuda(buf_hidden_.bf16(), buf_hidden2_.bf16(), dim, main_stream_);
            }
        } else if (cfg_.architecture == ModelArch::QWEN2) {
            // Standard Full GQA Attention (Ungated) Projections
            int q_dim = n_q_heads * head_dim;
            int kv_dim = n_kv_heads * head_dim;
            if (is_blackwell_tc_active_ && lw.w_q.dtype == "fp4") {
                gemm_fp4_blackwell_tensorcore_cuda(buf_q_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                    (const uint8_t*)lw.w_q.data, (const uint8_t*)lw.w_q_scale.data,
                                                    q_dim, dim, 1, main_stream_);
            } else {
                matmul_proj(buf_q_, buf_hidden2_, lw.w_q, lw.w_q_scale, q_dim, dim);
            }
            if (is_blackwell_tc_active_ && lw.w_k.dtype == "fp4") {
                gemm_fp4_blackwell_tensorcore_cuda(buf_gate_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                    (const uint8_t*)lw.w_k.data, (const uint8_t*)lw.w_k_scale.data,
                                                    kv_dim, dim, 1, main_stream_);
            } else {
                matmul_proj(buf_gate_, buf_hidden2_, lw.w_k, lw.w_k_scale, kv_dim, dim);
            }
            if (is_blackwell_tc_active_ && lw.w_v.dtype == "fp4") {
                gemm_fp4_blackwell_tensorcore_cuda(buf_up_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                    (const uint8_t*)lw.w_v.data, (const uint8_t*)lw.w_v_scale.data,
                                                    kv_dim, dim, 1, main_stream_);
            } else {
                matmul_proj(buf_up_, buf_hidden2_, lw.w_v, lw.w_v_scale, kv_dim, dim);
            }

            if (lw.gqa_q_bias.data) {
                add_bias_bf16_cuda(buf_q_.bf16(), lw.gqa_q_bias.bf16(), 1, q_dim, main_stream_);
            }
            if (lw.gqa_k_bias.data) {
                add_bias_bf16_cuda(buf_gate_.bf16(), lw.gqa_k_bias.bf16(), 1, kv_dim, main_stream_);
            }
            if (lw.gqa_v_bias.data) {
                add_bias_bf16_cuda(buf_up_.bf16(), lw.gqa_v_bias.bf16(), 1, kv_dim, main_stream_);
            }

            // 3 & 4. QK Norm + RoPE + GQA FP8 Decode (Ungated)
            qwen2_gqa_decode_fp8_cuda(
                buf_attn_out_.bf16(),
                buf_q_.bf16(),
                buf_gate_.bf16(),
                buf_up_.bf16(),
                lw.gqa_q_norm_w.data ? lw.gqa_q_norm_w.bf16() : nullptr,
                lw.gqa_k_norm_w.data ? lw.gqa_k_norm_w.bf16() : nullptr,
                lw.k_cache_gqa.u8(),
                lw.v_cache_gqa.u8(),
                n_q_heads, n_kv_heads, head_dim,
                buf_input_pos_.i32(), position, cfg_.max_seq_len,
                cfg_.rope_theta, cfg_.rms_norm_eps,
                buf_mrope_pos_.data ? buf_mrope_pos_.i32() : nullptr,
                main_stream_);

            // 5. Output Projection: in-place residual accumulation into buf_hidden_
            if (lw.w_o.dtype == "fp4") {
                if (is_blackwell_tc_active_) {
                    quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                   buf_attn_out_.bf16(), 1, q_dim, main_stream_);
                    gemm_fp4_residual_blackwell_tensorcore_cuda(buf_hidden_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                                (const uint8_t*)lw.w_o.data, (const uint8_t*)lw.w_o_scale.data,
                                                                dim, q_dim, 1, main_stream_);
                } else {
                    gemv_fp4_residual_cuda(buf_hidden_.bf16(), buf_attn_out_.bf16(),
                                            (const uint8_t*)lw.w_o.data, (const uint8_t*)lw.w_o_scale.data,
                                            dim, q_dim, main_stream_);
                }
            } else if (lw.w_o.dtype == "int4") {
                gemv_int4_residual_cuda(buf_hidden_.bf16(), buf_attn_out_.bf16(),
                                        (const uint8_t*)lw.w_o.data, lw.w_o_scale.bf16(),
                                        dim, q_dim, main_stream_);
            } else {
                matmul_proj(buf_hidden2_, buf_attn_out_, lw.w_o, lw.w_o_scale, dim, q_dim);
                vector_add_bf16_cuda(buf_hidden_.bf16(), buf_hidden2_.bf16(), dim, main_stream_);
            }
        } else {
            // Standard Full GQA Attention (Gated) Projections
            if (is_blackwell_tc_active_ && lw.w_q.dtype == "fp4") {
                gemm_fp4_blackwell_tensorcore_cuda(buf_q_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                    (const uint8_t*)lw.w_q.data, (const uint8_t*)lw.w_q_scale.data,
                                                    2 * n_q_heads * head_dim, dim, 1, main_stream_);
            } else {
                matmul_proj(buf_q_, buf_hidden2_, lw.w_q, lw.w_q_scale, 2 * n_q_heads * head_dim, dim);
            }
            if (is_blackwell_tc_active_ && lw.w_k.dtype == "fp4") {
                gemm_fp4_blackwell_tensorcore_cuda(buf_gate_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                    (const uint8_t*)lw.w_k.data, (const uint8_t*)lw.w_k_scale.data,
                                                    n_kv_heads * head_dim, dim, 1, main_stream_);
            } else {
                matmul_proj(buf_gate_, buf_hidden2_, lw.w_k, lw.w_k_scale, n_kv_heads * head_dim, dim);
            }
            if (is_blackwell_tc_active_ && lw.w_v.dtype == "fp4") {
                gemm_fp4_blackwell_tensorcore_cuda(buf_up_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                    (const uint8_t*)lw.w_v.data, (const uint8_t*)lw.w_v_scale.data,
                                                    n_kv_heads * head_dim, dim, 1, main_stream_);
            } else {
                matmul_proj(buf_up_, buf_hidden2_, lw.w_v, lw.w_v_scale, n_kv_heads * head_dim, dim);
            }

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

            // 5. Output Projection: in-place residual accumulation into buf_hidden_
            if (lw.w_o.dtype == "fp4") {
                if (is_blackwell_tc_active_) {
                    quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                   buf_attn_out_.bf16(), 1, n_q_heads * head_dim, main_stream_);
                    gemm_fp4_residual_blackwell_tensorcore_cuda(buf_hidden_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                                (const uint8_t*)lw.w_o.data, (const uint8_t*)lw.w_o_scale.data,
                                                                dim, n_q_heads * head_dim, 1, main_stream_);
                } else {
                    gemv_fp4_residual_cuda(buf_hidden_.bf16(), buf_attn_out_.bf16(),
                                            (const uint8_t*)lw.w_o.data, (const uint8_t*)lw.w_o_scale.data,
                                            dim, n_q_heads * head_dim, main_stream_);
                }
            } else if (lw.w_o.dtype == "int4") {
                gemv_int4_residual_cuda(buf_hidden_.bf16(), buf_attn_out_.bf16(),
                                        (const uint8_t*)lw.w_o.data, lw.w_o_scale.bf16(),
                                        dim, n_q_heads * head_dim, main_stream_);
            } else {
                matmul_proj(buf_hidden2_, buf_attn_out_, lw.w_o, lw.w_o_scale, dim, n_q_heads * head_dim);
                vector_add_bf16_cuda(buf_hidden_.bf16(), buf_hidden2_.bf16(), dim, main_stream_);
            }
        }

        // 7. FFN Pre-RMSNorm: buf_hidden_ -> buf_hidden2_
        if (cfg_.architecture == ModelArch::QWEN2) {
            rms_norm_cuda(buf_hidden2_.bf16(), buf_hidden_.bf16(),
                          lw.ffn_norm_w.bf16(), dim, cfg_.rms_norm_eps, main_stream_);
        } else {
            rms_norm_one_centered_cuda(buf_hidden2_.bf16(), buf_hidden_.bf16(),
                                       lw.ffn_norm_w.bf16(), dim, cfg_.rms_norm_eps, main_stream_);
        }

        // 8 & 9. Gate & Up projections + Fused SiLU(Gate) * Up
        if (lw.w_gate.dtype == "fp4") {
            if (is_blackwell_tc_active_) {
                quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                               buf_hidden2_.bf16(), 1, dim, main_stream_);
                gemm_fp4_swiglu_blackwell_tensorcore_cuda(buf_gate_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                          (const uint8_t*)lw.w_gate.data, (const uint8_t*)lw.w_gate_scale.data,
                                                          (const uint8_t*)lw.w_up.data, (const uint8_t*)lw.w_up_scale.data,
                                                          inter_size, dim, 1, cfg_.swiglu_limit, main_stream_);
            } else {
                gemv_fp4_swiglu_fused_cuda(buf_gate_.bf16(), buf_hidden2_.bf16(),
                                            (const uint8_t*)lw.w_gate.data, (const uint8_t*)lw.w_gate_scale.data,
                                            (const uint8_t*)lw.w_up.data, (const uint8_t*)lw.w_up_scale.data,
                                            inter_size, dim, cfg_.swiglu_limit, main_stream_);
            }
        } else if (lw.w_gate.dtype == "int4") {
            gemv_int4_swiglu_fused_cuda(buf_gate_.bf16(), buf_hidden2_.bf16(),
                                        (const uint8_t*)lw.w_gate.data, lw.w_gate_scale.bf16(),
                                        (const uint8_t*)lw.w_up.data, lw.w_up_scale.bf16(),
                                        inter_size, dim, cfg_.swiglu_limit, main_stream_);
        } else if (lw.w_gate.dtype == "int3") {
            gemv_int3_swiglu_fused_cuda(buf_gate_.bf16(), buf_hidden2_.bf16(),
                                        (const uint8_t*)lw.w_gate.data, lw.w_gate_scale.bf16(),
                                        (const uint8_t*)lw.w_up.data, lw.w_up_scale.bf16(),
                                        inter_size, dim, cfg_.swiglu_limit, main_stream_);
        } else {
            matmul_proj(buf_gate_, buf_hidden2_, lw.w_gate, lw.w_gate_scale, inter_size, dim);
            matmul_proj(buf_up_, buf_hidden2_, lw.w_up, lw.w_up_scale, inter_size, dim);
            silu_mul_cuda(buf_gate_.bf16(), buf_gate_.bf16(), buf_up_.bf16(), inter_size, cfg_.swiglu_limit, main_stream_);
        }

        // 10. Down projection: in-place residual accumulation into buf_hidden_
        if (lw.w_down.dtype == "fp4") {
            if (is_blackwell_tc_active_) {
                quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                               buf_gate_.bf16(), 1, inter_size, main_stream_);
                gemm_fp4_residual_blackwell_tensorcore_cuda(buf_hidden_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                            (const uint8_t*)lw.w_down.data, (const uint8_t*)lw.w_down_scale.data,
                                                            dim, inter_size, 1, main_stream_);
            } else {
                gemv_fp4_residual_cuda(buf_hidden_.bf16(), buf_gate_.bf16(),
                                        (const uint8_t*)lw.w_down.data, (const uint8_t*)lw.w_down_scale.data,
                                        dim, inter_size, main_stream_);
            }
        } else if (lw.w_down.dtype == "int4") {
            gemv_int4_residual_cuda(buf_hidden_.bf16(), buf_gate_.bf16(),
                                    (const uint8_t*)lw.w_down.data, lw.w_down_scale.bf16(),
                                    dim, inter_size, main_stream_);
        } else if (lw.w_down.dtype == "int3") {
            gemv_int3_residual_cuda(buf_hidden_.bf16(), buf_gate_.bf16(),
                                    (const uint8_t*)lw.w_down.data, lw.w_down_scale.bf16(),
                                    dim, inter_size, main_stream_);
        } else {
            matmul_proj(buf_hidden2_, buf_gate_, lw.w_down, lw.w_down_scale, dim, inter_size);
            vector_add_bf16_cuda(buf_hidden_.bf16(), buf_hidden2_.bf16(), dim, main_stream_);
        }
    }

    void forward_layer_qwen_batch(int layer_id, int position, int M, bool is_tree = false) {
        auto& lw = layers_[layer_id];
        int dim = cfg_.hidden_size;
        int n_q_heads = cfg_.num_attention_heads;
        int n_kv_heads = cfg_.num_key_value_heads;
        int head_dim = cfg_.head_dim;
        int inter_size = cfg_.intermediate_size > 0 ? cfg_.intermediate_size : cfg_.moe_intermediate_size;

        static bool s_layer_prof_done[64] = {false};
        bool prof = (M > 8 && !s_layer_prof_done[layer_id] && (layer_id == 0 || layer_id == 3));
        std::chrono::high_resolution_clock::time_point t0, t1, t2, t3, t4, t5, t6, t7;
        if (prof) {
            cudaStreamSynchronize(main_stream_);
            t0 = std::chrono::high_resolution_clock::now();
        }

        auto matmul_proj_batch = [&](GPUTensor& out, GPUTensor& in_vec, GPUTensor& weight, GPUTensor& scale, int N, int K, float beta = 0.0f) {
            if (weight.dtype == "fp4") {
                if (M == 1) {
                    gemv_fp4_cuda(out.bf16(), in_vec.bf16(), (const uint8_t*)weight.data, (const uint8_t*)scale.data, N, K, main_stream_);
                } else {
                    gemm_fp4_batch_cuda(out.bf16(), in_vec.bf16(), (const uint8_t*)weight.data, (const uint8_t*)scale.data, N, K, M, main_stream_);
                }
            } else if (weight.dtype == "int4") {
                gemm_int4_dequant(out.bf16(), M, N, K, in_vec.bf16(), (const uint8_t*)weight.data, scale.bf16(), 32, main_stream_, beta, &buf_dequant_, &buf_dequant_2_);
            } else if (weight.dtype == "int3") {
                gemm_int3_dequant(out.bf16(), M, N, K, in_vec.bf16(), (const uint8_t*)weight.data, scale.bf16(), 32, main_stream_, beta, &buf_dequant_, &buf_dequant_2_);
            } else {
                if (M <= 8) {
                    gemv_bf16_out_bf16_batch_cuda(out.bf16(), weight.bf16(), in_vec.bf16(), N, K, M, main_stream_);
                } else {
                    gemm_bf16(out.bf16(), M, N, K, in_vec.bf16(), weight.bf16(), 1.0f, beta);
                }
            }
        };

        // 1. Attention Pre-RMSNorm: [M, dim] -> [M, dim]
        if (cfg_.architecture == ModelArch::QWEN2) {
            rms_norm_cuda_batched(buf_hidden2_batch_.bf16(), buf_hidden_batch_.bf16(),
                                  lw.attn_norm_w.bf16(), M, dim, cfg_.rms_norm_eps, main_stream_);
        } else {
            rms_norm_one_centered_cuda_batched(buf_hidden2_batch_.bf16(), buf_hidden_batch_.bf16(),
                                               lw.attn_norm_w.bf16(), M, dim, cfg_.rms_norm_eps, main_stream_);
        }

        if (prof) {
            cudaStreamSynchronize(main_stream_);
            t1 = std::chrono::high_resolution_clock::now();
        }

        if (is_blackwell_tc_active_ && (lw.w_in_qkv.dtype == "fp4" || lw.w_q.dtype == "fp4")) {
            quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                           buf_hidden2_batch_.bf16(), M, dim, main_stream_);
        }

        if (lw.is_linear_attn) {
            // Qwen 3.8 Gated DeltaNet Linear Attention Projections for M tokens
            if (M > 8 && (lw.w_in_qkv.dtype == "int3" || lw.w_in_qkv.dtype == "int4") && (lw.w_in_z.dtype == "int3" || lw.w_in_z.dtype == "int4")) {
                // Dual-stream concurrent projections:
                // main_stream_: w_in_qkv -> buf_q_batch_ using buf_dequant_
                // side_stream_: w_in_z   -> buf_up_batch_ using buf_dequant_2_
                CUDA_CHECK(cudaEventRecord(main_event_, main_stream_));
                CUDA_CHECK(cudaStreamWaitEvent(side_stream_, main_event_, 0));

                if (lw.w_in_z.dtype == "int3") {
                    gemm_int3_dequant(buf_up_batch_.bf16(), M, 6144, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_in_z.data, lw.w_in_z_scale.bf16(), 32,
                                      side_stream_, 0.0f, &buf_dequant_2_, cublas_side_handle_);
                } else {
                    gemm_int4_dequant(buf_up_batch_.bf16(), M, 6144, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_in_z.data, lw.w_in_z_scale.bf16(), 32,
                                      side_stream_, 0.0f, &buf_dequant_2_, cublas_side_handle_);
                }

                gemm_bf16(buf_linear_a_batch_.bf16(), M, 48, dim, buf_hidden2_batch_.bf16(), lw.w_in_a.bf16(), 1.0f, 0.0f, side_stream_, cublas_side_handle_);
                gemm_bf16(buf_linear_b_batch_.bf16(), M, 48, dim, buf_hidden2_batch_.bf16(), lw.w_in_b.bf16(), 1.0f, 0.0f, side_stream_, cublas_side_handle_);
                CUDA_CHECK(cudaEventRecord(side_event_, side_stream_));

                if (lw.w_in_qkv.dtype == "int3") {
                    gemm_int3_dequant(buf_q_batch_.bf16(), M, 10240, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_in_qkv.data, lw.w_in_qkv_scale.bf16(), 32,
                                      main_stream_, 0.0f, &buf_dequant_, cublas_handle_);
                } else {
                    gemm_int4_dequant(buf_q_batch_.bf16(), M, 10240, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_in_qkv.data, lw.w_in_qkv_scale.bf16(), 32,
                                      main_stream_, 0.0f, &buf_dequant_, cublas_handle_);
                }

                CUDA_CHECK(cudaStreamWaitEvent(main_stream_, side_event_, 0));
            } else {
                if (is_blackwell_tc_active_ && lw.w_in_qkv.dtype == "fp4") {
                    gemm_fp4_blackwell_tensorcore_cuda(buf_q_batch_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                        (const uint8_t*)lw.w_in_qkv.data, (const uint8_t*)lw.w_in_qkv_scale.data,
                                                        10240, dim, M, main_stream_);
                } else {
                    matmul_proj_batch(buf_q_batch_, buf_hidden2_batch_, lw.w_in_qkv, lw.w_in_qkv_scale, 10240, dim);
                }
                if (is_blackwell_tc_active_ && lw.w_in_z.dtype == "fp4") {
                    gemm_fp4_blackwell_tensorcore_cuda(buf_up_batch_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                        (const uint8_t*)lw.w_in_z.data, (const uint8_t*)lw.w_in_z_scale.data,
                                                        6144, dim, M, main_stream_);
                } else {
                    matmul_proj_batch(buf_up_batch_, buf_hidden2_batch_, lw.w_in_z, lw.w_in_z_scale, 6144, dim);
                }
#if defined(__APPLE__)
                deltanet_in_proj_ab_batch_cuda(buf_linear_a_batch_.bf16(), buf_linear_b_batch_.bf16(),
                                               lw.w_in_a.bf16(), lw.w_in_b.bf16(), buf_hidden2_batch_.bf16(),
                                               48, dim, M, main_stream_);
#else
                if (M <= 8) {
                    gemv_bf16_out_bf16_batch_cuda(buf_linear_a_batch_.bf16(), lw.w_in_a.bf16(), buf_hidden2_batch_.bf16(), 48, dim, M, main_stream_);
                    gemv_bf16_out_bf16_batch_cuda(buf_linear_b_batch_.bf16(), lw.w_in_b.bf16(), buf_hidden2_batch_.bf16(), 48, dim, M, main_stream_);
                } else {
                    gemm_bf16(buf_linear_a_batch_.bf16(), M, 48, dim, buf_hidden2_batch_.bf16(), lw.w_in_a.bf16());
                    gemm_bf16(buf_linear_b_batch_.bf16(), M, 48, dim, buf_hidden2_batch_.bf16(), lw.w_in_b.bf16());
                }
#endif
            }

            if (prof) {
                cudaStreamSynchronize(main_stream_);
                t2 = std::chrono::high_resolution_clock::now();
            }

            if (is_tree) {
                deltanet_linear_attention_tree_cuda(
                    buf_attn_out_batch_.bf16(),
                    buf_q_batch_.bf16(),
                    buf_up_batch_.bf16(),
                    buf_linear_a_batch_.bf16(),
                    buf_linear_b_batch_.bf16(),
                    lw.conv1d_w.bf16(),
                    lw.conv_state.bf16(),
                    lw.conv_state.bf16(),
                    lw.conv_state_slots[0].bf16(),
                    lw.conv_state_slots[1].bf16(),
                    lw.conv_state_slots[2].bf16(),
                    lw.conv_state_slots[3].bf16(),
                    lw.conv_state_slots[4].bf16(),
                    lw.conv_state_slots[5].bf16(),
                    lw.conv_state_slots[6].bf16(),
                    lw.conv_state_slots[7].bf16(),
                    lw.A_log.bf16(),
                    lw.dt_bias.bf16(),
                    lw.linear_norm_w.bf16(),
                    lw.ssm_state.bf16(),
                    lw.ssm_state.bf16(),
                    lw.ssm_state_slots[0].bf16(),
                    lw.ssm_state_slots[1].bf16(),
                    lw.ssm_state_slots[2].bf16(),
                    lw.ssm_state_slots[3].bf16(),
                    lw.ssm_state_slots[4].bf16(),
                    lw.ssm_state_slots[5].bf16(),
                    lw.ssm_state_slots[6].bf16(),
                    lw.ssm_state_slots[7].bf16(),
                    buf_tree_parent_indices_.i32(),
                    16, 48, 128, M, main_stream_);
            } else {
                // DeltaNet SSM recurrence across M tokens (fully batched, zero intermediate VRAM roundtrips)
                deltanet_linear_attention_decode_batch_cuda(
                    buf_attn_out_batch_.bf16(),
                    buf_q_batch_.bf16(),
                    buf_up_batch_.bf16(),
                    buf_linear_a_batch_.bf16(),
                    buf_linear_b_batch_.bf16(),
                    lw.conv1d_w.bf16(),
                    lw.conv_state.bf16(),
                    lw.conv_state.bf16(),
                    (M > 1 && M <= 4) ? lw.conv_state_slots[0].bf16() : nullptr,
                    (M > 2 && M <= 4) ? lw.conv_state_slots[1].bf16() : nullptr,
                    (M > 3 && M <= 4) ? lw.conv_state_slots[2].bf16() : nullptr,
                    (M > 4 && M <= 4) ? lw.conv_state_slots[3].bf16() : nullptr,
                    lw.A_log.bf16(),
                    lw.dt_bias.bf16(),
                    lw.linear_norm_w.bf16(),
                    lw.ssm_state.bf16(),
                    lw.ssm_state.bf16(),
                    (M > 1 && M <= 4) ? lw.ssm_state_slots[0].bf16() : nullptr,
                    (M > 2 && M <= 4) ? lw.ssm_state_slots[1].bf16() : nullptr,
                    (M > 3 && M <= 4) ? lw.ssm_state_slots[2].bf16() : nullptr,
                    (M > 4 && M <= 4) ? lw.ssm_state_slots[3].bf16() : nullptr,
                    16, 48, 128, M, main_stream_);
            }

            if (prof) {
                cudaStreamSynchronize(main_stream_);
                t3 = std::chrono::high_resolution_clock::now();
            }

            // Output projection: [M, 6144] -> [M, 5120]
            if (lw.linear_out_proj.dtype == "fp4") {
                if (is_blackwell_tc_active_) {
                    quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                   buf_attn_out_batch_.bf16(), M, 6144, main_stream_);
                    gemm_fp4_residual_blackwell_tensorcore_cuda(buf_hidden_batch_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                                (const uint8_t*)lw.linear_out_proj.data, (const uint8_t*)lw.linear_out_proj_scale.data,
                                                                dim, 6144, M, main_stream_);
                } else if (M == 1) {
                    gemv_fp4_residual_cuda(buf_hidden_batch_.bf16(), buf_attn_out_batch_.bf16(),
                                           (const uint8_t*)lw.linear_out_proj.data, (const uint8_t*)lw.linear_out_proj_scale.data,
                                           dim, 6144, main_stream_);
                } else {
                    gemm_fp4_residual_batch_cuda(buf_hidden_batch_.bf16(), buf_attn_out_batch_.bf16(),
                                                 (const uint8_t*)lw.linear_out_proj.data, (const uint8_t*)lw.linear_out_proj_scale.data,
                                                 dim, 6144, M, main_stream_);
                }
            } else {
                if (M <= 8) {
                    matmul_proj_batch(buf_hidden2_batch_, buf_attn_out_batch_, lw.linear_out_proj, lw.linear_out_proj_scale, dim, 6144);
                    vector_add_bf16_cuda(buf_hidden_batch_.bf16(), buf_hidden2_batch_.bf16(), M * dim, main_stream_);
                } else {
                    matmul_proj_batch(buf_hidden_batch_, buf_attn_out_batch_, lw.linear_out_proj, lw.linear_out_proj_scale, dim, 6144, 1.0f);
                }
            }
        } else if (cfg_.architecture == ModelArch::QWEN2) {
            // Full GQA Attention Projections for M tokens (Ungated)
            int q_dim = n_q_heads * head_dim;
            int kv_dim = n_kv_heads * head_dim;
            if (M > 8 && (lw.w_q.dtype == "int3" || lw.w_q.dtype == "int4") && (lw.w_k.dtype == "int3" || lw.w_k.dtype == "int4") && (lw.w_v.dtype == "int3" || lw.w_v.dtype == "int4")) {
                CUDA_CHECK(cudaEventRecord(main_event_, main_stream_));
                CUDA_CHECK(cudaStreamWaitEvent(side_stream_, main_event_, 0));

                if (lw.w_k.dtype == "int3") {
                    gemm_int3_dequant(buf_gate_batch_.bf16(), M, kv_dim, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_k.data, lw.w_k_scale.bf16(), 32,
                                      side_stream_, 0.0f, &buf_dequant_2_, cublas_side_handle_);
                } else {
                    gemm_int4_dequant(buf_gate_batch_.bf16(), M, kv_dim, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_k.data, lw.w_k_scale.bf16(), 32,
                                      side_stream_, 0.0f, &buf_dequant_2_, cublas_side_handle_);
                }
                if (lw.w_v.dtype == "int3") {
                    gemm_int3_dequant(buf_up_batch_.bf16(), M, kv_dim, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_v.data, lw.w_v_scale.bf16(), 32,
                                      side_stream_, 0.0f, &buf_dequant_2_, cublas_side_handle_);
                } else {
                    gemm_int4_dequant(buf_up_batch_.bf16(), M, kv_dim, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_v.data, lw.w_v_scale.bf16(), 32,
                                      side_stream_, 0.0f, &buf_dequant_2_, cublas_side_handle_);
                }
                CUDA_CHECK(cudaEventRecord(side_event_, side_stream_));

                if (lw.w_q.dtype == "int3") {
                    gemm_int3_dequant(buf_q_batch_.bf16(), M, q_dim, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_q.data, lw.w_q_scale.bf16(), 32,
                                      main_stream_, 0.0f, &buf_dequant_, cublas_handle_);
                } else {
                    gemm_int4_dequant(buf_q_batch_.bf16(), M, q_dim, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_q.data, lw.w_q_scale.bf16(), 32,
                                      main_stream_, 0.0f, &buf_dequant_, cublas_handle_);
                }

                CUDA_CHECK(cudaStreamWaitEvent(main_stream_, side_event_, 0));
            } else {
                if (is_blackwell_tc_active_ && lw.w_q.dtype == "fp4") {
                    gemm_fp4_blackwell_tensorcore_cuda(buf_q_batch_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                        (const uint8_t*)lw.w_q.data, (const uint8_t*)lw.w_q_scale.data,
                                                        q_dim, dim, M, main_stream_);
                } else {
                    matmul_proj_batch(buf_q_batch_, buf_hidden2_batch_, lw.w_q, lw.w_q_scale, q_dim, dim);
                }
                if (is_blackwell_tc_active_ && lw.w_k.dtype == "fp4") {
                    gemm_fp4_blackwell_tensorcore_cuda(buf_gate_batch_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                        (const uint8_t*)lw.w_k.data, (const uint8_t*)lw.w_k_scale.data,
                                                        kv_dim, dim, M, main_stream_);
                } else {
                    matmul_proj_batch(buf_gate_batch_, buf_hidden2_batch_, lw.w_k, lw.w_k_scale, kv_dim, dim);
                }
                if (is_blackwell_tc_active_ && lw.w_v.dtype == "fp4") {
                    gemm_fp4_blackwell_tensorcore_cuda(buf_up_batch_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                        (const uint8_t*)lw.w_v.data, (const uint8_t*)lw.w_v_scale.data,
                                                        kv_dim, dim, M, main_stream_);
                } else {
                    matmul_proj_batch(buf_up_batch_, buf_hidden2_batch_, lw.w_v, lw.w_v_scale, kv_dim, dim);
                }
            }

            if (lw.gqa_q_bias.data) {
                add_bias_bf16_cuda(buf_q_batch_.bf16(), lw.gqa_q_bias.bf16(), M, q_dim, main_stream_);
            }
            if (lw.gqa_k_bias.data) {
                add_bias_bf16_cuda(buf_gate_batch_.bf16(), lw.gqa_k_bias.bf16(), M, kv_dim, main_stream_);
            }
            if (lw.gqa_v_bias.data) {
                add_bias_bf16_cuda(buf_up_batch_.bf16(), lw.gqa_v_bias.bf16(), M, kv_dim, main_stream_);
            }

            qwen2_gqa_decode_fp8_batch_cuda(
                buf_attn_out_batch_.bf16(),
                buf_q_batch_.bf16(),
                buf_gate_batch_.bf16(),
                buf_up_batch_.bf16(),
                lw.gqa_q_norm_w.data ? lw.gqa_q_norm_w.bf16() : nullptr,
                lw.gqa_k_norm_w.data ? lw.gqa_k_norm_w.bf16() : nullptr,
                lw.k_cache_gqa.u8(),
                lw.v_cache_gqa.u8(),
                n_q_heads, n_kv_heads, head_dim,
                buf_input_pos_batch_.data ? buf_input_pos_batch_.i32() : nullptr,
                position,
                M,
                cfg_.max_seq_len,
                cfg_.rope_theta, cfg_.rms_norm_eps,
                buf_mrope_pos_.data ? buf_mrope_pos_.i32() : nullptr,
                main_stream_);

            if (lw.w_o.dtype == "fp4") {
                if (is_blackwell_tc_active_) {
                    quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                   buf_attn_out_batch_.bf16(), M, q_dim, main_stream_);
                    gemm_fp4_residual_blackwell_tensorcore_cuda(buf_hidden_batch_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                                (const uint8_t*)lw.w_o.data, (const uint8_t*)lw.w_o_scale.data,
                                                                dim, q_dim, M, main_stream_);
                } else if (M == 1) {
                    gemv_fp4_residual_cuda(buf_hidden_batch_.bf16(), buf_attn_out_batch_.bf16(),
                                           (const uint8_t*)lw.w_o.data, (const uint8_t*)lw.w_o_scale.data,
                                           dim, q_dim, main_stream_);
                } else {
                    gemm_fp4_residual_batch_cuda(buf_hidden_batch_.bf16(), buf_attn_out_batch_.bf16(),
                                                 (const uint8_t*)lw.w_o.data, (const uint8_t*)lw.w_o_scale.data,
                                                 dim, q_dim, M, main_stream_);
                }
            } else {
                if (M <= 8) {
                    matmul_proj_batch(buf_hidden2_batch_, buf_attn_out_batch_, lw.w_o, lw.w_o_scale, dim, q_dim);
                    vector_add_bf16_cuda(buf_hidden_batch_.bf16(), buf_hidden2_batch_.bf16(), M * dim, main_stream_);
                } else {
                    matmul_proj_batch(buf_hidden_batch_, buf_attn_out_batch_, lw.w_o, lw.w_o_scale, dim, q_dim, 1.0f);
                }
            }
        } else {
            // Full GQA Attention Projections for M tokens
            if (M > 8 && (lw.w_q.dtype == "int3" || lw.w_q.dtype == "int4") && (lw.w_k.dtype == "int3" || lw.w_k.dtype == "int4") && (lw.w_v.dtype == "int3" || lw.w_v.dtype == "int4")) {
                int q_dim = 2 * n_q_heads * head_dim;
                int kv_dim = n_kv_heads * head_dim;
                CUDA_CHECK(cudaEventRecord(main_event_, main_stream_));
                CUDA_CHECK(cudaStreamWaitEvent(side_stream_, main_event_, 0));

                if (lw.w_k.dtype == "int3") {
                    gemm_int3_dequant(buf_gate_batch_.bf16(), M, kv_dim, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_k.data, lw.w_k_scale.bf16(), 32,
                                      side_stream_, 0.0f, &buf_dequant_2_, cublas_side_handle_);
                } else {
                    gemm_int4_dequant(buf_gate_batch_.bf16(), M, kv_dim, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_k.data, lw.w_k_scale.bf16(), 32,
                                      side_stream_, 0.0f, &buf_dequant_2_, cublas_side_handle_);
                }
                if (lw.w_v.dtype == "int3") {
                    gemm_int3_dequant(buf_up_batch_.bf16(), M, kv_dim, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_v.data, lw.w_v_scale.bf16(), 32,
                                      side_stream_, 0.0f, &buf_dequant_2_, cublas_side_handle_);
                } else {
                    gemm_int4_dequant(buf_up_batch_.bf16(), M, kv_dim, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_v.data, lw.w_v_scale.bf16(), 32,
                                      side_stream_, 0.0f, &buf_dequant_2_, cublas_side_handle_);
                }
                CUDA_CHECK(cudaEventRecord(side_event_, side_stream_));

                if (lw.w_q.dtype == "int3") {
                    gemm_int3_dequant(buf_q_batch_.bf16(), M, q_dim, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_q.data, lw.w_q_scale.bf16(), 32,
                                      main_stream_, 0.0f, &buf_dequant_, cublas_handle_);
                } else {
                    gemm_int4_dequant(buf_q_batch_.bf16(), M, q_dim, dim, buf_hidden2_batch_.bf16(),
                                      (const uint8_t*)lw.w_q.data, lw.w_q_scale.bf16(), 32,
                                      main_stream_, 0.0f, &buf_dequant_, cublas_handle_);
                }

                CUDA_CHECK(cudaStreamWaitEvent(main_stream_, side_event_, 0));
            } else {
                if (is_blackwell_tc_active_ && lw.w_q.dtype == "fp4") {
                    gemm_fp4_blackwell_tensorcore_cuda(buf_q_batch_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                        (const uint8_t*)lw.w_q.data, (const uint8_t*)lw.w_q_scale.data,
                                                        2 * n_q_heads * head_dim, dim, M, main_stream_);
                } else {
                    matmul_proj_batch(buf_q_batch_, buf_hidden2_batch_, lw.w_q, lw.w_q_scale, 2 * n_q_heads * head_dim, dim);
                }
                if (is_blackwell_tc_active_ && lw.w_k.dtype == "fp4") {
                    gemm_fp4_blackwell_tensorcore_cuda(buf_gate_batch_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                        (const uint8_t*)lw.w_k.data, (const uint8_t*)lw.w_k_scale.data,
                                                        n_kv_heads * head_dim, dim, M, main_stream_);
                } else {
                    matmul_proj_batch(buf_gate_batch_, buf_hidden2_batch_, lw.w_k, lw.w_k_scale, n_kv_heads * head_dim, dim);
                }
                if (is_blackwell_tc_active_ && lw.w_v.dtype == "fp4") {
                    gemm_fp4_blackwell_tensorcore_cuda(buf_up_batch_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                        (const uint8_t*)lw.w_v.data, (const uint8_t*)lw.w_v_scale.data,
                                                        n_kv_heads * head_dim, dim, M, main_stream_);
                } else {
                    matmul_proj_batch(buf_up_batch_, buf_hidden2_batch_, lw.w_v, lw.w_v_scale, n_kv_heads * head_dim, dim);
                }
            }

            if (prof) {
                cudaStreamSynchronize(main_stream_);
                t2 = std::chrono::high_resolution_clock::now();
            }

            if (is_tree) {
                qwen_gqa_decode_tree_fp8_batch_cuda(
                    buf_attn_out_batch_.bf16(),
                    buf_q_batch_.bf16(),
                    buf_gate_batch_.bf16(),
                    buf_up_batch_.bf16(),
                    lw.gqa_q_norm_w.bf16(),
                    lw.gqa_k_norm_w.bf16(),
                    lw.k_cache_gqa.u8(),
                    lw.v_cache_gqa.u8(),
                    n_q_heads, n_kv_heads, head_dim,
                    buf_input_pos_batch_.data ? buf_input_pos_batch_.i32() : nullptr,
                    buf_tree_ancestor_mask_.u64(),
                    position,
                    M,
                    cfg_.max_seq_len,
                    cfg_.rope_theta, cfg_.rms_norm_eps, main_stream_);
            } else {
                qwen_gqa_decode_gated_fp8_batch_cuda(
                    buf_attn_out_batch_.bf16(),
                    buf_q_batch_.bf16(),
                    buf_gate_batch_.bf16(),
                    buf_up_batch_.bf16(),
                    lw.gqa_q_norm_w.bf16(),
                    lw.gqa_k_norm_w.bf16(),
                    lw.k_cache_gqa.u8(),
                    lw.v_cache_gqa.u8(),
                    n_q_heads, n_kv_heads, head_dim,
                    buf_input_pos_batch_.data ? buf_input_pos_batch_.i32() : nullptr,
                    position,
                    M,
                    cfg_.max_seq_len,
                    cfg_.rope_theta, cfg_.rms_norm_eps, main_stream_);
            }

            if (prof) {
                cudaStreamSynchronize(main_stream_);
                t3 = std::chrono::high_resolution_clock::now();
            }

            if (lw.w_o.dtype == "fp4") {
                if (is_blackwell_tc_active_) {
                    quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                   buf_attn_out_batch_.bf16(), M, n_q_heads * head_dim, main_stream_);
                    gemm_fp4_residual_blackwell_tensorcore_cuda(buf_hidden_batch_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                                (const uint8_t*)lw.w_o.data, (const uint8_t*)lw.w_o_scale.data,
                                                                dim, n_q_heads * head_dim, M, main_stream_);
                } else if (M == 1) {
                    gemv_fp4_residual_cuda(buf_hidden_batch_.bf16(), buf_attn_out_batch_.bf16(),
                                           (const uint8_t*)lw.w_o.data, (const uint8_t*)lw.w_o_scale.data,
                                           dim, n_q_heads * head_dim, main_stream_);
                } else {
                    gemm_fp4_residual_batch_cuda(buf_hidden_batch_.bf16(), buf_attn_out_batch_.bf16(),
                                                 (const uint8_t*)lw.w_o.data, (const uint8_t*)lw.w_o_scale.data,
                                                 dim, n_q_heads * head_dim, M, main_stream_);
                }
            } else {
                if (M <= 8) {
                    matmul_proj_batch(buf_hidden2_batch_, buf_attn_out_batch_, lw.w_o, lw.w_o_scale, dim, n_q_heads * head_dim);
                    vector_add_bf16_cuda(buf_hidden_batch_.bf16(), buf_hidden2_batch_.bf16(), M * dim, main_stream_);
                } else {
                    matmul_proj_batch(buf_hidden_batch_, buf_attn_out_batch_, lw.w_o, lw.w_o_scale, dim, n_q_heads * head_dim, 1.0f);
                }
            }
        }

        if (prof) {
            cudaStreamSynchronize(main_stream_);
            t4 = std::chrono::high_resolution_clock::now();
        }

        if (cfg_.architecture == ModelArch::QWEN2) {
            rms_norm_cuda_batched(buf_hidden2_batch_.bf16(), buf_hidden_batch_.bf16(),
                                  lw.ffn_norm_w.bf16(), M, dim, cfg_.rms_norm_eps, main_stream_);
        } else {
            rms_norm_one_centered_cuda_batched(buf_hidden2_batch_.bf16(), buf_hidden_batch_.bf16(),
                                               lw.ffn_norm_w.bf16(), M, dim, cfg_.rms_norm_eps, main_stream_);
        }

        // FFN SwiGLU for M tokens simultaneously:
        if (lw.w_gate.dtype == "fp4") {
            if (is_blackwell_tc_active_) {
                quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                               buf_hidden2_batch_.bf16(), M, dim, main_stream_);
                gemm_fp4_swiglu_blackwell_tensorcore_cuda(buf_gate_batch_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                          (const uint8_t*)lw.w_gate.data, (const uint8_t*)lw.w_gate_scale.data,
                                                          (const uint8_t*)lw.w_up.data, (const uint8_t*)lw.w_up_scale.data,
                                                          inter_size, dim, M, cfg_.swiglu_limit, main_stream_);
            } else if (M == 1) {
                gemv_fp4_swiglu_fused_cuda(buf_gate_batch_.bf16(), buf_hidden2_batch_.bf16(),
                                           (const uint8_t*)lw.w_gate.data, (const uint8_t*)lw.w_gate_scale.data,
                                           (const uint8_t*)lw.w_up.data, (const uint8_t*)lw.w_up_scale.data,
                                           inter_size, dim, cfg_.swiglu_limit, main_stream_);
            } else {
                gemm_fp4_swiglu_fused_batch_cuda(buf_gate_batch_.bf16(), buf_hidden2_batch_.bf16(),
                                                  (const uint8_t*)lw.w_gate.data, (const uint8_t*)lw.w_gate_scale.data,
                                                  (const uint8_t*)lw.w_up.data, (const uint8_t*)lw.w_up_scale.data,
                                                  inter_size, dim, M, cfg_.swiglu_limit, main_stream_);
            }
        } else if (lw.w_gate.dtype == "int4" && M <= 8) {
            gemm_int4_swiglu_fused_batch_cuda(buf_gate_batch_.bf16(), buf_hidden2_batch_.bf16(),
                                              (const uint8_t*)lw.w_gate.data, lw.w_gate_scale.bf16(),
                                              (const uint8_t*)lw.w_up.data, lw.w_up_scale.bf16(),
                                              inter_size, dim, M, cfg_.swiglu_limit, main_stream_);
        } else if (lw.w_gate.dtype == "int3" && M <= 8) {
            gemm_int3_swiglu_fused_batch_cuda(buf_gate_batch_.bf16(), buf_hidden2_batch_.bf16(),
                                              (const uint8_t*)lw.w_gate.data, lw.w_gate_scale.bf16(),
                                              (const uint8_t*)lw.w_up.data, lw.w_up_scale.bf16(),
                                              inter_size, dim, M, cfg_.swiglu_limit, main_stream_);
        } else if (M > 8 && lw.w_gate.dtype == "int3" && lw.w_up.dtype == "int3") {
            // High-occupancy double-buffered pipelined MLP projections across buf_dequant_ & buf_dequant_2_ (48 MB tiles, 136 cuBLAS blocks)
            gemm_int3_dequant(buf_gate_batch_.bf16(), M, inter_size, dim, buf_hidden2_batch_.bf16(),
                              (const uint8_t*)lw.w_gate.data, lw.w_gate_scale.bf16(), 32,
                              main_stream_, 0.0f, &buf_dequant_, &buf_dequant_2_, cublas_handle_);

            gemm_int3_dequant(buf_up_batch_.bf16(), M, inter_size, dim, buf_hidden2_batch_.bf16(),
                              (const uint8_t*)lw.w_up.data, lw.w_up_scale.bf16(), 32,
                              main_stream_, 0.0f, &buf_dequant_, &buf_dequant_2_, cublas_handle_);

            if (prof) {
                cudaStreamSynchronize(main_stream_);
                t5 = std::chrono::high_resolution_clock::now();
            }
            silu_mul_cuda(buf_gate_batch_.bf16(), buf_gate_batch_.bf16(),
                          buf_up_batch_.bf16(), M * inter_size, cfg_.swiglu_limit, main_stream_);
        } else if (M > 8 && lw.w_gate.dtype == "int4" && lw.w_up.dtype == "int4") {
            gemm_int4_dequant(buf_gate_batch_.bf16(), M, inter_size, dim, buf_hidden2_batch_.bf16(),
                              (const uint8_t*)lw.w_gate.data, lw.w_gate_scale.bf16(), 32,
                              main_stream_, 0.0f, &buf_dequant_, &buf_dequant_2_, cublas_handle_);

            gemm_int4_dequant(buf_up_batch_.bf16(), M, inter_size, dim, buf_hidden2_batch_.bf16(),
                              (const uint8_t*)lw.w_up.data, lw.w_up_scale.bf16(), 32,
                              main_stream_, 0.0f, &buf_dequant_, &buf_dequant_2_, cublas_handle_);

            if (prof) {
                cudaStreamSynchronize(main_stream_);
                t5 = std::chrono::high_resolution_clock::now();
            }
            silu_mul_cuda(buf_gate_batch_.bf16(), buf_gate_batch_.bf16(),
                          buf_up_batch_.bf16(), M * inter_size, cfg_.swiglu_limit, main_stream_);
        } else {
            matmul_proj_batch(buf_gate_batch_, buf_hidden2_batch_, lw.w_gate, lw.w_gate_scale, inter_size, dim);
            matmul_proj_batch(buf_up_batch_, buf_hidden2_batch_, lw.w_up, lw.w_up_scale, inter_size, dim);
            if (prof) {
                cudaStreamSynchronize(main_stream_);
                t5 = std::chrono::high_resolution_clock::now();
            }
            silu_mul_cuda(buf_gate_batch_.bf16(), buf_gate_batch_.bf16(),
                          buf_up_batch_.bf16(), M * inter_size, cfg_.swiglu_limit, main_stream_);
        }

        if (prof) {
            cudaStreamSynchronize(main_stream_);
            t6 = std::chrono::high_resolution_clock::now();
        }

        // Down projection: [M, inter_size] -> [M, dim]
        if (lw.w_down.dtype == "fp4") {
            if (is_blackwell_tc_active_) {
                quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                               buf_gate_batch_.bf16(), M, inter_size, main_stream_);
                gemm_fp4_residual_blackwell_tensorcore_cuda(buf_hidden_batch_.bf16(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                            (const uint8_t*)lw.w_down.data, (const uint8_t*)lw.w_down_scale.data,
                                                            dim, inter_size, M, main_stream_);
            } else if (M == 1) {
                gemv_fp4_residual_cuda(buf_hidden_batch_.bf16(), buf_gate_batch_.bf16(),
                                       (const uint8_t*)lw.w_down.data, (const uint8_t*)lw.w_down_scale.data,
                                       dim, inter_size, main_stream_);
            } else {
                gemm_fp4_residual_batch_cuda(buf_hidden_batch_.bf16(), buf_gate_batch_.bf16(),
                                             (const uint8_t*)lw.w_down.data, (const uint8_t*)lw.w_down_scale.data,
                                             dim, inter_size, M, main_stream_);
            }
        } else {
            if (M <= 8) {
                matmul_proj_batch(buf_hidden2_batch_, buf_gate_batch_, lw.w_down, lw.w_down_scale, dim, inter_size);
                vector_add_bf16_cuda(buf_hidden_batch_.bf16(), buf_hidden2_batch_.bf16(), M * dim, main_stream_);
            } else {
                matmul_proj_batch(buf_hidden_batch_, buf_gate_batch_, lw.w_down, lw.w_down_scale, dim, inter_size, 1.0f);
            }
        }

        if (prof) {
            cudaStreamSynchronize(main_stream_);
            t7 = std::chrono::high_resolution_clock::now();
            s_layer_prof_done[layer_id] = true;
            float ms_norm = std::chrono::duration<float, std::milli>(t1 - t0).count();
            float ms_ain  = std::chrono::duration<float, std::milli>(t2 - t1).count();
            float ms_core = std::chrono::duration<float, std::milli>(t3 - t2).count();
            float ms_aout = std::chrono::duration<float, std::milli>(t4 - t3).count();
            float ms_min  = std::chrono::duration<float, std::milli>(t5 - t4).count();
            float ms_act  = std::chrono::duration<float, std::milli>(t6 - t5).count();
            float ms_down = std::chrono::duration<float, std::milli>(t7 - t6).count();
            float ms_tot  = std::chrono::duration<float, std::milli>(t7 - t0).count();
            LOG_INFO("[LAYER PROF L%d %s M=%d] total: %.2f ms | norm: %.2f | attn_in: %.2f | attn_core: %.2f | attn_out: %.2f | mlp_in: %.2f | swiglu: %.2f | mlp_down: %.2f",
                     layer_id, lw.is_linear_attn ? "DeltaNet" : "GQA", M, ms_tot, ms_norm, ms_ain, ms_core, ms_aout, ms_min, ms_act, ms_down);
        }
    }

    bool batch_graph_captured_[9] = {false};
    cudaGraph_t batch_graph_[9] = {nullptr};
    cudaGraphExec_t batch_graph_exec_[9] = {nullptr};

    void forward_token_batch_qwen_device_body(int position, int M, bool compute_logits = true, bool is_tree = false) {
        int dim = cfg_.hidden_size;
        // 1. Embedding lookup for M tokens
        if (embed_weight_.dtype == "fp4") {
            embedding_fp4_cuda(buf_hidden_batch_.bf16(), (const uint8_t*)embed_weight_.data, (const uint8_t*)embed_weight_scale_.data, buf_input_tokens_batch_.i32(), M, dim, main_stream_);
        } else if (embed_weight_.dtype == "int4") {
            embedding_int4_cuda(buf_hidden_batch_.bf16(), (uint8_t*)embed_weight_.data, embed_weight_scale_.bf16(), buf_input_tokens_batch_.i32(), M, dim, main_stream_);
        } else {
            embedding_cuda(buf_hidden_batch_.bf16(), embed_weight_.bf16(), buf_input_tokens_batch_.i32(), M, dim, main_stream_);
        }

        // Override with vision tower embedding for image tokens
        if (active_visual_embeddings_) {
            for (int m = 0; m < M; m++) {
                int pos = position + m;
                if (pos >= visual_embed_start_pos_ && pos < visual_embed_start_pos_ + visual_embed_num_tokens_) {
                    int v_idx = pos - visual_embed_start_pos_;
                    CUDA_CHECK(cudaMemcpyAsync(buf_hidden_batch_.bf16() + (size_t)m * dim,
                                               active_visual_embeddings_ + (size_t)v_idx * dim,
                                               dim * sizeof(__nv_bfloat16),
                                               cudaMemcpyDeviceToDevice, main_stream_));
                }
            }
        }

        // 2. Process each layer
        static int s_prof_pass = 0;
        bool do_prof = (M > 8 && s_prof_pass++ == 1);
        if (do_prof) {
            cudaStreamSynchronize(main_stream_);
            auto t_start = std::chrono::high_resolution_clock::now();
            float t_delta_net = 0.0f;
            float t_gqa = 0.0f;
            int n_delta_net = 0;
            int n_gqa = 0;
            for (int layer = 0; layer < cfg_.num_hidden_layers; layer++) {
                auto tl0 = std::chrono::high_resolution_clock::now();
                forward_layer_qwen_batch(layer, position, M, is_tree);
                cudaStreamSynchronize(main_stream_);
                auto tl1 = std::chrono::high_resolution_clock::now();
                float ms = std::chrono::duration<float, std::milli>(tl1 - tl0).count();
                if (layers_[layer].is_linear_attn) {
                    t_delta_net += ms;
                    n_delta_net++;
                } else {
                    t_gqa += ms;
                    n_gqa++;
                }
            }
            auto t_end = std::chrono::high_resolution_clock::now();
            float total_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
            LOG_INFO("[PREFILL PROFILE M=%d] Total 64 layers: %.2f ms (%.1f tok/s) | DeltaNet (%d layers): avg %.2f ms (sum %.1f ms) | GQA (%d layers): avg %.2f ms (sum %.1f ms)",
                     M, total_ms, M / (total_ms / 1000.0f), n_delta_net, t_delta_net / std::max(1, n_delta_net), t_delta_net, n_gqa, t_gqa / std::max(1, n_gqa), t_gqa);
        } else {
            for (int layer = 0; layer < cfg_.num_hidden_layers; layer++) {
                forward_layer_qwen_batch(layer, position, M, is_tree);
            }
        }

        if (!compute_logits) return;

        // 3. Final norm
        if (cfg_.architecture == ModelArch::QWEN2) {
            rms_norm_cuda_batched(buf_hidden_batch_.bf16(), buf_hidden_batch_.bf16(),
                                  norm_weight_.bf16(), M, dim, cfg_.rms_norm_eps, main_stream_);
        } else {
            rms_norm_one_centered_cuda_batched(buf_hidden_batch_.bf16(), buf_hidden_batch_.bf16(),
                                               norm_weight_.bf16(), M, dim, cfg_.rms_norm_eps, main_stream_);
        }

        // 4. Logits: hidden @ head_weight.T -> [M, vocab_size] in F32
        if (head_weight_.dtype == "fp4") {
            if (is_blackwell_tc_active_) {
                quantize_bf16_to_fp4_e2m1_cuda(buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                               buf_hidden_batch_.bf16(), M, dim, main_stream_);
                gemm_fp4_f32_blackwell_tensorcore_cuda(buf_logits_batch_.f32(), buf_activation_fp4_.u8(), buf_activation_scale_.u8(),
                                                       (const uint8_t*)head_weight_.data, (const uint8_t*)head_weight_scale_.data,
                                                       cfg_.vocab_size, dim, M, main_stream_);
            } else {
                gemm_fp4_f32_batch_cuda(buf_logits_batch_.f32(), buf_hidden_batch_.bf16(),
                                         (const uint8_t*)head_weight_.data, (const uint8_t*)head_weight_scale_.data,
                                         cfg_.vocab_size, dim, M, main_stream_);
            }
        } else if (head_weight_scale_.data) {
            gemm_int4_f32_batch_cuda(buf_logits_batch_.f32(), buf_hidden_batch_.bf16(),
                                     (const uint8_t*)head_weight_.data, head_weight_scale_.bf16(),
                                     cfg_.vocab_size, dim, M, main_stream_);
        } else {
            gemm_bf16_f32(buf_logits_batch_.f32(), M, cfg_.vocab_size, dim,
                          buf_hidden_batch_.bf16(), head_weight_.bf16());
        }
    }

    void forward_token_batch_qwen(const int* tokens_host, int position, int M) {
        if (!h_target_batch_tok_) {
            CUDA_CHECK(cudaMallocHost(&h_target_batch_tok_, 64 * sizeof(int32_t)));
            CUDA_CHECK(cudaMallocHost(&h_target_batch_pos_, 64 * sizeof(int32_t)));
        }
        for (int m = 0; m < M; m++) {
            h_target_batch_pos_[m] = position + m;
            h_target_batch_tok_[m] = tokens_host[m];
        }
        CUDA_CHECK(cudaMemcpyAsync(buf_input_pos_batch_.i32(), h_target_batch_pos_, M * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
        CUDA_CHECK(cudaMemcpyAsync(buf_input_tokens_batch_.i32(), h_target_batch_tok_, M * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));

        if (M >= 2 && M <= 8 && batch_graph_captured_[M]) {
            CUDA_CHECK(cudaGraphLaunch(batch_graph_exec_[M], main_stream_));
        } else {
            forward_token_batch_qwen_device_body(position, M);
        }
    }


    void forward_token_tree_qwen(const TreeTopology& tree, int position) {
        int M = tree.num_nodes;
        if (!h_target_batch_tok_) {
            CUDA_CHECK(cudaMallocHost(&h_target_batch_tok_, 64 * sizeof(int32_t)));
            CUDA_CHECK(cudaMallocHost(&h_target_batch_pos_, 64 * sizeof(int32_t)));
            CUDA_CHECK(cudaMallocHost(&h_tree_ancestor_mask_, 64 * sizeof(uint64_t)));
            CUDA_CHECK(cudaMallocHost(&h_tree_parent_, 64 * sizeof(int32_t)));
        }
        for (int m = 0; m < M; m++) {
            h_target_batch_pos_[m] = position + tree.depth[m];
            h_target_batch_tok_[m] = tree.tokens[m];
            h_tree_ancestor_mask_[m] = tree.ancestor_mask[m];
            h_tree_parent_[m] = tree.parent[m];
        }
        CUDA_CHECK(cudaMemcpyAsync(buf_input_pos_batch_.i32(), h_target_batch_pos_, M * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
        CUDA_CHECK(cudaMemcpyAsync(buf_input_tokens_batch_.i32(), h_target_batch_tok_, M * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
        CUDA_CHECK(cudaMemcpyAsync(buf_tree_ancestor_mask_.u64(), h_tree_ancestor_mask_, M * sizeof(uint64_t), cudaMemcpyHostToDevice, main_stream_));
        CUDA_CHECK(cudaMemcpyAsync(buf_tree_parent_indices_.i32(), h_tree_parent_, M * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));

        if (M == tree_graph_M_ && tree_graph_captured_) {
            CUDA_CHECK(cudaGraphLaunch(tree_graph_exec_, main_stream_));
        } else {
            forward_token_batch_qwen_device_body(position, M, /*compute_logits=*/true, /*is_tree=*/true);
        }
    }

    void init_batch_cuda_graphs() {
#if defined(__APPLE__)
        for (int m = 0; m <= 8; ++m) batch_graph_captured_[m] = false;
        return;
#endif
        if (cfg_.architecture != ModelArch::QWEN) return;
        std::vector<int> target_batch_sizes;
        if (gpu_caps_.is_vram_constrained) {
            int max_k = (mtp_k_ > 0) ? mtp_k_ : 1;
            int max_m = std::min(max_k + 1, 5);
            for (int m = 2; m <= max_m; ++m) {
                target_batch_sizes.push_back(m);
            }
        } else {
            int max_k = (mtp_k_ > 0) ? mtp_k_ : 4;
            int max_m = std::min(max_k + 1, 8);
            for (int m = 2; m <= max_m; ++m) {
                target_batch_sizes.push_back(m);
            }
        }
        for (int M : target_batch_sizes) {
            LOG_INFO("Warming up and capturing Batched Target CUDA Graph (M=%d)...", M);

            std::vector<int32_t> dummy_toks(M, 0);
            std::vector<int32_t> dummy_pos(M, 0);
            CUDA_CHECK(cudaMemcpy(buf_input_tokens_batch_.i32(), dummy_toks.data(), M * sizeof(int32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(buf_input_pos_batch_.i32(), dummy_pos.data(), M * sizeof(int32_t), cudaMemcpyHostToDevice));

            // Warmup
            for (int i = 0; i < 3; i++) {
                forward_token_batch_qwen_device_body(0, M);
            }
            CUDA_CHECK(cudaStreamSynchronize(main_stream_));

            cudaError_t cap_err = cudaStreamBeginCapture(main_stream_, cudaStreamCaptureModeGlobal);
            if (cap_err != cudaSuccess) {
                batch_graph_captured_[M] = false;
                continue;
            }

            forward_token_batch_qwen_device_body(0, M);

            cudaError_t end_err = cudaStreamEndCapture(main_stream_, &batch_graph_[M]);
            if (end_err != cudaSuccess) {
                batch_graph_captured_[M] = false;
                continue;
            }

            cudaError_t inst_err = cudaGraphInstantiate(&batch_graph_exec_[M], batch_graph_[M], nullptr, nullptr, 0);
            if (inst_err != cudaSuccess) {
                batch_graph_captured_[M] = false;
                continue;
            }

            batch_graph_captured_[M] = true;
            LOG_INFO("Batched Target CUDA Graph (M=%d) instantiated successfully!", M);
        }

        if (enable_tree_mtp_) {
            int M = std::max(2, std::min(4, tree_nodes_ + 1));
            tree_graph_M_ = M;
            LOG_INFO("Warming up and capturing Tree Attention CUDA Graph (nodes=%d, M=%d)...", tree_nodes_, M);
            std::vector<int32_t> dummy_toks(M, 0);
            std::vector<int32_t> dummy_pos(M, 0);
            std::vector<uint64_t> dummy_ancestors(M, 0ULL);
            std::vector<int32_t> dummy_parents(M, 0);

            for (int m = 0; m < M; m++) {
                dummy_parents[m] = (m == 0) ? -1 : (m == 3 ? 1 : 0);
                if (m == 0) dummy_ancestors[m] = 1ULL << 0;
                else if (m == 1) dummy_ancestors[m] = (1ULL << 0) | (1ULL << 1);
                else if (m == 2) dummy_ancestors[m] = (1ULL << 0) | (1ULL << 2);
                else if (m == 3) dummy_ancestors[m] = (1ULL << 0) | (1ULL << 1) | (1ULL << 3);
            }

            CUDA_CHECK(cudaMemcpy(buf_input_tokens_batch_.i32(), dummy_toks.data(), M * sizeof(int32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(buf_input_pos_batch_.i32(), dummy_pos.data(), M * sizeof(int32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(buf_tree_ancestor_mask_.u64(), dummy_ancestors.data(), M * sizeof(uint64_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(buf_tree_parent_indices_.i32(), dummy_parents.data(), M * sizeof(int32_t), cudaMemcpyHostToDevice));

            // Warmup
            for (int i = 0; i < 3; i++) {
                forward_token_batch_qwen_device_body(0, M, /*compute_logits=*/true, /*is_tree=*/true);
            }
            CUDA_CHECK(cudaStreamSynchronize(main_stream_));

            cudaError_t cap_err = cudaStreamBeginCapture(main_stream_, cudaStreamCaptureModeGlobal);
            if (cap_err != cudaSuccess) {
                tree_graph_captured_ = false;
                LOG_WARN("cudaStreamBeginCapture failed for tree graph: %s", cudaGetErrorString(cap_err));
            } else {
                forward_token_batch_qwen_device_body(0, M, /*compute_logits=*/true, /*is_tree=*/true);

                cudaError_t end_err = cudaStreamEndCapture(main_stream_, &tree_graph_);
                if (end_err != cudaSuccess) {
                    tree_graph_captured_ = false;
                    LOG_WARN("cudaStreamEndCapture failed for tree graph: %s", cudaGetErrorString(end_err));
                } else {
                    cudaError_t inst_err = cudaGraphInstantiate(&tree_graph_exec_, tree_graph_, nullptr, nullptr, 0);
                    if (inst_err != cudaSuccess) {
                        tree_graph_captured_ = false;
                        LOG_WARN("cudaGraphInstantiate failed for tree graph: %s", cudaGetErrorString(inst_err));
                    } else {
                        tree_graph_captured_ = true;
                        LOG_INFO("Tree Attention CUDA Graph (nodes=%d, M=%d) instantiated successfully!", tree_nodes_, M);
                    }
                }
            }
        }
    }

    inline void checkpoint_target_state() {
        if (target_ssm_pool_.data && target_ssm_slots_[0].data) {
            CUDA_CHECK(cudaMemcpyAsync(target_ssm_slots_[0].data, target_ssm_pool_.data, target_ssm_pool_.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
            CUDA_CHECK(cudaMemcpyAsync(target_conv_slots_[0].data, target_conv_pool_.data, target_conv_pool_.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
        }
    }

    inline void restore_target_state() {
        if (target_ssm_pool_.data && target_ssm_slots_[0].data) {
            CUDA_CHECK(cudaMemcpyAsync(target_ssm_pool_.data, target_ssm_slots_[0].data, target_ssm_pool_.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
            CUDA_CHECK(cudaMemcpyAsync(target_conv_pool_.data, target_conv_slots_[0].data, target_conv_pool_.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
        }
    }

    inline void commit_target_state_slot(int slot_idx) {
        if (slot_idx < 0 || slot_idx >= 8) return;
        if (!enable_tree_mtp_ && slot_idx >= 4) return;
        if (target_ssm_pool_.data && target_ssm_slots_[slot_idx].data) {
            CUDA_CHECK(cudaMemcpyAsync(target_ssm_pool_.data, target_ssm_slots_[slot_idx].data, target_ssm_pool_.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
            CUDA_CHECK(cudaMemcpyAsync(target_conv_pool_.data, target_conv_slots_[slot_idx].data, target_conv_pool_.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
        }
    }

    inline void compact_tree_kv_all_gqa_layers(const int* winning_slots, int pos_base, int L) {
        if (L <= 0 || !winning_slots) return;
        bool needs_compact = false;
        for (int k = 0; k < L; k++) {
            if (winning_slots[k] != (k + 1)) {
                needs_compact = true;
                break;
            }
        }
        if (!needs_compact) return;

        CUDA_CHECK(cudaMemcpyAsync(buf_tree_winning_slots_.i32(), winning_slots, L * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
        int n_kv_heads = cfg_.num_key_value_heads;
        int head_dim = cfg_.head_dim;
        for (int layer = 0; layer < cfg_.num_hidden_layers; layer++) {
            auto& lw = layers_[layer];
            if (!lw.is_linear_attn) {
                compact_tree_kv_cache_cuda(
                    lw.k_cache_gqa.u8(),
                    lw.v_cache_gqa.u8(),
                    buf_tree_winning_slots_.i32(),
                    pos_base,
                    L,
                    n_kv_heads,
                    head_dim,
                    main_stream_);
            }
        }
    }

    // ── Forward one layer ───────────────────────────────────────────────────

    void forward_layer(int layer_id, int token_id, int position) {
        if (cfg_.is_qwen()) {
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
                my_gate, buf_hidden_.bf16(),
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
            gemm_fp4_dequant(my_gate, 1, moe_inter, dim, buf_hidden_.bf16(), w1_data, w1_scale, w1_scale_cols, stream);
            gemm_fp4_dequant(my_up,   1, moe_inter, dim, buf_hidden_.bf16(), w3_data, w3_scale, w1_scale_cols, stream);
            silu_mul_cuda(my_gate, my_gate, my_up, moe_inter, cfg_.swiglu_limit, stream);
            gemm_fp4_dequant(my_down, 1, dim, moe_inter, my_gate, w2_data, w2_scale, w2_scale_cols, stream);
        }
    }

    void compute_logits() {
        int dim = cfg_.hidden_size;
        int vocab = cfg_.vocab_size;

        if (head_weight_.dtype == "fp4") {
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

                    // 1. RoPE and store KV into ring-buffer
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
                hc_head_reduce();
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

    // ── Sample from logits ──────────────────────────────────────────────────

    float current_rep_penalty_ = 1.0f;  // Set per-request by generate()

    int sample_token(float temperature, const std::vector<int>& history, int step = 0, bool is_reasoning = true,
                     int top_k = 1024, float top_p = 0.95f, float min_p = 0.0f) {
        return sample_from_logits_ptr(buf_logits_.f32(), temperature, history, step, is_reasoning, top_k, top_p, min_p);
    }

    int sample_from_logits_ptr(float* logits_ptr, float temperature, const std::vector<int>& history, int step = 0, bool is_reasoning = true,
                               int top_k = 1024, float top_p = 0.95f, float min_p = 0.0f) {
        int vocab = cfg_.vocab_size;
        static int32_t* h_sample_pin = nullptr;
        if (!h_sample_pin) {
            CUDA_CHECK(cudaMallocHost(&h_sample_pin, sizeof(int32_t)));
        }

        if (current_rep_penalty_ > 1.0f && !history.empty()) {
            static int32_t* d_rep_hist = nullptr;
            static const int MAX_REP_HIST = 256;
            if (!d_rep_hist) {
                CUDA_CHECK(cudaMalloc(&d_rep_hist, MAX_REP_HIST * sizeof(int32_t)));
            }
            int hist_len = (int)history.size();
            int n_penalize = std::min(hist_len, MAX_REP_HIST);
            const int* src_ptr = history.data() + (hist_len - n_penalize);
            CUDA_CHECK(cudaMemcpyAsync(d_rep_hist, src_ptr, n_penalize * sizeof(int32_t), cudaMemcpyHostToDevice, main_stream_));
            apply_repetition_penalty_cuda(logits_ptr, d_rep_hist, n_penalize, current_rep_penalty_, main_stream_);
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
#if defined(__APPLE__)
        sample_multinomial_f32_cuda(buf_argmax_out_.i32(), logits_ptr, vocab, temperature, r, min_p, main_stream_, top_k, top_p);
#else
        sample_multinomial_f32_cuda(buf_argmax_out_.i32(), logits_ptr, vocab, temperature, r, min_p, main_stream_);
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


static std::string get_message_content_string(const json& msg) {
    if (!msg.is_object() || !msg.contains("content")) return "";
    const auto& c = msg["content"];
    if (c.is_string()) {
        return c.get<std::string>();
    }
    if (c.is_array()) {
        std::string res;
        for (const auto& part : c) {
            if (part.is_string()) {
                res += part.get<std::string>();
            } else if (part.is_object()) {
                std::string ptype = part.value("type", "text");
                if (ptype == "text") {
                    res += part.value("text", "");
                }
            }
        }
        return res;
    }
    return "";
}

static bool message_has_image(const json& msg) {
    if (!msg.is_object()) return false;
    if (msg.contains("image") && !msg["image"].is_null()) return true;
    if (msg.contains("image_url") && !msg["image_url"].is_null()) return true;
    if (msg.contains("content") && msg["content"].is_array()) {
        for (const auto& part : msg["content"]) {
            if (part.is_object()) {
                std::string ptype = part.value("type", "");
                if (ptype == "image" || ptype == "image_url" || part.contains("image") || part.contains("image_url")) {
                    return true;
                }
            }
        }
    }
    return false;
}

static std::string extract_image_b64_from_message(const json& msg) {
    if (!msg.is_object()) return "";
    auto extract_from_url_str = [](const std::string& url) -> std::string {
        if (url.empty()) return "";
        size_t b64_pos = url.find("base64,");
        if (b64_pos != std::string::npos) {
            return url.substr(b64_pos + 7);
        }
        return url;
    };

    if (msg.contains("image")) {
        if (msg["image"].is_string()) {
            return extract_from_url_str(msg["image"].get<std::string>());
        } else if (msg["image"].is_object()) {
            return extract_from_url_str(msg["image"].value("url", ""));
        }
    }
    if (msg.contains("image_url")) {
        if (msg["image_url"].is_string()) {
            return extract_from_url_str(msg["image_url"].get<std::string>());
        } else if (msg["image_url"].is_object()) {
            return extract_from_url_str(msg["image_url"].value("url", ""));
        }
    }
    if (msg.contains("content") && msg["content"].is_array()) {
        for (const auto& part : msg["content"]) {
            if (part.is_object()) {
                std::string ptype = part.value("type", "");
                if (ptype == "image" || ptype == "image_url" || part.contains("image") || part.contains("image_url")) {
                    if (part.contains("image_url")) {
                        if (part["image_url"].is_string()) {
                            return extract_from_url_str(part["image_url"].get<std::string>());
                        } else if (part["image_url"].is_object()) {
                            return extract_from_url_str(part["image_url"].value("url", ""));
                        }
                    }
                    if (part.contains("image")) {
                        if (part["image"].is_string()) {
                            return extract_from_url_str(part["image"].get<std::string>());
                        } else if (part["image"].is_object()) {
                            return extract_from_url_str(part["image"].value("url", ""));
                        }
                    }
                    if (part.contains("url") && part["url"].is_string()) {
                        return extract_from_url_str(part["url"].get<std::string>());
                    }
                }
            }
        }
    }
    return "";
}

struct MultimodalPrompt {
    std::vector<int> prompt;
    std::string image_b64;
    int visual_start_pos = -1;
    int visual_num_tokens = 0;
};

static MultimodalPrompt apply_chat_template_multimodal(
    const json& messages,
    const BPETokenizer& tok,
    bool enable_thinking = true,
    const std::string& reasoning_effort = "high",
    const json& tools = json(),
    bool model_has_vision = false)
{
    MultimodalPrompt out;
    json resolved_tools = tools;
    if (!resolved_tools.empty()) {
        resolved_tools = resolve_canonical_tools(resolved_tools);
    }
    bool has_tools = (!resolved_tools.empty() && resolved_tools.is_array());
    int IM_START = tok.get_token_id("<|im_start|>");
    int IM_END = tok.get_token_id("<|im_end|>");
    bool is_qwen = (IM_START >= 0 && IM_END >= 0);
    std::string tools_system_prompt = has_tools ? build_dynamic_tools_prompt(resolved_tools, is_qwen) : "";
    has_tools = !tools_system_prompt.empty();

    if (IM_START >= 0 && IM_END >= 0) {
        // ChatML template (Qwen / Llama / SmolLM)
        std::vector<int> result;
        bool has_system = (!messages.empty() && messages[0].value("role", "") == "system");
        if (!has_system && (enable_thinking || has_tools)) {
            std::string sys_prompt = "You are a helpful assistant";
            if (has_tools) {
                sys_prompt += tools_system_prompt;
            }
            result.push_back(IM_START);
            auto sys_role = tok.encode("system\n" + sys_prompt);
            result.insert(result.end(), sys_role.begin(), sys_role.end());
            result.push_back(IM_END);
            auto nl_enc = tok.encode("\n");
            result.insert(result.end(), nl_enc.begin(), nl_enc.end());
        }
        for (size_t i = 0; i < messages.size(); i++) {
            std::string role = messages[i].value("role", "user");
            std::string content_str = "";
            std::string extracted_image_b64 = "";

            if (messages[i].contains("content")) {
                const auto& c = messages[i]["content"];
                if (c.is_string()) {
                    content_str = c.get<std::string>();
                } else if (c.is_array()) {
                    for (const auto& part : c) {
                        if (part.is_object()) {
                            std::string ptype = part.value("type", "text");
                            if (ptype == "text") {
                                content_str += part.value("text", "");
                            } else if ((ptype == "image_url" || ptype == "image") || part.contains("image_url") || part.contains("image")) {
                                std::string url;
                                if (part.contains("image_url")) {
                                    if (part["image_url"].is_string()) {
                                        url = part["image_url"].get<std::string>();
                                    } else if (part["image_url"].is_object()) {
                                        url = part["image_url"].value("url", "");
                                    }
                                } else if (part.contains("image")) {
                                    if (part["image"].is_string()) {
                                        url = part["image"].get<std::string>();
                                    } else if (part["image"].is_object()) {
                                        url = part["image"].value("url", "");
                                    }
                                }
                                if (!url.empty()) {
                                    size_t b64_pos = url.find("base64,");
                                    if (b64_pos != std::string::npos) {
                                        extracted_image_b64 = url.substr(b64_pos + 7);
                                    } else {
                                        extracted_image_b64 = url;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (extracted_image_b64.empty() && messages[i].contains("image") && messages[i]["image"].is_string()) {
                std::string url = messages[i]["image"].get<std::string>();
                size_t b64_pos = url.find("base64,");
                extracted_image_b64 = (b64_pos != std::string::npos) ? url.substr(b64_pos + 7) : url;
            }

            if (role == "system" && i == 0) {
                content_str = update_system_prompt_with_tools(content_str, tools_system_prompt, has_tools);
            }

            if (role == "tool" || role == "function") {
                result.push_back(IM_START);
                auto user_role = tok.encode("user\n<tool_response>\n" + content_str + "\n</tool_response>");
                result.insert(result.end(), user_role.begin(), user_role.end());
                result.push_back(IM_END);
                auto nl_enc = tok.encode("\n");
                result.insert(result.end(), nl_enc.begin(), nl_enc.end());
                continue;
            }

            result.push_back(IM_START);
            auto role_enc = tok.encode(role + "\n");
            result.insert(result.end(), role_enc.begin(), role_enc.end());

            // Multimodal image injection for Qwen Vision
            if (model_has_vision && !extracted_image_b64.empty() && role == "user") {
                int VISION_START = tok.get_token_id("<|vision_start|>");
                if (VISION_START < 0) VISION_START = 248053;
                int VISION_END = tok.get_token_id("<|vision_end|>");
                if (VISION_END < 0) VISION_END = 248054;
                int IMAGE_PAD = tok.get_token_id("<|image_pad|>");
                if (IMAGE_PAD < 0) IMAGE_PAD = 248056;

                out.image_b64 = extracted_image_b64;
                result.push_back(VISION_START);
                out.visual_start_pos = (int)result.size();
                out.visual_num_tokens = 576;
                result.insert(result.end(), 576, IMAGE_PAD);
                result.push_back(VISION_END);
            }

            std::string body = content_str;
            if (model_has_vision && !extracted_image_b64.empty() && role == "user") {
                std::string trimmed = body;
                while (!trimmed.empty() && (trimmed.front() == ' ' || trimmed.front() == '\n' || trimmed.front() == '\r' || trimmed.front() == '\t')) trimmed.erase(trimmed.begin());
                while (!trimmed.empty() && (trimmed.back() == ' ' || trimmed.back() == '\n' || trimmed.back() == '\r' || trimmed.back() == '\t' || trimmed.back() == '.' || trimmed.back() == '?' || trimmed.back() == '!')) trimmed.pop_back();
                std::string lower = trimmed;
                for (char& c : lower) c = (char)std::tolower((unsigned char)c);

                if (lower.empty() ||
                    lower == "describe this image" || lower == "describe the image" || lower == "describe image" ||
                    lower == "describe this picture" || lower == "describe picture" || lower == "describe" ||
                    lower == "what is this" || lower == "what is this image" || lower == "what is this picture" ||
                    lower == "what's this" || lower == "what do you see" || lower == "what do you see in this image" ||
                    lower == "what do you see in this picture" || lower == "analyze this image" || lower == "analyze image" ||
                    lower == "describe what is depicted in this photo" || lower == "describe what you see in this image") {
                    body = "What is shown in this image?";
                }
            }
            if (role == "assistant") {
                if (messages[i].contains("reasoning_content") && messages[i]["reasoning_content"].is_string()) {
                    std::string r_content = messages[i]["reasoning_content"].get<std::string>();
                    if (!r_content.empty()) {
                        body = "<think>\n" + r_content + "\n</think>\n" + body;
                    }
                }
                if (messages[i].contains("tool_calls") && messages[i]["tool_calls"].is_array()) {
                    for (const auto& tc : messages[i]["tool_calls"]) {
                        std::string fn_name = tc.value("function", json::object()).value("name", "");
                        json fn_args = tc.value("function", json::object()).value("arguments", json::object());
                        std::string args_str = fn_args.is_string() ? fn_args.get<std::string>() : fn_args.dump();
                        if (!body.empty() && body.back() != '\n') body += "\n";
                        body += "<tool_call>\n{\"name\": \"" + fn_name + "\", \"arguments\": " + args_str + "}\n</tool_call>\n";
                    }
                }
            }

            auto content_enc = tok.encode(body);
            result.insert(result.end(), content_enc.begin(), content_enc.end());
            result.push_back(IM_END);
            auto nl_enc = tok.encode("\n");
            result.insert(result.end(), nl_enc.begin(), nl_enc.end());
        }
        if (!messages.empty()) {
            std::string last_role = messages.back().value("role", "user");
            if (last_role == "user" || last_role == "tool" || last_role == "function") {
                result.push_back(IM_START);
                auto asst_enc = tok.encode("assistant\n");
                result.insert(result.end(), asst_enc.begin(), asst_enc.end());
                bool actual_thinking = enable_thinking && out.image_b64.empty();
                if (actual_thinking) {
                    auto think_enc = tok.encode("<think>\n");
                    result.insert(result.end(), think_enc.begin(), think_enc.end());
                } else {
                    auto think_enc = tok.encode("<think>\n\n</think>\n");
                    result.insert(result.end(), think_enc.begin(), think_enc.end());
                }
            }
        }
        out.prompt = result;
        return out;
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

    bool has_system = (!messages.empty() && messages[0].value("role", "") == "system");
    if (!has_system) {
        std::string default_sys = "";
        if (has_tools) {
            default_sys = g_base_system_prompt + tools_system_prompt;
        } else {
            default_sys = g_base_system_prompt;
        }
        if (!default_sys.empty()) {
            auto enc = tok.encode(default_sys);
            result.insert(result.end(), enc.begin(), enc.end());
        }
    }

    for (size_t i = 0; i < messages.size(); i++) {
        std::string role = messages[i].value("role", "user");
        std::string content = "";
        std::string extracted_image_b64 = "";

        if (messages[i].contains("content")) {
            const auto& c = messages[i]["content"];
            if (c.is_string()) {
                content = c.get<std::string>();
            } else if (c.is_array()) {
                for (const auto& part : c) {
                    if (part.is_object()) {
                        std::string type = part.value("type", "text");
                        if (type == "text") {
                            content += part.value("text", "");
                        } else if (type == "image_url" || type == "image") {
                            std::string url = "";
                            if (part.contains("image_url")) {
                                if (part["image_url"].is_string()) {
                                    url = part["image_url"].get<std::string>();
                                } else if (part["image_url"].is_object()) {
                                    url = part["image_url"].value("url", "");
                                }
                            } else if (part.contains("image")) {
                                if (part["image"].is_string()) {
                                    url = part["image"].get<std::string>();
                                } else if (part["image"].is_object()) {
                                    url = part["image"].value("url", "");
                                }
                            }
                            if (!url.empty()) {
                                size_t b64_pos = url.find("base64,");
                                extracted_image_b64 = (b64_pos != std::string::npos) ? url.substr(b64_pos + 7) : url;
                            }
                        }
                    }
                }
            }
        }
        if (content.empty()) {
            content = get_message_content_string(messages[i]);
        }
        if (extracted_image_b64.empty() && messages[i].contains("image") && messages[i]["image"].is_string()) {
            std::string url = messages[i]["image"].get<std::string>();
            size_t b64_pos = url.find("base64,");
            extracted_image_b64 = (b64_pos != std::string::npos) ? url.substr(b64_pos + 7) : url;
        }

        if (role == "system") {
            std::string sys_body = content;
            if (i == 0) {
                sys_body = update_system_prompt_with_tools(content, tools_system_prompt, has_tools);
            }
            auto enc = tok.encode(sys_body);
            result.insert(result.end(), enc.begin(), enc.end());
        } else if (role == "user") {
            result.push_back(USER);
            if (model_has_vision && !extracted_image_b64.empty()) {
                int IMAGE_TOKEN = tok.get_token_id("<｜image｜>");
                if (IMAGE_TOKEN < 0) IMAGE_TOKEN = 129279;

                out.image_b64 = extracted_image_b64;
                out.visual_start_pos = (int)result.size();
                out.visual_num_tokens = 576;
                result.insert(result.end(), 576, IMAGE_TOKEN);
                auto newline_enc = tok.encode("\n");
                result.insert(result.end(), newline_enc.begin(), newline_enc.end());

                size_t img_tag = content.find("<image>");
                if (img_tag != std::string::npos) {
                    content.erase(img_tag, 7);
                    if (img_tag < content.size() && content[img_tag] == '\n') {
                        content.erase(img_tag, 1);
                    }
                }
            }
            std::string prompt_text = content;
            if (model_has_vision && !extracted_image_b64.empty()) {
                std::string trimmed = prompt_text;
                while (!trimmed.empty() && (trimmed.front() == ' ' || trimmed.front() == '\n' || trimmed.front() == '\r' || trimmed.front() == '\t')) trimmed.erase(trimmed.begin());
                while (!trimmed.empty() && (trimmed.back() == ' ' || trimmed.back() == '\n' || trimmed.back() == '\r' || trimmed.back() == '\t' || trimmed.back() == '.' || trimmed.back() == '?' || trimmed.back() == '!')) trimmed.pop_back();
                std::string lower = trimmed;
                for (char& c : lower) c = (char)std::tolower((unsigned char)c);

                if (lower.empty() ||
                    lower == "describe this image" || lower == "describe the image" || lower == "describe image" ||
                    lower == "describe this picture" || lower == "describe picture" || lower == "describe" ||
                    lower == "what is this" || lower == "what is this image" || lower == "what is this picture" ||
                    lower == "what's this" || lower == "what do you see" || lower == "what do you see in this image" ||
                    lower == "what do you see in this picture" || lower == "analyze this image" || lower == "analyze image" ||
                    lower == "describe what is depicted in this photo" || lower == "describe what you see in this image") {
                    prompt_text = "What is shown in this image?";
                }
            }
            auto enc = tok.encode(prompt_text);
            result.insert(result.end(), enc.begin(), enc.end());
        } else if (role == "tool" || role == "function") {
            result.push_back(USER);
            auto enc = tok.encode("<tool_response>\n" + content + "\n</tool_response>");
            result.insert(result.end(), enc.begin(), enc.end());
        } else if (role == "assistant") {
            result.push_back(ASSISTANT);
            if (messages[i].contains("reasoning_content") && messages[i]["reasoning_content"].is_string()) {
                std::string r_content = messages[i]["reasoning_content"].get<std::string>();
                if (!r_content.empty()) {
                    result.push_back(THINK_BEGIN);
                    auto r_enc = tok.encode(r_content);
                    result.insert(result.end(), r_enc.begin(), r_enc.end());
                    result.push_back(THINK_END);
                } else {
                    result.push_back(THINK_END);
                }
            } else {
                result.push_back(THINK_END);
            }
            std::string body = content;
            if (messages[i].contains("tool_calls") && messages[i]["tool_calls"].is_array()) {
                for (const auto& tc : messages[i]["tool_calls"]) {
                    std::string fn_name = tc.value("function", json::object()).value("name", "");
                    json fn_args = tc.value("function", json::object()).value("arguments", json::object());
                    std::string args_str = fn_args.is_string() ? fn_args.get<std::string>() : fn_args.dump();
                    if (!body.empty() && body.back() != '\n') body += "\n";
                    body += "<tool_call>\n{\"name\": \"" + fn_name + "\", \"arguments\": " + args_str + "}\n</tool_call>\n";
                }
            }
            auto enc = tok.encode(body);
            result.insert(result.end(), enc.begin(), enc.end());
            result.push_back(EOS); // <｜end▁of▁sentence｜>
        }
    }

    // Add assistant prompt for generation
    if (!messages.empty()) {
        std::string last_role = messages.back().value("role", "user");
        if (last_role == "user" || last_role == "tool" || last_role == "function") {
            result.push_back(ASSISTANT);
            bool actual_thinking = enable_thinking && out.image_b64.empty();
            if (actual_thinking) {
                result.push_back(THINK_BEGIN);
            } else {
                result.push_back(THINK_BEGIN);
                result.push_back(THINK_END);
                auto nl = tok.encode("\n");
                result.insert(result.end(), nl.begin(), nl.end());
            }
        }
    }


    out.prompt = result;
    return out;
}

static std::vector<int> apply_chat_template(const json& messages, const BPETokenizer& tok, bool enable_thinking = true, const std::string& reasoning_effort = "high", const json& tools = json()) {
    return apply_chat_template_multimodal(messages, tok, enable_thinking, reasoning_effort, tools, false).prompt;
}


// ════════════════════════════════════════════════════════════════════════════════
//  HTTP Server (OpenAI-compatible)
// ════════════════════════════════════════════════════════════════════════════════

static std::mutex g_engine_mutex;  // serialize inference requests
static std::atomic<bool> g_engine_busy{false}; // active request lock

struct EngineBusyGuard {
    bool locked = false;
    EngineBusyGuard() {
        bool expected = false;
        locked = g_engine_busy.compare_exchange_strong(expected, true);
    }
    ~EngineBusyGuard() {
        if (locked) {
            g_engine_busy.store(false);
        }
    }
    bool is_locked() const { return locked; }
};

static int g_request_counter = 0;  // for unique request IDs
static json s_last_conv_messages = json::array();

static void rebuild_system_prefix(
    MoecherEngine& engine,
    const std::string& base_prompt,
    const json& active_tools,
    const std::string& custom_full_prompt = "")
{
    std::lock_guard<std::mutex> lock(g_engine_mutex);

    // Invalidate conversation-level continuation snapshot so next query doesn't reuse stale prompt
    engine.turn_kv_snapshot_.valid = false;
    s_last_conv_messages.clear();

    json resolved_tools = active_tools;
    if (!resolved_tools.empty()) {
        resolved_tools = resolve_canonical_tools(resolved_tools);
    }

    std::string full_prompt_text;
    if (!custom_full_prompt.empty()) {
        full_prompt_text = custom_full_prompt;
    } else {
        bool is_qwen = (engine.cfg_.architecture == ModelArch::QWEN);
        full_prompt_text = base_prompt + build_dynamic_tools_prompt(resolved_tools, is_qwen);
    }

    g_base_system_prompt = base_prompt;
    g_current_system_prompt = full_prompt_text;
    g_current_active_tools = resolved_tools;

    json default_messages = json::array({
        {{"role", "system"}, {"content", full_prompt_text}},
        {{"role", "user"}, {"content", ""}}
    });

    std::vector<int> prompt = apply_chat_template(default_messages, engine.tokenizer_, true, "high", resolved_tools);

    int user_start = (engine.cfg_.architecture == ModelArch::QWEN)
                         ? engine.tokenizer_.get_token_id("<|im_start|>")
                         : engine.tokenizer_.get_token_id("<｜User｜>");
    if (user_start < 0) {
        user_start = (engine.cfg_.architecture == ModelArch::QWEN) ? 151644 : 128803;
    }
    size_t sys_len = prompt.size();
    for (size_t i = 1; i < prompt.size(); i++) {
        if (prompt[i] == user_start) {
            sys_len = i;
            break;
        }
    }
    if (sys_len > 0) {
        std::vector<int> sys_tokens(prompt.begin(), prompt.begin() + sys_len);
        LOG_INFO("Rebuilding System KV Cache snapshot (%zu tokens, %zu active tools)...",
                 sys_tokens.size(), resolved_tools.size());
        engine.reset_all_kv_caches();
        engine.prefill_prefix(sys_tokens);
        // Note: prefill_prefix() already calls snapshot_system_kv() internally on line 4060
        LOG_INFO("System KV Cache snapshot pinned successfully (%zu tokens).", sys_tokens.size());
    } else {
        engine.reset_all_kv_caches();
        engine.system_kv_snapshot_.valid = false;
        LOG_INFO("System KV Cache snapshot cleared.");
    }
}

static std::vector<int> build_continuation_prompt(
    const MoecherEngine& engine,
    const std::vector<json>& new_messages,
    bool enable_thinking,
    const std::string& reasoning_effort)
{
    const auto& tok = engine.tokenizer_;
    std::vector<int> result = engine.turn_kv_snapshot_.tokens;

    if (engine.cfg_.architecture == ModelArch::QWEN) {
        int IM_START = tok.get_token_id("<|im_start|>");
        int IM_END = tok.get_token_id("<|im_end|>");
        auto nl = tok.encode("\n");
        result.push_back(IM_END);
        result.insert(result.end(), nl.begin(), nl.end());
        for (const auto& msg : new_messages) {
            std::string role = msg.value("role", "user");
            std::string content = get_message_content_string(msg);
            result.push_back(IM_START);
            if (role == "tool" || role == "function") {
                auto u_enc = tok.encode("user\n<tool_response>\n" + content + "\n</tool_response>");
                result.insert(result.end(), u_enc.begin(), u_enc.end());
            } else {
                auto u_enc = tok.encode("user\n" + content);
                result.insert(result.end(), u_enc.begin(), u_enc.end());
            }
            result.push_back(IM_END);
            result.insert(result.end(), nl.begin(), nl.end());
        }
        result.push_back(IM_START);
        auto asst_enc = tok.encode("assistant\n");
        result.insert(result.end(), asst_enc.begin(), asst_enc.end());
        if (enable_thinking) {
            auto th_enc = tok.encode("<think>\n");
            result.insert(result.end(), th_enc.begin(), th_enc.end());
        } else {
            auto th_enc = tok.encode("<think>\n\n</think>\n");
            result.insert(result.end(), th_enc.begin(), th_enc.end());
        }
    } else {
        // DeepSeek V4
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

        result.push_back(EOS);
        for (const auto& msg : new_messages) {
            std::string role = msg.value("role", "user");
            std::string content = get_message_content_string(msg);
            result.push_back(USER);
            if (role == "tool" || role == "function") {
                auto enc = tok.encode("<tool_response>\n" + content + "\n</tool_response>");
                result.insert(result.end(), enc.begin(), enc.end());
            } else {
                auto enc = tok.encode(content);
                result.insert(result.end(), enc.begin(), enc.end());
            }
        }
        result.push_back(ASSISTANT);
        if (enable_thinking) {
            result.push_back(THINK_BEGIN);
        } else {
            result.push_back(THINK_BEGIN);
            result.push_back(THINK_END);
        }
    }
    return result;
}

static bool check_and_build_continuation(
    const MoecherEngine& engine,
    const json& current_messages,
    const json& last_conv_messages,
    bool enable_thinking,
    const std::string& reasoning_effort,
    std::vector<int>& out_prompt)
{
    if (!engine.turn_kv_snapshot_.valid || engine.turn_kv_snapshot_.tokens.empty()) {
        return false;
    }
    if (last_conv_messages.empty() || current_messages.size() <= last_conv_messages.size()) {
        return false;
    }

    size_t prev_len = last_conv_messages.size();
    // Check if the previous messages match in role
    for (size_t i = 0; i < prev_len; i++) {
        std::string cur_role = current_messages[i].value("role", "");
        std::string last_role = last_conv_messages[i].value("role", "");
        if (cur_role != last_role) {
            return false;
        }
    }

    // Check if previous assistant message matches
    if (prev_len > 0) {
        std::string cur_asst = get_message_content_string(current_messages[prev_len - 1]);
        std::string last_asst = get_message_content_string(last_conv_messages[prev_len - 1]);
        if (!cur_asst.empty() && !last_asst.empty() && cur_asst != last_asst) {
            return false;
        }
    }

    // Collect new messages (user / tool)
    std::vector<json> new_msgs;
    for (size_t i = prev_len; i < current_messages.size(); i++) {
        std::string role = current_messages[i].value("role", "");
        if (role != "user" && role != "tool" && role != "function") {
            return false;
        }
        // If any incoming message has multimodal image content, bypass text-only continuation
        // so that the Vision Tower can process the image and embed visual tokens
        if (message_has_image(current_messages[i])) {
            return false;
        }
        new_msgs.push_back(current_messages[i]);
    }

    if (new_msgs.empty()) return false;

    out_prompt = build_continuation_prompt(engine, new_msgs, enable_thinking, reasoning_effort);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Context Window Pruning: Evict oldest turns when context limit is approached
// ─────────────────────────────────────────────────────────────────────────────
static json prune_messages_to_fit(
    const json& messages,
    const BPETokenizer& tok,
    int max_allowed_tokens,
    bool enable_thinking,
    const std::string& reasoning_effort,
    const json& tools)
{
    json pruned = messages;
    if (pruned.empty()) return pruned;

    std::vector<int> test_prompt = apply_chat_template(pruned, tok, enable_thinking, reasoning_effort, tools);
    if (test_prompt.size() <= (size_t)max_allowed_tokens) {
        return pruned;
    }

    LOG_WARN("[Context] Prompt length (%zu) exceeds context budget (%d). Pruning older turns...",
             test_prompt.size(), max_allowed_tokens);

    bool has_system = (pruned[0].value("role", "") == "system");
    size_t min_idx = has_system ? 1 : 0;

    // Prune oldest non-system messages
    while (pruned.size() > min_idx + 2) {
        pruned.erase(min_idx);
        test_prompt = apply_chat_template(pruned, tok, enable_thinking, reasoning_effort, tools);
        if (test_prompt.size() <= (size_t)max_allowed_tokens) {
            LOG_INFO("[Context] Pruning complete: retained %zu messages, prompt length %zu / %d",
                     pruned.size(), test_prompt.size(), max_allowed_tokens);
            return pruned;
        }
    }

    // If still too long, truncate oversized individual messages (e.g. huge file reads)
    while (test_prompt.size() > (size_t)max_allowed_tokens) {
        size_t largest_idx = 0;
        size_t largest_len = 0;
        for (size_t i = min_idx; i < pruned.size(); i++) {
            std::string content = get_message_content_string(pruned[i]);
            if (content.size() > largest_len) {
                largest_len = content.size();
                largest_idx = i;
            }
        }
        if (largest_len < 300) break; // Cannot truncate safely further

        std::string content = get_message_content_string(pruned[largest_idx]);
        size_t keep_len = (content.size() * 3) / 4;
        std::string notice = "\n[... Content truncated to fit context window ...]\n";
        if (keep_len > notice.size()) {
            pruned[largest_idx]["content"] = content.substr(0, keep_len - notice.size()) + notice;
        } else {
            pruned[largest_idx]["content"] = notice;
        }
        test_prompt = apply_chat_template(pruned, tok, enable_thinking, reasoning_effort, tools);
    }

    LOG_INFO("[Context] Pruning finished: final prompt length %zu / %d",
             test_prompt.size(), max_allowed_tokens);
    return pruned;
}

// ─────────────────────────────────────────────────────────────────────────────
// ToolTagStreamFilter: On-demand lookahead interceptor for streaming tool calls
// ─────────────────────────────────────────────────────────────────────────────
struct ToolTagStreamFilter {
    enum class State {
        STREAMING,
        BUFFERING_TAG,
        BUFFERING_TOOL
    };

    State state = State::STREAMING;
    std::string tag_candidate;
    std::string tool_buffer;
    std::string streamed_clean_text;
    bool in_tool_calls_container = false;
    std::function<bool(const std::string&)> on_status = nullptr;
    std::string current_tool_name;
    size_t last_reported_size = 0;

    bool feed(const std::string& text, const std::function<bool(const std::string&)>& emit_cb) {
        std::string safe_chunk;
        for (char c : text) {
            if (state == State::STREAMING) {
                if (c == '<') {
                    state = State::BUFFERING_TAG;
                    tag_candidate.clear();
                    tag_candidate.push_back(c);
                } else {
                    safe_chunk.push_back(c);
                }
            } else if (state == State::BUFFERING_TAG) {
                tag_candidate.push_back(c);

                if (c == '>') {
                    // Tag candidate is complete
                    if (is_opening_tool_tag(tag_candidate)) {
                        state = State::BUFFERING_TOOL;
                        if (is_container_tool_tag(tag_candidate)) {
                            in_tool_calls_container = true;
                        }
                        tool_buffer = tag_candidate;
                        tag_candidate.clear();
                        current_tool_name.clear();
                        last_reported_size = 0;
                        if (on_status) {
                            on_status("Preparing tool call & arguments...");
                        }
                    } else if (is_closing_or_standalone_tool_tag(tag_candidate)) {
                        // Stray / standalone tool tag (e.g. </tool_call>, <｜tool sep｜>, </｜DSML...>)
                        state = State::STREAMING;
                        tag_candidate.clear();
                    } else if (is_tool_tag_prefix(tag_candidate)) {
                        // Matched tool syntax
                        state = State::BUFFERING_TOOL;
                        tool_buffer = tag_candidate;
                        tag_candidate.clear();
                        current_tool_name.clear();
                        last_reported_size = 0;
                        if (on_status) {
                            on_status("Preparing tool call & arguments...");
                        }
                    } else {
                        // Normal HTML tag that happened to complete
                        safe_chunk += tag_candidate;
                        tag_candidate.clear();
                        state = State::STREAMING;
                    }
                } else if (is_tool_tag_prefix(tag_candidate)) {
                    // Still potentially a tool call tag, continue buffering
                } else {
                    // Diverged from tool tag syntax (e.g. <div>, <html>, <=, <!DOCTYPE)
                    if (c == '<') {
                        std::string prev = tag_candidate.substr(0, tag_candidate.size() - 1);
                        safe_chunk += prev;
                        tag_candidate = "<";
                    } else {
                        safe_chunk += tag_candidate;
                        tag_candidate.clear();
                        state = State::STREAMING;
                    }
                }
            } else if (state == State::BUFFERING_TOOL) {
                tool_buffer.push_back(c);

                // Early detection of tool name inside JSON arguments
                if (current_tool_name.empty() && tool_buffer.size() > 15) {
                    size_t npos = tool_buffer.find("\"name\"");
                    if (npos != std::string::npos) {
                        size_t colon = tool_buffer.find(':', npos);
                        if (colon != std::string::npos) {
                            size_t q1 = tool_buffer.find('"', colon);
                            if (q1 != std::string::npos) {
                                size_t q2 = tool_buffer.find('"', q1 + 1);
                                if (q2 != std::string::npos) {
                                    current_tool_name = tool_buffer.substr(q1 + 1, q2 - q1 - 1);
                                    if (on_status) {
                                        if (current_tool_name == "create_3d_model" || current_tool_name == "model_3d") {
                                            on_status("Generating 3D model geometry & Three.js code...");
                                        } else {
                                            on_status("Generating tool call: " + current_tool_name + "...");
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Periodic live progress reporting while tool arguments stream in
                if (tool_buffer.size() - last_reported_size >= 120) {
                    last_reported_size = tool_buffer.size();
                    if (on_status) {
                        int approx_tokens = (int)(tool_buffer.size() / 4);
                        if (current_tool_name == "create_3d_model" || current_tool_name == "model_3d") {
                            on_status("Generating 3D model geometry & Three.js code (" + std::to_string(approx_tokens) + " tokens)...");
                        } else if (!current_tool_name.empty()) {
                            on_status("Generating arguments for " + current_tool_name + " (" + std::to_string(approx_tokens) + " tokens)...");
                        } else {
                            on_status("Generating tool call parameters (" + std::to_string(approx_tokens) + " tokens)...");
                        }
                    }
                }

                if (in_tool_calls_container) {
                    if (ends_with_container_closing(tool_buffer)) {
                        state = State::STREAMING;
                        tool_buffer.clear();
                        current_tool_name.clear();
                        last_reported_size = 0;
                        in_tool_calls_container = false;
                    }
                } else {
                    if (ends_with_tool_closing(tool_buffer)) {
                        state = State::STREAMING;
                        tool_buffer.clear();
                        current_tool_name.clear();
                        last_reported_size = 0;
                    }
                }
            }
        }
        if (!safe_chunk.empty()) {
            streamed_clean_text += safe_chunk;
            return emit_cb(safe_chunk);
        }
        return true;
    }

    bool flush_remaining(const std::function<bool(const std::string&)>& emit_cb) {
        if (state == State::BUFFERING_TAG && !tag_candidate.empty()) {
            std::string to_flush = tag_candidate;
            tag_candidate.clear();
            state = State::STREAMING;
            // If it's a lonely '<' or '</', flush it.
            // If it's an uncompleted tool/DSML tag (e.g. "<tool_call" or "<｜DSML" or "< DSML"), suppress it!
            if (to_flush == "<" || to_flush == "</" || !is_tool_tag_prefix(to_flush)) {
                streamed_clean_text += to_flush;
                return emit_cb(to_flush);
            }
        }
        state = State::STREAMING;
        tool_buffer.clear();
        in_tool_calls_container = false;
        return true;
    }

private:
    static std::string to_lower_ascii(const std::string& s) {
        std::string res = s;
        for (char& ch : res) {
            if (ch >= 'A' && ch <= 'Z') ch = ch + ('a' - 'A');
        }
        return res;
    }

    static bool is_tool_tag_prefix(const std::string& s) {
        if (s.empty()) return false;
        if (s[0] != '<') return false;

        size_t idx = 1;
        if (idx < s.size() && s[idx] == '/') idx++;
        while (idx < s.size() && s[idx] == ' ') idx++;

        std::string rem = s.substr(idx);
        if (rem.empty()) return true;

        std::string rem_lower = to_lower_ascii(rem);

        static const std::vector<std::string> roots = {
            "tool_call",
            "tool_calls",
            "tools",
            "tool",
            "function_call",
            "call",
            "\xef\xbd\x9ctool call begin\xef\xbd\x9c",
            "\xef\xbd\x9ctool call end\xef\xbd\x9c",
            "\xef\xbd\x9ctool sep\xef\xbd\x9c",
            "\xef\xbd\x9ctool outputs begin\xef\xbd\x9c",
            "\xef\xbd\x9ctool outputs end\xef\xbd\x9c",
            "|tool call begin|",
            "|tool call end|",
            "|tool sep|",
            "|tool outputs begin|",
            "|tool outputs end|",
            "\xef\xbd\x9c" "dsml\xef\xbd\x9c" "tool_calls",
            "\xef\xbd\x9c" "dsml\xef\xbd\x9c" "invoke",
            "\xef\xbd\x9c" "dsml\xef\xbd\x9c" "parameter",
            "\xef\xbd\x9c" "dsml",
            "|dsml|tool_calls",
            "|dsml|invoke",
            "|dsml|parameter",
            "|dsml",
            "dsml:tool_calls",
            "dsml_tool_calls",
            "dsml:invoke",
            "dsml invoke",
            "dsml:parameter",
            "dsml"
        };

        for (const auto& root : roots) {
            if (rem_lower.size() <= root.size()) {
                if (root.compare(0, rem_lower.size(), rem_lower) == 0) return true;
            } else {
                if (rem_lower.compare(0, root.size(), root) == 0) return true;
            }
        }

        if (rem.size() == 1 && (unsigned char)rem[0] == 0xEF) return true;
        if (rem.size() == 2 && (unsigned char)rem[0] == 0xEF && (unsigned char)rem[1] == 0xBD) return true;

        return false;
    }

    static bool is_opening_tool_tag(const std::string& s) {
        std::string lower = to_lower_ascii(s);
        if (lower.find("</") != std::string::npos) return false;
        if (lower.find("end") != std::string::npos) return false;
        if (lower.find("sep") != std::string::npos) return false;

        if (lower.find("tool_call") != std::string::npos) return true;
        if (lower.find("tool_calls") != std::string::npos) return true;
        if (lower.find("tools") != std::string::npos) return true;
        if (lower.find("tool call begin") != std::string::npos) return true;
        if (lower.find("dsml") != std::string::npos) return true;
        if (lower.find("function_call") != std::string::npos) return true;
        if (lower.find("<call") != std::string::npos) return true;
        if (lower.find("<tool>") != std::string::npos || lower.find("<tool ") != std::string::npos) return true;

        return false;
    }

    static bool is_container_tool_tag(const std::string& s) {
        std::string lower = to_lower_ascii(s);
        if (lower.find("tool_calls") != std::string::npos) return true;
        if (lower.find("<tools>") != std::string::npos || lower.find("<tools ") != std::string::npos) return true;
        return false;
    }

    static bool is_closing_or_standalone_tool_tag(const std::string& s) {
        std::string lower = to_lower_ascii(s);
        if (lower.find("end") != std::string::npos) return true;
        if (lower.find("sep") != std::string::npos) return true;
        if (lower.find("</tool") != std::string::npos) return true;
        if (lower.find("</dsml") != std::string::npos) return true;
        if (lower.find("</\xef\xbd\x9c" "dsml") != std::string::npos) return true;
        if (lower.find("</|dsml") != std::string::npos) return true;
        if (lower.find("</call>") != std::string::npos) return true;
        if (lower.find("</function_call>") != std::string::npos) return true;
        return false;
    }

    static bool ends_with(const std::string& s, const std::string& suffix) {
        return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    static bool ends_with_container_closing(const std::string& s) {
        std::string lower = to_lower_ascii(s);
        static const std::vector<std::string> closings = {
            "</tool_calls>",
            "</tools>",
            "</\xef\xbd\x9c" "dsml\xef\xbd\x9c" "tool_calls>",
            "</|dsml|tool_calls>",
            "</dsml:tool_calls>",
            "</dsml_tool_calls>",
            "</dsml>",
            "</think>"
        };
        for (const auto& c : closings) {
            if (ends_with(lower, c)) return true;
        }
        return false;
    }

    static bool ends_with_tool_closing(const std::string& s) {
        std::string lower = to_lower_ascii(s);
        static const std::vector<std::string> closings = {
            "</tool_call>",
            "</tool_calls>",
            "</tools>",
            "</tool>",
            "</function_call>",
            "</call>",
            "<\xef\xbd\x9ctool call end\xef\xbd\x9c>",
            "<|tool call end|>",
            "<\xef\xbd\x9ctool outputs end\xef\xbd\x9c>",
            "<|tool outputs end|>",
            "</\xef\xbd\x9c" "dsml\xef\xbd\x9c" "tool_calls>",
            "</|dsml|tool_calls>",
            "</\xef\xbd\x9c" "dsml\xef\xbd\x9c" "invoke>",
            "</|dsml|invoke>",
            "</\xef\xbd\x9c" "dsml\xef\xbd\x9c" "parameter>",
            "</|dsml|parameter>",
            "</dsml:tool_calls>",
            "</dsml_tool_calls>",
            "</dsml>",
            "</think>"
        };
        for (const auto& c : closings) {
            if (ends_with(lower, c)) return true;
        }
        return false;
    }
};

static void run_server(MoecherEngine& engine, int port, int default_thinking_budget = 4096, int proxy_port = 8002, bool enable_forward_proxy = true, bool enable_system_proxy = false) {
    httplib::Server svr;
    svr.set_tcp_nodelay(true);
    svr.set_payload_max_length(64 * 1024 * 1024); // 64 MB
    svr.set_socket_options([](socket_t sock) {
        int opt = 1;
#ifdef _WIN32
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt));
#else
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const void*>(&opt), sizeof(opt));
#endif
    });

    if (enable_forward_proxy) {
        moecher::proxy::start_forward_proxy(proxy_port);
    }

    // Note: MCP (Model Context Protocol) Manager was initialized prior to pre-warming in main()

#ifdef _WIN32
    if (enable_system_proxy) {
        moecher::proxy::set_windows_system_proxy(true, "127.0.0.1:" + std::to_string(proxy_port));
        SetConsoleCtrlHandler([](DWORD) -> BOOL {
            moecher::proxy::set_windows_system_proxy(false, "");
            moecher::proxy::stop_forward_proxy();
            return FALSE;
        }, TRUE);
        std::atexit([]() {
            moecher::proxy::set_windows_system_proxy(false, "");
            moecher::proxy::stop_forward_proxy();
        });
    }
#endif

    // CORS headers and preflight
    svr.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE");
        res.set_header("Access-Control-Allow-Headers", "*");

        if (req.method == "OPTIONS") {
            res.status = 200;
            return httplib::Server::HandlerResponse::Handled;
        }

        // Direct HTTP Forward Proxy fallback on main port if accessed via forward proxy
        std::string target_url = req.path;
        if (target_url.rfind("http://", 0) == 0 || target_url.rfind("https://", 0) == 0) {
            std::string client_ua = req.get_header_value("User-Agent");
            std::string client_lang = req.get_header_value("Accept-Language");
            std::string client_sec_ua = req.get_header_value("Sec-Ch-Ua");
            std::string client_sec_mobile = req.get_header_value("Sec-Ch-Ua-Mobile");
            std::string client_sec_platform = req.get_header_value("Sec-Ch-Ua-Platform");
            std::string client_cookies = req.get_header_value("Cookie");

            moecher::tooling::set_client_browser_context(client_ua, client_lang, client_sec_ua, client_sec_mobile, client_sec_platform, client_cookies);

            auto doc = moecher::tooling::fetch_url_full(target_url, 30000, 5000000, "raw");
            res.status = 200;
            std::string ct = "text/html; charset=utf-8";
            if (moecher::tooling::is_raw_code_url(target_url)) {
                if (target_url.find(".json") != std::string::npos) ct = "application/json; charset=utf-8";
                else if (target_url.find(".js") != std::string::npos) ct = "application/javascript; charset=utf-8";
                else if (target_url.find(".css") != std::string::npos) ct = "text/css; charset=utf-8";
                else ct = "text/plain; charset=utf-8";
            }
            res.set_content(doc.raw_html.empty() ? doc.clean_text : doc.raw_html, ct);
            return httplib::Server::HandlerResponse::Handled;
        }

        return httplib::Server::HandlerResponse::Unhandled;
    });

    // Web UI endpoints & reverse proxy (statically linked from src/web_ui.cpp)
    moecher::web::register_web_routes(svr);

    // Agentic & Tooling endpoints (statically linked from src/agentic.cpp)
    moecher::agentic::register_agentic_routes(svr);

    svr.Post("/api/kv/reset", [&engine](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        {
            std::lock_guard<std::mutex> lock(g_engine_mutex);
            s_last_conv_messages.clear();
        }
        engine.turn_kv_snapshot_.valid = false;
        if (engine.system_kv_snapshot_.valid) {
            engine.restore_system_kv();
            LOG_INFO("[KV RESET] Restored pinned System KV snapshot of %zu tokens (ready for fresh conversation with 0ms prefill).",
                     engine.system_kv_snapshot_.tokens.size());
        } else {
            engine.reset_all_kv_caches();
            engine.cached_tokens_.clear();
            LOG_INFO("[KV RESET] Reset all KV caches to empty.");
        }
        json body = {
            {"status", "ok"},
            {"message", "KV cache reset successfully"},
            {"system_snapshot_restored", engine.system_kv_snapshot_.valid},
            {"cached_tokens", engine.cached_tokens_.size()}
        };
        res.set_content(body.dump(), "application/json");
    });

    svr.Get("/api/kv/reset", [&engine](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        {
            std::lock_guard<std::mutex> lock(g_engine_mutex);
            s_last_conv_messages.clear();
        }
        engine.turn_kv_snapshot_.valid = false;
        if (engine.system_kv_snapshot_.valid) {
            engine.restore_system_kv();
        } else {
            engine.reset_all_kv_caches();
            engine.cached_tokens_.clear();
        }
        json body = {
            {"status", "ok"},
            {"system_snapshot_restored", engine.system_kv_snapshot_.valid},
            {"cached_tokens", engine.cached_tokens_.size()}
        };
        res.set_content(body.dump(), "application/json");
    });

    svr.Options("/api/kv/reset", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.status = 204;
    });

    // ── System Prompt & Tool Configuration Endpoints ────────────────────────
    svr.Get("/api/system/prompt", [&engine](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        json body = {
            {"status", "ok"},
            {"server_instance_id", g_server_instance_id},
            {"base_prompt", g_base_system_prompt},
            {"system_prompt", g_current_system_prompt},
            {"active_tools", g_current_active_tools},
            {"tokens_count", engine.system_kv_snapshot_.valid ? engine.system_kv_snapshot_.tokens.size() : 0}
        };
        res.set_content(body.dump(), "application/json");
    });

    svr.Options("/api/system/prompt", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.status = 204;
    });

    svr.Post("/api/system/configure", [&engine](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        try {
            json j = json::parse(req.body);
            std::string base_prompt = j.value("base_prompt", g_base_system_prompt);
            std::string custom_full_prompt = j.value("system_prompt", j.value("custom_system_prompt", ""));

            json active_tools = g_current_active_tools;
            if (j.contains("tools")) {
                active_tools = j["tools"];
                if (active_tools.is_array() || active_tools.is_string()) {
                    active_tools = resolve_canonical_tools(active_tools);
                    // Retain discovered MCP tools unless caller explicitly passed an empty array []
                    if (!j["tools"].is_array() || !j["tools"].empty()) {
                        json mcp_tools = moecher::mcp::MCPManager::instance().get_openai_tools_schema();
                        for (const auto& mt : mcp_tools) {
                            active_tools.push_back(mt);
                        }
                    }
                }
                bool has_yt = false;
                if (active_tools.is_array()) {
                    for (const auto& item : active_tools) {
                        if (item.is_string() && item.get<std::string>() == "youtube_search") has_yt = true;
                        else if (item.is_object() && item.contains("function") && item["function"].value("name", "") == "youtube_search") has_yt = true;
                    }
                }
                moecher::tooling::g_fast_media_search = has_yt;
            }

            rebuild_system_prefix(engine, base_prompt, active_tools, custom_full_prompt);

            json body = {
                {"status", "ok"},
                {"server_instance_id", g_server_instance_id},
                {"base_prompt", g_base_system_prompt},
                {"system_prompt", g_current_system_prompt},
                {"active_tools", g_current_active_tools},
                {"tokens_count", engine.system_kv_snapshot_.valid ? engine.system_kv_snapshot_.tokens.size() : 0}
            };
            res.set_content(body.dump(), "application/json");
        } catch (const std::exception& e) {
            json err = {{"status", "error"}, {"message", e.what()}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
        }
    });

    svr.Options("/api/system/configure", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.status = 204;
    });

    auto get_system_resource_telemetry = []() -> json {
        json sys;
        
        // 1. GPU VRAM (CUDA)
        size_t free_vram = 0, total_vram = 0;
        cudaError_t cuda_err = cudaMemGetInfo(&free_vram, &total_vram);
        if (cuda_err == cudaSuccess && total_vram > 0) {
            sys["vram"] = {
                {"free_bytes", free_vram},
                {"total_bytes", total_vram},
                {"used_bytes", total_vram > free_vram ? total_vram - free_vram : 0},
                {"free_gb", (double)free_vram / (1024.0 * 1024.0 * 1024.0)},
                {"total_gb", (double)total_vram / (1024.0 * 1024.0 * 1024.0)},
                {"used_gb", (double)(total_vram > free_vram ? total_vram - free_vram : 0) / (1024.0 * 1024.0 * 1024.0)}
            };
        } else {
            sys["vram"] = {
                {"free_bytes", 0}, {"total_bytes", 0}, {"used_bytes", 0},
                {"free_gb", 0.0}, {"total_gb", 0.0}, {"used_gb", 0.0}
            };
        }

        // 2. System RAM
        size_t ram_total_bytes = 0, ram_available_bytes = 0;
#ifndef _WIN32
        std::ifstream meminfo("/proc/meminfo");
        if (meminfo.is_open()) {
            std::string line;
            size_t total_kb = 0, avail_kb = 0;
            while (std::getline(meminfo, line)) {
                if (line.rfind("MemTotal:", 0) == 0) {
                    sscanf(line.c_str(), "MemTotal: %zu kB", &total_kb);
                } else if (line.rfind("MemAvailable:", 0) == 0) {
                    sscanf(line.c_str(), "MemAvailable: %zu kB", &avail_kb);
                }
            }
            ram_total_bytes = total_kb * 1024;
            ram_available_bytes = avail_kb * 1024;
        }
#else
        MEMORYSTATUSEX memStatus;
        memStatus.dwLength = sizeof(memStatus);
        if (GlobalMemoryStatusEx(&memStatus)) {
            ram_total_bytes = memStatus.ullTotalPhys;
            ram_available_bytes = memStatus.ullAvailPhys;
        }
#endif
        sys["ram"] = {
            {"available_bytes", ram_available_bytes},
            {"total_bytes", ram_total_bytes},
            {"available_gb", (double)ram_available_bytes / (1024.0 * 1024.0 * 1024.0)},
            {"total_gb", (double)ram_total_bytes / (1024.0 * 1024.0 * 1024.0)}
        };

        // 3. SSD / Disk Storage
        size_t storage_available_bytes = 0, storage_total_bytes = 0;
        try {
            auto si = std::filesystem::space(".");
            storage_available_bytes = si.available;
            storage_total_bytes = si.capacity;
        } catch (...) {}
        sys["storage"] = {
            {"available_bytes", storage_available_bytes},
            {"total_bytes", storage_total_bytes},
            {"available_gb", (double)storage_available_bytes / (1024.0 * 1024.0 * 1024.0)},
            {"total_gb", (double)storage_total_bytes / (1024.0 * 1024.0 * 1024.0)}
        };

        return sys;
    };

    svr.Get("/v1/models", [&engine, get_system_resource_telemetry](const httplib::Request&, httplib::Response& res) {
        std::string model_id = engine.get_model_id();
        std::string display_name = engine.get_model_display_name();
        std::string arch_desc = (engine.cfg_.architecture == ModelArch::QWEN)
            ? "DeltaNet Linear Attention + MTP Speculative Decoding"
            : (model_id.find("coder") != std::string::npos
                ? "MoE Coder (64 Experts)"
                : "Sparse MoE (11,008 Experts) + MLA Latent Attention");
        json sys_res = get_system_resource_telemetry();
        json model_entry = {
            {"id", model_id},
            {"name", display_name},
            {"display_name", display_name},
            {"object", "model"},
            {"owned_by", "moecher"},
            {"architecture", arch_desc},
            {"max_context_length", engine.cfg_.max_seq_len},
            {"has_vision", engine.supports_vision()},
            {"has_vision_delegate", engine.has_vision_delegate()},
            {"active", true},
            {"resources", sys_res}
        };
        json body = {
            {"object", "list"},
            {"data", {model_entry}},
            {"resources", sys_res}
        };
        res.set_content(body.dump(), "application/json");
    });

    svr.Get("/api/model", [&engine, get_system_resource_telemetry](const httplib::Request&, httplib::Response& res) {
        std::string model_id = engine.get_model_id();
        std::string display_name = engine.get_model_display_name();
        std::string arch_desc = (engine.cfg_.architecture == ModelArch::QWEN)
            ? "DeltaNet Linear Attention + MTP Speculative Decoding"
            : (model_id.find("coder") != std::string::npos
                ? "MoE Coder (64 Experts)"
                : "Sparse MoE (11,008 Experts) + MLA Latent Attention");
        json sys_res = get_system_resource_telemetry();
        json body = {
            {"id", model_id},
            {"name", display_name},
            {"display_name", display_name},
            {"architecture", arch_desc},
            {"max_context_length", engine.cfg_.max_seq_len},
            {"vocab_size", engine.cfg_.vocab_size},
            {"max_tool_rounds", g_max_tool_rounds},
            {"has_vision", engine.supports_vision()},
            {"has_vision_delegate", engine.has_vision_delegate()},
            {"active", true},
            {"resources", sys_res}
        };
        res.set_content(body.dump(), "application/json");
    });

    svr.Get("/api/system/resources", [get_system_resource_telemetry](const httplib::Request&, httplib::Response& res) {
        json sys_res = get_system_resource_telemetry();
        res.set_content(sys_res.dump(), "application/json");
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


    // ── Zero Decode Bypass (ZDB) Endpoints ───────────────────────────────────
    auto handle_decide_request = [&engine](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "*");

        auto busy_guard = std::make_shared<EngineBusyGuard>();
        if (!busy_guard->is_locked()) {
            res.status = 429;
            res.set_content("{\"error\":\"Engine is currently busy processing another request.\"}", "application/json");
            return;
        }

        json request;
        try { request = json::parse(req.body); }
        catch (...) {
            res.status = 400;
            res.set_content("{\"error\":\"Invalid JSON\"}", "application/json");
            return;
        }

        std::vector<std::string> choices;
        if (request.contains("choices") && request["choices"].is_array()) {
            for (const auto& c : request["choices"]) {
                if (c.is_string()) choices.push_back(c.get<std::string>());
            }
        } else if (request.contains("decision_choices") && request["decision_choices"].is_array()) {
            for (const auto& c : request["decision_choices"]) {
                if (c.is_string()) choices.push_back(c.get<std::string>());
            }
        }

        if (choices.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"Missing or empty 'choices' array for Zero Decode Bypass\"}", "application/json");
            return;
        }

        float temperature = request.value("temperature", 1.0f);
        std::string format = request.value("format", "json");
        bool bypass_prefix_cache = request.value("bypass_prefix_cache", false);
        bool auto_fallback = request.value("auto_fallback", false);
        float min_confidence = request.value("min_confidence", 0.70f);
        float max_top1_gap = request.value("max_top1_gap", 1.5f);

        std::vector<int> prompt;
        if (request.contains("messages") && request["messages"].is_array()) {
            prompt = apply_chat_template(request["messages"], engine.tokenizer_, /*enable_thinking=*/false, /*reasoning_effort=*/"none");
        } else if (request.contains("prompt") && request["prompt"].is_string()) {
            prompt = engine.tokenizer_.encode(request["prompt"].get<std::string>());
        }

        if (prompt.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"Missing or empty 'messages' or 'prompt'\"}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(g_engine_mutex);
        auto result = engine.execute_zero_decode_bypass(prompt, choices, temperature, format, bypass_prefix_cache, min_confidence, max_top1_gap);

        if (auto_fallback && !result.gate_passed) {
            LOG_INFO("[ZDB-Arbiter] /v1/decide: Gate REJECTED System 1 fast path (top1_gap=%.2f, conf=%.2f [min=%.2f]). Falling back to System 2 decode...",
                     result.top1_gap, result.confidence, min_confidence);
            std::string sys2_text = engine.generate(prompt, 512, temperature);
            while (!sys2_text.empty() && (sys2_text.front() == ' ' || sys2_text.front() == '\n' || sys2_text.front() == '\r')) sys2_text.erase(0, 1);
            while (!sys2_text.empty() && (sys2_text.back() == ' ' || sys2_text.back() == '\n' || sys2_text.back() == '\r')) sys2_text.pop_back();

            std::string matched_choice = result.winning_choice;
            for (const auto& ch : choices) {
                if (sys2_text.find(ch) != std::string::npos) {
                    matched_choice = ch;
                    break;
                }
            }

            if (format == "text" || format == "choice") {
                res.set_content(matched_choice, "text/plain");
            } else {
                json out_json = {
                    {"decision", matched_choice},
                    {"system_2_output", sys2_text},
                    {"zero_decode_bypass", false},
                    {"decode_passes", engine.completion_token_count_},
                    {"arbiter", {
                        {"gate_passed", false},
                        {"system_path", "system_2_fallback"},
                        {"top1_gap", result.top1_gap},
                        {"candidate_margin", result.candidate_margin},
                        {"is_choice_top1", result.is_choice_top1},
                        {"fallback_triggered", true}
                    }},
                    {"global_top1", {
                        {"token_id", result.global_argmax_token_id},
                        {"token", result.global_argmax_token_str},
                        {"logit", result.global_argmax_logit},
                        {"is_candidate", result.is_choice_top1}
                    }},
                    {"probabilities", result.probabilities}
                };
                res.set_content(out_json.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
            }
            return;
        }

        if (format == "text" || format == "choice") {
            res.set_content(result.rendered_output, "text/plain");
        } else {
            res.set_content(result.rendered_output, "application/json");
        }
    };

    svr.Post("/v1/decide", handle_decide_request);
    svr.Post("/v1/classify", handle_decide_request);
    svr.Options("/v1/decide", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.status = 204;
    });
    svr.Options("/v1/classify", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.status = 204;
    });

    // Chat completions
    svr.Post("/v1/chat/completions",
        [&engine, default_thinking_budget](const httplib::Request& req, httplib::Response& res) {
            auto busy_guard = std::make_shared<EngineBusyGuard>();
            if (!busy_guard->is_locked()) {
                LOG_WARN("Rejecting /v1/chat/completions: engine is currently busy processing another request");
                res.status = 429;
                res.set_content("{\"error\":{\"message\":\"Engine is currently busy processing another request. Please wait.\",\"type\":\"busy\",\"code\":429}}", "application/json");
                return;
            }

            g_stop_requested.store(false);
            auto start = std::chrono::steady_clock::now();

            json request;
            try { request = json::parse(req.body); }
            catch (...) {
                res.status = 400;
                res.set_content("{\"error\":\"Invalid JSON\"}", "application/json");
                return;
            }

            // Capture and propagate active client browser context (User-Agent, Accept-Language, Client Hints, Cookies)
            std::string client_ua = req.get_header_value("X-Client-User-Agent");
            if (client_ua.empty()) client_ua = req.get_header_value("User-Agent");
            if (client_ua.empty() && request.contains("client_context") && request["client_context"].contains("user_agent")) {
                client_ua = request["client_context"].value("user_agent", "");
            }

            std::string client_lang = req.get_header_value("X-Client-Accept-Language");
            if (client_lang.empty()) client_lang = req.get_header_value("Accept-Language");
            if (client_lang.empty() && request.contains("client_context") && request["client_context"].contains("languages")) {
                client_lang = request["client_context"].value("languages", "");
            }

            std::string client_sec_ua = req.get_header_value("Sec-Ch-Ua");
            if (client_sec_ua.empty() && request.contains("client_context") && request["client_context"].contains("sec_ch_ua")) {
                client_sec_ua = request["client_context"].value("sec_ch_ua", "");
            }

            std::string client_sec_mobile = req.get_header_value("Sec-Ch-Ua-Mobile");
            if (client_sec_mobile.empty() && request.contains("client_context") && request["client_context"].contains("mobile")) {
                client_sec_mobile = request["client_context"].value("mobile", "");
            }

            std::string client_sec_platform = req.get_header_value("Sec-Ch-Ua-Platform");
            if (client_sec_platform.empty() && request.contains("client_context") && request["client_context"].contains("platform")) {
                client_sec_platform = request["client_context"].value("platform", "");
            }

            std::string client_cookies = req.get_header_value("X-Client-Cookie");
            if (client_cookies.empty()) client_cookies = req.get_header_value("Cookie");
            if (client_cookies.empty() && request.contains("client_context") && request["client_context"].contains("cookies")) {
                client_cookies = request["client_context"].value("cookies", "");
            }

            if (!client_ua.empty() || !client_cookies.empty()) {
                moecher::tooling::set_client_browser_context(client_ua, client_lang, client_sec_ua, client_sec_mobile, client_sec_platform, client_cookies);
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
            int max_tokens = request.value("max_tokens", 20000);
            bool stream = request.value("stream", false);
            float repetition_penalty = request.value("repetition_penalty", 1.0f);
            std::string reasoning_effort = request.value("reasoning_effort", "high");
            bool enable_thinking = true;
            int max_thinking_tokens = default_thinking_budget;

            json tools = json::array();
            if (g_enable_tools) {
                if (request.contains("tools")) {
                    if (request["tools"].is_array()) {
                        tools = resolve_canonical_tools(request["tools"]);
                    } else if (request["tools"].is_string()) {
                        tools = resolve_canonical_tools(request["tools"]);
                    }
                    // If the request explicitly passed an empty array "tools": [], honor it (user disabled all tools)
                    if (!request["tools"].is_array() || !request["tools"].empty()) {
                        json mcp_tools = moecher::mcp::MCPManager::instance().get_openai_tools_schema();
                        for (const auto& mt : mcp_tools) {
                            tools.push_back(mt);
                        }
                    }
                } else if (request.contains("enable_tools") && request["enable_tools"].is_boolean()) {
                    if (request["enable_tools"].get<bool>()) {
                        tools = resolve_canonical_tools("default");
                        json mcp_tools = moecher::mcp::MCPManager::instance().get_openai_tools_schema();
                        for (const auto& mt : mcp_tools) {
                            tools.push_back(mt);
                        }
                    }
                } else {
                    // Fallback to active tools configured via /api/system/configure
                    tools = g_current_active_tools;
                }
            }

            if (request.contains("enable_thinking") && request["enable_thinking"].is_boolean()) {
                enable_thinking = request["enable_thinking"].get<bool>();
            }

            engine.enable_tree_mtp_ = engine.default_enable_tree_mtp_;
            engine.tree_nodes_ = engine.default_tree_nodes_;
            engine.enable_mtp_ = engine.default_enable_mtp_;
            engine.mtp_k_ = engine.default_mtp_k_;
            engine.enable_spec_ = engine.default_enable_spec_;
            if (request.contains("enable_spec") && request["enable_spec"].is_boolean()) {
                engine.enable_spec_ = request["enable_spec"].get<bool>();
            }
            if (request.contains("enable_mtp") && request["enable_mtp"].is_boolean()) {
                engine.enable_mtp_ = request["enable_mtp"].get<bool>();
            } else if (request.contains("mtp") && request["mtp"].is_boolean()) {
                engine.enable_mtp_ = request["mtp"].get<bool>();
            }
            if (request.contains("mtp_k") && request["mtp_k"].is_number_integer()) {
                engine.mtp_k_ = std::clamp(request["mtp_k"].get<int>(), 0, 8);
            }
            if (request.contains("tree_mtp") && request["tree_mtp"].is_boolean()) {
                engine.enable_tree_mtp_ = request["tree_mtp"].get<bool>();
            } else if (request.contains("enable_tree_mtp") && request["enable_tree_mtp"].is_boolean()) {
                engine.enable_tree_mtp_ = request["enable_tree_mtp"].get<bool>();
            }
            if (request.contains("tree_nodes") && request["tree_nodes"].is_number_integer()) {
                engine.tree_nodes_ = std::clamp(request["tree_nodes"].get<int>(), 1, 4);
            } else if (request.contains("tree_branches") && request["tree_branches"].is_number_integer()) {
                engine.tree_nodes_ = std::clamp(request["tree_branches"].get<int>(), 1, 4);
            }

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

            bool req_has_image = false;
            if (messages.is_array()) {
                for (const auto& msg : messages) {
                    if (message_has_image(msg)) { req_has_image = true; break; }
                }
            }
            bool had_image_originally = req_has_image;

            if (engine.has_vision_delegate() && req_has_image && messages.is_array()) {
                std::lock_guard<std::mutex> lock(g_engine_mutex);
                for (size_t msg_idx = 0; msg_idx < messages.size(); ++msg_idx) {
                    auto& msg = messages[msg_idx];
                    if (message_has_image(msg)) {
                        std::string cur_text = get_message_content_string(msg);
                        bool already_has_perception = (cur_text.find("<visual_perception>") != std::string::npos);
                        bool is_historical_turn = (msg_idx + 1 < messages.size() && messages[msg_idx + 1].value("role", "") == "assistant");

                        if (!already_has_perception && !is_historical_turn) {
                            std::string img_b64 = extract_image_b64_from_message(msg);
                            if (!img_b64.empty()) {
                                LOG_INFO("[Vision Delegation] Delegating visual perception (%zu b64 bytes) to Qwen2.5-VL...", img_b64.size());
                                std::string perception = engine.vision_delegate_engine_->describe_image(img_b64, cur_text);
                                if (!perception.empty()) {
                                    if (cur_text.empty()) {
                                        cur_text = "Please describe the image in detail.";
                                    }
                                    msg["content"] = "[Visual Perception Subsystem Analysis]:\n<visual_perception>\n" + perception + "\n</visual_perception>\n\nBased on the visual observation above, answer the user request:\n" + cur_text;
                                    if (msg.contains("image")) msg.erase("image");
                                    if (msg.contains("image_url")) msg.erase("image_url");
                                    LOG_INFO("[Vision Delegation] Injected perception (%zu chars)", perception.size());
                                }
                            }
                        } else {
                            // Historical message or already processed: strip raw image to prevent re-delegation and save memory/context
                            if (!cur_text.empty()) {
                                msg["content"] = cur_text;
                            }
                            if (msg.contains("image")) msg.erase("image");
                            if (msg.contains("image_url")) msg.erase("image_url");
                        }
                    }
                }
                // Re-evaluate req_has_image now that delegated images are in text format
                req_has_image = false;
                for (const auto& msg : messages) {
                    if (message_has_image(msg)) { req_has_image = true; break; }
                }
            }

            if ((req_has_image || had_image_originally) && !request.contains("enable_thinking") && !request.contains("thinking") && !request.contains("reasoning_effort")) {
                enable_thinking = false;
                max_thinking_tokens = 0;
            }
            if ((req_has_image || had_image_originally) && !request.contains("tools")) {
                tools = json();
            }

            if (!enable_thinking) {
                max_thinking_tokens = 0;
            }

            int execution_timeout_ms = 60000;
            if (request.contains("execution_timeout_ms") && request["execution_timeout_ms"].is_number()) {
                execution_timeout_ms = request["execution_timeout_ms"].get<int>();
            } else if (request.contains("execution_timeout_sec") && request["execution_timeout_sec"].is_number()) {
                execution_timeout_ms = request["execution_timeout_sec"].get<int>() * 1000;
            }
            if (execution_timeout_ms <= 0) execution_timeout_ms = 60000;

            bool require_external_authorization = request.value("require_external_authorization", true);
            bool workspace_boundary_enforced = request.value("workspace_boundary_enforced", true);
            std::vector<std::string> authorized_paths;
            if (request.contains("authorized_paths") && request["authorized_paths"].is_array()) {
                for (const auto& ap : request["authorized_paths"]) {
                    if (ap.is_string()) authorized_paths.push_back(ap.get<std::string>());
                }
            }

            bool client_tool_exec = false;
            if (req.has_header("X-Client-Tool-Execution") && req.get_header_value("X-Client-Tool-Execution") == "true") {
                client_tool_exec = true;
            }
            if (request.contains("client_tool_execution") && request["client_tool_execution"].is_boolean()) {
                client_tool_exec = request["client_tool_execution"].get<bool>();
            }
            bool do_server_exec = g_server_exec && (!client_tool_exec || g_headless_browsing);

            int req_max_tool_rounds = g_max_tool_rounds;
            if (request.contains("max_tool_rounds") && request["max_tool_rounds"].is_number_integer()) {
                req_max_tool_rounds = std::max(1, request["max_tool_rounds"].get<int>());
            }

            bool bypass_prefix_cache = false;
            if (req.has_header("X-Bypass-Cache") && req.get_header_value("X-Bypass-Cache") == "true") {
                bypass_prefix_cache = true;
            }
            if (request.contains("no_cache") && request["no_cache"].is_boolean()) {
                bypass_prefix_cache = request["no_cache"].get<bool>();
            } else if (request.contains("bypass_cache") && request["bypass_cache"].is_boolean()) {
                bypass_prefix_cache = request["bypass_cache"].get<bool>();
            }

            LOG_INFO("Reasoning effort: %s, Thinking: %s, Thinking budget: %d, Tools: %zu, Server-Exec: %s (client_tool_exec: %s, headless: %s), Max-Tool-Rounds: %d, Timeout: %dms, Bypass-Cache: %s",
                     reasoning_effort.c_str(), enable_thinking ? "enabled" : "disabled", max_thinking_tokens,
                     tools.size(), (do_server_exec ? "enabled" : "disabled"), client_tool_exec ? "true" : "false", g_headless_browsing ? "true" : "false", req_max_tool_rounds, execution_timeout_ms,
                     bypass_prefix_cache ? "true" : "false");

            std::string req_id = "chatcmpl-moecher-" + std::to_string(++g_request_counter);
            std::string model_id = engine.get_model_id();

            // ── Zero Decode Bypass (ZDB) Handling in /v1/chat/completions ───────────
            bool req_zdb = request.value("zero_decode_bypass", false) || request.value("zdb", false);
            std::vector<std::string> zdb_choices;
            auto extract_choices = [&](const std::string& key) {
                if (request.contains(key) && request[key].is_array()) {
                    for (const auto& item : request[key]) {
                        if (item.is_string()) {
                            std::string s = item.get<std::string>();
                            if (!s.empty()) zdb_choices.push_back(s);
                        } else if (item.is_object() && item.contains("text") && item["text"].is_string()) {
                            std::string s = item["text"].get<std::string>();
                            if (!s.empty()) zdb_choices.push_back(s);
                        }
                    }
                }
            };
            extract_choices("decision_choices");
            extract_choices("classification_choices");
            extract_choices("candidate_labels");
            extract_choices("zdb_choices");
            if (zdb_choices.empty()) {
                extract_choices("choices");
            }

            if (req_zdb) {
                if (zdb_choices.empty()) {
                    res.status = 400;
                    res.set_content("{\"error\":\"zero_decode_bypass requires a non-empty 'choices', 'decision_choices', or 'candidate_labels' array.\"}", "application/json");
                    return;
                }

                bool auto_fallback = request.value("auto_fallback", true);
                float min_confidence = request.value("min_confidence", 0.70f);
                float max_top1_gap = request.value("max_top1_gap", 1.5f);
                std::string zdb_format = request.value("format", "choice");

                MoecherEngine::ZeroDecodeResult zdb_res;
                {
                    std::lock_guard<std::mutex> lock(g_engine_mutex);
                    // For ZDB in chat completions, apply chat template with thinking disabled for step-0 logit extraction
                    auto mm_prompt = apply_chat_template_multimodal(messages, engine.tokenizer_, false, "none", json(), false);
                    std::vector<int> prompt = mm_prompt.prompt;
                    zdb_res = engine.execute_zero_decode_bypass(prompt, zdb_choices, temperature, zdb_format, bypass_prefix_cache, min_confidence, max_top1_gap);
                }

                if (auto_fallback && !zdb_res.gate_passed) {
                    LOG_INFO("[ZDB-Arbiter] Gate REJECTED System 1 fast path (top1_gap=%.2f, conf=%.2f [min=%.2f], global_argmax='%s' [id=%d]). Falling back to System 2 autoregressive decode.",
                             zdb_res.top1_gap, zdb_res.confidence, min_confidence,
                             zdb_res.global_argmax_token_str.c_str(), zdb_res.global_argmax_token_id);
                    // Seamlessly fall through to standard stream/non-stream decode loops below!
                } else {
                    if (stream) {
                        time_t now_sec = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                        json usage_obj = {
                            {"prompt_tokens", zdb_res.prompt_tokens},
                            {"completion_tokens", 0},
                            {"total_tokens", zdb_res.prompt_tokens},
                            {"evaluated_prompt_tokens", zdb_res.prompt_tokens},
                            {"prefill_sec", zdb_res.prefill_sec},
                            {"prefill_tps", (zdb_res.prefill_sec > 0.0) ? (double)zdb_res.prompt_tokens / zdb_res.prefill_sec : 0.0},
                            {"decode_sec", 0.0},
                            {"decode_tps", 0.0}
                        };
                        json zdb_obj = {
                            {"enabled", true},
                            {"selected_choice", zdb_res.winning_choice},
                            {"winning_index", zdb_res.winning_index},
                            {"confidence", zdb_res.confidence},
                            {"decode_passes", 0},
                            {"probabilities", zdb_res.probabilities},
                            {"arbiter", {
                                {"gate_passed", zdb_res.gate_passed},
                                {"system_path", zdb_res.system_path},
                                {"top1_gap", zdb_res.top1_gap},
                                {"candidate_margin", zdb_res.candidate_margin},
                                {"is_choice_top1", zdb_res.is_choice_top1}
                            }},
                            {"global_top1", {
                                {"token_id", zdb_res.global_argmax_token_id},
                                {"token", zdb_res.global_argmax_token_str},
                                {"logit", zdb_res.global_argmax_logit},
                                {"is_candidate", zdb_res.is_choice_top1}
                            }},
                            {"timing", {
                                {"prefill_ms", zdb_res.prefill_sec * 1000.0},
                                {"total_ms", zdb_res.total_sec * 1000.0}
                            }}
                        };
                        json chunk1 = {
                            {"id", req_id},
                            {"object", "chat.completion.chunk"},
                            {"created", now_sec},
                            {"model", model_id},
                            {"choices", json::array({
                                {{"index", 0}, {"delta", {{"role", "assistant"}, {"content", zdb_res.rendered_output}}}, {"finish_reason", nullptr}}
                            })}
                        };
                        json chunk2 = {
                            {"id", req_id},
                            {"object", "chat.completion.chunk"},
                            {"created", now_sec},
                            {"model", model_id},
                            {"choices", json::array({
                                {{"index", 0}, {"delta", json::object()}, {"finish_reason", "stop"}}
                            })},
                            {"usage", usage_obj},
                            {"zero_decode_bypass", zdb_obj}
                        };
                        std::string sse = "data: " + chunk1.dump() + "\n\n" +
                                          "data: " + chunk2.dump() + "\n\n" +
                                          "data: [DONE]\n\n";
                        res.set_content(sse, "text/event-stream");
                        return;
                    } else {
                        json choice = {
                            {"index", 0},
                            {"message", {
                                {"role", "assistant"},
                                {"content", zdb_res.rendered_output}
                            }},
                            {"finish_reason", "stop"}
                        };

                        json response = {
                            {"id", req_id},
                            {"object", "chat.completion"},
                            {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                            {"model", model_id},
                            {"choices", {choice}},
                            {"usage", {
                                {"prompt_tokens", zdb_res.prompt_tokens},
                                {"completion_tokens", 0},
                                {"total_tokens", zdb_res.prompt_tokens},
                                {"evaluated_prompt_tokens", zdb_res.prompt_tokens},
                                {"prefill_sec", zdb_res.prefill_sec},
                                {"prefill_tps", (zdb_res.prefill_sec > 0.0) ? (double)zdb_res.prompt_tokens / zdb_res.prefill_sec : 0.0},
                                {"decode_sec", 0.0},
                                {"decode_tps", 0.0}
                            }},
                            {"zero_decode_bypass", {
                                {"enabled", true},
                                {"selected_choice", zdb_res.winning_choice},
                                {"winning_index", zdb_res.winning_index},
                                {"confidence", zdb_res.confidence},
                                {"decode_passes", 0},
                                {"probabilities", zdb_res.probabilities},
                                {"arbiter", {
                                    {"gate_passed", zdb_res.gate_passed},
                                    {"system_path", zdb_res.system_path},
                                    {"top1_gap", zdb_res.top1_gap},
                                    {"candidate_margin", zdb_res.candidate_margin},
                                    {"is_choice_top1", zdb_res.is_choice_top1}
                                }},
                                {"global_top1", {
                                    {"token_id", zdb_res.global_argmax_token_id},
                                    {"token", zdb_res.global_argmax_token_str},
                                    {"logit", zdb_res.global_argmax_logit},
                                    {"is_candidate", zdb_res.is_choice_top1}
                                }},
                                {"timing", {
                                    {"prefill_ms", zdb_res.prefill_sec * 1000.0},
                                    {"total_ms", zdb_res.total_sec * 1000.0}
                                }}
                            }}
                        };
                        res.set_content(response.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
                        return;
                    }
                }
            }

            if (stream) {
                // Fast SSE delta writer avoiding heap-allocated JSON ASTs
                auto send_sse_delta = [](httplib::DataSink& sink, const std::string& req_id, const std::string& model_id,
                                         const std::string& created_str, const char* field_name, const std::string& text) -> bool {
                    std::string sse;
                    sse.reserve(128 + text.size() * 2);
                    sse.append("data: {\"id\":\"");
                    sse.append(req_id);
                    sse.append("\",\"object\":\"chat.completion.chunk\",\"created\":");
                    sse.append(created_str);
                    sse.append(",\"model\":\"");
                    sse.append(model_id);
                    sse.append("\",\"choices\":[{\"index\":0,\"delta\":{\"");
                    sse.append(field_name);
                    sse.append("\":");
                    sse.append(json(text).dump());
                    sse.append("},\"finish_reason\":null}]}\n\n");
                    return sink.write(sse.data(), sse.size()) && !g_stop_requested.load();
                };

                // SSE streaming
                res.set_chunked_content_provider(
                    "text/event-stream",
                    [&engine, messages, tools, max_tokens, temperature, req_id, repetition_penalty, enable_thinking, max_thinking_tokens, top_p, min_p, top_k, reasoning_effort, model_id, execution_timeout_ms, require_external_authorization, workspace_boundary_enforced, authorized_paths, do_server_exec, req_max_tool_rounds, busy_guard, send_sse_delta, bypass_prefix_cache](size_t offset, httplib::DataSink &sink) {
                        if (offset > 0) return false;
                        std::lock_guard<std::mutex> lock(g_engine_mutex);

                        time_t created_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                        std::string created_time_str = std::to_string(created_time);

                        try {

                        json initial_chunk = {
                            {"id", req_id},
                            {"object", "chat.completion.chunk"},
                            {"created", created_time},
                            {"model", model_id},
                            {"choices", {{
                                {"index", 0},
                                {"delta", {{"role", "assistant"}}},
                                {"finish_reason", nullptr}
                            }}}
                        };
                        std::string sse = "data: " + initial_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                        sink.write(sse.data(), sse.size());

                        json current_messages = messages;
                        int max_tool_rounds = (do_server_exec && !tools.empty() && g_enable_tools) ? req_max_tool_rounds : 1;
                        std::string final_finish_reason = "stop";
                        std::string last_completed_content;
                        std::vector<json> round_new_tool_msgs;

                        int max_allowed_prompt = engine.cfg_.max_seq_len - max_tokens;
                        if (max_allowed_prompt < 1024) max_allowed_prompt = engine.cfg_.max_seq_len - 256;

                        for (int round = 0; round < max_tool_rounds; round++) {
                            std::vector<int> prompt;
                            bool is_continuation = false;
                            if (round == 0) {
                                is_continuation = check_and_build_continuation(engine, current_messages, s_last_conv_messages, enable_thinking, reasoning_effort, prompt);
                                if (is_continuation && prompt.size() > (size_t)max_allowed_prompt) {
                                    LOG_WARN("[Context] Continuation prompt length (%zu) exceeds context budget (%d). Forcing sliding-window pruning.",
                                             prompt.size(), max_allowed_prompt);
                                    is_continuation = false;
                                }
                            } else if (round > 0 && !round_new_tool_msgs.empty()) {
                                prompt = build_continuation_prompt(engine, round_new_tool_msgs, enable_thinking, reasoning_effort);
                                is_continuation = true;
                                if (prompt.size() > (size_t)max_allowed_prompt) {
                                    LOG_WARN("[Context] Tool continuation length (%zu) exceeds context budget (%d). Forcing sliding-window pruning.",
                                             prompt.size(), max_allowed_prompt);
                                    is_continuation = false;
                                }
                                round_new_tool_msgs.clear();
                            }
                            MultimodalPrompt mm_prompt;
                            const __nv_bfloat16* d_vis_out = nullptr;
                            if (!is_continuation) {
                                current_messages = prune_messages_to_fit(current_messages, engine.tokenizer_, max_allowed_prompt, enable_thinking, reasoning_effort, tools);
                                mm_prompt = apply_chat_template_multimodal(current_messages, engine.tokenizer_, enable_thinking, reasoning_effort, tools, engine.has_vision());
                                prompt = mm_prompt.prompt;
                                if (engine.has_vision() && !mm_prompt.image_b64.empty() && mm_prompt.visual_start_pos >= 0) {
                                    if (engine.ensure_vision_loaded()) {
                                        moecher::vision::ProcessedImage proc_img;
                                        int img_target_sz = engine.vision_tower_->image_size_;
                                        LOG_INFO("[Vision] Preprocessing image attachment for multimodal inference (%zu bytes b64, target_size=%d)...",
                                                 mm_prompt.image_b64.size(), img_target_sz);
                                        if (moecher::vision::preprocess_image_from_base64(mm_prompt.image_b64, proc_img, img_target_sz)) {
                                            d_vis_out = engine.vision_tower_->forward(proc_img, engine.main_stream_);
                                            LOG_INFO("[Vision] Vision Tower forward pass complete: 576 visual tokens projected to hidden size %d.", engine.cfg_.hidden_size);
                                        } else {
                                            LOG_WARN("[Vision] Failed to decode/preprocess image attachment!");
                                        }
                                    }
                                }
                            }
                            LOG_INFO("STREAM PROMPT [round %d] (len=%zu, continuation=%s, visual=%s)",
                                     round + 1, prompt.size(), is_continuation ? "true" : "false", (d_vis_out != nullptr) ? "active" : "none");

                            std::string round_content;
                            std::string round_reasoning;
                            ToolTagStreamFilter content_filter;
                            ToolTagStreamFilter reasoning_filter;

                            auto status_cb = [&](const std::string& status_msg) -> bool {
                                return send_sse_delta(sink, req_id, model_id, created_time_str, "status", status_msg);
                            };
                            content_filter.on_status = status_cb;
                            reasoning_filter.on_status = status_cb;

                            bool conv_has_image = false;
                            for (const auto& m : current_messages) {
                                if (message_has_image(m)) { conv_has_image = true; break; }
                            }
                            bool req_thinking = enable_thinking && !conv_has_image;
                            int req_thinking_tokens = req_thinking ? max_thinking_tokens : 0;

                            engine.generate(prompt, max_tokens, temperature, [&](const std::string& text, bool is_reasoning) -> bool {
                                if (g_stop_requested.load()) return false;
                                if (text.empty()) return true;

                                if (is_reasoning) {
                                    round_reasoning += text;
                                    if (g_enable_tools && !tools.empty()) {
                                        return reasoning_filter.feed(text, [&](const std::string& safe_text) -> bool {
                                            return send_sse_delta(sink, req_id, model_id, created_time_str, "reasoning_content", safe_text);
                                        });
                                    } else {
                                        return send_sse_delta(sink, req_id, model_id, created_time_str, "reasoning_content", text);
                                    }
                                } else {
                                    round_content += text;
                                    if (g_enable_tools && !tools.empty()) {
                                        return content_filter.feed(text, [&](const std::string& safe_text) -> bool {
                                            return send_sse_delta(sink, req_id, model_id, created_time_str, "content", safe_text);
                                        });
                                    } else {
                                        return send_sse_delta(sink, req_id, model_id, created_time_str, "content", text);
                                    }
                                }
                            }, repetition_penalty, req_thinking, req_thinking_tokens, top_p, min_p, top_k,
                               d_vis_out, mm_prompt.visual_start_pos, mm_prompt.visual_num_tokens, bypass_prefix_cache);

                            if (d_vis_out != nullptr && engine.gpu_caps_.is_vram_constrained) {
                                engine.unload_vision_tower();
                            }

                            final_finish_reason = engine.last_finish_reason_.empty() ? "stop" : engine.last_finish_reason_;

                            if (g_enable_tools && !tools.empty()) {
                                reasoning_filter.flush_remaining([&](const std::string& safe_text) -> bool {
                                    return send_sse_delta(sink, req_id, model_id, created_time_str, "reasoning_content", safe_text);
                                });
                                content_filter.flush_remaining([&](const std::string& safe_text) -> bool {
                                    return send_sse_delta(sink, req_id, model_id, created_time_str, "content", safe_text);
                                });
                            }

                            if (!g_enable_tools || tools.empty()) {
                                break;
                            }

                            std::string clean_content;
                            std::vector<moecher::tooling::ToolCall> round_tool_calls;
                            moecher::tooling::extract_tool_calls(round_content, clean_content, round_tool_calls);

                            if (!round_reasoning.empty()) {
                                std::string clean_reasoning;
                                std::vector<moecher::tooling::ToolCall> reasoning_calls;
                                moecher::tooling::extract_tool_calls(round_reasoning, clean_reasoning, reasoning_calls);
                                if (round_tool_calls.empty() && !reasoning_calls.empty()) {
                                    LOG_INFO("Extracted %zu tool call(s) from reasoning stream", reasoning_calls.size());
                                    round_tool_calls = reasoning_calls;
                                }
                                round_reasoning = clean_reasoning;
                            }

                            if (round_tool_calls.empty()) {
                                last_completed_content = clean_content;
                                std::string unstreamed_content = clean_content;
                                if (!content_filter.streamed_clean_text.empty()) {
                                    if (clean_content.size() >= content_filter.streamed_clean_text.size() &&
                                        clean_content.compare(0, content_filter.streamed_clean_text.size(), content_filter.streamed_clean_text) == 0) {
                                        unstreamed_content = clean_content.substr(content_filter.streamed_clean_text.size());
                                    } else if (clean_content.size() <= content_filter.streamed_clean_text.size()) {
                                        unstreamed_content.clear();
                                    }
                                }
                                if (!unstreamed_content.empty()) {
                                    json delta_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{"content", unstreamed_content}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_chunk = "data: " + delta_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_chunk.data(), sse_chunk.size());
                                }
                                break;
                            }

                            // Stream any unstreamed leading text from model before tool call
                            std::string unstreamed_intro = clean_content;
                            if (!content_filter.streamed_clean_text.empty()) {
                                if (clean_content.size() >= content_filter.streamed_clean_text.size() &&
                                    clean_content.compare(0, content_filter.streamed_clean_text.size(), content_filter.streamed_clean_text) == 0) {
                                    unstreamed_intro = clean_content.substr(content_filter.streamed_clean_text.size());
                                } else if (clean_content.size() <= content_filter.streamed_clean_text.size()) {
                                    unstreamed_intro.clear();
                                }
                            }
                            if (!unstreamed_intro.empty()) {
                                std::string field_name = enable_thinking ? "reasoning_content" : "content";
                                json intro_chunk = {
                                    {"id", req_id},
                                    {"object", "chat.completion.chunk"},
                                    {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                    {"model", model_id},
                                    {"choices", {{
                                        {"index", 0},
                                        {"delta", {{field_name, unstreamed_intro + "\n\n"}}},
                                        {"finish_reason", nullptr}
                                    }}}
                                };
                                std::string sse_intro = "data: " + intro_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                sink.write(sse_intro.data(), sse_intro.size());
                            }

                            if (!do_server_exec) {
                                final_finish_reason = "tool_calls";
                                json tc_arr = json::array();
                                for (size_t idx = 0; idx < round_tool_calls.size(); idx++) {
                                    const auto& tc = round_tool_calls[idx];
                                    tc_arr.push_back({
                                        {"index", (int)idx},
                                        {"id", tc.id},
                                        {"type", tc.type},
                                        {"function", {{"name", tc.name}, {"arguments", tc.arguments}}}
                                    });
                                }
                                json tc_chunk = {
                                    {"id", req_id},
                                    {"object", "chat.completion.chunk"},
                                    {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                    {"model", model_id},
                                    {"choices", {{
                                        {"index", 0},
                                        {"delta", {{"tool_calls", tc_arr}}},
                                        {"finish_reason", "tool_calls"}
                                    }}}
                                };
                                std::string sse_chunk = "data: " + tc_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                sink.write(sse_chunk.data(), sse_chunk.size());
                                break;
                            }

                            // Server-side tool execution
                            bool can_execute_all = true;
                            json assistant_msg = {{"role", "assistant"}, {"content", round_content}};
                            json tc_json_arr = json::array();
                            for (const auto& tc : round_tool_calls) {
                                tc_json_arr.push_back({
                                    {"id", tc.id},
                                    {"type", tc.type},
                                    {"function", {{"name", tc.name}, {"arguments", tc.arguments}}}
                                });
                            }
                            assistant_msg["tool_calls"] = tc_json_arr;
                            current_messages.push_back(assistant_msg);

                            std::string tool_delta_field = enable_thinking ? "reasoning_content" : "content";

                            auto is_path_authorized = [&](const std::string& p) -> bool {
                                if (!require_external_authorization && !workspace_boundary_enforced) return true;
                                if (moecher::tooling::is_path_inside_workspace(p)) return true;
                                for (const auto& ap : authorized_paths) {
                                    if (ap == "*" || ap == p) return true;
                                    if (!ap.empty() && moecher::tooling::is_path_inside_workspace(p, ap)) return true;
                                }
                                return false;
                            };

                            for (const auto& tc : round_tool_calls) {
                                std::string exec_output;
                                if (tc.name == "web_search" || tc.name == "google_search") {
                                    std::string query;
                                    int num_results = 5;
                                    std::string site = "";
                                    std::string provider = "";
                                    std::string api_key = "";
                                    std::string cx = "";
                                    std::string searx_url = "";
                                    try {
                                        json args = json::parse(tc.arguments);
                                        query = args.value("query", "");
                                        num_results = args.value("num_results", 5);
                                        site = args.value("site", "");
                                        provider = args.value("provider", "");
                                        api_key = args.value("api_key", "");
                                        cx = args.value("cx", "");
                                        searx_url = args.value("searxng_url", "");
                                    } catch (...) { query = tc.arguments; }

                                    std::string active_card =
                                        "\n\n<div class=\"tool-activity-block active\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"thinking-spinner\">progress_activity</span>\n"
                                        "  <span class=\"tool-action-label\">Searching Web</span>\n"
                                        "  <span class=\"tool-target-subtle\">" + query + "</span>\n"
                                        "</div>\n\n";

                                    json info_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, active_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_info = "data: " + info_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_info.data(), sse_info.size());

                                    LOG_INFO("Streaming Tool Exec: web_search('%s', num=%d, site='%s', provider='%s')", query.c_str(), num_results, site.c_str(), provider.c_str());
                                    auto doc = moecher::tooling::web_search_full(query, num_results, site, provider, api_key, cx, searx_url);
                                    exec_output = doc.clean_text;

                                    // Stream retrieved search cards to the frontend
                                    json doc_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {
                                                {"retrieved_document", {
                                                    {"id", tc.id},
                                                    {"url", doc.url},
                                                    {"title", doc.title.empty() ? ("Web Search: " + query) : doc.title},
                                                    {"html", doc.raw_html},
                                                    {"snippet", (doc.clean_text.size() > 300 ? doc.clean_text.substr(0, 300) + "..." : doc.clean_text)}
                                                }}
                                            }},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_doc = "data: " + doc_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_doc.data(), sse_doc.size());

                                    std::string completed_card =
                                        "\n\n<div class=\"tool-activity-block completed\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"material-symbols-outlined tool-done-icon\">search</span>\n"
                                        "  <span class=\"tool-action-label\">Web Searched</span>\n"
                                        "  <span class=\"tool-target-subtle\">" + query + "</span>\n"
                                        "</div>\n\n";

                                    json comp_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, completed_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_comp = "data: " + comp_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_comp.data(), sse_comp.size());

                                } else if (tc.name == "fetch_url") {
                                    std::string url;
                                    std::string mode = "text";
                                    std::string pattern = "";
                                    int offset = 0;
                                    int max_chars = 4000;
                                    try {
                                        json args = json::parse(tc.arguments);
                                        url = args.value("url", "");
                                        mode = args.value("mode", "text");
                                        pattern = args.value("pattern", "");
                                        offset = args.value("offset", 0);
                                        max_chars = args.value("max_chars", 4000);
                                    } catch (...) { url = tc.arguments; }

                                    std::string active_card =
                                        "\n\n<div class=\"tool-activity-block active\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"thinking-spinner\">progress_activity</span>\n"
                                        "  <span class=\"tool-action-label\">Searching Web</span>\n"
                                        "  <span class=\"tool-target-subtle\">" + url + "</span>\n"
                                        "</div>\n\n";

                                    json info_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, active_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_info = "data: " + info_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_info.data(), sse_info.size());

                                    LOG_INFO("Streaming Tool Exec: fetch_url('%s', mode='%s', pattern='%s')", url.c_str(), mode.c_str(), pattern.c_str());
                                    auto doc = moecher::tooling::fetch_url_full(url, std::min(execution_timeout_ms, 30000), max_chars, mode, pattern, offset);
                                    exec_output = doc.clean_text;

                                    // Stream the retrieved document metadata & HTML to the frontend for the Retrieved Documents panel
                                    json doc_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {
                                                {"retrieved_document", {
                                                    {"id", tc.id},
                                                    {"url", doc.url},
                                                    {"title", doc.title.empty() ? doc.url : doc.title},
                                                    {"html", doc.raw_html},
                                                    {"snippet", (doc.clean_text.size() > 300 ? doc.clean_text.substr(0, 300) + "..." : doc.clean_text)}
                                                }}
                                            }},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_doc = "data: " + doc_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_doc.data(), sse_doc.size());

                                    std::string completed_card =
                                        "\n\n<div class=\"tool-activity-block completed\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"material-symbols-outlined tool-done-icon\">travel_explore</span>\n"
                                        "  <span class=\"tool-action-label\">Searched</span>\n"
                                        "  <span class=\"tool-target-subtle\">" + doc.url + "</span>\n"
                                        "</div>\n\n";

                                    json comp_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, completed_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_comp = "data: " + comp_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_comp.data(), sse_comp.size());

                                } else if (tc.name == "read_file") {
                                    std::string path;
                                    int start_line = 1;
                                    int end_line = -1;
                                    try {
                                        json args = json::parse(tc.arguments);
                                        path = args.value("path", "");
                                        start_line = args.value("start_line", 1);
                                        end_line = args.value("end_line", -1);
                                    } catch (...) { path = tc.arguments; }

                                    std::string active_card =
                                        "\n\n<div class=\"tool-activity-block active\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"thinking-spinner\">progress_activity</span>\n"
                                        "  <span class=\"tool-action-label\">Reading file</span>\n"
                                        "  <span class=\"tool-target-subtle\">" + path + "</span>\n"
                                        "</div>\n\n";

                                    json info_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, active_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_info = "data: " + info_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_info.data(), sse_info.size());

                                    LOG_INFO("Streaming Tool Exec: read_file('%s', start=%d, end=%d)", path.c_str(), start_line, end_line);

                                    if (!is_path_authorized(path)) {
                                        exec_output = "[Authorization Required: The path '" + path + "' is outside the current workspace directory (" + moecher::tooling::get_workspace_directory() + "). Action requires explicit user authorization for path: " + path + "]";
                                        json auth_chunk = {
                                            {"id", req_id},
                                            {"object", "chat.completion.chunk"},
                                            {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                            {"model", model_id},
                                            {"choices", {{
                                                {"index", 0},
                                                {"delta", {
                                                    {"authorization_required", {
                                                        {"tool", "read_file"},
                                                        {"path", path},
                                                        {"id", tc.id}
                                                    }}
                                                }},
                                                {"finish_reason", nullptr}
                                            }}}
                                        };
                                        std::string sse_auth = "data: " + auth_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                        sink.write(sse_auth.data(), sse_auth.size());
                                    } else {
                                        exec_output = moecher::tooling::read_file(path, start_line, end_line);
                                    }

                                    std::string completed_card =
                                        "\n\n<div class=\"tool-activity-block completed\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"material-symbols-outlined tool-done-icon\">description</span>\n"
                                        "  <span class=\"tool-action-label\">Read file</span>\n"
                                        "  <span class=\"tool-target-subtle\">" + path + "</span>\n"
                                        "</div>\n\n";

                                    json comp_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, completed_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_comp = "data: " + comp_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_comp.data(), sse_comp.size());

                                } else if (tc.name == "write_file") {
                                    std::string path;
                                    std::string content;
                                    bool overwrite = true;
                                    try {
                                        json args = json::parse(tc.arguments);
                                        path = args.value("path", "");
                                        content = args.value("content", "");
                                        overwrite = args.value("overwrite", true);
                                    } catch (...) { path = tc.arguments; }

                                    std::string active_card =
                                        "\n\n<div class=\"tool-activity-block active\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"thinking-spinner\">progress_activity</span>\n"
                                        "  <span class=\"tool-action-label\">Writing file</span>\n"
                                        "  <span class=\"tool-target-subtle\">" + path + "</span>\n"
                                        "</div>\n\n";

                                    json info_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, active_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_info = "data: " + info_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_info.data(), sse_info.size());

                                    LOG_INFO("Streaming Tool Exec: write_file('%s', %zu bytes)", path.c_str(), content.size());

                                    if (!is_path_authorized(path)) {
                                        exec_output = "[Authorization Required: The path '" + path + "' is outside the current workspace directory (" + moecher::tooling::get_workspace_directory() + "). Action requires explicit user authorization for path: " + path + "]";
                                        json auth_chunk = {
                                            {"id", req_id},
                                            {"object", "chat.completion.chunk"},
                                            {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                            {"model", model_id},
                                            {"choices", {{
                                                {"index", 0},
                                                {"delta", {
                                                    {"authorization_required", {
                                                        {"tool", "write_file"},
                                                        {"path", path},
                                                        {"id", tc.id}
                                                    }}
                                                }},
                                                {"finish_reason", nullptr}
                                            }}}
                                        };
                                        std::string sse_auth = "data: " + auth_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                        sink.write(sse_auth.data(), sse_auth.size());
                                    } else {
                                        exec_output = moecher::tooling::write_file(path, content, overwrite);
                                    }

                                    std::string completed_card =
                                        "\n\n<div class=\"tool-activity-block completed\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"material-symbols-outlined tool-done-icon\">edit_note</span>\n"
                                        "  <span class=\"tool-action-label\">Wrote file</span>\n"
                                        "  <span class=\"tool-target-subtle\">" + path + "</span>\n"
                                        "</div>\n\n";

                                    json comp_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, completed_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_comp = "data: " + comp_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_comp.data(), sse_comp.size());

                                } else if (tc.name == "edit_file") {
                                    std::string path;
                                    std::string target_content;
                                    std::string replacement_content;
                                    try {
                                        json args = json::parse(tc.arguments);
                                        path = args.value("path", "");
                                        target_content = args.value("target_content", "");
                                        replacement_content = args.value("replacement_content", "");
                                    } catch (...) { path = tc.arguments; }

                                    std::string active_card =
                                        "\n\n<div class=\"tool-activity-block active\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"thinking-spinner\">progress_activity</span>\n"
                                        "  <span class=\"tool-action-label\">Editing file</span>\n"
                                        "  <span class=\"tool-target-subtle\">" + path + "</span>\n"
                                        "</div>\n\n";

                                    json info_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, active_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_info = "data: " + info_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_info.data(), sse_info.size());

                                    LOG_INFO("Streaming Tool Exec: edit_file('%s')", path.c_str());

                                    if (!is_path_authorized(path)) {
                                        exec_output = "[Authorization Required: The path '" + path + "' is outside the current workspace directory (" + moecher::tooling::get_workspace_directory() + "). Action requires explicit user authorization for path: " + path + "]";
                                        json auth_chunk = {
                                            {"id", req_id},
                                            {"object", "chat.completion.chunk"},
                                            {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                            {"model", model_id},
                                            {"choices", {{
                                                {"index", 0},
                                                {"delta", {
                                                    {"authorization_required", {
                                                        {"tool", "edit_file"},
                                                        {"path", path},
                                                        {"id", tc.id}
                                                    }}
                                                }},
                                                {"finish_reason", nullptr}
                                            }}}
                                        };
                                        std::string sse_auth = "data: " + auth_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                        sink.write(sse_auth.data(), sse_auth.size());
                                    } else {
                                        exec_output = moecher::tooling::edit_file(path, target_content, replacement_content);
                                    }

                                    std::string completed_card =
                                        "\n\n<div class=\"tool-activity-block completed\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"material-symbols-outlined tool-done-icon\">edit_square</span>\n"
                                        "  <span class=\"tool-action-label\">Edited file</span>\n"
                                        "  <span class=\"tool-target-subtle\">" + path + "</span>\n"
                                        "</div>\n\n";

                                    json comp_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, completed_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_comp = "data: " + comp_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_comp.data(), sse_comp.size());

                                } else if (tc.name == "execute_command") {
                                    std::string cmd;
                                    try {
                                        json args = json::parse(tc.arguments);
                                        cmd = args.value("command", "");
                                    } catch (...) { cmd = tc.arguments; }

                                    std::string active_card =
                                        "\n\n<div class=\"tool-activity-block active\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"thinking-spinner\">progress_activity</span>\n"
                                        "  <span class=\"tool-action-label\">Running command</span>\n"
                                        "  <span class=\"tool-target-subtle\">" + cmd + "</span>\n"
                                        "</div>\n\n";

                                    json info_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, active_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_info = "data: " + info_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_info.data(), sse_info.size());

                                    LOG_INFO("Streaming Tool Exec: execute_command('%s', timeout=%dms)", cmd.c_str(), execution_timeout_ms);
                                    exec_output = moecher::tooling::execute_system_command(cmd, execution_timeout_ms);

                                    std::string completed_card =
                                        "\n\n<div class=\"tool-activity-block completed\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"material-symbols-outlined tool-done-icon\">terminal</span>\n"
                                        "  <span class=\"tool-action-label\">Executed</span>\n"
                                        "  <span class=\"tool-target-subtle\">" + cmd + "</span>\n"
                                        "</div>\n\n";

                                    json comp_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, completed_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_comp = "data: " + comp_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_comp.data(), sse_comp.size());
                                } else if (moecher::mcp::MCPManager::instance().has_tool(tc.name)) {
                                    std::string s_id, raw_tname;
                                    moecher::mcp::MCPManager::instance().has_tool(tc.name, &s_id, &raw_tname);
                                    std::string active_card =
                                        "\n\n<div class=\"tool-activity-block active mcp-tool\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"thinking-spinner\">progress_activity</span>\n"
                                        "  <span class=\"tool-action-label\">[MCP: " + s_id + "] " + raw_tname + "</span>\n"
                                        "  <span class=\"tool-target-subtle\">" + (tc.arguments.size() > 80 ? tc.arguments.substr(0, 80) + "..." : tc.arguments) + "</span>\n"
                                        "</div>\n\n";

                                    json info_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, active_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_info = "data: " + info_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_info.data(), sse_info.size());

                                    json parsed_args = json::object();
                                    try { parsed_args = json::parse(tc.arguments); } catch (...) {}
                                    LOG_INFO("Streaming Tool Exec: MCP [%s:%s]", s_id.c_str(), raw_tname.c_str());
                                    exec_output = moecher::mcp::MCPManager::instance().call_tool(tc.name, parsed_args, execution_timeout_ms);

                                    std::string completed_card =
                                        "\n\n<div class=\"tool-activity-block completed mcp-tool\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"material-symbols-outlined tool-done-icon\">hub</span>\n"
                                        "  <span class=\"tool-action-label\">[MCP: " + s_id + "] " + raw_tname + "</span>\n"
                                        "  <span class=\"tool-target-subtle\">Completed</span>\n"
                                        "</div>\n\n";

                                    json comp_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, completed_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_comp = "data: " + comp_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_comp.data(), sse_comp.size());
                                } else {
                                    can_execute_all = false;
                                    break;
                                }

                                json tool_reply = {
                                    {"role", "tool"},
                                    {"tool_call_id", tc.id},
                                    {"content", exec_output}
                                };
                                current_messages.push_back(tool_reply);
                                round_new_tool_msgs.push_back(tool_reply);
                            }

                            if (!can_execute_all) {
                                final_finish_reason = "tool_calls";
                                break;
                            }

                            if (round + 1 < max_tool_rounds) {
                                json proc_chunk = {
                                    {"id", req_id},
                                    {"object", "chat.completion.chunk"},
                                    {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                    {"model", model_id},
                                    {"choices", {{
                                        {"index", 0},
                                        {"delta", {{"processing_status", "Encoding tool output & elaborating response"}}},
                                        {"finish_reason", nullptr}
                                    }}}
                                };
                                std::string sse_proc = "data: " + proc_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                sink.write(sse_proc.data(), sse_proc.size());
                            } else {
                                LOG_WARN("Tool execution reached max rounds limit (%d). Halting loop and notifying user.", max_tool_rounds);
                                std::string warn_msg = "\n\n> ⚠️ **Maximum tool execution rounds reached (" + std::to_string(max_tool_rounds) + "/" + std::to_string(max_tool_rounds) + ").** Execution stopped.";
                                json warn_chunk = {
                                    {"id", req_id},
                                    {"object", "chat.completion.chunk"},
                                    {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                    {"model", model_id},
                                    {"choices", {{
                                        {"index", 0},
                                        {"delta", {
                                            {"content", warn_msg},
                                            {"tool_limit_reached", true},
                                            {"max_tool_rounds", max_tool_rounds}
                                        }},
                                        {"finish_reason", nullptr}
                                    }}}
                                };
                                std::string sse_warn = "data: " + warn_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                sink.write(sse_warn.data(), sse_warn.size());
                            }
                        }

                        // Final finish chunk
                        json finish = {
                            {"id", req_id},
                            {"object", "chat.completion.chunk"},
                            {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                            {"model", model_id},
                            {"choices", {{
                                {"index", 0},
                                {"delta", json::object()},
                                {"finish_reason", final_finish_reason}
                            }}},
                            {"usage", {
                                {"prompt_tokens", engine.prompt_token_count_},
                                {"completion_tokens", engine.completion_token_count_},
                                {"total_tokens", engine.prompt_token_count_ + engine.completion_token_count_},
                                {"evaluated_prompt_tokens", engine.last_prefill_tokens_},
                                {"prefill_sec", engine.last_prefill_sec_},
                                {"prefill_tps", engine.last_prefill_tps_},
                                {"decode_sec", engine.last_decode_sec_},
                                {"decode_tps", engine.last_decode_tps_},
                                {"prompt_tokens_details", {
                                    {"cached_tokens", engine.last_cached_tokens_}
                                }}
                            }}
                        };
                        sse = "data: " + finish.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                        sink.write(sse.data(), sse.size());
                        sink.write("data: [DONE]\n\n", 14);
                        sink.done();

                        // Save conversation state for turn continuation
                        json conv_to_save = current_messages;
                        if (conv_to_save.empty() || conv_to_save.back().value("role", "") != "assistant") {
                            conv_to_save.push_back({{"role", "assistant"}, {"content", last_completed_content}});
                        }
                        s_last_conv_messages = conv_to_save;

                        if (g_track_experts && !engine.expert_freq_counts_.empty()) {
                            std::string fpath = engine.model_dir_.empty() ? "expert_freq.bin" : (engine.model_dir_ + "/expert_freq.bin");
                            engine.save_expert_freq(fpath);
                            engine.save_expert_profile(engine.model_dir_);
                        }
                        return true;
                        } catch (const std::exception& e) {
                            LOG_ERROR("[Streaming] Exception during request execution: %s", e.what());
                            send_sse_delta(sink, req_id, model_id, created_time_str, "content", std::string("\n\n[Server Error: ") + e.what() + "]\n");
                            sink.done();
                            return false;
                        }
                    }
                );
            } else {
                try {
                    std::string final_response_text;
                std::vector<moecher::tooling::ToolCall> emitted_tool_calls;
                json current_messages = messages;
                std::string finish_reason = "stop";
                std::vector<json> round_new_tool_msgs;

                int max_tool_rounds = (do_server_exec && !tools.empty() && g_enable_tools) ? req_max_tool_rounds : 1;

                int max_allowed_prompt = engine.cfg_.max_seq_len - max_tokens;
                if (max_allowed_prompt < 1024) max_allowed_prompt = engine.cfg_.max_seq_len - 256;

                for (int round = 0; round < max_tool_rounds; round++) {
                    std::vector<int> prompt;
                    bool is_continuation = false;
                    if (round == 0) {
                        std::lock_guard<std::mutex> lock(g_engine_mutex);
                        is_continuation = check_and_build_continuation(engine, current_messages, s_last_conv_messages, enable_thinking, reasoning_effort, prompt);
                        if (is_continuation && prompt.size() > (size_t)max_allowed_prompt) {
                            LOG_WARN("[Context] Continuation prompt length (%zu) exceeds context budget (%d). Forcing sliding-window pruning.",
                                     prompt.size(), max_allowed_prompt);
                            is_continuation = false;
                        }
                    } else if (round > 0 && !round_new_tool_msgs.empty()) {
                        std::lock_guard<std::mutex> lock(g_engine_mutex);
                        prompt = build_continuation_prompt(engine, round_new_tool_msgs, enable_thinking, reasoning_effort);
                        is_continuation = true;
                        if (prompt.size() > (size_t)max_allowed_prompt) {
                            LOG_WARN("[Context] Tool continuation length (%zu) exceeds context budget (%d). Forcing sliding-window pruning.",
                                     prompt.size(), max_allowed_prompt);
                            is_continuation = false;
                        }
                        round_new_tool_msgs.clear();
                    }
                    MultimodalPrompt mm_prompt;
                    const __nv_bfloat16* d_vis_out = nullptr;
                    if (!is_continuation) {
                        current_messages = prune_messages_to_fit(current_messages, engine.tokenizer_, max_allowed_prompt, enable_thinking, reasoning_effort, tools);
                        mm_prompt = apply_chat_template_multimodal(current_messages, engine.tokenizer_, enable_thinking, reasoning_effort, tools, engine.has_vision());
                        prompt = mm_prompt.prompt;
                        if (engine.has_vision() && !mm_prompt.image_b64.empty() && mm_prompt.visual_start_pos >= 0) {
                            if (engine.ensure_vision_loaded()) {
                                moecher::vision::ProcessedImage proc_img;
                                int img_target_sz = engine.vision_tower_->image_size_;
                                LOG_INFO("[Vision] Preprocessing image attachment for multimodal inference (%zu bytes b64, target_size=%d)...",
                                         mm_prompt.image_b64.size(), img_target_sz);
                                if (moecher::vision::preprocess_image_from_base64(mm_prompt.image_b64, proc_img, img_target_sz)) {
                                    d_vis_out = engine.vision_tower_->forward(proc_img, engine.main_stream_);
                                    LOG_INFO("[Vision] Vision Tower forward pass complete: 576 visual tokens projected to hidden size %d.", engine.cfg_.hidden_size);
                                } else {
                                    LOG_WARN("[Vision] Failed to decode/preprocess image attachment!");
                                }
                            }
                        }
                    }
                    LOG_INFO("PROMPT [round %d] (len=%zu, continuation=%s, visual=%s)",
                             round + 1, prompt.size(), is_continuation ? "true" : "false", (d_vis_out != nullptr) ? "active" : "none");

                    std::string response_text;
                    {
                        std::lock_guard<std::mutex> lock(g_engine_mutex);
                        bool conv_has_image = false;
                        for (const auto& m : current_messages) {
                            if (message_has_image(m)) { conv_has_image = true; break; }
                        }
                        bool req_thinking = enable_thinking && !conv_has_image;
                        int req_thinking_tokens = req_thinking ? max_thinking_tokens : 0;
                        response_text = engine.generate(prompt, max_tokens, temperature, nullptr, repetition_penalty, req_thinking, req_thinking_tokens, top_p, min_p, top_k,
                                                        d_vis_out, mm_prompt.visual_start_pos, mm_prompt.visual_num_tokens, bypass_prefix_cache);
                        finish_reason = engine.last_finish_reason_;

                        if (d_vis_out != nullptr && engine.gpu_caps_.is_vram_constrained) {
                            engine.unload_vision_tower();
                        }
                    }

                    if (!g_enable_tools || tools.empty()) {
                        final_response_text = response_text;
                        break;
                    }

                    std::string clean_content;
                    std::vector<moecher::tooling::ToolCall> round_tool_calls;
                    moecher::tooling::extract_tool_calls(response_text, clean_content, round_tool_calls);

                    if (round_tool_calls.empty()) {
                        final_response_text = clean_content;
                        break;
                    }

                    emitted_tool_calls = round_tool_calls;
                    final_response_text = clean_content;

                    if (!do_server_exec) {
                        finish_reason = "tool_calls";
                        break;
                    }

                    // Server-side tool execution
                    bool can_execute_all = true;
                    json assistant_msg = {{"role", "assistant"}, {"content", response_text}};
                    json tc_json_arr = json::array();
                    for (const auto& tc : round_tool_calls) {
                        tc_json_arr.push_back({
                            {"id", tc.id},
                            {"type", tc.type},
                            {"function", {{"name", tc.name}, {"arguments", tc.arguments}}}
                        });
                    }
                    assistant_msg["tool_calls"] = tc_json_arr;
                    current_messages.push_back(assistant_msg);

                    auto is_path_authorized = [&](const std::string& p) -> bool {
                        if (!require_external_authorization && !workspace_boundary_enforced) return true;
                        if (moecher::tooling::is_path_inside_workspace(p)) return true;
                        for (const auto& ap : authorized_paths) {
                            if (ap == "*" || ap == p) return true;
                            if (!ap.empty() && moecher::tooling::is_path_inside_workspace(p, ap)) return true;
                        }
                        return false;
                    };

                    for (const auto& tc : round_tool_calls) {
                        std::string exec_output;
                        if (tc.name == "web_search" || tc.name == "google_search") {
                            std::string query;
                            int num_results = 5;
                            std::string site = "";
                            std::string provider = "";
                            std::string api_key = "";
                            std::string cx = "";
                            std::string searx_url = "";
                            try {
                                json args = json::parse(tc.arguments);
                                query = args.value("query", "");
                                num_results = args.value("num_results", 5);
                                site = args.value("site", "");
                                provider = args.value("provider", "");
                                api_key = args.value("api_key", "");
                                cx = args.value("cx", "");
                                searx_url = args.value("searxng_url", "");
                            } catch (...) { query = tc.arguments; }
                            LOG_INFO("Executing built-in tool web_search: %s (num=%d, site=%s, provider=%s)", query.c_str(), num_results, site.c_str(), provider.c_str());
                            exec_output = moecher::tooling::web_search_content(query, num_results, site, provider, api_key, cx, searx_url);
                        } else if (tc.name == "fetch_url") {
                            std::string url;
                            std::string mode = "text";
                            std::string pattern = "";
                            int offset = 0;
                            int max_chars = 4000;
                            try {
                                json args = json::parse(tc.arguments);
                                url = args.value("url", "");
                                mode = args.value("mode", "text");
                                pattern = args.value("pattern", "");
                                offset = args.value("offset", 0);
                                max_chars = args.value("max_chars", 4000);
                            } catch (...) { url = tc.arguments; }
                            LOG_INFO("Executing built-in tool fetch_url: %s (mode=%s, pattern=%s)", url.c_str(), mode.c_str(), pattern.c_str());
                            exec_output = moecher::tooling::fetch_url_content(url, std::min(execution_timeout_ms, 30000), max_chars, mode, pattern, offset);
                        } else if (tc.name == "read_file") {
                            std::string path;
                            int start_line = 1;
                            int end_line = -1;
                            try {
                                json args = json::parse(tc.arguments);
                                path = args.value("path", "");
                                start_line = args.value("start_line", 1);
                                end_line = args.value("end_line", -1);
                            } catch (...) { path = tc.arguments; }
                            LOG_INFO("Executing built-in tool read_file: %s", path.c_str());
                            if (!is_path_authorized(path)) {
                                exec_output = "[Authorization Required: The path '" + path + "' is outside the current workspace directory (" + moecher::tooling::get_workspace_directory() + "). Action requires explicit user authorization for path: " + path + "]";
                            } else {
                                exec_output = moecher::tooling::read_file(path, start_line, end_line);
                            }
                        } else if (tc.name == "write_file") {
                            std::string path;
                            std::string content;
                            bool overwrite = true;
                            try {
                                json args = json::parse(tc.arguments);
                                path = args.value("path", "");
                                content = args.value("content", "");
                                overwrite = args.value("overwrite", true);
                            } catch (...) { path = tc.arguments; }
                            LOG_INFO("Executing built-in tool write_file: %s (%zu bytes)", path.c_str(), content.size());
                            if (!is_path_authorized(path)) {
                                exec_output = "[Authorization Required: The path '" + path + "' is outside the current workspace directory (" + moecher::tooling::get_workspace_directory() + "). Action requires explicit user authorization for path: " + path + "]";
                            } else {
                                exec_output = moecher::tooling::write_file(path, content, overwrite);
                            }
                        } else if (tc.name == "edit_file") {
                            std::string path;
                            std::string target_content;
                            std::string replacement_content;
                            try {
                                json args = json::parse(tc.arguments);
                                path = args.value("path", "");
                                target_content = args.value("target_content", "");
                                replacement_content = args.value("replacement_content", "");
                            } catch (...) { path = tc.arguments; }
                            LOG_INFO("Executing built-in tool edit_file: %s", path.c_str());
                            if (!is_path_authorized(path)) {
                                exec_output = "[Authorization Required: The path '" + path + "' is outside the current workspace directory (" + moecher::tooling::get_workspace_directory() + "). Action requires explicit user authorization for path: " + path + "]";
                            } else {
                                exec_output = moecher::tooling::edit_file(path, target_content, replacement_content);
                            }
                        } else if (tc.name == "execute_command") {
                            std::string cmd;
                            try {
                                json args = json::parse(tc.arguments);
                                cmd = args.value("command", "");
                            } catch (...) { cmd = tc.arguments; }
                            LOG_INFO("Executing built-in tool execute_command: %s (timeout=%dms)", cmd.c_str(), execution_timeout_ms);
                            exec_output = moecher::tooling::execute_system_command(cmd, execution_timeout_ms);
                        } else if (moecher::mcp::MCPManager::instance().has_tool(tc.name)) {
                            std::string s_id, raw_tname;
                            moecher::mcp::MCPManager::instance().has_tool(tc.name, &s_id, &raw_tname);
                            json parsed_args = json::object();
                            try { parsed_args = json::parse(tc.arguments); } catch (...) {}
                            LOG_INFO("Executing MCP tool: [%s:%s]", s_id.c_str(), raw_tname.c_str());
                            exec_output = moecher::mcp::MCPManager::instance().call_tool(tc.name, parsed_args, execution_timeout_ms);
                        } else {
                            can_execute_all = false;
                            break;
                        }

                        json tool_reply = {
                            {"role", "tool"},
                            {"tool_call_id", tc.id},
                            {"content", exec_output}
                        };
                        current_messages.push_back(tool_reply);
                        round_new_tool_msgs.push_back(tool_reply);
                    }

                    if (!can_execute_all) {
                        finish_reason = "tool_calls";
                        break;
                    }
                }

                if (do_server_exec && !emitted_tool_calls.empty() && finish_reason != "tool_calls") {
                    LOG_WARN("Tool execution reached max rounds limit (%d).", req_max_tool_rounds);
                    final_response_text += "\n\n> ⚠️ **Maximum tool execution rounds reached (" + std::to_string(req_max_tool_rounds) + "/" + std::to_string(req_max_tool_rounds) + ").** Execution stopped.";
                }

                json choice = {
                    {"index", 0},
                    {"message", {
                        {"role", "assistant"},
                        {"content", final_response_text}
                    }},
                    {"finish_reason", finish_reason}
                };

                if (!emitted_tool_calls.empty() && (!g_server_exec || finish_reason == "tool_calls")) {
                    json tc_arr = json::array();
                    for (const auto& tc : emitted_tool_calls) {
                        tc_arr.push_back({
                            {"id", tc.id},
                            {"type", tc.type},
                            {"function", {{"name", tc.name}, {"arguments", tc.arguments}}}
                        });
                    }
                    choice["message"]["tool_calls"] = tc_arr;
                }

                json response = {
                    {"id", req_id},
                    {"object", "chat.completion"},
                    {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                    {"model", model_id},
                    {"choices", {choice}},
                    {"usage", {
                        {"prompt_tokens", engine.prompt_token_count_},
                        {"completion_tokens", engine.completion_token_count_},
                        {"total_tokens", engine.prompt_token_count_ + engine.completion_token_count_},
                        {"evaluated_prompt_tokens", engine.last_prefill_tokens_},
                        {"prefill_sec", engine.last_prefill_sec_},
                        {"prefill_tps", engine.last_prefill_tps_},
                        {"decode_sec", engine.last_decode_sec_},
                        {"decode_tps", engine.last_decode_tps_},
                        {"prompt_tokens_details", {
                            {"cached_tokens", engine.last_cached_tokens_}
                        }}
                    }}
                };
                res.set_content(response.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");

                // Save conversation state for turn continuation
                {
                    std::lock_guard<std::mutex> lock(g_engine_mutex);
                    json conv_to_save = current_messages;
                    if (conv_to_save.empty() || conv_to_save.back().value("role", "") != "assistant") {
                        conv_to_save.push_back({{"role", "assistant"}, {"content", final_response_text}});
                    }
                    s_last_conv_messages = conv_to_save;
                }
                } catch (const std::exception& e) {
                    LOG_ERROR("[Completion] Exception during request execution: %s", e.what());
                    json err_resp = {{"error", {{"message", e.what()}, {"type", "server_error"}}}};
                    res.status = 500;
                    res.set_content(err_resp.dump(), "application/json");
                }
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

    LOG_INFO("Binding to 0.0.0.0:%d...", port);
    if (!svr.bind_to_port("0.0.0.0", port)) {
        LOG_ERROR("Failed to bind to 0.0.0.0:%d: %s (errno=%d)", port, strerror(errno), errno);
        moecher::mcp::MCPManager::instance().stop_all();
        return;
    }
    LOG_INFO("Server listening on port %d", port);
    LOG_INFO("version %s", MOECHER_VERSION);
    g_server_ready = true;
    svr.listen_after_bind();
    LOG_INFO("Server stopped on port %d", port);
    moecher::mcp::MCPManager::instance().stop_all();
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
#if defined(__APPLE__)
    bool enable_pld = true;
    int pld_draft_tokens = 4;
    bool enable_mtp = true;
    bool enable_tree_mtp = false;
    int tree_nodes = 2; // Default 2 alternative branches (Top-1 and Top-2, M=3 total nodes)
    bool enable_spec = true;
#else
    bool enable_pld = true;
    int pld_draft_tokens = 4;
    bool enable_mtp = true;
    bool enable_tree_mtp = true;
    int tree_nodes = 2;
    bool enable_spec = true;
#endif
    std::string test_prompt = "";
    bool benchmark_mode = false;
    float test_temp = 0.7f;
    int test_max_tokens = 150;
    bool test_thinking = true;
    int proxy_port = 8002;
    bool enable_forward_proxy = true;
    bool enable_system_proxy = false;
    std::string batched_prefill_mode = "auto";
    int prefill_chunk_size = 512;
    int max_seq_len_override = 0;
    int mtp_k = 0;
    bool enable_vision = true;

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --manifest, -m <path>       Path to moecher_manifest.json\n");
            printf("  --port, -p <port>           Server port (default: %d)\n", port);
            printf("  --proxy-port, -pp <port>    Proxy port (default: %d)\n", proxy_port);
            printf("  --max-vram, -V <gb>         Max VRAM budget in GB\n");
            printf("  --dram-cache-gb <gb>        DRAM cache budget in GB\n");
            printf("  --ctx, -c <tokens>          Max context length in tokens (default: 65536)\n");
            printf("  --max-seq-len <tokens>      Alias for --ctx\n");
            printf("  --no-mtp, --disable-mtp     Disable MTP self-speculative decoding\n");
            printf("  --no-tree-mtp               Disable Tree Attention on MTP speculative decoding\n");
            printf("  --tree-mtp                  Enable Tree Attention on MTP speculative decoding\n");
            printf("  --tree-nodes <n>            Number of alternative branches for Tree MTP (default: 2, range: 1..4)\n");
            printf("  --mtp-k <n>                 Max MTP draft depth (default: 1 on 16GB GPUs, 4 on >=24GB GPUs)\n");
            printf("  --no-vision                 Disable Vision Tower (saves ~990 MB VRAM on 16GB GPUs)\n");
            printf("  --budget <tokens>           Default thinking budget tokens\n");
            printf("  --max-tool-rounds <n>       Max sequential tool execution rounds per turn (default: %d)\n", g_max_tool_rounds);
            printf("  --help, -h                  Show this help message\n");
            return 0;
        } else if ((std::string(argv[i]) == "--manifest" || std::string(argv[i]) == "-m") && i + 1 < argc) {
            manifest_path = argv[++i];
        } else if ((std::string(argv[i]) == "--port" || std::string(argv[i]) == "-p") && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if ((std::string(argv[i]) == "--proxy-port" || std::string(argv[i]) == "-pp") && i + 1 < argc) {
            proxy_port = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--no-proxy" || std::string(argv[i]) == "--disable-proxy") {
            enable_forward_proxy = false;
        } else if (std::string(argv[i]) == "--system-proxy" || std::string(argv[i]) == "--auto-proxy") {
            enable_system_proxy = true;
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
        } else if ((std::string(argv[i]) == "--temp" || std::string(argv[i]) == "--temperature") && i + 1 < argc) {
            test_temp = std::stof(argv[++i]);
        } else if ((std::string(argv[i]) == "--max-tokens" || std::string(argv[i]) == "-n") && i + 1 < argc) {
            test_max_tokens = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--no-think" || std::string(argv[i]) == "--no-thinking") {
            test_thinking = false;
        } else if (std::string(argv[i]) == "--imatrix-dataset" && i + 1 < argc) {
            imatrix_dataset = argv[++i];
        } else if (std::string(argv[i]) == "--imatrix-out" && i + 1 < argc) {
            imatrix_out = argv[++i];
        } else if (std::string(argv[i]) == "--imatrix-max-tokens" && i + 1 < argc) {
            imatrix_max_tokens = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--test-prompt" && i + 1 < argc) {
            test_prompt = argv[++i];
        } else if (std::string(argv[i]) == "--benchmark") {
            benchmark_mode = true;
        } else if (std::string(argv[i]) == "--no-pld" || std::string(argv[i]) == "--disable-pld") {
            enable_pld = false;
        } else if (std::string(argv[i]) == "--no-mtp" || std::string(argv[i]) == "--disable-mtp") {
            enable_mtp = false;
        } else if (std::string(argv[i]) == "--no-tree-mtp" || std::string(argv[i]) == "--disable-tree-mtp") {
            enable_tree_mtp = false;
        } else if (std::string(argv[i]) == "--tree-mtp" || std::string(argv[i]) == "--enable-tree-mtp") {
            enable_tree_mtp = true;
        } else if ((std::string(argv[i]) == "--tree-nodes" || std::string(argv[i]) == "--tree-branches") && i + 1 < argc) {
            tree_nodes = std::clamp(std::stoi(argv[++i]), 1, 4);
        } else if ((std::string(argv[i]) == "--mtp-k" || std::string(argv[i]) == "--mtp-depth") && i + 1 < argc) {
            mtp_k = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--no-vision" || std::string(argv[i]) == "--disable-vision") {
            enable_vision = false;
        } else if (std::string(argv[i]) == "--no-spec" || std::string(argv[i]) == "--disable-spec") {
            enable_spec = false;
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
        } else if (std::string(argv[i]) == "--no-tools" || std::string(argv[i]) == "--disable-tools") {
            g_enable_tools = false;
        } else if (std::string(argv[i]) == "--tools") {
            g_enable_tools = true;
        } else if (std::string(argv[i]) == "--no-server-exec" || std::string(argv[i]) == "--disable-server-exec") {
            g_server_exec = false;
        } else if (std::string(argv[i]) == "--server-exec") {
            g_server_exec = true;
        } else if (std::string(argv[i]) == "--headless-browsing" || std::string(argv[i]) == "-headless-browsing" || std::string(argv[i]) == "--enable-headless-browsing") {
            g_headless_browsing = true;
            g_server_exec = true;
        } else if (std::string(argv[i]) == "--track-reset" ||
                   std::string(argv[i]) == "--reset-track" ||
                   std::string(argv[i]) == "-track-reset" ||
                   std::string(argv[i]) == "-reset-track") {
            g_track_reset = true;
            g_track_experts = true;
        } else if ((std::string(argv[i]) == "--batched-prefill") && i + 1 < argc) {
            batched_prefill_mode = argv[++i];
        } else if (std::string(argv[i]) == "--no-batched-prefill") {
            batched_prefill_mode = "off";
        } else if (std::string(argv[i]) == "--prefill-chunk-size" && i + 1 < argc) {
            prefill_chunk_size = std::stoi(argv[++i]);
        } else if ((std::string(argv[i]) == "--ctx" || std::string(argv[i]) == "--max-seq-len" || std::string(argv[i]) == "-c") && i + 1 < argc) {
            max_seq_len_override = std::stoi(argv[++i]);
        } else if ((std::string(argv[i]) == "--max-tool-rounds" || std::string(argv[i]) == "--max-rounds") && i + 1 < argc) {
            g_max_tool_rounds = std::max(1, std::stoi(argv[++i]));
        }
    }

    // Open log file
    g_log_file.open(log_path, std::ios::app);
    LOG_INFO("=== moecher starting ===");
    LOG_INFO("=== v%s ===", MOECHER_VERSION);
    LOG_INFO("Default thinking token budget: %d", default_thinking_budget);
    LOG_INFO("Tool calling support: %s", g_enable_tools ? "enabled" : "disabled");
    LOG_INFO("Max tool execution rounds: %d", g_max_tool_rounds);
    LOG_INFO("Server-side tool execution: %s (headless-browsing: %s)", g_server_exec ? "enabled" : "disabled", g_headless_browsing ? "enabled" : "disabled");

    MoecherEngine engine;
    engine.batched_prefill_mode_ = batched_prefill_mode;
    engine.prefill_chunk_size_ = prefill_chunk_size;
    engine.enable_pld_ = enable_pld;
    engine.pld_draft_tokens_ = pld_draft_tokens;
    engine.enable_mtp_ = enable_mtp;
    engine.default_enable_mtp_ = enable_mtp;
    engine.enable_tree_mtp_ = enable_tree_mtp;
    engine.tree_nodes_ = tree_nodes;
    engine.mtp_k_ = mtp_k;
    engine.default_mtp_k_ = mtp_k;
    engine.enable_spec_ = enable_spec;
    engine.default_enable_tree_mtp_ = enable_tree_mtp;
    engine.default_tree_nodes_ = tree_nodes;
    engine.default_enable_spec_ = enable_spec;
    engine.enable_vision_ = enable_vision;
    LOG_INFO("Speculative decoding: %s (PLD: %s, MTP: %s, Tree MTP: %s, tree_nodes=%d, mtp_k=%d)",
             enable_spec ? "enabled" : "disabled",
             enable_pld ? "enabled" : "disabled",
             enable_mtp ? "enabled" : "disabled",
             enable_tree_mtp ? "enabled" : "disabled",
             tree_nodes,
             mtp_k);

    if (!engine.load(manifest_path, max_vram_gb, dram_cache_gb, expert_dtype_override, buffered_io, max_seq_len_override)) {
        LOG_ERROR("Failed to load model");
        return 1;
    }

    if (!imatrix_dataset.empty() && !imatrix_out.empty()) {
        bool ok = engine.collect_imatrix(imatrix_dataset, imatrix_out, imatrix_max_tokens);
        return ok ? 0 : 1;
    }

    if (!test_prompt.empty() || benchmark_mode) {
        std::string prompt_str = test_prompt.empty() ? "Write a short story about an astronaut." : test_prompt;
        json messages = json::array({
            {{"role", "user"}, {"content", prompt_str}}
        });
        std::vector<int> prompt = apply_chat_template(messages, engine.tokenizer_, test_thinking, "high");
        LOG_INFO("Running test generation for prompt: '%s' (%zu prompt tokens, temp=%.2f, max_tokens=%d, thinking=%d)...",
                 prompt_str.c_str(), prompt.size(), test_temp, test_max_tokens, test_thinking ? 1 : 0);
        std::string result = engine.generate(prompt, test_max_tokens, test_temp, nullptr, 1.0f, test_thinking, default_thinking_budget, 0.95f, 0.0f, 1024);
        printf("\n\n--- Output ---\n%s\n--------------\n", result.c_str());
        return 0;
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

    // Initialize MCP (Model Context Protocol) Manager before pre-warming
    auto& mcp_mgr = moecher::mcp::MCPManager::instance();
    std::string mcp_config_file = moecher::tooling::get_workspace_directory() + "/mcp_servers.json";
    mcp_mgr.set_config_path(mcp_config_file);
    if (mcp_mgr.load_config()) {
        mcp_mgr.start_all_enabled();
    }

    // Pre-warm default system prompt and tooling into KV Cache for 0ms initial prefill latency
    if (g_enable_tools) {
        json default_tools = resolve_canonical_tools("default");
        json mcp_tools = mcp_mgr.get_openai_tools_schema();
        for (const auto& mt : mcp_tools) {
            default_tools.push_back(mt);
        }
        rebuild_system_prefix(engine, g_base_system_prompt, default_tools);
    }

    run_server(engine, port, default_thinking_budget, proxy_port, enable_forward_proxy, enable_system_proxy);

    // Save expert frequency & specialization profile on shutdown
    if (g_track_experts && !engine.expert_freq_counts_.empty()) {
        engine.save_expert_freq(freq_path);
        std::string model_dir = std::filesystem::path(manifest_path).parent_path().string();
        engine.save_expert_profile(model_dir);
    }
    return 0;
}
