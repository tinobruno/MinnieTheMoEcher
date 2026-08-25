#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <climits>
#include "../build/_deps/nlohmann_json-src/include/nlohmann/json.hpp"

using json = nlohmann::json;

#define LOG_INFO(fmt, ...) printf("[INFO] " fmt "\n", ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) printf("[WARN] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) printf("[ERROR] " fmt "\n", ##__VA_ARGS__)

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

int main() {
    BPETokenizer tok;
    tok.load("/home/tinobruno/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062/tokenizer.json");
    std::string prompt = "tell me about commodore Amiga, in particular about Amiga 3000 and Amiga 4000, and their UNIX variants.";
    auto ids = tok.encode(prompt);
    std::cout << "Moecher C++ encoded (" << ids.size() << "):" << std::endl;
    for (int id : ids) {
        std::cout << id << " ";
    }
    std::cout << std::endl;
    return 0;
}
