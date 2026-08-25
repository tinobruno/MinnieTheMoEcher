#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include "build/_deps/nlohmann_json-src/single_include/nlohmann/json.hpp"
using json = nlohmann::json;
class BPETokenizer {
public:
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<int, std::string> id_to_token_;
    std::vector<std::string> merges_;
    std::unordered_map<std::string, int> merge_rank_;
    std::vector<std::pair<std::string, int>> special_tokens_;
    char32_t byte_to_char_[256];
    std::unordered_map<char32_t, uint8_t> char_to_byte_;

    void build_byte_mapping() {
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
                byte_to_char_[b] = (char32_t)b;
            } else {
                byte_to_char_[b] = (char32_t)(256 + n);
                n++;
            }
        }
    }
    std::string bytes_to_unicode(const std::string& bytes) const {
        std::string result;
        for (unsigned char b : bytes) {
            char32_t c = byte_to_char_[b];
            if (c < 128) {
                result += (char)c;
            } else if (c < 0x800) {
                result += (char)(0xC0 | (c >> 6));
                result += (char)(0x80 | (c & 0x3F));
            } else {
                result += (char)(0xE0 | (c >> 12));
                result += (char)(0x80 | ((c >> 6) & 0x3F));
                result += (char)(0x80 | (c & 0x3F));
            }
        }
        return result;
    }
    std::vector<std::string> bpe(const std::string& token) const {
        std::vector<std::string> word;
        for (size_t i = 0; i < token.size(); ) {
            unsigned char c = (unsigned char)token[i];
            int char_len = 1;
            if ((c & 0xE0) == 0xC0) char_len = 2;
            else if ((c & 0xF0) == 0xE0) char_len = 3;
            else if ((c & 0xF8) == 0xF0) char_len = 4;
            word.push_back(token.substr(i, char_len));
            i += char_len;
        }
        while (word.size() > 1) {
            int best_rank = 1e9;
            int best_idx = -1;
            for (int i = 0; i < (int)word.size() - 1; i++) {
                std::string pair = word[i] + word[i+1];
                auto it = merge_rank_.find(pair);
                if (it != merge_rank_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_idx = i;
                }
            }
            if (best_idx == -1) break;
            word[best_idx] = word[best_idx] + word[best_idx+1];
            word.erase(word.begin() + best_idx + 1);
        }
        return word;
    }
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        std::string unicode_text = bytes_to_unicode(text);
        std::vector<std::string> words;
        std::string current;
        for (size_t i = 0; i < unicode_text.size(); ) {
            if (i + 1 < unicode_text.size() && (unsigned char)unicode_text[i] == 0xC4 && (unsigned char)unicode_text[i+1] == 0xA0) {
                if (!current.empty()) {
                    words.push_back(current);
                    current.clear();
                }
                current += unicode_text[i];
                current += unicode_text[i+1];
                i += 2;
            } else {
                current += unicode_text[i];
                i++;
            }
        }
        if (!current.empty()) words.push_back(current);
        
        for (const std::string& word : words) {
            std::vector<std::string> bpe_tokens = bpe(word);
            for (const std::string& tok : bpe_tokens) {
                auto it = token_to_id_.find(tok);
                if (it != token_to_id_.end()) {
                    ids.push_back(it->second);
                } else {
                    for (unsigned char b : tok) {
                        std::string byte_tok(1, b);
                        auto bit = token_to_id_.find(byte_tok);
                        if (bit != token_to_id_.end()) {
                            ids.push_back(bit->second);
                        }
                    }
                }
            }
        }
        return ids;
    }
};

int main() {
    BPETokenizer tok;
    tok.build_byte_mapping();
    std::ifstream f("/home/tinobruno/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1/tokenizer.json");
    if (!f.is_open()) return 1;
    json j; f >> j;
    auto vocab = j["model"]["vocab"];
    for (auto& item : vocab.items()) {
        tok.token_to_id_[item.key()] = item.value();
    }
    auto merges = j["model"]["merges"];
    for (int i=0; i<merges.size(); i++) {
        std::string m = merges[i];
        m.replace(m.find(" "), 1, "");
        tok.merge_rank_[m] = i;
    }
    
    std::string test = "What is 2+2?";
    auto ids = tok.encode(test);
    for (int id : ids) printf("%d ", id);
    printf("\n");
    return 0;
}
