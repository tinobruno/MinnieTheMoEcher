#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <climits>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class BPETokenizerTest {
public:
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<int, std::string> id_to_token_;
    std::unordered_map<std::string, int> merge_rank_;
    std::vector<std::pair<std::string, int>> special_tokens_;
    std::string byte_to_unicode_[256];

    void build_byte_mapping() {
        std::vector<int> bs;
        for (int b = '!'; b <= '~'; b++) bs.push_back(b);
        for (int b = 161; b <= 172; b++) bs.push_back(b);
        for (int b = 174; b <= 255; b++) bs.push_back(b);

        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                n++;
            }
        }

        for (size_t i = 0; i < bs.size(); i++) {
            int b = bs[i];
            int cp = cs[i];
            std::string utf8;
            if (cp < 0x80) {
                utf8 += (char)cp;
            } else if (cp < 0x800) {
                utf8 += (char)(0xC0 | (cp >> 6));
                utf8 += (char)(0x80 | (cp & 0x3F));
            }
            byte_to_unicode_[b] = utf8;
        }
    }

    bool load(const std::string& path) {
        std::ifstream f(path);
        json tok = json::parse(f);
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
                merge_rank_[merge_str] = (int)merge_rank_.size();
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
        return true;
    }

    std::string bytes_to_unicode(const std::string& bytes) const {
        std::string result;
        for (unsigned char b : bytes) {
            result += byte_to_unicode_[b];
        }
        return result;
    }

    void split_on_special(const std::string& text,
                          std::vector<std::pair<std::string, bool>>& out) const {
        size_t pos = 0;
        while (pos < text.size()) {
            bool found = false;
            for (auto& [st, id] : special_tokens_) {
                if (text.compare(pos, st.size(), st) == 0) {
                    out.push_back({st, true});
                    pos += st.size();
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
            if (seg.empty()) continue;

            std::string unicode_text = bytes_to_unicode(seg);

            // BPE merge on characters
            std::vector<std::string> tokens;
            size_t i = 0;
            while (i < unicode_text.size()) {
                unsigned char c = (unsigned char)unicode_text[i];
                int char_len = 1;
                if ((c & 0xE0) == 0xC0) char_len = 2;
                else if ((c & 0xF0) == 0xE0) char_len = 3;
                else if ((c & 0xF8) == 0xF0) char_len = 4;
                tokens.push_back(unicode_text.substr(i, char_len));
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
                if (it != token_to_id_.end()) ids.push_back(it->second);
            }
        }
        return ids;
    }
};

int main() {
    BPETokenizerTest tok;
    tok.load("/home/tinobruno/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062/tokenizer.json");

    std::string prompt = "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence" "\xef\xbd\x9c>"
                         "You are a helpful, friendly, and knowledgeable AI assistant. Answer clearly and concisely."
                         "<\xef\xbd\x9c" "User" "\xef\xbd\x9c>"
                         "What is the capital of Italy ?"
                         "<\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c>";

    auto ids = tok.encode(prompt);
    std::cout << "Test BPE Token Count: " << ids.size() << std::endl;
    std::cout << "Test BPE Token IDs: ";
    for (int id : ids) std::cout << id << " ";
    std::cout << std::endl;
    return 0;
}
