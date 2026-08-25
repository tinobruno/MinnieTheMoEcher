#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include <regex>
#include "json.hpp"
using json = nlohmann::json;

class BPETokenizer {
public:
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<int, std::string> id_to_token_;
    std::unordered_map<std::string, int> merges_;
    std::vector<std::pair<std::string, int>> special_tokens_;

    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return false;
        json tok;
        f >> tok;
        auto& model = tok["model"];
        auto& vocab = model["vocab"];
        for (auto& [k, v] : vocab.items()) {
            int id = v.get<int>();
            token_to_id_[k] = id;
            id_to_token_[id] = k;
        }
        if (model.contains("merges")) {
            for (auto& m : model["merges"]) {
                std::string merge = m.get<std::string>();
                merges_[merge] = merges_.size();
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
        return true;
    }

    std::vector<int> encode(const std::string& text) {
        std::vector<int> ids;
        size_t pos = 0;
        while (pos < text.length()) {
            bool matched = false;
            for (auto& st : special_tokens_) {
                if (text.compare(pos, st.first.length(), st.first) == 0) {
                    ids.push_back(st.second);
                    pos += st.first.length();
                    matched = true;
                    break;
                }
            }
            if (matched) continue;
            
            // just skip the rest for this basic test, we just want to know if special tokens match correctly
            // actually we can just print when we match a special token!
            pos++;
        }
        return ids;
    }
};

int main() {
    BPETokenizer tok;
    if (!tok.load("/home/tinobruno/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062/tokenizer.json")) return 1;
    
    std::ifstream ifs("build/prompt_dump.txt");
    std::string text((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    std::cout << "Text len: " << text.length() << std::endl;
    
    // Check if DSML special token is in added_tokens
    std::string dsml = "\xef\xbd\x9c" "DSML" "\xef\xbd\x9c";
    std::cout << "DSML token id: " << tok.token_to_id_[dsml] << std::endl;
    std::cout << "Assistant token id: " << tok.token_to_id_["<\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c>"] << std::endl;
    
    return 0;
}
