#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include "json.hpp"

using json = nlohmann::json;

#define LOG_INFO(...) 
#define LOG_WARN(...) 
#define LOG_ERROR(...) 

// Paste BPETokenizer methods
#include "tokenizer_snippet.h"

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
