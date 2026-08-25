#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#define main not_real_main
#include "../src/server_single.cpp"
#undef main

int main() {
    BPETokenizer tok;
    tok.load("tokenizer.json");
    std::string prompt = "tell me about commodore Amiga, in particular about Amiga 3000 and Amiga 4000, and their UNIX variants.";
    
    json msgs = json::array();
    msgs.push_back({{"role", "user"}, {"content", prompt}});
    
    auto tokens = apply_chat_template(msgs, tok, true, "high");
    std::cout << "Encoded tokens (" << tokens.size() << "):" << std::endl;
    for (int t : tokens) {
        std::cout << t << " -> '" << tok.decode({t}) << "'" << std::endl;
    }
    return 0;
}
