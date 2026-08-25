#include <iostream>
#include "src/server_single.cpp"
int main() {
    BPETokenizer tok("tokenizer.json");
    std::cout << "think id: " << tok.get_token_id("<think>") << std::endl;
    std::cout << "think str: " << tok.decode({tok.get_token_id("<think>")}) << std::endl;
    std::cout << "128804 str: " << tok.decode({128804}) << std::endl;
    std::cout << "128805 str: " << tok.decode({128805}) << std::endl;
}
