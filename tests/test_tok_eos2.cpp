#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>

// Quick test to inspect logits after Rome. and Rome.\n
int main() {
    std::cout << "Testing tokenization..." << std::endl;
    // Just run python script here? No, I want to print C++ tokenizer tokens.
    // wait, I can just change test_user_cases.py to print the tokens that server logs!
    return 0;
}
