#include <iostream>
#include <vector>
#include <string>
#include "tokenizer.h"

int main() {
    Tokenizer tokenizer;
    if (!tokenizer.load("moecher_manifest.json")) {
        std::cerr << "Failed to load tokenizer\n";
        return 1;
    }

    std::string prompt = "<\xef\xbd\x9c" "begin_of_sentence" "\xef\xbd\x9c><\xef\xbd\x9c" "User" "\xef\xbd\x9c>hello<\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c>";
    std::vector<int> tokens = tokenizer.encode(prompt);
    
    std::cout << "Tokens: ";
    for (int t : tokens) {
        std::cout << t << " ";
    }
    std::cout << "\n";
    return 0;
}
