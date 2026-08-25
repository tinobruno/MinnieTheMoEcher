import subprocess
with open("test_tok4.cpp", "w") as f:
    f.write("""
#include "tokenizer.h"
#include <iostream>
int main() {
    BPETokenizer tok("tokenizer.json");
    std::cout << "25254: '" << tok.decode({25254}) << "'" << std::endl;
    std::cout << "492: '" << tok.decode({492}) << "'" << std::endl;
    std::cout << "3167: '" << tok.decode({3167}) << "'" << std::endl;
    std::cout << "16214: '" << tok.decode({16214}) << "'" << std::endl;
}
""")
subprocess.run(["g++", "-O2", "-std=c++17", "test_tok4.cpp", "-Isrc", "-o", "test_tok4"])
subprocess.run(["./test_tok4"])
