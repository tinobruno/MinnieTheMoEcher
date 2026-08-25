#include "tokenizer.h"
#include <iostream>
int main() {
    BPETokenizer tok("tokenizer.json");
    std::cout << "1: '" << tok.decode({1}) << "'" << std::endl;
}
