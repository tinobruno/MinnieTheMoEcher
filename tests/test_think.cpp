#include <iostream>
#include <string>
#include "tokenizer.h"
int main() {
    BPETokenizer tok;
    tok.load("tokenizer.json"); // wait, there is no tokenizer.json
    return 0;
}
