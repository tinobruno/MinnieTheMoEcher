#include <iostream>
#include <string>

std::string strip_thinking(const std::string& text) {
    std::string res = text;
    while (true) {
        size_t start = res.find("<think>");
        if (start == std::string::npos) break;
        size_t end = res.find("</think>", start);
        if (end != std::string::npos) {
            res = res.substr(0, start) + res.substr(end + 8);
        } else {
            res = res.substr(0, start);
            break;
        }
    }
    // Trim leading whitespace
    size_t first_non_ws = res.find_first_not_of(" \n\r\t");
    if (first_non_ws != std::string::npos) {
        res = res.substr(first_non_ws);
    } else {
        res = "";
    }
    return res;
}

int main() {
    std::string s = "<think> I should think </think> The capital is Rome.";
    std::cout << strip_thinking(s) << std::endl;
    return 0;
}
