#include <iostream>
#include <string>
int main() {
    std::string ASSISTANT = "<\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c>";
    std::string result = ASSISTANT;
    result += "<think>\n";
    std::cout << "result size: " << result.size() << "\n";
    std::cout << "last 8: " << result.substr(result.size() - 8) << "\n";
    bool in_think_block = result.size() >= 8 && result.substr(result.size() - 8) == "<think>\n";
    std::cout << "in_think_block: " << in_think_block << "\n";
    return 0;
}
