#include <iostream>
#include <string>
int main() {
    std::string prompt = "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence" "\xef\xbd\x9c><\xef\xbd\x9c" "User" "\xef\xbd\x9c>Hello<\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c><think>\n";
    bool in_think_block = prompt.size() >= 8 && prompt.substr(prompt.size() - 8) == "<think>\n";
    std::cout << "Prompt len: " << prompt.size() << "\n";
    std::cout << "Last 8 chars: " << prompt.substr(prompt.size() - 8) << "\n";
    std::cout << "in_think_block: " << in_think_block << "\n";
    return 0;
}
