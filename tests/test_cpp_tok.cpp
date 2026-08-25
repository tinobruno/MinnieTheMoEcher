#include <iostream>
#include <string>

int main() {
    std::string ASSISTANT = "<\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c>";
    std::cout << ASSISTANT << std::endl;
    return 0;
}
