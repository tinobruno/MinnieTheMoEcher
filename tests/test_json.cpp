#include <iostream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

int main() {
    json chunk = {
        {"choices", {{
            {"delta", {{"content", "a"}}}
        }}}
    };
    std::cout << chunk.dump() << "\n";
    
    std::string text = "b";
    chunk["choices"][0]["delta"] = {{"content", text}};
    std::cout << chunk.dump() << "\n";
    
    chunk["choices"][0]["delta"]["content"] = "c";
    std::cout << chunk.dump() << "\n";
    return 0;
}
