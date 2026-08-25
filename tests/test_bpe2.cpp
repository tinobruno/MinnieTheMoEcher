#include <iostream>
#include <fstream>
#include <string>
#include "json.hpp"

using json = nlohmann::json;

int main() {
    std::ifstream f("gen.json");
    json j;
    f >> j;
    auto map = j["model"]["vocab"];
    std::string s = "<\xef\xbd\x9cend\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>";
    std::cout << "Special token string: " << s << "\n";
    if (map.contains(s)) {
        std::cout << "Contains! ID=" << map[s] << "\n";
    } else {
        std::cout << "Does not contain directly.\n";
    }
}
