#include <iostream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;
int main() {
    json tool = json::parse(R"({"name":"fetch_url","description":"Fetch and parse text content from a URL.","parameters":{"type":"object","properties":{"url":{"type":"string","description":"The URL to fetch."}},"required":["url"]}})");
    std::cout << tool.dump(2) << std::endl;
    return 0;
}
