#include <nlohmann/json.hpp>
#include <iostream>
using json = nlohmann::json;

int main() {
    try {
        std::string s = R"({"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "reasoning_content": "The user is asking about \"Hello\" - they want me to act as a helpful, concise assistant."}, {"role": "user", "content": "what is the commodore Amiga ?"}], "stream": true, "max_tokens": 100})";
        json request = json::parse(s);
        
        float temperature = request.value("temperature", 0.0f);
        int max_tokens = request.value("max_tokens", 512);
        bool stream = request.value("stream", false);
        float repetition_penalty = request.value("repetition_penalty", 1.1f);
        
        std::cout << "SUCCESS" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "EXCEPTION: " << e.what() << std::endl;
    }
    return 0;
}
