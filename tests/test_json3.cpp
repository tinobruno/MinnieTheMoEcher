#include <nlohmann/json.hpp>
#include <iostream>
using json = nlohmann::json;

std::string apply_chat_template(const json& request) {
    const json& messages = request["messages"];
    std::string result;

    for (size_t i = 0; i < messages.size(); i++) {
        std::string role = messages[i]["role"].get<std::string>();
        std::string content = messages[i].value("content", "");

        if (role == "system") {
            result += content;
            if (request.contains("tools") && request["tools"].is_array() && !request["tools"].empty()) {
                for (auto& tool : request["tools"]) {
                    if (tool.contains("function")) {
                        result += tool["function"].dump(-1, ' ', false, json::error_handler_t::replace) + "\n";
                    }
                }
            }
        } else if (role == "user") {
            result += content;
        } else if (role == "assistant") {
            if (messages[i].contains("reasoning_content")) {
                std::string reasoning = messages[i].value("reasoning_content", "");
                if (!reasoning.empty()) {
                    result += "<think>\n" + reasoning + "\n</think>\n";
                }
            }
            result += content;
        }
    }

    if (!messages.empty()) {
        std::string last_role = messages.back()["role"].get<std::string>();
    }

    return result;
}

int main() {
    try {
        std::string s = R"({"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "reasoning_content": "The user is asking about \"Hello\" - they want me to act as a helpful, concise assistant."}, {"role": "user", "content": "what is the commodore Amiga ?"}], "stream": true, "max_tokens": 100})";
        json request = json::parse(s);
        
        apply_chat_template(request);
        std::cout << "SUCCESS" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "EXCEPTION: " << e.what() << std::endl;
    }
    return 0;
}
