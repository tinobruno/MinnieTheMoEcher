import re
with open('src/server_single.cpp', 'r') as f:
    content = f.read()

pattern = r'(    if \(request\.contains\("tools"\) && request\["tools"\]\.is_array\(\) && !request\["tools"\]\.empty\(\)\) \{\n        std::string tools_str = "\\n\\n## Tools\\nYou have access to the following tools:\\n";).*?(        if \(system_prompt\.empty\(\)\) \{)'

replacement = r'''    if (request.contains("tools") && request["tools"].is_array() && !request["tools"].empty()) {
        std::string tools_str = "\n\n## Tools\n\nYou have access to the following tools:\n";
        for (auto& tool : request["tools"]) {
            if (tool.contains("function")) {
                auto& func = tool["function"];
                std::string name = func.value("name", "unknown_function");
                std::string desc = func.value("description", "");
                std::string params = func.value("parameters", json::object()).dump();
                tools_str += "\n### " + name + "\n\n" + desc + "\n\nParameters:\n" + params + "\n";
            }
        }
        
        if (system_prompt.empty()) {'''

content, count = re.subn(pattern, replacement, content, flags=re.DOTALL)
print(f"Replaced {count} times")
with open('src/server_single.cpp', 'w') as f:
    f.write(content)
