import re
with open('src/server_single.cpp', 'r') as f:
    content = f.read()

target = '''    if (request.contains("tools") && request["tools"].is_array() && !request["tools"].empty()) {
        std::string tools_str = "\\n\\n## Tools\\nYou have access to the following tools:\\n";
        for (auto& tool : request["tools"]) {
            if (tool.contains("function")) {
                auto& func = tool["function"];
                std::string name = func.value("name", "unknown_function");
                std::string desc = func.value("description", "");
                std::string params = func.value("parameters", json::object()).dump();
                tools_str += "\\n### " + name + "\\nDescription: " + desc + "\\n\\nParameters: " + params + "\\n";
            }
        }
        tools_str += "\\nIMPORTANT: ALWAYS adhere to this exact format for tool use:\\n";
        tools_str += "<\\xef\\xbd\\x9c" "tool\\xe2\\x96\\x81" "calls\\xe2\\x96\\x81" "begin\\xef\\xbd\\x9c>";
        tools_str += "<\\xef\\xbd\\x9c" "tool\\xe2\\x96\\x81" "call\\xe2\\x96\\x81" "begin\\xef\\xbd\\x9c>tool_call_name";
        tools_str += "<\\xef\\xbd\\x9c" "tool\\xe2\\x96\\x81" "sep\\xef\\xbd\\x9c>tool_call_arguments";
        tools_str += "<\\xef\\xbd\\x9c" "tool\\xe2\\x96\\x81" "call\\xe2\\x96\\x81" "end\\xef\\xbd\\x9c>";
        tools_str += "<\\xef\\xbd\\x9c" "tool\\xe2\\x96\\x81" "calls\\xe2\\x96\\x81" "end\\xef\\xbd\\x9c>\\n\\n";
        tools_str += "Where:\\n\\n";
        tools_str += "- `tool_call_name` must be an exact match to one of the available tools\\n";
        tools_str += "- `tool_call_arguments` must be a valid JSON object matching the tool's parameters schema\\n";
        tools_str += "- For multiple tool calls, chain them directly without separators or spaces\\n";
        
        if (system_prompt.empty()) {'''

replacement = '''    if (request.contains("tools") && request["tools"].is_array() && !request["tools"].empty()) {
        std::string tools_str = "\\n\\n## Tools\\n\\nYou have access to the following tools:\\n";
        for (auto& tool : request["tools"]) {
            if (tool.contains("function")) {
                auto& func = tool["function"];
                std::string name = func.value("name", "unknown_function");
                std::string desc = func.value("description", "");
                std::string params = func.value("parameters", json::object()).dump();
                tools_str += "\\n### " + name + "\\n\\n" + desc + "\\n\\nParameters:\\n" + params + "\\n";
            }
        }
        
        if (system_prompt.empty()) {'''

content = content.replace(target, replacement)
with open('src/server_single.cpp', 'w') as f:
    f.write(content)
