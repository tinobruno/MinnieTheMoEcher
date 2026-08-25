import re
import sys

with open('src/server_single.cpp', 'r') as f:
    content = f.read()

# 1. Update apply_chat_template signature and tool injection
s1 = """static std::string apply_chat_template(const json& messages, const BPETokenizer& tok) {"""
r1 = """static std::string apply_chat_template(const json& request, const BPETokenizer& tok) {
    auto& messages = request.contains("messages") ? request["messages"] : json::array();"""

content = content.replace(s1, r1)

s2 = """        if (role == "system") {
            // System message: raw content, no wrapper tokens (per official encoding)
            result += content;
        }"""
r2 = """        if (role == "system") {
            // System message: raw content, no wrapper tokens (per official encoding)
            result += content;
            if (request.contains("tools") && request["tools"].is_array() && !request["tools"].empty()) {
                std::string dsml = "\\xef\\xbd\\x9c" "DSML" "\\xef\\xbd\\x9c";
                result += "\\n\\n## Tools\\n\\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \\"<" + dsml + "tool_calls>\\" block like the following:\\n\\n";
                result += "<" + dsml + "tool_calls>\\n";
                result += "<" + dsml + "invoke name=\\"$TOOL_NAME\\">\\n";
                result += "<" + dsml + "parameter name=\\"$PARAMETER_NAME\\" string=\\"true|false\\">$PARAMETER_VALUE</" + dsml + "parameter>\\n";
                result += "...\\n";
                result += "</" + dsml + "invoke>\\n";
                result += "<" + dsml + "invoke name=\\"$TOOL_NAME2\\">\\n";
                result += "...\\n";
                result += "</" + dsml + "invoke>\\n";
                result += "</" + dsml + "tool_calls>\\n\\n";
                result += "String parameters should be specified as is and set `string=\\"true\\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\\"false\\"`.\\n\\n";
                result += "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\\n\\n";
                result += "Otherwise, output directly after </think> with tool calls or final response.\\n\\n";
                result += "### Available Tool Schemas\\n\\n";

                for (auto& tool : request["tools"]) {
                    if (tool.contains("function")) {
                        result += tool["function"].dump(-1, ' ', false, json::error_handler_t::replace) + "\\n";
                    }
                }
                result += "\\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\\n";
            }
        }"""
content = content.replace(s2, r2)

s3 = """    // Add assistant prompt for generation
    if (!messages.empty()) {
        std::string last_role = messages.back()["role"].get<std::string>();
        if (last_role == "user") {
            result += ASSISTANT;
            result += "</think>\\n";
        }
    }"""
r3 = """    // Add assistant prompt for generation
    if (!messages.empty()) {
        std::string last_role = messages.back()["role"].get<std::string>();
        if (last_role == "user") {
            result += ASSISTANT;
            result += "</think>\\n";
            if (request.contains("tool_choice") && request["tool_choice"].is_string() && request["tool_choice"].get<std::string>() == "required") {
                result += "<\\xef\\xbd\\x9cDSML\\xef\\xbd\\x9ctool_calls>\\n";
            }
        }
    }"""
content = content.replace(s3, r3)

# Fix endpoint call to apply_chat_template
s4 = """            // Apply chat template
            std::string prompt = apply_chat_template(messages, engine.tokenizer_);"""
r4 = """            // Apply chat template
            std::string prompt = apply_chat_template(request, engine.tokenizer_);"""
content = content.replace(s4, r4)

# 2. Add non-streaming tool parsing logic
s5 = """                json response = {
                    {"id", req_id},
                    {"object", "chat.completion"},
                    {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                    {"model", "deepseek-v4-flash"},
                    {"choices", {{
                        {"index", 0},
                        {"message", {{"role", "assistant"}, {"content", response_text}}},
                        {"finish_reason", engine.last_finish_reason_}
                    }}},"""
r5 = """                json tool_calls_json = json::array();
                std::string content_text = response_text;
                
                size_t tools_start = response_text.find("<\\xef\\xbd\\x9cDSML\\xef\\xbd\\x9ctool_calls>");
                if (tools_start != std::string::npos) {
                    content_text = response_text.substr(0, tools_start);
                    
                    std::string invoke_start = "<\\xef\\xbd\\x9cDSML\\xef\\xbd\\x9cinvoke name=\\\"";
                    std::string param_start = "<\\xef\\xbd\\x9cDSML\\xef\\xbd\\x9cparameter name=\\\"";
                    std::string param_end = "</\\xef\\xbd\\x9cDSML\\xef\\xbd\\x9cparameter>";
                    std::string invoke_end = "</\\xef\\xbd\\x9cDSML\\xef\\xbd\\x9cinvoke>";

                    size_t pos = tools_start;
                    int idx = 0;
                    while ((pos = response_text.find(invoke_start, pos)) != std::string::npos) {
                        pos += invoke_start.length();
                        size_t name_end = response_text.find("\\\">", pos);
                        if (name_end == std::string::npos) break;
                        std::string func_name = response_text.substr(pos, name_end - pos);
                        pos = name_end + 2;

                        size_t inv_end = response_text.find(invoke_end, pos);
                        if (inv_end == std::string::npos) inv_end = response_text.length();

                        json args_json = json::object();
                        size_t ppos = pos;
                        while ((ppos = response_text.find(param_start, ppos)) != std::string::npos && ppos < inv_end) {
                            ppos += param_start.length();
                            size_t pname_end = response_text.find("\\\"", ppos);
                            if (pname_end == std::string::npos) break;
                            std::string param_name = response_text.substr(ppos, pname_end - ppos);
                            ppos = pname_end + 1;

                            bool is_string = true;
                            if (response_text.substr(ppos, 9) == " string=\\\"") {
                                ppos += 9;
                                size_t pstr_end = response_text.find("\\\">", ppos);
                                if (pstr_end != std::string::npos) {
                                    std::string str_val = response_text.substr(ppos, pstr_end - ppos);
                                    is_string = (str_val == "true");
                                    ppos = pstr_end + 2;
                                }
                            } else {
                                size_t p_end = response_text.find(">", ppos);
                                if (p_end != std::string::npos) ppos = p_end + 1;
                            }

                            size_t pend = response_text.find(param_end, ppos);
                            if (pend == std::string::npos) break;

                            std::string param_val = response_text.substr(ppos, pend - ppos);
                            if (is_string) {
                                args_json[param_name] = param_val;
                            } else {
                                try {
                                    args_json[param_name] = json::parse(param_val);
                                } catch(...) {
                                    args_json[param_name] = param_val;
                                }
                            }
                            ppos = pend + param_end.length();
                        }

                        tool_calls_json.push_back({
                            {"index", idx++},
                            {"id", "call_" + std::to_string(rand())},
                            {"type", "function"},
                            {"function", {
                                {"name", func_name},
                                {"arguments", args_json.dump()}
                            }}
                        });

                        pos = inv_end + invoke_end.length();
                    }
                }

                json message_obj = {{"role", "assistant"}, {"content", content_text}};
                if (!tool_calls_json.empty()) {
                    message_obj["tool_calls"] = tool_calls_json;
                }

                json response = {
                    {"id", req_id},
                    {"object", "chat.completion"},
                    {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                    {"model", "deepseek-v4-flash"},
                    {"choices", {{
                        {"index", 0},
                        {"message", message_obj},
                        {"finish_reason", tool_calls_json.empty() ? engine.last_finish_reason_ : "tool_calls"}
                    }}},"""
content = content.replace(s5, r5)

with open('src/server_single.cpp', 'w') as f:
    f.write(content)

