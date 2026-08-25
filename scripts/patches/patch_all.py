with open('src/server_single.cpp', 'r') as f:
    content = f.read()

import re

# 1. Update apply_chat_template
pattern_template = r'(    if \(request\.contains\("tools"\) && request\["tools"\]\.is_array\(\) && !request\["tools"\]\.empty\(\)\) \{).*?(        result \+= tools_str;\n    \})'

replacement_template = r'''    if (request.contains("tools") && request["tools"].is_array() && !request["tools"].empty()) {
        std::string dsml = "\xef\xbd\x9c" "DSML" "\xef\xbd\x9c";
        std::string tools_str = "\n\n## Tools\n\n";
        tools_str += "You have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<" + dsml + "tool_calls>\" block like the following:\n\n";
        tools_str += "<" + dsml + "tool_calls>\n";
        tools_str += "<" + dsml + "invoke name=\"$TOOL_NAME\">\n";
        tools_str += "<" + dsml + "parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</" + dsml + "parameter>\n";
        tools_str += "...\n";
        tools_str += "</" + dsml + "invoke>\n";
        tools_str += "<" + dsml + "invoke name=\"$TOOL_NAME2\">\n";
        tools_str += "...\n";
        tools_str += "</" + dsml + "invoke>\n";
        tools_str += "</" + dsml + "tool_calls>\n\n";
        tools_str += "String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n";
        tools_str += "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\n";
        tools_str += "Otherwise, output directly after </think> with tool calls or final response.\n\n";
        tools_str += "### Available Tool Schemas\n\n";

        for (auto& tool : request["tools"]) {
            if (tool.contains("function")) {
                tools_str += tool["function"].dump(-1, ' ', false, json::error_handler_t::replace) + "\n";
            }
        }
        tools_str += "\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";

        result += tools_str;
    }'''

content, count = re.subn(pattern_template, replacement_template, content, flags=re.DOTALL)
print(f"Replaced apply_chat_template {count} times")

# 2. Update tool_choice requirement injection
pattern_req = r'(            if \(request\.contains\("tool_choice"\).*?\{\n                result \+= )".*?";\n            \}'
replacement_req = r'\1"<\xef\xbd\x9c" "DSML\xef\xbd\x9ctool_calls>\n";\n            }'

content, count = re.subn(pattern_req, replacement_req, content, flags=re.DOTALL)
print(f"Replaced tool_choice {count} times")

# 3. Update the tags
pattern_detect1 = r'tool_detect_buffer\.find\("<\\xef\\xbd\\x9c" "tool\\xe2\\x96\\x81calls\\xe2\\x96\\x81" "begin" "\\xef\\xbd\\x9c>"\)'
replacement_detect1 = r'tool_detect_buffer.find("<\xef\xbd\x9c" "DSML\xef\xbd\x9ctool_calls>")'
content, count = re.subn(pattern_detect1, replacement_detect1, content)

pattern_detect2 = r'tool_detect_buffer\.find\("<\\xef\\xbd\\x9c" "tool\\xe2\\x96\\x81calls\\xe2\\x96\\x81" "end" "\\xef\\xbd\\x9c>"\)'
replacement_detect2 = r'tool_detect_buffer.find("</\xef\xbd\x9c" "DSML\xef\xbd\x9ctool_calls>")'
content, count = re.subn(pattern_detect2, replacement_detect2, content)

# 4. Update the token type classification
pattern_class = r'(                    if \(in_think_block\) token_type = 1;\n                    else if \(in_tool_block \|\| tool_detect_buffer\.find\("<tool_call>"\) != std::string::npos \|\|\n                             tool_detect_buffer\.find\("<\\xef\\xbd\\x9c" "tool\\xe2\\x96\\x81calls\\xe2\\x96\\x81" "begin" "\\xef\\xbd\\x9c>"\) != std::string::npos \|\|\n                             tool_detect_buffer\.find\("<\\xef\\xbd\\x9c" "tool\\xe2\\x96\\x81call\\xe2\\x96\\x81" "begin" "\\xef\\xbd\\x9c>"\) != std::string::npos\) token_type = 2;)'
replacement_class = r'''                    if (in_think_block) token_type = 1;
                    else if (in_tool_block || tool_detect_buffer.find("<tool_call>") != std::string::npos ||
                             tool_detect_buffer.find("<\xef\xbd\x9c" "DSML\xef\xbd\x9ctool_calls>") != std::string::npos) token_type = 2;'''
content, count = re.subn(pattern_class, replacement_class, content, flags=re.DOTALL)
print(f"Replaced token_type class {count} times")


# 5. Replace parser logic
parse_pattern = r'(                            std::string begin_tag = "<\\xef\\xbd\\x9c" "tool\\xe2\\x96\\x81call\\xe2\\x96\\x81" "begin" "\\xef\\xbd\\x9c>";\n                            std::string end_tag = "<\\xef\\xbd\\x9c" "tool\\xe2\\x96\\x81call\\xe2\\x96\\x81" "end" "\\xef\\xbd\\x9c>";).*?(                                // Fallback: try parsing as a JSON array \(old format\))'

parse_replacement = r'''                            std::string invoke_start = "<\xef\xbd\x9c" "DSML\xef\xbd\x9cinvoke name=\"";
                            std::string param_start = "<\xef\xbd\x9c" "DSML\xef\xbd\x9cparameter name=\"";
                            std::string param_end = "</\xef\xbd\x9c" "DSML\xef\xbd\x9cparameter>";
                            std::string invoke_end = "</\xef\xbd\x9c" "DSML\xef\xbd\x9cinvoke>";
                            
                            size_t pos = 0;
                            int idx = 0;
                            while ((pos = tbuf.find(invoke_start, pos)) != std::string::npos) {
                                pos += invoke_start.length();
                                size_t name_end = tbuf.find("\">", pos);
                                if (name_end == std::string::npos) break;
                                std::string func_name = tbuf.substr(pos, name_end - pos);
                                pos = name_end + 2;
                                
                                size_t inv_end = tbuf.find(invoke_end, pos);
                                if (inv_end == std::string::npos) inv_end = tbuf.length();
                                
                                json args_json = json::object();
                                size_t ppos = pos;
                                while ((ppos = tbuf.find(param_start, ppos)) != std::string::npos && ppos < inv_end) {
                                    ppos += param_start.length();
                                    size_t pname_end = tbuf.find("\"", ppos);
                                    if (pname_end == std::string::npos) break;
                                    std::string param_name = tbuf.substr(ppos, pname_end - ppos);
                                    ppos = pname_end + 1;
                                    
                                    bool is_string = true;
                                    if (tbuf.substr(ppos, 9) == " string=\"") {
                                        ppos += 9;
                                        size_t pstr_end = tbuf.find("\">", ppos);
                                        if (pstr_end != std::string::npos) {
                                            std::string str_val = tbuf.substr(ppos, pstr_end - ppos);
                                            is_string = (str_val == "true");
                                            ppos = pstr_end + 2;
                                        }
                                    } else {
                                        size_t p_end = tbuf.find(">", ppos);
                                        if (p_end != std::string::npos) ppos = p_end + 1;
                                    }
                                    
                                    size_t pend = tbuf.find(param_end, ppos);
                                    if (pend == std::string::npos) break;
                                    
                                    std::string param_val = tbuf.substr(ppos, pend - ppos);
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
                            
                            if (tool_calls_json.empty()) {
\2'''

content, count2 = re.subn(parse_pattern, parse_replacement, content, flags=re.DOTALL)
print(f"Replaced parser logic {count2} times")

with open('src/server_single.cpp', 'w') as f:
    f.write(content)

