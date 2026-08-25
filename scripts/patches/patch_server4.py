with open('src/server_single.cpp', 'r') as f:
    content = f.read()

# Replace token type assignment
s1 = """                    if (in_think_block) token_type = 1;
                    else if (in_tool_block || tool_detect_buffer.find("<tool_call>") != std::string::npos ||
                             tool_detect_buffer.find("<\\xef\\xbd\\x9c" "tool\\xe2\\x96\\x81calls\\xe2\\x96\\x81" "begin" "\\xef\\xbd\\x9c>") != std::string::npos ||
                             tool_detect_buffer.find("<\\xef\\xbd\\x9c" "tool\\xe2\\x96\\x81call\\xe2\\x96\\x81" "begin" "\\xef\\xbd\\x9c>") != std::string::npos) token_type = 2;"""

s2 = """                    if (in_think_block) token_type = 1;
                    else if (in_tool_block || tool_detect_buffer.find("<tool_call>") != std::string::npos ||
                             tool_detect_buffer.find("<\\xef\\xbd\\x9cDSML\\xef\\xbd\\x9ctool_calls>") != std::string::npos) token_type = 2;"""

if s1 in content:
    content = content.replace(s1, s2)
    print("Replaced token type!")
else:
    print("Could not find token_type block to replace!")

# Replace parser block
parse_s1 = """                            std::string begin_tag = "<\\xef\\xbd\\x9c" "tool\\xe2\\x96\\x81call\\xe2\\x96\\x81" "begin" "\\xef\\xbd\\x9c>";
                            std::string end_tag = "<\\xef\\xbd\\x9c" "tool\\xe2\\x96\\x81call\\xe2\\x96\\x81" "end" "\\xef\\xbd\\x9c>";
                            
                            size_t pos = 0;
                            int idx = 0;
                            while ((pos = tbuf.find(begin_tag, pos)) != std::string::npos) {
                                pos += begin_tag.length();
                                size_t end_pos = tbuf.find(end_tag, pos);
                                std::string json_str;
                                if (end_pos != std::string::npos) {
                                    json_str = tbuf.substr(pos, end_pos - pos);
                                    pos = end_pos + end_tag.length();
                                } else {
                                    json_str = tbuf.substr(pos);
                                    pos = std::string::npos;
                                }
                                
                                try {
                                    json tc = json::parse(json_str);
                                    if (tc.contains("name") && tc.contains("arguments")) {
                                        std::string args_str;
                                        if (tc["arguments"].is_string()) {
                                            args_str = tc["arguments"].get<std::string>();
                                        } else {
                                            args_str = tc["arguments"].dump();
                                        }
                                        tool_calls_json.push_back({
                                            {"index", idx++},
                                            {"id", "call_" + std::to_string(rand())},
                                            {"type", "function"},
                                            {"function", {
                                                {"name", tc["name"].get<std::string>()},
                                                {"arguments", args_str}
                                            }}
                                        });
                                    }
                                } catch (const std::exception& e) {
                                    LOG_INFO("Failed to parse tool call JSON: %s\\nText: %s", e.what(), json_str.c_str());
                                }
                                
                                if (pos == std::string::npos) break;
                            }"""

parse_s2 = """                            std::string invoke_start = "<\\xef\\xbd\\x9cDSML\\xef\\xbd\\x9cinvoke name=\\\"";
                            std::string param_start = "<\\xef\\xbd\\x9cDSML\\xef\\xbd\\x9cparameter name=\\\"";
                            std::string param_end = "</\\xef\\xbd\\x9cDSML\\xef\\xbd\\x9cparameter>";
                            std::string invoke_end = "</\\xef\\xbd\\x9cDSML\\xef\\xbd\\x9cinvoke>";
                            
                            size_t pos = 0;
                            int idx = 0;
                            while ((pos = tbuf.find(invoke_start, pos)) != std::string::npos) {
                                pos += invoke_start.length();
                                size_t name_end = tbuf.find("\\\">", pos);
                                if (name_end == std::string::npos) break;
                                std::string func_name = tbuf.substr(pos, name_end - pos);
                                pos = name_end + 2;
                                
                                size_t inv_end = tbuf.find(invoke_end, pos);
                                if (inv_end == std::string::npos) inv_end = tbuf.length();
                                
                                json args_json = json::object();
                                size_t ppos = pos;
                                while ((ppos = tbuf.find(param_start, ppos)) != std::string::npos && ppos < inv_end) {
                                    ppos += param_start.length();
                                    size_t pname_end = tbuf.find("\\\"", ppos);
                                    if (pname_end == std::string::npos) break;
                                    std::string param_name = tbuf.substr(ppos, pname_end - ppos);
                                    ppos = pname_end + 1;
                                    
                                    bool is_string = true;
                                    if (tbuf.substr(ppos, 9) == " string=\\\"") {
                                        ppos += 9;
                                        size_t pstr_end = tbuf.find("\\\">", ppos);
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
                            }"""

if parse_s1 in content:
    content = content.replace(parse_s1, parse_s2)
    print("Replaced parser block!")
else:
    print("Could not find parser block to replace!")

with open('src/server_single.cpp', 'w') as f:
    f.write(content)

