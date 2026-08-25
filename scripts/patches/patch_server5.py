with open('src/server_single.cpp', 'r') as f:
    content = f.read()

import re

# token assignment
s_pattern = r'token_type = 2;\n'
# Let's find exactly lines 1415-1425
lines = content.splitlines()
target_lines = lines[1415:1430]
for i, line in enumerate(target_lines):
    if "token_type = 2;" in line:
        idx = 1415 + i
        # Replace the preceding 4 lines
        lines[idx-3] = '                    int token_type = 0;'
        lines[idx-2] = '                    if (in_think_block) token_type = 1;'
        lines[idx-1] = '                    else if (in_tool_block || tool_detect_buffer.find("<tool_call>") != std::string::npos ||'
        lines[idx] = '                             tool_detect_buffer.find("<\\xef\\xbd\\x9cDSML\\xef\\xbd\\x9ctool_calls>") != std::string::npos) token_type = 2;'
        break

# tools parsing JSON
parse_start_idx = -1
parse_end_idx = -1
for i, line in enumerate(lines):
    if 'std::string begin_tag = "<\\xef\\xbd\\x9c" "tool\\xe2\\x96\\x81call\\xe2\\x96\\x81" "begin" "\\xef\\xbd\\x9c>";' in line:
        parse_start_idx = i
    if '                                // Fallback: try parsing as a JSON array (old format)' in line:
        parse_end_idx = i
        break

if parse_start_idx != -1 and parse_end_idx != -1:
    replacement = """                            std::string invoke_start = "<\\xef\\xbd\\x9cDSML\\xef\\xbd\\x9cinvoke name=\\\"";
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
                            }
                            
                            if (tool_calls_json.empty()) {"""
    new_lines = lines[:parse_start_idx] + replacement.splitlines() + lines[parse_end_idx:]
    with open('src/server_single.cpp', 'w') as f:
        f.write("\n".join(new_lines) + "\n")
    print("Replaced token assign & parser logic via robust Python script!")
else:
    print(f"Could not find parse block! start={parse_start_idx} end={parse_end_idx}")

