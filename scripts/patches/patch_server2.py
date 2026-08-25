with open('src/server_single.cpp', 'r') as f:
    content = f.read()

s1 = '                    if (in_think_block) token_type = 1;\n                    else if (in_tool_block || tool_detect_buffer.find("<tool_call>") != std::string::npos ||\n                             tool_detect_buffer.find("<\\xef\\xbd\\x9c" "tool\\xe2\\x96\\x81calls\\xe2\\x96\\x81" "begin" "\\xef\\xbd\\x9c>") != std::string::npos ||\n                             tool_detect_buffer.find("<\\xef\\xbd\\x9c" "tool\\xe2\\x96\\x81call\\xe2\\x96\\x81" "begin" "\\xef\\xbd\\x9c>") != std::string::npos) token_type = 2;'

s2 = '                    if (in_think_block) token_type = 1;\n                    else if (in_tool_block || tool_detect_buffer.find("<tool_call>") != std::string::npos ||\n                             tool_detect_buffer.find("<\\xef\\xbd\\x9cDSML\\xef\\xbd\\x9ctool_calls>") != std::string::npos) token_type = 2;'

if s1 in content:
    content = content.replace(s1, s2)
    with open('src/server_single.cpp', 'w') as f:
        f.write(content)
    print("Replaced token_type block successfully!")
else:
    print("Failed to find token_type block!")

