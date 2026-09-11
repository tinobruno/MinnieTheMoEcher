#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "tool_exec.hpp"

using json = nlohmann::json;

void test_extract_tool_calls() {
    std::cout << "[TEST] Testing extract_tool_calls..." << std::endl;

    // 1. OpenAI-style XML <tool_call>
    std::string text1 = "I will check the files for you.\n<tool_call>\n{\"name\": \"execute_command\", \"arguments\": {\"command\": \"ls -la\"}}\n</tool_call>";
    std::string clean1;
    std::vector<moecher::tooling::ToolCall> tcs1;
    moecher::tooling::extract_tool_calls(text1, clean1, tcs1);

    assert(tcs1.size() == 1);
    assert(tcs1[0].name == "execute_command");
    assert(clean1.find("I will check the files for you.") != std::string::npos);
    std::cout << "  ✓ <tool_call> extraction passed!" << std::endl;

    // 2. Bare JSON tool call
    std::string text2 = "Let me run a command:\n```json\n{\"name\": \"execute_command\", \"arguments\": {\"command\": \"whoami\"}}\n```";
    std::string clean2;
    std::vector<moecher::tooling::ToolCall> tcs2;
    moecher::tooling::extract_tool_calls(text2, clean2, tcs2);
    assert(tcs2.size() == 1);
    assert(tcs2[0].name == "execute_command");
    std::cout << "  ✓ Bare JSON tool call extraction passed!" << std::endl;
}

void test_tools_resolution() {
    std::cout << "[TEST] Testing canonical tool schemas..." << std::endl;
    std::string tools_str = moecher::tooling::get_builtin_tools_json();
    json tools = json::parse(tools_str);
    assert(tools.is_array());
    assert(tools.size() >= 5);

    bool has_cmd = false;
    bool has_fetch = false;
    bool has_read = false;
    bool has_write = false;
    bool has_search = false;
    for (const auto& t : tools) {
        if (t.contains("function") && t["function"].contains("name")) {
            std::string name = t["function"]["name"].get<std::string>();
            if (name == "execute_command") has_cmd = true;
            if (name == "fetch_url") has_fetch = true;
            if (name == "read_file") has_read = true;
            if (name == "write_file") has_write = true;
            if (name == "web_search") has_search = true;
        }
    }
    assert(has_cmd);
    assert(has_fetch);
    assert(has_read);
    assert(has_write);
    assert(has_search);
    std::cout << "  ✓ Tool schemas verified (found " << tools.size() << " built-in tools)!" << std::endl;
}

void test_tool_execution() {
    std::cout << "[TEST] Testing local tool execution..." << std::endl;

    // 1. Local command execution
    std::string cmd_res = moecher::tooling::execute_system_command("echo 'MinnieTheMoEcher Agentic 2.03'");
    std::cout << "  System command result: " << cmd_res;
    assert(cmd_res.find("MinnieTheMoEcher Agentic 2.03") != std::string::npos);

    // 2. File write, read, edit
    std::string test_file = "test_agentic_tmp.txt";
    std::string w_res = moecher::tooling::write_file(test_file, "Line 1: Alpha\nLine 2: Beta\nLine 3: Gamma\n", true);
    assert(w_res.find("Successfully wrote") != std::string::npos || w_res.find("bytes") != std::string::npos);

    std::string r_res = moecher::tooling::read_file(test_file, 1, -1);
    assert(r_res.find("Alpha") != std::string::npos);
    assert(r_res.find("Beta") != std::string::npos);
    assert(r_res.find("Gamma") != std::string::npos);

    std::string e_res = moecher::tooling::edit_file(test_file, "Line 2: Beta\n", "Line 2: Beta Updated\n");
    assert(e_res.find("Successfully replaced") != std::string::npos || e_res.find("edited") != std::string::npos);

    std::string r2_res = moecher::tooling::read_file(test_file, 1, -1);
    assert(r2_res.find("Beta Updated") != std::string::npos);

    // Clean up
    moecher::tooling::execute_system_command("rm -f test_agentic_tmp.txt");

    std::cout << "  ✓ Local file and command tool execution passed!" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << " Running MinnieTheMoEcher Agentic Tests " << std::endl;
    std::cout << "========================================" << std::endl;
    test_extract_tool_calls();
    test_tools_resolution();
    test_tool_execution();
    std::cout << "\nALL AGENTIC UNIT TESTS PASSED SUCCESSFULLY! ✓✓✓" << std::endl;
    return 0;
}
