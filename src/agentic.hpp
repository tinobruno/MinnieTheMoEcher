#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <nlohmann/json.hpp>

namespace httplib {
    class Server;
}

namespace moecher::proxy {
    extern std::atomic<bool> g_proxy_running;
    extern std::atomic<int> g_proxy_port;

    bool start_forward_proxy(int proxy_port);
    void stop_forward_proxy();
    void set_windows_system_proxy(bool enable, const std::string& proxy_server);
}

namespace moecher::agentic {
    extern std::string g_base_system_prompt;
    extern std::string g_current_system_prompt;
    extern nlohmann::json g_current_active_tools;
    extern const std::string g_server_instance_id;

    nlohmann::json resolve_canonical_tools(const nlohmann::json& tools_input);
    std::string build_dynamic_tools_prompt(const nlohmann::json& resolved_tools, bool is_qwen = false);
    std::string update_system_prompt_with_tools(const std::string& original_content, const std::string& tools_system_prompt, bool has_tools);

    void register_agentic_routes(httplib::Server& svr);
}

// Global aliases for transparent static-linking compatibility
using moecher::agentic::resolve_canonical_tools;
using moecher::agentic::build_dynamic_tools_prompt;
using moecher::agentic::update_system_prompt_with_tools;
using moecher::agentic::g_base_system_prompt;
using moecher::agentic::g_current_system_prompt;
using moecher::agentic::g_current_active_tools;
using moecher::agentic::g_server_instance_id;
