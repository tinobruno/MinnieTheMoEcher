#pragma once

#include <string>
#include <nlohmann/json.hpp>

class MoecherEngine;

namespace moecher::server {

void rebuild_system_prefix(
    MoecherEngine& engine,
    const std::string& base_prompt,
    const nlohmann::json& active_tools,
    const std::string& custom_full_prompt = ""
);

void run_server(
    MoecherEngine& engine,
    int port,
    int default_thinking_budget = 4096,
    int proxy_port = 8002,
    bool enable_forward_proxy = true,
    bool enable_system_proxy = false
);

} // namespace moecher::server

using moecher::server::rebuild_system_prefix;
using moecher::server::run_server;
