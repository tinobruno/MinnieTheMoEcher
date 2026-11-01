#pragma once

#include <string>
#include <mutex>
#include <fstream>
#include <atomic>

extern bool g_log_experts;
extern bool g_log_tokens;
extern bool g_quiet;
extern bool g_server_ready;
extern bool g_track_experts;
extern bool g_track_reset;
extern bool g_enable_tools;
extern bool g_server_exec;
extern bool g_allow_file_write;
extern std::string g_workspace_root;
extern bool g_headless_browsing;
extern int g_max_tool_rounds;
extern bool g_intent_gating;
extern bool g_intent_gating_cli_explicit;
extern std::atomic<bool> g_stop_requested;
extern std::ofstream g_log_file;

void log_msg(const char* level, const char* fmt, ...);

#define LOG_INFO(...)  log_msg("INFO",  __VA_ARGS__)
#define LOG_WARN(...)  log_msg("WARN",  __VA_ARGS__)
#define LOG_ERROR(...) log_msg("ERROR", __VA_ARGS__)
