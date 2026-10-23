#include "logging.hpp"
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>

std::mutex g_log_mutex;
std::ofstream g_log_file;
bool g_log_experts = false;
bool g_log_tokens = true;
bool g_quiet = false;
bool g_server_ready = false;
bool g_track_experts = false;
bool g_track_reset = false;
bool g_enable_tools = false;
bool g_server_exec = false;
bool g_allow_file_write = false;
std::string g_workspace_root = "./workspace";
bool g_headless_browsing = false;
int g_max_tool_rounds = 10;
std::atomic<bool> g_stop_requested{false};

void log_msg(const char* level, const char* fmt, ...) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    char timebuf[64];
    strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", localtime(&time));

    char buf[4096];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    if (!g_quiet || !g_server_ready || strstr(buf, "[PREFILL STATS]") || strstr(buf, "[GENERATION STATS]") || strstr(buf, "[SPECULATIVE STATS]") || strstr(buf, "VERIFY TIMING") || strstr(buf, "MTP DEBUG") || strstr(buf, "SINGLE FWD TIMING") || strcmp(level, "ERROR") == 0) {
        fprintf(stderr, "[%s] [%s] %s\n", timebuf, level, buf);
        fflush(stderr);
    }
    if (g_log_file.is_open()) {
        g_log_file << "[" << timebuf << "] [" << level << "] " << buf << "\n";
        g_log_file.flush();
    }
}
