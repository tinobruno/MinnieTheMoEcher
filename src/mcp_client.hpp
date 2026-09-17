#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>
#include <atomic>
#include <cstring>
#include <nlohmann/json.hpp>
#include "version.hpp"

#if !defined(_WIN32) && !defined(_WIN64)
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#else
#include <io.h>
#include <process.h>
#ifndef _PID_T_DEFINED
#define _PID_T_DEFINED
typedef int pid_t;
#endif
#endif

namespace moecher {
namespace mcp {

using json = nlohmann::json;

// Logging helper (uses server standard if available)
inline void log_mcp(const char* level, const std::string& msg) {
    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char time_buf[32];
    std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    std::fprintf(stderr, "[%s] [%s] [MCP] %s\n", time_buf, level, msg.c_str());
    std::fflush(stderr);
}

inline std::string escape_shell_arg(const std::string& arg) {
    std::string out = "'";
    for (char c : arg) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

inline std::string execute_curl_command(const std::string& cmd, int timeout_ms = 30000) {
#if !defined(_WIN32) && !defined(_WIN64)
    int pipefd[2];
    if (pipe(pipefd) == -1) return "";
    pid_t pid = fork();
    if (pid == -1) {
        close(pipefd[0]);
        close(pipefd[1]);
        return "";
    }
    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            dup2(devnull, STDERR_FILENO);
            close(devnull);
        }
        close(pipefd[1]);
        execl("/bin/sh", "sh", "-c", cmd.c_str(), (char*)NULL);
        _exit(127);
    }
    close(pipefd[1]);
    std::string output;
    char buf[2048];
    int status;
    auto start = std::chrono::steady_clock::now();
    int flags = fcntl(pipefd[0], F_GETFL, 0);
    fcntl(pipefd[0], F_SETFL, flags | O_NONBLOCK);
    while (true) {
        auto el = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        if (el > timeout_ms) {
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            close(pipefd[0]);
            return "";
        }
        struct pollfd pfd;
        pfd.fd = pipefd[0];
        pfd.events = POLLIN;
        pfd.revents = 0;
        int p = poll(&pfd, 1, 100);
        if (p > 0 && (pfd.revents & POLLIN)) {
            ssize_t n = read(pipefd[0], buf, sizeof(buf) - 1);
            if (n > 0) {
                output.append(buf, n);
            } else if (n == 0) {
                break;
            }
        }
        if (waitpid(pid, &status, WNOHANG) == pid) {
            ssize_t n;
            while ((n = read(pipefd[0], buf, sizeof(buf) - 1)) > 0) {
                output.append(buf, n);
            }
            break;
        }
    }
    close(pipefd[0]);
    return output;
#else
    return "";
#endif
}

// Represents an MCP Server Configuration
struct MCPServerConfig {
    std::string id;
    std::string command;
    std::vector<std::string> args;
    std::string url = "";
    std::string transport_type = "stdio"; // "stdio", "streamable-http", "sse"
    std::unordered_map<std::string, std::string> env;
    bool enabled = true;
    std::string description = "";

    json to_json() const {
        json j = {
            {"command", command},
            {"args", args},
            {"env", env},
            {"enabled", enabled},
            {"transport_type", transport_type}
        };
        if (!url.empty()) j["url"] = url;
        if (!description.empty()) j["description"] = description;
        return j;
    }

    static MCPServerConfig from_json(const std::string& id, const json& j) {
        MCPServerConfig cfg;
        cfg.id = id;

        const json* target = &j;
        // Check if wrapped in "server" (official MCP registry schema 2025-10-17)
        if (j.contains("server") && j["server"].is_object()) {
            target = &j["server"];
            if (cfg.id.empty() || cfg.id == "custom" || cfg.id == "json-custom") {
                if (target->contains("name") && (*target)["name"].is_string()) {
                    std::string full_name = (*target)["name"].get<std::string>();
                    size_t slash_pos = full_name.find_last_of('/');
                    cfg.id = (slash_pos != std::string::npos) ? full_name.substr(slash_pos + 1) : full_name;
                }
            }
            if (target->contains("remotes") && (*target)["remotes"].is_array() && !(*target)["remotes"].empty()) {
                const auto& rem = (*target)["remotes"][0];
                if (rem.contains("url") && rem["url"].is_string()) {
                    cfg.url = rem["url"].get<std::string>();
                }
                if (rem.contains("type") && rem["type"].is_string()) {
                    cfg.transport_type = rem["type"].get<std::string>();
                }
            } else if (target->contains("packages") && (*target)["packages"].is_array() && !(*target)["packages"].empty()) {
                const auto& pkg = (*target)["packages"][0];
                std::string hint = pkg.value("runtimeHint", "");
                std::string reg = pkg.value("registryType", "");
                std::string id_pkg = pkg.value("identifier", "");
                if (hint.empty()) {
                    hint = (reg == "pypi") ? "uvx" : "npx";
                }
                cfg.command = hint;
                if (hint == "npx") {
                    cfg.args = {"-y", id_pkg};
                } else if (!id_pkg.empty()) {
                    cfg.args = {id_pkg};
                }
            }
        } else if (j.contains("mcpServers") && j["mcpServers"].is_object() && !j["mcpServers"].empty()) {
            auto first_it = j["mcpServers"].begin();
            if (cfg.id.empty() || cfg.id == "custom" || cfg.id == "json-custom") {
                cfg.id = first_it.key();
            }
            target = &(first_it.value());
        }

        if (target->contains("id") && (*target)["id"].is_string() && cfg.id.empty()) {
            cfg.id = (*target)["id"].get<std::string>();
        }
        if (target->contains("command") && (*target)["command"].is_string()) {
            cfg.command = (*target)["command"].get<std::string>();
        }
        if (target->contains("args") && (*target)["args"].is_array()) {
            for (const auto& a : (*target)["args"]) {
                if (a.is_string()) cfg.args.push_back(a.get<std::string>());
            }
        }
        if (target->contains("url") && (*target)["url"].is_string()) {
            cfg.url = (*target)["url"].get<std::string>();
            if (cfg.transport_type == "stdio") {
                cfg.transport_type = "streamable-http";
            }
        }
        if (target->contains("transport_type") && (*target)["transport_type"].is_string()) {
            cfg.transport_type = (*target)["transport_type"].get<std::string>();
        } else if (target->contains("transport") && (*target)["transport"].is_string()) {
            cfg.transport_type = (*target)["transport"].get<std::string>();
        }
        if (target->contains("env") && (*target)["env"].is_object()) {
            for (auto it = (*target)["env"].begin(); it != (*target)["env"].end(); ++it) {
                if (it.value().is_string()) {
                    cfg.env[it.key()] = it.value().get<std::string>();
                }
            }
        }
        if (target->contains("enabled") && (*target)["enabled"].is_boolean()) {
            cfg.enabled = (*target)["enabled"].get<bool>();
        }
        if (target->contains("description") && (*target)["description"].is_string()) {
            cfg.description = (*target)["description"].get<std::string>();
        } else if (target->contains("title") && (*target)["title"].is_string()) {
            cfg.description = (*target)["title"].get<std::string>();
        }
        return cfg;
    }
};

// Represents a Tool exposed by an MCP Server
struct MCPToolInfo {
    std::string server_id;
    std::string raw_name;
    std::string qualified_name; // e.g. "mcp__filesystem__read_file"
    std::string description;
    json input_schema;

    json to_openai_schema() const {
        return {
            {"type", "function"},
            {"function", {
                {"name", qualified_name},
                {"description", description.empty() ? ("MCP Tool from server '" + server_id + "'") : ("[" + server_id + " MCP] " + description)},
                {"parameters", input_schema.is_object() ? input_schema : json::object({{"type", "object"}, {"properties", json::object()}})}
            }}
        };
    }
};

// Represents an active MCP process communicating via stdio JSON-RPC 2.0
class MCPClientProcess {
public:
    explicit MCPClientProcess(MCPServerConfig config)
        : config_(std::move(config)) {}

    ~MCPClientProcess() {
        stop();
    }

    const MCPServerConfig& config() const { return config_; }
    void set_config(const MCPServerConfig& c) { config_ = c; }

    bool is_remote() const { return !config_.url.empty(); }

    bool is_running() {
        if (is_remote()) {
            return remote_ready_;
        }
#if !defined(_WIN32) && !defined(_WIN64)
        if (pid_ <= 0) return false;
        int status;
        pid_t res = waitpid(pid_, &status, WNOHANG);
        if (res == 0) return true; // still running
        if (res == pid_) {
            pid_ = -1;
            close_descriptors();
            return false;
        }
        return false;
#else
        return false;
#endif
    }

    int get_pid() const { return (int)pid_; }

    bool start() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (is_running()) return true;

        if (is_remote()) {
            log_mcp("INFO", "Connecting to remote MCP server '" + config_.id + "' [" + config_.transport_type + "]: " + config_.url);
            if (!perform_remote_handshake()) {
                log_mcp("ERROR", "Handshake failed for remote MCP server '" + config_.id + "': " + last_error_);
                remote_ready_ = false;
                return false;
            }
            remote_ready_ = true;
            refresh_tools_internal();
            return true;
        }

        if (config_.command.empty()) {
            last_error_ = "Command is empty";
            return false;
        }

#if !defined(_WIN32) && !defined(_WIN64)
        int stdin_pipe[2];
        int stdout_pipe[2];
        int stderr_pipe[2];

        if (pipe(stdin_pipe) == -1 || pipe(stdout_pipe) == -1 || pipe(stderr_pipe) == -1) {
            last_error_ = "Failed to create pipes";
            return false;
        }

        pid_t pid = fork();
        if (pid == -1) {
            last_error_ = "Failed to fork child process";
            close(stdin_pipe[0]); close(stdin_pipe[1]);
            close(stdout_pipe[0]); close(stdout_pipe[1]);
            close(stderr_pipe[0]); close(stderr_pipe[1]);
            return false;
        }

        if (pid == 0) {
            // Child process
            close(stdin_pipe[1]);   // Close write end
            dup2(stdin_pipe[0], STDIN_FILENO);
            close(stdin_pipe[0]);

            close(stdout_pipe[0]);  // Close read end
            dup2(stdout_pipe[1], STDOUT_FILENO);
            close(stdout_pipe[1]);

            close(stderr_pipe[0]);  // Close read end
            dup2(stderr_pipe[1], STDERR_FILENO);
            close(stderr_pipe[1]);

            // Set custom environment variables
            for (const auto& kv : config_.env) {
                setenv(kv.first.c_str(), kv.second.c_str(), 1);
            }

            // Build argv
            std::vector<char*> argv;
            argv.push_back(const_cast<char*>(config_.command.c_str()));
            for (const auto& arg : config_.args) {
                argv.push_back(const_cast<char*>(arg.c_str()));
            }
            argv.push_back(nullptr);

            execvp(config_.command.c_str(), argv.data());
            _exit(127);
        }

        // Parent process
        close(stdin_pipe[0]);
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);

        pid_ = pid;
        write_fd_ = stdin_pipe[1];
        read_fd_ = stdout_pipe[0];
        err_fd_ = stderr_pipe[0];

        // Set non-blocking on read descriptors
        int flags = fcntl(read_fd_, F_GETFL, 0);
        fcntl(read_fd_, F_SETFL, flags | O_NONBLOCK);
        flags = fcntl(err_fd_, F_GETFL, 0);
        fcntl(err_fd_, F_SETFL, flags | O_NONBLOCK);

        read_buffer_.clear();
        log_mcp("INFO", "Launched MCP server '" + config_.id + "' [PID " + std::to_string(pid_) + "]: " + config_.command);

        // Perform MCP JSON-RPC 2.0 handshake
        if (!perform_handshake()) {
            log_mcp("ERROR", "Handshake failed for MCP server '" + config_.id + "': " + last_error_);
            stop_internal();
            return false;
        }

        // Discover tools
        refresh_tools_internal();
        return true;
#else
        last_error_ = "Windows stdio MCP transport not implemented yet";
        return false;
#endif
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_internal();
    }

    const std::vector<MCPToolInfo>& get_tools() const {
        return tools_;
    }

    std::string get_last_error() const {
        return last_error_;
    }

    // Call tool over JSON-RPC tools/call
    std::string call_tool(const std::string& tool_name, const json& arguments, int timeout_ms = 60000) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!is_running()) {
            return "[MCP Error: Server '" + config_.id + "' is not running]";
        }

        json params = {
            {"name", tool_name},
            {"arguments", arguments.is_object() ? arguments : json::object()}
        };

        json resp;
        if (!send_request_internal("tools/call", params, resp, timeout_ms)) {
            return "[MCP Error: Server '" + config_.id + "' call failed: " + last_error_ + "]";
        }

        if (resp.contains("error")) {
            std::string errMsg = "Unknown error";
            if (resp["error"].contains("message")) {
                errMsg = resp["error"]["message"].get<std::string>();
            } else {
                errMsg = resp["error"].dump();
            }
            return "[MCP Tool Error (" + config_.id + ":" + tool_name + "): " + errMsg + "]";
        }

        if (resp.contains("result")) {
            const auto& res = resp["result"];
            std::string output = "";
            bool is_error = res.value("isError", false);

            if (res.contains("content") && res["content"].is_array()) {
                for (const auto& item : res["content"]) {
                    std::string type = item.value("type", "text");
                    if (type == "text" && item.contains("text")) {
                        if (!output.empty()) output += "\n";
                        output += item["text"].get<std::string>();
                    } else if (type == "resource" && item.contains("resource")) {
                        if (!output.empty()) output += "\n";
                        output += "[Resource: " + item["resource"].dump() + "]";
                    } else if (type == "image" && item.contains("data")) {
                        if (!output.empty()) output += "\n";
                        output += "[Image: " + item.value("mimeType", "image/png") + " (" + std::to_string(item["data"].get<std::string>().size()) + " b64 chars)]";
                    }
                }
            }

            if (output.empty()) {
                output = res.dump();
            }

            if (is_error) {
                return "[MCP Tool Returned Error]:\n" + output;
            }
            return output;
        }

        return resp.dump();
    }

    bool refresh_tools() {
        std::lock_guard<std::mutex> lock(mutex_);
        return refresh_tools_internal();
    }

private:
    MCPServerConfig config_;
    pid_t pid_ = -1;
    int write_fd_ = -1;
    int read_fd_ = -1;
    int err_fd_ = -1;
    std::string read_buffer_;
    std::vector<MCPToolInfo> tools_;
    std::string last_error_;
    std::atomic<uint64_t> request_counter_{1};
    std::mutex mutex_;
    bool remote_ready_ = false;

    void close_descriptors() {
#if !defined(_WIN32) && !defined(_WIN64)
        if (write_fd_ >= 0) { close(write_fd_); write_fd_ = -1; }
        if (read_fd_ >= 0)  { close(read_fd_);  read_fd_ = -1;  }
        if (err_fd_ >= 0)   { close(err_fd_);   err_fd_ = -1;   }
#endif
    }

    void stop_internal() {
        if (is_remote()) {
            remote_ready_ = false;
            tools_.clear();
            return;
        }
#if !defined(_WIN32) && !defined(_WIN64)
        if (pid_ > 0) {
            log_mcp("INFO", "Stopping MCP server '" + config_.id + "' [PID " + std::to_string(pid_) + "]");
            // Close write fd to send EOF to child stdin
            if (write_fd_ >= 0) {
                close(write_fd_);
                write_fd_ = -1;
            }

            // Give process up to 800ms to exit gracefully
            int status;
            bool exited = false;
            for (int i = 0; i < 8; ++i) {
                if (waitpid(pid_, &status, WNOHANG) == pid_) {
                    exited = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            if (!exited) {
                kill(pid_, SIGTERM);
                for (int i = 0; i < 5; ++i) {
                    if (waitpid(pid_, &status, WNOHANG) == pid_) {
                        exited = true;
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }

            if (!exited) {
                kill(pid_, SIGKILL);
                waitpid(pid_, &status, 0);
            }
            pid_ = -1;
        }
        close_descriptors();
        read_buffer_.clear();
        tools_.clear();
#endif
    }

    bool send_remote_request(const std::string& method, const json& params, json& out_response, int timeout_ms = 30000) {
        uint64_t req_id = request_counter_++;
        json req = {
            {"jsonrpc", "2.0"},
            {"id", req_id},
            {"method", method},
            {"params", params}
        };

        std::string curl_cmd = "curl -s -N -X POST " + escape_shell_arg(config_.url);
        curl_cmd += " -H " + escape_shell_arg("Content-Type: application/json");
        curl_cmd += " -H " + escape_shell_arg("Accept: application/json, text/event-stream");
        for (const auto& kv : config_.env) {
            curl_cmd += " -H " + escape_shell_arg(kv.first + ": " + kv.second);
        }
        curl_cmd += " -d " + escape_shell_arg(req.dump());

        std::string raw_res = execute_curl_command(curl_cmd, timeout_ms);
        if (raw_res.empty()) {
            last_error_ = "Remote endpoint returned empty response or timed out: " + config_.url;
            return false;
        }

        // 1. Try direct JSON parsing
        try {
            json parsed = json::parse(raw_res);
            if (parsed.is_object()) {
                out_response = std::move(parsed);
                return true;
            }
        } catch (...) {
            // Not standard single JSON, may be SSE stream
        }

        // 2. Parse Server-Sent Events (SSE) lines (data: { ... })
        std::istringstream stream(raw_res);
        std::string line;
        while (std::getline(stream, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
            if (line.rfind("data:", 0) == 0) {
                std::string data_content = line.substr(5);
                while (!data_content.empty() && data_content.front() == ' ') {
                    data_content.erase(0, 1);
                }
                if (data_content.empty()) continue;
                try {
                    json parsed = json::parse(data_content);
                    if (parsed.is_object() && (parsed.contains("result") || parsed.contains("error") || (parsed.contains("id") && parsed["id"] == req_id))) {
                        out_response = std::move(parsed);
                        return true;
                    }
                } catch (...) {
                    // Ignore malformed SSE lines
                }
            }
        }

        last_error_ = "Failed to parse remote JSON-RPC response from " + config_.url + ": " + (raw_res.size() > 200 ? raw_res.substr(0, 200) + "..." : raw_res);
        return false;
    }

    void send_remote_notification(const std::string& method, const json& params) {
        json req = {
            {"jsonrpc", "2.0"},
            {"method", method},
            {"params", params}
        };
        std::string curl_cmd = "curl -s -X POST " + escape_shell_arg(config_.url);
        curl_cmd += " -H " + escape_shell_arg("Content-Type: application/json");
        curl_cmd += " -H " + escape_shell_arg("Accept: application/json, text/event-stream");
        for (const auto& kv : config_.env) {
            curl_cmd += " -H " + escape_shell_arg(kv.first + ": " + kv.second);
        }
        curl_cmd += " -d " + escape_shell_arg(req.dump());
        execute_curl_command(curl_cmd, 5000);
    }

    bool perform_remote_handshake() {
        json init_params = {
            {"protocolVersion", "2024-11-05"},
            {"capabilities", {
                {"tools", json::object()}
            }},
            {"clientInfo", {
                {"name", "moecher"},
                {"version", moecher::VERSION}
            }}
        };

        json init_resp;
        if (!send_remote_request("initialize", init_params, init_resp, 15000)) {
            last_error_ = "Remote handshake initialize failed: " + last_error_;
            return false;
        }

        if (init_resp.contains("error")) {
            last_error_ = "Remote initialize returned error: " + init_resp["error"].dump();
            return false;
        }

        send_remote_notification("notifications/initialized", json::object());
        log_mcp("INFO", "Remote handshake succeeded with server '" + config_.id + "'");
        return true;
    }

    bool perform_handshake() {
        // Step 1: Send 'initialize'
        json init_params = {
            {"protocolVersion", "2024-11-05"},
            {"capabilities", {
                {"tools", json::object()}
            }},
            {"clientInfo", {
                {"name", "moecher"},
                {"version", moecher::VERSION}
            }}
        };

        json init_resp;
        if (!send_request_internal("initialize", init_params, init_resp, 15000)) {
            last_error_ = "Handshake initialize timed out or failed: " + last_error_;
            return false;
        }

        if (init_resp.contains("error")) {
            last_error_ = "Initialize returned error: " + init_resp["error"].dump();
            return false;
        }

        // Step 2: Send 'notifications/initialized'
        send_notification_internal("notifications/initialized", json::object());
        log_mcp("INFO", "Handshake succeeded with server '" + config_.id + "'");
        return true;
    }

    bool refresh_tools_internal() {
        tools_.clear();
        if (!is_running()) return false;

        json list_resp;
        if (!send_request_internal("tools/list", json::object(), list_resp, 10000)) {
            log_mcp("WARN", "tools/list failed for '" + config_.id + "': " + last_error_);
            return false;
        }

        if (list_resp.contains("result") && list_resp["result"].contains("tools") && list_resp["result"]["tools"].is_array()) {
            for (const auto& t : list_resp["result"]["tools"]) {
                if (!t.contains("name") || !t["name"].is_string()) continue;
                MCPToolInfo info;
                info.server_id = config_.id;
                info.raw_name = t["name"].get<std::string>();
                info.qualified_name = "mcp__" + config_.id + "__" + info.raw_name;
                info.description = t.value("description", "");
                if (t.contains("inputSchema") && t["inputSchema"].is_object()) {
                    info.input_schema = t["inputSchema"];
                } else {
                    info.input_schema = json::object({
                        {"type", "object"},
                        {"properties", json::object()}
                    });
                }
                tools_.push_back(std::move(info));
            }
            log_mcp("INFO", "Discovered " + std::to_string(tools_.size()) + " tools from server '" + config_.id + "'");
            return true;
        }
        return false;
    }

    void send_notification_internal(const std::string& method, const json& params) {
        if (is_remote()) {
            send_remote_notification(method, params);
            return;
        }
#if !defined(_WIN32) && !defined(_WIN64)
        if (write_fd_ < 0) return;
        json notif = {
            {"jsonrpc", "2.0"},
            {"method", method},
            {"params", params}
        };
        std::string line = notif.dump() + "\n";
        ssize_t written = write(write_fd_, line.data(), line.size());
        (void)written;
#endif
    }

    bool send_request_internal(const std::string& method, const json& params, json& out_response, int timeout_ms = 30000) {
        if (is_remote()) {
            return send_remote_request(method, params, out_response, timeout_ms);
        }
#if !defined(_WIN32) && !defined(_WIN64)
        if (write_fd_ < 0 || read_fd_ < 0) {
            last_error_ = "Process pipes not open";
            return false;
        }

        uint64_t req_id = request_counter_++;
        json req = {
            {"jsonrpc", "2.0"},
            {"id", req_id},
            {"method", method},
            {"params", params}
        };

        std::string line = req.dump() + "\n";
        ssize_t written = write(write_fd_, line.data(), line.size());
        if (written <= 0) {
            last_error_ = "Failed to write to server stdin: " + std::string(strerror(errno));
            return false;
        }

        auto start_time = std::chrono::steady_clock::now();

        while (true) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count();
            if (elapsed >= timeout_ms) {
                last_error_ = "Request '" + method + "' timed out after " + std::to_string(timeout_ms) + "ms";
                return false;
            }

            int remaining_timeout = timeout_ms - (int)elapsed;
            struct pollfd pfds[2];
            pfds[0].fd = read_fd_;
            pfds[0].events = POLLIN;
            pfds[0].revents = 0;

            pfds[1].fd = err_fd_;
            pfds[1].events = POLLIN;
            pfds[1].revents = 0;

            int poll_res = poll(pfds, 2, std::min(remaining_timeout, 200));
            if (poll_res < 0) {
                if (errno == EINTR) continue;
                last_error_ = "Poll failed: " + std::string(strerror(errno));
                return false;
            }

            // Read stderr for logging if available
            if (pfds[1].revents & POLLIN) {
                char err_buf[512];
                ssize_t err_bytes = read(err_fd_, err_buf, sizeof(err_buf) - 1);
                if (err_bytes > 0) {
                    err_buf[err_bytes] = '\0';
                    log_mcp("DEBUG", "[" + config_.id + ":stderr] " + std::string(err_buf));
                }
            }

            // Check if process died
            if (!is_running()) {
                last_error_ = "MCP server process died unexpectedly";
                return false;
            }

            if (pfds[0].revents & POLLIN) {
                char buf[1024];
                ssize_t bytes_read = read(read_fd_, buf, sizeof(buf));
                if (bytes_read > 0) {
                    read_buffer_.append(buf, bytes_read);

                    // Check for complete lines in read_buffer_
                    size_t newline_pos;
                    while ((newline_pos = read_buffer_.find('\n')) != std::string::npos) {
                        std::string msg_line = read_buffer_.substr(0, newline_pos);
                        read_buffer_.erase(0, newline_pos + 1);

                        // Strip trailing \r if any
                        if (!msg_line.empty() && msg_line.back() == '\r') {
                            msg_line.pop_back();
                        }
                        if (msg_line.empty()) continue;

                        try {
                            json parsed = json::parse(msg_line);
                            // Check if it's our response
                            if (parsed.contains("id") && parsed["id"] == req_id) {
                                out_response = std::move(parsed);
                                return true;
                            }
                            // Else if it's a notification or different id, log and continue
                            if (parsed.contains("method")) {
                                log_mcp("DEBUG", "Received notification from '" + config_.id + "': " + parsed["method"].get<std::string>());
                            }
                        } catch (const std::exception& e) {
                            log_mcp("WARN", "JSON parse error from '" + config_.id + "': " + std::string(e.what()) + " in line: " + msg_line);
                        }
                    }
                } else if (bytes_read == 0) {
                    last_error_ = "Server closed stdout pipe";
                    return false;
                }
            }
        }
#else
        last_error_ = "Windows stdio MCP transport not implemented yet";
        return false;
#endif
    }
};

// Top-level Manager for all MCP Servers and Tool Dispatching
class MCPManager {
public:
    static MCPManager& instance() {
        static MCPManager inst;
        return inst;
    }

    void set_config_path(const std::string& path) {
        std::lock_guard<std::mutex> lock(mutex_);
        config_path_ = path;
    }

    std::string get_config_path() const {
        return config_path_;
    }

    // Load servers from mcp_servers.json
    bool load_config(const std::string& path = "") {
        std::lock_guard<std::mutex> lock(mutex_);
        std::string target_path = path.empty() ? config_path_ : path;
        config_path_ = target_path;

        std::ifstream f(target_path);
        if (!f.is_open()) {
            log_mcp("INFO", "No existing MCP config file found at: " + target_path + " (will create on save)");
            return false;
        }

        json j;
        try {
            f >> j;
        } catch (const std::exception& e) {
            log_mcp("ERROR", "Failed to parse " + target_path + ": " + e.what());
            return false;
        }

        // Supports standard format: { "mcpServers": { "id": { "command": ..., "args": ... } } }
        json servers_obj = json::object();
        if (j.contains("mcpServers") && j["mcpServers"].is_object()) {
            servers_obj = j["mcpServers"];
        } else if (j.contains("servers") && j["servers"].is_object()) {
            servers_obj = j["servers"];
        }

        for (auto it = servers_obj.begin(); it != servers_obj.end(); ++it) {
            std::string s_id = it.key();
            MCPServerConfig cfg = MCPServerConfig::from_json(s_id, it.value());
            configs_[s_id] = cfg;
        }

        log_mcp("INFO", "Loaded " + std::to_string(configs_.size()) + " MCP server configurations from " + target_path);
        return true;
    }

    // Save configurations back to mcp_servers.json
    bool save_config() {
        std::lock_guard<std::mutex> lock(mutex_);
        json root = json::object();
        json mcp_servers = json::object();

        for (const auto& kv : configs_) {
            mcp_servers[kv.first] = kv.second.to_json();
        }
        root["mcpServers"] = mcp_servers;

        std::ofstream f(config_path_);
        if (!f.is_open()) {
            log_mcp("ERROR", "Failed to write MCP config to " + config_path_);
            return false;
        }
        f << root.dump(2) << std::endl;
        log_mcp("INFO", "Saved " + std::to_string(configs_.size()) + " MCP servers to " + config_path_);
        return true;
    }

    // Start all enabled servers
    void start_all_enabled() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& kv : configs_) {
            if (kv.second.enabled) {
                start_server_internal(kv.first);
            }
        }
    }

    // Stop all running servers
    void stop_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& kv : processes_) {
            kv.second->stop();
        }
        processes_.clear();
    }

    bool start_server(const std::string& id) {
        std::lock_guard<std::mutex> lock(mutex_);
        return start_server_internal(id);
    }

    void stop_server(const std::string& id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = processes_.find(id);
        if (it != processes_.end()) {
            it->second->stop();
            processes_.erase(it);
        }
    }

    bool restart_server(const std::string& id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = processes_.find(id);
        if (it != processes_.end()) {
            it->second->stop();
            processes_.erase(it);
        }
        return start_server_internal(id);
    }

    bool add_or_update_server(const MCPServerConfig& config, bool auto_start = true) {
        std::lock_guard<std::mutex> lock(mutex_);
        configs_[config.id] = config;

        // Stop existing process if running
        auto it = processes_.find(config.id);
        if (it != processes_.end()) {
            it->second->stop();
            processes_.erase(it);
        }

        if (config.enabled && auto_start) {
            start_server_internal(config.id);
        }
        return true;
    }

    bool remove_server(const std::string& id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = processes_.find(id);
        if (it != processes_.end()) {
            it->second->stop();
            processes_.erase(it);
        }
        configs_.erase(id);
        return true;
    }

    bool toggle_server(const std::string& id, bool enabled) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = configs_.find(id);
        if (it == configs_.end()) return false;

        it->second.enabled = enabled;
        if (!enabled) {
            auto pit = processes_.find(id);
            if (pit != processes_.end()) {
                pit->second->stop();
                processes_.erase(pit);
            }
        } else {
            start_server_internal(id);
        }
        return true;
    }

    // Aggregates tools from all active running servers
    std::vector<MCPToolInfo> get_all_tools() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<MCPToolInfo> all_tools;
        for (const auto& kv : processes_) {
            if (kv.second->is_running()) {
                const auto& server_tools = kv.second->get_tools();
                all_tools.insert(all_tools.end(), server_tools.begin(), server_tools.end());
            }
        }
        return all_tools;
    }

    // Returns OpenAI-compatible tools array for model prompt
    json get_openai_tools_schema() {
        std::vector<MCPToolInfo> tools = get_all_tools();
        json arr = json::array();
        for (const auto& t : tools) {
            arr.push_back(t.to_openai_schema());
        }
        return arr;
    }

    // Check if tool name belongs to any MCP server
    bool has_tool(const std::string& tool_name, std::string* out_server_id = nullptr, std::string* out_raw_name = nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);

        // 1. Check for qualified name: "mcp__<server_id>__<raw_name>"
        if (tool_name.rfind("mcp__", 0) == 0) {
            size_t second_delim = tool_name.find("__", 5);
            if (second_delim != std::string::npos) {
                std::string s_id = tool_name.substr(5, second_delim - 5);
                std::string r_name = tool_name.substr(second_delim + 2);
                auto it = processes_.find(s_id);
                if (it != processes_.end() && it->second->is_running()) {
                    if (out_server_id) *out_server_id = s_id;
                    if (out_raw_name) *out_raw_name = r_name;
                    return true;
                }
            }
        }

        // 2. Check for raw tool name match across running servers
        for (const auto& kv : processes_) {
            if (kv.second->is_running()) {
                for (const auto& t : kv.second->get_tools()) {
                    if (t.raw_name == tool_name || t.qualified_name == tool_name) {
                        if (out_server_id) *out_server_id = kv.first;
                        if (out_raw_name) *out_raw_name = t.raw_name;
                        return true;
                    }
                }
            }
        }
        return false;
    }

    // Execute tool call on appropriate server
    std::string call_tool(const std::string& tool_name, const json& arguments, int timeout_ms = 60000) {
        std::string server_id;
        std::string raw_name;

        if (!has_tool(tool_name, &server_id, &raw_name)) {
            return "[Error: MCP Tool '" + tool_name + "' not found or server is inactive]";
        }

        std::shared_ptr<MCPClientProcess> proc;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = processes_.find(server_id);
            if (it == processes_.end() || !it->second->is_running()) {
                return "[Error: MCP server '" + server_id + "' is not running]";
            }
            proc = it->second;
        }

        log_mcp("INFO", "Executing MCP tool '" + raw_name + "' on server '" + server_id + "'");
        return proc->call_tool(raw_name, arguments, timeout_ms);
    }

    // Status report JSON for UI dashboard
    json get_servers_status_json() {
        std::lock_guard<std::mutex> lock(mutex_);
        json arr = json::array();

        for (const auto& kv : configs_) {
            const auto& cfg = kv.second;
            bool is_running = false;
            int pid = -1;
            std::string err = "";
            json tools_arr = json::array();

            auto it = processes_.find(kv.first);
            if (it != processes_.end()) {
                is_running = it->second->is_running();
                pid = it->second->get_pid();
                err = it->second->get_last_error();
                for (const auto& t : it->second->get_tools()) {
                    tools_arr.push_back({
                        {"name", t.raw_name},
                        {"qualified_name", t.qualified_name},
                        {"description", t.description},
                        {"input_schema", t.input_schema}
                    });
                }
            }

            arr.push_back({
                {"id", cfg.id},
                {"command", cfg.command},
                {"args", cfg.args},
                {"url", cfg.url},
                {"transport_type", cfg.transport_type},
                {"env", cfg.env},
                {"enabled", cfg.enabled},
                {"description", cfg.description},
                {"running", is_running},
                {"pid", pid},
                {"error", err},
                {"tools_count", tools_arr.size()},
                {"tools", tools_arr}
            });
        }
        return arr;
    }

private:
    std::string config_path_ = "mcp_servers.json";
    std::unordered_map<std::string, MCPServerConfig> configs_;
    std::unordered_map<std::string, std::shared_ptr<MCPClientProcess>> processes_;
    std::mutex mutex_;

    MCPManager() = default;
    ~MCPManager() {
        stop_all();
    }

    bool start_server_internal(const std::string& id) {
        auto it = configs_.find(id);
        if (it == configs_.end()) return false;

        auto proc_it = processes_.find(id);
        if (proc_it != processes_.end() && proc_it->second->is_running()) {
            return true;
        }

        auto client = std::make_shared<MCPClientProcess>(it->second);
        bool ok = client->start();
        processes_[id] = client;
        return ok;
    }
};

} // namespace mcp
} // namespace moecher
