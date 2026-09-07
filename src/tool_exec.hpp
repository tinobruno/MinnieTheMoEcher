// tool_exec.hpp — Cross-platform system command execution and URL retrieval
// Supports both Windows (Win32 CreateProcess, WinHTTP) and Linux (POSIX fork/exec, curl/httplib)

#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <iostream>
#include <nlohmann/json.hpp>

#if defined(_WIN32) || defined(_WIN64)
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #include <winhttp.h>
    #pragma comment(lib, "winhttp.lib")
#else
    #include <unistd.h>
    #include <sys/types.h>
    #include <sys/wait.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <poll.h>
    #include <signal.h>
#endif

namespace moecher {
namespace tooling {

// ════════════════════════════════════════════════════════════════════════════════
//  HTML-to-Text Stripper (Cross-Platform)
// ════════════════════════════════════════════════════════════════════════════════

inline std::string strip_html_to_text(const std::string& html, size_t max_chars = 4000) {
    std::string text;
    text.reserve(std::min(html.size(), max_chars * 2));

    bool in_tag = false;
    bool in_script = false;
    bool in_style = false;
    std::string current_tag;

    for (size_t i = 0; i < html.size() && text.size() < max_chars; ++i) {
        char c = html[i];

        if (c == '<') {
            in_tag = true;
            current_tag.clear();
            continue;
        }

        if (in_tag) {
            if (c == '>') {
                in_tag = false;
                std::string lower_tag = current_tag;
                std::transform(lower_tag.begin(), lower_tag.end(), lower_tag.begin(),
                               [](unsigned char ch) { return (char)::tolower(ch); });

                if (lower_tag == "script" || lower_tag.rfind("script ", 0) == 0) {
                    in_script = true;
                } else if (lower_tag == "/script") {
                    in_script = false;
                } else if (lower_tag == "style" || lower_tag.rfind("style ", 0) == 0) {
                    in_style = true;
                } else if (lower_tag == "/style") {
                    in_style = false;
                } else if (lower_tag == "p" || lower_tag == "br" || lower_tag == "div" ||
                           lower_tag == "h1" || lower_tag == "h2" || lower_tag == "h3" ||
                           lower_tag == "li" || lower_tag == "tr") {
                    if (!text.empty() && text.back() != '\n') {
                        text.push_back('\n');
                    }
                }
            } else {
                current_tag.push_back(c);
            }
            continue;
        }

        if (in_script || in_style) {
            continue;
        }

        // Entity replacement
        if (c == '&') {
            if (html.compare(i, 4, "&lt;") == 0) { text.push_back('<'); i += 3; continue; }
            if (html.compare(i, 4, "&gt;") == 0) { text.push_back('>'); i += 3; continue; }
            if (html.compare(i, 5, "&amp;") == 0) { text.push_back('&'); i += 4; continue; }
            if (html.compare(i, 6, "&quot;") == 0) { text.push_back('"'); i += 5; continue; }
            if (html.compare(i, 6, "&nbsp;") == 0) { text.push_back(' '); i += 5; continue; }
        }

        if (c == '\r') continue;

        if (std::isspace((unsigned char)c)) {
            if (c == '\n') {
                if (!text.empty() && text.back() != '\n') {
                    text.push_back('\n');
                }
            } else {
                if (!text.empty() && text.back() != ' ' && text.back() != '\n') {
                    text.push_back(' ');
                }
            }
        } else {
            text.push_back(c);
        }
    }

    // Trim trailing whitespace
    while (!text.empty() && std::isspace((unsigned char)text.back())) {
        text.pop_back();
    }

    return text;
}

// ════════════════════════════════════════════════════════════════════════════════
//  System Command Execution & Security Guardrails (Cross-Platform)
// ════════════════════════════════════════════════════════════════════════════════

inline bool is_dangerous_command(const std::string& cmd, std::string& reason) {
    std::string lower = cmd;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return (char)::tolower(c); });

    auto contains_word = [&](const std::string& word) -> bool {
        size_t pos = 0;
        while ((pos = lower.find(word, pos)) != std::string::npos) {
            bool start_bound = (pos == 0 || (!std::isalnum((unsigned char)lower[pos - 1]) && lower[pos - 1] != '_' && lower[pos - 1] != '-'));
            size_t end_idx = pos + word.size();
            bool end_bound = (end_idx >= lower.size() || (!std::isalnum((unsigned char)lower[end_idx]) && lower[end_idx] != '_' && lower[end_idx] != '-'));
            if (start_bound && end_bound) return true;
            pos += word.size();
        }
        return false;
    };

    // Destructive disk / formatting
    if (contains_word("format") || contains_word("diskpart") || contains_word("fdisk") || contains_word("mkfs") || contains_word("dd")) {
        reason = "Disk partitioning or formatting commands are blocked by security policy.";
        return true;
    }

    // File deletion / directory destruction
    if (contains_word("del") || contains_word("erase") || contains_word("rmdir") || contains_word("rd") ||
        contains_word("rm") || contains_word("unlink") || contains_word("shred") || contains_word("wipe") ||
        contains_word("remove-item") || contains_word("clear-content")) {
        reason = "File or directory deletion commands are blocked by security policy.";
        return true;
    }

    // System power / reboot
    if (contains_word("shutdown") || contains_word("reboot") || contains_word("poweroff") ||
        contains_word("stop-computer") || contains_word("restart-computer")) {
        reason = "System shutdown and reboot commands are blocked by security policy.";
        return true;
    }

    // Registry / account / security modification
    if (lower.find("reg add") != std::string::npos || lower.find("reg delete") != std::string::npos ||
        lower.find("net user") != std::string::npos || lower.find("net localgroup") != std::string::npos ||
        contains_word("takeown") || contains_word("icacls") || contains_word("bcdedit")) {
        reason = "System registry, user accounts, and security policy modifications are blocked.";
        return true;
    }

    return false;
}

// ════════════════════════════════════════════════════════════════════════════════
//  Workspace Directory & Dedicated File Operations (Cross-Platform)
// ════════════════════════════════════════════════════════════════════════════════

namespace fs = std::filesystem;

inline std::string get_workspace_directory() {
    try {
        return fs::current_path().string();
    } catch (...) {
        return "";
    }
}

inline bool is_path_inside_workspace(const std::string& raw_path, const std::string& workspace_root = "") {
    try {
        if (raw_path.empty()) return true;
        fs::path ws = workspace_root.empty() ? fs::current_path() : fs::path(workspace_root);
        fs::path target = fs::absolute(fs::path(raw_path)).lexically_normal();
        fs::path root = fs::canonical(ws).lexically_normal();

        if (fs::exists(target)) {
            target = fs::canonical(target);
        } else if (target.has_parent_path() && fs::exists(target.parent_path())) {
            target = fs::canonical(target.parent_path()) / target.filename();
        }

        std::string target_str = target.string();
        std::string root_str = root.string();

#if defined(_WIN32) || defined(_WIN64)
        auto to_lower = [](std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return (char)::tolower(c); });
            return s;
        };
        target_str = to_lower(target_str);
        root_str = to_lower(root_str);
#endif

        if (target_str.rfind(root_str, 0) == 0) {
            if (target_str.size() == root_str.size() || 
                target_str[root_str.size()] == '/' || 
                target_str[root_str.size()] == '\\') {
                return true;
            }
        }
        return false;
    } catch (...) {
        return false;
    }
}

inline std::string read_file(const std::string& path, int start_line = 1, int end_line = -1, size_t max_bytes = 64000) {
    if (path.empty()) return "[Error: Empty file path]";
    try {
        fs::path p(path);
        if (!fs::exists(p)) {
            return "[Error: File does not exist: " + path + "]";
        }
        if (fs::is_directory(p)) {
            return "[Error: Path is a directory, not a file: " + path + "]";
        }

        std::ifstream file(p, std::ios::in | std::ios::binary);
        if (!file.is_open()) {
            return "[Error: Could not open file for reading: " + path + "]";
        }

        std::ostringstream ss;
        std::string line;
        int current_line = 1;
        size_t total_bytes = 0;

        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }

            if (current_line >= start_line && (end_line == -1 || current_line <= end_line)) {
                std::string formatted_line = std::to_string(current_line) + ": " + line + "\n";
                if (total_bytes + formatted_line.size() > max_bytes) {
                    ss << "\n... [Content truncated at " << max_bytes << " bytes. Specify start_line/end_line to view remaining content]";
                    break;
                }
                ss << formatted_line;
                total_bytes += formatted_line.size();
            }
            current_line++;
        }

        std::string result = ss.str();
        if (result.empty()) {
            return "[File is empty: " + path + "]";
        }
        return result;
    } catch (const std::exception& e) {
        return std::string("[Error reading file: ") + e.what() + "]";
    } catch (...) {
        return "[Error: Unknown exception while reading file: " + path + "]";
    }
}

inline std::string write_file(const std::string& path, const std::string& content, bool overwrite = true) {
    if (path.empty()) return "[Error: Empty file path]";
    try {
        fs::path p(path);
        if (fs::exists(p) && !overwrite) {
            return "[Error: File already exists and overwrite is false: " + path + "]";
        }
        if (p.has_parent_path()) {
            fs::create_directories(p.parent_path());
        }

        std::ofstream file(p, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            return "[Error: Could not open file for writing: " + path + "]";
        }

        file.write(content.data(), content.size());
        file.close();

        return "[Success: Successfully wrote " + std::to_string(content.size()) + " bytes to " + path + "]";
    } catch (const std::exception& e) {
        return std::string("[Error writing file: ") + e.what() + "]";
    } catch (...) {
        return "[Error: Unknown exception while writing file: " + path + "]";
    }
}

inline std::string edit_file(const std::string& path, const std::string& target_content, const std::string& replacement_content) {
    if (path.empty()) return "[Error: Empty file path]";
    if (target_content.empty()) return "[Error: Target content to replace cannot be empty]";
    try {
        fs::path p(path);
        if (!fs::exists(p)) {
            return "[Error: File does not exist: " + path + "]";
        }

        std::ifstream in(p, std::ios::in | std::ios::binary);
        if (!in.is_open()) {
            return "[Error: Could not open file for reading: " + path + "]";
        }
        std::ostringstream sstr;
        sstr << in.rdbuf();
        in.close();

        std::string full_text = sstr.str();

        size_t first_pos = full_text.find(target_content);
        if (first_pos == std::string::npos) {
            return "[Error: Target content not found in " + path + ". Ensure exact match including whitespace/indentation.]";
        }

        size_t second_pos = full_text.find(target_content, first_pos + target_content.size());
        if (second_pos != std::string::npos) {
            return "[Error: Target content is ambiguous (found multiple matches in " + path + "). Provide more surrounding context to make it unique.]";
        }

        full_text.replace(first_pos, target_content.size(), replacement_content);

        std::ofstream out(p, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            return "[Error: Could not open file for writing replacement: " + path + "]";
        }
        out.write(full_text.data(), full_text.size());
        out.close();

        return "[Success: Successfully replaced target chunk in " + path + "]";
    } catch (const std::exception& e) {
        return std::string("[Error editing file: ") + e.what() + "]";
    } catch (...) {
        return "[Error: Unknown exception while editing file: " + path + "]";
    }
}

inline std::string execute_system_command(const std::string& command, int timeout_ms = 60000, size_t max_output_bytes = 8192) {
    if (command.empty()) return "[Error: Empty command]";

    std::string block_reason;
    if (is_dangerous_command(command, block_reason)) {
        return "[Security Error: Execution blocked] " + block_reason;
    }

#if defined(_WIN32) || defined(_WIN64)
    HANDLE h_read_pipe = NULL;
    HANDLE h_write_pipe = NULL;

    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(SECURITY_ATTRIBUTES);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = NULL;

    if (!CreatePipe(&h_read_pipe, &h_write_pipe, &sa, 0)) {
        return "[Error: Failed to create pipe for command execution]";
    }
    SetHandleInformation(h_read_pipe, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.hStdOutput = h_write_pipe;
    si.hStdError = h_write_pipe;
    si.dwFlags |= STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;

    PROCESS_INFORMATION pi;
    ZeroMemory(&pi, sizeof(pi));

    // Execute via cmd.exe /c
    std::string full_cmd = "cmd.exe /c " + command;
    int wlen = MultiByteToWideChar(CP_UTF8, 0, full_cmd.c_str(), -1, NULL, 0);
    std::vector<wchar_t> wcmd(wlen);
    MultiByteToWideChar(CP_UTF8, 0, full_cmd.c_str(), -1, wcmd.data(), wlen);

    if (!CreateProcessW(NULL, wcmd.data(), NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) {
        CloseHandle(h_read_pipe);
        CloseHandle(h_write_pipe);
        return "[Error: Failed to create process]";
    }

    // Close write end in parent process so read pipe will signal EOF
    CloseHandle(h_write_pipe);

    // Wait for process with timeout
    DWORD wait_result = WaitForSingleObject(pi.hProcess, (DWORD)timeout_ms);
    if (wait_result == WAIT_TIMEOUT) {
        TerminateProcess(pi.hProcess, 1);
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
        CloseHandle(h_read_pipe);
        return "[Error: Command execution timed out after " + std::to_string(timeout_ms / 1000) + "s]";
    }

    DWORD exit_code = 0;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    // Read output from pipe
    std::string output;
    char buffer[1024];
    DWORD bytes_read = 0;

    while (ReadFile(h_read_pipe, buffer, sizeof(buffer), &bytes_read, NULL) && bytes_read > 0) {
        if (output.size() + bytes_read <= max_output_bytes) {
            output.append(buffer, bytes_read);
        } else {
            size_t remaining = max_output_bytes - output.size();
            if (remaining > 0) {
                output.append(buffer, remaining);
            }
            output += "\n... [Output truncated at " + std::to_string(max_output_bytes) + " bytes]";
            break;
        }
    }
    CloseHandle(h_read_pipe);

    if (output.empty()) {
        output = (exit_code == 0) ? "[Command completed with no output]" : "[Command exited with code " + std::to_string(exit_code) + "]";
    }
    return output;

#else
    // POSIX / Linux Implementation
    int pipefd[2];
    if (pipe(pipefd) == -1) {
        return "[Error: Failed to create pipe for command execution]";
    }

    pid_t pid = fork();
    if (pid == -1) {
        close(pipefd[0]);
        close(pipefd[1]);
        return "[Error: Failed to fork process]";
    }

    if (pid == 0) {
        // Child process
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);

        execl("/bin/sh", "sh", "-c", command.c_str(), (char*)NULL);
        _exit(127);
    }

    // Parent process
    close(pipefd[1]);

    // Set non-blocking on read pipe
    int flags = fcntl(pipefd[0], F_GETFL, 0);
    fcntl(pipefd[0], F_SETFL, flags | O_NONBLOCK);

    std::string output;
    auto start_time = std::chrono::steady_clock::now();
    bool timed_out = false;

    while (true) {
        struct pollfd pfd;
        pfd.fd = pipefd[0];
        pfd.events = POLLIN | POLLHUP;

        int elapsed = (int)std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time).count();
        int remaining_timeout = timeout_ms - elapsed;

        if (remaining_timeout <= 0) {
            timed_out = true;
            break;
        }

        int ret = poll(&pfd, 1, std::min(remaining_timeout, 200));
        if (ret > 0) {
            char buffer[1024];
            ssize_t n = read(pipefd[0], buffer, sizeof(buffer));
            if (n > 0) {
                if (output.size() + (size_t)n <= max_output_bytes) {
                    output.append(buffer, n);
                } else {
                    size_t rem = max_output_bytes - output.size();
                    if (rem > 0) output.append(buffer, rem);
                    output += "\n... [Output truncated at " + std::to_string(max_output_bytes) + " bytes]";
                    break;
                }
            } else if (n == 0) {
                break; // EOF
            }
        }

        int status;
        pid_t w = waitpid(pid, &status, WNOHANG);
        if (w == pid) {
            // Child exited, read remaining bytes
            char buffer[1024];
            while (true) {
                ssize_t n = read(pipefd[0], buffer, sizeof(buffer));
                if (n > 0 && output.size() + (size_t)n <= max_output_bytes) {
                    output.append(buffer, n);
                } else {
                    break;
                }
            }
            break;
        }
    }

    close(pipefd[0]);

    if (timed_out) {
        kill(pid, SIGKILL);
        waitpid(pid, NULL, 0);
        return "[Error: Command execution timed out after " + std::to_string(timeout_ms / 1000) + "s]";
    }

    if (output.empty()) {
        output = "[Command completed with no output]";
    }
    return output;
#endif
}

inline std::string sanitize_and_normalize_url(const std::string& raw_url) {
    std::string url = raw_url;
    // Trim whitespace and quotes/backticks
    size_t u_first = url.find_first_not_of(" \t\n\r\"'`");
    size_t u_last = url.find_last_not_of(" \t\n\r\"'`");
    if (u_first != std::string::npos && u_last != std::string::npos) {
        url = url.substr(u_first, u_last - u_first + 1);
    } else {
        return "";
    }

    // Rewrite search engines that block bot requests to DuckDuckGo HTML endpoint
    if (url.find("google.") != std::string::npos && url.find("/search") != std::string::npos) {
        size_t q_pos = url.find("q=");
        if (q_pos != std::string::npos) {
            std::string query = url.substr(q_pos + 2);
            size_t amp_pos = query.find('&');
            if (amp_pos != std::string::npos) query = query.substr(0, amp_pos);
            url = "https://html.duckduckgo.com/html/?q=" + query;
        }
    } else if (url.find("bing.com/search") != std::string::npos) {
        size_t q_pos = url.find("q=");
        if (q_pos != std::string::npos) {
            std::string query = url.substr(q_pos + 2);
            size_t amp_pos = query.find('&');
            if (amp_pos != std::string::npos) query = query.substr(0, amp_pos);
            url = "https://html.duckduckgo.com/html/?q=" + query;
        }
    }

    // Ensure scheme
    if (url.rfind("http://", 0) != 0 && url.rfind("https://", 0) != 0) {
        url = "https://" + url;
    }
    return url;
}

inline std::string url_decode(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        if (in[i] == '%' && i + 2 < in.size()) {
            auto hex_val = [](int c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                return -1;
            };
            int v1 = hex_val(in[i + 1]);
            int v2 = hex_val(in[i + 2]);
            if (v1 >= 0 && v2 >= 0) {
                out.push_back((char)((v1 << 4) | v2));
                i += 2;
                continue;
            }
        } else if (in[i] == '+') {
            out.push_back(' ');
            continue;
        }
        out.push_back(in[i]);
    }
    return out;
}

inline std::string parse_duckduckgo_results(const std::string& html, size_t max_results = 8) {
    std::ostringstream ss;
    size_t pos = 0;
    size_t count = 0;

    while (pos < html.size() && count < max_results) {
        size_t r_pos = html.find("class=\"result__body", pos);
        if (r_pos == std::string::npos) r_pos = html.find("class=\"result__title", pos);
        if (r_pos == std::string::npos) break;

        size_t a_start = html.find("<a ", r_pos);
        if (a_start == std::string::npos) { pos = r_pos + 20; continue; }

        size_t href_pos = html.find("href=\"", a_start);
        if (href_pos == std::string::npos) { pos = a_start + 5; continue; }
        href_pos += 6;

        size_t href_end = html.find('"', href_pos);
        if (href_end == std::string::npos) { pos = href_pos + 5; continue; }

        std::string raw_href = html.substr(href_pos, href_end - href_pos);
        std::string target_url = raw_href;

        // Decode uddg= target URL
        size_t uddg_pos = raw_href.find("uddg=");
        if (uddg_pos != std::string::npos) {
            std::string encoded = raw_href.substr(uddg_pos + 5);
            size_t amp = encoded.find('&');
            if (amp != std::string::npos) encoded = encoded.substr(0, amp);
            target_url = url_decode(encoded);
        } else if (target_url.rfind("//", 0) == 0) {
            target_url = "https:" + target_url;
        }

        // Title
        size_t tag_close = html.find('>', href_end);
        std::string title;
        if (tag_close != std::string::npos) {
            size_t a_end = html.find("</a>", tag_close);
            if (a_end != std::string::npos) {
                title = strip_html_to_text(html.substr(tag_close + 1, a_end - tag_close - 1), 200);
            }
        }

        // Snippet
        size_t snip_pos = html.find("class=\"result__snippet", href_end);
        std::string snippet;
        if (snip_pos != std::string::npos && snip_pos < href_end + 800) {
            size_t s_open = html.find('>', snip_pos);
            if (s_open != std::string::npos) {
                size_t s_close = html.find("</a>", s_open);
                if (s_close == std::string::npos) s_close = html.find("</div>", s_open);
                if (s_close != std::string::npos) {
                    snippet = strip_html_to_text(html.substr(s_open + 1, s_close - s_open - 1), 300);
                }
            }
        }

        if (!target_url.empty() && target_url.rfind("http", 0) == 0 && target_url.find("duckduckgo.com") == std::string::npos) {
            count++;
            ss << count << ". " << (title.empty() ? "Web Result" : title) << "\n";
            ss << "   URL: " << target_url << "\n";
            if (!snippet.empty()) {
                ss << "   Snippet: " << snippet << "\n";
            }
            ss << "\n";
        }

        pos = (snip_pos != std::string::npos && snip_pos > href_end) ? snip_pos + 50 : href_end + 50;
    }

    std::string res = ss.str();
    if (res.empty()) {
        return strip_html_to_text(html, 3000);
    }
    return res;
}

inline std::string extract_meta_tag(const std::string& html, const std::string& property_or_name) {
    std::string key_lower = property_or_name;
    std::transform(key_lower.begin(), key_lower.end(), key_lower.begin(), [](unsigned char c) { return (char)::tolower(c); });

    size_t pos = 0;
    while (pos < html.size()) {
        size_t meta_pos = html.find("<meta", pos);
        if (meta_pos == std::string::npos) meta_pos = html.find("<META", pos);
        if (meta_pos == std::string::npos) break;

        size_t tag_end = html.find('>', meta_pos);
        if (tag_end == std::string::npos) break;

        std::string tag = html.substr(meta_pos, tag_end - meta_pos + 1);
        std::string tag_lower = tag;
        std::transform(tag_lower.begin(), tag_lower.end(), tag_lower.begin(), [](unsigned char c) { return (char)::tolower(c); });

        if (tag_lower.find(key_lower) != std::string::npos) {
            size_t c_pos = tag_lower.find("content=");
            if (c_pos != std::string::npos) {
                size_t val_start = c_pos + 8;
                if (val_start < tag.size()) {
                    char quote = tag[val_start];
                    if (quote == '"' || quote == '\'') {
                        val_start++;
                        size_t val_end = tag.find(quote, val_start);
                        if (val_end != std::string::npos) {
                            return tag.substr(val_start, val_end - val_start);
                        }
                    } else {
                        size_t val_end = tag.find_first_of(" >", val_start);
                        if (val_end != std::string::npos) {
                            return tag.substr(val_start, val_end - val_start);
                        }
                    }
                }
            }
        }
        pos = tag_end + 1;
    }
    return "";
}

inline std::string extract_html_title(const std::string& html) {
    size_t t_open = html.find("<title");
    if (t_open == std::string::npos) t_open = html.find("<TITLE");
    if (t_open != std::string::npos) {
        size_t t_close_bracket = html.find('>', t_open);
        if (t_close_bracket != std::string::npos) {
            size_t t_end = html.find("</title>", t_close_bracket);
            if (t_end == std::string::npos) t_end = html.find("</TITLE>", t_close_bracket);
            if (t_end != std::string::npos && t_end > t_close_bracket + 1) {
                std::string title = html.substr(t_close_bracket + 1, t_end - t_close_bracket - 1);
                size_t f = title.find_first_not_of(" \t\n\r");
                size_t l = title.find_last_not_of(" \t\n\r");
                if (f != std::string::npos && l != std::string::npos) {
                    title = title.substr(f, l - f + 1);
                }
                if (!title.empty()) return title;
            }
        }
    }
    std::string og_title = extract_meta_tag(html, "og:title");
    if (!og_title.empty()) return og_title;
    std::string twitter_title = extract_meta_tag(html, "twitter:title");
    if (!twitter_title.empty()) return twitter_title;
    return "";
}

inline std::string extract_youtube_video_id(const std::string& url) {
    size_t v_pos = url.find("v=");
    if (v_pos != std::string::npos) {
        std::string id = url.substr(v_pos + 2);
        size_t amp = id.find_first_of("&#? /");
        if (amp != std::string::npos) id = id.substr(0, amp);
        if (!id.empty()) return id;
    }
    size_t youtu_be = url.find("youtu.be/");
    if (youtu_be != std::string::npos) {
        std::string id = url.substr(youtu_be + 9);
        size_t amp = id.find_first_of("&#? /");
        if (amp != std::string::npos) id = id.substr(0, amp);
        if (!id.empty()) return id;
    }
    size_t embed_pos = url.find("youtube.com/embed/");
    if (embed_pos != std::string::npos) {
        std::string id = url.substr(embed_pos + 18);
        size_t amp = id.find_first_of("&#? /");
        if (amp != std::string::npos) id = id.substr(0, amp);
        if (!id.empty()) return id;
    }
    return "";
}

inline std::string generate_youtube_preview_html(const std::string& video_id, const std::string& title, const std::string& desc, const std::string& author, const std::string& url) {
    std::string safe_title = title.empty() ? "YouTube Video" : title;
    std::ostringstream ss;
    ss << "<!DOCTYPE html>\n<html>\n<head>\n"
       << "  <meta charset=\"UTF-8\">\n"
       << "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
       << "  <title>" << safe_title << "</title>\n"
       << "  <style>\n"
       << "    * { box-sizing: border-box; }\n"
       << "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 24px; background: #0f0f0f; color: #f1f1f1; line-height: 1.5; }\n"
       << "    .yt-container { max-width: 900px; margin: 0 auto; }\n"
       << "    .video-wrapper { position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.7); background: #000; margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.1); }\n"
       << "    .video-wrapper iframe, .video-wrapper #player { position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none; }\n"
       << "    .video-title { font-size: 1.35rem; font-weight: 600; margin-bottom: 10px; line-height: 1.35; color: #ffffff; }\n"
       << "    .video-meta-bar { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; font-size: 0.9rem; color: #aaa; margin-bottom: 18px; padding-bottom: 14px; border-bottom: 1px solid rgba(255,255,255,0.12); }\n"
       << "    .video-author-badge { display: inline-flex; align-items: center; gap: 8px; font-weight: 500; color: #3ea6ff; }\n"
       << "    .video-actions a { display: inline-flex; align-items: center; gap: 6px; color: #fff; background: rgba(255,255,255,0.1); padding: 6px 14px; border-radius: 18px; text-decoration: none; font-size: 0.85rem; transition: background 0.2s; }\n"
       << "    .video-actions a:hover { background: rgba(255,255,255,0.25); }\n"
       << "    .video-desc-box { font-size: 0.92rem; line-height: 1.6; color: #e1e1e1; background: rgba(255,255,255,0.06); padding: 18px 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); white-space: pre-wrap; word-break: break-word; }\n"
       << "    .video-desc-header { font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.5px; color: #aaa; margin-bottom: 10px; font-weight: 600; }\n"
       << "  </style>\n"
       << "</head>\n<body>\n"
       << "  <div class=\"yt-container\">\n"
       << "    <div class=\"video-wrapper\">\n"
       << "      <div id=\"player\"></div>\n"
       << "    </div>\n"
       << "    <div class=\"video-title\">" << safe_title << "</div>\n"
       << "    <div class=\"video-meta-bar\">\n"
       << (!author.empty() ? ("      <div class=\"video-author-badge\">&#x1F464; " + author + "</div>\n") : "      <div></div>\n")
       << "      <div class=\"video-actions\">\n"
       << "        <a href=\"" << url << "\" target=\"_blank\">Watch on YouTube &#x2197;</a>\n"
       << "      </div>\n"
       << "    </div>\n"
       << (!desc.empty() ? ("    <div class=\"video-desc-box\">\n      <div class=\"video-desc-header\">Description</div>\n" + desc + "\n    </div>\n") : "")
       << "  </div>\n"
       << "  <script src=\"https://www.youtube.com/iframe_api\"></script>\n"
       << "  <script>\n"
       << "    var player;\n"
       << "    function onYouTubeIframeAPIReady() {\n"
       << "      player = new YT.Player('player', {\n"
       << "        videoId: '" << video_id << "',\n"
       << "        playerVars: {\n"
       << "          'autoplay': 1,\n"
       << "          'playsinline': 1,\n"
       << "          'enablejsapi': 1,\n"
       << "          'rel': 0\n"
       << "        },\n"
       << "        events: {\n"
       << "          'onReady': function(e) {\n"
       << "            try {\n"
       << "              e.target.unMute();\n"
       << "              e.target.setVolume(100);\n"
       << "              e.target.playVideo();\n"
       << "            } catch(err) {}\n"
       << "          }\n"
       << "        }\n"
       << "      });\n"
       << "    }\n"
       << "    setTimeout(function() {\n"
       << "      var container = document.getElementById('player');\n"
       << "      if (container && container.tagName !== 'IFRAME') {\n"
       << "        container.innerHTML = '<iframe src=\"https://www.youtube-nocookie.com/embed/" << video_id << "?autoplay=1&enablejsapi=1&rel=0\" allow=\"accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share\" allowfullscreen style=\"position:absolute;top:0;left:0;width:100%;height:100%;border:none;\"></iframe>';\n"
       << "      }\n"
       << "    }, 2500);\n"
       << "  </script>\n"
       << "</body>\n</html>";
    return ss.str();
}

struct RetrievedDocument {
    std::string url;
    std::string title;
    std::string clean_text;
    std::string raw_html;
};

inline RetrievedDocument fetch_url_full(const std::string& raw_url, int timeout_ms = 10000, size_t max_chars = 4000) {
    RetrievedDocument doc;
    doc.url = sanitize_and_normalize_url(raw_url);
    if (doc.url.empty()) {
        doc.clean_text = "[Error: Empty URL]";
        return doc;
    }

#if defined(_WIN32) || defined(_WIN64)
    // Windows Native WinHTTP Implementation
    int wlen = MultiByteToWideChar(CP_UTF8, 0, doc.url.c_str(), -1, NULL, 0);
    std::vector<wchar_t> wurl(wlen);
    MultiByteToWideChar(CP_UTF8, 0, doc.url.c_str(), -1, wurl.data(), wlen);

    URL_COMPONENTS url_comp;
    ZeroMemory(&url_comp, sizeof(url_comp));
    url_comp.dwStructSize = sizeof(url_comp);
    url_comp.dwHostNameLength = (DWORD)-1;
    url_comp.dwUrlPathLength = (DWORD)-1;
    url_comp.dwExtraInfoLength = (DWORD)-1;

    if (!WinHttpCrackUrl(wurl.data(), (DWORD)wurl.size(), 0, &url_comp)) {
        doc.clean_text = "[Error: Invalid URL format: " + doc.url + "]";
        return doc;
    }

    std::wstring host(url_comp.lpszHostName, url_comp.dwHostNameLength);
    std::wstring path(url_comp.lpszUrlPath, url_comp.dwUrlPathLength);
    if (url_comp.dwExtraInfoLength > 0) {
        path += std::wstring(url_comp.lpszExtraInfo, url_comp.dwExtraInfoLength);
    }
    if (path.empty()) path = L"/";

    bool is_https = (url_comp.nScheme == INTERNET_SCHEME_HTTPS);
    INTERNET_PORT port = url_comp.nPort;

    HINTERNET h_session = WinHttpOpen(L"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
                                      WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                      WINHTTP_NO_PROXY_NAME,
                                      WINHTTP_NO_PROXY_BYPASS, 0);
    if (!h_session) {
        doc.clean_text = "[Error: WinHttpOpen failed]";
        return doc;
    }

    WinHttpSetTimeouts(h_session, timeout_ms, timeout_ms, timeout_ms, timeout_ms);

    HINTERNET h_connect = WinHttpConnect(h_session, host.c_str(), port, 0);
    if (!h_connect) {
        WinHttpCloseHandle(h_session);
        doc.clean_text = "[Error: WinHttpConnect failed to connect to " + doc.url + "]";
        return doc;
    }

    DWORD open_flags = is_https ? WINHTTP_FLAG_SECURE : 0;
    HINTERNET h_request = WinHttpOpenRequest(h_connect, L"GET", path.c_str(),
                                            NULL, WINHTTP_NO_REFERER,
                                            WINHTTP_DEFAULT_ACCEPT_TYPES, open_flags);
    if (!h_request) {
        WinHttpCloseHandle(h_connect);
        WinHttpCloseHandle(h_session);
        doc.clean_text = "[Error: WinHttpOpenRequest failed]";
        return doc;
    }

    // Set auto-redirect
    DWORD opt_redirect = WINHTTP_OPTION_REDIRECT_POLICY_ALWAYS;
    WinHttpSetOption(h_request, WINHTTP_OPTION_REDIRECT_POLICY, &opt_redirect, sizeof(opt_redirect));

    LPCWSTR additional_headers = L"Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\nAccept-Language: en-US,en;q=0.9,it;q=0.8\r\nSec-Fetch-Dest: document\r\nSec-Fetch-Mode: navigate\r\n";
    BOOL send_ok = WinHttpSendRequest(h_request,
                                      additional_headers, (DWORD)-1L,
                                      WINHTTP_NO_REQUEST_DATA, 0, 0, 0);
    if (!send_ok || !WinHttpReceiveResponse(h_request, NULL)) {
        WinHttpCloseHandle(h_request);
        WinHttpCloseHandle(h_connect);
        WinHttpCloseHandle(h_session);
        doc.clean_text = "[Error: Failed to receive HTTP response from " + doc.url + "]";
        return doc;
    }

    DWORD status_code = 0;
    DWORD status_size = sizeof(status_code);
    WinHttpQueryHeaders(h_request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                        WINHTTP_HEADER_NAME_BY_INDEX, &status_code, &status_size, WINHTTP_NO_HEADER_INDEX);

    std::string raw_html;
    DWORD bytes_available = 0;
    while (WinHttpQueryDataAvailable(h_request, &bytes_available) && bytes_available > 0) {
        std::vector<char> temp_buf(bytes_available);
        DWORD bytes_read = 0;
        if (WinHttpReadData(h_request, temp_buf.data(), bytes_available, &bytes_read) && bytes_read > 0) {
            raw_html.append(temp_buf.data(), bytes_read);
            if (raw_html.size() > 2000000) break;
        } else {
            break;
        }
    }

    WinHttpCloseHandle(h_request);
    WinHttpCloseHandle(h_connect);
    WinHttpCloseHandle(h_session);

    if (status_code >= 400) {
        doc.raw_html = raw_html.empty() ? ("<!DOCTYPE html><html><body><h3>HTTP Error " + std::to_string(status_code) + "</h3><p>Could not load " + doc.url + "</p></body></html>") : raw_html;
        doc.title = extract_html_title(raw_html);
        doc.clean_text = "[HTTP Error " + std::to_string(status_code) + " when fetching " + doc.url + "]";
        return doc;
    }

#else
    // Linux Implementation: Use curl command helper
    std::string curl_cmd = "curl -sL --max-time " + std::to_string(timeout_ms / 1000) +
                           " -A \"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36\"" +
                           " -H \"Accept-Language: en-US,en;q=0.9,it;q=0.8\" \"" + doc.url + "\"";
    std::string raw_html = execute_system_command(curl_cmd, timeout_ms, 2000000);
    if (raw_html.rfind("[Error", 0) == 0) {
        doc.clean_text = raw_html;
        return doc;
    }
#endif

    // Metadata extraction (Title, OpenGraph, Description, Author)
    std::string yt_id = extract_youtube_video_id(doc.url);
    std::string meta_desc = extract_meta_tag(raw_html, "og:description");
    if (meta_desc.empty()) meta_desc = extract_meta_tag(raw_html, "description");
    if (meta_desc.empty()) meta_desc = extract_meta_tag(raw_html, "twitter:description");

    std::string meta_author = extract_meta_tag(raw_html, "author");
    if (meta_author.empty()) meta_author = extract_meta_tag(raw_html, "og:site_name");

    doc.title = extract_html_title(raw_html);
    if (doc.title.empty() && !yt_id.empty()) {
        doc.title = "YouTube Video (" + yt_id + ")";
    }

    if (!yt_id.empty()) {
        // YouTube Video: generate rich embedded player HTML and structured summary for the model
        doc.raw_html = generate_youtube_preview_html(yt_id, doc.title, meta_desc, meta_author, doc.url);

        std::string text_summary = "[Retrieved YouTube Video]\n";
        if (!doc.title.empty()) text_summary += "Title: " + doc.title + "\n";
        if (!meta_author.empty()) text_summary += "Channel/Author: " + meta_author + "\n";
        if (!meta_desc.empty()) text_summary += "Description: " + meta_desc + "\n";
        text_summary += "Video URL: " + doc.url + "\n";
        text_summary += "[Note: Video player and full page preview are available in the client preview.]";
        doc.clean_text = text_summary;
        return doc;
    }

    if (doc.url.find("duckduckgo.com") != std::string::npos) {
        doc.raw_html = raw_html;
        doc.title = extract_html_title(raw_html);
        if (doc.title.empty()) doc.title = "DuckDuckGo Search Results";
        doc.clean_text = "[Search Results]\n" + parse_duckduckgo_results(raw_html, 8);
        return doc;
    }

    // Standard Web Page
    doc.raw_html = raw_html;
    std::string body_text = strip_html_to_text(raw_html, max_chars);

    std::string combined_text;
    if (!doc.title.empty()) {
        combined_text += "Title: " + doc.title + "\n";
    }
    if (!meta_desc.empty() && (body_text.empty() || body_text.find(meta_desc.substr(0, std::min((size_t)30, meta_desc.size()))) == std::string::npos)) {
        combined_text += "Summary: " + meta_desc + "\n\n";
    }
    if (!body_text.empty()) {
        combined_text += body_text;
    }

    if (combined_text.empty()) {
        if (!doc.title.empty()) {
            doc.clean_text = "[Page Loaded: " + doc.title + "]\n[Note: Dynamic JavaScript content. Complete raw HTML is available in the client preview.]";
        } else {
            doc.clean_text = "[Web page loaded: " + doc.url + "]\n[Note: Dynamic JavaScript content. Complete raw HTML is available in the client preview.]";
        }
    } else {
        doc.clean_text = combined_text;
    }
    return doc;
}

inline std::string fetch_url_content(const std::string& raw_url, int timeout_ms = 10000, size_t max_chars = 4000) {
    RetrievedDocument doc = fetch_url_full(raw_url, timeout_ms, max_chars);
    return doc.clean_text;
}

// ════════════════════════════════════════════════════════════════════════════════
//  Built-in Tool Schemas & Definitions
// ════════════════════════════════════════════════════════════════════════════════

inline std::string get_builtin_tools_json() {
    return R"JSON([
  {
    "type": "function",
    "function": {
      "name": "fetch_url",
      "description": "Fetch and extract readable text content from a public web URL (HTTP/HTTPS), such as an article, documentation, Wikipedia page, GitHub repository, or news site.",
      "parameters": {
        "type": "object",
        "properties": {
          "url": {
            "type": "string",
            "description": "The complete HTTP or HTTPS URL to fetch (e.g. https://example.com or https://en.wikipedia.org/wiki/Special:Search?search=Topic)"
          }
        },
        "required": ["url"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "read_file",
      "description": "Read the text contents of a file on the local file system. Supports line numbering and viewing line ranges.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "The relative or absolute file path to read (e.g. 'src/main.cpp' or 'config.json')."
          },
          "start_line": {
            "type": "integer",
            "description": "Optional 1-indexed starting line number (default: 1)."
          },
          "end_line": {
            "type": "integer",
            "description": "Optional 1-indexed ending line number (default: -1 for entire file)."
          }
        },
        "required": ["path"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "write_file",
      "description": "Create a new file or completely overwrite an existing file with the provided text content.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "The relative or absolute file path to write."
          },
          "content": {
            "type": "string",
            "description": "The complete text content to write to the file."
          },
          "overwrite": {
            "type": "boolean",
            "description": "Whether to overwrite if file already exists (default: true)."
          }
        },
        "required": ["path", "content"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "edit_file",
      "description": "Perform a precise search-and-replace on a unique block of text within an existing file.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "The relative or absolute file path to modify."
          },
          "target_content": {
            "type": "string",
            "description": "The exact unique substring within the file to replace (must match exactly including whitespace)."
          },
          "replacement_content": {
            "type": "string",
            "description": "The new replacement text to insert in place of target_content."
          }
        },
        "required": ["path", "target_content", "replacement_content"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "execute_command",
      "description": "Execute a safe terminal or shell command on the local host system (e.g. dir, python script.py, git status, ninja -C build, cargo test) and return its output.",
      "parameters": {
        "type": "object",
        "properties": {
          "command": {
            "type": "string",
            "description": "The exact shell command line to execute."
          }
        },
        "required": ["command"]
      }
    }
  }
])JSON";
}

// ════════════════════════════════════════════════════════════════════════════════
//  Tool Call Extraction & Parsing (Omnivorous & Resilient)
// ════════════════════════════════════════════════════════════════════════════════

struct ToolCall {
    std::string id;
    std::string type = "function";
    std::string name;
    std::string arguments; // serialized JSON string
};

inline bool try_parse_tool_json(const std::string& str, ToolCall& tc, int idx) {
    try {
        nlohmann::json j = nlohmann::json::parse(str);
        if (!j.is_object()) return false;

        std::string fn_name;
        if (j.contains("name") && j["name"].is_string()) {
            fn_name = j["name"].get<std::string>();
        } else if (j.contains("function") && j["function"].is_string()) {
            fn_name = j["function"].get<std::string>();
        } else if (j.contains("function") && j["function"].is_object() && j["function"].contains("name")) {
            fn_name = j["function"]["name"].get<std::string>();
        }

        if (fn_name.empty()) return false;

        tc.name = fn_name;
        tc.id = "call_" + std::to_string(idx) + "_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count() % 1000000);
        tc.type = "function";

        if (j.contains("arguments")) {
            if (j["arguments"].is_string()) tc.arguments = j["arguments"].get<std::string>();
            else tc.arguments = j["arguments"].dump();
        } else if (j.contains("parameters")) {
            if (j["parameters"].is_string()) tc.arguments = j["parameters"].get<std::string>();
            else tc.arguments = j["parameters"].dump();
        } else if (j.contains("function") && j["function"].is_object() && j["function"].contains("arguments")) {
            if (j["function"]["arguments"].is_string()) tc.arguments = j["function"]["arguments"].get<std::string>();
            else tc.arguments = j["function"]["arguments"].dump();
        } else {
            // Collect all other keys into arguments (flat schema support like {"name": "fetch_url", "url": "..."})
            nlohmann::json args = nlohmann::json::object();
            for (auto it = j.begin(); it != j.end(); ++it) {
                if (it.key() != "name" && it.key() != "function" && it.key() != "type") {
                    args[it.key()] = it.value();
                }
            }
            tc.arguments = args.dump();
        }
        return true;
    } catch (...) {
        return false;
    }
}

inline void extract_tool_calls(
    const std::string& response_text,
    std::string& out_clean_content,
    std::vector<ToolCall>& out_tool_calls
) {
    out_tool_calls.clear();
    out_clean_content = response_text;

    int call_idx = 0;

    // 1. Check <tool_call> ... </tool_call>
    std::string start_tag = "<tool_call>";
    std::string end_tag = "</tool_call>";
    size_t tag_pos = response_text.find(start_tag);

    if (tag_pos != std::string::npos) {
        std::string clean_acc;
        size_t last_end = 0;
        size_t search_pos = 0;

        while (true) {
            size_t start = response_text.find(start_tag, search_pos);
            if (start == std::string::npos) break;

            size_t end = response_text.find(end_tag, start + start_tag.size());
            if (end == std::string::npos) end = response_text.size();

            clean_acc.append(response_text, last_end, start - last_end);
            size_t c_start = start + start_tag.size();
            size_t c_len = (end > c_start) ? (end - c_start) : 0;
            std::string raw_block = response_text.substr(c_start, c_len);

            ToolCall tc;
            if (try_parse_tool_json(raw_block, tc, ++call_idx)) {
                out_tool_calls.push_back(tc);
            }

            last_end = (end < response_text.size()) ? (end + end_tag.size()) : response_text.size();
            search_pos = last_end;
        }

        clean_acc.append(response_text, last_end, response_text.size() - last_end);
        // Clean trailing stray braces or backticks
        while (!clean_acc.empty() && (clean_acc.back() == '}' || clean_acc.back() == '`' || clean_acc.back() == ' ' || clean_acc.back() == '\n' || clean_acc.back() == '\r')) {
            clean_acc.pop_back();
        }
        out_clean_content = clean_acc;
        if (!out_tool_calls.empty()) return;
    }

    // 2. Check DeepSeek DSML tokens: <｜tool call begin｜> ... <｜tool call end｜>
    std::string dsml_start = "<｜tool call begin｜>";
    std::string dsml_end = "<｜tool call end｜>";
    size_t dsml_pos = response_text.find(dsml_start);
    if (dsml_pos != std::string::npos) {
        size_t dsml_pos_end = response_text.find(dsml_end, dsml_pos);
        if (dsml_pos_end != std::string::npos) {
            std::string dsml_block = response_text.substr(dsml_pos + dsml_start.size(), dsml_pos_end - (dsml_pos + dsml_start.size()));
            std::string sep = "<｜tool sep｜>";
            size_t sep1 = dsml_block.find(sep);
            if (sep1 != std::string::npos) {
                size_t sep2 = dsml_block.find(sep, sep1 + sep.size());
                if (sep2 != std::string::npos) {
                    std::string fn_name = dsml_block.substr(sep1 + sep.size(), sep2 - (sep1 + sep.size()));
                    std::string fn_args = dsml_block.substr(sep2 + sep.size());
                    ToolCall tc;
                    tc.id = "call_1_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count() % 1000000);
                    tc.type = "function";
                    tc.name = fn_name;
                    tc.arguments = fn_args;
                    out_tool_calls.push_back(tc);
                    out_clean_content = response_text.substr(0, dsml_pos);
                    return;
                }
            }
        }
    }

    // 3. Scan for any embedded JSON object { ... } containing "name" or "function"
    std::string text = response_text;
    std::string remaining;
    size_t scan_idx = 0;
    size_t text_len = text.size();

    while (scan_idx < text_len) {
        size_t open_brace = text.find('{', scan_idx);
        if (open_brace == std::string::npos) {
            remaining.append(text, scan_idx, text_len - scan_idx);
            break;
        }

        // Check if there is a matching closing brace
        int depth = 0;
        bool in_str = false;
        bool escape = false;
        size_t close_brace = std::string::npos;

        for (size_t i = open_brace; i < text_len; ++i) {
            char c = text[i];
            if (escape) {
                escape = false;
                continue;
            }
            if (c == '\\' && in_str) {
                escape = true;
                continue;
            }
            if (c == '"') {
                in_str = !in_str;
                continue;
            }
            if (!in_str) {
                if (c == '{') depth++;
                else if (c == '}') {
                    depth--;
                    if (depth == 0) {
                        close_brace = i;
                        break;
                    }
                }
            }
        }

        if (close_brace != std::string::npos) {
            std::string candidate = text.substr(open_brace, close_brace - open_brace + 1);
            ToolCall tc;
            if (try_parse_tool_json(candidate, tc, ++call_idx)) {
                out_tool_calls.push_back(tc);
                remaining.append(text, scan_idx, open_brace - scan_idx);
                scan_idx = close_brace + 1;
                // Skip trailing noise like extra closing braces or markdown fences
                while (scan_idx < text_len && (text[scan_idx] == '}' || text[scan_idx] == '`' || text[scan_idx] == ' ' || text[scan_idx] == '\n' || text[scan_idx] == '\r')) {
                    scan_idx++;
                }
                continue;
            }
        }

        // Not a valid tool json, advance by 1
        remaining.push_back(text[open_brace]);
        scan_idx = open_brace + 1;
    }

    if (!out_tool_calls.empty()) {
        // Clean leading and trailing whitespace
        size_t f = remaining.find_first_not_of(" \t\n\r");
        size_t l = remaining.find_last_not_of(" \t\n\r");
        if (f != std::string::npos && l != std::string::npos) {
            out_clean_content = remaining.substr(f, l - f + 1);
        } else {
            out_clean_content.clear();
        }
    }
}

} // namespace tooling
} // namespace moecher
