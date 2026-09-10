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
#include <mutex>
#include <unordered_set>
#include <unordered_map>
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

using json = nlohmann::json;

// ════════════════════════════════════════════════════════════════════════════════
//  Client Browser Context & Persistent Domain Cookie Jar
// ════════════════════════════════════════════════════════════════════════════════

inline std::mutex g_cookie_mutex;
inline std::string g_client_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0";
inline std::string g_client_accept_language = "en-US,en;q=0.9,it;q=0.8";
inline std::string g_client_sec_ch_ua = "\"Not(A:Brand\";v=\"99\", \"Microsoft Edge\";v=\"133\", \"Chromium\";v=\"133\"";
inline std::string g_client_sec_ch_ua_mobile = "?0";
inline std::string g_client_sec_ch_ua_platform = "\"Windows\"";

// Map: domain -> map of cookie_name -> cookie_value
inline std::unordered_map<std::string, std::unordered_map<std::string, std::string>> g_cookie_jar;
inline bool g_cookie_jar_loaded = false;

inline std::string normalize_domain(const std::string& host_or_domain) {
    std::string d = host_or_domain;
    for (auto& c : d) c = (char)tolower((unsigned char)c);
    while (!d.empty() && d.front() == '.') d.erase(0, 1);
    size_t colon = d.find(':');
    if (colon != std::string::npos) d = d.substr(0, colon);
    return d;
}

inline void save_cookie_jar_to_disk() {
    std::lock_guard<std::mutex> lock(g_cookie_mutex);
    try {
        json j = json::object();
        for (const auto& kv : g_cookie_jar) {
            j[kv.first] = kv.second;
        }
        std::string serialized = j.dump(2);
        std::ofstream f(".moecher_cookies.json");
        if (f.is_open()) {
            f << serialized;
        }
        std::ofstream f2("cookie_jar.json");
        if (f2.is_open()) {
            f2 << serialized;
        }
    } catch (...) {}
}

// ════════════════════════════════════════════════════════════════════════════════
//  Universal Web Search API Configuration (Tavily, Brave, SearXNG, Serper, Google)
// ════════════════════════════════════════════════════════════════════════════════

inline std::mutex g_config_mutex;
inline std::string g_search_provider = "tavily"; // "tavily", "brave", "searxng", "serper", "google"
inline std::string g_tavily_api_key = "";
inline std::string g_brave_api_key = "";
inline std::string g_serper_api_key = "";
inline std::string g_searxng_url = "https://searx.be";
inline std::string g_google_search_api_key = "";
inline std::string g_google_search_cx = "";
inline bool g_config_loaded = false;

inline void load_config_from_disk() {
    std::lock_guard<std::mutex> lock(g_config_mutex);
    if (g_config_loaded) return;
    g_config_loaded = true;

    // 1. Check environment variables
    const char* env_provider = std::getenv("SEARCH_PROVIDER");
    if (env_provider && strlen(env_provider) > 0) g_search_provider = env_provider;

    const char* env_tavily = std::getenv("TAVILY_API_KEY");
    if (env_tavily && strlen(env_tavily) > 0) g_tavily_api_key = env_tavily;

    const char* env_brave = std::getenv("BRAVE_API_KEY");
    if (env_brave && strlen(env_brave) > 0) g_brave_api_key = env_brave;

    const char* env_serper = std::getenv("SERPER_API_KEY");
    if (env_serper && strlen(env_serper) > 0) g_serper_api_key = env_serper;

    const char* env_searx = std::getenv("SEARXNG_URL");
    if (env_searx && strlen(env_searx) > 0) g_searxng_url = env_searx;

    const char* env_key = std::getenv("GOOGLE_SEARCH_API_KEY");
    if (!env_key) env_key = std::getenv("GOOGLE_API_KEY");
    if (env_key && strlen(env_key) > 0) g_google_search_api_key = env_key;

    const char* env_cx = std::getenv("GOOGLE_SEARCH_CX");
    if (!env_cx) env_cx = std::getenv("GOOGLE_CX");
    if (env_cx && strlen(env_cx) > 0) g_google_search_cx = env_cx;

    // 2. Check config file .moecher_config.json
    try {
        std::ifstream f(".moecher_config.json");
        if (f.is_open()) {
            json j = json::parse(f);
            if (j.contains("search_provider") && j["search_provider"].is_string()) {
                g_search_provider = j["search_provider"].get<std::string>();
            }
            if (j.contains("tavily_api_key") && j["tavily_api_key"].is_string() && g_tavily_api_key.empty()) {
                g_tavily_api_key = j["tavily_api_key"].get<std::string>();
            }
            if (j.contains("brave_api_key") && j["brave_api_key"].is_string() && g_brave_api_key.empty()) {
                g_brave_api_key = j["brave_api_key"].get<std::string>();
            }
            if (j.contains("serper_api_key") && j["serper_api_key"].is_string() && g_serper_api_key.empty()) {
                g_serper_api_key = j["serper_api_key"].get<std::string>();
            }
            if (j.contains("searxng_url") && j["searxng_url"].is_string()) {
                g_searxng_url = j["searxng_url"].get<std::string>();
            }
            if (j.contains("google_search_api_key") && j["google_search_api_key"].is_string() && g_google_search_api_key.empty()) {
                g_google_search_api_key = j["google_search_api_key"].get<std::string>();
            }
            if (j.contains("google_search_cx") && j["google_search_cx"].is_string() && g_google_search_cx.empty()) {
                g_google_search_cx = j["google_search_cx"].get<std::string>();
            }
        }
    } catch (...) {}
}

inline void save_config_to_disk() {
    std::lock_guard<std::mutex> lock(g_config_mutex);
    try {
        json j = json::object();
        std::ifstream rf(".moecher_config.json");
        if (rf.is_open()) {
            try { j = json::parse(rf); } catch (...) {}
            rf.close();
        }
        j["search_provider"] = g_search_provider;
        j["tavily_api_key"] = g_tavily_api_key;
        j["brave_api_key"] = g_brave_api_key;
        j["serper_api_key"] = g_serper_api_key;
        j["searxng_url"] = g_searxng_url;
        j["google_search_api_key"] = g_google_search_api_key;
        j["google_search_cx"] = g_google_search_cx;
        std::ofstream wf(".moecher_config.json");
        if (wf.is_open()) {
            wf << j.dump(2);
        }
    } catch (...) {}
}

inline void set_search_settings(
    const std::string& provider,
    const std::string& tavily_key,
    const std::string& brave_key,
    const std::string& serper_key,
    const std::string& searxng_url,
    const std::string& google_key,
    const std::string& google_cx
) {
    {
        std::lock_guard<std::mutex> lock(g_config_mutex);
        if (!provider.empty()) g_search_provider = provider;
        if (!tavily_key.empty()) g_tavily_api_key = tavily_key;
        if (!brave_key.empty()) g_brave_api_key = brave_key;
        if (!serper_key.empty()) g_serper_api_key = serper_key;
        if (!searxng_url.empty()) g_searxng_url = searxng_url;
        if (!google_key.empty()) g_google_search_api_key = google_key;
        if (!google_cx.empty()) g_google_search_cx = google_cx;
    }
    save_config_to_disk();
}

inline json get_search_settings() {
    load_config_from_disk();
    std::lock_guard<std::mutex> lock(g_config_mutex);
    json j;
    j["provider"] = g_search_provider;
    j["tavily_api_key"] = g_tavily_api_key;
    j["brave_api_key"] = g_brave_api_key;
    j["serper_api_key"] = g_serper_api_key;
    j["searxng_url"] = g_searxng_url.empty() ? "https://searx.be" : g_searxng_url;
    j["google_search_api_key"] = g_google_search_api_key;
    j["google_search_cx"] = g_google_search_cx;
    
    // Status indicator
    bool is_ready = false;
    if (g_search_provider == "tavily") is_ready = !g_tavily_api_key.empty();
    else if (g_search_provider == "brave") is_ready = !g_brave_api_key.empty();
    else if (g_search_provider == "serper") is_ready = !g_serper_api_key.empty();
    else if (g_search_provider == "searxng") is_ready = true; // No key needed!
    else if (g_search_provider == "google") is_ready = !g_google_search_api_key.empty() && !g_google_search_cx.empty();
    j["configured"] = is_ready;
    return j;
}

inline void set_google_search_credentials(const std::string& key, const std::string& cx) {
    {
        std::lock_guard<std::mutex> lock(g_config_mutex);
        if (!key.empty()) g_google_search_api_key = key;
        if (!cx.empty()) g_google_search_cx = cx;
    }
    save_config_to_disk();
}

inline std::pair<std::string, std::string> get_google_search_credentials() {
    load_config_from_disk();
    std::lock_guard<std::mutex> lock(g_config_mutex);
    return { g_google_search_api_key, g_google_search_cx };
}

inline void ensure_cookie_jar_loaded() {
    std::lock_guard<std::mutex> lock(g_cookie_mutex);
    if (g_cookie_jar_loaded) return;
    g_cookie_jar_loaded = true;

    // Default consent cookies for Google Search / YouTube to avoid bot/consent walls
    static const std::vector<std::string> google_domains = {
        "google.com", "google.it", "google.be", "google.co.uk", "google.de", "google.fr", "google.es", "youtube.com"
    };
    for (const auto& gd : google_domains) {
        g_cookie_jar[gd]["SOCS"] = "CAESHAgBEhJnd3NfMjAyNDA4MjAtMF9SQzIaAmVuIAEaBgiA_L20Bg";
        g_cookie_jar[gd]["CONSENT"] = "PENDING+999";
    }

    auto load_from_file = [](const std::string& filename) {
        try {
            std::ifstream f(filename);
            if (f.is_open()) {
                json j = json::parse(f);
                if (j.is_object()) {
                    for (auto it = j.begin(); it != j.end(); ++it) {
                        if (it.value().is_object()) {
                            for (auto cit = it.value().begin(); cit != it.value().end(); ++cit) {
                                if (cit.value().is_string()) {
                                    g_cookie_jar[it.key()][cit.key()] = cit.value().get<std::string>();
                                }
                            }
                        }
                    }
                }
            }
        } catch (...) {}
    };

    load_from_file(".moecher_cookies.json");
    load_from_file("cookie_jar.json");
}

inline void store_cookie(const std::string& domain, const std::string& name, const std::string& value) {
    if (domain.empty() || name.empty()) return;
    ensure_cookie_jar_loaded();
    std::string nd = normalize_domain(domain);
    {
        std::lock_guard<std::mutex> lock(g_cookie_mutex);
        g_cookie_jar[nd][name] = value;
    }
    save_cookie_jar_to_disk();
}

inline void parse_and_store_set_cookie_header(const std::string& target_url, const std::string& set_cookie_str) {
    if (set_cookie_str.empty()) return;
    ensure_cookie_jar_loaded();

    std::string host = "";
    if (!target_url.empty()) {
        size_t scheme_pos = target_url.find("://");
        if (scheme_pos != std::string::npos) {
            size_t host_start = scheme_pos + 3;
            size_t slash_pos = target_url.find('/', host_start);
            host = (slash_pos != std::string::npos) ? target_url.substr(host_start, slash_pos - host_start) : target_url.substr(host_start);
        } else {
            size_t slash_pos = target_url.find('/');
            host = (slash_pos != std::string::npos) ? target_url.substr(0, slash_pos) : target_url;
        }
    }
    host = normalize_domain(host);

    std::istringstream stream(set_cookie_str);
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        bool is_set_cookie = false;
        if (line.rfind("Set-Cookie:", 0) == 0 || line.rfind("set-cookie:", 0) == 0) {
            line = line.substr(11);
            is_set_cookie = true;
        } else if (line.rfind("Cookie:", 0) == 0 || line.rfind("cookie:", 0) == 0) {
            line = line.substr(7);
        }

        while (!line.empty() && line.front() == ' ') line.erase(0, 1);
        if (line.empty()) continue;

        if (is_set_cookie || line.find("domain=") != std::string::npos || line.find("Domain=") != std::string::npos ||
            line.find("path=") != std::string::npos || line.find("Path=") != std::string::npos ||
            line.find("expires=") != std::string::npos || line.find("Expires=") != std::string::npos) {
            // Set-Cookie single header format with attributes
            size_t first_semi = line.find(';');
            std::string name_value = (first_semi != std::string::npos) ? line.substr(0, first_semi) : line;
            size_t eq = name_value.find('=');
            if (eq == std::string::npos) continue;

            std::string name = name_value.substr(0, eq);
            std::string value = name_value.substr(eq + 1);
            while (!name.empty() && (name.front() == ' ' || name.front() == '\t')) name.erase(0, 1);
            while (!name.empty() && (name.back() == ' ' || name.back() == '\t')) name.pop_back();

            std::string domain = host;
            if (first_semi != std::string::npos) {
                std::string attrs = line.substr(first_semi + 1);
                std::string lower_attrs = attrs;
                for (auto& c : lower_attrs) c = (char)tolower((unsigned char)c);
                size_t d_pos = lower_attrs.find("domain=");
                if (d_pos != std::string::npos) {
                    size_t d_start = d_pos + 7;
                    size_t d_end = attrs.find(';', d_start);
                    domain = (d_end != std::string::npos) ? attrs.substr(d_start, d_end - d_start) : attrs.substr(d_start);
                    while (!domain.empty() && (domain.front() == ' ' || domain.front() == '.' || domain.front() == '\t')) domain.erase(0, 1);
                    while (!domain.empty() && (domain.back() == ' ' || domain.back() == '\t')) domain.pop_back();
                }
            }
            if (!domain.empty() && !name.empty()) {
                store_cookie(domain, name, value);
            }
        } else {
            // Standard Cookie header format: key1=val1; key2=val2; ...
            std::istringstream semi_stream(line);
            std::string pair_token;
            while (std::getline(semi_stream, pair_token, ';')) {
                while (!pair_token.empty() && (pair_token.front() == ' ' || pair_token.front() == '\t')) pair_token.erase(0, 1);
                while (!pair_token.empty() && (pair_token.back() == ' ' || pair_token.back() == '\t')) pair_token.pop_back();
                if (pair_token.empty()) continue;

                size_t eq = pair_token.find('=');
                if (eq == std::string::npos) continue;

                std::string k = pair_token.substr(0, eq);
                std::string v = pair_token.substr(eq + 1);
                while (!k.empty() && (k.front() == ' ' || k.front() == '\t')) k.erase(0, 1);
                while (!k.empty() && (k.back() == ' ' || k.back() == '\t')) k.pop_back();

                std::string lk = k;
                for (auto& c : lk) c = (char)tolower((unsigned char)c);
                if (lk == "path" || lk == "domain" || lk == "expires" || lk == "max-age" || lk == "samesite" || lk == "secure" || lk == "httponly" || lk == "priority") {
                    continue;
                }
                if (!host.empty() && !k.empty()) {
                    store_cookie(host, k, v);
                }
            }
        }
    }
}

inline std::string get_cookies_for_url(const std::string& url) {
    ensure_cookie_jar_loaded();
    std::string host = "";
    size_t scheme_pos = url.find("://");
    if (scheme_pos != std::string::npos) {
        size_t host_start = scheme_pos + 3;
        size_t slash_pos = url.find('/', host_start);
        host = (slash_pos != std::string::npos) ? url.substr(host_start, slash_pos - host_start) : url.substr(host_start);
    } else {
        size_t slash_pos = url.find('/');
        host = (slash_pos != std::string::npos) ? url.substr(0, slash_pos) : url;
    }
    host = normalize_domain(host);

    std::lock_guard<std::mutex> lock(g_cookie_mutex);
    std::unordered_map<std::string, std::string> matched_cookies;

    for (const auto& kv : g_cookie_jar) {
        const std::string& domain = kv.first;
        if (domain.empty()) continue;
        if (host == domain || (host.size() > domain.size() && host.compare(host.size() - domain.size(), domain.size(), domain) == 0 && host[host.size() - domain.size() - 1] == '.')) {
            for (const auto& ckv : kv.second) {
                matched_cookies[ckv.first] = ckv.second;
            }
        }
    }

    std::string cookie_header;
    for (const auto& kv : matched_cookies) {
        if (!cookie_header.empty()) cookie_header += "; ";
        cookie_header += kv.first + "=" + kv.second;
    }

    if ((host.find("google.") != std::string::npos || host.find("youtube.com") != std::string::npos) && matched_cookies.find("SOCS") == matched_cookies.end()) {
        if (!cookie_header.empty()) cookie_header += "; ";
        cookie_header += "SOCS=CAESHAgBEhJnd3NfMjAyNDA4MjAtMF9SQzIaAmVuIAEaBgiA_L20Bg; CONSENT=PENDING+999; 1P_JAR=2024-09-08-15";
    }

    return cookie_header;
}

inline void set_client_browser_context(const std::string& ua, const std::string& lang = "", const std::string& sec_ch_ua = "", const std::string& sec_ch_ua_mobile = "", const std::string& sec_ch_ua_platform = "", const std::string& raw_cookies = "") {
    if (!ua.empty()) g_client_user_agent = ua;
    if (!lang.empty()) g_client_accept_language = lang;
    if (!sec_ch_ua.empty()) g_client_sec_ch_ua = sec_ch_ua;
    if (!sec_ch_ua_mobile.empty()) g_client_sec_ch_ua_mobile = sec_ch_ua_mobile;
    if (!sec_ch_ua_platform.empty()) g_client_sec_ch_ua_platform = sec_ch_ua_platform;
    if (!raw_cookies.empty()) {
        parse_and_store_set_cookie_header("", raw_cookies);
    }
}

inline std::string get_client_user_agent() {
    return g_client_user_agent;
}

inline std::string get_client_accept_language() {
    return g_client_accept_language;
}


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

inline std::string base64_decode(const std::string& in) {
    static const std::string b64_table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    std::vector<int> T(256, -1);
    for (int i = 0; i < 64; i++) T[(unsigned char)b64_table[i]] = i;

    int val = 0, valb = -8;
    for (unsigned char c : in) {
        if (c == '=' || T[c] == -1) break;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(char((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

inline std::string url_encode(const std::string& in) {
    std::string out;
    out.reserve(in.size() * 3);
    for (char c : in) {
        if (isalnum((unsigned char)c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out += c;
        } else {
            char buf[8];
            snprintf(buf, sizeof(buf), "%%%02X", (unsigned char)c);
            out += buf;
        }
    }
    return out;
}

inline std::string resolve_redirect_url(const std::string& current_url, const std::string& location) {
    if (location.empty()) return current_url;
    if (location.rfind("http://", 0) == 0 || location.rfind("https://", 0) == 0) {
        return location;
    }
    if (location.rfind("//", 0) == 0) {
        std::string scheme = "https:";
        if (current_url.rfind("http://", 0) == 0) scheme = "http:";
        return scheme + location;
    }
    std::string origin = "";
    std::string dir = "";
    size_t proto = current_url.find("://");
    if (proto != std::string::npos) {
        size_t slash = current_url.find('/', proto + 3);
        if (slash != std::string::npos) {
            origin = current_url.substr(0, slash);
        } else {
            origin = current_url;
        }
    } else {
        origin = "https://www.linkedin.com";
    }
    if (location.front() == '/') {
        return origin + location;
    }
    size_t q_pos = current_url.find('?');
    std::string clean = (q_pos != std::string::npos) ? current_url.substr(0, q_pos) : current_url;
    size_t last_slash = clean.find_last_of('/');
    if (last_slash != std::string::npos && last_slash >= 8) {
        dir = clean.substr(0, last_slash + 1);
    } else {
        dir = origin + "/";
    }
    return dir + location;
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

    // Support search prefixes: "youtube: ...", "yt: ...", "search: ...", "google: ..."
    if (url.rfind("youtube:", 0) == 0 || url.rfind("yt:", 0) == 0) {
        size_t colon = url.find(':');
        std::string query = url.substr(colon + 1);
        while (!query.empty() && (query.front() == ' ' || query.front() == '\t')) query.erase(0, 1);
        std::string enc_query;
        for (char c : query) {
            if (c == ' ') enc_query += '+';
            else if (isalnum((unsigned char)c) || c == '-' || c == '_' || c == '.' || c == '~') enc_query += c;
            else {
                char buf[8];
                snprintf(buf, sizeof(buf), "%%%02X", (unsigned char)c);
                enc_query += buf;
            }
        }
        return "https://www.youtube.com/results?search_query=" + enc_query;
    }

    if (url.rfind("search:", 0) == 0 || url.rfind("google:", 0) == 0) {
        size_t colon = url.find(':');
        std::string query = url.substr(colon + 1);
        while (!query.empty() && (query.front() == ' ' || query.front() == '\t')) query.erase(0, 1);
        std::string lower_query = query;
        std::transform(lower_query.begin(), lower_query.end(), lower_query.begin(), [](unsigned char c){ return (char)::tolower(c); });
        std::string enc_query;
        for (char c : query) {
            if (c == ' ') enc_query += '+';
            else if (isalnum((unsigned char)c) || c == '-' || c == '_' || c == '.' || c == '~') enc_query += c;
            else {
                char buf[8];
                snprintf(buf, sizeof(buf), "%%%02X", (unsigned char)c);
                enc_query += buf;
            }
        }
        if (lower_query.find("youtube") != std::string::npos || lower_query.find("video") != std::string::npos ||
            lower_query.find("song") != std::string::npos || lower_query.find("music") != std::string::npos ||
            lower_query.find("audio") != std::string::npos || lower_query.find("canzone") != std::string::npos ||
            lower_query.find("brano") != std::string::npos || lower_query.find("de andre") != std::string::npos ||
            lower_query.find("play") != std::string::npos || lower_query.find("listen") != std::string::npos) {
            return "https://www.youtube.com/results?search_query=" + enc_query;
        }
        return "https://www.google.com/search?q=" + enc_query + "&hl=en&gl=us&gbv=1";
    }

    // If input is purely search keywords without scheme or domain dot
    if (url.find("://") == std::string::npos && url.find('.') == std::string::npos && url.find('/') == std::string::npos) {
        std::string lower_url = url;
        std::transform(lower_url.begin(), lower_url.end(), lower_url.begin(), [](unsigned char c){ return (char)::tolower(c); });
        std::string enc_query;
        for (char c : url) {
            if (c == ' ') enc_query += '+';
            else if (isalnum((unsigned char)c) || c == '-' || c == '_' || c == '.' || c == '~') enc_query += c;
            else {
                char buf[8];
                snprintf(buf, sizeof(buf), "%%%02X", (unsigned char)c);
                enc_query += buf;
            }
        }
        if (lower_url.find("youtube") != std::string::npos || lower_url.find("video") != std::string::npos ||
            lower_url.find("song") != std::string::npos || lower_url.find("music") != std::string::npos ||
            lower_url.find("audio") != std::string::npos || lower_url.find("canzone") != std::string::npos ||
            lower_url.find("brano") != std::string::npos || lower_url.find("de andre") != std::string::npos ||
            lower_url.find("play") != std::string::npos || lower_url.find("listen") != std::string::npos) {
            return "https://www.youtube.com/results?search_query=" + enc_query;
        }
        return "https://www.google.com/search?q=" + enc_query + "&hl=en&gl=us&gbv=1";
    }

    // Ensure scheme
    if (url.rfind("http://", 0) != 0 && url.rfind("https://", 0) != 0) {
        url = "https://" + url;
    }

    if (url.find("google.") != std::string::npos && url.find("/search") != std::string::npos) {
        if (url.find("hl=") == std::string::npos) {
            url += (url.find('?') == std::string::npos ? "?" : "&");
            url += "hl=en&gl=us";
        }
        if (url.find("gbv=") == std::string::npos) {
            url += (url.find('?') == std::string::npos ? "?" : "&");
            url += "gbv=1";
        }
    }

    return url;
}

inline std::string unescape_json_string(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        if (in[i] == '\\' && i + 1 < in.size()) {
            char next = in[i + 1];
            if (next == '"') { out.push_back('"'); i++; }
            else if (next == '\\') { out.push_back('\\'); i++; }
            else if (next == '/') { out.push_back('/'); i++; }
            else if (next == 'n') { out.push_back('\n'); i++; }
            else if (next == 'r') { out.push_back('\r'); i++; }
            else if (next == 't') { out.push_back('\t'); i++; }
            else if (next == 'u' && i + 5 < in.size()) {
                std::string hex_str = in.substr(i + 2, 4);
                try {
                    unsigned int cp = (unsigned int)std::stoul(hex_str, nullptr, 16);
                    if (cp < 0x80) {
                        out.push_back((char)cp);
                    } else if (cp < 0x800) {
                        out.push_back((char)(0xC0 | (cp >> 6)));
                        out.push_back((char)(0x80 | (cp & 0x3F)));
                    } else {
                        out.push_back((char)(0xE0 | (cp >> 12)));
                        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
                        out.push_back((char)(0x80 | (cp & 0x3F)));
                    }
                    i += 5;
                } catch (...) {
                    out.push_back(in[i]);
                }
            } else {
                out.push_back(in[i]);
            }
        } else {
            out.push_back(in[i]);
        }
    }
    return out;
}

inline std::string extract_json_ld_structured_data(const std::string& html, size_t max_chars = 3000) {
    std::string out;
    size_t pos = 0;
    while (pos < html.size() && out.size() < max_chars) {
        size_t ld_pos = html.find("type=\"application/ld+json\"", pos);
        if (ld_pos == std::string::npos) ld_pos = html.find("type='application/ld+json'", pos);
        if (ld_pos == std::string::npos) break;

        size_t tag_close = html.find('>', ld_pos);
        if (tag_close == std::string::npos) break;

        size_t script_end = html.find("</script>", tag_close);
        if (script_end == std::string::npos) break;

        std::string json_body = html.substr(tag_close + 1, script_end - (tag_close + 1));
        size_t f = json_body.find_first_not_of(" \t\r\n");
        size_t l = json_body.find_last_not_of(" \t\r\n");
        if (f != std::string::npos && l != std::string::npos && l >= f) {
            std::string trimmed = json_body.substr(f, l - f + 1);
            try {
                auto parsed = json::parse(trimmed);
                if (parsed.is_object()) {
                    std::string headline = parsed.value("headline", parsed.value("name", ""));
                    std::string desc = parsed.value("description", "");
                    std::string body = parsed.value("articleBody", parsed.value("text", ""));
                    if (!headline.empty() || !desc.empty() || !body.empty()) {
                        out += "[Structured Article / Page Data]\n";
                        if (!headline.empty()) out += "Headline: " + headline + "\n";
                        if (!desc.empty()) out += "Description: " + desc + "\n";
                        if (!body.empty()) out += "Content:\n" + body + "\n";
                        out += "\n";
                    }
                }
            } catch (...) {}
        }
        pos = script_end + 9;
    }
    return out;
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

inline std::string extract_pattern_contexts(const std::string& content, const std::string& pattern, size_t max_chars = 4000, size_t context_radius = 250) {
    if (pattern.empty()) {
        if (content.size() > max_chars) {
            return content.substr(0, max_chars) + "\n\n... [Content truncated to " + std::to_string(max_chars) + " characters. Full content cached to .moecher_web_cache.html]";
        }
        return content;
    }
    std::string lower_content = content;
    std::string lower_pattern = pattern;
    std::transform(lower_content.begin(), lower_content.end(), lower_content.begin(), [](unsigned char c){ return (char)::tolower(c); });
    std::transform(lower_pattern.begin(), lower_pattern.end(), lower_pattern.begin(), [](unsigned char c){ return (char)::tolower(c); });

    std::vector<size_t> matches;
    size_t pos = 0;
    while ((pos = lower_content.find(lower_pattern, pos)) != std::string::npos) {
        matches.push_back(pos);
        pos += lower_pattern.size();
    }

    if (matches.empty()) {
        return "[Pattern '" + pattern + "' not found in " + std::to_string(content.size()) + " bytes of document content. Full document cached to .moecher_web_cache.html]";
    }

    std::string out = "[Found " + std::to_string(matches.size()) + " occurrences of '" + pattern + "' in " + std::to_string(content.size()) + " bytes]:\n\n";
    size_t count = 0;
    for (size_t idx : matches) {
        count++;
        size_t start = (idx > context_radius) ? (idx - context_radius) : 0;
        size_t end = std::min(content.size(), idx + pattern.size() + context_radius);
        std::string snippet = content.substr(start, end - start);
        out += "--- [Match #" + std::to_string(count) + " (Byte offset " + std::to_string(idx) + ")] ---\n";
        out += snippet + "\n\n";
        if (out.size() >= max_chars) {
            out += "... [Truncated at " + std::to_string(max_chars) + " characters. Full content cached to .moecher_web_cache.html. Use execute_command with python/grep to inspect specific data.]";
            break;
        }
    }
    return out;
}

inline std::string extract_detected_resource_links(const std::string& html, const std::string& base_url, size_t max_links = 10) {
    std::string out;
    std::unordered_set<std::string> seen;
    size_t count = 0;

    // Detect YouTube video links
    size_t pos = 0;
    while (count < max_links && (pos = html.find("watch?v=", pos)) != std::string::npos) {
        size_t id_start = pos + 8;
        if (id_start + 11 <= html.size()) {
            std::string vid = html.substr(id_start, 11);
            bool valid = true;
            for (char c : vid) {
                if (!isalnum((unsigned char)c) && c != '-' && c != '_') { valid = false; break; }
            }
            if (valid && seen.insert(vid).second) {
                count++;
                if (out.empty()) out = "\n[Detected Video / Media Links]:\n";
                out += std::to_string(count) + ". https://www.youtube.com/watch?v=" + vid + "\n";
            }
        }
        pos += 8;
    }
    return out;
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
       << "      <iframe id=\"player\" src=\"https://www.youtube-nocookie.com/embed/" << video_id << "?autoplay=1&enablejsapi=1&rel=0\" allow=\"accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share\" allowfullscreen style=\"position:absolute;top:0;left:0;width:100%;height:100%;border:none;\"></iframe>\n"
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
       << "      try {\n"
       << "        player = new YT.Player('player', {\n"
       << "          events: {\n"
       << "            'onReady': function(e) {\n"
       << "              try { e.target.unMute(); e.target.setVolume(100); e.target.playVideo(); } catch(err) {}\n"
       << "            }\n"
       << "          }\n"
       << "        });\n"
       << "      } catch(e) {}\n"
       << "    }\n"
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

inline bool is_raw_code_url(const std::string& url) {
    std::string lower = url;
    for (auto& c : lower) c = (char)tolower(c);
    size_t q = lower.find('?');
    if (q != std::string::npos) lower = lower.substr(0, q);
    size_t hash = lower.find('#');
    if (hash != std::string::npos) lower = lower.substr(0, hash);

    static const std::vector<std::string> exts = {
        ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".json", ".css", ".scss",
        ".py", ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx", ".rs", ".go",
        ".java", ".cs", ".php", ".rb", ".sh", ".bash", ".bat", ".cmd", ".ps1",
        ".xml", ".yaml", ".yml", ".toml", ".sql", ".txt", ".md", ".log", ".ini", ".cfg"
    };
    for (const auto& ext : exts) {
        if (lower.size() >= ext.size() && lower.compare(lower.size() - ext.size(), ext.size(), ext) == 0) {
            return true;
        }
    }
    return false;
}

inline std::string extract_scripts_from_html(const std::string& html, const std::string& pattern = "", size_t offset = 0, size_t max_chars = 8000) {
    std::string result;
    size_t pos = 0;
    int count = 0;
    int matched_count = 0;

    std::string lower_pattern = pattern;
    std::transform(lower_pattern.begin(), lower_pattern.end(), lower_pattern.begin(), [](unsigned char c){ return (char)::tolower(c); });

    while (pos < html.size()) {
        size_t script_start = html.find("<script", pos);
        if (script_start == std::string::npos) break;
        size_t tag_end = html.find('>', script_start);
        if (tag_end == std::string::npos) break;

        std::string tag_header = html.substr(script_start, tag_end - script_start + 1);
        size_t script_end = html.find("</script>", tag_end);
        if (script_end == std::string::npos) break;

        std::string script_body = html.substr(tag_end + 1, script_end - (tag_end + 1));
        size_t first = script_body.find_first_not_of(" \t\r\n");
        size_t last = script_body.find_last_not_of(" \t\r\n");
        if (first != std::string::npos && last != std::string::npos && last >= first) {
            std::string trimmed = script_body.substr(first, last - first + 1);
            if (trimmed.size() > 5) {
                count++;
                bool matches = true;
                if (!lower_pattern.empty()) {
                    std::string lower_body = trimmed;
                    std::transform(lower_body.begin(), lower_body.end(), lower_body.begin(), [](unsigned char c){ return (char)::tolower(c); });
                    std::string lower_head = tag_header;
                    std::transform(lower_head.begin(), lower_head.end(), lower_head.begin(), [](unsigned char c){ return (char)::tolower(c); });
                    matches = (lower_body.find(lower_pattern) != std::string::npos || lower_head.find(lower_pattern) != std::string::npos);
                }

                if (matches) {
                    matched_count++;
                    if (matched_count > (int)offset) {
                        result += "--- [Script #" + std::to_string(count) + " (" + tag_header + ")] (length: " + std::to_string(trimmed.size()) + " chars) ---\n";
                        if (trimmed.size() > max_chars && pattern.empty()) {
                            result += trimmed.substr(0, max_chars) + "\n... [Script truncated. Use pattern filter or execute_command on .moecher_web_cache.html]\n\n";
                        } else {
                            result += trimmed + "\n\n";
                        }
                        if (result.size() >= max_chars) {
                            result = result.substr(0, max_chars) + "\n... [Scripts output reached " + std::to_string(max_chars) + " characters. Use execute_command on .moecher_web_cache.html for full data access]";
                            break;
                        }
                    }
                }
            }
        }
        pos = script_end + 9;
    }
    if (result.empty()) {
        if (!pattern.empty()) {
            return "[No script or data blocks matching pattern '" + pattern + "' found in document. Full document cached to .moecher_web_cache.html]";
        }
        return "[No inline script or data blocks found in document. Full document cached to .moecher_web_cache.html]";
    }
    return result;
}

inline std::string extract_links_from_html(const std::string& html, const std::string& base_url, const std::string& pattern = "", size_t offset = 0, size_t max_chars = 6000) {
    std::string result = "Links found on page (" + base_url + "):\n";
    size_t pos = 0;
    int count = 0;
    int matched_count = 0;

    std::string lower_pattern = pattern;
    std::transform(lower_pattern.begin(), lower_pattern.end(), lower_pattern.begin(), [](unsigned char c){ return (char)::tolower(c); });

    while (pos < html.size()) {
        size_t a_start = html.find("<a ", pos);
        if (a_start == std::string::npos) a_start = html.find("<a\t", pos);
        if (a_start == std::string::npos) a_start = html.find("<a\n", pos);
        if (a_start == std::string::npos) break;

        size_t tag_end = html.find('>', a_start);
        if (tag_end == std::string::npos) break;

        std::string tag_header = html.substr(a_start, tag_end - a_start + 1);
        size_t a_end = html.find("</a>", tag_end);
        std::string link_text;
        if (a_end != std::string::npos && a_end - tag_end < 300) {
            link_text = html.substr(tag_end + 1, a_end - (tag_end + 1));
            std::string clean_link_text;
            bool inside = false;
            for (char c : link_text) {
                if (c == '<') inside = true;
                else if (c == '>') inside = false;
                else if (!inside) clean_link_text += c;
            }
            link_text = clean_link_text;
            pos = a_end + 4;
        } else {
            pos = tag_end + 1;
        }

        size_t href_pos = tag_header.find("href=");
        if (href_pos != std::string::npos) {
            char quote = tag_header[href_pos + 5];
            size_t val_start = href_pos + 5;
            if (quote == '"' || quote == '\'') {
                val_start++;
                size_t val_end = tag_header.find(quote, val_start);
                if (val_end != std::string::npos) {
                    std::string href = tag_header.substr(val_start, val_end - val_start);
                    if (!href.empty() && href != "#" && href.find("javascript:") != 0) {
                        count++;
                        size_t f = link_text.find_first_not_of(" \t\r\n");
                        size_t l = link_text.find_last_not_of(" \t\r\n");
                        std::string t = (f != std::string::npos) ? link_text.substr(f, l - f + 1) : "";
                        if (t.empty()) t = "[No Anchor Text]";

                        bool matches = true;
                        if (!lower_pattern.empty()) {
                            std::string lower_href = href;
                            std::string lower_t = t;
                            std::transform(lower_href.begin(), lower_href.end(), lower_href.begin(), [](unsigned char c){ return (char)::tolower(c); });
                            std::transform(lower_t.begin(), lower_t.end(), lower_t.begin(), [](unsigned char c){ return (char)::tolower(c); });
                            matches = (lower_href.find(lower_pattern) != std::string::npos || lower_t.find(lower_pattern) != std::string::npos);
                        }

                        if (matches) {
                            matched_count++;
                            if (matched_count > (int)offset) {
                                result += std::to_string(matched_count) + ". " + t + " -> " + href + "\n";
                                if (result.size() >= max_chars) {
                                    result += "... [Links truncated at " + std::to_string(max_chars) + " characters]";
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if (matched_count == 0) {
        if (!pattern.empty()) return "[No links matching pattern '" + pattern + "' found on page]";
        return "[No links found on page]";
    }
    return result;
}

inline std::string fetch_url_with_browser_dom(const std::string& url, int timeout_ms = 10000) {
    std::string ua = get_client_user_agent();
    std::string lang = get_client_accept_language();

#if defined(_WIN32) || defined(_WIN64)
    std::string user_data_edge = "";
    std::string user_data_chrome = "";
    const char* local_app_data = getenv("LOCALAPPDATA");
    if (local_app_data && strlen(local_app_data) > 0) {
        user_data_edge = std::string(local_app_data) + "\\Microsoft\\Edge\\User Data";
        user_data_chrome = std::string(local_app_data) + "\\Google\\Chrome\\User Data";
    }

    struct BrowserConfig {
        std::string exe_path;
        std::string user_data;
    };

    const char* temp_env = getenv("TEMP");
    std::string temp_profile = (temp_env && strlen(temp_env) > 0) ? (std::string(temp_env) + "\\moecher_browser_profile") : ".moecher_browser_profile";

    std::vector<BrowserConfig> candidates;
    // 1. Primary Strategy: Isolated temporary profile (Avoids profile locks when Edge/Chrome is actively open)
    candidates.push_back({"C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe", temp_profile});
    candidates.push_back({"C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe", temp_profile});
    candidates.push_back({"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe", temp_profile});
    candidates.push_back({"C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe", temp_profile});

    // 2. Secondary Strategy: User profile if available
    if (!user_data_edge.empty()) {
        candidates.push_back({"C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe", user_data_edge});
        candidates.push_back({"C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe", user_data_edge});
    }
    if (!user_data_chrome.empty()) {
        candidates.push_back({"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe", user_data_chrome});
        candidates.push_back({"C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe", user_data_chrome});
    }

    for (const auto& b : candidates) {
        DWORD attrib = GetFileAttributesA(b.exe_path.c_str());
        if (attrib != INVALID_FILE_ATTRIBUTES && !(attrib & FILE_ATTRIBUTE_DIRECTORY)) {
            std::string cmd = "\"" + b.exe_path + "\" --headless=new --disable-gpu --user-data-dir=\"" + b.user_data + "\" --profile-directory=\"Default\" --disable-blink-features=AutomationControlled --disable-features=IsolateOrigins,site-per-process --no-first-run --no-default-browser-check --disable-dev-shm-usage --disable-infobars --window-size=1920,1080";
            if (!ua.empty()) {
                cmd += " \"--user-agent=" + ua + "\"";
            }
            if (!lang.empty()) {
                cmd += " \"--lang=" + lang + "\"";
            }
            cmd += " --dump-dom \"" + url + "\"";
            std::string rendered = execute_system_command(cmd, timeout_ms, 4000000);
            if (!rendered.empty() && (rendered.find("<html") != std::string::npos ||
                                      rendered.find("<!DOCTYPE") != std::string::npos ||
                                      rendered.find("<!doctype") != std::string::npos ||
                                      rendered.find("<HTML") != std::string::npos)) {
                return rendered;
            }
        }
    }
#else
    std::string profile_dir = "/tmp/moecher_browser_profile";
    std::string ua_flag = ua.empty() ? "" : (" \"--user-agent=" + ua + "\"");
    std::string lang_flag = lang.empty() ? "" : (" \"--lang=" + lang + "\"");
    std::string profile_flag = " \"--user-data-dir=" + profile_dir + "\"";
    std::string cmd = "google-chrome --headless=new --disable-gpu" + profile_flag + " --disable-blink-features=AutomationControlled --disable-features=IsolateOrigins,site-per-process --no-first-run --no-default-browser-check --disable-dev-shm-usage --window-size=1920,1080" + ua_flag + lang_flag + " --dump-dom \"" + url + "\" 2>/dev/null || "
                      "chromium-browser --headless=new --disable-gpu" + profile_flag + " --disable-blink-features=AutomationControlled --disable-features=IsolateOrigins,site-per-process --no-first-run --no-default-browser-check --disable-dev-shm-usage --window-size=1920,1080" + ua_flag + lang_flag + " --dump-dom \"" + url + "\" 2>/dev/null || "
                      "chromium --headless=new --disable-gpu" + profile_flag + " --disable-blink-features=AutomationControlled --disable-features=IsolateOrigins,site-per-process --no-first-run --no-default-browser-check --disable-dev-shm-usage --window-size=1920,1080" + ua_flag + lang_flag + " --dump-dom \"" + url + "\" 2>/dev/null || "
                      "microsoft-edge --headless=new --disable-gpu" + profile_flag + " --disable-blink-features=AutomationControlled --disable-features=IsolateOrigins,site-per-process --no-first-run --no-default-browser-check --disable-dev-shm-usage --window-size=1920,1080" + ua_flag + lang_flag + " --dump-dom \"" + url + "\" 2>/dev/null";
    std::string rendered = execute_system_command(cmd, timeout_ms, 4000000);
    if (!rendered.empty() && (rendered.find("<html") != std::string::npos ||
                              rendered.find("<!DOCTYPE") != std::string::npos ||
                              rendered.find("<!doctype") != std::string::npos ||
                              rendered.find("<HTML") != std::string::npos)) {
        return rendered;
    }
#endif
    return "";
}

inline std::string fetch_http_fast(const std::string& url, int timeout_ms = 4000) {
#if defined(_WIN32) || defined(_WIN64)
    int wlen = MultiByteToWideChar(CP_UTF8, 0, url.c_str(), -1, NULL, 0);
    std::vector<wchar_t> wurl(wlen);
    MultiByteToWideChar(CP_UTF8, 0, url.c_str(), -1, wurl.data(), wlen);

    URL_COMPONENTS url_comp;
    ZeroMemory(&url_comp, sizeof(url_comp));
    url_comp.dwStructSize = sizeof(url_comp);
    url_comp.dwHostNameLength = (DWORD)-1;
    url_comp.dwUrlPathLength = (DWORD)-1;
    url_comp.dwExtraInfoLength = (DWORD)-1;

    if (!WinHttpCrackUrl(wurl.data(), (DWORD)wurl.size(), 0, &url_comp)) {
        return "";
    }

    std::wstring host(url_comp.lpszHostName, url_comp.dwHostNameLength);
    std::wstring path(url_comp.lpszUrlPath, url_comp.dwUrlPathLength);
    if (url_comp.dwExtraInfoLength > 0) {
        path += std::wstring(url_comp.lpszExtraInfo, url_comp.dwExtraInfoLength);
    }
    if (path.empty()) path = L"/";

    bool is_https = (url_comp.nScheme == INTERNET_SCHEME_HTTPS);
    INTERNET_PORT port = url_comp.nPort;

    std::string ua = get_client_user_agent();
    std::string lang = get_client_accept_language();
    int ua_wlen = MultiByteToWideChar(CP_UTF8, 0, ua.c_str(), -1, NULL, 0);
    std::vector<wchar_t> w_ua(ua_wlen);
    MultiByteToWideChar(CP_UTF8, 0, ua.c_str(), -1, w_ua.data(), ua_wlen);

    HINTERNET h_session = WinHttpOpen(w_ua.empty() ? L"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" : w_ua.data(),
                                      WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                      WINHTTP_NO_PROXY_NAME,
                                      WINHTTP_NO_PROXY_BYPASS, 0);
    if (!h_session) return "";

    WinHttpSetTimeouts(h_session, timeout_ms, timeout_ms, timeout_ms, timeout_ms);

    HINTERNET h_connect = WinHttpConnect(h_session, host.c_str(), port, 0);
    if (!h_connect) {
        WinHttpCloseHandle(h_session);
        return "";
    }

    DWORD open_flags = is_https ? WINHTTP_FLAG_SECURE : 0;
    HINTERNET h_request = WinHttpOpenRequest(h_connect, L"GET", path.c_str(),
                                            NULL, WINHTTP_NO_REFERER,
                                            WINHTTP_DEFAULT_ACCEPT_TYPES, open_flags);
    if (!h_request) {
        WinHttpCloseHandle(h_connect);
        WinHttpCloseHandle(h_session);
        return "";
    }

    DWORD opt_redirect = WINHTTP_OPTION_REDIRECT_POLICY_ALWAYS;
    WinHttpSetOption(h_request, WINHTTP_OPTION_REDIRECT_POLICY, &opt_redirect, sizeof(opt_redirect));

    std::string cookies = get_cookies_for_url(url);
    std::string hdr_str = "User-Agent: " + ua + "\r\n"
                          "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8\r\n"
                          "Accept-Language: " + (lang.empty() ? "en-US,en;q=0.9,it;q=0.8" : lang) + "\r\n"
                          "Sec-Ch-Ua: " + g_client_sec_ch_ua + "\r\n"
                          "Sec-Ch-Ua-Mobile: " + g_client_sec_ch_ua_mobile + "\r\n"
                          "Sec-Ch-Ua-Platform: " + g_client_sec_ch_ua_platform + "\r\n"
                          "Sec-Fetch-Dest: document\r\n"
                          "Sec-Fetch-Mode: navigate\r\n"
                          "Sec-Fetch-Site: none\r\n"
                          "Sec-Fetch-User: ?1\r\n"
                          "Upgrade-Insecure-Requests: 1\r\n";
    if (!cookies.empty()) {
        hdr_str += "Cookie: " + cookies + "\r\n";
    }

    int hdr_wlen = MultiByteToWideChar(CP_UTF8, 0, hdr_str.c_str(), -1, NULL, 0);
    std::vector<wchar_t> w_hdr(hdr_wlen);
    MultiByteToWideChar(CP_UTF8, 0, hdr_str.c_str(), -1, w_hdr.data(), hdr_wlen);

    BOOL send_ok = WinHttpSendRequest(h_request, w_hdr.data(), (DWORD)-1L, WINHTTP_NO_REQUEST_DATA, 0, 0, 0);
    if (!send_ok || !WinHttpReceiveResponse(h_request, NULL)) {
        WinHttpCloseHandle(h_request);
        WinHttpCloseHandle(h_connect);
        WinHttpCloseHandle(h_session);
        return "";
    }

    // Intercept Set-Cookie response headers to maintain cookie jar
    DWORD raw_hdr_size = 0;
    WinHttpQueryHeaders(h_request, WINHTTP_QUERY_RAW_HEADERS_CRLF, WINHTTP_HEADER_NAME_BY_INDEX, NULL, &raw_hdr_size, WINHTTP_NO_HEADER_INDEX);
    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER && raw_hdr_size > 0) {
        std::vector<wchar_t> raw_hdr_buf(raw_hdr_size / sizeof(wchar_t) + 1, 0);
        if (WinHttpQueryHeaders(h_request, WINHTTP_QUERY_RAW_HEADERS_CRLF, WINHTTP_HEADER_NAME_BY_INDEX, raw_hdr_buf.data(), &raw_hdr_size, WINHTTP_NO_HEADER_INDEX)) {
            int utf8_len = WideCharToMultiByte(CP_UTF8, 0, raw_hdr_buf.data(), -1, NULL, 0, NULL, NULL);
            if (utf8_len > 0) {
                std::vector<char> utf8_buf(utf8_len, 0);
                WideCharToMultiByte(CP_UTF8, 0, raw_hdr_buf.data(), -1, utf8_buf.data(), utf8_len, NULL, NULL);
                parse_and_store_set_cookie_header(url, std::string(utf8_buf.data()));
            }
        }
    }

    std::string response_data;
    DWORD bytes_available = 0;
    while (WinHttpQueryDataAvailable(h_request, &bytes_available) && bytes_available > 0) {
        std::vector<char> temp_buf(bytes_available);
        DWORD bytes_read = 0;
        if (WinHttpReadData(h_request, temp_buf.data(), bytes_available, &bytes_read) && bytes_read > 0) {
            response_data.append(temp_buf.data(), bytes_read);
            if (response_data.size() > 1000000) break;
        } else {
            break;
        }
    }

    WinHttpCloseHandle(h_request);
    WinHttpCloseHandle(h_connect);
    WinHttpCloseHandle(h_session);
    return response_data;
#else
    std::string cookies = get_cookies_for_url(url);
    std::string cookie_flag = cookies.empty() ? "" : (" -H \"Cookie: " + cookies + "\"");
    std::string curl_cmd = "curl -sL --max-time " + std::to_string(timeout_ms / 1000) +
                           " -A \"" + (get_client_user_agent().empty() ? "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" : get_client_user_agent()) + "\"" +
                           " -H \"Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8\"" +
                           " -H \"Accept-Language: " + (get_client_accept_language().empty() ? "en-US,en;q=0.9" : get_client_accept_language()) + "\"" +
                           cookie_flag + " \"" + url + "\"";
    return execute_system_command(curl_cmd, timeout_ms, 500000);
#endif
}

inline std::string extract_search_results_from_html(const std::string& html, const std::string& url, size_t max_results = 8) {
    // 1. YouTube Search Results (youtube.com/results or ytInitialData)
    if (url.find("youtube.com/results") != std::string::npos || url.find("/results?search_query=") != std::string::npos || html.find("videoRenderer") != std::string::npos || html.find("ytInitialData") != std::string::npos) {
        std::string out;
        std::unordered_set<std::string> seen_vids;
        size_t count = 0;

        // Strategy A: ytInitialData JSON parsing (Extracts full video list in ~1ms)
        size_t data_pos = html.find("ytInitialData");
        if (data_pos != std::string::npos) {
            size_t j_start = html.find('{', data_pos);
            if (j_start != std::string::npos) {
                size_t j_end = html.find(";</script>", j_start);
                if (j_end == std::string::npos) j_end = html.find("</script>", j_start);
                if (j_end != std::string::npos) {
                    std::string j_str = html.substr(j_start, j_end - j_start);
                    try {
                        auto data = json::parse(j_str);
                        std::function<void(const json&)> find_renderers = [&](const json& j) {
                            if (count >= max_results) return;
                            if (j.is_object()) {
                                if (j.contains("videoRenderer")) {
                                    const auto& vr = j["videoRenderer"];
                                    std::string vid = vr.value("videoId", "");
                                    if (!vid.empty() && seen_vids.insert(vid).second) {
                                        std::string title;
                                        if (vr.contains("title")) {
                                            if (vr["title"].contains("runs") && vr["title"]["runs"].is_array() && !vr["title"]["runs"].empty()) {
                                                for (const auto& r : vr["title"]["runs"]) {
                                                    title += r.value("text", "");
                                                }
                                            } else if (vr["title"].contains("simpleText")) {
                                                title = vr["title"].value("simpleText", "");
                                            }
                                        }
                                        std::string channel;
                                        if (vr.contains("ownerText") && vr["ownerText"].contains("runs") && vr["ownerText"]["runs"].is_array() && !vr["ownerText"]["runs"].empty()) {
                                            channel = vr["ownerText"]["runs"][0].value("text", "");
                                        } else if (vr.contains("shortBylineText") && vr["shortBylineText"].contains("runs") && vr["shortBylineText"]["runs"].is_array() && !vr["shortBylineText"]["runs"].empty()) {
                                            channel = vr["shortBylineText"]["runs"][0].value("text", "");
                                        }
                                        std::string duration;
                                        if (vr.contains("lengthText")) {
                                            if (vr["lengthText"].contains("simpleText")) {
                                                duration = vr["lengthText"].value("simpleText", "");
                                            } else if (vr["lengthText"].contains("runs") && vr["lengthText"]["runs"].is_array() && !vr["lengthText"]["runs"].empty()) {
                                                duration = vr["lengthText"]["runs"][0].value("text", "");
                                            }
                                        }
                                        if (title.empty()) title = "YouTube Video (" + vid + ")";
                                        count++;
                                        if (out.empty()) out = "[YouTube Search Results]:\n";
                                        out += std::to_string(count) + ". " + title;
                                        if (!duration.empty()) out += " (" + duration + ")";
                                        if (!channel.empty()) out += " - " + channel;
                                        out += "\n   URL: https://www.youtube.com/watch?v=" + vid + "\n\n";
                                    }
                                }
                                for (auto it = j.begin(); it != j.end(); ++it) {
                                    if (it.value().is_structured()) find_renderers(it.value());
                                }
                            } else if (j.is_array()) {
                                for (const auto& elem : j) {
                                    if (count >= max_results) break;
                                    find_renderers(elem);
                                }
                            }
                        };
                        find_renderers(data);
                    } catch (...) {}
                }
            }
        }

        if (!out.empty()) return out;

        // Strategy B: Direct videoRenderer substring extraction
        size_t vr_pos = 0;
        while (count < max_results && (vr_pos = html.find("\"videoRenderer\":{\"videoId\":\"", vr_pos)) != std::string::npos) {
            size_t id_start = vr_pos + 28;
            if (id_start + 11 <= html.size()) {
                std::string vid = html.substr(id_start, 11);
                bool valid = true;
                for (char c : vid) {
                    if (!isalnum((unsigned char)c) && c != '-' && c != '_') { valid = false; break; }
                }

                if (valid && seen_vids.insert(vid).second) {
                    std::string title;
                    std::string duration;
                    std::string channel;

                    size_t block_end = std::min(html.size(), vr_pos + 1200);
                    std::string block = html.substr(vr_pos, block_end - vr_pos);

                    size_t t_pos = block.find("\"title\":{\"runs\":[{\"text\":\"");
                    if (t_pos != std::string::npos) {
                        size_t t_start = t_pos + 26;
                        size_t t_end = block.find('"', t_start);
                        if (t_end != std::string::npos) {
                            title = block.substr(t_start, t_end - t_start);
                        }
                    }

                    size_t ch_pos = block.find("\"longBylineText\":{\"runs\":[{\"text\":\"");
                    if (ch_pos == std::string::npos) ch_pos = block.find("\"ownerText\":{\"runs\":[{\"text\":\"");
                    if (ch_pos != std::string::npos) {
                        size_t ch_start = block.find("\"text\":\"", ch_pos);
                        if (ch_start != std::string::npos) {
                            ch_start += 8;
                            size_t ch_end = block.find('"', ch_start);
                            if (ch_end != std::string::npos) {
                                channel = block.substr(ch_start, ch_end - ch_start);
                            }
                        }
                    }

                    size_t dur_pos = block.find("\"lengthText\":{\"simpleText\":\"");
                    if (dur_pos != std::string::npos) {
                        size_t dur_start = dur_pos + 28;
                        size_t dur_end = block.find('"', dur_start);
                        if (dur_end != std::string::npos) {
                            duration = block.substr(dur_start, dur_end - dur_start);
                        }
                    }

                    if (title.empty()) title = "YouTube Video (" + vid + ")";
                    title = unescape_json_string(title);
                    if (!channel.empty()) channel = unescape_json_string(channel);

                    count++;
                    if (out.empty()) out = "[YouTube Search Results]:\n";
                    out += std::to_string(count) + ". " + title;
                    if (!duration.empty()) out += " (" + duration + ")";
                    if (!channel.empty()) out += " - " + channel;
                    out += "\n   URL: https://www.youtube.com/watch?v=" + vid + "\n\n";
                }
            }
            vr_pos += 28;
        }

        if (!out.empty()) return out;

        // Strategy C: DOM Anchor extraction fallback
        size_t pos = 0;
        while (count < max_results && (pos = html.find("watch?v=", pos)) != std::string::npos) {
            size_t id_start = pos + 8;
            if (id_start + 11 <= html.size()) {
                std::string vid = html.substr(id_start, 11);
                bool valid = true;
                for (char c : vid) {
                    if (!isalnum((unsigned char)c) && c != '-' && c != '_') { valid = false; break; }
                }

                if (valid && seen_vids.insert(vid).second) {
                    std::string label;
                    size_t a_start = (pos > 400) ? html.rfind("<a", pos) : 0;
                    if (a_start != std::string::npos && (pos - a_start) < 400) {
                        size_t tag_end = html.find('>', a_start);
                        if (tag_end != std::string::npos && tag_end > pos) {
                            std::string tag_str = html.substr(a_start, tag_end - a_start + 1);
                            size_t aria_pos = tag_str.find("aria-label=\"");
                            if (aria_pos != std::string::npos) {
                                size_t q_end = tag_str.find('"', aria_pos + 12);
                                if (q_end != std::string::npos) {
                                    label = tag_str.substr(aria_pos + 12, q_end - (aria_pos + 12));
                                }
                            }
                        }
                    }
                    if (label.empty()) label = "YouTube Video (" + vid + ")";
                    count++;
                    if (out.empty()) out = "[YouTube Search Results]:\n";
                    out += std::to_string(count) + ". " + label + "\n";
                    out += "   URL: https://www.youtube.com/watch?v=" + vid + "\n\n";
                }
            }
            pos += 8;
        }

        if (!out.empty()) return out;
    }

    // 2. Google / Web Search Results Parsing
    if (url.find("google.") != std::string::npos || url.find("duckduckgo.com") != std::string::npos) {
        std::string out;
        std::unordered_set<std::string> seen_urls;
        size_t count = 0;
        size_t pos = 0;

        while (count < max_results && pos < html.size()) {
            size_t h_pos = html.find("<h3", pos);
            if (h_pos == std::string::npos) h_pos = html.find("<h2", pos);
            if (h_pos == std::string::npos) break;

            size_t h_close = html.find('>', h_pos);
            if (h_close == std::string::npos) break;

            size_t h_end = html.find("</h3>", h_close);
            if (h_end == std::string::npos) h_end = html.find("</h2>", h_close);
            if (h_end == std::string::npos) break;

            std::string h_content = html.substr(h_close + 1, h_end - (h_close + 1));
            std::string target_url;

            size_t a_pos = h_content.find("href=\"");
            if (a_pos != std::string::npos) {
                size_t q_end = h_content.find('"', a_pos + 6);
                if (q_end != std::string::npos) target_url = h_content.substr(a_pos + 6, q_end - (a_pos + 6));
            } else {
                size_t lookback = (h_pos > 400) ? (h_pos - 400) : 0;
                std::string prec = html.substr(lookback, h_pos - lookback);
                size_t prec_a = prec.rfind("href=\"");
                if (prec_a != std::string::npos) {
                    size_t q_end = prec.find('"', prec_a + 6);
                    if (q_end != std::string::npos) target_url = prec.substr(prec_a + 6, q_end - (prec_a + 6));
                }
            }

            // Unwrap Google redirect wrappers (/url?q=... or /url?esrc=...)
            if (!target_url.empty() && target_url.find("/url?") != std::string::npos) {
                size_t qp = target_url.find("url=");
                if (qp == std::string::npos) qp = target_url.find("q=");
                if (qp != std::string::npos) {
                    size_t eq = target_url.find('=', qp);
                    if (eq != std::string::npos) {
                        std::string raw_target = target_url.substr(eq + 1);
                        size_t amp = raw_target.find('&');
                        if (amp != std::string::npos) raw_target = raw_target.substr(0, amp);
                        target_url = url_decode(raw_target);
                    }
                }
            }

            if (!target_url.empty() && target_url.rfind("http", 0) == 0 && target_url.find("google.com/search") == std::string::npos && target_url.find("google.it/search") == std::string::npos) {
                std::string clean_title;
                bool inside = false;
                for (char c : h_content) {
                    if (c == '<') inside = true;
                    else if (c == '>') inside = false;
                    else if (!inside) clean_title += c;
                }
                size_t f = clean_title.find_first_not_of(" \t\r\n");
                size_t l = clean_title.find_last_not_of(" \t\r\n");
                if (f != std::string::npos && l != std::string::npos) {
                    clean_title = clean_title.substr(f, l - f + 1);
                }

                if (!clean_title.empty() && clean_title != "Google" && seen_urls.insert(target_url).second) {
                    count++;
                    if (out.empty()) out = "[Google Search Results]:\n";
                    out += std::to_string(count) + ". " + clean_title + "\n";
                    out += "   URL: " + target_url + "\n\n";
                }
            }
            pos = h_end + 5;
        }

        if (!out.empty()) return out;
    }

    return "";
}

inline RetrievedDocument fetch_url_full(const std::string& raw_url, int timeout_ms = 10000, size_t max_chars = 4000, const std::string& mode = "text", const std::string& pattern = "", size_t offset = 0, const std::string& method = "GET", const std::string& post_body = "", const std::string& post_content_type = "") {
    RetrievedDocument doc;
    doc.url = sanitize_and_normalize_url(raw_url);
    if (doc.url.empty()) {
        doc.clean_text = "[Error: Empty URL]";
        return doc;
    }

    // 0. Fast direct YouTube video handler (Instant 0-latency playback & metadata resolution)
    std::string direct_yt_id = extract_youtube_video_id(doc.url);
    if (!direct_yt_id.empty() && doc.url.find("results") == std::string::npos && doc.url.find("search") == std::string::npos) {
        std::string title = "";
        std::string author = "";
        std::string desc = "";

        std::string oembed_url = "https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=" + direct_yt_id + "&format=json";
        std::string oembed_json = fetch_http_fast(oembed_url, 3000);
        if (!oembed_json.empty() && oembed_json.front() == '{') {
            try {
                auto oj = json::parse(oembed_json);
                title = oj.value("title", "");
                author = oj.value("author_name", "");
            } catch (...) {}
        }
        if (title.empty()) title = "YouTube Video (" + direct_yt_id + ")";

        doc.title = title;
        doc.raw_html = generate_youtube_preview_html(direct_yt_id, title, desc, author, doc.url);

        std::string text_summary = "[Retrieved YouTube Video]\n";
        text_summary += "Title: " + title + "\n";
        if (!author.empty()) text_summary += "Channel/Author: " + author + "\n";
        text_summary += "Video URL: https://www.youtube.com/watch?v=" + direct_yt_id + "\n";
        text_summary += "[Note: Interactive video player loaded in preview panel.]";
        doc.clean_text = text_summary;
        return doc;
    }

    // 0.5 Direct YouTube Query Handler for Music/Video Intent
    if (doc.url.find("google.") != std::string::npos || doc.url.find("/search") != std::string::npos) {
        size_t q_pos = doc.url.find("q=");
        if (q_pos != std::string::npos) {
            std::string query = doc.url.substr(q_pos + 2);
            size_t amp = query.find('&');
            if (amp != std::string::npos) query = query.substr(0, amp);
            std::string decoded_query = url_decode(query);
            std::string lower_query = decoded_query;
            std::transform(lower_query.begin(), lower_query.end(), lower_query.begin(), [](unsigned char c){ return (char)::tolower(c); });

            bool is_video_intent = (lower_query.find("youtube") != std::string::npos || 
                                    lower_query.find("video") != std::string::npos ||
                                    lower_query.find("song") != std::string::npos ||
                                    lower_query.find("music") != std::string::npos ||
                                    lower_query.find("audio") != std::string::npos ||
                                    lower_query.find("canzone") != std::string::npos ||
                                    lower_query.find("brano") != std::string::npos ||
                                    lower_query.find("listen") != std::string::npos ||
                                    lower_query.find("play") != std::string::npos ||
                                    lower_query.find("track") != std::string::npos ||
                                    lower_query.find("de andre") != std::string::npos);

            if (is_video_intent) {
                std::string yt_search_url = "https://www.youtube.com/results?search_query=" + query;
                std::string yt_html = fetch_http_fast(yt_search_url, 5000);
                if (yt_html.empty() || yt_html.find("videoRenderer") == std::string::npos) {
                    yt_html = fetch_url_with_browser_dom(yt_search_url, std::min(timeout_ms, 8000));
                }
                if (!yt_html.empty()) {
                    std::string yt_res = extract_search_results_from_html(yt_html, yt_search_url, 8);
                    if (!yt_res.empty()) {
                        doc.raw_html = yt_html;
                        doc.title = "YouTube Search Results (" + decoded_query + ")";
                        doc.clean_text = yt_res;
                        return doc;
                    }
                }
            }
        }
    }

    std::string raw_html;

    // 1. Primary Strategy: Live Browser DOM & JavaScript Execution (Edge / Chrome Engine)
    if (doc.url.find("youtube.com/results") != std::string::npos) {
        raw_html = fetch_http_fast(doc.url, 5000);
        if (raw_html.empty() || raw_html.find("videoRenderer") == std::string::npos) {
            raw_html = fetch_url_with_browser_dom(doc.url, std::min(timeout_ms, 8000));
        }
    } else if (!is_raw_code_url(doc.url) && method == "GET") {
        raw_html = fetch_url_with_browser_dom(doc.url, std::min(timeout_ms, 8000));
    }

    // If LinkedIn returned 999 authwall challenge script, follow the authwall redirect with cookie
    if (!raw_html.empty() && raw_html.find("authwall?trk=") != std::string::npos && doc.url.find("/authwall") == std::string::npos && doc.url.find("linkedin.com") != std::string::npos) {
        std::string authwall_url = "https://www.linkedin.com/authwall?trk=bf&trkInfo=bf&original_referer=&sessionRedirect=" + url_encode(doc.url);
        return fetch_url_full(authwall_url, timeout_ms, max_chars, mode, pattern, offset, method, post_body, post_content_type);
    }

    // 2. Fallback Strategy: Fast Native WinHTTP / curl with Modern Desktop Chrome Headers and Cookie Jar
    if (raw_html.empty()) {
#if defined(_WIN32) || defined(_WIN64)
    std::string cur_url = doc.url;
    std::string cur_method = method;
    std::string cur_post_body = post_body;
    std::string cur_post_ct = post_content_type;
    int max_redirects = 7;
    int redirect_count = 0;

    while (redirect_count < max_redirects) {
        int wlen = MultiByteToWideChar(CP_UTF8, 0, cur_url.c_str(), -1, NULL, 0);
        std::vector<wchar_t> wurl(wlen);
        MultiByteToWideChar(CP_UTF8, 0, cur_url.c_str(), -1, wurl.data(), wlen);

        URL_COMPONENTS url_comp;
        ZeroMemory(&url_comp, sizeof(url_comp));
        url_comp.dwStructSize = sizeof(url_comp);
        url_comp.dwHostNameLength = (DWORD)-1;
        url_comp.dwUrlPathLength = (DWORD)-1;
        url_comp.dwExtraInfoLength = (DWORD)-1;

        if (!WinHttpCrackUrl(wurl.data(), (DWORD)wurl.size(), 0, &url_comp)) {
            doc.clean_text = "[Error: Invalid URL format: " + cur_url + "]";
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

        std::string ua = get_client_user_agent();
        std::string lang = get_client_accept_language();
        int ua_wlen = MultiByteToWideChar(CP_UTF8, 0, ua.c_str(), -1, NULL, 0);
        std::vector<wchar_t> w_ua(ua_wlen);
        MultiByteToWideChar(CP_UTF8, 0, ua.c_str(), -1, w_ua.data(), ua_wlen);

        HINTERNET h_session = WinHttpOpen(w_ua.empty() ? L"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" : w_ua.data(),
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
            doc.clean_text = "[Error: WinHttpConnect failed to connect to " + cur_url + "]";
            return doc;
        }

        DWORD open_flags = is_https ? WINHTTP_FLAG_SECURE : 0;
        std::wstring w_method = L"GET";
        if (cur_method == "POST" || cur_method == "post") w_method = L"POST";
        else if (cur_method == "HEAD" || cur_method == "head") w_method = L"HEAD";

        HINTERNET h_request = WinHttpOpenRequest(h_connect, w_method.c_str(), path.c_str(),
                                                NULL, WINHTTP_NO_REFERER,
                                                WINHTTP_DEFAULT_ACCEPT_TYPES, open_flags);
        if (!h_request) {
            WinHttpCloseHandle(h_connect);
            WinHttpCloseHandle(h_session);
            doc.clean_text = "[Error: WinHttpOpenRequest failed]";
            return doc;
        }

        // Disable automatic redirect so we can intercept all intermediate Set-Cookie and Location headers
        DWORD opt_redirect = WINHTTP_OPTION_REDIRECT_POLICY_NEVER;
        WinHttpSetOption(h_request, WINHTTP_OPTION_REDIRECT_POLICY, &opt_redirect, sizeof(opt_redirect));

        std::string cookies = get_cookies_for_url(cur_url);
        std::string hdr_str = "User-Agent: " + ua + "\r\n"
                              "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8\r\n"
                              "Accept-Language: " + (lang.empty() ? "en-US,en;q=0.9,it;q=0.8" : lang) + "\r\n"
                              "Sec-Ch-Ua: " + g_client_sec_ch_ua + "\r\n"
                              "Sec-Ch-Ua-Mobile: " + g_client_sec_ch_ua_mobile + "\r\n"
                              "Sec-Ch-Ua-Platform: " + g_client_sec_ch_ua_platform + "\r\n"
                              "Sec-Fetch-Dest: document\r\n"
                              "Sec-Fetch-Mode: navigate\r\n"
                              "Sec-Fetch-Site: same-origin\r\n"
                              "Sec-Fetch-User: ?1\r\n"
                              "Upgrade-Insecure-Requests: 1\r\n";
        if (!cookies.empty()) {
            hdr_str += "Cookie: " + cookies + "\r\n";
        }
        if (!cur_post_ct.empty()) {
            hdr_str += "Content-Type: " + cur_post_ct + "\r\n";
        } else if (!cur_post_body.empty() && hdr_str.find("Content-Type:") == std::string::npos) {
            hdr_str += "Content-Type: application/x-www-form-urlencoded\r\n";
        }

        int hdr_wlen = MultiByteToWideChar(CP_UTF8, 0, hdr_str.c_str(), -1, NULL, 0);
        std::vector<wchar_t> w_hdr(hdr_wlen);
        MultiByteToWideChar(CP_UTF8, 0, hdr_str.c_str(), -1, w_hdr.data(), hdr_wlen);

        void* p_data = cur_post_body.empty() ? WINHTTP_NO_REQUEST_DATA : (void*)cur_post_body.data();
        DWORD dw_data_len = (DWORD)cur_post_body.size();

        BOOL send_ok = WinHttpSendRequest(h_request,
                                          w_hdr.data(), (DWORD)-1L,
                                          p_data, dw_data_len, dw_data_len, 0);
        if (!send_ok || !WinHttpReceiveResponse(h_request, NULL)) {
            WinHttpCloseHandle(h_request);
            WinHttpCloseHandle(h_connect);
            WinHttpCloseHandle(h_session);
            doc.clean_text = "[Error: Failed to receive HTTP response from " + cur_url + "]";
            return doc;
        }

        DWORD status_code = 0;
        DWORD status_size = sizeof(status_code);
        WinHttpQueryHeaders(h_request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                            WINHTTP_HEADER_NAME_BY_INDEX, &status_code, &status_size, WINHTTP_NO_HEADER_INDEX);

        // Save Set-Cookie response headers to cookie jar
        DWORD raw_hdr_size = 0;
        WinHttpQueryHeaders(h_request, WINHTTP_QUERY_RAW_HEADERS_CRLF, WINHTTP_HEADER_NAME_BY_INDEX, NULL, &raw_hdr_size, WINHTTP_NO_HEADER_INDEX);
        if (GetLastError() == ERROR_INSUFFICIENT_BUFFER && raw_hdr_size > 0) {
            std::vector<wchar_t> raw_hdr_buf(raw_hdr_size / sizeof(wchar_t) + 1, 0);
            if (WinHttpQueryHeaders(h_request, WINHTTP_QUERY_RAW_HEADERS_CRLF, WINHTTP_HEADER_NAME_BY_INDEX, raw_hdr_buf.data(), &raw_hdr_size, WINHTTP_NO_HEADER_INDEX)) {
                int utf8_len = WideCharToMultiByte(CP_UTF8, 0, raw_hdr_buf.data(), -1, NULL, 0, NULL, NULL);
                if (utf8_len > 0) {
                    std::vector<char> utf8_buf(utf8_len, 0);
                    WideCharToMultiByte(CP_UTF8, 0, raw_hdr_buf.data(), -1, utf8_buf.data(), utf8_len, NULL, NULL);
                    parse_and_store_set_cookie_header(cur_url, std::string(utf8_buf.data()));
                }
            }
        }

        // Check for redirects (301, 302, 303, 307, 308)
        if (status_code == 301 || status_code == 302 || status_code == 303 || status_code == 307 || status_code == 308) {
            DWORD loc_size = 0;
            WinHttpQueryHeaders(h_request, WINHTTP_QUERY_LOCATION, WINHTTP_HEADER_NAME_BY_INDEX, NULL, &loc_size, WINHTTP_NO_HEADER_INDEX);
            std::string next_loc = "";
            if (GetLastError() == ERROR_INSUFFICIENT_BUFFER && loc_size > 0) {
                std::vector<wchar_t> loc_buf(loc_size / sizeof(wchar_t) + 1, 0);
                if (WinHttpQueryHeaders(h_request, WINHTTP_QUERY_LOCATION, WINHTTP_HEADER_NAME_BY_INDEX, loc_buf.data(), &loc_size, WINHTTP_NO_HEADER_INDEX)) {
                    int utf8_len = WideCharToMultiByte(CP_UTF8, 0, loc_buf.data(), -1, NULL, 0, NULL, NULL);
                    if (utf8_len > 0) {
                        std::vector<char> utf8_buf(utf8_len, 0);
                        WideCharToMultiByte(CP_UTF8, 0, loc_buf.data(), -1, utf8_buf.data(), utf8_len, NULL, NULL);
                        next_loc = std::string(utf8_buf.data());
                    }
                }
            }
            WinHttpCloseHandle(h_request);
            WinHttpCloseHandle(h_connect);
            WinHttpCloseHandle(h_session);

            if (!next_loc.empty()) {
                cur_url = resolve_redirect_url(cur_url, next_loc);
                doc.url = cur_url;
                if (status_code == 301 || status_code == 302 || status_code == 303) {
                    cur_method = "GET";
                    cur_post_body.clear();
                    cur_post_ct.clear();
                }
                redirect_count++;
                continue;
            }
            break;
        }

        // Read body content for final response
        raw_html.clear();
        DWORD bytes_available = 0;
        while (WinHttpQueryDataAvailable(h_request, &bytes_available) && bytes_available > 0) {
            std::vector<char> temp_buf(bytes_available);
            DWORD bytes_read = 0;
            if (WinHttpReadData(h_request, temp_buf.data(), bytes_available, &bytes_read) && bytes_read > 0) {
                raw_html.append(temp_buf.data(), bytes_read);
                if (raw_html.size() > 4000000) break;
            } else {
                break;
            }
        }

        WinHttpCloseHandle(h_request);
        WinHttpCloseHandle(h_connect);
        WinHttpCloseHandle(h_session);

        // If LinkedIn returned 999, follow the authwall redirect with the stored trkCode cookie
        if (status_code == 999 && cur_url.find("linkedin.com") != std::string::npos && cur_url.find("/authwall") == std::string::npos) {
            std::string authwall_url = "https://www.linkedin.com/authwall?trk=bf&trkInfo=bf&original_referer=&sessionRedirect=" + url_encode(cur_url);
            return fetch_url_full(authwall_url, timeout_ms, max_chars, mode, pattern, offset, cur_method, cur_post_body, cur_post_ct);
        }

        if (status_code >= 400 && status_code != 429 && status_code != 999) {
            doc.raw_html = raw_html.empty() ? ("<!DOCTYPE html><html><body><h3>HTTP Error " + std::to_string(status_code) + "</h3><p>Could not load " + cur_url + "</p></body></html>") : raw_html;
            doc.title = extract_html_title(raw_html);
            doc.clean_text = "[HTTP Error " + std::to_string(status_code) + " when fetching " + cur_url + "]";
            return doc;
        }

        break;
    }

#else
    std::string ua = get_client_user_agent();
    std::string lang = get_client_accept_language();
    std::string cookies = get_cookies_for_url(doc.url);
    std::string cookie_flag = cookies.empty() ? "" : (" -H \"Cookie: " + cookies + "\"");
    std::string ua_flag = ua.empty() ? "" : (" -A \"" + ua + "\"");
    std::string lang_flag = lang.empty() ? "" : (" -H \"Accept-Language: " + lang + "\"");
    std::string curl_cmd = "curl -sL --max-time " + std::to_string(timeout_ms / 1000) +
                           ua_flag + lang_flag + cookie_flag +
                           " -H \"Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8\"" +
                           " -H \"Sec-Ch-Ua: \\\"Google Chrome\\\";v=\\\"133\\\", \\\"Chromium\\\";v=\\\"133\\\", \\\"Not_A Brand\\\";v=\\\"24\\\"\"" +
                           " -H \"Sec-Ch-Ua-Mobile: ?0\"" +
                           " -H \"Sec-Ch-Ua-Platform: \\\"Windows\\\"\"" +
                           " -H \"Sec-Fetch-Dest: document\"" +
                           " -H \"Sec-Fetch-Mode: navigate\"" +
                           " -H \"Sec-Fetch-Site: cross-site\"" +
                           " -H \"Sec-Fetch-User: ?1\"" +
                           " -H \"Upgrade-Insecure-Requests: 1\"" +
                           " \"" + doc.url + "\"";
    raw_html = execute_system_command(curl_cmd, timeout_ms, 4000000);
    if (raw_html.rfind("[Error", 0) == 0) {
        doc.clean_text = raw_html;
        return doc;
    }
#endif
    } // End of if (raw_html.empty()) fallback block

    // Cache raw document to .moecher_web_cache.html for LLM execute_command access
    try {
        std::ofstream cache_file(".moecher_web_cache.html", std::ios::binary | std::ios::trunc);
        if (cache_file.is_open()) {
            cache_file.write(raw_html.data(), raw_html.size());
            cache_file.close();
        }
    } catch (...) {}

    // Check if URL directly targets raw code/data or raw mode was requested
    bool is_code = is_raw_code_url(doc.url);
    if (is_code || mode == "raw") {
        doc.raw_html = raw_html;
        doc.title = is_code ? ("Source: " + doc.url) : extract_html_title(raw_html);
        if (!pattern.empty()) {
            doc.clean_text = extract_pattern_contexts(raw_html, pattern, max_chars * 2);
        } else if (raw_html.size() > max_chars * 2) {
            size_t start = std::min(offset, raw_html.size());
            size_t len = std::min(max_chars * 2, raw_html.size() - start);
            doc.clean_text = raw_html.substr(start, len) + "\n\n... [Source content truncated. Full content cached to .moecher_web_cache.html. Use execute_command with python/grep to inspect.]";
        } else {
            doc.clean_text = raw_html;
        }
        return doc;
    }

    if (mode == "scripts") {
        doc.raw_html = raw_html;
        doc.title = extract_html_title(raw_html);
        doc.clean_text = extract_scripts_from_html(raw_html, pattern, offset, max_chars * 2);
        return doc;
    }

    if (mode == "links") {
        doc.raw_html = raw_html;
        doc.title = extract_html_title(raw_html);
        doc.clean_text = extract_links_from_html(raw_html, doc.url, pattern, offset, max_chars * 2);
        return doc;
    }

    // Check if this is a Search Engine or YouTube search results page
    std::string search_results = extract_search_results_from_html(raw_html, doc.url, 8);
    if (!search_results.empty()) {
        doc.raw_html = raw_html;
        doc.title = extract_html_title(raw_html);
        doc.clean_text = search_results;
        return doc;
    }

    // Check if target requires interactive authentication / CAPTCHA challenge
    std::string lower_html = raw_html;
    std::transform(lower_html.begin(), lower_html.end(), lower_html.begin(), [](unsigned char c){ return (char)::tolower(c); });
    bool is_bot_challenge = (lower_html.find("unusual traffic") != std::string::npos || 
                             lower_html.find("traffico insolito") != std::string::npos ||
                             lower_html.find("enablejs?sei=") != std::string::npos || 
                             lower_html.find("solvesimplechallenge") != std::string::npos ||
                             lower_html.find("knitsail") != std::string::npos ||
                             lower_html.find("emsg=sg_rel") != std::string::npos ||
                             lower_html.find("having trouble accessing google search") != std::string::npos ||
                             (doc.url.find("google.") != std::string::npos && raw_html.size() < 20000 && raw_html.find("rso") == std::string::npos && raw_html.find("MjjYud") == std::string::npos && raw_html.find("<h3") == std::string::npos));

    if (is_bot_challenge) {
        doc.raw_html = raw_html;
        doc.title = "Authentication / Verification Required (" + doc.url + ")";
        doc.clean_text = "[Interactive Verification Required for " + doc.url + "]\n"
                         "Notice: Google or the target site presented a verification challenge / CAPTCHA.\n"
                         "The interactive verification page has been loaded into the client preview panel.\n"
                         "Once completed in the preview, authentication cookies will be captured automatically for subsequent requests.";
        return doc;
    }

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

    // Standard Web Page / Generic Mode (Clean Text extraction, Structured Data, Detected Resource Links)
    doc.raw_html = raw_html;
    std::string body_text = strip_html_to_text(raw_html, max_chars);
    std::string structured_data = extract_json_ld_structured_data(raw_html, max_chars);
    std::string detected_links = extract_detected_resource_links(raw_html, doc.url, 10);

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
    if (!structured_data.empty() && (body_text.size() < 200 || body_text.find(structured_data.substr(0, std::min((size_t)30, structured_data.size()))) == std::string::npos)) {
        if (!combined_text.empty()) combined_text += "\n\n";
        combined_text += structured_data;
    }
    if (!detected_links.empty()) {
        if (!combined_text.empty()) combined_text += "\n";
        combined_text += detected_links;
    }

    if (combined_text.empty()) {
        if (!doc.title.empty()) {
            doc.clean_text = "[Page Loaded: " + doc.title + "]\n[Note: Dynamic JavaScript content. Raw content cached to .moecher_web_cache.html. Use mode='scripts' or mode='links' to inspect.]";
        } else {
            doc.clean_text = "[Web page loaded: " + doc.url + "]\n[Note: Dynamic JavaScript content. Raw content cached to .moecher_web_cache.html. Use mode='scripts' or mode='links' to inspect.]";
        }
    } else {
        doc.clean_text = combined_text;
    }
    return doc;
}

inline std::string fetch_url_content(const std::string& raw_url, int timeout_ms = 10000, size_t max_chars = 4000, const std::string& mode = "text", const std::string& pattern = "", size_t offset = 0) {
    RetrievedDocument doc = fetch_url_full(raw_url, timeout_ms, max_chars, mode, pattern, offset);
    return doc.clean_text;
}

inline std::string escape_shell_arg(const std::string& arg) {
    std::string res = "'";
    for (char c : arg) {
        if (c == '\'') res += "'\\''";
        else res += c;
    }
    res += "'";
    return res;
}

inline std::string http_request_native(
    const std::string& method,
    const std::string& url_str,
    const std::string& body_data = "",
    const std::vector<std::string>& headers = {},
    int timeout_ms = 10000
) {
    std::string resp = "";
#if defined(_WIN32) || defined(_WIN64)
    int wlen = MultiByteToWideChar(CP_UTF8, 0, url_str.c_str(), -1, NULL, 0);
    if (wlen <= 0) return resp;
    std::vector<wchar_t> wurl(wlen);
    MultiByteToWideChar(CP_UTF8, 0, url_str.c_str(), -1, wurl.data(), wlen);

    URL_COMPONENTS url_comp;
    ZeroMemory(&url_comp, sizeof(url_comp));
    url_comp.dwStructSize = sizeof(url_comp);
    url_comp.dwHostNameLength = (DWORD)-1;
    url_comp.dwUrlPathLength = (DWORD)-1;
    url_comp.dwExtraInfoLength = (DWORD)-1;

    if (WinHttpCrackUrl(wurl.data(), (DWORD)wurl.size(), 0, &url_comp)) {
        std::wstring host(url_comp.lpszHostName, url_comp.dwHostNameLength);
        std::wstring path(url_comp.lpszUrlPath, url_comp.dwUrlPathLength);
        if (url_comp.dwExtraInfoLength > 0) path += std::wstring(url_comp.lpszExtraInfo, url_comp.dwExtraInfoLength);

        HINTERNET h_session = WinHttpOpen(L"MoecherSearch/2.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
        if (h_session) {
            WinHttpSetTimeouts(h_session, timeout_ms, timeout_ms, timeout_ms, timeout_ms);
            HINTERNET h_connect = WinHttpConnect(h_session, host.c_str(), url_comp.nPort, 0);
            if (h_connect) {
                std::wstring wmethod = (method == "POST") ? L"POST" : L"GET";
                DWORD flags = (url_comp.nScheme == INTERNET_SCHEME_HTTPS) ? WINHTTP_FLAG_SECURE : 0;
                HINTERNET h_request = WinHttpOpenRequest(h_connect, wmethod.c_str(), path.c_str(), NULL, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, flags);
                if (h_request) {
                    for (const auto& h : headers) {
                        int h_len = MultiByteToWideChar(CP_UTF8, 0, h.c_str(), -1, NULL, 0);
                        if (h_len > 0) {
                            std::vector<wchar_t> wh(h_len);
                            MultiByteToWideChar(CP_UTF8, 0, h.c_str(), -1, wh.data(), h_len);
                            WinHttpAddRequestHeaders(h_request, wh.data(), (ULONG)-1, WINHTTP_ADDREQ_FLAG_ADD | WINHTTP_ADDREQ_FLAG_REPLACE);
                        }
                    }

                    BOOL sent = FALSE;
                    if (method == "POST" && !body_data.empty()) {
                        sent = WinHttpSendRequest(h_request, WINHTTP_NO_ADDITIONAL_HEADERS, 0, (LPVOID)body_data.data(), (DWORD)body_data.size(), (DWORD)body_data.size(), 0);
                    } else {
                        sent = WinHttpSendRequest(h_request, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0);
                    }

                    if (sent && WinHttpReceiveResponse(h_request, NULL)) {
                        DWORD bytes_avail = 0;
                        while (WinHttpQueryDataAvailable(h_request, &bytes_avail) && bytes_avail > 0) {
                            std::vector<char> buf(bytes_avail);
                            DWORD bytes_read = 0;
                            if (WinHttpReadData(h_request, buf.data(), bytes_avail, &bytes_read) && bytes_read > 0) {
                                resp.append(buf.data(), bytes_read);
                            } else break;
                        }
                    }
                    WinHttpCloseHandle(h_request);
                }
                WinHttpCloseHandle(h_connect);
            }
            WinHttpCloseHandle(h_session);
        }
    }
#else
    std::string header_flags = "";
    for (const auto& h : headers) {
        header_flags += " -H \"" + h + "\"";
    }
    std::string curl_cmd;
    if (method == "POST") {
        curl_cmd = "curl -sL -X POST " + header_flags + " -d " + escape_shell_arg(body_data) + " --max-time 10 \"" + url_str + "\"";
    } else {
        curl_cmd = "curl -sL " + header_flags + " --max-time 10 \"" + url_str + "\"";
    }
    resp = execute_system_command(curl_cmd, 10000, 2000000);
#endif
    return resp;
}

// ════════════════════════════════════════════════════════════════════════════════
//  Search Provider Handlers (Tavily, Brave, SearXNG, Serper, Google)
// ════════════════════════════════════════════════════════════════════════════════

inline RetrievedDocument search_tavily(const std::string& clean_q, int num_results, const std::string& api_key) {
    RetrievedDocument doc;
    doc.url = "https://tavily.com";
    doc.title = "Tavily Search: " + clean_q;

    if (api_key.empty()) {
        doc.clean_text = "[Tavily Search API Key Not Configured]\n\n"
                         "Tavily provides 1,000 free web searches monthly for AI agents without requiring a credit card.\n"
                         "1. Get your free key at: https://tavily.com\n"
                         "2. Save your API key in the Web UI Settings panel (or set TAVILY_API_KEY).\n"
                         "Tip: You can also switch to 'SearXNG' in Settings to search with 0 API keys.";
        doc.raw_html = "<div style=\"padding:20px; font-family:sans-serif; color:#e2e8f0; background:#1e293b; border-radius:8px;\">"
                       "<h3>Tavily API Key Not Configured</h3>"
                       "<p>Get 1,000 free monthly searches at <a href=\"https://tavily.com\" target=\"_blank\" style=\"color:#60a5fa;\">tavily.com</a> (No credit card required).</p>"
                       "</div>";
        return doc;
    }

    json req_body = {
        {"api_key", api_key},
        {"query", clean_q},
        {"max_results", num_results},
        {"search_depth", "basic"},
        {"include_answer", true}
    };

    std::string resp = http_request_native("POST", "https://api.tavily.com/search", req_body.dump(), {"Content-Type: application/json"});
    if (resp.empty()) {
        doc.clean_text = "[Error: No response received from Tavily API]";
        return doc;
    }

    try {
        json j = json::parse(resp);
        if (j.contains("error")) {
            doc.clean_text = "[Tavily Error: " + j["error"].dump() + "]";
            return doc;
        }

        std::string direct_answer = j.value("answer", "");
        std::string text_out = "[Web Search Results (Tavily AI): \"" + clean_q + "\"]\n\n";
        if (!direct_answer.empty()) {
            text_out += "Direct AI Summary: " + direct_answer + "\n\n";
        }

        std::string html_cards = "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; color:#e2e8f0; background:#0f172a; padding:20px; border-radius:12px; max-width:860px; margin:0 auto;\">";
        html_cards += "<h2 style=\"margin:0 0 16px 0; font-size:18px; color:#60a5fa;\">Tavily AI Search: " + clean_q + "</h2>";
        if (!direct_answer.empty()) {
            html_cards += "<div style=\"background:rgba(59,130,246,0.15); border:1px solid rgba(59,130,246,0.3); border-radius:8px; padding:12px 14px; margin-bottom:16px; font-size:14px; color:#93c5fd;\"><strong>AI Summary:</strong> " + direct_answer + "</div>";
        }

        int count = 0;
        if (j.contains("results") && j["results"].is_array()) {
            for (const auto& item : j["results"]) {
                count++;
                std::string item_title = item.value("title", "");
                std::string item_link = item.value("url", "");
                std::string item_content = item.value("content", "");

                if (count == 1 && !item_link.empty()) {
                    doc.url = item_link;
                    doc.title = item_title;
                }

                text_out += std::to_string(count) + ". **" + item_title + "**\n";
                text_out += "   URL: " + item_link + "\n";
                text_out += "   Snippet: " + item_content + "\n\n";

                html_cards += "<div style=\"background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:14px 16px; margin-bottom:12px;\">";
                html_cards += "<a href=\"" + item_link + "\" target=\"_blank\" style=\"font-size:16px; font-weight:600; color:#93c5fd; text-decoration:none; display:inline-block; margin-bottom:6px;\">" + item_title + " &rarr;</a>";
                html_cards += "<p style=\"font-size:13px; color:#cbd5e1; margin:0; line-height:1.5;\">" + item_content + "</p>";
                html_cards += "</div>";
                if (count >= num_results) break;
            }
        }
        html_cards += "</div>";

        if (count == 0) {
            text_out += "No search results found for query: \"" + clean_q + "\"";
        }

        doc.clean_text = text_out;
        doc.raw_html = html_cards;
        return doc;
    } catch (const std::exception& e) {
        doc.clean_text = "[Error parsing Tavily response: " + std::string(e.what()) + "]";
        return doc;
    }
}

inline RetrievedDocument search_brave(const std::string& clean_q, int num_results, const std::string& api_key) {
    RetrievedDocument doc;
    doc.url = "https://search.brave.com/search?q=" + url_encode(clean_q);
    doc.title = "Brave Search: " + clean_q;

    if (api_key.empty()) {
        doc.clean_text = "[Brave Search API Key Not Configured]\n\n"
                         "Brave Search offers $5/month in free API credits (~1,000 searches).\n"
                         "1. Get your API key at: https://brave.com/search/api/\n"
                         "2. Save your API key in Settings (or set BRAVE_API_KEY).";
        doc.raw_html = "<div style=\"padding:20px; font-family:sans-serif; color:#e2e8f0; background:#1e293b; border-radius:8px;\">"
                       "<h3>Brave Search API Key Not Configured</h3>"
                       "<p>Configure your key at <a href=\"https://brave.com/search/api/\" target=\"_blank\" style=\"color:#60a5fa;\">brave.com/search/api</a>.</p>"
                       "</div>";
        return doc;
    }

    std::string api_url = "https://api.brave.com/res/v1/web/search?q=" + url_encode(clean_q) + "&count=" + std::to_string(num_results);
    std::string resp = http_request_native("GET", api_url, "", {"Accept: application/json", "X-Subscription-Token: " + api_key});

    if (resp.empty()) {
        doc.clean_text = "[Error: No response received from Brave Search API]";
        return doc;
    }

    try {
        json j = json::parse(resp);
        std::string text_out = "[Web Search Results (Brave): \"" + clean_q + "\"]\n\n";
        std::string html_cards = "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; color:#e2e8f0; background:#0f172a; padding:20px; border-radius:12px; max-width:860px; margin:0 auto;\">";
        html_cards += "<h2 style=\"margin:0 0 16px 0; font-size:18px; color:#60a5fa;\">Brave Search: " + clean_q + "</h2>";

        int count = 0;
        if (j.contains("web") && j["web"].is_object() && j["web"].contains("results") && j["web"]["results"].is_array()) {
            for (const auto& item : j["web"]["results"]) {
                count++;
                std::string item_title = item.value("title", "");
                std::string item_link = item.value("url", "");
                std::string item_desc = item.value("description", "");

                text_out += std::to_string(count) + ". **" + item_title + "**\n";
                text_out += "   URL: " + item_link + "\n";
                text_out += "   Snippet: " + item_desc + "\n\n";

                html_cards += "<div style=\"background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:14px 16px; margin-bottom:12px;\">";
                html_cards += "<a href=\"" + item_link + "\" target=\"_blank\" style=\"font-size:16px; font-weight:600; color:#93c5fd; text-decoration:none; display:inline-block; margin-bottom:6px;\">" + item_title + " &rarr;</a>";
                html_cards += "<p style=\"font-size:13px; color:#cbd5e1; margin:0; line-height:1.5;\">" + item_desc + "</p>";
                html_cards += "</div>";
                if (count >= num_results) break;
            }
        }
        html_cards += "</div>";

        if (count == 0) {
            text_out += "No search results found for query: \"" + clean_q + "\"";
        }

        doc.clean_text = text_out;
        doc.raw_html = html_cards;
        return doc;
    } catch (const std::exception& e) {
        doc.clean_text = "[Error parsing Brave response: " + std::string(e.what()) + "]";
        return doc;
    }
}

inline RetrievedDocument search_searxng(const std::string& clean_q, int num_results, const std::string& custom_instance_url) {
    RetrievedDocument doc;
    std::string base_url = !custom_instance_url.empty() ? custom_instance_url : (!g_searxng_url.empty() ? g_searxng_url : "https://searx.be");
    while (!base_url.empty() && base_url.back() == '/') base_url.pop_back();

    doc.url = base_url + "/search?q=" + url_encode(clean_q);
    doc.title = "SearXNG Search: " + clean_q;

    std::vector<std::string> candidate_instances = {
        base_url,
        "https://sx.xo.st",
        "https://searx.be",
        "https://searx.tiekoetter.com"
    };

    std::string resp = "";
    for (const auto& inst : candidate_instances) {
        if (inst.empty()) continue;
        std::string api_url = inst + "/search?q=" + url_encode(clean_q) + "&format=json&categories=general";
        std::string candidate_resp = http_request_native("GET", api_url, "", {"Accept: application/json", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}, 5000);
        if (!candidate_resp.empty() && candidate_resp.find("\"results\"") != std::string::npos) {
            resp = std::move(candidate_resp);
            break;
        }
    }

    if (resp.empty()) {
        doc.clean_text = "[SearXNG Search: No public instance response received. Please configure a custom instance (e.g. local Docker http://localhost:8080) or switch to Tavily AI (1,000 free queries/mo) in Settings.]";
        return doc;
    }

    try {
        json j = json::parse(resp);
        std::string text_out = "[Web Search Results (SearXNG Metasearch): \"" + clean_q + "\"]\n\n";
        std::string html_cards = "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; color:#e2e8f0; background:#0f172a; padding:20px; border-radius:12px; max-width:860px; margin:0 auto;\">";
        html_cards += "<h2 style=\"margin:0 0 16px 0; font-size:18px; color:#60a5fa;\">SearXNG Metasearch: " + clean_q + "</h2>";

        int count = 0;
        if (j.contains("results") && j["results"].is_array()) {
            for (const auto& item : j["results"]) {
                count++;
                std::string item_title = item.value("title", "");
                std::string item_link = item.value("url", "");
                std::string item_content = item.value("content", "");

                text_out += std::to_string(count) + ". **" + item_title + "**\n";
                text_out += "   URL: " + item_link + "\n";
                text_out += "   Snippet: " + item_content + "\n\n";

                html_cards += "<div style=\"background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:14px 16px; margin-bottom:12px;\">";
                html_cards += "<a href=\"" + item_link + "\" target=\"_blank\" style=\"font-size:16px; font-weight:600; color:#93c5fd; text-decoration:none; display:inline-block; margin-bottom:6px;\">" + item_title + " &rarr;</a>";
                html_cards += "<p style=\"font-size:13px; color:#cbd5e1; margin:0; line-height:1.5;\">" + item_content + "</p>";
                html_cards += "</div>";
                if (count >= num_results) break;
            }
        }
        html_cards += "</div>";

        if (count == 0) {
            text_out += "No search results found for query: \"" + clean_q + "\"";
        }

        doc.clean_text = text_out;
        doc.raw_html = html_cards;
        return doc;
    } catch (const std::exception& e) {
        doc.clean_text = "[Error parsing SearXNG response: " + std::string(e.what()) + "]";
        return doc;
    }
}

inline RetrievedDocument search_serper(const std::string& clean_q, int num_results, const std::string& api_key) {
    RetrievedDocument doc;
    doc.url = "https://google.com/search?q=" + url_encode(clean_q);
    doc.title = "Serper (Google Search): " + clean_q;

    if (api_key.empty()) {
        doc.clean_text = "[Serper.dev API Key Not Configured]\n\n"
                         "Serper.dev provides 2,500 free Google searches on signup.\n"
                         "1. Get your API key at: https://serper.dev\n"
                         "2. Save your API key in Settings (or set SERPER_API_KEY).";
        doc.raw_html = "<div style=\"padding:20px; font-family:sans-serif; color:#e2e8f0; background:#1e293b; border-radius:8px;\">"
                       "<h3>Serper.dev API Key Not Configured</h3>"
                       "<p>Get 2,500 free searches at <a href=\"https://serper.dev\" target=\"_blank\" style=\"color:#60a5fa;\">serper.dev</a>.</p>"
                       "</div>";
        return doc;
    }

    json req_body = {
        {"q", clean_q},
        {"num", num_results}
    };

    std::string resp = http_request_native("POST", "https://google.serper.dev/search", req_body.dump(), {"Content-Type: application/json", "X-API-KEY: " + api_key});
    if (resp.empty()) {
        doc.clean_text = "[Error: No response received from Serper API]";
        return doc;
    }

    try {
        json j = json::parse(resp);
        std::string text_out = "[Google Search Results (Serper.dev): \"" + clean_q + "\"]\n\n";
        std::string html_cards = "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; color:#e2e8f0; background:#0f172a; padding:20px; border-radius:12px; max-width:860px; margin:0 auto;\">";
        html_cards += "<h2 style=\"margin:0 0 16px 0; font-size:18px; color:#60a5fa;\">Google Search: " + clean_q + "</h2>";

        int count = 0;
        if (j.contains("organic") && j["organic"].is_array()) {
            for (const auto& item : j["organic"]) {
                count++;
                std::string item_title = item.value("title", "");
                std::string item_link = item.value("link", "");
                std::string item_snippet = item.value("snippet", "");

                text_out += std::to_string(count) + ". **" + item_title + "**\n";
                text_out += "   URL: " + item_link + "\n";
                text_out += "   Snippet: " + item_snippet + "\n\n";

                html_cards += "<div style=\"background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:14px 16px; margin-bottom:12px;\">";
                html_cards += "<a href=\"" + item_link + "\" target=\"_blank\" style=\"font-size:16px; font-weight:600; color:#93c5fd; text-decoration:none; display:inline-block; margin-bottom:6px;\">" + item_title + " &rarr;</a>";
                html_cards += "<p style=\"font-size:13px; color:#cbd5e1; margin:0; line-height:1.5;\">" + item_snippet + "</p>";
                html_cards += "</div>";
                if (count >= num_results) break;
            }
        }
        html_cards += "</div>";

        if (count == 0) {
            text_out += "No search results found for query: \"" + clean_q + "\"";
        }

        doc.clean_text = text_out;
        doc.raw_html = html_cards;
        return doc;
    } catch (const std::exception& e) {
        doc.clean_text = "[Error parsing Serper response: " + std::string(e.what()) + "]";
        return doc;
    }
}

inline RetrievedDocument search_google(const std::string& clean_q, int num_results, const std::string& custom_key, const std::string& custom_cx) {
    RetrievedDocument doc;
    std::string api_key = !custom_key.empty() ? custom_key : g_google_search_api_key;
    std::string cx = !custom_cx.empty() ? custom_cx : g_google_search_cx;

    doc.url = "https://www.google.com/search?q=" + url_encode(clean_q);
    doc.title = "Google Search: " + clean_q;

    if (api_key.empty() || cx.empty()) {
        doc.clean_text = "[Google Custom Search API Not Configured]\n\n"
                         "1. Enter Google API Key and Search Engine ID (CX) in Settings.\n"
                         "Tip: We recommend selecting 'Tavily' or 'SearXNG' in Settings for free, immediate web searching.";
        doc.raw_html = "<div style=\"padding:20px; font-family:sans-serif; color:#e2e8f0; background:#1e293b; border-radius:8px;\">"
                       "<h3>Google Custom Search API Not Configured</h3>"
                       "<p>Tip: Switch to <strong>Tavily</strong> (1,000 free queries/mo) or <strong>SearXNG</strong> (100% Free, No key needed) in Settings.</p>"
                       "</div>";
        return doc;
    }

    std::string api_url = "https://www.googleapis.com/customsearch/v1?key=" + url_encode(api_key) +
                          "&cx=" + url_encode(cx) +
                          "&q=" + url_encode(clean_q) +
                          "&num=" + std::to_string(num_results);

    std::string json_resp = http_request_native("GET", api_url);
    if (json_resp.empty()) {
        doc.clean_text = "[Error: No response received from Google Custom Search API]";
        return doc;
    }

    try {
        json j = json::parse(json_resp);
        if (j.contains("error")) {
            std::string err_msg = j["error"].is_object() ? j["error"].value("message", "Unknown error") : j["error"].dump();
            doc.clean_text = "[Google Search API Error: " + err_msg + "]";
            return doc;
        }

        std::string text_out = "[Google Search Results for: \"" + clean_q + "\"]\n\n";
        std::string html_cards = "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; color:#e2e8f0; background:#0f172a; padding:20px; border-radius:12px; max-width:860px; margin:0 auto;\">";
        html_cards += "<h2 style=\"margin:0 0 16px 0; font-size:18px; color:#60a5fa;\">Google Search: " + clean_q + "</h2>";

        int count = 0;
        if (j.contains("items") && j["items"].is_array()) {
            for (const auto& item : j["items"]) {
                count++;
                std::string item_title = item.value("title", "");
                std::string item_link = item.value("link", "");
                std::string item_snippet = item.value("snippet", "");

                text_out += std::to_string(count) + ". **" + item_title + "**\n";
                text_out += "   URL: " + item_link + "\n";
                text_out += "   Snippet: " + item_snippet + "\n\n";

                html_cards += "<div style=\"background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:14px 16px; margin-bottom:12px;\">";
                html_cards += "<a href=\"" + item_link + "\" target=\"_blank\" style=\"font-size:16px; font-weight:600; color:#93c5fd; text-decoration:none; display:inline-block; margin-bottom:6px;\">" + item_title + " &rarr;</a>";
                html_cards += "<p style=\"font-size:13px; color:#cbd5e1; margin:0; line-height:1.5;\">" + item_snippet + "</p>";
                html_cards += "</div>";
                if (count >= num_results) break;
            }
        }
        html_cards += "</div>";

        if (count == 0) {
            text_out += "No search results found for query: \"" + clean_q + "\"";
        }

        doc.clean_text = text_out;
        doc.raw_html = html_cards;
        return doc;
    } catch (const std::exception& e) {
        doc.clean_text = "[Error parsing Google Search API response: " + std::string(e.what()) + "]";
        return doc;
    }
}

// ════════════════════════════════════════════════════════════════════════════════
//  Universal Web Search Dispatcher
// ════════════════════════════════════════════════════════════════════════════════

inline RetrievedDocument web_search_full(
    const std::string& query,
    int num_results = 5,
    const std::string& site = "",
    const std::string& custom_provider = "",
    const std::string& custom_key = "",
    const std::string& custom_cx = "",
    const std::string& custom_searx_url = ""
) {
    load_config_from_disk();
    if (num_results < 1) num_results = 1;
    if (num_results > 10) num_results = 10;

    std::string clean_q = query;
    size_t f = clean_q.find_first_not_of(" \t\r\n");
    size_t l = clean_q.find_last_not_of(" \t\r\n");
    if (f != std::string::npos && l != std::string::npos) clean_q = clean_q.substr(f, l - f + 1);
    if (!site.empty()) clean_q += " site:" + site;

    std::string provider = !custom_provider.empty() ? custom_provider : g_search_provider;
    if (provider.empty()) provider = "tavily";

    if (provider == "tavily") {
        std::string key = !custom_key.empty() ? custom_key : g_tavily_api_key;
        return search_tavily(clean_q, num_results, key);
    } else if (provider == "brave") {
        std::string key = !custom_key.empty() ? custom_key : g_brave_api_key;
        return search_brave(clean_q, num_results, key);
    } else if (provider == "searxng" || provider == "searx") {
        std::string s_url = !custom_searx_url.empty() ? custom_searx_url : g_searxng_url;
        return search_searxng(clean_q, num_results, s_url);
    } else if (provider == "serper") {
        std::string key = !custom_key.empty() ? custom_key : g_serper_api_key;
        return search_serper(clean_q, num_results, key);
    } else if (provider == "google") {
        return search_google(clean_q, num_results, custom_key, custom_cx);
    }

    // Default fallback to Tavily
    return search_tavily(clean_q, num_results, g_tavily_api_key);
}

inline std::string web_search_content(
    const std::string& query,
    int num_results = 5,
    const std::string& site = "",
    const std::string& provider = "",
    const std::string& key = "",
    const std::string& cx = "",
    const std::string& searx_url = ""
) {
    RetrievedDocument doc = web_search_full(query, num_results, site, provider, key, cx, searx_url);
    return doc.clean_text;
}

// Backwards-compatible aliases
inline RetrievedDocument google_search_full(
    const std::string& query,
    int num_results = 5,
    const std::string& site = "",
    const std::string& custom_key = "",
    const std::string& custom_cx = ""
) {
    return web_search_full(query, num_results, site, "google", custom_key, custom_cx);
}

inline std::string google_search_content(
    const std::string& query,
    int num_results = 5,
    const std::string& site = "",
    const std::string& api_key = "",
    const std::string& cx = ""
) {
    return web_search_content(query, num_results, site, "google", api_key, cx);
}

// ════════════════════════════════════════════════════════════════════════════════
//  Built-in Tool Schemas & Definitions
// ════════════════════════════════════════════════════════════════════════════════

inline std::string get_builtin_tools_json() {
    return R"JSON([
  {
    "type": "function",
    "function": {
      "name": "web_search",
      "description": "Search the web to find relevant web pages, articles, documentation, research, and current information across the Internet using the active search provider (Tavily, Brave, SearXNG, Serper, or Google).",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "The search query string (e.g. 'latest quantum computing breakthroughs', 'Tino Bruno GitHub', 'llama.cpp documentation')."
          },
          "num_results": {
            "type": "integer",
            "description": "Optional number of search results to return (1-10, default: 5)."
          },
          "site": {
            "type": "string",
            "description": "Optional domain filter to restrict search to a specific website (e.g. 'github.com', 'wikipedia.org', 'arxiv.org')."
          }
        },
        "required": ["query"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "google_search",
      "description": "Search the web to find relevant web pages, articles, documentation, research, and current information (alias for web_search).",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "The search query string (e.g. 'latest quantum computing breakthroughs', 'Tino Bruno GitHub', 'llama.cpp documentation')."
          },
          "num_results": {
            "type": "integer",
            "description": "Optional number of search results to return (1-10, default: 5)."
          },
          "site": {
            "type": "string",
            "description": "Optional domain filter to restrict search to a specific website (e.g. 'github.com', 'wikipedia.org', 'arxiv.org')."
          }
        },
        "required": ["query"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "fetch_url",
      "description": "Fetch web content, inspect raw source code, or extract JavaScript/links from a public web URL (HTTP/HTTPS). Raw responses are cached to '.moecher_web_cache.html' for dynamic script/JSON parsing.",
      "parameters": {
        "type": "object",
        "properties": {
          "url": {
            "type": "string",
            "description": "The complete HTTP or HTTPS URL or search query to fetch (e.g. https://www.youtube.com/results?search_query=topic, https://www.google.com/search?q=quantum+computing, https://en.wikipedia.org/wiki/Special:Search?search=Topic, or https://raw.githubusercontent.com/...)"
          },
          "mode": {
            "type": "string",
            "enum": ["text", "raw", "scripts", "links"],
            "description": "Optional extraction mode: 'text' (default: clean readable article text, metadata, detected links), 'raw' (unmodified raw HTML/source code), 'scripts' (extracts inline JavaScript, script tags, and JSON-LD data), or 'links' (extracts all hyperlinks)."
          },
          "pattern": {
            "type": "string",
            "description": "Optional case-insensitive substring or keyword filter. In 'raw' mode, extracts surrounding context windows around matches; in 'scripts' mode, extracts scripts containing this keyword; in 'links' mode, extracts URLs/anchors containing this keyword."
          },
          "offset": {
            "type": "integer",
            "description": "Optional pagination offset for matches/scripts/links (default: 0)."
          },
          "max_chars": {
            "type": "integer",
            "description": "Optional maximum character length for the output (default: 4000)."
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
