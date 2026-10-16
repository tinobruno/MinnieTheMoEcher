#include "agentic.hpp"
#include "logging.hpp"
#include "tool_exec.hpp"
#include "mcp_client.hpp"

#include <httplib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <atomic>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <cctype>

#if defined(_WIN32) || defined(_WIN64)
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #include <winhttp.h>
    #include <wininet.h>
    #pragma comment(lib, "ws2_32.lib")
    #pragma comment(lib, "winhttp.lib")
    #pragma comment(lib, "wininet.lib")
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <netdb.h>
    #include <unistd.h>
    #include <arpa/inet.h>
    #include <sys/select.h>
    #include <sys/time.h>
#endif

using json = nlohmann::json;

// ════════════════════════════════════════════════════════════════════════════════
//  Forward Proxy Engine (HTTP & HTTPS CONNECT Tunneling)
// ════════════════════════════════════════════════════════════════════════════════

namespace moecher::proxy {

#ifndef SOCKET
#define SOCKET int
#endif
#ifndef INVALID_SOCKET
#define INVALID_SOCKET (-1)
#endif
#ifndef SOCKET_ERROR
#define SOCKET_ERROR (-1)
#endif
#ifndef closesocket
#define closesocket close
#endif

std::atomic<bool> g_proxy_running{false};
std::atomic<int> g_proxy_port{8002};
static SOCKET g_proxy_listen_sock = INVALID_SOCKET;

// Helper to reliably send all bytes over socket
static bool send_all(SOCKET s, const char* data, int len) {
    int total = 0;
    while (total < len) {
        int n = send(s, data + total, len - total, 0);
        if (n <= 0) return false;
        total += n;
    }
    return true;
}

// Forward Proxy Client Handler
static void handle_client_socket(SOCKET client_sock) {
#ifdef _WIN32
    DWORD timeout_ms = 15000;
    setsockopt(client_sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout_ms, sizeof(timeout_ms));
    setsockopt(client_sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout_ms, sizeof(timeout_ms));
#else
    struct timeval tv_to{15, 0};
    setsockopt(client_sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv_to, sizeof(tv_to));
    setsockopt(client_sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv_to, sizeof(tv_to));
#endif

    std::vector<char> req_vec(16384);
    int bytes_read = recv(client_sock, req_vec.data(), (int)req_vec.size() - 1, 0);
    if (bytes_read <= 0) {
        closesocket(client_sock);
        return;
    }
    req_vec[bytes_read] = '\0';
    std::string req_str(req_vec.data(), bytes_read);

    size_t line_end = req_str.find("\r\n");
    if (line_end == std::string::npos) {
        closesocket(client_sock);
        return;
    }

    std::string first_line = req_str.substr(0, line_end);
    std::istringstream iss(first_line);
    std::string method, target, version;
    iss >> method >> target >> version;

    if (method.empty() || target.empty()) {
        closesocket(client_sock);
        return;
    }

    std::transform(method.begin(), method.end(), method.begin(), ::toupper);

    std::cout << "[PROXY " << g_proxy_port.load() << "] " << method << " " << target << std::endl;

    if (method == "OPTIONS") {
        const char* preflight_resp = "HTTP/1.1 200 OK\r\n"
                                      "Access-Control-Allow-Origin: *\r\n"
                                      "Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS, HEAD\r\n"
                                      "Access-Control-Allow-Headers: *\r\n"
                                      "Access-Control-Expose-Headers: *\r\n"
                                      "Access-Control-Max-Age: 86400\r\n"
                                      "Content-Length: 0\r\n\r\n";
        send_all(client_sock, preflight_resp, (int)strlen(preflight_resp));
        closesocket(client_sock);
        return;
    }

    if (method == "CONNECT") {
        // HTTPS Tunneling: CONNECT host:port HTTP/1.1
        std::string host;
        int port = 443;
        size_t colon = target.find(':');
        if (colon != std::string::npos) {
            host = target.substr(0, colon);
            try { port = std::stoi(target.substr(colon + 1)); } catch (...) { port = 443; }
        } else {
            host = target;
        }

        struct addrinfo hints{}, *res = nullptr;
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_protocol = IPPROTO_TCP;

        if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res) {
            const char* err_resp = "HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n";
            send(client_sock, err_resp, (int)strlen(err_resp), 0);
            closesocket(client_sock);
            return;
        }

        SOCKET remote_sock = INVALID_SOCKET;
        for (struct addrinfo* ptr = res; ptr != nullptr; ptr = ptr->ai_next) {
            remote_sock = socket(ptr->ai_family, ptr->ai_socktype, ptr->ai_protocol);
            if (remote_sock == INVALID_SOCKET) continue;

#ifdef _WIN32
            DWORD r_timeout = 10000;
            setsockopt(remote_sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&r_timeout, sizeof(r_timeout));
            setsockopt(remote_sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&r_timeout, sizeof(r_timeout));
#else
            struct timeval r_tv{10, 0};
            setsockopt(remote_sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&r_tv, sizeof(r_tv));
            setsockopt(remote_sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&r_tv, sizeof(r_tv));
#endif

            if (connect(remote_sock, ptr->ai_addr, (int)ptr->ai_addrlen) == 0) {
                break;
            }
            closesocket(remote_sock);
            remote_sock = INVALID_SOCKET;
        }
        freeaddrinfo(res);

        if (remote_sock == INVALID_SOCKET) {
            const char* err_resp = "HTTP/1.1 504 Gateway Timeout\r\nContent-Length: 0\r\n\r\n";
            send(client_sock, err_resp, (int)strlen(err_resp), 0);
            closesocket(client_sock);
            return;
        }

        const char* ok_resp = "HTTP/1.1 200 Connection Established\r\n\r\n";
        if (!send_all(client_sock, ok_resp, (int)strlen(ok_resp))) {
            closesocket(remote_sock);
            closesocket(client_sock);
            return;
        }

        char relay_buf[16384];
        while (g_proxy_running.load()) {
            fd_set read_fds;
            FD_ZERO(&read_fds);
            FD_SET(client_sock, &read_fds);
            FD_SET(remote_sock, &read_fds);

            struct timeval tv;
            tv.tv_sec = 60;
            tv.tv_usec = 0;

            int max_fd = (int)std::max(client_sock, remote_sock);
            int activity = select(max_fd + 1, &read_fds, nullptr, nullptr, &tv);
            if (activity <= 0) break;

            if (FD_ISSET(client_sock, &read_fds)) {
                int n = recv(client_sock, relay_buf, sizeof(relay_buf), 0);
                if (n <= 0) break;
                if (!send_all(remote_sock, relay_buf, n)) break;
            }

            if (FD_ISSET(remote_sock, &read_fds)) {
                int n = recv(remote_sock, relay_buf, sizeof(relay_buf), 0);
                if (n <= 0) break;
                if (!send_all(client_sock, relay_buf, n)) break;
            }
        }

        closesocket(remote_sock);
        closesocket(client_sock);
        return;
    }

    // Forward HTTP Request (GET, POST, HEAD, PUT, DELETE)
    std::string target_url = target;
    std::string host_header = "";

    std::unordered_map<std::string, std::string> req_headers;
    size_t hdr_sep = req_str.find("\r\n\r\n");
    std::string raw_hdrs = (hdr_sep != std::string::npos) ? req_str.substr(0, hdr_sep) : req_str;
    std::string req_body = (hdr_sep != std::string::npos && hdr_sep + 4 < req_str.size()) ? req_str.substr(hdr_sep + 4) : "";

    std::istringstream hdr_stream(raw_hdrs);
    std::string hline;
    std::getline(hdr_stream, hline); // Skip first line
    while (std::getline(hdr_stream, hline)) {
        if (!hline.empty() && hline.back() == '\r') hline.pop_back();
        size_t cpos = hline.find(':');
        if (cpos != std::string::npos) {
            std::string hname = hline.substr(0, cpos);
            std::string hval = hline.substr(cpos + 1);
            while (!hval.empty() && hval.front() == ' ') hval.erase(0, 1);
            std::string hname_lower = hname;
            std::transform(hname_lower.begin(), hname_lower.end(), hname_lower.begin(), ::tolower);
            req_headers[hname_lower] = hval;
        }
    }

    if (req_headers.count("host")) {
        host_header = req_headers["host"];
    }

    if (target_url.rfind("/fetch?url=", 0) == 0) {
        target_url = moecher::tooling::url_decode(target_url.substr(11));
    } else if (target_url.rfind("/proxy?url=", 0) == 0) {
        target_url = moecher::tooling::url_decode(target_url.substr(11));
    } else if (target_url.rfind("/api/proxy?url=", 0) == 0) {
        target_url = moecher::tooling::url_decode(target_url.substr(15));
    } else if (target_url.rfind("/https://", 0) == 0) {
        target_url = target_url.substr(1);
    } else if (target_url.rfind("/http://", 0) == 0) {
        target_url = target_url.substr(1);
    } else if (target_url.rfind("http://", 0) != 0 && target_url.rfind("https://", 0) != 0) {
        if (!host_header.empty()) {
            target_url = "http://" + host_header + (target_url.front() == '/' ? target_url : ("/" + target_url));
        }
    }

    if (target_url.empty()) {
        closesocket(client_sock);
        return;
    }

    // Upgrade http to https for upstream fetch to bypass redirects and use TLS securely
    std::string upstream_url = target_url;
    if (upstream_url.rfind("http://", 0) == 0 &&
        upstream_url.find("localhost") == std::string::npos &&
        upstream_url.find("127.0.0.1") == std::string::npos) {
        upstream_url = "https://" + upstream_url.substr(7);
    }

#if defined(_WIN32) || defined(_WIN64)
    int wlen = MultiByteToWideChar(CP_UTF8, 0, upstream_url.c_str(), -1, NULL, 0);
    std::vector<wchar_t> wurl(wlen);
    MultiByteToWideChar(CP_UTF8, 0, upstream_url.c_str(), -1, wurl.data(), wlen);

    URL_COMPONENTS url_comp;
    ZeroMemory(&url_comp, sizeof(url_comp));
    url_comp.dwStructSize = sizeof(url_comp);
    url_comp.dwHostNameLength = (DWORD)-1;
    url_comp.dwUrlPathLength = (DWORD)-1;
    url_comp.dwExtraInfoLength = (DWORD)-1;

    if (!WinHttpCrackUrl(wurl.data(), (DWORD)wurl.size(), 0, &url_comp)) {
        const char* err_resp = "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n";
        send(client_sock, err_resp, (int)strlen(err_resp), 0);
        closesocket(client_sock);
        return;
    }

    std::wstring w_host(url_comp.lpszHostName, url_comp.dwHostNameLength);
    std::wstring w_path(url_comp.lpszUrlPath, url_comp.dwUrlPathLength);
    if (url_comp.dwExtraInfoLength > 0) {
        w_path += std::wstring(url_comp.lpszExtraInfo, url_comp.dwExtraInfoLength);
    }
    if (w_path.empty()) w_path = L"/";

    bool is_https = (url_comp.nScheme == INTERNET_SCHEME_HTTPS);
    INTERNET_PORT port = url_comp.nPort;

    std::string ua = req_headers.count("user-agent") ? req_headers["user-agent"] : moecher::tooling::get_client_user_agent();
    std::string lang = req_headers.count("accept-language") ? req_headers["accept-language"] : moecher::tooling::get_client_accept_language();
    int ua_wlen = MultiByteToWideChar(CP_UTF8, 0, ua.c_str(), -1, NULL, 0);
    std::vector<wchar_t> w_ua(ua_wlen);
    MultiByteToWideChar(CP_UTF8, 0, ua.c_str(), -1, w_ua.data(), ua_wlen);

    HINTERNET h_session = WinHttpOpen(w_ua.empty() ? L"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" : w_ua.data(),
                                      WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                      WINHTTP_NO_PROXY_NAME,
                                      WINHTTP_NO_PROXY_BYPASS, 0);
    if (!h_session) {
        const char* err_resp = "HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n";
        send(client_sock, err_resp, (int)strlen(err_resp), 0);
        closesocket(client_sock);
        return;
    }

    DWORD timeout_val = 15000;
    WinHttpSetTimeouts(h_session, timeout_val, timeout_val, timeout_val, timeout_val);

    HINTERNET h_connect = WinHttpConnect(h_session, w_host.c_str(), port, 0);
    if (!h_connect) {
        WinHttpCloseHandle(h_session);
        const char* err_resp = "HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n";
        send(client_sock, err_resp, (int)strlen(err_resp), 0);
        closesocket(client_sock);
        return;
    }

    DWORD open_flags = is_https ? WINHTTP_FLAG_SECURE : 0;
    std::wstring w_method = L"GET";
    if (method == "POST") w_method = L"POST";
    else if (method == "HEAD") w_method = L"HEAD";
    else if (method == "PUT") w_method = L"PUT";
    else if (method == "DELETE") w_method = L"DELETE";

    HINTERNET h_request = WinHttpOpenRequest(h_connect, w_method.c_str(), w_path.c_str(),
                                            NULL, WINHTTP_NO_REFERER,
                                            WINHTTP_DEFAULT_ACCEPT_TYPES, open_flags);
    if (!h_request) {
        WinHttpCloseHandle(h_connect);
        WinHttpCloseHandle(h_session);
        const char* err_resp = "HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n";
        send(client_sock, err_resp, (int)strlen(err_resp), 0);
        closesocket(client_sock);
        return;
    }

    DWORD opt_redirect = WINHTTP_OPTION_REDIRECT_POLICY_ALWAYS;
    WinHttpSetOption(h_request, WINHTTP_OPTION_REDIRECT_POLICY, &opt_redirect, sizeof(opt_redirect));

    // Combine request headers
    std::string cookies = req_headers.count("cookie") ? req_headers["cookie"] : moecher::tooling::get_cookies_for_url(upstream_url);
    if (upstream_url.find("google.") != std::string::npos) {
        if (cookies.find("SOCS=") == std::string::npos) {
            if (!cookies.empty()) cookies += "; ";
            cookies += "SOCS=CAESHAgBEhJnd3NfMjAyNDA4MjAtMF9SQzIaAmVuIAEaBgiA_L20Bg; CONSENT=PENDING+999";
        }
    }

    std::string hdr_str = "User-Agent: " + ua + "\r\n"
                          "Accept: " + (req_headers.count("accept") ? req_headers["accept"] : "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8") + "\r\n"
                          "Accept-Language: " + (lang.empty() ? "en-US,en;q=0.9" : lang) + "\r\n";
    if (req_headers.count("sec-ch-ua")) hdr_str += "Sec-Ch-Ua: " + req_headers["sec-ch-ua"] + "\r\n";
    else if (!moecher::tooling::g_client_sec_ch_ua.empty()) hdr_str += "Sec-Ch-Ua: " + moecher::tooling::g_client_sec_ch_ua + "\r\n";

    if (req_headers.count("sec-ch-ua-mobile")) hdr_str += "Sec-Ch-Ua-Mobile: " + req_headers["sec-ch-ua-mobile"] + "\r\n";
    else if (!moecher::tooling::g_client_sec_ch_ua_mobile.empty()) hdr_str += "Sec-Ch-Ua-Mobile: " + moecher::tooling::g_client_sec_ch_ua_mobile + "\r\n";

    if (req_headers.count("sec-ch-ua-platform")) hdr_str += "Sec-Ch-Ua-Platform: " + req_headers["sec-ch-ua-platform"] + "\r\n";
    else if (!moecher::tooling::g_client_sec_ch_ua_platform.empty()) hdr_str += "Sec-Ch-Ua-Platform: " + moecher::tooling::g_client_sec_ch_ua_platform + "\r\n";

    if (!cookies.empty()) hdr_str += "Cookie: " + cookies + "\r\n";
    if (req_headers.count("content-type")) hdr_str += "Content-Type: " + req_headers["content-type"] + "\r\n";

    int hdr_wlen = MultiByteToWideChar(CP_UTF8, 0, hdr_str.c_str(), -1, NULL, 0);
    std::vector<wchar_t> w_hdr(hdr_wlen);
    MultiByteToWideChar(CP_UTF8, 0, hdr_str.c_str(), -1, w_hdr.data(), hdr_wlen);

    void* p_data = req_body.empty() ? WINHTTP_NO_REQUEST_DATA : (void*)req_body.data();
    DWORD dw_data_len = (DWORD)req_body.size();

    BOOL send_ok = WinHttpSendRequest(h_request, w_hdr.data(), (DWORD)-1L, p_data, dw_data_len, dw_data_len, 0);
    if (!send_ok || !WinHttpReceiveResponse(h_request, NULL)) {
        WinHttpCloseHandle(h_request);
        WinHttpCloseHandle(h_connect);
        WinHttpCloseHandle(h_session);
        const char* err_resp = "HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n";
        send(client_sock, err_resp, (int)strlen(err_resp), 0);
        closesocket(client_sock);
        return;
    }

    DWORD status_code = 200;
    DWORD status_size = sizeof(status_code);
    WinHttpQueryHeaders(h_request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                        WINHTTP_HEADER_NAME_BY_INDEX, &status_code, &status_size, WINHTTP_NO_HEADER_INDEX);

    // Read upstream headers
    DWORD raw_hdr_size = 0;
    std::string raw_headers_str;
    WinHttpQueryHeaders(h_request, WINHTTP_QUERY_RAW_HEADERS_CRLF, WINHTTP_HEADER_NAME_BY_INDEX, NULL, &raw_hdr_size, WINHTTP_NO_HEADER_INDEX);
    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER && raw_hdr_size > 0) {
        std::vector<wchar_t> raw_hdr_buf(raw_hdr_size / sizeof(wchar_t) + 1, 0);
        if (WinHttpQueryHeaders(h_request, WINHTTP_QUERY_RAW_HEADERS_CRLF, WINHTTP_HEADER_NAME_BY_INDEX, raw_hdr_buf.data(), &raw_hdr_size, WINHTTP_NO_HEADER_INDEX)) {
            int utf8_len = WideCharToMultiByte(CP_UTF8, 0, raw_hdr_buf.data(), -1, NULL, 0, NULL, NULL);
            if (utf8_len > 0) {
                std::vector<char> utf8_buf(utf8_len, 0);
                WideCharToMultiByte(CP_UTF8, 0, raw_hdr_buf.data(), -1, utf8_buf.data(), utf8_len, NULL, NULL);
                raw_headers_str = std::string(utf8_buf.data());
                moecher::tooling::parse_and_store_set_cookie_header(upstream_url, raw_headers_str);
            }
        }
    }

    // Read body
    std::string resp_body;
    DWORD bytes_avail = 0;
    while (WinHttpQueryDataAvailable(h_request, &bytes_avail) && bytes_avail > 0) {
        std::vector<char> temp_buf(bytes_avail);
        DWORD bytes_read_chunk = 0;
        if (WinHttpReadData(h_request, temp_buf.data(), bytes_avail, &bytes_read_chunk) && bytes_read_chunk > 0) {
            resp_body.append(temp_buf.data(), bytes_read_chunk);
            if (resp_body.size() > 10000000) break;
        } else {
            break;
        }
    }

    WinHttpCloseHandle(h_request);
    WinHttpCloseHandle(h_connect);
    WinHttpCloseHandle(h_session);

    // Filter headers & build response
    std::string clean_headers;
    std::istringstream resp_hdr_iss(raw_headers_str);
    std::string rh_line;
    bool is_first_line = true;
    std::string status_line = "HTTP/1.1 200 OK\r\n";

    while (std::getline(resp_hdr_iss, rh_line)) {
        if (!rh_line.empty() && rh_line.back() == '\r') rh_line.pop_back();
        if (rh_line.empty()) continue;

        if (is_first_line) {
            is_first_line = false;
            if (status_code == 429 || status_code == 403) {
                status_line = "HTTP/1.1 200 OK\r\n";
            } else {
                status_line = rh_line + "\r\n";
            }
            continue;
        }

        std::string lower = rh_line;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        // Strip framing restrictions, CORS blockers, COOP/COEP/CORP
        if (lower.rfind("x-frame-options:", 0) == 0) continue;
        if (lower.rfind("content-security-policy:", 0) == 0) continue;
        if (lower.rfind("cross-origin-opener-policy:", 0) == 0) continue;
        if (lower.rfind("cross-origin-embedder-policy:", 0) == 0) continue;
        if (lower.rfind("cross-origin-resource-policy:", 0) == 0) continue;
        if (lower.rfind("transfer-encoding:", 0) == 0) continue;
        if (lower.rfind("content-length:", 0) == 0) continue;
        if (lower.rfind("connection:", 0) == 0) continue;
        if (lower.rfind("location:", 0) == 0 && lower.find("consent.google") != std::string::npos) continue;

        clean_headers += rh_line + "\r\n";
    }

    // HTML Shims: ensure solveSimpleChallenge is defined so Google challenge script doesn't throw ReferenceError
    if (resp_body.find("solveSimpleChallenge") != std::string::npos) {
        size_t ssc_pos = 0;
        while ((ssc_pos = resp_body.find("solveSimpleChallenge", ssc_pos)) != std::string::npos) {
            if (ssc_pos >= 3 && resp_body.substr(ssc_pos - 3, 3) == "if(") {
                resp_body.replace(ssc_pos, 20, "window.solveSimpleChallenge");
                ssc_pos += 27;
            } else if (ssc_pos >= 4 && resp_body.substr(ssc_pos - 4, 4) == "if (") {
                resp_body.replace(ssc_pos, 20, "window.solveSimpleChallenge");
                ssc_pos += 27;
            } else {
                ssc_pos += 20;
            }
        }
    }

    // Inject shim in HTML
    std::string lower_body = resp_body;
    std::transform(lower_body.begin(), lower_body.end(), lower_body.begin(), ::tolower);
    if (lower_body.find("<html") != std::string::npos || lower_body.find("<head") != std::string::npos) {
        std::string shim = "<script>window.solveSimpleChallenge=window.solveSimpleChallenge||function(){};</script>\n";
        size_t head_pos = lower_body.find("<head");
        if (head_pos != std::string::npos) {
            size_t tag_close = resp_body.find('>', head_pos);
            if (tag_close != std::string::npos) {
                resp_body.insert(tag_close + 1, "\n" + shim);
            }
        }
    }

    std::string cors_and_frame_headers = "Access-Control-Allow-Origin: *\r\n"
                                         "Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS, HEAD\r\n"
                                         "Access-Control-Allow-Headers: *\r\n"
                                         "Access-Control-Expose-Headers: *\r\n"
                                         "Content-Security-Policy: frame-ancestors *;\r\n"
                                         "Content-Length: " + std::to_string(resp_body.size()) + "\r\n"
                                         "Connection: close\r\n";

    std::string full_response = status_line + clean_headers + cors_and_frame_headers + "\r\n" + resp_body;
    send_all(client_sock, full_response.data(), (int)full_response.size());
    closesocket(client_sock);

#else
    // Fallback socket relay for non-Windows
    std::string host = host_header;
    int port = 80;
    if (host.empty() && target_url.rfind("http://", 0) == 0) {
        std::string no_proto = target_url.substr(7);
        size_t slash = no_proto.find('/');
        std::string host_port = (slash != std::string::npos) ? no_proto.substr(0, slash) : no_proto;
        size_t colon = host_port.find(':');
        if (colon != std::string::npos) {
            host = host_port.substr(0, colon);
            try { port = std::stoi(host_port.substr(colon + 1)); } catch (...) { port = 80; }
        } else {
            host = host_port;
        }
    }

    struct addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res) {
        const char* err_resp = "HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n";
        send(client_sock, err_resp, (int)strlen(err_resp), 0);
        closesocket(client_sock);
        return;
    }

    SOCKET remote_sock = INVALID_SOCKET;
    for (struct addrinfo* ptr = res; ptr != nullptr; ptr = ptr->ai_next) {
        remote_sock = socket(ptr->ai_family, ptr->ai_socktype, ptr->ai_protocol);
        if (remote_sock == INVALID_SOCKET) continue;
        if (connect(remote_sock, ptr->ai_addr, (int)ptr->ai_addrlen) == 0) break;
        closesocket(remote_sock);
        remote_sock = INVALID_SOCKET;
    }
    freeaddrinfo(res);

    if (remote_sock == INVALID_SOCKET) {
        const char* err_resp = "HTTP/1.1 504 Gateway Timeout\r\nContent-Length: 0\r\n\r\n";
        send(client_sock, err_resp, (int)strlen(err_resp), 0);
        closesocket(client_sock);
        return;
    }

    send_all(remote_sock, req_vec.data(), bytes_read);

    char relay_buf[16384];
    bool first_remote_chunk = true;
    while (g_proxy_running.load()) {
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(client_sock, &read_fds);
        FD_SET(remote_sock, &read_fds);

        struct timeval tv{30, 0};
        int max_fd = (int)std::max(client_sock, remote_sock);
        int activity = select(max_fd + 1, &read_fds, nullptr, nullptr, &tv);
        if (activity <= 0) break;

        if (FD_ISSET(client_sock, &read_fds)) {
            int n = recv(client_sock, relay_buf, sizeof(relay_buf), 0);
            if (n <= 0) break;
            if (!send_all(remote_sock, relay_buf, n)) break;
        }
        if (FD_ISSET(remote_sock, &read_fds)) {
            int n = recv(remote_sock, relay_buf, sizeof(relay_buf), 0);
            if (n <= 0) break;

            if (first_remote_chunk) {
                first_remote_chunk = false;
                std::string resp(relay_buf, n);
                size_t hdr_end = resp.find("\r\n\r\n");
                if (hdr_end != std::string::npos) {
                    std::string raw_headers = resp.substr(0, hdr_end);
                    std::string body = resp.substr(hdr_end + 4);

                    std::istringstream iss_r(raw_headers);
                    std::string line_r;
                    std::string clean_hdrs;

                    while (std::getline(iss_r, line_r)) {
                        if (!line_r.empty() && line_r.back() == '\r') line_r.pop_back();
                        if (line_r.empty()) continue;

                        std::string lower = line_r;
                        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                        if (lower.rfind("x-frame-options:", 0) == 0) continue;
                        if (lower.rfind("content-security-policy:", 0) == 0) continue;
                        if (lower.rfind("cross-origin-opener-policy:", 0) == 0) continue;
                        if (lower.rfind("cross-origin-embedder-policy:", 0) == 0) continue;
                        if (lower.rfind("cross-origin-resource-policy:", 0) == 0) continue;

                        clean_hdrs += line_r + "\r\n";
                    }

                    std::string cors_and_frame = "Access-Control-Allow-Origin: *\r\n"
                                                 "Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS, HEAD\r\n"
                                                 "Access-Control-Allow-Headers: *\r\n"
                                                 "Access-Control-Expose-Headers: *\r\n"
                                                 "Content-Security-Policy: frame-ancestors *;\r\n";
                    std::string modified = clean_hdrs + cors_and_frame + "\r\n" + body;
                    if (!send_all(client_sock, modified.data(), (int)modified.size())) break;
                    continue;
                }
            }
            if (!send_all(client_sock, relay_buf, n)) break;
        }
    }
    closesocket(remote_sock);
    closesocket(client_sock);
#endif
}

// Start Forward Proxy Listener bound to 0.0.0.0:proxy_port
bool start_forward_proxy(int proxy_port) {
    if (g_proxy_running.load()) return true;

#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif

    g_proxy_listen_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (g_proxy_listen_sock == INVALID_SOCKET) {
        LOG_ERROR("[Proxy] Failed to create proxy listen socket");
        return false;
    }

    int opt = 1;
    setsockopt(g_proxy_listen_sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

    sockaddr_in saddr{};
    saddr.sin_family = AF_INET;
    saddr.sin_addr.s_addr = INADDR_ANY; // 0.0.0.0 (All local and remote interfaces)
    saddr.sin_port = htons((u_short)proxy_port);

    if (bind(g_proxy_listen_sock, (sockaddr*)&saddr, sizeof(saddr)) == SOCKET_ERROR) {
        LOG_ERROR("[Proxy] Failed to bind forward proxy to 0.0.0.0:%d", proxy_port);
        closesocket(g_proxy_listen_sock);
        g_proxy_listen_sock = INVALID_SOCKET;
        return false;
    }

    if (listen(g_proxy_listen_sock, 128) == SOCKET_ERROR) {
        LOG_ERROR("[Proxy] Failed to listen on forward proxy socket");
        closesocket(g_proxy_listen_sock);
        g_proxy_listen_sock = INVALID_SOCKET;
        return false;
    }

    g_proxy_port.store(proxy_port);
    g_proxy_running.store(true);
    LOG_INFO("[Proxy] Forward Proxy (HTTP/HTTPS CONNECT) listening on 0.0.0.0:%d", proxy_port);

    std::thread proxy_thread([]() {
        while (g_proxy_running.load()) {
            sockaddr_in client_addr{};
#if defined(_WIN32)
            int addr_len = sizeof(client_addr);
#else
            socklen_t addr_len = sizeof(client_addr);
#endif
            SOCKET client_sock = accept(g_proxy_listen_sock, (sockaddr*)&client_addr, &addr_len);
            if (client_sock == INVALID_SOCKET) {
                if (!g_proxy_running.load()) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            std::thread(handle_client_socket, client_sock).detach();
        }
    });
    proxy_thread.detach();

    return true;
}

void stop_forward_proxy() {
    if (!g_proxy_running.load()) return;
    g_proxy_running.store(false);
    if (g_proxy_listen_sock != INVALID_SOCKET) {
        closesocket(g_proxy_listen_sock);
        g_proxy_listen_sock = INVALID_SOCKET;
    }
}

// Windows System Proxy Manager
void set_windows_system_proxy(bool enable, const std::string& proxy_server) {
#ifdef _WIN32
    HKEY hKey;
    if (RegOpenKeyExA(HKEY_CURRENT_USER, "Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings", 0, KEY_SET_VALUE, &hKey) == ERROR_SUCCESS) {
        DWORD dwEnable = enable ? 1 : 0;
        RegSetValueExA(hKey, "ProxyEnable", 0, REG_DWORD, (const BYTE*)&dwEnable, sizeof(dwEnable));
        if (enable && !proxy_server.empty()) {
            std::string server_val = "http=" + proxy_server + ";https=" + proxy_server;
            RegSetValueExA(hKey, "ProxyServer", 0, REG_SZ, (const BYTE*)server_val.c_str(), (DWORD)server_val.length() + 1);
            std::string override_val = "<local>;localhost;127.0.0.1;zetaorangews";
            RegSetValueExA(hKey, "ProxyOverride", 0, REG_SZ, (const BYTE*)override_val.c_str(), (DWORD)override_val.length() + 1);
        }
        RegCloseKey(hKey);

        HMODULE hWinINet = LoadLibraryA("wininet.dll");
        if (hWinINet) {
            typedef BOOL(WINAPI* pfnInternetSetOptionA)(HANDLE, DWORD, LPVOID, DWORD);
            pfnInternetSetOptionA pInternetSetOptionA = (pfnInternetSetOptionA)GetProcAddress(hWinINet, "InternetSetOptionA");
            if (pInternetSetOptionA) {
                pInternetSetOptionA(NULL, 39 /* INTERNET_OPTION_SETTINGS_CHANGED */, NULL, 0);
                pInternetSetOptionA(NULL, 37 /* INTERNET_OPTION_REFRESH */, NULL, 0);
            }
            FreeLibrary(hWinINet);
        }
        LOG_INFO("[Proxy] Local Windows System Proxy %s (server: %s)", enable ? "ENABLED" : "DISABLED", proxy_server.c_str());
    }
#endif
}

} // namespace moecher::proxy

// ════════════════════════════════════════════════════════════════════════════════
//  Agentic & Tool Subsystem Implementation
// ════════════════════════════════════════════════════════════════════════════════

namespace moecher::agentic {

std::string g_base_system_prompt = 
    "You are FrankensTin, a powerful multimodal vision-language AI assistant with direct visual perception for detailed scene analysis, vehicle recognition, optical character recognition (OCR), and 3D modeling.\n\n"
    "# Multimodal Visual Perception & Recognition Guidelines\n"
    "- Directly perceive and report visual objects, colors, vehicle makes, models, and parts.\n"
    "- For vehicles: clearly identify the vehicle make, vehicle color, and read the exact alphanumeric characters on its license plate accurately.\n"
    "- Answer directly, concisely, and factually based on the visual evidence, without meta-commentary.\n\n"
    "# High-Fidelity Analytic 3D Modeling Instructions\n"
    "When asked to create, model, or reconstruct in 3D (or reconstruct an object from an image), "
    "always output Three.js representation code defining `function createModel(scene, THREE, inputImage, helpers) { ... }` inside a ```javascript code block "
    "(never output a full HTML file), adding all meshes to `scene`.\n\n"
    "CRITICAL RULES FOR ANALYTIC MODELING, PART-SPECIFIC TEXTURING & UV MAPPING:\n"
    "1. SEMANTIC PART DECOMPOSITION: Never model an object as a single monolithic mesh. Deconstruct the subject into its distinct structural and anatomical parts (e.g. for an apple: body, stem, leaf; for a teapot: body, lid, knob, handle, spout; for vehicles/furniture: body, wheels, windows, panels). Model EVERY part as a separate named `THREE.Mesh` with its own geometry and material, or use `helpers.createPart(name, geometry, material, scene)`.\n"
    "2. PART-SPECIFIC TEXTURES & COLORS (DO NOT NAIVELY APPLY THE ENTIRE PHOTO!):\n"
    "   - NEVER blindly apply the entire input photo onto the whole model or primary mesh (which stretches background, shadows, and unrelated features across surfaces).\n"
    "   - For parts requiring surface textures: use `helpers.cropTexture(inputImage, uMin, vMin, uMax, vMax, options)` to crop clean sub-regions of the photo matching that part (normalized coordinates 0..1).\n"
    "   - For uniform, solid, or metallic parts: use `helpers.sampleColor(inputImage, u, v)` to sample realistic colors and set physically accurate PBR properties (roughness, metalness, clearcoat).\n"
    "3. ACCURATE UV MAPPING:\n"
    "   - Revolved / lathe / cylindrical parts: use cylindrical/lathe UVs (`helpers.applyCylindricalUV(geometry)` or standard Lathe UVs) with `wrapS: THREE.RepeatWrapping`.\n"
    "   - Flat / curved / extruded parts (leaves, labels, panels, wings): use `helpers.applyPlanarUV(geometry, 'z')` so cropped sub-textures map cleanly without distortion.\n"
    "   - Multi-sided / cubic parts: use `helpers.applyBoxUV(geometry)`.\n"
    "4. WATERTIGHT GEOMETRY & SEAMLESS INTEGRATION:\n"
    "   - Revolved profiles must start and end at x=0 (use `helpers.createWatertightLathe` or `helpers.createHollowVessel` for vessels with solid wall thickness).\n"
    "   - Tubes, pipes, stems, and handles must have sealed end caps (use `helpers.createCappedTube`).\n"
    "   - Attachments (stems, leaves, handles, spouts) must penetrate 5-10% deep into parent meshes to prevent floating seams or gaps.";

std::string g_current_system_prompt = "";
json g_current_active_tools = json::array();
const std::string g_server_instance_id = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());

json resolve_canonical_tools(const json& tools_input) {
    static const std::unordered_map<std::string, json> CANONICAL_TOOLS = {
        {"web_search", {
            {"type", "function"},
            {"function", {
                {"name", "web_search"},
                {"description", "Search the web for current information and news."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"query", {{"type", "string"}}},
                        {"num_results", {{"type", "integer"}}},
                        {"site", {{"type", "string"}}}
                    }},
                    {"required", json::array({"query"})}
                }}
            }}
        }},
        {"youtube_search", {
            {"type", "function"},
            {"function", {
                {"name", "youtube_search"},
                {"description", "Search YouTube for videos and music."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"query", {{"type", "string"}}},
                        {"num_results", {{"type", "integer"}}}
                    }},
                    {"required", json::array({"query"})}
                }}
            }}
        }},
        {"google_search", {
            {"type", "function"},
            {"function", {
                {"name", "google_search"},
                {"description", "Search the web using Google."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"query", {{"type", "string"}}},
                        {"num_results", {{"type", "integer"}}},
                        {"site", {{"type", "string"}}}
                    }},
                    {"required", json::array({"query"})}
                }}
            }}
        }},
        {"fetch_url", {
            {"type", "function"},
            {"function", {
                {"name", "fetch_url"},
                {"description", "Fetch text content from a web URL."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"url", {{"type", "string"}}},
                        {"mode", {{"type", "string"}, {"enum", {"text", "raw", "scripts", "links"}}}},
                        {"pattern", {{"type", "string"}}},
                        {"max_chars", {{"type", "integer"}}}
                    }},
                    {"required", json::array({"url"})}
                }}
            }}
        }},
        {"read_file", {
            {"type", "function"},
            {"function", {
                {"name", "read_file"},
                {"description", "Read file contents from local filesystem."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path", {{"type", "string"}}},
                        {"start_line", {{"type", "integer"}}},
                        {"end_line", {{"type", "integer"}}}
                    }},
                    {"required", json::array({"path"})}
                }}
            }}
        }},
        {"write_file", {
            {"type", "function"},
            {"function", {
                {"name", "write_file"},
                {"description", "Write or overwrite a file on local filesystem."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path", {{"type", "string"}}},
                        {"content", {{"type", "string"}}},
                        {"overwrite", {{"type", "boolean"}}}
                    }},
                    {"required", json::array({"path", "content"})}
                }}
            }}
        }},
        {"edit_file", {
            {"type", "function"},
            {"function", {
                {"name", "edit_file"},
                {"description", "Replace unique text in an existing file."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path", {{"type", "string"}}},
                        {"target_content", {{"type", "string"}}},
                        {"replacement_content", {{"type", "string"}}}
                    }},
                    {"required", json::array({"path", "target_content", "replacement_content"})}
                }}
            }}
        }},
        {"execute_command", {
            {"type", "function"},
            {"function", {
                {"name", "execute_command"},
                {"description", "Execute a local shell command."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"command", {{"type", "string"}}}
                    }},
                    {"required", json::array({"command"})}
                }}
            }}
        }},
        {"create_3d_model", {
            {"type", "function"},
            {"function", {
                {"name", "create_3d_model"},
                {"description", "Create and display an interactive 3D model in the 3D Studio using Three.js representation code."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"name", {{"type", "string"}, {"description", "Name of the 3D object or model"}}},
                        {"code", {{"type", "string"}, {"description", "Three.js JavaScript representation code defining function createModel(scene, THREE) { ... }"}}}
                    }},
                    {"required", json::array({"name", "code"})}
                }}
            }}
        }}
    };

    if (tools_input.is_string()) {
        std::string s = tools_input.get<std::string>();
        if (s == "default" || s == "all") {
            json arr = json::array();
            static const std::vector<std::string> default_order = {"web_search", "youtube_search", "google_search", "fetch_url", "create_3d_model", "read_file", "write_file", "edit_file", "execute_command"};
            for (const auto& name : default_order) {
                if (name == "execute_command" && (!g_server_exec
#ifndef ENABLE_UNSAFE_HOST_EXEC
                    || true
#endif
                )) {
                    continue;
                }
                if ((name == "write_file" || name == "edit_file") && !g_allow_file_write) {
                    continue;
                }
                if (CANONICAL_TOOLS.count(name)) arr.push_back(CANONICAL_TOOLS.at(name));
            }
            return arr;
        }
        if (CANONICAL_TOOLS.count(s)) {
            return json::array({CANONICAL_TOOLS.at(s)});
        }
        return json::array();
    }

    if (tools_input.is_array()) {
        json arr = json::array();
        for (const auto& item : tools_input) {
            if (item.is_string()) {
                std::string name = item.get<std::string>();
                if (CANONICAL_TOOLS.count(name)) {
                    arr.push_back(CANONICAL_TOOLS.at(name));
                }
            } else if (item.is_object()) {
                arr.push_back(item);
            }
        }
        return arr;
    }

    return json::array();
}

std::string build_dynamic_tools_prompt(const json& resolved_tools, bool is_qwen) {
    if (!resolved_tools.is_array() || resolved_tools.empty()) {
        return "";
    }

    bool has_yt = false;
    for (const auto& item : resolved_tools) {
        if (item.contains("function") && item["function"].value("name", "") == "youtube_search") {
            has_yt = true;
            break;
        }
    }

    std::string prompt =
        "\n\n# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n";
    for (const auto& item : resolved_tools) {
        prompt += item.dump() + "\n";
    }
    prompt += "</tools>\n\n";

    if (is_qwen) {
        prompt +=
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            "{\"name\": \"<function-name>\", \"arguments\": <args-json-object>}\n"
            "</tool_call>\n\n";
    }

    if (has_yt) {
        prompt +=
            "## Tool Usage Instructions:\n"
            "- When the user asks to play music, a song, or a video, invoke `youtube_search` directly. The video will automatically load and play in the user's preview panel with autoplay.\n";
    }

    prompt +=
        "## High-Fidelity Analytic 3D Modeling Instructions:\n"
        "- When asked to create, model, or reconstruct in 3D (or reconstruct an object from an image), always output Three.js representation code defining `function createModel(scene, THREE, inputImage, helpers) { ... }` inside a ```javascript code block (never output a full HTML file), adding all meshes to `scene`.\n"
        "- 1. SEMANTIC PART DECOMPOSITION: Never model an object as a single monolithic mesh. Deconstruct the object into its distinct structural and anatomical parts (e.g. for an apple: body, stem, leaf; for a teapot: vessel body, lid, knob, handle, spout; for vehicles/furniture: body, wheels, windows, panels). Model EVERY part as a separate named `THREE.Mesh` with its own geometry and material (or use `helpers.createPart(name, geometry, material, scene)`).\n"
        "- 2. PART-SPECIFIC TEXTURES & COLORS (DO NOT NAIVELY APPLY THE ENTIRE PHOTO!):\n"
        "  * NEVER blindly apply the entire input photo onto the whole model or primary mesh (which stretches background, shadows, and unrelated features across surfaces).\n"
        "  * For parts requiring surface textures (skin, patterns, labels, dials, leaves): use `helpers.cropTexture(inputImage, uMin, vMin, uMax, vMax, options)` to crop clean sub-regions of the photo matching that part (normalized coordinates 0..1).\n"
        "  * For uniform, solid, or metallic parts (stems, chrome handles, rims, bases): use `helpers.sampleColor(inputImage, u, v)` to sample realistic colors and set physically accurate PBR properties (`roughness`, `metalness`, `clearcoat`).\n"
        "- 3. ACCURATE UV MAPPING:\n"
        "  * For flat, curved, or extruded parts (leaves, labels, panels, wings): apply planar UVs using `helpers.applyPlanarUV(geometry, 'z')` so cropped sub-textures map cleanly without distortion.\n"
        "  * For cylindrical or revolved parts: use cylindrical UVs (`helpers.applyCylindricalUV(geometry)` or standard Lathe UVs) with `wrapS: THREE.RepeatWrapping`.\n"
        "  * For complex multi-sided parts: use `helpers.applyBoxUV(geometry)`.\n"
        "- 4. WATERTIGHT GEOMETRY & SEAMLESS INTEGRATION:\n"
        "  * Revolved profiles must start and end at x=0 (use `helpers.createWatertightLathe` or `helpers.createHollowVessel` for vessels with solid wall thickness).\n"
        "  * Tubes, pipes, stems, and handles must have sealed end caps (use `helpers.createCappedTube`).\n"
        "  * Attachments (stems, leaves, handles, spouts) must penetrate 5-10% deep into parent meshes to prevent floating seams or gaps.\n";

    return prompt;
}

std::string update_system_prompt_with_tools(const std::string& original_content, const std::string& tools_system_prompt, bool has_tools) {
    std::string content = original_content;
    size_t tools_pos = content.find("\n\n# Tools");
    if (tools_pos == std::string::npos) tools_pos = content.find("# Tools");
    if (tools_pos == std::string::npos) tools_pos = content.find("<tools>");
    if (tools_pos != std::string::npos) {
        content = content.substr(0, tools_pos);
    }
    while (!content.empty() && (content.back() == ' ' || content.back() == '\n' || content.back() == '\r')) {
        content.pop_back();
    }
    if (content.empty()) {
        content = g_base_system_prompt;
    }
    if (has_tools) {
        content += tools_system_prompt;
    }
    return content;
}

void register_agentic_routes(httplib::Server& svr) {
    svr.Get("/proxy.pac", [](const httplib::Request& req, httplib::Response& res) {
        std::string host_header = req.get_header_value("Host");
        std::string proxy_host = "127.0.0.1";
        if (!host_header.empty()) {
            size_t colon = host_header.find(':');
            proxy_host = (colon != std::string::npos) ? host_header.substr(0, colon) : host_header;
        }
        int p_port = moecher::proxy::g_proxy_port.load();
        std::string pac = "function FindProxyForURL(url, host) {\n"
                          "  if (shExpMatch(host, \"localhost\") || shExpMatch(host, \"127.0.0.1\") || shExpMatch(host, \"" + proxy_host + "\") || isPlainHostName(host) || isInNet(host, \"10.0.0.0\", \"255.0.0.0\") || isInNet(host, \"192.168.0.0\", \"255.255.0.0\")) {\n"
                          "    return \"DIRECT\";\n"
                          "  }\n"
                          "  return \"PROXY " + proxy_host + ":" + std::to_string(p_port) + "; DIRECT\";\n"
                          "}\n";
        res.set_content(pac, "application/x-ns-proxy-autoconfig");
    });

    // One-Click Remote Client Windows Setup Batch File
    svr.Get("/api/proxy/setup.bat", [](const httplib::Request& req, httplib::Response& res) {
        std::string host_header = req.get_header_value("Host");
        std::string proxy_host = "127.0.0.1";
        if (!host_header.empty()) {
            size_t colon = host_header.find(':');
            proxy_host = (colon != std::string::npos) ? host_header.substr(0, colon) : host_header;
        }
        int p_port = moecher::proxy::g_proxy_port.load();
        std::string bat = "@echo off\r\n"
                          "setlocal\r\n"
                          "chcp 65001 >nul\r\n"
                          "echo ===============================================================================\r\n"
                          "echo   Configuring Windows Client Proxy to route through MinnieTheMoEcher\r\n"
                          "echo   Proxy Server: " + proxy_host + ":" + std::to_string(p_port) + "\r\n"
                          "echo ===============================================================================\r\n"
                          "reg add \"HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings\" /v ProxyEnable /t REG_DWORD /d 1 /f >nul\r\n"
                          "reg add \"HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings\" /v ProxyServer /t REG_SZ /d \"http=" + proxy_host + ":" + std::to_string(p_port) + ";https=" + proxy_host + ":" + std::to_string(p_port) + "\" /f >nul\r\n"
                          "reg add \"HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings\" /v ProxyOverride /t REG_SZ /d \"<local>;localhost;127.0.0.1;" + proxy_host + "\" /f >nul\r\n"
                          "echo.\r\n"
                          "echo [SUCCESS] Windows System & Browser Proxy configured to " + proxy_host + ":" + std::to_string(p_port) + "!\r\n"
                          "echo Edge, Chrome, and system web browsers will now route external traffic transparently.\r\n"
                          "echo.\r\n"
                          "echo To disable later, run restore_proxy.bat or disable Proxy in Windows Settings.\r\n"
                          "echo.\r\n"
                          "pause\r\n";
        res.set_header("Content-Disposition", "attachment; filename=\"setup_proxy.bat\"");
        res.set_content(bat, "application/x-bat");
    });

    // One-Click Remote Client Windows Restore / Disable Batch File
    svr.Get("/api/proxy/restore.bat", [](const httplib::Request&, httplib::Response& res) {
        std::string bat = "@echo off\r\n"
                          "setlocal\r\n"
                          "chcp 65001 >nul\r\n"
                          "echo ===============================================================================\r\n"
                          "echo   Disabling Windows Client Proxy\r\n"
                          "echo ===============================================================================\r\n"
                          "reg add \"HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings\" /v ProxyEnable /t REG_DWORD /d 0 /f >nul\r\n"
                          "echo.\r\n"
                          "echo [SUCCESS] Windows System Proxy has been disabled.\r\n"
                          "echo.\r\n"
                          "pause\r\n";
        res.set_header("Content-Disposition", "attachment; filename=\"restore_proxy.bat\"");
        res.set_content(bat, "application/x-bat");
    });

    // Proxy Status API
    svr.Get("/api/proxy/status", [](const httplib::Request& req, httplib::Response& res) {
        std::string host_header = req.get_header_value("Host");
        std::string proxy_host = "127.0.0.1";
        if (!host_header.empty()) {
            size_t colon = host_header.find(':');
            proxy_host = (colon != std::string::npos) ? host_header.substr(0, colon) : host_header;
        }
        int p_port = moecher::proxy::g_proxy_port.load();
        bool is_running = moecher::proxy::g_proxy_running.load();
        json status = {
            {"status", is_running ? "active" : "disabled"},
            {"proxy_running", is_running},
            {"proxy_port", p_port},
            {"detected_host", proxy_host},
            {"proxy_host", proxy_host},
            {"remote_addr", req.remote_addr},
            {"pac_url", "http://" + host_header + "/proxy.pac"},
            {"setup_bat_url", "http://" + host_header + "/api/proxy/setup.bat"}
        };
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.set_content(status.dump(), "application/json");
    });

    svr.Post("/api/tool/execute", [](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "*");

        json req_data;
        try { req_data = json::parse(req.body); }
        catch (...) {
            res.status = 400;
            res.set_content("{\"error\":\"Invalid JSON\"}", "application/json");
            return;
        }

        std::string tool_name = req_data.value("name", "");
        json args = json::object();
        if (req_data.contains("arguments")) {
            if (req_data["arguments"].is_string()) {
                try { args = json::parse(req_data["arguments"].get<std::string>()); }
                catch (...) {}
            } else if (req_data["arguments"].is_object()) {
                args = req_data["arguments"];
            }
        }

        int timeout_ms = req_data.value("timeout_ms", 60000);
        bool require_external_authorization = req_data.value("require_external_authorization", true);
        bool workspace_boundary_enforced = req_data.value("workspace_boundary_enforced", true);
        std::vector<std::string> authorized_paths;
        if (req_data.contains("authorized_paths") && req_data["authorized_paths"].is_array()) {
            for (const auto& ap : req_data["authorized_paths"]) {
                if (ap.is_string()) authorized_paths.push_back(ap.get<std::string>());
            }
        }

        auto is_path_authorized = [&](const std::string& p) -> bool {
            if (!require_external_authorization && !workspace_boundary_enforced) return true;
            if (moecher::tooling::is_path_inside_workspace(p)) return true;
            for (const auto& ap : authorized_paths) {
                if (ap == "*" || ap == p) return true;
                if (!ap.empty() && moecher::tooling::is_path_inside_workspace(p, ap)) return true;
            }
            return false;
        };

        json resp_data = json::object();
        std::string output;

        if (!g_enable_tools) {
            resp_data["success"] = false;
            resp_data["output"] = "Error: Tool execution is disabled on this server.";
            res.set_content(resp_data.dump(), "application/json");
            return;
        }

        if (tool_name.rfind("mcp__", 0) != 0 && tool_name.rfind("tinobruno-", 0) != 0) {
            bool is_active = false;
            for (const auto& t : g_current_active_tools) {
                if (t.contains("function") && t["function"].value("name", "") == tool_name) {
                    is_active = true;
                    break;
                }
            }
            if (!is_active) {
                resp_data["success"] = false;
                resp_data["output"] = "Error: Tool '" + tool_name + "' is disabled in current settings.";
                res.set_content(resp_data.dump(), "application/json");
                return;
            }
        }

        if (tool_name == "read_file") {
            std::string path = args.value("path", "");
            int start_line = args.value("start_line", 1);
            int end_line = args.value("end_line", -1);
            if (!is_path_authorized(path)) {
                resp_data["authorization_required"] = {
                    {"tool", "read_file"},
                    {"path", path}
                };
                output = "[Authorization Required: The path '" + path + "' is outside the workspace directory (" + moecher::tooling::get_workspace_directory() + "). Action requires explicit user authorization.]";
            } else {
                output = moecher::tooling::read_file(path, start_line, end_line);
            }
        } else if (tool_name == "write_file") {
            std::string path = args.value("path", "");
            std::string content = args.value("content", "");
            bool overwrite = args.value("overwrite", true);
            if (!is_path_authorized(path)) {
                resp_data["authorization_required"] = {
                    {"tool", "write_file"},
                    {"path", path}
                };
                output = "[Authorization Required: The path '" + path + "' is outside the workspace directory (" + moecher::tooling::get_workspace_directory() + "). Action requires explicit user authorization.]";
            } else {
                output = moecher::tooling::write_file(path, content, overwrite);
            }
        } else if (tool_name == "edit_file") {
            std::string path = args.value("path", "");
            std::string target_content = args.value("target_content", "");
            std::string replacement_content = args.value("replacement_content", "");
            if (!is_path_authorized(path)) {
                resp_data["authorization_required"] = {
                    {"tool", "edit_file"},
                    {"path", path}
                };
                output = "[Authorization Required: The path '" + path + "' is outside the workspace directory (" + moecher::tooling::get_workspace_directory() + "). Action requires explicit user authorization.]";
            } else {
                output = moecher::tooling::edit_file(path, target_content, replacement_content);
            }
        } else if (tool_name == "execute_command") {
            std::string cmd = args.value("command", "");
            output = moecher::tooling::execute_system_command(cmd, timeout_ms);
        } else if (tool_name == "youtube_search") {
            std::string query = args.value("query", "");
            int num_results = args.value("num_results", 3);
            auto doc = moecher::tooling::search_youtube_direct(query, num_results);
            if (doc.clean_text.empty()) {
                // Fall back to configured search engine if direct YouTube gave 0 results
                std::string api_key = args.value("api_key", "");
                if (api_key.empty() && args.contains("tavily_api_key")) {
                    api_key = args.value("tavily_api_key", "");
                }
                std::string provider = args.value("provider", "");
                doc = moecher::tooling::web_search_full(query, num_results, "", provider, api_key);
            }
            output = doc.clean_text;
            resp_data["retrieved_document"] = {
                {"url", doc.url},
                {"title", doc.title.empty() ? ("YouTube: " + query) : doc.title},
                {"html", doc.raw_html},
                {"snippet", (doc.clean_text.size() > 300 ? doc.clean_text.substr(0, 300) + "..." : doc.clean_text)}
            };
        } else if (tool_name == "web_search" || tool_name == "google_search") {
            std::string query = args.value("query", "");
            int num_results = args.value("num_results", 5);
            std::string site = args.value("site", "");
            std::string provider = args.value("provider", "");
            std::string api_key = args.value("api_key", "");
            if (api_key.empty() && args.contains("tavily_api_key")) {
                api_key = args.value("tavily_api_key", "");
            }
            std::string cx = args.value("cx", "");
            if (cx.empty() && args.contains("google_search_cx")) {
                cx = args.value("google_search_cx", "");
            }
            std::string searx_url = args.value("searxng_url", "");
            auto doc = moecher::tooling::web_search_full(query, num_results, site, provider, api_key, cx, searx_url);
            output = doc.clean_text;
            resp_data["retrieved_document"] = {
                {"url", doc.url},
                {"title", doc.title.empty() ? ("Web Search: " + query) : doc.title},
                {"html", doc.raw_html},
                {"snippet", (doc.clean_text.size() > 300 ? doc.clean_text.substr(0, 300) + "..." : doc.clean_text)}
            };
        } else if (tool_name == "fetch_url") {
            std::string url = args.value("url", "");
            std::string mode = args.value("mode", "text");
            std::string pattern = args.value("pattern", "");
            int offset = args.value("offset", 0);
            int max_chars = args.value("max_chars", 4000);
            auto doc = moecher::tooling::fetch_url_full(url, std::min(timeout_ms, 30000), max_chars, mode, pattern, offset);
            output = doc.clean_text;
            resp_data["retrieved_document"] = {
                {"url", doc.url},
                {"title", doc.title.empty() ? doc.url : doc.title},
                {"html", doc.raw_html},
                {"snippet", (doc.clean_text.size() > 300 ? doc.clean_text.substr(0, 300) + "..." : doc.clean_text)}
            };
        } else if (moecher::mcp::MCPManager::instance().has_tool(tool_name)) {
            output = moecher::mcp::MCPManager::instance().call_tool(tool_name, args, timeout_ms);
        } else {
            output = "[Unknown tool: " + tool_name + "]";
        }

        resp_data["output"] = output;
        res.set_content(resp_data.dump(), "application/json");
    });
    svr.Options("/api/tool/execute", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.status = 204;
    });

    svr.Get("/api/settings/search_provider", [](const httplib::Request&, httplib::Response& res) {
        auto search_settings = moecher::tooling::get_search_settings();
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(search_settings.dump(), "application/json");
    });

    svr.Post("/api/settings/search_provider", [](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        try {
            json body = json::parse(req.body);
            std::string provider = body.value("provider", "tavily");
            std::string tavily_key = body.value("tavily_api_key", "");
            std::string brave_key = body.value("brave_api_key", "");
            std::string serper_key = body.value("serper_api_key", "");
            std::string searx_url = body.value("searxng_url", "https://searx.be");
            std::string google_key = body.value("google_search_api_key", "");
            std::string google_cx = body.value("google_search_cx", "");
            int fast_media = -1;
            if (body.contains("fast_media_search") && body["fast_media_search"].is_boolean()) {
                fast_media = body["fast_media_search"].get<bool>() ? 1 : 0;
            }

            moecher::tooling::set_search_settings(provider, tavily_key, brave_key, serper_key, searx_url, google_key, google_cx, fast_media);
            auto updated_settings = moecher::tooling::get_search_settings();
            res.set_content(updated_settings.dump(), "application/json");
        } catch (const std::exception& e) {
            json err = {{"status", "error"}, {"message", e.what()}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
        }
    });

    svr.Options("/api/settings/search_provider", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.status = 204;
    });

    svr.Get("/api/settings/google_search", [](const httplib::Request&, httplib::Response& res) {
        auto creds = moecher::tooling::get_google_search_credentials();
        bool configured = !creds.first.empty() && !creds.second.empty();
        std::string masked_key = "";
        if (!creds.first.empty()) {
            if (creds.first.size() > 8) {
                masked_key = creds.first.substr(0, 6) + "..." + creds.first.substr(creds.first.size() - 4);
            } else {
                masked_key = "********";
            }
        }
        json body = {
            {"configured", configured},
            {"api_key_set", !creds.first.empty()},
            {"cx_set", !creds.second.empty()},
            {"masked_api_key", masked_key},
            {"cx", creds.second}
        };
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(body.dump(), "application/json");
    });

    svr.Post("/api/settings/google_search", [](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        try {
            json body = json::parse(req.body);
            std::string key = body.value("api_key", "");
            std::string cx = body.value("cx", "");
            moecher::tooling::set_google_search_credentials(key, cx);
            auto creds = moecher::tooling::get_google_search_credentials();
            json resp_body = {
                {"status", "ok"},
                {"configured", !creds.first.empty() && !creds.second.empty()},
                {"cx", creds.second}
            };
            res.set_content(resp_body.dump(), "application/json");
        } catch (const std::exception& e) {
            json err = {{"status", "error"}, {"message", e.what()}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
        }
    });

    svr.Get("/v1/workspace", [](const httplib::Request&, httplib::Response& res) {
        json info = {
            {"workspace_directory", moecher::tooling::get_workspace_directory()},
            {"default_timeout_sec", 60}
        };
        res.set_content(info.dump(), "application/json");
    });

    // ── MCP (Model Context Protocol) API Endpoints ──────────────────────────────
    svr.Get("/v1/mcp/servers", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        json servers = moecher::mcp::MCPManager::instance().get_servers_status_json();
        json body = {
            {"object", "list"},
            {"data", servers}
        };
        res.set_content(body.dump(), "application/json");
    });
    svr.Options("/v1/mcp/servers", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.status = 204;
    });

    svr.Get("/v1/mcp/registry/search", [](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        std::string query = req.get_param_value("q");
        if (query.empty()) {
            query = req.get_param_value("search");
        }

        // If user pasted a URL like https://registry.modelcontextprotocol.io/?q=teams
        if (query.find("?q=") != std::string::npos) {
            size_t qpos = query.find("?q=");
            query = query.substr(qpos + 3);
            size_t amp = query.find('&');
            if (amp != std::string::npos) query = query.substr(0, amp);
        } else if (query.find("&q=") != std::string::npos) {
            size_t qpos = query.find("&q=");
            query = query.substr(qpos + 3);
            size_t amp = query.find('&');
            if (amp != std::string::npos) query = query.substr(0, amp);
        }

        std::string target_url;
        if (!query.empty()) {
            std::string encoded_q;
            for (char c : query) {
                if (isalnum((unsigned char)c) || c == '-' || c == '_' || c == '.' || c == '~') {
                    encoded_q += c;
                } else if (c == ' ') {
                    encoded_q += "%20";
                } else {
                    char hex[4];
                    snprintf(hex, sizeof(hex), "%%%02X", (unsigned char)c);
                    encoded_q += hex;
                }
            }
            target_url = "https://registry.modelcontextprotocol.io/v0.1/servers?search=" + encoded_q + "&version=latest";
        } else {
            target_url = "https://registry.modelcontextprotocol.io/v0.1/servers?limit=30&version=latest";
        }

        std::string cmd = "curl -s -m 15 -H " + moecher::mcp::escape_shell_arg("Accept: application/json") + " " + moecher::mcp::escape_shell_arg(target_url);
        std::string output = moecher::mcp::execute_curl_command(cmd, 15000);

        if (output.empty()) {
            res.status = 502;
            res.set_content("{\"servers\":[],\"error\":\"Failed to reach registry.modelcontextprotocol.io\"}", "application/json");
            return;
        }

        try {
            auto parsed = json::parse(output);
            res.set_content(parsed.dump(), "application/json");
        } catch (...) {
            res.status = 502;
            res.set_content("{\"servers\":[],\"error\":\"Invalid JSON returned from registry\"}", "application/json");
        }
    });

    svr.Options("/v1/mcp/registry/search", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.status = 204;
    });

    svr.Post("/v1/mcp/servers", [](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        json body;
        try { body = json::parse(req.body); }
        catch (...) {
            res.status = 400;
            res.set_content("{\"error\":\"Invalid JSON\"}", "application/json");
            return;
        }
        std::string s_id = body.value("id", "");
        if (s_id.empty() && body.contains("server") && body["server"].is_object() && body["server"].contains("name")) {
            std::string full_name = body["server"]["name"].get<std::string>();
            size_t slash_pos = full_name.find_last_of('/');
            s_id = (slash_pos != std::string::npos) ? full_name.substr(slash_pos + 1) : full_name;
        } else if (s_id.empty() && body.contains("mcpServers") && body["mcpServers"].is_object() && !body["mcpServers"].empty()) {
            s_id = body["mcpServers"].begin().key();
        }
        if (s_id.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"'id' is required or could not be determined from JSON manifest\"}", "application/json");
            return;
        }
        std::string old_id = body.value("old_id", "");
        if (!old_id.empty() && old_id != s_id) {
            moecher::mcp::MCPManager::instance().remove_server(old_id);
        }
        auto cfg = moecher::mcp::MCPServerConfig::from_json(s_id, body);
        bool ok = moecher::mcp::MCPManager::instance().add_or_update_server(cfg, cfg.enabled);
        moecher::mcp::MCPManager::instance().save_config();
        json resp = {
            {"status", ok ? "ok" : "error"},
            {"id", s_id},
            {"servers", moecher::mcp::MCPManager::instance().get_servers_status_json()}
        };
        res.set_content(resp.dump(), "application/json");
    });

    svr.Delete("/v1/mcp/servers", [](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        std::string s_id = req.get_param_value("id");
        if (s_id.empty() && !req.body.empty()) {
            try {
                json body = json::parse(req.body);
                s_id = body.value("id", "");
            } catch (...) {}
        }
        if (s_id.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"'id' parameter is required\"}", "application/json");
            return;
        }
        bool ok = moecher::mcp::MCPManager::instance().remove_server(s_id);
        moecher::mcp::MCPManager::instance().save_config();
        json resp = {
            {"status", ok ? "ok" : "error"},
            {"id", s_id},
            {"servers", moecher::mcp::MCPManager::instance().get_servers_status_json()}
        };
        res.set_content(resp.dump(), "application/json");
    });

    svr.Post("/v1/mcp/servers/restart", [](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        std::string s_id = req.get_param_value("id");
        if (s_id.empty() && !req.body.empty()) {
            try {
                json body = json::parse(req.body);
                s_id = body.value("id", "");
            } catch (...) {}
        }
        if (s_id.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"'id' is required\"}", "application/json");
            return;
        }
        bool ok = moecher::mcp::MCPManager::instance().restart_server(s_id);
        json resp = {
            {"status", ok ? "ok" : "error"},
            {"id", s_id},
            {"servers", moecher::mcp::MCPManager::instance().get_servers_status_json()}
        };
        res.set_content(resp.dump(), "application/json");
    });
    svr.Options("/v1/mcp/servers/restart", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.status = 204;
    });

    svr.Post("/v1/mcp/servers/toggle", [](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        json body;
        try { body = json::parse(req.body); } catch (...) {}
        std::string s_id = body.value("id", req.get_param_value("id"));
        bool enabled = body.value("enabled", true);
        if (s_id.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"'id' is required\"}", "application/json");
            return;
        }
        bool ok = moecher::mcp::MCPManager::instance().toggle_server(s_id, enabled);
        moecher::mcp::MCPManager::instance().save_config();
        json resp = {
            {"status", ok ? "ok" : "error"},
            {"id", s_id},
            {"enabled", enabled},
            {"servers", moecher::mcp::MCPManager::instance().get_servers_status_json()}
        };
        res.set_content(resp.dump(), "application/json");
    });
    svr.Options("/v1/mcp/servers/toggle", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.status = 204;
    });

    svr.Get("/v1/mcp/tools", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        json tools_schema = moecher::mcp::MCPManager::instance().get_openai_tools_schema();
        json body = {
            {"object", "list"},
            {"data", tools_schema}
        };
        res.set_content(body.dump(), "application/json");
    });
    svr.Options("/v1/mcp/tools", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.status = 204;
    });
}

} // namespace moecher::agentic
