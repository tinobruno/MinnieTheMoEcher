#include "web_ui.hpp"
#include "embedded_web.hpp"
#include "version.hpp"
#include "tool_exec.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <string_view>
#include <cctype>

using json = nlohmann::json;

namespace moecher::web {

void register_web_routes(httplib::Server& svr) {
    // Web UI endpoints: serve disk assets when present (live development), fallback to embedded
    auto serve_asset = [](const std::string& disk_path, std::string_view embedded, const char* mime, httplib::Response& res) {
        std::ifstream in(disk_path, std::ios::binary);
        if (in.is_open()) {
            std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            res.set_content(content, mime);
        } else if (!embedded.empty()) {
            res.set_content(std::string(embedded), mime);
        }
        res.set_header("Cache-Control", "no-cache, no-store, must-revalidate");
        res.set_header("Pragma", "no-cache");
        res.set_header("Expires", "0");
    };

    svr.Get("/", [&serve_asset](const httplib::Request&, httplib::Response& res) {
        serve_asset("web/index.html", moecher::embedded_web::INDEX_HTML(), "text/html; charset=utf-8", res);
    });
    svr.Get("/index.html", [&serve_asset](const httplib::Request&, httplib::Response& res) {
        serve_asset("web/index.html", moecher::embedded_web::INDEX_HTML(), "text/html; charset=utf-8", res);
    });
    svr.Get("/style.css", [&serve_asset](const httplib::Request&, httplib::Response& res) {
        serve_asset("web/style.css", moecher::embedded_web::STYLE_CSS(), "text/css; charset=utf-8", res);
    });
    svr.Get("/script.js", [&serve_asset](const httplib::Request&, httplib::Response& res) {
        serve_asset("web/script.js", moecher::embedded_web::SCRIPT_JS(), "application/javascript; charset=utf-8", res);
    });
    svr.Get("/sw.js", [&serve_asset](const httplib::Request&, httplib::Response& res) {
        res.set_header("Service-Worker-Allowed", "/");
        serve_asset("web/sw.js", moecher::embedded_web::SW_JS(), "application/javascript; charset=utf-8", res);
    });
    svr.Get("/pdf.min.js", [&serve_asset](const httplib::Request&, httplib::Response& res) {
        serve_asset("web/pdf.min.js", "", "application/javascript; charset=utf-8", res);
    });
    svr.Get("/pdf.worker.min.js", [&serve_asset](const httplib::Request&, httplib::Response& res) {
        serve_asset("web/pdf.worker.min.js", "", "application/javascript; charset=utf-8", res);
    });

    // Health & status check
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(std::string("{\"status\":\"ok\",\"engine\":\"moecher\",\"version\":\"") + moecher::VERSION + "\"}", "application/json");
    });

    // Local Rewriting Reverse Proxy Endpoint for Browser Preview & Cross-Origin Unblocking
    auto handle_proxy_request = [](const httplib::Request& req, httplib::Response& res) {
        std::string target_url;
        if (req.has_param("url")) {
            target_url = req.get_param_value("url");
        }
        if (target_url.empty()) {
            res.status = 400;
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE, PATCH, HEAD");
            res.set_header("Access-Control-Allow-Headers", "*");
            res.set_header("Access-Control-Expose-Headers", "*");
            res.set_header("Access-Control-Allow-Credentials", "true");
            res.set_content("{\"error\":\"Missing ?url= parameter\"}", "application/json");
            return;
        }

        // Fix target_url protocol if protocol-relative
        if (target_url.rfind("//", 0) == 0) {
            target_url = "https:" + target_url;
        } else if (target_url.rfind("http://", 0) != 0 && target_url.rfind("https://", 0) != 0) {
            target_url = "https://" + target_url;
        }

        // Capture incoming client headers & cookies to update browser impersonation context
        std::string client_ua = req.get_header_value("X-Client-User-Agent");
        if (client_ua.empty()) client_ua = req.get_header_value("User-Agent");

        std::string client_lang = req.get_header_value("X-Client-Accept-Language");
        if (client_lang.empty()) client_lang = req.get_header_value("Accept-Language");

        std::string client_sec_ua = req.get_header_value("Sec-Ch-Ua");
        std::string client_sec_mobile = req.get_header_value("Sec-Ch-Ua-Mobile");
        std::string client_sec_platform = req.get_header_value("Sec-Ch-Ua-Platform");
        std::string client_cookies = req.get_header_value("X-Client-Cookie");
        if (client_cookies.empty()) client_cookies = req.get_header_value("Cookie");

        moecher::tooling::set_client_browser_context(client_ua, client_lang, client_sec_ua, client_sec_mobile, client_sec_platform, client_cookies);
        if (!client_cookies.empty()) {
            moecher::tooling::parse_and_store_set_cookie_header(target_url, client_cookies);
        }

        std::string post_body = (req.method == "POST") ? req.body : "";
        std::string post_ct = req.get_header_value("Content-Type");

        auto doc = moecher::tooling::fetch_url_full(target_url, 30000, 5000000, "raw", "", 0, req.method, post_body, post_ct);

        std::string lookup_url = doc.url.empty() ? target_url : doc.url;

        // Compute domain_origin (e.g. "https://www.linkedin.com")
        std::string domain_origin = "https://www.google.com";
        size_t proto_pos = lookup_url.find("://");
        if (proto_pos != std::string::npos) {
            size_t slash_pos = lookup_url.find('/', proto_pos + 3);
            if (slash_pos != std::string::npos) {
                domain_origin = lookup_url.substr(0, slash_pos);
            } else {
                domain_origin = lookup_url;
            }
        }

        // Compute directory base_url (e.g. "https://www.linkedin.com/in/")
        std::string base_url = domain_origin + "/";
        size_t query_pos = lookup_url.find('?');
        std::string clean_lookup = (query_pos != std::string::npos) ? lookup_url.substr(0, query_pos) : lookup_url;
        size_t last_slash = clean_lookup.find_last_of('/');
        if (last_slash != std::string::npos && last_slash >= 8) {
            base_url = clean_lookup.substr(0, last_slash + 1);
        }

        std::string html = doc.raw_html.empty() ? doc.clean_text : doc.raw_html;

        // Determine MIME Content-Type
        std::string ct = "text/html; charset=utf-8";
        std::string lower_target = target_url;
        for (auto& c : lower_target) c = (char)std::tolower((unsigned char)c);
        size_t q_pos = lower_target.find('?');
        if (q_pos != std::string::npos) lower_target = lower_target.substr(0, q_pos);
        size_t h_pos = lower_target.find('#');
        if (h_pos != std::string::npos) lower_target = lower_target.substr(0, h_pos);

        bool is_html_target = false;
        if (lower_target.size() >= 5 && lower_target.rfind(".json") == lower_target.size() - 5) ct = "application/json; charset=utf-8";
        else if (lower_target.size() >= 3 && lower_target.rfind(".js") == lower_target.size() - 3) ct = "application/javascript; charset=utf-8";
        else if (lower_target.size() >= 4 && lower_target.rfind(".mjs") == lower_target.size() - 4) ct = "application/javascript; charset=utf-8";
        else if (lower_target.size() >= 4 && lower_target.rfind(".css") == lower_target.size() - 4) ct = "text/css; charset=utf-8";
        else if (lower_target.size() >= 4 && lower_target.rfind(".png") == lower_target.size() - 4) ct = "image/png";
        else if (lower_target.size() >= 4 && lower_target.rfind(".jpg") == lower_target.size() - 4) ct = "image/jpeg";
        else if (lower_target.size() >= 5 && lower_target.rfind(".jpeg") == lower_target.size() - 5) ct = "image/jpeg";
        else if (lower_target.size() >= 4 && lower_target.rfind(".gif") == lower_target.size() - 4) ct = "image/gif";
        else if (lower_target.size() >= 5 && lower_target.rfind(".webp") == lower_target.size() - 5) ct = "image/webp";
        else if (lower_target.size() >= 4 && lower_target.rfind(".svg") == lower_target.size() - 4) ct = "image/svg+xml";
        else if (lower_target.size() >= 4 && lower_target.rfind(".ico") == lower_target.size() - 4) ct = "image/x-icon";
        else if (lower_target.size() >= 6 && lower_target.rfind(".woff2") == lower_target.size() - 6) ct = "font/woff2";
        else if (lower_target.size() >= 5 && lower_target.rfind(".woff") == lower_target.size() - 5) ct = "font/woff";
        else if (lower_target.size() >= 4 && lower_target.rfind(".ttf") == lower_target.size() - 4) ct = "font/ttf";
        else if (lower_target.size() >= 4 && lower_target.rfind(".xml") == lower_target.size() - 4) ct = "application/xml; charset=utf-8";
        else if (lower_target.size() >= 4 && lower_target.rfind(".txt") == lower_target.size() - 4) ct = "text/plain; charset=utf-8";
        else {
            is_html_target = true;
            ct = "text/html; charset=utf-8";
        }

        if (is_html_target) {
            // Strip or neutralize any restrictive CSP or X-Frame-Options meta tags
            std::string lower_doc = html;
            for (auto& c : lower_doc) c = (char)std::tolower((unsigned char)c);

            auto strip_meta_tag = [&](const std::string& pattern) {
                size_t mpos = 0;
                while ((mpos = lower_doc.find(pattern, mpos)) != std::string::npos) {
                    size_t tag_start = lower_doc.rfind("<meta", mpos);
                    if (tag_start != std::string::npos && mpos - tag_start < 250) {
                        size_t tag_end = lower_doc.find('>', mpos);
                        if (tag_end != std::string::npos) {
                            std::string spaces(tag_end - tag_start + 1, ' ');
                            html.replace(tag_start, tag_end - tag_start + 1, spaces);
                            lower_doc.replace(tag_start, tag_end - tag_start + 1, spaces);
                            mpos = tag_end + 1;
                            continue;
                        }
                    }
                    mpos += pattern.size();
                }
            };

            strip_meta_tag("content-security-policy");
            strip_meta_tag("x-frame-options");

            // Fix unquoted/undeclared solveSimpleChallenge checks in inline HTML handlers
            size_t ssc_pos = 0;
            while ((ssc_pos = html.find("solveSimpleChallenge", ssc_pos)) != std::string::npos) {
                if (ssc_pos >= 3 && html.substr(ssc_pos - 3, 3) == "if(") {
                    html.replace(ssc_pos, 20, "window.solveSimpleChallenge");
                    ssc_pos += 27;
                } else if (ssc_pos >= 4 && html.substr(ssc_pos - 4, 4) == "if (") {
                    html.replace(ssc_pos, 20, "window.solveSimpleChallenge");
                    ssc_pos += 27;
                } else {
                    ssc_pos += 20;
                }
            }

            // Neutralize LinkedIn location.host checks in inline 999 challenge scripts
            if (domain_origin.find("linkedin.com") != std::string::npos) {
                size_t lh_pos = 0;
                while ((lh_pos = html.find("location.host", lh_pos)) != std::string::npos) {
                    html.replace(lh_pos, 13, "\"www.linkedin.com\"");
                    lh_pos += 18;
                }
            }

            std::string clean_base_host = domain_origin;

            // Rewrite meta refresh tags to route through proxy
            size_t meta_refresh_pos = 0;
            std::string lower_for_meta = html;
            for (auto& c : lower_for_meta) c = (char)std::tolower((unsigned char)c);
            while ((meta_refresh_pos = lower_for_meta.find("url=/", meta_refresh_pos)) != std::string::npos) {
                size_t path_start = meta_refresh_pos + 4; // points to '/'
                size_t path_end = html.find_first_of("\"' >", path_start);
                if (path_end != std::string::npos) {
                    std::string rel_path = html.substr(path_start, path_end - path_start);
                    std::string proxied_url = "/api/proxy?url=" + moecher::tooling::url_encode(clean_base_host + rel_path);
                    html.replace(path_start, path_end - path_start, proxied_url);
                    lower_for_meta = html;
                    for (auto& c : lower_for_meta) c = (char)std::tolower((unsigned char)c);
                    meta_refresh_pos += proxied_url.size() + 4;
                } else {
                    meta_refresh_pos += 5;
                }
            }

            // Inject <base href="..."> and bulletproof JavaScript network & navigation proxy shims
            std::string lower_html = html;
            for (auto& c : lower_html) c = (char)std::tolower((unsigned char)c);

            if (lower_html.find("<html") != std::string::npos || lower_html.find("<body") != std::string::npos ||
                lower_html.find("<!doctype") != std::string::npos) {
                std::string shim = "<base href=\"" + base_url + "\">\n"
                                   "<script>\n"
                                   "(function(){\n"
                                   "  try {\n"
                                   "    Object.defineProperty(window, 'top', { get: function(){ return window.self; }, set: function(){} });\n"
                                   "    Object.defineProperty(window, 'parent', { get: function(){ return window.self; }, set: function(){} });\n"
                                   "    Object.defineProperty(window, 'frameElement', { get: function(){ return null; }, set: function(){} });\n"
                                   "  } catch(e){}\n"
                                   "  window.solveSimpleChallenge = window.solveSimpleChallenge || function(){};\n"
                                   "  var baseDir = \"" + base_url + "\";\n"
                                   "  var domainOrigin = \"" + domain_origin + "\";\n"
                                   "  var proxyOrigin = window.location.origin;\n"
                                   "  function isTelemetry(u){\n"
                                   "    if(!u || typeof u !== 'string') return false;\n"
                                   "    return u.indexOf('/li/track') !== -1 || u.indexOf('litms') !== -1 ||\n"
                                   "           u.indexOf('page-view-heartbeat') !== -1 || u.indexOf('pve-pat') !== -1 ||\n"
                                   "           u.indexOf('realtime') !== -1 || u.indexOf('csp-report') !== -1 ||\n"
                                   "           u.indexOf('google-analytics') !== -1 || u.indexOf('googletagmanager') !== -1 ||\n"
                                   "           u.indexOf('doubleclick.net') !== -1 || u.indexOf('clarity.ms') !== -1 ||\n"
                                   "           u.indexOf('facebook.net') !== -1;\n"
                                   "  }\n"
                                   "  function toProxy(u){\n"
                                   "    if(!u || typeof u !== 'string') return u;\n"
                                   "    if(u.startsWith('data:') || u.startsWith('blob:') || u.startsWith('javascript:') || u.startsWith('mailto:') || u.startsWith('tel:') || u.startsWith('about:')) return u;\n"
                                   "    if(u.indexOf(domainOrigin + '/api/proxy') !== -1){\n"
                                   "      u = u.replace(domainOrigin + '/api/proxy', proxyOrigin + '/api/proxy');\n"
                                   "    }\n"
                                   "    if(u.startsWith(proxyOrigin + '/api/proxy')) return u;\n"
                                   "    if(u.startsWith('/api/proxy')) return proxyOrigin + u;\n"
                                   "    var full = u;\n"
                                   "    if(proxyOrigin && full.startsWith(proxyOrigin)){\n"
                                   "      full = full.substring(proxyOrigin.length);\n"
                                   "    }\n"
                                   "    if(full.startsWith('//')) full = 'https:' + full;\n"
                                   "    else if(full.startsWith('/')) full = domainOrigin + full;\n"
                                   "    else if(full.indexOf('://') === -1) full = baseDir + full;\n"
                                   "    return proxyOrigin + '/api/proxy?url=' + encodeURIComponent(full);\n"
                                   "  }\n"
                                   "  if (window.fetch) {\n"
                                   "    var _origFetch = window.fetch;\n"
                                   "    window.fetch = function(resource, init) {\n"
                                   "      try {\n"
                                   "        var urlStr = typeof resource === 'string' ? resource : (resource && resource.url ? resource.url : '');\n"
                                   "        if (isTelemetry(urlStr)) {\n"
                                   "          return Promise.resolve(new Response('{\"status\":\"ok\"}', { status: 200, headers: { 'Content-Type': 'application/json' } }));\n"
                                   "        }\n"
                                   "        if (typeof resource === 'string') {\n"
                                   "          return _origFetch.call(this, toProxy(resource), init);\n"
                                   "        } else if (resource && resource.url) {\n"
                                   "          var newReq = new Request(toProxy(resource.url), resource);\n"
                                   "          return _origFetch.call(this, newReq, init);\n"
                                   "        }\n"
                                   "      } catch(e){}\n"
                                   "      return _origFetch.apply(this, arguments);\n"
                                   "    };\n"
                                   "  }\n"
                                   "  if (window.XMLHttpRequest) {\n"
                                   "    var _origOpen = XMLHttpRequest.prototype.open;\n"
                                   "    var _origSend = XMLHttpRequest.prototype.send;\n"
                                   "    XMLHttpRequest.prototype.open = function(method, url, async, user, password) {\n"
                                   "      this._isTele = isTelemetry(url);\n"
                                   "      try { url = toProxy(url); } catch(e){}\n"
                                   "      return _origOpen.call(this, method, url, async === undefined ? true : async, user, password);\n"
                                   "    };\n"
                                   "    XMLHttpRequest.prototype.send = function(body) {\n"
                                   "      if (this._isTele) {\n"
                                   "        var self = this;\n"
                                   "        setTimeout(function(){\n"
                                   "          try {\n"
                                   "            Object.defineProperty(self, 'readyState', { value: 4, writable: true });\n"
                                   "            Object.defineProperty(self, 'status', { value: 200, writable: true });\n"
                                   "            Object.defineProperty(self, 'responseText', { value: '{\"status\":\"ok\"}', writable: true });\n"
                                   "            if (typeof self.onreadystatechange === 'function') self.onreadystatechange();\n"
                                   "            if (typeof self.onload === 'function') self.onload();\n"
                                   "          }catch(e){}\n"
                                   "        }, 10);\n"
                                   "        return;\n"
                                   "      }\n"
                                   "      return _origSend.call(this, body);\n"
                                   "    };\n"
                                   "  }\n"
                                   "  window.addEventListener('message', function(e){\n"
                                   "    if (e.data && e.data.type === 'challengeAnswer') {\n"
                                   "      var challengeToken = e.data.token;\n"
                                   "      if (typeof window.solveSimpleChallenge === 'function') {\n"
                                   "        window.solveSimpleChallenge(challengeToken);\n"
                                   "      }\n"
                                   "    }\n"
                                   "  }, true);\n"
                                   "})();\n"
                                   "</script>\n";

                size_t head_pos = lower_html.find("<head");
                if (head_pos != std::string::npos) {
                    size_t tag_close = html.find('>', head_pos);
                    if (tag_close != std::string::npos) {
                        html.insert(tag_close + 1, "\n" + shim);
                    } else {
                        html.insert(head_pos + 5, "\n" + shim);
                    }
                } else {
                    size_t html_pos = lower_html.find("<html");
                    if (html_pos != std::string::npos) {
                        size_t tag_close = html.find('>', html_pos);
                        if (tag_close != std::string::npos) {
                            html.insert(tag_close + 1, "\n<head>" + shim + "</head>\n");
                        } else {
                            html.insert(html_pos + 5, "\n<head>" + shim + "</head>\n");
                        }
                    } else {
                        html = "<head>" + shim + "</head>\n" + html;
                    }
                }
            }
        }

        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE, PATCH, HEAD");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.set_header("Access-Control-Expose-Headers", "*");
        res.set_header("Access-Control-Allow-Credentials", "true");
        res.set_header("X-Frame-Options", "ALLOWALL");
        res.set_header("Content-Security-Policy", "frame-ancestors *;");

        res.set_content(html, ct);
    };

    svr.Get("/api/proxy", handle_proxy_request);
    svr.Post("/api/proxy", handle_proxy_request);
    svr.Options("/api/proxy", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE, PATCH, HEAD");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.set_header("Access-Control-Expose-Headers", "*");
        res.set_header("Access-Control-Allow-Credentials", "true");
        res.status = 204;
    });

    // Fallback catch-all handlers for third-party navigation (such as /authwall, /checkpoint, /uas/*, etc.)
    auto handle_catchall_proxy = [](const httplib::Request& req, httplib::Response& res) {
        std::string upstream_host = "https://www.linkedin.com";

        if (req.has_param("sessionRedirect")) {
            std::string redirect_param = req.get_param_value("sessionRedirect");
            size_t p_pos = redirect_param.find("://");
            if (p_pos != std::string::npos) {
                size_t s_pos = redirect_param.find('/', p_pos + 3);
                if (s_pos != std::string::npos) {
                    upstream_host = redirect_param.substr(0, s_pos);
                } else {
                    upstream_host = redirect_param;
                }
            }
        } else if (req.has_header("Referer")) {
            std::string referer = req.get_header_value("Referer");
            size_t proxy_idx = referer.find("/api/proxy?url=");
            if (proxy_idx != std::string::npos) {
                std::string embedded_url = moecher::tooling::url_decode(referer.substr(proxy_idx + 15));
                size_t p_pos = embedded_url.find("://");
                if (p_pos != std::string::npos) {
                    size_t s_pos = embedded_url.find('/', p_pos + 3);
                    if (s_pos != std::string::npos) upstream_host = embedded_url.substr(0, s_pos);
                    else upstream_host = embedded_url;
                }
            }
        }

        std::string query_str = "";
        if (!req.params.empty()) {
            bool first = true;
            for (const auto& kv : req.params) {
                query_str += (first ? "?" : "&") + moecher::tooling::url_encode(kv.first) + "=" + moecher::tooling::url_encode(kv.second);
                first = false;
            }
        }

        std::string full_target = upstream_host + req.path + query_str;
        std::string proxied_url = "/api/proxy?url=" + moecher::tooling::url_encode(full_target);

        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE, PATCH, HEAD");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.set_header("Access-Control-Expose-Headers", "*");
        res.set_header("Access-Control-Allow-Credentials", "true");
        res.set_header("Location", proxied_url);
        res.status = 302;
    };

    svr.Get("/authwall", handle_catchall_proxy);
    svr.Post("/authwall", handle_catchall_proxy);
    svr.Get(R"(/uas/.*)", handle_catchall_proxy);
    svr.Post(R"(/uas/.*)", handle_catchall_proxy);
    svr.Get(R"(/checkpoint/.*)", handle_catchall_proxy);
    svr.Post(R"(/checkpoint/.*)", handle_catchall_proxy);
    svr.Get(R"(/signup/.*)", handle_catchall_proxy);
    svr.Post(R"(/signup/.*)", handle_catchall_proxy);
    svr.Get(R"(/legal/.*)", handle_catchall_proxy);
    svr.Post(R"(/legal/.*)", handle_catchall_proxy);
}

} // namespace moecher::web
