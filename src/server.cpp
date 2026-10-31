#include "server.hpp"
#include "engine.hpp"
#include "logging.hpp"
#include "agentic.hpp"
#include "web_ui.hpp"
#include "multimodal.hpp"
#include "zdb.hpp"
#include "version.hpp"
#include "platform/platform_io.hpp"
#include "mcp_client.hpp"
#include "tool_exec.hpp"

#ifndef CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH
#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH (64 * 1024 * 1024)
#endif
#ifndef CPPHTTPLIB_PAYLOAD_MAX_LENGTH
#define CPPHTTPLIB_PAYLOAD_MAX_LENGTH (64 * 1024 * 1024)
#endif
#ifndef CPPHTTPLIB_HEADER_MAX_LENGTH
#define CPPHTTPLIB_HEADER_MAX_LENGTH (1024 * 1024)
#endif
#ifndef CPPHTTPLIB_REQUEST_URI_MAX_LENGTH
#define CPPHTTPLIB_REQUEST_URI_MAX_LENGTH (1024 * 1024)
#endif

#include <httplib.h>
#ifdef _WIN32
#include <wininet.h>
#endif

using json = nlohmann::json;
using moecher::zdb::ZeroDecodeCandidate;
using moecher::zdb::ZeroDecodeResult;

namespace moecher::server {

static std::mutex g_engine_mutex;  // serialize inference requests
static std::atomic<bool> g_engine_busy{false}; // active request lock

struct EngineBusyGuard {
    bool locked = false;
    EngineBusyGuard() {
        bool expected = false;
        locked = g_engine_busy.compare_exchange_strong(expected, true);
    }
    ~EngineBusyGuard() {
        if (locked) {
            g_engine_busy.store(false);
        }
    }
    bool is_locked() const { return locked; }
};

static int g_request_counter = 0;  // for unique request IDs
static json s_last_conv_messages = json::array();

// ── Persistent Authoritative Server Configuration ───────────────────────────
struct ServerConfigState {
    bool zdb_enabled = false;
    std::vector<std::string> zdb_choices = {"APPROVED", "REJECTED"};
    bool zdb_auto_fallback = true;
    float zdb_min_confidence = 0.70f;
    float zdb_max_top1_gap = 1.50f;

    float temperature = 0.6f;
    float repetition_penalty = -1.0f;
    std::string reasoning_effort = "medium";
    int thinking_budget = 4096;
    bool thinking_enabled = true;
    int max_tokens = 4096;

    std::vector<std::string> active_tools = {
        "web_search", "youtube_search", "google_search", "fetch_url",
        "create_3d_model", "read_file", "write_file", "edit_file", "execute_command"
    };

    json to_json() const {
        return json{
            {"zdb_enabled", zdb_enabled},
            {"zdb_choices", zdb_choices},
            {"zdb_auto_fallback", zdb_auto_fallback},
            {"zdb_min_confidence", zdb_min_confidence},
            {"zdb_max_top1_gap", zdb_max_top1_gap},
            {"temperature", temperature},
            {"repetition_penalty", repetition_penalty},
            {"reasoning_effort", reasoning_effort},
            {"thinking_budget", thinking_budget},
            {"thinking_enabled", thinking_enabled},
            {"max_tokens", max_tokens},
            {"active_tools", active_tools}
        };
    }

    void from_json(const json& j) {
        if (j.contains("zdb_enabled") && j["zdb_enabled"].is_boolean()) zdb_enabled = j["zdb_enabled"].get<bool>();
        if (j.contains("zdb_choices") && j["zdb_choices"].is_array()) {
            zdb_choices.clear();
            for (const auto& item : j["zdb_choices"]) {
                if (item.is_string() && !item.get<std::string>().empty()) zdb_choices.push_back(item.get<std::string>());
            }
            if (zdb_choices.empty()) zdb_choices = {"APPROVED", "REJECTED"};
        }
        if (j.contains("zdb_auto_fallback") && j["zdb_auto_fallback"].is_boolean()) zdb_auto_fallback = j["zdb_auto_fallback"].get<bool>();
        if (j.contains("zdb_min_confidence") && j["zdb_min_confidence"].is_number()) zdb_min_confidence = j["zdb_min_confidence"].get<float>();
        if (j.contains("zdb_max_top1_gap") && j["zdb_max_top1_gap"].is_number()) zdb_max_top1_gap = j["zdb_max_top1_gap"].get<float>();
        if (j.contains("temperature") && j["temperature"].is_number()) temperature = j["temperature"].get<float>();
        if (j.contains("repetition_penalty") && j["repetition_penalty"].is_number()) repetition_penalty = j["repetition_penalty"].get<float>();
        if (j.contains("reasoning_effort") && j["reasoning_effort"].is_string()) reasoning_effort = j["reasoning_effort"].get<std::string>();
        if (j.contains("thinking_budget") && j["thinking_budget"].is_number_integer()) thinking_budget = j["thinking_budget"].get<int>();
        if (j.contains("thinking_enabled") && j["thinking_enabled"].is_boolean()) thinking_enabled = j["thinking_enabled"].get<bool>();
        if (j.contains("max_tokens") && j["max_tokens"].is_number_integer()) max_tokens = j["max_tokens"].get<int>();
        if (j.contains("active_tools") && j["active_tools"].is_array()) {
            active_tools.clear();
            for (const auto& item : j["active_tools"]) {
                if (item.is_string()) active_tools.push_back(item.get<std::string>());
            }
        }
    }
};

static ServerConfigState g_server_config;
static std::mutex g_server_config_mutex;

static void load_server_config(const std::string& path = "server_config.json") {
    std::lock_guard<std::mutex> lock(g_server_config_mutex);
    try {
        std::ifstream f(path);
        if (f.is_open()) {
            json j = json::parse(f);
            g_server_config.from_json(j);
            LOG_INFO("[Config] Loaded persistent server configuration from %s (zdb_enabled=%s, temp=%.2f, rep_pen=%.2f)",
                     path.c_str(), g_server_config.zdb_enabled ? "true" : "false",
                     g_server_config.temperature, g_server_config.repetition_penalty);
        }
    } catch (const std::exception& e) {
        LOG_WARN("[Config] Could not parse %s: %s (using defaults)", path.c_str(), e.what());
    }
}

static void save_server_config(const std::string& path = "server_config.json") {
    std::lock_guard<std::mutex> lock(g_server_config_mutex);
    try {
        std::ofstream f(path);
        if (f.is_open()) {
            f << g_server_config.to_json().dump(2);
        }
    } catch (const std::exception& e) {
        LOG_WARN("[Config] Failed to save %s: %s", path.c_str(), e.what());
    }
}

void rebuild_system_prefix(
    MoecherEngine& engine,
    const std::string& base_prompt,
    const json& active_tools,
    const std::string& custom_full_prompt)
{
    std::lock_guard<std::mutex> lock(g_engine_mutex);

    json resolved_tools = (g_enable_tools && !active_tools.empty()) ? resolve_canonical_tools(active_tools) : json::array();

    std::string full_prompt_text;
    if (!custom_full_prompt.empty()) {
        full_prompt_text = custom_full_prompt;
    } else {
        bool is_qwen = (engine.cfg_.architecture == ModelArch::QWEN || engine.cfg_.architecture == ModelArch::NEMOTRON_35);
        if (g_enable_tools && !resolved_tools.empty()) {
            full_prompt_text = base_prompt + build_dynamic_tools_prompt(resolved_tools, is_qwen);
        } else {
            full_prompt_text = base_prompt;
        }
    }

    // If no system prompt and no active tools are configured, clear snapshot
    if (full_prompt_text.empty() && resolved_tools.empty()) {
        engine.reset_all_kv_caches();
        engine.system_kv_snapshot_.valid = false;
        engine.turn_kv_snapshot_.valid = false;
        s_last_conv_messages.clear();
        g_base_system_prompt.clear();
        g_current_system_prompt.clear();
        g_current_active_tools.clear();
        LOG_INFO("No system prompt or active tools configured; System KV Cache snapshot cleared.");
        return;
    }

    // Idempotency guard: if the system prompt and tools have not changed and snapshot is valid, skip redundant rebuild
    if (engine.system_kv_snapshot_.valid &&
        g_base_system_prompt == base_prompt &&
        g_current_system_prompt == full_prompt_text &&
        g_current_active_tools == resolved_tools) {
        LOG_INFO("System KV Cache snapshot already up-to-date (%zu tokens, %zu active tools). Skipping redundant rebuild.",
                 engine.system_kv_snapshot_.tokens.size(), resolved_tools.size());
        return;
    }

    // Invalidate conversation-level continuation snapshot so next query doesn't reuse stale prompt
    engine.turn_kv_snapshot_.valid = false;
    s_last_conv_messages.clear();

    g_base_system_prompt = base_prompt;
    g_current_system_prompt = full_prompt_text;
    g_current_active_tools = resolved_tools;

    json default_messages = json::array({
        {{"role", "system"}, {"content", full_prompt_text}},
        {{"role", "user"}, {"content", ""}}
    });

    std::vector<int> prompt = engine.apply_chat_template(default_messages, true, "high", resolved_tools);

    bool is_chatml = (engine.cfg_.architecture == ModelArch::QWEN || engine.cfg_.architecture == ModelArch::QWEN2 || engine.cfg_.architecture == ModelArch::NEMOTRON_35);
    int user_start = is_chatml
                         ? engine.tokenizer_.get_token_id("<|im_start|>")
                         : engine.tokenizer_.get_token_id("<｜User｜>");
    if (user_start < 0) {
        user_start = is_chatml ? 151644 : 128803;
    }
    size_t sys_len = prompt.size();
    for (size_t i = 1; i < prompt.size(); i++) {
        if (prompt[i] == user_start) {
            sys_len = i;
            break;
        }
    }
    if (sys_len > 0) {
        std::vector<int> sys_tokens(prompt.begin(), prompt.begin() + sys_len);
        LOG_INFO("Rebuilding System KV Cache snapshot (%zu tokens, %zu active tools)...",
                 sys_tokens.size(), resolved_tools.size());
        engine.reset_all_kv_caches();
        engine.prefill_prefix(sys_tokens);
        // Note: prefill_prefix() already calls snapshot_system_kv() internally on line 4060
        LOG_INFO("System KV Cache snapshot pinned successfully (%zu tokens).", sys_tokens.size());
    } else {
        engine.reset_all_kv_caches();
        engine.system_kv_snapshot_.valid = false;
        LOG_INFO("System KV Cache snapshot cleared.");
    }
}

static std::vector<int> build_continuation_prompt(
    const MoecherEngine& engine,
    const std::vector<json>& new_messages,
    bool enable_thinking,
    const std::string& reasoning_effort)
{
    const auto& tok = engine.tokenizer_;
    std::vector<int> result = engine.turn_kv_snapshot_.tokens;

    bool is_chatml = (engine.cfg_.architecture == ModelArch::QWEN || engine.cfg_.architecture == ModelArch::QWEN2 || engine.cfg_.architecture == ModelArch::NEMOTRON_35);
    if (is_chatml) {
        int IM_START = tok.get_token_id("<|im_start|>");
        int IM_END = tok.get_token_id("<|im_end|>");
        auto nl = tok.encode("\n");
        result.push_back(IM_END);
        result.insert(result.end(), nl.begin(), nl.end());
        for (const auto& msg : new_messages) {
            std::string role = msg.value("role", "user");
            std::string content = get_message_content_string(msg);
            result.push_back(IM_START);
            if (role == "tool" || role == "function") {
                auto u_enc = tok.encode("user\n<tool_response>\n" + content + "\n</tool_response>");
                result.insert(result.end(), u_enc.begin(), u_enc.end());
            } else {
                if (engine.cfg_.is_nemotron() && enable_thinking && reasoning_effort != "none") {
                    content += "\n\n{reasoning effort: efficient}";
                }
                auto u_enc = tok.encode("user\n" + content);
                result.insert(result.end(), u_enc.begin(), u_enc.end());
            }
            result.push_back(IM_END);
            result.insert(result.end(), nl.begin(), nl.end());
        }
        result.push_back(IM_START);
        auto asst_enc = tok.encode("assistant\n");
        result.insert(result.end(), asst_enc.begin(), asst_enc.end());
        if (engine.cfg_.is_nemotron()) {
            if (enable_thinking) {
                auto th_enc = tok.encode("<think>\n");
                result.insert(result.end(), th_enc.begin(), th_enc.end());
            } else {
                auto th_enc = tok.encode("<think></think>");
                result.insert(result.end(), th_enc.begin(), th_enc.end());
            }
        } else {
            if (enable_thinking) {
                auto th_enc = tok.encode("<think>\n");
                result.insert(result.end(), th_enc.begin(), th_enc.end());
            } else {
                auto th_enc = tok.encode("<think>\n\n</think>\n");
                result.insert(result.end(), th_enc.begin(), th_enc.end());
            }
        }
    } else {
        // DeepSeek V4
        int EOS = tok.get_token_id("<｜end of sentence｜>");
        if (EOS < 0) EOS = tok.get_token_id("<｜end\xe2\x96\x81of\xe2\x96\x81sentence｜>");
        if (EOS < 1) EOS = 1;

        int USER = tok.get_token_id("<｜User｜>");
        if (USER < 0) USER = 128803;

        int ASSISTANT = tok.get_token_id("<｜Assistant｜>");
        if (ASSISTANT < 0) ASSISTANT = 128804;

        int THINK_BEGIN = tok.get_token_id("<think>");
        if (THINK_BEGIN < 0) THINK_BEGIN = 128821;

        int THINK_END = tok.get_token_id("</think>");
        if (THINK_END < 0) THINK_END = 128822;

        result.push_back(EOS);
        for (const auto& msg : new_messages) {
            std::string role = msg.value("role", "user");
            std::string content = get_message_content_string(msg);
            result.push_back(USER);
            if (role == "tool" || role == "function") {
                auto enc = tok.encode("<tool_response>\n" + content + "\n</tool_response>");
                result.insert(result.end(), enc.begin(), enc.end());
            } else {
                auto enc = tok.encode(content);
                result.insert(result.end(), enc.begin(), enc.end());
            }
        }
        result.push_back(ASSISTANT);
        if (enable_thinking) {
            result.push_back(THINK_BEGIN);
        } else {
            result.push_back(THINK_END);
        }
    }
    return result;
}

static bool check_and_build_continuation(
    const MoecherEngine& engine,
    const json& current_messages,
    const json& last_conv_messages,
    bool enable_thinking,
    const std::string& reasoning_effort,
    std::vector<int>& out_prompt,
    bool had_image_originally = false)
{
    if (had_image_originally) {
        return false;
    }
    if (!engine.turn_kv_snapshot_.valid || engine.turn_kv_snapshot_.tokens.empty()) {
        return false;
    }
    if (last_conv_messages.empty() || current_messages.size() <= last_conv_messages.size()) {
        return false;
    }

    size_t prev_len = last_conv_messages.size();
    // Check if ALL previous messages match strictly in role AND content
    for (size_t i = 0; i < prev_len; i++) {
        std::string cur_role = current_messages[i].value("role", "");
        std::string last_role = last_conv_messages[i].value("role", "");
        if (cur_role != last_role) {
            return false;
        }
        std::string cur_content = get_message_content_string(current_messages[i]);
        std::string last_content = get_message_content_string(last_conv_messages[i]);
        if (cur_content != last_content) {
            return false;
        }
    }

    // Collect new messages (user / tool)
    std::vector<json> new_msgs;
    for (size_t i = prev_len; i < current_messages.size(); i++) {
        std::string role = current_messages[i].value("role", "");
        if (role != "user" && role != "tool" && role != "function") {
            return false;
        }
        // If any incoming message has multimodal image content or visual perception analysis,
        // bypass text-only continuation so that the visual observation is evaluated cleanly.
        std::string content = get_message_content_string(current_messages[i]);
        if (content.find("<visual_perception>") != std::string::npos || message_has_image(current_messages[i])) {
            return false;
        }
        new_msgs.push_back(current_messages[i]);
    }

    if (new_msgs.empty()) return false;

    out_prompt = build_continuation_prompt(engine, new_msgs, enable_thinking, reasoning_effort);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Context Window Pruning: Evict oldest turns when context limit is approached
// ─────────────────────────────────────────────────────────────────────────────
static json prune_messages_to_fit(
    const json& messages,
    const MoecherEngine& engine,
    int max_allowed_tokens,
    bool enable_thinking,
    const std::string& reasoning_effort,
    const json& tools)
{
    json pruned = messages;
    if (pruned.empty()) return pruned;

    std::vector<int> test_prompt = engine.apply_chat_template(pruned, enable_thinking, reasoning_effort, tools);
    if (test_prompt.size() <= (size_t)max_allowed_tokens) {
        return pruned;
    }

    LOG_WARN("[Context] Prompt length (%zu) exceeds context budget (%d). Pruning older turns...",
             test_prompt.size(), max_allowed_tokens);

    bool has_system = (pruned[0].value("role", "") == "system");
    size_t min_idx = has_system ? 1 : 0;

    // Prune oldest non-system messages
    while (pruned.size() > min_idx + 2) {
        pruned.erase(min_idx);
        test_prompt = engine.apply_chat_template(pruned, enable_thinking, reasoning_effort, tools);
        if (test_prompt.size() <= (size_t)max_allowed_tokens) {
            LOG_INFO("[Context] Pruning complete: retained %zu messages, prompt length %zu / %d",
                     pruned.size(), test_prompt.size(), max_allowed_tokens);
            return pruned;
        }
    }

    // If still too long, truncate oversized individual messages (e.g. huge file reads)
    while (test_prompt.size() > (size_t)max_allowed_tokens) {
        size_t largest_idx = 0;
        size_t largest_len = 0;
        for (size_t i = min_idx; i < pruned.size(); i++) {
            std::string content = get_message_content_string(pruned[i]);
            if (content.size() > largest_len) {
                largest_len = content.size();
                largest_idx = i;
            }
        }
        if (largest_len < 300) break; // Cannot truncate safely further

        std::string content = get_message_content_string(pruned[largest_idx]);
        size_t keep_len = (content.size() * 3) / 4;
        std::string notice = "\n[... Content truncated to fit context window ...]\n";
        if (keep_len > notice.size()) {
            pruned[largest_idx]["content"] = content.substr(0, keep_len - notice.size()) + notice;
        } else {
            pruned[largest_idx]["content"] = notice;
        }
        test_prompt = engine.apply_chat_template(pruned, enable_thinking, reasoning_effort, tools);
    }

    LOG_INFO("[Context] Pruning finished: final prompt length %zu / %d",
             test_prompt.size(), max_allowed_tokens);
    return pruned;
}

// Returns the length of the longest prefix of str that contains only complete, valid UTF-8 sequences.
// If str ends with an incomplete multi-byte sequence, that trailing partial sequence is excluded.
inline size_t valid_utf8_prefix_len(const std::string& str) {
    size_t i = 0;
    size_t len = str.size();
    size_t last_valid = 0;
    while (i < len) {
        unsigned char c = (unsigned char)str[i];
        if (c < 0x80) {
            i += 1;
            last_valid = i;
        } else if ((c & 0xE0) == 0xC0) {
            if (c < 0xC2) {
                i += 1;
                last_valid = i;
            } else if (i + 1 < len) {
                unsigned char c1 = (unsigned char)str[i + 1];
                if ((c1 & 0xC0) == 0x80) {
                    i += 2;
                    last_valid = i;
                } else {
                    i += 1;
                    last_valid = i;
                }
            } else {
                break;
            }
        } else if ((c & 0xF0) == 0xE0) {
            if (i + 2 < len) {
                unsigned char c1 = (unsigned char)str[i + 1];
                unsigned char c2 = (unsigned char)str[i + 2];
                if ((c1 & 0xC0) == 0x80 && (c2 & 0xC0) == 0x80) {
                    i += 3;
                    last_valid = i;
                } else {
                    i += 1;
                    last_valid = i;
                }
            } else if (i + 1 < len) {
                unsigned char c1 = (unsigned char)str[i + 1];
                if ((c1 & 0xC0) != 0x80) {
                    i += 1;
                    last_valid = i;
                } else {
                    break;
                }
            } else {
                break;
            }
        } else if ((c & 0xF8) == 0xF0) {
            if (c > 0xF4) {
                i += 1;
                last_valid = i;
            } else if (i + 3 < len) {
                unsigned char c1 = (unsigned char)str[i + 1];
                unsigned char c2 = (unsigned char)str[i + 2];
                unsigned char c3 = (unsigned char)str[i + 3];
                if ((c1 & 0xC0) == 0x80 && (c2 & 0xC0) == 0x80 && (c3 & 0xC0) == 0x80) {
                    i += 4;
                    last_valid = i;
                } else {
                    i += 1;
                    last_valid = i;
                }
            } else {
                break;
            }
        } else {
            i += 1;
            last_valid = i;
        }
    }
    return last_valid;
}

// ─────────────────────────────────────────────────────────────────────────────
// ToolTagStreamFilter: On-demand lookahead interceptor for streaming tool calls
// ─────────────────────────────────────────────────────────────────────────────
struct ToolTagStreamFilter {
    enum class State {
        STREAMING,
        BUFFERING_TAG,
        BUFFERING_TOOL
    };

    State state = State::STREAMING;
    std::string tag_candidate;
    std::string tool_buffer;
    std::string streamed_clean_text;
    bool in_tool_calls_container = false;
    std::function<bool(const std::string&)> on_status = nullptr;
    std::string current_tool_name;
    size_t last_reported_size = 0;

    bool feed(const std::string& text, const std::function<bool(const std::string&)>& emit_cb) {
        std::string safe_chunk;
        for (char c : text) {
            if (state == State::STREAMING) {
                if (c == '<') {
                    state = State::BUFFERING_TAG;
                    tag_candidate.clear();
                    tag_candidate.push_back(c);
                } else {
                    safe_chunk.push_back(c);
                }
            } else if (state == State::BUFFERING_TAG) {
                tag_candidate.push_back(c);

                if (c == '>') {
                    // Tag candidate is complete
                    if (is_opening_tool_tag(tag_candidate)) {
                        state = State::BUFFERING_TOOL;
                        if (is_container_tool_tag(tag_candidate)) {
                            in_tool_calls_container = true;
                        }
                        tool_buffer = tag_candidate;
                        tag_candidate.clear();
                        current_tool_name.clear();
                        last_reported_size = 0;
                        if (on_status) {
                            on_status("Preparing tool call & arguments...");
                        }
                    } else if (is_closing_or_standalone_tool_tag(tag_candidate)) {
                        // Stray / standalone tool tag (e.g. </tool_call>, <｜tool sep｜>, </｜DSML...>)
                        state = State::STREAMING;
                        tag_candidate.clear();
                    } else if (is_tool_tag_prefix(tag_candidate)) {
                        // Matched tool syntax
                        state = State::BUFFERING_TOOL;
                        tool_buffer = tag_candidate;
                        tag_candidate.clear();
                        current_tool_name.clear();
                        last_reported_size = 0;
                        if (on_status) {
                            on_status("Preparing tool call & arguments...");
                        }
                    } else {
                        // Normal HTML tag that happened to complete
                        safe_chunk += tag_candidate;
                        tag_candidate.clear();
                        state = State::STREAMING;
                    }
                } else if (is_tool_tag_prefix(tag_candidate)) {
                    // Still potentially a tool call tag, continue buffering
                } else {
                    // Diverged from tool tag syntax (e.g. <div>, <html>, <=, <!DOCTYPE)
                    if (c == '<') {
                        std::string prev = tag_candidate.substr(0, tag_candidate.size() - 1);
                        safe_chunk += prev;
                        tag_candidate = "<";
                    } else {
                        safe_chunk += tag_candidate;
                        tag_candidate.clear();
                        state = State::STREAMING;
                    }
                }
            } else if (state == State::BUFFERING_TOOL) {
                tool_buffer.push_back(c);

                // Early detection of tool name inside JSON arguments
                if (current_tool_name.empty() && tool_buffer.size() > 15) {
                    size_t npos = tool_buffer.find("\"name\"");
                    if (npos != std::string::npos) {
                        size_t colon = tool_buffer.find(':', npos);
                        if (colon != std::string::npos) {
                            size_t q1 = tool_buffer.find('"', colon);
                            if (q1 != std::string::npos) {
                                size_t q2 = tool_buffer.find('"', q1 + 1);
                                if (q2 != std::string::npos) {
                                    current_tool_name = tool_buffer.substr(q1 + 1, q2 - q1 - 1);
                                    if (on_status) {
                                        if (current_tool_name == "create_3d_model" || current_tool_name == "model_3d") {
                                            on_status("Generating 3D model geometry & Three.js code...");
                                        } else {
                                            on_status("Generating tool call: " + current_tool_name + "...");
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Periodic live progress reporting while tool arguments stream in
                if (tool_buffer.size() - last_reported_size >= 120) {
                    last_reported_size = tool_buffer.size();
                    if (on_status) {
                        int approx_tokens = (int)(tool_buffer.size() / 4);
                        if (current_tool_name == "create_3d_model" || current_tool_name == "model_3d") {
                            on_status("Generating 3D model geometry & Three.js code (" + std::to_string(approx_tokens) + " tokens)...");
                        } else if (!current_tool_name.empty()) {
                            on_status("Generating arguments for " + current_tool_name + " (" + std::to_string(approx_tokens) + " tokens)...");
                        } else {
                            on_status("Generating tool call parameters (" + std::to_string(approx_tokens) + " tokens)...");
                        }
                    }
                }

                if (in_tool_calls_container) {
                    if (ends_with_container_closing(tool_buffer)) {
                        state = State::STREAMING;
                        tool_buffer.clear();
                        current_tool_name.clear();
                        last_reported_size = 0;
                        in_tool_calls_container = false;
                    }
                } else {
                    if (ends_with_tool_closing(tool_buffer)) {
                        state = State::STREAMING;
                        tool_buffer.clear();
                        current_tool_name.clear();
                        last_reported_size = 0;
                    }
                }
            }
        }
        if (!safe_chunk.empty()) {
            streamed_clean_text += safe_chunk;
            return emit_cb(safe_chunk);
        }
        return true;
    }

    bool flush_remaining(const std::function<bool(const std::string&)>& emit_cb) {
        if (state == State::BUFFERING_TAG && !tag_candidate.empty()) {
            std::string to_flush = tag_candidate;
            tag_candidate.clear();
            state = State::STREAMING;
            // If it's a lonely '<' or '</', flush it.
            // If it's an uncompleted tool/DSML tag (e.g. "<tool_call" or "<｜DSML" or "< DSML"), suppress it!
            if (to_flush == "<" || to_flush == "</" || !is_tool_tag_prefix(to_flush)) {
                streamed_clean_text += to_flush;
                return emit_cb(to_flush);
            }
        }
        state = State::STREAMING;
        tool_buffer.clear();
        in_tool_calls_container = false;
        return true;
    }

private:
    static std::string to_lower_ascii(const std::string& s) {
        std::string res = s;
        for (char& ch : res) {
            if (ch >= 'A' && ch <= 'Z') ch = ch + ('a' - 'A');
        }
        return res;
    }

    static bool is_tool_tag_prefix(const std::string& s) {
        if (s.empty()) return false;
        if (s[0] != '<') return false;

        size_t idx = 1;
        if (idx < s.size() && s[idx] == '/') idx++;
        while (idx < s.size() && s[idx] == ' ') idx++;

        std::string rem = s.substr(idx);
        if (rem.empty()) return true;

        std::string rem_lower = to_lower_ascii(rem);

        static const std::vector<std::string> roots = {
            "tool_call",
            "tool_calls",
            "tools",
            "tool",
            "function_call",
            "call",
            "\xef\xbd\x9ctool call begin\xef\xbd\x9c",
            "\xef\xbd\x9ctool call end\xef\xbd\x9c",
            "\xef\xbd\x9ctool sep\xef\xbd\x9c",
            "\xef\xbd\x9ctool outputs begin\xef\xbd\x9c",
            "\xef\xbd\x9ctool outputs end\xef\xbd\x9c",
            "|tool call begin|",
            "|tool call end|",
            "|tool sep|",
            "|tool outputs begin|",
            "|tool outputs end|",
            "\xef\xbd\x9c" "dsml\xef\xbd\x9c" "tool_calls",
            "\xef\xbd\x9c" "dsml\xef\xbd\x9c" "invoke",
            "\xef\xbd\x9c" "dsml\xef\xbd\x9c" "parameter",
            "\xef\xbd\x9c" "dsml",
            "|dsml|tool_calls",
            "|dsml|invoke",
            "|dsml|parameter",
            "|dsml",
            "dsml:tool_calls",
            "dsml_tool_calls",
            "dsml:invoke",
            "dsml invoke",
            "dsml:parameter",
            "dsml"
        };

        for (const auto& root : roots) {
            if (rem_lower.size() <= root.size()) {
                if (root.compare(0, rem_lower.size(), rem_lower) == 0) return true;
            } else {
                if (rem_lower.compare(0, root.size(), root) == 0) return true;
            }
        }

        if (rem.size() == 1 && (unsigned char)rem[0] == 0xEF) return true;
        if (rem.size() == 2 && (unsigned char)rem[0] == 0xEF && (unsigned char)rem[1] == 0xBD) return true;

        return false;
    }

    static bool is_opening_tool_tag(const std::string& s) {
        std::string lower = to_lower_ascii(s);
        if (lower.find("</") != std::string::npos) return false;
        if (lower.find("end") != std::string::npos) return false;
        if (lower.find("sep") != std::string::npos) return false;

        if (lower.find("tool_call") != std::string::npos) return true;
        if (lower.find("tool_calls") != std::string::npos) return true;
        if (lower.find("tools") != std::string::npos) return true;
        if (lower.find("tool call begin") != std::string::npos) return true;
        if (lower.find("dsml") != std::string::npos) return true;
        if (lower.find("function_call") != std::string::npos) return true;
        if (lower.find("<call") != std::string::npos) return true;
        if (lower.find("<tool>") != std::string::npos || lower.find("<tool ") != std::string::npos) return true;

        return false;
    }

    static bool is_container_tool_tag(const std::string& s) {
        std::string lower = to_lower_ascii(s);
        if (lower.find("tool_calls") != std::string::npos) return true;
        if (lower.find("<tools>") != std::string::npos || lower.find("<tools ") != std::string::npos) return true;
        return false;
    }

    static bool is_closing_or_standalone_tool_tag(const std::string& s) {
        std::string lower = to_lower_ascii(s);
        if (lower.find("end") != std::string::npos) return true;
        if (lower.find("sep") != std::string::npos) return true;
        if (lower.find("</tool") != std::string::npos) return true;
        if (lower.find("</dsml") != std::string::npos) return true;
        if (lower.find("</\xef\xbd\x9c" "dsml") != std::string::npos) return true;
        if (lower.find("</|dsml") != std::string::npos) return true;
        if (lower.find("</call>") != std::string::npos) return true;
        if (lower.find("</function_call>") != std::string::npos) return true;
        return false;
    }

    static bool ends_with(const std::string& s, const std::string& suffix) {
        return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    static bool ends_with_container_closing(const std::string& s) {
        std::string lower = to_lower_ascii(s);
        static const std::vector<std::string> closings = {
            "</tool_calls>",
            "</tools>",
            "</\xef\xbd\x9c" "dsml\xef\xbd\x9c" "tool_calls>",
            "</|dsml|tool_calls>",
            "</dsml:tool_calls>",
            "</dsml_tool_calls>",
            "</dsml>",
            "</think>"
        };
        for (const auto& c : closings) {
            if (ends_with(lower, c)) return true;
        }
        return false;
    }

    static bool ends_with_tool_closing(const std::string& s) {
        std::string lower = to_lower_ascii(s);
        static const std::vector<std::string> closings = {
            "</tool_call>",
            "</tool_calls>",
            "</tools>",
            "</tool>",
            "</function_call>",
            "</call>",
            "<\xef\xbd\x9ctool call end\xef\xbd\x9c>",
            "<|tool call end|>",
            "<\xef\xbd\x9ctool outputs end\xef\xbd\x9c>",
            "<|tool outputs end|>",
            "</\xef\xbd\x9c" "dsml\xef\xbd\x9c" "tool_calls>",
            "</|dsml|tool_calls>",
            "</\xef\xbd\x9c" "dsml\xef\xbd\x9c" "invoke>",
            "</|dsml|invoke>",
            "</\xef\xbd\x9c" "dsml\xef\xbd\x9c" "parameter>",
            "</|dsml|parameter>",
            "</dsml:tool_calls>",
            "</dsml_tool_calls>",
            "</dsml>",
            "</think>"
        };
        for (const auto& c : closings) {
            if (ends_with(lower, c)) return true;
        }
        return false;
    }
};

static bool is_conversational_greeting(const std::string& text) {
    std::string trimmed = text;
    while (!trimmed.empty() && (isspace((unsigned char)trimmed.front()) || ispunct((unsigned char)trimmed.front()))) trimmed.erase(0, 1);
    while (!trimmed.empty() && (isspace((unsigned char)trimmed.back()) || ispunct((unsigned char)trimmed.back()))) trimmed.pop_back();
    std::string lower;
    for (char c : trimmed) lower += (char)tolower((unsigned char)c);
    return (lower == "hello" || lower == "hi" || lower == "hey" || lower == "greetings" ||
            lower == "good morning" || lower == "good afternoon" || lower == "good evening" ||
            lower == "howdy" || lower == "yo" || lower == "sup" ||
            lower == "who are you" || lower == "what is your name" || lower == "how are you" ||
            lower == "help" || lower == "test");
}

void run_server(MoecherEngine& engine, int port, int default_thinking_budget, int proxy_port, bool enable_forward_proxy, bool enable_system_proxy) {
    load_server_config("server_config.json");
    // Whenever the server is started, reset temp, repetition penalty, and anti-loop to proper model defaults
    nlohmann::json rec_sampling = engine.get_model_recommended_sampling();
    float model_default_temp = rec_sampling.value("temperature", engine.cfg_.default_temperature >= 0.0f ? engine.cfg_.default_temperature : 0.0f);
    float model_default_rep = rec_sampling.value("repetition_penalty", engine.cfg_.default_repetition_penalty > 0.0f ? engine.cfg_.default_repetition_penalty : 1.00f);
    bool model_anti_loop = rec_sampling.value("anti_loop", false);
    g_server_config.temperature = model_default_temp;
    g_server_config.repetition_penalty = model_default_rep;
    engine.enable_anti_loop_ = model_anti_loop;
    engine.default_enable_anti_loop_ = model_anti_loop;
    save_server_config("server_config.json");
    LOG_INFO("[Config] Reset server parameters on startup to model defaults: temp=%.2f, rep_penalty=%.2f, anti_loop=%s",
             g_server_config.temperature, g_server_config.repetition_penalty, model_anti_loop ? "enabled" : "disabled");
    httplib::Server svr;
    svr.set_tcp_nodelay(true);
    svr.set_payload_max_length(64 * 1024 * 1024); // 64 MB
    svr.set_socket_options([](socket_t sock) {
        int opt = 1;
#ifdef _WIN32
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt));
#else
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const void*>(&opt), sizeof(opt));
#endif
    });

    if (enable_forward_proxy) {
        moecher::proxy::start_forward_proxy(proxy_port);
    }

    // Note: MCP (Model Context Protocol) Manager was initialized prior to pre-warming in main()

#ifdef _WIN32
    if (enable_system_proxy) {
        moecher::proxy::set_windows_system_proxy(true, "127.0.0.1:" + std::to_string(proxy_port));
        SetConsoleCtrlHandler([](DWORD) -> BOOL {
            moecher::proxy::set_windows_system_proxy(false, "");
            moecher::proxy::stop_forward_proxy();
            return FALSE;
        }, TRUE);
        std::atexit([]() {
            moecher::proxy::set_windows_system_proxy(false, "");
            moecher::proxy::stop_forward_proxy();
        });
    }
#endif

    // CORS headers and preflight
    svr.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE");
        res.set_header("Access-Control-Allow-Headers", "*");

        if (req.method == "OPTIONS") {
            res.status = 200;
            return httplib::Server::HandlerResponse::Handled;
        }

        // Direct HTTP Forward Proxy fallback on main port if accessed via forward proxy
        std::string target_url = req.path;
        if (target_url.rfind("http://", 0) == 0 || target_url.rfind("https://", 0) == 0) {
            std::string client_ua = req.get_header_value("User-Agent");
            std::string client_lang = req.get_header_value("Accept-Language");
            std::string client_sec_ua = req.get_header_value("Sec-Ch-Ua");
            std::string client_sec_mobile = req.get_header_value("Sec-Ch-Ua-Mobile");
            std::string client_sec_platform = req.get_header_value("Sec-Ch-Ua-Platform");
            std::string client_cookies = req.get_header_value("Cookie");

            moecher::tooling::set_client_browser_context(client_ua, client_lang, client_sec_ua, client_sec_mobile, client_sec_platform, client_cookies);

            auto doc = moecher::tooling::fetch_url_full(target_url, 30000, 5000000, "raw");
            res.status = 200;
            std::string ct = "text/html; charset=utf-8";
            if (moecher::tooling::is_raw_code_url(target_url)) {
                if (target_url.find(".json") != std::string::npos) ct = "application/json; charset=utf-8";
                else if (target_url.find(".js") != std::string::npos) ct = "application/javascript; charset=utf-8";
                else if (target_url.find(".css") != std::string::npos) ct = "text/css; charset=utf-8";
                else ct = "text/plain; charset=utf-8";
            }
            res.set_content(doc.raw_html.empty() ? doc.clean_text : doc.raw_html, ct);
            return httplib::Server::HandlerResponse::Handled;
        }

        return httplib::Server::HandlerResponse::Unhandled;
    });

    // Web UI endpoints & reverse proxy (statically linked from src/web_ui.cpp)
    moecher::web::register_web_routes(svr);

    // Agentic & Tooling endpoints (statically linked from src/agentic.cpp)
    moecher::agentic::register_agentic_routes(svr);

    svr.Post("/api/kv/reset", [&engine](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        bool restored = false;
        {
            std::lock_guard<std::mutex> lock(g_engine_mutex);
            s_last_conv_messages.clear();
            engine.turn_kv_snapshot_.valid = false;
            if (engine.system_kv_snapshot_.valid) {
                engine.restore_system_kv();
                restored = true;
                LOG_INFO("[KV RESET] Restored pinned System KV snapshot of %zu tokens (ready for fresh conversation with 0ms prefill).",
                         engine.system_kv_snapshot_.tokens.size());
            } else {
                engine.reset_all_kv_caches();
                engine.cached_tokens_.clear();
                LOG_INFO("[KV RESET] Reset all KV caches to empty.");
            }
        }
        json body = {
            {"status", "ok"},
            {"message", "KV cache reset successfully"},
            {"system_snapshot_restored", restored},
            {"cached_tokens", engine.cached_tokens_.size()}
        };
        res.set_content(body.dump(), "application/json");
    });

    svr.Get("/api/kv/reset", [&engine](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        bool restored = false;
        {
            std::lock_guard<std::mutex> lock(g_engine_mutex);
            s_last_conv_messages.clear();
            engine.turn_kv_snapshot_.valid = false;
            if (engine.system_kv_snapshot_.valid) {
                engine.restore_system_kv();
                restored = true;
            } else {
                engine.reset_all_kv_caches();
                engine.cached_tokens_.clear();
            }
        }
        json body = {
            {"status", "ok"},
            {"system_snapshot_restored", restored},
            {"cached_tokens", engine.cached_tokens_.size()}
        };
        res.set_content(body.dump(), "application/json");
    });

    svr.Options("/api/kv/reset", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.status = 204;
    });

    // ── System Prompt & Tool Configuration Endpoints ────────────────────────
    svr.Get("/api/system/prompt", [&engine](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        json body = {
            {"status", "ok"},
            {"server_instance_id", g_server_instance_id},
            {"tools_enabled", g_enable_tools},
            {"base_prompt", g_base_system_prompt},
            {"system_prompt", g_current_system_prompt},
            {"active_tools", g_enable_tools ? g_current_active_tools : json::array()},
            {"repetition_penalty", g_server_config.repetition_penalty > 0.0f ? g_server_config.repetition_penalty : 1.00f},
            {"anti_loop", engine.default_enable_anti_loop_},
            {"tokens_count", engine.system_kv_snapshot_.valid ? engine.system_kv_snapshot_.tokens.size() : 0}
        };
        res.set_content(body.dump(), "application/json");
    });

    svr.Options("/api/system/prompt", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.status = 204;
    });

    // ── Persistent Configuration Endpoints ──────────────────────────────────
    svr.Get("/api/system/config", [&engine](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        json body;
        {
            std::lock_guard<std::mutex> lock(g_server_config_mutex);
            body = g_server_config.to_json();
        }
        body["status"] = "ok";
        body["server_instance_id"] = g_server_instance_id;
        body["model_id"] = engine.get_model_id();
        body["is_nemotron"] = engine.cfg_.is_nemotron();
        body["has_vision"] = engine.has_vision();
        body["system_prompt"] = g_current_system_prompt;
        body["base_prompt"] = g_base_system_prompt;
        body["tokens_count"] = engine.system_kv_snapshot_.valid ? engine.system_kv_snapshot_.tokens.size() : 0;
        res.set_content(body.dump(), "application/json");
    });

    svr.Post("/api/system/config", [&engine](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        try {
            json patch = json::parse(req.body);
            {
                std::lock_guard<std::mutex> lock(g_server_config_mutex);
                g_server_config.from_json(patch);
            }
            save_server_config("server_config.json");

            if (patch.contains("active_tools") || patch.contains("tools") || patch.contains("system_prompt") || patch.contains("base_prompt")) {
                std::string base_prompt = patch.value("base_prompt", g_base_system_prompt);
                std::string custom_full_prompt = patch.value("system_prompt", "");
                json tools = json::array();
                if (g_enable_tools) {
                    tools = patch.contains("active_tools") ? patch["active_tools"] : (patch.contains("tools") ? patch["tools"] : g_current_active_tools);
                }
                rebuild_system_prefix(engine, base_prompt, tools, custom_full_prompt);
            }

            json body;
            {
                std::lock_guard<std::mutex> lock(g_server_config_mutex);
                body = g_server_config.to_json();
            }
            body["status"] = "ok";
            body["server_instance_id"] = g_server_instance_id;
            body["tools_enabled"] = g_enable_tools;
            res.set_content(body.dump(), "application/json");
        } catch (const std::exception& e) {
            json err = {{"status", "error"}, {"message", e.what()}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
        }
    });

    svr.Options("/api/system/config", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.status = 204;
    });

    svr.Get("/api/system/configure", [&engine](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        json body = {
            {"status", "ok"},
            {"server_instance_id", g_server_instance_id},
            {"tools_enabled", g_enable_tools},
            {"base_prompt", g_base_system_prompt},
            {"system_prompt", g_current_system_prompt},
            {"active_tools", g_enable_tools ? g_current_active_tools : json::array()},
            {"repetition_penalty", g_server_config.repetition_penalty > 0.0f ? g_server_config.repetition_penalty : 1.00f},
            {"tokens_count", engine.system_kv_snapshot_.valid ? engine.system_kv_snapshot_.tokens.size() : 0}
        };
        res.set_content(body.dump(), "application/json");
    });

    svr.Post("/api/system/configure", [&engine](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        try {
            json j = json::parse(req.body);
            std::string base_prompt = j.value("base_prompt", g_base_system_prompt);
            std::string custom_full_prompt = j.value("system_prompt", j.value("custom_system_prompt", ""));

            json active_tools = json::array();
            if (g_enable_tools) {
                active_tools = g_current_active_tools;
                if (j.contains("tools")) {
                    active_tools = j["tools"];
                    if (active_tools.is_array() || active_tools.is_string()) {
                        active_tools = resolve_canonical_tools(active_tools);
                        // Retain discovered MCP tools unless caller explicitly passed an empty array []
                        if (!j["tools"].is_array() || !j["tools"].empty()) {
                            json mcp_tools = moecher::mcp::MCPManager::instance().get_openai_tools_schema();
                            for (const auto& mt : mcp_tools) {
                                active_tools.push_back(mt);
                            }
                        }
                    }
                    bool has_yt = false;
                    if (active_tools.is_array()) {
                        for (const auto& item : active_tools) {
                            if (item.is_string() && item.get<std::string>() == "youtube_search") has_yt = true;
                            else if (item.is_object() && item.contains("function") && item["function"].value("name", "") == "youtube_search") has_yt = true;
                        }
                    }
                    moecher::tooling::g_fast_media_search = has_yt;
                }
            }

            rebuild_system_prefix(engine, base_prompt, active_tools, custom_full_prompt);

            {
                std::lock_guard<std::mutex> lock(g_server_config_mutex);
                if (j.contains("repetition_penalty") && j["repetition_penalty"].is_number()) {
                    g_server_config.repetition_penalty = j["repetition_penalty"].get<float>();
                }
                if (g_enable_tools && active_tools.is_array()) {
                    g_server_config.active_tools.clear();
                    for (const auto& t : active_tools) {
                        if (t.is_string()) g_server_config.active_tools.push_back(t.get<std::string>());
                        else if (t.is_object() && t.contains("function") && t["function"].contains("name"))
                            g_server_config.active_tools.push_back(t["function"]["name"].get<std::string>());
                    }
                }
            }
            save_server_config("server_config.json");

            json body = {
                {"status", "ok"},
                {"server_instance_id", g_server_instance_id},
                {"tools_enabled", g_enable_tools},
                {"base_prompt", g_base_system_prompt},
                {"system_prompt", g_current_system_prompt},
                {"active_tools", g_enable_tools ? g_current_active_tools : json::array()},
                {"repetition_penalty", g_server_config.repetition_penalty > 0.0f ? g_server_config.repetition_penalty : 1.00f},
                {"tokens_count", engine.system_kv_snapshot_.valid ? engine.system_kv_snapshot_.tokens.size() : 0}
            };
            res.set_content(body.dump(), "application/json");
        } catch (const std::exception& e) {
            json err = {{"status", "error"}, {"message", e.what()}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
        }
    });

    svr.Options("/api/system/configure", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.status = 204;
    });

    auto get_system_resource_telemetry = []() -> json {
        json sys;
        
        // 1. GPU VRAM (CUDA)
        size_t free_vram = 0, total_vram = 0;
        cudaError_t cuda_err = cudaMemGetInfo(&free_vram, &total_vram);
        if (cuda_err == cudaSuccess && total_vram > 0) {
            sys["vram"] = {
                {"free_bytes", free_vram},
                {"total_bytes", total_vram},
                {"used_bytes", total_vram > free_vram ? total_vram - free_vram : 0},
                {"free_gb", (double)free_vram / (1024.0 * 1024.0 * 1024.0)},
                {"total_gb", (double)total_vram / (1024.0 * 1024.0 * 1024.0)},
                {"used_gb", (double)(total_vram > free_vram ? total_vram - free_vram : 0) / (1024.0 * 1024.0 * 1024.0)}
            };
        } else {
            sys["vram"] = {
                {"free_bytes", 0}, {"total_bytes", 0}, {"used_bytes", 0},
                {"free_gb", 0.0}, {"total_gb", 0.0}, {"used_gb", 0.0}
            };
        }

        // 2. System RAM
        size_t ram_total_bytes = 0, ram_available_bytes = 0;
#ifndef _WIN32
        std::ifstream meminfo("/proc/meminfo");
        if (meminfo.is_open()) {
            std::string line;
            size_t total_kb = 0, avail_kb = 0;
            while (std::getline(meminfo, line)) {
                if (line.rfind("MemTotal:", 0) == 0) {
                    sscanf(line.c_str(), "MemTotal: %zu kB", &total_kb);
                } else if (line.rfind("MemAvailable:", 0) == 0) {
                    sscanf(line.c_str(), "MemAvailable: %zu kB", &avail_kb);
                }
            }
            ram_total_bytes = total_kb * 1024;
            ram_available_bytes = avail_kb * 1024;
        }
#else
        MEMORYSTATUSEX memStatus;
        memStatus.dwLength = sizeof(memStatus);
        if (GlobalMemoryStatusEx(&memStatus)) {
            ram_total_bytes = memStatus.ullTotalPhys;
            ram_available_bytes = memStatus.ullAvailPhys;
        }
#endif
        sys["ram"] = {
            {"available_bytes", ram_available_bytes},
            {"total_bytes", ram_total_bytes},
            {"available_gb", (double)ram_available_bytes / (1024.0 * 1024.0 * 1024.0)},
            {"total_gb", (double)ram_total_bytes / (1024.0 * 1024.0 * 1024.0)}
        };

        // 3. SSD / Disk Storage
        size_t storage_available_bytes = 0, storage_total_bytes = 0;
        try {
            auto si = std::filesystem::space(".");
            storage_available_bytes = si.available;
            storage_total_bytes = si.capacity;
        } catch (...) {}
        sys["storage"] = {
            {"available_bytes", storage_available_bytes},
            {"total_bytes", storage_total_bytes},
            {"available_gb", (double)storage_available_bytes / (1024.0 * 1024.0 * 1024.0)},
            {"total_gb", (double)storage_total_bytes / (1024.0 * 1024.0 * 1024.0)}
        };

        return sys;
    };

    svr.Get("/v1/models", [&engine, get_system_resource_telemetry](const httplib::Request&, httplib::Response& res) {
        std::string model_id = engine.get_model_id();
        std::string display_name = engine.get_model_display_name();
        std::string arch_desc = engine.cfg_.is_nemotron()
            ? "Hybrid Mamba-2 SSM + GQA Attention + Sparse MoE (NVFP4)"
            : (engine.cfg_.is_deepseek_v41()
                ? "Sparse MoE (15,360 Experts) + Single-Pass Mega-mHC (NVFP4)"
                : ((engine.cfg_.architecture == ModelArch::QWEN)
                    ? "DeltaNet Linear Attention + MTP Speculative Decoding"
                    : (model_id.find("coder") != std::string::npos
                        ? "MoE Coder (64 Experts)"
                        : "Sparse MoE (11,008 Experts) + MLA Latent Attention")));
        json sys_res = get_system_resource_telemetry();
        json rec_sampling = engine.get_model_recommended_sampling();
        json model_entry = {
            {"id", model_id},
            {"name", display_name},
            {"display_name", display_name},
            {"object", "model"},
            {"owned_by", "moecher"},
            {"architecture", arch_desc},
            {"max_context_length", engine.cfg_.max_seq_len},
            {"has_vision", engine.supports_vision()},
            {"has_vision_delegate", engine.has_vision_delegate()},
            {"default_system_prompt", engine.get_model_default_system_prompt()},
            {"recommended_sampling", rec_sampling},
            {"active", true},
            {"resources", sys_res}
        };
        json body = {
            {"object", "list"},
            {"data", {model_entry}},
            {"recommended_sampling", rec_sampling},
            {"resources", sys_res}
        };
        res.set_content(body.dump(), "application/json");
    });

    svr.Get("/api/model", [&engine, get_system_resource_telemetry](const httplib::Request&, httplib::Response& res) {
        std::string model_id = engine.get_model_id();
        std::string display_name = engine.get_model_display_name();
        std::string arch_desc = engine.cfg_.is_nemotron()
            ? "Hybrid Mamba-2 SSM + GQA Attention + Sparse MoE (NVFP4)"
            : (engine.cfg_.is_deepseek_v41()
                ? "Sparse MoE (15,360 Experts) + Single-Pass Mega-mHC (NVFP4)"
                : ((engine.cfg_.architecture == ModelArch::QWEN)
                    ? "DeltaNet Linear Attention + MTP Speculative Decoding"
                    : (model_id.find("coder") != std::string::npos
                        ? "MoE Coder (64 Experts)"
                        : "Sparse MoE (11,008 Experts) + MLA Latent Attention")));
        json sys_res = get_system_resource_telemetry();
        json rec_sampling = engine.get_model_recommended_sampling();
        json body = {
            {"id", model_id},
            {"name", display_name},
            {"display_name", display_name},
            {"architecture", arch_desc},
            {"max_context_length", engine.cfg_.max_seq_len},
            {"vocab_size", engine.cfg_.vocab_size},
            {"max_tool_rounds", g_max_tool_rounds},
            {"has_vision", engine.supports_vision()},
            {"has_vision_delegate", engine.has_vision_delegate()},
            {"default_system_prompt", engine.get_model_default_system_prompt()},
            {"recommended_sampling", rec_sampling},
            {"active", true},
            {"resources", sys_res}
        };
        res.set_content(body.dump(), "application/json");
    });

    svr.Get("/api/system/resources", [get_system_resource_telemetry](const httplib::Request&, httplib::Response& res) {
        json sys_res = get_system_resource_telemetry();
        res.set_content(sys_res.dump(), "application/json");
    });

    // Expert Specialization & Profile endpoints
    svr.Get("/v1/experts/profile", [&engine](const httplib::Request&, httplib::Response& res) {
        json profile = engine.generate_expert_profile_json();
        res.set_content(profile.dump(), "application/json");
    });
    svr.Get("/v1/experts/stats", [&engine](const httplib::Request&, httplib::Response& res) {
        json profile = engine.generate_expert_profile_json();
        res.set_content(profile.dump(), "application/json");
    });


    // ── Zero Decode Bypass (ZDB) Endpoints ───────────────────────────────────
    auto handle_decide_request = [&engine](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "*");

        auto busy_guard = std::make_shared<EngineBusyGuard>();
        if (!busy_guard->is_locked()) {
            res.status = 429;
            res.set_content("{\"error\":\"Engine is currently busy processing another request.\"}", "application/json");
            return;
        }

        json request;
        try { request = json::parse(req.body); }
        catch (...) {
            res.status = 400;
            res.set_content("{\"error\":\"Invalid JSON\"}", "application/json");
            return;
        }

        std::vector<std::string> choices;
        if (request.contains("choices") && request["choices"].is_array()) {
            for (const auto& c : request["choices"]) {
                if (c.is_string()) choices.push_back(c.get<std::string>());
            }
        } else if (request.contains("decision_choices") && request["decision_choices"].is_array()) {
            for (const auto& c : request["decision_choices"]) {
                if (c.is_string()) choices.push_back(c.get<std::string>());
            }
        }

        if (choices.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"Missing or empty 'choices' array for Zero Decode Bypass\"}", "application/json");
            return;
        }

        float temperature = request.value("temperature", 1.0f);
        std::string format = request.value("format", "json");
        bool bypass_prefix_cache = request.value("bypass_prefix_cache", false);
        bool auto_fallback = request.value("auto_fallback", false);
        float min_confidence = request.value("min_confidence", 0.70f);
        float max_top1_gap = request.value("max_top1_gap", 1.5f);

        std::vector<int> prompt;
        if (request.contains("messages") && request["messages"].is_array()) {
            prompt = engine.apply_chat_template(request["messages"], /*enable_thinking=*/false, /*reasoning_effort=*/"none", json());
        } else if (request.contains("prompt") && request["prompt"].is_string()) {
            prompt = engine.tokenizer_.encode(request["prompt"].get<std::string>());
        }

        if (prompt.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"Missing or empty 'messages' or 'prompt'\"}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(g_engine_mutex);
        auto result = engine.execute_zero_decode_bypass(prompt, choices, temperature, format, bypass_prefix_cache, min_confidence, max_top1_gap);

        if (auto_fallback && !result.gate_passed) {
            LOG_INFO("[ZDB-Arbiter] /v1/decide: Gate REJECTED System 1 fast path (top1_gap=%.2f, conf=%.2f [min=%.2f]). Falling back to System 2 decode...",
                     result.top1_gap, result.confidence, min_confidence);
            std::string sys2_text = engine.generate(prompt, 512, temperature);
            while (!sys2_text.empty() && (sys2_text.front() == ' ' || sys2_text.front() == '\n' || sys2_text.front() == '\r')) sys2_text.erase(0, 1);
            while (!sys2_text.empty() && (sys2_text.back() == ' ' || sys2_text.back() == '\n' || sys2_text.back() == '\r')) sys2_text.pop_back();

            std::string matched_choice = result.winning_choice;
            for (const auto& ch : choices) {
                if (sys2_text.find(ch) != std::string::npos) {
                    matched_choice = ch;
                    break;
                }
            }

            if (format == "text" || format == "choice") {
                res.set_content(matched_choice, "text/plain");
            } else {
                json out_json = moecher::zdb::build_zdb_fallback_response_json(result, matched_choice, sys2_text, engine.completion_token_count_);
                res.set_content(out_json.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
            }
            return;
        }

        if (format == "text" || format == "choice") {
            res.set_content(result.rendered_output, "text/plain");
        } else {
            res.set_content(result.rendered_output, "application/json");
        }
    };

    svr.Post("/v1/decide", handle_decide_request);
    svr.Post("/v1/classify", handle_decide_request);
    svr.Options("/v1/decide", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.status = 204;
    });
    svr.Options("/v1/classify", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.status = 204;
    });

    // Chat completions
    svr.Post("/v1/chat/completions",
        [&engine, default_thinking_budget](const httplib::Request& req, httplib::Response& res) {
            auto busy_guard = std::make_shared<EngineBusyGuard>();
            if (!busy_guard->is_locked()) {
                LOG_WARN("Rejecting /v1/chat/completions: engine is currently busy processing another request");
                res.status = 429;
                res.set_content("{\"error\":{\"message\":\"Engine is currently busy processing another request. Please wait.\",\"type\":\"busy\",\"code\":429}}", "application/json");
                return;
            }

            g_stop_requested.store(false);
            auto start = std::chrono::steady_clock::now();

            json request;
            try { request = json::parse(req.body); }
            catch (...) {
                res.status = 400;
                res.set_content("{\"error\":\"Invalid JSON\"}", "application/json");
                return;
            }

            // Capture and propagate active client browser context (User-Agent, Accept-Language, Client Hints, Cookies)
            std::string client_ua = req.get_header_value("X-Client-User-Agent");
            if (client_ua.empty()) client_ua = req.get_header_value("User-Agent");
            if (client_ua.empty() && request.contains("client_context") && request["client_context"].contains("user_agent")) {
                client_ua = request["client_context"].value("user_agent", "");
            }

            std::string client_lang = req.get_header_value("X-Client-Accept-Language");
            if (client_lang.empty()) client_lang = req.get_header_value("Accept-Language");
            if (client_lang.empty() && request.contains("client_context") && request["client_context"].contains("languages")) {
                client_lang = request["client_context"].value("languages", "");
            }

            std::string client_sec_ua = req.get_header_value("Sec-Ch-Ua");
            if (client_sec_ua.empty() && request.contains("client_context") && request["client_context"].contains("sec_ch_ua")) {
                client_sec_ua = request["client_context"].value("sec_ch_ua", "");
            }

            std::string client_sec_mobile = req.get_header_value("Sec-Ch-Ua-Mobile");
            if (client_sec_mobile.empty() && request.contains("client_context") && request["client_context"].contains("mobile")) {
                client_sec_mobile = request["client_context"].value("mobile", "");
            }

            std::string client_sec_platform = req.get_header_value("Sec-Ch-Ua-Platform");
            if (client_sec_platform.empty() && request.contains("client_context") && request["client_context"].contains("platform")) {
                client_sec_platform = request["client_context"].value("platform", "");
            }

            std::string client_cookies = req.get_header_value("X-Client-Cookie");
            if (client_cookies.empty()) client_cookies = req.get_header_value("Cookie");
            if (client_cookies.empty() && request.contains("client_context") && request["client_context"].contains("cookies")) {
                client_cookies = request["client_context"].value("cookies", "");
            }

            if (!client_ua.empty() || !client_cookies.empty()) {
                moecher::tooling::set_client_browser_context(client_ua, client_lang, client_sec_ua, client_sec_mobile, client_sec_platform, client_cookies);
            }

            // Log request
            LOG_INFO("REQUEST: %s", req.body.c_str());
            {
                static std::mutex s_log_mutex;
                std::lock_guard<std::mutex> log_lock(s_log_mutex);
                std::ofstream log_file("client_request.log", std::ios::app);
                if (log_file.is_open()) {
                    auto now = std::chrono::system_clock::now();
                    std::time_t t = std::chrono::system_clock::to_time_t(now);
                    char time_buf[64];
                    std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
                    log_file << "[" << time_buf << "] " << req.body << std::endl;
                }
            }

            auto& messages = request["messages"];
            float temperature = request.value("temperature", 1.0f);
            float top_p = request.value("top_p", 0.95f);
            float min_p = request.value("min_p", engine.cfg_.is_deepseek() ? 0.0f : 0.05f);
            int top_k = request.value("top_k", 1024);
            int max_tokens = request.value("max_tokens", 20000);
            bool stream = request.value("stream", false);
            nlohmann::json rec = engine.get_model_recommended_sampling();
            float default_rep = g_server_config.repetition_penalty > 0.0f ? g_server_config.repetition_penalty :
                (engine.cfg_.default_repetition_penalty > 0.0f ? engine.cfg_.default_repetition_penalty :
                rec.value("repetition_penalty", 1.00f));
            float repetition_penalty = request.value("repetition_penalty", default_rep);
            float presence_penalty = request.value("presence_penalty", 0.0f);
            float frequency_penalty = request.value("frequency_penalty", 0.0f);
            if (repetition_penalty <= 1.0f && (presence_penalty > 0.0f || frequency_penalty > 0.0f)) {
                repetition_penalty = 1.0f + std::max(presence_penalty, frequency_penalty) * 0.1f;
            }
            std::vector<std::string> stop_words;
            if (request.contains("stop")) {
                if (request["stop"].is_string()) {
                    std::string s = request["stop"].get<std::string>();
                    if (!s.empty()) stop_words.push_back(s);
                } else if (request["stop"].is_array()) {
                    for (const auto& item : request["stop"]) {
                        if (item.is_string()) {
                            std::string s = item.get<std::string>();
                            if (!s.empty()) stop_words.push_back(s);
                        }
                    }
                }
            }
            if (messages.is_array() && !messages.empty() && messages[0].value("role", "") == "system") {
                std::string sys_txt = messages[0].value("content", "");
                if (sys_txt.empty()) {
                    messages.erase(0);
                }
            }
            std::string reasoning_effort = request.value("reasoning_effort", "high");
            bool enable_thinking = true;
            int max_thinking_tokens = default_thinking_budget;

            json tools = json::array();
            if (g_enable_tools) {
                if (request.contains("tools")) {
                    if (request["tools"].is_array()) {
                        tools = resolve_canonical_tools(request["tools"]);
                    } else if (request["tools"].is_string()) {
                        tools = resolve_canonical_tools(request["tools"]);
                    }
                    // If the request explicitly passed an empty array "tools": [], honor it (user disabled all tools)
                    if (!request["tools"].is_array() || !request["tools"].empty()) {
                        json mcp_tools = moecher::mcp::MCPManager::instance().get_openai_tools_schema();
                        for (const auto& mt : mcp_tools) {
                            tools.push_back(mt);
                        }
                    }
                } else if (request.contains("enable_tools") && request["enable_tools"].is_boolean()) {
                    if (request["enable_tools"].get<bool>()) {
                        tools = resolve_canonical_tools("default");
                        json mcp_tools = moecher::mcp::MCPManager::instance().get_openai_tools_schema();
                        for (const auto& mt : mcp_tools) {
                            tools.push_back(mt);
                        }
                    }
                } else {
                    // Fallback to active tools configured via /api/system/configure
                    tools = g_current_active_tools;
                }
            }

            if (request.contains("enable_thinking") && request["enable_thinking"].is_boolean()) {
                enable_thinking = request["enable_thinking"].get<bool>();
            }

            engine.enable_tree_mtp_ = engine.default_enable_tree_mtp_;
            engine.tree_nodes_ = engine.default_tree_nodes_;
            engine.enable_mtp_ = engine.default_enable_mtp_;
            engine.mtp_k_ = engine.default_mtp_k_;
            engine.enable_spec_ = engine.default_enable_spec_;
            if (request.contains("enable_spec") && request["enable_spec"].is_boolean()) {
                engine.enable_spec_ = request["enable_spec"].get<bool>();
            }
            if (request.contains("enable_mtp") && request["enable_mtp"].is_boolean()) {
                engine.enable_mtp_ = request["enable_mtp"].get<bool>();
            } else if (request.contains("mtp") && request["mtp"].is_boolean()) {
                engine.enable_mtp_ = request["mtp"].get<bool>();
            }
            if (request.contains("mtp_k") && request["mtp_k"].is_number_integer()) {
                int req_k = request["mtp_k"].get<int>();
                engine.mtp_k_ = (req_k > 0) ? std::clamp(req_k, 1, 8) : engine.default_mtp_k_;
            }
            if (request.contains("tree_mtp") && request["tree_mtp"].is_boolean()) {
                engine.enable_tree_mtp_ = request["tree_mtp"].get<bool>();
            } else if (request.contains("enable_tree_mtp") && request["enable_tree_mtp"].is_boolean()) {
                engine.enable_tree_mtp_ = request["enable_tree_mtp"].get<bool>();
            }
            if (request.contains("tree_nodes") && request["tree_nodes"].is_number_integer()) {
                engine.tree_nodes_ = std::clamp(request["tree_nodes"].get<int>(), 1, 4);
            } else if (request.contains("tree_branches") && request["tree_branches"].is_number_integer()) {
                engine.tree_nodes_ = std::clamp(request["tree_branches"].get<int>(), 1, 4);
            }

            engine.enable_anti_loop_ = engine.default_enable_anti_loop_;
            if (request.contains("enable_anti_loop") && request["enable_anti_loop"].is_boolean()) {
                engine.enable_anti_loop_ = request["enable_anti_loop"].get<bool>();
            } else if (request.contains("anti_loop") && request["anti_loop"].is_boolean()) {
                engine.enable_anti_loop_ = request["anti_loop"].get<bool>();
            }

            if (reasoning_effort == "none") {
                enable_thinking = false;
                max_thinking_tokens = 0;
            }

            if (request.contains("thinking")) {
                if (request["thinking"].is_boolean()) {
                    enable_thinking = request["thinking"].get<bool>();
                    if (!enable_thinking) max_thinking_tokens = 0;
                } else if (request["thinking"].is_object()) {
                    if (request["thinking"].contains("type") && request["thinking"]["type"] == "disabled") {
                        enable_thinking = false;
                        max_thinking_tokens = 0;
                    }
                    if (enable_thinking && request["thinking"].contains("budget_tokens") && request["thinking"]["budget_tokens"].is_number_integer()) {
                        max_thinking_tokens = request["thinking"]["budget_tokens"].get<int>();
                        if (max_thinking_tokens <= 0) enable_thinking = false;
                    }
                }
            }
            if (request.contains("chat_template_kwargs") && request["chat_template_kwargs"].is_object()) {
                if (request["chat_template_kwargs"].contains("thinking") && request["chat_template_kwargs"]["thinking"].is_boolean()) {
                    enable_thinking = request["chat_template_kwargs"]["thinking"].get<bool>();
                    if (!enable_thinking) max_thinking_tokens = 0;
                }
            }
            if (enable_thinking && request.contains("max_thinking_tokens") && request["max_thinking_tokens"].is_number_integer()) {
                max_thinking_tokens = request["max_thinking_tokens"].get<int>();
                if (max_thinking_tokens <= 0) enable_thinking = false;
            }
            if (enable_thinking && request.contains("thinking_budget") && request["thinking_budget"].is_number_integer()) {
                max_thinking_tokens = request["thinking_budget"].get<int>();
                if (max_thinking_tokens <= 0) enable_thinking = false;
            }

            if (reasoning_effort == "none" || !enable_thinking || max_thinking_tokens <= 0) {
                enable_thinking = false;
                max_thinking_tokens = 0;
                reasoning_effort = "none";
            }

            bool req_has_image = false;
            if (messages.is_array()) {
                for (const auto& msg : messages) {
                    if (message_has_image(msg)) { req_has_image = true; break; }
                }
            }
            bool had_image_originally = req_has_image;

            if (engine.has_vision_delegate() && req_has_image && messages.is_array()) {
                std::lock_guard<std::mutex> lock(g_engine_mutex);
                for (size_t msg_idx = 0; msg_idx < messages.size(); ++msg_idx) {
                    auto& msg = messages[msg_idx];
                    if (message_has_image(msg)) {
                        std::string cur_text = get_message_content_string(msg);
                        bool already_has_perception = (cur_text.find("<visual_perception>") != std::string::npos);
                        bool is_historical_turn = (msg_idx + 1 < messages.size() && messages[msg_idx + 1].value("role", "") == "assistant");

                        if (!already_has_perception && !is_historical_turn) {
                            std::string img_b64 = extract_image_b64_from_message(msg);
                            if (!img_b64.empty()) {
                                LOG_INFO("[Vision Delegation] Delegating visual perception (%zu b64 bytes) to Qwen2.5-VL...", img_b64.size());
                                std::string perception = engine.describe_image(img_b64, cur_text);
                                if (!perception.empty()) {
                                    if (cur_text.empty()) {
                                        cur_text = "Please describe the image in detail.";
                                    }
                                    msg["content"] = "[Visual Perception Subsystem Analysis]:\n<visual_perception>\n" + perception + "\n</visual_perception>\n\nBased on the visual observation above, answer the user request:\n" + cur_text;
                                    if (msg.contains("image")) msg.erase("image");
                                    if (msg.contains("image_url")) msg.erase("image_url");
                                    LOG_INFO("[Vision Delegation] Injected perception (%zu chars)", perception.size());
                                }
                            }
                        } else {
                            // Historical message or already processed: strip raw image to prevent re-delegation and save memory/context
                            if (!cur_text.empty()) {
                                msg["content"] = cur_text;
                            }
                            if (msg.contains("image")) msg.erase("image");
                            if (msg.contains("image_url")) msg.erase("image_url");
                        }
                    }
                }
                // If a new image was analyzed in this turn, clean bulky <visual_perception> blocks
                // from prior historical turns to prevent earlier image context from competing with or
                // cross-contaminating the current turn's visual perception.
                if (had_image_originally) {
                    for (size_t msg_idx = 0; msg_idx + 1 < messages.size(); ++msg_idx) {
                        std::string prior_text = get_message_content_string(messages[msg_idx]);
                        if (prior_text.find("<visual_perception>") != std::string::npos) {
                            size_t vp_start = prior_text.find("<visual_perception>");
                            size_t vp_end = prior_text.find("</visual_perception>");
                            if (vp_start != std::string::npos && vp_end != std::string::npos) {
                                vp_end += 20; // length of </visual_perception>
                                std::string marker = "answer the user request:\n";
                                size_t q_pos = prior_text.find(marker, vp_end);
                                std::string user_q = "";
                                if (q_pos != std::string::npos) {
                                    user_q = prior_text.substr(q_pos + marker.size());
                                }
                                messages[msg_idx]["content"] = "[User attached an image in prior turn]\n" + (user_q.empty() ? "" : user_q);
                            }
                        }
                    }
                }
                // Re-evaluate req_has_image now that delegated images are in text format
                req_has_image = false;
                for (const auto& msg : messages) {
                    if (message_has_image(msg)) { req_has_image = true; break; }
                }
            }

            if ((req_has_image || had_image_originally) && !request.contains("enable_thinking") && !request.contains("thinking") && !request.contains("reasoning_effort")) {
                enable_thinking = false;
                max_thinking_tokens = 0;
            }
            if ((req_has_image || had_image_originally) && !request.contains("tools")) {
                tools = json();
            }

            if (!enable_thinking || reasoning_effort == "none" || max_thinking_tokens <= 0) {
                enable_thinking = false;
                max_thinking_tokens = 0;
                reasoning_effort = "none";
            }

            int execution_timeout_ms = 60000;
            if (request.contains("execution_timeout_ms") && request["execution_timeout_ms"].is_number()) {
                execution_timeout_ms = request["execution_timeout_ms"].get<int>();
            } else if (request.contains("execution_timeout_sec") && request["execution_timeout_sec"].is_number()) {
                execution_timeout_ms = request["execution_timeout_sec"].get<int>() * 1000;
            }
            if (execution_timeout_ms <= 0) execution_timeout_ms = 60000;

            bool require_external_authorization = request.value("require_external_authorization", true);
            bool workspace_boundary_enforced = request.value("workspace_boundary_enforced", true);
            std::vector<std::string> authorized_paths;
            if (request.contains("authorized_paths") && request["authorized_paths"].is_array()) {
                for (const auto& ap : request["authorized_paths"]) {
                    if (ap.is_string()) authorized_paths.push_back(ap.get<std::string>());
                }
            }

            bool client_tool_exec = false;
            if (req.has_header("X-Client-Tool-Execution") && req.get_header_value("X-Client-Tool-Execution") == "true") {
                client_tool_exec = true;
            }
            if (request.contains("client_tool_execution") && request["client_tool_execution"].is_boolean()) {
                client_tool_exec = request["client_tool_execution"].get<bool>();
            }
            bool do_server_exec = g_server_exec && (!client_tool_exec || g_headless_browsing);

            int req_max_tool_rounds = g_max_tool_rounds;
            if (request.contains("max_tool_rounds") && request["max_tool_rounds"].is_number_integer()) {
                req_max_tool_rounds = std::max(1, request["max_tool_rounds"].get<int>());
            }

            bool bypass_prefix_cache = false;
            if (req.has_header("X-Bypass-Cache") && req.get_header_value("X-Bypass-Cache") == "true") {
                bypass_prefix_cache = true;
            }
            if (request.contains("no_cache") && request["no_cache"].is_boolean()) {
                bypass_prefix_cache = request["no_cache"].get<bool>();
            } else if (request.contains("bypass_cache") && request["bypass_cache"].is_boolean()) {
                bypass_prefix_cache = request["bypass_cache"].get<bool>();
            } else if (request.contains("bypass_prefix_cache") && request["bypass_prefix_cache"].is_boolean()) {
                bypass_prefix_cache = request["bypass_prefix_cache"].get<bool>();
            }

            LOG_INFO("Reasoning effort: %s, Thinking: %s, Thinking budget: %d, Tools: %zu, Anti-Loop: %s, Stop-Words: %zu, Server-Exec: %s (client_tool_exec: %s, headless: %s), Max-Tool-Rounds: %d, Timeout: %dms, Bypass-Cache: %s",
                     reasoning_effort.c_str(), enable_thinking ? "enabled" : "disabled", max_thinking_tokens,
                     tools.size(), (engine.enable_anti_loop_ ? "enabled" : "disabled"), stop_words.size(),
                     (do_server_exec ? "enabled" : "disabled"), client_tool_exec ? "true" : "false", g_headless_browsing ? "true" : "false", req_max_tool_rounds, execution_timeout_ms,
                     bypass_prefix_cache ? "true" : "false");

            std::string req_id = "chatcmpl-moecher-" + std::to_string(++g_request_counter);
            std::string model_id = engine.get_model_id();

            // ── Zero Decode Bypass (ZDB) Handling in /v1/chat/completions ───────────
            bool req_zdb = request.value("zero_decode_bypass", false) || request.value("zdb", false);
            std::vector<std::string> zdb_choices;
            auto extract_choices = [&](const std::string& key) {
                if (request.contains(key) && request[key].is_array()) {
                    for (const auto& item : request[key]) {
                        if (item.is_string()) {
                            std::string s = item.get<std::string>();
                            if (!s.empty()) zdb_choices.push_back(s);
                        } else if (item.is_object() && item.contains("text") && item["text"].is_string()) {
                            std::string s = item["text"].get<std::string>();
                            if (!s.empty()) zdb_choices.push_back(s);
                        }
                    }
                }
            };
            extract_choices("decision_choices");
            extract_choices("classification_choices");
            extract_choices("candidate_labels");
            extract_choices("zdb_choices");
            if (zdb_choices.empty()) {
                extract_choices("choices");
            }

            if (req_zdb) {
                if (zdb_choices.empty()) {
                    res.status = 400;
                    res.set_content("{\"error\":\"zero_decode_bypass requires a non-empty 'choices', 'decision_choices', or 'candidate_labels' array.\"}", "application/json");
                    return;
                }

                bool auto_fallback = request.value("auto_fallback", true);
                float min_confidence = request.value("min_confidence", 0.70f);
                float max_top1_gap = request.value("max_top1_gap", 1.5f);
                std::string zdb_format = request.value("format", "choice");

                MoecherEngine::ZeroDecodeResult zdb_res;
                {
                    std::lock_guard<std::mutex> lock(g_engine_mutex);
                    // For ZDB in chat completions, apply chat template with thinking disabled for step-0 logit extraction
                    auto mm_prompt = engine.format_chat_prompt(messages, false, "none", json(), false);
                    std::vector<int> prompt = mm_prompt.prompt;
                    zdb_res = engine.execute_zero_decode_bypass(prompt, zdb_choices, temperature, zdb_format, bypass_prefix_cache, min_confidence, max_top1_gap);
                }

                if (auto_fallback && !zdb_res.gate_passed) {
                    LOG_INFO("[ZDB-Arbiter] Gate REJECTED System 1 fast path (top1_gap=%.2f, conf=%.2f [min=%.2f], global_argmax='%s' [id=%d]). Falling back to System 2 autoregressive decode.",
                             zdb_res.top1_gap, zdb_res.confidence, min_confidence,
                             zdb_res.global_argmax_token_str.c_str(), zdb_res.global_argmax_token_id);
                    // Seamlessly fall through to standard stream/non-stream decode loops below!
                } else {
                    if (stream) {
                        time_t now_sec = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                        json usage_obj = {
                            {"prompt_tokens", zdb_res.prompt_tokens},
                            {"completion_tokens", 0},
                            {"total_tokens", zdb_res.prompt_tokens},
                            {"evaluated_prompt_tokens", zdb_res.prompt_tokens},
                            {"prefill_sec", zdb_res.prefill_sec},
                            {"prefill_tps", (zdb_res.prefill_sec > 0.0) ? (double)zdb_res.prompt_tokens / zdb_res.prefill_sec : 0.0},
                            {"decode_sec", 0.0},
                            {"decode_tps", 0.0}
                        };
                        json zdb_obj = moecher::zdb::build_zdb_json_object(zdb_res);
                        json chunk1 = {
                            {"id", req_id},
                            {"object", "chat.completion.chunk"},
                            {"created", now_sec},
                            {"model", model_id},
                            {"choices", json::array({
                                {{"index", 0}, {"delta", {{"role", "assistant"}, {"content", zdb_res.rendered_output}}}, {"finish_reason", nullptr}}
                            })}
                        };
                        json chunk2 = {
                            {"id", req_id},
                            {"object", "chat.completion.chunk"},
                            {"created", now_sec},
                            {"model", model_id},
                            {"choices", json::array({
                                {{"index", 0}, {"delta", json::object()}, {"finish_reason", "stop"}}
                            })},
                            {"usage", usage_obj},
                            {"zero_decode_bypass", zdb_obj}
                        };
                        std::string sse = "data: " + chunk1.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n" +
                                          "data: " + chunk2.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n" +
                                          "data: [DONE]\n\n";
                        res.set_content(sse, "text/event-stream");
                        return;
                    } else {
                        json choice = {
                            {"index", 0},
                            {"message", {
                                {"role", "assistant"},
                                {"content", zdb_res.rendered_output}
                            }},
                            {"finish_reason", "stop"}
                        };

                        json response = {
                            {"id", req_id},
                            {"object", "chat.completion"},
                            {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                            {"model", model_id},
                            {"choices", {choice}},
                            {"usage", {
                                {"prompt_tokens", zdb_res.prompt_tokens},
                                {"completion_tokens", 0},
                                {"total_tokens", zdb_res.prompt_tokens},
                                {"evaluated_prompt_tokens", zdb_res.prompt_tokens},
                                {"prefill_sec", zdb_res.prefill_sec},
                                {"prefill_tps", (zdb_res.prefill_sec > 0.0) ? (double)zdb_res.prompt_tokens / zdb_res.prefill_sec : 0.0},
                                {"decode_sec", 0.0},
                                {"decode_tps", 0.0}
                            }},
                            {"zero_decode_bypass", moecher::zdb::build_zdb_json_object(zdb_res)}
                        };
                        res.set_content(response.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
                        return;
                    }
                }
            }

            if (stream) {
                // Fast SSE delta writer avoiding heap-allocated JSON ASTs
                auto send_sse_delta = [](httplib::DataSink& sink, const std::string& req_id, const std::string& model_id,
                                         const std::string& created_str, const char* field_name, const std::string& text) -> bool {
                    std::string sse;
                    sse.reserve(128 + text.size() * 2);
                    sse.append("data: {\"id\":\"");
                    sse.append(req_id);
                    sse.append("\",\"object\":\"chat.completion.chunk\",\"created\":");
                    sse.append(created_str);
                    sse.append(",\"model\":\"");
                    sse.append(model_id);
                    sse.append("\",\"choices\":[{\"index\":0,\"delta\":{\"");
                    sse.append(field_name);
                    sse.append("\":");
                    sse.append(json(text).dump(-1, ' ', false, json::error_handler_t::replace));
                    sse.append("},\"finish_reason\":null}]}\n\n");
                    return sink.write(sse.data(), sse.size()) && !g_stop_requested.load();
                };

                // SSE streaming
                res.set_chunked_content_provider(
                    "text/event-stream",
                    [&engine, messages, tools, max_tokens, temperature, req_id, repetition_penalty, enable_thinking, max_thinking_tokens, top_p, min_p, top_k, reasoning_effort, model_id, execution_timeout_ms, require_external_authorization, workspace_boundary_enforced, authorized_paths, do_server_exec, req_max_tool_rounds, busy_guard, send_sse_delta, bypass_prefix_cache, had_image_originally, stop_words](size_t offset, httplib::DataSink &sink) {
                        if (offset > 0) return false;
                        std::lock_guard<std::mutex> lock(g_engine_mutex);

                        time_t created_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                        std::string created_time_str = std::to_string(created_time);

                        try {

                        json initial_chunk = {
                            {"id", req_id},
                            {"object", "chat.completion.chunk"},
                            {"created", created_time},
                            {"model", model_id},
                            {"choices", {{
                                {"index", 0},
                                {"delta", {{"role", "assistant"}}},
                                {"finish_reason", nullptr}
                            }}}
                        };
                        std::string sse = "data: " + initial_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                        if (!sink.write(sse.data(), sse.size())) {
                            LOG_INFO("Client disconnected before initial chunk could be delivered; aborting.");
                            return false;
                        }

                        json current_messages = messages;
                        int max_tool_rounds = (do_server_exec && !tools.empty() && g_enable_tools) ? req_max_tool_rounds : 1;
                        std::string final_finish_reason = "stop";
                        std::string last_completed_content;
                        std::vector<json> round_new_tool_msgs;

                        int max_allowed_prompt = engine.cfg_.max_seq_len - max_tokens;
                        if (max_allowed_prompt < 1024) max_allowed_prompt = engine.cfg_.max_seq_len - 256;

                        for (int round = 0; round < max_tool_rounds; round++) {
                            bool conv_has_image = false;
                            for (const auto& m : current_messages) {
                                if (message_has_image(m)) { conv_has_image = true; break; }
                            }
                            bool is_greeting = false;
                            if (!engine.cfg_.is_nemotron() && !current_messages.empty()) {
                                const auto& last_m = current_messages.back();
                                if (last_m.value("role", "") == "user" && last_m.contains("content") && last_m["content"].is_string()) {
                                    is_greeting = is_conversational_greeting(last_m["content"].get<std::string>());
                                }
                            }
                            bool req_thinking = enable_thinking && !conv_has_image && !is_greeting;
                            int req_thinking_tokens = req_thinking ? max_thinking_tokens : 0;
                            bool eff_bypass_prefix_cache = bypass_prefix_cache || had_image_originally;

                            std::vector<int> prompt;
                            bool is_continuation = false;
                            if (round == 0) {
                                is_continuation = check_and_build_continuation(engine, current_messages, s_last_conv_messages, req_thinking, reasoning_effort, prompt, had_image_originally);
                                if (is_continuation && prompt.size() > (size_t)max_allowed_prompt) {
                                    LOG_WARN("[Context] Continuation prompt length (%zu) exceeds context budget (%d). Forcing sliding-window pruning.",
                                             prompt.size(), max_allowed_prompt);
                                    is_continuation = false;
                                }
                            } else if (round > 0 && !round_new_tool_msgs.empty()) {
                                prompt = build_continuation_prompt(engine, round_new_tool_msgs, req_thinking, reasoning_effort);
                                is_continuation = true;
                                if (prompt.size() > (size_t)max_allowed_prompt) {
                                    LOG_WARN("[Context] Tool continuation length (%zu) exceeds context budget (%d). Forcing sliding-window pruning.",
                                             prompt.size(), max_allowed_prompt);
                                    is_continuation = false;
                                }
                                round_new_tool_msgs.clear();
                            }
                            MultimodalPrompt mm_prompt;
                            const __nv_bfloat16* d_vis_out = nullptr;
                            if (!is_continuation) {
                                current_messages = prune_messages_to_fit(current_messages, engine, max_allowed_prompt, req_thinking, reasoning_effort, tools);
                                mm_prompt = engine.format_chat_prompt(current_messages, req_thinking, reasoning_effort, tools, engine.has_vision());
                                prompt = mm_prompt.prompt;
                                if (engine.has_vision() && !mm_prompt.image_b64.empty() && mm_prompt.visual_start_pos >= 0) {
                                    if (engine.ensure_vision_loaded()) {
                                        moecher::vision::ProcessedImage proc_img;
                                        int img_target_sz = engine.vision_tower_->image_size_;
                                        LOG_INFO("[Vision] Preprocessing image attachment for multimodal inference (%zu bytes b64, target_size=%d)...",
                                                 mm_prompt.image_b64.size(), img_target_sz);
                                        if (moecher::vision::preprocess_image_from_base64(mm_prompt.image_b64, proc_img, img_target_sz)) {
                                            d_vis_out = engine.vision_tower_->forward(proc_img, engine.main_stream_);
                                            LOG_INFO("[Vision] Vision Tower forward pass complete: 576 visual tokens projected to hidden size %d.", engine.cfg_.hidden_size);
                                        } else {
                                            LOG_WARN("[Vision] Failed to decode/preprocess image attachment!");
                                        }
                                    }
                                }
                            }
                            LOG_INFO("STREAM PROMPT [round %d] (len=%zu, continuation=%s, visual=%s)",
                                     round + 1, prompt.size(), is_continuation ? "true" : "false", (d_vis_out != nullptr) ? "active" : "none");

                            std::string round_content;
                            std::string round_reasoning;
                            ToolTagStreamFilter content_filter;
                            ToolTagStreamFilter reasoning_filter;

                            auto status_cb = [&](const std::string& status_msg) -> bool {
                                return send_sse_delta(sink, req_id, model_id, created_time_str, "status", status_msg);
                            };
                            content_filter.on_status = status_cb;
                            reasoning_filter.on_status = status_cb;

                            std::string utf8_reasoning_buf;
                            std::string utf8_content_buf;

                            auto emit_buffered_reasoning = [&](const std::string& chunk, bool flush_all = false) -> bool {
                                utf8_reasoning_buf += chunk;
                                size_t valid_len = flush_all ? utf8_reasoning_buf.size() : valid_utf8_prefix_len(utf8_reasoning_buf);
                                if (valid_len > 0) {
                                    std::string to_send = utf8_reasoning_buf.substr(0, valid_len);
                                    utf8_reasoning_buf.erase(0, valid_len);
                                    return send_sse_delta(sink, req_id, model_id, created_time_str, "reasoning_content", to_send);
                                }
                                return true;
                            };

                            auto emit_buffered_content = [&](const std::string& chunk, bool flush_all = false) -> bool {
                                utf8_content_buf += chunk;
                                size_t valid_len = flush_all ? utf8_content_buf.size() : valid_utf8_prefix_len(utf8_content_buf);
                                if (valid_len > 0) {
                                    std::string to_send = utf8_content_buf.substr(0, valid_len);
                                    utf8_content_buf.erase(0, valid_len);
                                    return send_sse_delta(sink, req_id, model_id, created_time_str, "content", to_send);
                                }
                                return true;
                            };

                            engine.generate(prompt, max_tokens, temperature, [&](const std::string& text, bool is_reasoning) -> bool {
                                if (g_stop_requested.load()) return false;
                                if (text.empty()) return true;

                                if (is_reasoning) {
                                    round_reasoning += text;
                                    if (g_enable_tools && !tools.empty()) {
                                        return reasoning_filter.feed(text, [&](const std::string& safe_text) -> bool {
                                            return emit_buffered_reasoning(safe_text);
                                        });
                                    } else {
                                        reasoning_filter.streamed_clean_text += text;
                                        return emit_buffered_reasoning(text);
                                    }
                                } else {
                                    round_content += text;
                                    if (g_enable_tools && !tools.empty()) {
                                        return content_filter.feed(text, [&](const std::string& safe_text) -> bool {
                                            return emit_buffered_content(safe_text);
                                        });
                                    } else {
                                        content_filter.streamed_clean_text += text;
                                        return emit_buffered_content(text);
                                    }
                                }
                            }, repetition_penalty, req_thinking, req_thinking_tokens, top_p, min_p, top_k,
                               d_vis_out, mm_prompt.visual_start_pos, mm_prompt.visual_num_tokens, eff_bypass_prefix_cache, stop_words);

                            if (d_vis_out != nullptr && engine.gpu_caps_.is_vram_constrained) {
                                engine.unload_vision_tower();
                            }

                            final_finish_reason = engine.last_finish_reason_.empty() ? "stop" : engine.last_finish_reason_;
                            last_completed_content = round_content;

                            if (g_enable_tools && !tools.empty()) {
                                reasoning_filter.flush_remaining([&](const std::string& safe_text) -> bool {
                                    return emit_buffered_reasoning(safe_text);
                                });
                                content_filter.flush_remaining([&](const std::string& safe_text) -> bool {
                                    return emit_buffered_content(safe_text);
                                });
                            }
                            emit_buffered_reasoning("", true);
                            emit_buffered_content("", true);

                            if (!g_enable_tools || tools.empty()) {
                                break;
                            }

                            std::string clean_content;
                            std::vector<moecher::tooling::ToolCall> round_tool_calls;
                            moecher::tooling::extract_tool_calls(round_content, clean_content, round_tool_calls);

                            if (round_tool_calls.empty()) {
                                last_completed_content = clean_content;
                                std::string unstreamed_content = clean_content;
                                if (!content_filter.streamed_clean_text.empty()) {
                                    if (clean_content.size() >= content_filter.streamed_clean_text.size() &&
                                        clean_content.compare(0, content_filter.streamed_clean_text.size(), content_filter.streamed_clean_text) == 0) {
                                        unstreamed_content = clean_content.substr(content_filter.streamed_clean_text.size());
                                    } else if (clean_content.size() <= content_filter.streamed_clean_text.size()) {
                                        unstreamed_content.clear();
                                    }
                                }
                                if (!unstreamed_content.empty()) {
                                    json delta_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{"content", unstreamed_content}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_chunk = "data: " + delta_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_chunk.data(), sse_chunk.size());
                                }
                                break;
                            }

                            // Stream any unstreamed leading text from model before tool call
                            std::string unstreamed_intro = clean_content;
                            if (!content_filter.streamed_clean_text.empty()) {
                                if (clean_content.size() >= content_filter.streamed_clean_text.size() &&
                                    clean_content.compare(0, content_filter.streamed_clean_text.size(), content_filter.streamed_clean_text) == 0) {
                                    unstreamed_intro = clean_content.substr(content_filter.streamed_clean_text.size());
                                } else if (clean_content.size() <= content_filter.streamed_clean_text.size()) {
                                    unstreamed_intro.clear();
                                }
                            }
                            if (!unstreamed_intro.empty()) {
                                std::string field_name = req_thinking ? "reasoning_content" : "content";
                                json intro_chunk = {
                                    {"id", req_id},
                                    {"object", "chat.completion.chunk"},
                                    {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                    {"model", model_id},
                                    {"choices", {{
                                        {"index", 0},
                                        {"delta", {{field_name, unstreamed_intro + "\n\n"}}},
                                        {"finish_reason", nullptr}
                                    }}}
                                };
                                std::string sse_intro = "data: " + intro_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                sink.write(sse_intro.data(), sse_intro.size());
                            }

                            if (!do_server_exec) {
                                final_finish_reason = "tool_calls";
                                json tc_arr = json::array();
                                for (size_t idx = 0; idx < round_tool_calls.size(); idx++) {
                                    const auto& tc = round_tool_calls[idx];
                                    tc_arr.push_back({
                                        {"index", (int)idx},
                                        {"id", tc.id},
                                        {"type", tc.type},
                                        {"function", {{"name", tc.name}, {"arguments", tc.arguments}}}
                                    });
                                }
                                json tc_chunk = {
                                    {"id", req_id},
                                    {"object", "chat.completion.chunk"},
                                    {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                    {"model", model_id},
                                    {"choices", {{
                                        {"index", 0},
                                        {"delta", {{"tool_calls", tc_arr}}},
                                        {"finish_reason", "tool_calls"}
                                    }}}
                                };
                                std::string sse_chunk = "data: " + tc_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                sink.write(sse_chunk.data(), sse_chunk.size());
                                break;
                            }

                            // Server-side tool execution
                            bool can_execute_all = true;
                            json assistant_msg = {{"role", "assistant"}, {"content", round_content}};
                            json tc_json_arr = json::array();
                            for (const auto& tc : round_tool_calls) {
                                tc_json_arr.push_back({
                                    {"id", tc.id},
                                    {"type", tc.type},
                                    {"function", {{"name", tc.name}, {"arguments", tc.arguments}}}
                                });
                            }
                            assistant_msg["tool_calls"] = tc_json_arr;
                            current_messages.push_back(assistant_msg);

                            std::string tool_delta_field = req_thinking ? "reasoning_content" : "content";

                            auto is_path_authorized = [&](const std::string& p) -> bool {
                                if (!require_external_authorization && !workspace_boundary_enforced) return true;
                                if (moecher::tooling::is_path_inside_workspace(p)) return true;
                                for (const auto& ap : authorized_paths) {
                                    if (ap == "*" || ap == p) return true;
                                    if (!ap.empty() && moecher::tooling::is_path_inside_workspace(p, ap)) return true;
                                }
                                return false;
                            };

                            for (const auto& tc : round_tool_calls) {
                                std::string exec_output;
                                if (tc.name == "web_search" || tc.name == "google_search") {
                                    std::string query;
                                    int num_results = 5;
                                    std::string site = "";
                                    std::string provider = "";
                                    std::string api_key = "";
                                    std::string cx = "";
                                    std::string searx_url = "";
                                    try {
                                        json args = json::parse(tc.arguments);
                                        query = args.value("query", "");
                                        num_results = args.value("num_results", 5);
                                        site = args.value("site", "");
                                        provider = args.value("provider", "");
                                        api_key = args.value("api_key", "");
                                        cx = args.value("cx", "");
                                        searx_url = args.value("searxng_url", "");
                                    } catch (...) { query = tc.arguments; }

                                    std::string active_card =
                                        "\n\n<div class=\"tool-activity-block active\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"thinking-spinner\">progress_activity</span>\n"
                                        "  <span class=\"tool-action-label\">Searching Web</span>\n"
                                        "  <span class=\"tool-target-subtle\">" + query + "</span>\n"
                                        "</div>\n\n";

                                    json info_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, active_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_info = "data: " + info_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_info.data(), sse_info.size());

                                    LOG_INFO("Streaming Tool Exec: web_search('%s', num=%d, site='%s', provider='%s')", query.c_str(), num_results, site.c_str(), provider.c_str());
                                    auto doc = moecher::tooling::web_search_full(query, num_results, site, provider, api_key, cx, searx_url);
                                    exec_output = doc.clean_text;

                                    // Stream retrieved search cards to the frontend
                                    json doc_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {
                                                {"retrieved_document", {
                                                    {"id", tc.id},
                                                    {"url", doc.url},
                                                    {"title", doc.title.empty() ? ("Web Search: " + query) : doc.title},
                                                    {"html", doc.raw_html},
                                                    {"snippet", (doc.clean_text.size() > 300 ? doc.clean_text.substr(0, 300) + "..." : doc.clean_text)}
                                                }}
                                            }},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_doc = "data: " + doc_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_doc.data(), sse_doc.size());

                                    std::string completed_card =
                                        "\n\n<div class=\"tool-activity-block completed\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"material-symbols-outlined tool-done-icon\">search</span>\n"
                                        "  <span class=\"tool-action-label\">Web Searched</span>\n"
                                        "  <span class=\"tool-target-subtle\">" + query + "</span>\n"
                                        "</div>\n\n";

                                    json comp_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, completed_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_comp = "data: " + comp_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_comp.data(), sse_comp.size());

                                } else if (tc.name == "fetch_url") {
                                    std::string url;
                                    std::string mode = "text";
                                    std::string pattern = "";
                                    int offset = 0;
                                    int max_chars = 4000;
                                    try {
                                        json args = json::parse(tc.arguments);
                                        url = args.value("url", "");
                                        mode = args.value("mode", "text");
                                        pattern = args.value("pattern", "");
                                        offset = args.value("offset", 0);
                                        max_chars = args.value("max_chars", 4000);
                                    } catch (...) { url = tc.arguments; }

                                    std::string active_card =
                                        "\n\n<div class=\"tool-activity-block active\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"thinking-spinner\">progress_activity</span>\n"
                                        "  <span class=\"tool-action-label\">Searching Web</span>\n"
                                        "  <span class=\"tool-target-subtle\">" + url + "</span>\n"
                                        "</div>\n\n";

                                    json info_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, active_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_info = "data: " + info_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_info.data(), sse_info.size());

                                    LOG_INFO("Streaming Tool Exec: fetch_url('%s', mode='%s', pattern='%s')", url.c_str(), mode.c_str(), pattern.c_str());
                                    auto doc = moecher::tooling::fetch_url_full(url, std::min(execution_timeout_ms, 30000), max_chars, mode, pattern, offset);
                                    exec_output = doc.clean_text;

                                    // Stream the retrieved document metadata & HTML to the frontend for the Retrieved Documents panel
                                    json doc_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {
                                                {"retrieved_document", {
                                                    {"id", tc.id},
                                                    {"url", doc.url},
                                                    {"title", doc.title.empty() ? doc.url : doc.title},
                                                    {"html", doc.raw_html},
                                                    {"snippet", (doc.clean_text.size() > 300 ? doc.clean_text.substr(0, 300) + "..." : doc.clean_text)}
                                                }}
                                            }},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_doc = "data: " + doc_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_doc.data(), sse_doc.size());

                                    std::string completed_card =
                                        "\n\n<div class=\"tool-activity-block completed\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"material-symbols-outlined tool-done-icon\">travel_explore</span>\n"
                                        "  <span class=\"tool-action-label\">Searched</span>\n"
                                        "  <span class=\"tool-target-subtle\">" + doc.url + "</span>\n"
                                        "</div>\n\n";

                                    json comp_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, completed_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_comp = "data: " + comp_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_comp.data(), sse_comp.size());

                                } else if (tc.name == "read_file") {
                                    std::string path;
                                    int start_line = 1;
                                    int end_line = -1;
                                    try {
                                        json args = json::parse(tc.arguments);
                                        path = args.value("path", "");
                                        start_line = args.value("start_line", 1);
                                        end_line = args.value("end_line", -1);
                                    } catch (...) { path = tc.arguments; }

                                    std::string active_card =
                                        "\n\n<div class=\"tool-activity-block active\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"thinking-spinner\">progress_activity</span>\n"
                                        "  <span class=\"tool-action-label\">Reading file</span>\n"
                                        "  <span class=\"tool-target-subtle\">" + path + "</span>\n"
                                        "</div>\n\n";

                                    json info_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, active_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_info = "data: " + info_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_info.data(), sse_info.size());

                                    LOG_INFO("Streaming Tool Exec: read_file('%s', start=%d, end=%d)", path.c_str(), start_line, end_line);

                                    if (!is_path_authorized(path)) {
                                        exec_output = "[Authorization Required: The path '" + path + "' is outside the current workspace directory (" + moecher::tooling::get_workspace_directory() + "). Action requires explicit user authorization for path: " + path + "]";
                                        json auth_chunk = {
                                            {"id", req_id},
                                            {"object", "chat.completion.chunk"},
                                            {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                            {"model", model_id},
                                            {"choices", {{
                                                {"index", 0},
                                                {"delta", {
                                                    {"authorization_required", {
                                                        {"tool", "read_file"},
                                                        {"path", path},
                                                        {"id", tc.id}
                                                    }}
                                                }},
                                                {"finish_reason", nullptr}
                                            }}}
                                        };
                                        std::string sse_auth = "data: " + auth_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                        sink.write(sse_auth.data(), sse_auth.size());
                                    } else {
                                        exec_output = moecher::tooling::read_file(path, start_line, end_line);
                                    }

                                    std::string completed_card =
                                        "\n\n<div class=\"tool-activity-block completed\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"material-symbols-outlined tool-done-icon\">description</span>\n"
                                        "  <span class=\"tool-action-label\">Read file</span>\n"
                                        "  <span class=\"tool-target-subtle\">" + path + "</span>\n"
                                        "</div>\n\n";

                                    json comp_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, completed_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_comp = "data: " + comp_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_comp.data(), sse_comp.size());

                                } else if (tc.name == "write_file") {
                                    std::string path;
                                    std::string content;
                                    bool overwrite = true;
                                    try {
                                        json args = json::parse(tc.arguments);
                                        path = args.value("path", "");
                                        content = args.value("content", "");
                                        overwrite = args.value("overwrite", true);
                                    } catch (...) { path = tc.arguments; }

                                    std::string active_card =
                                        "\n\n<div class=\"tool-activity-block active\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"thinking-spinner\">progress_activity</span>\n"
                                        "  <span class=\"tool-action-label\">Writing file</span>\n"
                                        "  <span class=\"tool-target-subtle\">" + path + "</span>\n"
                                        "</div>\n\n";

                                    json info_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, active_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_info = "data: " + info_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_info.data(), sse_info.size());

                                    LOG_INFO("Streaming Tool Exec: write_file('%s', %zu bytes)", path.c_str(), content.size());

                                    if (!is_path_authorized(path)) {
                                        exec_output = "[Authorization Required: The path '" + path + "' is outside the current workspace directory (" + moecher::tooling::get_workspace_directory() + "). Action requires explicit user authorization for path: " + path + "]";
                                        json auth_chunk = {
                                            {"id", req_id},
                                            {"object", "chat.completion.chunk"},
                                            {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                            {"model", model_id},
                                            {"choices", {{
                                                {"index", 0},
                                                {"delta", {
                                                    {"authorization_required", {
                                                        {"tool", "write_file"},
                                                        {"path", path},
                                                        {"id", tc.id}
                                                    }}
                                                }},
                                                {"finish_reason", nullptr}
                                            }}}
                                        };
                                        std::string sse_auth = "data: " + auth_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                        sink.write(sse_auth.data(), sse_auth.size());
                                    } else {
                                        exec_output = moecher::tooling::write_file(path, content, overwrite);
                                    }

                                    std::string completed_card =
                                        "\n\n<div class=\"tool-activity-block completed\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"material-symbols-outlined tool-done-icon\">edit_note</span>\n"
                                        "  <span class=\"tool-action-label\">Wrote file</span>\n"
                                        "  <span class=\"tool-target-subtle\">" + path + "</span>\n"
                                        "</div>\n\n";

                                    json comp_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, completed_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_comp = "data: " + comp_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_comp.data(), sse_comp.size());

                                } else if (tc.name == "edit_file") {
                                    std::string path;
                                    std::string target_content;
                                    std::string replacement_content;
                                    try {
                                        json args = json::parse(tc.arguments);
                                        path = args.value("path", "");
                                        target_content = args.value("target_content", "");
                                        replacement_content = args.value("replacement_content", "");
                                    } catch (...) { path = tc.arguments; }

                                    std::string active_card =
                                        "\n\n<div class=\"tool-activity-block active\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"thinking-spinner\">progress_activity</span>\n"
                                        "  <span class=\"tool-action-label\">Editing file</span>\n"
                                        "  <span class=\"tool-target-subtle\">" + path + "</span>\n"
                                        "</div>\n\n";

                                    json info_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, active_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_info = "data: " + info_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_info.data(), sse_info.size());

                                    LOG_INFO("Streaming Tool Exec: edit_file('%s')", path.c_str());

                                    if (!is_path_authorized(path)) {
                                        exec_output = "[Authorization Required: The path '" + path + "' is outside the current workspace directory (" + moecher::tooling::get_workspace_directory() + "). Action requires explicit user authorization for path: " + path + "]";
                                        json auth_chunk = {
                                            {"id", req_id},
                                            {"object", "chat.completion.chunk"},
                                            {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                            {"model", model_id},
                                            {"choices", {{
                                                {"index", 0},
                                                {"delta", {
                                                    {"authorization_required", {
                                                        {"tool", "edit_file"},
                                                        {"path", path},
                                                        {"id", tc.id}
                                                    }}
                                                }},
                                                {"finish_reason", nullptr}
                                            }}}
                                        };
                                        std::string sse_auth = "data: " + auth_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                        sink.write(sse_auth.data(), sse_auth.size());
                                    } else {
                                        exec_output = moecher::tooling::edit_file(path, target_content, replacement_content);
                                    }

                                    std::string completed_card =
                                        "\n\n<div class=\"tool-activity-block completed\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"material-symbols-outlined tool-done-icon\">edit_square</span>\n"
                                        "  <span class=\"tool-action-label\">Edited file</span>\n"
                                        "  <span class=\"tool-target-subtle\">" + path + "</span>\n"
                                        "</div>\n\n";

                                    json comp_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, completed_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_comp = "data: " + comp_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_comp.data(), sse_comp.size());

                                } else if (tc.name == "execute_command") {
                                    std::string cmd;
                                    try {
                                        json args = json::parse(tc.arguments);
                                        cmd = args.value("command", "");
                                    } catch (...) { cmd = tc.arguments; }

                                    std::string active_card =
                                        "\n\n<div class=\"tool-activity-block active\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"thinking-spinner\">progress_activity</span>\n"
                                        "  <span class=\"tool-action-label\">Running command</span>\n"
                                        "  <span class=\"tool-target-subtle\">" + cmd + "</span>\n"
                                        "</div>\n\n";

                                    json info_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, active_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_info = "data: " + info_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_info.data(), sse_info.size());

                                    LOG_INFO("Streaming Tool Exec: execute_command('%s', timeout=%dms)", cmd.c_str(), execution_timeout_ms);
                                    exec_output = moecher::tooling::execute_system_command(cmd, execution_timeout_ms);

                                    std::string completed_card =
                                        "\n\n<div class=\"tool-activity-block completed\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"material-symbols-outlined tool-done-icon\">terminal</span>\n"
                                        "  <span class=\"tool-action-label\">Executed</span>\n"
                                        "  <span class=\"tool-target-subtle\">" + cmd + "</span>\n"
                                        "</div>\n\n";

                                    json comp_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, completed_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_comp = "data: " + comp_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_comp.data(), sse_comp.size());
                                } else if (moecher::mcp::MCPManager::instance().has_tool(tc.name)) {
                                    std::string s_id, raw_tname;
                                    moecher::mcp::MCPManager::instance().has_tool(tc.name, &s_id, &raw_tname);
                                    std::string active_card =
                                        "\n\n<div class=\"tool-activity-block active mcp-tool\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"thinking-spinner\">progress_activity</span>\n"
                                        "  <span class=\"tool-action-label\">[MCP: " + s_id + "] " + raw_tname + "</span>\n"
                                        "  <span class=\"tool-target-subtle\">" + (tc.arguments.size() > 80 ? tc.arguments.substr(0, 80) + "..." : tc.arguments) + "</span>\n"
                                        "</div>\n\n";

                                    json info_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, active_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_info = "data: " + info_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_info.data(), sse_info.size());

                                    json parsed_args = json::object();
                                    try { parsed_args = json::parse(tc.arguments); } catch (...) {}
                                    LOG_INFO("Streaming Tool Exec: MCP [%s:%s]", s_id.c_str(), raw_tname.c_str());
                                    exec_output = moecher::mcp::MCPManager::instance().call_tool(tc.name, parsed_args, execution_timeout_ms);

                                    std::string completed_card =
                                        "\n\n<div class=\"tool-activity-block completed mcp-tool\" id=\"tool-act-" + tc.id + "\">\n"
                                        "  <span class=\"material-symbols-outlined tool-done-icon\">hub</span>\n"
                                        "  <span class=\"tool-action-label\">[MCP: " + s_id + "] " + raw_tname + "</span>\n"
                                        "  <span class=\"tool-target-subtle\">Completed</span>\n"
                                        "</div>\n\n";

                                    json comp_chunk = {
                                        {"id", req_id},
                                        {"object", "chat.completion.chunk"},
                                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                        {"model", model_id},
                                        {"choices", {{
                                            {"index", 0},
                                            {"delta", {{tool_delta_field, completed_card}}},
                                            {"finish_reason", nullptr}
                                        }}}
                                    };
                                    std::string sse_comp = "data: " + comp_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                    sink.write(sse_comp.data(), sse_comp.size());
                                } else {
                                    can_execute_all = false;
                                    break;
                                }

                                json tool_reply = {
                                    {"role", "tool"},
                                    {"tool_call_id", tc.id},
                                    {"content", exec_output}
                                };
                                current_messages.push_back(tool_reply);
                                round_new_tool_msgs.push_back(tool_reply);
                            }

                            if (!can_execute_all) {
                                final_finish_reason = "tool_calls";
                                break;
                            }

                            if (round + 1 < max_tool_rounds) {
                                json proc_chunk = {
                                    {"id", req_id},
                                    {"object", "chat.completion.chunk"},
                                    {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                    {"model", model_id},
                                    {"choices", {{
                                        {"index", 0},
                                        {"delta", {{"processing_status", "Encoding tool output & elaborating response"}}},
                                        {"finish_reason", nullptr}
                                    }}}
                                };
                                std::string sse_proc = "data: " + proc_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                sink.write(sse_proc.data(), sse_proc.size());
                            } else {
                                LOG_WARN("Tool execution reached max rounds limit (%d). Halting loop and notifying user.", max_tool_rounds);
                                std::string warn_msg = "\n\n> ⚠️ **Maximum tool execution rounds reached (" + std::to_string(max_tool_rounds) + "/" + std::to_string(max_tool_rounds) + ").** Execution stopped.";
                                json warn_chunk = {
                                    {"id", req_id},
                                    {"object", "chat.completion.chunk"},
                                    {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                                    {"model", model_id},
                                    {"choices", {{
                                        {"index", 0},
                                        {"delta", {
                                            {"content", warn_msg},
                                            {"tool_limit_reached", true},
                                            {"max_tool_rounds", max_tool_rounds}
                                        }},
                                        {"finish_reason", nullptr}
                                    }}}
                                };
                                std::string sse_warn = "data: " + warn_chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                                sink.write(sse_warn.data(), sse_warn.size());
                            }
                        }

                        // Final finish chunk
                        json finish = {
                            {"id", req_id},
                            {"object", "chat.completion.chunk"},
                            {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                            {"model", model_id},
                            {"choices", {{
                                {"index", 0},
                                {"delta", json::object()},
                                {"finish_reason", final_finish_reason}
                            }}},
                            {"usage", {
                                {"prompt_tokens", engine.prompt_token_count_},
                                {"completion_tokens", engine.completion_token_count_},
                                {"total_tokens", engine.prompt_token_count_ + engine.completion_token_count_},
                                {"evaluated_prompt_tokens", engine.last_prefill_tokens_},
                                {"prefill_sec", engine.last_prefill_sec_},
                                {"prefill_tps", engine.last_prefill_tps_},
                                {"decode_sec", engine.last_decode_sec_},
                                {"decode_tps", engine.last_decode_tps_},
                                {"prompt_tokens_details", {
                                    {"cached_tokens", engine.last_cached_tokens_}
                                }}
                            }}
                        };
                        sse = "data: " + finish.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                        sink.write(sse.data(), sse.size());
                        sink.write("data: [DONE]\n\n", 14);
                        sink.done();

                        // Save conversation state for turn continuation
                        json conv_to_save = current_messages;
                        if (conv_to_save.empty() || conv_to_save.back().value("role", "") != "assistant") {
                            conv_to_save.push_back({{"role", "assistant"}, {"content", last_completed_content}});
                        }
                        s_last_conv_messages = conv_to_save;

                        if (g_track_experts && !engine.expert_freq_counts_.empty()) {
                            std::string fpath = engine.model_dir_.empty() ? "expert_freq.bin" : (engine.model_dir_ + "/expert_freq.bin");
                            engine.save_expert_freq(fpath);
                            engine.save_expert_profile(engine.model_dir_);
                        }
                        return true;
                        } catch (const std::exception& e) {
                            LOG_ERROR("[Streaming] Exception during request execution: %s", e.what());
                            send_sse_delta(sink, req_id, model_id, created_time_str, "content", std::string("\n\n[Server Error: ") + e.what() + "]\n");
                            sink.done();
                            return false;
                        }
                    }
                );
            } else {
                try {
                    std::string final_response_text;
                std::vector<moecher::tooling::ToolCall> emitted_tool_calls;
                json current_messages = messages;
                std::string finish_reason = "stop";
                std::vector<json> round_new_tool_msgs;

                int max_tool_rounds = (do_server_exec && !tools.empty() && g_enable_tools) ? req_max_tool_rounds : 1;

                int max_allowed_prompt = engine.cfg_.max_seq_len - max_tokens;
                if (max_allowed_prompt < 1024) max_allowed_prompt = engine.cfg_.max_seq_len - 256;

                for (int round = 0; round < max_tool_rounds; round++) {
                    bool conv_has_image = false;
                    for (const auto& m : current_messages) {
                        if (message_has_image(m)) { conv_has_image = true; break; }
                    }
                    bool is_greeting = false;
                    if (!engine.cfg_.is_nemotron() && !current_messages.empty()) {
                        const auto& last_m = current_messages.back();
                        if (last_m.value("role", "") == "user" && last_m.contains("content") && last_m["content"].is_string()) {
                            is_greeting = is_conversational_greeting(last_m["content"].get<std::string>());
                        }
                    }
                    bool req_thinking = enable_thinking && !conv_has_image && !is_greeting;
                    int req_thinking_tokens = req_thinking ? max_thinking_tokens : 0;
                    bool eff_bypass_prefix_cache = bypass_prefix_cache || had_image_originally;

                    std::vector<int> prompt;
                    bool is_continuation = false;
                    if (round == 0) {
                        std::lock_guard<std::mutex> lock(g_engine_mutex);
                        is_continuation = check_and_build_continuation(engine, current_messages, s_last_conv_messages, req_thinking, reasoning_effort, prompt, had_image_originally);
                        if (is_continuation && prompt.size() > (size_t)max_allowed_prompt) {
                            LOG_WARN("[Context] Continuation prompt length (%zu) exceeds context budget (%d). Forcing sliding-window pruning.",
                                     prompt.size(), max_allowed_prompt);
                            is_continuation = false;
                        }
                    } else if (round > 0 && !round_new_tool_msgs.empty()) {
                        std::lock_guard<std::mutex> lock(g_engine_mutex);
                        prompt = build_continuation_prompt(engine, round_new_tool_msgs, req_thinking, reasoning_effort);
                        is_continuation = true;
                        if (prompt.size() > (size_t)max_allowed_prompt) {
                            LOG_WARN("[Context] Tool continuation length (%zu) exceeds context budget (%d). Forcing sliding-window pruning.",
                                     prompt.size(), max_allowed_prompt);
                            is_continuation = false;
                        }
                        round_new_tool_msgs.clear();
                    }
                    MultimodalPrompt mm_prompt;
                    const __nv_bfloat16* d_vis_out = nullptr;
                    if (!is_continuation) {
                        current_messages = prune_messages_to_fit(current_messages, engine, max_allowed_prompt, req_thinking, reasoning_effort, tools);
                        mm_prompt = engine.format_chat_prompt(current_messages, req_thinking, reasoning_effort, tools, engine.has_vision());
                        prompt = mm_prompt.prompt;
                        if (engine.has_vision() && !mm_prompt.image_b64.empty() && mm_prompt.visual_start_pos >= 0) {
                            if (engine.ensure_vision_loaded()) {
                                moecher::vision::ProcessedImage proc_img;
                                int img_target_sz = engine.vision_tower_->image_size_;
                                LOG_INFO("[Vision] Preprocessing image attachment for multimodal inference (%zu bytes b64, target_size=%d)...",
                                         mm_prompt.image_b64.size(), img_target_sz);
                                if (moecher::vision::preprocess_image_from_base64(mm_prompt.image_b64, proc_img, img_target_sz)) {
                                    d_vis_out = engine.vision_tower_->forward(proc_img, engine.main_stream_);
                                    LOG_INFO("[Vision] Vision Tower forward pass complete: 576 visual tokens projected to hidden size %d.", engine.cfg_.hidden_size);
                                } else {
                                    LOG_WARN("[Vision] Failed to decode/preprocess image attachment!");
                                }
                            }
                        }
                    }
                    LOG_INFO("PROMPT [round %d] (len=%zu, continuation=%s, visual=%s)",
                             round + 1, prompt.size(), is_continuation ? "true" : "false", (d_vis_out != nullptr) ? "active" : "none");
                    std::string decoded_p = engine.tokenizer_.decode(prompt);
                    LOG_INFO("PROMPT text: \"%s\"", decoded_p.c_str());
                    std::string ids_str = "";
                    for (size_t pi = 0; pi < std::min((size_t)30, prompt.size()); pi++) ids_str += std::to_string(prompt[pi]) + " ";
                    LOG_INFO("PROMPT first tokens: %s", ids_str.c_str());

                    std::string response_text;
                    {
                        std::lock_guard<std::mutex> lock(g_engine_mutex);
                        response_text = engine.generate(prompt, max_tokens, temperature, nullptr, repetition_penalty, req_thinking, req_thinking_tokens, top_p, min_p, top_k,
                                                        d_vis_out, mm_prompt.visual_start_pos, mm_prompt.visual_num_tokens, eff_bypass_prefix_cache, stop_words);
                        finish_reason = engine.last_finish_reason_;

                        if (d_vis_out != nullptr && engine.gpu_caps_.is_vram_constrained) {
                            engine.unload_vision_tower();
                        }
                    }

                    if (!g_enable_tools || tools.empty()) {
                        final_response_text = response_text;
                        break;
                    }

                    std::string clean_content;
                    std::vector<moecher::tooling::ToolCall> round_tool_calls;
                    moecher::tooling::extract_tool_calls(response_text, clean_content, round_tool_calls);

                    if (round_tool_calls.empty()) {
                        final_response_text = clean_content;
                        break;
                    }

                    emitted_tool_calls = round_tool_calls;
                    final_response_text = clean_content;

                    if (!do_server_exec) {
                        finish_reason = "tool_calls";
                        break;
                    }

                    // Server-side tool execution
                    bool can_execute_all = true;
                    json assistant_msg = {{"role", "assistant"}, {"content", response_text}};
                    json tc_json_arr = json::array();
                    for (const auto& tc : round_tool_calls) {
                        tc_json_arr.push_back({
                            {"id", tc.id},
                            {"type", tc.type},
                            {"function", {{"name", tc.name}, {"arguments", tc.arguments}}}
                        });
                    }
                    assistant_msg["tool_calls"] = tc_json_arr;
                    current_messages.push_back(assistant_msg);

                    auto is_path_authorized = [&](const std::string& p) -> bool {
                        if (!require_external_authorization && !workspace_boundary_enforced) return true;
                        if (moecher::tooling::is_path_inside_workspace(p)) return true;
                        for (const auto& ap : authorized_paths) {
                            if (ap == "*" || ap == p) return true;
                            if (!ap.empty() && moecher::tooling::is_path_inside_workspace(p, ap)) return true;
                        }
                        return false;
                    };

                    for (const auto& tc : round_tool_calls) {
                        std::string exec_output;
                        if (tc.name == "web_search" || tc.name == "google_search") {
                            std::string query;
                            int num_results = 5;
                            std::string site = "";
                            std::string provider = "";
                            std::string api_key = "";
                            std::string cx = "";
                            std::string searx_url = "";
                            try {
                                json args = json::parse(tc.arguments);
                                query = args.value("query", "");
                                num_results = args.value("num_results", 5);
                                site = args.value("site", "");
                                provider = args.value("provider", "");
                                api_key = args.value("api_key", "");
                                cx = args.value("cx", "");
                                searx_url = args.value("searxng_url", "");
                            } catch (...) { query = tc.arguments; }
                            LOG_INFO("Executing built-in tool web_search: %s (num=%d, site=%s, provider=%s)", query.c_str(), num_results, site.c_str(), provider.c_str());
                            exec_output = moecher::tooling::web_search_content(query, num_results, site, provider, api_key, cx, searx_url);
                        } else if (tc.name == "fetch_url") {
                            std::string url;
                            std::string mode = "text";
                            std::string pattern = "";
                            int offset = 0;
                            int max_chars = 4000;
                            try {
                                json args = json::parse(tc.arguments);
                                url = args.value("url", "");
                                mode = args.value("mode", "text");
                                pattern = args.value("pattern", "");
                                offset = args.value("offset", 0);
                                max_chars = args.value("max_chars", 4000);
                            } catch (...) { url = tc.arguments; }
                            LOG_INFO("Executing built-in tool fetch_url: %s (mode=%s, pattern=%s)", url.c_str(), mode.c_str(), pattern.c_str());
                            exec_output = moecher::tooling::fetch_url_content(url, std::min(execution_timeout_ms, 30000), max_chars, mode, pattern, offset);
                        } else if (tc.name == "read_file") {
                            std::string path;
                            int start_line = 1;
                            int end_line = -1;
                            try {
                                json args = json::parse(tc.arguments);
                                path = args.value("path", "");
                                start_line = args.value("start_line", 1);
                                end_line = args.value("end_line", -1);
                            } catch (...) { path = tc.arguments; }
                            LOG_INFO("Executing built-in tool read_file: %s", path.c_str());
                            if (!is_path_authorized(path)) {
                                exec_output = "[Authorization Required: The path '" + path + "' is outside the current workspace directory (" + moecher::tooling::get_workspace_directory() + "). Action requires explicit user authorization for path: " + path + "]";
                            } else {
                                exec_output = moecher::tooling::read_file(path, start_line, end_line);
                            }
                        } else if (tc.name == "write_file") {
                            std::string path;
                            std::string content;
                            bool overwrite = true;
                            try {
                                json args = json::parse(tc.arguments);
                                path = args.value("path", "");
                                content = args.value("content", "");
                                overwrite = args.value("overwrite", true);
                            } catch (...) { path = tc.arguments; }
                            LOG_INFO("Executing built-in tool write_file: %s (%zu bytes)", path.c_str(), content.size());
                            if (!is_path_authorized(path)) {
                                exec_output = "[Authorization Required: The path '" + path + "' is outside the current workspace directory (" + moecher::tooling::get_workspace_directory() + "). Action requires explicit user authorization for path: " + path + "]";
                            } else {
                                exec_output = moecher::tooling::write_file(path, content, overwrite);
                            }
                        } else if (tc.name == "edit_file") {
                            std::string path;
                            std::string target_content;
                            std::string replacement_content;
                            try {
                                json args = json::parse(tc.arguments);
                                path = args.value("path", "");
                                target_content = args.value("target_content", "");
                                replacement_content = args.value("replacement_content", "");
                            } catch (...) { path = tc.arguments; }
                            LOG_INFO("Executing built-in tool edit_file: %s", path.c_str());
                            if (!is_path_authorized(path)) {
                                exec_output = "[Authorization Required: The path '" + path + "' is outside the current workspace directory (" + moecher::tooling::get_workspace_directory() + "). Action requires explicit user authorization for path: " + path + "]";
                            } else {
                                exec_output = moecher::tooling::edit_file(path, target_content, replacement_content);
                            }
                        } else if (tc.name == "execute_command") {
                            std::string cmd;
                            try {
                                json args = json::parse(tc.arguments);
                                cmd = args.value("command", "");
                            } catch (...) { cmd = tc.arguments; }
                            LOG_INFO("Executing built-in tool execute_command: %s (timeout=%dms)", cmd.c_str(), execution_timeout_ms);
                            exec_output = moecher::tooling::execute_system_command(cmd, execution_timeout_ms);
                        } else if (moecher::mcp::MCPManager::instance().has_tool(tc.name)) {
                            std::string s_id, raw_tname;
                            moecher::mcp::MCPManager::instance().has_tool(tc.name, &s_id, &raw_tname);
                            json parsed_args = json::object();
                            try { parsed_args = json::parse(tc.arguments); } catch (...) {}
                            LOG_INFO("Executing MCP tool: [%s:%s]", s_id.c_str(), raw_tname.c_str());
                            exec_output = moecher::mcp::MCPManager::instance().call_tool(tc.name, parsed_args, execution_timeout_ms);
                        } else {
                            can_execute_all = false;
                            break;
                        }

                        json tool_reply = {
                            {"role", "tool"},
                            {"tool_call_id", tc.id},
                            {"content", exec_output}
                        };
                        current_messages.push_back(tool_reply);
                        round_new_tool_msgs.push_back(tool_reply);
                    }

                    if (!can_execute_all) {
                        finish_reason = "tool_calls";
                        break;
                    }
                }

                if (do_server_exec && !emitted_tool_calls.empty() && finish_reason != "tool_calls") {
                    LOG_WARN("Tool execution reached max rounds limit (%d).", req_max_tool_rounds);
                    final_response_text += "\n\n> ⚠️ **Maximum tool execution rounds reached (" + std::to_string(req_max_tool_rounds) + "/" + std::to_string(req_max_tool_rounds) + ").** Execution stopped.";
                }

                json choice = {
                    {"index", 0},
                    {"message", {
                        {"role", "assistant"},
                        {"content", final_response_text}
                    }},
                    {"finish_reason", finish_reason}
                };
                if (!engine.last_reasoning_text_.empty()) {
                    choice["message"]["reasoning_content"] = engine.last_reasoning_text_;
                }

                if (!emitted_tool_calls.empty() && (!g_server_exec || finish_reason == "tool_calls")) {
                    json tc_arr = json::array();
                    for (const auto& tc : emitted_tool_calls) {
                        tc_arr.push_back({
                            {"id", tc.id},
                            {"type", tc.type},
                            {"function", {{"name", tc.name}, {"arguments", tc.arguments}}}
                        });
                    }
                    choice["message"]["tool_calls"] = tc_arr;
                }

                json response = {
                    {"id", req_id},
                    {"object", "chat.completion"},
                    {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                    {"model", model_id},
                    {"choices", {choice}},
                    {"usage", {
                        {"prompt_tokens", engine.prompt_token_count_},
                        {"completion_tokens", engine.completion_token_count_},
                        {"total_tokens", engine.prompt_token_count_ + engine.completion_token_count_},
                        {"evaluated_prompt_tokens", engine.last_prefill_tokens_},
                        {"prefill_sec", engine.last_prefill_sec_},
                        {"prefill_tps", engine.last_prefill_tps_},
                        {"decode_sec", engine.last_decode_sec_},
                        {"decode_tps", engine.last_decode_tps_},
                        {"prompt_tokens_details", {
                            {"cached_tokens", engine.last_cached_tokens_}
                        }}
                    }}
                };
                res.set_content(response.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");

                // Save conversation state for turn continuation
                {
                    std::lock_guard<std::mutex> lock(g_engine_mutex);
                    json conv_to_save = current_messages;
                    if (conv_to_save.empty() || conv_to_save.back().value("role", "") != "assistant") {
                        conv_to_save.push_back({{"role", "assistant"}, {"content", final_response_text}});
                    }
                    s_last_conv_messages = conv_to_save;
                }
                } catch (const std::exception& e) {
                    LOG_ERROR("[Completion] Exception during request execution: %s", e.what());
                    json err_resp = {{"error", {{"message", e.what()}, {"type", "server_error"}}}};
                    res.status = 500;
                    res.set_content(err_resp.dump(), "application/json");
                }
            }
            auto end = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
            LOG_INFO("RESPONSE (%.0fms)", elapsed_ms);

            if (g_track_experts && !engine.expert_freq_counts_.empty()) {
                std::string fpath = engine.model_dir_.empty() ? "expert_freq.bin" : (engine.model_dir_ + "/expert_freq.bin");
                engine.save_expert_freq(fpath);
                engine.save_expert_profile(engine.model_dir_);
            }
        });

    // Stop / abort generation endpoints
    svr.Post("/v1/chat/stop", [](const httplib::Request&, httplib::Response& res) {
        LOG_WARN("Received stop request on /v1/chat/stop");
        g_stop_requested.store(true);
        json body = {{"status", "ok"}, {"message", "Generation stopping"}};
        res.set_content(body.dump(), "application/json");
    });
    svr.Post("/v1/stop", [](const httplib::Request&, httplib::Response& res) {
        LOG_WARN("Received stop request on /v1/stop");
        g_stop_requested.store(true);
        json body = {{"status", "ok"}, {"message", "Generation stopping"}};
        res.set_content(body.dump(), "application/json");
    });

    LOG_INFO("Binding to 0.0.0.0:%d...", port);
    if (!svr.bind_to_port("0.0.0.0", port)) {
        LOG_ERROR("Failed to bind to 0.0.0.0:%d: %s (errno=%d)", port, strerror(errno), errno);
        moecher::mcp::MCPManager::instance().stop_all();
        return;
    }
    LOG_INFO("Server listening on port %d", port);
    LOG_INFO("version %s", MOECHER_VERSION);
    g_server_ready = true;
    svr.listen_after_bind();
    LOG_INFO("Server stopped on port %d", port);
    moecher::mcp::MCPManager::instance().stop_all();
}


} // namespace moecher::server
