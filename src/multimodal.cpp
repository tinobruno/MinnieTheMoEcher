#include "multimodal.hpp"
#include "agentic.hpp"
#include "logging.hpp"

#include <algorithm>
#include <cctype>

using json = nlohmann::json;

namespace moecher::multimodal {



std::string get_message_content_string(const json& msg) {
    if (!msg.is_object() || !msg.contains("content")) return "";
    const auto& c = msg["content"];
    if (c.is_string()) {
        return c.get<std::string>();
    }
    if (c.is_array()) {
        std::string res;
        for (const auto& part : c) {
            if (part.is_string()) {
                res += part.get<std::string>();
            } else if (part.is_object()) {
                std::string ptype = part.value("type", "text");
                if (ptype == "text") {
                    res += part.value("text", "");
                }
            }
        }
        return res;
    }
    return "";
}

bool message_has_image(const json& msg) {
    if (!msg.is_object()) return false;
    if (msg.contains("image") && !msg["image"].is_null()) return true;
    if (msg.contains("image_url") && !msg["image_url"].is_null()) return true;
    if (msg.contains("content") && msg["content"].is_array()) {
        for (const auto& part : msg["content"]) {
            if (part.is_object()) {
                std::string ptype = part.value("type", "");
                if (ptype == "image" || ptype == "image_url" || part.contains("image") || part.contains("image_url")) {
                    return true;
                }
            }
        }
    }
    return false;
}

std::string extract_image_b64_from_message(const json& msg) {
    if (!msg.is_object()) return "";
    auto extract_from_url_str = [](const std::string& url) -> std::string {
        if (url.empty()) return "";
        size_t b64_pos = url.find("base64,");
        if (b64_pos != std::string::npos) {
            return url.substr(b64_pos + 7);
        }
        return url;
    };

    if (msg.contains("image")) {
        if (msg["image"].is_string()) {
            return extract_from_url_str(msg["image"].get<std::string>());
        } else if (msg["image"].is_object()) {
            return extract_from_url_str(msg["image"].value("url", ""));
        }
    }
    if (msg.contains("image_url")) {
        if (msg["image_url"].is_string()) {
            return extract_from_url_str(msg["image_url"].get<std::string>());
        } else if (msg["image_url"].is_object()) {
            return extract_from_url_str(msg["image_url"].value("url", ""));
        }
    }
    if (msg.contains("content") && msg["content"].is_array()) {
        for (const auto& part : msg["content"]) {
            if (part.is_object()) {
                std::string ptype = part.value("type", "");
                if (ptype == "image" || ptype == "image_url" || part.contains("image") || part.contains("image_url")) {
                    if (part.contains("image_url")) {
                        if (part["image_url"].is_string()) {
                            return extract_from_url_str(part["image_url"].get<std::string>());
                        } else if (part["image_url"].is_object()) {
                            return extract_from_url_str(part["image_url"].value("url", ""));
                        }
                    }
                    if (part.contains("image")) {
                        if (part["image"].is_string()) {
                            return extract_from_url_str(part["image"].get<std::string>());
                        } else if (part["image"].is_object()) {
                            return extract_from_url_str(part["image"].value("url", ""));
                        }
                    }
                    if (part.contains("url") && part["url"].is_string()) {
                        return extract_from_url_str(part["url"].get<std::string>());
                    }
                }
            }
        }
    }
    return "";
}

MultimodalPrompt apply_chat_template_multimodal(
    const json& messages,
    const BPETokenizer& tok,
    bool enable_thinking,
    const std::string& reasoning_effort,
    const json& tools,
    bool model_has_vision,
    bool is_nemotron,
    bool is_deepseek_v41)
{
    MultimodalPrompt out;
    json resolved_tools = tools;
    if (!resolved_tools.empty()) {
        resolved_tools = resolve_canonical_tools(resolved_tools);
    }
    bool has_tools = (!resolved_tools.empty() && resolved_tools.is_array());
    int IM_START = tok.get_token_id("<|im_start|>");
    int IM_END = tok.get_token_id("<|im_end|>");
    bool is_qwen = (IM_START >= 0 && IM_END >= 0);
    std::string tools_system_prompt = has_tools ? build_dynamic_tools_prompt(resolved_tools, is_qwen) : "";
    has_tools = !tools_system_prompt.empty();

    if (IM_START >= 0 && IM_END >= 0) {
        // ChatML template (Qwen / Llama / SmolLM)
        std::vector<int> result;
        bool has_system = (!messages.empty() && messages[0].value("role", "") == "system");
        if (!has_system) {
            std::string sys_prompt = "";
            if (is_nemotron) {
                std::string base = g_base_system_prompt.empty() ? "You are a helpful assistant." : g_base_system_prompt;
                sys_prompt = has_tools ? (base + tools_system_prompt) : base;
            } else if (has_tools) {
                sys_prompt = tools_system_prompt;
            } else if (enable_thinking) {
                sys_prompt = "You are a helpful assistant";
            }
            if (!sys_prompt.empty()) {
                result.push_back(IM_START);
                auto sys_role = tok.encode("system\n" + sys_prompt);
                result.insert(result.end(), sys_role.begin(), sys_role.end());
                result.push_back(IM_END);
                auto nl_enc = tok.encode("\n");
                result.insert(result.end(), nl_enc.begin(), nl_enc.end());
            }
        }
        int last_user_idx = -1;
        for (int i = 0; i < (int)messages.size(); i++) {
            if (messages[i].value("role", "") == "user") {
                last_user_idx = i;
            }
        }

        for (size_t i = 0; i < messages.size(); i++) {
            std::string role = messages[i].value("role", "user");
            std::string content_str = "";
            std::string extracted_image_b64 = "";

            if (messages[i].contains("content")) {
                const auto& c = messages[i]["content"];
                if (c.is_string()) {
                    content_str = c.get<std::string>();
                } else if (c.is_array()) {
                    for (const auto& part : c) {
                        if (part.is_object()) {
                            std::string ptype = part.value("type", "text");
                            if (ptype == "text") {
                                content_str += part.value("text", "");
                            } else if ((ptype == "image_url" || ptype == "image") || part.contains("image_url") || part.contains("image")) {
                                std::string url;
                                if (part.contains("image_url")) {
                                    if (part["image_url"].is_string()) {
                                        url = part["image_url"].get<std::string>();
                                    } else if (part["image_url"].is_object()) {
                                        url = part["image_url"].value("url", "");
                                    }
                                } else if (part.contains("image")) {
                                    if (part["image"].is_string()) {
                                        url = part["image"].get<std::string>();
                                    } else if (part["image"].is_object()) {
                                        url = part["image"].value("url", "");
                                    }
                                }
                                if (!url.empty()) {
                                    size_t b64_pos = url.find("base64,");
                                    if (b64_pos != std::string::npos) {
                                        extracted_image_b64 = url.substr(b64_pos + 7);
                                    } else {
                                        extracted_image_b64 = url;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (extracted_image_b64.empty() && messages[i].contains("image") && messages[i]["image"].is_string()) {
                std::string url = messages[i]["image"].get<std::string>();
                size_t b64_pos = url.find("base64,");
                extracted_image_b64 = (b64_pos != std::string::npos) ? url.substr(b64_pos + 7) : url;
            }

            if (role == "system" && i == 0) {
                content_str = update_system_prompt_with_tools(content_str, tools_system_prompt, has_tools);
            }

            if (role == "tool" || role == "function") {
                result.push_back(IM_START);
                auto user_role = tok.encode("user\n<tool_response>\n" + content_str + "\n</tool_response>");
                result.insert(result.end(), user_role.begin(), user_role.end());
                result.push_back(IM_END);
                auto nl_enc = tok.encode("\n");
                result.insert(result.end(), nl_enc.begin(), nl_enc.end());
                continue;
            }

            result.push_back(IM_START);
            auto role_enc = tok.encode(role + "\n");
            result.insert(result.end(), role_enc.begin(), role_enc.end());

            // Multimodal image injection for Qwen Vision
            if (model_has_vision && !extracted_image_b64.empty() && role == "user") {
                int VISION_START = tok.get_token_id("<|vision_start|>");
                if (VISION_START < 0) VISION_START = 248053;
                int VISION_END = tok.get_token_id("<|vision_end|>");
                if (VISION_END < 0) VISION_END = 248054;
                int IMAGE_PAD = tok.get_token_id("<|image_pad|>");
                if (IMAGE_PAD < 0) IMAGE_PAD = 248056;

                out.image_b64 = extracted_image_b64;
                result.push_back(VISION_START);
                out.visual_start_pos = (int)result.size();
                out.visual_num_tokens = 576;
                result.insert(result.end(), 576, IMAGE_PAD);
                result.push_back(VISION_END);
            }

            std::string body = content_str;
            if (is_nemotron && role == "user" && (int)i == last_user_idx) {
                if (enable_thinking && (reasoning_effort == "efficient" || reasoning_effort == "medium" || reasoning_effort == "low")) {
                    body += "\n\n{reasoning effort: efficient}";
                }
            }
            if (model_has_vision && !extracted_image_b64.empty() && role == "user") {
                std::string trimmed = body;
                while (!trimmed.empty() && (trimmed.front() == ' ' || trimmed.front() == '\n' || trimmed.front() == '\r' || trimmed.front() == '\t')) trimmed.erase(trimmed.begin());
                while (!trimmed.empty() && (trimmed.back() == ' ' || trimmed.back() == '\n' || trimmed.back() == '\r' || trimmed.back() == '\t' || trimmed.back() == '.' || trimmed.back() == '?' || trimmed.back() == '!')) trimmed.pop_back();
                std::string lower = trimmed;
                for (char& c : lower) c = (char)std::tolower((unsigned char)c);

                if (lower.empty() ||
                    lower == "describe this image" || lower == "describe the image" || lower == "describe image" ||
                    lower == "describe this picture" || lower == "describe picture" || lower == "describe" ||
                    lower == "what is this" || lower == "what is this image" || lower == "what is this picture" ||
                    lower == "what's this" || lower == "what do you see" || lower == "what do you see in this image" ||
                    lower == "what do you see in this picture" || lower == "analyze this image" || lower == "analyze image" ||
                    lower == "describe what is depicted in this photo" || lower == "describe what you see in this image") {
                    body = "What is shown in this image?";
                }
            }
            if (role == "assistant") {
                if (messages[i].contains("reasoning_content") && messages[i]["reasoning_content"].is_string()) {
                    std::string r_content = messages[i]["reasoning_content"].get<std::string>();
                    if (!r_content.empty()) {
                        body = "<think>\n" + r_content + "\n</think>\n" + body;
                    }
                } else if (is_nemotron) {
                    if (body.find("<think>") == std::string::npos && body.find("</think>") == std::string::npos) {
                        body = "<think></think>" + body;
                    }
                }
                if (messages[i].contains("tool_calls") && messages[i]["tool_calls"].is_array()) {
                    for (const auto& tc : messages[i]["tool_calls"]) {
                        std::string fn_name = tc.value("function", json::object()).value("name", "");
                        json fn_args = tc.value("function", json::object()).value("arguments", json::object());
                        std::string args_str = fn_args.is_string() ? fn_args.get<std::string>() : fn_args.dump();
                        if (!body.empty() && body.back() != '\n') body += "\n";
                        body += "<tool_call>\n{\"name\": \"" + fn_name + "\", \"arguments\": " + args_str + "}\n</tool_call>\n";
                    }
                }
            }

            auto content_enc = tok.encode(body);
            result.insert(result.end(), content_enc.begin(), content_enc.end());
            result.push_back(IM_END);
            auto nl_enc = tok.encode("\n");
            result.insert(result.end(), nl_enc.begin(), nl_enc.end());
        }
        if (!messages.empty()) {
            std::string last_role = messages.back().value("role", "user");
            if (last_role == "user" || last_role == "tool" || last_role == "function") {
                result.push_back(IM_START);
                auto asst_enc = tok.encode("assistant\n");
                result.insert(result.end(), asst_enc.begin(), asst_enc.end());
                if (is_nemotron) {
                    if (enable_thinking) {
                        auto think_enc = tok.encode("<think>\n");
                        result.insert(result.end(), think_enc.begin(), think_enc.end());
                    } else {
                        auto think_enc = tok.encode("<think></think>");
                        result.insert(result.end(), think_enc.begin(), think_enc.end());
                    }
                } else {
                    bool actual_thinking = enable_thinking && out.image_b64.empty();
                    if (actual_thinking) {
                        auto think_enc = tok.encode("<think>\n");
                        result.insert(result.end(), think_enc.begin(), think_enc.end());
                    } else {
                        auto think_enc = tok.encode("<think>\n\n</think>\n");
                        result.insert(result.end(), think_enc.begin(), think_enc.end());
                    }
                }
            }
        }
        out.prompt = result;
        return out;
    }

    int BOS = tok.get_token_id("<｜begin of sentence｜>");
    if (BOS < 0) BOS = tok.get_token_id("<｜begin\xe2\x96\x81of\xe2\x96\x81sentence｜>");
    if (BOS < 0) BOS = 0;

    int EOS = tok.get_token_id("<｜end of sentence｜>");
    if (EOS < 0) EOS = tok.get_token_id("<｜end\xe2\x96\x81of\xe2\x96\x81sentence｜>");
    if (EOS < 1) EOS = 1;

    int SYSTEM = tok.get_token_id("<｜System｜>");
    if (SYSTEM < 0) SYSTEM = tok.get_token_id("<｜System\xe2\x96\x81｜>");
    if (SYSTEM < 0) SYSTEM = 128799;

    int USER = tok.get_token_id("<｜User｜>");
    if (USER < 0) USER = 128803;

    int ASSISTANT = tok.get_token_id("<｜Assistant｜>");
    if (ASSISTANT < 0) ASSISTANT = 128804;

    int THINK_BEGIN = tok.get_token_id("<think>");
    if (THINK_BEGIN < 0) THINK_BEGIN = 128821;

    int THINK_END = tok.get_token_id("</think>");
    if (THINK_END < 0) THINK_END = 128822;

    bool has_image = false;
    for (const auto& msg : messages) {
        if (message_has_image(msg)) {
            has_image = true;
            break;
        }
    }
    bool actual_thinking = enable_thinking && !has_image;

    int budget = 75;
    std::string effort_lower = reasoning_effort;
    for (char& c : effort_lower) c = (char)std::tolower((unsigned char)c);
    if (effort_lower == "low") {
        budget = 50;
    } else if (effort_lower == "high" || effort_lower.empty()) {
        budget = 75;
    } else if (effort_lower == "max") {
        budget = 100;
    } else {
        try {
            int parsed = std::stoi(reasoning_effort);
            if (parsed >= 1 && parsed <= 100) {
                budget = parsed;
            }
        } catch (...) {}
    }

    std::string reasoning_effort_prompt = "";
    if (actual_thinking && is_deepseek_v41) {
        reasoning_effort_prompt = "Reasoning Effort: " + std::to_string(budget) +
            " (range 1-100, the higher the value, the more thorough the reasoning)\n\n";
    }

    std::vector<int> result;
    result.push_back(BOS);

    bool has_system = (!messages.empty() && messages[0].value("role", "") == "system");
    std::string default_sys = "";
    if (has_tools) {
        default_sys = g_base_system_prompt + tools_system_prompt;
    } else if (!g_base_system_prompt.empty()) {
        default_sys = g_base_system_prompt;
    }

    // Lead conversation with <｜System｜> if (is_deepseek_v41 && actual_thinking) OR first message is system OR default_sys is non-empty
    if ((is_deepseek_v41 && actual_thinking) || has_system || !default_sys.empty()) {
        result.push_back(SYSTEM);
        std::string leading_sys = reasoning_effort_prompt;
        if (has_system) {
            std::string content = messages[0].contains("content") ? get_message_content_string(messages[0]) : "";
            std::string sys_body = update_system_prompt_with_tools(content, tools_system_prompt, has_tools);
            leading_sys += sys_body;
        } else if (!default_sys.empty()) {
            leading_sys += default_sys;
        }
        if (!leading_sys.empty()) {
            auto enc = tok.encode(leading_sys);
            result.insert(result.end(), enc.begin(), enc.end());
        }
    }

    for (size_t i = 0; i < messages.size(); i++) {
        std::string role = messages[i].value("role", "user");
        std::string content = "";
        std::string extracted_image_b64 = "";

        if (messages[i].contains("content")) {
            const auto& c = messages[i]["content"];
            if (c.is_string()) {
                content = c.get<std::string>();
            } else if (c.is_array()) {
                for (const auto& part : c) {
                    if (part.is_object()) {
                        std::string type = part.value("type", "text");
                        if (type == "text") {
                            content += part.value("text", "");
                        } else if (type == "image_url" || type == "image") {
                            std::string url = "";
                            if (part.contains("image_url")) {
                                if (part["image_url"].is_string()) {
                                    url = part["image_url"].get<std::string>();
                                } else if (part["image_url"].is_object()) {
                                    url = part["image_url"].value("url", "");
                                }
                            } else if (part.contains("image")) {
                                if (part["image"].is_string()) {
                                    url = part["image"].get<std::string>();
                                } else if (part["image"].is_object()) {
                                    url = part["image"].value("url", "");
                                }
                            }
                            if (!url.empty()) {
                                size_t b64_pos = url.find("base64,");
                                extracted_image_b64 = (b64_pos != std::string::npos) ? url.substr(b64_pos + 7) : url;
                            }
                        }
                    }
                }
            }
        }
        if (content.empty()) {
            content = get_message_content_string(messages[i]);
        }
        if (extracted_image_b64.empty() && messages[i].contains("image") && messages[i]["image"].is_string()) {
            std::string url = messages[i]["image"].get<std::string>();
            size_t b64_pos = url.find("base64,");
            extracted_image_b64 = (b64_pos != std::string::npos) ? url.substr(b64_pos + 7) : url;
        }

        if (role == "system") {
            if (i > 0) {
                // Mid-conversation system message
                result.push_back(SYSTEM);
                std::string sys_body = content;
                if (!sys_body.empty() && sys_body.back() != '\n') sys_body += "\n";
                auto enc = tok.encode(sys_body);
                result.insert(result.end(), enc.begin(), enc.end());
            }
            // Note: i == 0 was already handled above in the leading <｜System｜> block!
        } else if (role == "user") {
            result.push_back(USER);
            if (model_has_vision && !extracted_image_b64.empty()) {
                int IMAGE_TOKEN = tok.get_token_id("<｜image｜>");
                if (IMAGE_TOKEN < 0) IMAGE_TOKEN = 129279;

                out.image_b64 = extracted_image_b64;
                out.visual_start_pos = (int)result.size();
                out.visual_num_tokens = 576;
                result.insert(result.end(), 576, IMAGE_TOKEN);
                auto newline_enc = tok.encode("\n");
                result.insert(result.end(), newline_enc.begin(), newline_enc.end());

                size_t img_tag = content.find("<image>");
                if (img_tag != std::string::npos) {
                    content.erase(img_tag, 7);
                    if (img_tag < content.size() && content[img_tag] == '\n') {
                        content.erase(img_tag, 1);
                    }
                }
            }
            std::string prompt_text = content;
            if (model_has_vision && !extracted_image_b64.empty()) {
                std::string trimmed = prompt_text;
                while (!trimmed.empty() && (trimmed.front() == ' ' || trimmed.front() == '\n' || trimmed.front() == '\r' || trimmed.front() == '\t')) trimmed.erase(trimmed.begin());
                while (!trimmed.empty() && (trimmed.back() == ' ' || trimmed.back() == '\n' || trimmed.back() == '\r' || trimmed.back() == '\t' || trimmed.back() == '.' || trimmed.back() == '?' || trimmed.back() == '!')) trimmed.pop_back();
                std::string lower = trimmed;
                for (char& c : lower) c = (char)std::tolower((unsigned char)c);

                if (lower.empty() ||
                    lower == "describe this image" || lower == "describe the image" || lower == "describe image" ||
                    lower == "describe this picture" || lower == "describe picture" || lower == "describe" ||
                    lower == "what is this" || lower == "what is this image" || lower == "what is this picture" ||
                    lower == "what's this" || lower == "what do you see" || lower == "what do you see in this image" ||
                    lower == "what do you see in this picture" || lower == "analyze this image" || lower == "analyze image" ||
                    lower == "describe what is depicted in this photo" || lower == "describe what you see in this image") {
                    prompt_text = "What is shown in this image?";
                }
            }
            auto enc = tok.encode(prompt_text);
            result.insert(result.end(), enc.begin(), enc.end());
        } else if (role == "tool" || role == "function") {
            result.push_back(USER);
            auto enc = tok.encode("<tool_result>\n" + content + "\n</tool_result>");
            result.insert(result.end(), enc.begin(), enc.end());
        } else if (role == "assistant") {
            result.push_back(ASSISTANT);
            if (messages[i].contains("reasoning_content") && messages[i]["reasoning_content"].is_string()) {
                std::string r_content = messages[i]["reasoning_content"].get<std::string>();
                if (!r_content.empty()) {
                    result.push_back(THINK_BEGIN);
                    auto r_enc = tok.encode(r_content);
                    result.insert(result.end(), r_enc.begin(), r_enc.end());
                    result.push_back(THINK_END);
                } else if (is_deepseek_v41) {
                    result.push_back(THINK_END);
                }
            } else if (is_deepseek_v41) {
                result.push_back(THINK_END);
            }
            std::string body = content;
            if (messages[i].contains("tool_calls") && messages[i]["tool_calls"].is_array()) {
                for (const auto& tc : messages[i]["tool_calls"]) {
                    std::string fn_name = tc.value("function", json::object()).value("name", "");
                    json fn_args = tc.value("function", json::object()).value("arguments", json::object());
                    std::string args_str = fn_args.is_string() ? fn_args.get<std::string>() : fn_args.dump();
                    if (!body.empty() && body.back() != '\n') body += "\n";
                    body += "<tool_call>\n{\"name\": \"" + fn_name + "\", \"arguments\": " + args_str + "}\n</tool_call>\n";
                }
            }
            auto enc = tok.encode(body);
            result.insert(result.end(), enc.begin(), enc.end());
            result.push_back(EOS); // <｜end of sentence｜>
        }
    }

    // Add assistant prompt for generation
    if (!messages.empty()) {
        std::string last_role = messages.back().value("role", "user");
        if (last_role == "user" || last_role == "tool" || last_role == "function" || (last_role == "system" && messages.size() > 1)) {
            result.push_back(ASSISTANT);
            if (actual_thinking) {
                result.push_back(THINK_BEGIN);
            } else if (is_deepseek_v41) {
                result.push_back(THINK_END);
            } else {
                result.push_back(THINK_BEGIN);
                result.push_back(THINK_END);
                auto nl = tok.encode("\n");
                result.insert(result.end(), nl.begin(), nl.end());
            }
        }
    }

    out.prompt = result;
    return out;
}

std::vector<int> apply_chat_template(
    const json& messages,
    const BPETokenizer& tok,
    bool enable_thinking,
    const std::string& reasoning_effort,
    const json& tools,
    bool is_nemotron,
    bool is_deepseek_v41)
{
    return apply_chat_template_multimodal(messages, tok, enable_thinking, reasoning_effort, tools, false, is_nemotron, is_deepseek_v41).prompt;
}

} // namespace moecher::multimodal
