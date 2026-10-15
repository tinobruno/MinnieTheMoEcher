#pragma once

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "tokenizer.hpp"

namespace moecher::multimodal {

struct MultimodalPrompt {
    std::vector<int> prompt;
    std::string image_b64;
    int visual_start_pos = -1;
    int visual_num_tokens = 0;
};

std::string get_message_content_string(const nlohmann::json& msg);
bool message_has_image(const nlohmann::json& msg);
std::string extract_image_b64_from_message(const nlohmann::json& msg);

MultimodalPrompt apply_chat_template_multimodal(
    const nlohmann::json& messages,
    const BPETokenizer& tok,
    bool enable_thinking = true,
    const std::string& reasoning_effort = "high",
    const nlohmann::json& tools = nlohmann::json(),
    bool model_has_vision = false
);

std::vector<int> apply_chat_template(
    const nlohmann::json& messages,
    const BPETokenizer& tok,
    bool enable_thinking = true,
    const std::string& reasoning_effort = "high",
    const nlohmann::json& tools = nlohmann::json()
);

} // namespace moecher::multimodal

// Global aliases for transparent static-linking compatibility
using moecher::multimodal::MultimodalPrompt;
using moecher::multimodal::get_message_content_string;
using moecher::multimodal::message_has_image;
using moecher::multimodal::extract_image_b64_from_message;
using moecher::multimodal::apply_chat_template_multimodal;
using moecher::multimodal::apply_chat_template;
