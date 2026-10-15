#pragma once

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include "tokenizer.hpp"

namespace moecher::zdb {

struct ZeroDecodeCandidate {
    std::string choice;
    std::vector<int> token_ids;
    float raw_logit = -1e9f;
    float probability = 0.0f;
};

struct ZeroDecodeResult {
    int winning_index = -1;
    std::string winning_choice;
    float confidence = 0.0f;
    std::vector<ZeroDecodeCandidate> candidates;
    std::map<std::string, float> probabilities;
    int global_argmax_token_id = -1;
    std::string global_argmax_token_str;
    float global_argmax_logit = -1e9f;
    bool is_choice_top1 = false;
    float top1_gap = 0.0f;           // distance between global argmax logit and best candidate logit
    float candidate_margin = 0.0f;   // distance between #1 candidate logit and #2 candidate logit
    bool gate_passed = false;        // true if candidate passes System 1 confidence & alignment arbiter
    std::string system_path = "system_1_zdb"; // "system_1_zdb" or "system_2_fallback"
    double prefill_sec = 0.0;
    double total_sec = 0.0;
    size_t prompt_tokens = 0;
    int decode_passes = 0; // strictly 0
    std::string rendered_output;
};

// Maps candidate choices to token IDs (including case and space variations)
// and returns the deduplicated list of all token IDs that need to be read from GPU.
std::vector<int> collect_candidate_tokens(
    const std::vector<std::string>& candidate_choices,
    const BPETokenizer& tok,
    int vocab_size,
    std::vector<std::vector<int>>& out_choice_tokens
);

// Evaluates candidate choices given GPU-extracted logit lookup, finds winners, computes
// temperature-scaled softmax, performs hybrid arbiter gating checks, and renders JSON or text output.
ZeroDecodeResult score_and_underwrite_candidates(
    const std::vector<std::string>& candidate_choices,
    const std::vector<std::vector<int>>& choice_tokens,
    const std::unordered_map<int, float>& logit_lookup,
    int global_argmax_token_id,
    const std::string& global_argmax_token_str,
    float global_argmax_logit,
    float temperature,
    float min_confidence,
    float max_top1_gap,
    const std::string& output_format
);

// Builds the nested zero_decode_bypass JSON object used in /v1/chat/completions
nlohmann::json build_zdb_json_object(const ZeroDecodeResult& res);

// Builds fallback JSON response when System 1 gate fails and System 2 autoregressive decode triggers
nlohmann::json build_zdb_fallback_response_json(
    const ZeroDecodeResult& res,
    const std::string& matched_choice,
    const std::string& sys2_text,
    int completion_token_count
);

} // namespace moecher::zdb
