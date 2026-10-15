#include "zdb.hpp"
#include "logging.hpp"

#include <cmath>
#include <algorithm>
#include <cctype>

using json = nlohmann::json;

namespace moecher::zdb {

std::vector<int> collect_candidate_tokens(
    const std::vector<std::string>& candidate_choices,
    const BPETokenizer& tok,
    int vocab_size,
    std::vector<std::vector<int>>& out_choice_tokens
) {
    out_choice_tokens.resize(candidate_choices.size());
    std::vector<int> all_tokens_to_read;

    for (size_t c_idx = 0; c_idx < candidate_choices.size(); c_idx++) {
        const std::string& choice = candidate_choices[c_idx];
        auto& choice_toks = out_choice_tokens[c_idx];

        auto add_tok = [&](const std::vector<int>& toks) {
            if (!toks.empty()) {
                int t = toks[0];
                if (t >= 0 && t < vocab_size) {
                    if (std::find(choice_toks.begin(), choice_toks.end(), t) == choice_toks.end()) {
                        choice_toks.push_back(t);
                    }
                    if (std::find(all_tokens_to_read.begin(), all_tokens_to_read.end(), t) == all_tokens_to_read.end()) {
                        all_tokens_to_read.push_back(t);
                    }
                }
            }
        };

        add_tok(tok.encode(choice));
        add_tok(tok.encode(" " + choice));

        std::string lower_c = choice;
        for (char& ch : lower_c) ch = (char)::tolower((unsigned char)ch);
        if (lower_c != choice) {
            add_tok(tok.encode(lower_c));
            add_tok(tok.encode(" " + lower_c));
        }

        std::string upper_c = choice;
        for (char& ch : upper_c) ch = (char)::toupper((unsigned char)ch);
        if (upper_c != choice) {
            add_tok(tok.encode(upper_c));
            add_tok(tok.encode(" " + upper_c));
        }

        std::string title_c = choice;
        if (!title_c.empty()) {
            title_c[0] = (char)::toupper((unsigned char)title_c[0]);
            for (size_t i = 1; i < title_c.size(); i++) title_c[i] = (char)::tolower((unsigned char)title_c[i]);
            if (title_c != choice && title_c != lower_c && title_c != upper_c) {
                add_tok(tok.encode(title_c));
                add_tok(tok.encode(" " + title_c));
            }
        }
    }

    return all_tokens_to_read;
}

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
) {
    ZeroDecodeResult result;
    result.decode_passes = 0;
    result.global_argmax_token_id = global_argmax_token_id;
    result.global_argmax_token_str = global_argmax_token_str;
    result.global_argmax_logit = global_argmax_logit;

    result.candidates.resize(candidate_choices.size());
    float max_choice_logit = -1e9f;

    for (size_t c_idx = 0; c_idx < candidate_choices.size(); c_idx++) {
        auto& cand = result.candidates[c_idx];
        cand.choice = candidate_choices[c_idx];
        cand.token_ids = choice_tokens[c_idx];
        cand.raw_logit = -1e9f;

        for (int tid : cand.token_ids) {
            auto it = logit_lookup.find(tid);
            if (it != logit_lookup.end()) {
                float l_val = it->second;
                if (l_val > cand.raw_logit) {
                    cand.raw_logit = l_val;
                }
            }
            if (tid == global_argmax_token_id) {
                result.is_choice_top1 = true;
            }
        }

        if (cand.raw_logit > max_choice_logit) {
            max_choice_logit = cand.raw_logit;
            result.winning_index = (int)c_idx;
            result.winning_choice = cand.choice;
        }
    }

    // Softmax normalization across candidate choices
    float eff_temp = std::max(temperature, 0.01f);
    float sum_exp = 0.0f;
    for (auto& cand : result.candidates) {
        float exp_val = std::exp((cand.raw_logit - max_choice_logit) / eff_temp);
        cand.probability = exp_val;
        sum_exp += exp_val;
    }

    if (sum_exp > 0.0f) {
        for (auto& cand : result.candidates) {
            cand.probability /= sum_exp;
            result.probabilities[cand.choice] = cand.probability;
        }
    }

    if (result.winning_index >= 0 && result.winning_index < (int)result.candidates.size()) {
        result.confidence = result.candidates[result.winning_index].probability;
    }

    // ── Hybrid Arbiter (System 1 vs System 2 Confidence & Alignment) ───────
    float second_choice_logit = -1e9f;
    for (size_t c_idx = 0; c_idx < result.candidates.size(); c_idx++) {
        if ((int)c_idx != result.winning_index && result.candidates[c_idx].raw_logit > second_choice_logit) {
            second_choice_logit = result.candidates[c_idx].raw_logit;
        }
    }
    result.candidate_margin = max_choice_logit - second_choice_logit;

    if (result.is_choice_top1) {
        result.top1_gap = 0.0f;
    } else {
        result.top1_gap = std::max(0.0f, result.global_argmax_logit - max_choice_logit);
    }

    // Arbiter gating decision:
    // Distinguish surface formatting tokens (e.g. "**", "*", "\"", "#", ":", "-")
    // from genuine hesitation/scratchpad tokens (e.g. "\n\n", "<think>", "Based", "I").
    bool is_formatting_token = (result.global_argmax_token_str == "**" ||
                                result.global_argmax_token_str == "*" ||
                                result.global_argmax_token_str == "#" ||
                                result.global_argmax_token_str == "##" ||
                                result.global_argmax_token_str == "\"" ||
                                result.global_argmax_token_str == ":" ||
                                result.global_argmax_token_str == "-");

    float effective_gap_tolerance = is_formatting_token ? std::max(max_top1_gap, 2.0f) : max_top1_gap;

    // Pass System 1 if candidate was global top-1 OR if top1_gap is within tolerance AND confidence >= min_confidence
    result.gate_passed = (result.is_choice_top1 || result.top1_gap <= effective_gap_tolerance) &&
                         (result.confidence >= min_confidence);
    result.system_path = result.gate_passed ? "system_1_zdb" : "system_2_fallback";

    // Direct CPU Memory Stitch
    if (output_format == "text" || output_format == "choice") {
        result.rendered_output = result.winning_choice;
    } else {
        json out_json = {
            {"decision", result.winning_choice},
            {"winning_index", result.winning_index},
            {"confidence", result.confidence},
            {"probabilities", result.probabilities},
            {"zero_decode_bypass", true},
            {"decode_passes", 0},
            {"arbiter", {
                {"gate_passed", result.gate_passed},
                {"system_path", result.system_path},
                {"top1_gap", result.top1_gap},
                {"candidate_margin", result.candidate_margin},
                {"is_choice_top1", result.is_choice_top1}
            }},
            {"global_top1", {
                {"token_id", result.global_argmax_token_id},
                {"token", result.global_argmax_token_str},
                {"logit", result.global_argmax_logit},
                {"is_candidate", result.is_choice_top1}
            }},
            {"timing", {
                {"prefill_ms", result.prefill_sec * 1000.0},
                {"total_ms", result.total_sec * 1000.0}
            }}
        };
        result.rendered_output = out_json.dump(-1, ' ', false, json::error_handler_t::replace);
    }

    return result;
}

json build_zdb_json_object(const ZeroDecodeResult& res) {
    return json{
        {"enabled", true},
        {"selected_choice", res.winning_choice},
        {"winning_index", res.winning_index},
        {"confidence", res.confidence},
        {"decode_passes", 0},
        {"probabilities", res.probabilities},
        {"arbiter", {
            {"gate_passed", res.gate_passed},
            {"system_path", res.system_path},
            {"top1_gap", res.top1_gap},
            {"candidate_margin", res.candidate_margin},
            {"is_choice_top1", res.is_choice_top1}
        }},
        {"global_top1", {
            {"token_id", res.global_argmax_token_id},
            {"token", res.global_argmax_token_str},
            {"logit", res.global_argmax_logit},
            {"is_candidate", res.is_choice_top1}
        }},
        {"timing", {
            {"prefill_ms", res.prefill_sec * 1000.0},
            {"total_ms", res.total_sec * 1000.0}
        }}
    };
}

json build_zdb_fallback_response_json(
    const ZeroDecodeResult& res,
    const std::string& matched_choice,
    const std::string& sys2_text,
    int completion_token_count
) {
    return json{
        {"decision", matched_choice},
        {"system_2_output", sys2_text},
        {"zero_decode_bypass", false},
        {"decode_passes", completion_token_count},
        {"arbiter", {
            {"gate_passed", false},
            {"system_path", "system_2_fallback"},
            {"top1_gap", res.top1_gap},
            {"candidate_margin", res.candidate_margin},
            {"is_choice_top1", res.is_choice_top1},
            {"fallback_triggered", true}
        }},
        {"global_top1", {
            {"token_id", res.global_argmax_token_id},
            {"token", res.global_argmax_token_str},
            {"logit", res.global_argmax_logit},
            {"is_candidate", res.is_choice_top1}
        }},
        {"probabilities", res.probabilities}
    };
}

} // namespace moecher::zdb
