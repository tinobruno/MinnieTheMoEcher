#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <nlohmann/json.hpp>

class BPETokenizer {
public:
    bool load(const std::string& path);
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;
    std::string decode_token_str(int id) const;
    int get_token_id(const std::string& token) const;

private:
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<int, std::string> id_to_token_;
    std::vector<std::string> merges_;
    std::unordered_map<std::string, int> merge_rank_;
    std::vector<std::pair<std::string, int>> special_tokens_;

    uint32_t byte_to_char_[256];
    std::unordered_map<char32_t, uint8_t> char_to_byte_;

    void build_byte_mapping();
    std::string bytes_to_unicode(const std::string& bytes) const;
    std::string decode_token(const std::string& token) const;
    void split_on_special(const std::string& text, std::vector<std::pair<std::string, bool>>& out) const;
    void bpe_emit_piece(const std::string& raw_piece, std::vector<int>& ids) const;
    void joyai_tokenize_segment(const std::string& text, std::vector<int>& out) const;
};
