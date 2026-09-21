#pragma once
#include <string>
#include <vector>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

namespace moecher::vision {

// ── Base64 Decoding ─────────────────────────────────────────────────────────

static inline std::vector<uint8_t> base64_decode(const std::string& in) {
    std::string clean = in;
    size_t comma = clean.find(',');
    if (comma != std::string::npos) {
        clean = clean.substr(comma + 1);
    }

    static const std::string b64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<int> T(256, -1);
    for (int i = 0; i < 64; i++) T[(unsigned char)b64_chars[i]] = i;

    std::vector<uint8_t> out;
    out.reserve(clean.size() * 3 / 4);

    int val = 0, valb = -8;
    for (unsigned char c : clean) {
        if (c == '\r' || c == '\n' || c == ' ' || c == '\t') continue;
        if (c == '=') break;
        if (T[c] == -1) continue;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back((uint8_t)((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

// ── Image Preprocessor for Qwen2-VL ─────────────────────────────────────────

struct ProcessedImage {
    int width = 768;
    int height = 768;
    int channels = 3;
    int temporal = 2;
    std::vector<float> data; // [T=2, C=3, H=768, W=768] in float32
};

static inline bool preprocess_image(
    const uint8_t* raw_bytes,
    size_t raw_len,
    ProcessedImage& out_img,
    int target_size = 768)
{
    if (!raw_bytes || raw_len == 0) return false;

    int orig_w = 0, orig_h = 0, orig_comp = 0;
    stbi_uc* rgb = stbi_load_from_memory(raw_bytes, (int)raw_len, &orig_w, &orig_h, &orig_comp, 3);
    if (!rgb) {
        return false;
    }

    // Letterbox resize to target_size x target_size preserving aspect ratio
    float scale = std::min((float)target_size / orig_w, (float)target_size / orig_h);
    int new_w = std::clamp((int)std::round(orig_w * scale), 1, target_size);
    int new_h = std::clamp((int)std::round(orig_h * scale), 1, target_size);

    std::vector<uint8_t> resized_rgb(new_w * new_h * 3);
    stbir_resize_uint8_srgb(rgb, orig_w, orig_h, orig_w * 3,
                            resized_rgb.data(), new_w, new_h, new_w * 3,
                            STBIR_RGB);
    stbi_image_free(rgb);

    // Create target_size x target_size canvas with neutral padding (128)
    std::vector<uint8_t> canvas(target_size * target_size * 3, 128);
    int pad_x = (target_size - new_w) / 2;
    int pad_y = (target_size - new_h) / 2;

    for (int y = 0; y < new_h; y++) {
        int dst_y = pad_y + y;
        const uint8_t* src_row = resized_rgb.data() + y * (new_w * 3);
        uint8_t* dst_row = canvas.data() + (dst_y * target_size + pad_x) * 3;
        std::memcpy(dst_row, src_row, new_w * 3);
    }

    // Qwen3.5 / Qwen2.5-VL standard normalization (preprocessor_config.json):
    // mean = [0.5, 0.5, 0.5]
    // std  = [0.5, 0.5, 0.5]
    const float mean[3] = {0.5f, 0.5f, 0.5f};
    const float std_dev[3] = {0.5f, 0.5f, 0.5f};

    size_t plane_size = (size_t)target_size * target_size;
    size_t frame_size = 3 * plane_size;
    out_img.width = target_size;
    out_img.height = target_size;
    out_img.channels = 3;
    out_img.temporal = 2;
    out_img.data.resize(2 * frame_size);

    float* t0_ptr = out_img.data.data();
    float* t1_ptr = out_img.data.data() + frame_size;

    for (int y = 0; y < target_size; y++) {
        for (int x = 0; x < target_size; x++) {
            size_t src_idx = (y * target_size + x) * 3;
            size_t dst_offset = y * target_size + x;

            for (int c = 0; c < 3; c++) {
                float val = ((float)canvas[src_idx + c] / 255.0f - mean[c]) / std_dev[c];
                t0_ptr[c * plane_size + dst_offset] = val;
                t1_ptr[c * plane_size + dst_offset] = val; // temporal duplication for image
            }
        }
    }

    return true;
}

static inline bool preprocess_image_from_base64(
    const std::string& base64_str,
    ProcessedImage& out_img,
    int target_size = 768)
{
    std::vector<uint8_t> raw = base64_decode(base64_str);
    if (raw.empty()) return false;
    return preprocess_image(raw.data(), raw.size(), out_img, target_size);
}

} // namespace moecher::vision
