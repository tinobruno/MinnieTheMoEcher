#pragma once

// ============================================================================
// Universal Hybrid CPU MoE Expert Compute Engine (AVX2 + F16C + FMA)
// ============================================================================
// Computes L1-VRAM-missed MoE experts (e.g. L2.5 Unpinned RAM and overflow
// L2 Pinned RAM misses) in-place on the CPU in parallel with GPU L1-Hit
// execution and PCIe DMA on copy_stream (True Split-Bus Parallelism).
//
// Supports all GGML & GSQ quantization types used across MinnieTheMoEcher models:
//   - Q2_0 (42), IQ1_M (29), IQ2_XXS (16), IQ2_XS (17), IQ2_S (22),
//     IQ3_XXS (18), IQ3_S (21), IQ4_NL (20), IQ4_XS (23)
//   - Q4_K (12), Q5_K (13), Q6_K (14), Q4_0 (2), Q4_1 (3), Q5_0 (6),
//     Q5_1 (7), Q8_0 (8)
//   - BF16 (30), F16 (1), F32 (0)
// ============================================================================

#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#if !defined(_MSC_VER) && !defined(__forceinline)
#define __forceinline inline __attribute__((always_inline))
#endif

namespace moecher {
namespace cpu_moe {

// Pull in GGML block structs and host-side lookup tables (iq2xxs_grid, iq2xs_grid,
// iq2s_grid, iq3xxs_grid, iq3s_grid, iq1s_grid, ksigns64, kvalues_iq4nl) in an
// isolated namespace so there are never symbol collisions with CUDA translation units.
namespace ggml_host {
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "../cuda/ggml-common.h"
#undef GGML_COMMON_DECL_CPP
#undef GGML_COMMON_IMPL_CPP
#undef GGML_COMMON_DECL
#undef GGML_COMMON_IMPL
#undef GGML_TABLE_BEGIN
#undef GGML_TABLE_END
} // namespace ggml_host

using ggml_host::block_q8_1;
using ggml_host::block_q2_0;
using ggml_host::block_q4_0;
using ggml_host::block_q4_1;
using ggml_host::block_q5_0;
using ggml_host::block_q5_1;
using ggml_host::block_q8_0;
using ggml_host::block_q4_K;
using ggml_host::block_q5_K;
using ggml_host::block_q6_K;
using ggml_host::block_iq1_m;
using ggml_host::block_iq2_xxs;
using ggml_host::block_iq2_xs;
using ggml_host::block_iq2_s;
using ggml_host::block_iq3_xxs;
using ggml_host::block_iq3_s;
using ggml_host::block_iq4_nl;
using ggml_host::block_iq4_xs;
using ggml_host::ksigns64;
using ggml_host::iq2xxs_grid;
using ggml_host::iq2xs_grid;
using ggml_host::iq2s_grid;
using ggml_host::iq3xxs_grid;
using ggml_host::iq3s_grid;
using ggml_host::iq1s_grid;
using ggml_host::kvalues_iq4nl;

// ============================================================================
// Fast Hardware FP16 / BF16 & AVX2 Integer Dot-Product Primitives
// ============================================================================

static __forceinline float cpu_fp16_to_fp32(uint16_t h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
}

static __forceinline uint16_t cpu_fp32_to_fp16(float f) {
    return (uint16_t)_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT), 0);
}

static __forceinline float cpu_bf16_to_fp32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

static __forceinline int hsum_epi32_128(__m128i x) {
    const __m128i s64 = _mm_add_epi32(x, _mm_shuffle_epi32(x, _MM_SHUFFLE(1, 0, 3, 2)));
    const __m128i s32 = _mm_add_epi32(s64, _mm_shuffle_epi32(s64, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s32);
}

static __forceinline int hsum_epi32_256(__m256i x) {
    return hsum_epi32_128(_mm_add_epi32(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1)));
}

// 32-element unsigned uint8 * signed int8 -> int32 dot product
static __forceinline int avx2_dot32_u8_s8(__m256i w_u8, __m256i q_s8) {
    const __m256i prod16 = _mm256_maddubs_epi16(w_u8, q_s8);
    const __m256i prod32 = _mm256_madd_epi16(prod16, _mm256_set1_epi16(1));
    return hsum_epi32_256(prod32);
}

// Dual 16-element unsigned uint8 * signed int8 -> (sumi0, sumi1) dot product
static __forceinline void avx2_dot16x2_u8_s8(__m256i w_u8, __m256i q_s8, int* sumi0, int* sumi1) {
    const __m256i prod16 = _mm256_maddubs_epi16(w_u8, q_s8);
    const __m256i prod32 = _mm256_madd_epi16(prod16, _mm256_set1_epi16(1));
    *sumi0 = hsum_epi32_128(_mm256_castsi256_si128(prod32));
    *sumi1 = hsum_epi32_128(_mm256_extracti128_si256(prod32, 1));
}

// 32-element signed int8 * signed int8 -> int32 dot product (for a, b in [-127, +127])
static __forceinline int avx2_dot32_s8_s8(__m256i a_s8, __m256i b_s8) {
    const __m256i abs_a    = _mm256_abs_epi8(a_s8);
    const __m256i signed_b = _mm256_sign_epi8(b_s8, a_s8);
    return avx2_dot32_u8_s8(abs_a, signed_b);
}

// Dual 16-element signed int8 * signed int8 -> (sumi0, sumi1) dot product
static __forceinline void avx2_dot16x2_s8_s8(__m256i a_s8, __m256i b_s8, int* sumi0, int* sumi1) {
    const __m256i abs_a    = _mm256_abs_epi8(a_s8);
    const __m256i signed_b = _mm256_sign_epi8(b_s8, a_s8);
    avx2_dot16x2_u8_s8(abs_a, signed_b, sumi0, sumi1);
}

// Sum of all 32 signed int8 elements in q_s8
static __forceinline int avx2_sum32_s8(__m256i q_s8) {
    return avx2_dot32_u8_s8(_mm256_set1_epi8(1), q_s8);
}

// Unpack 32 sign bits (4 bytes) into 32 signed bytes (+1 if bit==0, -1 if bit==1)
static __forceinline __m256i unpack_signs32_to_pm1(uint32_t s32) {
    const __m256i v = _mm256_set1_epi32((int)s32);
    const __m256i shuf = _mm256_set_epi64x(
        0x0303030303030303ULL,
        0x0202020202020202ULL,
        0x0101010101010101ULL,
        0x0000000000000000ULL);
    const __m256i rep  = _mm256_shuffle_epi8(v, shuf);
    const __m256i mask = _mm256_set1_epi64x((long long)0x8040201008040201ULL);
    const __m256i neg  = _mm256_cmpeq_epi8(_mm256_and_si256(rep, mask), mask);
    return _mm256_or_si256(neg, _mm256_set1_epi8(1));
}

// ============================================================================
// Per-Row AVX2 Quantized Dot-Product Kernels (Matching qwen4_kernels.cu)
// ============================================================================

// TY = 42: Q2_0 (64 weights per block_q2_0)
static __forceinline float row_dot_q2_0_avx2(const uint8_t* row, const block_q8_1* x, int n_cols) {
    const int nb = n_cols / 64;
    const block_q2_0* bq2 = (const block_q2_0*)row;
    float sumf = 0.0f;
    for (int kbx = 0; kbx < nb; ++kbx) {
        const float d2 = cpu_fp16_to_fp32(bq2[kbx].d);
        const block_q8_1* bq8 = x + kbx * 2;
        for (int iqs = 0; iqs < 2; ++iqs) {
            const __m256i v  = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(bq2[kbx].qs + iqs * 8)));
            const __m256i v0 = _mm256_or_si256(v, _mm256_slli_epi32(v, 12));
            const __m256i v1 = _mm256_or_si256(_mm256_slli_epi32(v, 6), _mm256_slli_epi32(v, 18));
            const __m256i u8 = _mm256_or_si256(
                _mm256_and_si256(v0, _mm256_set1_epi32(0x00030003)),
                _mm256_and_si256(v1, _mm256_set1_epi32(0x03000300)));
            const __m256i w_s8 = _mm256_sub_epi8(u8, _mm256_set1_epi8(1));
            const __m256i q_s8 = _mm256_loadu_si256((const __m256i*)bq8[iqs].qs);
            const int sumi = avx2_dot32_s8_s8(w_s8, q_s8);
            const float d8 = cpu_fp16_to_fp32(bq8[iqs].data.data.d);
            sumf += d2 * d8 * (float)sumi;
        }
    }
    return sumf;
}

// TY = 16: IQ2_XXS (256 weights per block_iq2_xxs)
static __forceinline float row_dot_iq2_xxs_avx2(const uint8_t* row, const block_q8_1* x, int n_cols) {
    const int nb = n_cols / 256;
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*)row;
    float sumf = 0.0f;
    for (int kbx = 0; kbx < nb; ++kbx) {
        const float d2 = cpu_fp16_to_fp32(bq2[kbx].d);
        const block_q8_1* bq8 = x + kbx * 8;
        for (int b = 0; b < 8; ++b) {
            const uint8_t* aux8 = (const uint8_t*)(bq2[kbx].qs + 4 * b);
            uint32_t aux32;
            std::memcpy(&aux32, bq2[kbx].qs + 4 * b + 2, sizeof(uint32_t));
            const __m256i grid_u8 = _mm256_set_epi64x(
                (long long)iq2xxs_grid[aux8[3]],
                (long long)iq2xxs_grid[aux8[2]],
                (long long)iq2xxs_grid[aux8[1]],
                (long long)iq2xxs_grid[aux8[0]]);
            const __m256i signs = _mm256_or_si256(
                _mm256_set_epi64x(
                    (long long)ksigns64[(aux32 >> 21) & 0x7F],
                    (long long)ksigns64[(aux32 >> 14) & 0x7F],
                    (long long)ksigns64[(aux32 >>  7) & 0x7F],
                    (long long)ksigns64[(aux32 >>  0) & 0x7F]),
                _mm256_set1_epi8(1));
            const __m256i q8 = _mm256_loadu_si256((const __m256i*)bq8[b].qs);
            int sumi = avx2_dot32_u8_s8(grid_u8, _mm256_sign_epi8(q8, signs));
            const int ls = (int)((aux32 >> 27) | 1u);
            sumi = sumi * ls / 8;
            sumf += (d2 * cpu_fp16_to_fp32(bq8[b].data.data.d)) * (float)sumi;
        }
    }
    return sumf;
}

// TY = 17: IQ2_XS (256 weights per block_iq2_xs)
static __forceinline float row_dot_iq2_xs_avx2(const uint8_t* row, const block_q8_1* x, int n_cols) {
    const int nb = n_cols / 256;
    const block_iq2_xs* bq2 = (const block_iq2_xs*)row;
    float sumf = 0.0f;
    for (int kbx = 0; kbx < nb; ++kbx) {
        const float d2 = cpu_fp16_to_fp32(bq2[kbx].d);
        const block_q8_1* bq8 = x + kbx * 8;
        for (int b = 0; b < 8; ++b) {
            const uint16_t* q2 = bq2[kbx].qs + 4 * b;
            const int ls0 = bq2[kbx].scales[b] & 0x0F;
            const int ls1 = bq2[kbx].scales[b] >> 4;
            const __m256i grid_u8 = _mm256_set_epi64x(
                (long long)iq2xs_grid[q2[3] & 0x1FF],
                (long long)iq2xs_grid[q2[2] & 0x1FF],
                (long long)iq2xs_grid[q2[1] & 0x1FF],
                (long long)iq2xs_grid[q2[0] & 0x1FF]);
            const __m256i signs = _mm256_or_si256(
                _mm256_set_epi64x(
                    (long long)ksigns64[q2[3] >> 9],
                    (long long)ksigns64[q2[2] >> 9],
                    (long long)ksigns64[q2[1] >> 9],
                    (long long)ksigns64[q2[0] >> 9]),
                _mm256_set1_epi8(1));
            const __m256i q8 = _mm256_loadu_si256((const __m256i*)bq8[b].qs);
            int sumi0 = 0, sumi1 = 0;
            avx2_dot16x2_u8_s8(grid_u8, _mm256_sign_epi8(q8, signs), &sumi0, &sumi1);
            const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
            sumf += (d2 * cpu_fp16_to_fp32(bq8[b].data.data.d)) * (float)sumi;
        }
    }
    return sumf;
}

// TY = 22: IQ2_S (256 weights per block_iq2_s)
static __forceinline float row_dot_iq2_s_avx2(const uint8_t* row, const block_q8_1* x, int n_cols) {
    const int nb = n_cols / 256;
    const block_iq2_s* bq2 = (const block_iq2_s*)row;
    float sumf = 0.0f;
    for (int kbx = 0; kbx < nb; ++kbx) {
        const float d2 = cpu_fp16_to_fp32(bq2[kbx].d);
        const block_q8_1* bq8 = x + kbx * 8;
        for (int b = 0; b < 8; ++b) {
            const uint8_t* qs = bq2[kbx].qs + 4 * b;
            const int qh = bq2[kbx].qh[b];
            uint32_t signs32;
            std::memcpy(&signs32, bq2[kbx].qs + 32 + 4 * b, sizeof(uint32_t));
            const int ls0 = bq2[kbx].scales[b] & 0x0F;
            const int ls1 = bq2[kbx].scales[b] >> 4;
            const __m256i grid_u8 = _mm256_set_epi64x(
                (long long)iq2s_grid[qs[3] | ((qh << 2) & 0x300)],
                (long long)iq2s_grid[qs[2] | ((qh << 4) & 0x300)],
                (long long)iq2s_grid[qs[1] | ((qh << 6) & 0x300)],
                (long long)iq2s_grid[qs[0] | ((qh << 8) & 0x300)]);
            const __m256i signs = unpack_signs32_to_pm1(signs32);
            const __m256i q8 = _mm256_loadu_si256((const __m256i*)bq8[b].qs);
            int sumi0 = 0, sumi1 = 0;
            avx2_dot16x2_u8_s8(grid_u8, _mm256_sign_epi8(q8, signs), &sumi0, &sumi1);
            const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
            sumf += (d2 * cpu_fp16_to_fp32(bq8[b].data.data.d)) * (float)sumi;
        }
    }
    return sumf;
}

// TY = 18: IQ3_XXS (256 weights per block_iq3_xxs)
static __forceinline float row_dot_iq3_xxs_avx2(const uint8_t* row, const block_q8_1* x, int n_cols) {
    const int nb = n_cols / 256;
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*)row;
    float sumf = 0.0f;
    for (int kbx = 0; kbx < nb; ++kbx) {
        const float d3 = cpu_fp16_to_fp32(bq3[kbx].d);
        const block_q8_1* bq8 = x + kbx * 8;
        for (int b = 0; b < 8; ++b) {
            const uint8_t* q3 = bq3[kbx].qs + 8 * b;
            uint32_t aux32;
            std::memcpy(&aux32, bq3[kbx].qs + 64 + 4 * b, sizeof(uint32_t));
            const __m256i grid_u8 = _mm256_set_epi32(
                (int)iq3xxs_grid[q3[7]], (int)iq3xxs_grid[q3[6]],
                (int)iq3xxs_grid[q3[5]], (int)iq3xxs_grid[q3[4]],
                (int)iq3xxs_grid[q3[3]], (int)iq3xxs_grid[q3[2]],
                (int)iq3xxs_grid[q3[1]], (int)iq3xxs_grid[q3[0]]);
            const __m256i signs = _mm256_or_si256(
                _mm256_set_epi64x(
                    (long long)ksigns64[(aux32 >> 21) & 0x7F],
                    (long long)ksigns64[(aux32 >> 14) & 0x7F],
                    (long long)ksigns64[(aux32 >>  7) & 0x7F],
                    (long long)ksigns64[(aux32 >>  0) & 0x7F]),
                _mm256_set1_epi8(1));
            const __m256i q8 = _mm256_loadu_si256((const __m256i*)bq8[b].qs);
            int sumi = avx2_dot32_u8_s8(grid_u8, _mm256_sign_epi8(q8, signs));
            const int ls = (int)(aux32 >> 28);
            sumi = (ls * sumi + sumi / 2) / 2;
            sumf += (d3 * cpu_fp16_to_fp32(bq8[b].data.data.d)) * (float)sumi;
        }
    }
    return sumf;
}

// TY = 21: IQ3_S (256 weights per block_iq3_s)
static __forceinline float row_dot_iq3_s_avx2(const uint8_t* row, const block_q8_1* x, int n_cols) {
    const int nb = n_cols / 256;
    const block_iq3_s* bq3 = (const block_iq3_s*)row;
    float sumf = 0.0f;
    for (int kbx = 0; kbx < nb; ++kbx) {
        const float d3 = cpu_fp16_to_fp32(bq3[kbx].d);
        const block_q8_1* bq8 = x + kbx * 8;
        for (int b = 0; b < 8; ++b) {
            const uint8_t* qs = bq3[kbx].qs + 8 * b;
            const int qh = bq3[kbx].qh[b];
            uint32_t signs32;
            std::memcpy(&signs32, bq3[kbx].signs + 4 * b, sizeof(uint32_t));
            const __m256i grid_u8 = _mm256_set_epi32(
                (int)iq3s_grid[qs[7] | ((qh << 1) & 0x100)],
                (int)iq3s_grid[qs[6] | ((qh << 2) & 0x100)],
                (int)iq3s_grid[qs[5] | ((qh << 3) & 0x100)],
                (int)iq3s_grid[qs[4] | ((qh << 4) & 0x100)],
                (int)iq3s_grid[qs[3] | ((qh << 5) & 0x100)],
                (int)iq3s_grid[qs[2] | ((qh << 6) & 0x100)],
                (int)iq3s_grid[qs[1] | ((qh << 7) & 0x100)],
                (int)iq3s_grid[qs[0] | ((qh << 8) & 0x100)]);
            const __m256i signs = unpack_signs32_to_pm1(signs32);
            const __m256i q8 = _mm256_loadu_si256((const __m256i*)bq8[b].qs);
            int sumi = avx2_dot32_u8_s8(grid_u8, _mm256_sign_epi8(q8, signs));
            sumi *= 1 + 2 * ((bq3[kbx].scales[b / 2] >> ((b & 1) * 4)) & 0x0F);
            sumf += (d3 * cpu_fp16_to_fp32(bq8[b].data.data.d)) * (float)sumi;
        }
    }
    return sumf;
}

// TY = 29: IQ1_M (256 weights per block_iq1_m)
static __forceinline float row_dot_iq1_m_avx2(const uint8_t* row, const block_q8_1* x, int n_cols) {
    const int nb = n_cols / 256;
    const block_iq1_m* bq1 = (const block_iq1_m*)row;
    float sumf = 0.0f;
    for (int kbx = 0; kbx < nb; ++kbx) {
        const uint16_t* sc = (const uint16_t*)bq1[kbx].scales;
        const uint16_t u16_scale = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000));
        const float d_super = cpu_fp16_to_fp32(u16_scale);
        const block_q8_1* bq8 = x + kbx * 8;
        for (int iqs = 0; iqs < 8; ++iqs) {
            const uint8_t* qs = bq1[kbx].qs + 4 * iqs;
            const uint8_t qh0 = bq1[kbx].qh[2 * iqs + 0];
            const uint8_t qh1 = bq1[kbx].qh[2 * iqs + 1];
            const int qhl0 = (qh0 >> 0) & 0x0F;
            const int qhl1 = (qh0 >> 4) & 0x0F;
            const int qhl2 = (qh1 >> 0) & 0x0F;
            const int qhl3 = (qh1 >> 4) & 0x0F;
            const __m256i g_s8 = _mm256_set_epi64x(
                (long long)iq1s_grid[qs[3] | ((qhl3 & 0x07) << 8)],
                (long long)iq1s_grid[qs[2] | ((qhl2 & 0x07) << 8)],
                (long long)iq1s_grid[qs[1] | ((qhl1 & 0x07) << 8)],
                (long long)iq1s_grid[qs[0] | ((qhl0 & 0x07) << 8)]);
            const __m256i delta_pm1 = _mm256_set_epi64x(
                (qhl3 & 0x08) ? -0x0101010101010101LL : 0x0101010101010101LL,
                (qhl2 & 0x08) ? -0x0101010101010101LL : 0x0101010101010101LL,
                (qhl1 & 0x08) ? -0x0101010101010101LL : 0x0101010101010101LL,
                (qhl0 & 0x08) ? -0x0101010101010101LL : 0x0101010101010101LL);
            const __m256i w8 = _mm256_add_epi8(_mm256_sign_epi8(_mm256_set1_epi8(8), g_s8), delta_pm1);
            const __m256i q8 = _mm256_loadu_si256((const __m256i*)bq8[iqs].qs);
            int dot0 = 0, dot1 = 0;
            avx2_dot16x2_s8_s8(w8, q8, &dot0, &dot1);
            const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
            const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
            const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
            const float d = d_super * cpu_fp16_to_fp32(bq8[iqs].data.data.d);
            sumf += d * (0.125f * (float)(dot0 * sc0 + dot1 * sc1));
        }
    }
    return sumf;
}

static __forceinline void cpu_k_scale_min(const uint8_t* scales8, int bq8_offset, uint16_t aux[2]) {
    const uint16_t* scales = (const uint16_t*)scales8;
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = (uint32_t)-(int32_t)(j >= 2);
    aux[0] = (uint16_t)(((s0 & 0x3f3fu) & ~hi) | ((((s4 >> 0) & 0x0f0fu) | ((s0 & 0xc0c0u) >> 2)) & hi));
    aux[1] = (uint16_t)(((s2 & 0x3f3fu) & ~hi) | ((((s4 >> 4) & 0x0f0fu) | ((s2 & 0xc0c0u) >> 2)) & hi));
}

// TY = 12: Q4_K (256 weights per block_q4_K)
static __forceinline float row_dot_q4_K_avx2(const uint8_t* row, const block_q8_1* x, int n_cols) {
    const int nb = n_cols / 256;
    const block_q4_K* bq4 = (const block_q4_K*)row;
    float sumf = 0.0f;
    for (int kbx = 0; kbx < nb; ++kbx) {
        const float d4 = cpu_fp16_to_fp32(bq4[kbx].data.data.d);
        const float m4 = cpu_fp16_to_fp32(bq4[kbx].data.data.dmin);
        const block_q8_1* bq8 = x + kbx * 8;
        float sumf_d = 0.0f;
        float sumf_m = 0.0f;
        for (int j = 0; j < 4; ++j) {
            const int bq8_offset = 2 * j;
            uint16_t aux[2];
            cpu_k_scale_min(bq4[kbx].scales, bq8_offset, aux);
            const uint8_t* sc = (const uint8_t*)aux;
            const uint8_t* m  = sc + 2;
            const __m256i qs32  = _mm256_loadu_si256((const __m256i*)(bq4[kbx].qs + 32 * j));
            const __m256i w0_u8 = _mm256_and_si256(qs32, _mm256_set1_epi8(0x0F));
            const __m256i w1_u8 = _mm256_and_si256(_mm256_srli_epi16(qs32, 4), _mm256_set1_epi8(0x0F));

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)bq8[bq8_offset + 0].qs);
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)bq8[bq8_offset + 1].qs);
            const float d8_0 = cpu_fp16_to_fp32(bq8[bq8_offset + 0].data.data.d);
            const float d8_1 = cpu_fp16_to_fp32(bq8[bq8_offset + 1].data.data.d);

            const int dot1_0 = avx2_dot32_u8_s8(w0_u8, q8_0);
            const int dot2_0 = avx2_sum32_s8(q8_0);
            sumf_d += d8_0 * (float)(dot1_0 * sc[0]);
            sumf_m += d8_0 * (float)(dot2_0 * m[0]);

            const int dot1_1 = avx2_dot32_u8_s8(w1_u8, q8_1);
            const int dot2_1 = avx2_sum32_s8(q8_1);
            sumf_d += d8_1 * (float)(dot1_1 * sc[1]);
            sumf_m += d8_1 * (float)(dot2_1 * m[1]);
        }
        sumf += d4 * sumf_d - m4 * sumf_m;
    }
    return sumf;
}

// TY = 13: Q5_K (256 weights per block_q5_K)
static __forceinline float row_dot_q5_K_avx2(const uint8_t* row, const block_q8_1* x, int n_cols) {
    const int nb = n_cols / 256;
    const block_q5_K* bq5 = (const block_q5_K*)row;
    float sumf = 0.0f;
    const __m256i qh_LoadedMask0 = _mm256_set1_epi8(0x01);
    (void)qh_LoadedMask0;
    for (int kbx = 0; kbx < nb; ++kbx) {
        const float d5 = cpu_fp16_to_fp32(bq5[kbx].data.data.d);
        const float m5 = cpu_fp16_to_fp32(bq5[kbx].data.data.dmin);
        const block_q8_1* bq8 = x + kbx * 8;
        const __m256i qh32 = _mm256_loadu_si256((const __m256i*)bq5[kbx].qh);
        float sumf_d = 0.0f;
        float sumf_m = 0.0f;
        for (int j = 0; j < 4; ++j) {
            const int bq8_offset = 2 * j;
            uint16_t aux[2];
            cpu_k_scale_min(bq5[kbx].scales, bq8_offset, aux);
            const uint8_t* sc = (const uint8_t*)aux;
            const uint8_t* m  = sc + 2;
            const __m256i qs32  = _mm256_loadu_si256((const __m256i*)(bq5[kbx].qs + 32 * j));
            const __m256i h0    = _mm256_and_si256(_mm256_slli_epi16(_mm256_srli_epi16(qh32, bq8_offset + 0), 4), _mm256_set1_epi8(0x10));
            const __m256i h1    = _mm256_and_si256(_mm256_slli_epi16(_mm256_srli_epi16(qh32, bq8_offset + 1), 4), _mm256_set1_epi8(0x10));
            const __m256i w0_u8 = _mm256_or_si256(_mm256_and_si256(qs32, _mm256_set1_epi8(0x0F)), h0);
            const __m256i w1_u8 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qs32, 4), _mm256_set1_epi8(0x0F)), h1);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)bq8[bq8_offset + 0].qs);
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)bq8[bq8_offset + 1].qs);
            const float d8_0 = cpu_fp16_to_fp32(bq8[bq8_offset + 0].data.data.d);
            const float d8_1 = cpu_fp16_to_fp32(bq8[bq8_offset + 1].data.data.d);

            sumf_d += d8_0 * (float)(avx2_dot32_u8_s8(w0_u8, q8_0) * sc[0]);
            sumf_m += d8_0 * (float)(avx2_sum32_s8(q8_0) * m[0]);
            sumf_d += d8_1 * (float)(avx2_dot32_u8_s8(w1_u8, q8_1) * sc[1]);
            sumf_m += d8_1 * (float)(avx2_sum32_s8(q8_1) * m[1]);
        }
        sumf += d5 * sumf_d - m5 * sumf_m;
    }
    return sumf;
}

// TY = 14: Q6_K (256 weights per block_q6_K)
static __forceinline float row_dot_q6_K_avx2(const uint8_t* row, const block_q8_1* x, int n_cols) {
    const int nb = n_cols / 256;
    const block_q6_K* bq6 = (const block_q6_K*)row;
    float sumf = 0.0f;
    for (int kbx = 0; kbx < nb; ++kbx) {
        const float d6 = cpu_fp16_to_fp32(bq6[kbx].d);
        const block_q8_1* bq8 = x + kbx * 8;
        float block_sum = 0.0f;
        for (int half_idx = 0; half_idx < 2; ++half_idx) {
            const uint8_t* ql = bq6[kbx].ql + 64 * half_idx;
            const uint8_t* qh = bq6[kbx].qh + 32 * half_idx;
            const int8_t*  sc = bq6[kbx].scales + 8 * half_idx;
            const block_q8_1* q8_half = bq8 + 4 * half_idx;
            const __m256i ql0 = _mm256_loadu_si256((const __m256i*)(ql + 0));
            const __m256i ql1 = _mm256_loadu_si256((const __m256i*)(ql + 32));
            const __m256i qh0 = _mm256_loadu_si256((const __m256i*)qh);

            const __m256i w0 = _mm256_sub_epi8(
                _mm256_or_si256(_mm256_and_si256(ql0, _mm256_set1_epi8(0x0F)),
                                _mm256_slli_epi16(_mm256_and_si256(qh0, _mm256_set1_epi8(0x03)), 4)),
                _mm256_set1_epi8(32));
            const __m256i w1 = _mm256_sub_epi8(
                _mm256_or_si256(_mm256_and_si256(ql1, _mm256_set1_epi8(0x0F)),
                                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh0, 2), _mm256_set1_epi8(0x03)), 4)),
                _mm256_set1_epi8(32));
            const __m256i w2 = _mm256_sub_epi8(
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql0, 4), _mm256_set1_epi8(0x0F)),
                                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh0, 4), _mm256_set1_epi8(0x03)), 4)),
                _mm256_set1_epi8(32));
            const __m256i w3 = _mm256_sub_epi8(
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql1, 4), _mm256_set1_epi8(0x0F)),
                                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh0, 6), _mm256_set1_epi8(0x03)), 4)),
                _mm256_set1_epi8(32));

            const __m256i ws[4] = {w0, w1, w2, w3};
            for (int b = 0; b < 4; ++b) {
                int s0 = 0, s1 = 0;
                const __m256i q8 = _mm256_loadu_si256((const __m256i*)q8_half[b].qs);
                avx2_dot16x2_s8_s8(ws[b], q8, &s0, &s1);
                const float d8 = cpu_fp16_to_fp32(q8_half[b].data.data.d);
                block_sum += d8 * (float)(s0 * sc[2 * b + 0] + s1 * sc[2 * b + 1]);
            }
        }
        sumf += d6 * block_sum;
    }
    return sumf;
}

// TY = 20: IQ4_NL (32 weights per block_iq4_nl)
static __forceinline float row_dot_iq4_nl_avx2(const uint8_t* row, const block_q8_1* x, int n_cols) {
    const int nb = n_cols / 32;
    const block_iq4_nl* bq4 = (const block_iq4_nl*)row;
    const __m128i lut128 = _mm_loadu_si128((const __m128i*)kvalues_iq4nl);
    float sumf = 0.0f;
    for (int kbx = 0; kbx < nb; ++kbx) {
        const __m128i qs16 = _mm_loadu_si128((const __m128i*)bq4[kbx].qs);
        const __m128i lo   = _mm_shuffle_epi8(lut128, _mm_and_si128(qs16, _mm_set1_epi8(0x0F)));
        const __m128i hi   = _mm_shuffle_epi8(lut128, _mm_and_si128(_mm_srli_epi16(qs16, 4), _mm_set1_epi8(0x0F)));
        const __m256i w_s8 = _mm256_set_m128i(hi, lo);
        const __m256i q8   = _mm256_loadu_si256((const __m256i*)x[kbx].qs);
        const int sumi = avx2_dot32_s8_s8(w_s8, q8);
        sumf += (cpu_fp16_to_fp32(bq4[kbx].d) * cpu_fp16_to_fp32(x[kbx].data.data.d)) * (float)sumi;
    }
    return sumf;
}

// TY = 23: IQ4_XS (256 weights per block_iq4_xs)
static __forceinline float row_dot_iq4_xs_avx2(const uint8_t* row, const block_q8_1* x, int n_cols) {
    const int nb = n_cols / 256;
    const block_iq4_xs* bq4 = (const block_iq4_xs*)row;
    const __m128i lut128 = _mm_loadu_si128((const __m128i*)kvalues_iq4nl);
    float sumf = 0.0f;
    for (int kbx = 0; kbx < nb; ++kbx) {
        const float d4 = cpu_fp16_to_fp32(bq4[kbx].d);
        const block_q8_1* bq8 = x + kbx * 8;
        for (int b = 0; b < 8; ++b) {
            const int iqs = 4 * b;
            const __m128i qs16 = _mm_loadu_si128((const __m128i*)(bq4[kbx].qs + 16 * b));
            const __m128i lo   = _mm_shuffle_epi8(lut128, _mm_and_si128(qs16, _mm_set1_epi8(0x0F)));
            const __m128i hi   = _mm_shuffle_epi8(lut128, _mm_and_si128(_mm_srli_epi16(qs16, 4), _mm_set1_epi8(0x0F)));
            const __m256i w_s8 = _mm256_set_m128i(hi, lo);
            const __m256i q8   = _mm256_loadu_si256((const __m256i*)bq8[b].qs);
            int sumi = avx2_dot32_s8_s8(w_s8, q8);
            const int ls = ((bq4[kbx].scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4[kbx].scales_h >> (iqs / 2)) & 0x03) << 4);
            sumi *= ls - 32;
            sumf += (d4 * cpu_fp16_to_fp32(bq8[b].data.data.d)) * (float)sumi;
        }
    }
    return sumf;
}

// TY = 2: Q4_0 (32 weights per block_q4_0)
static __forceinline float row_dot_q4_0_avx2(const uint8_t* row, const block_q8_1* x, int n_cols) {
    const int nb = n_cols / 32;
    const block_q4_0* bq4 = (const block_q4_0*)row;
    float sumf = 0.0f;
    for (int kbx = 0; kbx < nb; ++kbx) {
        const __m128i qs16 = _mm_loadu_si128((const __m128i*)bq4[kbx].qs);
        const __m128i lo   = _mm_sub_epi8(_mm_and_si128(qs16, _mm_set1_epi8(0x0F)), _mm_set1_epi8(8));
        const __m128i hi   = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(qs16, 4), _mm_set1_epi8(0x0F)), _mm_set1_epi8(8));
        const __m256i w_s8 = _mm256_set_m128i(hi, lo);
        const __m256i q8   = _mm256_loadu_si256((const __m256i*)x[kbx].qs);
        const int sumi = avx2_dot32_s8_s8(w_s8, q8);
        sumf += (cpu_fp16_to_fp32(bq4[kbx].d) * cpu_fp16_to_fp32(x[kbx].data.data.d)) * (float)sumi;
    }
    return sumf;
}

// TY = 8: Q8_0 (32 weights per block_q8_0)
static __forceinline float row_dot_q8_0_avx2(const uint8_t* row, const block_q8_1* x, int n_cols) {
    const int nb = n_cols / 32;
    const block_q8_0* bq8 = (const block_q8_0*)row;
    float sumf = 0.0f;
    for (int kbx = 0; kbx < nb; ++kbx) {
        const __m256i w_s8 = _mm256_loadu_si256((const __m256i*)bq8[kbx].qs);
        const __m256i q_s8 = _mm256_loadu_si256((const __m256i*)x[kbx].qs);
        const int sumi = avx2_dot32_s8_s8(w_s8, q_s8);
        sumf += (cpu_fp16_to_fp32(bq8[kbx].d) * cpu_fp16_to_fp32(x[kbx].data.data.d)) * (float)sumi;
    }
    return sumf;
}

// TY = 1: F16 (IEEE 754 FP16 weights)
static __forceinline float row_dot_f16_avx2(const uint8_t* row, const block_q8_1* x, int n_cols) {
    const int nb = n_cols / 32;
    const uint16_t* w16 = (const uint16_t*)row;
    float sumf = 0.0f;
    for (int kbx = 0; kbx < nb; ++kbx) {
        const float d8 = cpu_fp16_to_fp32(x[kbx].data.data.d);
        __m256 acc = _mm256_setzero_ps();
        for (int g = 0; g < 4; ++g) {
            const __m128i h8 = _mm_loadu_si128((const __m128i*)(w16 + kbx * 32 + g * 8));
            const __m256  wf = _mm256_cvtph_ps(h8);
            const __m128i q8 = _mm_loadl_epi64((const __m128i*)(x[kbx].qs + g * 8));
            const __m256  qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
            acc = _mm256_fmadd_ps(wf, qf, acc);
        }
        const __m128 lo = _mm256_castps256_ps128(acc);
        const __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s4 = _mm_add_ps(lo, hi);
        s4 = _mm_hadd_ps(s4, s4);
        s4 = _mm_hadd_ps(s4, s4);
        sumf += d8 * _mm_cvtss_f32(s4);
    }
    return sumf;
}

// TY = 30: BF16 (BFloat16 weights)
static __forceinline float row_dot_bf16_avx2(const uint8_t* row, const block_q8_1* x, int n_cols) {
    const int nb = n_cols / 32;
    const uint16_t* w16 = (const uint16_t*)row;
    float sumf = 0.0f;
    for (int kbx = 0; kbx < nb; ++kbx) {
        const float d8 = cpu_fp16_to_fp32(x[kbx].data.data.d);
        __m256 acc = _mm256_setzero_ps();
        for (int g = 0; g < 4; ++g) {
            const __m128i bf8 = _mm_loadu_si128((const __m128i*)(w16 + kbx * 32 + g * 8));
            const __m256i u32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf8), 16);
            const __m256  wf  = _mm256_castsi256_ps(u32);
            const __m128i q8  = _mm_loadl_epi64((const __m128i*)(x[kbx].qs + g * 8));
            const __m256  qf  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
            acc = _mm256_fmadd_ps(wf, qf, acc);
        }
        const __m128 lo = _mm256_castps256_ps128(acc);
        const __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s4 = _mm_add_ps(lo, hi);
        s4 = _mm_hadd_ps(s4, s4);
        s4 = _mm_hadd_ps(s4, s4);
        sumf += d8 * _mm_cvtss_f32(s4);
    }
    return sumf;
}

// Dispatch any GGML/GSQ type row dot product against block_q8_1 input vector
static __forceinline float cpu_mmvq_row_dot(int32_t ggml_type, const uint8_t* row, const block_q8_1* x, int n_cols) {
    switch (ggml_type) {
        case 42: return row_dot_q2_0_avx2(row, x, n_cols);
        case 29: return row_dot_iq1_m_avx2(row, x, n_cols);
        case 16: return row_dot_iq2_xxs_avx2(row, x, n_cols);
        case 17: return row_dot_iq2_xs_avx2(row, x, n_cols);
        case 22: return row_dot_iq2_s_avx2(row, x, n_cols);
        case 18: return row_dot_iq3_xxs_avx2(row, x, n_cols);
        case 21: return row_dot_iq3_s_avx2(row, x, n_cols);
        case 20: return row_dot_iq4_nl_avx2(row, x, n_cols);
        case 23: return row_dot_iq4_xs_avx2(row, x, n_cols);
        case 12: return row_dot_q4_K_avx2(row, x, n_cols);
        case 13: return row_dot_q5_K_avx2(row, x, n_cols);
        case 14: return row_dot_q6_K_avx2(row, x, n_cols);
        case 2:  return row_dot_q4_0_avx2(row, x, n_cols);
        case 8:  return row_dot_q8_0_avx2(row, x, n_cols);
        case 1:  return row_dot_f16_avx2(row, x, n_cols);
        case 30: return row_dot_bf16_avx2(row, x, n_cols);
        default: return 0.0f;
    }
}

// Fused SwiGLU + Q8_1 quantization for one 32-element block (matching swiglu_and_q8_1_kernel in qwen4_kernels.cu)
static __forceinline void cpu_swiglu_and_q8_1_block32(
    const float* __restrict__ gate32,
    const float* __restrict__ up32,
    block_q8_1* __restrict__ out_blk)
{
    float xi[32];
    float amax = 0.0f;
    float sum  = 0.0f;
    for (int i = 0; i < 32; ++i) {
        const float g = gate32[i];
        const float u = up32[i];
        float h = (g / (1.0f + std::exp(-g))) * u;
        if (!std::isfinite(h)) h = 0.0f;
        else if (h >  65504.0f) h =  65504.0f;
        else if (h < -65504.0f) h = -65504.0f;
        xi[i] = h;
        const float a = std::fabs(h);
        if (a > amax) amax = a;
        sum += h;
    }
    const float d = amax / 127.0f;
    const float inv_d = (amax == 0.0f) ? 0.0f : (127.0f / amax);
    for (int i = 0; i < 32; ++i) {
        out_blk->qs[i] = (amax == 0.0f) ? 0 : (int8_t)std::roundf(xi[i] * inv_d);
    }
    out_blk->data.data.d = cpu_fp32_to_fp16(d);
    out_blk->data.data.s = cpu_fp32_to_fp16(sum);
}

// ============================================================================
// Persistent Low-Latency Hybrid CPU MoE Expert Thread Pool
// ============================================================================
// - Uses an active-window spin barrier (~0.1 us wakeup during generation) that
//   automatically parks on a condition_variable after ~15 ms of idle time (0% CPU
//   usage when idle or when --no-cpu-experts is set).
// - Supports both:
//   1) Asynchronous 2-stage MoE expert evaluation (launch_experts_async + assist_and_wait)
//   2) Parallel unpinned memory gather (parallel_memcpy) when CPU compute is disabled.

struct CpuExpertTaskDesc {
    const uint8_t* gate_ptr = nullptr;
    const uint8_t* up_ptr   = nullptr;
    const uint8_t* down_ptr = nullptr;
    float          weight   = 0.0f;
};

struct CpuMemcpyTaskDesc {
    const uint8_t* src    = nullptr;
    uint8_t*       dst    = nullptr;
    size_t         nbytes = 0;
};

class CpuMoEExpertPool {
public:
    static constexpr int MAX_CPU_EXPERTS = 10;
    static constexpr int MAX_INTER_DIM   = 16384;

    static CpuMoEExpertPool& instance() {
        static CpuMoEExpertPool pool;
        return pool;
    }

    int num_threads() const { return n_threads_; }

    // Synchronous parallel evaluation of `n_experts` routed experts across caller + worker threads.
    // Uses lock-free work-unit completion counters (`s1_done`, `s2_done`) so an OS-preempted worker
    // thread can never stall Stage 1 -> Stage 2 transition or completion.
    void compute_experts_sync(
        int n_experts,
        const CpuExpertTaskDesc* tasks,
        int32_t gate_type, size_t gate_row_bytes,
        int32_t up_type,   size_t up_row_bytes,
        int32_t down_type, size_t down_row_bytes,
        const void* h_x_q8_1,
        float* h_out_f32,
        int hidden_dim,
        int inter_dim)
    {
        if (n_experts <= 0) return;
        const uint64_t next_ep = epoch_.load(std::memory_order_relaxed) + 1;
        JobSlot& slot = slots_[next_ep & (RING_SLOTS - 1)];

        slot.mode           = Mode::COMPUTE_EXPERTS;
        slot.n_experts      = n_experts;
        for (int e = 0; e < n_experts; ++e) {
            slot.tasks[e] = tasks[e];
        }
        slot.gate_type      = gate_type;
        slot.gate_row_bytes = gate_row_bytes;
        slot.up_type        = up_type;
        slot.up_row_bytes   = up_row_bytes;
        slot.down_type      = down_type;
        slot.down_row_bytes = down_row_bytes;
        slot.x_q8_1         = (const block_q8_1*)h_x_q8_1;
        slot.out_f32        = h_out_f32;
        slot.hidden_dim     = hidden_dim;
        slot.inter_dim      = inter_dim;
        slot.inter_blocks   = inter_dim / 32;
        slot.s1_total       = n_experts * slot.inter_blocks;
        slot.s2_total       = (hidden_dim + 31) / 32; // 32-row chunks (80 chunks for 2560)

        slot.s1_next.store(0, std::memory_order_relaxed);
        slot.s1_done.store(0, std::memory_order_relaxed);
        slot.s2_next.store(0, std::memory_order_relaxed);
        slot.s2_done.store(0, std::memory_order_relaxed);

        dispatch_epoch(next_ep);
        run_expert_stages(slot);
        while (slot.s2_done.load(std::memory_order_acquire) < slot.s2_total) {
            _mm_pause();
        }
    }

    // Parallel memcpy gather striped in 8 slices (~256 KB each) per expert blob across worker threads.
    void parallel_memcpy(const CpuMemcpyTaskDesc* tasks, int count) {
        if (count <= 0) return;
        constexpr int SLICES_PER_TASK = 8;
        const uint64_t next_ep = epoch_.load(std::memory_order_relaxed) + 1;
        JobSlot& slot = slots_[next_ep & (RING_SLOTS - 1)];

        slot.mode          = Mode::MEMCPY_GATHER;
        slot.memcpy_tasks  = tasks;
        slot.memcpy_count  = count;
        slot.memcpy_chunks = count * SLICES_PER_TASK;
        slot.s1_next.store(0, std::memory_order_relaxed);
        slot.s2_done.store(0, std::memory_order_relaxed);

        dispatch_epoch(next_ep);
        run_memcpy_stage(slot);
        while (slot.s2_done.load(std::memory_order_acquire) < slot.memcpy_chunks) {
            _mm_pause();
        }
    }

private:
    static constexpr int RING_SLOTS = 8;

    enum class Mode : int {
        COMPUTE_EXPERTS = 0,
        MEMCPY_GATHER   = 1
    };

    struct alignas(64) JobSlot {
        std::atomic<int>         s1_next{0};
        std::atomic<int>         s1_done{0};
        std::atomic<int>         s2_next{0};
        std::atomic<int>         s2_done{0};
        Mode                     mode           = Mode::COMPUTE_EXPERTS;
        int                      n_experts      = 0;
        CpuExpertTaskDesc        tasks[MAX_CPU_EXPERTS];
        int32_t                  gate_type      = 0;
        size_t                   gate_row_bytes = 0;
        int32_t                  up_type        = 0;
        size_t                   up_row_bytes   = 0;
        int32_t                  down_type      = 0;
        size_t                   down_row_bytes = 0;
        const block_q8_1*        x_q8_1         = nullptr;
        float*                   out_f32        = nullptr;
        int                      hidden_dim     = 0;
        int                      inter_dim      = 0;
        int                      inter_blocks   = 0;
        int                      s1_total       = 0;
        int                      s2_total       = 0;
        const CpuMemcpyTaskDesc* memcpy_tasks   = nullptr;
        int                      memcpy_count   = 0;
        int                      memcpy_chunks  = 0;
    };

    CpuMoEExpertPool() {
        const int hw = (int)std::thread::hardware_concurrency();
        // Default to 10 threads (1 caller + 9 workers): evenly divides 20 Stage-1 blocks and 80 Stage-2 chunks
        // while leaving cores free for the CUDA driver and OS scheduler.
        n_threads_ = std::clamp(hw >= 16 ? 10 : (hw / 2), 2, 10);
        const int n_workers = n_threads_ - 1;
        workers_.resize(n_workers);
        for (int t = 0; t < n_workers; ++t) {
            workers_[t] = std::thread([this]() { worker_loop(); });
        }
    }

    ~CpuMoEExpertPool() {
        stop_.store(true, std::memory_order_release);
        epoch_.fetch_add(1, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(cv_mtx_);
            cv_.notify_all();
        }
        for (auto& th : workers_) {
            if (th.joinable()) th.join();
        }
    }

    void dispatch_epoch(uint64_t next_ep) {
        epoch_.store(next_ep, std::memory_order_release);
        if (sleeping_workers_.load(std::memory_order_acquire) > 0) {
            std::lock_guard<std::mutex> lk(cv_mtx_);
            cv_.notify_all();
        }
    }

    void run_expert_stages(JobSlot& slot) {
        const int n_exp          = slot.n_experts;
        const int n_inter_blocks = slot.inter_blocks;
        const int s1_total       = slot.s1_total;
        const int32_t g_type     = slot.gate_type;
        const size_t  g_rb       = slot.gate_row_bytes;
        const int32_t u_type     = slot.up_type;
        const size_t  u_rb       = slot.up_row_bytes;
        const block_q8_1* x_q8   = slot.x_q8_1;
        const int hidden_dim     = slot.hidden_dim;

        // Stage 1: Fused Gate + Up + SwiGLU + Q8_1 Quantization (32 rows per work unit)
        while (true) {
            const int u = slot.s1_next.fetch_add(1, std::memory_order_relaxed);
            if (u >= s1_total) break;
            const int e  = u / n_inter_blocks;
            const int ib = u % n_inter_blocks;
            const int r0 = ib * 32;
            const uint8_t* g_base = slot.tasks[e].gate_ptr + (size_t)r0 * g_rb;
            const uint8_t* u_base = slot.tasks[e].up_ptr   + (size_t)r0 * u_rb;

            float gate32[32];
            float up32[32];
            for (int i = 0; i < 32; ++i) {
                gate32[i] = cpu_mmvq_row_dot(g_type, g_base + (size_t)i * g_rb, x_q8, hidden_dim);
                up32[i]   = cpu_mmvq_row_dot(u_type, u_base + (size_t)i * u_rb, x_q8, hidden_dim);
            }
            cpu_swiglu_and_q8_1_block32(gate32, up32, &inter_q8_1_[e * (MAX_INTER_DIM / 32) + ib]);
            slot.s1_done.fetch_add(1, std::memory_order_release);
        }

        // Wait for all `s1_total` Stage-1 work units to finish (never waits for preempted idle threads!)
        while (slot.s1_done.load(std::memory_order_acquire) < s1_total) {
            _mm_pause();
        }

        // Stage 2: Down Projection + Weighted Accumulation across `n_exp` experts (32 rows per work unit)
        const int s2_total    = slot.s2_total;
        const int32_t d_type  = slot.down_type;
        const size_t  d_rb    = slot.down_row_bytes;
        const int inter_dim   = slot.inter_dim;
        float* out_f32        = slot.out_f32;

        while (true) {
            const int c = slot.s2_next.fetch_add(1, std::memory_order_relaxed);
            if (c >= s2_total) break;
            const int r_start = c * 32;
            const int r_end   = std::min(r_start + 32, hidden_dim);
            for (int r = r_start; r < r_end; ++r) {
                float acc = 0.0f;
                for (int e = 0; e < n_exp; ++e) {
                    const uint8_t* d_row = slot.tasks[e].down_ptr + (size_t)r * d_rb;
                    const block_q8_1* iq8 = &inter_q8_1_[e * (MAX_INTER_DIM / 32)];
                    acc += slot.tasks[e].weight * cpu_mmvq_row_dot(d_type, d_row, iq8, inter_dim);
                }
                out_f32[r] = acc;
            }
            slot.s2_done.fetch_add(1, std::memory_order_release);
        }
    }

    void run_memcpy_stage(JobSlot& slot) {
        constexpr int SLICES_PER_TASK = 8;
        const int total_units = slot.memcpy_chunks;
        while (true) {
            const int u = slot.s1_next.fetch_add(1, std::memory_order_relaxed);
            if (u >= total_units) break;
            const int tidx  = u / SLICES_PER_TASK;
            const int slice = u % SLICES_PER_TASK;
            const auto& task = slot.memcpy_tasks[tidx];
            const size_t chunk_sz = ((task.nbytes + SLICES_PER_TASK - 1) / SLICES_PER_TASK + 63) & ~size_t(63);
            const size_t offset   = (size_t)slice * chunk_sz;
            if (offset < task.nbytes) {
                const size_t len = std::min(chunk_sz, task.nbytes - offset);
                std::memcpy(task.dst + offset, task.src + offset, len);
            }
            slot.s2_done.fetch_add(1, std::memory_order_release);
        }
    }

    void worker_loop() {
        uint64_t last_ep = 0;
        uint32_t spin = 0;
        while (!stop_.load(std::memory_order_relaxed)) {
            uint64_t ep = epoch_.load(std::memory_order_acquire);
            if (ep == last_ep) {
                if (++spin < 1000000u) {
                    _mm_pause();
                } else {
                    // Idle for > ~10 ms: park on condition variable (0% idle CPU usage)
                    sleeping_workers_.fetch_add(1, std::memory_order_release);
                    {
                        std::unique_lock<std::mutex> lk(cv_mtx_);
                        cv_.wait(lk, [&]() {
                            return stop_.load(std::memory_order_relaxed) ||
                                   epoch_.load(std::memory_order_acquire) != last_ep;
                        });
                    }
                    sleeping_workers_.fetch_sub(1, std::memory_order_release);
                    spin = 0;
                }
                continue;
            }
            spin = 0;
            last_ep = ep;
            if (stop_.load(std::memory_order_relaxed)) break;

            JobSlot& slot = slots_[ep & (RING_SLOTS - 1)];
            if (slot.mode == Mode::COMPUTE_EXPERTS) {
                run_expert_stages(slot);
            } else {
                run_memcpy_stage(slot);
            }
        }
    }

    int n_threads_ = 1;
    std::vector<std::thread> workers_;
    std::atomic<bool> stop_{false};
    alignas(64) std::atomic<uint64_t> epoch_{0};
    alignas(64) std::atomic<int>      sleeping_workers_{0};
    std::mutex                        cv_mtx_;
    std::condition_variable           cv_;

    JobSlot                  slots_[RING_SLOTS];
    alignas(64) block_q8_1   inter_q8_1_[MAX_CPU_EXPERTS * (MAX_INTER_DIM / 32)];
};

} // namespace cpu_moe
} // namespace moecher
