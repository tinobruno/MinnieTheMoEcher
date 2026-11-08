#include "qwen4_kernels.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

namespace {

// ============================================================================
// Device Numeric Helpers (BF16 / FP16 / Warp Reductions / RoPE)
// ============================================================================

__device__ __forceinline__ float qwen4_bf16_to_fp32(uint16_t b) {
    return __uint_as_float((uint32_t)b << 16);
}

__device__ __forceinline__ uint16_t qwen4_fp32_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return 0x7FC0u;
    return (uint16_t)((u + (0x7FFFu + ((u >> 16) & 1u))) >> 16);
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, mask, 32);
    }
    return v;
}

template <int NWARPS>
__device__ __forceinline__ float block_sum(float v, float* smem) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    v = warp_sum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < NWARPS) ? smem[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ void rope_neox_elem(
    float* v,
    int i,
    int n_rot,
    int pos,
    float freq_base)
{
    const int half = n_rot / 2;
    if (i >= half) return;
    const float freq = (float)pos * powf(freq_base, -2.0f * (float)i / (float)n_rot);
    float s, c;
    sincosf(freq, &s, &c);
    const float v0 = v[i];
    const float v1 = v[i + half];
    v[i]        = v0 * c - v1 * s;
    v[i + half] = v0 * s + v1 * c;
}

// ============================================================================
// Activation Quantization: FP32 -> block_q8_1 (with finite clamp)
// ============================================================================

__device__ __forceinline__ float q8_1_finite(float xi) {
    if (!isfinite(xi)) return 0.0f;
    if (xi >  65504.0f) return  65504.0f;
    if (xi < -65504.0f) return -65504.0f;
    return xi;
}

__global__ void quantize_q8_1_kernel(
    const float* __restrict__ x,
    block_q8_1* __restrict__ y,
    int n_blocks)
{
    const int ib = blockIdx.x * blockDim.y + threadIdx.y;
    if (ib >= n_blocks) return;

    const int i = threadIdx.x; // 0..31
    const float xi = q8_1_finite(x[ib * QK8_1 + i]);

    float amax = fabsf(xi);
    float sum = xi;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask, 32));
        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, mask, 32);
    }

    const float d = amax / 127.0f;
    const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / d);
    y[ib].qs[i] = q;
    if (i == 0) {
        y[ib].ds = make_half2(__float2half(d), __float2half(sum));
    }
}

// ============================================================================
// Quantized Matrix-Vector Multiplication (MMVQ) - All Q / K / IQ / Q2_0 Types
// ============================================================================

__device__ __forceinline__ int ggml_cuda_dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const int8_t* pa = (const int8_t*)&a;
    const int8_t* pb = (const int8_t*)&b;
    return c + pa[0]*pb[0] + pa[1]*pb[1] + pa[2]*pb[2] + pa[3]*pb[3];
#endif
}

__device__ __forceinline__ int get_int_b2(const void* x, const int& i32) {
    const uint16_t* x16 = (const uint16_t*)x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}

__device__ __forceinline__ int get_int_b4(const void* x, const int& i32) {
    return ((const int*)x)[i32];
}

__device__ __forceinline__ uint32_t unpack_ksigns(const uint8_t v) {
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ (p << 7);
    return s * 0x01010101u;
}

__device__ __forceinline__ int2 get_int_from_table_16(const int& q4, const int8_t* table) {
    const uint32_t* table32 = (const uint32_t*)table;
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210u | (((uint32_t)q4 & 0x88888888u) >> 1));
    #pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(table32[0], table32[1], (uint32_t)q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], (uint32_t)q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

__device__ __forceinline__ float vec_dot_q2_0_q8_1(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_1,
    const int& kbx,
    const int& iqs)
{
    const block_q2_0* bq2_0 = (const block_q2_0*)vbq + kbx;
    const float d2 = __half2float(bq2_0->d);
    const int16_t* qs = (const int16_t*)bq2_0->qs + iqs * 4;
    const block_q8_1* bq8_1_chunk = bq8_1 + iqs;
    int sumi = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = get_int_b4(bq8_1_chunk->qs, j * 2 + 0);
        const int v = get_int_b4(bq8_1_chunk->qs, j * 2 + 1);
        const int qe = __byte_perm(0x020100FFu, 0x020100FFu, q >> 0);
        const int qo = __byte_perm(0x020100FFu, 0x020100FFu, q >> 2);
        const int qx = __byte_perm(qe, qo, 0x5140);
        const int qy = __byte_perm(qe, qo, 0x7362);
        sumi = ggml_cuda_dp4a(u, qx, sumi);
        sumi = ggml_cuda_dp4a(v, qy, sumi);
    }
    const float d8 = __low2float(bq8_1_chunk->ds);
    return d2 * d8 * (float)sumi;
}

__device__ __forceinline__ float vec_dot_iq2_xxs_q8_1(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_1,
    const int& kbx,
    const int& iqs)
{
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*)vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*)&q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
    #pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*)iq2xxs_grid)[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201u, 0);
        const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 0);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        const int signs1 = __vcmpne4(signs & 0x80402010u, 0);
        const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }
    const int ls = (aux32 >> 27) | 1;
    sumi = sumi * ls / 8;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * (float)sumi;
}

__device__ __forceinline__ float vec_dot_iq2_xs_q8_1(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_1,
    const int& kbx,
    const int& iqs)
{
    const block_iq2_xs* bq2 = (const block_iq2_xs*)vbq + kbx;
    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t* q2 = (const uint16_t*)&q2_packed;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*)iq2xs_grid)[q2[l0 / 2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
        const int signs0 = __vcmpne4(signs & 0x08040201u, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010u, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * (float)sumi;
}

__device__ __forceinline__ float vec_dot_iq2_s_q8_1(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_1,
    const int& kbx,
    const int& iqs)
{
    const block_iq2_s* bq2 = (const block_iq2_s*)vbq + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const uint8_t* qs = (const uint8_t*)&qs_packed;
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*)&signs_packed_32;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int* grid_pos = (const int*)(iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
        const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * (float)sumi;
}

__device__ __forceinline__ float vec_dot_iq3_xxs_q8_1(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_1,
    const int& kbx,
    const int& iqs)
{
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*)vbq + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*)&q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201u, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010u, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * (float)sumi;
}

__device__ __forceinline__ float vec_dot_iq3_s_q8_1(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_1,
    const int& kbx,
    const int& iqs)
{
    const block_iq3_s* bq3 = (const block_iq3_s*)vbq + kbx;
    const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* qs = (const uint8_t*)&qs_packed;
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*)&signs_packed_32;
    int sumi = 0;
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
            iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * (float)sumi;
}

__device__ __forceinline__ float vec_dot_iq1_m_q8_1(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_1,
    const int& kbx,
    const int& iqs)
{
    const block_iq1_m* bq1 = (const block_iq1_m*)vbq + kbx;
    const int qs_packed = get_int_b4(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*)&qs_packed;
    int sumi[2] = {0, 0};
    float sumf[2] = {0.0f, 0.0f};
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
        const int grid = iq1s_grid_gpu[qs[l0 / 2] | ((qhl & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi[l0 / 4] = ggml_cuda_dp4a(grid0, u0, sumi[l0 / 4]);
        sumi[l0 / 4] = ggml_cuda_dp4a(grid1, u1, sumi[l0 / 4]);
        const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        int sumy = 0;
        sumy = ggml_cuda_dp4a(u0, 0x01010101, sumy);
        sumy = ggml_cuda_dp4a(u1, 0x01010101, sumy);
        sumf[l0 / 4] += delta * (float)sumy;
    }
    const uint16_t* sc = (const uint16_t*)bq1->scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d = __half2float(scale.f16) * __low2float(bq8_1[iqs].ds);
    const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
    const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
    return d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1);
}

__device__ __forceinline__ float vec_dot_iq4_nl_q8_1(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_1,
    const int& kbx,
    const int& iqs)
{
    const block_iq4_nl* bq4 = (const block_iq4_nl*)vbq + kbx;
    const int* q8 = (const int*)bq8_1->qs + iqs;
    int sumi = 0;
    #pragma unroll
    for (int l = 0; l < 2; ++l) {
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }
    const float d = __half2float(bq4->d) * __low2float(bq8_1->ds);
    return d * (float)sumi;
}

__device__ __forceinline__ float vec_dot_iq4_xs_q8_1(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_1,
    const int& kbx,
    const int& iqs)
{
    const block_iq4_xs* bq4 = (const block_iq4_xs*)vbq + kbx;
    int sumi = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int u0 = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
        sumi = ggml_cuda_dp4a(v.x, u0, sumi);
        sumi = ggml_cuda_dp4a(v.y, u1, sumi);
    }
    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = __half2float(bq4->d) * __low2float(bq8_1[iqs / 4].ds);
    return d * (float)sumi;
}

constexpr int VDR_Q4_K = 2, VDR_Q5_K = 2, VDR_Q5_1 = 2, VDR_Q5_0 = 2, VDR_Q4_1 = 2, VDR_Q4_0 = 2, VDR_Q8_0 = 2, VDR_Q6_K = 1;

__device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int* __restrict__ v,
    const int* __restrict__ u,
    const uint8_t* __restrict__ sc,
    const uint8_t* __restrict__ m,
    const half2& dm4,
    const float* __restrict__ d8)
{
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
    #pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;
        const int dot1 = ggml_cuda_dp4a(v1i, u[2 * i + 1], ggml_cuda_dp4a(v0i, u[2 * i + 0], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2 * i + 1], ggml_cuda_dp4a(0x01010101, u[2 * i + 0], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

__device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int* __restrict__ vl,
    const int* __restrict__ vh,
    const int* __restrict__ u,
    const uint8_t* __restrict__ sc,
    const uint8_t* __restrict__ m,
    const half2& dm5,
    const float* __restrict__ d8)
{
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
    #pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0F0F0F0F;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = ggml_cuda_dp4a(v0i, u[2 * i + 0], ggml_cuda_dp4a(v1i, u[2 * i + 1], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2 * i + 0], ggml_cuda_dp4a(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm5f = __half22float2(dm5);
    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

__device__ __forceinline__ void k_scale_min(const uint8_t* scales8, int bq8_offset, uint16_t aux[2]) {
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

__device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_1,
    const int& kbx,
    const int& iqs)
{
    const block_q4_K* bq4_K = (const block_q4_K*)vbq + kbx;
    int v[2];
    int u[2 * QR4_K];
    float d8[QR4_K];
    const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    const int* q4 = (const int*)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = q4[0];
    v[1] = q4[4];
    uint16_t aux[2];
    k_scale_min(bq4_K->scales, bq8_offset, aux);
    const uint8_t* sc = (const uint8_t*)aux;
    const uint8_t* m = sc + 2;
    #pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*)bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}

__device__ __forceinline__ float vec_dot_q5_K_q8_1(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_1,
    const int& kbx,
    const int& iqs)
{
    const block_q5_K* bq5_K = (const block_q5_K*)vbq + kbx;
    int vl[2];
    int vh[2];
    int u[2 * QR5_K];
    float d8[QR5_K];
    const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
    const int* ql = (const int*)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = (const int*)(bq5_K->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;
    uint16_t aux[2];
    k_scale_min(bq5_K->scales, bq8_offset, aux);
    const uint8_t* sc = (const uint8_t*)aux;
    const uint8_t* m = sc + 2;
    #pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*)bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
}

__device__ __forceinline__ float vec_dot_q5_0_q8_1(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_1,
    const int& kbx,
    const int& iqs)
{
    const block_q5_0* bq5_0 = (const block_q5_0*)vbq + kbx;
    int sumi = 0, sumu = 0;
    #pragma unroll
    for (int i = 0; i < VDR_Q5_0; ++i) {
        const int vl = get_int_b2(bq5_0->qs, iqs + i);
        const int vh = get_int_b2(bq5_0->qh, 0) >> (4 * (iqs + i));
        const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI5_0);
        int vi0 = (vl >> 0) & 0x0F0F0F0F;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi0, u0, sumi);
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi1, u1, sumi);
        sumu = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, sumu));
    }
    const float d5 = __half2float(bq5_0->d);
    const float d8 = __low2float(bq8_1->ds);
    return d5 * d8 * (float)(sumi - 16 * sumu);
}

__device__ __forceinline__ int q4_0_centred(const int v) {
    const uint32_t b = (uint32_t)v ^ 0x08080808u;
    return (int)(b | ((b & 0x08080808u) * 0x1Eu));
}

__device__ __forceinline__ float vec_dot_q4_0_q8_1(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_1,
    const int& kbx,
    const int& iqs)
{
    const block_q4_0* bq4 = (const block_q4_0*)vbq + kbx;
    int sumi = 0;
    #pragma unroll
    for (int i = 0; i < VDR_Q4_0; ++i) {
        const int v = get_int_b2(bq4->qs, iqs + i);
        sumi = ggml_cuda_dp4a(q4_0_centred((v >> 0) & 0x0F0F0F0F), get_int_b4(bq8_1->qs, iqs + i), sumi);
        sumi = ggml_cuda_dp4a(q4_0_centred((v >> 4) & 0x0F0F0F0F), get_int_b4(bq8_1->qs, iqs + i + QI4_0), sumi);
    }
    return __half2float(bq4->d) * __low2float(bq8_1->ds) * (float)sumi;
}

__device__ __forceinline__ float vec_dot_q4_1_q8_1(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_1,
    const int& kbx,
    const int& iqs)
{
    const block_q4_1* bq4_1 = (const block_q4_1*)vbq + kbx;
    int sumi = 0, sumu = 0;
    #pragma unroll
    for (int i = 0; i < VDR_Q4_1; ++i) {
        const int v = get_int_b4(bq4_1->qs, iqs + i);
        const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI4_1);
        sumi = ggml_cuda_dp4a((v >> 0) & 0x0F0F0F0F, u0, sumi);
        sumi = ggml_cuda_dp4a((v >> 4) & 0x0F0F0F0F, u1, sumi);
        sumu = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, sumu));
    }
    const float2 dm4 = __half22float2(bq4_1->dm);
    const float d8 = __low2float(bq8_1->ds);
    return sumi * (dm4.x * d8) + sumu * (dm4.y * d8);
}

__device__ __forceinline__ float vec_dot_q5_1_q8_1(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_1,
    const int& kbx,
    const int& iqs)
{
    const block_q5_1* bq5_1 = (const block_q5_1*)vbq + kbx;
    int sumi = 0, sumu = 0;
    #pragma unroll
    for (int i = 0; i < VDR_Q5_1; ++i) {
        const int vl = get_int_b4(bq5_1->qs, iqs + i);
        const int vh = get_int_b4(bq5_1->qh, 0) >> (4 * (iqs + i));
        const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI5_1);
        int vi0 = (vl >> 0) & 0x0F0F0F0F;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi0, u0, sumi);
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi1, u1, sumi);
        sumu = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, sumu));
    }
    const float2 dm5 = __half22float2(bq5_1->dm);
    const float d8 = __low2float(bq8_1->ds);
    return sumi * (dm5.x * d8) + sumu * (dm5.y * d8);
}

__device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq(
    const int vl,
    const int vh,
    const int* __restrict__ u,
    const int8_t* __restrict__ scales,
    const float d,
    const float* __restrict__ d8)
{
    float sumf = 0.0f;
    #pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4(vil | vih, 0x20202020);
        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc);
    }
    return d * sumf;
}

__device__ __forceinline__ float vec_dot_q6_K_q8_1(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_1,
    const int& kbx,
    const int& iqs)
{
    const block_q6_K* bq6_K = (const block_q6_K*)vbq + kbx;
    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
    const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
    const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));
    const int vl = get_int_b2(bq6_K->ql, iqs);
    const int vh = get_int_b2(bq6_K->qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_shift;
    const int8_t* scales = bq6_K->scales + scale_offset;
    int u[QR6_K];
    float d8[QR6_K];
    #pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i] = get_int_b4(bq8_1[bq8_offset + 2 * i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + 2 * i].ds);
    }
    return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, __half2float(bq6_K->d), d8);
}

__device__ __forceinline__ float vec_dot_q8_0_q8_1(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_1,
    const int& kbx,
    const int& iqs)
{
    const block_q8_0* bq8_0 = (const block_q8_0*)vbq + kbx;
    int sumi = 0;
    #pragma unroll
    for (int i = 0; i < VDR_Q8_0; ++i) {
        sumi = ggml_cuda_dp4a(get_int_b2(bq8_0->qs, iqs + i), get_int_b4(bq8_1->qs, iqs + i), sumi);
    }
    const float d8_0 = __half2float(bq8_0->d), d8_1 = __low2float(bq8_1->ds);
    return d8_0 * d8_1 * (float)sumi;
}

template <int TY> struct Fmt;
template<> struct Fmt<16> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<17> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<18> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<20> { static constexpr int qk = 32, ipb = 2, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_nl_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<21> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<22> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<23> { static constexpr int qk = 256, ipb = 8, step = 4;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_xs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<29> { static constexpr int qk = 256, ipb = 8, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq1_m_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<42> { static constexpr int qk = 64, ipb = 2, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q2_0_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<12> { static constexpr int qk = 256, ipb = QI4_K / VDR_Q4_K, step = VDR_Q4_K;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q4_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<13> { static constexpr int qk = 256, ipb = QI5_K / VDR_Q5_K, step = VDR_Q5_K;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<14> { static constexpr int qk = 256, ipb = QI6_K / VDR_Q6_K, step = VDR_Q6_K;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q6_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<7> { static constexpr int qk = 32, ipb = QI5_1 / VDR_Q5_1, step = VDR_Q5_1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_1_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<2> { static constexpr int qk = 32, ipb = QI4_0 / VDR_Q4_0, step = VDR_Q4_0;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q4_0_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<3> { static constexpr int qk = 32, ipb = QI4_1 / VDR_Q4_1, step = VDR_Q4_1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q4_1_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<6> { static constexpr int qk = 32, ipb = QI5_0 / VDR_Q5_0, step = VDR_Q5_0;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_0_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<8> { static constexpr int qk = 32, ipb = QI8_0 / VDR_Q8_0, step = VDR_Q8_0;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q8_0_q8_1(v, y, kbx, iqs); } };

template <int TY>
__device__ __forceinline__ float row_dot(const uint8_t* row, const block_q8_1* x, int nb, int lane) {
    using F = Fmt<TY>;
    float s = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        s += F::dot(row, x + kbx * (F::qk / 32), kbx, iqs);
    }
    return warp_sum(s);
}

template <int TY>
__global__ void __launch_bounds__(128) mmvq_kernel(
    const uint8_t* __restrict__ w,
    size_t row_bytes,
    const block_q8_1* __restrict__ x,
    float* __restrict__ y,
    int n_in,
    int n_out)
{
    const int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= n_out) return;
    const int lane = threadIdx.x;
    const int nb = n_in / Fmt<TY>::qk;
    const uint8_t* wr = w + (size_t)row * row_bytes;
    const float s = row_dot<TY>(wr, x, nb, lane);
    if (lane == 0) y[row] = s;
}

// Batched MMVQ across 10 routed experts (blockIdx.y = 0..9)
template <int TY>
__global__ void __launch_bounds__(128) mmvq_batch10_kernel(
    const uint8_t* __restrict__ base_w,
    size_t expert_stride_bytes,
    const void* const* __restrict__ expert_ptrs,
    const int32_t* __restrict__ d_top_idx,
    size_t row_bytes,
    const block_q8_1* __restrict__ x,
    size_t x_expert_stride_blocks,
    float* __restrict__ y,
    int n_in,
    int n_out)
{
    const int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= n_out) return;
    const int k = blockIdx.y; // 0..9
    const uint8_t* w_exp = (base_w != nullptr)
        ? (base_w + (size_t)(d_top_idx != nullptr ? d_top_idx[k] : k) * expert_stride_bytes)
        : (const uint8_t*)expert_ptrs[k];
    const block_q8_1* x_exp = x + (size_t)k * x_expert_stride_blocks;
    const int lane = threadIdx.x;
    const int nb = n_in / Fmt<TY>::qk;
    const uint8_t* wr = w_exp + (size_t)row * row_bytes;
    const float s = row_dot<TY>(wr, x_exp, nb, lane);
    if (lane == 0) y[(size_t)k * n_out + row] = s;
}

// ============================================================================
// BF16 / F16 / F32 Matrix-Vector Multiplication
// ============================================================================

__global__ void __launch_bounds__(128) bf16_gemv_kernel(
    const uint16_t* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ y,
    int n_rows,
    int n_cols)
{
    const int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= n_rows) return;
    const int lane = threadIdx.x;
    const uint16_t* w_row = W + (size_t)row * n_cols;
    float acc = 0.0f;
    for (int j = lane; j < n_cols; j += 32) {
        acc += qwen4_bf16_to_fp32(w_row[j]) * x[j];
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

__global__ void __launch_bounds__(128) f16_gemv_kernel(
    const half* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ y,
    int n_rows,
    int n_cols)
{
    const int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= n_rows) return;
    const int lane = threadIdx.x;
    const half* w_row = W + (size_t)row * n_cols;
    float acc = 0.0f;
    for (int j = lane; j < n_cols; j += 32) {
        acc += __half2float(w_row[j]) * x[j];
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

__global__ void __launch_bounds__(128) f32_gemv_kernel(
    const float* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ y,
    int n_rows,
    int n_cols)
{
    const int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= n_rows) return;
    const int lane = threadIdx.x;
    const float* w_row = W + (size_t)row * n_cols;
    float acc = 0.0f;
    for (int j = lane; j < n_cols; j += 32) {
        acc += w_row[j] * x[j];
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

// ============================================================================
// Token Embedding Dequantize + 4-Stream Hyper-Connection Broadcast
// ============================================================================

__global__ void embed_to_hc_kernel(
    int32_t embd_type,
    const uint8_t* __restrict__ row_ptr,
    float* __restrict__ d_R,
    int n_embd,
    int n_streams)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_embd) return;

    float v = 0.0f;
    if (embd_type == QWEN4_GGML_TYPE_IQ4_XS) {
        const block_iq4_xs* bq = (const block_iq4_xs*)row_ptr + (col / 256);
        const int in_b = col % 256;
        const int ib = in_b / 32;
        const int in_sub = in_b % 32;
        const int j = in_sub % 16;
        const int is_high = in_sub / 16;
        const int ls = ((bq->scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F) | (((bq->scales_h >> (2 * ib)) & 3) << 4);
        const float dl = __half2float(bq->d) * (float)(ls - 32);
        const uint8_t qbyte = bq->qs[ib * 16 + j];
        const uint8_t nib = is_high ? (qbyte >> 4) : (qbyte & 0x0F);
        v = dl * (float)kvalues_iq4nl[nib];
    } else if (embd_type == QWEN4_GGML_TYPE_IQ4_NL) {
        const block_iq4_nl* bq = (const block_iq4_nl*)row_ptr + (col / 32);
        const int in_b = col % 32;
        const int j = in_b % 16;
        const int is_high = in_b / 16;
        const uint8_t qbyte = bq->qs[j];
        const uint8_t nib = is_high ? (qbyte >> 4) : (qbyte & 0x0F);
        v = __half2float(bq->d) * (float)kvalues_iq4nl[nib];
    } else if (embd_type == QWEN4_GGML_TYPE_Q4_K) {
        const block_q4_K* bq = (const block_q4_K*)row_ptr + (col / 256);
        const int in_b = col % 256;
        const int group64 = in_b / 64;
        const int in_64 = in_b % 64;
        const int is_high = in_64 / 32;
        const int l = in_64 % 32;
        const int is = group64 * 2 + is_high;
        const uint8_t* q = bq->scales;
        uint8_t sc, m;
        if (is < 4) {
            sc = q[is] & 63;
            m  = q[is + 4] & 63;
        } else {
            sc = (q[is + 4] & 0x0F) | ((q[is - 4] >> 6) << 4);
            m  = (q[is + 4] >> 4)   | ((q[is - 0] >> 6) << 4);
        }
        const float2 dm = __half22float2(bq->dm);
        const uint8_t qbyte = bq->qs[group64 * 32 + l];
        const uint8_t nib = is_high ? (qbyte >> 4) : (qbyte & 0x0F);
        v = dm.x * (float)sc * (float)nib - dm.y * (float)m;
    } else if (embd_type == QWEN4_GGML_TYPE_Q6_K) {
        const block_q6_K* bq = (const block_q6_K*)row_ptr + (col / 256);
        const int in_b = col % 256;
        const int half128 = in_b / 128;
        const int in_128 = in_b % 128;
        const int q_group = in_128 / 32; // 0..3
        const int l = in_128 % 32;
        const int is = l / 16;
        const uint8_t* ql = bq->ql + half128 * 64;
        const uint8_t* qh = bq->qh + half128 * 32;
        const int8_t* sc = bq->scales + half128 * 8;
        const uint8_t ql_byte = ql[l + (q_group & 1) * 32];
        const uint8_t low4 = (q_group >= 2) ? (ql_byte >> 4) : (ql_byte & 0x0F);
        const uint8_t high2 = (qh[l] >> (2 * q_group)) & 3;
        const int q6 = (int)(low4 | (high2 << 4)) - 32;
        v = __half2float(bq->d) * (float)sc[is + q_group * 2] * (float)q6;
    } else if (embd_type == QWEN4_GGML_TYPE_Q8_0) {
        const block_q8_0* bq = (const block_q8_0*)row_ptr + (col / 32);
        v = __half2float(bq->d) * (float)bq->qs[col % 32];
    } else if (embd_type == QWEN4_GGML_TYPE_Q4_0) {
        const block_q4_0* bq = (const block_q4_0*)row_ptr + (col / 32);
        const int in_b = col % 32;
        const uint8_t qbyte = bq->qs[in_b % 16];
        const int nib = (in_b >= 16) ? (qbyte >> 4) : (qbyte & 0x0F);
        v = __half2float(bq->d) * (float)(nib - 8);
    } else if (embd_type == QWEN4_GGML_TYPE_BF16) {
        v = qwen4_bf16_to_fp32(((const uint16_t*)row_ptr)[col]);
    } else if (embd_type == QWEN4_GGML_TYPE_F16) {
        v = __half2float(((const half*)row_ptr)[col]);
    } else if (embd_type == QWEN4_GGML_TYPE_F32) {
        v = ((const float*)row_ptr)[col];
    }

    for (int s = 0; s < n_streams; ++s) {
        d_R[s * n_embd + col] = v;
    }
}

__global__ void vec_to_hc_kernel(
    const float* __restrict__ d_embd,
    float* __restrict__ d_R,
    int n_embd,
    int n_streams)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_embd) return;
    const float v = d_embd[col];
    for (int s = 0; s < n_streams; ++s) {
        d_R[s * n_embd + col] = v;
    }
}

// ============================================================================
// 4-Stream Hyper-Connection (HC Read & HC Write) - matches hc.cuh
// ============================================================================

constexpr int QWEN4_N_STREAMS    = 4;
constexpr int QWEN4_N_EMBD       = 2560;
constexpr int QWEN4_STREAM_TOTAL = QWEN4_N_STREAMS * QWEN4_N_EMBD; // 10240
constexpr int QWEN4_HC_RANK      = 320;
constexpr int QWEN4_GR_WARPS     = 8; // 256 threads

// 4 blocks (one per stream of 2560 floats): per-stream RMSNorm -> normed[4, 2560]
__global__ void gr_stream_rmsnorm_bf16_kernel(
    const float* __restrict__ R,
    const float* __restrict__ norm_w,
    float* __restrict__ normed,
    float eps)
{
    __shared__ float smem[QWEN4_GR_WARPS];
    const int s = blockIdx.x;
    const float* rs = R + s * QWEN4_N_EMBD;
    const float* ws = norm_w + s * QWEN4_N_EMBD;
    float* out = normed + s * QWEN4_N_EMBD;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < QWEN4_N_EMBD; i += blockDim.x) {
        const float v = rs[i];
        ss += v * v;
    }
    ss = block_sum<QWEN4_GR_WARPS>(ss, smem);
    const float scale = rsqrtf(ss / (float)QWEN4_N_EMBD + eps);

    for (int i = threadIdx.x; i < QWEN4_N_EMBD; i += blockDim.x) {
        out[i] = rs[i] * scale * ws[i];
    }
}

// 320 blocks (one per rank row): h[r] = silu(dot(down_bf16[r], normed) / 4.0f)
__global__ void gr_down_gemv_kernel(
    const uint16_t* __restrict__ down_bf16,
    const float* __restrict__ normed,
    float* __restrict__ h)
{
    __shared__ float smem[QWEN4_GR_WARPS];
    const int r = blockIdx.x;
    const uint16_t* row = down_bf16 + (size_t)r * QWEN4_STREAM_TOTAL;

    float acc = 0.0f;
    for (int j = threadIdx.x; j < QWEN4_STREAM_TOTAL; j += blockDim.x) {
        acc += qwen4_bf16_to_fp32(row[j]) * normed[j];
    }
    acc = block_sum<QWEN4_GR_WARPS>(acc, smem);
    if (threadIdx.x == 0) {
        const float x = acc * (1.0f / (float)QWEN4_N_STREAMS);
        h[r] = x / (1.0f + expf(-x));
    }
}

// 2560 blocks (one per output dim j): computes g[s, j] = dot(up_bf16[s*2560+j], h) and x_out[j] = mean_s(sigmoid(g[s, j]) * normed[s, j])
__global__ void gr_up_and_mean_kernel(
    const uint16_t* __restrict__ up_bf16,
    const float* __restrict__ h,
    const float* __restrict__ normed,
    float* __restrict__ x_out)
{
    const int j = blockIdx.x; // 0..2559
    const int s = threadIdx.y; // 0..3 (4 warps per block, 1 warp per stream s)
    const int lane = threadIdx.x; // 0..31

    const int row_idx = s * QWEN4_N_EMBD + j;
    const uint16_t* row = up_bf16 + (size_t)row_idx * QWEN4_HC_RANK;

    float acc = 0.0f;
    for (int k = lane; k < QWEN4_HC_RANK; k += 32) {
        acc += qwen4_bf16_to_fp32(row[k]) * h[k];
    }
    acc = warp_sum(acc);

    __shared__ float s_sum[4];
    if (lane == 0) {
        const float sig = 1.0f / (1.0f + expf(-acc));
        s_sum[s] = sig * normed[row_idx];
    }
    __syncthreads();
    if (s == 0 && lane == 0) {
        x_out[j] = 0.25f * (s_sum[0] + s_sum[1] + s_sum[2] + s_sum[3]);
    }
}

// 4 blocks (one per stream): computes inj[s] = dot(inject_bf16[s], normed) and R[s, j] += 2 * sigmoid(inj[s] / 4) * sub_out[j]
__global__ void gr_inject_and_write_kernel(
    const uint16_t* __restrict__ inject_bf16,
    const float* __restrict__ normed,
    const float* __restrict__ sub_out,
    float* __restrict__ R)
{
    __shared__ float smem[QWEN4_GR_WARPS];
    const int s = blockIdx.x; // 0..3
    const uint16_t* row = inject_bf16 + (size_t)s * QWEN4_STREAM_TOTAL;

    float acc = 0.0f;
    for (int j = threadIdx.x; j < QWEN4_STREAM_TOTAL; j += blockDim.x) {
        acc += qwen4_bf16_to_fp32(row[j]) * normed[j];
    }
    const float inj = block_sum<QWEN4_GR_WARPS>(acc, smem);
    const float gate = 2.0f / (1.0f + expf(-inj * (1.0f / (float)QWEN4_N_STREAMS)));

    float* rs = R + s * QWEN4_N_EMBD;
    for (int j = threadIdx.x; j < QWEN4_N_EMBD; j += blockDim.x) {
        rs[j] += gate * sub_out[j];
    }
}

// ============================================================================
// Layer 1 PLE + 16-Head N-Gram Gated Dilated Conv Block (matches kernels.cu L791-846)
// ============================================================================

constexpr int QWEN4_PLE_HIST = 9;
constexpr int QWEN4_PLE_DIL  = 3;

// 80 blocks x 16 threads: dequantize 16 * 5 = 80 IQ4_NL blocks (1440 B) -> 2560 floats
__global__ void ple_iq4nl_dequant_kernel(
    const block_iq4_nl* __restrict__ src,
    float* __restrict__ dst)
{
    const int b = blockIdx.x; // 0..79
    const int j = threadIdx.x; // 0..15
    const float d = __half2float(src[b].d);
    const uint8_t q = src[b].qs[j];
    float* y = dst + b * 32;
    y[j]      = d * (float)kvalues_iq4nl[q & 0x0F];
    y[j + 16] = d * (float)kvalues_iq4nl[q >> 4];
}

// 4 blocks (c = 0..3), 256 threads: grouped RMSNorm of key[c] & R[c], signed-sqrt sigmoid gate, gated val & conv norm
__global__ void ple_gate_kernel(
    const float* __restrict__ R,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ nk,
    const float* __restrict__ nq,
    const float* __restrict__ nc,
    float* __restrict__ gated,
    float* __restrict__ normd,
    float eps)
{
    __shared__ float smem[QWEN4_GR_WARPS];
    const int c = blockIdx.x; // 0..3
    const int base = c * QWEN4_N_EMBD;

    float sk = 0.0f, sq = 0.0f;
    for (int i = threadIdx.x; i < QWEN4_N_EMBD; i += blockDim.x) {
        const float kv = key[base + i];
        const float qv = R[base + i];
        sk += kv * kv;
        sq += qv * qv;
    }
    const float rk = rsqrtf(block_sum<QWEN4_GR_WARPS>(sk, smem) / (float)QWEN4_N_EMBD + eps);
    const float rq = rsqrtf(block_sum<QWEN4_GR_WARPS>(sq, smem) / (float)QWEN4_N_EMBD + eps);

    float dot = 0.0f;
    for (int i = threadIdx.x; i < QWEN4_N_EMBD; i += blockDim.x) {
        const float kn = key[base + i] * rk * nk[base + i];
        const float qn = R[base + i]   * rq * nq[base + i];
        dot += kn * qn;
    }
    const float s = block_sum<QWEN4_GR_WARPS>(dot, smem) * rsqrtf((float)QWEN4_N_EMBD);
    const float sgn = (s > 0.0f) ? 1.0f : ((s < 0.0f) ? -1.0f : 0.0f);
    const float gate = 1.0f / (1.0f + expf(-sgn * sqrtf(fmaxf(fabsf(s), 1e-6f))));

    float sg = 0.0f;
    for (int i = threadIdx.x; i < QWEN4_N_EMBD; i += blockDim.x) {
        const float gv = value[i] * gate;
        gated[base + i] = gv;
        sg += gv * gv;
    }
    const float rg = rsqrtf(block_sum<QWEN4_GR_WARPS>(sg, smem) / (float)QWEN4_N_EMBD + eps);
    for (int i = threadIdx.x; i < QWEN4_N_EMBD; i += blockDim.x) {
        normd[base + i] = gated[base + i] * rg * nc[base + i];
    }
}

// 40 blocks x 256 threads (ch = 0..10239): dilated causal conv1d (dilation 3 over 9 history rows + current), silu, residual update
__global__ void ple_conv_update_kernel(
    float* __restrict__ R,
    const float* __restrict__ gated,
    const float* __restrict__ normd,
    const half* __restrict__ conv_w_f16,
    float* __restrict__ hist)
{
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= QWEN4_STREAM_TOTAL) return;

    float pad[QWEN4_PLE_HIST + 1];
    #pragma unroll
    for (int i = 0; i < QWEN4_PLE_HIST; ++i) {
        pad[i] = hist[i * QWEN4_STREAM_TOTAL + ch];
    }
    pad[QWEN4_PLE_HIST] = normd[ch];

    const half* w = conv_w_f16 + ch * 4;
    float s = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        s += __half2float(w[k]) * pad[QWEN4_PLE_HIST - (3 - k) * QWEN4_PLE_DIL];
    }
    const float silu_s = s / (1.0f + expf(-s));
    R[ch] += gated[ch] + silu_s;

    #pragma unroll
    for (int i = 0; i < QWEN4_PLE_HIST; ++i) {
        hist[i * QWEN4_STREAM_TOTAL + ch] = pad[i + 1];
    }
}

// ============================================================================
// 36 Gated DeltaNet (GDN) Layers
// ============================================================================

constexpr int QWEN4_GDN_N_K_HEAD = 16;
constexpr int QWEN4_GDN_N_V_HEAD = 48;
constexpr int QWEN4_GDN_HEAD_DIM = 128;
constexpr int QWEN4_GDN_QK_DIM   = QWEN4_GDN_N_K_HEAD * QWEN4_GDN_HEAD_DIM; // 2048
constexpr int QWEN4_GDN_V_DIM    = QWEN4_GDN_N_V_HEAD * QWEN4_GDN_HEAD_DIM; // 6144
constexpr int QWEN4_GDN_QKV_DIM  = 2 * QWEN4_GDN_QK_DIM + QWEN4_GDN_V_DIM;  // 10240

__global__ void gdn_conv_silu_kernel(
    float* __restrict__ conv_state,
    const float* __restrict__ qkv,
    const float* __restrict__ conv_w,
    float* __restrict__ qkv_conv)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= QWEN4_GDN_QKV_DIM) return;

    const float s0 = conv_state[0 * QWEN4_GDN_QKV_DIM + c];
    const float s1 = conv_state[1 * QWEN4_GDN_QKV_DIM + c];
    const float s2 = conv_state[2 * QWEN4_GDN_QKV_DIM + c];
    const float s3 = qkv[c];

    conv_state[0 * QWEN4_GDN_QKV_DIM + c] = s1;
    conv_state[1 * QWEN4_GDN_QKV_DIM + c] = s2;
    conv_state[2 * QWEN4_GDN_QKV_DIM + c] = s3;

    const float* w = conv_w + c * 4;
    const float v = s0 * w[0] + s1 * w[1] + s2 * w[2] + s3 * w[3];
    qkv_conv[c] = v / (1.0f + expf(-v));
}

// Fused alpha + beta BF16 GEMV + gate/beta activation (48 rows x 2)
__global__ void __launch_bounds__(128) gdn_alpha_beta_gemv_kernel(
    const uint16_t* __restrict__ w_alpha_bf16,
    const uint16_t* __restrict__ w_beta_bf16,
    const float* __restrict__ x,
    const float* __restrict__ ssm_a,
    const float* __restrict__ dt_bias,
    float* __restrict__ gate,
    float* __restrict__ beta)
{
    const int h = blockIdx.x * 4 + threadIdx.y; // 0..47
    if (h >= QWEN4_GDN_N_V_HEAD) return;
    const int lane = threadIdx.x;

    const uint16_t* wa = w_alpha_bf16 + (size_t)h * QWEN4_N_EMBD;
    const uint16_t* wb = w_beta_bf16  + (size_t)h * QWEN4_N_EMBD;

    float acc_a = 0.0f, acc_b = 0.0f;
    for (int j = lane; j < QWEN4_N_EMBD; j += 32) {
        const float xj = x[j];
        acc_a += qwen4_bf16_to_fp32(wa[j]) * xj;
        acc_b += qwen4_bf16_to_fp32(wb[j]) * xj;
    }
    acc_a = warp_sum(acc_a);
    acc_b = warp_sum(acc_b);

    if (lane == 0) {
        const float x_dt = acc_a + dt_bias[h];
        const float sp = (x_dt > 20.0f) ? x_dt : logf(1.0f + expf(x_dt));
        gate[h] = ssm_a[h] * sp;
        beta[h] = 1.0f / (1.0f + expf(-acc_b));
    }
}

// 32 blocks (0..15 = Q heads, 16..31 = K heads), 128 threads
__global__ void gdn_qk_l2norm_kernel(float* __restrict__ qkv_conv, float eps) {
    __shared__ float smem[4];
    const int b = blockIdx.x; // 0..31
    const int d = threadIdx.x; // 0..127
    float* head = qkv_conv + b * QWEN4_GDN_HEAD_DIM;
    const float v = head[d];
    const float ss = block_sum<4>(v * v, smem);
    const float q_scale = (b < QWEN4_GDN_N_K_HEAD) ? rsqrtf((float)QWEN4_GDN_HEAD_DIM) : 1.0f;
    head[d] = v * rsqrtf(ss + eps) * q_scale;
}

// 48 blocks (one per V head), 128 threads (one per col j = 0..127)
// Modulo head mapping: qk_head = h % 16
__global__ void gdn_recurrence_kernel(
    float* __restrict__ ssm_state,
    const float* __restrict__ qkv_conv,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ raw_out)
{
    __shared__ float q_sh[QWEN4_GDN_HEAD_DIM];
    __shared__ float k_sh[QWEN4_GDN_HEAD_DIM];

    const int h = blockIdx.x; // 0..47
    const int j = threadIdx.x; // 0..127
    const int qk_h = h % QWEN4_GDN_N_K_HEAD;

    q_sh[j] = qkv_conv[qk_h * QWEN4_GDN_HEAD_DIM + j];
    k_sh[j] = qkv_conv[QWEN4_GDN_QK_DIM + qk_h * QWEN4_GDN_HEAD_DIM + j];
    __syncthreads();

    float* S = ssm_state + (size_t)h * (QWEN4_GDN_HEAD_DIM * QWEN4_GDN_HEAD_DIM);
    const float decay = expf(gate[h]);
    const float b = beta[h];
    const float v_j = qkv_conv[2 * QWEN4_GDN_QK_DIM + h * QWEN4_GDN_HEAD_DIM + j];

    float kv_j = 0.0f;
    for (int i = 0; i < QWEN4_GDN_HEAD_DIM; ++i) {
        kv_j += (S[i * QWEN4_GDN_HEAD_DIM + j] * decay) * k_sh[i];
    }

    const float delta_j = (v_j - kv_j) * b;

    float out_j = 0.0f;
    for (int i = 0; i < QWEN4_GDN_HEAD_DIM; ++i) {
        const float s_ij = S[i * QWEN4_GDN_HEAD_DIM + j] * decay + k_sh[i] * delta_j;
        S[i * QWEN4_GDN_HEAD_DIM + j] = s_ij;
        out_j += s_ij * q_sh[i];
    }
    raw_out[h * QWEN4_GDN_HEAD_DIM + j] = out_j;
}

// 48 blocks, 128 threads: per-head RMSNorm * ssm_norm * sigmoid(z) (Flash-Next gates with sigmoid(z))
__global__ void gdn_norm_gate_kernel(
    const float* __restrict__ raw_out,
    const float* __restrict__ ssm_norm,
    const float* __restrict__ z,
    float* __restrict__ out,
    float eps)
{
    __shared__ float smem[4];
    const int h = blockIdx.x; // 0..47
    const int d = threadIdx.x; // 0..127
    const int idx = h * QWEN4_GDN_HEAD_DIM + d;

    const float v = raw_out[idx];
    const float ss = block_sum<4>(v * v, smem);
    const float normed = v * rsqrtf(ss / (float)QWEN4_GDN_HEAD_DIM + eps) * ssm_norm[d];
    const float zv = z[idx];
    const float sig_z = 1.0f / (1.0f + expf(-zv));
    out[idx] = normed * sig_z;
}

// ============================================================================
// 12 Query-Sparse Attention (QSA) Layers
// ============================================================================

constexpr int QWEN4_QSA_N_HEAD    = 24;
constexpr int QWEN4_QSA_N_KV      = 2;
constexpr int QWEN4_QSA_HEAD_DIM  = 256;
constexpr int QWEN4_QSA_N_ROT     = 64;
constexpr float QWEN4_QSA_THETA   = 10000000.0f;
constexpr int QWEN4_QSA_IDX_HEADS = 4;
constexpr int QWEN4_QSA_IDX_DIM   = 128;
constexpr int QWEN4_QSA_IDX_BLOCK = 4;
constexpr int QWEN4_QSA_IDX_TOPK  = 2048;
constexpr int QWEN4_QSA_MAX_SEL   = QWEN4_QSA_IDX_TOPK + QWEN4_QSA_IDX_BLOCK - 1; // 2051

// 24 blocks, 256 threads: split q_gate[24, 512] -> q[24, 256] + gate[24, 256], RMSNorm + partial NeoX RoPE on q
__global__ void qsa_prep_q_kernel(
    const float* __restrict__ q_gate,
    const float* __restrict__ q_norm,
    float* __restrict__ q,
    float* __restrict__ gate,
    int pos,
    float eps)
{
    __shared__ float smem[8];
    const int h = blockIdx.x; // 0..23
    const int d = threadIdx.x; // 0..255

    const float* src = q_gate + h * (2 * QWEN4_QSA_HEAD_DIM);
    float* qh = q + h * QWEN4_QSA_HEAD_DIM;
    float* gh = gate + h * QWEN4_QSA_HEAD_DIM;

    const float v = src[d];
    gh[d] = src[QWEN4_QSA_HEAD_DIM + d];

    const float ss = block_sum<8>(v * v, smem);
    qh[d] = v * rsqrtf(ss / (float)QWEN4_QSA_HEAD_DIM + eps) * q_norm[d];
    __syncthreads();

    rope_neox_elem(qh, d, QWEN4_QSA_N_ROT, pos, QWEN4_QSA_THETA);
}

// 2 blocks, 256 threads: RMSNorm + partial NeoX RoPE on k[2, 256], store k and v into BF16 KV cache at `pos`
__global__ void qsa_prep_kv_kernel(
    const float* __restrict__ k_in,
    const float* __restrict__ v_in,
    const float* __restrict__ k_norm,
    uint16_t* __restrict__ k_cache,
    uint16_t* __restrict__ v_cache,
    int pos,
    float eps)
{
    __shared__ float smem[8];
    __shared__ float s_kh[QWEN4_QSA_HEAD_DIM];
    const int h = blockIdx.x; // 0..1
    const int d = threadIdx.x; // 0..255

    uint16_t* kh = k_cache + (size_t)pos * (QWEN4_QSA_N_KV * QWEN4_QSA_HEAD_DIM) + h * QWEN4_QSA_HEAD_DIM;
    uint16_t* vh = v_cache + (size_t)pos * (QWEN4_QSA_N_KV * QWEN4_QSA_HEAD_DIM) + h * QWEN4_QSA_HEAD_DIM;

    const float kv = k_in[h * QWEN4_QSA_HEAD_DIM + d];
    vh[d] = qwen4_fp32_to_bf16(v_in[h * QWEN4_QSA_HEAD_DIM + d]);

    const float ss = block_sum<8>(kv * kv, smem);
    s_kh[d] = kv * rsqrtf(ss / (float)QWEN4_QSA_HEAD_DIM + eps) * k_norm[d];
    __syncthreads();

    rope_neox_elem(s_kh, d, QWEN4_QSA_N_ROT, pos, QWEN4_QSA_THETA);
    __syncthreads();
    kh[d] = qwen4_fp32_to_bf16(s_kh[d]);
}

// 1 block, 128 threads: append idx_raw[128] into indexer tail/dead/pooled
__global__ void qsa_indexer_append_kernel(
    const float* __restrict__ idx_raw,
    const float* __restrict__ idx_k_norm,
    float* __restrict__ idx_pooled,
    float* __restrict__ idx_tail,
    float* __restrict__ idx_dead,
    int pos,
    float eps)
{
    __shared__ float smem[4];
    const int d = threadIdx.x; // 0..127
    const int slot = pos % QWEN4_QSA_IDX_BLOCK;
    idx_tail[slot * QWEN4_QSA_IDX_DIM + d] = idx_raw[d];

    if (pos == 0) {
        const float v = idx_raw[d] * 0.25f;
        const float ss = block_sum<4>(v * v, smem);
        idx_dead[d] = v * rsqrtf(ss / (float)QWEN4_QSA_IDX_DIM + eps) * idx_k_norm[d];
    }

    if (slot == QWEN4_QSA_IDX_BLOCK - 1) {
        const int b = pos / QWEN4_QSA_IDX_BLOCK;
        const float v = 0.25f * (
            idx_tail[0 * QWEN4_QSA_IDX_DIM + d] +
            idx_tail[1 * QWEN4_QSA_IDX_DIM + d] +
            idx_tail[2 * QWEN4_QSA_IDX_DIM + d] +
            idx_tail[3 * QWEN4_QSA_IDX_DIM + d]);
        const float ss = block_sum<4>(v * v, smem);
        float* pb = idx_pooled + (size_t)b * QWEN4_QSA_IDX_DIM;
        pb[d] = v * rsqrtf(ss / (float)QWEN4_QSA_IDX_DIM + eps) * idx_k_norm[d];
        __syncthreads();
        rope_neox_elem(pb, d, QWEN4_QSA_N_ROT, b * QWEN4_QSA_IDX_BLOCK, QWEN4_QSA_THETA);
    }
}

// 4 blocks, 128 threads: RMSNorm + partial NeoX RoPE on q_idx[4, 128]
__global__ void qsa_indexer_prep_q_kernel(
    float* __restrict__ q_idx,
    const float* __restrict__ idx_q_norm,
    int pos,
    float eps)
{
    __shared__ float smem[4];
    const int h = blockIdx.x; // 0..3
    const int d = threadIdx.x; // 0..127
    float* qh = q_idx + h * QWEN4_QSA_IDX_DIM;
    const float v = qh[d];
    const float ss = block_sum<4>(v * v, smem);
    qh[d] = v * rsqrtf(ss / (float)QWEN4_QSA_IDX_DIM + eps) * idx_q_norm[d];
    __syncthreads();
    rope_neox_elem(qh, d, QWEN4_QSA_N_ROT, pos, QWEN4_QSA_THETA);
}

// Score blocks b = 0..n_blocks-1 (n_blocks blocks, 128 threads)
__global__ void qsa_indexer_score_blocks_kernel(
    const float* __restrict__ idx_pooled,
    const float* __restrict__ q_idx,
    float* __restrict__ block_scores,
    int n_kv,
    int n_blocks)
{
    __shared__ float smem[4];
    const int b = blockIdx.x;
    if (b >= n_blocks) return;
    const int d = threadIdx.x; // 0..127

    const int start = b * QWEN4_QSA_IDX_BLOCK;
    const int valid = (n_kv - start < QWEN4_QSA_IDX_BLOCK) ? (n_kv - start) : QWEN4_QSA_IDX_BLOCK;
    if (valid < QWEN4_QSA_IDX_BLOCK) {
        if (d == 0) block_scores[b] = 1e9f;
        return;
    }

    const float k_val = idx_pooled[(size_t)b * QWEN4_QSA_IDX_DIM + d];
    float total = 0.0f;
    for (int h = 0; h < QWEN4_QSA_IDX_HEADS; ++h) {
        const float prod = k_val * q_idx[h * QWEN4_QSA_IDX_DIM + d];
        const float dot = block_sum<4>(prod, smem);
        if (d == 0 && dot > 0.0f) total += dot;
    }
    if (d == 0) block_scores[b] = total;
}

// GPU top-block selector (1 block of 256 threads): selects top `n_sel` cells in descending (score, -b) order
__global__ void qsa_indexer_select_topk_kernel(
    float* __restrict__ block_scores,
    int32_t* __restrict__ d_sel_indices,
    int n_kv,
    int n_blocks,
    int n_sel)
{
    __shared__ float s_val[256];
    __shared__ int   s_idx[256];
    __shared__ int   s_written;

    const int tid = threadIdx.x;
    if (tid == 0) s_written = 0;
    __syncthreads();

    while (s_written < n_sel) {
        float best_v = -1e30f;
        int   best_b = 0x7FFFFFFF;
        for (int b = tid; b < n_blocks; b += 256) {
            const float v = block_scores[b];
            if (v > best_v || (v == best_v && b < best_b)) {
                best_v = v;
                best_b = b;
            }
        }
        s_val[tid] = best_v;
        s_idx[tid] = best_b;
        __syncthreads();

        for (int stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float v2 = s_val[tid + stride];
                const int   b2 = s_idx[tid + stride];
                const float v1 = s_val[tid];
                const int   b1 = s_idx[tid];
                if (v2 > v1 || (v2 == v1 && b2 < b1)) {
                    s_val[tid] = v2;
                    s_idx[tid] = b2;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            const int win_b = s_idx[0];
            block_scores[win_b] = -1e30f;
            const int start = win_b * QWEN4_QSA_IDX_BLOCK;
            int w = s_written;
            for (int t = 0; t < QWEN4_QSA_IDX_BLOCK && (start + t) < n_kv && w < n_sel; ++t) {
                d_sel_indices[w++] = start + t;
            }
            s_written = w;
        }
        __syncthreads();
    }
}

// 24 blocks (one per query head), 256 threads (one per head_dim d = 0..255)
// Online softmax attention over either dense 0..n_kv-1 (sel_indices == nullptr) or sparse sel_indices[0..n_sel-1]
// Group-major GQA mapping: kv_h = h / 12
__global__ void qsa_attend_gate_kernel(
    const float* __restrict__ q,
    const float* __restrict__ gate,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    const int32_t* __restrict__ sel_indices,
    int n_sel,
    float* __restrict__ attn_gated)
{
    __shared__ float smem[8];
    const int h = blockIdx.x; // 0..23
    const int d = threadIdx.x; // 0..255
    const int kv_h = h / (QWEN4_QSA_N_HEAD / QWEN4_QSA_N_KV); // h / 12
    const float scale = rsqrtf((float)QWEN4_QSA_HEAD_DIM); // 1/16 = 0.0625f

    const float q_d = q[h * QWEN4_QSA_HEAD_DIM + d];
    float m = -1e30f;
    float l = 0.0f;
    float acc_d = 0.0f;

    for (int s = 0; s < n_sel; ++s) {
        const int pos = (sel_indices != nullptr) ? sel_indices[s] : s;
        const size_t kv_off = (size_t)pos * (QWEN4_QSA_N_KV * QWEN4_QSA_HEAD_DIM) + kv_h * QWEN4_QSA_HEAD_DIM;
        const float k_d = qwen4_bf16_to_fp32(k_cache[kv_off + d]);
        const float score = block_sum<8>(q_d * k_d, smem) * scale;

        const float m_new = fmaxf(m, score);
        const float alpha = expf(m - m_new);
        const float p     = expf(score - m_new);
        l = l * alpha + p;
        acc_d = acc_d * alpha + p * qwen4_bf16_to_fp32(v_cache[kv_off + d]);
        m = m_new;
    }

    const float g = gate[h * QWEN4_QSA_HEAD_DIM + d];
    const float sig_g = 1.0f / (1.0f + expf(-g));
    attn_gated[h * QWEN4_QSA_HEAD_DIM + d] = (l > 0.0f ? (acc_d / l) : 0.0f) * sig_g;
}

// ============================================================================
// MoE Router, Shared Expert, & Batched 10-Expert SwiGLU
// ============================================================================

constexpr int QWEN4_N_EXPERTS        = 256;
constexpr int QWEN4_N_EXPERTS_ACTIVE = 10;
constexpr int QWEN4_MOE_FF_DIM       = 640;

__global__ void router_top10_kernel(
    float* __restrict__ logits,
    int32_t* __restrict__ top_idx,
    float* __restrict__ top_w)
{
    __shared__ float probs[QWEN4_N_EXPERTS];
    __shared__ float smem[8];

    const int e = threadIdx.x; // 0..255
    const int warp = e / 32;
    const int lane = e & 31;

    float v = logits[e];
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, mask, 32));
    }
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < 8) ? smem[lane] : -1e30f;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, mask, 32));
        }
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    const float max_l = smem[0];

    const float p = expf(logits[e] - max_l);
    const float sum_p = block_sum<8>(p, smem);
    const float prob_e = p / sum_p;
    probs[e] = prob_e;
    logits[e] = prob_e;
    __syncthreads();

    if (e == 0) {
        float w_sum = 0.0f;
        for (int k = 0; k < QWEN4_N_EXPERTS_ACTIVE; ++k) {
            int best_i = 0;
            float best_p = probs[0];
            for (int i = 1; i < QWEN4_N_EXPERTS; ++i) {
                if (probs[i] > best_p) {
                    best_p = probs[i];
                    best_i = i;
                }
            }
            top_idx[k] = best_i;
            top_w[k] = best_p;
            w_sum += best_p;
            probs[best_i] = -1.0f;
        }
        const float denom = fmaxf(w_sum, 0.00006103515625f); // 2^-14 clamp
        for (int k = 0; k < QWEN4_N_EXPERTS_ACTIVE; ++k) {
            top_w[k] /= denom;
        }
    }
}

// Fused SwiGLU + Q8_1 quantization across `gridDim.x` experts (1 for shared expert, 10 for routed experts)
// Launch: <<<n_experts, 640>>> (20 warps of 32 threads = 640 threads per block)
__global__ void __launch_bounds__(640) swiglu_and_q8_1_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    block_q8_1* __restrict__ out_q8_1)
{
    const int k = blockIdx.x; // 0..n_experts-1
    const int tid = threadIdx.x; // 0..639
    const int warp = tid / 32; // 0..19
    const int lane = tid & 31; // 0..31

    const float g = gate[k * QWEN4_MOE_FF_DIM + tid];
    const float u = up[k * QWEN4_MOE_FF_DIM + tid];
    const float h = (g / (1.0f + expf(-g))) * u;
    const float xi = q8_1_finite(h);

    float amax = fabsf(xi);
    float sum = xi;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask, 32));
        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, mask, 32);
    }

    const float d = amax / 127.0f;
    const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / d);
    block_q8_1* blk = out_q8_1 + k * (QWEN4_MOE_FF_DIM / 32) + warp;
    blk->qs[lane] = q;
    if (lane == 0) {
        blk->ds = make_half2(__float2half(d), __float2half(sum));
    }
}

// Shared expert scalar gate dot + scale into y_moe[2560]
__global__ void shared_expert_gate_scale_kernel(
    const float* __restrict__ x_f32,
    const uint16_t* __restrict__ gate_inp_shexp_bf16,
    const float* __restrict__ down_shexp,
    float* __restrict__ y_moe)
{
    __shared__ float smem[QWEN4_GR_WARPS];
    float dot = 0.0f;
    for (int j = threadIdx.x; j < QWEN4_N_EMBD; j += blockDim.x) {
        dot += x_f32[j] * qwen4_bf16_to_fp32(gate_inp_shexp_bf16[j]);
    }
    dot = block_sum<QWEN4_GR_WARPS>(dot, smem);
    const float g = 1.0f / (1.0f + expf(-dot));

    for (int j = threadIdx.x; j < QWEN4_N_EMBD; j += blockDim.x) {
        y_moe[j] = g * down_shexp[j];
    }
}

// Weighted accumulation of the routed experts' down outputs (plus optional CPU MoE output) into y_moe[2560]
__global__ void accum_routed10_kernel(
    const float* __restrict__ down_all,
    const float* __restrict__ d_top_w,
    const int32_t* __restrict__ d_slot,
    int n_gpu_experts,
    const float* __restrict__ d_cpu_moe_out,
    float* __restrict__ y_moe)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= QWEN4_N_EMBD) return;

    float acc = (d_cpu_moe_out != nullptr) ? d_cpu_moe_out[j] : 0.0f;
    for (int k = 0; k < n_gpu_experts; ++k) {
        const int orig_k = (d_slot != nullptr) ? d_slot[k] : k;
        acc += d_top_w[orig_k] * down_all[k * QWEN4_N_EMBD + j];
    }
    y_moe[j] += acc;
}

} // anonymous namespace

// ============================================================================
// Public Host API Implementations
// ============================================================================

size_t qwen4_ggml_row_bytes(int32_t ggml_type, int64_t n_cols) {
    switch (ggml_type) {
        case QWEN4_GGML_TYPE_F32:     return (size_t)n_cols * 4;
        case QWEN4_GGML_TYPE_F16:     return (size_t)n_cols * 2;
        case QWEN4_GGML_TYPE_BF16:    return (size_t)n_cols * 2;
        case QWEN4_GGML_TYPE_Q4_0:    return (size_t)(n_cols / QK4_0) * sizeof(block_q4_0);
        case QWEN4_GGML_TYPE_Q4_1:    return (size_t)(n_cols / QK4_1) * sizeof(block_q4_1);
        case QWEN4_GGML_TYPE_Q5_0:    return (size_t)(n_cols / QK5_0) * sizeof(block_q5_0);
        case QWEN4_GGML_TYPE_Q5_1:    return (size_t)(n_cols / QK5_1) * sizeof(block_q5_1);
        case QWEN4_GGML_TYPE_Q8_0:    return (size_t)(n_cols / QK8_0) * sizeof(block_q8_0);
        case QWEN4_GGML_TYPE_Q2_K:    return (size_t)(n_cols / QK_K)  * sizeof(block_q2_K);
        case QWEN4_GGML_TYPE_Q3_K:    return (size_t)(n_cols / QK_K)  * sizeof(block_q3_K);
        case QWEN4_GGML_TYPE_Q4_K:    return (size_t)(n_cols / QK_K)  * sizeof(block_q4_K);
        case QWEN4_GGML_TYPE_Q5_K:    return (size_t)(n_cols / QK_K)  * sizeof(block_q5_K);
        case QWEN4_GGML_TYPE_Q6_K:    return (size_t)(n_cols / QK_K)  * sizeof(block_q6_K);
        case QWEN4_GGML_TYPE_IQ2_XXS: return (size_t)(n_cols / QK_K)  * sizeof(block_iq2_xxs);
        case QWEN4_GGML_TYPE_IQ2_XS:  return (size_t)(n_cols / QK_K)  * sizeof(block_iq2_xs);
        case QWEN4_GGML_TYPE_IQ3_XXS: return (size_t)(n_cols / QK_K)  * sizeof(block_iq3_xxs);
        case QWEN4_GGML_TYPE_IQ1_S:   return (size_t)(n_cols / QK_K)  * sizeof(block_iq1_s);
        case QWEN4_GGML_TYPE_IQ4_NL:  return (size_t)(n_cols / QK4_NL)* sizeof(block_iq4_nl);
        case QWEN4_GGML_TYPE_IQ3_S:   return (size_t)(n_cols / QK_K)  * sizeof(block_iq3_s);
        case QWEN4_GGML_TYPE_IQ2_S:   return (size_t)(n_cols / QK_K)  * sizeof(block_iq2_s);
        case QWEN4_GGML_TYPE_IQ4_XS:  return (size_t)(n_cols / QK_K)  * sizeof(block_iq4_xs);
        case QWEN4_GGML_TYPE_IQ1_M:   return (size_t)(n_cols / QK_K)  * sizeof(block_iq1_m);
        case QWEN4_GGML_TYPE_Q1_0:    return (size_t)(n_cols / QK1_0) * sizeof(block_q1_0);
        case QWEN4_GGML_TYPE_Q2_0:    return (size_t)(n_cols / QK2_0) * sizeof(block_q2_0);
        default: return 0;
    }
}

const char* qwen4_ggml_type_name(int32_t ggml_type) {
    switch (ggml_type) {
        case QWEN4_GGML_TYPE_F32:     return "F32";
        case QWEN4_GGML_TYPE_F16:     return "F16";
        case QWEN4_GGML_TYPE_BF16:    return "BF16";
        case QWEN4_GGML_TYPE_Q4_0:    return "Q4_0";
        case QWEN4_GGML_TYPE_Q4_1:    return "Q4_1";
        case QWEN4_GGML_TYPE_Q5_0:    return "Q5_0";
        case QWEN4_GGML_TYPE_Q5_1:    return "Q5_1";
        case QWEN4_GGML_TYPE_Q8_0:    return "Q8_0";
        case QWEN4_GGML_TYPE_Q2_K:    return "Q2_K";
        case QWEN4_GGML_TYPE_Q3_K:    return "Q3_K";
        case QWEN4_GGML_TYPE_Q4_K:    return "Q4_K";
        case QWEN4_GGML_TYPE_Q5_K:    return "Q5_K";
        case QWEN4_GGML_TYPE_Q6_K:    return "Q6_K";
        case QWEN4_GGML_TYPE_IQ2_XXS: return "IQ2_XXS";
        case QWEN4_GGML_TYPE_IQ2_XS:  return "IQ2_XS";
        case QWEN4_GGML_TYPE_IQ3_XXS: return "IQ3_XXS";
        case QWEN4_GGML_TYPE_IQ1_S:   return "IQ1_S";
        case QWEN4_GGML_TYPE_IQ4_NL:  return "IQ4_NL";
        case QWEN4_GGML_TYPE_IQ3_S:   return "IQ3_S";
        case QWEN4_GGML_TYPE_IQ2_S:   return "IQ2_S";
        case QWEN4_GGML_TYPE_IQ4_XS:  return "IQ4_XS";
        case QWEN4_GGML_TYPE_IQ1_M:   return "IQ1_M";
        case QWEN4_GGML_TYPE_Q1_0:    return "Q1_0";
        case QWEN4_GGML_TYPE_Q2_0:    return "Q2_0";
        default: return "UNKNOWN";
    }
}

size_t qwen4_q8_1_bytes(int64_t n_cols) {
    return (size_t)(n_cols / QK8_1) * sizeof(block_q8_1);
}

void qwen4_quantize_q8_1_cuda(const float* x, void* vx_q8_1, int n_cols, cudaStream_t stream) {
    const int n_blocks = n_cols / QK8_1;
    const dim3 block(32, 4);
    const dim3 grid((n_blocks + 3) / 4);
    quantize_q8_1_kernel<<<grid, block, 0, stream>>>(x, (block_q8_1*)vx_q8_1, n_blocks);
}

void qwen4_mmvq_cuda(
    int32_t ggml_type,
    const void* W,
    const void* vx_q8_1,
    float* y,
    int n_rows,
    int n_cols,
    cudaStream_t stream)
{
    const size_t row_bytes = qwen4_ggml_row_bytes(ggml_type, n_cols);
    const dim3 block(32, 4);
    const dim3 grid((n_rows + 3) / 4);
    const uint8_t* w_ptr = (const uint8_t*)W;
    const block_q8_1* x_ptr = (const block_q8_1*)vx_q8_1;

    switch (ggml_type) {
        case 2:  mmvq_kernel<2> <<<grid, block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows); break;
        case 3:  mmvq_kernel<3> <<<grid, block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows); break;
        case 6:  mmvq_kernel<6> <<<grid, block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows); break;
        case 7:  mmvq_kernel<7> <<<grid, block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows); break;
        case 8:  mmvq_kernel<8> <<<grid, block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows); break;
        case 12: mmvq_kernel<12><<<grid, block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows); break;
        case 13: mmvq_kernel<13><<<grid, block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows); break;
        case 14: mmvq_kernel<14><<<grid, block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows); break;
        case 16: mmvq_kernel<16><<<grid, block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows); break;
        case 17: mmvq_kernel<17><<<grid, block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows); break;
        case 18: mmvq_kernel<18><<<grid, block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows); break;
        case 20: mmvq_kernel<20><<<grid, block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows); break;
        case 21: mmvq_kernel<21><<<grid, block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows); break;
        case 22: mmvq_kernel<22><<<grid, block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows); break;
        case 23: mmvq_kernel<23><<<grid, block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows); break;
        case 29: mmvq_kernel<29><<<grid, block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows); break;
        case 42: mmvq_kernel<42><<<grid, block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows); break;
        default:
            std::fprintf(stderr, "[Qwen4] Unsupported MMVQ ggml_type=%d\n", ggml_type);
            break;
    }
}

void qwen4_bf16_gemv_cuda(
    const uint16_t* W_bf16,
    const float* x,
    float* y,
    int n_rows,
    int n_cols,
    cudaStream_t stream)
{
    const dim3 block(32, 4);
    const dim3 grid((n_rows + 3) / 4);
    bf16_gemv_kernel<<<grid, block, 0, stream>>>(W_bf16, x, y, n_rows, n_cols);
}

void qwen4_project_cuda(
    int32_t ggml_type,
    const void* W,
    const float* x_f32,
    const void* x_q8_1,
    float* y,
    int n_rows,
    int n_cols,
    cudaStream_t stream)
{
    if (ggml_type == QWEN4_GGML_TYPE_BF16) {
        qwen4_bf16_gemv_cuda((const uint16_t*)W, x_f32, y, n_rows, n_cols, stream);
    } else if (ggml_type == QWEN4_GGML_TYPE_F16) {
        const dim3 block(32, 4);
        const dim3 grid((n_rows + 3) / 4);
        f16_gemv_kernel<<<grid, block, 0, stream>>>((const half*)W, x_f32, y, n_rows, n_cols);
    } else if (ggml_type == QWEN4_GGML_TYPE_F32) {
        const dim3 block(32, 4);
        const dim3 grid((n_rows + 3) / 4);
        f32_gemv_kernel<<<grid, block, 0, stream>>>((const float*)W, x_f32, y, n_rows, n_cols);
    } else {
        qwen4_mmvq_cuda(ggml_type, W, x_q8_1, y, n_rows, n_cols, stream);
    }
}

void qwen4_embed_to_hc_cuda(
    int32_t embd_type,
    const void* token_embd,
    int token_id,
    float* d_R,
    int n_embd,
    int n_streams,
    cudaStream_t stream)
{
    const size_t row_bytes = qwen4_ggml_row_bytes(embd_type, n_embd);
    const uint8_t* row_ptr = (const uint8_t*)token_embd + (size_t)token_id * row_bytes;
    embed_to_hc_kernel<<<(n_embd + 255) / 256, 256, 0, stream>>>(embd_type, row_ptr, d_R, n_embd, n_streams);
}

void qwen4_vec_to_hc_cuda(
    const float* d_embd,
    float* d_R,
    int n_embd,
    int n_streams,
    cudaStream_t stream)
{
    vec_to_hc_kernel<<<(n_embd + 255) / 256, 256, 0, stream>>>(d_embd, d_R, n_embd, n_streams);
}

void qwen4_gr_read_cuda(
    const float* R,
    const float* norm_w,
    const uint16_t* down_bf16,
    const uint16_t* up_bf16,
    float* x_out,
    float* tmp_buf,
    float eps,
    cudaStream_t stream)
{
    float* normed = tmp_buf;
    float* h = tmp_buf + QWEN4_STREAM_TOTAL;

    gr_stream_rmsnorm_bf16_kernel<<<QWEN4_N_STREAMS, 256, 0, stream>>>(R, norm_w, normed, eps);
    gr_down_gemv_kernel<<<QWEN4_HC_RANK, 256, 0, stream>>>(down_bf16, normed, h);
    const dim3 block_up(32, QWEN4_N_STREAMS);
    gr_up_and_mean_kernel<<<QWEN4_N_EMBD, block_up, 0, stream>>>(up_bf16, h, normed, x_out);
}

void qwen4_gr_write_cuda(
    float* R,
    const float* norm_w,
    const uint16_t* inject_bf16,
    const float* sub_out,
    float* tmp_buf,
    float eps,
    cudaStream_t stream)
{
    float* normed = tmp_buf;
    gr_stream_rmsnorm_bf16_kernel<<<QWEN4_N_STREAMS, 256, 0, stream>>>(R, norm_w, normed, eps);
    gr_inject_and_write_kernel<<<QWEN4_N_STREAMS, 256, 0, stream>>>(inject_bf16, normed, sub_out, R);
}

void qwen4_ple_iq4nl_dequant_cuda(
    const void* src16_iq4nl,
    float* dst_f32,
    cudaStream_t stream)
{
    ple_iq4nl_dequant_kernel<<<QWEN4_N_EMBD / 32, 16, 0, stream>>>(
        (const block_iq4_nl*)src16_iq4nl, dst_f32);
}

void qwen4_ple_forward_cuda(
    float* R,
    const float* ple_norm_q,
    const float* ple_norm_k,
    const float* ple_norm_c,
    const void* ple_key,
    int32_t ple_key_type,
    const void* ple_val,
    int32_t ple_val_type,
    const uint16_t* ple_conv_f16,
    const float* emb_flat,
    float* conv_state,
    float* scratch,
    void* q8_scratch,
    float eps,
    cudaStream_t stream)
{
    float* key_out = scratch;                                   // [10240]
    float* val_out = key_out + QWEN4_STREAM_TOTAL;              // [2560]
    float* gated   = val_out + QWEN4_N_EMBD;                    // [10240]
    float* normd   = gated   + QWEN4_STREAM_TOTAL;              // [10240]

    if ((ple_key_type != QWEN4_GGML_TYPE_BF16 && ple_key_type != QWEN4_GGML_TYPE_F16 && ple_key_type != QWEN4_GGML_TYPE_F32) ||
        (ple_val_type != QWEN4_GGML_TYPE_BF16 && ple_val_type != QWEN4_GGML_TYPE_F16 && ple_val_type != QWEN4_GGML_TYPE_F32)) {
        qwen4_quantize_q8_1_cuda(emb_flat, q8_scratch, QWEN4_N_EMBD, stream);
    }

    qwen4_project_cuda(ple_key_type, ple_key, emb_flat, q8_scratch, key_out, QWEN4_STREAM_TOTAL, QWEN4_N_EMBD, stream);
    qwen4_project_cuda(ple_val_type, ple_val, emb_flat, q8_scratch, val_out, QWEN4_N_EMBD, QWEN4_N_EMBD, stream);

    ple_gate_kernel<<<QWEN4_N_STREAMS, 256, 0, stream>>>(
        R, key_out, val_out, ple_norm_k, ple_norm_q, ple_norm_c, gated, normd, eps);

    ple_conv_update_kernel<<<(QWEN4_STREAM_TOTAL + 255) / 256, 256, 0, stream>>>(
        R, gated, normd, (const half*)ple_conv_f16, conv_state);
}

void qwen4_gdn_decode_cuda(
    const float* x_f32,
    const void* x_q8_1,
    const void* w_qkv, int32_t qkv_type,
    const void* w_gate, int32_t gate_type,
    const uint16_t* w_alpha_bf16,
    const uint16_t* w_beta_bf16,
    const float* w_conv_f32,
    const float* ssm_a_f32,
    const float* dt_bias_f32,
    const float* ssm_norm_f32,
    const void* w_out, int32_t out_type,
    float* conv_state,
    float* ssm_state,
    float* y_out,
    float* scratch,
    void* q8_scratch,
    float eps,
    cudaStream_t stream)
{
    float* qkv      = scratch;                                  // [10240]
    float* qkv_conv = qkv + QWEN4_GDN_QKV_DIM;                  // [10240]
    float* z        = qkv_conv + QWEN4_GDN_QKV_DIM;             // [6144]
    float* raw_out  = z + QWEN4_GDN_V_DIM;                      // [6144]
    float* gate     = raw_out + QWEN4_GDN_V_DIM;                // [48]
    float* beta     = gate + QWEN4_GDN_N_V_HEAD;                // [48]

    qwen4_project_cuda(qkv_type, w_qkv, x_f32, x_q8_1, qkv, QWEN4_GDN_QKV_DIM, QWEN4_N_EMBD, stream);
    qwen4_project_cuda(gate_type, w_gate, x_f32, x_q8_1, z, QWEN4_GDN_V_DIM, QWEN4_N_EMBD, stream);

    const dim3 block_ab(32, 4);
    const dim3 grid_ab((QWEN4_GDN_N_V_HEAD + 3) / 4);
    gdn_alpha_beta_gemv_kernel<<<grid_ab, block_ab, 0, stream>>>(
        w_alpha_bf16, w_beta_bf16, x_f32, ssm_a_f32, dt_bias_f32, gate, beta);

    gdn_conv_silu_kernel<<<QWEN4_GDN_QKV_DIM / 256, 256, 0, stream>>>(conv_state, qkv, w_conv_f32, qkv_conv);
    gdn_qk_l2norm_kernel<<<2 * QWEN4_GDN_N_K_HEAD, QWEN4_GDN_HEAD_DIM, 0, stream>>>(qkv_conv, eps);
    gdn_recurrence_kernel<<<QWEN4_GDN_N_V_HEAD, QWEN4_GDN_HEAD_DIM, 0, stream>>>(ssm_state, qkv_conv, gate, beta, raw_out);
    gdn_norm_gate_kernel<<<QWEN4_GDN_N_V_HEAD, QWEN4_GDN_HEAD_DIM, 0, stream>>>(raw_out, ssm_norm_f32, z, z, eps);

    if (out_type != QWEN4_GGML_TYPE_BF16 && out_type != QWEN4_GGML_TYPE_F16 && out_type != QWEN4_GGML_TYPE_F32) {
        qwen4_quantize_q8_1_cuda(z, q8_scratch, QWEN4_GDN_V_DIM, stream);
    }
    qwen4_project_cuda(out_type, w_out, z, q8_scratch, y_out, QWEN4_N_EMBD, QWEN4_GDN_V_DIM, stream);
}

void qwen4_qsa_decode_cuda(
    const float* x_f32,
    const void* x_q8_1,
    const void* w_q, int32_t q_type,
    const void* w_k, int32_t k_type,
    const void* w_v, int32_t v_type,
    const float* q_norm_f32,
    const float* k_norm_f32,
    const void* w_out, int32_t out_type,
    const uint16_t* w_idx_k_bf16,
    const uint16_t* w_idx_q_bf16,
    const float* idx_k_norm_f32,
    const float* idx_q_norm_f32,
    uint16_t* k_cache,
    uint16_t* v_cache,
    float* idx_pooled,
    float* idx_tail,
    float* idx_dead,
    int pos,
    int max_seq_len,
    float* y_out,
    float* scratch,
    int32_t* d_sel_indices,
    void* q8_scratch,
    float eps,
    cudaStream_t stream)
{
    (void)max_seq_len;
    constexpr int Q_GATE_DIM = QWEN4_QSA_N_HEAD * 2 * QWEN4_QSA_HEAD_DIM; // 12288
    constexpr int Q_DIM      = QWEN4_QSA_N_HEAD * QWEN4_QSA_HEAD_DIM;     // 6144
    constexpr int KV_DIM     = QWEN4_QSA_N_KV * QWEN4_QSA_HEAD_DIM;       // 512
    constexpr int IDX_Q_DIM  = QWEN4_QSA_IDX_HEADS * QWEN4_QSA_IDX_DIM;   // 512

    float* q_gate       = scratch;                          // [12288]
    float* q            = q_gate + Q_GATE_DIM;              // [6144]
    float* gate         = q + Q_DIM;                        // [6144]
    float* k_cur        = gate + Q_DIM;                     // [512]
    float* v_cur        = k_cur + KV_DIM;                   // [512]
    float* attn_gated   = v_cur + KV_DIM;                   // [6144]
    float* q_idx        = attn_gated + Q_DIM;               // [512]
    float* idx_raw      = q_idx + IDX_Q_DIM;                // [128]
    float* block_scores = idx_raw + QWEN4_QSA_IDX_DIM;      // [max_blocks]

    qwen4_project_cuda(q_type, w_q, x_f32, x_q8_1, q_gate, Q_GATE_DIM, QWEN4_N_EMBD, stream);
    qwen4_project_cuda(k_type, w_k, x_f32, x_q8_1, k_cur,  KV_DIM,     QWEN4_N_EMBD, stream);
    qwen4_project_cuda(v_type, w_v, x_f32, x_q8_1, v_cur,  KV_DIM,     QWEN4_N_EMBD, stream);

    qsa_prep_q_kernel<<<QWEN4_QSA_N_HEAD, QWEN4_QSA_HEAD_DIM, 0, stream>>>(q_gate, q_norm_f32, q, gate, pos, eps);
    qsa_prep_kv_kernel<<<QWEN4_QSA_N_KV, QWEN4_QSA_HEAD_DIM, 0, stream>>>(k_cur, v_cur, k_norm_f32, k_cache, v_cache, pos, eps);

    // Update indexer keys for current token
    qwen4_bf16_gemv_cuda(w_idx_k_bf16, x_f32, idx_raw, QWEN4_QSA_IDX_DIM, QWEN4_N_EMBD, stream);
    qsa_indexer_append_kernel<<<1, QWEN4_QSA_IDX_DIM, 0, stream>>>(
        idx_raw, idx_k_norm_f32, idx_pooled, idx_tail, idx_dead, pos, eps);

    const int n_kv = pos + 1;
    if (n_kv <= QWEN4_QSA_MAX_SEL) {
        // Exact full causal attention over 0..pos (all cells fit in budget 2051)
        qsa_attend_gate_kernel<<<QWEN4_QSA_N_HEAD, QWEN4_QSA_HEAD_DIM, 0, stream>>>(
            q, gate, k_cache, v_cache, nullptr, n_kv, attn_gated);
    } else {
        // Sparse top-2051 block selection on GPU
        const int n_blocks = (n_kv + QWEN4_QSA_IDX_BLOCK - 1) / QWEN4_QSA_IDX_BLOCK;
        const int n_sel = QWEN4_QSA_MAX_SEL;
        qwen4_bf16_gemv_cuda(w_idx_q_bf16, x_f32, q_idx, IDX_Q_DIM, QWEN4_N_EMBD, stream);
        qsa_indexer_prep_q_kernel<<<QWEN4_QSA_IDX_HEADS, QWEN4_QSA_IDX_DIM, 0, stream>>>(q_idx, idx_q_norm_f32, pos, eps);
        qsa_indexer_score_blocks_kernel<<<n_blocks, QWEN4_QSA_IDX_DIM, 0, stream>>>(
            idx_pooled, q_idx, block_scores, n_kv, n_blocks);
        qsa_indexer_select_topk_kernel<<<1, 256, 0, stream>>>(
            block_scores, d_sel_indices, n_kv, n_blocks, n_sel);
        qsa_attend_gate_kernel<<<QWEN4_QSA_N_HEAD, QWEN4_QSA_HEAD_DIM, 0, stream>>>(
            q, gate, k_cache, v_cache, d_sel_indices, n_sel, attn_gated);
    }

    if (out_type != QWEN4_GGML_TYPE_BF16 && out_type != QWEN4_GGML_TYPE_F16 && out_type != QWEN4_GGML_TYPE_F32) {
        qwen4_quantize_q8_1_cuda(attn_gated, q8_scratch, Q_DIM, stream);
    }
    qwen4_project_cuda(out_type, w_out, attn_gated, q8_scratch, y_out, QWEN4_N_EMBD, Q_DIM, stream);
}

void qwen4_router_top10_cuda(
    const uint16_t* gate_inp_bf16,
    const float* x_f32,
    float* scratch_logits,
    int32_t* d_top_idx,
    float* d_top_w,
    cudaStream_t stream)
{
    qwen4_bf16_gemv_cuda(gate_inp_bf16, x_f32, scratch_logits, QWEN4_N_EXPERTS, QWEN4_N_EMBD, stream);
    router_top10_kernel<<<1, QWEN4_N_EXPERTS, 0, stream>>>(scratch_logits, d_top_idx, d_top_w);
}

void qwen4_shared_expert_cuda(
    const float* x_f32,
    const void* x_q8_1,
    const void* w_gate_shexp, int32_t gate_shexp_type,
    const void* w_up_shexp, int32_t up_shexp_type,
    const void* w_down_shexp, int32_t down_shexp_type,
    const uint16_t* w_gate_inp_shexp_bf16,
    float* y_moe,
    float* scratch,
    void* q8_scratch,
    cudaStream_t stream)
{
    float* gate_sh = scratch;                           // [640]
    float* up_sh   = gate_sh + QWEN4_MOE_FF_DIM;        // [640]
    float* down_sh = up_sh + QWEN4_MOE_FF_DIM;          // [2560]

    qwen4_mmvq_cuda(gate_shexp_type, w_gate_shexp, x_q8_1, gate_sh, QWEN4_MOE_FF_DIM, QWEN4_N_EMBD, stream);
    qwen4_mmvq_cuda(up_shexp_type,   w_up_shexp,   x_q8_1, up_sh,   QWEN4_MOE_FF_DIM, QWEN4_N_EMBD, stream);

    swiglu_and_q8_1_kernel<<<1, QWEN4_MOE_FF_DIM, 0, stream>>>(gate_sh, up_sh, (block_q8_1*)q8_scratch);
    qwen4_mmvq_cuda(down_shexp_type, w_down_shexp, q8_scratch, down_sh, QWEN4_N_EMBD, QWEN4_MOE_FF_DIM, stream);

    shared_expert_gate_scale_kernel<<<1, 256, 0, stream>>>(x_f32, w_gate_inp_shexp_bf16, down_sh, y_moe);
}

namespace {

void launch_mmvq_batch10(
    int32_t ggml_type,
    const void* base_w,
    size_t expert_stride_bytes,
    const void* const* d_expert_ptrs,
    const int32_t* d_top_idx,
    const void* vx_q8_1,
    size_t x_expert_stride_blocks,
    float* y,
    int n_rows,
    int n_cols,
    int n_experts,
    cudaStream_t stream)
{
    if (n_experts <= 0) return;
    const size_t row_bytes = qwen4_ggml_row_bytes(ggml_type, n_cols);
    const dim3 block(32, 4);
    const dim3 grid((n_rows + 3) / 4, n_experts);
    const uint8_t* bw = (const uint8_t*)base_w;
    const block_q8_1* x_ptr = (const block_q8_1*)vx_q8_1;

    switch (ggml_type) {
        case 2:  mmvq_batch10_kernel<2> <<<grid, block, 0, stream>>>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows); break;
        case 3:  mmvq_batch10_kernel<3> <<<grid, block, 0, stream>>>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows); break;
        case 6:  mmvq_batch10_kernel<6> <<<grid, block, 0, stream>>>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows); break;
        case 7:  mmvq_batch10_kernel<7> <<<grid, block, 0, stream>>>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows); break;
        case 8:  mmvq_batch10_kernel<8> <<<grid, block, 0, stream>>>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows); break;
        case 12: mmvq_batch10_kernel<12><<<grid, block, 0, stream>>>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows); break;
        case 13: mmvq_batch10_kernel<13><<<grid, block, 0, stream>>>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows); break;
        case 14: mmvq_batch10_kernel<14><<<grid, block, 0, stream>>>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows); break;
        case 16: mmvq_batch10_kernel<16><<<grid, block, 0, stream>>>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows); break;
        case 17: mmvq_batch10_kernel<17><<<grid, block, 0, stream>>>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows); break;
        case 18: mmvq_batch10_kernel<18><<<grid, block, 0, stream>>>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows); break;
        case 20: mmvq_batch10_kernel<20><<<grid, block, 0, stream>>>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows); break;
        case 21: mmvq_batch10_kernel<21><<<grid, block, 0, stream>>>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows); break;
        case 22: mmvq_batch10_kernel<22><<<grid, block, 0, stream>>>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows); break;
        case 23: mmvq_batch10_kernel<23><<<grid, block, 0, stream>>>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows); break;
        case 29: mmvq_batch10_kernel<29><<<grid, block, 0, stream>>>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows); break;
        case 42: mmvq_batch10_kernel<42><<<grid, block, 0, stream>>>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows); break;
        default:
            std::fprintf(stderr, "[Qwen4] Unsupported batch10 MMVQ ggml_type=%d\n", ggml_type);
            break;
    }
}

} // anonymous namespace

void qwen4_routed_experts_range_cuda(
    const void* x_q8_1,
    const void* base_gate, size_t gate_stride_bytes, int32_t gate_type,
    const void* base_up, size_t up_stride_bytes, int32_t up_type,
    const void* base_down, size_t down_stride_bytes, int32_t down_type,
    const Qwen4ActiveExpertPtrs* d_ptrs,
    const int32_t* d_top_idx,
    const float* d_top_w,
    int start_k,
    int count_k,
    bool do_accum,
    float* y_moe,
    float* scratch,
    void* q8_scratch,
    cudaStream_t stream,
    const float* d_cpu_moe_out)
{
    float* gate_all = scratch;                                                  // [10, 640]
    float* up_all   = gate_all + QWEN4_N_EXPERTS_ACTIVE * QWEN4_MOE_FF_DIM;     // [10, 640]
    float* down_all = up_all   + QWEN4_N_EXPERTS_ACTIVE * QWEN4_MOE_FF_DIM;     // [10, 2560]

    if (count_k > 0) {
        const void* const* gate_ptrs = (d_ptrs != nullptr) ? (d_ptrs->gate + start_k) : nullptr;
        const void* const* up_ptrs   = (d_ptrs != nullptr) ? (d_ptrs->up   + start_k) : nullptr;
        const void* const* down_ptrs = (d_ptrs != nullptr) ? (d_ptrs->down + start_k) : nullptr;
        const int32_t* top_idx_range = (d_top_idx != nullptr) ? (d_top_idx + start_k) : nullptr;

        float* gate_range = gate_all + (size_t)start_k * QWEN4_MOE_FF_DIM;
        float* up_range   = up_all   + (size_t)start_k * QWEN4_MOE_FF_DIM;
        float* down_range = down_all + (size_t)start_k * QWEN4_N_EMBD;
        block_q8_1* q8_range = (block_q8_1*)q8_scratch + (size_t)start_k * (QWEN4_MOE_FF_DIM / QK8_1);

        launch_mmvq_batch10(
            gate_type, base_gate, gate_stride_bytes, gate_ptrs, top_idx_range,
            x_q8_1, 0, gate_range, QWEN4_MOE_FF_DIM, QWEN4_N_EMBD, count_k, stream);

        launch_mmvq_batch10(
            up_type, base_up, up_stride_bytes, up_ptrs, top_idx_range,
            x_q8_1, 0, up_range, QWEN4_MOE_FF_DIM, QWEN4_N_EMBD, count_k, stream);

        swiglu_and_q8_1_kernel<<<count_k, QWEN4_MOE_FF_DIM, 0, stream>>>(
            gate_range, up_range, q8_range);

        launch_mmvq_batch10(
            down_type, base_down, down_stride_bytes, down_ptrs, top_idx_range,
            q8_range, QWEN4_MOE_FF_DIM / QK8_1, down_range, QWEN4_N_EMBD, QWEN4_MOE_FF_DIM, count_k, stream);
    }

    if (do_accum) {
        const int32_t* d_slot = (d_ptrs != nullptr) ? d_ptrs->slot : nullptr;
        const int n_gpu_total = start_k + count_k;
        accum_routed10_kernel<<<QWEN4_N_EMBD / 256, 256, 0, stream>>>(
            down_all, d_top_w, d_slot, n_gpu_total, d_cpu_moe_out, y_moe);
    }
}

void qwen4_routed_experts_batch10_cuda(
    const void* x_q8_1,
    const void* base_gate, size_t gate_stride_bytes, int32_t gate_type,
    const void* base_up, size_t up_stride_bytes, int32_t up_type,
    const void* base_down, size_t down_stride_bytes, int32_t down_type,
    const Qwen4ActiveExpertPtrs* d_ptrs,
    const int32_t* d_top_idx,
    const float* d_top_w,
    float* y_moe,
    float* scratch,
    void* q8_scratch,
    cudaStream_t stream)
{
    qwen4_routed_experts_range_cuda(
        x_q8_1,
        base_gate, gate_stride_bytes, gate_type,
        base_up, up_stride_bytes, up_type,
        base_down, down_stride_bytes, down_type,
        d_ptrs, d_top_idx, d_top_w,
        0, QWEN4_N_EXPERTS_ACTIVE, true,
        y_moe, scratch, q8_scratch, stream);
}
