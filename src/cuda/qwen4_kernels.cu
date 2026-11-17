#include "qwen4_kernels.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

namespace {

static cublasHandle_t g_qwen4_cublas[2]        = {nullptr, nullptr};
static uint16_t*      g_qwen4_bf16_ws[2]       = {nullptr, nullptr};
static size_t         g_qwen4_bf16_ws_bytes[2] = {0, 0};

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

__device__ __forceinline__ float dot8_bf16_u4(const __nv_bfloat162 q_bf2[4], const uint4 k_u4) {
    const __nv_bfloat162 k0 = *reinterpret_cast<const __nv_bfloat162*>(&k_u4.x);
    const __nv_bfloat162 k1 = *reinterpret_cast<const __nv_bfloat162*>(&k_u4.y);
    const __nv_bfloat162 k2 = *reinterpret_cast<const __nv_bfloat162*>(&k_u4.z);
    const __nv_bfloat162 k3 = *reinterpret_cast<const __nv_bfloat162*>(&k_u4.w);
    __nv_bfloat162 p0 = __hmul2(q_bf2[0], k0);
    __nv_bfloat162 p1 = __hmul2(q_bf2[1], k1);
    p0 = __hfma2(q_bf2[2], k2, p0);
    p1 = __hfma2(q_bf2[3], k3, p1);
    const float2 f0 = __bfloat1622float2(__hadd2(p0, p1));
    return f0.x + f0.y;
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

__device__ __forceinline__ int kvalues_iq4nl_lut(int nib) {
    constexpr uint32_t t0 = 0xBFAD9881u;
    constexpr uint32_t t1 = 0xF6EADDCFu;
    constexpr uint32_t t2 = 0x26190D01u;
    constexpr uint32_t t3 = 0x71594535u;
    const uint32_t w = (nib < 8) ? ((nib < 4) ? t0 : t1) : ((nib < 12) ? t2 : t3);
    return (int)(int8_t)(w >> ((nib & 3) * 8));
}

__device__ __forceinline__ int qwen4_get_u8(int v, int b) {
    return ((uint32_t)v >> (8 * b)) & 0xFFu;
}
__device__ __forceinline__ int qwen4_get_u8_2(int2 v, int b) {
    return ((uint32_t)((b < 4) ? v.x : v.y) >> (8 * (b & 3))) & 0xFFu;
}
__device__ __forceinline__ int qwen4_get_u16_2(int2 v, int h) {
    return ((uint32_t)((h < 2) ? v.x : v.y) >> (16 * (h & 1))) & 0xFFFFu;
}

__device__ __forceinline__ int2 get_int_from_table_16(const int& q4, const int8_t* table) {
    (void)table;
    constexpr uint32_t t0 = 0xBFAD9881u;
    constexpr uint32_t t1 = 0xF6EADDCFu;
    constexpr uint32_t t2 = 0x26190D01u;
    constexpr uint32_t t3 = 0x71594535u;
    const uint32_t q4u = (uint32_t)q4;
    const uint32_t sel = (0x32103210u | ((q4u & 0x88888888u) >> 1));
    const uint32_t low0  = __byte_perm(t0, t1, q4u);
    const uint32_t high0 = __byte_perm(t2, t3, q4u);
    const uint32_t tmp0  = __byte_perm(low0, high0, sel);
    const uint32_t q4h   = q4u >> 16;
    const uint32_t low1  = __byte_perm(t0, t1, q4h);
    const uint32_t high1 = __byte_perm(t2, t3, q4h);
    const uint32_t tmp1  = __byte_perm(low1, high1, sel >> 16);
    return make_int2(__byte_perm(tmp0, tmp1, 0x6420), __byte_perm(tmp0, tmp1, 0x7531));
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
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
    #pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = __ldg(((const uint2*)iq2xxs_grid) + qwen4_get_u8(q2, k0 / 2));
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
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int q2_val = qwen4_get_u16_2(q2_packed, l0 / 2);
        const uint2 grid_pos = __ldg(((const uint2*)iq2xs_grid) + (q2_val & 0x1FF));
        const uint32_t signs = unpack_ksigns(q2_val >> 9);
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
    const int iqs2 = iqs >> 1;
    const int qs_packed = get_int_b2(bq2->qs, iqs2);
    const int qh = bq2->qh[iqs2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs2);
    const int sc0 = 2 * (bq2->scales[iqs2] & 0x0F) + 1;
    const int sc1 = 2 * (bq2->scales[iqs2] >> 4) + 1;
    int sumi0 = 0, sumi1 = 0;
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qs_b = qwen4_get_u8(qs_packed, l0 / 2);
        const int sp_b = qwen4_get_u8(signs_packed_32, l0 / 2);
        const uint2 grid_pos = __ldg((const uint2*)(iq2s_grid + (qs_b | ((qh << (8 - l0)) & 0x300))));
        const int signs0 = __vcmpne4(((sp_b & 0x03) << 7) | ((sp_b & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((sp_b & 0x30) << 3) | ((sp_b & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4((int)grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4((int)grid_pos.y ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const float d = 0.125f * __half2float(bq2->d) * __low2float(bq8_1[iqs2].ds);
    return d * (float)(sumi0 * sc0 + sumi1 * sc1);
}

__device__ __forceinline__ float vec_dot_iq3_xxs_q8_1(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_1,
    const int& kbx,
    const int& iqs)
{
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*)vbq + kbx;
    const int iqs2 = iqs >> 1;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs2);
    int sumi = 0;
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            __ldg(iq3xxs_grid + qwen4_get_u8_2(q3_packed, l0 + 0)),
            __ldg(iq3xxs_grid + qwen4_get_u8_2(q3_packed, l0 + 1)));
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201u, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010u, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    const int ls = aux32 >> 28;
    const float d = 0.25f * __half2float(bq3->d) * __low2float(bq8_1[iqs2].ds) * (float)(2 * ls + 1);
    return d * (float)sumi;
}

__device__ __forceinline__ float vec_dot_iq3_s_q8_1(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_1,
    const int& kbx,
    const int& iqs)
{
    const block_iq3_s* bq3 = (const block_iq3_s*)vbq + kbx;
    const int iqs2 = iqs >> 1;
    const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const int qh = bq3->qh[iqs2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs2);
    int sumi = 0;
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int sp_b = qwen4_get_u8(signs_packed_32, l0 / 2);
        const int2 grid_pos = make_int2(
            __ldg(iq3s_grid + (qwen4_get_u8_2(qs_packed, l0 + 0) | ((qh << (8 - l0)) & 0x100))),
            __ldg(iq3s_grid + (qwen4_get_u8_2(qs_packed, l0 + 1) | ((qh << (7 - l0)) & 0x100))));
        const int signs0 = __vcmpne4(((sp_b & 0x03) << 7) | ((sp_b & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((sp_b & 0x30) << 3) | ((sp_b & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs >> 2] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs2].ds);
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
    int sumi0 = 0, sumi1 = 0;
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
        const int grid = __ldg(iq1s_grid_gpu + (qwen4_get_u8(qs_packed, l0 / 2) | ((qhl & 0x07) << 8)));
        const int offset = (qhl & 0x08) ? 0x09090909 : 0x07070707;
        const int grid0 = __vsub4((grid << 3) & 0x78787878, offset);
        const int grid1 = __vsub4((grid >> 1) & 0x78787878, offset);
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid0, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid1, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid0, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid1, u1, sumi1);
        }
    }
    const uint16_t* sc = (const uint16_t*)bq1->scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d = 0.125f * __half2float(scale.f16) * __low2float(bq8_1[iqs].ds);
    const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
    const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
    return d * (float)(sumi0 * sc0 + sumi1 * sc1);
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

constexpr int VDR_Q4_K = 2, VDR_Q5_K = 2, VDR_Q5_1 = 2, VDR_Q5_0 = 2, VDR_Q4_1 = 2, VDR_Q4_0 = 2, VDR_Q8_0 = 2, VDR_Q6_K = 2;

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
    int sumi = 0;
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
        sumi = ggml_cuda_dp4a(__vsub4(vi0, 0x10101010), u0, sumi);
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = ggml_cuda_dp4a(__vsub4(vi1, 0x10101010), u1, sumi);
    }
    const float d5 = __half2float(bq5_0->d);
    const float d8 = __low2float(bq8_1->ds);
    return d5 * d8 * (float)sumi;
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

__device__ __forceinline__ float vec_dot_q6_K_q8_1(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_1,
    const int& kbx,
    const int& iqs)
{
    const block_q6_K* bq6_K = (const block_q6_K*)vbq + kbx;
    const int bq8_offset   = ((iqs >> 2) & 4) | ((iqs >> 3) & 1);
    const int scale_offset = ((iqs >> 1) & 8) | ((iqs >> 2) & 3);
    const int vh_shift     = (iqs >> 2) & 2;
    const int qh_idx       = ((iqs >> 1) & 8) | (iqs & 6);
    const int q8_sub       = iqs & 6;

    const int vl0 = get_int_b2(bq6_K->ql, iqs + 0);
    const int vl1 = get_int_b2(bq6_K->ql, iqs + 1);
    const int vh0 = get_int_b2(bq6_K->qh, qh_idx + 0) >> vh_shift;
    const int vh1 = get_int_b2(bq6_K->qh, qh_idx + 1) >> vh_shift;
    const int8_t* scales = bq6_K->scales + scale_offset;

    float sumf = 0.0f;
    #pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const block_q8_1* bq8 = bq8_1 + bq8_offset + 2 * i;
        const int u0 = get_int_b4(bq8->qs, q8_sub + 0);
        const int u1 = get_int_b4(bq8->qs, q8_sub + 1);
        const float d8 = __low2float(bq8->ds);
        const int vi0 = __vsubss4(((vl0 >> (4 * i)) & 0x0F0F0F0F) | (((vh0 >> (4 * i)) << 4) & 0x30303030), 0x20202020);
        const int vi1 = __vsubss4(((vl1 >> (4 * i)) & 0x0F0F0F0F) | (((vh1 >> (4 * i)) << 4) & 0x30303030), 0x20202020);
        const int dot = ggml_cuda_dp4a(vi1, u1, ggml_cuda_dp4a(vi0, u0, 0));
        sumf += d8 * (float)(dot * (int)scales[4 * i]);
    }
    return __half2float(bq6_K->d) * sumf;
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

template <int TILE_M>
__device__ __forceinline__ void vec_dot_q2_0_q8_1_tileM(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc)
{
    const block_q2_0* bq2_0 = (const block_q2_0*)vbq + kbx;
    const float d2 = __half2float(bq2_0->d);
    const int16_t* qs = (const int16_t*)bq2_0->qs + iqs * 4;
    int sumi[TILE_M] = {0};
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int qe = __byte_perm(0x020100FFu, 0x020100FFu, q >> 0);
        const int qo = __byte_perm(0x020100FFu, 0x020100FFu, q >> 2);
        const int qx = __byte_perm(qe, qo, 0x5140);
        const int qy = __byte_perm(qe, qo, 0x7362);
        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const block_q8_1* bq8 = bq8_base + t * q8_stride + iqs;
            const int u = get_int_b4(bq8->qs, j * 2 + 0);
            const int v = get_int_b4(bq8->qs, j * 2 + 1);
            sumi[t] = ggml_cuda_dp4a(u, qx, sumi[t]);
            sumi[t] = ggml_cuda_dp4a(v, qy, sumi[t]);
        }
    }
    #pragma unroll
    for (int t = 0; t < TILE_M; ++t) {
        const float d8 = __low2float((bq8_base + t * q8_stride + iqs)->ds);
        acc[t] += (d2 * d8) * (float)sumi[t];
    }
}

template <int TILE_M>
__device__ __forceinline__ void vec_dot2_q2_0_q8_1_tileM(
    const void* __restrict__ vbq0,
    const void* __restrict__ vbq1,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc0,
    float* __restrict__ acc1)
{
    const block_q2_0* b0 = (const block_q2_0*)vbq0 + kbx;
    const block_q2_0* b1 = (const block_q2_0*)vbq1 + kbx;
    const float d2_0 = __half2float(b0->d);
    const float d2_1 = __half2float(b1->d);
    const int16_t* qs0 = (const int16_t*)b0->qs + iqs * 4;
    const int16_t* qs1 = (const int16_t*)b1->qs + iqs * 4;
    int sum0[TILE_M] = {0};
    int sum1[TILE_M] = {0};
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q0 = qs0[j];
        const int qe0 = __byte_perm(0x020100FFu, 0x020100FFu, q0 >> 0);
        const int qo0 = __byte_perm(0x020100FFu, 0x020100FFu, q0 >> 2);
        const int qx0 = __byte_perm(qe0, qo0, 0x5140);
        const int qy0 = __byte_perm(qe0, qo0, 0x7362);

        const int q1 = qs1[j];
        const int qe1 = __byte_perm(0x020100FFu, 0x020100FFu, q1 >> 0);
        const int qo1 = __byte_perm(0x020100FFu, 0x020100FFu, q1 >> 2);
        const int qx1 = __byte_perm(qe1, qo1, 0x5140);
        const int qy1 = __byte_perm(qe1, qo1, 0x7362);

        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const block_q8_1* bq8 = bq8_base + t * q8_stride + iqs;
            const int u = get_int_b4(bq8->qs, j * 2 + 0);
            const int v = get_int_b4(bq8->qs, j * 2 + 1);
            sum0[t] = ggml_cuda_dp4a(u, qx0, sum0[t]);
            sum0[t] = ggml_cuda_dp4a(v, qy0, sum0[t]);
            sum1[t] = ggml_cuda_dp4a(u, qx1, sum1[t]);
            sum1[t] = ggml_cuda_dp4a(v, qy1, sum1[t]);
        }
    }
    #pragma unroll
    for (int t = 0; t < TILE_M; ++t) {
        const float d8 = __low2float((bq8_base + t * q8_stride + iqs)->ds);
        acc0[t] += (d2_0 * d8) * (float)sum0[t];
        acc1[t] += (d2_1 * d8) * (float)sum1[t];
    }
}

template <int TILE_M>
__device__ __forceinline__ void vec_dot_iq2_s_q8_1_tileM(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc)
{
    const block_iq2_s* bq2 = (const block_iq2_s*)vbq + kbx;
    const int iqs2 = iqs >> 1;
    const int qs_packed = get_int_b2(bq2->qs, iqs2);
    const int qh = bq2->qh[iqs2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs2);
    const int sc0 = 2 * (bq2->scales[iqs2] & 0x0F) + 1;
    const int sc1 = 2 * (bq2->scales[iqs2] >> 4) + 1;
    const float d2 = 0.125f * __half2float(bq2->d);

    int sumi0[TILE_M] = {0};
    int sumi1[TILE_M] = {0};
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qs_b = qwen4_get_u8(qs_packed, l0 / 2);
        const int sp_b = qwen4_get_u8(signs_packed_32, l0 / 2);
        const uint2 grid_pos = __ldg((const uint2*)(iq2s_grid + (qs_b | ((qh << (8 - l0)) & 0x300))));
        const int signs0 = __vcmpne4(((sp_b & 0x03) << 7) | ((sp_b & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((sp_b & 0x30) << 3) | ((sp_b & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4((int)grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4((int)grid_pos.y ^ signs1, signs1);
        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const block_q8_1* bq8 = bq8_base + t * q8_stride + iqs2;
            const int u0 = get_int_b4(bq8->qs, l0 + 0);
            const int u1 = get_int_b4(bq8->qs, l0 + 1);
            if (l0 < 4) {
                sumi0[t] = ggml_cuda_dp4a(grid_l, u0, sumi0[t]);
                sumi0[t] = ggml_cuda_dp4a(grid_h, u1, sumi0[t]);
            } else {
                sumi1[t] = ggml_cuda_dp4a(grid_l, u0, sumi1[t]);
                sumi1[t] = ggml_cuda_dp4a(grid_h, u1, sumi1[t]);
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < TILE_M; ++t) {
        const float d8 = __low2float((bq8_base + t * q8_stride + iqs2)->ds);
        acc[t] += (d2 * d8) * (float)(sumi0[t] * sc0 + sumi1[t] * sc1);
    }
}

template <int TILE_M>
__device__ __forceinline__ void vec_dot2_iq2_s_q8_1_tileM(
    const void* __restrict__ vbq0,
    const void* __restrict__ vbq1,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc0,
    float* __restrict__ acc1)
{
    const block_iq2_s* b0 = (const block_iq2_s*)vbq0 + kbx;
    const block_iq2_s* b1 = (const block_iq2_s*)vbq1 + kbx;
    const int qs_p0 = get_int_b2(b0->qs, iqs / 2), qh0 = b0->qh[iqs / 2], sp_p0 = get_int_b2(b0->qs, QK_K / 32 + iqs / 2);
    const int qs_p1 = get_int_b2(b1->qs, iqs / 2), qh1 = b1->qh[iqs / 2], sp_p1 = get_int_b2(b1->qs, QK_K / 32 + iqs / 2);
    const int sc0_0 = 2 * (b0->scales[iqs / 2] & 0x0F) + 1, sc1_0 = 2 * (b0->scales[iqs / 2] >> 4) + 1;
    const int sc0_1 = 2 * (b1->scales[iqs / 2] & 0x0F) + 1, sc1_1 = 2 * (b1->scales[iqs / 2] >> 4) + 1;
    const float d2_0 = 0.125f * __half2float(b0->d), d2_1 = 0.125f * __half2float(b1->d);

    int s0_0[TILE_M] = {0}, s1_0[TILE_M] = {0};
    int s0_1[TILE_M] = {0}, s1_1[TILE_M] = {0};
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qsb0 = qwen4_get_u8(qs_p0, l0 / 2), spb0 = qwen4_get_u8(sp_p0, l0 / 2);
        const uint2 gp0 = __ldg((const uint2*)(iq2s_grid + (qsb0 | ((qh0 << (8 - l0)) & 0x300))));
        const int sg00 = __vcmpne4(((spb0 & 0x03) << 7) | ((spb0 & 0x0C) << 21), 0);
        const int sg01 = __vcmpne4(((spb0 & 0x30) << 3) | ((spb0 & 0xC0) << 17), 0);
        const int gl0 = __vsub4((int)gp0.x ^ sg00, sg00);
        const int gh0 = __vsub4((int)gp0.y ^ sg01, sg01);

        const int qsb1 = qwen4_get_u8(qs_p1, l0 / 2), spb1 = qwen4_get_u8(sp_p1, l0 / 2);
        const uint2 gp1 = __ldg((const uint2*)(iq2s_grid + (qsb1 | ((qh1 << (8 - l0)) & 0x300))));
        const int sg10 = __vcmpne4(((spb1 & 0x03) << 7) | ((spb1 & 0x0C) << 21), 0);
        const int sg11 = __vcmpne4(((spb1 & 0x30) << 3) | ((spb1 & 0xC0) << 17), 0);
        const int gl1 = __vsub4((int)gp1.x ^ sg10, sg10);
        const int gh1 = __vsub4((int)gp1.y ^ sg11, sg11);

        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const block_q8_1* bq8 = bq8_base + t * q8_stride + (iqs / 2);
            const int u0 = get_int_b4(bq8->qs, l0 + 0);
            const int u1 = get_int_b4(bq8->qs, l0 + 1);
            if (l0 < 4) {
                s0_0[t] = ggml_cuda_dp4a(gl0, u0, s0_0[t]);
                s0_0[t] = ggml_cuda_dp4a(gh0, u1, s0_0[t]);
                s0_1[t] = ggml_cuda_dp4a(gl1, u0, s0_1[t]);
                s0_1[t] = ggml_cuda_dp4a(gh1, u1, s0_1[t]);
            } else {
                s1_0[t] = ggml_cuda_dp4a(gl0, u0, s1_0[t]);
                s1_0[t] = ggml_cuda_dp4a(gh0, u1, s1_0[t]);
                s1_1[t] = ggml_cuda_dp4a(gl1, u0, s1_1[t]);
                s1_1[t] = ggml_cuda_dp4a(gh1, u1, s1_1[t]);
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < TILE_M; ++t) {
        const float d8 = __low2float((bq8_base + t * q8_stride + (iqs / 2))->ds);
        acc0[t] += (d2_0 * d8) * (float)(s0_0[t] * sc0_0 + s1_0[t] * sc1_0);
        acc1[t] += (d2_1 * d8) * (float)(s0_1[t] * sc0_1 + s1_1[t] * sc1_1);
    }
}

template <int TILE_M>
__device__ __forceinline__ void vec_dot_iq3_xxs_q8_1_tileM(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc)
{
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*)vbq + kbx;
    const int iqs2 = iqs >> 1;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs2);
    const int ls = aux32 >> 28;
    const float d3 = 0.25f * __half2float(bq3->d) * (float)(2 * ls + 1);

    int sumi[TILE_M] = {0};
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            __ldg(iq3xxs_grid + qwen4_get_u8_2(q3_packed, l0 + 0)),
            __ldg(iq3xxs_grid + qwen4_get_u8_2(q3_packed, l0 + 1)));
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201u, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int signs1 = __vcmpne4(signs & 0x80402010u, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const block_q8_1* bq8 = bq8_base + t * q8_stride + iqs2;
            const int u0 = get_int_b4(bq8->qs, l0 + 0);
            const int u1 = get_int_b4(bq8->qs, l0 + 1);
            sumi[t] = ggml_cuda_dp4a(grid_l, u0, sumi[t]);
            sumi[t] = ggml_cuda_dp4a(grid_h, u1, sumi[t]);
        }
    }
    #pragma unroll
    for (int t = 0; t < TILE_M; ++t) {
        const float d8 = __low2float((bq8_base + t * q8_stride + iqs2)->ds);
        acc[t] += (d3 * d8) * (float)sumi[t];
    }
}

template <int TILE_M>
__device__ __forceinline__ void vec_dot2_iq3_xxs_q8_1_tileM(
    const void* __restrict__ vbq0,
    const void* __restrict__ vbq1,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc0,
    float* __restrict__ acc1)
{
    const block_iq3_xxs* b0 = (const block_iq3_xxs*)vbq0 + kbx;
    const block_iq3_xxs* b1 = (const block_iq3_xxs*)vbq1 + kbx;
    const int2 q3_p0 = make_int2(get_int_b2(b0->qs, iqs), get_int_b2(b0->qs, iqs + 1));
    const int2 q3_p1 = make_int2(get_int_b2(b1->qs, iqs), get_int_b2(b1->qs, iqs + 1));
    const uint32_t aux0 = get_int_b2(b0->qs, QK_K / 16 + iqs / 2);
    const uint32_t aux1 = get_int_b2(b1->qs, QK_K / 16 + iqs / 2);
    const int ls0 = aux0 >> 28, ls1 = aux1 >> 28;
    const float d3_0 = 0.25f * __half2float(b0->d) * (float)(2 * ls0 + 1);
    const float d3_1 = 0.25f * __half2float(b1->d) * (float)(2 * ls1 + 1);

    int sum0[TILE_M] = {0}, sum1[TILE_M] = {0};
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 gp0 = make_int2(__ldg(iq3xxs_grid + qwen4_get_u8_2(q3_p0, l0 + 0)), __ldg(iq3xxs_grid + qwen4_get_u8_2(q3_p0, l0 + 1)));
        const uint32_t sgn0 = unpack_ksigns(aux0 >> (7 * l0 / 2));
        const int sg00 = __vcmpne4(sgn0 & 0x08040201u, 0), sg01 = __vcmpne4(sgn0 & 0x80402010u, 0);
        const int gl0 = __vsub4(gp0.x ^ sg00, sg00), gh0 = __vsub4(gp0.y ^ sg01, sg01);

        const int2 gp1 = make_int2(__ldg(iq3xxs_grid + qwen4_get_u8_2(q3_p1, l0 + 0)), __ldg(iq3xxs_grid + qwen4_get_u8_2(q3_p1, l0 + 1)));
        const uint32_t sgn1 = unpack_ksigns(aux1 >> (7 * l0 / 2));
        const int sg10 = __vcmpne4(sgn1 & 0x08040201u, 0), sg11 = __vcmpne4(sgn1 & 0x80402010u, 0);
        const int gl1 = __vsub4(gp1.x ^ sg10, sg10), gh1 = __vsub4(gp1.y ^ sg11, sg11);

        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const block_q8_1* bq8 = bq8_base + t * q8_stride + (iqs / 2);
            const int u0 = get_int_b4(bq8->qs, l0 + 0);
            const int u1 = get_int_b4(bq8->qs, l0 + 1);
            sum0[t] = ggml_cuda_dp4a(gl0, u0, sum0[t]);
            sum0[t] = ggml_cuda_dp4a(gh0, u1, sum0[t]);
            sum1[t] = ggml_cuda_dp4a(gl1, u0, sum1[t]);
            sum1[t] = ggml_cuda_dp4a(gh1, u1, sum1[t]);
        }
    }
    #pragma unroll
    for (int t = 0; t < TILE_M; ++t) {
        const float d8 = __low2float((bq8_base + t * q8_stride + (iqs / 2))->ds);
        acc0[t] += (d3_0 * d8) * (float)sum0[t];
        acc1[t] += (d3_1 * d8) * (float)sum1[t];
    }
}

template <int TILE_M>
__device__ __forceinline__ void vec_dot_iq3_s_q8_1_tileM(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc)
{
    const block_iq3_s* bq3 = (const block_iq3_s*)vbq + kbx;
    const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
    const int sc_mult = 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d3 = __half2float(bq3->d) * (float)sc_mult;

    int sumi[TILE_M] = {0};
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int sp_b = qwen4_get_u8(signs_packed_32, l0 / 2);
        const int2 grid_pos = make_int2(
            __ldg(iq3s_grid + (qwen4_get_u8_2(qs_packed, l0 + 0) | ((qh << (8 - l0)) & 0x100))),
            __ldg(iq3s_grid + (qwen4_get_u8_2(qs_packed, l0 + 1) | ((qh << (7 - l0)) & 0x100))));
        const int signs0 = __vcmpne4(((sp_b & 0x03) << 7) | ((sp_b & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((sp_b & 0x30) << 3) | ((sp_b & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const block_q8_1* bq8 = bq8_base + t * q8_stride + (iqs / 2);
            const int u0 = get_int_b4(bq8->qs, l0 + 0);
            const int u1 = get_int_b4(bq8->qs, l0 + 1);
            sumi[t] = ggml_cuda_dp4a(grid_l, u0, sumi[t]);
            sumi[t] = ggml_cuda_dp4a(grid_h, u1, sumi[t]);
        }
    }
    #pragma unroll
    for (int t = 0; t < TILE_M; ++t) {
        const float d8 = __low2float((bq8_base + t * q8_stride + (iqs / 2))->ds);
        acc[t] += (d3 * d8) * (float)sumi[t];
    }
}

template <int TILE_M>
__device__ __forceinline__ void vec_dot2_iq3_s_q8_1_tileM(
    const void* __restrict__ vbq0,
    const void* __restrict__ vbq1,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc0,
    float* __restrict__ acc1)
{
    const block_iq3_s* b0 = (const block_iq3_s*)vbq0 + kbx;
    const block_iq3_s* b1 = (const block_iq3_s*)vbq1 + kbx;
    const int2 qs_p0 = make_int2(get_int_b2(b0->qs, iqs + 0), get_int_b2(b0->qs, iqs + 1));
    const int2 qs_p1 = make_int2(get_int_b2(b1->qs, iqs + 0), get_int_b2(b1->qs, iqs + 1));
    const int qh0 = b0->qh[iqs / 2], qh1 = b1->qh[iqs / 2];
    const int sp_p0 = get_int_b2(b0->signs, iqs / 2), sp_p1 = get_int_b2(b1->signs, iqs / 2);
    const int sc0 = 1 + 2 * ((b0->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const int sc1 = 1 + 2 * ((b1->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d3_0 = __half2float(b0->d) * (float)sc0;
    const float d3_1 = __half2float(b1->d) * (float)sc1;

    int sum0[TILE_M] = {0}, sum1[TILE_M] = {0};
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int spb0 = qwen4_get_u8(sp_p0, l0 / 2);
        const int2 gp0 = make_int2(
            __ldg(iq3s_grid + (qwen4_get_u8_2(qs_p0, l0 + 0) | ((qh0 << (8 - l0)) & 0x100))),
            __ldg(iq3s_grid + (qwen4_get_u8_2(qs_p0, l0 + 1) | ((qh0 << (7 - l0)) & 0x100))));
        const int sg00 = __vcmpne4(((spb0 & 0x03) << 7) | ((spb0 & 0x0C) << 21), 0);
        const int sg01 = __vcmpne4(((spb0 & 0x30) << 3) | ((spb0 & 0xC0) << 17), 0);
        const int gl0 = __vsub4(gp0.x ^ sg00, sg00), gh0 = __vsub4(gp0.y ^ sg01, sg01);

        const int spb1 = qwen4_get_u8(sp_p1, l0 / 2);
        const int2 gp1 = make_int2(
            __ldg(iq3s_grid + (qwen4_get_u8_2(qs_p1, l0 + 0) | ((qh1 << (8 - l0)) & 0x100))),
            __ldg(iq3s_grid + (qwen4_get_u8_2(qs_p1, l0 + 1) | ((qh1 << (7 - l0)) & 0x100))));
        const int sg10 = __vcmpne4(((spb1 & 0x03) << 7) | ((spb1 & 0x0C) << 21), 0);
        const int sg11 = __vcmpne4(((spb1 & 0x30) << 3) | ((spb1 & 0xC0) << 17), 0);
        const int gl1 = __vsub4(gp1.x ^ sg10, sg10), gh1 = __vsub4(gp1.y ^ sg11, sg11);

        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const block_q8_1* bq8 = bq8_base + t * q8_stride + (iqs / 2);
            const int u0 = get_int_b4(bq8->qs, l0 + 0);
            const int u1 = get_int_b4(bq8->qs, l0 + 1);
            sum0[t] = ggml_cuda_dp4a(gl0, u0, sum0[t]);
            sum0[t] = ggml_cuda_dp4a(gh0, u1, sum0[t]);
            sum1[t] = ggml_cuda_dp4a(gl1, u0, sum1[t]);
            sum1[t] = ggml_cuda_dp4a(gh1, u1, sum1[t]);
        }
    }
    #pragma unroll
    for (int t = 0; t < TILE_M; ++t) {
        const float d8 = __low2float((bq8_base + t * q8_stride + (iqs / 2))->ds);
        acc0[t] += (d3_0 * d8) * (float)sum0[t];
        acc1[t] += (d3_1 * d8) * (float)sum1[t];
    }
}

template <int TILE_M>
__device__ __forceinline__ void vec_dot_iq4_nl_q8_1_tileM(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc)
{
    const block_iq4_nl* bq4 = (const block_iq4_nl*)vbq + kbx;
    const float d4 = __half2float(bq4->d);
    int sumi[TILE_M] = {0};
    #pragma unroll
    for (int l = 0; l < 2; ++l) {
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const int* q8 = (const int*)(bq8_base + t * q8_stride)->qs + iqs;
            sumi[t] = ggml_cuda_dp4a(v.x, q8[l + 0], sumi[t]);
            sumi[t] = ggml_cuda_dp4a(v.y, q8[l + 4], sumi[t]);
        }
    }
    #pragma unroll
    for (int t = 0; t < TILE_M; ++t) {
        const float d8 = __low2float((bq8_base + t * q8_stride)->ds);
        acc[t] += (d4 * d8) * (float)sumi[t];
    }
}

template <int TILE_M>
__device__ __forceinline__ void vec_dot2_iq4_nl_q8_1_tileM(
    const void* __restrict__ vbq0,
    const void* __restrict__ vbq1,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc0,
    float* __restrict__ acc1)
{
    const block_iq4_nl* b0 = (const block_iq4_nl*)vbq0 + kbx;
    const block_iq4_nl* b1 = (const block_iq4_nl*)vbq1 + kbx;
    const float d4_0 = __half2float(b0->d);
    const float d4_1 = __half2float(b1->d);
    int sum0[TILE_M] = {0};
    int sum1[TILE_M] = {0};
    #pragma unroll
    for (int l = 0; l < 2; ++l) {
        const int2 v0 = get_int_from_table_16(get_int_b2(b0->qs, iqs + l), kvalues_iq4nl);
        const int2 v1 = get_int_from_table_16(get_int_b2(b1->qs, iqs + l), kvalues_iq4nl);
        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const int* q8 = (const int*)(bq8_base + t * q8_stride)->qs + iqs;
            const int u0 = q8[l + 0];
            const int u1 = q8[l + 4];
            sum0[t] = ggml_cuda_dp4a(v0.x, u0, sum0[t]);
            sum0[t] = ggml_cuda_dp4a(v0.y, u1, sum0[t]);
            sum1[t] = ggml_cuda_dp4a(v1.x, u0, sum1[t]);
            sum1[t] = ggml_cuda_dp4a(v1.y, u1, sum1[t]);
        }
    }
    #pragma unroll
    for (int t = 0; t < TILE_M; ++t) {
        const float d8 = __low2float((bq8_base + t * q8_stride)->ds);
        acc0[t] += (d4_0 * d8) * (float)sum0[t];
        acc1[t] += (d4_1 * d8) * (float)sum1[t];
    }
}

template <int TILE_M>
__device__ __forceinline__ void vec_dot_iq4_xs_q8_1_tileM(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc)
{
    const block_iq4_xs* bq4 = (const block_iq4_xs*)vbq + kbx;
    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    const float d4 = __half2float(bq4->d) * (float)(ls - 32);

    int sumi[TILE_M] = {0};
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const block_q8_1* bq8 = bq8_base + t * q8_stride + (iqs / 4);
            const int u0 = get_int_b4(bq8->qs, j + 0);
            const int u1 = get_int_b4(bq8->qs, j + 4);
            sumi[t] = ggml_cuda_dp4a(v.x, u0, sumi[t]);
            sumi[t] = ggml_cuda_dp4a(v.y, u1, sumi[t]);
        }
    }
    #pragma unroll
    for (int t = 0; t < TILE_M; ++t) {
        const float d8 = __low2float((bq8_base + t * q8_stride + (iqs / 4))->ds);
        acc[t] += (d4 * d8) * (float)sumi[t];
    }
}

template <int TILE_M>
__device__ __forceinline__ void vec_dot2_iq4_xs_q8_1_tileM(
    const void* __restrict__ vbq0,
    const void* __restrict__ vbq1,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc0,
    float* __restrict__ acc1)
{
    const block_iq4_xs* b0 = (const block_iq4_xs*)vbq0 + kbx;
    const block_iq4_xs* b1 = (const block_iq4_xs*)vbq1 + kbx;
    const int ls0 = ((b0->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((b0->scales_h >> (iqs / 2)) & 0x03) << 4);
    const int ls1 = ((b1->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((b1->scales_h >> (iqs / 2)) & 0x03) << 4);
    const float d4_0 = __half2float(b0->d) * (float)(ls0 - 32);
    const float d4_1 = __half2float(b1->d) * (float)(ls1 - 32);

    int sum0[TILE_M] = {0}, sum1[TILE_M] = {0};
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int2 v0 = get_int_from_table_16(get_int_b4(b0->qs, iqs + j), kvalues_iq4nl);
        const int2 v1 = get_int_from_table_16(get_int_b4(b1->qs, iqs + j), kvalues_iq4nl);
        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const block_q8_1* bq8 = bq8_base + t * q8_stride + (iqs / 4);
            const int u0 = get_int_b4(bq8->qs, j + 0);
            const int u1 = get_int_b4(bq8->qs, j + 4);
            sum0[t] = ggml_cuda_dp4a(v0.x, u0, sum0[t]);
            sum0[t] = ggml_cuda_dp4a(v0.y, u1, sum0[t]);
            sum1[t] = ggml_cuda_dp4a(v1.x, u0, sum1[t]);
            sum1[t] = ggml_cuda_dp4a(v1.y, u1, sum1[t]);
        }
    }
    #pragma unroll
    for (int t = 0; t < TILE_M; ++t) {
        const float d8 = __low2float((bq8_base + t * q8_stride + (iqs / 4))->ds);
        acc0[t] += (d4_0 * d8) * (float)sum0[t];
        acc1[t] += (d4_1 * d8) * (float)sum1[t];
    }
}

__device__ __forceinline__ uint32_t k_scale_min_u32(const uint8_t* scales8, int bq8_offset) {
    const uint16_t* scales = (const uint16_t*)scales8;
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = (uint32_t)-(int32_t)(j >= 2);
    const uint32_t a0 = (((s0 & 0x3f3fu) & ~hi) | ((((s4 >> 0) & 0x0f0fu) | ((s0 & 0xc0c0u) >> 2)) & hi)) & 0xFFFFu;
    const uint32_t a1 = (((s2 & 0x3f3fu) & ~hi) | ((((s4 >> 4) & 0x0f0fu) | ((s2 & 0xc0c0u) >> 2)) & hi)) & 0xFFFFu;
    return a0 | (a1 << 16);
}

template <int TILE_M>
__device__ __forceinline__ void vec_dot_q4_K_q8_1_tileM(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc)
{
    const block_q4_K* bq4_K = (const block_q4_K*)vbq + kbx;
    const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    const int q8_sub = (iqs / 2) % 4;
    const int* q4 = (const int*)(bq4_K->qs + 16 * bq8_offset + 4 * q8_sub);
    const int v0 = q4[0];
    const int v1 = q4[4];
    const uint32_t aux32 = k_scale_min_u32(bq4_K->scales, bq8_offset);
    const float2 dm4f = __half22float2(bq4_K->dm);

    #pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v0 >> (4 * i)) & 0x0F0F0F0F;
        const int v1i = (v1 >> (4 * i)) & 0x0F0F0F0F;
        const float sci = dm4f.x * (float)qwen4_get_u8(aux32, i);
        const float mi  = dm4f.y * (float)qwen4_get_u8(aux32, i + 2);
        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const block_q8_1* bq8i = bq8_base + t * q8_stride + bq8_offset + i;
            const float d8 = __low2float(bq8i->ds);
            const int* q8 = (const int*)bq8i->qs + q8_sub;
            const int u0 = q8[0];
            const int u1 = q8[4];
            const int dot1 = ggml_cuda_dp4a(v1i, u1, ggml_cuda_dp4a(v0i, u0, 0));
            const int dot2 = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, 0));
            acc[t] += d8 * (sci * (float)dot1 - mi * (float)dot2);
        }
    }
}

template <int TILE_M>
__device__ __forceinline__ void vec_dot2_q4_K_q8_1_tileM(
    const void* __restrict__ vbq0,
    const void* __restrict__ vbq1,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc0,
    float* __restrict__ acc1)
{
    const block_q4_K* b0 = (const block_q4_K*)vbq0 + kbx;
    const block_q4_K* b1 = (const block_q4_K*)vbq1 + kbx;
    const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    const int q8_sub = (iqs / 2) % 4;
    const int* q4_0 = (const int*)(b0->qs + 16 * bq8_offset + 4 * q8_sub);
    const int* q4_1 = (const int*)(b1->qs + 16 * bq8_offset + 4 * q8_sub);
    const int v0_0 = q4_0[0], v1_0 = q4_0[4];
    const int v0_1 = q4_1[0], v1_1 = q4_1[4];
    const uint32_t aux0 = k_scale_min_u32(b0->scales, bq8_offset);
    const uint32_t aux1 = k_scale_min_u32(b1->scales, bq8_offset);
    const float2 dm0 = __half22float2(b0->dm);
    const float2 dm1 = __half22float2(b1->dm);

    #pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i_0 = (v0_0 >> (4 * i)) & 0x0F0F0F0F;
        const int v1i_0 = (v1_0 >> (4 * i)) & 0x0F0F0F0F;
        const int v0i_1 = (v0_1 >> (4 * i)) & 0x0F0F0F0F;
        const int v1i_1 = (v1_1 >> (4 * i)) & 0x0F0F0F0F;
        const float sci0 = dm0.x * (float)qwen4_get_u8(aux0, i);
        const float mi0  = dm0.y * (float)qwen4_get_u8(aux0, i + 2);
        const float sci1 = dm1.x * (float)qwen4_get_u8(aux1, i);
        const float mi1  = dm1.y * (float)qwen4_get_u8(aux1, i + 2);
        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const block_q8_1* bq8i = bq8_base + t * q8_stride + bq8_offset + i;
            const float d8 = __low2float(bq8i->ds);
            const int* q8 = (const int*)bq8i->qs + q8_sub;
            const int u0 = q8[0];
            const int u1 = q8[4];
            const int dot1_0 = ggml_cuda_dp4a(v1i_0, u1, ggml_cuda_dp4a(v0i_0, u0, 0));
            const int dot1_1 = ggml_cuda_dp4a(v1i_1, u1, ggml_cuda_dp4a(v0i_1, u0, 0));
            const float fdot2 = (float)ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, 0));
            acc0[t] += d8 * (sci0 * (float)dot1_0 - mi0 * fdot2);
            acc1[t] += d8 * (sci1 * (float)dot1_1 - mi1 * fdot2);
        }
    }
}

template <int TILE_M>
__device__ __forceinline__ void vec_dot_q5_K_q8_1_tileM(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc)
{
    const block_q5_K* bq5_K = (const block_q5_K*)vbq + kbx;
    const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
    const int q8_sub = (iqs / 2) % 4;
    const int* ql = (const int*)(bq5_K->qs + 16 * bq8_offset + 4 * q8_sub);
    const int* qh = (const int*)(bq5_K->qh + 4 * q8_sub);
    const int vl0 = ql[0];
    const int vl1 = ql[4];
    const int vh0 = qh[0] >> bq8_offset;
    const int vh1 = qh[4] >> bq8_offset;
    const uint32_t aux32 = k_scale_min_u32(bq5_K->scales, bq8_offset);
    const float2 dm5f = __half22float2(bq5_K->dm);

    #pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int v0i = ((vl0 >> (4 * i)) & 0x0F0F0F0F) | (((vh0 >> i) << 4) & 0x10101010);
        const int v1i = ((vl1 >> (4 * i)) & 0x0F0F0F0F) | (((vh1 >> i) << 4) & 0x10101010);
        const float sci = dm5f.x * (float)qwen4_get_u8(aux32, i);
        const float mi  = dm5f.y * (float)qwen4_get_u8(aux32, i + 2);
        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const block_q8_1* bq8i = bq8_base + t * q8_stride + bq8_offset + i;
            const float d8 = __low2float(bq8i->ds);
            const int* q8 = (const int*)bq8i->qs + q8_sub;
            const int u0 = q8[0];
            const int u1 = q8[4];
            const int dot1 = ggml_cuda_dp4a(v0i, u0, ggml_cuda_dp4a(v1i, u1, 0));
            const int dot2 = ggml_cuda_dp4a(0x01010101, u0, ggml_cuda_dp4a(0x01010101, u1, 0));
            acc[t] += d8 * (sci * (float)dot1 - mi * (float)dot2);
        }
    }
}

template <int TILE_M>
__device__ __forceinline__ void vec_dot2_q5_K_q8_1_tileM(
    const void* __restrict__ vbq0,
    const void* __restrict__ vbq1,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc0,
    float* __restrict__ acc1)
{
    const block_q5_K* b0 = (const block_q5_K*)vbq0 + kbx;
    const block_q5_K* b1 = (const block_q5_K*)vbq1 + kbx;
    const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
    const int q8_sub = (iqs / 2) % 4;
    const int* ql0 = (const int*)(b0->qs + 16 * bq8_offset + 4 * q8_sub);
    const int* qh0 = (const int*)(b0->qh + 4 * q8_sub);
    const int* ql1 = (const int*)(b1->qs + 16 * bq8_offset + 4 * q8_sub);
    const int* qh1 = (const int*)(b1->qh + 4 * q8_sub);
    const int vl0_0 = ql0[0], vl1_0 = ql0[4], vh0_0 = qh0[0] >> bq8_offset, vh1_0 = qh0[4] >> bq8_offset;
    const int vl0_1 = ql1[0], vl1_1 = ql1[4], vh0_1 = qh1[0] >> bq8_offset, vh1_1 = qh1[4] >> bq8_offset;
    const uint32_t aux0 = k_scale_min_u32(b0->scales, bq8_offset);
    const uint32_t aux1 = k_scale_min_u32(b1->scales, bq8_offset);
    const float2 dm0 = __half22float2(b0->dm);
    const float2 dm1 = __half22float2(b1->dm);

    #pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int v0i_0 = ((vl0_0 >> (4 * i)) & 0x0F0F0F0F) | (((vh0_0 >> i) << 4) & 0x10101010);
        const int v1i_0 = ((vl1_0 >> (4 * i)) & 0x0F0F0F0F) | (((vh1_0 >> i) << 4) & 0x10101010);
        const int v0i_1 = ((vl0_1 >> (4 * i)) & 0x0F0F0F0F) | (((vh0_1 >> i) << 4) & 0x10101010);
        const int v1i_1 = ((vl1_1 >> (4 * i)) & 0x0F0F0F0F) | (((vh1_1 >> i) << 4) & 0x10101010);
        const float sci0 = dm0.x * (float)qwen4_get_u8(aux0, i);
        const float mi0  = dm0.y * (float)qwen4_get_u8(aux0, i + 2);
        const float sci1 = dm1.x * (float)qwen4_get_u8(aux1, i);
        const float mi1  = dm1.y * (float)qwen4_get_u8(aux1, i + 2);
        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const block_q8_1* bq8i = bq8_base + t * q8_stride + bq8_offset + i;
            const float d8 = __low2float(bq8i->ds);
            const int* q8 = (const int*)bq8i->qs + q8_sub;
            const int u0 = q8[0];
            const int u1 = q8[4];
            const int dot1_0 = ggml_cuda_dp4a(v0i_0, u0, ggml_cuda_dp4a(v1i_0, u1, 0));
            const int dot1_1 = ggml_cuda_dp4a(v0i_1, u0, ggml_cuda_dp4a(v1i_1, u1, 0));
            const float fdot2 = (float)ggml_cuda_dp4a(0x01010101, u0, ggml_cuda_dp4a(0x01010101, u1, 0));
            acc0[t] += d8 * (sci0 * (float)dot1_0 - mi0 * fdot2);
            acc1[t] += d8 * (sci1 * (float)dot1_1 - mi1 * fdot2);
        }
    }
}

template <int TILE_M>
__device__ __forceinline__ void vec_dot_q6_K_q8_1_tileM(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc)
{
    const block_q6_K* bq6_K = (const block_q6_K*)vbq + kbx;
    const int bq8_offset   = ((iqs >> 2) & 4) | ((iqs >> 3) & 1);
    const int scale_offset = ((iqs >> 1) & 8) | ((iqs >> 2) & 3);
    const int vh_shift     = (iqs >> 2) & 2;
    const int qh_idx       = ((iqs >> 1) & 8) | (iqs & 6);
    const int q8_sub       = iqs & 6;

    const int vl0 = get_int_b2(bq6_K->ql, iqs + 0);
    const int vl1 = get_int_b2(bq6_K->ql, iqs + 1);
    const int vh0 = get_int_b2(bq6_K->qh, qh_idx + 0) >> vh_shift;
    const int vh1 = get_int_b2(bq6_K->qh, qh_idx + 1) >> vh_shift;
    const int8_t* scales = bq6_K->scales + scale_offset;
    const float d6 = __half2float(bq6_K->d);

    #pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const float d_sc = d6 * (float)scales[4 * i];
        const int vi0 = __vsubss4(((vl0 >> (4 * i)) & 0x0F0F0F0F) | (((vh0 >> (4 * i)) << 4) & 0x30303030), 0x20202020);
        const int vi1 = __vsubss4(((vl1 >> (4 * i)) & 0x0F0F0F0F) | (((vh1 >> (4 * i)) << 4) & 0x30303030), 0x20202020);
        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const block_q8_1* bq8 = bq8_base + t * q8_stride + bq8_offset + 2 * i;
            const int u0 = get_int_b4(bq8->qs, q8_sub + 0);
            const int u1 = get_int_b4(bq8->qs, q8_sub + 1);
            const float d8 = __low2float(bq8->ds);
            const int dot = ggml_cuda_dp4a(vi1, u1, ggml_cuda_dp4a(vi0, u0, 0));
            acc[t] += (d8 * d_sc) * (float)dot;
        }
    }
}

template <int TILE_M>
__device__ __forceinline__ void vec_dot2_q6_K_q8_1_tileM(
    const void* __restrict__ vbq0,
    const void* __restrict__ vbq1,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc0,
    float* __restrict__ acc1)
{
    const block_q6_K* b0 = (const block_q6_K*)vbq0 + kbx;
    const block_q6_K* b1 = (const block_q6_K*)vbq1 + kbx;
    const int bq8_offset   = ((iqs >> 2) & 4) | ((iqs >> 3) & 1);
    const int scale_offset = ((iqs >> 1) & 8) | ((iqs >> 2) & 3);
    const int vh_shift     = (iqs >> 2) & 2;
    const int qh_idx       = ((iqs >> 1) & 8) | (iqs & 6);
    const int q8_sub       = iqs & 6;

    const int vl0_0 = get_int_b2(b0->ql, iqs + 0), vl0_1 = get_int_b2(b0->ql, iqs + 1);
    const int vh0_0 = get_int_b2(b0->qh, qh_idx + 0) >> vh_shift, vh0_1 = get_int_b2(b0->qh, qh_idx + 1) >> vh_shift;
    const int vl1_0 = get_int_b2(b1->ql, iqs + 0), vl1_1 = get_int_b2(b1->ql, iqs + 1);
    const int vh1_0 = get_int_b2(b1->qh, qh_idx + 0) >> vh_shift, vh1_1 = get_int_b2(b1->qh, qh_idx + 1) >> vh_shift;
    const int8_t* sc0 = b0->scales + scale_offset;
    const int8_t* sc1 = b1->scales + scale_offset;
    const float d6_0 = __half2float(b0->d);
    const float d6_1 = __half2float(b1->d);

    #pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const float d_sc0 = d6_0 * (float)sc0[4 * i];
        const float d_sc1 = d6_1 * (float)sc1[4 * i];
        const int vi0_0 = __vsubss4(((vl0_0 >> (4 * i)) & 0x0F0F0F0F) | (((vh0_0 >> (4 * i)) << 4) & 0x30303030), 0x20202020);
        const int vi0_1 = __vsubss4(((vl0_1 >> (4 * i)) & 0x0F0F0F0F) | (((vh0_1 >> (4 * i)) << 4) & 0x30303030), 0x20202020);
        const int vi1_0 = __vsubss4(((vl1_0 >> (4 * i)) & 0x0F0F0F0F) | (((vh1_0 >> (4 * i)) << 4) & 0x30303030), 0x20202020);
        const int vi1_1 = __vsubss4(((vl1_1 >> (4 * i)) & 0x0F0F0F0F) | (((vh1_1 >> (4 * i)) << 4) & 0x30303030), 0x20202020);
        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const block_q8_1* bq8 = bq8_base + t * q8_stride + bq8_offset + 2 * i;
            const int u0 = get_int_b4(bq8->qs, q8_sub + 0);
            const int u1 = get_int_b4(bq8->qs, q8_sub + 1);
            const float d8 = __low2float(bq8->ds);
            acc0[t] += (d8 * d_sc0) * (float)ggml_cuda_dp4a(vi0_1, u1, ggml_cuda_dp4a(vi0_0, u0, 0));
            acc1[t] += (d8 * d_sc1) * (float)ggml_cuda_dp4a(vi1_1, u1, ggml_cuda_dp4a(vi1_0, u0, 0));
        }
    }
}

template <int TILE_M>
__device__ __forceinline__ void vec_dot_iq1_m_q8_1_tileM(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc)
{
    const block_iq1_m* bq1 = (const block_iq1_m*)vbq + kbx;
    const int qs_packed = get_int_b4(bq1->qs, iqs);
    int sumi0[TILE_M] = {0};
    int sumi1[TILE_M] = {0};
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
        const int grid = __ldg(iq1s_grid_gpu + (qwen4_get_u8(qs_packed, l0 / 2) | ((qhl & 0x07) << 8)));
        const int offset = (qhl & 0x08) ? 0x09090909 : 0x07070707;
        const int grid0 = __vsub4((grid << 3) & 0x78787878, offset);
        const int grid1 = __vsub4((grid >> 1) & 0x78787878, offset);
        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const block_q8_1* bq8 = bq8_base + t * q8_stride + iqs;
            const int u0 = get_int_b4(bq8->qs, l0 + 0);
            const int u1 = get_int_b4(bq8->qs, l0 + 1);
            if (l0 < 4) {
                sumi0[t] = ggml_cuda_dp4a(grid0, u0, sumi0[t]);
                sumi0[t] = ggml_cuda_dp4a(grid1, u1, sumi0[t]);
            } else {
                sumi1[t] = ggml_cuda_dp4a(grid0, u0, sumi1[t]);
                sumi1[t] = ggml_cuda_dp4a(grid1, u1, sumi1[t]);
            }
        }
    }
    const uint16_t* sc = (const uint16_t*)bq1->scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d1 = 0.125f * __half2float(scale.f16);
    const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
    const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
    #pragma unroll
    for (int t = 0; t < TILE_M; ++t) {
        const float d8 = __low2float((bq8_base + t * q8_stride + iqs)->ds);
        acc[t] += (d1 * d8) * (float)(sumi0[t] * sc0 + sumi1[t] * sc1);
    }
}

template <int TILE_M>
__device__ __forceinline__ void vec_dot2_iq1_m_q8_1_tileM(
    const void* __restrict__ vbq0,
    const void* __restrict__ vbq1,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc0,
    float* __restrict__ acc1)
{
    const block_iq1_m* b0 = (const block_iq1_m*)vbq0 + kbx;
    const block_iq1_m* b1 = (const block_iq1_m*)vbq1 + kbx;
    const int qs_p0 = get_int_b4(b0->qs, iqs);
    const int qs_p1 = get_int_b4(b1->qs, iqs);
    int s0_0[TILE_M] = {0}, s1_0[TILE_M] = {0};
    int s0_1[TILE_M] = {0}, s1_1[TILE_M] = {0};
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl0 = b0->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
        const int gr0  = __ldg(iq1s_grid_gpu + (qwen4_get_u8(qs_p0, l0 / 2) | ((qhl0 & 0x07) << 8)));
        const int off0 = (qhl0 & 0x08) ? 0x09090909 : 0x07070707;
        const int g00  = __vsub4((gr0 << 3) & 0x78787878, off0);
        const int g01  = __vsub4((gr0 >> 1) & 0x78787878, off0);

        const int qhl1 = b1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
        const int gr1  = __ldg(iq1s_grid_gpu + (qwen4_get_u8(qs_p1, l0 / 2) | ((qhl1 & 0x07) << 8)));
        const int off1 = (qhl1 & 0x08) ? 0x09090909 : 0x07070707;
        const int g10  = __vsub4((gr1 << 3) & 0x78787878, off1);
        const int g11  = __vsub4((gr1 >> 1) & 0x78787878, off1);

        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const block_q8_1* bq8 = bq8_base + t * q8_stride + iqs;
            const int u0 = get_int_b4(bq8->qs, l0 + 0);
            const int u1 = get_int_b4(bq8->qs, l0 + 1);
            if (l0 < 4) {
                s0_0[t] = ggml_cuda_dp4a(g00, u0, s0_0[t]);
                s0_0[t] = ggml_cuda_dp4a(g01, u1, s0_0[t]);
                s0_1[t] = ggml_cuda_dp4a(g10, u0, s0_1[t]);
                s0_1[t] = ggml_cuda_dp4a(g11, u1, s0_1[t]);
            } else {
                s1_0[t] = ggml_cuda_dp4a(g00, u0, s1_0[t]);
                s1_0[t] = ggml_cuda_dp4a(g01, u1, s1_0[t]);
                s1_1[t] = ggml_cuda_dp4a(g10, u0, s1_1[t]);
                s1_1[t] = ggml_cuda_dp4a(g11, u1, s1_1[t]);
            }
        }
    }
    const uint16_t* sc_0 = (const uint16_t*)b0->scales;
    const uint16_t* sc_1 = (const uint16_t*)b1->scales;
    iq1m_scale_t scale0, scale1;
    scale0.u16 = (sc_0[0] >> 12) | ((sc_0[1] >> 8) & 0x00F0) | ((sc_0[2] >> 4) & 0x0F00) | (sc_0[3] & 0xF000);
    scale1.u16 = (sc_1[0] >> 12) | ((sc_1[1] >> 8) & 0x00F0) | ((sc_1[2] >> 4) & 0x0F00) | (sc_1[3] & 0xF000);
    const float d1_0 = 0.125f * __half2float(scale0.f16);
    const float d1_1 = 0.125f * __half2float(scale1.f16);
    const int tmp0 = sc_0[iqs / 2] >> (6 * (iqs % 2));
    const int tmp1 = sc_1[iqs / 2] >> (6 * (iqs % 2));
    const int sc0_0 = 2 * ((tmp0 >> 0) & 0x07) + 1, sc1_0 = 2 * ((tmp0 >> 3) & 0x07) + 1;
    const int sc0_1 = 2 * ((tmp1 >> 0) & 0x07) + 1, sc1_1 = 2 * ((tmp1 >> 3) & 0x07) + 1;
    #pragma unroll
    for (int t = 0; t < TILE_M; ++t) {
        const float d8 = __low2float((bq8_base + t * q8_stride + iqs)->ds);
        acc0[t] += (d1_0 * d8) * (float)(s0_0[t] * sc0_0 + s1_0[t] * sc1_0);
        acc1[t] += (d1_1 * d8) * (float)(s0_1[t] * sc0_1 + s1_1[t] * sc1_1);
    }
}

template <int TILE_M>
__device__ __forceinline__ void vec_dot_iq2_xxs_q8_1_tileM(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc)
{
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*)vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi[TILE_M] = {0};
    #pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = __ldg(((const uint2*)iq2xxs_grid) + qwen4_get_u8(q2, k0 / 2));
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201u, 0);
        const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
        const int signs1 = __vcmpne4(signs & 0x80402010u, 0);
        const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const block_q8_1* bq8 = bq8_base + t * q8_stride + (iqs / 2);
            const int u0 = get_int_b4(bq8->qs, k0 + 0);
            const int u1 = get_int_b4(bq8->qs, k0 + 1);
            sumi[t] = ggml_cuda_dp4a(grid0, u0, sumi[t]);
            sumi[t] = ggml_cuda_dp4a(grid1, u1, sumi[t]);
        }
    }
    const int ls = (aux32 >> 27) | 1;
    const float d2 = __half2float(bq2->d);
    #pragma unroll
    for (int t = 0; t < TILE_M; ++t) {
        const float d8 = __low2float((bq8_base + t * q8_stride + (iqs / 2))->ds);
        acc[t] += (d2 * d8) * (float)(sumi[t] * ls / 8);
    }
}

template <int TILE_M>
__device__ __forceinline__ void vec_dot_iq2_xs_q8_1_tileM(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc)
{
    const block_iq2_xs* bq2 = (const block_iq2_xs*)vbq + kbx;
    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0[TILE_M] = {0}, sumi1[TILE_M] = {0};
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int q2_val = qwen4_get_u16_2(q2_packed, l0 / 2);
        const uint2 grid_pos = __ldg(((const uint2*)iq2xs_grid) + (q2_val & 0x1FF));
        const uint32_t signs = unpack_ksigns(q2_val >> 9);
        const int signs0 = __vcmpne4(signs & 0x08040201u, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int signs1 = __vcmpne4(signs & 0x80402010u, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const block_q8_1* bq8 = bq8_base + t * q8_stride + (iqs / 2);
            const int u0 = get_int_b4(bq8->qs, l0 + 0);
            const int u1 = get_int_b4(bq8->qs, l0 + 1);
            if (l0 < 4) {
                sumi0[t] = ggml_cuda_dp4a(grid_l, u0, sumi0[t]);
                sumi0[t] = ggml_cuda_dp4a(grid_h, u1, sumi0[t]);
            } else {
                sumi1[t] = ggml_cuda_dp4a(grid_l, u0, sumi1[t]);
                sumi1[t] = ggml_cuda_dp4a(grid_h, u1, sumi1[t]);
            }
        }
    }
    const float d2 = __half2float(bq2->d);
    #pragma unroll
    for (int t = 0; t < TILE_M; ++t) {
        const int sumi = (sumi0[t] * ls0 + sumi1[t] * ls1 + (sumi0[t] + sumi1[t]) / 2) / 4;
        const float d8 = __low2float((bq8_base + t * q8_stride + (iqs / 2))->ds);
        acc[t] += (d2 * d8) * (float)sumi;
    }
}

template <int TILE_M>
__device__ __forceinline__ void vec_dot_q4_0_q8_1_tileM(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc)
{
    const block_q4_0* bq4 = (const block_q4_0*)vbq + kbx;
    const float d4 = __half2float(bq4->d);
    int sumi[TILE_M] = {0};
    #pragma unroll
    for (int i = 0; i < VDR_Q4_0; ++i) {
        const int v = get_int_b2(bq4->qs, iqs + i);
        const int vi0 = q4_0_centred((v >> 0) & 0x0F0F0F0F);
        const int vi1 = q4_0_centred((v >> 4) & 0x0F0F0F0F);
        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const block_q8_1* bq8 = bq8_base + t * q8_stride;
            sumi[t] = ggml_cuda_dp4a(vi0, get_int_b4(bq8->qs, iqs + i), sumi[t]);
            sumi[t] = ggml_cuda_dp4a(vi1, get_int_b4(bq8->qs, iqs + i + QI4_0), sumi[t]);
        }
    }
    #pragma unroll
    for (int t = 0; t < TILE_M; ++t) {
        const float d8 = __low2float((bq8_base + t * q8_stride)->ds);
        acc[t] += (d4 * d8) * (float)sumi[t];
    }
}

template <int TILE_M>
__device__ __forceinline__ void vec_dot_q4_1_q8_1_tileM(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc)
{
    const block_q4_1* bq4_1 = (const block_q4_1*)vbq + kbx;
    const float2 dm4 = __half22float2(bq4_1->dm);
    int sumi[TILE_M] = {0}, sumu[TILE_M] = {0};
    #pragma unroll
    for (int i = 0; i < VDR_Q4_1; ++i) {
        const int v = get_int_b4(bq4_1->qs, iqs + i);
        const int vi0 = (v >> 0) & 0x0F0F0F0F;
        const int vi1 = (v >> 4) & 0x0F0F0F0F;
        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const block_q8_1* bq8 = bq8_base + t * q8_stride;
            const int u0 = get_int_b4(bq8->qs, iqs + i), u1 = get_int_b4(bq8->qs, iqs + i + QI4_1);
            sumi[t] = ggml_cuda_dp4a(vi0, u0, sumi[t]);
            sumi[t] = ggml_cuda_dp4a(vi1, u1, sumi[t]);
            sumu[t] = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, sumu[t]));
        }
    }
    #pragma unroll
    for (int t = 0; t < TILE_M; ++t) {
        const float d8 = __low2float((bq8_base + t * q8_stride)->ds);
        acc[t] += sumi[t] * (dm4.x * d8) + sumu[t] * (dm4.y * d8);
    }
}

template <int TILE_M>
__device__ __forceinline__ void vec_dot_q5_0_q8_1_tileM(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc)
{
    const block_q5_0* bq5_0 = (const block_q5_0*)vbq + kbx;
    const float d5 = __half2float(bq5_0->d);
    int sumi[TILE_M] = {0};
    #pragma unroll
    for (int i = 0; i < VDR_Q5_0; ++i) {
        const int vl = get_int_b2(bq5_0->qs, iqs + i);
        const int vh = get_int_b2(bq5_0->qh, 0) >> (4 * (iqs + i));
        int vi0 = (vl >> 0) & 0x0F0F0F0F;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        vi0 = __vsub4(vi0, 0x10101010);
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        vi1 = __vsub4(vi1, 0x10101010);
        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const block_q8_1* bq8 = bq8_base + t * q8_stride;
            sumi[t] = ggml_cuda_dp4a(vi0, get_int_b4(bq8->qs, iqs + i), sumi[t]);
            sumi[t] = ggml_cuda_dp4a(vi1, get_int_b4(bq8->qs, iqs + i + QI5_0), sumi[t]);
        }
    }
    #pragma unroll
    for (int t = 0; t < TILE_M; ++t) {
        const float d8 = __low2float((bq8_base + t * q8_stride)->ds);
        acc[t] += (d5 * d8) * (float)sumi[t];
    }
}

template <int TILE_M>
__device__ __forceinline__ void vec_dot_q5_1_q8_1_tileM(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc)
{
    const block_q5_1* bq5_1 = (const block_q5_1*)vbq + kbx;
    const float2 dm5 = __half22float2(bq5_1->dm);
    int sumi[TILE_M] = {0}, sumu[TILE_M] = {0};
    #pragma unroll
    for (int i = 0; i < VDR_Q5_1; ++i) {
        const int vl = get_int_b4(bq5_1->qs, iqs + i);
        const int vh = get_int_b4(bq5_1->qh, 0) >> (4 * (iqs + i));
        int vi0 = (vl >> 0) & 0x0F0F0F0F;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const block_q8_1* bq8 = bq8_base + t * q8_stride;
            const int u0 = get_int_b4(bq8->qs, iqs + i), u1 = get_int_b4(bq8->qs, iqs + i + QI5_1);
            sumi[t] = ggml_cuda_dp4a(vi0, u0, sumi[t]);
            sumi[t] = ggml_cuda_dp4a(vi1, u1, sumi[t]);
            sumu[t] = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, sumu[t]));
        }
    }
    #pragma unroll
    for (int t = 0; t < TILE_M; ++t) {
        const float d8 = __low2float((bq8_base + t * q8_stride)->ds);
        acc[t] += sumi[t] * (dm5.x * d8) + sumu[t] * (dm5.y * d8);
    }
}

template <int TILE_M>
__device__ __forceinline__ void vec_dot_q8_0_q8_1_tileM(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc)
{
    const block_q8_0* bq8_0 = (const block_q8_0*)vbq + kbx;
    const float d8_0 = __half2float(bq8_0->d);
    int sumi[TILE_M] = {0};
    #pragma unroll
    for (int i = 0; i < VDR_Q8_0; ++i) {
        const int v = get_int_b2(bq8_0->qs, iqs + i);
        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const block_q8_1* bq8 = bq8_base + t * q8_stride;
            sumi[t] = ggml_cuda_dp4a(v, get_int_b4(bq8->qs, iqs + i), sumi[t]);
        }
    }
    #pragma unroll
    for (int t = 0; t < TILE_M; ++t) {
        const float d8_1 = __low2float((bq8_base + t * q8_stride)->ds);
        acc[t] += (d8_0 * d8_1) * (float)sumi[t];
    }
}

template <int TILE_M>
__device__ __forceinline__ void vec_dot2_q8_0_q8_1_tileM(
    const void* __restrict__ vbq0,
    const void* __restrict__ vbq1,
    const block_q8_1* __restrict__ bq8_base,
    int q8_stride,
    int kbx,
    int iqs,
    float* __restrict__ acc0,
    float* __restrict__ acc1)
{
    const block_q8_0* b0 = (const block_q8_0*)vbq0 + kbx;
    const block_q8_0* b1 = (const block_q8_0*)vbq1 + kbx;
    const float d0 = __half2float(b0->d);
    const float d1 = __half2float(b1->d);
    int sum0[TILE_M] = {0}, sum1[TILE_M] = {0};
    #pragma unroll
    for (int i = 0; i < VDR_Q8_0; ++i) {
        const int v0 = get_int_b2(b0->qs, iqs + i);
        const int v1 = get_int_b2(b1->qs, iqs + i);
        #pragma unroll
        for (int t = 0; t < TILE_M; ++t) {
            const int u = get_int_b4((bq8_base + t * q8_stride)->qs, iqs + i);
            sum0[t] = ggml_cuda_dp4a(v0, u, sum0[t]);
            sum1[t] = ggml_cuda_dp4a(v1, u, sum1[t]);
        }
    }
    #pragma unroll
    for (int t = 0; t < TILE_M; ++t) {
        const float d8_1 = __low2float((bq8_base + t * q8_stride)->ds);
        acc0[t] += (d0 * d8_1) * (float)sum0[t];
        acc1[t] += (d1 * d8_1) * (float)sum1[t];
    }
}

template <int TY> struct Fmt;
template<> struct Fmt<16> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ __forceinline__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xxs_q8_1(v, y, kbx, iqs); }
    template <int TILE_M> __device__ __forceinline__ static void dot_tileM(const void* v, const block_q8_1* y, int stride, int kbx, int iqs, float* acc) {
        vec_dot_iq2_xxs_q8_1_tileM<TILE_M>(v, y, stride, kbx, iqs, acc);
    }
    template <int TILE_M> __device__ __forceinline__ static void dot2_tileM(const void* v0, const void* v1, const block_q8_1* y, int stride, int kbx, int iqs, float* acc0, float* acc1) {
        vec_dot_iq2_xxs_q8_1_tileM<TILE_M>(v0, y, stride, kbx, iqs, acc0);
        vec_dot_iq2_xxs_q8_1_tileM<TILE_M>(v1, y, stride, kbx, iqs, acc1);
    } };
template<> struct Fmt<17> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ __forceinline__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xs_q8_1(v, y, kbx, iqs); }
    template <int TILE_M> __device__ __forceinline__ static void dot_tileM(const void* v, const block_q8_1* y, int stride, int kbx, int iqs, float* acc) {
        vec_dot_iq2_xs_q8_1_tileM<TILE_M>(v, y, stride, kbx, iqs, acc);
    }
    template <int TILE_M> __device__ __forceinline__ static void dot2_tileM(const void* v0, const void* v1, const block_q8_1* y, int stride, int kbx, int iqs, float* acc0, float* acc1) {
        vec_dot_iq2_xs_q8_1_tileM<TILE_M>(v0, y, stride, kbx, iqs, acc0);
        vec_dot_iq2_xs_q8_1_tileM<TILE_M>(v1, y, stride, kbx, iqs, acc1);
    } };
template<> struct Fmt<18> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ __forceinline__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_xxs_q8_1(v, y, kbx, iqs); }
    template <int TILE_M> __device__ __forceinline__ static void dot_tileM(const void* v, const block_q8_1* y, int stride, int kbx, int iqs, float* acc) {
        vec_dot_iq3_xxs_q8_1_tileM<TILE_M>(v, y, stride, kbx, iqs, acc);
    }
    template <int TILE_M> __device__ __forceinline__ static void dot2_tileM(const void* v0, const void* v1, const block_q8_1* y, int stride, int kbx, int iqs, float* acc0, float* acc1) {
        vec_dot2_iq3_xxs_q8_1_tileM<TILE_M>(v0, v1, y, stride, kbx, iqs, acc0, acc1);
    } };
template<> struct Fmt<20> { static constexpr int qk = 32, ipb = 2, step = 2;
    __device__ __forceinline__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_nl_q8_1(v, y, kbx, iqs); }
    template <int TILE_M> __device__ __forceinline__ static void dot_tileM(const void* v, const block_q8_1* y, int stride, int kbx, int iqs, float* acc) {
        vec_dot_iq4_nl_q8_1_tileM<TILE_M>(v, y, stride, kbx, iqs, acc);
    }
    template <int TILE_M> __device__ __forceinline__ static void dot2_tileM(const void* v0, const void* v1, const block_q8_1* y, int stride, int kbx, int iqs, float* acc0, float* acc1) {
        vec_dot2_iq4_nl_q8_1_tileM<TILE_M>(v0, v1, y, stride, kbx, iqs, acc0, acc1);
    } };
template<> struct Fmt<21> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ __forceinline__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_s_q8_1(v, y, kbx, iqs); }
    template <int TILE_M> __device__ __forceinline__ static void dot_tileM(const void* v, const block_q8_1* y, int stride, int kbx, int iqs, float* acc) {
        vec_dot_iq3_s_q8_1_tileM<TILE_M>(v, y, stride, kbx, iqs, acc);
    }
    template <int TILE_M> __device__ __forceinline__ static void dot2_tileM(const void* v0, const void* v1, const block_q8_1* y, int stride, int kbx, int iqs, float* acc0, float* acc1) {
        vec_dot2_iq3_s_q8_1_tileM<TILE_M>(v0, v1, y, stride, kbx, iqs, acc0, acc1);
    } };
template<> struct Fmt<22> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ __forceinline__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_s_q8_1(v, y, kbx, iqs); }
    template <int TILE_M> __device__ __forceinline__ static void dot_tileM(const void* v, const block_q8_1* y, int stride, int kbx, int iqs, float* acc) {
        vec_dot_iq2_s_q8_1_tileM<TILE_M>(v, y, stride, kbx, iqs, acc);
    }
    template <int TILE_M> __device__ __forceinline__ static void dot2_tileM(const void* v0, const void* v1, const block_q8_1* y, int stride, int kbx, int iqs, float* acc0, float* acc1) {
        vec_dot2_iq2_s_q8_1_tileM<TILE_M>(v0, v1, y, stride, kbx, iqs, acc0, acc1);
    } };
template<> struct Fmt<23> { static constexpr int qk = 256, ipb = 8, step = 4;
    __device__ __forceinline__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_xs_q8_1(v, y, kbx, iqs); }
    template <int TILE_M> __device__ __forceinline__ static void dot_tileM(const void* v, const block_q8_1* y, int stride, int kbx, int iqs, float* acc) {
        vec_dot_iq4_xs_q8_1_tileM<TILE_M>(v, y, stride, kbx, iqs, acc);
    }
    template <int TILE_M> __device__ __forceinline__ static void dot2_tileM(const void* v0, const void* v1, const block_q8_1* y, int stride, int kbx, int iqs, float* acc0, float* acc1) {
        vec_dot2_iq4_xs_q8_1_tileM<TILE_M>(v0, v1, y, stride, kbx, iqs, acc0, acc1);
    } };
template<> struct Fmt<29> { static constexpr int qk = 256, ipb = 8, step = 1;
    __device__ __forceinline__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq1_m_q8_1(v, y, kbx, iqs); }
    template <int TILE_M> __device__ __forceinline__ static void dot_tileM(const void* v, const block_q8_1* y, int stride, int kbx, int iqs, float* acc) {
        vec_dot_iq1_m_q8_1_tileM<TILE_M>(v, y, stride, kbx, iqs, acc);
    }
    template <int TILE_M> __device__ __forceinline__ static void dot2_tileM(const void* v0, const void* v1, const block_q8_1* y, int stride, int kbx, int iqs, float* acc0, float* acc1) {
        vec_dot2_iq1_m_q8_1_tileM<TILE_M>(v0, v1, y, stride, kbx, iqs, acc0, acc1);
    } };
template<> struct Fmt<42> { static constexpr int qk = 64, ipb = 2, step = 1;
    __device__ __forceinline__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q2_0_q8_1(v, y, kbx, iqs); }
    template <int TILE_M> __device__ __forceinline__ static void dot_tileM(const void* v, const block_q8_1* y, int stride, int kbx, int iqs, float* acc) {
        vec_dot_q2_0_q8_1_tileM<TILE_M>(v, y, stride, kbx, iqs, acc);
    }
    template <int TILE_M> __device__ __forceinline__ static void dot2_tileM(const void* v0, const void* v1, const block_q8_1* y, int stride, int kbx, int iqs, float* acc0, float* acc1) {
        vec_dot2_q2_0_q8_1_tileM<TILE_M>(v0, v1, y, stride, kbx, iqs, acc0, acc1);
    } };
template<> struct Fmt<12> { static constexpr int qk = 256, ipb = QI4_K / VDR_Q4_K, step = VDR_Q4_K;
    __device__ __forceinline__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q4_K_q8_1(v, y, kbx, iqs); }
    template <int TILE_M> __device__ __forceinline__ static void dot_tileM(const void* v, const block_q8_1* y, int stride, int kbx, int iqs, float* acc) {
        vec_dot_q4_K_q8_1_tileM<TILE_M>(v, y, stride, kbx, iqs, acc);
    }
    template <int TILE_M> __device__ __forceinline__ static void dot2_tileM(const void* v0, const void* v1, const block_q8_1* y, int stride, int kbx, int iqs, float* acc0, float* acc1) {
        vec_dot2_q4_K_q8_1_tileM<TILE_M>(v0, v1, y, stride, kbx, iqs, acc0, acc1);
    } };
template<> struct Fmt<13> { static constexpr int qk = 256, ipb = QI5_K / VDR_Q5_K, step = VDR_Q5_K;
    __device__ __forceinline__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_K_q8_1(v, y, kbx, iqs); }
    template <int TILE_M> __device__ __forceinline__ static void dot_tileM(const void* v, const block_q8_1* y, int stride, int kbx, int iqs, float* acc) {
        vec_dot_q5_K_q8_1_tileM<TILE_M>(v, y, stride, kbx, iqs, acc);
    }
    template <int TILE_M> __device__ __forceinline__ static void dot2_tileM(const void* v0, const void* v1, const block_q8_1* y, int stride, int kbx, int iqs, float* acc0, float* acc1) {
        vec_dot2_q5_K_q8_1_tileM<TILE_M>(v0, v1, y, stride, kbx, iqs, acc0, acc1);
    } };
template<> struct Fmt<14> { static constexpr int qk = 256, ipb = QI6_K / VDR_Q6_K, step = VDR_Q6_K;
    __device__ __forceinline__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q6_K_q8_1(v, y, kbx, iqs); }
    template <int TILE_M> __device__ __forceinline__ static void dot_tileM(const void* v, const block_q8_1* y, int stride, int kbx, int iqs, float* acc) {
        vec_dot_q6_K_q8_1_tileM<TILE_M>(v, y, stride, kbx, iqs, acc);
    }
    template <int TILE_M> __device__ __forceinline__ static void dot2_tileM(const void* v0, const void* v1, const block_q8_1* y, int stride, int kbx, int iqs, float* acc0, float* acc1) {
        vec_dot2_q6_K_q8_1_tileM<TILE_M>(v0, v1, y, stride, kbx, iqs, acc0, acc1);
    } };
template<> struct Fmt<7> { static constexpr int qk = 32, ipb = QI5_1 / VDR_Q5_1, step = VDR_Q5_1;
    __device__ __forceinline__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_1_q8_1(v, y, kbx, iqs); }
    template <int TILE_M> __device__ __forceinline__ static void dot_tileM(const void* v, const block_q8_1* y, int stride, int kbx, int iqs, float* acc) {
        vec_dot_q5_1_q8_1_tileM<TILE_M>(v, y, stride, kbx, iqs, acc);
    }
    template <int TILE_M> __device__ __forceinline__ static void dot2_tileM(const void* v0, const void* v1, const block_q8_1* y, int stride, int kbx, int iqs, float* acc0, float* acc1) {
        vec_dot_q5_1_q8_1_tileM<TILE_M>(v0, y, stride, kbx, iqs, acc0);
        vec_dot_q5_1_q8_1_tileM<TILE_M>(v1, y, stride, kbx, iqs, acc1);
    } };
template<> struct Fmt<2> { static constexpr int qk = 32, ipb = QI4_0 / VDR_Q4_0, step = VDR_Q4_0;
    __device__ __forceinline__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q4_0_q8_1(v, y, kbx, iqs); }
    template <int TILE_M> __device__ __forceinline__ static void dot_tileM(const void* v, const block_q8_1* y, int stride, int kbx, int iqs, float* acc) {
        vec_dot_q4_0_q8_1_tileM<TILE_M>(v, y, stride, kbx, iqs, acc);
    }
    template <int TILE_M> __device__ __forceinline__ static void dot2_tileM(const void* v0, const void* v1, const block_q8_1* y, int stride, int kbx, int iqs, float* acc0, float* acc1) {
        vec_dot_q4_0_q8_1_tileM<TILE_M>(v0, y, stride, kbx, iqs, acc0);
        vec_dot_q4_0_q8_1_tileM<TILE_M>(v1, y, stride, kbx, iqs, acc1);
    } };
template<> struct Fmt<3> { static constexpr int qk = 32, ipb = QI4_1 / VDR_Q4_1, step = VDR_Q4_1;
    __device__ __forceinline__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q4_1_q8_1(v, y, kbx, iqs); }
    template <int TILE_M> __device__ __forceinline__ static void dot_tileM(const void* v, const block_q8_1* y, int stride, int kbx, int iqs, float* acc) {
        vec_dot_q4_1_q8_1_tileM<TILE_M>(v, y, stride, kbx, iqs, acc);
    }
    template <int TILE_M> __device__ __forceinline__ static void dot2_tileM(const void* v0, const void* v1, const block_q8_1* y, int stride, int kbx, int iqs, float* acc0, float* acc1) {
        vec_dot_q4_1_q8_1_tileM<TILE_M>(v0, y, stride, kbx, iqs, acc0);
        vec_dot_q4_1_q8_1_tileM<TILE_M>(v1, y, stride, kbx, iqs, acc1);
    } };
template<> struct Fmt<6> { static constexpr int qk = 32, ipb = QI5_0 / VDR_Q5_0, step = VDR_Q5_0;
    __device__ __forceinline__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_0_q8_1(v, y, kbx, iqs); }
    template <int TILE_M> __device__ __forceinline__ static void dot_tileM(const void* v, const block_q8_1* y, int stride, int kbx, int iqs, float* acc) {
        vec_dot_q5_0_q8_1_tileM<TILE_M>(v, y, stride, kbx, iqs, acc);
    }
    template <int TILE_M> __device__ __forceinline__ static void dot2_tileM(const void* v0, const void* v1, const block_q8_1* y, int stride, int kbx, int iqs, float* acc0, float* acc1) {
        vec_dot_q5_0_q8_1_tileM<TILE_M>(v0, y, stride, kbx, iqs, acc0);
        vec_dot_q5_0_q8_1_tileM<TILE_M>(v1, y, stride, kbx, iqs, acc1);
    } };
template<> struct Fmt<8> { static constexpr int qk = 32, ipb = QI8_0 / VDR_Q8_0, step = VDR_Q8_0;
    __device__ __forceinline__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q8_0_q8_1(v, y, kbx, iqs); }
    template <int TILE_M> __device__ __forceinline__ static void dot_tileM(const void* v, const block_q8_1* y, int stride, int kbx, int iqs, float* acc) {
        vec_dot_q8_0_q8_1_tileM<TILE_M>(v, y, stride, kbx, iqs, acc);
    }
    template <int TILE_M> __device__ __forceinline__ static void dot2_tileM(const void* v0, const void* v1, const block_q8_1* y, int stride, int kbx, int iqs, float* acc0, float* acc1) {
        vec_dot2_q8_0_q8_1_tileM<TILE_M>(v0, v1, y, stride, kbx, iqs, acc0, acc1);
    } };

template <int LANES>
__device__ __forceinline__ float subwarp_sum(float v) {
    #pragma unroll
    for (int mask = LANES / 2; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, mask, LANES);
    }
    return v;
}

__device__ __forceinline__ void copy_q8_1_smem(
    block_q8_1* __restrict__ dst,
    const block_q8_1* __restrict__ src,
    int n_blocks,
    int tid)
{
    uint32_t* d32 = (uint32_t*)dst;
    const uint32_t* s32 = (const uint32_t*)src;
    const int n_u32 = n_blocks * (int)(sizeof(block_q8_1) / sizeof(uint32_t));
    for (int i = tid; i < n_u32; i += 256) {
        d32[i] = s32[i];
    }
}

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

template <int TY, int LANES>
__device__ __forceinline__ float row_dot_subwarp(const uint8_t* row, const block_q8_1* x, int nb, int sub_lane) {
    using F = Fmt<TY>;
    float s = 0.0f;
    if constexpr (LANES >= F::ipb && (LANES % F::ipb) == 0) {
        constexpr int kbx_step = LANES / F::ipb;
        constexpr int q8_per_blk = F::qk / 32;
        const int iqs = F::step * (sub_lane & (F::ipb - 1));
        for (int kbx = sub_lane / F::ipb; kbx < nb; kbx += kbx_step) {
            s += F::dot(row, x + kbx * q8_per_blk, kbx, iqs);
        }
    } else {
        for (int k = sub_lane; k < nb * F::ipb; k += LANES) {
            const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
            s += F::dot(row, x + kbx * (F::qk / 32), kbx, iqs);
        }
    }
    return subwarp_sum<LANES>(s);
}

template <int TY, int LANES>
__device__ __forceinline__ void row_dot2_subwarp(
    const uint8_t* row0,
    const uint8_t* row1,
    const block_q8_1* x,
    int nb,
    int sub_lane,
    float& out0,
    float& out1)
{
    using F = Fmt<TY>;
    float s0[1] = {0.0f}, s1[1] = {0.0f};
    if constexpr (LANES >= F::ipb && (LANES % F::ipb) == 0) {
        constexpr int kbx_step = LANES / F::ipb;
        constexpr int q8_per_blk = F::qk / 32;
        const int iqs = F::step * (sub_lane & (F::ipb - 1));
        for (int kbx = sub_lane / F::ipb; kbx < nb; kbx += kbx_step) {
            F::template dot2_tileM<1>(row0, row1, x + kbx * q8_per_blk, 0, kbx, iqs, s0, s1);
        }
    } else {
        for (int k = sub_lane; k < nb * F::ipb; k += LANES) {
            const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
            const block_q8_1* sx = x + kbx * (F::qk / 32);
            F::template dot2_tileM<1>(row0, row1, sx, 0, kbx, iqs, s0, s1);
        }
    }
    out0 = subwarp_sum<LANES>(s0[0]);
    out1 = subwarp_sum<LANES>(s1[0]);
}

template <int LANES>
__device__ __noinline__ float row_dot_cold_subwarp(
    int32_t ggml_type,
    const uint8_t* row,
    const block_q8_1* x,
    int n_in,
    int sub_lane)
{
    switch (ggml_type) {
        case 20: return row_dot_subwarp<20, LANES>(row, x, n_in / Fmt<20>::qk, sub_lane);
        case 8:  return row_dot_subwarp<8,  LANES>(row, x, n_in / Fmt<8>::qk,  sub_lane);
        case 22: return row_dot_subwarp<22, LANES>(row, x, n_in / Fmt<22>::qk, sub_lane);
        case 18: return row_dot_subwarp<18, LANES>(row, x, n_in / Fmt<18>::qk, sub_lane);
        case 21: return row_dot_subwarp<21, LANES>(row, x, n_in / Fmt<21>::qk, sub_lane);
        case 42: return row_dot_subwarp<42, LANES>(row, x, n_in / Fmt<42>::qk, sub_lane);
        case 29: return row_dot_subwarp<29, LANES>(row, x, n_in / Fmt<29>::qk, sub_lane);
        case 16: return row_dot_subwarp<16, LANES>(row, x, n_in / Fmt<16>::qk, sub_lane);
        case 17: return row_dot_subwarp<17, LANES>(row, x, n_in / Fmt<17>::qk, sub_lane);
        case 2:  return row_dot_subwarp<2,  LANES>(row, x, n_in / Fmt<2>::qk,  sub_lane);
        case 3:  return row_dot_subwarp<3,  LANES>(row, x, n_in / Fmt<3>::qk,  sub_lane);
        case 6:  return row_dot_subwarp<6,  LANES>(row, x, n_in / Fmt<6>::qk,  sub_lane);
        case 7:  return row_dot_subwarp<7,  LANES>(row, x, n_in / Fmt<7>::qk,  sub_lane);
        default: return 0.0f;
    }
}

template <int LANES>
__device__ __forceinline__ float row_dot_any_subwarp(
    int32_t ggml_type,
    const uint8_t* row,
    const block_q8_1* x,
    int n_in,
    int sub_lane)
{
    if (n_in == 640) {
        switch (ggml_type) {
            case 20: return row_dot_subwarp<20, LANES>(row, x, n_in / Fmt<20>::qk, sub_lane);
            case 8:  return row_dot_subwarp<8,  LANES>(row, x, n_in / Fmt<8>::qk,  sub_lane);
            default: return row_dot_cold_subwarp<LANES>(ggml_type, row, x, n_in, sub_lane);
        }
    }
    switch (ggml_type) {
        case 14: return row_dot_subwarp<14, LANES>(row, x, n_in / Fmt<14>::qk, sub_lane);
        case 13: return row_dot_subwarp<13, LANES>(row, x, n_in / Fmt<13>::qk, sub_lane);
        case 12: return row_dot_subwarp<12, LANES>(row, x, n_in / Fmt<12>::qk, sub_lane);
        case 23: return row_dot_subwarp<23, LANES>(row, x, n_in / Fmt<23>::qk, sub_lane);
        default: return row_dot_cold_subwarp<LANES>(ggml_type, row, x, n_in, sub_lane);
    }
}

template <int LANES>
__device__ __forceinline__ void row_dot2_any_subwarp(
    int32_t ggml_type,
    const uint8_t* row0,
    const uint8_t* row1,
    const block_q8_1* x,
    int n_in,
    int sub_lane,
    float& out0,
    float& out1)
{
    switch (ggml_type) {
        case 14: row_dot2_subwarp<14, LANES>(row0, row1, x, n_in / Fmt<14>::qk, sub_lane, out0, out1); break;
        case 12: row_dot2_subwarp<12, LANES>(row0, row1, x, n_in / Fmt<12>::qk, sub_lane, out0, out1); break;
        case 23: row_dot2_subwarp<23, LANES>(row0, row1, x, n_in / Fmt<23>::qk, sub_lane, out0, out1); break;
        case 13: row_dot2_subwarp<13, LANES>(row0, row1, x, n_in / Fmt<13>::qk, sub_lane, out0, out1); break;
        case 20: row_dot2_subwarp<20, LANES>(row0, row1, x, n_in / Fmt<20>::qk, sub_lane, out0, out1); break;
        case 8:  row_dot2_subwarp<8,  LANES>(row0, row1, x, n_in / Fmt<8>::qk,  sub_lane, out0, out1); break;
        default:
            out0 = row_dot_cold_subwarp<LANES>(ggml_type, row0, x, n_in, sub_lane);
            out1 = row_dot_cold_subwarp<LANES>(ggml_type, row1, x, n_in, sub_lane);
            break;
    }
}

// Sub-warp MMVQ with shared-memory x_q8_1 caching (256 threads = 8 warps per block)
template <int TY, int LANES>
__global__ void __launch_bounds__(256) mmvq_subwarp_kernel(
    const uint8_t* __restrict__ w,
    size_t row_bytes,
    const block_q8_1* __restrict__ x,
    float* __restrict__ y,
    int n_in,
    int n_out)
{
    __shared__ block_q8_1 s_x[192]; // up to n_in = 6144 (192 blocks = 6912 B)
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int n_q8 = n_in / 32;
    const block_q8_1* x_use = x;
    if (n_q8 <= 192) {
        copy_q8_1_smem(s_x, x, n_q8, tid);
        __syncthreads();
        x_use = s_x;
    }

    constexpr int ROWS_PER_BLOCK = 256 / LANES;
    const int sub_id   = tid / LANES;
    const int sub_lane = tid & (LANES - 1);
    const int row      = blockIdx.x * ROWS_PER_BLOCK + sub_id;
    if (row >= n_out) return;

    const int nb = n_in / Fmt<TY>::qk;
    const uint8_t* wr = w + (size_t)row * row_bytes;
    const float s = row_dot_subwarp<TY, LANES>(wr, x_use, nb, sub_lane);
    if (sub_lane == 0) {
        y[row] = s;
    }
}

// 2-row-per-subwarp MMVQ for large matrices (e.g. output.weight [248320, 2560], w_out [2560, 6144], ple_key [10240, 2560])
template <int TY, int LANES, int MAX_Q8 = 80>
__global__ void __launch_bounds__(256) mmvq_subwarp_2row_kernel(
    const uint8_t* __restrict__ w,
    size_t row_bytes,
    const block_q8_1* __restrict__ x,
    float* __restrict__ y,
    int n_in,
    int n_out)
{
    __shared__ block_q8_1 s_x[MAX_Q8];
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int n_q8 = n_in / 32;
    copy_q8_1_smem(s_x, x, n_q8, tid);
    __syncthreads();

    constexpr int SUBWARPS_PER_BLOCK = 256 / LANES;
    const int sub_id   = tid / LANES;
    const int sub_lane = tid & (LANES - 1);
    const int row0     = blockIdx.x * (2 * SUBWARPS_PER_BLOCK) + sub_id;
    const int row1     = row0 + SUBWARPS_PER_BLOCK;
    if (row0 >= n_out) return;

    const int nb = n_in / Fmt<TY>::qk;
    const uint8_t* wr0 = w + (size_t)row0 * row_bytes;
    if (row1 < n_out) {
        const uint8_t* wr1 = w + (size_t)row1 * row_bytes;
        float s0 = 0.0f, s1 = 0.0f;
        row_dot2_subwarp<TY, LANES>(wr0, wr1, s_x, nb, sub_lane, s0, s1);
        if (sub_lane == 0) {
            y[row0] = s0;
            y[row1] = s1;
        }
    } else {
        const float s0 = row_dot_subwarp<TY, LANES>(wr0, s_x, nb, sub_lane);
        if (sub_lane == 0) y[row0] = s0;
    }
}

// Batched MMVQ across `gridDim.y` routed experts with sub-warp lane packing + shared-memory x_q8_1
template <int TY, int LANES>
__global__ void __launch_bounds__(256) mmvq_batch10_subwarp_kernel(
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
    __shared__ block_q8_1 s_x[80]; // n_in <= 2560 (80 blocks = 2880 B)
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int k = blockIdx.y;
    const block_q8_1* x_exp = x + (size_t)k * x_expert_stride_blocks;
    const int n_q8 = n_in / 32;
    const block_q8_1* x_use = x_exp;
    if (n_q8 <= 80) {
        copy_q8_1_smem(s_x, x_exp, n_q8, tid);
        __syncthreads();
        x_use = s_x;
    }

    constexpr int ROWS_PER_BLOCK = 256 / LANES;
    const int sub_id   = tid / LANES;
    const int sub_lane = tid & (LANES - 1);
    const int row      = blockIdx.x * ROWS_PER_BLOCK + sub_id;
    if (row >= n_out) return;

    const uint8_t* w_exp = (base_w != nullptr)
        ? (base_w + (size_t)(d_top_idx != nullptr ? d_top_idx[k] : k) * expert_stride_bytes)
        : (const uint8_t*)expert_ptrs[k];
    const int nb = n_in / Fmt<TY>::qk;
    const uint8_t* wr = w_exp + (size_t)row * row_bytes;
    const float s = row_dot_subwarp<TY, LANES>(wr, x_use, nb, sub_lane);
    if (sub_lane == 0) y[(size_t)k * n_out + row] = s;
}

// Fused Gate + Up + SwiGLU + Q8_1 quantization across `gridDim.y` experts (1 for shared, 1..10 for routed)
// Launch: grid = dim3(20, count_k), block = dim3(32, 8) (256 threads)
// Each block bx = 0..19 computes 32 rows (bx * 32 .. bx * 32 + 31) of silu(gate) * up and writes out_q8_1[k * 20 + bx]!
template <int TY, int LANES>
__global__ void __launch_bounds__(256) mmvq_gate_up_swiglu_q8_1_batch_kernel(
    const uint8_t* __restrict__ base_gate,
    size_t gate_stride_bytes,
    const void* const* __restrict__ gate_ptrs,
    const uint8_t* __restrict__ base_up,
    size_t up_stride_bytes,
    const void* const* __restrict__ up_ptrs,
    const int32_t* __restrict__ d_top_idx,
    size_t row_bytes,
    const block_q8_1* __restrict__ x,
    block_q8_1* __restrict__ out_q8_1,
    int n_in)
{
    __shared__ block_q8_1 s_x[80];
    __shared__ float s_swiglu[32];

    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int n_q8 = n_in / 32;
    copy_q8_1_smem(s_x, x, n_q8, tid);
    __syncthreads();

    const int bx = blockIdx.x; // 0..19
    const int k  = blockIdx.y; // 0..count_k-1
    const int eid = (d_top_idx != nullptr) ? d_top_idx[k] : k;

    const uint8_t* wg_exp = (base_gate != nullptr)
        ? (base_gate + (size_t)eid * gate_stride_bytes)
        : (const uint8_t*)gate_ptrs[k];
    const uint8_t* wu_exp = (base_up != nullptr)
        ? (base_up + (size_t)eid * up_stride_bytes)
        : (const uint8_t*)up_ptrs[k];

    constexpr int SUBWARPS_PER_BLOCK = 256 / LANES; // 16 when LANES=16, 8 when LANES=32
    constexpr int PASSES = 32 / SUBWARPS_PER_BLOCK; // 2 when LANES=16, 4 when LANES=32
    const int sub_id   = tid / LANES;
    const int sub_lane = tid & (LANES - 1);
    const int nb       = n_in / Fmt<TY>::qk;

    #pragma unroll
    for (int p = 0; p < PASSES; ++p) {
        const int r   = p * SUBWARPS_PER_BLOCK + sub_id;
        const int row = bx * 32 + r;
        float g = 0.0f, u = 0.0f;
        row_dot2_subwarp<TY, LANES>(
            wg_exp + (size_t)row * row_bytes,
            wu_exp + (size_t)row * row_bytes,
            s_x, nb, sub_lane, g, u);
        if (sub_lane == 0) {
            s_swiglu[r] = (g / (1.0f + expf(-g))) * u;
        }
    }
    __syncthreads();

    if (threadIdx.y == 0) {
        const int lane = threadIdx.x;
        const float xi = q8_1_finite(s_swiglu[lane]);
        float amax = fabsf(xi);
        float sum  = xi;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask, 32));
            sum += __shfl_xor_sync(0xFFFFFFFFu, sum, mask, 32);
        }
        const float d = amax * (1.0f / 127.0f);
        const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / d);
        block_q8_1* blk = out_q8_1 + k * 20 + bx;
        blk->qs[lane] = q;
        if (lane == 0) {
            blk->ds = make_half2(__float2half(d), __float2half(sum));
        }
    }
}

// Multi-token Tile-M MMVQ for Prefill with K-chunked shared memory (2 rows per subwarp via dot2_tileM, 18.0 KB for TILE_M=16)
template <int TY, int LANES, int TILE_M>
__global__ void __launch_bounds__(256, 2) mmvq_tileM_kernel(
    const uint8_t* __restrict__ w,
    size_t row_bytes,
    const block_q8_1* __restrict__ x,
    float* __restrict__ y,
    int M,
    int n_in,
    int n_out)
{
    constexpr int CHUNK_Q8 = 32; // 32 Q8_1 blocks = 1024 columns (18,432 B for TILE_M = 16 -> 2 blocks/SM!)
    __shared__ block_q8_1 s_x[TILE_M * CHUNK_Q8];

    const int tid     = threadIdx.y * 32 + threadIdx.x;
    const int m_base  = blockIdx.y * TILE_M;
    const int m_count = (M - m_base < TILE_M) ? (M - m_base) : TILE_M;
    const int n_q8    = n_in / 32;

    constexpr int ROWS_PER_BLOCK = 256 / LANES;
    const int sub_id   = tid / LANES;
    const int sub_lane = tid & (LANES - 1);
    const int row0     = blockIdx.x * (2 * ROWS_PER_BLOCK) + sub_id;
    const int row1     = row0 + ROWS_PER_BLOCK;

    using F = Fmt<TY>;
    const uint8_t* wr0 = (row0 < n_out) ? (w + (size_t)row0 * row_bytes) : nullptr;
    const uint8_t* wr1 = (row1 < n_out) ? (w + (size_t)row1 * row_bytes) : wr0;

    float acc0[TILE_M], acc1[TILE_M];
    #pragma unroll
    for (int m = 0; m < TILE_M; ++m) {
        acc0[m] = 0.0f;
        acc1[m] = 0.0f;
    }

    for (int q8_start = 0; q8_start < n_q8; q8_start += CHUNK_Q8) {
        const int cur_q8 = (n_q8 - q8_start < CHUNK_Q8) ? (n_q8 - q8_start) : CHUNK_Q8;
        const int cur_u32 = cur_q8 * 9;
        for (int m = threadIdx.y; m < TILE_M; m += 8) {
            uint32_t* dst_m = (uint32_t*)(s_x + m * CHUNK_Q8);
            if (m < m_count) {
                const uint32_t* src_m = (const uint32_t*)(x + (size_t)(m_base + m) * n_q8 + q8_start);
                for (int i = threadIdx.x; i < cur_u32; i += 32) {
                    dst_m[i] = src_m[i];
                }
            } else {
                for (int i = threadIdx.x; i < cur_u32; i += 32) {
                    dst_m[i] = 0u;
                }
            }
        }
        __syncthreads();

        if (row1 < n_out) {
            const int kbx_base = q8_start / (F::qk / 32);
            const int cur_nb   = cur_q8 / (F::qk / 32);
            for (int k = sub_lane; k < cur_nb * F::ipb; k += LANES) {
                const int kbx_local = k / F::ipb;
                const int iqs       = F::step * (k % F::ipb);
                const int q8_off    = kbx_local * (F::qk / 32);
                F::template dot2_tileM<TILE_M>(wr0, wr1, s_x + q8_off, CHUNK_Q8, kbx_base + kbx_local, iqs, acc0, acc1);
            }
        } else if (row0 < n_out) {
            const int kbx_base = q8_start / (F::qk / 32);
            const int cur_nb   = cur_q8 / (F::qk / 32);
            for (int k = sub_lane; k < cur_nb * F::ipb; k += LANES) {
                const int kbx_local = k / F::ipb;
                const int iqs       = F::step * (k % F::ipb);
                const int q8_off    = kbx_local * (F::qk / 32);
                F::template dot_tileM<TILE_M>(wr0, s_x + q8_off, CHUNK_Q8, kbx_base + kbx_local, iqs, acc0);
            }
        }
        if (q8_start + CHUNK_Q8 < n_q8) {
            __syncthreads();
        }
    }

    if (row0 >= n_out) return;
    #pragma unroll
    for (int m = 0; m < TILE_M; ++m) {
        const float s0 = subwarp_sum<LANES>(acc0[m]);
        const float s1 = subwarp_sum<LANES>(acc1[m]);
        if (sub_lane == 0 && m < m_count) {
            float* ym = y + (size_t)(m_base + m) * n_out;
            ym[row0] = s0;
            if (row1 < n_out) ym[row1] = s1;
        }
    }
}

// ============================================================================
// BF16 / F16 / F32 Matrix-Vector & Batched Matrix-Matrix Multiplication
// ============================================================================

__device__ __forceinline__ float dot_bf16x8_fp32x8(uint4 w_u4, float4 x0, float4 x1) {
    float acc = 0.0f;
    acc += __uint_as_float(w_u4.x << 16)         * x0.x;
    acc += __uint_as_float(w_u4.x & 0xFFFF0000u) * x0.y;
    acc += __uint_as_float(w_u4.y << 16)         * x0.z;
    acc += __uint_as_float(w_u4.y & 0xFFFF0000u) * x0.w;
    acc += __uint_as_float(w_u4.z << 16)         * x1.x;
    acc += __uint_as_float(w_u4.z & 0xFFFF0000u) * x1.y;
    acc += __uint_as_float(w_u4.w << 16)         * x1.z;
    acc += __uint_as_float(w_u4.w & 0xFFFF0000u) * x1.w;
    return acc;
}

__global__ void __launch_bounds__(256) bf16_gemv_kernel(
    const uint16_t* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ y,
    int n_rows,
    int n_cols)
{
    const int lane = threadIdx.x;
    const int row  = blockIdx.x * 8 + threadIdx.y;
    if (row >= n_rows) return;

    const uint16_t* w_row = W + (size_t)row * n_cols;
    float acc = 0.0f;
    if ((n_cols & 7) == 0 && ((((uintptr_t)w_row | (uintptr_t)x) & 15) == 0)) {
        const int n_vec8 = n_cols / 8;
        const uint4* w_u4 = (const uint4*)w_row;
        const float4* x_f4 = (const float4*)x;
        for (int j = lane; j < n_vec8; j += 32) {
            const float4 x0 = __ldg(x_f4 + j * 2 + 0);
            const float4 x1 = __ldg(x_f4 + j * 2 + 1);
            acc += dot_bf16x8_fp32x8(w_u4[j], x0, x1);
        }
    } else {
        for (int j = lane; j < n_cols; j += 32) {
            acc += qwen4_bf16_to_fp32(w_row[j]) * __ldg(x + j);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

// Batched Tile-M BF16 GEMM for Prefill (reuses uint4 BF16 weight loads across up to TILE_M tokens)
template <int TILE_M>
__global__ void __launch_bounds__(256) bf16_gemm_tileM_kernel(
    const uint16_t* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int n_rows,
    int n_cols)
{
    __shared__ float s_x[TILE_M][512];

    const int row     = blockIdx.x * 8 + threadIdx.y;
    const int m_base  = blockIdx.y * TILE_M;
    const int m_count = (M - m_base < TILE_M) ? (M - m_base) : TILE_M;
    if (m_count <= 0) return;

    const int lane = threadIdx.x;
    const int tid  = threadIdx.y * 32 + lane;
    const uint16_t* w_row = (row < n_rows) ? (W + (size_t)row * n_cols) : nullptr;
    float acc[TILE_M];
    #pragma unroll
    for (int m = 0; m < TILE_M; ++m) acc[m] = 0.0f;

    if ((n_cols & 7) == 0 && ((((uintptr_t)W | (uintptr_t)x) & 15) == 0)) {
        for (int cBase = 0; cBase < n_cols; cBase += 512) {
            const int chunk_cols = (n_cols - cBase < 512) ? (n_cols - cBase) : 512;
            const int n_f4 = chunk_cols >> 2;
            if (tid < n_f4) {
                #pragma unroll
                for (int m = 0; m < TILE_M; ++m) {
                    if (m < m_count) {
                        const float4* xm_f4 = (const float4*)(x + (size_t)(m_base + m) * n_cols + cBase);
                        ((float4*)s_x[m])[tid] = xm_f4[tid];
                    } else {
                        ((float4*)s_x[m])[tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    }
                }
            }
            __syncthreads();

            if (row < n_rows) {
                const uint4* w_u4 = (const uint4*)(w_row + cBase);
                const int n_vec8 = chunk_cols >> 3;
                for (int j = lane; j < n_vec8; j += 32) {
                    const uint4 wv = w_u4[j];
                    const float w0 = __uint_as_float(wv.x << 16);
                    const float w1 = __uint_as_float(wv.x & 0xFFFF0000u);
                    const float w2 = __uint_as_float(wv.y << 16);
                    const float w3 = __uint_as_float(wv.y & 0xFFFF0000u);
                    const float w4 = __uint_as_float(wv.z << 16);
                    const float w5 = __uint_as_float(wv.z & 0xFFFF0000u);
                    const float w6 = __uint_as_float(wv.w << 16);
                    const float w7 = __uint_as_float(wv.w & 0xFFFF0000u);
                    #pragma unroll
                    for (int m = 0; m < TILE_M; ++m) {
                        const float4* sx_f4 = (const float4*)s_x[m];
                        const float4 x0 = sx_f4[j * 2 + 0];
                        const float4 x1 = sx_f4[j * 2 + 1];
                        acc[m] += w0 * x0.x + w1 * x0.y + w2 * x0.z + w3 * x0.w
                                + w4 * x1.x + w5 * x1.y + w6 * x1.z + w7 * x1.w;
                    }
                }
            }
            __syncthreads();
        }
    } else if (row < n_rows) {
        for (int j = lane; j < n_cols; j += 32) {
            const float wv = qwen4_bf16_to_fp32(w_row[j]);
            #pragma unroll
            for (int m = 0; m < TILE_M; ++m) {
                if (m < m_count) {
                    acc[m] += wv * x[(size_t)(m_base + m) * n_cols + j];
                }
            }
        }
    }

    if (row < n_rows) {
        #pragma unroll
        for (int m = 0; m < TILE_M; ++m) {
            acc[m] = warp_sum(acc[m]);
            if (lane == 0 && m < m_count) {
                y[(size_t)(m_base + m) * n_rows + row] = acc[m];
            }
        }
    }
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

// Vectorized FP32 -> BF16 conversion (4 floats -> 1 uint2 per thread) for Tensor Core GEMM
__global__ void f32_to_bf16_vec4_kernel(
    const float* __restrict__ src,
    uint16_t* __restrict__ dst,
    int n_vec4)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vec4) return;
    const float4 v = ((const float4*)src)[idx];
    const uint32_t p0 = (uint32_t)qwen4_fp32_to_bf16(v.x) | ((uint32_t)qwen4_fp32_to_bf16(v.y) << 16);
    const uint32_t p1 = (uint32_t)qwen4_fp32_to_bf16(v.z) | ((uint32_t)qwen4_fp32_to_bf16(v.w) << 16);
    ((uint2*)dst)[idx] = make_uint2(p0, p1);
}

// Fast 1D 256-Thread Vectorized Weight Dequantization (4 columns -> 1 uint2 BF16 per thread, constant-divisor fast path, 32-bit loads!)
template <int TY>
__global__ void __launch_bounds__(256) dequant_matrix_to_bf16_vec4_kernel(
    const uint8_t* __restrict__ w_raw,
    size_t row_bytes,
    uint16_t* __restrict__ w_bf16,
    int n_rows,
    int n_cols4)
{
    const int idx = blockIdx.x * 256 + threadIdx.x;
    const int total_vec4 = n_rows * n_cols4;
    if (idx >= total_vec4) return;

    int row, col4;
    if (n_cols4 == 640) {
        row  = idx / 640;
        col4 = idx - row * 640;
    } else if (n_cols4 == 1536) {
        row  = idx / 1536;
        col4 = idx - row * 1536;
    } else if (n_cols4 == 160) {
        row  = idx / 160;
        col4 = idx - row * 160;
    } else {
        row  = idx / n_cols4;
        col4 = idx - row * n_cols4;
    }
    const int col = col4 * 4;
    const uint8_t* row_ptr = w_raw + (size_t)row * row_bytes;

    float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
    if (TY == QWEN4_GGML_TYPE_Q8_0) {
        const block_q8_0* bq = (const block_q8_0*)row_ptr + (col >> 5);
        const int in_b = col & 31;
        const float d = __half2float(bq->d);
        const int q4 = get_int_b2(bq->qs, in_b >> 2);
        v0 = d * (float)(int8_t)(q4 >> 0);
        v1 = d * (float)(int8_t)(q4 >> 8);
        v2 = d * (float)(int8_t)(q4 >> 16);
        v3 = d * (float)(int8_t)(q4 >> 24);
    } else if (TY == QWEN4_GGML_TYPE_IQ4_NL) {
        const block_iq4_nl* bq = (const block_iq4_nl*)row_ptr + (col >> 5);
        const int in_b = col & 31;
        const int j = in_b & 15;
        const int shift = (in_b >> 4) * 4;
        const float d = __half2float(bq->d);
        const int q4 = get_int_b2(bq->qs, j >> 2) >> shift;
        v0 = d * (float)kvalues_iq4nl_lut((q4 >> 0)  & 0x0F);
        v1 = d * (float)kvalues_iq4nl_lut((q4 >> 8)  & 0x0F);
        v2 = d * (float)kvalues_iq4nl_lut((q4 >> 16) & 0x0F);
        v3 = d * (float)kvalues_iq4nl_lut((q4 >> 24) & 0x0F);
    } else if (TY == QWEN4_GGML_TYPE_IQ4_XS) {
        const block_iq4_xs* bq = (const block_iq4_xs*)row_ptr + (col >> 8);
        const int in_b = col & 255;
        const int ib = in_b >> 5;
        const int in_sub = in_b & 31;
        const int j = in_sub & 15;
        const int shift = (in_sub >> 4) * 4;
        const int ls = ((bq->scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F) | (((bq->scales_h >> (2 * ib)) & 3) << 4);
        const float dl = __half2float(bq->d) * (float)(ls - 32);
        const int q4 = get_int_b4(bq->qs, (ib * 16 + j) >> 2) >> shift;
        v0 = dl * (float)kvalues_iq4nl_lut((q4 >> 0)  & 0x0F);
        v1 = dl * (float)kvalues_iq4nl_lut((q4 >> 8)  & 0x0F);
        v2 = dl * (float)kvalues_iq4nl_lut((q4 >> 16) & 0x0F);
        v3 = dl * (float)kvalues_iq4nl_lut((q4 >> 24) & 0x0F);
    } else if (TY == QWEN4_GGML_TYPE_Q4_K) {
        const block_q4_K* bq = (const block_q4_K*)row_ptr + (col >> 8);
        const int in_b = col & 255;
        const int group64 = in_b >> 6;
        const int in_64 = in_b & 63;
        const int is_high = in_64 >> 5;
        const int l = in_64 & 31;
        const int is = group64 * 2 + is_high;
        const uint32_t sc_m = k_scale_min_u32(bq->scales, is);
        const float2 dm = __half22float2(bq->dm);
        const float d_eff = dm.x * (float)qwen4_get_u8(sc_m, is_high);
        const float m_eff = dm.y * (float)qwen4_get_u8(sc_m, is_high + 2);
        const int q4 = get_int_b4(bq->qs, (group64 * 32 + l) >> 2) >> (is_high * 4);
        v0 = d_eff * (float)((q4 >> 0)  & 0x0F) - m_eff;
        v1 = d_eff * (float)((q4 >> 8)  & 0x0F) - m_eff;
        v2 = d_eff * (float)((q4 >> 16) & 0x0F) - m_eff;
        v3 = d_eff * (float)((q4 >> 24) & 0x0F) - m_eff;
    } else if (TY == QWEN4_GGML_TYPE_Q5_K) {
        const block_q5_K* bq = (const block_q5_K*)row_ptr + (col >> 8);
        const int in_b = col & 255;
        const int group64 = in_b >> 6;
        const int in_64 = in_b & 63;
        const int is_high = in_64 >> 5;
        const int l = in_64 & 31;
        const int is = group64 * 2 + is_high;
        const uint32_t sc_m = k_scale_min_u32(bq->scales, is);
        const float2 dm = __half22float2(bq->dm);
        const float d_eff = dm.x * (float)qwen4_get_u8(sc_m, is_high);
        const float m_eff = dm.y * (float)qwen4_get_u8(sc_m, is_high + 2);
        const int ql4 = get_int_b4(bq->qs, (group64 * 32 + l) >> 2) >> (is_high * 4);
        const int qh4 = get_int_b4(bq->qh, l >> 2) >> is;
        v0 = d_eff * (float)(((ql4 >> 0)  & 0x0F) | (((qh4 >> 0)  & 1) << 4)) - m_eff;
        v1 = d_eff * (float)(((ql4 >> 8)  & 0x0F) | (((qh4 >> 8)  & 1) << 4)) - m_eff;
        v2 = d_eff * (float)(((ql4 >> 16) & 0x0F) | (((qh4 >> 16) & 1) << 4)) - m_eff;
        v3 = d_eff * (float)(((ql4 >> 24) & 0x0F) | (((qh4 >> 24) & 1) << 4)) - m_eff;
    } else if (TY == QWEN4_GGML_TYPE_Q6_K) {
        const block_q6_K* bq = (const block_q6_K*)row_ptr + (col >> 8);
        const int in_b = col & 255;
        const int half128 = in_b >> 7;
        const int in_128 = in_b & 127;
        const int q_group = in_128 >> 5;
        const int l = in_128 & 31;
        const int is = l >> 4;
        const int ql4 = get_int_b2(bq->ql + half128 * 64 + (q_group & 1) * 32, l >> 2) >> ((q_group >= 2) ? 4 : 0);
        const int qh4 = get_int_b2(bq->qh + half128 * 32, l >> 2) >> (2 * q_group);
        const int8_t* sc = bq->scales + half128 * 8;
        const float d_eff = __half2float(bq->d) * (float)sc[is + q_group * 2];
        v0 = d_eff * (float)((int)(((ql4 >> 0)  & 0x0F) | (((qh4 >> 0)  & 3) << 4)) - 32);
        v1 = d_eff * (float)((int)(((ql4 >> 8)  & 0x0F) | (((qh4 >> 8)  & 3) << 4)) - 32);
        v2 = d_eff * (float)((int)(((ql4 >> 16) & 0x0F) | (((qh4 >> 16) & 3) << 4)) - 32);
        v3 = d_eff * (float)((int)(((ql4 >> 24) & 0x0F) | (((qh4 >> 24) & 3) << 4)) - 32);
    }

    const uint32_t p0 = (uint32_t)qwen4_fp32_to_bf16(v0) | ((uint32_t)qwen4_fp32_to_bf16(v1) << 16);
    const uint32_t p1 = (uint32_t)qwen4_fp32_to_bf16(v2) | ((uint32_t)qwen4_fp32_to_bf16(v3) << 16);
    ((uint2*)w_bf16)[idx] = make_uint2(p0, p1);
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

__global__ void embed_to_hc_batchM_kernel(
    int32_t embd_type,
    const uint8_t* __restrict__ token_embd,
    size_t row_bytes,
    const int32_t* __restrict__ d_token_ids,
    float* __restrict__ d_R,
    int n_embd,
    int n_streams)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int m   = blockIdx.y;
    if (col >= n_embd) return;

    const int token_id = (d_token_ids != nullptr) ? d_token_ids[m] : m;
    const uint8_t* row_ptr = token_embd + (size_t)token_id * row_bytes;

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
        uint8_t sc, m_sc;
        if (is < 4) {
            sc   = q[is] & 63;
            m_sc = q[is + 4] & 63;
        } else {
            sc   = (q[is + 4] & 0x0F) | ((q[is - 4] >> 6) << 4);
            m_sc = (q[is + 4] >> 4)   | ((q[is - 0] >> 6) << 4);
        }
        const float2 dm = __half22float2(bq->dm);
        const uint8_t qbyte = bq->qs[group64 * 32 + l];
        const uint8_t nib = is_high ? (qbyte >> 4) : (qbyte & 0x0F);
        v = dm.x * (float)sc * (float)nib - dm.y * (float)m_sc;
    } else if (embd_type == QWEN4_GGML_TYPE_Q6_K) {
        const block_q6_K* bq = (const block_q6_K*)row_ptr + (col / 256);
        const int in_b = col % 256;
        const int half128 = in_b / 128;
        const int in_128 = in_b % 128;
        const int q_group = in_128 / 32;
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

    float* r_m = d_R + (size_t)m * (n_streams * n_embd);
    for (int s = 0; s < n_streams; ++s) {
        r_m[s * n_embd + col] = v;
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
// 4-Stream Hyper-Connection (HC Read & HC Write) - Vectorized + Fused Q8_1
// ============================================================================

constexpr int QWEN4_N_STREAMS    = 4;
constexpr int QWEN4_N_EMBD       = 2560;
constexpr int QWEN4_STREAM_TOTAL = QWEN4_N_STREAMS * QWEN4_N_EMBD; // 10240
constexpr int QWEN4_HC_RANK      = 320;
constexpr int QWEN4_GR_WARPS     = 8; // 256 threads

// 4 * M blocks (one per stream of 2560 floats): optional fused GR Write + per-stream RMSNorm -> normed[M, 4, 2560]
// Launch: <<<4 * M, 640>>> (20 warps per block = 1 float4 per thread, zero loop, zero branch divergence!)
__global__ void __launch_bounds__(640) gr_stream_rmsnorm_bf16_kernel(
    float* __restrict__ R,
    const float* __restrict__ norm_w,
    float* __restrict__ normed,
    float eps,
    const float* __restrict__ prev_sub_out = nullptr,
    const float* __restrict__ prev_gates = nullptr,
    uint16_t* __restrict__ normed_bf16_out = nullptr)
{
    __shared__ float smem[20];
    const int blk = blockIdx.x;
    const int m   = blk >> 2;
    const int s   = blk & 3;
    const int tid = threadIdx.x; // 0..639

    float4*       rs4  = (float4*)(R      + (size_t)blk * QWEN4_N_EMBD);
    const float4* ws4  = (const float4*)(norm_w + (size_t)s   * QWEN4_N_EMBD);
    float4*       out4 = (float4*)(normed + (size_t)blk * QWEN4_N_EMBD);

    float4 v = rs4[tid];
    if (prev_sub_out != nullptr && prev_gates != nullptr) {
        const float g = prev_gates[m * QWEN4_N_STREAMS + s];
        const float4 sub = ((const float4*)(prev_sub_out + (size_t)m * QWEN4_N_EMBD))[tid];
        v.x += g * sub.x;
        v.y += g * sub.y;
        v.z += g * sub.z;
        v.w += g * sub.w;
        rs4[tid] = v;
    }

    float ss = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    ss = block_sum<20>(ss, smem);
    const float scale = rsqrtf(ss * (1.0f / (float)QWEN4_N_EMBD) + eps);

    const float4 w = ws4[tid];
    const float4 out_v = make_float4(v.x * scale * w.x, v.y * scale * w.y, v.z * scale * w.z, v.w * scale * w.w);
    out4[tid] = out_v;
    if (normed_bf16_out != nullptr) {
        const uint32_t p0 = (uint32_t)qwen4_fp32_to_bf16(out_v.x) | ((uint32_t)qwen4_fp32_to_bf16(out_v.y) << 16);
        const uint32_t p1 = (uint32_t)qwen4_fp32_to_bf16(out_v.z) | ((uint32_t)qwen4_fp32_to_bf16(out_v.w) << 16);
        ((uint2*)(normed_bf16_out + (size_t)blk * QWEN4_N_EMBD))[tid] = make_uint2(p0, p1);
    }
}

// Fused Down-Projection (320 rows) + Optional Inject Gate Precomputation (4 rows) — 2 rows per block!
// Launch: grid = 162 (when inject_bf16 != nullptr) or 160, block = 256 (8 warps/block = 16 warps/SM, 5 unrolled uint4 loads/thread)
__global__ void __launch_bounds__(256) gr_down_inject_gemv_kernel(
    const uint16_t* __restrict__ down_bf16,
    const uint16_t* __restrict__ inject_bf16,
    const float* __restrict__ normed,
    float* __restrict__ h,
    float* __restrict__ gates_out)
{
    __shared__ float smem0[8];
    __shared__ float smem1[8];
    const int r0  = blockIdx.x * 2 + 0;
    const int r1  = blockIdx.x * 2 + 1;
    const int tid = threadIdx.x;

    const uint16_t* row0 = (r0 < QWEN4_HC_RANK)
        ? (down_bf16 + (size_t)r0 * QWEN4_STREAM_TOTAL)
        : (inject_bf16 + (size_t)(r0 - QWEN4_HC_RANK) * QWEN4_STREAM_TOTAL);
    const uint16_t* row1 = (r1 < QWEN4_HC_RANK)
        ? (down_bf16 + (size_t)r1 * QWEN4_STREAM_TOTAL)
        : (inject_bf16 + (size_t)(r1 - QWEN4_HC_RANK) * QWEN4_STREAM_TOTAL);

    const uint4*  w0_u4 = (const uint4*)row0;
    const uint4*  w1_u4 = (const uint4*)row1;
    const float4* x_f4  = (const float4*)normed;
    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll 5
    for (int j = tid; j < QWEN4_STREAM_TOTAL / 8; j += 256) {
        const float4 x0 = __ldg(x_f4 + j * 2 + 0);
        const float4 x1 = __ldg(x_f4 + j * 2 + 1);
        acc0 += dot_bf16x8_fp32x8(w0_u4[j], x0, x1);
        acc1 += dot_bf16x8_fp32x8(w1_u4[j], x0, x1);
    }
    acc0 = warp_sum(acc0);
    acc1 = warp_sum(acc1);
    if ((tid & 31) == 0) {
        smem0[tid >> 5] = acc0;
        smem1[tid >> 5] = acc1;
    }
    __syncthreads();

    if (tid == 0) {
        const float x0 = ((smem0[0] + smem0[1] + smem0[2] + smem0[3]) +
                          (smem0[4] + smem0[5] + smem0[6] + smem0[7])) * (1.0f / (float)QWEN4_N_STREAMS);
        const float x1 = ((smem1[0] + smem1[1] + smem1[2] + smem1[3]) +
                          (smem1[4] + smem1[5] + smem1[6] + smem1[7])) * (1.0f / (float)QWEN4_N_STREAMS);
        if (r0 < QWEN4_HC_RANK) {
            h[r0] = x0 / (1.0f + __expf(-x0));
            h[r1] = x1 / (1.0f + __expf(-x1));
        } else if (gates_out != nullptr) {
            gates_out[r0 - QWEN4_HC_RANK] = 2.0f / (1.0f + __expf(-x0));
            gates_out[r1 - QWEN4_HC_RANK] = 2.0f / (1.0f + __expf(-x1));
        }
    }
}

// Tensor-Core Post-Down Activation (h_raw -> SiLU(h_raw/4) in FP32 & BF16) + Optional 4-Row Inject Gate
// Launch: grid = dim3(inject_bf16 != nullptr ? 2 : 1, M), block = 256
__global__ void __launch_bounds__(256) gr_down_act_and_inject_kernel(
    const float* __restrict__ h_raw,
    float* __restrict__ h_out,
    uint16_t* __restrict__ h_bf16_out,
    const uint16_t* __restrict__ inject_bf16,
    const float* __restrict__ normed,
    float* __restrict__ gates_out,
    int down_rows = QWEN4_HC_RANK)
{
    __shared__ float smem[8];
    const int bx  = blockIdx.x;
    const int m   = blockIdx.y;
    const int tid = threadIdx.x;

    if (bx == 0) {
        const float* hr    = h_raw      + (size_t)m * down_rows;
        float*       hm    = h_out      + (size_t)m * QWEN4_HC_RANK;
        uint16_t*    hbf16 = h_bf16_out + (size_t)m * QWEN4_HC_RANK;
        for (int r = tid; r < QWEN4_HC_RANK; r += 256) {
            const float x = hr[r] * (1.0f / (float)QWEN4_N_STREAMS);
            const float val = x / (1.0f + expf(-x));
            hm[r]    = val;
            hbf16[r] = qwen4_fp32_to_bf16(val);
        }
        if (down_rows > QWEN4_HC_RANK && gates_out != nullptr && tid < QWEN4_N_STREAMS) {
            const float x = hr[QWEN4_HC_RANK + tid] * (1.0f / (float)QWEN4_N_STREAMS);
            gates_out[(size_t)m * QWEN4_N_STREAMS + tid] = 2.0f / (1.0f + expf(-x));
        }
    } else if (inject_bf16 != nullptr && gates_out != nullptr) {
        const int s       = tid >> 6; // 0..3 (64 threads = 2 warps per stream)
        const int sub_tid = tid & 63; // 0..63
        const uint4*  w_u4 = (const uint4*)(inject_bf16 + (size_t)s * QWEN4_STREAM_TOTAL);
        const float4* x_f4 = (const float4*)(normed + (size_t)m * QWEN4_STREAM_TOTAL);
        float acc = 0.0f;
        #pragma unroll 4
        for (int j = sub_tid; j < QWEN4_STREAM_TOTAL / 8; j += 64) {
            const float4 x0 = __ldg(x_f4 + j * 2 + 0);
            const float4 x1 = __ldg(x_f4 + j * 2 + 1);
            acc += dot_bf16x8_fp32x8(w_u4[j], x0, x1);
        }
        acc = warp_sum(acc);
        if ((tid & 31) == 0) smem[tid >> 5] = acc;
        __syncthreads();
        if (sub_tid == 0) {
            const float tot = smem[s * 2 + 0] + smem[s * 2 + 1];
            const float x   = tot * (1.0f / (float)QWEN4_N_STREAMS);
            gates_out[(size_t)m * QWEN4_N_STREAMS + s] = 2.0f / (1.0f + expf(-x));
        }
    }
}

// Tensor-Core Post-Up 4-Stream Sigmoid-Gated Mean + Optional Q8_1 Quantization (zero shared memory, zero barriers!)
// Launch: grid = dim3(QWEN4_N_EMBD / 128, M), block = dim3(32, 4)
__global__ void __launch_bounds__(128) gr_up_gate_mean_q8_1_batchM_kernel(
    const float* __restrict__ up_out,
    const float* __restrict__ normed,
    float* __restrict__ x_out,
    block_q8_1* __restrict__ x_q8_1_out)
{
    const int bx   = blockIdx.x * 4 + threadIdx.y; // 0..79
    const int lane = threadIdx.x;                  // 0..31
    const int col  = bx * 32 + lane;               // 0..2559
    const int m    = blockIdx.y;                   // 0..M-1

    const float* up_m = up_out + (size_t)m * QWEN4_STREAM_TOTAL + col;
    const float* nm_m = normed + (size_t)m * QWEN4_STREAM_TOTAL + col;

    float val = 0.0f;
    #pragma unroll
    for (int s = 0; s < QWEN4_N_STREAMS; ++s) {
        const float a = up_m[s * QWEN4_N_EMBD];
        const float g = 1.0f / (1.0f + expf(-a));
        val += g * nm_m[s * QWEN4_N_EMBD];
    }
    val *= 0.25f;
    x_out[(size_t)m * QWEN4_N_EMBD + col] = val;

    if (x_q8_1_out != nullptr) {
        const float xi = q8_1_finite(val);
        float amax = fabsf(xi);
        float sum  = xi;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask, 32));
            sum += __shfl_xor_sync(0xFFFFFFFFu, sum, mask, 32);
        }
        const float d = amax * (1.0f / 127.0f);
        const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / d);
        block_q8_1* blk = x_q8_1_out + (size_t)m * (QWEN4_N_EMBD / 32) + bx;
        blk->qs[lane] = q;
        if (lane == 0) {
            blk->ds = make_half2(__float2half(d), __float2half(sum));
        }
    }
}

// Fused Batched Down-Projection (320 rows) + Optional Inject Gate Precomputation (4 rows) + Activation
// Launch: grid = dim3(inject_bf16 != nullptr ? 21 : 20, (M + TILE_M - 1) / TILE_M), block = dim3(32, 8)
// Each warp computes 2 rows (16 rows/block), reusing s_x[TILE_M, 512] in shared memory -> cuts normed reads by 16x and BF16 weight reads by TILE_M!
template <int TILE_M>
__global__ void __launch_bounds__(256) gr_down_inject_tileM_kernel(
    const uint16_t* __restrict__ down_bf16,
    const uint16_t* __restrict__ inject_bf16,
    const float* __restrict__ normed,
    float* __restrict__ h,
    float* __restrict__ gates_out,
    int M)
{
    __shared__ float s_x[TILE_M][512];

    const int r0      = blockIdx.x * 16 + threadIdx.y * 2 + 0;
    const int r1      = r0 + 1;
    const int m_base  = blockIdx.y * TILE_M;
    const int m_count = (M - m_base < TILE_M) ? (M - m_base) : TILE_M;
    if (m_count <= 0) return;

    const int lane = threadIdx.x;
    const int tid  = threadIdx.y * 32 + lane;

    const uint16_t* row0 = nullptr;
    if (r0 < QWEN4_HC_RANK) {
        row0 = down_bf16 + (size_t)r0 * QWEN4_STREAM_TOTAL;
    } else if (r0 < QWEN4_HC_RANK + QWEN4_N_STREAMS && inject_bf16 != nullptr) {
        row0 = inject_bf16 + (size_t)(r0 - QWEN4_HC_RANK) * QWEN4_STREAM_TOTAL;
    }
    const uint16_t* row1 = nullptr;
    if (r1 < QWEN4_HC_RANK) {
        row1 = down_bf16 + (size_t)r1 * QWEN4_STREAM_TOTAL;
    } else if (r1 < QWEN4_HC_RANK + QWEN4_N_STREAMS && inject_bf16 != nullptr) {
        row1 = inject_bf16 + (size_t)(r1 - QWEN4_HC_RANK) * QWEN4_STREAM_TOTAL;
    }

    float acc0[TILE_M];
    float acc1[TILE_M];
    #pragma unroll
    for (int m = 0; m < TILE_M; ++m) {
        acc0[m] = 0.0f;
        acc1[m] = 0.0f;
    }

    for (int cBase = 0; cBase < QWEN4_STREAM_TOTAL; cBase += 512) {
        const int m_half = tid >> 7;
        const int col_f4 = tid & 127;
        #pragma unroll
        for (int mBase = 0; mBase < TILE_M; mBase += 2) {
            const int m = mBase + m_half;
            if (m < TILE_M) {
                if (m < m_count) {
                    const float4* xm_f4 = (const float4*)(normed + (size_t)(m_base + m) * QWEN4_STREAM_TOTAL + cBase);
                    ((float4*)s_x[m])[col_f4] = xm_f4[col_f4];
                } else {
                    ((float4*)s_x[m])[col_f4] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        }
        __syncthreads();

        if (row0 != nullptr) {
            const uint4* w0_u4 = (const uint4*)(row0 + cBase);
            const uint4* w1_u4 = (row1 != nullptr) ? (const uint4*)(row1 + cBase) : w0_u4;
            #pragma unroll
            for (int j = lane; j < 512 / 8; j += 32) {
                const uint4 wv0 = w0_u4[j];
                const float w00 = __uint_as_float(wv0.x << 16);
                const float w01 = __uint_as_float(wv0.x & 0xFFFF0000u);
                const float w02 = __uint_as_float(wv0.y << 16);
                const float w03 = __uint_as_float(wv0.y & 0xFFFF0000u);
                const float w04 = __uint_as_float(wv0.z << 16);
                const float w05 = __uint_as_float(wv0.z & 0xFFFF0000u);
                const float w06 = __uint_as_float(wv0.w << 16);
                const float w07 = __uint_as_float(wv0.w & 0xFFFF0000u);

                const uint4 wv1 = w1_u4[j];
                const float w10 = __uint_as_float(wv1.x << 16);
                const float w11 = __uint_as_float(wv1.x & 0xFFFF0000u);
                const float w12 = __uint_as_float(wv1.y << 16);
                const float w13 = __uint_as_float(wv1.y & 0xFFFF0000u);
                const float w14 = __uint_as_float(wv1.z << 16);
                const float w15 = __uint_as_float(wv1.z & 0xFFFF0000u);
                const float w16 = __uint_as_float(wv1.w << 16);
                const float w17 = __uint_as_float(wv1.w & 0xFFFF0000u);

                #pragma unroll
                for (int m = 0; m < TILE_M; ++m) {
                    const float4* sx_f4 = (const float4*)s_x[m];
                    const float4 x0 = sx_f4[j * 2 + 0];
                    const float4 x1 = sx_f4[j * 2 + 1];
                    acc0[m] += w00 * x0.x + w01 * x0.y + w02 * x0.z + w03 * x0.w
                             + w04 * x1.x + w05 * x1.y + w06 * x1.z + w07 * x1.w;
                    acc1[m] += w10 * x0.x + w11 * x0.y + w12 * x0.z + w13 * x0.w
                             + w14 * x1.x + w15 * x1.y + w16 * x1.z + w17 * x1.w;
                }
            }
        }
        __syncthreads();
    }

    if (row0 != nullptr) {
        #pragma unroll
        for (int m = 0; m < TILE_M; ++m) {
            acc0[m] = warp_sum(acc0[m]);
            acc1[m] = warp_sum(acc1[m]);
            if (lane == 0 && m < m_count) {
                const float x0 = acc0[m] * (1.0f / (float)QWEN4_N_STREAMS);
                if (r0 < QWEN4_HC_RANK) {
                    h[(size_t)(m_base + m) * QWEN4_HC_RANK + r0] = x0 / (1.0f + expf(-x0));
                } else if (gates_out != nullptr) {
                    gates_out[(size_t)(m_base + m) * QWEN4_N_STREAMS + (r0 - QWEN4_HC_RANK)] = 2.0f / (1.0f + expf(-x0));
                }
                if (row1 != nullptr) {
                    const float x1 = acc1[m] * (1.0f / (float)QWEN4_N_STREAMS);
                    if (r1 < QWEN4_HC_RANK) {
                        h[(size_t)(m_base + m) * QWEN4_HC_RANK + r1] = x1 / (1.0f + expf(-x1));
                    } else if (gates_out != nullptr) {
                        gates_out[(size_t)(m_base + m) * QWEN4_N_STREAMS + (r1 - QWEN4_HC_RANK)] = 2.0f / (1.0f + expf(-x1));
                    }
                }
            }
        }
    }
}

// Fused Up-Projection + 4-Stream Sigmoid-Gated Mean + Optional Q8_1 Quantization (Decode M = 1)
// Launch: grid = dim3(80, 1), block = 1024 (32 warps of 32 lanes; 1 stream per 8-lane subwarp -> 32 warps/SM!)
__global__ void __launch_bounds__(1024) gr_up_mean_q8_1_kernel(
    const uint16_t* __restrict__ up_bf16,
    const float* __restrict__ h,
    const float* __restrict__ normed,
    float* __restrict__ x_out,
    block_q8_1* __restrict__ x_q8_1_out,
    block_q8_1* __restrict__ x_q8_1_peer_out = nullptr)
{
    __shared__ float sh[QWEN4_HC_RANK]; // 320 floats
    __shared__ float s_x_out[32];

    const int tid = threadIdx.x; // 0..1023
    const int bx  = blockIdx.x;  // 0..79
    const int m   = blockIdx.y;  // 0..M-1

    const float* h_m      = h      + (size_t)m * QWEN4_HC_RANK;
    const float* normed_m = normed + (size_t)m * QWEN4_STREAM_TOTAL;
    float*       x_out_m  = x_out  + (size_t)m * QWEN4_N_EMBD;

    if (tid < QWEN4_HC_RANK) {
        sh[tid] = h_m[tid];
    }
    __syncthreads();

    const int c      = tid >> 5;       // 0..31 (one column per 32-lane warp)
    const int s      = (tid >> 3) & 3; // 0..3  (one stream per 8-lane subwarp)
    const int k_lane = tid & 7;        // 0..7
    const int col    = bx * 32 + c;

    const uint4* r_s  = (const uint4*)(up_bf16 + (size_t)(s * QWEN4_N_EMBD + col) * QWEN4_HC_RANK);
    const float4* sh4 = (const float4*)sh;

    float a = 0.0f;
    #pragma unroll
    for (int k = k_lane; k < QWEN4_HC_RANK / 8; k += 8) {
        const float4 h0 = sh4[k * 2 + 0];
        const float4 h1 = sh4[k * 2 + 1];
        a += dot_bf16x8_fp32x8(r_s[k], h0, h1);
    }

    a = subwarp_sum<8>(a);

    const float g = 1.0f / (1.0f + expf(-a));
    float val = g * normed_m[s * QWEN4_N_EMBD + col];
    val += __shfl_xor_sync(0xFFFFFFFFu, val, 8, 32);
    val += __shfl_xor_sync(0xFFFFFFFFu, val, 16, 32);

    if ((tid & 31) == 0) {
        const float out_v = 0.25f * val;
        x_out_m[col] = out_v;
        s_x_out[c]   = out_v;
    }

    if (x_q8_1_out != nullptr) {
        __syncthreads();
        if (tid < 32) {
            const float xi = q8_1_finite(s_x_out[tid]);
            float amax = fabsf(xi);
            float sum  = xi;
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask, 32));
                sum += __shfl_xor_sync(0xFFFFFFFFu, sum, mask, 32);
            }
            const float d = amax * (1.0f / 127.0f);
            const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / d);
            const half2 ds_val = make_half2(__float2half(d), __float2half(sum));
            block_q8_1* blk = x_q8_1_out + m * (QWEN4_N_EMBD / 32) + bx;
            blk->qs[tid] = q;
            if (tid == 0) {
                blk->ds = ds_val;
            }
            if (x_q8_1_peer_out != nullptr) {
                block_q8_1* pblk = x_q8_1_peer_out + m * (QWEN4_N_EMBD / 32) + bx;
                pblk->qs[tid] = q;
                if (tid == 0) {
                    pblk->ds = ds_val;
                }
            }
        }
    }
}

// Batched Tiled Up-Projection + 4-Stream Sigmoid-Gated Mean + Optional Q8_1 Quantization (Prefill M > 1)
// Launch: grid = dim3(80, (M + TILE_M - 1) / TILE_M), block = 256
// Loads up_bf16[4, 2560, 320] ONCE per TILE_M tokens (cutting 6.55 MB/call BF16 weight reads by 8x!)
template <int TILE_M>
__global__ void __launch_bounds__(256) gr_up_mean_q8_1_tileM_kernel(
    const uint16_t* __restrict__ up_bf16,
    const float* __restrict__ h,
    const float* __restrict__ normed,
    float* __restrict__ x_out,
    block_q8_1* __restrict__ x_q8_1_out,
    int M)
{
    __shared__ float sh[TILE_M][QWEN4_HC_RANK]; // 8 * 320 floats = 10 KB
    __shared__ float s_x_out[TILE_M][32];       // 8 * 32 floats  = 1 KB

    const int tid     = threadIdx.x;
    const int bx      = blockIdx.x; // 0..79
    const int m_base  = blockIdx.y * TILE_M;
    const int m_count = (M - m_base < TILE_M) ? (M - m_base) : TILE_M;
    if (m_count <= 0) return;

    for (int m = 0; m < m_count; ++m) {
        const float* h_m = h + (size_t)(m_base + m) * QWEN4_HC_RANK;
        for (int i = tid; i < QWEN4_HC_RANK; i += 256) {
            sh[m][i] = h_m[i];
        }
    }
    __syncthreads();

    const int c        = tid >> 3; // 0..31 (one column per 8-lane sub-warp)
    const int sub_lane = tid & 7;  // 0..7
    const int col      = bx * 32 + c;

    float val[TILE_M];
    #pragma unroll
    for (int m = 0; m < TILE_M; ++m) val[m] = 0.0f;

    #pragma unroll
    for (int s = 0; s < QWEN4_N_STREAMS; ++s) {
        const uint4* rs = (const uint4*)(up_bf16 + (size_t)(s * QWEN4_N_EMBD + col) * QWEN4_HC_RANK);
        float a[TILE_M];
        #pragma unroll
        for (int m = 0; m < TILE_M; ++m) a[m] = 0.0f;

        #pragma unroll
        for (int k = sub_lane; k < QWEN4_HC_RANK / 8; k += 8) {
            const uint4 wv = rs[k];
            const float w0 = __uint_as_float(wv.x << 16);
            const float w1 = __uint_as_float(wv.x & 0xFFFF0000u);
            const float w2 = __uint_as_float(wv.y << 16);
            const float w3 = __uint_as_float(wv.y & 0xFFFF0000u);
            const float w4 = __uint_as_float(wv.z << 16);
            const float w5 = __uint_as_float(wv.z & 0xFFFF0000u);
            const float w6 = __uint_as_float(wv.w << 16);
            const float w7 = __uint_as_float(wv.w & 0xFFFF0000u);
            #pragma unroll
            for (int m = 0; m < TILE_M; ++m) {
                if (m < m_count) {
                    const float4* shm4 = (const float4*)sh[m];
                    const float4 h0 = shm4[k * 2 + 0];
                    const float4 h1 = shm4[k * 2 + 1];
                    a[m] += w0 * h0.x + w1 * h0.y + w2 * h0.z + w3 * h0.w
                          + w4 * h1.x + w5 * h1.y + w6 * h1.z + w7 * h1.w;
                }
            }
        }

        #pragma unroll
        for (int m = 0; m < TILE_M; ++m) {
            a[m] = subwarp_sum<8>(a[m]);
            if (sub_lane == 0 && m < m_count) {
                const float g = 1.0f / (1.0f + expf(-a[m]));
                val[m] += g * normed[(size_t)(m_base + m) * QWEN4_STREAM_TOTAL + s * QWEN4_N_EMBD + col];
            }
        }
    }

    if (sub_lane == 0) {
        #pragma unroll
        for (int m = 0; m < TILE_M; ++m) {
            if (m < m_count) {
                const float out_v = 0.25f * val[m];
                x_out[(size_t)(m_base + m) * QWEN4_N_EMBD + col] = out_v;
                s_x_out[m][c] = out_v;
            }
        }
    }

    if (x_q8_1_out != nullptr) {
        __syncthreads();
        const int warp = tid >> 5; // 0..7
        const int lane = tid & 31; // 0..31
        for (int m = warp; m < m_count; m += 8) {
            const float xi = q8_1_finite(s_x_out[m][lane]);
            float amax = fabsf(xi);
            float sum  = xi;
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask, 32));
                sum += __shfl_xor_sync(0xFFFFFFFFu, sum, mask, 32);
            }
            const float d = amax * (1.0f / 127.0f);
            const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / d);
            block_q8_1* blk = x_q8_1_out + (size_t)(m_base + m) * (QWEN4_N_EMBD / 32) + bx;
            blk->qs[lane] = q;
            if (lane == 0) {
                blk->ds = make_half2(__float2half(d), __float2half(sum));
            }
        }
    }
}

// Fast 3-block float4 residual update using precomputed gates[m, 0..3]
// Launch: grid = dim3(3, M), block = 256
__global__ void gr_apply_write_vec4_kernel(
    float* __restrict__ R,
    const float* __restrict__ sub_out,
    const float* __restrict__ gates)
{
    const int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx4 >= QWEN4_N_EMBD / 4) return;
    const int m = blockIdx.y;

    const float4 g   = ((const float4*)gates)[m];
    const float4 sub = ((const float4*)(sub_out + (size_t)m * QWEN4_N_EMBD))[idx4];
    float* r_m = R + (size_t)m * QWEN4_STREAM_TOTAL;

    float4* r0 = ((float4*)(r_m + 0 * QWEN4_N_EMBD)) + idx4;
    float4* r1 = ((float4*)(r_m + 1 * QWEN4_N_EMBD)) + idx4;
    float4* r2 = ((float4*)(r_m + 2 * QWEN4_N_EMBD)) + idx4;
    float4* r3 = ((float4*)(r_m + 3 * QWEN4_N_EMBD)) + idx4;

    float4 v0 = *r0; v0.x += g.x * sub.x; v0.y += g.x * sub.y; v0.z += g.x * sub.z; v0.w += g.x * sub.w; *r0 = v0;
    float4 v1 = *r1; v1.x += g.y * sub.x; v1.y += g.y * sub.y; v1.z += g.y * sub.z; v1.w += g.y * sub.w; *r1 = v1;
    float4 v2 = *r2; v2.x += g.z * sub.x; v2.y += g.z * sub.y; v2.z += g.z * sub.z; v2.w += g.z * sub.w; *r2 = v2;
    float4 v3 = *r3; v3.x += g.w * sub.x; v3.y += g.w * sub.y; v3.z += g.w * sub.z; v3.w += g.w * sub.w; *r3 = v3;
}

// Fallback 4 blocks (one per stream) when gates were not precomputed
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

// 80 * M blocks x 32 threads: dequantize M * 80 IQ4_NL blocks -> M * 2560 floats + optional fused Q8_1 quantization
__global__ void ple_iq4nl_dequant_kernel(
    const block_iq4_nl* __restrict__ src,
    float* __restrict__ dst,
    block_q8_1* __restrict__ q8_out = nullptr)
{
    const int b    = blockIdx.x;  // 0 .. M*80 - 1
    const int lane = threadIdx.x; // 0..31
    const float d  = __half2float(src[b].d);
    const uint8_t q = src[b].qs[lane & 15];
    const float val = d * (float)kvalues_iq4nl[(lane < 16) ? (q & 0x0F) : (q >> 4)];
    dst[b * 32 + lane] = val;

    if (q8_out != nullptr) {
        const float xi = q8_1_finite(val);
        float amax = fabsf(xi);
        float sum  = xi;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask, 32));
            sum += __shfl_xor_sync(0xFFFFFFFFu, sum, mask, 32);
        }
        const float sc = amax / 127.0f;
        const int8_t qv = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / sc);
        q8_out[b].qs[lane] = qv;
        if (lane == 0) {
            q8_out[b].ds = make_half2(__float2half(sc), __float2half(sum));
        }
    }
}

// dim3(4, M) blocks (c = 0..3, m = 0..M-1), 256 threads: grouped RMSNorm of key[m, c] & R[m, c], signed-sqrt sigmoid gate, gated val & conv norm
// When conv_w_f16 != nullptr && hist != nullptr (M == 1 decode), fuses the dilated causal conv1d update directly into R & hist!
__global__ void ple_gate_kernel(
    float* __restrict__ R,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ nk,
    const float* __restrict__ nq,
    const float* __restrict__ nc,
    float* __restrict__ gated,
    float* __restrict__ normd,
    float eps,
    const half* __restrict__ conv_w_f16 = nullptr,
    float* __restrict__ hist = nullptr)
{
    __shared__ float smem[QWEN4_GR_WARPS];
    const int c = blockIdx.x; // 0..3
    const int m = blockIdx.y; // 0..M-1
    const int stream_off = c * QWEN4_N_EMBD;
    const size_t tok_off = (size_t)m * QWEN4_STREAM_TOTAL + stream_off;
    const float* val_m   = value + (size_t)m * QWEN4_N_EMBD;

    float sk = 0.0f, sq = 0.0f;
    for (int i = threadIdx.x; i < QWEN4_N_EMBD; i += blockDim.x) {
        const float kv = key[tok_off + i];
        const float qv = R[tok_off + i];
        sk += kv * kv;
        sq += qv * qv;
    }
    const float rk = rsqrtf(block_sum<QWEN4_GR_WARPS>(sk, smem) / (float)QWEN4_N_EMBD + eps);
    const float rq = rsqrtf(block_sum<QWEN4_GR_WARPS>(sq, smem) / (float)QWEN4_N_EMBD + eps);

    float dot = 0.0f;
    for (int i = threadIdx.x; i < QWEN4_N_EMBD; i += blockDim.x) {
        const float kn = key[tok_off + i] * rk * nk[stream_off + i];
        const float qn = R[tok_off + i]   * rq * nq[stream_off + i];
        dot += kn * qn;
    }
    const float s = block_sum<QWEN4_GR_WARPS>(dot, smem) * rsqrtf((float)QWEN4_N_EMBD);
    const float sgn = (s > 0.0f) ? 1.0f : ((s < 0.0f) ? -1.0f : 0.0f);
    const float gate = 1.0f / (1.0f + expf(-sgn * sqrtf(fmaxf(fabsf(s), 1e-6f))));

    float sg = 0.0f;
    for (int i = threadIdx.x; i < QWEN4_N_EMBD; i += blockDim.x) {
        const float gv = val_m[i] * gate;
        if (conv_w_f16 == nullptr) gated[tok_off + i] = gv;
        sg += gv * gv;
    }
    const float rg = rsqrtf(block_sum<QWEN4_GR_WARPS>(sg, smem) / (float)QWEN4_N_EMBD + eps);
    if (conv_w_f16 != nullptr && hist != nullptr) {
        for (int i = threadIdx.x; i < QWEN4_N_EMBD; i += blockDim.x) {
            const int ch   = stream_off + i;
            const float gv = val_m[i] * gate;
            const float nv = gv * rg * nc[ch];
            const half* w  = conv_w_f16 + ch * 4;
            const float p0 = hist[0 * QWEN4_STREAM_TOTAL + ch];
            const float p1 = hist[1 * QWEN4_STREAM_TOTAL + ch];
            const float p2 = hist[2 * QWEN4_STREAM_TOTAL + ch];
            const float p3 = hist[3 * QWEN4_STREAM_TOTAL + ch];
            const float p4 = hist[4 * QWEN4_STREAM_TOTAL + ch];
            const float p5 = hist[5 * QWEN4_STREAM_TOTAL + ch];
            const float p6 = hist[6 * QWEN4_STREAM_TOTAL + ch];
            const float p7 = hist[7 * QWEN4_STREAM_TOTAL + ch];
            const float p8 = hist[8 * QWEN4_STREAM_TOTAL + ch];
            const float sc = __half2float(w[0]) * p0 + __half2float(w[1]) * p3 + __half2float(w[2]) * p6 + __half2float(w[3]) * nv;
            R[ch] += gv + (sc / (1.0f + expf(-sc)));
            hist[0 * QWEN4_STREAM_TOTAL + ch] = p1;
            hist[1 * QWEN4_STREAM_TOTAL + ch] = p2;
            hist[2 * QWEN4_STREAM_TOTAL + ch] = p3;
            hist[3 * QWEN4_STREAM_TOTAL + ch] = p4;
            hist[4 * QWEN4_STREAM_TOTAL + ch] = p5;
            hist[5 * QWEN4_STREAM_TOTAL + ch] = p6;
            hist[6 * QWEN4_STREAM_TOTAL + ch] = p7;
            hist[7 * QWEN4_STREAM_TOTAL + ch] = p8;
            hist[8 * QWEN4_STREAM_TOTAL + ch] = nv;
        }
    } else {
        for (int i = threadIdx.x; i < QWEN4_N_EMBD; i += blockDim.x) {
            normd[tok_off + i] = gated[tok_off + i] * rg * nc[stream_off + i];
        }
    }
}

// 40 blocks x 256 threads (ch = 0..10239): steps dilated causal conv1d over M tokens in registers and updates R + hist
__global__ void ple_conv_update_kernel(
    float* __restrict__ R,
    const float* __restrict__ gated,
    const float* __restrict__ normd,
    const half* __restrict__ conv_w_f16,
    float* __restrict__ hist,
    int M)
{
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= QWEN4_STREAM_TOTAL) return;

    float pad[QWEN4_PLE_HIST + 1];
    #pragma unroll
    for (int i = 0; i < QWEN4_PLE_HIST; ++i) {
        pad[i] = hist[i * QWEN4_STREAM_TOTAL + ch];
    }

    const half* w = conv_w_f16 + ch * 4;
    const float w0 = __half2float(w[0]);
    const float w1 = __half2float(w[1]);
    const float w2 = __half2float(w[2]);
    const float w3 = __half2float(w[3]);

    for (int m = 0; m < M; ++m) {
        const size_t off = (size_t)m * QWEN4_STREAM_TOTAL + ch;
        pad[QWEN4_PLE_HIST] = normd[off];

        const float s =
            w0 * pad[QWEN4_PLE_HIST - 3 * QWEN4_PLE_DIL] +
            w1 * pad[QWEN4_PLE_HIST - 2 * QWEN4_PLE_DIL] +
            w2 * pad[QWEN4_PLE_HIST - 1 * QWEN4_PLE_DIL] +
            w3 * pad[QWEN4_PLE_HIST];
        const float silu_s = s / (1.0f + expf(-s));
        R[off] += gated[off] + silu_s;

        #pragma unroll
        for (int i = 0; i < QWEN4_PLE_HIST; ++i) {
            pad[i] = pad[i + 1];
        }
    }

    #pragma unroll
    for (int i = 0; i < QWEN4_PLE_HIST; ++i) {
        hist[i * QWEN4_STREAM_TOTAL + ch] = pad[i];
    }
}

// ============================================================================
// 36 Gated DeltaNet (GDN) Layers (Fused Preprocess + Single-Pass Register State)
// ============================================================================

constexpr int QWEN4_GDN_N_K_HEAD = 16;
constexpr int QWEN4_GDN_N_V_HEAD = 48;
constexpr int QWEN4_GDN_HEAD_DIM = 128;
constexpr int QWEN4_GDN_QK_DIM   = QWEN4_GDN_N_K_HEAD * QWEN4_GDN_HEAD_DIM; // 2048
constexpr int QWEN4_GDN_V_DIM    = QWEN4_GDN_N_V_HEAD * QWEN4_GDN_HEAD_DIM; // 6144
constexpr int QWEN4_GDN_QKV_DIM  = 2 * QWEN4_GDN_QK_DIM + QWEN4_GDN_V_DIM;  // 10240

// Fused single-token GDN causal conv1d + SiLU + Q/K Head L2Norm
// Launch: <<<80, 128>>> (blocks 0..15 = Q heads, 16..31 = K heads, 32..79 = V heads)
__global__ void __launch_bounds__(128) gdn_conv_silu_l2norm_kernel(
    float* __restrict__ conv_state,
    const float* __restrict__ qkv,
    const float* __restrict__ conv_w,
    float* __restrict__ qkv_conv,
    float eps)
{
    __shared__ float smem[4];
    const int b = blockIdx.x;  // 0..79
    const int d = threadIdx.x; // 0..127
    const int c = b * QWEN4_GDN_HEAD_DIM + d;

    const float s0 = conv_state[0 * QWEN4_GDN_QKV_DIM + c];
    const float s1 = conv_state[1 * QWEN4_GDN_QKV_DIM + c];
    const float s2 = conv_state[2 * QWEN4_GDN_QKV_DIM + c];
    const float s3 = qkv[c];

    conv_state[0 * QWEN4_GDN_QKV_DIM + c] = s1;
    conv_state[1 * QWEN4_GDN_QKV_DIM + c] = s2;
    conv_state[2 * QWEN4_GDN_QKV_DIM + c] = s3;

    const float4 w = ((const float4*)conv_w)[c];
    const float v = s0 * w.x + s1 * w.y + s2 * w.z + s3 * w.w;
    float act = v / (1.0f + expf(-v));

    if (b < 2 * QWEN4_GDN_N_K_HEAD) {
        const float ss = block_sum<4>(act * act, smem);
        const float q_scale = (b < QWEN4_GDN_N_K_HEAD) ? rsqrtf((float)QWEN4_GDN_HEAD_DIM) : 1.0f;
        act = act * rsqrtf(ss + eps) * q_scale;
    }
    qkv_conv[c] = act;
}

// Batched M-token GDN causal conv1d + SiLU + Fused In-Place Q/K L2-Normalization
// Launch: <<<QWEN4_GDN_QKV_DIM / 128, 128>>> (80 blocks of 128 threads: b=0..15 Q heads, b=16..31 K heads, b=32..79 V heads)
__global__ void __launch_bounds__(128) gdn_conv_silu_batchM_kernel(
    float* __restrict__ conv_state,
    const float* __restrict__ qkv,
    const float* __restrict__ conv_w,
    float* __restrict__ qkv_conv,
    int M,
    float eps)
{
    __shared__ float smem[4];
    const int b = blockIdx.x; // 0..79
    const int c = b * 128 + threadIdx.x;
    if (c >= QWEN4_GDN_QKV_DIM) return;

    float s0 = conv_state[0 * QWEN4_GDN_QKV_DIM + c];
    float s1 = conv_state[1 * QWEN4_GDN_QKV_DIM + c];
    float s2 = conv_state[2 * QWEN4_GDN_QKV_DIM + c];
    const float4 w = ((const float4*)conv_w)[c];
    const bool is_qk = (b < 2 * QWEN4_GDN_N_K_HEAD);
    const float q_scale = (b < QWEN4_GDN_N_K_HEAD) ? rsqrtf((float)QWEN4_GDN_HEAD_DIM) : 1.0f;

    if (is_qk) {
        for (int m = 0; m < M; ++m) {
            const size_t off = (size_t)m * QWEN4_GDN_QKV_DIM + c;
            const float s3 = qkv[off];
            const float v = s0 * w.x + s1 * w.y + s2 * w.z + s3 * w.w;
            float act = v / (1.0f + __expf(-v));
            const float ss = block_sum<4>(act * act, smem);
            qkv_conv[off] = act * rsqrtf(ss + eps) * q_scale;
            s0 = s1;
            s1 = s2;
            s2 = s3;
        }
    } else {
        for (int m = 0; m < M; ++m) {
            const size_t off = (size_t)m * QWEN4_GDN_QKV_DIM + c;
            const float s3 = qkv[off];
            const float v = s0 * w.x + s1 * w.y + s2 * w.z + s3 * w.w;
            qkv_conv[off] = v / (1.0f + __expf(-v));
            s0 = s1;
            s1 = s2;
            s2 = s3;
        }
    }

    conv_state[0 * QWEN4_GDN_QKV_DIM + c] = s0;
    conv_state[1 * QWEN4_GDN_QKV_DIM + c] = s1;
    conv_state[2 * QWEN4_GDN_QKV_DIM + c] = s2;
}

// Vectorized Fused alpha + beta BF16 GEMV + gate/beta activation
// Launch: grid = dim3(6, M), block = dim3(32, 8) (8 warps per block = 8 V heads per block)
__global__ void __launch_bounds__(256) gdn_alpha_beta_gemv_kernel(
    const uint16_t* __restrict__ w_alpha_bf16,
    const uint16_t* __restrict__ w_beta_bf16,
    const float* __restrict__ x,
    const float* __restrict__ ssm_a,
    const float* __restrict__ dt_bias,
    float* __restrict__ gate,
    float* __restrict__ beta)
{
    __shared__ float s_x[QWEN4_N_EMBD];
    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int lane = threadIdx.x;
    const int m    = blockIdx.y;

    const float4* x_in4 = (const float4*)(x + (size_t)m * QWEN4_N_EMBD);
    for (int i = tid; i < QWEN4_N_EMBD / 4; i += 256) {
        ((float4*)s_x)[i] = x_in4[i];
    }
    __syncthreads();

    const int h = blockIdx.x * 8 + threadIdx.y; // 0..47
    if (h >= QWEN4_GDN_N_V_HEAD) return;

    const uint4*  wa4 = (const uint4*)(w_alpha_bf16 + (size_t)h * QWEN4_N_EMBD);
    const uint4*  wb4 = (const uint4*)(w_beta_bf16  + (size_t)h * QWEN4_N_EMBD);
    const float4* sx4 = (const float4*)s_x;

    float acc_a = 0.0f, acc_b = 0.0f;
    #pragma unroll
    for (int j = lane; j < QWEN4_N_EMBD / 8; j += 32) {
        const float4 x0 = sx4[j * 2 + 0];
        const float4 x1 = sx4[j * 2 + 1];
        acc_a += dot_bf16x8_fp32x8(wa4[j], x0, x1);
        acc_b += dot_bf16x8_fp32x8(wb4[j], x0, x1);
    }
    acc_a = warp_sum(acc_a);
    acc_b = warp_sum(acc_b);

    if (lane == 0) {
        const float x_dt = acc_a + dt_bias[h];
        const float sp = (x_dt > 20.0f) ? x_dt : logf(1.0f + expf(x_dt));
        gate[(size_t)m * QWEN4_GDN_N_V_HEAD + h] = ssm_a[h] * sp;
        beta[(size_t)m * QWEN4_GDN_N_V_HEAD + h] = 1.0f / (1.0f + expf(-acc_b));
    }
}

// Batched Tile-M Fused alpha + beta BF16 GEMM + gate/beta activation for Prefill (cuts w_alpha/w_beta BF16 reads by TILE_M!)
template <int TILE_M>
__global__ void __launch_bounds__(256) gdn_alpha_beta_tileM_kernel(
    const uint16_t* __restrict__ w_alpha_bf16,
    const uint16_t* __restrict__ w_beta_bf16,
    const float* __restrict__ x,
    const float* __restrict__ ssm_a,
    const float* __restrict__ dt_bias,
    float* __restrict__ gate,
    float* __restrict__ beta,
    int M)
{
    __shared__ float s_x[TILE_M][512];
    const int tid     = threadIdx.y * 32 + threadIdx.x;
    const int lane    = threadIdx.x;
    const int h       = blockIdx.x * 8 + threadIdx.y; // 0..47
    const int m_base  = blockIdx.y * TILE_M;
    const int m_count = (M - m_base < TILE_M) ? (M - m_base) : TILE_M;
    if (m_count <= 0) return;

    float acc_a[TILE_M];
    float acc_b[TILE_M];
    #pragma unroll
    for (int m = 0; m < TILE_M; ++m) {
        acc_a[m] = 0.0f;
        acc_b[m] = 0.0f;
    }

    const uint16_t* wa_row = (h < QWEN4_GDN_N_V_HEAD) ? (w_alpha_bf16 + (size_t)h * QWEN4_N_EMBD) : nullptr;
    const uint16_t* wb_row = (h < QWEN4_GDN_N_V_HEAD) ? (w_beta_bf16  + (size_t)h * QWEN4_N_EMBD) : nullptr;

    for (int cBase = 0; cBase < QWEN4_N_EMBD; cBase += 512) {
        const int m_half = tid >> 7;
        const int col_f4 = tid & 127;
        #pragma unroll
        for (int mBase = 0; mBase < TILE_M; mBase += 2) {
            const int m = mBase + m_half;
            if (m < TILE_M) {
                if (m < m_count) {
                    const float4* xm_f4 = (const float4*)(x + (size_t)(m_base + m) * QWEN4_N_EMBD + cBase);
                    ((float4*)s_x[m])[col_f4] = xm_f4[col_f4];
                } else {
                    ((float4*)s_x[m])[col_f4] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        }
        __syncthreads();

        if (h < QWEN4_GDN_N_V_HEAD) {
            const uint4* wa_u4 = (const uint4*)(wa_row + cBase);
            const uint4* wb_u4 = (const uint4*)(wb_row + cBase);
            #pragma unroll
            for (int j = lane; j < 512 / 8; j += 32) {
                const uint4 wva = wa_u4[j];
                const float wa0 = __uint_as_float(wva.x << 16);
                const float wa1 = __uint_as_float(wva.x & 0xFFFF0000u);
                const float wa2 = __uint_as_float(wva.y << 16);
                const float wa3 = __uint_as_float(wva.y & 0xFFFF0000u);
                const float wa4 = __uint_as_float(wva.z << 16);
                const float wa5 = __uint_as_float(wva.z & 0xFFFF0000u);
                const float wa6 = __uint_as_float(wva.w << 16);
                const float wa7 = __uint_as_float(wva.w & 0xFFFF0000u);

                const uint4 wvb = wb_u4[j];
                const float wb0 = __uint_as_float(wvb.x << 16);
                const float wb1 = __uint_as_float(wvb.x & 0xFFFF0000u);
                const float wb2 = __uint_as_float(wvb.y << 16);
                const float wb3 = __uint_as_float(wvb.y & 0xFFFF0000u);
                const float wb4 = __uint_as_float(wvb.z << 16);
                const float wb5 = __uint_as_float(wvb.z & 0xFFFF0000u);
                const float wb6 = __uint_as_float(wvb.w << 16);
                const float wb7 = __uint_as_float(wvb.w & 0xFFFF0000u);

                #pragma unroll
                for (int m = 0; m < TILE_M; ++m) {
                    const float4* sx_f4 = (const float4*)s_x[m];
                    const float4 x0 = sx_f4[j * 2 + 0];
                    const float4 x1 = sx_f4[j * 2 + 1];
                    acc_a[m] += wa0 * x0.x + wa1 * x0.y + wa2 * x0.z + wa3 * x0.w
                              + wa4 * x1.x + wa5 * x1.y + wa6 * x1.z + wa7 * x1.w;
                    acc_b[m] += wb0 * x0.x + wb1 * x0.y + wb2 * x0.z + wb3 * x0.w
                              + wb4 * x1.x + wb5 * x1.y + wb6 * x1.z + wb7 * x1.w;
                }
            }
        }
        __syncthreads();
    }

    if (h < QWEN4_GDN_N_V_HEAD) {
        const float a_h = ssm_a[h];
        const float b_h = dt_bias[h];
        #pragma unroll
        for (int m = 0; m < TILE_M; ++m) {
            acc_a[m] = warp_sum(acc_a[m]);
            acc_b[m] = warp_sum(acc_b[m]);
            if (lane == 0 && m < m_count) {
                const float x_dt = acc_a[m] + b_h;
                const float sp = (x_dt > 20.0f) ? x_dt : logf(1.0f + expf(x_dt));
                gate[(size_t)(m_base + m) * QWEN4_GDN_N_V_HEAD + h] = a_h * sp;
                beta[(size_t)(m_base + m) * QWEN4_GDN_N_V_HEAD + h] = 1.0f / (1.0f + expf(-acc_b[m]));
            }
        }
    }
}

// Fused GDN Decode Input Projections:
//   Blocks 0 .. 639:   w_qkv  [10240, 2560] -> qkv_conv[10240] (16 rows/block with LANES=16 + fused causal conv1d + SiLU!)
//   Blocks 640 .. 1023: w_gate [6144,  2560] -> z[6144]         (16 rows/block with LANES=16)
//   Blocks 1024 .. 1029: w_alpha_bf16 + w_beta_bf16 [48, 2560] -> gate[48] + beta[48] (8 heads/block)
// Launch: <<<1030, dim3(32, 8)>>>
template <int LANES>
__global__ void __launch_bounds__(256) gdn_in_proj_fused_kernel(
    const float* __restrict__ x_f32,
    const block_q8_1* __restrict__ x_q8_1,
    const uint8_t* __restrict__ w_qkv, int32_t qkv_type, size_t qkv_row_bytes,
    const uint8_t* __restrict__ w_gate, int32_t gate_type, size_t gate_row_bytes,
    const uint16_t* __restrict__ w_alpha_bf16,
    const uint16_t* __restrict__ w_beta_bf16,
    const float* __restrict__ w_conv_f32,
    const float* __restrict__ ssm_a,
    const float* __restrict__ dt_bias,
    float* __restrict__ conv_state,
    float* __restrict__ qkv_conv,
    float* __restrict__ z,
    float* __restrict__ gate,
    float* __restrict__ beta)
{
    __shared__ block_q8_1 s_x[80]; // 2,880 B -> 6 blocks/SM = 48 warps/SM (100% occupancy!)

    constexpr int ROWS_PER_BLOCK = 256 / LANES; // 16 when LANES=16
    constexpr int QKV_BLOCKS     = QWEN4_GDN_QKV_DIM / ROWS_PER_BLOCK; // 640
    constexpr int GATE_BLOCKS    = QWEN4_GDN_V_DIM   / ROWS_PER_BLOCK; // 384

    const int bx  = blockIdx.x;
    const int tid = threadIdx.y * 32 + threadIdx.x;

    if (bx < QKV_BLOCKS + GATE_BLOCKS) {
        copy_q8_1_smem(s_x, x_q8_1, QWEN4_N_EMBD / 32, tid);
        __syncthreads();

        const int sub_id   = tid / LANES;
        const int sub_lane = tid & (LANES - 1);
        if (bx < QKV_BLOCKS) {
            const int row = bx * ROWS_PER_BLOCK + sub_id;
            const float s3 = row_dot_any_subwarp<LANES>(
                qkv_type, w_qkv + (size_t)row * qkv_row_bytes, s_x, QWEN4_N_EMBD, sub_lane);
            if (sub_lane == 0) {
                const float s0 = conv_state[0 * QWEN4_GDN_QKV_DIM + row];
                const float s1 = conv_state[1 * QWEN4_GDN_QKV_DIM + row];
                const float s2 = conv_state[2 * QWEN4_GDN_QKV_DIM + row];
                conv_state[0 * QWEN4_GDN_QKV_DIM + row] = s1;
                conv_state[1 * QWEN4_GDN_QKV_DIM + row] = s2;
                conv_state[2 * QWEN4_GDN_QKV_DIM + row] = s3;
                const float4 w = ((const float4*)w_conv_f32)[row];
                const float v = s0 * w.x + s1 * w.y + s2 * w.z + s3 * w.w;
                qkv_conv[row] = v / (1.0f + expf(-v));
            }
        } else {
            const int row = (bx - QKV_BLOCKS) * ROWS_PER_BLOCK + sub_id;
            const float sum = row_dot_any_subwarp<LANES>(
                gate_type, w_gate + (size_t)row * gate_row_bytes, s_x, QWEN4_N_EMBD, sub_lane);
            if (sub_lane == 0) z[row] = sum;
        }
    } else {
        const int lane = threadIdx.x;
        const int h = (bx - (QKV_BLOCKS + GATE_BLOCKS)) * 8 + threadIdx.y; // 0..47
        if (h >= QWEN4_GDN_N_V_HEAD) return;

        const uint4*  wa4 = (const uint4*)(w_alpha_bf16 + (size_t)h * QWEN4_N_EMBD);
        const uint4*  wb4 = (const uint4*)(w_beta_bf16  + (size_t)h * QWEN4_N_EMBD);
        const float4* sx4 = (const float4*)x_f32;

        float acc_a = 0.0f, acc_b = 0.0f;
        #pragma unroll
        for (int j = lane; j < QWEN4_N_EMBD / 8; j += 32) {
            const float4 x0 = __ldg(sx4 + j * 2 + 0);
            const float4 x1 = __ldg(sx4 + j * 2 + 1);
            acc_a += dot_bf16x8_fp32x8(wa4[j], x0, x1);
            acc_b += dot_bf16x8_fp32x8(wb4[j], x0, x1);
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
}

// Single-Pass Register-Resident GDN State Recurrence + Head RMSNorm + Sigmoid Gate + Q8_1 Quantization
// Supports both M = 1 (decode) and M > 1 (batched prefill, keeping S[128, 128] in registers across all M steps!)
// Uses double-buffered q_sh/k_sh/s_qk_norm so rg=1 prefetches & L2-normalizes step (m+1) concurrently with step m's
// state update & smem_norm reduction, cutting __syncthreads() per token step from 4 down to 2!
// Launch: <<<48, dim3(128, 4)>>> (512 threads per block: j = 0..127, rg = 0..3; 48 blocks on 82 SMs = 1 block/SM = 128 regs/thread)
__global__ void __launch_bounds__(512, 1) gdn_recurrence_norm_q8_1_kernel(
    float* __restrict__ ssm_state,
    const float* __restrict__ qkv_conv,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const float* __restrict__ ssm_norm,
    float* __restrict__ z,
    block_q8_1* __restrict__ q8_out,
    int M,
    float eps,
    bool l2norm_qk = false)
{
    __shared__ float q_sh[2][QWEN4_GDN_HEAD_DIM];
    __shared__ float k_sh[2][QWEN4_GDN_HEAD_DIM];
    __shared__ float s_red_kv[4][QWEN4_GDN_HEAD_DIM];
    __shared__ float s_red_out[4][QWEN4_GDN_HEAD_DIM];
    __shared__ float smem_norm[4];
    __shared__ float s_qk_norm[2][3][4];

    const int h      = blockIdx.x;  // 0..47
    const int j      = threadIdx.x; // 0..127
    const int rg     = threadIdx.y; // 0..3
    const int qk_h   = h % QWEN4_GDN_N_K_HEAD;
    const int i_base = rg * 32;

    float* S = ssm_state + (size_t)h * (QWEN4_GDN_HEAD_DIM * QWEN4_GDN_HEAD_DIM);
    float s_reg[32];
    #pragma unroll
    for (int u = 0; u < 32; ++u) {
        s_reg[u] = S[(i_base + u) * QWEN4_GDN_HEAD_DIM + j];
    }

    const float ssm_norm_j = (rg == 0) ? ssm_norm[j] : 0.0f;

    auto load_qk_step = [&](int m_step, int buf) {
        const float* qkv_m = qkv_conv + (size_t)m_step * QWEN4_GDN_QKV_DIM;
        const float qv = qkv_m[qk_h * QWEN4_GDN_HEAD_DIM + j];
        const float kv = qkv_m[QWEN4_GDN_QK_DIM + qk_h * QWEN4_GDN_HEAD_DIM + j];
        q_sh[buf][j] = qv;
        k_sh[buf][j] = kv;
        const float kq_w = warp_sum(qv * kv);
        if (!l2norm_qk) {
            if ((j & 31) == 0) s_qk_norm[buf][2][j >> 5] = kq_w;
        } else {
            const float q_ss = warp_sum(qv * qv);
            const float k_ss = warp_sum(kv * kv);
            if ((j & 31) == 0) {
                s_qk_norm[buf][0][j >> 5] = q_ss;
                s_qk_norm[buf][1][j >> 5] = k_ss;
                s_qk_norm[buf][2][j >> 5] = kq_w;
            }
        }
    };

    if (M > 0) {
        if (rg == 0) {
            load_qk_step(0, 0);
        }
        __syncthreads();
    }

    for (int m = 0; m < M; ++m) {
        const int buf = m & 1;
        const float* qkv_m = qkv_conv + (size_t)m * QWEN4_GDN_QKV_DIM;

        const float q_scale = l2norm_qk
            ? (rsqrtf((s_qk_norm[buf][0][0] + s_qk_norm[buf][0][1] + s_qk_norm[buf][0][2] + s_qk_norm[buf][0][3]) + eps) * rsqrtf((float)QWEN4_GDN_HEAD_DIM))
            : 1.0f;
        const float k_scale = l2norm_qk
            ? rsqrtf((s_qk_norm[buf][1][0] + s_qk_norm[buf][1][1] + s_qk_norm[buf][1][2] + s_qk_norm[buf][1][3]) + eps)
            : 1.0f;
        const float kq_dot = (s_qk_norm[buf][2][0] + s_qk_norm[buf][2][1] + s_qk_norm[buf][2][2] + s_qk_norm[buf][2][3]) * (q_scale * k_scale);

        const float decay = expf(gate[(size_t)m * QWEN4_GDN_N_V_HEAD + h]);
        const float b     = beta[(size_t)m * QWEN4_GDN_N_V_HEAD + h];
        const float v_j   = qkv_m[2 * QWEN4_GDN_QK_DIM + h * QWEN4_GDN_HEAD_DIM + j];

        float kv_part = 0.0f;
        float qv_part = 0.0f;
        #pragma unroll
        for (int u = 0; u < 32; ++u) {
            const float s_dec = s_reg[u] * decay;
            s_reg[u] = s_dec;
            kv_part += s_dec * k_sh[buf][i_base + u];
            qv_part += s_dec * q_sh[buf][i_base + u];
        }
        s_red_kv[rg][j]  = kv_part;
        s_red_out[rg][j] = qv_part;
        __syncthreads();

        const float kv_j    = (s_red_kv[0][j]  + s_red_kv[1][j]  + s_red_kv[2][j]  + s_red_kv[3][j])  * k_scale;
        const float qv_j    = (s_red_out[0][j] + s_red_out[1][j] + s_red_out[2][j] + s_red_out[3][j]) * q_scale;
        const float delta_j = (v_j - kv_j) * b;
        const float dk_j    = delta_j * k_scale;
        const float out_j   = qv_j + delta_j * kq_dot;

        if (rg == 0) {
            const float w_ss = warp_sum(out_j * out_j);
            if ((j & 31) == 0) smem_norm[j >> 5] = w_ss;
        } else if (rg == 1 && (m + 1 < M)) {
            load_qk_step(m + 1, buf ^ 1);
        }

        if (m + 1 == M) {
            #pragma unroll
            for (int u = 0; u < 32; ++u) {
                S[(i_base + u) * QWEN4_GDN_HEAD_DIM + j] = s_reg[u] + k_sh[buf][i_base + u] * dk_j;
            }
            if (rg != 0) return;
            asm volatile("bar.sync 1, 128;");
        } else {
            #pragma unroll
            for (int u = 0; u < 32; ++u) {
                s_reg[u] += k_sh[buf][i_base + u] * dk_j;
            }
            __syncthreads();
        }

        if (rg == 0) {
            const float total_ss = smem_norm[0] + smem_norm[1] + smem_norm[2] + smem_norm[3];
            const float rstd = rsqrtf(total_ss * (1.0f / (float)QWEN4_GDN_HEAD_DIM) + eps);
            const size_t idx = (size_t)m * QWEN4_GDN_V_DIM + h * QWEN4_GDN_HEAD_DIM + j;
            const float normed = out_j * rstd * ssm_norm_j;
            const float zv = z[idx];
            const float gated_val = normed / (1.0f + expf(-zv));
            z[idx] = gated_val;

            if (q8_out != nullptr) {
                const int lane = j & 31;
                const float xi = q8_1_finite(gated_val);
                float amax = fabsf(xi);
                float sum  = xi;
                #pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1) {
                    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask, 32));
                    sum += __shfl_xor_sync(0xFFFFFFFFu, sum, mask, 32);
                }
                const float d = amax * (1.0f / 127.0f);
                const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / d);
                block_q8_1* blk = q8_out + (size_t)m * (QWEN4_GDN_V_DIM / 32) + h * 4 + (j >> 5);
                blk->qs[lane] = q;
                if (lane == 0) {
                    blk->ds = make_half2(__float2half(d), __float2half(sum));
                }
            }
        }
    }
}

// Warp-Self-Contained Zero-Barrier GDN State Recurrence for Batched Prefill (M > 1):
//   - Launch: <<<192, dim3(32, 8)>>> (4 blocks per V-head = 192 blocks across all 82 SMs; 8 warps/block = 4 cols/warp)
//   - Each 8-lane subwarp (j_sub = lane >> 3, sub_lane = lane & 7) owns 1 column j = j_base + j_sub (16 rows s_reg[4] float4s per lane).
//   - Row mapping i0 = (g * 8 + sub_lane) * 4 makes s_q/s_k shared memory reads 100% bank-conflict-free (stride = 1 float4 = 4 banks)!
//   - Uses pre-normalized Q/K from gdn_conv_silu_batchM_kernel, eliminating 96x redundant Q/K L2-norm reductions!
__global__ void __launch_bounds__(256) gdn_recurrence_warp_batchM_kernel(
    float* __restrict__ ssm_state,
    const float* __restrict__ qkv_conv,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ out_buf,
    int M,
    float eps)
{
    (void)eps;
    __shared__ float4 s_q[8][32];
    __shared__ float4 s_k[8][32];

    const int h        = blockIdx.x >> 2;             // 0..47
    const int j_blk    = (blockIdx.x & 3) * 32;       // 0, 32, 64, 96
    const int lane     = threadIdx.x;                 // 0..31
    const int warp     = threadIdx.y;                 // 0..7
    const int j_base   = j_blk + warp * 4;            // 0..124 (4 columns per warp)
    const int j_sub    = lane >> 3;                   // 0..3
    const int sub_lane = lane & 7;                    // 0..7
    const int j_col    = j_base + j_sub;              // column owned by this 8-lane subwarp
    const int i_base   = lane * 4;                    // 0..124 (4 rows per lane for q/k load)
    const int qk_h     = h % QWEN4_GDN_N_K_HEAD;

    float* S = ssm_state + (size_t)h * (QWEN4_GDN_HEAD_DIM * QWEN4_GDN_HEAD_DIM);
    float4 s_reg[4];
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        const int i0 = (g * 8 + sub_lane) * 4;
        s_reg[g] = make_float4(
            S[(i0 + 0) * QWEN4_GDN_HEAD_DIM + j_col],
            S[(i0 + 1) * QWEN4_GDN_HEAD_DIM + j_col],
            S[(i0 + 2) * QWEN4_GDN_HEAD_DIM + j_col],
            S[(i0 + 3) * QWEN4_GDN_HEAD_DIM + j_col]);
    }

    for (int m = 0; m < M; ++m) {
        const float* qkv_m = qkv_conv + (size_t)m * QWEN4_GDN_QKV_DIM;
        const float4 q4 = *(const float4*)(qkv_m + qk_h * QWEN4_GDN_HEAD_DIM + i_base);
        const float4 k4 = *(const float4*)(qkv_m + QWEN4_GDN_QK_DIM + qk_h * QWEN4_GDN_HEAD_DIM + i_base);
        const float kq_dot = warp_sum(q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w);

        s_q[warp][lane] = q4;
        s_k[warp][lane] = k4;
        __syncwarp();

        const float decay = __expf(gate[(size_t)m * QWEN4_GDN_N_V_HEAD + h]);
        const float b     = beta[(size_t)m * QWEN4_GDN_N_V_HEAD + h];
        const float v_col = qkv_m[2 * QWEN4_GDN_QK_DIM + h * QWEN4_GDN_HEAD_DIM + j_col];

        float kv = 0.0f, qv = 0.0f;
        float4 kg_reg[4];
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            const float4 qg = s_q[warp][g * 8 + sub_lane];
            const float4 kg = s_k[warp][g * 8 + sub_lane];
            kg_reg[g] = kg;
            s_reg[g].x *= decay; kv += s_reg[g].x * kg.x; qv += s_reg[g].x * qg.x;
            s_reg[g].y *= decay; kv += s_reg[g].y * kg.y; qv += s_reg[g].y * qg.y;
            s_reg[g].z *= decay; kv += s_reg[g].z * kg.z; qv += s_reg[g].z * qg.z;
            s_reg[g].w *= decay; kv += s_reg[g].w * kg.w; qv += s_reg[g].w * qg.w;
        }

        #pragma unroll
        for (int mask = 4; mask > 0; mask >>= 1) {
            kv += __shfl_xor_sync(0xFFFFFFFFu, kv, mask, 8);
            qv += __shfl_xor_sync(0xFFFFFFFFu, qv, mask, 8);
        }

        const float d = (v_col - kv) * b;
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            s_reg[g].x += kg_reg[g].x * d;
            s_reg[g].y += kg_reg[g].y * d;
            s_reg[g].z += kg_reg[g].z * d;
            s_reg[g].w += kg_reg[g].w * d;
        }

        if (sub_lane == 0) {
            out_buf[(size_t)m * QWEN4_GDN_V_DIM + h * QWEN4_GDN_HEAD_DIM + j_col] = qv + d * kq_dot;
        }
        __syncwarp();
    }

    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        const int i0 = (g * 8 + sub_lane) * 4;
        S[(i0 + 0) * QWEN4_GDN_HEAD_DIM + j_col] = s_reg[g].x;
        S[(i0 + 1) * QWEN4_GDN_HEAD_DIM + j_col] = s_reg[g].y;
        S[(i0 + 2) * QWEN4_GDN_HEAD_DIM + j_col] = s_reg[g].z;
        S[(i0 + 3) * QWEN4_GDN_HEAD_DIM + j_col] = s_reg[g].w;
    }
}

// Batched GDN Head RMSNorm + Sigmoid Gate + Q8_1 / BF16 Output across all (48, M) heads in parallel!
// Launch: <<<dim3(48, M), 128>>>
__global__ void __launch_bounds__(128) gdn_head_rmsnorm_gate_q8_1_batchM_kernel(
    const float* __restrict__ out_buf,
    const float* __restrict__ ssm_norm,
    float* __restrict__ z,
    block_q8_1* __restrict__ q8_out,
    float eps,
    uint16_t* __restrict__ bf16_out = nullptr)
{
    __shared__ float smem[4];
    const int h = blockIdx.x;  // 0..47
    const int m = blockIdx.y;  // 0..M-1
    const int j = threadIdx.x; // 0..127

    const size_t idx = (size_t)m * QWEN4_GDN_V_DIM + h * QWEN4_GDN_HEAD_DIM + j;
    const float out_j = out_buf[idx];
    const float ss = block_sum<4>(out_j * out_j, smem);
    const float rstd = rsqrtf(ss * (1.0f / (float)QWEN4_GDN_HEAD_DIM) + eps);
    const float normed = out_j * rstd * ssm_norm[j];
    const float zv = z[idx];
    const float gated_val = normed / (1.0f + expf(-zv));
    if (bf16_out != nullptr) {
        bf16_out[idx] = qwen4_fp32_to_bf16(gated_val);
    } else if (q8_out == nullptr) {
        z[idx] = gated_val;
    }

    if (q8_out != nullptr) {
        const int lane = j & 31;
        const float xi = q8_1_finite(gated_val);
        float amax = fabsf(xi);
        float sum  = xi;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask, 32));
            sum += __shfl_xor_sync(0xFFFFFFFFu, sum, mask, 32);
        }
        const float d = amax * (1.0f / 127.0f);
        const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / d);
        block_q8_1* blk = q8_out + (size_t)m * (QWEN4_GDN_V_DIM / 32) + h * 4 + (j >> 5);
        blk->qs[lane] = q;
        if (lane == 0) {
            blk->ds = make_half2(__float2half(d), __float2half(sum));
        }
    }
}

// ============================================================================
// 12 Query-Sparse Attention (QSA) Layers (GQA-12:1 Split-KV Flash-Decoding)
// ============================================================================

constexpr int QWEN4_QSA_N_HEAD           = 24;
constexpr int QWEN4_QSA_N_KV             = 2;
constexpr int QWEN4_QSA_HEAD_DIM         = 256;
constexpr int QWEN4_QSA_N_ROT            = 64;
constexpr float QWEN4_QSA_THETA          = 10000000.0f;
constexpr int QWEN4_QSA_IDX_HEADS        = 4;
constexpr int QWEN4_QSA_IDX_DIM          = 128;
constexpr int QWEN4_QSA_IDX_BLOCK        = 4;
constexpr int QWEN4_QSA_IDX_TOPK         = 2048;
constexpr int QWEN4_QSA_MAX_SEL          = QWEN4_QSA_IDX_TOPK + QWEN4_QSA_IDX_BLOCK - 1; // 2051
constexpr int QWEN4_QSA_SPARSE_THRESHOLD = 2304; // Full exact attention up to 2304 tokens (36 split chunks)
constexpr int QWEN4_QSA_SPLIT_CHUNK      = 64;

// Fused QSA Decode Input Projections:
//   Blocks 0 .. 767:   w_q          [12288, 2560] -> q_gate[12288]  (16 rows/block with LANES=16)
//   Blocks 768 .. 799: w_k          [512,   2560] -> k_cur[512]     (16 rows/block with LANES=16)
//   Blocks 800 .. 831: w_v          [512,   2560] -> v_cur[512]     (16 rows/block with LANES=16)
//   Blocks 832 .. 839: w_idx_k_bf16 [128,   2560] -> idx_raw[128]   (16 rows/block)
//   Blocks 840 .. 871: w_idx_q_bf16 [512,   2560] -> q_idx[512]     (16 rows/block, when w_idx_q_bf16 != nullptr)
// Launch: <<<840 or 872, dim3(32, 8)>>>
template <int LANES>
__global__ void __launch_bounds__(256) qsa_in_proj_fused_kernel(
    const float* __restrict__ x_f32,
    const block_q8_1* __restrict__ x_q8_1,
    const uint8_t* __restrict__ w_q, int32_t q_type, size_t q_row_bytes,
    const uint8_t* __restrict__ w_k, int32_t k_type, size_t k_row_bytes,
    const uint8_t* __restrict__ w_v, int32_t v_type, size_t v_row_bytes,
    const uint16_t* __restrict__ w_idx_k_bf16,
    const uint16_t* __restrict__ w_idx_q_bf16,
    float* __restrict__ q_gate,
    float* __restrict__ k_cur,
    float* __restrict__ v_cur,
    float* __restrict__ idx_raw,
    float* __restrict__ q_idx)
{
    __shared__ block_q8_1 s_x[80]; // 2,880 B -> 6 blocks/SM = 48 warps/SM (100% occupancy!)

    constexpr int ROWS_PER_BLOCK = 256 / LANES; // 16 when LANES=16
    constexpr int Q_BLOCKS     = (QWEN4_QSA_N_HEAD * 2 * QWEN4_QSA_HEAD_DIM) / ROWS_PER_BLOCK; // 768
    constexpr int K_BLOCKS     = (QWEN4_QSA_N_KV * QWEN4_QSA_HEAD_DIM)       / ROWS_PER_BLOCK; // 32
    constexpr int V_BLOCKS     = (QWEN4_QSA_N_KV * QWEN4_QSA_HEAD_DIM)       / ROWS_PER_BLOCK; // 32
    constexpr int QUANT_BLOCKS = Q_BLOCKS + K_BLOCKS + V_BLOCKS; // 832

    const int bx  = blockIdx.x;
    const int tid = threadIdx.y * 32 + threadIdx.x;

    if (bx < QUANT_BLOCKS) {
        copy_q8_1_smem(s_x, x_q8_1, QWEN4_N_EMBD / 32, tid);
        __syncthreads();

        const int sub_id   = tid / LANES;
        const int sub_lane = tid & (LANES - 1);
        if (bx < Q_BLOCKS) {
            const int row = bx * ROWS_PER_BLOCK + sub_id;
            const float sum = row_dot_any_subwarp<LANES>(
                q_type, w_q + (size_t)row * q_row_bytes, s_x, QWEN4_N_EMBD, sub_lane);
            if (sub_lane == 0) q_gate[row] = sum;
        } else if (bx < Q_BLOCKS + K_BLOCKS) {
            const int row = (bx - Q_BLOCKS) * ROWS_PER_BLOCK + sub_id;
            const float sum = row_dot_any_subwarp<LANES>(
                k_type, w_k + (size_t)row * k_row_bytes, s_x, QWEN4_N_EMBD, sub_lane);
            if (sub_lane == 0) k_cur[row] = sum;
        } else {
            const int row = (bx - (Q_BLOCKS + K_BLOCKS)) * ROWS_PER_BLOCK + sub_id;
            const float sum = row_dot_any_subwarp<LANES>(
                v_type, w_v + (size_t)row * v_row_bytes, s_x, QWEN4_N_EMBD, sub_lane);
            if (sub_lane == 0) v_cur[row] = sum;
        }
    } else {
        const int lane = threadIdx.x;
        const float4* sx4 = (const float4*)x_f32;
        const int bf16_bx = bx - QUANT_BLOCKS;
        if (bf16_bx < 8) {
            const int r0 = bf16_bx * 16 + threadIdx.y * 2 + 0; // 0..126
            const int r1 = r0 + 1;
            const uint4* w0 = (const uint4*)(w_idx_k_bf16 + (size_t)r0 * QWEN4_N_EMBD);
            const uint4* w1 = (const uint4*)(w_idx_k_bf16 + (size_t)r1 * QWEN4_N_EMBD);
            float acc0 = 0.0f, acc1 = 0.0f;
            #pragma unroll 4
            for (int j = lane; j < QWEN4_N_EMBD / 8; j += 32) {
                const float4 x0 = __ldg(sx4 + j * 2 + 0);
                const float4 x1 = __ldg(sx4 + j * 2 + 1);
                acc0 += dot_bf16x8_fp32x8(w0[j], x0, x1);
                acc1 += dot_bf16x8_fp32x8(w1[j], x0, x1);
            }
            acc0 = warp_sum(acc0);
            acc1 = warp_sum(acc1);
            if (lane == 0) {
                idx_raw[r0] = acc0;
                idx_raw[r1] = acc1;
            }
        } else if (w_idx_q_bf16 != nullptr) {
            const int r0 = (bf16_bx - 8) * 16 + threadIdx.y * 2 + 0; // 0..510
            const int r1 = r0 + 1;
            const uint4* w0 = (const uint4*)(w_idx_q_bf16 + (size_t)r0 * QWEN4_N_EMBD);
            const uint4* w1 = (const uint4*)(w_idx_q_bf16 + (size_t)r1 * QWEN4_N_EMBD);
            float acc0 = 0.0f, acc1 = 0.0f;
            #pragma unroll 4
            for (int j = lane; j < QWEN4_N_EMBD / 8; j += 32) {
                const float4 x0 = __ldg(sx4 + j * 2 + 0);
                const float4 x1 = __ldg(sx4 + j * 2 + 1);
                acc0 += dot_bf16x8_fp32x8(w0[j], x0, x1);
                acc1 += dot_bf16x8_fp32x8(w1[j], x0, x1);
            }
            acc0 = warp_sum(acc0);
            acc1 = warp_sum(acc1);
            if (lane == 0) {
                q_idx[r0] = acc0;
                q_idx[r1] = acc1;
            }
        }
    }
}

// Fused QSA Q-Prep (bx = 0..23) + KV-Prep (bx = 24..25) + Parallel Indexer-Append (bx = 26) + optional Indexer-Q-Prep (bx = 27..30)
// Launch: grid = dim3(27 or 31, M), block = 256
__global__ void __launch_bounds__(256) qsa_prep_qkv_idx_kernel(
    const float* __restrict__ q_gate,
    const float* __restrict__ q_norm,
    float* __restrict__ q,
    float* __restrict__ gate,
    const float* __restrict__ k_in,
    const float* __restrict__ v_in,
    const float* __restrict__ k_norm,
    uint16_t* __restrict__ k_cache,
    uint16_t* __restrict__ v_cache,
    const float* __restrict__ idx_raw,
    const float* __restrict__ idx_k_norm,
    float* __restrict__ idx_pooled,
    float* __restrict__ idx_tail,
    float* __restrict__ idx_dead,
    float* __restrict__ q_idx,
    const float* __restrict__ idx_q_norm,
    int pos_start,
    int M,
    float eps)
{
    __shared__ float smem[8];
    __shared__ float s_kh[QWEN4_QSA_HEAD_DIM];

    const int bx = blockIdx.x;
    const int m  = blockIdx.y;
    const int d  = threadIdx.x; // 0..255

    if (bx < QWEN4_QSA_N_HEAD) {
        const int h   = bx;
        const int pos = pos_start + m;
        const size_t head_idx = (size_t)m * QWEN4_QSA_N_HEAD + h;
        const float* src = q_gate + head_idx * (2 * QWEN4_QSA_HEAD_DIM);
        float* qh = q    + head_idx * QWEN4_QSA_HEAD_DIM;
        float* gh = gate + head_idx * QWEN4_QSA_HEAD_DIM;

        const float v = src[d];
        gh[d] = src[QWEN4_QSA_HEAD_DIM + d];

        const float ss = block_sum<8>(v * v, smem);
        qh[d] = v * rsqrtf(ss / (float)QWEN4_QSA_HEAD_DIM + eps) * q_norm[d];
        __syncthreads();

        rope_neox_elem(qh, d, QWEN4_QSA_N_ROT, pos, QWEN4_QSA_THETA);
    } else if (bx < QWEN4_QSA_N_HEAD + QWEN4_QSA_N_KV) {
        const int h   = bx - QWEN4_QSA_N_HEAD;
        const int pos = pos_start + m;
        const size_t in_off = ((size_t)m * QWEN4_QSA_N_KV + h) * QWEN4_QSA_HEAD_DIM;
        uint16_t* kh = k_cache + (size_t)pos * (QWEN4_QSA_N_KV * QWEN4_QSA_HEAD_DIM) + h * QWEN4_QSA_HEAD_DIM;
        uint16_t* vh = v_cache + (size_t)pos * (QWEN4_QSA_N_KV * QWEN4_QSA_HEAD_DIM) + h * QWEN4_QSA_HEAD_DIM;

        const float kv = k_in[in_off + d];
        vh[d] = qwen4_fp32_to_bf16(v_in[in_off + d]);

        const float ss = block_sum<8>(kv * kv, smem);
        s_kh[d] = kv * rsqrtf(ss / (float)QWEN4_QSA_HEAD_DIM + eps) * k_norm[d];
        __syncthreads();

        rope_neox_elem(s_kh, d, QWEN4_QSA_N_ROT, pos, QWEN4_QSA_THETA);
        __syncthreads();
        kh[d] = qwen4_fp32_to_bf16(s_kh[d]);
    } else if (bx == 26) {
        if (d >= QWEN4_QSA_IDX_DIM) return;
        const int pos  = pos_start + m;
        const int slot = pos & (QWEN4_QSA_IDX_BLOCK - 1);
        const float kn = idx_k_norm[d];

        if (pos == 0) {
            const float v = idx_raw[(size_t)m * QWEN4_QSA_IDX_DIM + d] * 0.25f;
            float w_ss = warp_sum(v * v);
            if ((d & 31) == 0) smem[d >> 5] = w_ss;
            __syncthreads();
            if (d < 32) {
                float tot = (d < 4) ? smem[d] : 0.0f;
                tot = warp_sum(tot);
                if (d == 0) smem[0] = tot;
            }
            __syncthreads();
            idx_dead[d] = v * rsqrtf(smem[0] / (float)QWEN4_QSA_IDX_DIM + eps) * kn;
        }

        if (slot == QWEN4_QSA_IDX_BLOCK - 1) {
            const int b = pos / QWEN4_QSA_IDX_BLOCK;
            float sum4 = 0.0f;
            #pragma unroll
            for (int s = 0; s < QWEN4_QSA_IDX_BLOCK; ++s) {
                const int rel = m - (QWEN4_QSA_IDX_BLOCK - 1 - s);
                sum4 += (rel >= 0)
                    ? idx_raw[(size_t)rel * QWEN4_QSA_IDX_DIM + d]
                    : idx_tail[s * QWEN4_QSA_IDX_DIM + d];
            }
            const float v = 0.25f * sum4;
            float w_ss = warp_sum(v * v);
            if ((d & 31) == 0) smem[d >> 5] = w_ss;
            __syncthreads();
            if (d < 32) {
                float tot = (d < 4) ? smem[d] : 0.0f;
                tot = warp_sum(tot);
                if (d == 0) smem[0] = tot;
            }
            __syncthreads();
            float* pb = idx_pooled + (size_t)b * QWEN4_QSA_IDX_DIM;
            pb[d] = v * rsqrtf(smem[0] / (float)QWEN4_QSA_IDX_DIM + eps) * kn;
            __syncthreads();
            rope_neox_elem(pb, d, QWEN4_QSA_N_ROT, b * QWEN4_QSA_IDX_BLOCK, QWEN4_QSA_THETA);
        }

        const int first_complete_m = (pos_start & 3) ? (3 - (pos_start & 3)) : 0;
        const int tail_updater_m   = (first_complete_m < M) ? first_complete_m : 0;
        if (m == tail_updater_m) {
            const int start_step = (M > QWEN4_QSA_IDX_BLOCK) ? (M - QWEN4_QSA_IDX_BLOCK) : 0;
            for (int step = start_step; step < M; ++step) {
                const int s_slot = (pos_start + step) & (QWEN4_QSA_IDX_BLOCK - 1);
                idx_tail[s_slot * QWEN4_QSA_IDX_DIM + d] = idx_raw[(size_t)step * QWEN4_QSA_IDX_DIM + d];
            }
        }
    } else if (q_idx != nullptr && d < QWEN4_QSA_IDX_DIM) {
        const int h   = bx - 27; // 0..3
        const int pos = pos_start + m;
        float* qh = q_idx + ((size_t)m * QWEN4_QSA_IDX_HEADS + h) * QWEN4_QSA_IDX_DIM;
        const float v = qh[d];
        float w_ss = warp_sum(v * v);
        if ((d & 31) == 0) smem[d >> 5] = w_ss;
        __syncthreads();
        if (d < 32) {
            float tot = (d < 4) ? smem[d] : 0.0f;
            tot = warp_sum(tot);
            if (d == 0) smem[0] = tot;
        }
        __syncthreads();
        qh[d] = v * rsqrtf(smem[0] / (float)QWEN4_QSA_IDX_DIM + eps) * idx_q_norm[d];
        __syncthreads();
        rope_neox_elem(qh, d, QWEN4_QSA_N_ROT, pos, QWEN4_QSA_THETA);
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

__device__ int g_qwen4_qsa_score_sync = 0;

// Warp-parallel float4 block scorer (8 blocks per 256-thread CTA) + optional fused last-block Top-Block Selector!
// Launch: <<<(n_blocks + 7) / 8, dim3(32, 8)>>>
__global__ void __launch_bounds__(256) qsa_indexer_score_blocks_kernel(
    const float* __restrict__ idx_pooled,
    const float* __restrict__ q_idx,
    float* __restrict__ block_scores,
    int n_kv,
    int n_blocks,
    int32_t* __restrict__ d_sel_indices = nullptr,
    int n_sel = 0)
{
    const int b    = blockIdx.x * 8 + threadIdx.y;
    const int lane = threadIdx.x; // 0..31
    if (b < n_blocks) {
        const int start = b * QWEN4_QSA_IDX_BLOCK;
        const int valid = (n_kv - start < QWEN4_QSA_IDX_BLOCK) ? (n_kv - start) : QWEN4_QSA_IDX_BLOCK;
        if (valid < QWEN4_QSA_IDX_BLOCK) {
            if (lane == 0) block_scores[b] = 1e9f;
        } else {
            const float4 k4 = ((const float4*)(idx_pooled + (size_t)b * QWEN4_QSA_IDX_DIM))[lane];
            float total = 0.0f;
            #pragma unroll
            for (int h = 0; h < QWEN4_QSA_IDX_HEADS; ++h) {
                const float4 q4 = ((const float4*)(q_idx + h * QWEN4_QSA_IDX_DIM))[lane];
                const float prod = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
                const float dot  = warp_sum(prod);
                if (dot > 0.0f) total += dot;
            }
            if (lane == 0) block_scores[b] = total;
        }
    }
    if (d_sel_indices == nullptr) return;

    __shared__ float s_min[8];
    __shared__ float s_max[8];
    __shared__ int   s_cnt[2][8];
    __shared__ float s_lo;
    __shared__ float s_hi;
    __shared__ int   s_written;
    __shared__ bool  s_is_last;

    const int warp = threadIdx.y;
    const int tid  = warp * 32 + lane;
    __syncthreads();

    if (tid == 0) {
        if (gridDim.x > 1) {
            __threadfence();
            const int old = atomicAdd(&g_qwen4_qsa_score_sync, 1);
            s_is_last = (old == (int)gridDim.x - 1);
            if (s_is_last) g_qwen4_qsa_score_sync = 0;
        } else {
            __threadfence_block();
            s_is_last = true;
        }
    }
    __syncthreads();
    if (!s_is_last) return;

    const int full_blocks = n_blocks - 1;
    const int tail_start  = full_blocks * QWEN4_QSA_IDX_BLOCK;
    const int tail_valid  = n_kv - tail_start;

    if (tid == 0) {
        int w = 0;
        for (int t = 0; t < tail_valid && w < n_sel; ++t) {
            d_sel_indices[w++] = tail_start + t;
        }
        s_written = w;
    }
    __syncthreads();

    const int need_cells  = n_sel - s_written;
    const int need_blocks = (need_cells + QWEN4_QSA_IDX_BLOCK - 1) / QWEN4_QSA_IDX_BLOCK;
    if (need_blocks <= 0 || full_blocks <= 0) return;

    if (full_blocks <= need_blocks) {
        for (int blk = tid; blk < full_blocks; blk += 256) {
            const int pos   = atomicAdd(&s_written, QWEN4_QSA_IDX_BLOCK);
            const int start = blk * QWEN4_QSA_IDX_BLOCK;
            #pragma unroll
            for (int t = 0; t < QWEN4_QSA_IDX_BLOCK; ++t) {
                if (pos + t < n_sel) d_sel_indices[pos + t] = start + t;
            }
        }
        return;
    }

    float l_min = 1e30f;
    float l_max = -1e30f;
    float sc0 = -1e30f, sc1 = -1e30f, sc2 = -1e30f, sc3 = -1e30f;
    if (full_blocks <= 1024) {
        if (tid < full_blocks)       { sc0 = __ldcg(block_scores + tid);       l_min = fminf(l_min, sc0); l_max = fmaxf(l_max, sc0); }
        if (tid + 256 < full_blocks) { sc1 = __ldcg(block_scores + tid + 256); l_min = fminf(l_min, sc1); l_max = fmaxf(l_max, sc1); }
        if (tid + 512 < full_blocks) { sc2 = __ldcg(block_scores + tid + 512); l_min = fminf(l_min, sc2); l_max = fmaxf(l_max, sc2); }
        if (tid + 768 < full_blocks) { sc3 = __ldcg(block_scores + tid + 768); l_min = fminf(l_min, sc3); l_max = fmaxf(l_max, sc3); }
    } else {
        for (int blk = tid; blk < full_blocks; blk += 256) {
            const float v = __ldcg(block_scores + blk);
            l_min = fminf(l_min, v);
            l_max = fmaxf(l_max, v);
        }
    }
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        l_min = fminf(l_min, __shfl_xor_sync(0xFFFFFFFFu, l_min, mask, 32));
        l_max = fmaxf(l_max, __shfl_xor_sync(0xFFFFFFFFu, l_max, mask, 32));
    }
    if (lane == 0) {
        s_min[warp] = l_min;
        s_max[warp] = l_max;
    }
    __syncthreads();

    if (warp == 0) {
        l_min = (lane < 8) ? s_min[lane] : 1e30f;
        l_max = (lane < 8) ? s_max[lane] : -1e30f;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            l_min = fminf(l_min, __shfl_xor_sync(0xFFFFFFFFu, l_min, mask, 32));
            l_max = fmaxf(l_max, __shfl_xor_sync(0xFFFFFFFFu, l_max, mask, 32));
        }
        if (lane == 0) {
            s_lo = l_min;
            s_hi = l_max + 1e-4f;
        }
    }
    __syncthreads();

    float lo = s_lo;
    float hi = s_hi;
    #pragma unroll 1
    for (int iter = 0; iter < 16; ++iter) {
        const float mid = 0.5f * (lo + hi);
        int c = 0;
        if (full_blocks <= 1024) {
            c = (sc0 >= mid) + (sc1 >= mid) + (sc2 >= mid) + (sc3 >= mid);
        } else {
            for (int blk = tid; blk < full_blocks; blk += 256) {
                if (__ldcg(block_scores + blk) >= mid) c++;
            }
        }
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            c += __shfl_xor_sync(0xFFFFFFFFu, c, mask, 32);
        }
        if (lane == 0) s_cnt[iter & 1][warp] = c;
        __syncthreads();

        const int* sc = s_cnt[iter & 1];
        const int tot = sc[0] + sc[1] + sc[2] + sc[3] + sc[4] + sc[5] + sc[6] + sc[7];
        if (tot >= need_blocks) lo = mid;
        else                    hi = mid;
    }

    for (int blk = tid; blk < full_blocks; blk += 256) {
        if (__ldcg(block_scores + blk) >= hi) {
            const int pos   = atomicAdd(&s_written, QWEN4_QSA_IDX_BLOCK);
            const int start = blk * QWEN4_QSA_IDX_BLOCK;
            #pragma unroll
            for (int t = 0; t < QWEN4_QSA_IDX_BLOCK; ++t) {
                if (pos + t < n_sel) d_sel_indices[pos + t] = start + t;
            }
        }
    }
    __syncthreads();

    if (s_written < n_sel) {
        for (int blk = tid; blk < full_blocks; blk += 256) {
            const float v = __ldcg(block_scores + blk);
            if (v >= lo && v < hi) {
                const int pos = atomicAdd(&s_written, QWEN4_QSA_IDX_BLOCK);
                if (pos < n_sel) {
                    const int start = blk * QWEN4_QSA_IDX_BLOCK;
                    #pragma unroll
                    for (int t = 0; t < QWEN4_QSA_IDX_BLOCK; ++t) {
                        if (pos + t < n_sel) d_sel_indices[pos + t] = start + t;
                    }
                }
            }
        }
    }
}

// Fast Parallel Binary-Search Top-Block Selector (1 block of 256 threads):
// Finds the score threshold for the top `n_sel` cells in 24 binary search iterations (~50 barriers total vs 4,617 previously!)
__global__ void __launch_bounds__(256) qsa_indexer_select_topk_kernel(
    float* __restrict__ block_scores,
    int32_t* __restrict__ d_sel_indices,
    int n_kv,
    int n_blocks,
    int n_sel)
{
    __shared__ float s_min[8];
    __shared__ float s_max[8];
    __shared__ int   s_cnt[2][8];
    __shared__ float s_lo;
    __shared__ float s_hi;
    __shared__ int   s_written;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    // 1. Always emit the tail block (b = n_blocks - 1) first so the most recent tokens are unconditionally included
    const int full_blocks = n_blocks - 1;
    const int tail_start  = full_blocks * QWEN4_QSA_IDX_BLOCK;
    const int tail_valid  = n_kv - tail_start;

    if (tid == 0) {
        int w = 0;
        for (int t = 0; t < tail_valid && w < n_sel; ++t) {
            d_sel_indices[w++] = tail_start + t;
        }
        s_written = w;
    }
    __syncthreads();

    const int need_cells  = n_sel - s_written;
    const int need_blocks = (need_cells + QWEN4_QSA_IDX_BLOCK - 1) / QWEN4_QSA_IDX_BLOCK;
    if (need_blocks <= 0 || full_blocks <= 0) return;

    if (full_blocks <= need_blocks) {
        for (int b = tid; b < full_blocks; b += 256) {
            const int pos   = atomicAdd(&s_written, QWEN4_QSA_IDX_BLOCK);
            const int start = b * QWEN4_QSA_IDX_BLOCK;
            #pragma unroll
            for (int t = 0; t < QWEN4_QSA_IDX_BLOCK; ++t) {
                if (pos + t < n_sel) d_sel_indices[pos + t] = start + t;
            }
        }
        return;
    }

    // 2. Find [min_s, max_s] across b = 0 .. full_blocks - 1
    float l_min = 1e30f;
    float l_max = -1e30f;
    float sc0 = -1e30f, sc1 = -1e30f, sc2 = -1e30f, sc3 = -1e30f;
    if (full_blocks <= 1024) {
        if (tid < full_blocks)       { sc0 = block_scores[tid];       l_min = fminf(l_min, sc0); l_max = fmaxf(l_max, sc0); }
        if (tid + 256 < full_blocks) { sc1 = block_scores[tid + 256]; l_min = fminf(l_min, sc1); l_max = fmaxf(l_max, sc1); }
        if (tid + 512 < full_blocks) { sc2 = block_scores[tid + 512]; l_min = fminf(l_min, sc2); l_max = fmaxf(l_max, sc2); }
        if (tid + 768 < full_blocks) { sc3 = block_scores[tid + 768]; l_min = fminf(l_min, sc3); l_max = fmaxf(l_max, sc3); }
    } else {
        for (int b = tid; b < full_blocks; b += 256) {
            const float v = block_scores[b];
            l_min = fminf(l_min, v);
            l_max = fmaxf(l_max, v);
        }
    }
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        l_min = fminf(l_min, __shfl_xor_sync(0xFFFFFFFFu, l_min, mask, 32));
        l_max = fmaxf(l_max, __shfl_xor_sync(0xFFFFFFFFu, l_max, mask, 32));
    }
    if (lane == 0) {
        s_min[warp] = l_min;
        s_max[warp] = l_max;
    }
    __syncthreads();

    if (warp == 0) {
        l_min = (lane < 8) ? s_min[lane] : 1e30f;
        l_max = (lane < 8) ? s_max[lane] : -1e30f;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            l_min = fminf(l_min, __shfl_xor_sync(0xFFFFFFFFu, l_min, mask, 32));
            l_max = fmaxf(l_max, __shfl_xor_sync(0xFFFFFFFFu, l_max, mask, 32));
        }
        if (lane == 0) {
            s_lo = l_min;
            s_hi = l_max + 1e-4f;
        }
    }
    __syncthreads();

    // 3. 16-iteration 1-barrier double-buffered parallel range binary search
    float lo = s_lo;
    float hi = s_hi;
    #pragma unroll 1
    for (int iter = 0; iter < 16; ++iter) {
        const float mid = 0.5f * (lo + hi);
        int c = 0;
        if (full_blocks <= 1024) {
            c = (sc0 >= mid) + (sc1 >= mid) + (sc2 >= mid) + (sc3 >= mid);
        } else {
            for (int b = tid; b < full_blocks; b += 256) {
                if (block_scores[b] >= mid) c++;
            }
        }
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            c += __shfl_xor_sync(0xFFFFFFFFu, c, mask, 32);
        }
        if (lane == 0) s_cnt[iter & 1][warp] = c;
        __syncthreads();

        const int* sc = s_cnt[iter & 1];
        const int tot = sc[0] + sc[1] + sc[2] + sc[3] + sc[4] + sc[5] + sc[6] + sc[7];
        if (tot >= need_blocks) lo = mid;
        else                    hi = mid;
    }

    // 4. Pass 1: Emit all blocks strictly >= hi (guaranteed < need_blocks)
    for (int b = tid; b < full_blocks; b += 256) {
        if (block_scores[b] >= hi) {
            const int pos   = atomicAdd(&s_written, QWEN4_QSA_IDX_BLOCK);
            const int start = b * QWEN4_QSA_IDX_BLOCK;
            #pragma unroll
            for (int t = 0; t < QWEN4_QSA_IDX_BLOCK; ++t) {
                if (pos + t < n_sel) d_sel_indices[pos + t] = start + t;
            }
        }
    }
    __syncthreads();

    // 5. Pass 2: Fill remaining cells from blocks in [lo, hi)
    if (s_written < n_sel) {
        for (int b = tid; b < full_blocks; b += 256) {
            const float v = block_scores[b];
            if (v >= lo && v < hi) {
                const int pos = atomicAdd(&s_written, QWEN4_QSA_IDX_BLOCK);
                if (pos < n_sel) {
                    const int start = b * QWEN4_QSA_IDX_BLOCK;
                    #pragma unroll
                    for (int t = 0; t < QWEN4_QSA_IDX_BLOCK; ++t) {
                        if (pos + t < n_sel) d_sel_indices[pos + t] = start + t;
                    }
                }
            }
        }
    }
}

// GQA-12:1 Split-KV Flash-Decoding Chunk Kernel
// Launch: grid = dim3(n_chunks, 4, M), block = dim3(32, 8) (8 warps per block)
// Each group g = 0..3 processes 6 query heads sharing kv_h = g >> 1, loading K/V once via uint4 (8x BF16) per 6 query heads!
__global__ void __launch_bounds__(256, 2) qsa_split_kv_chunk_kernel(
    const float* __restrict__ q,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    const int32_t* __restrict__ sel_indices,
    int pos_start,
    int n_sel_fixed,
    float* __restrict__ partial_acc,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int split_chunk = QWEN4_QSA_SPLIT_CHUNK)
{
    __shared__ float s_m[6][8];
    __shared__ float s_l[6][8];
    __shared__ float s_acc[2][8][QWEN4_QSA_HEAD_DIM]; // 16.4 KB/block -> 2 blocks/SM (32.8 KB <= 48 KB SM static smem limit = 164 blocks/wave!)

    const int chunk_id = blockIdx.x;
    const int g        = blockIdx.y; // 0..3
    const int m_tok    = blockIdx.z; // 0..M-1
    const int n_chunks = gridDim.x;
    const int h_base   = g * 6;
    const int kv_h     = g >> 1;

    const int n_sel   = (sel_indices != nullptr) ? n_sel_fixed : (pos_start + m_tok + 1);
    const int s_start = chunk_id * split_chunk;
    const int s_end   = (s_start + split_chunk < n_sel) ? (s_start + split_chunk) : n_sel;

    const int lane   = threadIdx.x; // 0..31
    const int warp   = threadIdx.y; // 0..7
    const int tid    = warp * 32 + lane;
    const int d_base = lane * 8;

    if (s_start >= s_end) {
        if (warp == 0 && lane < 6) {
            const size_t p_idx = ((size_t)m_tok * n_chunks + chunk_id) * QWEN4_QSA_N_HEAD + (h_base + lane);
            partial_m[p_idx] = -1e30f;
            partial_l[p_idx] = 0.0f;
        }
        return;
    }

    constexpr float scale = 0.0625f; // 1 / sqrt(256)
    __nv_bfloat162 q_bf2[6][4];
    #pragma unroll
    for (int qh = 0; qh < 6; ++qh) {
        const float4* q_f4 = (const float4*)(q + ((size_t)m_tok * QWEN4_QSA_N_HEAD + (h_base + qh)) * QWEN4_QSA_HEAD_DIM + d_base);
        const float4 q0 = q_f4[0];
        const float4 q1 = q_f4[1];
        q_bf2[qh][0] = __floats2bfloat162_rn(q0.x * scale, q0.y * scale);
        q_bf2[qh][1] = __floats2bfloat162_rn(q0.z * scale, q0.w * scale);
        q_bf2[qh][2] = __floats2bfloat162_rn(q1.x * scale, q1.y * scale);
        q_bf2[qh][3] = __floats2bfloat162_rn(q1.z * scale, q1.w * scale);
    }

    float m_w[6];
    float l_w[6];
    float acc_w[6][8];
    #pragma unroll
    for (int qh = 0; qh < 6; ++qh) {
        m_w[qh] = -1e30f;
        l_w[qh] = 0.0f;
        #pragma unroll
        for (int u = 0; u < 8; ++u) acc_w[qh][u] = 0.0f;
    }

    for (int s = s_start + warp; s < s_end; s += 8) {
        const int kv_pos = (sel_indices != nullptr) ? sel_indices[s] : s;
        const size_t kv_off = (size_t)kv_pos * (QWEN4_QSA_N_KV * QWEN4_QSA_HEAD_DIM) + (size_t)kv_h * QWEN4_QSA_HEAD_DIM + d_base;
        const uint4 k_u4 = __ldg((const uint4*)(k_cache + kv_off));
        const uint4 v_u4 = __ldg((const uint4*)(v_cache + kv_off));
        const float v0 = __uint_as_float(v_u4.x << 16);
        const float v1 = __uint_as_float(v_u4.x & 0xFFFF0000u);
        const float v2 = __uint_as_float(v_u4.y << 16);
        const float v3 = __uint_as_float(v_u4.y & 0xFFFF0000u);
        const float v4 = __uint_as_float(v_u4.z << 16);
        const float v5 = __uint_as_float(v_u4.z & 0xFFFF0000u);
        const float v6 = __uint_as_float(v_u4.w << 16);
        const float v7 = __uint_as_float(v_u4.w & 0xFFFF0000u);

        #pragma unroll
        for (int qh = 0; qh < 6; ++qh) {
            const float score = warp_sum(dot8_bf16_u4(q_bf2[qh], k_u4));
            if (score > m_w[qh]) {
                const float alpha = __expf(m_w[qh] - score);
                m_w[qh] = score;
                l_w[qh] = l_w[qh] * alpha + 1.0f;
                acc_w[qh][0] = acc_w[qh][0] * alpha + v0;
                acc_w[qh][1] = acc_w[qh][1] * alpha + v1;
                acc_w[qh][2] = acc_w[qh][2] * alpha + v2;
                acc_w[qh][3] = acc_w[qh][3] * alpha + v3;
                acc_w[qh][4] = acc_w[qh][4] * alpha + v4;
                acc_w[qh][5] = acc_w[qh][5] * alpha + v5;
                acc_w[qh][6] = acc_w[qh][6] * alpha + v6;
                acc_w[qh][7] = acc_w[qh][7] * alpha + v7;
            } else if (score > m_w[qh] - 20.0f) {
                const float p = __expf(score - m_w[qh]);
                l_w[qh] += p;
                acc_w[qh][0] += p * v0;
                acc_w[qh][1] += p * v1;
                acc_w[qh][2] += p * v2;
                acc_w[qh][3] += p * v3;
                acc_w[qh][4] += p * v4;
                acc_w[qh][5] += p * v5;
                acc_w[qh][6] += p * v6;
                acc_w[qh][7] += p * v7;
            }
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (int qh = 0; qh < 6; ++qh) {
            s_m[qh][warp] = m_w[qh];
            s_l[qh][warp] = l_w[qh];
        }
    }
    __syncthreads();

    float m_max[6];
    float w_scale[6];
    #pragma unroll
    for (int qh = 0; qh < 6; ++qh) {
        float mm = -1e30f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            mm = fmaxf(mm, s_m[qh][w]);
        }
        m_max[qh]   = mm;
        w_scale[qh] = (l_w[qh] > 0.0f) ? expf(m_w[qh] - mm) : 0.0f;
    }

    const size_t p_base = ((size_t)m_tok * n_chunks + chunk_id) * QWEN4_QSA_N_HEAD + h_base;
    if (warp == 0 && lane < 6) {
        const int qh = lane;
        const float mm = m_max[qh];
        float l_tot = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            if (s_l[qh][w] > 0.0f) l_tot += s_l[qh][w] * expf(s_m[qh][w] - mm);
        }
        partial_m[p_base + qh] = mm;
        partial_l[p_base + qh] = l_tot;
    }

    #pragma unroll
    for (int part = 0; part < 3; ++part) {
        const int qh0 = part * 2 + 0;
        const int qh1 = part * 2 + 1;

        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            s_acc[0][warp][d_base + u] = acc_w[qh0][u] * w_scale[qh0];
            s_acc[1][warp][d_base + u] = acc_w[qh1][u] * w_scale[qh1];
        }
        __syncthreads();

        float sum0 = 0.0f, sum1 = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            sum0 += s_acc[0][w][tid];
            sum1 += s_acc[1][w][tid];
        }
        partial_acc[(p_base + qh0) * QWEN4_QSA_HEAD_DIM + tid] = sum0;
        partial_acc[(p_base + qh1) * QWEN4_QSA_HEAD_DIM + tid] = sum1;
        if (part < 2) {
            __syncthreads();
        }
    }
}

// Merge Split-KV Chunks + Apply Sigmoid Gate + Quantize to Q8_1
// Uses Warp-0 Parallel Softmax Reduction into s_beta[36] so the 256-thread partial_acc loop has ZERO expf/fmaxf/branch dependencies!
// Launch: grid = dim3(24, M), block = 256
__global__ void __launch_bounds__(256) qsa_merge_chunks_gate_q8_1_kernel(
    const float* __restrict__ partial_acc,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ gate,
    int n_chunks,
    float* __restrict__ attn_gated,
    block_q8_1* __restrict__ q8_out)
{
    __shared__ float s_beta[36];
    const int h     = blockIdx.x;  // 0..23
    const int m_tok = blockIdx.y;  // 0..M-1
    const int d     = threadIdx.x; // 0..255

    if (d < 32 && n_chunks <= 32) {
        const size_t p_idx = ((size_t)m_tok * n_chunks + d) * QWEN4_QSA_N_HEAD + h;
        const float lc = (d < n_chunks) ? __ldg(partial_l + p_idx) : 0.0f;
        const float mc = (lc > 0.0f)    ? __ldg(partial_m + p_idx) : -1e30f;
        float m_max = mc;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            m_max = fmaxf(m_max, __shfl_xor_sync(0xFFFFFFFFu, m_max, mask, 32));
        }
        const float w_c = (lc > 0.0f) ? expf(mc - m_max) : 0.0f;
        float l_tot = lc * w_c;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            l_tot += __shfl_xor_sync(0xFFFFFFFFu, l_tot, mask, 32);
        }
        s_beta[d] = (l_tot > 0.0f) ? (w_c / l_tot) : 0.0f;
    } else if (d == 0 && n_chunks > 32) {
        float m_max = -1e30f;
        for (int c = 0; c < n_chunks; ++c) {
            const size_t p_idx = ((size_t)m_tok * n_chunks + c) * QWEN4_QSA_N_HEAD + h;
            if (partial_l[p_idx] > 0.0f) m_max = fmaxf(m_max, partial_m[p_idx]);
        }
        float l_tot = 0.0f;
        for (int c = 0; c < n_chunks; ++c) {
            const size_t p_idx = ((size_t)m_tok * n_chunks + c) * QWEN4_QSA_N_HEAD + h;
            const float lc = partial_l[p_idx];
            const float w_c = (lc > 0.0f) ? expf(partial_m[p_idx] - m_max) : 0.0f;
            s_beta[c] = w_c;
            l_tot += lc * w_c;
        }
        const float inv_l = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;
        for (int c = 0; c < n_chunks; ++c) s_beta[c] *= inv_l;
    }
    __syncthreads();

    float acc_d = 0.0f;
    const float* pa_d = partial_acc + ((size_t)m_tok * n_chunks * QWEN4_QSA_N_HEAD + h) * QWEN4_QSA_HEAD_DIM + d;
    constexpr size_t c_stride = (size_t)QWEN4_QSA_N_HEAD * QWEN4_QSA_HEAD_DIM;
    for (int c = 0; c < n_chunks; ++c) {
        acc_d += __ldg(pa_d + c * c_stride) * s_beta[c];
    }

    const size_t out_idx = ((size_t)m_tok * QWEN4_QSA_N_HEAD + h) * QWEN4_QSA_HEAD_DIM + d;
    const float g = __ldg(gate + out_idx);
    const float val = acc_d / (1.0f + expf(-g));
    attn_gated[out_idx] = val;

    if (q8_out != nullptr) {
        const int lane = d & 31;
        const float xi = q8_1_finite(val);
        float amax = fabsf(xi);
        float sum  = xi;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask, 32));
            sum += __shfl_xor_sync(0xFFFFFFFFu, sum, mask, 32);
        }
        const float sc = amax * (1.0f / 127.0f);
        const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / sc);
        block_q8_1* blk = q8_out + ((size_t)m_tok * QWEN4_QSA_N_HEAD + h) * (QWEN4_QSA_HEAD_DIM / 32) + (d >> 5);
        blk->qs[lane] = q;
        if (lane == 0) {
            blk->ds = make_half2(__float2half(sc), __float2half(sum));
        }
    }
}

// Single-Pass Fused Batched Flash-Attention + Sigmoid Gate + Q8_1 Quantization for QSA Prefill (pos_start + M <= 2048)
// Launch: grid = dim3(8, (M + 1) / 2), block = dim3(32, 8) (256 threads)
// Each block processes 2 query tokens (m0, m1) x 3 query heads (h_base + 0..2) sharing kv_h = g >> 2,
// streaming K/V cache ONCE across all 6 (token, head) pairs and writing attn_gated + q8_out directly!
__global__ void __launch_bounds__(256) qsa_prefill_fused_attn_gate_q8_1_kernel(
    const float* __restrict__ q,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    const float* __restrict__ gate,
    int pos_start,
    int M,
    float* __restrict__ attn_gated,
    block_q8_1* __restrict__ q8_out,
    uint16_t* __restrict__ bf16_out = nullptr)
{
    __shared__ float s_m[6][8];
    __shared__ float s_l[6][8];
    __shared__ float s_acc[2][8][QWEN4_QSA_HEAD_DIM];

    const int g      = blockIdx.x; // 0..7 (3 query heads per group)
    const int pair_m = blockIdx.y; // 0..(M+1)/2 - 1
    const int m0     = pair_m * 2;
    const int m1     = m0 + 1;
    const bool has_m1 = (m1 < M);
    const int h_base = g * 3;
    const int kv_h   = g >> 2;

    const int n_kv0  = pos_start + m0 + 1;
    const int n_kv1  = has_m1 ? (pos_start + m1 + 1) : n_kv0;

    const int lane   = threadIdx.x; // 0..31
    const int warp   = threadIdx.y; // 0..7
    const int tid    = warp * 32 + lane;
    const int d_base = lane * 8;

    constexpr float scale = 0.0625f; // 1 / sqrt(256)
    __nv_bfloat162 q_bf2[6][4];
    #pragma unroll
    for (int qh = 0; qh < 3; ++qh) {
        const float4* q0_f4 = (const float4*)(q + ((size_t)m0 * QWEN4_QSA_N_HEAD + (h_base + qh)) * QWEN4_QSA_HEAD_DIM + d_base);
        const float4 a0 = q0_f4[0];
        const float4 a1 = q0_f4[1];
        q_bf2[qh][0] = __floats2bfloat162_rn(a0.x * scale, a0.y * scale);
        q_bf2[qh][1] = __floats2bfloat162_rn(a0.z * scale, a0.w * scale);
        q_bf2[qh][2] = __floats2bfloat162_rn(a1.x * scale, a1.y * scale);
        q_bf2[qh][3] = __floats2bfloat162_rn(a1.z * scale, a1.w * scale);
        if (has_m1) {
            const float4* q1_f4 = (const float4*)(q + ((size_t)m1 * QWEN4_QSA_N_HEAD + (h_base + qh)) * QWEN4_QSA_HEAD_DIM + d_base);
            const float4 b0 = q1_f4[0];
            const float4 b1 = q1_f4[1];
            q_bf2[3 + qh][0] = __floats2bfloat162_rn(b0.x * scale, b0.y * scale);
            q_bf2[3 + qh][1] = __floats2bfloat162_rn(b0.z * scale, b0.w * scale);
            q_bf2[3 + qh][2] = __floats2bfloat162_rn(b1.x * scale, b1.y * scale);
            q_bf2[3 + qh][3] = __floats2bfloat162_rn(b1.z * scale, b1.w * scale);
        } else {
            q_bf2[3 + qh][0] = __float2bfloat162_rn(0.0f);
            q_bf2[3 + qh][1] = __float2bfloat162_rn(0.0f);
            q_bf2[3 + qh][2] = __float2bfloat162_rn(0.0f);
            q_bf2[3 + qh][3] = __float2bfloat162_rn(0.0f);
        }
    }

    float m_w[6];
    float l_w[6];
    float acc_w[6][8];
    #pragma unroll
    for (int s_idx = 0; s_idx < 6; ++s_idx) {
        m_w[s_idx] = -1e30f;
        l_w[s_idx] = 0.0f;
        #pragma unroll
        for (int u = 0; u < 8; ++u) acc_w[s_idx][u] = 0.0f;
    }

    for (int s = warp; s < n_kv1; s += 8) {
        const size_t kv_off = (size_t)s * (QWEN4_QSA_N_KV * QWEN4_QSA_HEAD_DIM) + (size_t)kv_h * QWEN4_QSA_HEAD_DIM + d_base;
        const uint4 k_u4 = __ldg((const uint4*)(k_cache + kv_off));
        const uint4 v_u4 = __ldg((const uint4*)(v_cache + kv_off));

        if (s < n_kv0) {
            #pragma unroll
            for (int qh = 0; qh < 3; ++qh) {
                const float score = warp_sum(dot8_bf16_u4(q_bf2[qh], k_u4));
                if (score > m_w[qh]) {
                    const float alpha = __expf(m_w[qh] - score);
                    m_w[qh] = score;
                    l_w[qh] = l_w[qh] * alpha + 1.0f;
                    acc_w[qh][0] = acc_w[qh][0] * alpha + __uint_as_float(v_u4.x << 16);
                    acc_w[qh][1] = acc_w[qh][1] * alpha + __uint_as_float(v_u4.x & 0xFFFF0000u);
                    acc_w[qh][2] = acc_w[qh][2] * alpha + __uint_as_float(v_u4.y << 16);
                    acc_w[qh][3] = acc_w[qh][3] * alpha + __uint_as_float(v_u4.y & 0xFFFF0000u);
                    acc_w[qh][4] = acc_w[qh][4] * alpha + __uint_as_float(v_u4.z << 16);
                    acc_w[qh][5] = acc_w[qh][5] * alpha + __uint_as_float(v_u4.z & 0xFFFF0000u);
                    acc_w[qh][6] = acc_w[qh][6] * alpha + __uint_as_float(v_u4.w << 16);
                    acc_w[qh][7] = acc_w[qh][7] * alpha + __uint_as_float(v_u4.w & 0xFFFF0000u);
                } else if (score > m_w[qh] - 20.0f) {
                    const float p = __expf(score - m_w[qh]);
                    l_w[qh] += p;
                    acc_w[qh][0] += p * __uint_as_float(v_u4.x << 16);
                    acc_w[qh][1] += p * __uint_as_float(v_u4.x & 0xFFFF0000u);
                    acc_w[qh][2] += p * __uint_as_float(v_u4.y << 16);
                    acc_w[qh][3] += p * __uint_as_float(v_u4.y & 0xFFFF0000u);
                    acc_w[qh][4] += p * __uint_as_float(v_u4.z << 16);
                    acc_w[qh][5] += p * __uint_as_float(v_u4.z & 0xFFFF0000u);
                    acc_w[qh][6] += p * __uint_as_float(v_u4.w << 16);
                    acc_w[qh][7] += p * __uint_as_float(v_u4.w & 0xFFFF0000u);
                }
            }
        }

        if (has_m1) {
            #pragma unroll
            for (int qh = 3; qh < 6; ++qh) {
                const float score = warp_sum(dot8_bf16_u4(q_bf2[qh], k_u4));
                if (score > m_w[qh]) {
                    const float alpha = __expf(m_w[qh] - score);
                    m_w[qh] = score;
                    l_w[qh] = l_w[qh] * alpha + 1.0f;
                    acc_w[qh][0] = acc_w[qh][0] * alpha + __uint_as_float(v_u4.x << 16);
                    acc_w[qh][1] = acc_w[qh][1] * alpha + __uint_as_float(v_u4.x & 0xFFFF0000u);
                    acc_w[qh][2] = acc_w[qh][2] * alpha + __uint_as_float(v_u4.y << 16);
                    acc_w[qh][3] = acc_w[qh][3] * alpha + __uint_as_float(v_u4.y & 0xFFFF0000u);
                    acc_w[qh][4] = acc_w[qh][4] * alpha + __uint_as_float(v_u4.z << 16);
                    acc_w[qh][5] = acc_w[qh][5] * alpha + __uint_as_float(v_u4.z & 0xFFFF0000u);
                    acc_w[qh][6] = acc_w[qh][6] * alpha + __uint_as_float(v_u4.w << 16);
                    acc_w[qh][7] = acc_w[qh][7] * alpha + __uint_as_float(v_u4.w & 0xFFFF0000u);
                } else if (score > m_w[qh] - 20.0f) {
                    const float p = __expf(score - m_w[qh]);
                    l_w[qh] += p;
                    acc_w[qh][0] += p * __uint_as_float(v_u4.x << 16);
                    acc_w[qh][1] += p * __uint_as_float(v_u4.x & 0xFFFF0000u);
                    acc_w[qh][2] += p * __uint_as_float(v_u4.y << 16);
                    acc_w[qh][3] += p * __uint_as_float(v_u4.y & 0xFFFF0000u);
                    acc_w[qh][4] += p * __uint_as_float(v_u4.z << 16);
                    acc_w[qh][5] += p * __uint_as_float(v_u4.z & 0xFFFF0000u);
                    acc_w[qh][6] += p * __uint_as_float(v_u4.w << 16);
                    acc_w[qh][7] += p * __uint_as_float(v_u4.w & 0xFFFF0000u);
                }
            }
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (int s_idx = 0; s_idx < 6; ++s_idx) {
            s_m[s_idx][warp] = m_w[s_idx];
            s_l[s_idx][warp] = l_w[s_idx];
        }
    }
    __syncthreads();

    #pragma unroll
    for (int pair = 0; pair < 3; ++pair) {
        const int s0 = pair * 2 + 0;
        const int s1 = pair * 2 + 1;

        float m_max0 = -1e30f, m_max1 = -1e30f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            m_max0 = fmaxf(m_max0, s_m[s0][w]);
            m_max1 = fmaxf(m_max1, s_m[s1][w]);
        }

        float l_tot0 = 0.0f, l_tot1 = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            if (s_l[s0][w] > 0.0f) l_tot0 += s_l[s0][w] * expf(s_m[s0][w] - m_max0);
            if (s_l[s1][w] > 0.0f) l_tot1 += s_l[s1][w] * expf(s_m[s1][w] - m_max1);
        }

        const float w_scale0 = (l_w[s0] > 0.0f) ? expf(m_w[s0] - m_max0) : 0.0f;
        const float w_scale1 = (l_w[s1] > 0.0f) ? expf(m_w[s1] - m_max1) : 0.0f;

        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            s_acc[0][warp][d_base + u] = acc_w[s0][u] * w_scale0;
            s_acc[1][warp][d_base + u] = acc_w[s1][u] * w_scale1;
        }
        __syncthreads();

        float sum0 = 0.0f, sum1 = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            sum0 += s_acc[0][w][tid];
            sum1 += s_acc[1][w][tid];
        }

        const int m_tok0 = (s0 < 3) ? m0 : m1;
        const int h0     = h_base + ((s0 < 3) ? s0 : (s0 - 3));
        const bool valid0 = (s0 < 3) || has_m1;
        if (valid0) {
            const size_t out_idx0 = ((size_t)m_tok0 * QWEN4_QSA_N_HEAD + h0) * QWEN4_QSA_HEAD_DIM + tid;
            const float g0 = gate[out_idx0];
            const float val0 = (l_tot0 > 0.0f ? (sum0 / l_tot0) : 0.0f) / (1.0f + expf(-g0));
            if (bf16_out != nullptr) {
                bf16_out[out_idx0] = qwen4_fp32_to_bf16(val0);
            } else if (q8_out == nullptr) {
                attn_gated[out_idx0] = val0;
            }
            if (q8_out != nullptr) {
                const float xi = q8_1_finite(val0);
                float amax = fabsf(xi);
                float sum  = xi;
                #pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1) {
                    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask, 32));
                    sum += __shfl_xor_sync(0xFFFFFFFFu, sum, mask, 32);
                }
                const float sc = amax * (1.0f / 127.0f);
                const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / sc);
                block_q8_1* blk = q8_out + ((size_t)m_tok0 * QWEN4_QSA_N_HEAD + h0) * (QWEN4_QSA_HEAD_DIM / 32) + warp;
                blk->qs[lane] = q;
                if (lane == 0) {
                    blk->ds = make_half2(__float2half(sc), __float2half(sum));
                }
            }
        }

        const int m_tok1 = (s1 < 3) ? m0 : m1;
        const int h1     = h_base + ((s1 < 3) ? s1 : (s1 - 3));
        const bool valid1 = (s1 < 3) || has_m1;
        if (valid1) {
            const size_t out_idx1 = ((size_t)m_tok1 * QWEN4_QSA_N_HEAD + h1) * QWEN4_QSA_HEAD_DIM + tid;
            const float g1 = gate[out_idx1];
            const float val1 = (l_tot1 > 0.0f ? (sum1 / l_tot1) : 0.0f) / (1.0f + expf(-g1));
            if (bf16_out != nullptr) {
                bf16_out[out_idx1] = qwen4_fp32_to_bf16(val1);
            } else if (q8_out == nullptr) {
                attn_gated[out_idx1] = val1;
            }
            if (q8_out != nullptr) {
                const float xi = q8_1_finite(val1);
                float amax = fabsf(xi);
                float sum  = xi;
                #pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1) {
                    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask, 32));
                    sum += __shfl_xor_sync(0xFFFFFFFFu, sum, mask, 32);
                }
                const float sc = amax * (1.0f / 127.0f);
                const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / sc);
                block_q8_1* blk = q8_out + ((size_t)m_tok1 * QWEN4_QSA_N_HEAD + h1) * (QWEN4_QSA_HEAD_DIM / 32) + warp;
                blk->qs[lane] = q;
                if (lane == 0) {
                    blk->ds = make_half2(__float2half(sc), __float2half(sum));
                }
            }
        }

        if (pair < 2) {
            __syncthreads();
        }
    }
}

// ============================================================================
// MoE Router, Shared Expert, & Batched / Grouped Routed Experts
// ============================================================================

constexpr int QWEN4_N_EXPERTS        = 256;
constexpr int QWEN4_N_EXPERTS_ACTIVE = 10;
constexpr int QWEN4_MOE_FF_DIM       = 640;

__device__ int g_qwen4_router_sync = 0;

// Fused Router (256 rows, blocks 0..63, 4 rows/block, 1 row/warp) + Shared Expert Gate (1 row, block 64) BF16 GEMV
// + Optional Fused Last-Block Register-Only Top-10 Softmax Selection (65 blocks <= 82 SMs = 1 single wave!)
// Launch: <<<65, 128>>>
template <bool FUSE_TOP10>
__global__ void __launch_bounds__(128) router_and_shexp_gemv_kernel(
    const uint16_t* __restrict__ gate_inp_bf16,
    const uint16_t* __restrict__ gate_inp_shexp_bf16,
    const float* __restrict__ x,
    float* __restrict__ logits_and_shexp,
    int32_t* __restrict__ d_top_idx = nullptr,
    float* __restrict__ d_top_w = nullptr,
    int32_t* __restrict__ d_host_top_idx = nullptr,
    float* __restrict__ d_host_top_w = nullptr)
{
    const int bx   = blockIdx.x;  // 0..64
    const int tid  = threadIdx.x; // 0..127
    const int warp = tid >> 5;    // 0..3
    const int lane = tid & 31;    // 0..31

    if (bx < 64) {
        const int r = bx * 4 + warp;
        const uint4*  w  = (const uint4*)(gate_inp_bf16 + (size_t)r * QWEN4_N_EMBD);
        const float4* x4 = (const float4*)x;
        float acc = 0.0f;
        #pragma unroll
        for (int j = lane; j < QWEN4_N_EMBD / 8; j += 32) {
            const float4 x0 = __ldg(x4 + j * 2 + 0);
            const float4 x1 = __ldg(x4 + j * 2 + 1);
            acc += dot_bf16x8_fp32x8(w[j], x0, x1);
        }
        acc = warp_sum(acc);
        if (lane == 0) {
            logits_and_shexp[r] = acc;
        }
        if (!FUSE_TOP10) {
            return;
        } else {
            __shared__ bool s_is_last;
            __syncthreads();
            if (tid == 0) {
                __threadfence();
                const int old = atomicAdd(&g_qwen4_router_sync, 1);
                s_is_last = (old == 63);
                if (old == 63) g_qwen4_router_sync = 0;
            }
            __syncthreads();
            if (!s_is_last) return;

            if (warp == 0) {
                const float4 l0 = __ldcg(((const float4*)logits_and_shexp) + lane * 2 + 0);
                const float4 l1 = __ldcg(((const float4*)logits_and_shexp) + lane * 2 + 1);
                float p_reg[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
                int   i_reg[8];
                float vmax = -1e30f;
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    i_reg[u] = lane * 8 + u;
                    vmax = fmaxf(vmax, p_reg[u]);
                }
                #pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1) {
                    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFFu, vmax, mask, 32));
                }
                float lsum = 0.0f;
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    p_reg[u] = expf(p_reg[u] - vmax);
                    lsum += p_reg[u];
                }
                const float sum_p = warp_sum(lsum);
                const float inv_sum_p = 1.0f / sum_p;
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    p_reg[u] *= inv_sum_p;
                }

                float w_sum = 0.0f;
                int   sel_i = 0;
                float sel_p = 0.0f;
                #pragma unroll
                for (int k = 0; k < QWEN4_N_EXPERTS_ACTIVE; ++k) {
                    float best_p = p_reg[0];
                    int   best_i = i_reg[0];
                    #pragma unroll
                    for (int u = 1; u < 8; ++u) {
                        if (p_reg[u] > best_p) {
                            best_p = p_reg[u];
                            best_i = i_reg[u];
                        }
                    }
                    #pragma unroll
                    for (int mask = 16; mask > 0; mask >>= 1) {
                        const float o_p = __shfl_xor_sync(0xFFFFFFFFu, best_p, mask, 32);
                        const int   o_i = __shfl_xor_sync(0xFFFFFFFFu, best_i, mask, 32);
                        if (o_p > best_p || (o_p == best_p && o_i < best_i)) {
                            best_p = o_p;
                            best_i = o_i;
                        }
                    }
                    w_sum += best_p;
                    if (lane == k) {
                        sel_i = best_i;
                        sel_p = best_p;
                    }
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        if (i_reg[u] == best_i) p_reg[u] = -1.0f;
                    }
                }
                if (lane < QWEN4_N_EXPERTS_ACTIVE) {
                    const float denom = fmaxf(w_sum, 0.00006103515625f);
                    const float nw    = sel_p / denom;
                    d_top_idx[lane]   = sel_i;
                    d_top_w[lane]     = nw;
                    if (d_host_top_idx != nullptr) d_host_top_idx[lane] = sel_i;
                    if (d_host_top_w   != nullptr) d_host_top_w[lane]   = nw;
                }
            }
        }
    } else if (gate_inp_shexp_bf16 != nullptr) {
        __shared__ float smem0[4];
        const uint4*  w4 = (const uint4*)gate_inp_shexp_bf16;
        const float4* x4 = (const float4*)x;
        float acc = 0.0f;
        #pragma unroll
        for (int j = tid; j < QWEN4_N_EMBD / 8; j += 128) {
            acc += dot_bf16x8_fp32x8(w4[j], __ldg(x4 + j * 2 + 0), __ldg(x4 + j * 2 + 1));
        }
        acc = block_sum<4>(acc, smem0);
        if (tid == 0) {
            logits_and_shexp[QWEN4_N_EXPERTS] = 1.0f / (1.0f + __expf(-acc));
        }
    }
}

// Fused 11-Expert Decode Gate + Up + SwiGLU + Q8_1 (k = 0..9 Routed Experts + k = 10 Shared Expert)
// Launch: grid = dim3(20, 11), block = dim3(32, 8) (256 threads)
template <int TY_EXP, int LANES>
__global__ void __launch_bounds__(256) mmvq_gate_up_swiglu_q8_1_shexp_routed_kernel(
    const block_q8_1* __restrict__ x_q8_1,
    const uint8_t* __restrict__ w_gate_shexp, int32_t gate_shexp_type, size_t gate_shexp_row_bytes,
    const uint8_t* __restrict__ w_up_shexp,   int32_t up_shexp_type,   size_t up_shexp_row_bytes,
    const uint8_t* __restrict__ base_vram_exp,
    const uint8_t* const* __restrict__ d_vram_exp_table,
    size_t gate_exp_bytes,
    size_t expert_blob_bytes,
    size_t exp_row_bytes,
    const int32_t* __restrict__ d_top_idx,
    block_q8_1* __restrict__ out_q8_1)
{
    const int k = blockIdx.y; // 0..9 = routed, 10 = shared expert
    const uint8_t* wg_exp = nullptr;
    const uint8_t* wu_exp = nullptr;

    if (k < QWEN4_N_EXPERTS_ACTIVE) {
        const int eid = d_top_idx[k];
        const uint8_t* exp_base = (base_vram_exp != nullptr)
            ? (base_vram_exp + (size_t)eid * expert_blob_bytes)
            : d_vram_exp_table[eid];
        if (exp_base == nullptr) return; // L1 miss on 1-GPU -> exits in 1 cycle!
        wg_exp = exp_base;
        wu_exp = exp_base + gate_exp_bytes;
    } else {
        wg_exp = w_gate_shexp;
        wu_exp = w_up_shexp;
    }

    __shared__ block_q8_1 s_x[80];
    __shared__ float s_swiglu[32];

    const int tid = threadIdx.y * 32 + threadIdx.x;
    copy_q8_1_smem(s_x, x_q8_1, QWEN4_N_EMBD / 32, tid);
    __syncthreads();

    const int bx = blockIdx.x; // 0..19

    if (k < QWEN4_N_EXPERTS_ACTIVE) {
        constexpr int SUBWARPS_PER_BLOCK = 256 / LANES;
        constexpr int PASSES = 32 / SUBWARPS_PER_BLOCK;
        const int sub_id   = tid / LANES;
        const int sub_lane = tid & (LANES - 1);
        constexpr int nb = QWEN4_N_EMBD / Fmt<TY_EXP>::qk;
        #pragma unroll
        for (int p = 0; p < PASSES; ++p) {
            const int r   = p * SUBWARPS_PER_BLOCK + sub_id;
            const int row = bx * 32 + r;
            float g = 0.0f, u = 0.0f;
            row_dot2_subwarp<TY_EXP, LANES>(
                wg_exp + (size_t)row * exp_row_bytes,
                wu_exp + (size_t)row * exp_row_bytes,
                s_x, nb, sub_lane, g, u);
            if (sub_lane == 0) {
                s_swiglu[r] = (g / (1.0f + expf(-g))) * u;
            }
        }
    } else {
        constexpr int LANES_SH = 16;
        constexpr int SUBWARPS_SH = 256 / LANES_SH; // 16
        constexpr int PASSES_SH = 32 / SUBWARPS_SH; // 2
        const int sub_id   = tid / LANES_SH;
        const int sub_lane = tid & (LANES_SH - 1);
        if (gate_shexp_type == up_shexp_type) {
            #pragma unroll
            for (int p = 0; p < PASSES_SH; ++p) {
                const int r   = p * SUBWARPS_SH + sub_id;
                const int row = bx * 32 + r;
                float g = 0.0f, u = 0.0f;
                if (gate_shexp_type == TY_EXP) {
                    row_dot2_subwarp<TY_EXP, LANES_SH>(
                        wg_exp + (size_t)row * gate_shexp_row_bytes,
                        wu_exp + (size_t)row * up_shexp_row_bytes,
                        s_x, QWEN4_N_EMBD / Fmt<TY_EXP>::qk, sub_lane, g, u);
                } else {
                    row_dot2_any_subwarp<LANES_SH>(
                        gate_shexp_type,
                        wg_exp + (size_t)row * gate_shexp_row_bytes,
                        wu_exp + (size_t)row * up_shexp_row_bytes,
                        s_x, QWEN4_N_EMBD, sub_lane, g, u);
                }
                if (sub_lane == 0) {
                    s_swiglu[r] = (g / (1.0f + expf(-g))) * u;
                }
            }
        } else {
            #pragma unroll
            for (int p = 0; p < PASSES_SH; ++p) {
                const int r   = p * SUBWARPS_SH + sub_id;
                const int row = bx * 32 + r;
                const float g = (gate_shexp_type == TY_EXP)
                    ? row_dot_subwarp<TY_EXP, LANES_SH>(wg_exp + (size_t)row * gate_shexp_row_bytes, s_x, QWEN4_N_EMBD / Fmt<TY_EXP>::qk, sub_lane)
                    : row_dot_any_subwarp<LANES_SH>(gate_shexp_type, wg_exp + (size_t)row * gate_shexp_row_bytes, s_x, QWEN4_N_EMBD, sub_lane);
                const float u = (up_shexp_type == TY_EXP)
                    ? row_dot_subwarp<TY_EXP, LANES_SH>(wu_exp + (size_t)row * up_shexp_row_bytes, s_x, QWEN4_N_EMBD / Fmt<TY_EXP>::qk, sub_lane)
                    : row_dot_any_subwarp<LANES_SH>(up_shexp_type, wu_exp + (size_t)row * up_shexp_row_bytes, s_x, QWEN4_N_EMBD, sub_lane);
                if (sub_lane == 0) {
                    s_swiglu[r] = (g / (1.0f + expf(-g))) * u;
                }
            }
        }
    }
    __syncthreads();

    if (threadIdx.y == 0) {
        const int lane = threadIdx.x;
        const float xi = q8_1_finite(s_swiglu[lane]);
        float amax = fabsf(xi);
        float sum  = xi;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask, 32));
            sum += __shfl_xor_sync(0xFFFFFFFFu, sum, mask, 32);
        }
        const float d = amax * (1.0f / 127.0f);
        const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / d);
        block_q8_1* blk = out_q8_1 + k * 20 + bx;
        blk->qs[lane] = q;
        if (lane == 0) {
            blk->ds = make_half2(__float2half(d), __float2half(sum));
        }
    }
}

// Fused 11-Expert Decode Down-Projection (k = 0..9 Routed Experts + k = 10 Shared Expert, 2 rows per subwarp -> 440 blocks = 1 single wave!)
template <int TY_EXP, int LANES>
__global__ void __launch_bounds__(256) mmvq_down_shexp_routed_kernel(
    const block_q8_1* __restrict__ in_q8_1,
    const uint8_t* __restrict__ w_down_shexp, int32_t down_shexp_type, size_t down_shexp_row_bytes,
    const uint8_t* __restrict__ base_vram_exp,
    const uint8_t* const* __restrict__ d_vram_exp_table,
    size_t down_exp_offset_bytes,
    size_t expert_blob_bytes,
    size_t exp_down_row_bytes,
    const int32_t* __restrict__ d_top_idx,
    float* __restrict__ y_down_11)
{
    constexpr int SUBWARPS_PER_BLOCK = 256 / LANES;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int sub_id   = tid / LANES;
    const int sub_lane = tid & (LANES - 1);
    const int row0     = blockIdx.x * (2 * SUBWARPS_PER_BLOCK) + sub_id;
    const int row1     = row0 + SUBWARPS_PER_BLOCK;
    const int k        = blockIdx.y; // 0..9 = routed, 10 = shared expert

    const uint8_t* wd_exp = nullptr;
    if (k < QWEN4_N_EXPERTS_ACTIVE) {
        const int eid = d_top_idx[k];
        const uint8_t* exp_base = (base_vram_exp != nullptr)
            ? (base_vram_exp + (size_t)eid * expert_blob_bytes)
            : d_vram_exp_table[eid];
        if (exp_base == nullptr) {
            if (sub_lane == 0) {
                if (row0 < QWEN4_N_EMBD) y_down_11[(size_t)k * QWEN4_N_EMBD + row0] = 0.0f;
                if (row1 < QWEN4_N_EMBD) y_down_11[(size_t)k * QWEN4_N_EMBD + row1] = 0.0f;
            }
            return;
        }
        wd_exp = exp_base + down_exp_offset_bytes;
    } else {
        wd_exp = w_down_shexp;
    }

    __shared__ block_q8_1 s_x[20];
    copy_q8_1_smem(s_x, in_q8_1 + k * 20, QWEN4_MOE_FF_DIM / 32, tid);
    __syncthreads();

    if (row0 >= QWEN4_N_EMBD) return;

    float s0 = 0.0f, s1 = 0.0f;
    if (k < QWEN4_N_EXPERTS_ACTIVE) {
        constexpr int nb = QWEN4_MOE_FF_DIM / Fmt<TY_EXP>::qk;
        const uint8_t* wr0 = wd_exp + (size_t)row0 * exp_down_row_bytes;
        const uint8_t* wr1 = (row1 < QWEN4_N_EMBD) ? (wd_exp + (size_t)row1 * exp_down_row_bytes) : wr0;
        row_dot2_subwarp<TY_EXP, LANES>(wr0, wr1, s_x, nb, sub_lane, s0, s1);
    } else {
        const uint8_t* wr0 = wd_exp + (size_t)row0 * down_shexp_row_bytes;
        const uint8_t* wr1 = (row1 < QWEN4_N_EMBD) ? (wd_exp + (size_t)row1 * down_shexp_row_bytes) : wr0;
        if (down_shexp_type == 20) {
            row_dot2_subwarp<20, LANES>(wr0, wr1, s_x, QWEN4_MOE_FF_DIM / Fmt<20>::qk, sub_lane, s0, s1);
        } else if (down_shexp_type == 8) {
            row_dot2_subwarp<8, LANES>(wr0, wr1, s_x, QWEN4_MOE_FF_DIM / Fmt<8>::qk, sub_lane, s0, s1);
        } else {
            row_dot2_any_subwarp<LANES>(down_shexp_type, wr0, wr1, s_x, QWEN4_MOE_FF_DIM, sub_lane, s0, s1);
        }
    }
    if (sub_lane == 0) {
        y_down_11[(size_t)k * QWEN4_N_EMBD + row0] = s0;
        if (row1 < QWEN4_N_EMBD) y_down_11[(size_t)k * QWEN4_N_EMBD + row1] = s1;
    }
}

// Down-Projection for `n_gpu_misses` missed routed experts, writing directly into `y_down_11[d_slot[gm] * 2560]` (2 rows per subwarp)
// Also updates `d_vram_exp_table` in VRAM for any newly cached LRU slots, eliminating a separate H2D cudaMemcpyAsync per layer!
template <int TY, int LANES>
__global__ void __launch_bounds__(256) mmvq_down_misses_kernel(
    const Qwen4ActiveExpertPtrs* __restrict__ d_miss_ptrs,
    size_t row_bytes,
    const block_q8_1* __restrict__ in_q8_1,
    float* __restrict__ y_down_11,
    uint8_t** __restrict__ d_vram_exp_table)
{
    __shared__ block_q8_1 s_x[20];
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int gm  = blockIdx.y;
    if (blockIdx.x == 0 && gm == 0 && tid == 0 && d_vram_exp_table != nullptr) {
        const int n_misses = (int)gridDim.y;
        for (int g = 0; g < n_misses; ++g) {
            const int old_e = d_miss_ptrs->old_eid[g];
            const int new_e = d_miss_ptrs->new_eid[g];
            if (old_e >= 0) d_vram_exp_table[old_e] = nullptr;
            if (new_e >= 0) d_vram_exp_table[new_e] = (uint8_t*)d_miss_ptrs->gate[g];
        }
    }
    copy_q8_1_smem(s_x, in_q8_1 + gm * 20, QWEN4_MOE_FF_DIM / 32, tid);
    __syncthreads();

    constexpr int SUBWARPS_PER_BLOCK = 256 / LANES;
    const int sub_id   = tid / LANES;
    const int sub_lane = tid & (LANES - 1);
    const int row0     = blockIdx.x * (2 * SUBWARPS_PER_BLOCK) + sub_id;
    const int row1     = row0 + SUBWARPS_PER_BLOCK;
    if (row0 >= QWEN4_N_EMBD) return;

    const uint8_t* wd_exp = (const uint8_t*)d_miss_ptrs->down[gm];
    const int orig_k      = d_miss_ptrs->slot[gm];
    constexpr int nb      = QWEN4_MOE_FF_DIM / Fmt<TY>::qk;
    const uint8_t* wr0    = wd_exp + (size_t)row0 * row_bytes;
    const uint8_t* wr1    = (row1 < QWEN4_N_EMBD) ? (wd_exp + (size_t)row1 * row_bytes) : wr0;
    float s0 = 0.0f, s1 = 0.0f;
    row_dot2_subwarp<TY, LANES>(wr0, wr1, s_x, nb, sub_lane, s0, s1);
    if (sub_lane == 0) {
        y_down_11[(size_t)orig_k * QWEN4_N_EMBD + row0] = s0;
        if (row1 < QWEN4_N_EMBD) y_down_11[(size_t)orig_k * QWEN4_N_EMBD + row1] = s1;
    }
}

// Fused Shared Expert Gate Scale + 10 Routed Experts Weighted Accumulation + 4-Stream Gated Residual Write into R[4, 2560]
// + Optional Fused Per-Stream RMSNorm for Next Layer's GR Read (eliminating 46 kernel launches & 46 R round-trips per token!)
// Launch: <<<4, 640>>> (4 blocks of 20 warps = 1 float4 per thread per stream)
__global__ void __launch_bounds__(640) moe_accum_gr_write_and_norm_kernel(
    float* __restrict__ R,
    const float* __restrict__ down_11,
    const float* __restrict__ d_top_w,
    const float* __restrict__ d_shexp_gate,
    const float* __restrict__ d_cpu_moe_out,
    const float* __restrict__ next_norm_w,
    float* __restrict__ next_normed,
    float eps)
{
    __shared__ float smem[20];
    const int s    = blockIdx.x;  // 0..3
    const int idx4 = threadIdx.x; // 0..639

    const float4* down4 = (const float4*)down_11;
    const float sh_w    = __ldg(d_shexp_gate);
    const float4 sh4    = __ldg(down4 + 10 * (QWEN4_N_EMBD / 4) + idx4);
    float4 acc = make_float4(sh_w * sh4.x, sh_w * sh4.y, sh_w * sh4.z, sh_w * sh4.w);

    #pragma unroll
    for (int k = 0; k < QWEN4_N_EXPERTS_ACTIVE; ++k) {
        const float wk   = __ldg(d_top_w + k);
        const float4 dk4 = __ldg(down4 + k * (QWEN4_N_EMBD / 4) + idx4);
        acc.x += wk * dk4.x;
        acc.y += wk * dk4.y;
        acc.z += wk * dk4.z;
        acc.w += wk * dk4.w;
    }
    if (d_cpu_moe_out != nullptr) {
        const float4 cpu4 = ((const float4*)d_cpu_moe_out)[idx4];
        acc.x += cpu4.x;
        acc.y += cpu4.y;
        acc.z += cpu4.z;
        acc.w += cpu4.w;
    }

    const float gs = (R + QWEN4_STREAM_TOTAL)[s];
    float4* rs4 = (float4*)(R + (size_t)s * QWEN4_N_EMBD);
    float4 r4 = rs4[idx4];
    r4.x += gs * acc.x;
    r4.y += gs * acc.y;
    r4.z += gs * acc.z;
    r4.w += gs * acc.w;
    rs4[idx4] = r4;

    if (next_norm_w != nullptr && next_normed != nullptr) {
        float ss = r4.x * r4.x + r4.y * r4.y + r4.z * r4.z + r4.w * r4.w;
        ss = block_sum<20>(ss, smem);
        const float scale = rsqrtf(ss * (1.0f / (float)QWEN4_N_EMBD) + eps);
        const float4 w4 = ((const float4*)(next_norm_w + (size_t)s * QWEN4_N_EMBD))[idx4];
        ((float4*)(next_normed + (size_t)s * QWEN4_N_EMBD))[idx4] = make_float4(
            r4.x * scale * w4.x, r4.y * scale * w4.y, r4.z * scale * w4.z, r4.w * scale * w4.w);
    }
}
__global__ void router_top10_kernel(
    float* __restrict__ logits,
    int32_t* __restrict__ top_idx,
    float* __restrict__ top_w,
    int32_t* __restrict__ host_top_idx = nullptr,
    float* __restrict__ host_top_w = nullptr)
{
    __shared__ float probs[QWEN4_N_EXPERTS];
    __shared__ float smem[8];

    const int m    = blockIdx.x;
    const int e    = threadIdx.x; // 0..255
    const int warp = e / 32;
    const int lane = e & 31;

    float* logits_m        = logits  + (size_t)m * QWEN4_N_EXPERTS;
    int32_t* top_idx_m     = top_idx + (size_t)m * QWEN4_N_EXPERTS_ACTIVE;
    float* top_w_m         = top_w   + (size_t)m * QWEN4_N_EXPERTS_ACTIVE;
    int32_t* host_idx_m    = (host_top_idx != nullptr) ? (host_top_idx + (size_t)m * QWEN4_N_EXPERTS_ACTIVE) : nullptr;
    float* host_w_m        = (host_top_w   != nullptr) ? (host_top_w   + (size_t)m * QWEN4_N_EXPERTS_ACTIVE) : nullptr;

    float v = logits_m[e];
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

    const float p = expf(logits_m[e] - max_l);
    const float sum_p = block_sum<8>(p, smem);
    const float prob_e = p / sum_p;
    probs[e]    = prob_e;
    logits_m[e] = prob_e;
    __syncthreads();

    if (warp == 0) {
        float p_reg[8];
        int   i_reg[8];
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int idx = lane * 8 + u;
            p_reg[u] = probs[idx];
            i_reg[u] = idx;
        }
        float w_sum = 0.0f;
        int   sel_i = 0;
        float sel_p = 0.0f;
        #pragma unroll
        for (int k = 0; k < QWEN4_N_EXPERTS_ACTIVE; ++k) {
            float best_p = p_reg[0];
            int   best_i = i_reg[0];
            #pragma unroll
            for (int u = 1; u < 8; ++u) {
                if (p_reg[u] > best_p) {
                    best_p = p_reg[u];
                    best_i = i_reg[u];
                }
            }
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                const float o_p = __shfl_xor_sync(0xFFFFFFFFu, best_p, mask, 32);
                const int   o_i = __shfl_xor_sync(0xFFFFFFFFu, best_i, mask, 32);
                if (o_p > best_p || (o_p == best_p && o_i < best_i)) {
                    best_p = o_p;
                    best_i = o_i;
                }
            }
            w_sum += best_p;
            if (lane == k) {
                sel_i = best_i;
                sel_p = best_p;
            }
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                if (i_reg[u] == best_i) p_reg[u] = -1.0f;
            }
        }
        if (lane < QWEN4_N_EXPERTS_ACTIVE) {
            const float denom = fmaxf(w_sum, 0.00006103515625f); // 2^-14 clamp
            const float nw    = sel_p / denom;
            top_idx_m[lane]   = sel_i;
            top_w_m[lane]     = nw;
            if (host_idx_m != nullptr) host_idx_m[lane] = sel_i;
            if (host_w_m   != nullptr) host_w_m[lane]   = nw;
        }
    }
}

// Fused SwiGLU + Q8_1 quantization across `gridDim.x` experts (1 for shared expert, 10 for routed experts)
// Launch: <<<n_experts, 640>>> (20 warps of 32 threads = 640 threads per block)
__global__ void __launch_bounds__(640) swiglu_and_q8_1_kernel(
    float* __restrict__ gate,
    const float* __restrict__ up,
    block_q8_1* __restrict__ out_q8_1,
    uint16_t* __restrict__ bf16_out = nullptr)
{
    const int k = blockIdx.x; // 0..n_experts-1
    const int tid = threadIdx.x; // 0..639
    const int warp = tid / 32; // 0..19
    const int lane = tid & 31; // 0..31

    const size_t idx = (size_t)k * QWEN4_MOE_FF_DIM + tid;
    const float g = gate[idx];
    const float u = up[idx];
    const float h = (g / (1.0f + expf(-g))) * u;
    if (bf16_out != nullptr) {
        bf16_out[idx] = qwen4_fp32_to_bf16(h);
    } else if (out_q8_1 == nullptr) {
        gate[idx] = h;
    }
    if (out_q8_1 == nullptr) return;
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
    block_q8_1* blk = out_q8_1 + (size_t)k * (QWEN4_MOE_FF_DIM / 32) + warp;
    blk->qs[lane] = q;
    if (lane == 0) {
        blk->ds = make_half2(__float2half(d), __float2half(sum));
    }
}

// Shared expert scalar gate dot + scale into y_moe[M, 2560] (M blocks, 256 threads, uint4 + float4 vectorized; supports in-place down_shexp == y_moe)
__global__ void shared_expert_gate_scale_kernel(
    const float* __restrict__ x_f32,
    const uint16_t* __restrict__ gate_inp_shexp_bf16,
    const float* down_shexp,
    float* y_moe)
{
    __shared__ float smem[QWEN4_GR_WARPS];
    const int m = blockIdx.x;
    const float4* x4 = (const float4*)(x_f32 + (size_t)m * QWEN4_N_EMBD);
    const uint4*  w4 = (const uint4*)gate_inp_shexp_bf16;

    float dot = 0.0f;
    for (int j = threadIdx.x; j < QWEN4_N_EMBD / 8; j += blockDim.x) {
        dot += dot_bf16x8_fp32x8(w4[j], x4[j * 2 + 0], x4[j * 2 + 1]);
    }
    dot = block_sum<QWEN4_GR_WARPS>(dot, smem);
    const float g = 1.0f / (1.0f + expf(-dot));

    const float4* d4 = (const float4*)(down_shexp + (size_t)m * QWEN4_N_EMBD);
    float4*       y4 = (float4*)(y_moe + (size_t)m * QWEN4_N_EMBD);
    for (int j = threadIdx.x; j < QWEN4_N_EMBD / 4; j += blockDim.x) {
        const float4 dv = d4[j];
        y4[j] = make_float4(g * dv.x, g * dv.y, g * dv.z, g * dv.w);
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

// GPU Expert Grouping Kernel for Batched Prefill (supports any M up to 2064+):
// Groups M * 10 routed expert assignments by expert ID (e = 0..255) into contiguous 8-token tile tasks (0 bytes stack frame!)
// Uses O(M * 10 / 256) two-pass shared-memory atomic scatter instead of O(M * 10) per-expert linear scan.
// When d_vram_exp_table != nullptr (1-GPU L1/Mapped Cache mode), groups all L1 VRAM experts plus single-tile (<= 8 tokens) Mapped L2 experts!
// Launch: <<<1, 256>>>
__global__ void __launch_bounds__(256) build_expert_groups_kernel(
    const int32_t* __restrict__ d_top_idx,
    const float* __restrict__ d_top_w,
    int M,
    Qwen4MoETileTask* __restrict__ d_tasks,
    int32_t* __restrict__ d_num_tasks,
    const uint8_t* const* __restrict__ d_vram_exp_table = nullptr,
    size_t gate_exp_bytes = 0,
    size_t up_exp_bytes = 0,
    const void** __restrict__ d_gate_ptrs = nullptr,
    const void** __restrict__ d_up_ptrs = nullptr,
    const void** __restrict__ d_down_ptrs = nullptr,
    const uint8_t* const* __restrict__ d_decode_exp_table = nullptr)
{
    __shared__ int s_exp_cnt[256];       // 1,024 B
    __shared__ int s_exp_write[256];     // 1,024 B
    __shared__ int s_exp_base_slot[256]; // 1,024 B
    __shared__ int s_task_cnt;

    const int e = threadIdx.x; // 0..255
    s_exp_cnt[e]       = 0;
    s_exp_write[e]     = 0;
    s_exp_base_slot[e] = -1;
    if (e == 0) s_task_cnt = 0;
    __syncthreads();

    const int total_pairs = M * QWEN4_N_EXPERTS_ACTIVE;
    for (int i = e; i < total_pairs; i += 256) {
        const int32_t id = d_top_idx[i];
        if ((unsigned)id < 256u) {
            atomicAdd(&s_exp_cnt[id], 1);
        }
    }
    __syncthreads();

    const int total_cnt        = s_exp_cnt[e];
    const uint8_t* vram_base   = (d_vram_exp_table != nullptr) ? d_vram_exp_table[e] : nullptr;
    const uint8_t* mapped_base = (vram_base == nullptr && d_decode_exp_table != nullptr) ? d_decode_exp_table[e] : nullptr;
    const bool is_vram_exp     = (d_vram_exp_table == nullptr) || (vram_base != nullptr);

    if (total_cnt > 0 && is_vram_exp) {
        if (d_vram_exp_table != nullptr) {
            d_gate_ptrs[e] = vram_base;
            d_up_ptrs[e]   = vram_base + gate_exp_bytes;
            d_down_ptrs[e] = vram_base + gate_exp_bytes + up_exp_bytes;
        }
        const int n_tiles   = (total_cnt + QWEN4_MOE_TILE_M - 1) / QWEN4_MOE_TILE_M;
        const int base_slot = atomicAdd(&s_task_cnt, n_tiles);
        s_exp_base_slot[e]  = base_slot;
        for (int t = 0; t < n_tiles; ++t) {
            const int rem = total_cnt - t * QWEN4_MOE_TILE_M;
            const int nt  = (rem < QWEN4_MOE_TILE_M) ? rem : QWEN4_MOE_TILE_M;
            d_tasks[base_slot + t].exp_id = e;
            d_tasks[base_slot + t].n_toks = nt;
            if (nt < QWEN4_MOE_TILE_M) {
                for (int j = nt; j < QWEN4_MOE_TILE_M; ++j) {
                    d_tasks[base_slot + t].tok_idx[j] = 0;
                    d_tasks[base_slot + t].tok_w[j]   = 0.0f;
                }
            }
        }
    } else if (total_cnt > 0 && total_cnt <= ((M <= 64) ? QWEN4_MOE_TILE_M : 2) && mapped_base != nullptr) {
        d_gate_ptrs[e] = mapped_base;
        d_up_ptrs[e]   = mapped_base + gate_exp_bytes;
        d_down_ptrs[e] = mapped_base + gate_exp_bytes + up_exp_bytes;
        const int base_slot = atomicAdd(&s_task_cnt, 1);
        s_exp_base_slot[e]  = base_slot;
        d_tasks[base_slot].exp_id = e;
        d_tasks[base_slot].n_toks = total_cnt;
        if (total_cnt < QWEN4_MOE_TILE_M) {
            for (int j = total_cnt; j < QWEN4_MOE_TILE_M; ++j) {
                d_tasks[base_slot].tok_idx[j] = 0;
                d_tasks[base_slot].tok_w[j]   = 0.0f;
            }
        }
    }
    __syncthreads();

    for (int i = e; i < total_pairs; i += 256) {
        const int32_t id = d_top_idx[i];
        if ((unsigned)id < 256u) {
            const int base_slot = s_exp_base_slot[id];
            if (base_slot >= 0) {
                const int c = atomicAdd(&s_exp_write[id], 1);
                const int m = (int)(((uint32_t)i * 52429u) >> 19); // exact i / 10 for i < 262139
                const int slot = base_slot + (c / QWEN4_MOE_TILE_M);
                const int lane = c % QWEN4_MOE_TILE_M;
                d_tasks[slot].tok_idx[lane] = m;
                d_tasks[slot].tok_w[lane]   = d_top_w[i];
            }
        }
    }
    if (e == 0) {
        *d_num_tasks = s_task_cnt;
    }
}

template <int TY, int LANES, int ACTIVE_M>
__device__ __forceinline__ void eval_gate_up_row_tileM(
    const uint8_t* __restrict__ wg_row,
    const uint8_t* __restrict__ wu_row,
    const block_q8_1* __restrict__ s_x,
    int n_q8,
    int nb,
    int sub_lane,
    int n_toks,
    int r,
    float s_up[QWEN4_MOE_TILE_M][32])
{
    using F = Fmt<TY>;
    float acc_g[ACTIVE_M];
    float acc_u[ACTIVE_M];
    #pragma unroll
    for (int t = 0; t < ACTIVE_M; ++t) {
        acc_g[t] = 0.0f;
        acc_u[t] = 0.0f;
    }

    if constexpr (LANES >= F::ipb && (LANES % F::ipb) == 0) {
        constexpr int kbx_step = LANES / F::ipb;
        constexpr int q8_per_blk = F::qk / 32;
        const int iqs = F::step * (sub_lane & (F::ipb - 1));
        for (int kbx = sub_lane / F::ipb; kbx < nb; kbx += kbx_step) {
            F::template dot2_tileM<ACTIVE_M>(wg_row, wu_row, s_x + kbx * q8_per_blk, n_q8, kbx, iqs, acc_g, acc_u);
        }
    } else {
        for (int k = sub_lane; k < nb * F::ipb; k += LANES) {
            const int kbx    = k / F::ipb;
            const int iqs    = F::step * (k % F::ipb);
            const int q8_off = kbx * (F::qk / 32);
            F::template dot2_tileM<ACTIVE_M>(wg_row, wu_row, s_x + q8_off, n_q8, kbx, iqs, acc_g, acc_u);
        }
    }
    #pragma unroll
    for (int t = 0; t < ACTIVE_M; ++t) {
        const float g = subwarp_sum<LANES>(acc_g[t]);
        const float u = subwarp_sum<LANES>(acc_u[t]);
        if (sub_lane == 0 && t < n_toks) {
            s_up[t][r] = (g / (1.0f + __expf(-g))) * u;
        }
    }
}

template <int TY, int LANES, int ACTIVE_M>
__device__ __forceinline__ void eval_gate_up_row_2chunk_tileM(
    const uint8_t* __restrict__ wg_row,
    const uint8_t* __restrict__ wu_row,
    const block_q8_1* __restrict__ x_all,
    block_q8_1* __restrict__ s_x,
    const Qwen4MoETileTask& s_task,
    int n_q8,
    int nb,
    int sub_lane,
    int n_toks,
    int r,
    float s_up[QWEN4_MOE_TILE_M][32])
{
    using F = Fmt<TY>;
    constexpr int CHUNK_Q8   = 40;
    constexpr int q8_per_blk = F::qk / 32;
    constexpr int cur_nb     = CHUNK_Q8 / q8_per_blk;
    constexpr int cur_u32    = CHUNK_Q8 * 9;

    float acc_g[ACTIVE_M];
    float acc_u[ACTIVE_M];
    #pragma unroll
    for (int t = 0; t < ACTIVE_M; ++t) {
        acc_g[t] = 0.0f;
        acc_u[t] = 0.0f;
    }

    #pragma unroll
    for (int c = 0; c < 2; ++c) {
        const int q8_start = c * CHUNK_Q8;
        const int kbx_base = c * cur_nb;
        for (int t = threadIdx.y; t < ACTIVE_M; t += 8) {
            uint32_t* dst_t = (uint32_t*)(s_x + t * CHUNK_Q8);
            if (t < n_toks) {
                const uint32_t* src_t = (const uint32_t*)(x_all + (size_t)s_task.tok_idx[t] * n_q8 + q8_start);
                for (int i = threadIdx.x; i < cur_u32; i += 32) {
                    dst_t[i] = src_t[i];
                }
            } else {
                for (int i = threadIdx.x; i < cur_u32; i += 32) {
                    dst_t[i] = 0u;
                }
            }
        }
        __syncthreads();

        if constexpr (LANES >= F::ipb && (LANES % F::ipb) == 0) {
            constexpr int kbx_step = LANES / F::ipb;
            const int iqs = F::step * (sub_lane & (F::ipb - 1));
            for (int kbx_loc = sub_lane / F::ipb; kbx_loc < cur_nb; kbx_loc += kbx_step) {
                F::template dot2_tileM<ACTIVE_M>(wg_row, wu_row, s_x + kbx_loc * q8_per_blk, CHUNK_Q8, kbx_base + kbx_loc, iqs, acc_g, acc_u);
            }
        } else {
            for (int k = sub_lane; k < cur_nb * F::ipb; k += LANES) {
                const int kbx_loc = k / F::ipb;
                const int iqs     = F::step * (k % F::ipb);
                const int q8_off  = kbx_loc * q8_per_blk;
                F::template dot2_tileM<ACTIVE_M>(wg_row, wu_row, s_x + q8_off, CHUNK_Q8, kbx_base + kbx_loc, iqs, acc_g, acc_u);
            }
        }
        if (c == 0) {
            __syncthreads();
        }
    }

    #pragma unroll
    for (int t = 0; t < ACTIVE_M; ++t) {
        const float g = subwarp_sum<LANES>(acc_g[t]);
        const float u = subwarp_sum<LANES>(acc_u[t]);
        if (sub_lane == 0 && t < n_toks) {
            s_up[t][r] = (g / (1.0f + __expf(-g))) * u;
        }
    }
}

template <int TY, int LANES, int ACTIVE_M>
__device__ __forceinline__ void eval_down2_accum_tileM(
    const uint8_t* __restrict__ wr0,
    const uint8_t* __restrict__ wr1,
    const block_q8_1* __restrict__ s_x,
    int n_q8,
    int nb,
    int sub_lane,
    int n_toks,
    int row0,
    int row1,
    int n_out,
    const Qwen4MoETileTask& s_task,
    float* __restrict__ y_moe)
{
    using F = Fmt<TY>;
    float acc0[ACTIVE_M];
    float acc1[ACTIVE_M];
    #pragma unroll
    for (int t = 0; t < ACTIVE_M; ++t) {
        acc0[t] = 0.0f;
        acc1[t] = 0.0f;
    }

    if constexpr (LANES >= F::ipb && (LANES % F::ipb) == 0) {
        constexpr int kbx_step = LANES / F::ipb;
        constexpr int q8_per_blk = F::qk / 32;
        const int iqs = F::step * (sub_lane & (F::ipb - 1));
        for (int kbx = sub_lane / F::ipb; kbx < nb; kbx += kbx_step) {
            F::template dot2_tileM<ACTIVE_M>(wr0, wr1, s_x + kbx * q8_per_blk, n_q8, kbx, iqs, acc0, acc1);
        }
    } else {
        for (int k = sub_lane; k < nb * F::ipb; k += LANES) {
            const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
            const int q8_off = kbx * (F::qk / 32);
            F::template dot2_tileM<ACTIVE_M>(wr0, wr1, s_x + q8_off, n_q8, kbx, iqs, acc0, acc1);
        }
    }
    #pragma unroll
    for (int t = 0; t < ACTIVE_M; ++t) {
        const float s0 = subwarp_sum<LANES>(acc0[t]);
        const float s1 = subwarp_sum<LANES>(acc1[t]);
        if (sub_lane == 0 && t < n_toks) {
            const float w = s_task.tok_w[t];
            float* dst = y_moe + (size_t)s_task.tok_idx[t] * n_out;
            atomicAdd(&dst[row0], s0 * w);
            atomicAdd(&dst[row1], s1 * w);
        }
    }
}

// Grouped Routed Expert Gate + Up + SwiGLU + Q8_1 Kernel (processes up to 16 tokens per expert task with 24.6KB smem = 2 blocks/SM!)
// Launch: grid = dim3(20, max_tasks), block = dim3(32, 8) (256 threads)
template <int TY, int LANES>
__global__ void __launch_bounds__(256, 2) mmvq_gate_up_swiglu_q8_1_tile4_kernel(
    const uint8_t* __restrict__ base_gate,
    size_t gate_stride_bytes,
    const void* const* __restrict__ gate_ptrs,
    const uint8_t* __restrict__ base_up,
    size_t up_stride_bytes,
    const void* const* __restrict__ up_ptrs,
    size_t row_bytes,
    const block_q8_1* __restrict__ x_all,
    const Qwen4MoETileTask* __restrict__ d_tasks,
    const int32_t* __restrict__ d_num_tasks,
    block_q8_1* __restrict__ out_q8_1,
    int n_in)
{
    const int task_id = blockIdx.y;
    if (d_num_tasks != nullptr && task_id >= *d_num_tasks) return;

    __shared__ block_q8_1 s_x[8 * 80];           // 23,040 B (holds 8*80 in 1 pass OR 16*40 in 2 passes!)
    __shared__ float s_up[QWEN4_MOE_TILE_M][32]; // 2,048 B
    __shared__ Qwen4MoETileTask s_task;

    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int n_q8 = n_in / 32; // 80
    if (tid == 0) s_task = d_tasks[task_id];
    __syncthreads();

    const int n_toks   = s_task.n_toks;
    const int active_m = (n_toks == 1) ? 1 : ((n_toks == 2) ? 2 : ((n_toks <= 4) ? 4 : ((n_toks <= 6) ? 6 : ((n_toks <= 8) ? 8 : ((n_toks <= 12) ? 12 : 16)))));

    const int bx  = blockIdx.x; // 0..19
    const int eid = s_task.exp_id;

    const uint8_t* wg_exp = (base_gate != nullptr)
        ? (base_gate + (size_t)eid * gate_stride_bytes)
        : (const uint8_t*)gate_ptrs[eid];
    const uint8_t* wu_exp = (base_up != nullptr)
        ? (base_up + (size_t)eid * up_stride_bytes)
        : (const uint8_t*)up_ptrs[eid];

    constexpr int SUBWARPS_PER_BLOCK = 256 / LANES;
    constexpr int PASSES = 32 / SUBWARPS_PER_BLOCK;
    const int sub_id   = tid / LANES;
    const int sub_lane = tid & (LANES - 1);
    using F = Fmt<TY>;
    const int nb = n_in / F::qk;

    if (active_m <= 8) {
        const int cur_u32 = n_q8 * 9;
        const int t       = threadIdx.y;
        if (t < active_m) {
            uint32_t* dst_t = (uint32_t*)(s_x + t * n_q8);
            if (t < n_toks) {
                const uint32_t* src_t = (const uint32_t*)(x_all + (size_t)s_task.tok_idx[t] * n_q8);
                for (int i = threadIdx.x; i < cur_u32; i += 32) {
                    dst_t[i] = src_t[i];
                }
            } else {
                for (int i = threadIdx.x; i < cur_u32; i += 32) {
                    dst_t[i] = 0u;
                }
            }
        }
        __syncthreads();

        for (int p = 0; p < PASSES; ++p) {
            const int r   = p * SUBWARPS_PER_BLOCK + sub_id;
            const int row = bx * 32 + r;
            const uint8_t* wg_row = wg_exp + (size_t)row * row_bytes;
            const uint8_t* wu_row = wu_exp + (size_t)row * row_bytes;

            if (active_m == 1) {
                eval_gate_up_row_tileM<TY, LANES, 1>(wg_row, wu_row, s_x, n_q8, nb, sub_lane, n_toks, r, s_up);
            } else if (active_m == 2) {
                eval_gate_up_row_tileM<TY, LANES, 2>(wg_row, wu_row, s_x, n_q8, nb, sub_lane, n_toks, r, s_up);
            } else if (active_m == 4) {
                eval_gate_up_row_tileM<TY, LANES, 4>(wg_row, wu_row, s_x, n_q8, nb, sub_lane, n_toks, r, s_up);
            } else if (active_m == 6) {
                eval_gate_up_row_tileM<TY, LANES, 6>(wg_row, wu_row, s_x, n_q8, nb, sub_lane, n_toks, r, s_up);
            } else {
                eval_gate_up_row_tileM<TY, LANES, 8>(wg_row, wu_row, s_x, n_q8, nb, sub_lane, n_toks, r, s_up);
            }
        }
    } else {
        for (int p = 0; p < PASSES; ++p) {
            const int r   = p * SUBWARPS_PER_BLOCK + sub_id;
            const int row = bx * 32 + r;
            const uint8_t* wg_row = wg_exp + (size_t)row * row_bytes;
            const uint8_t* wu_row = wu_exp + (size_t)row * row_bytes;

            if (active_m == 12) {
                eval_gate_up_row_2chunk_tileM<TY, LANES, 12>(wg_row, wu_row, x_all, s_x, s_task, n_q8, nb, sub_lane, n_toks, r, s_up);
            } else {
                eval_gate_up_row_2chunk_tileM<TY, LANES, 16>(wg_row, wu_row, x_all, s_x, s_task, n_q8, nb, sub_lane, n_toks, r, s_up);
            }
        }
    }
    __syncthreads();

    for (int t = threadIdx.y; t < n_toks; t += 8) {
        const int lane = threadIdx.x;
        const float xi = q8_1_finite(s_up[t][lane]);
        float amax = fabsf(xi);
        float sum  = xi;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask, 32));
            sum += __shfl_xor_sync(0xFFFFFFFFu, sum, mask, 32);
        }
        const float d = amax * (1.0f / 127.0f);
        const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / d);
        block_q8_1* blk = out_q8_1 + ((size_t)task_id * QWEN4_MOE_TILE_M + t) * 20 + bx;
        blk->qs[lane] = q;
        if (lane == 0) {
            blk->ds = make_half2(__float2half(d), __float2half(sum));
        }
    }
}

// Grouped Routed Expert Down-Projection + Weighted Atomic Accumulation into y_moe[M, 2560] (2 rows per subwarp, up to 16 tokens per task!)
// Launch: grid = dim3((2560 + 2 * ROWS_PER_BLOCK - 1) / (2 * ROWS_PER_BLOCK), max_tasks), block = dim3(32, 8)
template <int TY, int LANES>
__global__ void __launch_bounds__(256, 2) mmvq_down_accum_tile4_kernel(
    const uint8_t* __restrict__ base_down,
    size_t down_stride_bytes,
    const void* const* __restrict__ down_ptrs,
    size_t row_bytes,
    const block_q8_1* __restrict__ in_q8_1,
    const Qwen4MoETileTask* __restrict__ d_tasks,
    const int32_t* __restrict__ d_num_tasks,
    float* __restrict__ y_moe,
    int n_in,
    int n_out)
{
    const int task_id = blockIdx.y;
    if (d_num_tasks != nullptr && task_id >= *d_num_tasks) return;

    __shared__ block_q8_1 s_x[QWEN4_MOE_TILE_M * 20]; // 16 tokens * 20 blocks = 11,520 B (11.25 KB!)
    __shared__ Qwen4MoETileTask s_task;

    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int n_q8 = n_in / 32; // 20
    if (tid == 0) s_task = d_tasks[task_id];
    __syncthreads();

    const int n_toks   = s_task.n_toks;
    const int active_m = (n_toks == 1) ? 1 : ((n_toks == 2) ? 2 : ((n_toks <= 4) ? 4 : ((n_toks <= 6) ? 6 : ((n_toks <= 8) ? 8 : ((n_toks <= 12) ? 12 : 16)))));
    const block_q8_1* x_base = in_q8_1 + (size_t)task_id * QWEN4_MOE_TILE_M * n_q8;
    const uint32_t* src_u32 = (const uint32_t*)x_base;
    uint32_t* dst_u32 = (uint32_t*)s_x;
    const int valid_u32 = n_toks * n_q8 * 9;
    const int max_u32   = active_m * n_q8 * 9;
    for (int i = tid; i < max_u32; i += 256) {
        dst_u32[i] = (i < valid_u32) ? src_u32[i] : 0u;
    }
    __syncthreads();

    constexpr int ROWS_PER_BLOCK = 256 / LANES;
    const int sub_id   = tid / LANES;
    const int sub_lane = tid & (LANES - 1);
    const int row0     = blockIdx.x * (2 * ROWS_PER_BLOCK) + sub_id;
    const int row1     = row0 + ROWS_PER_BLOCK;
    if (row0 >= n_out) return;

    const int eid = s_task.exp_id;
    const uint8_t* wd_exp = (base_down != nullptr)
        ? (base_down + (size_t)eid * down_stride_bytes)
        : (const uint8_t*)down_ptrs[eid];
    const uint8_t* wr0 = wd_exp + (size_t)row0 * row_bytes;
    const uint8_t* wr1 = (row1 < n_out) ? (wd_exp + (size_t)row1 * row_bytes) : wr0;

    using F = Fmt<TY>;
    const int nb = n_in / F::qk;

    if (active_m == 1) {
        eval_down2_accum_tileM<TY, LANES, 1>(wr0, wr1, s_x, n_q8, nb, sub_lane, n_toks, row0, row1, n_out, s_task, y_moe);
    } else if (active_m == 2) {
        eval_down2_accum_tileM<TY, LANES, 2>(wr0, wr1, s_x, n_q8, nb, sub_lane, n_toks, row0, row1, n_out, s_task, y_moe);
    } else if (active_m == 4) {
        eval_down2_accum_tileM<TY, LANES, 4>(wr0, wr1, s_x, n_q8, nb, sub_lane, n_toks, row0, row1, n_out, s_task, y_moe);
    } else if (active_m == 6) {
        eval_down2_accum_tileM<TY, LANES, 6>(wr0, wr1, s_x, n_q8, nb, sub_lane, n_toks, row0, row1, n_out, s_task, y_moe);
    } else if (active_m == 8) {
        eval_down2_accum_tileM<TY, LANES, 8>(wr0, wr1, s_x, n_q8, nb, sub_lane, n_toks, row0, row1, n_out, s_task, y_moe);
    } else if (active_m == 12) {
        eval_down2_accum_tileM<TY, LANES, 12>(wr0, wr1, s_x, n_q8, nb, sub_lane, n_toks, row0, row1, n_out, s_task, y_moe);
    } else {
        eval_down2_accum_tileM<TY, LANES, 16>(wr0, wr1, s_x, n_q8, nb, sub_lane, n_toks, row0, row1, n_out, s_task, y_moe);
    }
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

void qwen4_quantize_q8_1_batchM_cuda(const float* x, void* vx_q8_1, int M, int n_cols, cudaStream_t stream) {
    if (M <= 0) return;
    const int n_blocks = (M * n_cols) / QK8_1;
    const dim3 block(32, 4);
    const dim3 grid((n_blocks + 3) / 4);
    quantize_q8_1_kernel<<<grid, block, 0, stream>>>(x, (block_q8_1*)vx_q8_1, n_blocks);
}

namespace {

template <int TY>
void launch_mmvq_subwarp_typed(
    const uint8_t* w_ptr,
    size_t row_bytes,
    const block_q8_1* x_ptr,
    float* y,
    int n_cols,
    int n_rows,
    cudaStream_t stream)
{
    using F = Fmt<TY>;
    const int total_iters = (n_cols / F::qk) * F::ipb;
    const dim3 block(32, 8); // 256 threads
    if (n_rows >= 32768 && n_cols == 2560 && (total_iters & 15) == 0) {
        const dim3 grid((n_rows + 31) / 32);
        mmvq_subwarp_2row_kernel<TY, 16, 80><<<grid, block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows);
    } else if (n_cols >= 6144 && (total_iters & 31) == 0) {
        const dim3 grid((n_rows + 7) / 8);
        mmvq_subwarp_kernel<TY, 32><<<grid, block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows);
    } else if ((total_iters & 15) == 0) {
        const dim3 grid((n_rows + 15) / 16);
        mmvq_subwarp_kernel<TY, 16><<<grid, block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows);
    } else if ((total_iters & 7) == 0) {
        const dim3 grid((n_rows + 31) / 32);
        mmvq_subwarp_kernel<TY, 8><<<grid, block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows);
    } else {
        const dim3 grid((n_rows + 63) / 64);
        mmvq_subwarp_kernel<TY, 4><<<grid, block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows);
    }
}

template <int TY>
void launch_mmvq_tileM_typed(
    const uint8_t* w_ptr,
    size_t row_bytes,
    const block_q8_1* x_ptr,
    float* y,
    int M,
    int n_cols,
    int n_rows,
    cudaStream_t stream)
{
    using F = Fmt<TY>;
    const int total_iters = (n_cols / F::qk) * F::ipb;
    const dim3 block(32, 8);
    if (M >= 12) {
        constexpr int TILE_M = 16; // 18.0 KB smem via CHUNK_Q8=32 across all n_cols (640, 2560, 6144) -> 2 blocks/SM!
        const int gy = (M + TILE_M - 1) / TILE_M;
        if ((total_iters & 15) == 0) {
            mmvq_tileM_kernel<TY, 16, TILE_M><<<dim3((n_rows + 31) / 32, gy), block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, M, n_cols, n_rows);
        } else if ((total_iters & 7) == 0) {
            mmvq_tileM_kernel<TY, 8, TILE_M><<<dim3((n_rows + 63) / 64, gy), block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, M, n_cols, n_rows);
        } else {
            mmvq_tileM_kernel<TY, 4, TILE_M><<<dim3((n_rows + 127) / 128, gy), block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, M, n_cols, n_rows);
        }
    } else {
        constexpr int TILE_M = 8;
        const int gy = (M + TILE_M - 1) / TILE_M;
        if ((total_iters & 15) == 0) {
            mmvq_tileM_kernel<TY, 16, TILE_M><<<dim3((n_rows + 31) / 32, gy), block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, M, n_cols, n_rows);
        } else if ((total_iters & 7) == 0) {
            mmvq_tileM_kernel<TY, 8, TILE_M><<<dim3((n_rows + 63) / 64, gy), block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, M, n_cols, n_rows);
        } else {
            mmvq_tileM_kernel<TY, 4, TILE_M><<<dim3((n_rows + 127) / 128, gy), block, 0, stream>>>(w_ptr, row_bytes, x_ptr, y, M, n_cols, n_rows);
        }
    }
}

template <int TY>
void launch_mmvq_batch10_typed(
    const uint8_t* bw,
    size_t expert_stride_bytes,
    const void* const* d_expert_ptrs,
    const int32_t* d_top_idx,
    size_t row_bytes,
    const block_q8_1* x_ptr,
    size_t x_expert_stride_blocks,
    float* y,
    int n_cols,
    int n_rows,
    int n_experts,
    cudaStream_t stream)
{
    using F = Fmt<TY>;
    const int total_iters = (n_cols / F::qk) * F::ipb;
    const dim3 block(32, 8);
    if ((total_iters & 15) == 0) {
        mmvq_batch10_subwarp_kernel<TY, 16><<<dim3((n_rows + 15) / 16, n_experts), block, 0, stream>>>(
            bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows);
    } else if ((total_iters & 7) == 0) {
        mmvq_batch10_subwarp_kernel<TY, 8><<<dim3((n_rows + 31) / 32, n_experts), block, 0, stream>>>(
            bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows);
    } else {
        mmvq_batch10_subwarp_kernel<TY, 4><<<dim3((n_rows + 63) / 64, n_experts), block, 0, stream>>>(
            bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows);
    }
}

template <int TY>
void launch_mmvq_gate_up_swiglu_typed(
    const uint8_t* base_gate,
    size_t gate_stride_bytes,
    const void* const* gate_ptrs,
    const uint8_t* base_up,
    size_t up_stride_bytes,
    const void* const* up_ptrs,
    const int32_t* d_top_idx,
    size_t row_bytes,
    const block_q8_1* x_ptr,
    block_q8_1* out_q8_1,
    int n_cols,
    int n_experts,
    cudaStream_t stream)
{
    const dim3 block(32, 8);
    const dim3 grid(QWEN4_MOE_FF_DIM / 32, n_experts); // (20, n_experts)
    mmvq_gate_up_swiglu_q8_1_batch_kernel<TY, 8><<<grid, block, 0, stream>>>(
        base_gate, gate_stride_bytes, gate_ptrs, base_up, up_stride_bytes, up_ptrs, d_top_idx, row_bytes, x_ptr, out_q8_1, n_cols);
}

template <int TY_EXP>
void launch_mmvq_gate_up_shexp_routed_typed(
    const block_q8_1* x_q8_1,
    const uint8_t* w_gate_shexp, int32_t gate_shexp_type, size_t gate_shexp_row_bytes,
    const uint8_t* w_up_shexp,   int32_t up_shexp_type,   size_t up_shexp_row_bytes,
    const uint8_t* base_vram_exp,
    const uint8_t* const* d_vram_exp_table,
    size_t gate_exp_bytes,
    size_t expert_blob_bytes,
    size_t exp_row_bytes,
    const int32_t* d_top_idx,
    block_q8_1* out_q8_1,
    cudaStream_t stream)
{
    const dim3 block(32, 8);
    const dim3 grid(QWEN4_MOE_FF_DIM / 32, QWEN4_N_EXPERTS_ACTIVE + 1); // (20, 11)
    mmvq_gate_up_swiglu_q8_1_shexp_routed_kernel<TY_EXP, 8><<<grid, block, 0, stream>>>(
        x_q8_1, w_gate_shexp, gate_shexp_type, gate_shexp_row_bytes,
        w_up_shexp, up_shexp_type, up_shexp_row_bytes,
        base_vram_exp, d_vram_exp_table, gate_exp_bytes, expert_blob_bytes, exp_row_bytes,
        d_top_idx, out_q8_1);
}

template <int TY_EXP>
void launch_mmvq_down_shexp_routed_typed(
    const block_q8_1* in_q8_1,
    const uint8_t* w_down_shexp, int32_t down_shexp_type, size_t down_shexp_row_bytes,
    const uint8_t* base_vram_exp,
    const uint8_t* const* d_vram_exp_table,
    size_t down_exp_offset_bytes,
    size_t expert_blob_bytes,
    size_t exp_down_row_bytes,
    const int32_t* d_top_idx,
    float* y_down_11,
    cudaStream_t stream)
{
    using F = Fmt<TY_EXP>;
    const int total_iters = (QWEN4_MOE_FF_DIM / F::qk) * F::ipb;
    const dim3 block(32, 8);
    if ((total_iters & 15) == 0) {
        mmvq_down_shexp_routed_kernel<TY_EXP, 16><<<dim3(QWEN4_N_EMBD / 32, QWEN4_N_EXPERTS_ACTIVE + 1), block, 0, stream>>>(
            in_q8_1, w_down_shexp, down_shexp_type, down_shexp_row_bytes,
            base_vram_exp, d_vram_exp_table, down_exp_offset_bytes, expert_blob_bytes, exp_down_row_bytes,
            d_top_idx, y_down_11);
    } else if ((total_iters & 7) == 0) {
        mmvq_down_shexp_routed_kernel<TY_EXP, 8><<<dim3(QWEN4_N_EMBD / 64, QWEN4_N_EXPERTS_ACTIVE + 1), block, 0, stream>>>(
            in_q8_1, w_down_shexp, down_shexp_type, down_shexp_row_bytes,
            base_vram_exp, d_vram_exp_table, down_exp_offset_bytes, expert_blob_bytes, exp_down_row_bytes,
            d_top_idx, y_down_11);
    } else {
        mmvq_down_shexp_routed_kernel<TY_EXP, 4><<<dim3(QWEN4_N_EMBD / 128, QWEN4_N_EXPERTS_ACTIVE + 1), block, 0, stream>>>(
            in_q8_1, w_down_shexp, down_shexp_type, down_shexp_row_bytes,
            base_vram_exp, d_vram_exp_table, down_exp_offset_bytes, expert_blob_bytes, exp_down_row_bytes,
            d_top_idx, y_down_11);
    }
}

template <int TY_EXP>
void launch_mmvq_down_misses_typed(
    const Qwen4ActiveExpertPtrs* d_miss_ptrs,
    size_t row_bytes,
    const block_q8_1* in_q8_1,
    float* y_down_11,
    int n_gpu_misses,
    uint8_t** d_vram_exp_table,
    cudaStream_t stream)
{
    using F = Fmt<TY_EXP>;
    const int total_iters = (QWEN4_MOE_FF_DIM / F::qk) * F::ipb;
    const dim3 block(32, 8);
    if ((total_iters & 7) == 0) {
        mmvq_down_misses_kernel<TY_EXP, 8><<<dim3(QWEN4_N_EMBD / 64, n_gpu_misses), block, 0, stream>>>(
            d_miss_ptrs, row_bytes, in_q8_1, y_down_11, d_vram_exp_table);
    } else {
        mmvq_down_misses_kernel<TY_EXP, 4><<<dim3(QWEN4_N_EMBD / 128, n_gpu_misses), block, 0, stream>>>(
            d_miss_ptrs, row_bytes, in_q8_1, y_down_11, d_vram_exp_table);
    }
}

template <int TY>
void launch_mmvq_gate_up_tile4_typed(
    const uint8_t* base_gate,
    size_t gate_stride_bytes,
    const void* const* gate_ptrs,
    const uint8_t* base_up,
    size_t up_stride_bytes,
    const void* const* up_ptrs,
    size_t row_bytes,
    const block_q8_1* x_all,
    const Qwen4MoETileTask* d_tasks,
    const int32_t* d_num_tasks,
    int max_tasks,
    block_q8_1* out_q8_1,
    int n_cols,
    cudaStream_t stream)
{
    if (max_tasks <= 0) return;
    const dim3 block(32, 8);
    const dim3 grid(QWEN4_MOE_FF_DIM / 32, max_tasks);
    mmvq_gate_up_swiglu_q8_1_tile4_kernel<TY, 8><<<grid, block, 0, stream>>>(
        base_gate, gate_stride_bytes, gate_ptrs, base_up, up_stride_bytes, up_ptrs, row_bytes, x_all, d_tasks, d_num_tasks, out_q8_1, n_cols);
}

template <int TY>
void launch_mmvq_down_tile4_typed(
    const uint8_t* base_down,
    size_t down_stride_bytes,
    const void* const* down_ptrs,
    size_t row_bytes,
    const block_q8_1* in_q8_1,
    const Qwen4MoETileTask* d_tasks,
    const int32_t* d_num_tasks,
    int max_tasks,
    float* y_moe,
    int n_cols,
    int n_rows,
    cudaStream_t stream)
{
    if (max_tasks <= 0) return;
    using F = Fmt<TY>;
    const int total_iters = (n_cols / F::qk) * F::ipb;
    const dim3 block(32, 8);
    if ((total_iters & 7) == 0) {
        mmvq_down_accum_tile4_kernel<TY, 8><<<dim3((n_rows + 63) / 64, max_tasks), block, 0, stream>>>(
            base_down, down_stride_bytes, down_ptrs, row_bytes, in_q8_1, d_tasks, d_num_tasks, y_moe, n_cols, n_rows);
    } else {
        mmvq_down_accum_tile4_kernel<TY, 4><<<dim3((n_rows + 127) / 128, max_tasks), block, 0, stream>>>(
            base_down, down_stride_bytes, down_ptrs, row_bytes, in_q8_1, d_tasks, d_num_tasks, y_moe, n_cols, n_rows);
    }
}

void launch_mmvq_gate_up_swiglu_batch(
    int32_t ggml_type,
    const void* base_gate, size_t gate_stride_bytes, const void* const* gate_ptrs,
    const void* base_up,   size_t up_stride_bytes,   const void* const* up_ptrs,
    const int32_t* d_top_idx,
    const void* vx_q8_1,
    void* vout_q8_1,
    int n_cols,
    int n_experts,
    cudaStream_t stream)
{
    if (n_experts <= 0) return;
    const size_t row_bytes = qwen4_ggml_row_bytes(ggml_type, n_cols);
    const uint8_t* bg = (const uint8_t*)base_gate;
    const uint8_t* bu = (const uint8_t*)base_up;
    const block_q8_1* x_ptr = (const block_q8_1*)vx_q8_1;
    block_q8_1* out_ptr = (block_q8_1*)vout_q8_1;

    switch (ggml_type) {
        case 2:  launch_mmvq_gate_up_swiglu_typed<2> (bg, gate_stride_bytes, gate_ptrs, bu, up_stride_bytes, up_ptrs, d_top_idx, row_bytes, x_ptr, out_ptr, n_cols, n_experts, stream); break;
        case 3:  launch_mmvq_gate_up_swiglu_typed<3> (bg, gate_stride_bytes, gate_ptrs, bu, up_stride_bytes, up_ptrs, d_top_idx, row_bytes, x_ptr, out_ptr, n_cols, n_experts, stream); break;
        case 6:  launch_mmvq_gate_up_swiglu_typed<6> (bg, gate_stride_bytes, gate_ptrs, bu, up_stride_bytes, up_ptrs, d_top_idx, row_bytes, x_ptr, out_ptr, n_cols, n_experts, stream); break;
        case 7:  launch_mmvq_gate_up_swiglu_typed<7> (bg, gate_stride_bytes, gate_ptrs, bu, up_stride_bytes, up_ptrs, d_top_idx, row_bytes, x_ptr, out_ptr, n_cols, n_experts, stream); break;
        case 8:  launch_mmvq_gate_up_swiglu_typed<8> (bg, gate_stride_bytes, gate_ptrs, bu, up_stride_bytes, up_ptrs, d_top_idx, row_bytes, x_ptr, out_ptr, n_cols, n_experts, stream); break;
        case 12: launch_mmvq_gate_up_swiglu_typed<12>(bg, gate_stride_bytes, gate_ptrs, bu, up_stride_bytes, up_ptrs, d_top_idx, row_bytes, x_ptr, out_ptr, n_cols, n_experts, stream); break;
        case 13: launch_mmvq_gate_up_swiglu_typed<13>(bg, gate_stride_bytes, gate_ptrs, bu, up_stride_bytes, up_ptrs, d_top_idx, row_bytes, x_ptr, out_ptr, n_cols, n_experts, stream); break;
        case 14: launch_mmvq_gate_up_swiglu_typed<14>(bg, gate_stride_bytes, gate_ptrs, bu, up_stride_bytes, up_ptrs, d_top_idx, row_bytes, x_ptr, out_ptr, n_cols, n_experts, stream); break;
        case 16: launch_mmvq_gate_up_swiglu_typed<16>(bg, gate_stride_bytes, gate_ptrs, bu, up_stride_bytes, up_ptrs, d_top_idx, row_bytes, x_ptr, out_ptr, n_cols, n_experts, stream); break;
        case 17: launch_mmvq_gate_up_swiglu_typed<17>(bg, gate_stride_bytes, gate_ptrs, bu, up_stride_bytes, up_ptrs, d_top_idx, row_bytes, x_ptr, out_ptr, n_cols, n_experts, stream); break;
        case 18: launch_mmvq_gate_up_swiglu_typed<18>(bg, gate_stride_bytes, gate_ptrs, bu, up_stride_bytes, up_ptrs, d_top_idx, row_bytes, x_ptr, out_ptr, n_cols, n_experts, stream); break;
        case 20: launch_mmvq_gate_up_swiglu_typed<20>(bg, gate_stride_bytes, gate_ptrs, bu, up_stride_bytes, up_ptrs, d_top_idx, row_bytes, x_ptr, out_ptr, n_cols, n_experts, stream); break;
        case 21: launch_mmvq_gate_up_swiglu_typed<21>(bg, gate_stride_bytes, gate_ptrs, bu, up_stride_bytes, up_ptrs, d_top_idx, row_bytes, x_ptr, out_ptr, n_cols, n_experts, stream); break;
        case 22: launch_mmvq_gate_up_swiglu_typed<22>(bg, gate_stride_bytes, gate_ptrs, bu, up_stride_bytes, up_ptrs, d_top_idx, row_bytes, x_ptr, out_ptr, n_cols, n_experts, stream); break;
        case 23: launch_mmvq_gate_up_swiglu_typed<23>(bg, gate_stride_bytes, gate_ptrs, bu, up_stride_bytes, up_ptrs, d_top_idx, row_bytes, x_ptr, out_ptr, n_cols, n_experts, stream); break;
        case 29: launch_mmvq_gate_up_swiglu_typed<29>(bg, gate_stride_bytes, gate_ptrs, bu, up_stride_bytes, up_ptrs, d_top_idx, row_bytes, x_ptr, out_ptr, n_cols, n_experts, stream); break;
        case 42: launch_mmvq_gate_up_swiglu_typed<42>(bg, gate_stride_bytes, gate_ptrs, bu, up_stride_bytes, up_ptrs, d_top_idx, row_bytes, x_ptr, out_ptr, n_cols, n_experts, stream); break;
        default:
            std::fprintf(stderr, "[Qwen4] Unsupported fused gate+up ggml_type=%d\n", ggml_type);
            break;
    }
}

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
    const uint8_t* bw = (const uint8_t*)base_w;
    const block_q8_1* x_ptr = (const block_q8_1*)vx_q8_1;

    switch (ggml_type) {
        case 2:  launch_mmvq_batch10_typed<2> (bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows, n_experts, stream); break;
        case 3:  launch_mmvq_batch10_typed<3> (bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows, n_experts, stream); break;
        case 6:  launch_mmvq_batch10_typed<6> (bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows, n_experts, stream); break;
        case 7:  launch_mmvq_batch10_typed<7> (bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows, n_experts, stream); break;
        case 8:  launch_mmvq_batch10_typed<8> (bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows, n_experts, stream); break;
        case 12: launch_mmvq_batch10_typed<12>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows, n_experts, stream); break;
        case 13: launch_mmvq_batch10_typed<13>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows, n_experts, stream); break;
        case 14: launch_mmvq_batch10_typed<14>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows, n_experts, stream); break;
        case 16: launch_mmvq_batch10_typed<16>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows, n_experts, stream); break;
        case 17: launch_mmvq_batch10_typed<17>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows, n_experts, stream); break;
        case 18: launch_mmvq_batch10_typed<18>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows, n_experts, stream); break;
        case 20: launch_mmvq_batch10_typed<20>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows, n_experts, stream); break;
        case 21: launch_mmvq_batch10_typed<21>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows, n_experts, stream); break;
        case 22: launch_mmvq_batch10_typed<22>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows, n_experts, stream); break;
        case 23: launch_mmvq_batch10_typed<23>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows, n_experts, stream); break;
        case 29: launch_mmvq_batch10_typed<29>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows, n_experts, stream); break;
        case 42: launch_mmvq_batch10_typed<42>(bw, expert_stride_bytes, d_expert_ptrs, d_top_idx, row_bytes, x_ptr, x_expert_stride_blocks, y, n_cols, n_rows, n_experts, stream); break;
        default:
            std::fprintf(stderr, "[Qwen4] Unsupported batch10 MMVQ ggml_type=%d\n", ggml_type);
            break;
    }
}

} // anonymous namespace

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
    const uint8_t* w_ptr = (const uint8_t*)W;
    const block_q8_1* x_ptr = (const block_q8_1*)vx_q8_1;

    switch (ggml_type) {
        case 2:  launch_mmvq_subwarp_typed<2> (w_ptr, row_bytes, x_ptr, y, n_cols, n_rows, stream); break;
        case 3:  launch_mmvq_subwarp_typed<3> (w_ptr, row_bytes, x_ptr, y, n_cols, n_rows, stream); break;
        case 6:  launch_mmvq_subwarp_typed<6> (w_ptr, row_bytes, x_ptr, y, n_cols, n_rows, stream); break;
        case 7:  launch_mmvq_subwarp_typed<7> (w_ptr, row_bytes, x_ptr, y, n_cols, n_rows, stream); break;
        case 8:  launch_mmvq_subwarp_typed<8> (w_ptr, row_bytes, x_ptr, y, n_cols, n_rows, stream); break;
        case 12: launch_mmvq_subwarp_typed<12>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows, stream); break;
        case 13: launch_mmvq_subwarp_typed<13>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows, stream); break;
        case 14: launch_mmvq_subwarp_typed<14>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows, stream); break;
        case 16: launch_mmvq_subwarp_typed<16>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows, stream); break;
        case 17: launch_mmvq_subwarp_typed<17>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows, stream); break;
        case 18: launch_mmvq_subwarp_typed<18>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows, stream); break;
        case 20: launch_mmvq_subwarp_typed<20>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows, stream); break;
        case 21: launch_mmvq_subwarp_typed<21>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows, stream); break;
        case 22: launch_mmvq_subwarp_typed<22>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows, stream); break;
        case 23: launch_mmvq_subwarp_typed<23>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows, stream); break;
        case 29: launch_mmvq_subwarp_typed<29>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows, stream); break;
        case 42: launch_mmvq_subwarp_typed<42>(w_ptr, row_bytes, x_ptr, y, n_cols, n_rows, stream); break;
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
    const dim3 block(32, 8);
    const dim3 grid((n_rows + 7) / 8);
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

void qwen4_bind_device_workspace_cuda(
    int dev,
    void* cublas_handle,
    void* d_bf16_workspace,
    size_t workspace_bytes)
{
    if (dev < 0 || dev >= 2) return;
    g_qwen4_cublas[dev]        = (cublasHandle_t)cublas_handle;
    g_qwen4_bf16_ws[dev]       = (uint16_t*)d_bf16_workspace;
    g_qwen4_bf16_ws_bytes[dev] = workspace_bytes;
}

namespace {

static inline bool qwen4_can_cublas_gemm(int32_t ggml_type, int M, int dev) {
    if (dev < 0 || dev >= 2 || g_qwen4_cublas[dev] == nullptr || g_qwen4_bf16_ws[dev] == nullptr) return false;
    if (ggml_type == QWEN4_GGML_TYPE_BF16) return (M >= 16);
    if (M < 64) return false;
    return (ggml_type == QWEN4_GGML_TYPE_Q8_0 ||
            ggml_type == QWEN4_GGML_TYPE_Q4_K ||
            ggml_type == QWEN4_GGML_TYPE_Q5_K ||
            ggml_type == QWEN4_GGML_TYPE_Q6_K ||
            ggml_type == QWEN4_GGML_TYPE_IQ4_NL ||
            ggml_type == QWEN4_GGML_TYPE_IQ4_XS);
}

bool qwen4_try_cublas_gemm_batchM(
    int32_t ggml_type,
    const void* W,
    const float* x_f32,
    float* y,
    int M,
    int n_rows,
    int n_cols,
    cudaStream_t stream,
    bool reuse_x_bf16 = false)
{
    if ((x_f32 == nullptr && !reuse_x_bf16) || (n_cols & 3) != 0) return false;
    const bool is_bf16 = (ggml_type == QWEN4_GGML_TYPE_BF16);
    if (!is_bf16 && M < 64) return false;
    if (is_bf16 && M < 16) return false;
    if (!is_bf16 &&
        ggml_type != QWEN4_GGML_TYPE_Q8_0 &&
        ggml_type != QWEN4_GGML_TYPE_Q4_K &&
        ggml_type != QWEN4_GGML_TYPE_Q5_K &&
        ggml_type != QWEN4_GGML_TYPE_Q6_K &&
        ggml_type != QWEN4_GGML_TYPE_IQ4_NL &&
        ggml_type != QWEN4_GGML_TYPE_IQ4_XS) {
        return false;
    }

    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 2) return false;
    cublasHandle_t handle = g_qwen4_cublas[dev];
    uint16_t* ws_base     = g_qwen4_bf16_ws[dev];
    const size_t ws_bytes = g_qwen4_bf16_ws_bytes[dev];
    if (handle == nullptr || ws_base == nullptr) return false;

    constexpr size_t MAX_W_ELEMS = 31457280ULL; // 60.0 MiB (fits 12288 x 2560 BF16)
    const size_t w_elems  = is_bf16 ? 0ULL : ((size_t)n_rows * (size_t)n_cols);
    const size_t x_elems  = (size_t)M * (size_t)n_cols;
    if (w_elems > MAX_W_ELEMS) return false;
    const size_t x_ws_off = (w_elems <= 15728640ULL && n_cols > 2560) ? 15728640ULL : MAX_W_ELEMS;
    if ((x_ws_off + x_elems) * sizeof(uint16_t) > ws_bytes) return false;

    const bool is_shexp = ((size_t)n_rows * (size_t)n_cols == 1638400ULL) || (is_bf16 && n_rows == 256 && n_cols == 2560);
    uint16_t* d_w_bf16 = is_shexp ? (ws_base - 35553280ULL) : ws_base;
    uint16_t* d_x_bf16 = is_shexp ? (d_w_bf16 + 1638400ULL)  : (ws_base + x_ws_off);

    if (!reuse_x_bf16) {
        const int x_vec4 = (int)(x_elems / 4);
        f32_to_bf16_vec4_kernel<<<(x_vec4 + 255) / 256, 256, 0, stream>>>(x_f32, d_x_bf16, x_vec4);
    }

    const uint16_t* w_bf16_ptr = (const uint16_t*)W;
    if (!is_bf16) {
        const size_t row_bytes = qwen4_ggml_row_bytes(ggml_type, n_cols);
        const int n_cols4 = n_cols / 4;
        const int w_vec4 = (int)(w_elems / 4);
        const int grid_w = (w_vec4 + 255) / 256;
        const uint8_t* w_raw = (const uint8_t*)W;
        switch (ggml_type) {
            case QWEN4_GGML_TYPE_Q8_0:   dequant_matrix_to_bf16_vec4_kernel<QWEN4_GGML_TYPE_Q8_0>  <<<grid_w, 256, 0, stream>>>(w_raw, row_bytes, d_w_bf16, n_rows, n_cols4); break;
            case QWEN4_GGML_TYPE_Q4_K:   dequant_matrix_to_bf16_vec4_kernel<QWEN4_GGML_TYPE_Q4_K>  <<<grid_w, 256, 0, stream>>>(w_raw, row_bytes, d_w_bf16, n_rows, n_cols4); break;
            case QWEN4_GGML_TYPE_Q5_K:   dequant_matrix_to_bf16_vec4_kernel<QWEN4_GGML_TYPE_Q5_K>  <<<grid_w, 256, 0, stream>>>(w_raw, row_bytes, d_w_bf16, n_rows, n_cols4); break;
            case QWEN4_GGML_TYPE_Q6_K:   dequant_matrix_to_bf16_vec4_kernel<QWEN4_GGML_TYPE_Q6_K>  <<<grid_w, 256, 0, stream>>>(w_raw, row_bytes, d_w_bf16, n_rows, n_cols4); break;
            case QWEN4_GGML_TYPE_IQ4_NL: dequant_matrix_to_bf16_vec4_kernel<QWEN4_GGML_TYPE_IQ4_NL><<<grid_w, 256, 0, stream>>>(w_raw, row_bytes, d_w_bf16, n_rows, n_cols4); break;
            case QWEN4_GGML_TYPE_IQ4_XS: dequant_matrix_to_bf16_vec4_kernel<QWEN4_GGML_TYPE_IQ4_XS><<<grid_w, 256, 0, stream>>>(w_raw, row_bytes, d_w_bf16, n_rows, n_cols4); break;
            default: return false;
        }
        w_bf16_ptr = d_w_bf16;
    }

    cublasSetStream(handle, stream);
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    const cublasStatus_t st = cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        n_rows,
        M,
        n_cols,
        &alpha,
        w_bf16_ptr, CUDA_R_16BF, n_cols,
        d_x_bf16,   CUDA_R_16BF, n_cols,
        &beta,
        y,          CUDA_R_32F,  n_rows,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return (st == CUBLAS_STATUS_SUCCESS);
}

} // anonymous namespace

void qwen4_project_batchM_cuda(
    int32_t ggml_type,
    const void* W,
    const float* x_f32,
    const void* x_q8_1,
    float* y,
    int M,
    int n_rows,
    int n_cols,
    cudaStream_t stream,
    bool reuse_x_bf16)
{
    if (M <= 0) return;
    if (M == 1) {
        qwen4_project_cuda(ggml_type, W, x_f32, x_q8_1, y, n_rows, n_cols, stream);
        return;
    }
    if (qwen4_try_cublas_gemm_batchM(ggml_type, W, x_f32, y, M, n_rows, n_cols, stream, reuse_x_bf16)) {
        return;
    }
    if (ggml_type == QWEN4_GGML_TYPE_BF16) {
        const dim3 block(32, 8);
        if (M >= 12) {
            constexpr int TILE_M = 16;
            const dim3 grid((n_rows + 7) / 8, (M + TILE_M - 1) / TILE_M);
            bf16_gemm_tileM_kernel<TILE_M><<<grid, block, 0, stream>>>((const uint16_t*)W, x_f32, y, M, n_rows, n_cols);
        } else {
            constexpr int TILE_M = 8;
            const dim3 grid((n_rows + 7) / 8, (M + TILE_M - 1) / TILE_M);
            bf16_gemm_tileM_kernel<TILE_M><<<grid, block, 0, stream>>>((const uint16_t*)W, x_f32, y, M, n_rows, n_cols);
        }
    } else if (ggml_type == QWEN4_GGML_TYPE_F16 || ggml_type == QWEN4_GGML_TYPE_F32) {
        for (int m = 0; m < M; ++m) {
            qwen4_project_cuda(ggml_type, W, x_f32 + (size_t)m * n_cols, nullptr, y + (size_t)m * n_rows, n_rows, n_cols, stream);
        }
    } else {
        const size_t row_bytes = qwen4_ggml_row_bytes(ggml_type, n_cols);
        const uint8_t* w_ptr = (const uint8_t*)W;
        const block_q8_1* x_ptr = (const block_q8_1*)x_q8_1;
        switch (ggml_type) {
            case 2:  launch_mmvq_tileM_typed<2> (w_ptr, row_bytes, x_ptr, y, M, n_cols, n_rows, stream); break;
            case 3:  launch_mmvq_tileM_typed<3> (w_ptr, row_bytes, x_ptr, y, M, n_cols, n_rows, stream); break;
            case 6:  launch_mmvq_tileM_typed<6> (w_ptr, row_bytes, x_ptr, y, M, n_cols, n_rows, stream); break;
            case 7:  launch_mmvq_tileM_typed<7> (w_ptr, row_bytes, x_ptr, y, M, n_cols, n_rows, stream); break;
            case 8:  launch_mmvq_tileM_typed<8> (w_ptr, row_bytes, x_ptr, y, M, n_cols, n_rows, stream); break;
            case 12: launch_mmvq_tileM_typed<12>(w_ptr, row_bytes, x_ptr, y, M, n_cols, n_rows, stream); break;
            case 13: launch_mmvq_tileM_typed<13>(w_ptr, row_bytes, x_ptr, y, M, n_cols, n_rows, stream); break;
            case 14: launch_mmvq_tileM_typed<14>(w_ptr, row_bytes, x_ptr, y, M, n_cols, n_rows, stream); break;
            case 16: launch_mmvq_tileM_typed<16>(w_ptr, row_bytes, x_ptr, y, M, n_cols, n_rows, stream); break;
            case 17: launch_mmvq_tileM_typed<17>(w_ptr, row_bytes, x_ptr, y, M, n_cols, n_rows, stream); break;
            case 18: launch_mmvq_tileM_typed<18>(w_ptr, row_bytes, x_ptr, y, M, n_cols, n_rows, stream); break;
            case 20: launch_mmvq_tileM_typed<20>(w_ptr, row_bytes, x_ptr, y, M, n_cols, n_rows, stream); break;
            case 21: launch_mmvq_tileM_typed<21>(w_ptr, row_bytes, x_ptr, y, M, n_cols, n_rows, stream); break;
            case 22: launch_mmvq_tileM_typed<22>(w_ptr, row_bytes, x_ptr, y, M, n_cols, n_rows, stream); break;
            case 23: launch_mmvq_tileM_typed<23>(w_ptr, row_bytes, x_ptr, y, M, n_cols, n_rows, stream); break;
            case 29: launch_mmvq_tileM_typed<29>(w_ptr, row_bytes, x_ptr, y, M, n_cols, n_rows, stream); break;
            case 42: launch_mmvq_tileM_typed<42>(w_ptr, row_bytes, x_ptr, y, M, n_cols, n_rows, stream); break;
            default:
                std::fprintf(stderr, "[Qwen4] Unsupported batchM MMVQ ggml_type=%d\n", ggml_type);
                break;
        }
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

void qwen4_embed_to_hc_batchM_cuda(
    int32_t embd_type,
    const void* token_embd,
    const int32_t* d_token_ids,
    int M,
    float* d_R,
    int n_embd,
    int n_streams,
    cudaStream_t stream)
{
    if (M <= 0) return;
    const size_t row_bytes = qwen4_ggml_row_bytes(embd_type, n_embd);
    const dim3 grid((n_embd + 255) / 256, M);
    embed_to_hc_batchM_kernel<<<grid, 256, 0, stream>>>(
        embd_type, (const uint8_t*)token_embd, row_bytes, d_token_ids, d_R, n_embd, n_streams);
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
    float* R,
    const float* norm_w,
    const uint16_t* down_bf16,
    const uint16_t* up_bf16,
    float* x_out,
    float* tmp_buf,
    float eps,
    cudaStream_t stream,
    const uint16_t* inject_bf16,
    void* x_q8_1_out,
    const float* prev_sub_out,
    bool pre_normed,
    void* x_q8_1_peer_out)
{
    float* normed = tmp_buf;
    float* h      = tmp_buf + QWEN4_STREAM_TOTAL;
    float* gates_save = (inject_bf16 != nullptr) ? (R + QWEN4_STREAM_TOTAL) : nullptr;
    const float* prev_gates = (prev_sub_out != nullptr) ? (R + QWEN4_STREAM_TOTAL) : nullptr;

    if (!pre_normed) {
        gr_stream_rmsnorm_bf16_kernel<<<QWEN4_N_STREAMS, 640, 0, stream>>>(R, norm_w, normed, eps, prev_sub_out, prev_gates);
    }
    const int n_down_blocks = (inject_bf16 != nullptr) ? 162 : 160;
    gr_down_inject_gemv_kernel<<<n_down_blocks, 256, 0, stream>>>(
        down_bf16, inject_bf16, normed, h, gates_save);
    gr_up_mean_q8_1_kernel<<<dim3(QWEN4_N_EMBD / 32, 1), 1024, 0, stream>>>(
        up_bf16, h, normed, x_out, (block_q8_1*)x_q8_1_out, (block_q8_1*)x_q8_1_peer_out);
}

void qwen4_gr_write_cuda(
    float* R,
    const float* norm_w,
    const uint16_t* inject_bf16,
    const float* sub_out,
    float* tmp_buf,
    float eps,
    cudaStream_t stream,
    bool precomputed_gates)
{
    if (precomputed_gates) {
        const float* gates_save = R + QWEN4_STREAM_TOTAL;
        gr_apply_write_vec4_kernel<<<dim3(3, 1), 256, 0, stream>>>(R, sub_out, gates_save);
    } else {
        float* normed = tmp_buf;
        gr_stream_rmsnorm_bf16_kernel<<<QWEN4_N_STREAMS, 640, 0, stream>>>(R, norm_w, normed, eps);
        gr_inject_and_write_kernel<<<QWEN4_N_STREAMS, 256, 0, stream>>>(inject_bf16, normed, sub_out, R);
    }
}

void qwen4_gr_read_batchM_cuda(
    float* R,
    const float* norm_w,
    const uint16_t* down_bf16,
    const uint16_t* up_bf16,
    float* x_out,
    int M,
    float* tmp_buf,
    float eps,
    cudaStream_t stream,
    const uint16_t* inject_bf16,
    void* x_q8_1_out,
    const float* prev_sub_out)
{
    if (M <= 0) return;
    if (M == 1) {
        qwen4_gr_read_cuda(R, norm_w, down_bf16, up_bf16, x_out, tmp_buf, eps, stream, inject_bf16, x_q8_1_out, prev_sub_out);
        return;
    }
    float* gates_save_base       = (inject_bf16 != nullptr) ? (R + (size_t)M * QWEN4_STREAM_TOTAL) : nullptr;
    const float* prev_gates_base = (prev_sub_out != nullptr) ? (R + (size_t)M * QWEN4_STREAM_TOTAL) : nullptr;

    int dev = 0;
    cudaGetDevice(&dev);
    constexpr size_t GR_UP_OUT_ELEMS = 21135360ULL; // Fits [1032, 10240] FP32 output (21,135,360 uint16s)

    for (int m0 = 0; m0 < M; m0 += 1032) {
        const int sub_m         = (M - m0 > 1032) ? 1032 : (M - m0);
        float* R_sub            = R + (size_t)m0 * QWEN4_STREAM_TOTAL;
        float* x_sub            = x_out + (size_t)m0 * QWEN4_N_EMBD;
        void* q8_sub            = (x_q8_1_out != nullptr) ? ((uint8_t*)x_q8_1_out + (size_t)m0 * 2880) : nullptr;
        const float* prev_sub   = (prev_sub_out != nullptr) ? (prev_sub_out + (size_t)m0 * QWEN4_N_EMBD) : nullptr;
        float* gates_save       = (gates_save_base != nullptr) ? (gates_save_base + (size_t)m0 * QWEN4_N_STREAMS) : nullptr;
        const float* prev_gates = (prev_gates_base != nullptr) ? (prev_gates_base + (size_t)m0 * QWEN4_N_STREAMS) : nullptr;
        float* normed           = tmp_buf;                                         // [sub_m, 10240]
        float* h                = normed + (size_t)sub_m * QWEN4_STREAM_TOTAL;     // [sub_m, 320]

        const size_t need_ws_bytes = (GR_UP_OUT_ELEMS + (size_t)sub_m * QWEN4_STREAM_TOTAL) * sizeof(uint16_t);
        if (sub_m >= 16 && dev >= 0 && dev < 2 &&
            g_qwen4_cublas[dev] != nullptr &&
            g_qwen4_bf16_ws[dev] != nullptr &&
            g_qwen4_bf16_ws_bytes[dev] >= need_ws_bytes &&
            (size_t)sub_m * QWEN4_STREAM_TOTAL * sizeof(float) <= GR_UP_OUT_ELEMS * sizeof(uint16_t)) {

            cublasHandle_t handle = g_qwen4_cublas[dev];
            float*    d_up_out = (float*)g_qwen4_bf16_ws[dev];
            uint16_t* d_x_bf16 = g_qwen4_bf16_ws[dev] + GR_UP_OUT_ELEMS;

            gr_stream_rmsnorm_bf16_kernel<<<sub_m * QWEN4_N_STREAMS, 640, 0, stream>>>(
                R_sub, norm_w, normed, eps, prev_sub, prev_gates, d_x_bf16);

            cublasSetStream(handle, stream);
            const float alpha = 1.0f;
            const float beta  = 0.0f;
            cublasGemmEx(
                handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                QWEN4_HC_RANK, sub_m, QWEN4_STREAM_TOTAL,
                &alpha,
                down_bf16, CUDA_R_16BF, QWEN4_STREAM_TOTAL,
                d_x_bf16,  CUDA_R_16BF, QWEN4_STREAM_TOTAL,
                &beta,
                d_up_out,  CUDA_R_32F,  QWEN4_HC_RANK,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);

            gr_down_act_and_inject_kernel<<<dim3(inject_bf16 != nullptr ? 2 : 1, sub_m), 256, 0, stream>>>(
                d_up_out, h, d_x_bf16, inject_bf16, normed, gates_save);

            cublasGemmEx(
                handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                QWEN4_STREAM_TOTAL, sub_m, QWEN4_HC_RANK,
                &alpha,
                up_bf16,  CUDA_R_16BF, QWEN4_HC_RANK,
                d_x_bf16, CUDA_R_16BF, QWEN4_HC_RANK,
                &beta,
                d_up_out, CUDA_R_32F,  QWEN4_STREAM_TOTAL,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);

            gr_up_gate_mean_q8_1_batchM_kernel<<<dim3(QWEN4_N_EMBD / 128, sub_m), dim3(32, 4), 0, stream>>>(
                d_up_out, normed, x_sub, (block_q8_1*)q8_sub);
            continue;
        }

        gr_stream_rmsnorm_bf16_kernel<<<sub_m * QWEN4_N_STREAMS, 640, 0, stream>>>(R_sub, norm_w, normed, eps, prev_sub, prev_gates);

        const int n_down_blocks = (inject_bf16 != nullptr) ? 21 : 20;
        constexpr int TILE_M = 8;
        const int gy = (sub_m + TILE_M - 1) / TILE_M;
        gr_down_inject_tileM_kernel<TILE_M><<<dim3(n_down_blocks, gy), dim3(32, 8), 0, stream>>>(
            down_bf16, inject_bf16, normed, h, gates_save, sub_m);
        gr_up_mean_q8_1_tileM_kernel<TILE_M><<<dim3(QWEN4_N_EMBD / 32, gy), 256, 0, stream>>>(
            up_bf16, h, normed, x_sub, (block_q8_1*)q8_sub, sub_m);
    }
}

void qwen4_gr_write_batchM_cuda(
    float* R,
    const float* sub_out,
    int M,
    cudaStream_t stream)
{
    if (M <= 0) return;
    const float* gates_save = R + (size_t)M * QWEN4_STREAM_TOTAL;
    gr_apply_write_vec4_kernel<<<dim3(3, M), 256, 0, stream>>>(R, sub_out, gates_save);
}

void qwen4_ple_iq4nl_dequant_cuda(
    const void* src16_iq4nl,
    float* dst_f32,
    cudaStream_t stream,
    void* dst_q8_1)
{
    ple_iq4nl_dequant_kernel<<<QWEN4_N_EMBD / 32, 32, 0, stream>>>(
        (const block_iq4_nl*)src16_iq4nl, dst_f32, (block_q8_1*)dst_q8_1);
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

    qwen4_project_cuda(ple_key_type, ple_key, emb_flat, q8_scratch, key_out, QWEN4_STREAM_TOTAL, QWEN4_N_EMBD, stream);
    qwen4_project_cuda(ple_val_type, ple_val, emb_flat, q8_scratch, val_out, QWEN4_N_EMBD, QWEN4_N_EMBD, stream);

    ple_gate_kernel<<<dim3(QWEN4_N_STREAMS, 1), 256, 0, stream>>>(
        R, key_out, val_out, ple_norm_k, ple_norm_q, ple_norm_c, gated, normd, eps, (const half*)ple_conv_f16, conv_state);
}

void qwen4_ple_forward_batchM_cuda(
    float* R,
    const float* ple_norm_q,
    const float* ple_norm_k,
    const float* ple_norm_c,
    const void* ple_key,
    int32_t ple_key_type,
    const void* ple_val,
    int32_t ple_val_type,
    const uint16_t* ple_conv_f16,
    const void* d_ple_rows_iq4nl,
    float* emb_flat,
    float* conv_state,
    int M,
    float* scratch,
    void* q8_scratch,
    float eps,
    cudaStream_t stream)
{
    if (M <= 0) return;
    float* key_out = scratch;                                           // [M, 10240]
    float* val_out = key_out + (size_t)M * QWEN4_STREAM_TOTAL;          // [M, 2560]
    float* gated   = val_out + (size_t)M * QWEN4_N_EMBD;                // [M, 10240]
    float* normd   = gated   + (size_t)M * QWEN4_STREAM_TOTAL;          // [M, 10240]
    float* emb_buf = (M <= 64 && emb_flat != nullptr) ? emb_flat : normd; // Reuses normd region during Phase 1!

    const bool need_q8 = (M < 64) &&
        ((ple_key_type != QWEN4_GGML_TYPE_BF16 && ple_key_type != QWEN4_GGML_TYPE_F16 && ple_key_type != QWEN4_GGML_TYPE_F32) ||
         (ple_val_type != QWEN4_GGML_TYPE_BF16 && ple_val_type != QWEN4_GGML_TYPE_F16 && ple_val_type != QWEN4_GGML_TYPE_F32));
    ple_iq4nl_dequant_kernel<<<M * (QWEN4_N_EMBD / 32), 32, 0, stream>>>(
        (const block_iq4_nl*)d_ple_rows_iq4nl, emb_buf, need_q8 ? (block_q8_1*)q8_scratch : nullptr);

    qwen4_project_batchM_cuda(ple_key_type, ple_key, emb_buf, q8_scratch, key_out, M, QWEN4_STREAM_TOTAL, QWEN4_N_EMBD, stream, false);
    qwen4_project_batchM_cuda(ple_val_type, ple_val, emb_buf, q8_scratch, val_out, M, QWEN4_N_EMBD, QWEN4_N_EMBD, stream, (M >= 64));

    if (M == 1) {
        ple_gate_kernel<<<dim3(QWEN4_N_STREAMS, 1), 256, 0, stream>>>(
            R, key_out, val_out, ple_norm_k, ple_norm_q, ple_norm_c, gated, normd, eps, (const half*)ple_conv_f16, conv_state);
    } else {
        ple_gate_kernel<<<dim3(QWEN4_N_STREAMS, M), 256, 0, stream>>>(
            R, key_out, val_out, ple_norm_k, ple_norm_q, ple_norm_c, gated, normd, eps);

        ple_conv_update_kernel<<<(QWEN4_STREAM_TOTAL + 255) / 256, 256, 0, stream>>>(
            R, gated, normd, (const half*)ple_conv_f16, conv_state, M);
    }
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
    float* gate     = z + QWEN4_GDN_V_DIM;                      // [48]
    float* beta     = gate + QWEN4_GDN_N_V_HEAD;                // [48]

    const bool qkv_quant  = (qkv_type  != QWEN4_GGML_TYPE_BF16 && qkv_type  != QWEN4_GGML_TYPE_F16 && qkv_type  != QWEN4_GGML_TYPE_F32);
    const bool gate_quant = (gate_type != QWEN4_GGML_TYPE_BF16 && gate_type != QWEN4_GGML_TYPE_F16 && gate_type != QWEN4_GGML_TYPE_F32);
    const bool quant_out  = (out_type  != QWEN4_GGML_TYPE_BF16 && out_type  != QWEN4_GGML_TYPE_F16 && out_type  != QWEN4_GGML_TYPE_F32);
    if (qkv_quant && gate_quant) {
        const size_t qkv_rb  = qwen4_ggml_row_bytes(qkv_type,  QWEN4_N_EMBD);
        const size_t gate_rb = qwen4_ggml_row_bytes(gate_type, QWEN4_N_EMBD);
        gdn_in_proj_fused_kernel<16><<<1030, dim3(32, 8), 0, stream>>>(
            x_f32, (const block_q8_1*)x_q8_1,
            (const uint8_t*)w_qkv,  qkv_type,  qkv_rb,
            (const uint8_t*)w_gate, gate_type, gate_rb,
            w_alpha_bf16, w_beta_bf16, w_conv_f32, ssm_a_f32, dt_bias_f32,
            conv_state, qkv_conv, z, gate, beta);

        gdn_recurrence_norm_q8_1_kernel<<<QWEN4_GDN_N_V_HEAD, dim3(QWEN4_GDN_HEAD_DIM, 4), 0, stream>>>(
            ssm_state, qkv_conv, gate, beta, ssm_norm_f32, z, quant_out ? (block_q8_1*)q8_scratch : nullptr, 1, eps, true);
    } else {
        qwen4_project_cuda(qkv_type, w_qkv, x_f32, x_q8_1, qkv, QWEN4_GDN_QKV_DIM, QWEN4_N_EMBD, stream);
        qwen4_project_cuda(gate_type, w_gate, x_f32, x_q8_1, z, QWEN4_GDN_V_DIM, QWEN4_N_EMBD, stream);
        gdn_alpha_beta_gemv_kernel<<<dim3(6, 1), dim3(32, 8), 0, stream>>>(
            w_alpha_bf16, w_beta_bf16, x_f32, ssm_a_f32, dt_bias_f32, gate, beta);

        gdn_conv_silu_l2norm_kernel<<<QWEN4_GDN_QKV_DIM / QWEN4_GDN_HEAD_DIM, QWEN4_GDN_HEAD_DIM, 0, stream>>>(
            conv_state, qkv, w_conv_f32, qkv_conv, eps);

        gdn_recurrence_norm_q8_1_kernel<<<QWEN4_GDN_N_V_HEAD, dim3(QWEN4_GDN_HEAD_DIM, 4), 0, stream>>>(
            ssm_state, qkv_conv, gate, beta, ssm_norm_f32, z, quant_out ? (block_q8_1*)q8_scratch : nullptr, 1, eps, false);
    }

    qwen4_project_cuda(out_type, w_out, z, q8_scratch, y_out, QWEN4_N_EMBD, QWEN4_GDN_V_DIM, stream);
}

void qwen4_gdn_prefill_batchM_cuda(
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
    int M,
    float* y_out,
    float* scratch,
    void* q8_scratch,
    float eps,
    cudaStream_t stream)
{
    if (M <= 0) return;
    if (M == 1) {
        qwen4_gdn_decode_cuda(
            x_f32, x_q8_1, w_qkv, qkv_type, w_gate, gate_type, w_alpha_bf16, w_beta_bf16,
            w_conv_f32, ssm_a_f32, dt_bias_f32, ssm_norm_f32, w_out, out_type,
            conv_state, ssm_state, y_out, scratch, q8_scratch, eps, stream);
        return;
    }
    if (M > 1032) {
        for (int m0 = 0; m0 < M; m0 += 1032) {
            const int sub_m = (M - m0 > 1032) ? 1032 : (M - m0);
            qwen4_gdn_prefill_batchM_cuda(
                x_f32 + (size_t)m0 * QWEN4_N_EMBD,
                (x_q8_1 != nullptr) ? ((const uint8_t*)x_q8_1 + (size_t)m0 * 2880) : nullptr,
                w_qkv, qkv_type, w_gate, gate_type, w_alpha_bf16, w_beta_bf16,
                w_conv_f32, ssm_a_f32, dt_bias_f32, ssm_norm_f32, w_out, out_type,
                conv_state, ssm_state, sub_m,
                y_out + (size_t)m0 * QWEN4_N_EMBD,
                scratch, q8_scratch, eps, stream);
        }
        return;
    }

    float* qkv     = scratch;                                              // [M, 10240]
    float* z       = qkv     + (size_t)M * QWEN4_GDN_QKV_DIM;              // [M, 6144]
    float* gate    = z       + (size_t)M * QWEN4_GDN_V_DIM;                // [M, 48]
    float* beta    = gate    + (size_t)M * QWEN4_GDN_N_V_HEAD;             // [M, 48]
    float* out_buf = beta    + (size_t)M * QWEN4_GDN_N_V_HEAD;             // [M, 6144]

    qwen4_project_batchM_cuda(qkv_type, w_qkv, x_f32, x_q8_1, qkv, M, QWEN4_GDN_QKV_DIM, QWEN4_N_EMBD, stream, false);
    qwen4_project_batchM_cuda(gate_type, w_gate, x_f32, x_q8_1, z, M, QWEN4_GDN_V_DIM, QWEN4_N_EMBD, stream, (M >= 64));

    constexpr int AB_TILE_M = 8;
    gdn_alpha_beta_tileM_kernel<AB_TILE_M><<<dim3(6, (M + AB_TILE_M - 1) / AB_TILE_M), dim3(32, 8), 0, stream>>>(
        w_alpha_bf16, w_beta_bf16, x_f32, ssm_a_f32, dt_bias_f32, gate, beta, M);

    gdn_conv_silu_batchM_kernel<<<QWEN4_GDN_QKV_DIM / 128, 128, 0, stream>>>(
        conv_state, qkv, w_conv_f32, qkv, M, eps);

    int dev = 0;
    cudaGetDevice(&dev);
    const bool fuse_out_bf16 = qwen4_can_cublas_gemm(out_type, M, dev);
    uint16_t* bf16_out = fuse_out_bf16 ? (g_qwen4_bf16_ws[dev] + 15728640ULL) : nullptr;
    const bool quant_out = (!fuse_out_bf16 && M < 64) && (out_type != QWEN4_GGML_TYPE_BF16 && out_type != QWEN4_GGML_TYPE_F16 && out_type != QWEN4_GGML_TYPE_F32);
    gdn_recurrence_warp_batchM_kernel<<<QWEN4_GDN_N_V_HEAD * 4, dim3(32, 8), 0, stream>>>(
        ssm_state, qkv, gate, beta, out_buf, M, eps);
    gdn_head_rmsnorm_gate_q8_1_batchM_kernel<<<dim3(QWEN4_GDN_N_V_HEAD, M), QWEN4_GDN_HEAD_DIM, 0, stream>>>(
        out_buf, ssm_norm_f32, z, quant_out ? (block_q8_1*)q8_scratch : nullptr, eps, bf16_out);

    qwen4_project_batchM_cuda(out_type, w_out, z, q8_scratch, y_out, M, QWEN4_N_EMBD, QWEN4_GDN_V_DIM, stream, fuse_out_bf16);
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
    constexpr int Q_GATE_DIM = QWEN4_QSA_N_HEAD * 2 * QWEN4_QSA_HEAD_DIM; // 12288
    constexpr int Q_DIM      = QWEN4_QSA_N_HEAD * QWEN4_QSA_HEAD_DIM;     // 6144
    constexpr int KV_DIM     = QWEN4_QSA_N_KV * QWEN4_QSA_HEAD_DIM;       // 512
    constexpr int IDX_Q_DIM  = QWEN4_QSA_IDX_HEADS * QWEN4_QSA_IDX_DIM;   // 512
    const size_t max_blocks  = ((size_t)max_seq_len + 3) / 4 + 256;

    float* q_gate       = scratch;                          // [12288]
    float* q            = q_gate + Q_GATE_DIM;              // [6144]
    float* gate         = q + Q_DIM;                        // [6144]
    float* k_cur        = gate + Q_DIM;                     // [512]
    float* v_cur        = k_cur + KV_DIM;                   // [512]
    float* attn_gated   = v_cur + KV_DIM;                   // [6144]
    float* q_idx        = attn_gated + Q_DIM;               // [512]
    float* idx_raw      = q_idx + IDX_Q_DIM;                // [128]
    float* block_scores = idx_raw + QWEN4_QSA_IDX_DIM;      // [max_blocks]
    float* partial_m    = block_scores + max_blocks;        // [68 * 24]
    float* partial_l    = partial_m + 68 * QWEN4_QSA_N_HEAD;// [68 * 24]
    float* partial_acc  = partial_l + 68 * QWEN4_QSA_N_HEAD;// [68 * 24 * 256]

    const int n_kv = pos + 1;
    const bool need_idx_q = (n_kv > QWEN4_QSA_SPARSE_THRESHOLD);
    const bool qkv_quant =
        (q_type != QWEN4_GGML_TYPE_BF16 && q_type != QWEN4_GGML_TYPE_F16 && q_type != QWEN4_GGML_TYPE_F32) &&
        (k_type != QWEN4_GGML_TYPE_BF16 && k_type != QWEN4_GGML_TYPE_F16 && k_type != QWEN4_GGML_TYPE_F32) &&
        (v_type != QWEN4_GGML_TYPE_BF16 && v_type != QWEN4_GGML_TYPE_F16 && v_type != QWEN4_GGML_TYPE_F32);

    if (qkv_quant) {
        const size_t q_rb = qwen4_ggml_row_bytes(q_type, QWEN4_N_EMBD);
        const size_t k_rb = qwen4_ggml_row_bytes(k_type, QWEN4_N_EMBD);
        const size_t v_rb = qwen4_ggml_row_bytes(v_type, QWEN4_N_EMBD);
        const int n_in_blocks = need_idx_q ? 872 : 840;
        qsa_in_proj_fused_kernel<16><<<n_in_blocks, dim3(32, 8), 0, stream>>>(
            x_f32, (const block_q8_1*)x_q8_1,
            (const uint8_t*)w_q, q_type, q_rb,
            (const uint8_t*)w_k, k_type, k_rb,
            (const uint8_t*)w_v, v_type, v_rb,
            w_idx_k_bf16, need_idx_q ? w_idx_q_bf16 : nullptr,
            q_gate, k_cur, v_cur, idx_raw, need_idx_q ? q_idx : nullptr);
    } else {
        qwen4_project_cuda(q_type, w_q, x_f32, x_q8_1, q_gate, Q_GATE_DIM, QWEN4_N_EMBD, stream);
        qwen4_project_cuda(k_type, w_k, x_f32, x_q8_1, k_cur,  KV_DIM,     QWEN4_N_EMBD, stream);
        qwen4_project_cuda(v_type, w_v, x_f32, x_q8_1, v_cur,  KV_DIM,     QWEN4_N_EMBD, stream);
        qwen4_bf16_gemv_cuda(w_idx_k_bf16, x_f32, idx_raw, QWEN4_QSA_IDX_DIM, QWEN4_N_EMBD, stream);
        if (need_idx_q) {
            qwen4_bf16_gemv_cuda(w_idx_q_bf16, x_f32, q_idx, IDX_Q_DIM, QWEN4_N_EMBD, stream);
        }
    }

    const int n_prep_blocks = need_idx_q ? 31 : 27;
    qsa_prep_qkv_idx_kernel<<<dim3(n_prep_blocks, 1), QWEN4_QSA_HEAD_DIM, 0, stream>>>(
        q_gate, q_norm_f32, q, gate,
        k_cur, v_cur, k_norm_f32, k_cache, v_cache,
        idx_raw, idx_k_norm_f32, idx_pooled, idx_tail, idx_dead,
        need_idx_q ? q_idx : nullptr, idx_q_norm_f32,
        pos, 1, eps);

    const bool quant_out = (out_type != QWEN4_GGML_TYPE_BF16 && out_type != QWEN4_GGML_TYPE_F16 && out_type != QWEN4_GGML_TYPE_F32);
    if (!need_idx_q) {
        const int split_chunk = (n_kv <= 1280) ? QWEN4_QSA_SPLIT_CHUNK : (((n_kv + 19) / 20 + 7) & ~7);
        const int n_chunks = (n_kv + split_chunk - 1) / split_chunk;
        qsa_split_kv_chunk_kernel<<<dim3(n_chunks, 4, 1), dim3(32, 8), 0, stream>>>(
            q, k_cache, v_cache, nullptr, pos, n_kv, partial_acc, partial_m, partial_l, split_chunk);
        qsa_merge_chunks_gate_q8_1_kernel<<<dim3(QWEN4_QSA_N_HEAD, 1), QWEN4_QSA_HEAD_DIM, 0, stream>>>(
            partial_acc, partial_m, partial_l, gate, n_chunks, attn_gated, quant_out ? (block_q8_1*)q8_scratch : nullptr);
    } else {
        const int n_blocks = (n_kv + QWEN4_QSA_IDX_BLOCK - 1) / QWEN4_QSA_IDX_BLOCK;
        const int n_sel = QWEN4_QSA_MAX_SEL;
        const int split_chunk = 112;
        const int n_chunks = (n_sel + split_chunk - 1) / split_chunk;
        qsa_indexer_score_blocks_kernel<<<(n_blocks + 7) / 8, dim3(32, 8), 0, stream>>>(
            idx_pooled, q_idx, block_scores, n_kv, n_blocks, d_sel_indices, n_sel);
        qsa_split_kv_chunk_kernel<<<dim3(n_chunks, 4, 1), dim3(32, 8), 0, stream>>>(
            q, k_cache, v_cache, d_sel_indices, pos, n_sel, partial_acc, partial_m, partial_l, split_chunk);
        qsa_merge_chunks_gate_q8_1_kernel<<<dim3(QWEN4_QSA_N_HEAD, 1), QWEN4_QSA_HEAD_DIM, 0, stream>>>(
            partial_acc, partial_m, partial_l, gate, n_chunks, attn_gated, quant_out ? (block_q8_1*)q8_scratch : nullptr);
    }

    qwen4_project_cuda(out_type, w_out, attn_gated, q8_scratch, y_out, QWEN4_N_EMBD, Q_DIM, stream);
}

void qwen4_qsa_prefill_batchM_cuda(
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
    int pos_start,
    int M,
    int max_seq_len,
    float* y_out,
    float* scratch,
    int32_t* d_sel_indices,
    void* q8_scratch,
    float eps,
    cudaStream_t stream)
{
    if (M <= 0) return;
    if (M == 1) {
        qwen4_qsa_decode_cuda(
            x_f32, x_q8_1, w_q, q_type, w_k, k_type, w_v, v_type, q_norm_f32, k_norm_f32,
            w_out, out_type, w_idx_k_bf16, w_idx_q_bf16, idx_k_norm_f32, idx_q_norm_f32,
            k_cache, v_cache, idx_pooled, idx_tail, idx_dead, pos_start, max_seq_len,
            y_out, scratch, d_sel_indices, q8_scratch, eps, stream);
        return;
    }
    if (M > 1032) {
        for (int m0 = 0; m0 < M; m0 += 1032) {
            const int sub_m = (M - m0 > 1032) ? 1032 : (M - m0);
            qwen4_qsa_prefill_batchM_cuda(
                x_f32 + (size_t)m0 * QWEN4_N_EMBD,
                (x_q8_1 != nullptr) ? ((const uint8_t*)x_q8_1 + (size_t)m0 * 2880) : nullptr,
                w_q, q_type, w_k, k_type, w_v, v_type, q_norm_f32, k_norm_f32,
                w_out, out_type, w_idx_k_bf16, w_idx_q_bf16, idx_k_norm_f32, idx_q_norm_f32,
                k_cache, v_cache, idx_pooled, idx_tail, idx_dead,
                pos_start + m0, sub_m, max_seq_len,
                y_out + (size_t)m0 * QWEN4_N_EMBD,
                scratch, d_sel_indices, q8_scratch, eps, stream);
        }
        return;
    }

    constexpr int Q_GATE_DIM = QWEN4_QSA_N_HEAD * 2 * QWEN4_QSA_HEAD_DIM; // 12288
    constexpr int Q_DIM      = QWEN4_QSA_N_HEAD * QWEN4_QSA_HEAD_DIM;     // 6144
    constexpr int KV_DIM     = QWEN4_QSA_N_KV * QWEN4_QSA_HEAD_DIM;       // 512
    constexpr int IDX_Q_DIM  = QWEN4_QSA_IDX_HEADS * QWEN4_QSA_IDX_DIM;   // 512
    const size_t max_blocks  = ((size_t)max_seq_len + 3) / 4 + 256;

    // Lifetime-partitioned scratch layout (fits M <= 1032 in < 20.4M floats during input GEMMs):
    //   Phase 1 (Input GEMMs): q_gate [M, 12288] + k_cur [M, 512] + v_cur [M, 512] + idx_raw [M, 128] = 13,440 * M floats
    //   Phase 2 (QKV Prep):    reads Phase 1 -> writes q [M, 6144] + gate [M, 6144] at offset 13,440 * M
    //   Phase 3 (Attention):   reads q + gate -> writes attn_gated [M, 6144] + q_idx + partials reusing Phase 1 region!
    float* q_gate       = scratch;                                  // [M, 12288]
    float* k_cur        = q_gate + (size_t)M * Q_GATE_DIM;          // [M, 512]
    float* v_cur        = k_cur + (size_t)M * KV_DIM;               // [M, 512]
    float* idx_raw      = v_cur + (size_t)M * KV_DIM;               // [M, 128]
    float* q            = idx_raw + (size_t)M * QWEN4_QSA_IDX_DIM;  // [M, 6144]
    float* gate         = q + (size_t)M * Q_DIM;                    // [M, 6144]
    float* attn_gated   = scratch;                                  // [M, 6144] (reuses q_gate after qsa_prep_qkv_idx_kernel)
    float* q_idx        = attn_gated + (size_t)M * Q_DIM;           // [M, 512]
    float* block_scores = q_idx + (size_t)M * IDX_Q_DIM;            // [max_blocks]
    float* partial_m    = block_scores + max_blocks;                // [8 * 36 * 24]
    float* partial_l    = partial_m + 8 * 36 * QWEN4_QSA_N_HEAD;    // [8 * 36 * 24]
    float* partial_acc  = partial_l + 8 * 36 * QWEN4_QSA_N_HEAD;    // [8 * 36 * 24 * 256]

    qwen4_project_batchM_cuda(q_type, w_q, x_f32, x_q8_1, q_gate, M, Q_GATE_DIM, QWEN4_N_EMBD, stream, false);
    qwen4_project_batchM_cuda(k_type, w_k, x_f32, x_q8_1, k_cur,  M, KV_DIM,     QWEN4_N_EMBD, stream, (M >= 64));
    qwen4_project_batchM_cuda(v_type, w_v, x_f32, x_q8_1, v_cur,  M, KV_DIM,     QWEN4_N_EMBD, stream, (M >= 64));
    qwen4_project_batchM_cuda(QWEN4_GGML_TYPE_BF16, w_idx_k_bf16, x_f32, nullptr, idx_raw, M, QWEN4_QSA_IDX_DIM, QWEN4_N_EMBD, stream, (M >= 64));

    qsa_prep_qkv_idx_kernel<<<dim3(27, M), QWEN4_QSA_HEAD_DIM, 0, stream>>>(
        q_gate, q_norm_f32, q, gate,
        k_cur, v_cur, k_norm_f32, k_cache, v_cache,
        idx_raw, idx_k_norm_f32, idx_pooled, idx_tail, idx_dead,
        nullptr, nullptr,
        pos_start, M, eps);

    int dev = 0;
    cudaGetDevice(&dev);
    const int m_direct = (pos_start >= QWEN4_QSA_SPARSE_THRESHOLD)
        ? 0
        : ((pos_start + M <= QWEN4_QSA_SPARSE_THRESHOLD) ? M : (QWEN4_QSA_SPARSE_THRESHOLD - pos_start));
    const bool fuse_out_bf16 = (m_direct == M) && qwen4_can_cublas_gemm(out_type, M, dev);
    uint16_t* bf16_out = fuse_out_bf16 ? (g_qwen4_bf16_ws[dev] + 15728640ULL) : nullptr;
    const bool quant_out = (!fuse_out_bf16 && M < 64) && (out_type != QWEN4_GGML_TYPE_BF16 && out_type != QWEN4_GGML_TYPE_F16 && out_type != QWEN4_GGML_TYPE_F32);
    if (m_direct > 0) {
        qsa_prefill_fused_attn_gate_q8_1_kernel<<<dim3(8, (m_direct + 1) / 2), dim3(32, 8), 0, stream>>>(
            q, k_cache, v_cache, gate, pos_start, m_direct, attn_gated, quant_out ? (block_q8_1*)q8_scratch : nullptr, bf16_out);
    }
    if (m_direct < M) {
        const int m_rem = M - m_direct;
        qwen4_project_batchM_cuda(
            QWEN4_GGML_TYPE_BF16, w_idx_q_bf16,
            x_f32 + (size_t)m_direct * QWEN4_N_EMBD, nullptr,
            q_idx + (size_t)m_direct * IDX_Q_DIM,
            m_rem, IDX_Q_DIM, QWEN4_N_EMBD, stream);
        for (int m = m_direct; m < M; ++m) {
            const int pos  = pos_start + m;
            const int n_kv = pos + 1;
            float* q_m    = q + (size_t)m * Q_DIM;
            float* gate_m = gate + (size_t)m * Q_DIM;
            float* out_m  = attn_gated + (size_t)m * Q_DIM;
            block_q8_1* q8_m = quant_out ? ((block_q8_1*)q8_scratch + (size_t)m * (Q_DIM / 32)) : nullptr;
            const int n_blocks = (n_kv + QWEN4_QSA_IDX_BLOCK - 1) / QWEN4_QSA_IDX_BLOCK;
            const int n_sel = QWEN4_QSA_MAX_SEL;
            const int split_chunk = 128;
            const int n_chunks = (n_sel + split_chunk - 1) / split_chunk;
            float* q_idx_m = q_idx + (size_t)m * IDX_Q_DIM;
            qsa_indexer_prep_q_kernel<<<QWEN4_QSA_IDX_HEADS, QWEN4_QSA_IDX_DIM, 0, stream>>>(q_idx_m, idx_q_norm_f32, pos, eps);
            qsa_indexer_score_blocks_kernel<<<(n_blocks + 7) / 8, dim3(32, 8), 0, stream>>>(
                idx_pooled, q_idx_m, block_scores, n_kv, n_blocks);
            qsa_indexer_select_topk_kernel<<<1, 256, 0, stream>>>(
                block_scores, d_sel_indices, n_kv, n_blocks, n_sel);
            qsa_split_kv_chunk_kernel<<<dim3(n_chunks, 4, 1), dim3(32, 8), 0, stream>>>(
                q_m, k_cache, v_cache, d_sel_indices, pos, n_sel, partial_acc, partial_m, partial_l, split_chunk);
            qsa_merge_chunks_gate_q8_1_kernel<<<dim3(QWEN4_QSA_N_HEAD, 1), QWEN4_QSA_HEAD_DIM, 0, stream>>>(
                partial_acc, partial_m, partial_l, gate_m, n_chunks, out_m, q8_m);
        }
    }

    qwen4_project_batchM_cuda(out_type, w_out, attn_gated, q8_scratch, y_out, M, QWEN4_N_EMBD, Q_DIM, stream, fuse_out_bf16);
}

void qwen4_router_top10_cuda(
    const uint16_t* gate_inp_bf16,
    const float* x_f32,
    float* scratch_logits,
    int32_t* d_top_idx,
    float* d_top_w,
    cudaStream_t stream,
    const uint16_t* w_gate_inp_shexp_bf16,
    int32_t* d_host_top_idx,
    float* d_host_top_w)
{
    const int n_blocks = (w_gate_inp_shexp_bf16 != nullptr) ? 65 : 64;
    router_and_shexp_gemv_kernel<true><<<n_blocks, 128, 0, stream>>>(
        gate_inp_bf16, w_gate_inp_shexp_bf16, x_f32, scratch_logits,
        d_top_idx, d_top_w, d_host_top_idx, d_host_top_w);
}

void qwen4_moe_decode_shexp_and_l1_cuda(
    const void* x_q8_1,
    const void* w_gate_shexp, int32_t gate_shexp_type,
    const void* w_up_shexp, int32_t up_shexp_type,
    const void* w_down_shexp, int32_t down_shexp_type,
    const uint8_t* base_vram_exp,
    const uint8_t* const* d_vram_exp_table,
    size_t gate_exp_bytes, size_t up_exp_bytes, size_t expert_blob_bytes,
    int32_t gate_exp_type, int32_t up_exp_type, int32_t down_exp_type,
    const int32_t* d_top_idx,
    float* d_down_11,
    void* q8_scratch_11,
    cudaStream_t stream)
{
    (void)up_exp_type;
    const size_t gs_rb  = qwen4_ggml_row_bytes(gate_shexp_type, QWEN4_N_EMBD);
    const size_t us_rb  = qwen4_ggml_row_bytes(up_shexp_type,   QWEN4_N_EMBD);
    const size_t ds_rb  = qwen4_ggml_row_bytes(down_shexp_type, QWEN4_MOE_FF_DIM);
    const size_t exp_rb = qwen4_ggml_row_bytes(gate_exp_type,   QWEN4_N_EMBD);
    const size_t dn_rb  = qwen4_ggml_row_bytes(down_exp_type,   QWEN4_MOE_FF_DIM);
    const size_t dn_off = gate_exp_bytes + up_exp_bytes;

    const block_q8_1* x_ptr = (const block_q8_1*)x_q8_1;
    block_q8_1* q8_interm   = (block_q8_1*)q8_scratch_11;
    const uint8_t* wgs = (const uint8_t*)w_gate_shexp;
    const uint8_t* wus = (const uint8_t*)w_up_shexp;
    const uint8_t* wds = (const uint8_t*)w_down_shexp;

    switch (gate_exp_type) {
        case 2:  launch_mmvq_gate_up_shexp_routed_typed<2> (x_ptr, wgs, gate_shexp_type, gs_rb, wus, up_shexp_type, us_rb, base_vram_exp, d_vram_exp_table, gate_exp_bytes, expert_blob_bytes, exp_rb, d_top_idx, q8_interm, stream); break;
        case 3:  launch_mmvq_gate_up_shexp_routed_typed<3> (x_ptr, wgs, gate_shexp_type, gs_rb, wus, up_shexp_type, us_rb, base_vram_exp, d_vram_exp_table, gate_exp_bytes, expert_blob_bytes, exp_rb, d_top_idx, q8_interm, stream); break;
        case 6:  launch_mmvq_gate_up_shexp_routed_typed<6> (x_ptr, wgs, gate_shexp_type, gs_rb, wus, up_shexp_type, us_rb, base_vram_exp, d_vram_exp_table, gate_exp_bytes, expert_blob_bytes, exp_rb, d_top_idx, q8_interm, stream); break;
        case 7:  launch_mmvq_gate_up_shexp_routed_typed<7> (x_ptr, wgs, gate_shexp_type, gs_rb, wus, up_shexp_type, us_rb, base_vram_exp, d_vram_exp_table, gate_exp_bytes, expert_blob_bytes, exp_rb, d_top_idx, q8_interm, stream); break;
        case 8:  launch_mmvq_gate_up_shexp_routed_typed<8> (x_ptr, wgs, gate_shexp_type, gs_rb, wus, up_shexp_type, us_rb, base_vram_exp, d_vram_exp_table, gate_exp_bytes, expert_blob_bytes, exp_rb, d_top_idx, q8_interm, stream); break;
        case 12: launch_mmvq_gate_up_shexp_routed_typed<12>(x_ptr, wgs, gate_shexp_type, gs_rb, wus, up_shexp_type, us_rb, base_vram_exp, d_vram_exp_table, gate_exp_bytes, expert_blob_bytes, exp_rb, d_top_idx, q8_interm, stream); break;
        case 13: launch_mmvq_gate_up_shexp_routed_typed<13>(x_ptr, wgs, gate_shexp_type, gs_rb, wus, up_shexp_type, us_rb, base_vram_exp, d_vram_exp_table, gate_exp_bytes, expert_blob_bytes, exp_rb, d_top_idx, q8_interm, stream); break;
        case 14: launch_mmvq_gate_up_shexp_routed_typed<14>(x_ptr, wgs, gate_shexp_type, gs_rb, wus, up_shexp_type, us_rb, base_vram_exp, d_vram_exp_table, gate_exp_bytes, expert_blob_bytes, exp_rb, d_top_idx, q8_interm, stream); break;
        case 16: launch_mmvq_gate_up_shexp_routed_typed<16>(x_ptr, wgs, gate_shexp_type, gs_rb, wus, up_shexp_type, us_rb, base_vram_exp, d_vram_exp_table, gate_exp_bytes, expert_blob_bytes, exp_rb, d_top_idx, q8_interm, stream); break;
        case 17: launch_mmvq_gate_up_shexp_routed_typed<17>(x_ptr, wgs, gate_shexp_type, gs_rb, wus, up_shexp_type, us_rb, base_vram_exp, d_vram_exp_table, gate_exp_bytes, expert_blob_bytes, exp_rb, d_top_idx, q8_interm, stream); break;
        case 18: launch_mmvq_gate_up_shexp_routed_typed<18>(x_ptr, wgs, gate_shexp_type, gs_rb, wus, up_shexp_type, us_rb, base_vram_exp, d_vram_exp_table, gate_exp_bytes, expert_blob_bytes, exp_rb, d_top_idx, q8_interm, stream); break;
        case 20: launch_mmvq_gate_up_shexp_routed_typed<20>(x_ptr, wgs, gate_shexp_type, gs_rb, wus, up_shexp_type, us_rb, base_vram_exp, d_vram_exp_table, gate_exp_bytes, expert_blob_bytes, exp_rb, d_top_idx, q8_interm, stream); break;
        case 21: launch_mmvq_gate_up_shexp_routed_typed<21>(x_ptr, wgs, gate_shexp_type, gs_rb, wus, up_shexp_type, us_rb, base_vram_exp, d_vram_exp_table, gate_exp_bytes, expert_blob_bytes, exp_rb, d_top_idx, q8_interm, stream); break;
        case 22: launch_mmvq_gate_up_shexp_routed_typed<22>(x_ptr, wgs, gate_shexp_type, gs_rb, wus, up_shexp_type, us_rb, base_vram_exp, d_vram_exp_table, gate_exp_bytes, expert_blob_bytes, exp_rb, d_top_idx, q8_interm, stream); break;
        case 23: launch_mmvq_gate_up_shexp_routed_typed<23>(x_ptr, wgs, gate_shexp_type, gs_rb, wus, up_shexp_type, us_rb, base_vram_exp, d_vram_exp_table, gate_exp_bytes, expert_blob_bytes, exp_rb, d_top_idx, q8_interm, stream); break;
        case 29: launch_mmvq_gate_up_shexp_routed_typed<29>(x_ptr, wgs, gate_shexp_type, gs_rb, wus, up_shexp_type, us_rb, base_vram_exp, d_vram_exp_table, gate_exp_bytes, expert_blob_bytes, exp_rb, d_top_idx, q8_interm, stream); break;
        case 42: launch_mmvq_gate_up_shexp_routed_typed<42>(x_ptr, wgs, gate_shexp_type, gs_rb, wus, up_shexp_type, us_rb, base_vram_exp, d_vram_exp_table, gate_exp_bytes, expert_blob_bytes, exp_rb, d_top_idx, q8_interm, stream); break;
        default: break;
    }

    switch (down_exp_type) {
        case 2:  launch_mmvq_down_shexp_routed_typed<2> (q8_interm, wds, down_shexp_type, ds_rb, base_vram_exp, d_vram_exp_table, dn_off, expert_blob_bytes, dn_rb, d_top_idx, d_down_11, stream); break;
        case 3:  launch_mmvq_down_shexp_routed_typed<3> (q8_interm, wds, down_shexp_type, ds_rb, base_vram_exp, d_vram_exp_table, dn_off, expert_blob_bytes, dn_rb, d_top_idx, d_down_11, stream); break;
        case 6:  launch_mmvq_down_shexp_routed_typed<6> (q8_interm, wds, down_shexp_type, ds_rb, base_vram_exp, d_vram_exp_table, dn_off, expert_blob_bytes, dn_rb, d_top_idx, d_down_11, stream); break;
        case 7:  launch_mmvq_down_shexp_routed_typed<7> (q8_interm, wds, down_shexp_type, ds_rb, base_vram_exp, d_vram_exp_table, dn_off, expert_blob_bytes, dn_rb, d_top_idx, d_down_11, stream); break;
        case 8:  launch_mmvq_down_shexp_routed_typed<8> (q8_interm, wds, down_shexp_type, ds_rb, base_vram_exp, d_vram_exp_table, dn_off, expert_blob_bytes, dn_rb, d_top_idx, d_down_11, stream); break;
        case 20: launch_mmvq_down_shexp_routed_typed<20>(q8_interm, wds, down_shexp_type, ds_rb, base_vram_exp, d_vram_exp_table, dn_off, expert_blob_bytes, dn_rb, d_top_idx, d_down_11, stream); break;
        case 42: launch_mmvq_down_shexp_routed_typed<42>(q8_interm, wds, down_shexp_type, ds_rb, base_vram_exp, d_vram_exp_table, dn_off, expert_blob_bytes, dn_rb, d_top_idx, d_down_11, stream); break;
        default: break;
    }
}

void qwen4_moe_decode_misses_cuda(
    const void* x_q8_1,
    const Qwen4ActiveExpertPtrs* d_miss_ptrs,
    int n_gpu_misses,
    int32_t gate_exp_type, int32_t up_exp_type, int32_t down_exp_type,
    float* d_down_11,
    void* q8_scratch_11,
    cudaStream_t stream,
    uint8_t** d_vram_exp_table)
{
    if (n_gpu_misses <= 0) return;
    (void)up_exp_type;
    launch_mmvq_gate_up_swiglu_batch(
        gate_exp_type,
        nullptr, 0, d_miss_ptrs->gate,
        nullptr, 0, d_miss_ptrs->up,
        nullptr, x_q8_1, q8_scratch_11, QWEN4_N_EMBD, n_gpu_misses, stream);

    const size_t dn_rb = qwen4_ggml_row_bytes(down_exp_type, QWEN4_MOE_FF_DIM);
    const block_q8_1* q8_interm = (const block_q8_1*)q8_scratch_11;
    switch (down_exp_type) {
        case 2:  launch_mmvq_down_misses_typed<2> (d_miss_ptrs, dn_rb, q8_interm, d_down_11, n_gpu_misses, d_vram_exp_table, stream); break;
        case 3:  launch_mmvq_down_misses_typed<3> (d_miss_ptrs, dn_rb, q8_interm, d_down_11, n_gpu_misses, d_vram_exp_table, stream); break;
        case 6:  launch_mmvq_down_misses_typed<6> (d_miss_ptrs, dn_rb, q8_interm, d_down_11, n_gpu_misses, d_vram_exp_table, stream); break;
        case 7:  launch_mmvq_down_misses_typed<7> (d_miss_ptrs, dn_rb, q8_interm, d_down_11, n_gpu_misses, d_vram_exp_table, stream); break;
        case 8:  launch_mmvq_down_misses_typed<8> (d_miss_ptrs, dn_rb, q8_interm, d_down_11, n_gpu_misses, d_vram_exp_table, stream); break;
        case 20: launch_mmvq_down_misses_typed<20>(d_miss_ptrs, dn_rb, q8_interm, d_down_11, n_gpu_misses, d_vram_exp_table, stream); break;
        case 42: launch_mmvq_down_misses_typed<42>(d_miss_ptrs, dn_rb, q8_interm, d_down_11, n_gpu_misses, d_vram_exp_table, stream); break;
        default: break;
    }
}

void qwen4_moe_accum_gr_write_cuda(
    float* R,
    const float* d_down_11,
    const float* d_top_w,
    const float* d_shexp_gate,
    const float* d_cpu_moe_out,
    cudaStream_t stream,
    const float* next_norm_w,
    float* next_normed,
    float eps)
{
    moe_accum_gr_write_and_norm_kernel<<<4, 640, 0, stream>>>(
        R, d_down_11, d_top_w, d_shexp_gate, d_cpu_moe_out, next_norm_w, next_normed, eps);
}

void qwen4_router_top10_batchM_cuda(
    const uint16_t* gate_inp_bf16,
    const float* x_f32,
    int M,
    float* scratch_logits,
    int32_t* d_top_idx,
    float* d_top_w,
    cudaStream_t stream,
    int32_t* d_host_top_idx,
    float* d_host_top_w)
{
    if (M <= 0) return;
    qwen4_project_batchM_cuda(
        QWEN4_GGML_TYPE_BF16, gate_inp_bf16,
        x_f32, nullptr,
        scratch_logits, M, QWEN4_N_EXPERTS, QWEN4_N_EMBD, stream);
    router_top10_kernel<<<M, QWEN4_N_EXPERTS, 0, stream>>>(
        scratch_logits,
        d_top_idx,
        d_top_w,
        d_host_top_idx,
        d_host_top_w);
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

    if (gate_shexp_type == up_shexp_type) {
        launch_mmvq_gate_up_swiglu_batch(
            gate_shexp_type,
            w_gate_shexp, 0, nullptr,
            w_up_shexp,   0, nullptr,
            nullptr, x_q8_1, q8_scratch, QWEN4_N_EMBD, 1, stream);
    } else {
        qwen4_mmvq_cuda(gate_shexp_type, w_gate_shexp, x_q8_1, gate_sh, QWEN4_MOE_FF_DIM, QWEN4_N_EMBD, stream);
        qwen4_mmvq_cuda(up_shexp_type,   w_up_shexp,   x_q8_1, up_sh,   QWEN4_MOE_FF_DIM, QWEN4_N_EMBD, stream);
        swiglu_and_q8_1_kernel<<<1, QWEN4_MOE_FF_DIM, 0, stream>>>(gate_sh, up_sh, (block_q8_1*)q8_scratch);
    }

    qwen4_mmvq_cuda(down_shexp_type, w_down_shexp, q8_scratch, down_sh, QWEN4_N_EMBD, QWEN4_MOE_FF_DIM, stream);
    shared_expert_gate_scale_kernel<<<1, 256, 0, stream>>>(x_f32, w_gate_inp_shexp_bf16, down_sh, y_moe);
}

void qwen4_shared_expert_batchM_cuda(
    const float* x_f32,
    const void* x_q8_1,
    const void* w_gate_shexp, int32_t gate_shexp_type,
    const void* w_up_shexp, int32_t up_shexp_type,
    const void* w_down_shexp, int32_t down_shexp_type,
    const uint16_t* w_gate_inp_shexp_bf16,
    int M,
    float* y_moe,
    float* scratch,
    void* q8_scratch,
    cudaStream_t stream)
{
    if (M <= 0) return;
    if (M == 1) {
        qwen4_shared_expert_cuda(
            x_f32, x_q8_1, w_gate_shexp, gate_shexp_type, w_up_shexp, up_shexp_type,
            w_down_shexp, down_shexp_type, w_gate_inp_shexp_bf16, y_moe, scratch, q8_scratch, stream);
        return;
    }

    for (int m0 = 0; m0 < M; m0 += 2080) {
        const int sub_m     = (M - m0 > 2080) ? 2080 : (M - m0);
        const float* x_sub  = x_f32 + (size_t)m0 * QWEN4_N_EMBD;
        const void* q8_sub  = (x_q8_1 != nullptr) ? ((const uint8_t*)x_q8_1 + (size_t)m0 * 2880) : nullptr;
        float* y_sub        = y_moe + (size_t)m0 * QWEN4_N_EMBD;
        float* gate_sh      = scratch;                                    // [sub_m, 640]
        float* up_sh        = gate_sh + (size_t)sub_m * QWEN4_MOE_FF_DIM; // [sub_m, 640]

        qwen4_project_batchM_cuda(gate_shexp_type, w_gate_shexp, x_sub, q8_sub, gate_sh, sub_m, QWEN4_MOE_FF_DIM, QWEN4_N_EMBD, stream, (m0 == 0 && sub_m >= 64));
        qwen4_project_batchM_cuda(up_shexp_type,   w_up_shexp,   x_sub, q8_sub, up_sh,   sub_m, QWEN4_MOE_FF_DIM, QWEN4_N_EMBD, stream, (sub_m >= 64));

        int dev = 0;
        cudaGetDevice(&dev);
        const bool fuse_down_bf16 = qwen4_can_cublas_gemm(down_shexp_type, sub_m, dev);
        uint16_t* bf16_down = fuse_down_bf16 ? (g_qwen4_bf16_ws[dev] - 35553280ULL + 1638400ULL) : nullptr;
        const bool quant_down = (!fuse_down_bf16 && sub_m < 64) && (down_shexp_type != QWEN4_GGML_TYPE_BF16 && down_shexp_type != QWEN4_GGML_TYPE_F16 && down_shexp_type != QWEN4_GGML_TYPE_F32);
        swiglu_and_q8_1_kernel<<<sub_m, QWEN4_MOE_FF_DIM, 0, stream>>>(gate_sh, up_sh, quant_down ? (block_q8_1*)q8_scratch : nullptr, bf16_down);
        qwen4_project_batchM_cuda(down_shexp_type, w_down_shexp, gate_sh, q8_scratch, y_sub, sub_m, QWEN4_N_EMBD, QWEN4_MOE_FF_DIM, stream, fuse_down_bf16);

        shared_expert_gate_scale_kernel<<<sub_m, 256, 0, stream>>>(x_sub, w_gate_inp_shexp_bf16, y_sub, y_sub);
    }
}

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

        if (gate_type == up_type) {
            launch_mmvq_gate_up_swiglu_batch(
                gate_type,
                base_gate, gate_stride_bytes, gate_ptrs,
                base_up,   up_stride_bytes,   up_ptrs,
                top_idx_range, x_q8_1, q8_range, QWEN4_N_EMBD, count_k, stream);
        } else {
            launch_mmvq_batch10(
                gate_type, base_gate, gate_stride_bytes, gate_ptrs, top_idx_range,
                x_q8_1, 0, gate_range, QWEN4_MOE_FF_DIM, QWEN4_N_EMBD, count_k, stream);

            launch_mmvq_batch10(
                up_type, base_up, up_stride_bytes, up_ptrs, top_idx_range,
                x_q8_1, 0, up_range, QWEN4_MOE_FF_DIM, QWEN4_N_EMBD, count_k, stream);

            swiglu_and_q8_1_kernel<<<count_k, QWEN4_MOE_FF_DIM, 0, stream>>>(
                gate_range, up_range, q8_range);
        }

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

void qwen4_routed_experts_tasks_cuda(
    const void* x_q8_1,
    const void* base_gate, size_t gate_stride_bytes, int32_t gate_type,
    const void* base_up, size_t up_stride_bytes, int32_t up_type,
    const void* base_down, size_t down_stride_bytes, int32_t down_type,
    const void* const* d_gate_ptrs,
    const void* const* d_up_ptrs,
    const void* const* d_down_ptrs,
    const Qwen4MoETileTask* d_tasks,
    const int32_t* d_num_tasks,
    int max_tasks,
    float* y_moe,
    void* q8_scratch,
    cudaStream_t stream,
    int phase)
{
    if (max_tasks <= 0) return;
    const size_t gu_row_bytes = qwen4_ggml_row_bytes(gate_type, QWEN4_N_EMBD);
    const size_t dn_row_bytes = qwen4_ggml_row_bytes(down_type, QWEN4_MOE_FF_DIM);
    const uint8_t* bg = (const uint8_t*)base_gate;
    const uint8_t* bu = (const uint8_t*)base_up;
    const uint8_t* bd = (const uint8_t*)base_down;
    const block_q8_1* x_all = (const block_q8_1*)x_q8_1;
    block_q8_1* q8_interm   = (block_q8_1*)q8_scratch;
    (void)up_type;

    if (phase != 2) {
        switch (gate_type) {
            case 2:  launch_mmvq_gate_up_tile4_typed<2> (bg, gate_stride_bytes, d_gate_ptrs, bu, up_stride_bytes, d_up_ptrs, gu_row_bytes, x_all, d_tasks, d_num_tasks, max_tasks, q8_interm, QWEN4_N_EMBD, stream); break;
            case 3:  launch_mmvq_gate_up_tile4_typed<3> (bg, gate_stride_bytes, d_gate_ptrs, bu, up_stride_bytes, d_up_ptrs, gu_row_bytes, x_all, d_tasks, d_num_tasks, max_tasks, q8_interm, QWEN4_N_EMBD, stream); break;
            case 6:  launch_mmvq_gate_up_tile4_typed<6> (bg, gate_stride_bytes, d_gate_ptrs, bu, up_stride_bytes, d_up_ptrs, gu_row_bytes, x_all, d_tasks, d_num_tasks, max_tasks, q8_interm, QWEN4_N_EMBD, stream); break;
            case 7:  launch_mmvq_gate_up_tile4_typed<7> (bg, gate_stride_bytes, d_gate_ptrs, bu, up_stride_bytes, d_up_ptrs, gu_row_bytes, x_all, d_tasks, d_num_tasks, max_tasks, q8_interm, QWEN4_N_EMBD, stream); break;
            case 8:  launch_mmvq_gate_up_tile4_typed<8> (bg, gate_stride_bytes, d_gate_ptrs, bu, up_stride_bytes, d_up_ptrs, gu_row_bytes, x_all, d_tasks, d_num_tasks, max_tasks, q8_interm, QWEN4_N_EMBD, stream); break;
            case 12: launch_mmvq_gate_up_tile4_typed<12>(bg, gate_stride_bytes, d_gate_ptrs, bu, up_stride_bytes, d_up_ptrs, gu_row_bytes, x_all, d_tasks, d_num_tasks, max_tasks, q8_interm, QWEN4_N_EMBD, stream); break;
            case 13: launch_mmvq_gate_up_tile4_typed<13>(bg, gate_stride_bytes, d_gate_ptrs, bu, up_stride_bytes, d_up_ptrs, gu_row_bytes, x_all, d_tasks, d_num_tasks, max_tasks, q8_interm, QWEN4_N_EMBD, stream); break;
            case 14: launch_mmvq_gate_up_tile4_typed<14>(bg, gate_stride_bytes, d_gate_ptrs, bu, up_stride_bytes, d_up_ptrs, gu_row_bytes, x_all, d_tasks, d_num_tasks, max_tasks, q8_interm, QWEN4_N_EMBD, stream); break;
            case 16: launch_mmvq_gate_up_tile4_typed<16>(bg, gate_stride_bytes, d_gate_ptrs, bu, up_stride_bytes, d_up_ptrs, gu_row_bytes, x_all, d_tasks, d_num_tasks, max_tasks, q8_interm, QWEN4_N_EMBD, stream); break;
            case 17: launch_mmvq_gate_up_tile4_typed<17>(bg, gate_stride_bytes, d_gate_ptrs, bu, up_stride_bytes, d_up_ptrs, gu_row_bytes, x_all, d_tasks, d_num_tasks, max_tasks, q8_interm, QWEN4_N_EMBD, stream); break;
            case 18: launch_mmvq_gate_up_tile4_typed<18>(bg, gate_stride_bytes, d_gate_ptrs, bu, up_stride_bytes, d_up_ptrs, gu_row_bytes, x_all, d_tasks, d_num_tasks, max_tasks, q8_interm, QWEN4_N_EMBD, stream); break;
            case 20: launch_mmvq_gate_up_tile4_typed<20>(bg, gate_stride_bytes, d_gate_ptrs, bu, up_stride_bytes, d_up_ptrs, gu_row_bytes, x_all, d_tasks, d_num_tasks, max_tasks, q8_interm, QWEN4_N_EMBD, stream); break;
            case 21: launch_mmvq_gate_up_tile4_typed<21>(bg, gate_stride_bytes, d_gate_ptrs, bu, up_stride_bytes, d_up_ptrs, gu_row_bytes, x_all, d_tasks, d_num_tasks, max_tasks, q8_interm, QWEN4_N_EMBD, stream); break;
            case 22: launch_mmvq_gate_up_tile4_typed<22>(bg, gate_stride_bytes, d_gate_ptrs, bu, up_stride_bytes, d_up_ptrs, gu_row_bytes, x_all, d_tasks, d_num_tasks, max_tasks, q8_interm, QWEN4_N_EMBD, stream); break;
            case 23: launch_mmvq_gate_up_tile4_typed<23>(bg, gate_stride_bytes, d_gate_ptrs, bu, up_stride_bytes, d_up_ptrs, gu_row_bytes, x_all, d_tasks, d_num_tasks, max_tasks, q8_interm, QWEN4_N_EMBD, stream); break;
            case 29: launch_mmvq_gate_up_tile4_typed<29>(bg, gate_stride_bytes, d_gate_ptrs, bu, up_stride_bytes, d_up_ptrs, gu_row_bytes, x_all, d_tasks, d_num_tasks, max_tasks, q8_interm, QWEN4_N_EMBD, stream); break;
            case 42: launch_mmvq_gate_up_tile4_typed<42>(bg, gate_stride_bytes, d_gate_ptrs, bu, up_stride_bytes, d_up_ptrs, gu_row_bytes, x_all, d_tasks, d_num_tasks, max_tasks, q8_interm, QWEN4_N_EMBD, stream); break;
            default: break;
        }
    }

    if (phase != 1) {
        switch (down_type) {
            case 2:  launch_mmvq_down_tile4_typed<2> (bd, down_stride_bytes, d_down_ptrs, dn_row_bytes, q8_interm, d_tasks, d_num_tasks, max_tasks, y_moe, QWEN4_MOE_FF_DIM, QWEN4_N_EMBD, stream); break;
            case 3:  launch_mmvq_down_tile4_typed<3> (bd, down_stride_bytes, d_down_ptrs, dn_row_bytes, q8_interm, d_tasks, d_num_tasks, max_tasks, y_moe, QWEN4_MOE_FF_DIM, QWEN4_N_EMBD, stream); break;
            case 6:  launch_mmvq_down_tile4_typed<6> (bd, down_stride_bytes, d_down_ptrs, dn_row_bytes, q8_interm, d_tasks, d_num_tasks, max_tasks, y_moe, QWEN4_MOE_FF_DIM, QWEN4_N_EMBD, stream); break;
            case 7:  launch_mmvq_down_tile4_typed<7> (bd, down_stride_bytes, d_down_ptrs, dn_row_bytes, q8_interm, d_tasks, d_num_tasks, max_tasks, y_moe, QWEN4_MOE_FF_DIM, QWEN4_N_EMBD, stream); break;
            case 8:  launch_mmvq_down_tile4_typed<8> (bd, down_stride_bytes, d_down_ptrs, dn_row_bytes, q8_interm, d_tasks, d_num_tasks, max_tasks, y_moe, QWEN4_MOE_FF_DIM, QWEN4_N_EMBD, stream); break;
            case 12: launch_mmvq_down_tile4_typed<12>(bd, down_stride_bytes, d_down_ptrs, dn_row_bytes, q8_interm, d_tasks, d_num_tasks, max_tasks, y_moe, QWEN4_MOE_FF_DIM, QWEN4_N_EMBD, stream); break;
            case 13: launch_mmvq_down_tile4_typed<13>(bd, down_stride_bytes, d_down_ptrs, dn_row_bytes, q8_interm, d_tasks, d_num_tasks, max_tasks, y_moe, QWEN4_MOE_FF_DIM, QWEN4_N_EMBD, stream); break;
            case 14: launch_mmvq_down_tile4_typed<14>(bd, down_stride_bytes, d_down_ptrs, dn_row_bytes, q8_interm, d_tasks, d_num_tasks, max_tasks, y_moe, QWEN4_MOE_FF_DIM, QWEN4_N_EMBD, stream); break;
            case 16: launch_mmvq_down_tile4_typed<16>(bd, down_stride_bytes, d_down_ptrs, dn_row_bytes, q8_interm, d_tasks, d_num_tasks, max_tasks, y_moe, QWEN4_MOE_FF_DIM, QWEN4_N_EMBD, stream); break;
            case 17: launch_mmvq_down_tile4_typed<17>(bd, down_stride_bytes, d_down_ptrs, dn_row_bytes, q8_interm, d_tasks, d_num_tasks, max_tasks, y_moe, QWEN4_MOE_FF_DIM, QWEN4_N_EMBD, stream); break;
            case 18: launch_mmvq_down_tile4_typed<18>(bd, down_stride_bytes, d_down_ptrs, dn_row_bytes, q8_interm, d_tasks, d_num_tasks, max_tasks, y_moe, QWEN4_MOE_FF_DIM, QWEN4_N_EMBD, stream); break;
            case 20: launch_mmvq_down_tile4_typed<20>(bd, down_stride_bytes, d_down_ptrs, dn_row_bytes, q8_interm, d_tasks, d_num_tasks, max_tasks, y_moe, QWEN4_MOE_FF_DIM, QWEN4_N_EMBD, stream); break;
            case 21: launch_mmvq_down_tile4_typed<21>(bd, down_stride_bytes, d_down_ptrs, dn_row_bytes, q8_interm, d_tasks, d_num_tasks, max_tasks, y_moe, QWEN4_MOE_FF_DIM, QWEN4_N_EMBD, stream); break;
            case 22: launch_mmvq_down_tile4_typed<22>(bd, down_stride_bytes, d_down_ptrs, dn_row_bytes, q8_interm, d_tasks, d_num_tasks, max_tasks, y_moe, QWEN4_MOE_FF_DIM, QWEN4_N_EMBD, stream); break;
            case 23: launch_mmvq_down_tile4_typed<23>(bd, down_stride_bytes, d_down_ptrs, dn_row_bytes, q8_interm, d_tasks, d_num_tasks, max_tasks, y_moe, QWEN4_MOE_FF_DIM, QWEN4_N_EMBD, stream); break;
            case 29: launch_mmvq_down_tile4_typed<29>(bd, down_stride_bytes, d_down_ptrs, dn_row_bytes, q8_interm, d_tasks, d_num_tasks, max_tasks, y_moe, QWEN4_MOE_FF_DIM, QWEN4_N_EMBD, stream); break;
            case 42: launch_mmvq_down_tile4_typed<42>(bd, down_stride_bytes, d_down_ptrs, dn_row_bytes, q8_interm, d_tasks, d_num_tasks, max_tasks, y_moe, QWEN4_MOE_FF_DIM, QWEN4_N_EMBD, stream); break;
            default: break;
        }
    }
}

void qwen4_routed_experts_resident_batchM_cuda(
    const void* x_q8_1,
    const void* base_gate, size_t gate_stride_bytes, int32_t gate_type,
    const void* base_up, size_t up_stride_bytes, int32_t up_type,
    const void* base_down, size_t down_stride_bytes, int32_t down_type,
    const int32_t* d_top_idx,
    const float* d_top_w,
    int M,
    float* y_moe,
    float* scratch,
    void* q8_scratch,
    cudaStream_t stream)
{
    if (M <= 0) return;
    if (M == 1) {
        qwen4_routed_experts_batch10_cuda(
            x_q8_1, base_gate, gate_stride_bytes, gate_type, base_up, up_stride_bytes, up_type,
            base_down, down_stride_bytes, down_type, nullptr, d_top_idx, d_top_w, y_moe, scratch, q8_scratch, stream);
        return;
    }
    if (M > 576) {
        for (int m0 = 0; m0 < M; m0 += 576) {
            const int sub_m = (M - m0 > 576) ? 576 : (M - m0);
            qwen4_routed_experts_resident_batchM_cuda(
                (const uint8_t*)x_q8_1 + (size_t)m0 * 2880,
                base_gate, gate_stride_bytes, gate_type,
                base_up, up_stride_bytes, up_type,
                base_down, down_stride_bytes, down_type,
                d_top_idx + (size_t)m0 * QWEN4_N_EXPERTS_ACTIVE,
                d_top_w + (size_t)m0 * QWEN4_N_EXPERTS_ACTIVE,
                sub_m,
                y_moe + (size_t)m0 * QWEN4_N_EMBD,
                scratch, q8_scratch, stream);
        }
        return;
    }

    Qwen4MoETileTask* d_tasks = (Qwen4MoETileTask*)scratch;
    int32_t* d_num_tasks      = (int32_t*)(d_tasks + 1024);
    block_q8_1* q8_interm     = (block_q8_1*)(d_num_tasks + 64);
    (void)q8_scratch;

    build_expert_groups_kernel<<<1, 256, 0, stream>>>(d_top_idx, d_top_w, M, d_tasks, d_num_tasks);

    const int max_tasks = std::min(640, std::min(256, M * QWEN4_N_EXPERTS_ACTIVE) + (M * QWEN4_N_EXPERTS_ACTIVE) / QWEN4_MOE_TILE_M);
    qwen4_routed_experts_tasks_cuda(
        x_q8_1, base_gate, gate_stride_bytes, gate_type, base_up, up_stride_bytes, up_type,
        base_down, down_stride_bytes, down_type, nullptr, nullptr, nullptr,
        d_tasks, d_num_tasks, max_tasks, y_moe, q8_interm, stream);
}

void qwen4_routed_experts_l1_hits_batchM_cuda(
    const void* x_q8_1,
    const uint8_t* const* d_vram_exp_table,
    size_t gate_exp_bytes, size_t up_exp_bytes,
    int32_t gate_type, int32_t up_type, int32_t down_type,
    const int32_t* d_top_idx,
    const float* d_top_w,
    int M,
    float* y_moe,
    float* scratch,
    cudaStream_t stream,
    const uint8_t* const* d_decode_exp_table,
    int exact_tasks,
    int phase)
{
    if (M <= 0 || d_vram_exp_table == nullptr || exact_tasks == 0) return;
    Qwen4MoETileTask* d_tasks = (Qwen4MoETileTask*)scratch;
    int32_t* d_num_tasks      = (int32_t*)(d_tasks + 1520);
    const void** d_gate_ptrs  = (const void**)(d_num_tasks + 64);
    const void** d_up_ptrs    = d_gate_ptrs + 256;
    const void** d_down_ptrs  = d_up_ptrs + 256;
    block_q8_1* q8_interm     = (block_q8_1*)(d_down_ptrs + 256);

    if (phase != 2) {
        build_expert_groups_kernel<<<1, 256, 0, stream>>>(
            d_top_idx, d_top_w, M, d_tasks, d_num_tasks,
            d_vram_exp_table, gate_exp_bytes, up_exp_bytes,
            d_gate_ptrs, d_up_ptrs, d_down_ptrs, d_decode_exp_table);
    }

    const int max_tasks = (exact_tasks > 0)
        ? exact_tasks
        : std::min(1520, std::min(256, M * QWEN4_N_EXPERTS_ACTIVE) + (M * QWEN4_N_EXPERTS_ACTIVE) / QWEN4_MOE_TILE_M);
    qwen4_routed_experts_tasks_cuda(
        x_q8_1, nullptr, 0, gate_type, nullptr, 0, up_type,
        nullptr, 0, down_type, d_gate_ptrs, d_up_ptrs, d_down_ptrs,
        d_tasks, (exact_tasks > 0) ? nullptr : d_num_tasks, max_tasks, y_moe, q8_interm, stream, phase);
}

namespace {

__global__ void __launch_bounds__(1024) qwen4_argmax_f32_masked_kernel(
    const float* __restrict__ logits,
    int vocab_size,
    int mask0,
    int mask1,
    int32_t* __restrict__ out_idx,
    int idx_offset)
{
    __shared__ float   s_val[32];
    __shared__ int32_t s_idx[32];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    float   best_v = -1e30f;
    int32_t best_i = 0;

    const int n4 = vocab_size >> 2;
    const float4* l4 = (const float4*)logits;
    for (int i4 = tid; i4 < n4; i4 += 1024) {
        const float4 v4 = __ldg(l4 + i4);
        const int base = i4 << 2;
        const float a0 = (base + 0 == mask0 || base + 0 == mask1) ? -1e30f : v4.x;
        const float a1 = (base + 1 == mask0 || base + 1 == mask1) ? -1e30f : v4.y;
        const float a2 = (base + 2 == mask0 || base + 2 == mask1) ? -1e30f : v4.z;
        const float a3 = (base + 3 == mask0 || base + 3 == mask1) ? -1e30f : v4.w;
        if (a0 > best_v) { best_v = a0; best_i = base + 0; }
        if (a1 > best_v) { best_v = a1; best_i = base + 1; }
        if (a2 > best_v) { best_v = a2; best_i = base + 2; }
        if (a3 > best_v) { best_v = a3; best_i = base + 3; }
    }
    for (int i = (n4 << 2) + tid; i < vocab_size; i += 1024) {
        const float v = (i == mask0 || i == mask1) ? -1e30f : __ldg(logits + i);
        if (v > best_v) { best_v = v; best_i = i; }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        const float   ov = __shfl_xor_sync(0xFFFFFFFFu, best_v, mask, 32);
        const int32_t oi = __shfl_xor_sync(0xFFFFFFFFu, best_i, mask, 32);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }
    if (lane == 0) {
        s_val[warp] = best_v;
        s_idx[warp] = best_i;
    }
    __syncthreads();

    if (warp == 0) {
        best_v = s_val[lane];
        best_i = s_idx[lane];
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            const float   ov = __shfl_xor_sync(0xFFFFFFFFu, best_v, mask, 32);
            const int32_t oi = __shfl_xor_sync(0xFFFFFFFFu, best_i, mask, 32);
            if (ov > best_v || (ov == best_v && oi < best_i)) {
                best_v = ov;
                best_i = oi;
            }
        }
        if (lane == 0) {
            out_idx[0] = best_i + idx_offset;
            ((float*)out_idx)[1] = best_v;
        }
    }
}

} // anonymous namespace

void qwen4_argmax_f32_cuda(
    const float* d_logits,
    int vocab_size,
    int mask_id0,
    int mask_id1,
    int32_t* d_out_idx,
    cudaStream_t stream,
    int idx_offset)
{
    const int rel_mask0 = (mask_id0 >= 0) ? (mask_id0 - idx_offset) : -1;
    const int rel_mask1 = (mask_id1 >= 0) ? (mask_id1 - idx_offset) : -1;
    qwen4_argmax_f32_masked_kernel<<<1, 1024, 0, stream>>>(
        d_logits, vocab_size, rel_mask0, rel_mask1, d_out_idx, idx_offset);
}

