import sys
content = open('src/cuda/activations.cu').read()

old_e8m0 = """__device__ __forceinline__ float e8m0_to_float(uint8_t bits) {
    if (bits == 0) return 0.0f;
    return exp2f((float)bits - 127.0f);
}"""

new_e8m0 = """__device__ __forceinline__ float e8m0_to_float(uint8_t bits) {
    if (bits == 0) return 0.0f;
    uint32_t val = (uint32_t)bits << 23;
    return __uint_as_float(val);
}"""

content = content.replace(old_e8m0, new_e8m0)

old_fp8 = """__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t bits) {
    uint32_t sign = (bits >> 7) & 1;
    uint32_t exp  = (bits >> 3) & 0xF;
    uint32_t mant = bits & 0x7;
    float val;
    if (exp == 0) {
        // Subnormal: (-1)^sign * 2^(-6) * (mant/8)
        val = ldexpf((float)mant, -9);  // mant * 2^-3 * 2^-6 = mant * 2^-9
    } else if (exp == 15 && mant == 7) {
        val = __int_as_float(0x7FC00000);  // NaN
    } else {
        // Normal: (-1)^sign * 2^(exp-7) * (1 + mant/8)
        val = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
    }
    return sign ? -val : val;
}"""

new_fp8 = """__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t bits) {
    uint32_t sign = (bits & 0x80) << 24;
    uint32_t exp  = (bits & 0x78) >> 3;
    uint32_t mant = bits & 0x07;
    if (exp == 0) {
        if (mant == 0) return __uint_as_float(sign);
        float val = (float)mant;
        uint32_t val_i = __float_as_uint(val);
        val_i -= (9 << 23);
        return __uint_as_float(val_i | sign);
    } else if (exp == 15 && mant == 7) {
        return __uint_as_float(sign | 0x7FC00000);
    } else {
        uint32_t f_exp = (exp + 120) << 23;
        uint32_t f_mant = mant << 20;
        return __uint_as_float(sign | f_exp | f_mant);
    }
}"""

content = content.replace(old_fp8, new_fp8)

open('src/cuda/activations.cu', 'w').write(content)
