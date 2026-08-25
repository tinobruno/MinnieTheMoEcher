import math
import torch

def yarn_find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )

def yarn_find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)

def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

dim = 64
base = 10000
scaling_factor = 40.0
beta_fast = 32
beta_slow = 1
original_max_position_embeddings = 4096

freq_extra = 1.0 / (
    base
    ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
)
freq_inter = 1.0 / (
    scaling_factor
    * base
    ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
)

low, high = yarn_find_correction_range(
    beta_fast,
    beta_slow,
    dim,
    base,
    original_max_position_embeddings,
)
inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2)
inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

print("YaRN Inv Freqs (Last 8):", inv_freq[-8:].tolist())

base_inv_freq = 1.0 / (10000 ** (torch.arange(0, 64, 2).float() / 64))
print("Base Inv Freqs (Last 8):", base_inv_freq[-8:].tolist())
