import torch
import math

class DeepseekV3RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

def yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)

def yarn_linear_ramp_mask(min, max, dim):
    if min == max: max += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    return torch.clamp(linear_func, 0, 1)

class DeepseekV3YarnRotaryEmbedding(DeepseekV3RotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0, original_max_position_embeddings=4096, beta_fast=32, beta_slow=1, mscale=1, mscale_all_dim=0):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base, device)
        freq_extra = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        freq_inter = 1.0 / (self.scaling_factor * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        low, high = yarn_find_correction_range(self.beta_fast, self.beta_slow, dim, self.base, self.original_max_position_embeddings)
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(device=device, dtype=torch.float32)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

config = type("Config", (), {})()
config.qk_rope_head_dim = 64
config.max_position_embeddings = 4096
config.original_max_position_embeddings = 4096
config.rope_theta = 10000
config.rope_scaling = {
    "type": "yarn",
    "factor": 40.0,
    "original_max_position_embeddings": 4096,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
}
rope = DeepseekV3YarnRotaryEmbedding(dim=64, max_position_embeddings=4096, base=10000, device="cpu", scaling_factor=40.0, beta_fast=32, beta_slow=1, mscale=1, mscale_all_dim=1, original_max_position_embeddings=4096)
