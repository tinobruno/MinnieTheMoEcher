import torch

def rotate_half_moecher(x):
    # moecher pairs i and i + d//2
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope_moecher(x, cos, sin, inverse=False):
    if inverse:
        sin = -sin
    half = x.shape[-1] // 2
    # cos and sin are repeated
    cos_full = torch.cat([cos, cos], dim=-1)
    sin_full = torch.cat([sin, sin], dim=-1)
    return x * cos_full + rotate_half_moecher(x) * sin_full

b, h, s, d = 1, 1, 1, 4
q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
# freq is length 2
cos = torch.tensor([[[[0.5, 0.6]]]])
sin = torch.tensor([[[[0.8, 0.8]]]])

q_rot = apply_rope_moecher(q, cos, sin)
print("q_rot:", q_rot)

attn_out = q_rot

attn_out_inv = apply_rope_moecher(attn_out, cos, sin, inverse=True)
print("attn_out_inv:", attn_out_inv)
