import torch

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, cos, sin):
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed

def apply_rope_moecher(x, cos, sin, inverse=False):
    if inverse:
        sin = -sin
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    c = cos[..., :half]
    s = sin[..., :half]
    rx1 = x1 * c - x2 * s
    rx2 = x2 * c + x1 * s
    return torch.cat([rx1, rx2], dim=-1)

q = torch.randn(1, 1, 1, 4)
k = torch.randn(1, 1, 1, 4)
angle = torch.tensor([0.1, 0.2])
cos = torch.cat([angle.cos(), angle.cos()], dim=-1).view(1, 1, 1, 4)
sin = torch.cat([angle.sin(), angle.sin()], dim=-1).view(1, 1, 1, 4)

hf_q = apply_rotary_pos_emb(q, cos, sin)
hf_k = apply_rotary_pos_emb(k, cos, sin)
hf_dot = (hf_q * hf_k).sum()

mo_q = apply_rope_moecher(q, cos, sin)
mo_k = apply_rope_moecher(k, cos, sin)
mo_dot = (mo_q * mo_k).sum()

print("HF Dot:", hf_dot.item())
print("MO Dot:", mo_dot.item())
