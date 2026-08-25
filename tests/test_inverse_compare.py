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

q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
angle = torch.tensor([0.1, 0.2])
cos = torch.cat([angle.cos(), angle.cos()], dim=-1).view(1, 1, 1, 4)
sin = torch.cat([angle.sin(), angle.sin()], dim=-1).view(1, 1, 1, 4)

# Forward pass
hf_q_rot = apply_rotary_pos_emb(q, cos, sin)
mo_q_rot = apply_rope_moecher(q, cos, sin)

print("Forward hf:", hf_q_rot)
print("Forward mo:", mo_q_rot)

# Assume attention output is just the V vector (K=V)
hf_attn_out = hf_q_rot
mo_attn_out = mo_q_rot

# Inverse pass
hf_inv = apply_rotary_pos_emb(hf_attn_out, cos, -sin)
mo_inv = apply_rope_moecher(mo_attn_out, cos, sin, inverse=True)

print("Inverse hf:", hf_inv)
print("Inverse mo:", mo_inv)
