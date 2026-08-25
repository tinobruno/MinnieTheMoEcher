import torch
import math

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, cos, sin):
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed

b, h, s, d = 1, 1, 1, 4
q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])

# Valid cos and sin (c^2 + s^2 = 1)
angle = torch.tensor([0.1, 0.2])
cos = torch.cat([angle.cos(), angle.cos()], dim=-1).view(1, 1, 1, 4)
sin = torch.cat([angle.sin(), angle.sin()], dim=-1).view(1, 1, 1, 4)

q_rot = apply_rotary_pos_emb(q, cos, sin)
print("q_rot:", q_rot)

attn_out = q_rot

# apply inverse rope EXACTLY as in hf_attention.py:
attn_out_inv = apply_rotary_pos_emb(attn_out, cos, -sin)
print("attn_out_inv:", attn_out_inv)

