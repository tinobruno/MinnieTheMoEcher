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

q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
# Use symbols for clarity: c=1, s=0
cos = torch.tensor([[[[1.0, 1.0, 1.0, 1.0]]]])
sin = torch.tensor([[[[0.0, 0.0, 0.0, 0.0]]]])

# 1. Forward apply
q_rot = apply_rotary_pos_emb(q, cos, sin)
print("Forward (c=1, s=0):", q_rot)

# 2. Assume attention simply passes V=K through
attn_out = q_rot

# 3. Inverse apply
attn_out_inv = apply_rotary_pos_emb(attn_out, cos, -sin)
print("Inverse (c=1, s=0):", attn_out_inv)

