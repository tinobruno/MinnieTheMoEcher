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

b, h, s, d = 1, 1, 1, 4
q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
cos = torch.tensor([[[[0.5, 0.6, 0.5, 0.6]]]])
sin = torch.tensor([[[[0.8, 0.8, 0.8, 0.8]]]])

q_rot = apply_rotary_pos_emb(q, cos, sin)
print("q_rot:", q_rot)

# simulate attention with K=V (so V is q_rot)
attn_out = q_rot

# apply inverse rope
attn_out_inv = apply_rotary_pos_emb(attn_out, cos, -sin)
print("attn_out_inv:", attn_out_inv)
