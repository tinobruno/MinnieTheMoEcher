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

torch.manual_seed(42)
k = torch.randn(1, 1, 1, 4)
angle = torch.tensor([0.1, 0.2])
cos = torch.cat([angle.cos(), angle.cos()], dim=-1).view(1, 1, 1, 4)
sin = torch.cat([angle.sin(), angle.sin()], dim=-1).view(1, 1, 1, 4)

hf_k = apply_rotary_pos_emb(k, cos, sin)
hf_inv = apply_rotary_pos_emb(hf_k, cos, -sin)

print("Original k:\n", k.flatten())
print("HF Inv:\n", hf_inv.flatten())
